"""P0/P1 warm-start packet readout screen; no torch imports in metadata mode."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

from e3_gate_study import (activate_gpu, checked_gpu, guard_graph_loads, identity,
                           inside, read, require, runtime, sha, write)
import run_fusion_gate_pair as source_pair
from summarize_fusion_gate_pair import analyze as analyze_source

dim = source_pair.parent
pair = dim.pair
HERE = Path(__file__).resolve().parent
MODULE = HERE.parent / 'src/xgnid/packet_pooling.py'
VARIANTS = {'P0': 'mean', 'P1': 'attention'}
TENSORS = ('flow64', 'packet64', 'ptr', 'labels')


def check_config(c):
    expected = {'schema': 'xgnid.packet-pool-pair.v1', 'variants': VARIANTS,
        'source_variant': 'D32-Gate', 'source_epoch': 62,
        'source_macro_f1': .9732070542814468,
        'source_best_sha256': '7815875a111d3e94ceefd0e9ef6cb64dedaf6ff80723645f1ec035b376af31ff',
        'seed': 42, 'width': 32, 'encoder_mode': 'frozen_eval',
        'inputs': 'frozen_flow64_and_ragged_packet64_train_val_only',
        'score_network': [64, 16, 1], 'score_output_bias': False,
        'score_output_initialization': 'zero', 'optimizer': 'fresh_Adam',
        'head_lr': 1e-5, 'pool_lr': 1e-4, 'weight_decay': 0., 'batch_size': 512,
        'eval_batch_size': 16, 'workers': 2, 'epochs': 100, 'patience': 20,
        'include_epoch_zero': True, 'probe_batches': 3, 'loss': 'plain_final_NLL',
        'budget_above_stronger_reference_pp': .10, 'training_jobs': 2,
        'test_evaluated': False, 'automatic_followup': False}
    require(all(c.get(k) == v for k, v in expected.items()), 'Unregistered packet-pool protocol')


def architecture(mode):
    require(mode in VARIANTS.values(), 'Unknown pooling mode')
    return {**dim.architecture(32), 'packet_pooling': mode,
        'pool_parameters': 1056 if mode == 'attention' else 0,
        'trainable_parameters': 8200 if mode == 'attention' else 7144,
        'packet_score': [64, 16, 1] if mode == 'attention' else None}


def metadata(args):
    c = read(args.config)
    check_config(c)
    original = inside(args.config.parent, c['source_protocol'])
    require(sha(original) == c['source_config_sha256'], 'Source protocol changed')
    previous = source_pair.metadata(SimpleNamespace(repo=args.repo, config=original))
    root = inside(args.repo / 'outputs', inside(args.repo, c['source_directory']))
    require(read(root / 'status.json') == {'status': 'completed', 'exit_code': 0}, 'D32 source incomplete')
    s = read(root / 'summary.json')
    analyze_source(s)
    require(s['protocol'] == previous[0] == read(root / 'config.json')
            and s['binding'] == previous[-1] == read(root / 'binding.json')
            and s['encoder_parameters_and_BN_unchanged'] and s['shared_cache_unchanged']
            and s['source_artifacts_unchanged'], 'D32 source binding/state changed')
    m = s['models']['D32-Gate']
    require(m == read(root / 'D32-Gate/metrics.json') and m['best_epoch'] == c['source_epoch']
            and m['architecture'] == source_pair.architecture('dynamic')
            and abs(m['best_validation']['macro_f1'] - c['source_macro_f1']) < 1e-12
            and sha(root / 'D32-Gate/best.pt') == m['checkpoint_sha256']['best.pt'] == c['source_best_sha256']
            and sha(root / 'D32-Gate/validation_predictions.jsonl.gz') == m['predictions_sha256'],
            'Wrong mature D32-Gate checkpoint/score/predictions')
    cfg = read(root / 'D32-Gate/config.json')
    require(cfg['binding'] == s['binding'] and cfg['protocol'] == previous[0]
            and cfg['variant'] == 'D32-Gate' and cfg['architecture'] == m['architecture'], 'Source arm config differs')
    replay = read(root / 'D32-Gate/graph_replay.json')
    require(replay == s['graph_replays']['D32-Gate'] and replay['strict_state_load']
            and replay['confusion_matrix_reproduced'], 'Source graph replay incomplete')
    files = ['status.json', 'config.json', 'binding.json', 'summary.json',
             'D32-Gate/best.pt', 'D32-Gate/config.json', 'D32-Gate/metrics.json',
             'D32-Gate/validation_predictions.jsonl.gz', 'D32-Gate/graph_replay.json']
    binding = {'config_sha256': sha(args.config), 'source_pair_binding': previous[-1],
        'source': {'directory': str(root), 'file_sha256': {n: sha(root / n) for n in files}},
        'sources': {n: sha(p) for n, p in {'scripts/run_packet_pool_pair.py': Path(__file__),
            'src/xgnid/packet_pooling.py': MODULE,
            'scripts/summarize_packet_pool_pair.py': HERE / 'summarize_packet_pool_pair.py'}.items()}}
    return c, previous, root, s, binding


def load_classes():
    Head, _ = dim.load_classes()
    name = 'xgnid.packet_pooling'
    spec = importlib.util.spec_from_file_location(name, MODULE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return Head, module.PacketReadoutClassifier


def build(core, classes, data, device, n0, state, mode):
    Head, Graph = classes
    backbone = dim.build_backbone(core, data, device, n0)
    head = dim.make_head(core, Head, 32, device, dim.initialization_streams(42))
    rng = pair.rng_state(core)
    graph = Graph(backbone.flow_encoder, backbone.packet_encoder, head, mode).to(device)
    require(pair.rng_hash(core, rng) == pair.rng_hash(core, pair.rng_state(core)), 'Pool construction changed RNG')
    initial = graph.state_dict()
    common = {n: t for n, t in initial.items() if not n.startswith('packet_pool.')}
    require(set(common) == set(state), 'Source state layout changed')
    initial.update(state)
    graph.load_state_dict(initial, strict=True)
    require(core.state_hash((n, t) for n, t in graph.state_dict().items() if n in common)
            == core.state_hash(state.items()), 'Source common weights were changed')
    require(sum(p.numel() for p in graph.parameters() if p.requires_grad)
            == architecture(mode)['trainable_parameters'], 'Wrong trainable capacity')
    return graph.eval()


def optimizer(core, graph, c):
    head = list(graph.fusion.parameters())
    pool = list(graph.packet_pool.parameters()) if graph.packet_pool is not None else []
    groups = [{'name': 'existing_head', 'params': head, 'lr': c['head_lr']}]
    if pool:
        groups.append({'name': 'new_packet_pool', 'params': pool, 'lr': c['pool_lr']})
    flat = head + pool
    enc = {id(p) for e in (graph.flow_encoder, graph.packet_encoder) for p in e.parameters()}
    require(len({id(p) for p in flat}) == len(flat)
            and {id(p) for p in flat} == {id(p) for p in graph.parameters() if p.requires_grad}
            and not enc.intersection(id(p) for p in flat), 'Optimizer omitted/duplicated/leaked parameters')
    require(sum(p.numel() for p in head) == 7144 and sum(p.numel() for p in pool)
            == architecture(graph.pooling)['pool_parameters'], 'Parameter groups differ')
    opt = core.torch.optim.Adam(groups, weight_decay=c['weight_decay'])
    require(not opt.state, 'Adam must start clean')
    return opt


def loader(core, data, part, c):
    return core.DataLoader(core.GraphRows(data, core.select_rows(data, part)), batch_size=16,
        shuffle=False, num_workers=c['workers'], pin_memory=True, persistent_workers=True,
        multiprocessing_context='fork')


def tensor_hash(core, cache):
    return core.state_hash((n, cache[n]) for n in TENSORS)


def extract(core, args, model, data, device, part, c, encoder_hash, split_binding):
    rows = core.select_rows(data, part)
    ids, flows, nodes, labels, counts, variation = [], [], [], [], [], []
    started = time.perf_counter()
    for step, batch in enumerate(loader(core, data, part, c), 1):
        batch = batch.to(device)
        flow, packet, index, size = model.encode(batch)
        current = rows[len(ids):len(ids) + size]
        require(list(batch.sample_id) == [r['sample_id'] for r in current]
                and batch.y.view(-1).tolist() == [r['label'] for r in current], 'Extraction order/labels differ')
        require(flow.shape == (size, 64) and packet.ndim == 2 and packet.size(1) == 64
                and bool(core.torch.isfinite(flow).all()) and bool(core.torch.isfinite(packet).all()),
                'Invalid frozen representations')
        n = core.torch.bincount(index, minlength=size)
        require(bool((n > 0).all()) and bool(core.torch.equal(index,
                    core.torch.repeat_interleave(core.torch.arange(size, device=device), n))),
                'Packet nodes must be present and grouped in original graph order')
        for j in range(size):
            x = packet[index == j]
            # RMS Euclidean distance to the graph mean; zero for a singleton.
            variation.append(float((x - x.mean(0)).square().sum(-1).mean().sqrt()))
        ids.extend(batch.sample_id)
        flows.append(flow.cpu())
        nodes.append(packet.cpu())
        labels.append(batch.y.view(-1).cpu())
        counts.extend(n.tolist())
        if step == 1 or step % 200 == 0:
            dim.progress(args, 'extract_packet_nodes', partition=part, samples=len(ids), total=len(rows))
    require(len(ids) == len(rows), 'Incomplete extraction')
    cache = {'ids': ids, 'flow64': core.torch.cat(flows), 'packet64': core.torch.cat(nodes),
        'labels': core.torch.cat(labels), 'ptr': core.torch.tensor([0] + counts, dtype=core.torch.long).cumsum(0),
        'packet_rms_distance': variation,
        'metadata': {'schema': 'xgnid.packet-node64-cache.v1', 'partition': part,
            'encoder_sha256': encoder_hash, 'split_binding': split_binding,
            'sample_order_sha256': identity(ids), 'samples': len(rows)}}
    path = args.output / (part + '_packet_nodes.pt')
    core.checkpoint(path, cache)
    receipt = {'sha256': sha(path), 'tensor_sha256': tensor_hash(core, cache),
               'metadata': cache['metadata'], 'wall_seconds': time.perf_counter() - started}
    dim.assert_frozen(core, model, encoder_hash)
    return cache, receipt


def validate_cache(core, cache, rows, part, encoder_hash, split_binding):
    n = len(rows)
    require(cache['ids'] == [r['sample_id'] for r in rows]
            and len(set(cache['ids'])) == n and cache['labels'].tolist() == [r['label'] for r in rows]
            and cache['metadata'] == {'schema': 'xgnid.packet-node64-cache.v1', 'partition': part,
                'encoder_sha256': encoder_hash, 'split_binding': split_binding,
                'sample_order_sha256': identity(cache['ids']), 'samples': n}, 'Node cache identity differs')
    ptr = cache['ptr']
    require(ptr.dtype == core.torch.int64 and ptr.shape == (n + 1,) and int(ptr[0]) == 0
            and bool((ptr[1:] > ptr[:-1]).all()) and int(ptr[-1]) == len(cache['packet64'])
            and cache['flow64'].shape == (n, 64) and cache['packet64'].shape == (int(ptr[-1]), 64)
            and cache['labels'].dtype == core.torch.int64
            and all(bool(core.torch.isfinite(cache[k]).all()) for k in ('flow64', 'packet64')),
            'Node cache ragged shape/content invalid')
    require(len(cache['packet_rms_distance']) == n
            and all(math.isfinite(v) and v >= 0 for v in cache['packet_rms_distance']), 'Invalid packet diagnostics')


def minibatch(core, cache, numbers, device):
    chunks = [cache['packet64'][int(cache['ptr'][i]):int(cache['ptr'][i + 1])] for i in numbers]
    lengths = core.torch.tensor([len(x) for x in chunks], device=device)
    index = core.torch.repeat_interleave(core.torch.arange(len(numbers), device=device), lengths)
    return (cache['flow64'][numbers].to(device), core.torch.cat(chunks).to(device), index, len(numbers)), \
        cache['labels'][numbers].to(device)


def diagnostics(core, cache):
    result = {}
    for y, label in enumerate(core.LABELS):
        ix = [i for i, v in enumerate(cache['labels'].tolist()) if v == y]
        counts = [int(cache['ptr'][i + 1] - cache['ptr'][i]) for i in ix]
        multi = [cache['packet_rms_distance'][i] for i, n in zip(ix, counts) if n > 1]
        histogram = {str(n): counts.count(n) for n in sorted(set(counts))}
        result[label] = {'graphs': len(ix), 'packet_count_histogram': histogram,
            'multi_packet_graphs': len(multi), 'multi_packet_fraction': len(multi) / len(ix),
            'max_packets': max(counts), 'mean_packets': sum(counts) / len(ix),
            'varying_multi_packet_graphs': sum(v > 1e-7 for v in multi),
            'multi_packet_rms_mean': sum(multi) / len(multi) if multi else None,
            'multi_packet_rms_max': max(multi) if multi else None}
    return result


def evaluate(core, model, cache, device, export=False):
    model.eval()
    cm, exports = [[0] * 8 for _ in range(8)], []
    with core.torch.no_grad():
        for offset in range(0, len(cache['ids']), 16):
            numbers = list(range(offset, min(offset + 16, len(cache['ids']))))
            features, labels = minibatch(core, cache, numbers, device)
            scores, alpha = model.forward_features(*features)
            require(scores.shape == (len(numbers), 8) and alpha.shape == (len(numbers), 32)
                    and bool(core.torch.isfinite(scores).all()), 'Invalid evaluation')
            beta = model.packet_weights(features[1], features[2], features[3]) if export else None
            for j, (i, y, p) in enumerate(zip(numbers, labels.tolist(), scores.argmax(-1).tolist())):
                cm[y][p] += 1
                if export:
                    b = beta[features[2] == j]
                    require(bool(core.torch.isfinite(b).all()) and abs(float(b.sum()) - 1.) < 1e-5,
                            'Packet attention is not normalized within graph')
                    exports.append({'sample_id': cache['ids'][i], 'label': y, 'prediction': p,
                        'flow_weights': alpha[j].tolist(), 'packet_attention': b.tolist(),
                        'packet_count': len(b), 'max_abs_attention_from_uniform': float((b - 1 / len(b)).abs().max())})
    metrics = dim.frozen.matrix_report(cm, core.LABELS)
    dim.verify_metrics(metrics)
    return metrics, exports


def initial_equivalence(core, models, cache, device, gold, expected):
    error = 0.
    for model in models.values():
        model.eval()
    with core.torch.no_grad():
        for offset in range(0, len(cache['ids']), 16):
            features, _ = minibatch(core, cache, list(range(offset, min(offset + 16, len(cache['ids'])))), device)
            a, aw = models['P0'].forward_features(*features)
            b, bw = models['P1'].forward_features(*features)
            require(core.torch.allclose(a, b, atol=1e-6, rtol=1e-6)
                    and core.torch.allclose(aw, bw, atol=1e-6, rtol=1e-6), 'Non-equivalent epoch0 readout')
            error = max(error, float((a - b).abs().max()))
            beta = models['P1'].packet_weights(features[1], features[2], features[3])
            uniform = models['P0'].packet_weights(features[1], features[2], features[3])
            require(core.torch.allclose(beta, uniform, atol=1e-7, rtol=1e-7), 'Initial packet weights nonuniform')
    for name, model in models.items():
        m, exports = evaluate(core, model, cache, device, export=True)
        require(m['confusion_matrix'] == expected['confusion_matrix'], 'Epoch0 source score differs: ' + name)
        require([(r['sample_id'], r['label'], r['prediction']) for r in exports]
                == [(r['sample_id'], r['label'], r['prediction']) for r in gold], 'Epoch0 source predictions differ')
    return {'samples': len(cache['ids']), 'max_abs_P0_P1_log_probability_error': error,
        'P0_P1_outputs_allclose': True, 'source_predictions_identical': True,
        'uniform_packet_attention': True, 'epoch_zero_eligible': True}


def probe(core, args, model, data, cache, device, c, encoder_hash):
    clean = {n: t.detach().clone() for n, t in model.state_dict().items()}
    digest, rng = core.state_hash(clean.items()), pair.rng_state(core)
    # A singleton cannot train packet scores. Select informative TRAIN graphs.
    eligible = [i for i, v in enumerate(cache['packet_rms_distance'])
                if int(cache['ptr'][i + 1] - cache['ptr'][i]) > 1 and v > 1e-7]
    require(bool(eligible), 'No varying multi-packet training representations; inspect packet_diagnostics.json')
    numbers = eligible[:512]
    rows = core.select_rows(data, 'train')
    batch = core.Batch.from_data_list([data.load(rows[i]) for i in numbers]).to(device)
    require(list(batch.sample_id) == [cache['ids'][i] for i in numbers]
            and batch.y.view(-1).tolist() == cache['labels'][numbers].tolist(), 'Probe raw/cache sample order differs')
    receipts = []
    try:
        opt = optimizer(core, model, c)
        for step in range(1, c['probe_batches'] + 1):
            model.train()
            before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
            opt.zero_grad(set_to_none=True)
            raw = model(batch)
            features, y = minibatch(core, cache, numbers, device)
            cached, _ = model.forward_features(*features)
            require(core.torch.allclose(raw, cached, atol=1e-5, rtol=1e-5), 'Raw/node-cache probe differs')
            loss = core.F.nll_loss(raw, y)
            require(bool(core.torch.isfinite(loss)) and raw.requires_grad, 'Pool is disconnected')
            loss.backward()
            require(all(p.grad is None or bool(core.torch.isfinite(p.grad).all()) for p in model.parameters()),
                    'Nonfinite gradient')
            norms = {n: float(p.grad.abs().sum()) if p.grad is not None else 0.
                     for n, p in model.named_parameters() if p.requires_grad}
            opt.step()
            changed = {n: not core.torch.equal(before[n], p) for n, p in model.named_parameters() if p.requires_grad}
            with core.torch.no_grad():
                beta = model.packet_weights(features[1], features[2], features[3])
                counts = core.torch.bincount(features[2], minlength=len(numbers))
                uniform = counts[features[2]].to(beta.dtype).reciprocal()
            receipts.append({'step': step, 'loss': float(loss.detach()), 'gradient_l1': norms,
                'changed': changed, 'max_abs_attention_from_uniform': float((beta - uniform).abs().max())})
            dim.assert_frozen(core, model, encoder_hash)
        prefixes = ['fusion.flow_projection.', 'fusion.packet_projection.', 'fusion.gate.', 'fusion.shared_head.']
        if model.pooling == 'attention':
            prefixes += ['packet_pool.aggregation.gate_nn.0.', 'packet_pool.aggregation.gate_nn.2.']
            require(any(r['max_abs_attention_from_uniform'] > 0 for r in receipts), 'Packet weights never learned')
        for prefix in prefixes:
            require(any(any(n.startswith(prefix) and v > 0 for n, v in r['gradient_l1'].items()) for r in receipts)
                    and any(any(n.startswith(prefix) and v for n, v in r['changed'].items()) for r in receipts),
                    'No gradient/update after probe: ' + prefix)
    finally:
        model.load_state_dict(clean, strict=True)
        model.zero_grad(set_to_none=True)
        model.eval()
        pair.restore_rng(core, rng)
        require(core.state_hash(model.state_dict().items()) == digest
                and pair.rng_hash(core, rng) == pair.rng_hash(core, pair.rng_state(core)), 'Probe restoration failed')
        dim.assert_frozen(core, model, encoder_hash)
    return {'updates': receipts, 'train_graphs': len(numbers), 'informative_train_graphs': len(eligible),
        'discarded': True, 'clean_model_and_rng_restored': True, 'optimizer_discarded': True}


def train(core, args, model, caches, device, c, binding, encoder_hash, name, expected_orders=None):
    root = args.output / name
    root.mkdir()
    cfg = {'variant': name, 'architecture': architecture(model.pooling), 'protocol': c,
           'binding': binding, 'labels': list(core.LABELS), 'frozen_encoder_sha256': encoder_hash}
    write(root / 'config.json', cfg)
    initial, _ = evaluate(core, model, caches['val'], device)
    write(root / 'epoch_zero.json', initial)
    core.seed_all(42)
    start_rng = pair.rng_hash(core, pair.rng_state(core))
    opt = optimizer(core, model, c)
    sampler = core.TrainBatches(len(caches['train']['ids']), c['batch_size'], c['seed'])
    payload = {'model_state': model.state_dict(), 'optimizer_state': opt.state_dict(), 'epoch': 0,
               'validation': initial, 'config': cfg, 'rng': pair.rng_state(core),
               'sampler_rng': sampler.generator.get_state()}
    core.checkpoint(root / 'epoch_zero.pt', payload)
    core.checkpoint(root / 'best.pt', payload)
    best, selected, stale = initial['macro_f1'], 0, 0
    orders, history, updates = [], [], 0
    started = time.perf_counter()
    with (root / 'epochs.jsonl').open('x', encoding='utf-8') as log:
        for epoch in range(1, c['epochs'] + 1):
            model.train()
            count, steps, total = 0, 0, 0.
            for numbers in sampler:
                if steps == 0 and expected_orders is not None and epoch <= len(expected_orders):
                    require(sampler.last_order_sha256 == expected_orders[epoch - 1], 'Paired sample order differs')
                features, labels = minibatch(core, caches['train'], numbers, device)
                opt.zero_grad(set_to_none=True)
                scores, _ = model.forward_features(*features)
                loss = core.F.nll_loss(scores, labels)
                require(bool(core.torch.isfinite(loss)), 'Nonfinite training loss')
                loss.backward()
                require(all(p.grad is None or bool(core.torch.isfinite(p.grad).all()) for p in model.parameters()),
                        'Nonfinite training gradient')
                opt.step()
                total += float(loss.detach()) * len(numbers)
                count += len(numbers)
                steps += 1
            require(count == len(caches['train']['ids']) == 107621 and steps == len(sampler) == 211,
                    'Incomplete training epoch')
            dim.assert_frozen(core, model, encoder_hash)
            metrics, _ = evaluate(core, model, caches['val'], device)
            best, stale, improved = pair.selection_state(metrics['macro_f1'], best, stale)
            if improved:
                selected = epoch
            updates += steps
            orders.append(sampler.last_order_sha256)
            payload = {'model_state': model.state_dict(), 'optimizer_state': opt.state_dict(), 'epoch': epoch,
                'validation': metrics, 'config': cfg, 'rng': pair.rng_state(core),
                'sampler_rng': sampler.generator.get_state()}
            core.checkpoint(root / 'last.pt', payload)
            if improved:
                core.checkpoint(root / 'best.pt', payload)
            row = {'epoch': epoch, 'train_nll': total / count, 'val_macro_f1': metrics['macro_f1'],
                'train_order_sha256': orders[-1], 'optimizer_updates': steps,
                'best_epoch': selected, 'stale_epochs': stale}
            history.append(row)
            log.write(json.dumps(row) + '\n')
            log.flush()
            dim.progress(args, 'training', variant=name, **row)
            if stale >= c['patience']:
                break
    core.sync(device)
    elapsed = time.perf_counter() - started
    saved = core.torch.load(root / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(saved['model_state'], strict=True)
    measured, exports = evaluate(core, model, caches['val'], device, export=True)
    require(saved['epoch'] == selected and measured['confusion_matrix'] == saved['validation']['confusion_matrix'],
            'Selected cached checkpoint differs')
    pair.save_predictions(root / 'validation_predictions.jsonl.gz', exports)
    result = {'architecture': cfg['architecture'], 'epoch_zero': initial, 'best_validation': measured,
        'last_validation': metrics, 'best_epoch': selected, 'epochs': epoch, 'optimizer_updates': updates,
        'orders': orders, 'epoch_zero_eligible': True, 'selected_epoch_zero': selected == 0,
        'training_start_rng_sha256': start_rng, 'training_wall_seconds': elapsed,
        'training_evidence': dim.trained_evidence(initial['macro_f1'], history),
        'parameter_groups': [{'name': g['name'], 'lr': g['lr'], 'parameters': sum(p.numel() for p in g['params'])}
                             for g in opt.param_groups],
        'checkpoint_sha256': {n: sha(root / n) for n in ('best.pt', 'last.pt')},
        'predictions_sha256': sha(root / 'validation_predictions.jsonl.gz')}
    check_history(history, result, c)
    write(root / 'metrics.json', result)
    return result, exports


def check_history(history, result, c):
    require(bool(history) and len(history) == result['epochs'] <= c['epochs']
            and result['epoch_zero_eligible'] is True, 'Incomplete epoch history')
    best, selected, stale = result['epoch_zero']['macro_f1'], 0, 0
    for epoch, row in enumerate(history, 1):
        require(row['epoch'] == epoch and math.isfinite(row['val_macro_f1'])
                and 0 <= row['val_macro_f1'] <= 1 and row['optimizer_updates'] == 211
                and row['train_order_sha256'] == result['orders'][epoch - 1], 'Invalid epoch ledger')
        best, stale, improved = pair.selection_state(row['val_macro_f1'], best, stale)
        if improved:
            selected = epoch
        require(row['best_epoch'] == selected and row['stale_epochs'] == stale
                and (epoch == len(history) or stale < c['patience']), 'Selection ledger differs')
    require(selected == result['best_epoch'] and result['selected_epoch_zero'] == (selected == 0)
            and abs(best - result['best_validation']['macro_f1']) < 1e-12
            and abs(history[-1]['val_macro_f1'] - result['last_validation']['macro_f1']) < 1e-12
            and len(result['orders']) == len(history) and result['optimizer_updates'] == len(history) * 211
            and (stale >= c['patience'] or len(history) == c['epochs']), 'Selection/budget mismatch')


def graph_replay(core, args, model, data, cache, device, c, expected, encoder_hash, original=None):
    """Compare every raw validation graph with its cached-node output."""
    model.eval()
    if original is not None:
        original.eval()
    offset, error, source_error = 0, 0., 0.
    cm, exports = [[0] * 8 for _ in range(8)], []
    started = time.perf_counter()
    with core.torch.no_grad():
        for step, batch in enumerate(loader(core, data, 'val', c), 1):
            batch = batch.to(device)
            raw = model(batch)
            size = batch.num_graphs
            numbers = list(range(offset, offset + size))
            features, labels = minibatch(core, cache, numbers, device)
            require(list(batch.sample_id) == cache['ids'][offset:offset + size]
                    and batch.y.view(-1).tolist() == labels.tolist(), 'Raw replay order/labels differ')
            cached, _ = model.forward_features(*features)
            require(core.torch.allclose(raw, cached, atol=1e-5, rtol=1e-5)
                    and bool((raw.argmax(-1) == cached.argmax(-1)).all()), 'Raw/cache output differs')
            error = max(error, float((raw - cached).abs().max()))
            if original is not None:
                old = original(batch)
                require(core.torch.allclose(raw, old, atol=1e-6, rtol=1e-6)
                        and bool((raw.argmax(-1) == old.argmax(-1)).all()), 'Original D32 raw output differs')
                source_error = max(source_error, float((raw - old).abs().max()))
            for sid, y, p in zip(batch.sample_id, labels.tolist(), raw.argmax(-1).tolist()):
                cm[y][p] += 1
                exports.append({'sample_id': sid, 'label': y, 'prediction': p})
            offset += size
            if step == 1 or step % 200 == 0:
                dim.progress(args, 'raw_validation_replay', pooling=model.pooling, samples=offset)
    require(offset == len(cache['ids']) == 13446 and cm == expected['confusion_matrix'], 'Raw replay score differs')
    dim.assert_frozen(core, model, encoder_hash)
    return {'samples': offset, 'outputs_allclose': True, 'max_abs_raw_cached_error': error,
        'predictions_identical': True, 'confusion_matrix_reproduced': True,
        'strict_state_load': True, 'max_abs_original_D32_error': source_error if original is not None else None,
        'encoder_parameters_and_BN_unchanged': True, 'wall_seconds': time.perf_counter() - started}, exports


def run(args, meta):
    started = time.perf_counter()
    c, previous, root, source, binding = meta
    _, spec, ref, nroot, _, _ = previous[1]
    gpu = checked_gpu(args.physical_gpu, args.allow_shared_gpu)
    activate_gpu(gpu)
    # The archived entry uses the server checkout only for frozen core helpers.
    scripts = str(args.repo / 'scripts')
    sys.path.insert(0, scripts)
    try:
        import train_grouped as core
    finally:
        sys.path.remove(scripts)
    device = runtime(core)
    write(args.output / 'gpu.json', gpu)
    data = core.FrozenGraphs(inside(args.repo, spec['split_directory']), ref['binding']['split_id'])
    load_counts = guard_graph_loads(data, {'train', 'val'})
    n0 = core.torch.load(nroot / 'N0/best.pt', map_location=device, weights_only=False)
    mature = core.torch.load(root / 'D32-Gate/best.pt', map_location=device, weights_only=False)
    expected = source['models']['D32-Gate']['best_validation']
    require(mature['epoch'] == c['source_epoch'] and mature['validation']['confusion_matrix'] == expected['confusion_matrix']
            and mature['config']['binding'] == source['binding'] and mature['config']['variant'] == 'D32-Gate'
            and mature['config']['labels'] == list(core.LABELS) and mature['config']['protocol'] == previous[0]
            and mature['config']['architecture'] == source_pair.architecture('dynamic'),
            'Mature checkpoint payload differs')
    classes = load_classes()
    models = {name: build(core, classes, data, device, n0, mature['model_state'], mode)
              for name, mode in VARIANTS.items()}
    encoder_hash = core.state_hash(dim.encoder_values(models['P0']))
    require(encoder_hash == previous[3]['binding']['encoder_source']['encoder_sha256'], 'Source encoders changed')
    for model in models.values():
        dim.assert_frozen(core, model, encoder_hash)
        optimizer(core, model, c)
    caches, receipts = {}, {}
    for part in ('train', 'val'):
        caches[part], receipts[part] = extract(core, args, models['P0'], data, device, part, c, encoder_hash, ref['binding'])
        validate_cache(core, caches[part], core.select_rows(data, part), part, encoder_hash, ref['binding'])
    write(args.output / 'cache_receipts.json', receipts)
    write(args.output / 'packet_diagnostics.json', {p: diagnostics(core, x) for p, x in caches.items()})
    gold = dim.predictions(root / 'D32-Gate/validation_predictions.jsonl.gz')
    initialization = initial_equivalence(core, models, caches['val'], device, gold, expected)
    # Use the unchanged historical class as an independent raw-graph reference.
    _, Original = dim.load_classes()
    original = Original(models['P0'].flow_encoder, models['P0'].packet_encoder, models['P0'].fusion).eval()
    initial_replays = {}
    for name, model in models.items():
        initial_replays[name], _ = graph_replay(core, args, model, data, caches['val'], device, c,
                                               expected, encoder_hash, original)
    del original
    initialization['raw_graph_replays'] = initial_replays
    write(args.output / 'initialization.json', initialization)
    probes = {}
    for name, model in models.items():
        dim.progress(args, 'raw_training_gradient_probe', variant=name)
        probes[name] = probe(core, args, model, data, caches['train'], device, c, encoder_hash)
        path = args.output / (name + '_preflight.pt')
        core.checkpoint(path, {'model_state': model.state_dict(), 'architecture': architecture(model.pooling)})
        fresh = build(core, classes, data, device, n0, mature['model_state'], model.pooling)
        fresh.load_state_dict(core.torch.load(path, map_location=device, weights_only=False)['model_state'], strict=True)
        features, _ = minibatch(core, caches['val'], list(range(16)), device)
        with core.torch.no_grad():
            require(core.torch.allclose(model.forward_features(*features)[0], fresh.forward_features(*features)[0],
                                       atol=1e-6, rtol=1e-6), 'Preflight checkpoint roundtrip failed')
        dim.assert_frozen(core, fresh, encoder_hash)
        del fresh
    write(args.output / 'probes.json', probes)
    source_pair.assert_source_unchanged(binding)
    preflight_seconds = time.perf_counter() - started
    write(args.output / 'status.json', {'status': 'running', 'phase': 'training', 'tensor_preflight_passed': True})
    results, exports, replays = {}, {}, {}
    for name, model in models.items():
        orders = results['P0']['orders'] if name == 'P1' else None
        results[name], exports[name] = train(core, args, model, caches, device, c, binding, encoder_hash, name, orders)
        saved = core.torch.load(args.output / name / 'best.pt', map_location=device, weights_only=False)
        require(saved['config']['binding'] == binding and saved['config']['variant'] == name
                and saved['config']['architecture'] == architecture(model.pooling), 'Best checkpoint identity differs')
        replay = build(core, classes, data, device, n0, mature['model_state'], model.pooling)
        replay.load_state_dict(saved['model_state'], strict=True)
        replays[name], raw = graph_replay(core, args, replay, data, caches['val'], device, c,
            saved['validation'], encoder_hash)
        require([r['prediction'] for r in raw] == [r['prediction'] for r in exports[name]], 'Best raw predictions differ')
        write(args.output / name / 'graph_replay.json', replays[name])
        del replay
    common = min(len(m['orders']) for m in results.values())
    require(results['P0']['orders'][:common] == results['P1']['orders'][:common]
            and results['P0']['training_start_rng_sha256'] == results['P1']['training_start_rng_sha256'], 'Pair RNG differs')
    for part, cache in caches.items():
        require(tensor_hash(core, cache) == receipts[part]['tensor_sha256']
                and sha(args.output / (part + '_packet_nodes.pt')) == receipts[part]['sha256'], 'Node cache changed')
    for model in models.values():
        dim.assert_frozen(core, model, encoder_hash)
    source_pair.assert_source_unchanged(binding)
    from summarize_joint_fusion32 import changed_samples
    from summarize_fusion_gate_pair import prediction_transitions
    from summarize_packet_pool_pair import summarize
    comparisons = {'P1_vs_P0': (exports['P0'], exports['P1']),
                   'P0_vs_D32': (gold, exports['P0']), 'P1_vs_D32': (gold, exports['P1'])}
    changes, transitions = {}, {}
    for name, (a, b) in comparisons.items():
        changes[name] = pair.compare_predictions(a, b)
        transitions[name] = prediction_transitions(a, b, core.LABELS)
        pair.save_predictions(args.output / (name + '_changed_samples.jsonl.gz'), changed_samples(a, b, core.LABELS))
    summary = {'status': 'completed', 'exit_code': 0, 'protocol': c, 'binding': binding,
        'source_model': source['models']['D32-Gate'], 'models': results, 'initialization': initialization,
        'probes': probes, 'packet_diagnostics': read(args.output / 'packet_diagnostics.json'),
        'comparisons': changes, 'prediction_transitions': transitions, 'graph_replays': replays,
        'common_epochs_order_checked': common, 'encoder_parameters_and_BN_unchanged': True,
        'node_caches_unchanged': True, 'source_artifacts_unchanged': True, 'tensor_preflight_passed': True,
        'preflight_wall_seconds': preflight_seconds, 'cache_receipts': receipts, 'graph_load_counts': load_counts,
        'discarded_probe_updates': 6, 'training_jobs': 2, 'test_evaluated': False, 'automatic_followup': False}
    write(args.output / 'summary.json', summary)
    print(summarize(args.output), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path.cwd())
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=('check', 'run'), default='check')
    p.add_argument('--physical-gpu', type=int, default=1)
    p.add_argument('--allow-shared-gpu', action='store_true')
    args = p.parse_args()
    args.repo, args.config = args.repo.resolve(), args.config.resolve()
    args.output = inside(args.repo / 'outputs', inside(args.repo, args.output))
    meta = metadata(args)
    require(not args.output.exists() and args.output != meta[2] and meta[2] not in args.output.parents,
            'Refuse overwrite/resume/source output reuse')
    if args.mode == 'check':
        print(json.dumps({'status': 'metadata_checks_passed', 'binding': meta[-1],
                          'training_started': False, 'tensor_checks_executed': False}), flush=True)
        return
    require(os.name == 'posix', 'Formal training runs on the Linux wx server only')
    args.output.mkdir(parents=True)
    write(args.output / 'config.json', meta[0])
    write(args.output / 'binding.json', meta[-1])
    write(args.output / 'status.json', {'status': 'running', 'phase': 'preflight'})
    try:
        run(args, meta)
        write(args.output / 'status.json', {'status': 'completed', 'exit_code': 0})
        write(args.output / 'progress.json', {'status': 'completed', 'training_jobs': 2})
    except BaseException:
        write(args.output / 'status.json', {'status': 'failed', 'exit_code': 1})
        write(args.output / 'progress.json', {'status': 'failed', 'exit_code': 1})
        (args.output / 'error.log').write_text(traceback.format_exc(), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
