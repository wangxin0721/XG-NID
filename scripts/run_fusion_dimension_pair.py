"""Two fresh 8/32 fusion heads on the same frozen N0-seed42 encoders.

Metadata mode has no torch imports. Formal training is Linux/server-only.
No old head weights, old 8D caches, test graphs, retries or automatic next run.
"""
from __future__ import annotations

import argparse
import importlib.util
from itertools import islice
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
import run_e2_concat_pair as pair
import run_frozen_fusion as frozen
from run_fusion_seed_repeats import predictions, verify_metrics
from run_joint_gate import trained_evidence

HERE = Path(__file__).resolve().parent
MODULE = HERE.parent / 'src/xgnid/fusion_dimension.py'
VARIANTS = {'D8': 8, 'D32': 32}
COMPONENTS = ('flow_projection', 'packet_projection', 'gate_hidden', 'gate_output', 'head_hidden', 'head_output')


def check_config(c):
    expected = {'schema': 'xgnid.fusion-dimension-pair.v1', 'variants': VARIANTS,
        'seed': 42, 'encoder_hidden_dim': 64, 'encoder_mode': 'frozen_eval',
        'head_initialization': 'fresh_default_Linear_named_seed42_streams_zero_gate_output',
        'gate_hidden_dim': 16, 'gate_input': 'two_projected_vectors_entropy_concentration_margins_cosine_2d_plus_5',
        'head_hidden_dim': 32, 'classes': 8, 'cache_dim': 64, 'cache_batch_size': 16,
        'eval_batch_size': 16, 'batch_size': 512, 'workers': 2, 'optimizer': 'fresh_Adam',
        'lr': .001, 'weight_decay': 0., 'epochs': 100, 'patience': 20,
        'include_epoch_zero': False, 'probe_batches': 3, 'loss': 'plain_final_NLL',
        'statistics_residual': False, 'primary_comparison': 'D32_minus_D8',
        'class_tradeoff_review_required': True, 'test_evaluated': False,
        'automatic_followup': False, 'training_jobs': 2}
    require(all(c.get(k) == v for k, v in expected.items()), 'Unregistered dimension-pair protocol')


def initialization_streams(seed):
    require(seed == 42, 'This screen uses only seed42')
    return {name: int(identity({'seed': seed, 'component': name})[:15], 16) for name in COMPONENTS}


def architecture(width):
    require(width in (8, 32), 'Unregistered fusion dimension')
    return {'width': width, 'projection': [64, width], 'gate': [2 * width + 5, 16, width],
            'shared_head': [width, 32, 8], 'trainable_parameters': 211 * width + 392}


def metadata(args):
    c = read(args.config)
    check_config(c)
    # Only reuse reference validation; none of the old frozen-head training or
    # projected 8D caches is used in this experiment.
    source = SimpleNamespace(repo=args.repo, config=inside(args.config.parent, c['reference_protocol']))
    source_spec, ref, nroot, parent_binding = frozen.metadata(source)
    require(c['n0_best_sha256'] == ref['model']['checkpoint_sha256']['best.pt'], 'Wrong encoder checkpoint')
    verify_metrics(ref['model']['best_validation'])
    h = c['historical_J2']
    root = inside(args.repo / 'outputs', inside(args.repo, h['directory']))
    require(read(root / 'status.json') == {'status': 'completed', 'exit_code': 0}, 'Historical J2 incomplete')
    summary = read(root / 'summary.json')
    m = summary['joint']['J2']
    require(summary['status'] == 'completed' and summary['test_evaluated'] is False
            and summary['N0'] == ref['model'] and read(root / 'J2/metrics.json') == m,
            'Historical J2 identity differs')
    require(sha(inside(args.config.parent, h['config'])) == h['config_sha256']
            == summary['binding']['joint_config_sha256']
            and sha(HERE / 'run_joint_gate.py') == summary['binding']['joint_source_sha256']
            and sha(root / 'J2/best.pt') == h['best_sha256'] == m['checkpoint_sha256']
            and abs(m['best_validation']['macro_f1'] - h['macro_f1']) < 1e-12,
            'Historical J2 config/checkpoint/score changed')
    verify_metrics(m['best_validation'])
    binding = {'config_sha256': sha(args.config), 'N0_reference_validation': parent_binding,
        'historical_J2_summary_sha256': sha(root / 'summary.json'),
        'historical_J2_best_sha256': h['best_sha256'], 'initialization_streams': initialization_streams(c['seed']),
        'sources': {name: sha(path) for name, path in {
            'scripts/run_fusion_dimension_pair.py': Path(__file__), 'src/xgnid/fusion_dimension.py': MODULE,
            'scripts/summarize_fusion_dimension_pair.py': HERE / 'summarize_fusion_dimension_pair.py'}.items()}}
    return c, source_spec, ref, nroot, m, binding


def progress(args, phase, **details):
    value = {'status': 'running', 'phase': phase, 'pid': os.getpid(), **details}
    write(args.output / 'progress.json', value)
    print(json.dumps(value), flush=True)


def load_classes():
    name = 'xgnid.fusion_dimension'
    spec = importlib.util.spec_from_file_location(name, MODULE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.FusionHead, module.FrozenEncoderClassifier


def build_backbone(core, data, device, saved):
    core.seed_all(42)
    warm = core.Batch.from_data_list([data.load(r) for r in data.training_candidates()[:16]]).to(device)
    model = core.build_experiment('E2', warm).to(device)
    core.initialize(model, 'E2', warm)
    model.load_state_dict(saved['model_state'], strict=True)
    model.requires_grad_(False)
    return model.eval()


def encoder_values(graph):
    return [(n, t) for n, t in graph.state_dict().items() if n.startswith(('flow_encoder.', 'packet_encoder.'))]


def assert_frozen(core, graph, expected):
    for encoder in (graph.flow_encoder, graph.packet_encoder):
        require(not any(m.training for m in encoder.modules())
                and all(not p.requires_grad and p.grad is None for p in encoder.parameters()),
                'Frozen encoder mode/gradient invariant broken')
    require(core.state_hash(encoder_values(graph)) == expected, 'Frozen encoder parameters/BN buffers changed')


def cache_header(part, source, rows):
    require(part in ('train', 'val'), 'Test representations are forbidden')
    return {'schema': 'xgnid.encoder64-cache.v1', 'partition': part, 'embedding_dim': 64,
            'source': source, 'samples': len(rows), 'sample_order_sha256': identity([r['sample_id'] for r in rows])}


def validate_cache_header(header, part, source, rows):
    require(header == cache_header(part, source, rows), 'Cache identity/dimension/order differs')


def extract_cache(core, args, model, data, device, part, source, gold=None, expected_n0=None):
    rows = core.select_rows(data, part)
    loader = core.DataLoader(core.GraphRows(data, rows), batch_size=16, shuffle=False, num_workers=2,
                           pin_memory=True, persistent_workers=True, multiprocessing_context='fork')
    pieces = {k: [] for k in ('flow64', 'packet64', 'labels')}
    ids, exports = [], []
    before = core.state_hash(model.state_dict().items())
    started = time.perf_counter()
    max_error = 0.
    n0_replay = None
    with core.torch.no_grad():
        for step, batch in enumerate(loader, 1):
            batch = batch.to(device)
            hf, hp, zf, zp = model._branch_outputs(batch)
            require(hf.shape == hp.shape == (batch.num_graphs, 64)
                    and bool(core.torch.isfinite(hf).all()) and bool(core.torch.isfinite(hp).all()), 'Bad 64D encoder output')
            if part == 'val':
                scores = core.F.log_softmax(model.shared_head(.5 * zf + .5 * zp), dim=-1)
                if step == 1:
                    original = model(batch)
                    require(core.torch.allclose(original, scores, atol=1e-5, rtol=1e-5), 'N0 definition differs')
                    max_error = float((original - scores).abs().max())
                exports += [{'sample_id': s, 'label': y, 'prediction': p} for s, y, p in
                            zip(batch.sample_id, batch.y.view(-1).tolist(), scores.argmax(-1).tolist())]
            pieces['flow64'].append(hf.cpu())
            pieces['packet64'].append(hp.cpu())
            pieces['labels'].append(batch.y.view(-1).cpu())
            ids.extend(batch.sample_id)
            if step == 1 or step % 25 == 0 or len(ids) == len(rows):
                elapsed = time.perf_counter() - started
                progress(args, 'cache_' + part + '_64', samples=len(ids), total=len(rows), batches=step,
                         elapsed_seconds=elapsed, estimated_remaining_seconds=elapsed * (len(rows) - len(ids)) / len(ids))
    cache = {k: core.torch.cat(v) for k, v in pieces.items()}
    cache.update(ids=ids, metadata=cache_header(part, source, rows))
    require(ids == [r['sample_id'] for r in rows] and cache['labels'].tolist() == [r['label'] for r in rows],
            'Cache samples/labels differ from frozen split')
    if part == 'val':
        frozen.verify_order(gold, rows)
        require([r['prediction'] for r in exports] == [r['prediction'] for r in gold], 'N0 validation replay failed')
        cm = [[0] * 8 for _ in range(8)]
        for row in exports:
            cm[row['label']][row['prediction']] += 1
        n0_replay = frozen.matrix_report(cm, core.LABELS)
        require(cm == expected_n0['confusion_matrix']
                and abs(n0_replay['macro_f1'] - expected_n0['macro_f1']) < 1e-12,
                'N0 validation confusion matrix/Macro-F1 differs')
    require(core.state_hash(model.state_dict().items()) == before, 'Caching changed frozen N0 state')
    path = args.output / (part + '_encoder64.pt')
    core.checkpoint(path, cache)
    return cache, {'metadata': cache['metadata'], 'sha256': sha(path),
                   'wall_seconds': time.perf_counter() - started,
                   'N0_first_batch_max_error': max_error if part == 'val' else None,
                   'N0_replayed_validation': n0_replay}


def make_head(core, cls, width, device, streams):
    rng = pair.rng_state(core)
    head = cls(width, streams).to(device=device, dtype=core.torch.float32)
    require(pair.rng_hash(core, rng) == pair.rng_hash(core, pair.rng_state(core)), 'Head construction leaked RNG')
    require(sum(p.numel() for p in head.parameters()) == architecture(width)['trainable_parameters'], 'Wrong head capacity')
    return head


def optimizer(core, graph, c):
    enc = {id(p) for e in (graph.flow_encoder, graph.packet_encoder) for p in e.parameters()}
    params = list(graph.fusion.parameters())
    ids = {id(p) for p in params}
    require(len(ids) == len(params) and not ids.intersection(enc)
            and ids == {id(p) for p in graph.parameters() if p.requires_grad}, 'Optimizer omitted/duplicated/leaked parameters')
    return core.torch.optim.Adam(params, lr=c['lr'], weight_decay=c['weight_decay'])


def evaluate(core, head, cache):
    head.eval()
    cm = [[0] * 8 for _ in range(8)]
    exports, weights = [], []
    with core.torch.no_grad():
        for offset in range(0, len(cache['ids']), 16):
            ix = slice(offset, offset + 16)
            scores, alpha = head(cache['flow64'][ix], cache['packet64'][ix])
            require(scores.shape == (len(cache['ids'][ix]), 8) and alpha.shape == (len(cache['ids'][ix]), head.width)
                    and bool(core.torch.isfinite(scores).all())
                    and bool(core.torch.isfinite(alpha).all()), 'Invalid cached prediction')
            for s, y, p, a in zip(cache['ids'][ix], cache['labels'][ix].tolist(), scores.argmax(-1).tolist(), alpha.tolist()):
                cm[y][p] += 1
                exports.append({'sample_id': s, 'label': y, 'prediction': p,
                                'flow_weights': a, 'packet_weights': [1 - v for v in a]})
            weights.append(alpha)
    a = core.torch.cat(weights)
    metrics = frozen.matrix_report(cm, core.LABELS)
    metrics['flow_weight_distribution'] = {'mean': a.mean(0).tolist(), 'std': a.std(0, unbiased=False).tolist(),
                                         'min': a.min(0).values.tolist(), 'max': a.max(0).values.tolist(),
                                         'max_abs_deviation_from_half': float((a - .5).abs().max())}
    return metrics, exports


def validate_equal_start(distribution):
    # A parallel mean reduction need not return exactly 0.5, even for a tensor
    # filled with exactly 0.5. Validate the elementwise maximum error instead.
    error = distribution['max_abs_deviation_from_half']
    require(math.isfinite(error) and 0 <= error <= 1e-7,
            'Fresh epoch0 gate weights differ from equal fusion: max_abs_error=' + str(error))
    return {'elementwise_atol': 1e-7, 'max_abs_error': error,
            'observed_mean': distribution['mean'], 'observed_min': distribution['min'],
            'observed_max': distribution['max'], 'legacy_aggregate_exact_check_would_pass':
                all(v == .5 for key in ('mean', 'min', 'max') for v in distribution[key])}


def assert_zero_gate_output(core, head):
    require(bool((head.gate[-1].weight == 0).all()) and bool((head.gate[-1].bias == 0).all()),
            'Fresh gate output parameters are not zero')


def probe(core, graph, data, cache, device, c, encoder_hash):
    started = time.perf_counter()
    clean = {n: t.detach().clone() for n, t in graph.state_dict().items()}
    digest = core.state_hash(clean.items())
    rng = pair.rng_state(core)
    records = []
    try:
        rows = core.select_rows(data, 'train')
        indices = list(islice(core.TrainBatches(len(rows), 512, 42), 3))
        loader = core.DataLoader(core.GraphRows(data, rows), batch_sampler=indices, num_workers=0)
        opt = optimizer(core, graph, c)
        for step, (batch, numbers) in enumerate(zip(loader, indices), 1):
            index = core.torch.tensor(numbers, device=device)
            batch = batch.to(device)
            require(list(batch.sample_id) == [cache['ids'][i] for i in numbers], 'Probe data order differs')
            graph.train()
            require(graph.fusion.training, 'Fresh heads must train')
            with core.torch.no_grad():
                cached, alpha = graph.fusion(cache['flow64'][index], cache['packet64'][index])
                if step == 1:
                    require(bool((alpha == .5).all()), 'Fresh gate did not initialize to equal weights')
            before = {n: p.detach().clone() for n, p in graph.fusion.named_parameters()}
            opt.zero_grad(set_to_none=True)
            scores = graph(batch)
            require(core.torch.allclose(scores, cached, atol=1e-5, rtol=1e-5), 'Raw graph/cache predictions differ')
            loss = core.F.nll_loss(scores, batch.y.view(-1))
            require(bool(core.torch.isfinite(loss)), 'Nonfinite probe loss')
            loss.backward()
            require(all(p.grad is None or bool(core.torch.isfinite(p.grad).all()) for p in graph.parameters()),
                    'Nonfinite probe gradient')
            norms = {n: float(p.grad.abs().sum()) if p.grad is not None else 0. for n, p in graph.fusion.named_parameters()}
            opt.step()
            changed = {n: not core.torch.equal(before[n], p) for n, p in graph.fusion.named_parameters()}
            for prefix in ('flow_projection.', 'packet_projection.', 'shared_head.', 'gate.2.'):
                require(any(v > 0 for n, v in norms.items() if n.startswith(prefix))
                        and any(v for n, v in changed.items() if n.startswith(prefix)), 'No gradient/update for ' + prefix)
            assert_frozen(core, graph, encoder_hash)
            records.append({'step': step, 'loss': float(loss.detach()), 'gradients_l1': norms,
                            'parameters_changed': changed, 'raw_cache_max_error': float((scores - cached).abs().max())})
        require(len(records) == 3 and any(r['gradients_l1']['gate.0.weight'] > 0
                and r['parameters_changed']['gate.0.weight'] for r in records[1:]), 'Gate hidden layer never trained')
    finally:
        graph.load_state_dict(clean, strict=True)
        graph.zero_grad(set_to_none=True)
        graph.eval()
        pair.restore_rng(core, rng)
        require(core.state_hash(graph.state_dict().items()) == digest
                and pair.rng_hash(core, rng) == pair.rng_hash(core, pair.rng_state(core)), 'Probe leaked state/RNG')
        assert_frozen(core, graph, encoder_hash)
    return {'updates': records, 'discarded': True, 'clean_state_and_rng_restored': True,
            'encoder_parameters_and_BN_unchanged': True, 'wall_seconds': time.perf_counter() - started,
            'initial_head_sha256': core.state_hash(graph.fusion.state_dict().items())}


def check_history(history, metrics, c):
    require(c['include_epoch_zero'] is False, 'This pair selects trained epochs only')
    require([r['epoch'] for r in history] == list(range(1, metrics['epochs'] + 1))
            and 1 <= len(history) <= c['epochs'] and all(math.isfinite(r['val_macro_f1'])
            and 0 <= r['val_macro_f1'] <= 1 for r in history), 'Incomplete/invalid training history')
    best, selected, stale = -math.inf, None, 0
    for row in history:
        require(row['optimizer_updates'] == 211, 'Incomplete epoch')
        best, stale, improved = pair.selection_state(row['val_macro_f1'], best, stale)
        if improved:
            selected = row['epoch']
        require(row['best_epoch'] == selected and row['stale_epochs'] == stale
                and (row['epoch'] == len(history) or stale < c['patience']), 'Selection history differs')
    require(selected == metrics['best_epoch'] and selected >= 1 and metrics['selected_epoch_zero'] is False
            and abs(best - metrics['best_validation']['macro_f1']) < 1e-12
            and metrics['optimizer_updates'] == len(history) * 211
            and (stale >= c['patience'] or len(history) == c['epochs']), 'Selection/budget mismatch')


def train(core, args, c, graph, caches, variant, binding, encoder_hash, expected_orders=None):
    folder = args.output / variant
    folder.mkdir()
    assert_zero_gate_output(core, graph.fusion)
    initial, _ = evaluate(core, graph.fusion, caches['val'])
    receipt = validate_equal_start(initial['flow_weight_distribution'])
    write(folder / 'equal_weight_start.json', receipt)
    print(json.dumps({'phase': 'elementwise_equal_gate_check', 'variant': variant, **receipt}), flush=True)
    cfg = {'variant': variant, 'architecture': architecture(VARIANTS[variant]), 'protocol': c,
           'binding': binding, 'labels': list(core.LABELS), 'optimizer_state': 'reset',
           'frozen_encoder_sha256': encoder_hash, 'heads_loaded_from_N0_or_J2': False}
    write(folder / 'config.json', cfg)
    write(folder / 'epoch_zero.json', initial)
    write(folder / 'probe.json', args.probes[variant])
    core.seed_all(42)
    start_rng = pair.rng_hash(core, pair.rng_state(core))
    opt = optimizer(core, graph, c)
    sampler = core.TrainBatches(len(caches['train']['ids']), 512, 42)
    best, best_epoch, stale = -math.inf, None, 0
    orders, history = [], []
    payload = {'model_state': graph.state_dict(), 'epoch': 0, 'validation': initial, 'config': cfg}
    # Epoch0 is recorded for this pair, but never eligible for selection.
    core.checkpoint(folder / 'epoch_zero.pt', payload)
    started = time.perf_counter()
    with (folder / 'epochs.jsonl').open('x', encoding='utf-8') as log:
        for epoch in range(1, c['epochs'] + 1):
            graph.train()
            steps, count, total_loss = 0, 0, 0.
            for numbers in sampler:
                if steps == 0 and expected_orders is not None and epoch <= len(expected_orders):
                    require(sampler.last_order_sha256 == expected_orders[epoch - 1], 'Paired data order differs')
                index = core.torch.tensor(numbers, device=caches['train']['flow64'].device)
                opt.zero_grad(set_to_none=True)
                scores, _ = graph.fusion(caches['train']['flow64'][index], caches['train']['packet64'][index])
                loss = core.F.nll_loss(scores, caches['train']['labels'][index])
                require(bool(core.torch.isfinite(loss)), 'Nonfinite train loss')
                loss.backward()
                require(all(p.grad is None or bool(core.torch.isfinite(p.grad).all()) for p in graph.parameters()),
                        'Nonfinite train gradients')
                opt.step()
                count += len(numbers)
                steps += 1
                total_loss += float(loss.detach()) * len(numbers)
            require(count == 107621 and steps == 211, 'Incomplete epoch')
            assert_frozen(core, graph, encoder_hash)
            metrics, _ = evaluate(core, graph.fusion, caches['val'])
            best, stale, improved = pair.selection_state(metrics['macro_f1'], best, stale)
            if improved:
                best_epoch = epoch
            orders.append(sampler.last_order_sha256)
            payload = {'model_state': graph.state_dict(), 'optimizer_state': opt.state_dict(),
                'epoch': epoch, 'validation': metrics, 'config': cfg,
                'rng': pair.rng_state(core), 'sampler_rng': sampler.generator.get_state()}
            core.checkpoint(folder / 'last.pt', payload)
            if improved:
                core.checkpoint(folder / 'best.pt', payload)
            row = {'epoch': epoch, 'train_nll': total_loss / count, 'val_macro_f1': metrics['macro_f1'],
                   'optimizer_updates': steps, 'train_order_sha256': orders[-1], 'best_epoch': best_epoch,
                   'stale_epochs': stale}
            history.append(row)
            log.write(json.dumps(row) + '\n')
            log.flush()
            progress(args, 'training', variant=variant, **row)
            if stale >= c['patience']:
                break
    core.sync(caches['val']['flow64'].device)
    elapsed = time.perf_counter() - started
    selected = core.torch.load(folder / 'best.pt', map_location=caches['val']['flow64'].device, weights_only=False)
    graph.load_state_dict(selected['model_state'], strict=True)
    measured, exports = evaluate(core, graph.fusion, caches['val'])
    require(measured['confusion_matrix'] == selected['validation']['confusion_matrix'], 'Best cached validation replay failed')
    verify_metrics(measured)
    pair.save_predictions(folder / 'validation_predictions.jsonl.gz', exports)
    result = {'variant': variant, 'architecture': cfg['architecture'], 'epoch_zero': initial,
        'best_validation': measured, 'last_validation': metrics, 'best_epoch': best_epoch, 'epochs': epoch,
        'selected_epoch_zero': best_epoch == 0, 'optimizer_updates': epoch * 211, 'orders': orders,
        'epoch_zero_eligible': False,
        'training_start_rng_sha256': start_rng, 'training_wall_seconds': elapsed,
        'training_evidence': trained_evidence(initial['macro_f1'], history),
        'checkpoint_sha256': {name: sha(folder / name) for name in ('best.pt', 'last.pt')},
        'predictions_sha256': sha(folder / 'validation_predictions.jsonl.gz'), 'test_evaluated': False}
    check_history(history, result, c)
    write(folder / 'metrics.json', result)
    return result, exports


def graph_replay(core, args, cls, data, device, saved_n0, saved, cache, streams, encoder_hash):
    started = time.perf_counter()
    Head, Graph = cls
    width = saved['config']['architecture']['width']
    require(saved['config']['architecture'] == architecture(width) and saved['config']['labels'] == list(core.LABELS),
            'Checkpoint architecture/label mapping differs')
    backbone = build_backbone(core, data, device, saved_n0)
    head = make_head(core, Head, width, device, streams)
    graph = Graph(backbone.flow_encoder, backbone.packet_encoder, head).to(device)
    graph.load_state_dict(saved['model_state'], strict=True)
    graph.eval()
    assert_frozen(core, graph, encoder_hash)
    rows = core.select_rows(data, 'val')
    loader = core.DataLoader(core.GraphRows(data, rows), batch_size=16, shuffle=False, num_workers=2)
    cm, exports, offset, error = [[0] * 8 for _ in range(8)], [], 0, 0.
    with core.torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            ix = slice(offset, offset + batch.num_graphs)
            require(list(batch.sample_id) == cache['ids'][ix], 'Graph replay order differs')
            raw = graph(batch)
            cached, alpha = head(cache['flow64'][ix], cache['packet64'][ix])
            require(core.torch.allclose(raw, cached, atol=1e-5, rtol=1e-5)
                    and bool((raw.argmax(-1) == cached.argmax(-1)).all())
                    and core.torch.allclose(graph.last_flow_weights, alpha, atol=1e-5, rtol=1e-5),
                    'Full graph/cache replay differs')
            error = max(error, float((raw - cached).abs().max()))
            for s, y, p, a in zip(batch.sample_id, batch.y.view(-1).tolist(), raw.argmax(-1).tolist(), graph.last_flow_weights.tolist()):
                cm[y][p] += 1
                exports.append({'sample_id': s, 'label': y, 'prediction': p,
                                'flow_weights': a, 'packet_weights': [1 - x for x in a]})
            offset += batch.num_graphs
    frozen.verify_order(exports, rows)
    require(cm == saved['validation']['confusion_matrix'], 'Full graph checkpoint metric differs')
    assert_frozen(core, graph, encoder_hash)
    return {'samples': offset, 'max_abs_raw_cached_error': error, 'strict_state_load': True,
            'confusion_matrix_reproduced': True, 'frozen_encoder_unchanged': True,
            'wall_seconds': time.perf_counter() - started}, exports


def run(args, meta):
    preflight_started = time.perf_counter()
    c, source_spec, ref, nroot, historic, binding = meta
    gpu = checked_gpu(args.physical_gpu, args.allow_shared_gpu)
    activate_gpu(gpu)
    scripts = str(args.repo / 'scripts')
    sys.path.insert(0, scripts)
    try:
        import train_grouped as core
    finally:
        sys.path.remove(scripts)
    device = runtime(core)
    write(args.output / 'gpu.json', gpu)
    data = core.FrozenGraphs(inside(args.repo, source_spec['split_directory']), ref['binding']['split_id'])
    guard_graph_loads(data, {'train', 'val'})
    saved_n0 = core.torch.load(nroot / 'N0/best.pt', map_location=device, weights_only=False)
    require(saved_n0['epoch'] == 5 and saved_n0['config']['binding'] == ref['binding']
            and saved_n0['validation']['confusion_matrix'] == ref['model']['best_validation']['confusion_matrix'],
            'N0 checkpoint payload differs')
    backbone = build_backbone(core, data, device, saved_n0)
    encoder_hash = core.state_hash(encoder_values(backbone))
    source = {'encoder_sha256': encoder_hash, 'N0_best_sha256': c['n0_best_sha256'],
              'split_id': ref['binding']['split_id'], 'manifest_sha256': ref['binding']['manifest_sha256']}
    gold = predictions(nroot / 'N0/validation_predictions.jsonl.gz')
    cache_started = time.perf_counter()
    caches, receipts = {}, {}
    for part in ('train', 'val'):
        cached, receipts[part] = extract_cache(core, args, backbone, data, device, part, source,
            gold if part == 'val' else None, ref['model']['best_validation'] if part == 'val' else None)
        # Exercise on-disk cache restoration/identity before using it in either arm.
        restored = core.torch.load(args.output / (part + '_encoder64.pt'), map_location='cpu', weights_only=False)
        rows = core.select_rows(data, part)
        validate_cache_header(restored['metadata'], part, source, rows)
        require(restored['ids'] == cached['ids'] and all(core.torch.equal(restored[k], cached[k])
                for k in ('flow64', 'packet64', 'labels')), 'Cache roundtrip differs')
        caches[part] = {k: v.to(device) if core.torch.is_tensor(v) else v for k, v in restored.items()}
    cache_seconds = time.perf_counter() - cache_started
    cache_tensor_hashes = {part: core.state_hash([(k, cache[k]) for k in ('flow64', 'packet64', 'labels')])
                           for part, cache in caches.items()}
    write(args.output / 'cache_receipts.json', receipts)
    classes = load_classes()
    Head, Graph = classes
    streams = initialization_streams(c['seed'])
    models = {}
    initial = {}
    for name, width in VARIANTS.items():
        head = make_head(core, Head, width, device, streams)
        models[name] = Graph(backbone.flow_encoder, backbone.packet_encoder, head).to(device)
        initial[name] = core.state_hash(head.state_dict().items())
    require(all(core.torch.equal(a, b) for a, b in zip(models['D8'].fusion.shared_head[2].parameters(),
                                                     models['D32'].fusion.shared_head[2].parameters())),
            'Common output layer init was displaced by dimension-dependent RNG draws')
    write(args.output / 'initialization.json', {'architecture': {k: architecture(v) for k, v in VARIANTS.items()},
        'streams': streams, 'head_state_sha256': initial, 'all_heads_fresh': True,
        'shared_output_layer_initialization_identical': True, 'encoder_sha256': encoder_hash,
        'epoch_zero_is_fresh_model': True})
    args.probes = {}
    roundtrips = {}
    for name, graph in models.items():
        progress(args, 'raw_graph_gradient_probe', variant=name)
        args.probes[name] = probe(core, graph, data, caches['train'], device, c, encoder_hash)
        require(core.state_hash(graph.fusion.state_dict().items()) == initial[name], 'Probe head reset differs')
    write(args.output / 'probes.json', args.probes)
    # A strict full-state reload with a real graph also precedes formal training.
    warm = core.Batch.from_data_list([data.load(r) for r in data.training_candidates()[:16]]).to(device)
    for name, graph in models.items():
        path = args.output / (name + '_preflight.pt')
        core.checkpoint(path, {'model_state': graph.state_dict(), 'architecture': architecture(VARIANTS[name])})
        saved = core.torch.load(path, map_location=device, weights_only=False)
        fresh_head = make_head(core, Head, VARIANTS[name], device, streams)
        fresh = Graph(backbone.flow_encoder, backbone.packet_encoder, fresh_head).to(device)
        fresh.load_state_dict(saved['model_state'], strict=True)
        fresh.eval()
        graph.eval()
        assert_frozen(core, fresh, encoder_hash)
        require(core.state_hash(fresh.fusion.state_dict().items()) == initial[name], 'Roundtrip changed initialized heads')
        with core.torch.no_grad():
            left, right = fresh(warm), graph(warm)
            require(core.torch.allclose(left, right, atol=1e-5, rtol=1e-5)
                    and bool((left.argmax(-1) == right.argmax(-1)).all()), 'Preflight checkpoint replay failed')
        roundtrips[name] = {'strict_reload': True, 'samples': 16, 'checkpoint_sha256': sha(path)}
    write(args.output / 'preflight_roundtrips.json', roundtrips)
    preflight_seconds = time.perf_counter() - preflight_started
    write(args.output / 'status.json', {'status': 'running', 'phase': 'paired_training', 'tensor_preflight_passed': True})
    binding = {**binding, 'encoder_source': source, 'caches': receipts, 'cache_tensor_sha256': cache_tensor_hashes}
    write(args.output / 'binding.json', binding)
    results, exports = {}, {}
    for name in VARIANTS:
        results[name], exports[name] = train(core, args, c, models[name], caches, name, binding, encoder_hash,
            results['D8']['orders'] if name == 'D32' else None)
        progress(args, 'full_graph_best_checkpoint_replay', variant=name)
        require(sha(args.output / name / 'best.pt') == results[name]['checkpoint_sha256']['best.pt'],
                'Selected checkpoint changed before graph replay')
        checkpoint = core.torch.load(args.output / name / 'best.pt', map_location=device, weights_only=False)
        require(checkpoint['config']['binding'] == binding and checkpoint['config']['variant'] == name,
                'Checkpoint binding/variant differs')
        replay, raw_rows = graph_replay(core, args, classes, data, device, saved_n0, checkpoint, caches['val'], streams, encoder_hash)
        require([r['prediction'] for r in raw_rows] == [r['prediction'] for r in exports[name]], 'Paired exports differ')
        write(args.output / name / 'graph_replay.json', replay)
        require(all(core.state_hash([(k, cache[k]) for k in ('flow64', 'packet64', 'labels')]) == cache_tensor_hashes[part]
                    for part, cache in caches.items()), 'Shared representations changed during head training/replay')
    common = min(len(r['orders']) for r in results.values())
    require(results['D8']['orders'][:common] == results['D32']['orders'][:common]
            and results['D8']['training_start_rng_sha256'] == results['D32']['training_start_rng_sha256'],
            'Pair sample order/training RNG differs')
    delta = 100 * (results['D32']['best_validation']['macro_f1'] - results['D8']['best_validation']['macro_f1'])
    summary = {'status': 'completed', 'exit_code': 0, 'binding': binding, 'protocol': c, 'models': results,
        'N0_source': ref['model'], 'historical_J2': historic, 'D32_minus_D8_pp': delta,
        'D32_minus_historical_J2_pp': 100 * (results['D32']['best_validation']['macro_f1'] - historic['best_validation']['macro_f1']),
        'D32_versus_D8': pair.compare_predictions(exports['D8'], exports['D32']),
        'D32_versus_N0': pair.compare_predictions(gold, exports['D32']),
        'common_epochs_order_checked': common, 'shared_cache_wall_seconds': cache_seconds,
        'preflight_including_cache_wall_seconds': preflight_seconds,
        'tensor_preflight_passed': True, 'frozen_encoder_unchanged': True, 'shared_cache_unchanged': True,
        'probe_optimizer_updates_discarded': 6, 'training_jobs': 2,
        'test_evaluated': False, 'automatic_followup': False, 'class_tradeoff_review_required': True,
        'attribution': 'This pair compares fresh dimension-dependent projection/gate/head pipelines on fixed N0 encoders. '
                       'It changes gate input/output widths and parameter capacity together; it does not isolate a scalar '
                       'dimension effect or establish the contribution of dynamic versus equal fusion. Historical J2 uses a different training protocol.'}
    write(args.output / 'summary.json', summary)
    from summarize_fusion_dimension_pair import summarize
    summarize(args.output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path.cwd())
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=('check', 'run'), default='check')
    p.add_argument('--physical-gpu', type=int, default=1)
    p.add_argument('--allow-shared-gpu', action='store_true')
    args = p.parse_args()
    args.repo = args.repo.resolve()
    args.config = (args.repo / args.config).resolve()
    meta = metadata(args)
    if args.mode == 'check':
        print(json.dumps({'status': 'metadata_checks_passed', 'binding': meta[-1], 'training_started': False,
                          'tensor_checks_executed': False}), flush=True)
        return
    require(os.name == 'posix', 'Formal training runs on Linux wx server only')
    args.output = inside(args.repo / 'outputs', inside(args.repo, args.output))
    require(not args.output.exists(), 'Refuse overwrite/resume')
    args.output.mkdir(parents=True)
    write(args.output / 'config.json', meta[0])
    write(args.output / 'binding.json', meta[-1])
    write(args.output / 'status.json', {'status': 'running', 'phase': 'preflight_and_encoder_cache'})
    progress(args, 'loading_frozen_N0')
    try:
        run(args, meta)
        write(args.output / 'status.json', {'status': 'completed', 'exit_code': 0})
        write(args.output / 'progress.json', {'status': 'completed', 'training_jobs': 2})
    except BaseException:
        write(args.output / 'status.json', {'status': 'failed', 'exit_code': 1})
        previous = read(args.output / 'progress.json')
        write(args.output / 'progress.json', {**previous, 'status': 'failed', 'exit_code': 1})
        (args.output / 'error.log').write_text(traceback.format_exc(), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
