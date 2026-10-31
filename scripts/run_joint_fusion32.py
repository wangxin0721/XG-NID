"""Two raw-graph 32D joint fine-tunes, each starting from its own mature head.

Metadata checks require no torch. Server-only tensor probes block training.
Historical model definitions, protocols and artifacts remain unchanged.
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

from e3_gate_study import (activate_gpu, checked_gpu, guard_graph_loads, inside,
                           read, require, runtime, sha, write)
import run_fusion_gate_pair as control
from summarize_fusion_gate_pair import analyze as analyze_source

dim = control.parent
pair = dim.pair
HERE = Path(__file__).resolve().parent
MODULE = HERE.parent / 'src/xgnid/joint_fusion32.py'
VARIANTS = {'J32-Equal': ('D32-Equal', 'equal'), 'J32-Gate': ('D32-Gate', 'dynamic')}


def check_config(c):
    expected = {'schema': 'xgnid.joint-fusion32.v1', 'seed': 42, 'width': 32,
        'encoder_mode': 'eval_with_gradients', 'encoder_buffers': 'fixed',
        'inputs': 'raw_train_val_graphs_only', 'optimizer': 'fresh_Adam',
        'encoder_lr': 1e-6, 'head_lr': 1e-5, 'gate_lr': 1e-5, 'weight_decay': 0.,
        'batch_size': 512, 'eval_batch_size': 16, 'workers': 2, 'epochs': 100,
        'patience': 20, 'include_epoch_zero': True, 'probe_batches': 3, 'loss': 'plain_final_NLL',
        'budget_gate_above_stronger_reference_pp': .10, 'head_reinitialization': False,
        'warmup_training': False, 'training_jobs': 2, 'test_evaluated': False, 'automatic_followup': False}
    require(all(c.get(k) == v for k, v in expected.items()), 'Unregistered J32 protocol')
    require(set(c['variants']) == set(VARIANTS), 'Only J32-Equal and J32-Gate are registered')
    for name, (source, mode) in VARIANTS.items():
        require(c['variants'][name]['source'] == source and c['variants'][name]['mode'] == mode,
                'Wrong source/fusion pairing')


def metadata(args):
    c = read(args.config)
    check_config(c)
    original = inside(args.config.parent, c['source_protocol'])
    require(sha(original) == c['source_config_sha256'], 'Source 32D protocol changed')
    previous = control.metadata(SimpleNamespace(repo=args.repo, config=original))
    source = inside(args.repo / 'outputs', inside(args.repo, c['source_directory']))
    require(read(source / 'status.json') == {'status': 'completed', 'exit_code': 0}, 'Source 32D pair incomplete')
    s = read(source / 'summary.json')
    analyze_source(s)
    require(s['protocol'] == previous[0] == read(source / 'config.json')
            and s['binding'] == previous[-1] == read(source / 'binding.json')
            and s['encoder_parameters_and_BN_unchanged'] and s['shared_cache_unchanged']
            and s['source_artifacts_unchanged'], 'Source configuration/binding/state differs')
    files = ['status.json', 'config.json', 'binding.json', 'summary.json', 'gpu.json',
             'initialization.json', 'probes.json', 'preflight_roundtrips.json']
    for name, (old, mode) in VARIANTS.items():
        m, selected = s['models'][old], c['variants'][name]
        require(m == read(source / old / 'metrics.json') and m['architecture'] == control.architecture(mode)
                and m['best_epoch'] == selected['epoch']
                and abs(m['best_validation']['macro_f1'] - selected['macro_f1']) < 1e-12
                and sha(source / old / 'best.pt') == m['checkpoint_sha256']['best.pt'] == selected['best_sha256']
                and sha(source / old / 'validation_predictions.jsonl.gz') == m['predictions_sha256'],
                'Source checkpoint/metrics/predictions changed: ' + old)
        cfg = read(source / old / 'config.json')
        require(cfg['binding'] == s['binding'] and cfg['protocol'] == previous[0]
                and cfg['variant'] == old and cfg['architecture'] == control.architecture(mode), 'Source arm config differs')
        require(read(source / old / 'graph_replay.json') == s['graph_replays'][old]
                and s['graph_replays'][old]['strict_state_load']
                and s['graph_replays'][old]['confusion_matrix_reproduced'], 'Source graph replay incomplete')
        for filename in ('best.pt', 'last.pt', 'config.json', 'metrics.json', 'epochs.jsonl',
                         'validation_predictions.jsonl.gz', 'graph_replay.json'):
            files.append(old + '/' + filename)
    binding = {'config_sha256': sha(args.config), 'source_pair_binding': previous[-1],
        'source': {'directory': str(source), 'file_sha256': {n: sha(source / n) for n in files}},
        'sources': {n: sha(p) for n, p in {'scripts/run_joint_fusion32.py': Path(__file__),
            'src/xgnid/joint_fusion32.py': MODULE,
            'scripts/summarize_joint_fusion32.py': HERE / 'summarize_joint_fusion32.py'}.items()}}
    return c, previous, source, s, binding


def group_name(name, mode):
    if name.startswith(('flow_encoder.', 'packet_encoder.')):
        return 'encoder'
    if name.startswith(('fusion.flow_projection.', 'fusion.packet_projection.', 'fusion.shared_head.')):
        return 'head'
    if name.startswith('fusion.gate.'):
        return 'gate' if mode == 'dynamic' else 'unused_gate'
    raise ValueError('Unrecognized parameter: ' + name)


def groups(model, mode, c):
    require(mode in ('equal', 'dynamic'), 'Unknown fusion mode')
    buckets, roles = {k: [] for k in ('encoder', 'head', 'gate')}, {}
    for name, p in model.named_parameters(remove_duplicate=False):
        kind = group_name(name, mode)
        if id(p) in roles:
            require(roles[id(p)] == kind, 'Shared parameter has conflicting optimization roles: ' + name)
            continue
        roles[id(p)] = kind
        p.requires_grad_(kind != 'unused_gate')
        if kind != 'unused_gate':
            buckets[kind].append(p)
    result = [{'name': k, 'params': ps, 'lr': c[k + '_lr']} for k, ps in buckets.items() if ps]
    flat = [p for g in result for p in g['params']]
    require(len({id(p) for p in flat}) == len(flat)
            and {id(p) for p in flat} == {id(p) for p in model.parameters() if p.requires_grad},
            'Optimizer omitted or duplicated trainable parameters')
    head_count = sum(p.numel() for p in buckets['head'])
    gate_count = sum(p.numel() for p in buckets['gate'])
    require(head_count == 5480 and gate_count == (1664 if mode == 'dynamic' else 0)
            and bool(buckets['encoder']), 'Unexpected 32D parameter grouping')
    return result


def optimizer(core, model, mode, c):
    opt = core.torch.optim.Adam(groups(model, mode, c), weight_decay=c['weight_decay'])
    require(not opt.state, 'Fine-tuning must reset Adam state')
    return opt


def buffer_values(model):
    return [(n, t) for n, t in model.named_buffers() if n.startswith(('flow_encoder.', 'packet_encoder.'))]


def assert_modes_buffers(core, model, expected):
    require(all(not m.training for e in (model.flow_encoder, model.packet_encoder) for m in e.modules()),
            'Encoder eval mode was lost')
    require(all(p.requires_grad for e in (model.flow_encoder, model.packet_encoder) for p in e.parameters()),
            'Encoder gradients were disabled')
    require(core.state_hash(buffer_values(model)) == expected, 'Encoder/BN buffers changed')


def load_classes():
    heads, _ = control.load_classes()
    name = 'xgnid.joint_fusion32'
    spec = importlib.util.spec_from_file_location(name, MODULE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return heads, module.JointEncoderClassifier


def build(core, classes, data, device, n0, state, mode, c):
    Head, Graph = classes[0][mode], classes[1]
    backbone = dim.build_backbone(core, data, device, n0)
    head = dim.make_head(core, Head, 32, device, dim.initialization_streams(42))
    model = Graph(backbone.flow_encoder, backbone.packet_encoder, head).to(device)
    model.load_state_dict(state, strict=True)
    groups(model, mode, c)
    return model.eval()


def loaders(core, data, c, sampler=None):
    kwargs = dict(num_workers=c['workers'], pin_memory=True, persistent_workers=True, multiprocessing_context='fork')
    if sampler is not None:
        return core.DataLoader(core.GraphRows(data, core.select_rows(data, 'train')), batch_sampler=sampler, **kwargs)
    return core.DataLoader(core.GraphRows(data, core.select_rows(data, 'val')), batch_size=c['eval_batch_size'],
                           shuffle=False, **kwargs)


def evaluate(core, model, loader, device, expected_rows, args=None, phase='validation', export=False):
    model.eval()
    cm, exports, weights, offset = [[0] * 8 for _ in range(8)], [], [], 0
    started = time.perf_counter()
    with core.torch.no_grad():
        for step, batch in enumerate(loader, 1):
            batch = batch.to(device)
            scores = model(batch)
            alpha = model.last_flow_weights
            require(scores.shape == (batch.num_graphs, 8) and alpha.shape == (batch.num_graphs, 32)
                    and bool(core.torch.isfinite(scores).all()) and bool(core.torch.isfinite(alpha).all()),
                    'Invalid raw-graph evaluation output')
            current = expected_rows[offset:offset + batch.num_graphs]
            require(list(batch.sample_id) == [r['sample_id'] for r in current]
                    and batch.y.view(-1).tolist() == [r['label'] for r in current], 'Evaluation sample/label order differs')
            for sid, y, p, a in zip(batch.sample_id, batch.y.view(-1).tolist(), scores.argmax(-1).tolist(), alpha.tolist()):
                cm[y][p] += 1
                if export:
                    exports.append({'sample_id': sid, 'label': y, 'prediction': p,
                                    'flow_weights': a, 'packet_weights': [1-x for x in a]})
            weights.append(alpha.cpu())
            offset += batch.num_graphs
            if args is not None and (step == 1 or step % 200 == 0):
                dim.progress(args, phase, samples=offset, total=len(expected_rows), elapsed_seconds=time.perf_counter()-started)
    require(offset == len(expected_rows) == 13446, 'Incomplete validation')
    a = core.torch.cat(weights)
    metrics = dim.frozen.matrix_report(cm, core.LABELS)
    metrics['flow_weight_distribution'] = {'mean': a.mean(0).tolist(), 'std': a.std(0, unbiased=False).tolist(),
        'min': a.min(0).values.tolist(), 'max': a.max(0).values.tolist(),
        'max_abs_deviation_from_half': float((a-.5).abs().max())}
    metrics['wall_seconds_including_graph_io'] = time.perf_counter() - started
    dim.verify_metrics(metrics)
    return metrics, exports


def probe(core, model, data, device, c, mode, expected_buffers):
    clean = {n: t.detach().clone() for n, t in model.state_dict().items()}
    clean_hash, rng = core.state_hash(clean.items()), pair.rng_state(core)
    receipts, started = [], time.perf_counter()
    try:
        rows = core.select_rows(data, 'train')
        indices = list(islice(core.TrainBatches(len(rows), 512, 42), 3))
        loader = core.DataLoader(core.GraphRows(data, rows), batch_sampler=indices, num_workers=0)
        opt = optimizer(core, model, mode, c)
        prefixes = ['flow_encoder.', 'packet_encoder.', 'fusion.flow_projection.',
                    'fusion.packet_projection.', 'fusion.shared_head.']
        if mode == 'dynamic':
            prefixes += ['fusion.gate.0.', 'fusion.gate.2.']
        for step, (batch, numbers) in enumerate(zip(loader, indices), 1):
            model.train()
            require(model.fusion.training, 'Trainable fusion modules must train')
            assert_modes_buffers(core, model, expected_buffers)
            require(list(batch.sample_id) == [rows[i]['sample_id'] for i in numbers], 'Probe sample order differs')
            batch = batch.to(device)
            before = {n: p.detach().clone() for n, p in model.named_parameters()}
            opt.zero_grad(set_to_none=True)
            scores = model(batch)
            require(scores.requires_grad, 'Raw graph forward disabled autograd')
            if mode == 'equal':
                require(bool((model.last_flow_weights == .5).all()), 'Equal fusion changed')
            loss = core.F.nll_loss(scores, batch.y.view(-1))
            require(bool(core.torch.isfinite(loss)), 'Nonfinite probe loss')
            loss.backward()
            require(all(p.grad is None or bool(core.torch.isfinite(p.grad).all()) for p in model.parameters()), 'Bad gradient')
            norms = {n: float(p.grad.abs().sum()) if p.grad is not None else 0. for n, p in model.named_parameters()}
            opt.step()
            changed = {n: not core.torch.equal(before[n], p) for n, p in model.named_parameters()}
            if mode == 'equal':
                require(all(p.grad is None and not p.requires_grad for p in model.fusion.gate.parameters())
                        and not any(v for n, v in changed.items() if n.startswith('fusion.gate.')), 'Unused gate trained')
            assert_modes_buffers(core, model, expected_buffers)
            receipts.append({'step': step, 'loss': float(loss.detach()), 'gradients_l1': norms, 'parameters_changed': changed})
        # Check the full three-update window; one zero gradient step is not a failed path.
        for prefix in prefixes:
            require(any(any(v > 0 for n, v in r['gradients_l1'].items() if n.startswith(prefix))
                        and any(v for n, v in r['parameters_changed'].items() if n.startswith(prefix)) for r in receipts),
                    'Missing gradient/update across three batches: ' + prefix)
    finally:
        model.load_state_dict(clean, strict=True)
        model.zero_grad(set_to_none=True)
        model.eval()
        pair.restore_rng(core, rng)
        require(core.state_hash(model.state_dict().items()) == clean_hash
                and pair.rng_hash(core, rng) == pair.rng_hash(core, pair.rng_state(core)), 'Probe state/RNG restore failed')
        assert_modes_buffers(core, model, expected_buffers)
    require(len(receipts) == 3, 'Incomplete probe')
    return {'updates': receipts, 'discarded': True, 'full_state_and_rng_restored': True,
        'encoder_parameters_updated_in_probe': True, 'encoder_BN_buffers_unchanged': True,
        'wall_seconds': time.perf_counter()-started}


def check_history(history, result, c):
    require(c['include_epoch_zero'] is True and [r['epoch'] for r in history] == list(range(1, result['epochs']+1))
            and 1 <= len(history) <= c['epochs'], 'Invalid joint history')
    best, selected, stale = result['epoch_zero']['macro_f1'], 0, 0
    for r in history:
        require(math.isfinite(r['val_macro_f1']) and 0 <= r['val_macro_f1'] <= 1 and r['optimizer_updates'] == 211,
                'Invalid epoch metrics/updates')
        best, stale, improved = pair.selection_state(r['val_macro_f1'], best, stale)
        if improved:
            selected = r['epoch']
        require(r['best_epoch'] == selected and r['stale_epochs'] == stale
                and (r['epoch'] == len(history) or stale < c['patience']), 'Selection history differs')
    require(result['best_epoch'] == selected and result['selected_epoch_zero'] == (selected == 0)
            and abs(best-result['best_validation']['macro_f1']) < 1e-12
            and result['optimizer_updates'] == 211*len(history)
            and result['orders'] == [r['train_order_sha256'] for r in history]
            and (stale >= c['patience'] or len(history) == c['epochs']), 'Selection/budget differs')


def train(core, args, c, model, data, device, name, binding, zero, expected_buffers, expected_orders=None):
    mode = VARIANTS[name][1]
    folder = args.output / name
    folder.mkdir()
    gs = groups(model, mode, c)
    group_report = [{'name': g['name'], 'lr': g['lr'], 'parameters': sum(p.numel() for p in g['params'])} for g in gs]
    cfg = {'variant': name, 'mode': mode, 'fusion_architecture': control.architecture(mode), 'protocol': c,
        'binding': binding, 'labels': list(core.LABELS), 'optimizer_reset': True,
        'encoder_policy': 'eval_with_gradients_fixed_buffers', 'parameter_groups': group_report,
        'source_checkpoint': c['variants'][name]}
    write(folder / 'config.json', cfg)
    write(folder / 'parameter_groups.json', group_report)
    write(folder / 'epoch_zero.json', zero)
    write(folder / 'probe.json', args.probes[name])
    sampler = core.TrainBatches(len(core.select_rows(data, 'train')), c['batch_size'], 42)
    loader, val = loaders(core, data, c, sampler), loaders(core, data, c)
    val_rows = core.select_rows(data, 'val')
    core.seed_all(42)
    start_rng = pair.rng_hash(core, pair.rng_state(core))
    opt = optimizer(core, model, mode, c)
    best, best_epoch, stale = zero['macro_f1'], 0, 0
    orders, history = [], []
    payload = {'model_state': model.state_dict(), 'epoch': 0, 'validation': zero, 'config': cfg}
    core.checkpoint(folder / 'epoch_zero.pt', payload)
    core.checkpoint(folder / 'best.pt', payload)
    core.sync(device)
    started = time.perf_counter()
    with (folder / 'epochs.jsonl').open('x', encoding='utf-8') as log:
        for epoch in range(1, c['epochs']+1):
            model.train()
            assert_modes_buffers(core, model, expected_buffers)
            steps, count, loss_sum = 0, 0, 0.
            for batch in loader:
                if steps == 0 and expected_orders is not None and epoch <= len(expected_orders):
                    require(sampler.last_order_sha256 == expected_orders[epoch-1], 'Paired sample order differs')
                batch = batch.to(device)
                opt.zero_grad(set_to_none=True)
                scores = model(batch)
                loss = core.F.nll_loss(scores, batch.y.view(-1))
                require(bool(core.torch.isfinite(loss)), 'Nonfinite train loss')
                loss.backward()
                require(all(p.grad is None or bool(core.torch.isfinite(p.grad).all()) for p in model.parameters()),
                        'Nonfinite training gradient')
                opt.step()
                steps += 1
                count += batch.num_graphs
                loss_sum += float(loss.detach())*batch.num_graphs
                if steps == 1 or steps % 50 == 0:
                    dim.progress(args, 'joint_training', variant=name, epoch=epoch, batch=steps, batches=211,
                                 train_nll=loss_sum/count, elapsed_seconds=time.perf_counter()-started)
            require(steps == 211 and count == 107621, 'Incomplete training epoch')
            assert_modes_buffers(core, model, expected_buffers)
            metrics, _ = evaluate(core, model, val, device, val_rows, args, name + '_validation')
            best, stale, improved = pair.selection_state(metrics['macro_f1'], best, stale)
            if improved:
                best_epoch = epoch
            orders.append(sampler.last_order_sha256)
            payload = {'model_state': model.state_dict(), 'optimizer_state': opt.state_dict(), 'epoch': epoch,
                'validation': metrics, 'config': cfg, 'rng': pair.rng_state(core), 'sampler_rng': sampler.generator.get_state()}
            core.checkpoint(folder / 'last.pt', payload)
            if improved:
                core.checkpoint(folder / 'best.pt', payload)
            row = {'epoch': epoch, 'train_nll': loss_sum/count, 'val_macro_f1': metrics['macro_f1'],
                'optimizer_updates': steps, 'train_order_sha256': orders[-1], 'best_epoch': best_epoch, 'stale_epochs': stale}
            history.append(row)
            log.write(json.dumps(row)+'\n')
            log.flush()
            dim.progress(args, 'epoch_completed', variant=name, **row)
            if stale >= c['patience']:
                break
    core.sync(device)
    elapsed = time.perf_counter()-started
    result = {'best_validation': zero if best_epoch == 0 else None, 'last_validation': metrics,
        'best_epoch': best_epoch, 'epochs': epoch, 'optimizer_updates': epoch*211, 'orders': orders,
        'epoch_zero': zero, 'epoch_zero_eligible': True, 'selected_epoch_zero': best_epoch == 0,
        'training_start_rng_sha256': start_rng, 'training_wall_seconds': elapsed,
        'parameter_groups': group_report, 'joint_trainable_parameters': sum(g['parameters'] for g in group_report),
        'training_evidence': dim.trained_evidence(zero['macro_f1'], history),
        'checkpoint_sha256': {n: sha(folder/n) for n in ('best.pt', 'last.pt')}, 'test_evaluated': False}
    selected = core.torch.load(folder/'best.pt', map_location=device, weights_only=False)
    result['best_validation'] = selected['validation']
    check_history(history, result, c)
    return result


def replay(core, args, classes, data, device, n0, name, c, result, binding, buffers):
    started = time.perf_counter()
    folder, mode = args.output/name, VARIANTS[name][1]
    saved = core.torch.load(folder/'best.pt', map_location=device, weights_only=False)
    require(sha(folder/'best.pt') == result['checkpoint_sha256']['best.pt']
            and saved['config'] == read(folder/'config.json') and saved['config']['binding'] == binding
            and saved['config']['labels'] == list(core.LABELS) and saved['config']['mode'] == mode
            and saved['config']['fusion_architecture'] == control.architecture(mode)
            and saved['epoch'] == result['best_epoch'], 'Best checkpoint config/identity differs')
    fresh = build(core, classes, data, device, n0, saved['model_state'], mode, c)
    assert_modes_buffers(core, fresh, buffers)
    metrics, exports = evaluate(core, fresh, loaders(core, data, c), device, core.select_rows(data, 'val'), args,
                               name+'_best_raw_graph_replay', export=True)
    require(metrics['confusion_matrix'] == saved['validation']['confusion_matrix']
            and abs(metrics['macro_f1']-saved['validation']['macro_f1']) < 1e-12, 'Raw checkpoint replay failed')
    pair.save_predictions(folder/'validation_predictions.jsonl.gz', exports)
    result['best_validation'] = metrics
    result['predictions_sha256'] = sha(folder/'validation_predictions.jsonl.gz')
    if mode == 'equal':
        require(metrics['flow_weight_distribution']['max_abs_deviation_from_half'] == 0., 'Equal fusion changed')
    assert_modes_buffers(core, fresh, buffers)
    receipt = {'samples': len(exports), 'strict_state_load': True, 'raw_graphs_only': True,
        'confusion_matrix_reproduced': True, 'encoder_BN_buffers_unchanged': True,
        'wall_seconds': time.perf_counter()-started}
    write(folder/'graph_replay.json', receipt)
    write(folder/'metrics.json', result)
    return exports, receipt


def run(args, meta):
    preflight_started = time.perf_counter()
    c, previous, source, old, binding = meta
    _, spec, ref, nroot, _, _ = previous[1]
    gpu = checked_gpu(args.physical_gpu, args.allow_shared_gpu)
    activate_gpu(gpu)
    scripts = str(args.repo/'scripts')
    sys.path.insert(0, scripts)
    try:
        import train_grouped as core
    finally:
        sys.path.remove(scripts)
    device = runtime(core)
    write(args.output/'gpu.json', gpu)
    data = core.FrozenGraphs(inside(args.repo, spec['split_directory']), ref['binding']['split_id'])
    guard_graph_loads(data, {'train', 'val'})
    n0 = core.torch.load(nroot/'N0/best.pt', map_location=device, weights_only=False)
    require(n0['config']['binding'] == ref['binding'] and n0['epoch'] == 5, 'N0 backbone config differs')
    classes = load_classes()
    models, zeros, buffers, starts = {}, {}, {}, {}
    args.probes = {}
    for name, (oldname, mode) in VARIANTS.items():
        dim.progress(args, 'load_own_best_and_reproduce_epoch_zero', variant=name)
        saved = core.torch.load(source/oldname/'best.pt', map_location=device, weights_only=False)
        require(saved['config'] == read(source/oldname/'config.json')
                and saved['epoch'] == c['variants'][name]['epoch']
                and saved['validation']['confusion_matrix'] == old['models'][oldname]['best_validation']['confusion_matrix']
                and saved['config']['labels'] == list(core.LABELS), 'Source checkpoint payload differs')
        model = build(core, classes, data, device, n0, saved['model_state'], mode, c)
        require(core.state_hash(model.state_dict().items()) == core.state_hash(saved['model_state'].items()),
                'Mature checkpoint was not loaded exactly')
        require(core.state_hash(dim.encoder_values(model)) == old['initialization']['encoder_sha256'],
                'Source mature heads do not contain the registered frozen N0 encoders')
        models[name] = model
        buffers[name] = core.state_hash(buffer_values(model))
        starts[name] = core.state_hash(model.state_dict().items())
        zeros[name], exports = evaluate(core, model, loaders(core, data, c), device,
            core.select_rows(data, 'val'), args, name+'_epoch_zero', export=True)
        original_exports = dim.predictions(source/oldname/'validation_predictions.jsonl.gz')
        require(zeros[name]['confusion_matrix'] == old['models'][oldname]['best_validation']['confusion_matrix']
                and abs(zeros[name]['macro_f1']-c['variants'][name]['macro_f1']) < 1e-12
                and [r['sample_id'] for r in exports] == [r['sample_id'] for r in original_exports]
                and [r['prediction'] for r in exports] == [r['prediction'] for r in original_exports],
                'Own epoch0 predictions/metrics do not reproduce source')
        dim.progress(args, 'three_raw_graph_update_probe', variant=name)
        args.probes[name] = probe(core, model, data, device, c, mode, buffers[name])
        require(core.state_hash(model.state_dict().items()) == starts[name], 'Probe contaminated starting checkpoint')
        # Roundtrip into a separately constructed model before formal training.
        path = args.output/(name+'_preflight.pt')
        core.checkpoint(path, {'model_state': model.state_dict(), 'mode': mode})
        restored = core.torch.load(path, map_location=device, weights_only=False)
        fresh = build(core, classes, data, device, n0, restored['model_state'], mode, c)
        warm = core.Batch.from_data_list([data.load(r) for r in data.training_candidates()[:16]]).to(device)
        with core.torch.no_grad():
            a, b = model(warm), fresh(warm)
            require(core.torch.allclose(a, b, atol=1e-5, rtol=1e-5)
                    and bool((a.argmax(-1) == b.argmax(-1)).all()), 'Initial strict raw-graph roundtrip failed')
        assert_modes_buffers(core, fresh, buffers[name])
    require(not {id(p) for p in models['J32-Equal'].parameters()}.intersection(
        id(p) for p in models['J32-Gate'].parameters()), 'Arms unexpectedly share mutable parameters')
    write(args.output/'initial_checks.json', {n: {'own_start_reproduced': True, 'full_state_sha256': starts[n],
        'encoder_buffer_sha256': buffers[n], 'strict_roundtrip': True, 'epoch_zero': zeros[n]} for n in VARIANTS})
    write(args.output/'probes.json', args.probes)
    control.assert_source_unchanged(binding)
    preflight_seconds = time.perf_counter()-preflight_started
    write(args.output/'status.json', {'status': 'running', 'phase': 'paired_training', 'tensor_preflight_passed': True})
    results, exports, replays = {}, {}, {}
    for name in VARIANTS:
        results[name] = train(core, args, c, models[name], data, device, name, binding, zeros[name], buffers[name],
            results['J32-Equal']['orders'] if name == 'J32-Gate' else None)
        exports[name], replays[name] = replay(core, args, classes, data, device, n0, name, c, results[name], binding, buffers[name])
    common = min(r['epochs'] for r in results.values())
    require(results['J32-Equal']['orders'][:common] == results['J32-Gate']['orders'][:common]
            and results['J32-Equal']['training_start_rng_sha256'] == results['J32-Gate']['training_start_rng_sha256'],
            'Joint pair sample order/RNG differs')
    control.assert_source_unchanged(binding)
    from summarize_joint_fusion32 import changed_samples, summarize
    from summarize_fusion_gate_pair import prediction_transitions
    comparisons = {n: pair.compare_predictions(dim.predictions(source/VARIANTS[n][0]/'validation_predictions.jsonl.gz'), exports[n])
                   for n in VARIANTS}
    for name in VARIANTS:
        pair.save_predictions(args.output/name/'changes_from_own_start.jsonl.gz', changed_samples(
            dim.predictions(source/VARIANTS[name][0]/'validation_predictions.jsonl.gz'), exports[name], core.LABELS))
    pair.save_predictions(args.output/'gate_vs_equal_changes.jsonl.gz', changed_samples(
        exports['J32-Equal'], exports['J32-Gate'], core.LABELS))
    summary = {'status': 'completed', 'exit_code': 0, 'binding': binding, 'protocol': c, 'models': results,
        'source_models': old['models'], 'own_start_changes': comparisons,
        'own_start_transitions': {n: prediction_transitions(dim.predictions(source/VARIANTS[n][0]/'validation_predictions.jsonl.gz'),
                                 exports[n], core.LABELS) for n in VARIANTS},
        'Gate_versus_Equal': pair.compare_predictions(exports['J32-Equal'], exports['J32-Gate']),
        'Gate_versus_Equal_transitions': prediction_transitions(exports['J32-Equal'], exports['J32-Gate'], core.LABELS),
        'common_epochs_order_checked': common, 'graph_replays': replays,
        'tensor_preflight_passed': True, 'encoder_BN_buffers_unchanged': True, 'source_artifacts_unchanged': True,
        'encoder_parameters_trainable': True, 'raw_graph_training': True, 'frozen_cache_used_for_training': False,
        'discarded_probe_updates': 6, 'preflight_wall_seconds': preflight_seconds,
        'training_jobs': 2, 'test_evaluated': False, 'automatic_followup': False}
    write(args.output/'summary.json', summary)
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
    args.output = inside(args.repo/'outputs', inside(args.repo, args.output))
    meta = metadata(args)
    require(not args.output.exists() and args.output != meta[2] and meta[2] not in args.output.parents,
            'Refuse overwrite/resume or source output reuse')
    if args.mode == 'check':
        print(json.dumps({'status': 'metadata_checks_passed', 'starts': meta[0]['variants'],
            'training_started': False, 'tensor_checks_executed': False}), flush=True)
        return
    require(os.name == 'posix', 'Formal training runs on the Linux wx server only')
    args.output.mkdir(parents=True)
    write(args.output/'config.json', meta[0])
    write(args.output/'binding.json', meta[-1])
    write(args.output/'status.json', {'status': 'running', 'phase': 'preflight'})
    dim.progress(args, 'validate_two_mature_starts')
    try:
        run(args, meta)
        write(args.output/'status.json', {'status': 'completed', 'exit_code': 0})
        write(args.output/'progress.json', {'status': 'completed', 'training_jobs': 2})
    except BaseException:
        write(args.output/'status.json', {'status': 'failed', 'exit_code': 1})
        write(args.output/'progress.json', {**read(args.output/'progress.json'), 'status': 'failed', 'exit_code': 1})
        (args.output/'error.log').write_text(traceback.format_exc(), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
