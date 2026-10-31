"""Train a matched equal/dynamic 32D fusion pair from the clean D32 start.

No torch imports in metadata mode. Server probes precede both new trainings.
Original scripts, caches, checkpoints and reports are read-only dependencies.
"""
from __future__ import annotations

import argparse
import importlib.util
from itertools import islice
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import FunctionType, SimpleNamespace

from e3_gate_study import (activate_gpu, checked_gpu, guard_graph_loads, inside,
                           read, require, runtime, sha, write)
import run_fusion_dimension_pair as parent

HERE = Path(__file__).resolve().parent
MODULE = HERE.parent / 'src/xgnid/fusion_gate_control.py'
VARIANTS = {'D32-Equal': 'equal', 'D32-Gate': 'dynamic'}


def check_config(c):
    expected = {'schema': 'xgnid.fusion-gate-pair.v1', 'variants': VARIANTS,
        'seed': 42, 'width': 32, 'encoder_mode': 'frozen_eval',
        'initialization': 'strict_reuse_clean_source_D32_preflight_and_epoch_zero',
        'optimizer': 'fresh_Adam', 'lr': .001, 'weight_decay': 0., 'batch_size': 512,
        'eval_batch_size': 16, 'workers': 2, 'epochs': 100, 'patience': 20,
        'include_epoch_zero': False, 'probe_batches': 3, 'loss': 'plain_final_NLL',
        'budget_gate_minus_equal_pp': .10, 'historical_rerun_material_difference_pp': .05,
        'statistics_residual': False, 'training_jobs': 2, 'test_evaluated': False,
        'automatic_followup': False}
    require(all(c.get(k) == v for k, v in expected.items()), 'Unregistered equal/gate pair protocol')


def architecture(mode, width=32):
    require(mode in ('equal', 'dynamic') and width == 32, 'Only the registered 32D controls are supported')
    return {**parent.architecture(width), 'fusion': mode,
        'trainable_parameters': 5480 if mode == 'equal' else 7144,
        'stored_head_parameters': 7144, 'unused_frozen_gate_parameters': 1664 if mode == 'equal' else 0}


def metadata(args):
    c = read(args.config)
    check_config(c)
    original = inside(args.config.parent, c['source_protocol'])
    require(sha(original) == c['source_config_sha256'], 'Original dimension protocol changed')
    base = parent.metadata(SimpleNamespace(repo=args.repo, config=original))
    source = inside(args.repo / 'outputs', inside(args.repo, c['source_directory']))
    require(read(source / 'status.json') == {'status': 'completed', 'exit_code': 0}, 'Source pair is not complete')
    s = read(source / 'summary.json')
    require(s['status'] == 'completed' and s['exit_code'] == 0 and s['test_evaluated'] is False
            and s['tensor_preflight_passed'] and s['frozen_encoder_unchanged'] and s['shared_cache_unchanged'],
            'Source pair invariants failed')
    b = read(source / 'binding.json')
    require(b == s['binding'] and s['protocol'] == base[0] == read(source / 'config.json'), 'Source config/binding differs')
    extras = {'encoder_source', 'caches', 'cache_tensor_sha256', 'recovery'}
    require({k: v for k, v in b.items() if k not in extras} == base[-1], 'Source code/reference binding changed')
    m = s['models']['D32']
    require(m == read(source / 'D32/metrics.json') and m['architecture'] == parent.architecture(32)
            and m['epoch_zero_eligible'] is False and m['selected_epoch_zero'] is False
            and abs(m['best_validation']['macro_f1'] - c['source_D32_macro_f1']) < 1e-12
            and sha(source / 'D32/best.pt') == c['source_D32_best_sha256'] == m['checkpoint_sha256']['best.pt'],
            'Historical D32 checkpoint/score differs')
    for key in ('epoch_zero', 'best_validation', 'last_validation'):
        parent.verify_metrics(m[key])
    history = [json.loads(line) for line in (source / 'D32/epochs.jsonl').read_text(encoding='utf-8').splitlines()]
    parent.check_history(history, m, base[0])
    require(m['orders'] == [r['train_order_sha256'] for r in history], 'Source training order ledger differs')
    init, rounds, receipts = [read(source / n) for n in ('initialization.json', 'preflight_roundtrips.json', 'cache_receipts.json')]
    require(init['streams'] == parent.initialization_streams(42) and init['encoder_sha256'] == b['encoder_source']['encoder_sha256']
            and init['all_heads_fresh'] and rounds['D32']['strict_reload'] is True,
            'Source initialization provenance differs')
    require(sha(source / 'D32_preflight.pt') == rounds['D32']['checkpoint_sha256'], 'Clean D32 snapshot changed')
    for part in ('train', 'val'):
        require(receipts[part] == b['caches'][part]
                and sha(source / (part + '_encoder64.pt')) == receipts[part]['sha256'] == c['source_cache_sha256'][part]
                and receipts[part]['metadata']['source'] == b['encoder_source']
                and receipts[part]['metadata']['embedding_dim'] == 64, 'Source cache mismatch')
    require(receipts['val']['N0_replayed_validation']['confusion_matrix']
            == base[2]['model']['best_validation']['confusion_matrix'], 'Source N0 reproduction differs')
    files = ['status.json', 'summary.json', 'binding.json', 'config.json', 'gpu.json', 'initialization.json',
             'cache_receipts.json', 'preflight_roundtrips.json', 'D32_preflight.pt', 'D32/epoch_zero.pt',
             'D32/best.pt', 'D32/last.pt', 'D32/metrics.json', 'D32/epochs.jsonl',
             'D32/validation_predictions.jsonl.gz', 'train_encoder64.pt', 'val_encoder64.pt']
    receipt = {'directory': str(source), 'file_sha256': {n: sha(source / n) for n in files}}
    binding = {'config_sha256': sha(args.config), 'original_protocol_binding': base[-1], 'source': receipt,
        'sources': {n: sha(p) for n, p in {'scripts/run_fusion_gate_pair.py': Path(__file__),
            'src/xgnid/fusion_gate_control.py': MODULE,
            'scripts/summarize_fusion_gate_pair.py': HERE / 'summarize_fusion_gate_pair.py'}.items()}}
    return c, base, source, s, binding


def optimizer(core, graph, c):
    params = [p for p in graph.fusion.parameters() if p.requires_grad]
    ids = {id(p) for p in params}
    enc = {id(p) for e in (graph.flow_encoder, graph.packet_encoder) for p in e.parameters()}
    require(len(ids) == len(params) and ids == {id(p) for p in graph.parameters() if p.requires_grad}
            and not ids.intersection(enc), 'Optimizer parameters omitted, duplicated or leaked')
    require(sum(p.numel() for p in params) == architecture(graph.fusion.fusion_mode)['trainable_parameters'],
            'Wrong active parameter count')
    return core.torch.optim.Adam(params, lr=c['lr'], weight_decay=c['weight_decay'])


def adapted(function, mode):
    """Reuse the unchanged selection/training/replay implementation with explicit controls."""
    namespace = dict(function.__globals__, VARIANTS={n: 32 for n in VARIANTS}, optimizer=optimizer,
                     architecture=lambda width: architecture(mode, width))
    return FunctionType(function.__code__, namespace, function.__name__, function.__defaults__)


def load_classes():
    Dynamic, Graph = parent.load_classes()
    name = 'xgnid.fusion_gate_control'
    spec = importlib.util.spec_from_file_location(name, MODULE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return {'equal': module.EqualFusionHead, 'dynamic': Dynamic}, Graph


def probe(core, graph, data, cache, device, c, encoder_hash):
    """Probe raw graphs for three updates, then restore parameters, buffers and RNG."""
    started = time.perf_counter()
    clean = {n: t.detach().clone() for n, t in graph.state_dict().items()}
    digest, rng = core.state_hash(clean.items()), parent.pair.rng_state(core)
    records, mode = [], graph.fusion.fusion_mode
    try:
        rows = core.select_rows(data, 'train')
        indices = list(islice(core.TrainBatches(len(rows), 512, 42), c['probe_batches']))
        loader = core.DataLoader(core.GraphRows(data, rows), batch_sampler=indices, num_workers=0)
        opt = optimizer(core, graph, c)
        for step, (batch, numbers) in enumerate(zip(loader, indices), 1):
            graph.train()
            require(graph.fusion.training, 'Fusion head is not in training mode')
            index = core.torch.tensor(numbers, device=device)
            batch = batch.to(device)
            require(list(batch.sample_id) == [cache['ids'][i] for i in numbers], 'Probe sample order differs')
            with core.torch.no_grad():
                cached, alpha = graph.fusion(cache['flow64'][index], cache['packet64'][index])
                if step == 1 or mode == 'equal':
                    require(bool((alpha == .5).all()), 'Equal fusion invariant failed')
            before = {n: p.detach().clone() for n, p in graph.fusion.named_parameters()}
            opt.zero_grad(set_to_none=True)
            scores = graph(batch)
            require(core.torch.allclose(scores, cached, atol=1e-5, rtol=1e-5), 'Raw/cache probe mismatch')
            loss = core.F.nll_loss(scores, batch.y.view(-1))
            require(bool(core.torch.isfinite(loss)), 'Nonfinite probe loss')
            loss.backward()
            require(all(p.grad is None or bool(core.torch.isfinite(p.grad).all()) for p in graph.parameters()),
                    'Nonfinite gradient')
            norms = {n: float(p.grad.abs().sum()) if p.grad is not None else 0. for n, p in graph.fusion.named_parameters()}
            opt.step()
            changed = {n: not core.torch.equal(before[n], p) for n, p in graph.fusion.named_parameters()}
            prefixes = ['flow_projection.', 'packet_projection.', 'shared_head.']
            if mode == 'dynamic':
                prefixes.append('gate.2.')
            for prefix in prefixes:
                require(any(v > 0 for n, v in norms.items() if n.startswith(prefix))
                        and any(v for n, v in changed.items() if n.startswith(prefix)), 'Missing gradient/update: ' + prefix)
            if mode == 'equal':
                require(all(not p.requires_grad and p.grad is None for p in graph.fusion.gate.parameters())
                        and not any(v for n, v in changed.items() if n.startswith('gate.')), 'Equal gate unexpectedly trained')
            parent.assert_frozen(core, graph, encoder_hash)
            records.append({'step': step, 'loss': float(loss.detach()), 'gradients_l1': norms,
                'parameters_changed': changed, 'raw_cache_max_error': float((scores - cached).abs().max())})
        require(len(records) == 3, 'Incomplete gradient probe')
        if mode == 'dynamic':
            require(any(r['gradients_l1']['gate.0.weight'] > 0 and r['parameters_changed']['gate.0.weight']
                        for r in records[1:]), 'Hidden gate never updated after the zero-output first step')
    finally:
        graph.load_state_dict(clean, strict=True)
        graph.zero_grad(set_to_none=True)
        graph.eval()
        parent.pair.restore_rng(core, rng)
        require(core.state_hash(graph.state_dict().items()) == digest
                and parent.pair.rng_hash(core, rng) == parent.pair.rng_hash(core, parent.pair.rng_state(core)),
                'Probe failed to restore model/RNG')
        parent.assert_frozen(core, graph, encoder_hash)
    return {'updates': records, 'discarded': True, 'clean_state_and_rng_restored': True,
        'encoder_parameters_and_BN_unchanged': True, 'wall_seconds': time.perf_counter() - started,
        'initial_head_sha256': core.state_hash(graph.fusion.state_dict().items())}


def assert_source_unchanged(binding):
    source = Path(binding['source']['directory'])
    for name, digest in binding['source']['file_sha256'].items():
        require(sha(source / name) == digest, 'Source artifact modified: ' + name)


def execution_comparison(args, c, old, results, core):
    # Use the bound successful source, rather than assuming a server checkout path.
    historical = [json.loads(line) for line in (Path(args.source) / 'D32/epochs.jsonl').read_text(encoding='utf-8').splitlines()]
    rerun = [json.loads(line) for line in (args.output / 'D32-Gate/epochs.jsonl').read_text(encoding='utf-8').splitlines()]
    common = min(len(historical), len(rerun))
    checks = ('seed', 'optimizer', 'lr', 'weight_decay', 'batch_size', 'eval_batch_size',
              'workers', 'epochs', 'patience', 'include_epoch_zero', 'loss')
    differences = {key: {'source': old['protocol'][key], 'new': c[key]} for key in checks if old['protocol'][key] != c[key]}
    require(not differences, 'Dynamic rerun common training protocol differs')
    return {'common_protocol_differences': differences, 'source_and_rerun_initial_head_identical': True,
        'source_and_rerun_training_rng_identical': results['D32-Gate']['training_start_rng_sha256']
            == old['models']['D32']['training_start_rng_sha256'],
        'same_model_definition_verified_by_source_sha256': True,
        'same_64D_cache_verified_by_sha256': True, 'common_epochs_with_historical_D32': common,
        'max_common_epoch_macro_f1_abs_difference': max(abs(a['val_macro_f1'] - b['val_macro_f1'])
                                                       for a, b in zip(historical, rerun)),
        'max_common_epoch_train_nll_abs_difference': max(abs(a['train_nll'] - b['train_nll'])
                                                        for a, b in zip(historical, rerun)),
        'source_gpu': read(Path(args.source) / 'gpu.json'), 'rerun_gpu': read(args.output / 'gpu.json'),
        'runtime': {'torch': core.torch.__version__, 'pyg': core.torch_geometric.__version__,
                    'python': sys.version.split()[0], 'cuda': core.torch.version.cuda},
        'unexplained_difference_policy': 'Review hardware/numerical execution if scores materially differ despite '
                                        'matching verified inputs and settings; do not infer a cause from the score alone.'}


def compare_initial(core, left, right, cache, expected):
    a, ea = parent.evaluate(core, left, cache)
    b, eb = parent.evaluate(core, right, cache)
    error = 0.
    with core.torch.no_grad():
        for offset in range(0, len(cache['ids']), 16):
            ix = slice(offset, offset + 16)
            x, alpha = left(cache['flow64'][ix], cache['packet64'][ix])
            y, beta = right(cache['flow64'][ix], cache['packet64'][ix])
            require(core.torch.allclose(x, y, atol=1e-6, rtol=1e-6)
                    and bool((alpha == .5).all()) and bool((beta == .5).all()), 'Initial pair outputs differ')
            error = max(error, float((x - y).abs().max()))
    require(a['confusion_matrix'] == b['confusion_matrix'] == expected['confusion_matrix']
            and [r['prediction'] for r in ea] == [r['prediction'] for r in eb], 'Initial predictions/CM differ')
    return {'full_validation_samples': len(ea), 'outputs_allclose': True, 'predictions_identical': True,
        'confusion_matrices_identical_to_source_epoch_zero': True, 'max_abs_log_probability_error': error,
        'epoch_zero_eligible': False, 'metrics': a}


def run(args, meta):
    started = time.perf_counter()
    c, base, source, old, binding = meta
    args.source = source
    _, spec, ref, nroot, _, _ = base
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
    data = core.FrozenGraphs(inside(args.repo, spec['split_directory']), ref['binding']['split_id'])
    guard_graph_loads(data, {'train', 'val'})
    saved_n0 = core.torch.load(nroot / 'N0/best.pt', map_location=device, weights_only=False)
    require(saved_n0['config']['binding'] == ref['binding'] and saved_n0['epoch'] == 5
            and saved_n0['validation']['confusion_matrix'] == ref['model']['best_validation']['confusion_matrix'],
            'N0 payload differs')
    backbone = parent.build_backbone(core, data, device, saved_n0)
    encoder_hash = core.state_hash(parent.encoder_values(backbone))
    require(encoder_hash == old['binding']['encoder_source']['encoder_sha256'], 'Source encoder parameters/BN differ')
    cache_started = time.perf_counter()
    caches = {}
    for part in ('train', 'val'):
        parent.progress(args, 'load_validate_shared_64D_cache', partition=part)
        cache = core.torch.load(source / (part + '_encoder64.pt'), map_location='cpu', weights_only=False)
        rows = core.select_rows(data, part)
        parent.validate_cache_header(cache['metadata'], part, old['binding']['encoder_source'], rows)
        require(cache['ids'] == [r['sample_id'] for r in rows] and cache['labels'].tolist() == [r['label'] for r in rows],
                'Cache samples/labels differ')
        require(cache['flow64'].shape == cache['packet64'].shape == (len(rows), 64)
                and cache['labels'].dtype == core.torch.int64
                and all(bool(core.torch.isfinite(cache[k]).all()) for k in ('flow64', 'packet64'))
                and core.state_hash([(k, cache[k]) for k in ('flow64', 'packet64', 'labels')])
                == old['binding']['cache_tensor_sha256'][part], 'Cache tensor contents differ')
        caches[part] = {k: v.to(device) if core.torch.is_tensor(v) else v for k, v in cache.items()}
    cache_seconds = time.perf_counter() - cache_started
    # Both clean files must agree, independently of the mature best checkpoint.
    initial = core.torch.load(source / 'D32_preflight.pt', map_location=device, weights_only=False)
    epoch0 = core.torch.load(source / 'D32/epoch_zero.pt', map_location=device, weights_only=False)
    require(initial['architecture'] == parent.architecture(32) and epoch0['epoch'] == 0
            and epoch0['config']['architecture'] == initial['architecture']
            and epoch0['config']['binding'] == old['binding']
            and epoch0['validation']['confusion_matrix'] == old['models']['D32']['epoch_zero']['confusion_matrix']
            and core.state_hash(initial['model_state'].items()) == core.state_hash(epoch0['model_state'].items()),
            'Clean initial snapshots disagree; refuse mature/probe weights')
    heads, Graph = load_classes()
    streams = parent.initialization_streams(42)
    expected_head = read(source / 'initialization.json')['head_state_sha256']['D32']
    models = {}
    for name, mode in VARIANTS.items():
        head = parent.make_head(core, heads[mode], 32, device, streams)
        head.fusion_mode = mode
        require(core.state_hash(head.state_dict().items()) == expected_head, 'Named initialization no longer matches source')
        graph = Graph(backbone.flow_encoder, backbone.packet_encoder, head).to(device)
        graph.load_state_dict(initial['model_state'], strict=True)
        graph.eval()
        require(core.state_hash(head.state_dict().items()) == expected_head, 'Loaded initial head differs')
        parent.assert_zero_gate_output(core, head)
        parent.assert_frozen(core, graph, encoder_hash)
        optimizer(core, graph, c)  # Validate grouping before any probes/training.
        models[name] = graph
    initialization = compare_initial(core, models['D32-Equal'].fusion, models['D32-Gate'].fusion,
                                     caches['val'], old['models']['D32']['epoch_zero'])
    initialization.update(source_clean_file='D32_preflight.pt', clean_head_sha256=expected_head,
        public_initialization_identical=True, encoder_sha256=encoder_hash,
        architectures={n: architecture(m) for n, m in VARIANTS.items()})
    write(args.output / 'initialization.json', initialization)
    args.probes = {}
    roundtrips = {}
    warm = core.Batch.from_data_list([data.load(r) for r in data.training_candidates()[:16]]).to(device)
    for name, mode in VARIANTS.items():
        parent.progress(args, 'raw_graph_gradient_probe', variant=name)
        graph = models[name]
        args.probes[name] = probe(core, graph, data, caches['train'], device, c, encoder_hash)
        require(core.state_hash(graph.fusion.state_dict().items()) == expected_head, 'Probe leaked into training start')
        path = args.output / (name + '_preflight.pt')
        core.checkpoint(path, {'model_state': graph.state_dict(), 'architecture': architecture(mode)})
        restored = core.torch.load(path, map_location=device, weights_only=False)
        fresh = parent.make_head(core, heads[mode], 32, device, streams)
        fresh.fusion_mode = mode
        replay = Graph(backbone.flow_encoder, backbone.packet_encoder, fresh).to(device)
        replay.load_state_dict(restored['model_state'], strict=True)
        replay.eval()
        with core.torch.no_grad():
            left, right = graph(warm), replay(warm)
            require(core.torch.allclose(left, right, atol=1e-6, rtol=1e-6)
                    and bool((left.argmax(-1) == right.argmax(-1)).all()), 'Initial raw checkpoint replay failed')
        parent.assert_frozen(core, replay, encoder_hash)
        roundtrips[name] = {'strict_reload': True, 'samples': 16, 'checkpoint_sha256': sha(path)}
    write(args.output / 'probes.json', args.probes)
    write(args.output / 'preflight_roundtrips.json', roundtrips)
    preflight_seconds = time.perf_counter() - started
    assert_source_unchanged(binding)
    write(args.output / 'status.json', {'status': 'running', 'phase': 'paired_training', 'tensor_preflight_passed': True})
    results, exports, replays = {}, {}, {}
    for name, mode in VARIANTS.items():
        expected = results['D32-Equal']['orders'] if mode == 'dynamic' else old['models']['D32']['orders']
        results[name], exports[name] = adapted(parent.train, mode)(
            core, args, c, models[name], caches, name, binding, encoder_hash, expected)
        common_old = min(len(results[name]['orders']), len(old['models']['D32']['orders']))
        require(results[name]['orders'][:common_old] == old['models']['D32']['orders'][:common_old]
                and results[name]['training_start_rng_sha256'] == old['models']['D32']['training_start_rng_sha256'],
                'Training RNG/order differs from original D32')
        parent.progress(args, 'full_graph_best_checkpoint_replay', variant=name)
        saved = core.torch.load(args.output / name / 'best.pt', map_location=device, weights_only=False)
        require(saved['config']['binding'] == binding and saved['config']['variant'] == name
                and sha(args.output / name / 'best.pt') == results[name]['checkpoint_sha256']['best.pt'], 'Checkpoint identity differs')
        replays[name], raw = adapted(parent.graph_replay, mode)(core, args, (heads[mode], Graph), data,
            device, saved_n0, saved, caches['val'], streams, encoder_hash)
        require([r['prediction'] for r in raw] == [r['prediction'] for r in exports[name]], 'Raw exported predictions differ')
        write(args.output / name / 'graph_replay.json', replays[name])
    common = min(len(r['orders']) for r in results.values())
    require(results['D32-Equal']['orders'][:common] == results['D32-Gate']['orders'][:common]
            and results['D32-Equal']['training_start_rng_sha256'] == results['D32-Gate']['training_start_rng_sha256'],
            'Paired order/RNG differs')
    for part, cache in caches.items():
        require(core.state_hash([(k, cache[k]) for k in ('flow64', 'packet64', 'labels')])
                == old['binding']['cache_tensor_sha256'][part], 'Shared cache mutated')
    require(core.state_hash(models['D32-Equal'].fusion.gate.state_dict().items()) == core.state_hash(
        [(k[len('fusion.gate.'):], v) for k, v in initial['model_state'].items() if k.startswith('fusion.gate.')]),
        'Unused equal gate parameters changed')
    parent.assert_frozen(core, models['D32-Equal'], encoder_hash)
    parent.assert_frozen(core, models['D32-Gate'], encoder_hash)
    assert_source_unchanged(binding)
    from summarize_fusion_gate_pair import prediction_transitions, summarize
    summary = {'status': 'completed', 'exit_code': 0, 'binding': binding, 'protocol': c, 'models': results,
        'historical_D32': old['models']['D32'],
        'Gate_versus_Equal': parent.pair.compare_predictions(exports['D32-Equal'], exports['D32-Gate']),
        'prediction_transitions': prediction_transitions(exports['D32-Equal'], exports['D32-Gate'], core.LABELS),
        'Gate_versus_historical_D32': parent.pair.compare_predictions(
            parent.predictions(source / 'D32/validation_predictions.jsonl.gz'), exports['D32-Gate']),
        'execution_comparison': execution_comparison(args, c, old, results, core),
        'common_epochs_order_checked': common, 'initialization': initialization, 'graph_replays': replays,
        'source_artifacts_unchanged': True, 'encoder_parameters_and_BN_unchanged': True,
        'shared_cache_unchanged': True, 'tensor_preflight_passed': True,
        'shared_cache_reuse_wall_seconds': cache_seconds, 'preflight_including_cache_wall_seconds': preflight_seconds,
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
    require(args.output != meta[2] and meta[2] not in args.output.parents
            and not args.output.exists(), 'Refuse overwrite/resume or source output reuse')
    if args.mode == 'check':
        print(json.dumps({'status': 'metadata_checks_passed', 'binding': meta[-1],
                          'training_started': False, 'tensor_checks_executed': False}), flush=True)
        return
    require(os.name == 'posix', 'Formal training runs on the Linux wx server only')
    args.output.mkdir(parents=True)
    write(args.output / 'config.json', meta[0])
    write(args.output / 'binding.json', meta[-1])
    write(args.output / 'status.json', {'status': 'running', 'phase': 'preflight'})
    parent.progress(args, 'load_frozen_N0_and_clean_D32')
    try:
        run(args, meta)
        write(args.output / 'status.json', {'status': 'completed', 'exit_code': 0})
        write(args.output / 'progress.json', {'status': 'completed', 'training_jobs': 2})
    except BaseException:
        write(args.output / 'status.json', {'status': 'failed', 'exit_code': 1})
        write(args.output / 'progress.json', {**read(args.output / 'progress.json'), 'status': 'failed', 'exit_code': 1})
        (args.output / 'error.log').write_text(traceback.format_exc(), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
