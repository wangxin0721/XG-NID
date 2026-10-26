"""Frozen seed42 A/C/B probe and validation screening; run on Linux wx only.

queue runs one joint probe, then A/C/B sequentially. No test inference, no hour
budget, no resume/overwrite and no changes to the historical experiment files.
"""
from __future__ import annotations

import argparse
import gc
import gzip
from itertools import islice
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from e3_gate_study import (activate_gpu, checked_gpu, guard_graph_loads, identity,
    inside, matrix_metrics, read, reference_metadata, require, runtime, sha, write)
from fusion_acb import (VARIANTS, build_model, check_config, parameter_counts,
    rng_state, update, validate_probe)
from run_e3_gate_ablation_server import diagnostic_receipt, snapshot
from diagnose_gate_candidate import class_metrics, finish_groups, group_name, groups

HERE = Path(__file__).resolve().parent
SOURCE_FILES = ('run_fusion_acb_server.py', 'fusion_acb.py', 'e3_gate_study.py',
                'run_e3_gate_ablation_server.py', 'diagnose_gate_candidate.py', 'diagnose_e3_gate_server.py')


def metadata(args):
    config, design = read(args.config), read(args.design)
    check_config(config, design)
    old_spec = read(args.reference_config)
    reference = reference_metadata(args.repo, old_spec)
    original_path = inside(args.repo, config['original_diagnostic'])
    original = diagnostic_receipt(original_path, reference)
    evidence = {}
    z_initialization = None
    for variant, paths in config['completed_candidates'].items():
        diag_path, cache_path, screen_path = [inside(args.repo, paths[k]) for k in ('diagnostic', 'cache_analysis', 'screen')]
        diag, cache, screen = read(diag_path), read(cache_path), read(screen_path)
        for path in (diag_path, cache_path, screen_path):
            status = read(path.parent / 'status.json')
            require((status.get('status'), status.get('exit_code')) == ('completed', 0), 'Evidence process incomplete: ' + str(path))
        require(diag['schema'] == 'xgnid.gate-candidate-diagnostic.v1'
                and cache['schema'] == 'xgnid.candidate-fusion-cache-analysis.v1', 'Wrong evidence type')
        for r in (diag, cache, screen):
            require(r['status'] == 'completed' and r['exit_code'] == 0 and r['variant'] == variant
                    and r['test_evaluated'] is False, 'Wrong candidate or test boundary')
        require(diag['saved_validation_reproduced'] and diag['model_state_unchanged']
                and diag['receipt']['reference'] == reference and diag['receipt']['diagnostic'] == original,
                'Candidate reference or validation identity differs')
        require(cache['samples'] == 13446 and all(cache['checks'].values()) and cache['partition'] == 'val', 'Cache not accepted')
        require(cache['report_sha256'] == {'original': sha(original_path), 'candidate': sha(diag_path)}, 'Cache source reports changed')
        require(cache['cache_sha256']['original'] == read(original_path)['models']['E3']['cache_sha256'], 'Cache uses another original E3')
        require(cache['cache_sha256']['candidate'] == diag['cache_sha256'] == sha(diag_path.parent / 'validation_cache.jsonl.gz')
                and cache['cache_sha256']['scan'] == diag['scan_sha256'] == sha(diag_path.parent / 'webbased_packet_only_scan.jsonl.gz'),
                'Candidate cache bytes changed')
        folder = screen_path.parent / variant
        metrics, cfg = read(folder / 'metrics.json'), read(folder / 'config.json')
        require(screen['model'] == metrics and cfg['binding'] == screen['binding'], 'Screen documents disagree')
        require(sha(folder / 'config.json') == metrics['config_sha256'] == diag['receipt']['config_sha256']
                and sha(folder / 'best.pt') == metrics['checkpoint_sha256']['best.pt'] == diag['receipt']['best_sha256'],
                'Candidate checkpoint/config changed')
        require(metrics['best_validation']['confusion_matrix'] == diag['metrics']['fusion']['confusion_matrix'], 'Evidence predictions differ')
        require(cfg['controls'] == old_spec['controls'] and (cfg['epochs'], cfg['workers'], cfg['evaluation_batch_size'], cfg['initialization_batch_size'])
                == (100, 2, 16, 16), 'Old candidate training controls differ')
        require(cfg['binding']['helper_sha256'] == sha(HERE / 'e3_gate_study.py')
                and cfg['binding']['entry_sha256'] == sha(HERE / 'run_e3_gate_ablation_server.py'), 'Old candidate sources differ')
        epochs_path = folder / 'epochs.jsonl'
        epochs = [json.loads(line) for line in epochs_path.read_text(encoding='utf-8').splitlines()]
        require(len(epochs) == metrics['stopped_epoch'] and [r['epoch'] for r in epochs] == list(range(1, len(epochs)+1))
                and all(len(r['train_order_sha256']) == 64 for r in epochs), 'Candidate epoch order log incomplete')
        evidence[variant] = {'diagnostic_sha256': sha(diag_path), 'cache_analysis_sha256': sha(cache_path),
                             'screen_sha256': sha(screen_path), 'best_sha256': diag['receipt']['best_sha256'],
                             'epochs_sha256': sha(epochs_path)}
        if variant == 'E3a_equal_init':
            z_initialization = metrics['initialization']
    require(set(evidence) == {'E3a', 'E3a_equal_init'}, 'Both completed candidate studies required')
    binding = {'schema': config['schema'], 'config_sha256': sha(args.config), 'design_sha256': sha(args.design),
               'reference_config_sha256': sha(args.reference_config), 'reference': reference,
               'original_diagnostic': original, 'candidate_evidence': evidence,
               'sources': {name: sha(HERE / name) for name in SOURCE_FILES}, 'seed': 42}
    return config, old_spec, binding, z_initialization


def optimizer(core, model):
    # Same parameter iteration/Adam defaults as E3a-Z; unused params have grad=None.
    return core.torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0.)


def probe(core, args, frozen, device, binding, z_initialization, gpu):
    results = {}
    rows = core.select_rows(frozen, 'train')
    for variant in VARIANTS:
        model, cert = build_model(core, frozen, device, variant, z_initialization)
        opt = optimizer(core, model)
        batches = list(islice(core.TrainBatches(len(rows), 512, 42), 3))
        loader = core.DataLoader(core.GraphRows(frozen, rows), batch_sampler=batches, num_workers=0)
        before = core.state_hash(model.named_parameters())
        gate_before = core.state_hash(model.gate.named_parameters())
        updates = []
        for index, batch in enumerate(loader):
            batch = batch.to(device)
            require(batch.num_graphs == 512, 'Probe batch must have 512 graphs')
            item = update(core, model, opt, batch, variant, probe=True, initial=index == 0)
            if index == 0:
                require(item['fusion_weight_mean'] == [.5, .5], 'Initial weights are not equal')
            item['batch'] = index + 1
            updates.append(item)
            print(json.dumps({'variant': variant, 'probe': item}), flush=True)
        changed = before != core.state_hash(model.named_parameters())
        fusion_changed = gate_before != core.state_hash(model.gate.named_parameters())
        require(changed and len(updates) == 3, 'No model update or incomplete probe')
        if variant != 'A_equal':
            require(fusion_changed and max(v or 0 for item in updates for v in item['gate_gradients'].values()) > 0,
                    'Learned fusion parameter did not update')
            if variant == 'B_dynamic':
                require(any((item['gate_gradients'].get('net.0.weight') or 0) > 0 for item in updates[1:]),
                        'Dynamic gate upstream layers never received gradients')
        else:
            require(not fusion_changed, 'Unused A gate changed')
        results[variant] = {'initialization': cert, 'parameters': parameter_counts(model, variant),
                            'updates': updates, 'optimizer_update_verified': changed,
                            'fusion_parameter_update_verified': fusion_changed,
                            'auxiliary_gradient_paths_verified': True}
        del model, opt, loader, batch
        gc.collect()
        core.torch.cuda.empty_cache()
    result = {'status': 'completed', 'exit_code': 0, 'mode': 'probe', 'binding': binding,
              'variants': results, 'hardware': gpu, 'test_evaluated': False}
    validate_probe(result, {'status': 'completed', 'exit_code': 0, 'mode': 'probe'}, binding)
    return result


def best_diagnostic(core, model, loader, rows, device, output, expected):
    """One final validation pass; no scan or fitted post-hoc weights."""
    model.eval()
    matrices = {k: [[0]*8 for _ in range(8)] for k in ('fusion', 'flow', 'packet')}
    grouped, by_class = groups(), {c: groups() for c in core.LABELS}
    count, ordered = 0, []
    state_before = core.state_hash(model.state_dict().items())
    with gzip.open(output / 'validation_predictions.jsonl.gz', 'xt', encoding='utf-8') as stream, core.torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            scores = model(batch)
            cache = model._last_cache
            logits = {'fusion': scores, 'flow': model.shared_head(cache['flow_logits']),
                      'packet': model.shared_head(cache['packet_logits'])}
            require(all(bool(core.torch.isfinite(v).all()) for v in logits.values()), 'Nonfinite diagnostic scores')
            predictions = {k: v.argmax(-1).cpu().tolist() for k, v in logits.items()}
            weights = cache['gate_weights'].cpu().tolist()
            for j, y in enumerate(batch.y.view(-1).tolist()):
                row = rows[count]
                require(row['sample_id'] == batch.sample_id[j] and row['label'] == y, 'Validation identity mismatch')
                p = {k: v[j] for k, v in predictions.items()}
                g = group_name(p['flow'] == y, p['packet'] == y)
                for k in matrices:
                    matrices[k][y][p[k]] += 1
                for target in (grouped[g], by_class[core.LABELS[y]][g]):
                    target['samples'] += 1
                    target['fusion_correct'] += int(p['fusion'] == y)
                    target['fusion_wrong'] += int(p['fusion'] != y)
                stream.write(json.dumps({'sample_id': row['sample_id'], 'node_hash': row['node_hash'],
                    'label': y, 'predictions': p, 'final_weights': weights[j], 'group': g}) + '\n')
                ordered.append((row['sample_id'], y))
                count += 1
    require(count == 13446 and matrices['fusion'] == expected['confusion_matrix'], 'Best checkpoint validation not reproduced')
    require(core.state_hash(model.state_dict().items()) == state_before, 'Diagnostic altered model state')
    return {'samples': count, 'metrics': {k: matrix_metrics(v) for k, v in matrices.items()},
            'per_class': {k: class_metrics(v, core.LABELS) for k, v in matrices.items()},
            'groups': finish_groups(grouped), 'groups_by_class': {k: finish_groups(v) for k, v in by_class.items()},
            'ordered_validation_identity_sha256': identity(ordered),
            'predictions_sha256': sha(output / 'validation_predictions.jsonl.gz'),
            'saved_validation_reproduced': True, 'test_evaluated': False}


def screen(core, args, frozen, device, binding, z_initialization, gpu):
    require(args.probe_report is not None and args.variant in VARIANTS, 'Screen needs joint probe and variant')
    report, status = read(args.probe_report), read(args.probe_report.parent / 'status.json')
    validate_probe(report, status, binding)
    z_path = inside(args.repo, read(args.config)['completed_candidates']['E3a_equal_init']['screen'])
    z_orders = [json.loads(line)['train_order_sha256'] for line in
                (z_path.parent / 'E3a_equal_init/epochs.jsonl').read_text(encoding='utf-8').splitlines()]
    model, cert = build_model(core, frozen, device, args.variant, z_initialization)
    require(cert == report['variants'][args.variant]['initialization'], 'Screen/probe initialization differs')
    opt = optimizer(core, model)
    rows = {p: core.select_rows(frozen, p) for p in ('train', 'val')}
    sampler = core.TrainBatches(len(rows['train']), 512, 42)
    kwargs = {'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'multiprocessing_context': 'fork'}
    train_loader = core.DataLoader(core.GraphRows(frozen, rows['train']), batch_sampler=sampler, **kwargs)
    val_loader = core.DataLoader(core.GraphRows(frozen, rows['val']), batch_size=16, shuffle=False, **kwargs)
    cfg = {'schema': 'xgnid.fusion-acb-run.v1', 'variant': args.variant, 'binding': binding,
           'initialization': cert, 'parameters': parameter_counts(model, args.variant),
           'probe_sha256': sha(args.probe_report), 'lambda': .05, 'batch_size': 512,
           'evaluation_batch_size': 16, 'initialization_batch_size': 16, 'seed': 42,
           'epochs': 100, 'patience': 12, 'workers': 2, 'lr': .001, 'test_evaluated': False}
    write(args.output / 'config.json', cfg)
    best, best_epoch, stale, updates, stopped = -1., 0, 0, 0, 0
    started = time.perf_counter()
    with (args.output / 'epochs.jsonl').open('x', encoding='utf-8') as log:
        for epoch in range(1, 101):
            core.sync(device)
            tick = time.perf_counter()
            core.torch.cuda.reset_peak_memory_stats(device)
            sums = {k: 0. for k in ('loss_total', 'loss_fusion', 'loss_flow', 'loss_packet')}
            count, epoch_updates = 0, 0
            for batch in train_loader:
                batch = batch.to(device)
                item = update(core, model, opt, batch, args.variant)
                for k in sums:
                    sums[k] += item[k] * batch.num_graphs
                count += batch.num_graphs
                epoch_updates += 1
            core.sync(device)
            train_seconds = time.perf_counter() - tick
            require(count == 107621 and epoch_updates == len(sampler) == 211, 'Incomplete training epoch')
            if epoch <= len(z_orders):
                require(sampler.last_order_sha256 == z_orders[epoch-1], 'Epoch data order differs from paired E3a-Z')
            updates += epoch_updates
            validation, _ = core.evaluate(model, val_loader, device)
            improved = validation['macro_f1'] > best
            if improved:
                best, best_epoch, stale = validation['macro_f1'], epoch, 0
            else:
                stale += 1
            entry = {'epoch': epoch, **{k: v/count for k, v in sums.items()}, 'train_samples': count,
                     'train_seconds': train_seconds, 'train_order_sha256': sampler.last_order_sha256,
                     'optimizer_updates': epoch_updates, 'cumulative_optimizer_updates': updates,
                     'val_accuracy': validation['accuracy'], 'val_macro_f1': validation['macro_f1'],
                     'val_weighted_f1': validation['weighted_f1'], 'gate_weights': validation['gate_weights'],
                     'best_epoch': best_epoch, 'stale_epochs': stale,
                     'peak_cuda_allocated_bytes': core.torch.cuda.max_memory_allocated(device),
                     'other_compute_processes_snapshot': snapshot(gpu)}
            payload = {'model_state': model.state_dict(), 'optimizer_state': opt.state_dict(),
                       'config': cfg, 'epoch': epoch, 'best_epoch': best_epoch, 'best_val_macro_f1': best,
                       'validation': validation, 'sampler_rng_state': sampler.generator.get_state(),
                       'rng_state': rng_state(core), 'optimizer_updates': updates, 'stale_epochs': stale}
            core.checkpoint(args.output / 'last.pt', payload)
            if improved:
                core.checkpoint(args.output / 'best.pt', payload)
            log.write(json.dumps(entry) + '\n')
            log.flush()
            print(json.dumps({'variant': args.variant, **entry}), flush=True)
            stopped = epoch
            if stale >= 12:
                break
    selected = core.torch.load(args.output / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(selected['model_state'], strict=True)
    detail = best_diagnostic(core, model, val_loader, rows['val'], device, args.output, selected['validation'])
    expected_ids = read(inside(args.repo, read(args.config)['original_diagnostic']))['models']['E3']['ordered_validation_identity_sha256']
    require(detail['ordered_validation_identity_sha256'] == expected_ids, 'Validation ordering changed')
    write(args.output / 'diagnostics.json', detail)
    metrics = {'variant': args.variant, 'best_epoch': best_epoch, 'stopped_epoch': stopped,
               'best_validation': selected['validation'], 'parameters': cfg['parameters'], 'initialization': cert,
               'optimizer_updates': updates, 'training_samples_seen': 107621 * stopped,
               'wall_seconds': time.perf_counter() - started, 'config_sha256': sha(args.output / 'config.json'),
               'checkpoint_sha256': {f: sha(args.output / f) for f in ('best.pt', 'last.pt')},
               'diagnostics_sha256': sha(args.output / 'diagnostics.json'), 'test': None, 'test_evaluated': False}
    write(args.output / 'metrics.json', metrics)
    return {'status': 'completed', 'exit_code': 0, 'mode': 'screen', 'variant': args.variant,
            'binding': binding, 'hardware': gpu, 'model': metrics, 'test_evaluated': False}


def queue(args, binding):
    base = [sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--repo', str(args.repo),
            '--config', str(args.config), '--design', str(args.design), '--reference-config', str(args.reference_config),
            '--physical-gpu', str(args.physical_gpu)]
    if args.allow_shared_gpu:
        base.append('--allow-shared-gpu')
    jobs = [('probe', ['--mode', 'probe'])] + [(v, ['--mode', 'screen', '--variant', v,
            '--probe-report', str(args.output / 'probe/summary.json')]) for v in VARIANTS]
    for name, extra in jobs:
        print(json.dumps({'starting': name, 'output': str(args.output / name)}), flush=True)
        subprocess.run(base + extra + ['--output', str(args.output / name)], check=True)
    models = {v: read(args.output / v / 'metrics.json') for v in VARIANTS}
    orders = {v: [json.loads(line)['train_order_sha256'] for line in
                  (args.output / v / 'epochs.jsonl').read_text(encoding='utf-8').splitlines()] for v in VARIANTS}
    for i in range(max(map(len, orders.values()))):
        require(len({values[i] for values in orders.values() if i < len(values)}) == 1, 'Paired epoch orders differ')
    z_path = inside(args.repo, read(args.config)['completed_candidates']['E3a_equal_init']['screen'])
    z_orders = [json.loads(line)['train_order_sha256'] for line in
                (z_path.parent / 'E3a_equal_init/epochs.jsonl').read_text(encoding='utf-8').splitlines()]
    overlap = min(len(z_orders), len(orders['B_dynamic']))
    require(overlap > 0 and z_orders[:overlap] == orders['B_dynamic'][:overlap], 'B/E3a-Z epoch orders differ')
    scores = {v: m['best_validation']['macro_f1'] for v, m in models.items()}
    return {'status': 'completed', 'exit_code': 0, 'mode': 'queue', 'binding': binding,
            'models': models, 'paired_epoch_orders_match': True, 'validation_macro_f1': scores,
            'B_lambda_zero_reference': {'model': 'E3a-Z', 'initial_state_matched': True,
                'paired_epoch_orders_match': True, 'overlapping_epochs': overlap,
                'same_forward_optimizer_and_stopping_rule': True,
                'intended_training_change': 'lambda=0 -> .05 for CE(H(z)) auxiliary supervision',
                'different_early_stopping_epochs_can_change_total_exposure': True,
                'timing_comparison_is_not_controlled': True},
            'B_minus_A_percentage_points': 100*(scores['B_dynamic']-scores['A_equal']),
            'B_minus_C_percentage_points': 100*(scores['B_dynamic']-scores['C_global']), 'test_evaluated': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--config', type=Path, default=HERE.parent / 'configs/fusion_acb_execution_20261026.json')
    parser.add_argument('--design', type=Path, default=HERE.parent / 'configs/fusion_acb_study_20261023.json')
    parser.add_argument('--reference-config', type=Path, default=HERE.parent / 'configs/e3_gate_study.json')
    parser.add_argument('--mode', choices=('check', 'probe', 'screen', 'queue'), default='check')
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--probe-report', type=Path)
    parser.add_argument('--physical-gpu', type=int, choices=(0, 1))
    parser.add_argument('--allow-shared-gpu', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    require(os.name == 'posix', 'Execute on the Linux wx server only')
    for k in ('repo', 'config', 'design', 'reference_config'):
        setattr(args, k, getattr(args, k).resolve())
    config, old_spec, binding, z_initialization = metadata(args)
    if args.mode == 'check':
        print(json.dumps({'status': 'metadata_checks_passed', 'binding': binding,
                          'gpu_touched': False, 'training_started': False}), flush=True)
        return 0
    require(args.output is not None and args.physical_gpu is not None, 'Explicit output and physical GPU required')
    args.output = inside(args.repo / 'outputs', args.output.resolve())
    require(args.output != args.repo / 'outputs', 'Use a new child output directory')
    args.output.mkdir(exist_ok=False)
    write(args.output / 'status.json', {'status': 'running', 'mode': args.mode, 'pid': os.getpid(), 'variant': args.variant})
    try:
        if args.mode == 'queue':
            result = queue(args, binding)
        else:
            gpu = checked_gpu(args.physical_gpu, args.allow_shared_gpu)
            activate_gpu(gpu)
            sys.path.insert(0, str(args.repo / 'scripts'))
            import train_grouped as core
            device = runtime(core)
            frozen = core.FrozenGraphs(inside(args.repo, old_spec['split_directory']), old_spec['split_id'])
            guard_graph_loads(frozen, {'train'} if args.mode == 'probe' else {'train', 'val'})
            result = (probe if args.mode == 'probe' else screen)(core, args, frozen, device, binding, z_initialization, gpu)
        write(args.output / 'summary.json', result)
        write(args.output / 'status.json', {'status': 'completed', 'exit_code': 0, 'mode': args.mode, 'variant': args.variant})
        print(json.dumps({'status': 'completed', 'mode': args.mode, 'output': str(args.output), 'test_evaluated': False}), flush=True)
    except BaseException as exc:
        write(args.output / 'status.json', {'status': 'failed', 'exit_code': 130 if isinstance(exc, KeyboardInterrupt) else 1,
                                          'mode': args.mode, 'variant': args.variant})
        (args.output / 'error.log').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    return 0


if __name__ == '__main__':
    sys.exit(main())
