"""Independent batch1024 B-H seed42 probe and validation screening; run on Linux wx only.

queue runs a disposable probe, one fresh B-H training, then offline analysis. No test inference, no hour
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
from fusion_acb import rng_state, update, parameter_counts
from fusion_bh import (VARIANT, CONTROLS, build_model, initial_equivalence,
                       probe_update, validate_probe)
from fusion_bh_b1024_metadata import metadata
VARIANTS = (VARIANT,)
from run_e3_gate_ablation_server import diagnostic_receipt, snapshot
from diagnose_gate_candidate import class_metrics, finish_groups, group_name, groups

HERE = Path(__file__).resolve().parent
SOURCE_FILES = ('run_fusion_acb_server.py', 'fusion_acb.py', 'e3_gate_study.py',
                'run_e3_gate_ablation_server.py', 'diagnose_gate_candidate.py', 'diagnose_e3_gate_server.py')


def optimizer(core, model):
    # Same parameter iteration/Adam defaults as E3a-Z; unused params have grad=None.
    return core.torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0.)


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
    z_path = inside(args.repo, read(args.config)['acb_root']) / 'B_dynamic'
    z_orders = [json.loads(line)['train_order_sha256'] for line in
                (z_path / 'epochs.jsonl').read_text(encoding='utf-8').splitlines()]
    model, cert = build_model(core, frozen, device, z_initialization)
    require(cert == report['initialization'], 'Screen/probe initialization differs')
    opt = optimizer(core, model)
    rows = {p: core.select_rows(frozen, p) for p in ('train', 'val')}
    sampler = core.TrainBatches(len(rows['train']), 1024, 42)
    kwargs = {'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'multiprocessing_context': 'fork'}
    train_loader = core.DataLoader(core.GraphRows(frozen, rows['train']), batch_sampler=sampler, **kwargs)
    val_loader = core.DataLoader(core.GraphRows(frozen, rows['val']), batch_size=16, shuffle=False, **kwargs)
    cfg = {'schema': 'xgnid.fusion-bh-run.v1', 'variant': args.variant, 'binding': binding,
           'initialization': cert, 'parameters': parameter_counts(model, 'B_dynamic'),
           'probe_sha256': sha(args.probe_report), 'lambda': .05, 'batch_size': 1024,
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
                item = update(core, model, opt, batch, 'B_dynamic')
                for k in sums:
                    sums[k] += item[k] * batch.num_graphs
                count += batch.num_graphs
                epoch_updates += 1
            core.sync(device)
            train_seconds = time.perf_counter() - tick
            require(count == 107621 and epoch_updates == len(sampler) == 106, 'Incomplete training epoch')
            if epoch <= len(z_orders):
                require(sampler.last_order_sha256 == z_orders[epoch-1], 'Epoch data order differs from paired B')
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



def probe(core, args, frozen, device, binding, expected_b, gpu):
    model, cert = build_model(core, frozen, device, expected_b)
    opt = optimizer(core, model)
    captured = {}
    def capture_input(module, values):
        captured['input'] = values[0]
    def capture_output(module, values, output):
        captured['output'] = output
    hooks = [model.gate.net[0].register_forward_pre_hook(capture_input),
             model.register_forward_hook(capture_output)]
    rows = core.select_rows(frozen, 'train')
    batches = list(islice(core.TrainBatches(len(rows),1024,42),3))
    loader = core.DataLoader(core.GraphRows(frozen,rows),batch_sampler=batches,num_workers=0)
    updates = []
    try:
        for index, batch in enumerate(loader):
            batch = batch.to(device)
            require(batch.num_graphs == 1024, 'Probe batch must contain 1024 graphs')
            if index == 0:
                audit = initial_equivalence(core,model,batch,captured)
            item = probe_update(core,model,opt,batch,captured)
            item['batch'] = index+1
            updates.append(item)
            print(json.dumps({'probe':item}),flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    require(len(updates)==3, 'Probe needs exactly three updates')
    def learns(name, items):
        return any((r['gate_gradients'].get(name) or 0)>0 and r['gate_parameter_changes'][name]>0 for r in items)
    checks = {'initial_equivalence':audit['initial_output_and_all_losses_match_B'] and audit['switch_back_restores_B'],
              'label_independence':audit['label_permutation_does_not_change_gate_or_output'],
              'four_statistics_only':audit['only_four_input_statistics_changed'],
              'auxiliary_gradients':all(bool(r['auxiliary_gradient_paths']) for r in updates),
              'gate_update':learns('net.2.weight',updates),
              'upstream_gate_update':learns('net.0.weight',updates[1:]),
              'new_H_gradient_path':any(r['H_via_gate_statistics_gradient_max']>0 for r in updates[1:])}
    result = {'status':'completed','exit_code':0,'mode':'probe','variant':VARIANT,'binding':binding,
              'initialization':cert,'audit':audit,'updates':updates,'checks':checks,'hardware':gpu,'test_evaluated':False}
    validate_probe(result,result,binding)
    return result


def queue(args,binding):
    base = [sys.executable,'-B','-u',str(Path(__file__).resolve()),'--repo',str(args.repo),
            '--config',str(args.config),'--reference-config',str(args.reference_config),
            '--physical-gpu',str(args.physical_gpu)]
    if args.allow_shared_gpu:
        base.append('--allow-shared-gpu')
    subprocess.run(base+['--mode','probe','--output',str(args.output/'probe')],check=True)
    subprocess.run(base+['--mode','screen','--probe-report',str(args.output/'probe/summary.json'),
                        '--output',str(args.output/VARIANT)],check=True)
    from analyze_fusion_bh_b1024_results import analyze
    analysis = analyze(args.repo,args.output,read(args.config),binding)
    write(args.output/'analysis.json',analysis)
    return {'status':'completed','exit_code':0,'mode':'queue','variant':VARIANT,'binding':binding,
            'model':read(args.output/VARIANT/'metrics.json'),'analysis_sha256':sha(args.output/'analysis.json'),
            'decision':analysis['decision'],'test_evaluated':False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--config', type=Path, default=HERE.parent / 'configs/fusion_bh_b1024_study_20261027.json')
    parser.add_argument('--design', type=Path, default=HERE.parent / 'configs/fusion_acb_study_20261023.json')
    parser.add_argument('--reference-config', type=Path, default=HERE.parent / 'configs/e3_gate_study.json')
    parser.add_argument('--mode', choices=('check', 'probe', 'screen', 'queue'), default='check')
    parser.add_argument('--variant', choices=VARIANTS, default=VARIANT)
    parser.add_argument('--probe-report', type=Path)
    parser.add_argument('--physical-gpu', type=int, choices=(0, 1))
    parser.add_argument('--allow-shared-gpu', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    for k in ('repo', 'config', 'design', 'reference_config'):
        setattr(args, k, getattr(args, k).resolve())
    config, old_spec, binding, z_initialization = metadata(args)
    if args.mode == 'check':
        print(json.dumps({'status': 'metadata_checks_passed', 'binding': binding,
                          'gpu_touched': False, 'training_started': False}), flush=True)
        return 0
    require(os.name == 'posix', 'Training runs only on Linux wx')
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
