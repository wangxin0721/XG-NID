"""Validation-only E3a/E3a-Z diagnostics. No optimizer, test inference or training.

Import and --mode check use only the standard library. Real inference is Linux
server-only. The original study/helper/runner bytes remain unchanged.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

from e3_gate_study import (activate_gpu, checked_gpu, choose_alpha, guard_graph_loads,
    identity, inside, matrix_metrics, read, reference_metadata, require, runtime, sha, write)
from diagnose_e3_gate_server import Probabilities
from run_e3_gate_ablation_server import compare_initialization, diagnostic_receipt, new_model

GROUPS = ('both_correct', 'flow_only_correct', 'packet_only_correct', 'both_wrong')


def group_name(flow_correct, packet_correct):
    return ('both_correct' if flow_correct and packet_correct else
            'flow_only_correct' if flow_correct else 'packet_only_correct' if packet_correct else 'both_wrong')


def prediction_stats(probabilities, logits, label):
    require(len(probabilities) == len(logits) == 8 and 0 <= label < 8, 'Wrong prediction dimensions')
    require(all(math.isfinite(v) for v in probabilities + logits), 'Nonfinite prediction')
    require(all(0 <= p <= 1 for p in probabilities) and abs(sum(probabilities) - 1) < 1e-5, 'Invalid probabilities')
    order = sorted(range(8), key=lambda k: (-probabilities[k], k))
    mean = sum(logits) / 8
    return {'prediction': order[0], 'top1_top2_probability_margin': probabilities[order[0]] - probabilities[order[1]],
            'true_class_probability_margin': probabilities[label] - max(p for k, p in enumerate(probabilities) if k != label),
            'strongest_wrong_class': max((k for k in range(8) if k != label), key=lambda k: probabilities[k]),
            'centered_logit_l2': math.sqrt(sum((v - mean) ** 2 for v in logits))}


def grid_runs(predictions, label):
    """Contiguous correct GRID points only; not continuous or oracle predictions."""
    require(len(predictions) == 21, 'Need 21 grid points')
    result = []
    for i, predicted in enumerate(predictions):
        if predicted == label:
            if i and predictions[i - 1] == label:
                result[-1][1] = i / 20
            else:
                result.append([i / 20, i / 20])
    return result


def groups():
    return {g: {'samples': 0, 'fusion_correct': 0, 'fusion_wrong': 0} for g in GROUPS}


def finish_groups(values):
    total = sum(v['samples'] for v in values.values())
    return {key: {**v, 'fraction_of_population': v['samples'] / total if total else None,
                  'within_group_fusion_accuracy': v['fusion_correct'] / v['samples'] if v['samples'] else None}
            for key, v in values.items()}


def class_metrics(cm, labels):
    result = {}
    for i, label in enumerate(labels):
        support, predicted, tp = sum(cm[i]), sum(row[i] for row in cm), cm[i][i]
        result[label] = {'support': support, 'true_positive': tp, 'false_negative': support - tp,
                         'false_positive': predicted - tp, 'precision': tp / predicted if predicted else 0.,
                         'recall': tp / support if support else 0.,
                         'f1': 2 * tp / (support + predicted) if support + predicted else 0.,
                         'true_class_predicted_as': dict(zip(labels, cm[i])),
                         'other_classes_predicted_here': {name: cm[j][i] for j, name in enumerate(labels) if j != i}}
    return result


def completed_candidate(summary, status, cfg, metrics, variant):
    require((summary.get('status'), summary.get('exit_code'), summary.get('mode'), summary.get('variant')) ==
            ('completed', 0, 'screen', variant), 'Candidate screening is incomplete or wrong variant')
    require((status.get('status'), status.get('exit_code'), status.get('mode'), status.get('variant')) ==
            ('completed', 0, 'screen', variant), 'Candidate process is not completed')
    require(summary['model'] == metrics and summary['binding'] == cfg['binding'], 'Candidate documents disagree')
    require(metrics['variant'] == cfg['variant'] == variant and metrics['mode'] == cfg['mode'] == 'screen', 'Wrong model config')
    require(metrics['test'] is None and metrics['test_evaluated'] is False and summary['test_evaluated'] is False,
            'Candidate screening evaluated test')
    require(metrics['initialization'] == cfg['initialization'] and metrics['evaluation_batch_size'] == 16,
            'Initialization/evaluation controls differ')


def metadata(args):
    spec = read(args.config)
    reference = reference_metadata(args.repo, spec)
    diagnostic = diagnostic_receipt(args.diagnostic_report, reference)
    root = inside(args.repo / 'outputs', args.screen_report.resolve()).parent
    summary, status = read(args.screen_report), read(root / 'status.json')
    folder = root / args.variant
    cfg, metrics = read(folder / 'config.json'), read(folder / 'metrics.json')
    completed_candidate(summary, status, cfg, metrics, args.variant)
    binding = summary['binding']
    require(binding['reference'] == reference and binding['diagnostic'] == diagnostic, 'Candidate reference/diagnostic changed')
    require(binding['variant'] == args.variant and binding['spec_sha256'] == sha(args.config), 'Study or variant differs')
    require(binding['helper_sha256'] == sha(Path(__file__).with_name('e3_gate_study.py')) and
            binding['entry_sha256'] == sha(Path(__file__).with_name('run_e3_gate_ablation_server.py')), 'Candidate training implementation differs')
    require(sha(folder / 'config.json') == metrics['config_sha256'], 'Candidate config changed')
    for name in ('best.pt', 'last.pt'):
        require(sha(folder / name) == metrics['checkpoint_sha256'][name], 'Candidate checkpoint changed: ' + name)
    require(cfg['controls'] == spec['controls'] and (cfg['epochs'], cfg['workers'], cfg['evaluation_batch_size'], cfg['initialization_batch_size']) ==
            (100, 2, 16, 16), 'Candidate controls differ')
    receipt = {'summary_sha256': sha(args.screen_report), 'config_sha256': sha(folder / 'config.json'),
               'metrics_sha256': sha(folder / 'metrics.json'), 'best_sha256': sha(folder / 'best.pt'),
               'variant': args.variant, 'reference': reference, 'diagnostic': diagnostic}
    return spec, folder, cfg, metrics, receipt


def diagnose(core, args, spec, folder, cfg, saved, receipt, gpu):
    torch = core.torch
    device = runtime(core)
    frozen = core.FrozenGraphs(inside(args.repo, spec['split_directory']), spec['split_id'])
    loads = guard_graph_loads(frozen, {'train', 'val'})
    model, cert = new_model(core, frozen, device, args.variant)
    initialization = compare_initialization(cert, saved['initialization'])
    checkpoint = torch.load(folder / 'best.pt', map_location=device, weights_only=False)
    require(checkpoint['config'] == cfg and checkpoint['epoch'] == saved['best_epoch'] and
            checkpoint['validation'] == saved['best_validation'], 'Checkpoint contents differ')
    model.load_state_dict(checkpoint['model_state'], strict=True)
    del checkpoint
    model.eval()
    before = core.state_hash(model.state_dict().items())
    rows = core.select_rows(frozen, 'val')
    loader = core.DataLoader(core.GraphRows(frozen, rows), batch_size=16, shuffle=False, num_workers=0)
    primary = ('fusion', 'flow', 'packet', 'equal')
    matrices = {key: [[0]*8 for _ in range(8)] for key in primary + tuple('alpha_%02d' % i for i in range(21))}
    calibration = {key: Probabilities() for key in primary}
    grouped, by_class = groups(), {label: groups() for label in core.LABELS}
    web_packet = {'samples': 0, 'fusion_wrong': 0, 'packet_weight_below_half': 0,
                  'packet_weight_equal_half': 0, 'packet_weight_above_half': 0,
                  'wrong_despite_packet_weight_above_half': 0}
    captured = {}
    handle = model.gate.register_forward_hook(lambda module, values, result: captured.update(raw=result))
    ordered, count = [], 0
    cache_path = args.output / 'validation_cache.jsonl.gz'
    scan_path = args.output / 'webbased_packet_only_scan.jsonl.gz'
    forward_seconds = 0.
    tick = time.perf_counter()
    try:
        with gzip.open(cache_path, 'xt', encoding='utf-8') as stream, gzip.open(scan_path, 'xt', encoding='utf-8') as scans, torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                captured.clear()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                output = model(batch)
                end.record()
                end.synchronize()
                forward_seconds += start.elapsed_time(end) / 1000
                cache = model._last_cache
                zf, zp, weights = cache['flow_logits'], cache['packet_logits'], cache['gate_weights']
                require(torch.equal(weights, captured['raw']), 'Final weights are not the pure gate')
                logits = {'fusion': model.shared_head(weights[:, :1]*zf + weights[:, 1:]*zp),
                          'flow': model.shared_head(zf), 'packet': model.shared_head(zp),
                          'equal': model.shared_head(.5*(zf+zp))}
                require(bool(torch.allclose(core.F.log_softmax(logits['fusion'], dim=-1), output, atol=1e-6, rtol=0)), 'Actual fusion path mismatch')
                for i in range(21):
                    logits['alpha_%02d' % i] = model.shared_head((i/20)*zf + (1-i/20)*zp)
                require(all(bool(torch.isfinite(v).all()) for v in logits.values()), 'Nonfinite logits')
                probs = {key: value.softmax(-1).cpu().tolist() for key, value in logits.items()}
                raw_logits = {key: logits[key].cpu().tolist() for key in primary}
                ws, zfs, zps = weights.cpu().tolist(), zf.cpu().tolist(), zp.cpu().tolist()
                for j, label in enumerate(batch.y.view(-1).tolist()):
                    row = rows[count]
                    sample_id = batch.sample_id[j]
                    require(sample_id == row['sample_id'] and label == row['label'], 'Validation identity mismatch')
                    ordered.append((sample_id, label))
                    predictions = {key: max(range(8), key=lambda k: values[j][k]) for key, values in probs.items()}
                    for key, predicted in predictions.items():
                        matrices[key][label][predicted] += 1
                    for key in primary:
                        calibration[key].add(probs[key][j], label)
                    group = group_name(predictions['flow'] == label, predictions['packet'] == label)
                    correct = predictions['fusion'] == label
                    for target in (grouped[group], by_class[core.LABELS[label]][group]):
                        target['samples'] += 1
                        target['fusion_correct'] += int(correct)
                        target['fusion_wrong'] += int(not correct)
                    w = ws[j]
                    require(len(w) == 2 and all(math.isfinite(v) and 0 <= v <= 1 for v in w) and abs(sum(w)-1)<1e-5, 'Invalid final weights')
                    grid_pred = [predictions['alpha_%02d' % i] for i in range(21)]
                    item = {'sample_id': sample_id, 'label': label, 'node_hash': row['node_hash'], 'group': group,
                            'final_weights': w, 'z_flow_representation': zfs[j], 'z_packet_representation': zps[j],
                            'probabilities': {key: probs[key][j] for key in primary},
                            'classification_statistics': {key: prediction_stats(probs[key][j], raw_logits[key][j], label) for key in primary},
                            'alpha_flow_grid_predictions': grid_pred,
                            'alpha_flow_grid_true_class_margins': [probs['alpha_%02d' % i][j][label] - max(p for k,p in enumerate(probs['alpha_%02d' % i][j]) if k != label) for i in range(21)],
                            'correct_grid_runs_not_continuous_intervals': grid_runs(grid_pred, label),
                            'label_based_scan_is_diagnostic_only': True}
                    stream.write(json.dumps(item, allow_nan=False)+'\n')
                    if core.LABELS[label] == 'WebBased' and group == 'packet_only_correct':
                        web_packet['samples'] += 1
                        web_packet['fusion_wrong'] += int(not correct)
                        suffix = 'below_half' if w[1] < .5 else 'above_half' if w[1] > .5 else 'equal_half'
                        web_packet['packet_weight_' + suffix] += 1
                        web_packet['wrong_despite_packet_weight_above_half'] += int(not correct and w[1] > .5)
                        scans.write(json.dumps(item, allow_nan=False)+'\n')
                    count += 1
                if count % 1024 == 0:
                    print(json.dumps({'variant': args.variant, 'validation_samples': count}), flush=True)
    finally:
        handle.remove()
    require(count == 13446 and loads['test'] == 0 and len(set(s for s,y in ordered)) == count, 'Incomplete/duplicate validation or test touched')
    expected_id = read(args.diagnostic_report)['models']['E3']['ordered_validation_identity_sha256']
    require(identity(ordered) == expected_id, 'Validation ordering differs from reference')
    result_metrics = {key: matrix_metrics(cm) for key,cm in matrices.items()}
    require(matrices['fusion'] == saved['best_validation']['confusion_matrix'], 'Saved validation matrix not reproduced')
    require(all(abs(result_metrics['fusion'][k]-saved['best_validation'][k]) <= 1e-10 for k in ('accuracy','macro_f1','weighted_f1')), 'Saved validation metrics not reproduced')
    require(core.state_hash(model.state_dict().items()) == before and sha(folder/'best.pt') == receipt['best_sha256'], 'Model/checkpoint changed')
    chosen = choose_alpha({i: result_metrics['alpha_%02d' % i] for i in range(21)})
    return {'schema': 'xgnid.gate-candidate-diagnostic.v1', 'status': 'completed', 'exit_code': 0,
            'variant': args.variant, 'partition': 'val', 'receipt': receipt, 'hardware': gpu,
            'initialization_comparison': initialization, 'saved_validation_reproduced': True, 'model_state_unchanged': True,
            'metrics': result_metrics, 'per_class': {key: class_metrics(matrices[key], core.LABELS) for key in primary},
            'groups': finish_groups(grouped), 'groups_by_class': {k: finish_groups(v) for k,v in by_class.items()},
            'webbased_packet_only': web_packet, 'calibration': {k:v.result() for k,v in calibration.items()},
            'selected_global_alpha_on_validation_grid': chosen/20,
            'scan_interpretation': 'H(alpha*z_flow+(1-alpha)*z_packet), 21 grid points; not continuous regions, not trained global C, no per-sample oracle metric or predictions',
            'branch_interpretation': 'H(z) interventions inside the joint model, not separately trained classifiers; z statistics are representation features',
            'parameter_count': sum(p.numel() for p in model.parameters()), 'evaluation_batch_size': 16,
            'forward_seconds_cuda_events_excluding_diagnostic_heads': forward_seconds,
            'timing_scope': 'this shared-capable run only, no isolated benchmark claim',
            'wall_seconds': time.perf_counter()-tick, 'graph_load_counts': loads,
            'ordered_validation_identity_sha256': identity(ordered), 'cache_sha256': sha(cache_path), 'scan_sha256': sha(scan_path),
            'source_sha256': {name: sha(Path(__file__).with_name(name)) for name in
                              ('diagnose_gate_candidate.py','e3_gate_study.py','diagnose_e3_gate_server.py','run_e3_gate_ablation_server.py')},
            'test_evaluated': False, 'training_started': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path.cwd())
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--screen-report', type=Path, required=True)
    p.add_argument('--diagnostic-report', type=Path, required=True)
    p.add_argument('--variant', choices=('E3a','E3a_equal_init'), required=True)
    p.add_argument('--mode', choices=('check','diagnose'), default='check')
    p.add_argument('--physical-gpu', type=int, choices=(0,1))
    p.add_argument('--allow-shared-gpu', action='store_true')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    args.repo = args.repo.resolve()
    spec, folder, cfg, saved, receipt = metadata(args)
    if args.mode == 'check':
        print(json.dumps({'status':'metadata_checks_passed','variant':args.variant,'receipt':receipt,'gpu_touched':False}, indent=2))
        return 0
    require(os.name == 'posix', 'Real diagnosis only on Linux server')
    require(args.physical_gpu is not None and args.output is not None, 'Explicit GPU and fresh output required')
    args.output = inside(args.repo/'outputs', args.output.resolve())
    require(args.output.name.startswith('e3_gate_candidate_') and not args.output.exists(), 'Use a fresh outputs/e3_gate_candidate_* directory')
    gpu = checked_gpu(args.physical_gpu, args.allow_shared_gpu)
    activate_gpu(gpu)
    args.output.mkdir()
    write(args.output/'status.json', {'status':'running','variant':args.variant,'mode':'diagnose','pid':os.getpid()})
    try:
        sys.path.insert(0, str(args.repo/'scripts'))
        import train_grouped as core
        result = diagnose(core,args,spec,folder,cfg,saved,receipt,gpu)
        write(args.output/'report.json', result)
        write(args.output/'status.json', {'status':'completed','exit_code':0,'variant':args.variant,'mode':'diagnose'})
        print(json.dumps({'status':'completed','report':str(args.output/'report.json'),'test_evaluated':False}),flush=True)
    except BaseException as exc:
        write(args.output/'status.json', {'status':'failed','exit_code':130 if isinstance(exc,KeyboardInterrupt) else 1,'variant':args.variant,'mode':'diagnose'})
        (args.output/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
        raise
    return 0


if __name__ == '__main__':
    sys.exit(main())
