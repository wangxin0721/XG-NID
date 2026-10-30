"""Read-only last-checkpoint interventions after a failed residual screen.

Four fixed validation replays; no optimizer, backward, training or selection.
All reports go to a new directory, never into the completed experiment.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

from e3_gate_study import (activate_gpu, checked_gpu, guard_graph_loads, inside,
                           read, require, runtime, sha, write)
import run_flow_stats_residual as screen

CASES = ('last_full', 'last_stats_off', 'last_with_start_BN', 'start_with_last_stats')


def is_encoder_bn_buffer(name):
    return name.startswith(('flow_encoder.', 'packet_encoder.')) and name.endswith(
        ('.running_mean', '.running_var', '.num_batches_tracked'))


def compose_state(start, trained, case):
    """Change only the explicitly named buffers/module, keeping all state keys."""
    require(case in CASES and set(start) == set(trained), 'Wrong case/state structure')
    if case == 'start_with_last_stats':
        state = dict(start)
        keys = [k for k in start if k.startswith('flow_stats_mlp.')]
        require(len(keys) == 4, 'Expected two-layer statistics MLP')
        state.update({k: trained[k] for k in keys})
    else:
        state = dict(trained)
        if case == 'last_with_start_BN':
            keys = [k for k in start if is_encoder_bn_buffer(k)]
            require(keys and len(keys) % 3 == 0, 'Missing encoder BN buffers')
            state.update({k: start[k] for k in keys})
    return state


def source_receipt(args):
    meta = screen.metadata(args)
    c = meta[0]
    root = inside(args.repo / 'outputs', inside(args.repo, args.training_output))
    require(read(root / 'status.json') == {'status': 'completed', 'exit_code': 0},
            'Source experiment is incomplete')
    summary = read(root / 'summary.json')
    require(summary['status'] == 'completed' and summary['exit_code'] == 0
            and summary['binding'] == meta[-1] and summary['protocol'] == c
            and summary['test_evaluated'] is False and summary['tensor_preflight_passed'] is True,
            'Source binding/config/preflight differs from delivered experiment')
    require(read(root / 'config.json') == c and read(root / 'binding.json') == meta[-1],
            'Source config/binding files disagree')
    folder = root / screen.VARIANT
    require(read(folder / 'metrics.json') == summary['candidate'], 'Source metrics disagree')
    screen.history_evidence(folder, summary['candidate'])
    for name, digest in summary['artifacts'].items():
        require(sha(inside(folder, name)) == digest, 'Source artifact changed: ' + name)
    require(summary['artifacts']['best.pt'] == summary['candidate']['checkpoint_sha256'],
            'Source best checkpoint identity differs')
    return meta, root, summary, {'training_directory': str(root),
        'training_summary_sha256': sha(root / 'summary.json'),
        'last_sha256': summary['artifacts']['last.pt'],
        'best_sha256': summary['artifacts']['best.pt'],
        'diagnostic_source_sha256': sha(Path(__file__)), 'source_files_will_be_modified': False}


def metrics_row(name, m, diag):
    return {'case': name, 'macro_f1': m['macro_f1'], 'accuracy': m['accuracy'],
            'errors': m['samples'] - sum(m['confusion_matrix'][i][i] for i in range(8)),
            'mean_residual_to_graph_norm': diag['distributions']['residual_to_graph_norm']['mean'],
            'p95_residual_to_graph_norm': diag['distributions']['residual_to_graph_norm']['quantiles']['0.95'],
            'per_class': m['per_class']}


def report(output, summary):
    lines = ['| replay | validation Macro-F1 | errors | mean residual/graph norm | p95 ratio |',
             '|---|---:|---:|---:|---:|']
    for row in summary['rows']:
        lines.append(f"| {row['case']} | {100*row['macro_f1']:.6f}% | {row['errors']} | "
                     f"{row['mean_residual_to_graph_norm']:.6g} | {row['p95_residual_to_graph_norm']:.6g} |")
    lines += ['', f"Source selected epoch {summary['selected_epoch']}; diagnosing trained epoch {summary['last_epoch']}.",
              f"Frozen V2: {100*summary['frozen_V2_macro_f1']:.6f}%; lower-LR J2: {100*summary['lower_lr_J2_macro_f1']:.6f}%.",
              'last_stats_off: keep last parameters/BN buffers, disable residual only.',
              'last_with_start_BN: keep last parameters/residual; restore only encoder BN buffers from the V2 start.',
              'start_with_last_stats: keep V2 public parameters/BN; attach the last trained statistics MLP.',
              'These are fixed inference interventions, not separately trained ablations or candidate selection.',
              'A norm ratio alone cannot prove harmful magnitude. Inspect output/class changes together.',
              'No training, no backward, no test evaluation, no automatic followup. Source files unchanged.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return '\n'.join(lines)


def run(args, receipt):
    meta, training_root, source, binding = receipt
    _, registry, refs, stage, gate_report, _, _, _ = meta
    gpu = checked_gpu(args.physical_gpu, args.allow_shared_gpu)
    activate_gpu(gpu)
    server_scripts = str(args.repo / 'scripts')
    sys.path.insert(0, server_scripts)
    try:
        import train_grouped as core
    finally:
        sys.path.remove(server_scripts)
    device = runtime(core)
    write(args.output / 'gpu.json', gpu)
    data = core.FrozenGraphs(inside(args.repo, registry['split_directory']), registry['split_id'])
    guard_graph_loads(data, {'train', 'val'})
    rows = core.select_rows(data, 'val')
    loader = core.DataLoader(core.GraphRows(data, rows), batch_size=16, shuffle=False, num_workers=2)
    n0 = core.torch.load(inside(args.repo, refs['42']['directory']) / 'best.pt', map_location='cpu', weights_only=False)
    gate = core.torch.load(stage / 'V2/best.pt', map_location='cpu', weights_only=False)
    require(gate['epoch'] == gate_report['metrics']['best_epoch'], 'Wrong gate epoch')
    model = screen.build(core, data, device, n0, gate['fusion_state'], screen.candidate_class())
    start = {n: t.detach().cpu().clone() for n, t in model.state_dict().items()}
    folder = training_root / screen.VARIANT
    last = core.torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)
    require(last['epoch'] == source['candidate']['epochs']
            and last['config']['binding'] == source['binding']
            and last['config']['protocol'] == source['protocol']
            and last['config']['variant'] == screen.VARIANT
            and last['validation']['confusion_matrix'] == source['candidate']['last_validation']['confusion_matrix'],
            'Last checkpoint payload/config differs')
    model.requires_grad_(False)
    results, table = {}, []
    started = time.perf_counter()
    for case in CASES:
        state = compose_state(start, last['model_state'], case)
        model.load_state_dict(state, strict=True)
        model.flow_stats_enabled = case != 'last_stats_off'
        model.eval()
        before = core.state_hash(model.state_dict().items())
        write(args.output / 'progress.json', {'status': 'running', 'phase': 'validation_replay', 'case': case})
        print(json.dumps({'phase': 'validation_replay', 'case': case}), flush=True)
        m, exports, diag = screen.diagnostics(core, model, loader, device, rows)
        require(core.state_hash(model.state_dict().items()) == before, 'Replay mutated model state')
        if case == 'last_full':
            require(m['confusion_matrix'] == last['validation']['confusion_matrix']
                    and abs(m['macro_f1'] - last['validation']['macro_f1']) < 1e-12,
                    'Last checkpoint validation failed to reproduce')
        case_root = args.output / case
        case_root.mkdir()
        screen.pair.save_predictions(case_root / 'validation_diagnostics.jsonl.gz', exports)
        write(case_root / 'metrics.json', m)
        write(case_root / 'diagnostics.json', diag)
        results[case] = {'metrics': m, 'diagnostics': diag,
                         'versus_frozen_V2': screen.pair.compare_predictions(
                             screen.predictions(stage / 'V2/validation_predictions.jsonl.gz'), exports)}
        table.append(metrics_row(case, m, diag))
    # Re-check original artifacts after every replay. No write goes to that tree.
    for name, digest in source['artifacts'].items():
        require(sha(inside(folder, name)) == digest, 'Source artifact changed during diagnosis')
    require(sha(training_root / 'summary.json') == binding['training_summary_sha256'], 'Source summary changed')
    summary = {'status': 'completed', 'exit_code': 0, 'binding': binding, 'results': results, 'rows': table,
        'selected_epoch': source['candidate']['best_epoch'], 'last_epoch': last['epoch'],
        'frozen_V2_macro_f1': source['frozen_V2']['best_validation']['macro_f1'],
        'lower_lr_J2_macro_f1': source['reference_joint']['J2']['best_validation']['macro_f1'],
        'preflight_gradient_receipt': read(training_root / 'probe.json'),
        'diagnostic_wall_seconds': time.perf_counter() - started,
        'training_updates': 0, 'test_evaluated': False, 'source_artifacts_unchanged': True,
        'automatic_followup': False}
    write(args.output / 'summary.json', summary)
    print(report(args.output, summary), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path.cwd())
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--training-output', type=Path, default=Path('outputs/flow_stats_residual_joint_s42_v1'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=('check', 'run'), default='check')
    p.add_argument('--physical-gpu', type=int, default=1)
    p.add_argument('--allow-shared-gpu', action='store_true')
    args = p.parse_args()
    args.repo = args.repo.resolve()
    args.config = (args.repo / args.config).resolve()
    receipt = source_receipt(args)
    if args.mode == 'check':
        print(json.dumps({'status': 'source_metadata_checks_passed', 'binding': receipt[-1],
                          'tensor_checks_executed': False, 'training_started': False}), flush=True)
        return
    require(os.name == 'posix', 'Replay runs on the original Linux wx server')
    args.output = inside(args.repo / 'outputs', inside(args.repo, args.output))
    require(args.output != receipt[1] and receipt[1] not in args.output.parents,
            'Diagnostic output must be outside the completed training directory')
    require(not args.output.exists(), 'Refuse diagnostic output overwrite')
    args.output.mkdir(parents=True)
    write(args.output / 'binding.json', receipt[-1])
    write(args.output / 'status.json', {'status': 'running'})
    try:
        run(args, receipt)
        write(args.output / 'status.json', {'status': 'completed', 'exit_code': 0})
        write(args.output / 'progress.json', {'status': 'completed', 'training_updates': 0})
    except BaseException:
        write(args.output / 'status.json', {'status': 'failed', 'exit_code': 1})
        (args.output / 'error.log').write_text(traceback.format_exc(), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
