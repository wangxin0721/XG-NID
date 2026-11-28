"""Run only R1-R3 for 20 epochs from locked Ordered; no new architecture or test."""
import argparse
import hashlib
import math
from pathlib import Path
import signal
import subprocess
import traceback

from filelock import FileLock
import torch

import node_training_diagnostic as diagnostic
import extract_node_training_trajectory as historical

parent, training = diagnostic.parent, diagnostic.training
raw, require, archive = diagnostic.raw, diagnostic.require, diagnostic.archive
OUTPUT = 'node_training_diagnostic_v1'
CONFIG = raw.ROOT/('configs/'+OUTPUT+'.json')


def lock(repo, root, download):
    settings = raw.read(CONFIG)
    fixed = {'backbone_seed': 42, 'sampling_seed': 42, 'new_module_seed': 42,
             'batch_size': 512, 'eval_batch_size': 16, 'training_samples': 107621, 'validation_samples': 13446,
             'encoder_learning_rate': 1e-7, 'readout_learning_rate': 1e-5,
             'fusion_learning_rate': 1e-5, 'ordered_learning_rate': 1e-4, 'new_learning_rate': 1e-4,
             'diagnostic_samples_per_class': 64, 'diagnostic_sampling_seed': 420064,
             'extension_gain_pp': .05, 'loss': 'plain_final_eight_class_NLL',
             'adam_betas': [.9,.999], 'adam_eps': 1e-8, 'weight_decay': 0.}
    require(all(settings.get(k) == v for k,v in fixed.items()), 'Registered diagnostic settings differ')
    require(settings['epochs'] == 20 and settings['arms'] == list(diagnostic.ARMS)
            and settings['updates_per_arm'] == 4220 and settings['total_updates'] == 12660
            and not settings['test_access'] and not settings['automatic_extension'], 'Registered diagnostic protocol differs')
    _, source_node_protocol, old, source_protocol, inputs, manifest = parent.lock(repo, root/'source_node_protocol', download)
    names = ('configs/node_training_diagnostic_v1.json', 'scripts/node_training_diagnostic.py',
             'scripts/run_node_training_diagnostic.py', 'scripts/audit_node_training_diagnostic.py',
             'scripts/extract_node_training_trajectory.py', 'scripts/launch_node_training_diagnostic.py',
             'scripts/launch_node_training_diagnostic.sh', 'scripts/flow_packet_node_training.py')
    hashes = {n: hashlib.sha256((raw.ROOT/n).read_bytes().replace(b'\r\n', b'\n')).hexdigest() for n in names}
    protocol = {'settings': settings, 'code_commit': source_node_protocol['code_commit'],
                'source_node_protocol_sha256': archive.canonical_sha(source_node_protocol),
                'files_canonical_sha256': hashes,
                'validation_cohort_sha256': manifest['validation_cohort_sha256'], 'test_access': False}
    parent.sources.immutable(root/'config.json', settings)
    parent.sources.immutable(root/'protocol_lock.json', protocol)
    check = subprocess.run(['git', 'cat-file', '-e', historical.COMMIT+':'+historical.PATH], cwd=repo,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if check.returncode:
        require(download, 'Fetch the exact historical result commit first')
        subprocess.run(['git', 'fetch', '--no-tags', 'origin', historical.COMMIT], cwd=repo, check=True)
    trajectory = historical.extract(repo, root/'historical')
    expected = raw.read(raw.ROOT/'docs/node_training_diagnostic_v1/archive_trajectory/trajectory.json')
    require(trajectory == expected, 'Historical trajectory bytes/values differ')
    return settings, protocol, old, source_protocol, inputs, manifest


def run(repo, data_root, root, download=True, resume=True, lock_only=False, probe_only=False):
    settings, protocol, old, source_protocol, inputs, manifest = lock(repo, root, download)
    if lock_only:
        raw.write(root/'status.json', {'status': 'source_locked', 'test_evaluated': False})
        return
    if (root/'artifact_manifest.json').exists():
        require(resume, 'Completed diagnostic exists; use resume')
        archive.verify_sealed(root)
        return
    context = parent.sources.context(repo, data_root, root/'source_node_protocol'/'sources', old, source_protocol, inputs, 42)
    _, source, references, data, rows, device, _, paths, binding, before, guard = context
    models, reference = diagnostic.make_models(source, references['joint_ordered'], settings, device)
    initial = {n: training.transaction.cpu_state(m) for n, m in models.items()}
    subset = diagnostic.diagnostic_subset(rows['train'], settings)
    parent.sources.immutable(root/'diagnostic_subset.json', subset)
    binding = {**binding, 'diagnostic_protocol_sha256': archive.canonical_sha(protocol),
               'diagnostic_subset_sha256': archive.canonical_sha(subset), 'arms': list(models)}
    parent.sources.immutable(root/'provenance.json', binding)
    # One sample per class for disposable gradient tests; all eight classes present.
    probe_rows = [next(r for r in subset if r['label'] == label) for label in range(8)]
    diagnostic.gradient_probe(models, data, probe_rows, settings, device, root)
    epoch0, zero = diagnostic.observe(models, reference, initial, data, rows, subset, settings, device, root, 0)
    spec = manifest['seeds']['42']['models']['joint_ordered']
    expected = parent.sources.locked.stripped(inputs.rows(spec['validation_predictions_path']))
    for name in ('Ordered', *models):
        actual = [{'sample_id': r['sample_id'], 'label': r['label'], 'prediction': r['predictions'][name]} for r in epoch0]
        parent.sources.fixed_test.compare_replay(actual, expected, root, name+'_source_replay.json')
    frozen_initial = {n: training.frozen_digest(m) for n, m in models.items()}
    reference_digest = raw.state_digest(reference.state_dict())
    raw.write(root/'server_probe.json', {'passed': True, 'epoch0_prediction_replay': True,
              'adjustment_gradient_through_frozen_GNN': True, 'test_graph_loads': guard['test'],
              'trainable_parameters': {n: sum(p.numel() for p in m.parameters() if p.requires_grad) for n,m in models.items()},
              'learning_rates': {n: {g['name']: g['lr'] for g in diagnostic.optimizer(m, settings).param_groups} for n,m in models.items()},
              'optimizers_start_fresh_from_Ordered': True})
    if probe_only:
        raw.write(root/'status.json', {'status': 'probe_passed', 'test_evaluated': False})
        return
    raw.write(root/'status.json', {'status': 'training', 'test_evaluated': False})
    def measure(current, epoch):
        return diagnostic.observe(current, reference, initial, data, rows, subset, settings, device, root, epoch)
    state = training.fit(models, data, rows, settings, binding, device, root, resume=resume,
                         optimizer_fn=diagnostic.optimizer, evaluate_fn=measure)
    require(state['completed_epoch'] == settings['epochs'] and state['updates'] == settings['updates_per_arm'], 'Diagnostic budget incomplete')
    for n, m in models.items():
        require(training.frozen_digest(m) == frozen_initial[n], 'Original frozen state changed')
        require(all(int(v['step']) == settings['updates_per_arm'] for v in state['optimizers'][n]['state'].values()), 'Adam updates differ')
        m.load_state_dict(state['best'][n]['state'], strict=True)
    selected, metrics = diagnostic.observe(models, reference, initial, data, rows, subset, settings, device, root, 'selected')
    for n in models:
        require([r['predictions'][n] for r in selected] == state['best'][n]['predictions'], 'Selected raw replay differs')
        previous = state['best'][n]['validation']
        training.metrics.check_metric(metrics[n], previous)
        require(all(math.isclose(metrics[n][key], previous[key], abs_tol=1e-6, rel_tol=1e-5)
                    for key in ('nll', 'fixed_train_nll'))
                and metrics[n]['parameter_delta'] == previous['parameter_delta'],
                'Selected observations differ from selection epoch')
    source_metric = diagnostic.record_metrics(selected, 'Ordered')
    best = {'Ordered': source_metric, **metrics}
    historical_rows = raw.read(root/'historical/trajectory.json')['selected']
    old_by_epoch = {r['epoch']: r for r in historical_rows}
    r1_deltas = {str(e): 100*(state['history'][e]['validation']['R1']['macro_f1']-
                             old_by_epoch[e]['validation_macro_f1']['S1']) for e in (1, 2, 5, 10, 20)}
    summary = {'schema': 'xgnid.node-diagnostic-results.v1', 'settings': settings,
               'best': best, 'best_epochs': {n: state['best'][n]['epoch'] for n in models},
               'last': state['history'][-1]['validation'], 'updates_each': state['updates'],
               'decision': diagnostic.decision(best, settings), 'R1_minus_historical_S1_pp': r1_deltas,
               'R1_comparison_note': 'Matched epochs and source; no preregistered numerical definition of clearly reduced degradation, so inspect trajectories without an invented binary rule.',
               'test_evaluated': False, 'validation_samples': len(selected), 'diagnostic_samples': len(subset)}
    raw.save_predictions(root/'selected_validation.jsonl.gz', selected)
    raw.write(root/'summary.json', summary)
    lines = ['# 20轮训练策略诊断', '', '| 组别 | 最佳轮 | 最佳验证 Macro-F1 % | 验证 NLL | 固定训练子集 NLL |',
             '|---|---:|---:|---:|---:|']
    for n in ('Ordered', *models):
        metric = best[n]
        lines.append(f"| {n} | {summary['best_epochs'].get(n, 0)} | {100*metric['macro_f1']:.9f} | {metric['nll']:.9f} | {metrics[n]['fixed_train_nll'] if n in metrics else raw.read(root/'observations/epoch_0/Ordered.json')['fixed_train_nll']:.9f} |")
    lines += ['', 'R3延长投入门槛：'+str(summary['decision']), '',
              'R1与旧S1相同轮次的差值（pp）：'+str(r1_deltas),
              '结构比较以R3−R2为主；R1仅诊断训练策略。20轮未通过不等于方法永久无效。无自动扩展、无测试集访问。',
              '全部逐类指标、修复/退步计数、NLL和参数变化见history.json与observations/。']
    (root/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
    parent.sources.locked.verify_frozen(source, references, before)
    parent.frozen.evaluation.verify_inputs(paths)
    require(raw.state_digest(reference.state_dict()) == reference_digest and guard['test'] == 0, 'Source changed or test loaded')
    raw.write(root/'final_verification.json', {'passed': True, 'updates_each': settings['updates_per_arm'], 'total_updates': settings['total_updates'],
              'selected_raw_replay': True, 'test_graph_loads': 0, 'frozen_parameters_and_BN_unchanged': True})
    raw.write(root/'status.json', {'status': 'completed', 'test_evaluated': False})
    files = {p.relative_to(root).as_posix(): raw.sha(p) for p in sorted(root.rglob('*'))
             if p.is_file() and p.relative_to(root).parts[0] != 'publication'
             and p.name != 'artifact_manifest.json' and not p.name.endswith(('.tmp', '.lock', '.pid'))}
    raw.write(root/'artifact_manifest.json', {'files_sha256': files})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path.cwd())
    p.add_argument('--data-root', type=Path)
    p.add_argument('--resume', choices=('auto', 'never'), default='auto')
    p.add_argument('--download-inputs', action='store_true')
    g = p.add_mutually_exclusive_group()
    g.add_argument('--lock-only', action='store_true')
    g.add_argument('--probe-only', action='store_true')
    a = p.parse_args()
    root = a.repo.resolve()/'outputs'/OUTPUT
    root.mkdir(parents=True, exist_ok=True)
    (a.repo/'tmp').mkdir(exist_ok=True)
    def stop(signum, frame):
        raise KeyboardInterrupt('signal '+str(signum))
    signal.signal(signal.SIGTERM, stop)
    with FileLock(str(a.repo/'tmp'/(OUTPUT+'.runner.lock')), timeout=0):
        try:
            run(a.repo.resolve(), a.data_root, root, a.download_inputs, a.resume == 'auto', a.lock_only, a.probe_only)
        except BaseException as exc:
            raw.write(root/'failure.json', {'exception': repr(exc), 'traceback': traceback.format_exc(), 'test_evaluated': False})
            raise


if __name__ == '__main__':
    main()
