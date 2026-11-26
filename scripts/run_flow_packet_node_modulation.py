"""Seed42 S0-S3 screen on raw graphs; no test loading or automatic expansion."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import traceback

from filelock import FileLock
import torch
from torch.nn import functional as F
from torch_geometric.data import Batch

import prepare_conditional_complement as sources
import flow_packet_node_training as training
from xgnid.flow_packet_node_modulation import ARMS, FlowPacketNodeClassifier

raw, frozen, require, archive = sources.raw, sources.frozen, sources.require, sources.archive
OUTPUT = 'flow_packet_node_modulation_v1'
CONFIG = raw.ROOT / ('configs/' + OUTPUT + '.json')


def lock(repo, root, download):
    settings = raw.read(CONFIG)
    require(settings['arms'] == list(ARMS) and settings['trained_arms'] == ['S1', 'S2', 'S3']
            and settings['test_access'] is False and settings['automatic_replication'] is False,
            'Experiment scope differs')
    require(settings['epochs'] == 200 and settings['batch_size'] == 512
            and settings['updates_per_arm'] == 42200 and settings['new_parameters_each'] == 3008,
            'Registered budget differs')
    original = raw.read(raw.ROOT / settings['source_settings'])
    inputs, source_protocol, manifest = sources.lock(repo, root / 'sources', original, download)
    # Include the full tracked Python dependency closure, not just the new modules.
    names = subprocess.check_output(['git', 'ls-files', 'src', 'scripts', 'configs'],
                                    cwd=raw.ROOT, text=True).splitlines()
    names = sorted(set(names) | {'src/xgnid/flow_packet_node_modulation.py',
        'scripts/flow_packet_node_training.py', 'scripts/run_flow_packet_node_modulation.py',
        'scripts/launch_flow_packet_node_modulation.py', 'scripts/audit_flow_packet_node_archive.py',
        'configs/flow_packet_node_modulation_v1.json'})
    hashes = {n: hashlib.sha256((raw.ROOT / n).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
              for n in names if n.endswith(('.py', '.json'))}
    protocol = {'schema': 'xgnid.node-protocol.v1', 'settings': settings,
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=raw.ROOT, text=True).strip(),
        'source_files_canonical_sha256': hashes,
        'source_protocol_sha256': archive.canonical_sha(source_protocol),
        'source_checkpoint': original['seeds']['42']['checkpoints']['joint_ordered'],
        'source_binding': original['seeds']['42']['source_binding'],
        'validation_cohort_sha256': manifest['validation_cohort_sha256'],
        'scope': 'seed42_development_validation_only', 'test_access': False}
    sources.immutable(root / 'config.json', settings)
    sources.immutable(root / 'protocol_lock.json', protocol)
    return settings, protocol, original, source_protocol, inputs, manifest


def make_models(source, ordered, device, settings):
    models = {n: FlowPacketNodeClassifier(source, ordered, n, settings['new_module_seed']).to(device)
              for n in ARMS}
    counts = {n: {'trainable': sum(p.numel() for p in m.parameters() if p.requires_grad),
                  'new': sum(p.numel() for p in m.adjustment.parameters()) if m.adjustment else 0,
                  'inference': sum(p.numel() for p in m.parameters())} for n, m in models.items()}
    require(counts['S0']['trainable'] == 0 and counts['S2']['new'] == counts['S3']['new'] == 3008
            and counts['S2']['trainable'] == counts['S3']['trainable'] == counts['S1']['trainable'] + 3008,
            'Parameter matching differs')
    return models, counts


def probe(models, source, ordered, data, rows, settings, device, root):
    expected = rows[:min(16, len(rows))]
    batch = Batch.from_data_list([data.load(r) for r in expected])
    training.identity(batch, expected)
    links = training.topology(batch, device)
    batch = batch.to(device)
    features, _ = frozen.encode(source, batch, device)
    with torch.no_grad():
        reference = ordered(features)[0]
        errors = {}
        for n, m in models.items():
            m.eval()
            scores = m(batch, links)
            errors[n] = float((scores - reference).abs().max())
            require(torch.allclose(scores, reference, atol=1e-6, rtol=1e-6)
                    and torch.equal(scores.argmax(-1), reference.argmax(-1)), 'Raw epoch0 mismatch: ' + n)
    gradients = {}
    for n in ('S1', 'S2', 'S3'):
        m = deepcopy(models[n]).train()
        before = training.frozen_digest(m)
        opt = training.optimizer(m, settings)
        steps = []
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            loss = F.nll_loss(m(batch, links), batch.y.view(-1))
            loss.backward()
            norms = {g['name']: sum(float(p.grad.square().sum()) for p in g['params'] if p.grad is not None)**.5
                     for g in opt.param_groups}
            require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in m.parameters()),
                    'Nonfinite probe gradient')
            require(norms['packet2'] > 0 and bool(torch.isfinite(loss)), 'Missing Packet2 gradient')
            if m.adjustment is not None:
                require(norms['adjustment'] > 0, 'Missing adjustment gradient')
            steps.append(norms)
            opt.step()
        require(training.frozen_digest(m) == before, 'Probe changed frozen weights/BN')
        gradients[n] = steps
    raw.write(root / 'gradient_probe.json', {'passed': True, 'raw_epoch0_max_errors': errors,
              'two_discarded_updates': gradients, 'frozen_weights_and_BN_unchanged': True})


def summarize(records, metadata, settings):
    measured = {n: training.metrics.metrics(records, n) for n in ARMS}
    comparisons = {}
    for after, before in (('S1', 'S0'), ('S2', 'S1'), ('S3', 'S0'), ('S3', 'S1'), ('S3', 'S2')):
        per_class = {label: {'repaired': 0, 'regressed': 0, 'wrong_changed': 0}
                     for label in training.metrics.LABELS}
        for r in records:
            a, b, y = r['predictions'][after], r['predictions'][before], r['label']
            if a != b:
                kind = 'repaired' if a == y else 'regressed' if b == y else 'wrong_changed'
                per_class[training.metrics.LABELS[y]][kind] += 1
        comparisons[after + '-' + before] = {
            'delta_macro_f1_pp': 100 * (measured[after]['macro_f1'] - measured[before]['macro_f1']),
            'counts': {k: sum(v[k] for v in per_class.values()) for k in ('repaired', 'regressed', 'wrong_changed')},
            'per_class': per_class}
    passed = all(measured['S3']['macro_f1'] > measured[n]['macro_f1'] for n in ('S1', 'S2'))
    return {'schema': 'xgnid.node-results.v1', 'backbone_seed': 42, 'partition': 'val',
            'samples': len(records), 'metrics': measured, 'comparisons': comparisons, 'models': metadata,
            'decision': {'S3_exceeds_both_controls': passed,
                         'interpretation': 'preliminary_seed42_evidence' if passed else 'no_extra_benefit_demonstrated',
                         'automatic_replication': False, 'test_authorized': False},
            'test_evaluated': False, 'settings': settings}


def run(repo, data_root, root, download=False, resume=True, lock_only=False, probe_only=False):
    settings, protocol, old, source_protocol, inputs, manifest = lock(repo, root, download)
    if lock_only:
        raw.write(root / 'status.json', {'status': 'source_locked', 'test_evaluated': False})
        return
    if (root / 'artifact_manifest.json').exists():
        require(resume, 'Completed run exists; use --resume auto')
        archive.verify_sealed(root)
        return
    context = sources.context(repo, data_root, root / 'sources', old, source_protocol, inputs, 42)
    _, source, refs, data, rows, device, config, paths, binding, before, guard = context
    models, costs = make_models(source, refs['joint_ordered'], device, settings)
    binding = {**binding, 'node_protocol_sha256': archive.canonical_sha(protocol)}
    sources.immutable(root / 'provenance.json', binding)
    probe(models, source, refs['joint_ordered'], data, rows['train'], settings, device, root)
    # All validation identities and initial predictions must replay the locked Ordered.
    initial, _ = training.evaluate(models, data, rows['val'], settings, device, root, 'full_epoch0_replay')
    spec = manifest['seeds']['42']['models']['joint_ordered']
    expected = sources.locked.stripped(inputs.rows(spec['validation_predictions_path']))
    for name in ARMS:
        actual = [dict(sample_id=r['sample_id'], label=r['label'], prediction=r['predictions'][name]) for r in initial]
        sources.fixed_test.compare_replay(actual, expected, root, name + '_source_replay.json')
    raw.write(root / 'server_probe.json', {'passed': True, 'samples': len(initial),
              'raw_validation_replay': True, 'parameter_counts': costs, 'test_graph_loads': guard['test']})
    sources.locked.verify_frozen(source, refs, before)
    frozen.evaluation.verify_inputs(paths)
    if probe_only:
        raw.write(root / 'status.json', {'status': 'probe_passed', 'test_evaluated': False})
        return
    raw.write(root / 'status.json', {'status': 'training', 'test_evaluated': False})
    trained = {n: models[n] for n in settings['trained_arms']}
    state = training.fit(trained, data, rows, settings, binding, device, root, resume=resume)
    sampler = training.transaction.PairedSampler(settings['training_samples'], settings['batch_size'], settings['sampling_seed'])
    require(state['updates'] == settings['updates_per_arm'] and state['completed_epoch'] == settings['epochs'],
            'Formal training budget incomplete')
    require([r['epoch'] for r in state['history']] == list(range(settings['epochs'] + 1)), 'Missing validation epochs')
    for row in state['history'][1:]:
        sampler.start(row['epoch'])
        require(row['order_sha256'] == sampler.order_sha256 and row['samples_per_arm'] == sampler.size
                and row['updates_per_arm'] == len(sampler), 'Paired sample/update order differs')
        sampler.cursor = sampler.size
    metadata = {'S0': {**costs['S0'], 'best_epoch': 0, 'updates': 0}}
    frozen_before = {n: training.frozen_digest(m) for n, m in trained.items()}
    for n, m in trained.items():
        best = max(state['history'], key=lambda r: r['validation'][n]['macro_f1'])
        require(state['best'][n]['epoch'] == best['epoch'], 'Earliest best selection differs')
        require(all(int(v['step']) == settings['updates_per_arm'] for v in state['optimizers'][n]['state'].values()),
                'Final Adam update count differs')
        m.load_state_dict(state['best'][n]['state'], strict=True)
        require(training.frozen_digest(m) == frozen_before[n], 'Best checkpoint changed frozen weights/BN')
        metadata[n] = {**costs[n], 'best_epoch': best['epoch'], 'updates': state['updates'],
                       'last_validation': state['history'][-1]['validation'][n],
                       'training_seconds': state['training_seconds'][n]}
    records, _ = training.evaluate(models, data, rows['val'], settings, device, root, 'selected_raw_replay')
    require([r['predictions']['S0'] for r in records] == [r['predictions']['S0'] for r in initial], 'S0 changed')
    for n in trained:
        require([r['predictions'][n] for r in records] == state['best'][n]['predictions'], 'Selected replay differs')
    sources.locked.verify_frozen(source, refs, before)
    frozen.evaluation.verify_inputs(paths)
    require(guard['test'] == 0, 'Test graph loaded')
    raw.save_predictions(root / 'validation_predictions.jsonl.gz', records)
    summary = summarize(records, metadata, settings)
    raw.write(root / 'summary.json', summary)
    lines = ['# Flow / Packet 节点调制：seed42 验证筛选', '',
             'S0 固定 Ordered；S1 普通微调；S2 参数匹配节点 MLP；S3 Flow 条件节点调制。', '',
             '| 组别 | 最佳 epoch | Macro-F1 (%) | 新增参数 |', '|---|---:|---:|---:|']
    for n in ARMS:
        lines.append(f"| {n} | {metadata[n]['best_epoch']} | {100*summary['metrics'][n]['macro_f1']:.9f} | {costs[n]['new']} |")
    lines += ['', 'S3 同时超过 S1、S2：' + str(summary['decision']['S3_exceeds_both_controls']), '',
              '仅 seed42 验证证据，未访问测试集。各类取舍和修复/退步计数见 summary.json。',
              '当前失败不能外推为双分支互补路线行不通；未通过时仅说明这一节点调制实现没有证明额外收益。']
    (root / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    raw.write(root / 'final_verification.json', {'passed': True, 'selected_raw_replay': True,
              'source_weights_and_BN_unchanged': True, 'test_graph_loads': 0, 'updates_each': state['updates']})
    raw.write(root / 'status.json', {'status': 'completed', 'test_evaluated': False})
    # The launcher's log remains open until this process exits. Exclude the entire
    # publication subtree (and the shell's PID file) from the computation seal.
    files = {p.relative_to(root).as_posix(): raw.sha(p) for p in sorted(root.rglob('*'))
             if p.is_file() and p.relative_to(root).parts[0] != 'publication'
             and p.name != 'artifact_manifest.json' and not p.name.endswith(('.tmp', '.lock', '.pid'))}
    raw.write(root / 'artifact_manifest.json', {'files_sha256': files})


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
    repo = a.repo.resolve()
    root = repo / 'outputs' / OUTPUT
    root.mkdir(parents=True, exist_ok=True)
    (repo / 'tmp').mkdir(exist_ok=True)
    def stop(signum, frame):
        raise KeyboardInterrupt('signal ' + str(signum))
    signal.signal(signal.SIGTERM, stop)
    with FileLock(str(repo / 'tmp' / (OUTPUT + '.runner.lock')), timeout=0):
        try:
            run(repo, a.data_root, root, a.download_inputs, a.resume == 'auto', a.lock_only, a.probe_only)
        except BaseException as exc:
            raw.write(root / 'failure.json', {'exception': repr(exc), 'traceback': traceback.format_exc(),
                      'test_evaluated': False})
            raise


if __name__ == '__main__':
    main()
