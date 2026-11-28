"""Independently recompute per-epoch diagnostics from exact Git/LFS snapshot bytes."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import run_node_training_diagnostic as run


def compare_observation(actual, saved):
    run.training.metrics.check_metric(actual, saved)
    run.require(math.isclose(actual['nll'], saved['nll'], abs_tol=1e-6, rel_tol=1e-5)
                and actual['source_comparison'] == saved['source_comparison'], 'NLL or source changes differ')


def verify(inputs):
    settings = inputs.json('config.json')
    protocol = inputs.json('protocol_lock.json')
    run.require(settings == protocol['settings'] and settings['test_access'] is False, 'Config or scope differs')
    if inputs.json('status.json')['status'] != 'completed':
        return {'passed': True, 'completed_training_audited': False, 'test_predictions_loaded': False}
    summary = inputs.json('summary.json')
    history = inputs.json('history.json')
    subset = inputs.json('diagnostic_subset.json')
    run.require(len(subset) == 8*settings['diagnostic_samples_per_class'] and
                set(r['label'] for r in subset) == set(range(8)) and
                all(r['partition'] == 'train' for r in subset), 'Diagnostic subset differs')
    run.require([r['epoch'] for r in history] == list(range(21)), 'Incomplete 20-epoch history')
    sampler = run.training.transaction.PairedSampler(settings['training_samples'], settings['batch_size'], settings['sampling_seed'])
    cohort = None
    for row in history:
        epoch = row['epoch']
        if epoch:
            sampler.start(epoch)
            run.require(row['order_sha256'] == sampler.order_sha256 and row['updates_per_arm'] == len(sampler)
                        and row['samples_per_arm'] == sampler.size and row['cumulative_updates_per_arm'] == epoch*len(sampler),
                        'Sampling/update budget differs')
            sampler.cursor = sampler.size
        folder = 'observations/epoch_'+str(epoch)+'/'
        val, fixed = inputs.rows(folder+'validation.jsonl.gz'), inputs.rows(folder+'fixed_train.jsonl.gz')
        current = [(r['sample_id'], r['label']) for r in val]
        run.require(len(val) == settings['validation_samples'] and len({sid for sid, _ in current}) == len(current), 'Invalid validation cohort')
        run.require(run.archive.canonical_sha(current) == protocol['validation_cohort_sha256'], 'Wrong validation identities')
        run.require(cohort is None or current == cohort, 'Cohort changed across epochs')
        cohort = current
        run.require([(r['sample_id'], r['label']) for r in fixed] == [(r['sample_id'], r['label']) for r in subset],
                    'Fixed training identities changed')
        for part in (val, fixed):
            run.require(all(r['label'] in range(8) and set(r['predictions']) == {'Ordered', *run.diagnostic.ARMS}
                            and set(r['true_log_probability']) == set(r['predictions'])
                            and all(p in range(8) for p in r['predictions'].values())
                            and all(math.isfinite(x) and x <= 1e-6 for x in r['true_log_probability'].values())
                            for r in part), 'Invalid prediction/probability export')
        for name in run.diagnostic.ARMS:
            actual = run.diagnostic.record_metrics(val, name)
            saved = row['validation'][name]
            compare_observation(actual, saved)
            run.require(math.isclose(run.diagnostic.record_metrics(fixed, name)['nll'], saved['fixed_train_nll'],
                                    abs_tol=1e-6, rel_tol=1e-5), 'Fixed train NLL differs')
            if name in ('R2', 'R3'):
                run.require(all(v['absolute_l2'] == 0 and v['changed_elements'] == 0
                                for k, v in saved['parameter_delta'].items() if k != 'adjustment'), 'Original parameters moved')
    selected = inputs.rows('selected_validation.jsonl.gz')
    run.require([(r['sample_id'], r['label']) for r in selected] == cohort, 'Selected cohort differs')
    compare_observation(run.diagnostic.record_metrics(selected, 'Ordered'), summary['best']['Ordered'])
    selected_train = inputs.rows('observations/epoch_selected/fixed_train.jsonl.gz')
    run.require([(r['sample_id'], r['label']) for r in selected_train] == [(r['sample_id'],r['label']) for r in subset],
                'Selected training subset changed')
    for name in run.diagnostic.ARMS:
        best = max(history, key=lambda r: r['validation'][name]['macro_f1'])
        run.require(summary['best_epochs'][name] == best['epoch'], 'Earliest best selection differs')
        compare_observation(run.diagnostic.record_metrics(selected, name), summary['best'][name])
        run.training.metrics.check_metric(summary['best'][name], best['validation'][name])
        run.require(math.isclose(run.diagnostic.record_metrics(selected_train,name)['nll'],
                                summary['best'][name]['fixed_train_nll'],abs_tol=1e-6,rel_tol=1e-5),
                    'Selected fixed training NLL differs')
    run.require(run.diagnostic.decision(summary['best'], settings) == summary['decision'], 'Extension gate differs')
    old = inputs.json('historical/trajectory.json')
    deltas = {str(r['epoch']): 100*(history[r['epoch']]['validation']['R1']['macro_f1']-
                                   r['validation_macro_f1']['S1']) for r in old['selected'] if r['epoch'] in (1,2,5,10,20)}
    run.require(deltas == summary['R1_minus_historical_S1_pp'], 'R1 historical comparison differs')
    run.require(inputs.json('final_verification.json')['passed'] and summary['updates_each'] == settings['updates_per_arm']
                and not summary['test_evaluated'], 'Missing budget/scope verification')
    return {'passed': True, 'completed_training_audited': True, 'epochs': 20, 'updates_each': settings['updates_per_arm'],
            'test_predictions_loaded': False, 'samples': len(selected), 'decision': summary['decision'],
            'scope': 'predictions/NLL/selection/budget recomputed; tensor changes are server receipts, not independently reloaded here'}


def audit(repo, commit, remote='origin'):
    run.require(len(commit) == 40 and all(c in '0123456789abcdef' for c in commit), 'Exact commit required')
    env = run.archive.lfs_environment(repo)
    subprocess.run(['git', 'fetch', '--no-tags', remote, commit], cwd=repo, env=env, check=True)
    prefix = 'outputs/'+run.OUTPUT+'/'
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit, '--', prefix], cwd=repo, env=env, text=True).splitlines()
    selected = [n for n in names if n[len(prefix):] in ('config.json', 'protocol_lock.json', 'status.json',
        'summary.json', 'history.json', 'diagnostic_subset.json', 'final_verification.json',
        'selected_validation.jsonl.gz', 'historical/trajectory.json') or
        (n.startswith(prefix+'observations/') and n.endswith('.jsonl.gz'))]
    spec = {'source_result_commit': commit, 'source_directory': prefix.rstrip('/'), 'inputs': {}}
    for path in selected:
        payload = subprocess.check_output(['git', 'show', commit+':'+path], cwd=repo, env=env)
        if payload.startswith(b'version https://git-lfs.github.com/spec/v1\n'):
            lines = payload.decode().splitlines()
            meta = {'sha256': lines[1].split(':')[1], 'bytes': int(lines[2].split()[1])}
        else:
            meta = {'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}
        spec['inputs'][path[len(prefix):]] = meta
    inputs = run.archive.Archive(repo, spec, snapshot=True, env=env)
    missing = [prefix+n for n in spec['inputs'] if isinstance(inputs.locate(n), Path) and not inputs.locate(n).is_file()]
    if missing:
        subprocess.run(['git', 'lfs', 'fetch', '--include='+','.join(missing), '--exclude=', remote, commit], cwd=repo, env=env, check=True)
    inputs.verify_all()
    return {'result_commit': commit, 'input_bytes_sha256_verified': True, **verify(inputs)}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path.cwd())
    p.add_argument('--commit', required=True)
    p.add_argument('--remote', default='origin')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    result = audit(a.repo.resolve(), a.commit, a.remote)
    if a.output:
        run.raw.write(a.output, result)
    print(json.dumps(result, ensure_ascii=False))
