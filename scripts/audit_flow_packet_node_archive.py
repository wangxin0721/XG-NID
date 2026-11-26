"""Read the exact GitHub result snapshot and recompute integer-prediction metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import run_flow_packet_node_modulation as run


def audit(repo, commit, remote='origin'):
    require, archive = run.require, run.archive
    require(len(commit) == 40 and all(c in '0123456789abcdef' for c in commit), 'Exact SHA required')
    env = archive.lfs_environment(repo)
    subprocess.run(['git', 'fetch', '--no-tags', remote, commit], cwd=repo, env=env, check=True)
    prefix = 'outputs/' + run.OUTPUT + '/'
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit, '--', prefix],
                                    cwd=repo, env=env, text=True).splitlines()
    required = ['config.json', 'protocol_lock.json', 'status.json']
    completed = prefix + 'summary.json' in names
    if completed:
        required += ['summary.json', 'validation_predictions.jsonl.gz', 'final_verification.json', 'history.json']
    spec = {'source_result_commit': commit, 'source_directory': prefix.rstrip('/'), 'inputs': {}}
    for name in required:
        require(prefix + name in names, 'Published input missing: ' + name)
        data = subprocess.check_output(['git', 'show', commit + ':' + prefix + name], cwd=repo, env=env)
        if data.startswith(b'version https://git-lfs.github.com/spec/v1\n'):
            lines = data.decode().splitlines()
            meta = {'sha256': lines[1].split(':')[1], 'bytes': int(lines[2].split()[1])}
        else:
            meta = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
        spec['inputs'][name] = meta
    inputs = archive.Archive(repo, spec, snapshot=True, env=env)
    missing = [prefix+n for n in required if isinstance(inputs.locate(n), Path) and not inputs.locate(n).is_file()]
    if missing:
        subprocess.run(['git', 'lfs', 'fetch', '--include=' + ','.join(missing), '--exclude=', remote, commit],
                       cwd=repo, env=env, check=True)
    inputs.verify_all()
    settings = inputs.json('config.json')
    require(settings == inputs.json('protocol_lock.json')['settings'], 'Published config/protocol mismatch')
    result = {'passed': True, 'result_commit': commit, 'completed_training_audited': completed,
              'test_predictions_loaded': False, 'input_bytes_sha256_verified': True}
    if completed:
        rows = inputs.rows('validation_predictions.jsonl.gz')
        require(len(rows) == settings['validation_samples'] and len({r['sample_id'] for r in rows}) == len(rows),
                'Published cohort size/identities differ')
        require(archive.canonical_sha([(r['sample_id'], r['label']) for r in rows]) ==
                inputs.json('protocol_lock.json')['validation_cohort_sha256'], 'Published validation cohort differs')
        require(all(r['label'] in range(8) and set(r['predictions']) == set(run.ARMS)
                    and all(p in range(8) for p in r['predictions'].values()) for r in rows), 'Invalid predictions')
        saved = inputs.json('summary.json')
        measured = run.summarize(rows, saved['models'], settings)
        require(measured == saved, 'GitHub metric/pair/decision recomputation differs')
        history = inputs.json('history.json')
        require([r['epoch'] for r in history] == list(range(settings['epochs'] + 1)), 'Incomplete epoch history')
        sampler = run.training.transaction.PairedSampler(settings['training_samples'], settings['batch_size'], settings['sampling_seed'])
        for row in history[1:]:
            sampler.start(row['epoch'])
            require(row['order_sha256'] == sampler.order_sha256 and row['samples_per_arm'] == sampler.size
                    and row['updates_per_arm'] == len(sampler)
                    and row['cumulative_updates_per_arm'] == row['epoch'] * len(sampler), 'Published training budget/order differs')
            sampler.cursor = sampler.size
        for n in settings['trained_arms']:
            best = max(history, key=lambda r: r['validation'][n]['macro_f1'])
            require(best['epoch'] == saved['models'][n]['best_epoch'], 'Published earliest-best epoch differs')
            run.training.metrics.check_metric(saved['metrics'][n], best['validation'][n])
        require(inputs.json('status.json')['status'] == 'completed'
                and inputs.json('final_verification.json')['passed'], 'Missing completion verification')
        result.update(samples=len(rows), comparisons=saved['comparisons'], decision=saved['decision'])
    return result


def main():
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


if __name__ == '__main__':
    main()
