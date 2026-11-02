"""Audit both known pre-training failures; reuse the original node caches."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import traceback

from e3_gate_study import inside, read, require, sha, write
import recover_packet_pool_pair as first
import run_packet_pool_pair as parent

PREVIOUS_COMMIT = '2bf57a5352a5ed65918b759dfb33459bcaadd770'
PREVIOUS_SOURCES = {
    'scripts/run_packet_pool_pair.py': 'b090fc0afd15e37d585985c8d8de9685153e61c511451f54de4c26ea26b38884',
    'src/xgnid/packet_pooling.py': 'd0e2a846b8bb0705ca4b42e3030ac37cd42a203866a901f575149724a88a27db',
    'scripts/summarize_packet_pool_pair.py': '302ec2196356f743493c07de4a7518e1ed44f44e5c43ecc724f3939f276a7550',
    'scripts/recover_packet_pool_pair.py': '5ebbbf6a351421111b1d80bbb0753f51c09eab3670c056f9287c4ba8b298340f'}


def previous_failure_receipt(previous, meta, original):
    require(read(previous / 'status.json') == {'status': 'failed', 'exit_code': 1}, 'Previous recovery is not failed')
    error = (previous / 'error.log').read_text(encoding='utf-8')
    require(first.FAILURE in error and 'in initial_equivalence' in error, 'Previous recovery failed at a different stage')
    require(all((previous / n).is_file() for n in first.REQUIRED)
            and (previous / 'recovery_receipt.json').is_file()
            and (previous / 'initialization_comparison.json').is_file(), 'Previous recovery evidence incomplete')
    require(not any(p.is_dir() for p in previous.iterdir())
            and not any((previous / n).exists() for n in ('summary.json', 'initialization.json', 'probes.json'))
            and {p.name for p in previous.glob('*.pt')} == {'train_packet_nodes.pt', 'val_packet_nodes.pt'},
            'Previous recovery contains later work; refuse to discard progress')
    expected = copy.deepcopy(meta[-1])
    expected['sources'] = dict(PREVIOUS_SOURCES)
    expected['cache_recovery'] = original
    require(read(previous / 'binding.json') == expected and read(previous / 'config.json') == meta[0]
            and read(previous / 'recovery_receipt.json') == original, 'Previous recovery binding/source changed')
    diagnostic = read(previous / 'initialization_comparison.json')
    require(diagnostic['samples'] == 13446 and diagnostic['atol'] == diagnostic['rtol'] == 1e-6
            and diagnostic['new_failed_batches'] > 0 and diagnostic['new_prediction_mismatches'] == 0,
            'This recovery requires the known numeric-only initial failure')
    reused = read(previous / 'cache_receipts.json')
    require(set(reused) == {'train', 'val'}, 'Previous cache partitions differ')
    for part, expected_cache in original['cache_receipts'].items():
        r = reused[part]
        require(r['reused'] is True and r['metadata'] == expected_cache['metadata']
                and r['tensor_sha256'] == expected_cache['tensor_sha256']
                and r['sha256'] == expected_cache['sha256'] == sha(previous / (part + '_packet_nodes.pt')),
                'Previous copied cache changed')
    return {'directory': str(previous), 'source_commit': PREVIOUS_COMMIT,
        'file_sha256': {p.name: sha(p) for p in sorted(previous.iterdir()) if p.is_file()},
        'initialization_comparison': diagnostic, 'formal_training_updates_before_failure': 0,
        'probe_updates_before_failure': 0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path.cwd())
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--failed-run', type=Path, required=True)
    p.add_argument('--previous-recovery', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=('check', 'run'), default='check')
    p.add_argument('--physical-gpu', type=int, default=1)
    p.add_argument('--allow-shared-gpu', action='store_true')
    args = p.parse_args()
    args.repo, args.config = args.repo.resolve(), args.config.resolve()
    for name in ('failed_run', 'previous_recovery', 'output'):
        setattr(args, name, inside(args.repo / 'outputs', inside(args.repo, getattr(args, name))))
    paths = (args.failed_run, args.previous_recovery, args.output)
    require(len(set(paths)) == 3 and not args.output.exists()
            and all(a not in b.parents for a in paths for b in paths if a != b), 'Refuse overwrite or nested recovery')
    meta = parent.metadata(args)
    original = first.source_receipt(args.failed_run, meta)
    previous = previous_failure_receipt(args.previous_recovery, meta, original)
    for source in (args.failed_run, args.previous_recovery):
        first.ensure_inactive(source, args.output)
    binding = copy.deepcopy(meta[-1])
    binding['cache_recovery'], binding['previous_recovery_failure'] = original, previous
    binding['sources']['scripts/recover_packet_pool_pair.py'] = sha(Path(first.__file__))
    binding['sources']['scripts/recover_packet_pool_pair_v2.py'] = sha(Path(__file__))
    meta = (*meta[:-1], binding)
    if args.mode == 'check':
        print(json.dumps({'status': 'recovery_v2_metadata_checks_passed',
                          'source': str(args.failed_run), 'previous_recovery': str(args.previous_recovery),
                          'formal_updates_discarded': 0, 'training_started': False}), flush=True)
        return
    require(os.name == 'posix', 'Formal recovery runs on the Linux wx server')
    args.output.mkdir(parents=True)
    write(args.output / 'config.json', meta[0])
    write(args.output / 'binding.json', binding)
    write(args.output / 'recovery_receipt.json', original)
    write(args.output / 'previous_failure_receipt.json', previous)
    write(args.output / 'cache_memory_estimates.json', {part: {**r['memory_estimate'],
        'measurement_scope': 'original_failed_preparation', 'source_directory': original['directory']}
        for part, r in original['cache_receipts'].items()})
    write(args.output / 'status.json', {'status': 'running', 'phase': 'cuda_readout_preflight_then_cache_reuse'})
    def verify_sources():
        first.assert_source_unchanged(original)
        first.assert_source_unchanged(previous)
    try:
        parent.run(args, meta, cache_provider=first.cache_provider(original), verify_cache_source=verify_sources)
        verify_sources()
        write(args.output / 'status.json', {'status': 'completed', 'exit_code': 0})
        write(args.output / 'progress.json', {'status': 'completed', 'training_jobs': 2})
    except BaseException:
        write(args.output / 'status.json', {'status': 'failed', 'exit_code': 1})
        write(args.output / 'progress.json', {'status': 'failed', 'exit_code': 1})
        (args.output / 'error.log').write_text(traceback.format_exc(), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
