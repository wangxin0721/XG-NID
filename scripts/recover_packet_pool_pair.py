"""Reuse caches only after the fbf812e pre-training epoch0 readout failure."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import time
import traceback

from e3_gate_study import inside, read, require, sha, write
import run_packet_pool_pair as parent

ORIGINAL_CONFIG = '70bb0c6b0e3c01f5797fbd567b17d7b9c3a324c9deb8f8ccc1c51f7c12ca0333'
ORIGINAL_SOURCES = {
    'scripts/run_packet_pool_pair.py': '29f8d91553b4f232cd46748565b9c0e360a5be780cb61df9445aa6d6e9fdddbb',
    'src/xgnid/packet_pooling.py': '475cd89d9ab52ddcc360ee5a8c02e8a1601b783a917fcaca56354b33fb2ef8b1',
    'scripts/summarize_packet_pool_pair.py': 'a234e5045112c50a04fccfa17dfac5725ec6774eec41c77d40229a23d48863c7'}
FAILURE = 'ValueError: Non-equivalent epoch0 readout'
REQUIRED = ('status.json', 'progress.json', 'config.json', 'binding.json', 'gpu.json', 'error.log',
            'cache_receipts.json', 'cache_memory_estimates.json', 'packet_diagnostics.json',
            'train_packet_nodes.pt', 'val_packet_nodes.pt')


def source_receipt(source, meta):
    c, previous, _, _, current = meta
    require(current['config_sha256'] == ORIGINAL_CONFIG, 'Recovery requires the unchanged P0/P1 protocol')
    require(read(source / 'status.json') == {'status': 'failed', 'exit_code': 1}, 'Source is not the failed preparation')
    error = (source / 'error.log').read_text(encoding='utf-8')
    require(FAILURE in error and 'in initial_equivalence' in error, 'Different failure requires a separate diagnosis')
    require(all((source / n).is_file() for n in REQUIRED), 'Incomplete prepared cache artifacts')
    require(not any(p.is_dir() for p in source.iterdir()), 'Source has model directories; refuse possible training progress')
    forbidden = ('summary.json', 'initialization.json', 'probes.json', 'P0_preflight.pt', 'P1_preflight.pt')
    require(not any((source / n).exists() for n in forbidden)
            and {p.name for p in source.glob('*.pt')} == {'train_packet_nodes.pt', 'val_packet_nodes.pt'},
            'Source reached a later phase; refuse to discard any training/probe work')
    require(read(source / 'config.json') == c, 'Source configuration changed')
    expected = copy.deepcopy(current)
    expected['sources'] = dict(ORIGINAL_SOURCES)
    require(read(source / 'binding.json') == expected, 'Failed source is not the bound fbf812e preparation')
    receipts = read(source / 'cache_receipts.json')
    memory = read(source / 'cache_memory_estimates.json')
    require(set(receipts) == set(memory) == {'train', 'val'}, 'Prepared cache partitions differ')
    ref = previous[1][2]
    encoder_hash = previous[3]['binding']['encoder_source']['encoder_sha256']
    for part, size in (('train', 107621), ('val', 13446)):
        r, h = receipts[part], receipts[part]['metadata']
        require(h['schema'] == 'xgnid.packet-node64-cache.v1' and h['partition'] == part
                and h['samples'] == size and h['encoder_sha256'] == encoder_hash
                and h['split_binding'] == ref['binding'] and bool(h['sample_order_sha256']), 'Cache provenance differs')
        require(r['sha256'] == sha(source / (part + '_packet_nodes.pt')) and bool(r['tensor_sha256']),
                'Prepared cache file changed')
        require(r['graph_load_stage'] == parent.graph_load_stage('extract_' + part, part, size, c['workers'])
                and r['memory_estimate'] == memory[part] and memory[part]['graph_samples'] == size,
                'Original extraction receipt differs')
    files = sorted(p for p in source.iterdir() if p.is_file())
    return {'directory': str(source), 'file_sha256': {p.name: sha(p) for p in files},
        'cache_receipts': receipts, 'formal_training_updates_before_failure': 0,
        'probe_updates_before_failure': 0, 'reused_graph_samples': {'train': 107621, 'val': 13446},
        'original_cache_extraction_wall_seconds': sum(r['wall_seconds'] for r in receipts.values()),
        'original_source_commit': 'fbf812ef01a69781d2ee734e7e53a95fb16506ed'}


def ensure_inactive(source, output):
    require(Path('/proc').is_dir(), 'Recovery process audit requires Linux')
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit() or int(folder.name) == os.getpid():
            continue
        try:
            args = (folder / 'cmdline').read_bytes().decode(errors='replace').split('\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if not any(Path(x).name in ('run_packet_pool_pair.py', 'recover_packet_pool_pair.py') for x in args):
            continue
        paths = [x.split('=', 1)[-1] for x in args]
        require(not any(x == str(root) or x.startswith(str(root) + '/')
                        for root in (source, output) for x in paths),
                'Source/recovery process still active: ' + folder.name)


def assert_source_unchanged(receipt):
    source = Path(receipt['directory'])
    require(sorted(p.name for p in source.iterdir() if p.is_file()) == sorted(receipt['file_sha256']),
            'Failed source file inventory changed')
    require(not any(p.is_dir() for p in source.iterdir()), 'New training directory appeared in failed source')
    for name, digest in receipt['file_sha256'].items():
        require(sha(source / name) == digest, 'Failed source artifact modified: ' + name)


def cache_provider(receipt):
    def reuse(core, args, model, data, device, part, c, encoder_hash, split_binding, retained_bytes=0):
        started = time.perf_counter()
        source = Path(receipt['directory']) / (part + '_packet_nodes.pt')
        destination = args.output / source.name
        expected = receipt['cache_receipts'][part]
        require(not destination.exists() and sha(source) == expected['sha256'], 'Refuse changed cache or overwrite')
        parent.dim.progress(args, 'copy_and_validate_prepared_packet_nodes', partition=part)
        shutil.copyfile(source, destination)
        require(sha(destination) == expected['sha256'], 'Copied cache file differs')
        cached = core.torch.load(destination, map_location='cpu', weights_only=False)
        parent.validate_cache(core, cached, core.select_rows(data, part), part, encoder_hash, split_binding)
        require(parent.tensor_hash(core, cached) == expected['tensor_sha256']
                and cached['metadata'] == expected['metadata'], 'Copied cache tensors/provenance differ')
        parent.dim.assert_frozen(core, model, encoder_hash)
        return cached, {**copy.deepcopy(expected), 'reused': True, 'reused_from': str(source),
            'original_extraction_wall_seconds': expected['wall_seconds'],
            'wall_seconds': time.perf_counter() - started,
            'memory_estimate_scope': 'original_failed_preparation_not_current_copy_or_load'}
    return reuse


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path.cwd())
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--failed-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=('check', 'run'), default='check')
    p.add_argument('--physical-gpu', type=int, default=1)
    p.add_argument('--allow-shared-gpu', action='store_true')
    args = p.parse_args()
    args.repo, args.config = args.repo.resolve(), args.config.resolve()
    args.failed_run = inside(args.repo / 'outputs', inside(args.repo, args.failed_run))
    args.output = inside(args.repo / 'outputs', inside(args.repo, args.output))
    require(not args.output.exists() and args.output != args.failed_run
            and args.failed_run not in args.output.parents and args.output not in args.failed_run.parents,
            'Refuse overwrite/resume or nested recovery output')
    meta = parent.metadata(args)
    receipt = source_receipt(args.failed_run, meta)
    ensure_inactive(args.failed_run, args.output)
    binding = copy.deepcopy(meta[-1])
    binding['cache_recovery'] = receipt
    binding['sources']['scripts/recover_packet_pool_pair.py'] = sha(Path(__file__))
    meta = (*meta[:-1], binding)
    if args.mode == 'check':
        print(json.dumps({'status': 'recovery_metadata_checks_passed', 'cache_recovery': receipt,
                          'training_started': False, 'tensor_checks_executed': False}), flush=True)
        return
    require(os.name == 'posix', 'Recovery runs on the Linux wx server only')
    args.output.mkdir(parents=True)
    write(args.output / 'config.json', meta[0])
    write(args.output / 'binding.json', binding)
    write(args.output / 'recovery_receipt.json', receipt)
    write(args.output / 'cache_memory_estimates.json', {part: {**r['memory_estimate'],
        'measurement_scope': 'original_failed_preparation', 'source_directory': receipt['directory']}
        for part, r in receipt['cache_receipts'].items()})
    write(args.output / 'status.json', {'status': 'running', 'phase': 'reuse_prepared_caches'})
    try:
        parent.run(args, meta, cache_provider=cache_provider(receipt),
                   verify_cache_source=lambda: assert_source_unchanged(receipt))
        assert_source_unchanged(receipt)
        write(args.output / 'status.json', {'status': 'completed', 'exit_code': 0})
        write(args.output / 'progress.json', {'status': 'completed', 'training_jobs': 2})
    except BaseException:
        write(args.output / 'status.json', {'status': 'failed', 'exit_code': 1})
        write(args.output / 'progress.json', {'status': 'failed', 'exit_code': 1})
        (args.output / 'error.log').write_text(traceback.format_exc(), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
