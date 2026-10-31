"""Recover only the known pre-training equal-weight assertion failure.

Reuse validated 64D caches, clean checkpoints and completed gradient probes.
Preserve every source artifact; train into a new output tree.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import time
import traceback
from types import FunctionType

from e3_gate_study import inside, read, require, sha, write
import run_fusion_dimension_pair as parent

ORIGINAL_SOURCES = {
    'scripts/run_fusion_dimension_pair.py': '023b43b1534c8c7fe6279cc4d09c8d0fd05828182fb5099637949bf2c2d20b3e',
    'scripts/summarize_fusion_dimension_pair.py': '0aef5121afdc4e568646dca49657585b175a1eb975ba41029abf506dbd86b2ac',
    'src/xgnid/fusion_dimension.py': 'eb8dd5c91a63b424bd421c29e333ff24492e5474cdabe77408abde5f81bb703d'}
ORIGINAL_CONFIG = '07635d84a1c00038c5dbbf550e5b0a5fa15761f95d63b0f159e8820e8eb37b2a'
FAILURE = 'ValueError: Fresh epoch0 fusion is not equal-weight'


def source_receipt(source, meta):
    c, _, ref, _, _, current = meta
    require(current['config_sha256'] == ORIGINAL_CONFIG, 'Only the unchanged 100/20 protocol can recover')
    require(read(source / 'status.json') == {'status': 'failed', 'exit_code': 1}, 'Source is not the failed experiment')
    require(FAILURE in (source / 'error.log').read_text(encoding='utf-8'), 'Different failure needs separate recovery')
    require(not (source / 'summary.json').exists() and not (source / 'D32').exists(), 'Unexpected later-phase source state')
    require((source / 'D8').is_dir() and not list((source / 'D8').iterdir()),
            'D8 has artifacts: refuse to discard any possible training progress')
    require(read(source / 'config.json') == c, 'Source training configuration changed')
    binding = read(source / 'binding.json')
    expected = copy.deepcopy(current)
    expected['sources'] = dict(ORIGINAL_SOURCES)
    dynamic = {'encoder_source', 'caches', 'cache_tensor_sha256'}
    require({k: v for k, v in binding.items() if k not in dynamic} == expected, 'Original source binding differs')
    enc = binding['encoder_source']
    require(enc['N0_best_sha256'] == c['n0_best_sha256']
            and enc['split_id'] == ref['binding']['split_id']
            and enc['manifest_sha256'] == ref['binding']['manifest_sha256'], 'Encoder/split identity differs')
    receipts = read(source / 'cache_receipts.json')
    require(binding['caches'] == receipts and set(receipts) == {'train', 'val'}, 'Cache receipts disagree')
    for part in ('train', 'val'):
        receipt = receipts[part]
        h = receipt['metadata']
        require(h['schema'] == 'xgnid.encoder64-cache.v1' and h['embedding_dim'] == 64
                and h['partition'] == part and h['source'] == enc
                and h['samples'] == (107621 if part == 'train' else 13446), 'Cache is not matching 64D data')
        require(sha(source / (part + '_encoder64.pt')) == receipt['sha256'], 'Cached tensor file changed')
    require(receipts['val']['N0_replayed_validation']['confusion_matrix'] == ref['model']['best_validation']['confusion_matrix'],
            'Original N0 replay did not match')
    init, probes, rounds = [read(source / name) for name in ('initialization.json', 'probes.json', 'preflight_roundtrips.json')]
    require(init['architecture'] == {k: parent.architecture(v) for k, v in parent.VARIANTS.items()}
            and init['streams'] == parent.initialization_streams(42) and init['all_heads_fresh'] is True
            and init['shared_output_layer_initialization_identical'] is True
            and init['encoder_sha256'] == enc['encoder_sha256'], 'Original initialization differs')
    for name in parent.VARIANTS:
        p, r = probes[name], rounds[name]
        require(p['discarded'] is True and p['clean_state_and_rng_restored'] is True
                and p['encoder_parameters_and_BN_unchanged'] is True and len(p['updates']) == 3
                and p['initial_head_sha256'] == init['head_state_sha256'][name], 'Original probe was not completed/restored')
        require(r['strict_reload'] is True and r['samples'] == 16
                and sha(source / (name + '_preflight.pt')) == r['checkpoint_sha256'], 'Initial checkpoint identity differs')
    names = ['status.json', 'progress.json', 'gpu.json', 'error.log', 'config.json', 'binding.json', 'cache_receipts.json',
             'initialization.json', 'probes.json', 'preflight_roundtrips.json',
             'train_encoder64.pt', 'val_encoder64.pt', 'D8_preflight.pt', 'D32_preflight.pt']
    return {'source': str(source), 'source_file_sha256': {n: sha(source / n) for n in names},
        'cache_receipts': receipts, 'initialization': init, 'probes': probes,
        'encoder_source': enc, 'cache_tensor_sha256': binding['cache_tensor_sha256'],
        'formal_training_updates_before_failure': 0, 'gradient_probe_updates_reused': 6,
        'original_cache_extraction_wall_seconds': sum(x['wall_seconds'] for x in receipts.values()),
        'recovery_entry_sha256': sha(Path(__file__))}


def ensure_inactive(source, output):
    require(Path('/proc').is_dir(), 'Recovery process audit requires Linux')
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit() or int(folder.name) == os.getpid():
            continue
        try:
            args = (folder / 'cmdline').read_bytes().decode(errors='replace').split('\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if not any(Path(x).name in ('run_fusion_dimension_pair.py', 'recover_fusion_dimension_pair.py') for x in args):
            continue
        require(not any(x == str(root) or x.startswith(str(root) + '/') for root in (source, output) for x in args),
                'Original/recovery process is still active: ' + folder.name)


def recovered_runtime(args, receipt):
    source = Path(receipt['source'])
    saved_initial = {}

    def reuse_cache(core, work, model, data, device, part, identity, gold=None, expected_n0=None):
        parent.progress(work, 'reuse_encoder64_cache', partition=part)
        cached = core.torch.load(work.output / (part + '_encoder64.pt'), map_location='cpu', weights_only=False)
        parent.validate_cache_header(cached['metadata'], part, identity, core.select_rows(data, part))
        require(core.state_hash([(k, cached[k]) for k in ('flow64', 'packet64', 'labels')])
                == receipt['cache_tensor_sha256'][part], 'Cached tensor contents differ')
        rows = core.select_rows(data, part)
        require(cached['ids'] == [r['sample_id'] for r in rows]
                and cached['labels'].tolist() == [r['label'] for r in rows], 'Cache sample/label order differs')
        return cached, {**copy.deepcopy(receipt['cache_receipts'][part]), 'reused': True,
                        'source_file': str(source / (part + '_encoder64.pt'))}

    def reuse_initial(core, cls, width, device, streams):
        head = parent.make_head(core, cls, width, device, streams)
        name = next(k for k, v in parent.VARIANTS.items() if v == width)
        if name not in saved_initial:
            saved = core.torch.load(source / (name + '_preflight.pt'), map_location='cpu', weights_only=False)
            require(saved['architecture'] == parent.architecture(width), 'Saved initial architecture differs')
            require(core.state_hash([(k, v) for k, v in saved['model_state'].items()
                    if k.startswith(('flow_encoder.', 'packet_encoder.'))]) == receipt['encoder_source']['encoder_sha256'],
                    'Saved initial encoder/BN state differs')
            saved_initial[name] = {k[len('fusion.'):]: v for k, v in saved['model_state'].items() if k.startswith('fusion.')}
        expected = receipt['initialization']['head_state_sha256'][name]
        require(core.state_hash(head.state_dict().items()) == expected, 'Fresh initialization no longer reproduces source')
        head.load_state_dict(saved_initial[name], strict=True)
        require(core.state_hash(head.state_dict().items()) == expected, 'Source initial head differs')
        parent.assert_zero_gate_output(core, head)
        return head

    def reuse_probe(core, graph, data, cache, device, c, encoder_hash):
        name = next(k for k, v in parent.VARIANTS.items() if v == graph.fusion.width)
        parent.assert_frozen(core, graph, encoder_hash)
        require(core.state_hash(graph.fusion.state_dict().items())
                == receipt['initialization']['head_state_sha256'][name], 'Source probe state leaked into head')
        return {**copy.deepcopy(receipt['probes'][name]), 'reused': True, 'new_probe_optimizer_updates': 0,
                'source': str(source / 'probes.json')}

    def recovery_write(path, value):
        if path == args.output / 'summary.json':
            for name, digest in receipt['source_file_sha256'].items():
                require(sha(source / name) == digest, 'Original artifact changed during recovery: ' + name)
            value = {**value, 'cache_reused': True, 'cache_time_scope': 'recovery_load_and_validate',
                'recovery_source': str(source), 'source_artifacts_unchanged': True,
                'original_cache_extraction_wall_seconds': receipt['original_cache_extraction_wall_seconds'],
                'probe_optimizer_updates_discarded': 0, 'gradient_probe_updates_reused': 6,
                'new_probe_optimizer_updates': 0}
        write(path, value)

    def recovery_progress(work, phase, **details):
        if phase == 'raw_graph_gradient_probe':
            phase = 'reuse_completed_gradient_probe'
        parent.progress(work, phase, **details)

    ns = dict(parent.run.__globals__, extract_cache=reuse_cache, make_head=reuse_initial,
              probe=reuse_probe, write=recovery_write, progress=recovery_progress)
    return FunctionType(parent.run.__code__, ns, 'run_recovered_pair')


def execute(args, meta, receipt):
    require(args.output != args.source and args.source not in args.output.parents, 'Recovery needs a separate tree')
    require(not args.output.exists(), 'Refuse recovery output overwrite/resume')
    args.output.mkdir(parents=True)
    write(args.output / 'status.json', {'status': 'running', 'phase': 'cache_recovery'})
    write(args.output / 'progress.json', {'status': 'running', 'phase': 'cache_recovery'})
    binding = {**meta[-1], 'recovery': {k: v for k, v in receipt.items() if k not in
               ('cache_receipts', 'initialization', 'probes', 'encoder_source', 'cache_tensor_sha256')}}
    current = (*meta[:-1], binding)
    write(args.output / 'config.json', meta[0])
    write(args.output / 'binding.json', binding)
    write(args.output / 'recovery_plan.json', receipt)
    started = time.perf_counter()
    try:
        for part in ('train', 'val'):
            filename = part + '_encoder64.pt'
            shutil.copyfile(args.source / filename, args.output / filename)
            require(sha(args.output / filename) == receipt['source_file_sha256'][filename], 'Recovered cache copy differs')
        write(args.output / 'recovery_cache_copy.json', {'wall_seconds': time.perf_counter() - started,
                                                       'source_files_modified': False})
        recovered_runtime(args, receipt)(args, current)
        for name, digest in receipt['source_file_sha256'].items():
            require(sha(args.source / name) == digest, 'Original artifact changed during recovery: ' + name)
        write(args.output / 'status.json', {'status': 'completed', 'exit_code': 0})
        write(args.output / 'progress.json', {'status': 'completed', 'training_jobs': 2})
    except BaseException:
        write(args.output / 'status.json', {'status': 'failed', 'exit_code': 1})
        write(args.output / 'progress.json', {'status': 'failed', 'exit_code': 1})
        (args.output / 'error.log').write_text(traceback.format_exc(), encoding='utf-8')
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path.cwd())
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=('check', 'run'), default='check')
    p.add_argument('--physical-gpu', type=int, default=1)
    p.add_argument('--allow-shared-gpu', action='store_true')
    args = p.parse_args()
    args.repo = args.repo.resolve()
    args.config = (args.repo / args.config).resolve()
    args.source = inside(args.repo / 'outputs', inside(args.repo, args.source))
    args.output = inside(args.repo / 'outputs', inside(args.repo, args.output))
    require(args.output != args.source and args.source not in args.output.parents, 'Recovery must use a separate tree')
    meta = parent.metadata(args)
    receipt = source_receipt(args.source, meta)
    require(os.name == 'posix', 'Execute recovery checks on the Linux server')
    ensure_inactive(args.source, args.output)
    if args.mode == 'check':
        print(json.dumps({'status': 'recovery_metadata_checks_passed', 'formal_updates_before_failure': 0,
            'cache_reused': True, 'initial_states_and_probes_reused': True, 'training_started': False,
            'tensor_checks_executed': False, 'source': str(args.source)}), flush=True)
        return
    execute(args, meta, receipt)


if __name__ == '__main__':
    main()
