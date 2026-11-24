"""Lock twelve validation-best models, replay validation, then evaluate fixed test; no training."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import time
import traceback

from filelock import FileLock
import torch
import torch_geometric
from torch_geometric.loader import DataLoader

import fixed_transition_test_lock as locked
import fixed_transition_test_results as report

raw, frozen, original, require = locked.raw, locked.frozen, locked.original, locked.require
OUTPUT_NAME = 'fixed_transition_final_test_v1'
CONFIG = raw.ROOT / ('configs/' + OUTPUT_NAME + '.json')


def environment(device):
    return {'python': platform.python_version(), 'torch': str(torch.__version__),
        'pyg': torch_geometric.__version__, 'cuda': torch.version.cuda, 'platform': platform.platform(),
        'device': torch.cuda.get_device_name(device) if device.type == 'cuda' else str(device)}


def load_data(repo, graph_root, settings):
    split = repo / settings['split_directory']
    paths = {split / 'split.json': settings['split_json_sha256']}
    frozen.evaluation.verify_inputs(paths)
    data = raw.FrozenGraphs(split, settings['split_id'], data_root=graph_root)
    require(data.metadata['sample_manifest_sha256'] == settings['manifest_sha256'], 'Sample manifest differs')
    paths[split / data.metadata['sample_manifest']] = settings['manifest_sha256']
    frozen.evaluation.verify_inputs(paths)
    require(data.metadata['partition_sizes'] == {'train': 107621, 'val': 13446, 'test': 13448},
            'Fixed partition counts differ')
    config = {'seed': 42, 'workers': settings['workers'], 'eval_batch_size': settings['eval_batch_size'],
              'partition_sizes': {'val': settings['validation_samples'], 'test': settings['test_samples']}}
    rows = {p: frozen.evaluation.evaluation_rows(data, p, config) for p in ('val', 'test')}
    require(not {r['sample_id'] for r in rows['val']}.intersection(r['sample_id'] for r in rows['test']),
            'Validation/test overlap')
    return data, rows, config, paths


@torch.no_grad()
def extract_test_cache(source, data, rows, config, device, output):
    # Keep the historical training loader's train/val guard intact. Test access
    # uses the existing, explicitly test-only dataset instead.
    loader = DataLoader(frozen.evaluation.TestGraphs(data, rows), batch_size=config['eval_batch_size'],
        shuffle=False, num_workers=config['workers'], generator=torch.Generator().manual_seed(242))
    chunks, predictions, offset = [], [], 0
    for step, batch in enumerate(loader, 1):
        stop = offset + batch.num_graphs
        expected = rows[offset:stop]
        require(list(zip(batch.sample_id, batch.y.view(-1).tolist())) ==
                [(r['sample_id'], r['label']) for r in expected], 'Test graph IDs/labels/order differ')
        features, _ = frozen.encode(source, batch, device)
        chunks.append(original.Features(*(v.cpu() for v in features)))
        predictions.extend(original.baseline(source, features).argmax(-1).tolist())
        offset = stop
        if step == 1 or step % 200 == 0 or stop == len(rows):
            raw.progress(output, 'cache_fixed_test', samples=stop, total=len(rows))
    require(offset == len(rows), 'Incomplete fixed test extraction')
    cache = original.FrozenCache(chunks, [r['label'] for r in rows])
    counts = locked.results.diagnostics.context_counts(cache.features)
    coverage = {'samples': len(rows), 'valid_transitions': int(cache.features.lengths.sum()),
        'ordered_contexts': int(counts.sum()), 'flows_with_ordered_contexts': int((counts > 0).sum()),
        'flows_without_transitions': int((cache.features.lengths == 0).sum()),
        'context_coverage_by_class': {label: {'samples': sum(r['label'] == i for r in rows),
            'with_context': sum(r['label'] == i and n > 0 for r, n in zip(rows, counts.tolist()))}
            for i, label in enumerate(locked.LABELS)}}
    return cache, original.prediction_records(rows, predictions), coverage


def bound_cache(source, data, rows, part, config, device, output, binding):
    """Only frozen features, once per backbone/partition, with explicit payload integrity."""
    path = output / ('frozen_' + part + '_features.pt')
    receipt_path = output / ('cache_' + part + '_receipt.json')
    if path.is_file() and receipt_path.is_file():
        receipt = raw.read(receipt_path)
        require(receipt['binding'] == binding and receipt['sha256'] == raw.sha(path),
                'Frozen cache source/environment/hash differs')
        saved = torch.load(path, map_location='cpu', weights_only=True)
        require(saved['binding'] == binding and saved['ids'] == [r['sample_id'] for r in rows]
                and saved['labels'] == [r['label'] for r in rows], 'Frozen cache cohort differs')
        features = original.Features(*(saved['features'][k] for k in original.Features._fields))
        return original.FrozenCache([features], saved['labels']), saved['reference'], saved['coverage']
    # A crash between cache and receipt publication causes extraction to be repeated, never trusted blindly.
    require(part in ('val', 'test'), 'Fixed evaluation cannot extract training features')
    if part == 'test':
        cache, reference, coverage = extract_test_cache(source, data, rows, config, device, output)
    else:
        cache, reference, coverage = frozen.cache_split(source, data, rows, part, config, device, output)
    locked.stability.training.parent.atomic_save(path, {'binding': binding,
        'ids': [r['sample_id'] for r in rows], 'labels': [r['label'] for r in rows],
        'features': cache.features._asdict(), 'reference': reference, 'coverage': coverage})
    raw.write(receipt_path, {'binding': binding, 'sha256': raw.sha(path),
        'partition': part, 'samples': len(rows), 'frozen_modules_only': True})
    return cache, reference, coverage


@torch.no_grad()
def predict(models, cache, rows, batch_size, device):
    records = []
    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        features, _ = cache.batch(range(start, stop), device)
        records.extend(locked.results.diagnostics.observe(models, features, rows[start:stop]))
    return records


def compare_replay(actual, expected, output, filename):
    actual, expected = locked.stripped(actual), locked.stripped(expected)
    differences = [(a, b) for a, b in zip(actual, expected) if a != b]
    receipt = {'passed': actual == expected, 'samples': len(actual), 'expected_samples': len(expected),
        'prediction_digest': frozen.joint.prediction_digest(actual),
        'expected_prediction_digest': frozen.joint.prediction_digest(expected),
        'first_differences': differences[:20], 'comparison_fields': ['sample_id', 'label', 'prediction']}
    raw.write(output / filename, receipt)
    require(receipt['passed'], 'Fixed-checkpoint replay differs: ' + str(output / filename))
    return receipt


def run_seed(repo, graph_root, output, settings, protocol, manifest, inputs, seed, resume=False, probe_only=False):
    folder = output / ('seed' + str(seed))
    folder.mkdir(exist_ok=True)
    require(raw.read(output / 'checkpoint_lock_receipt.json')['passed'], 'Lock all checkpoints before graph loading')
    seal = folder / 'artifact_manifest.json'
    if seal.exists():
        require(resume, 'Completed fixed test exists; use --resume for verification/reuse')
        old = raw.read(folder / 'provenance.json')
        require(old['protocol_lock_sha256'] == locked.archive.canonical_sha(protocol), 'Completed test protocol differs')
        require(graph_root is None or str(graph_root.resolve()) == old['binding']['graph_root'], 'Graph root changed')
        completed = locked.archive.verify_sealed(folder)
        require(completed['backbone_seed'] == seed
                and completed['source_binding'] == manifest['seeds'][str(seed)]['source_binding']
                and completed['model_metadata'] == manifest['seeds'][str(seed)]['models'],
                'Completed fixed-test backbone/checkpoint identities differ')
        return {'status': 'completed', 'backbone_seed': seed, 'test_evaluated': True,
                'reused_completed_test': True, 'new_optimizer_updates': 0, 'samples_per_model': settings['test_samples']}
    started = time.monotonic()
    device = frozen.independent.d32.baseline.runtime(frozen.independent.core)
    torch.set_num_threads(2)
    data, rows, config, paths = load_data(repo, graph_root, settings)
    counts = frozen.independent.d32.guard_graph_loads(data, {'val'} if probe_only else {'val', 'test'})
    require(locked.archive.canonical_sha([(r['sample_id'], r['label']) for r in rows['test']]) == manifest['test_cohort_sha256'],
            'Current fixed test cohort differs from the locked historical cohort')
    require(locked.archive.canonical_sha([(r['sample_id'], r['label']) for r in rows['val']]) == manifest['validation_cohort_sha256'],
            'Current validation cohort differs from locked selections')
    source, models, source_config = locked.restore_models(inputs, settings, seed, device)
    before = locked.frozen_digests(source, models)
    binding = {'protocol_lock_sha256': locked.archive.canonical_sha(protocol),
        'source': settings['seeds'][str(seed)]['source_binding'], 'environment': environment(device),
        'graph_root': str(data.root), 'input_sha256': {str(p.relative_to(repo)): h for p, h in paths.items()},
        'eval_batch_size': settings['eval_batch_size'], 'workers': settings['workers']}
    provenance = {'protocol_lock_sha256': binding['protocol_lock_sha256'], 'binding': binding,
                  'source_C_checkpoint_config': source_config, 'initial_model_state_sha256': before,
                  'new_training_runs': 0, 'new_optimizer_updates': 0}
    path = folder / 'provenance.json'
    if path.exists():
        require(resume and raw.read(path) == provenance, 'Fixed-test resume source/config/environment differs')
    raw.write(path, provenance)
    val, reference, coverage = bound_cache(source, data, rows['val'], 'val', config, device, folder, binding)
    records = predict(models, val, rows['val'], settings['eval_batch_size'], device)
    exported = locked.results.diagnostics.exports(records)
    validation_receipts = {}
    for name in locked.MODELS:
        expected_path = manifest['seeds'][str(seed)]['models'][name]['validation_predictions_path']
        validation_receipts[name] = compare_replay(exported[name], inputs.rows(expected_path), folder,
                                                   name + '_validation_replay.json')
        metric = frozen.joint.study.validate_predictions(exported[name], rows['val'], probabilities=False)
        require(metric == manifest['seeds'][str(seed)]['models'][name]['validation'], 'Validation metrics differ')
    raw.write(folder / 'selected_validation_replay.json', validation_receipts)
    locked.verify_frozen(source, models, before)
    del val, records
    if probe_only:
        require(counts['train'] == counts['test'] == 0, 'Probe accessed a forbidden graph partition')
        raw.write(folder / 'server_probe.json', {'passed': True, 'validation_samples_per_model': len(rows['val']),
            'models': 4, 'test_evaluated': False, 'test_graph_loads': 0, 'new_optimizer_updates': 0})
        return {'status': 'server_probe_passed', 'backbone_seed': seed, 'test_evaluated': False,
                'new_optimizer_updates': 0}
    raw.write(folder / 'test_started.json', {'checkpoint_manifest_sha256': protocol['checkpoint_manifest_sha256'],
        'models_locked_before_test': True, 'started_utc': datetime.now(timezone.utc).isoformat()})
    test, reference, test_coverage = bound_cache(source, data, rows['test'], 'test', config, device, folder, binding)
    # Verify same-weight historical C before interpreting any new-head fixed-test results.
    baseline = compare_replay(reference, locked.historical_predictions(inputs, settings, seed), folder,
                              'historical_base_C_test_replay.json')
    records = predict(models, test, rows['test'], settings['eval_batch_size'], device)
    exported = locked.results.diagnostics.exports(records)
    compare_replay(exported['base_C'], reference, folder, 'shared_cache_base_C_replay.json')
    locked.verify_frozen(source, models, before)
    frozen.evaluation.verify_inputs(paths)
    inputs.verify_all()
    require(counts['train'] == 0, 'Fixed test loaded training graphs')
    summary = report.write_seed(folder, records, rows['test'], seed, manifest['seeds'][str(seed)],
        {'environment': binding['environment'], 'wall_seconds': time.monotonic() - started,
         'main_process_graph_load_counters': counts,
         'graph_guard_scope': 'all dataset calls, including worker processes; counters are process-local',
         'shared_frozen_features': True})
    raw.write(folder / 'coverage.json', {'val': coverage, 'test': test_coverage})
    raw.write(folder / 'final_verification.json', {'passed': True, 'models': 4,
        'samples_per_model': len(rows['test']), 'all_validation_replays_passed': True,
        'same_weight_historical_C_predictions_identical': baseline['passed'],
        'frozen_parameters_buffers_modes_unchanged': True, 'source_inputs_unchanged': True,
        'new_training_runs': 0, 'new_optimizer_updates': 0, 'train_graph_loads': counts['train'],
        'independent_prediction_audit_passed': summary['independent_verification']['passed']})
    raw.write(folder / 'status.json', {'status': 'completed', 'backbone_seed': seed,
        'test_evaluated': True, 'new_optimizer_updates': 0})
    locked.archive.seal(folder)
    return {'status': 'completed', 'backbone_seed': seed, 'test_evaluated': True,
            'models': 4, 'samples_per_model': len(rows['test']), 'new_optimizer_updates': 0}


def run_phase(repo, graph_root, output, settings, phase, resume=False, probe_only=False, download=False):
    inputs, protocol, manifest = locked.lock_inputs(repo, output, settings, download)
    if phase == 'lock':
        return {'status': 'checkpoint_lock_completed', 'models_locked': 12,
                'test_evaluated': False, 'new_optimizer_updates': 0}
    if phase in ('seed42', 'seed43', 'seed44'):
        return run_seed(repo, graph_root, output, settings, protocol, manifest, inputs,
                        int(phase[-2:]), resume, probe_only)
    require(phase == 'aggregate' and not probe_only, 'Unexpected fixed-test phase')
    summaries, cohort = {}, None
    for seed in (42, 43, 44):
        folder = output / ('seed' + str(seed))
        require(raw.read(folder / 'provenance.json')['protocol_lock_sha256'] == locked.archive.canonical_sha(protocol),
                'Seed output uses another locked protocol')
        s = locked.archive.verify_sealed(folder)
        require(s['backbone_seed'] == seed and s['source_binding'] == manifest['seeds'][str(seed)]['source_binding']
                and s['model_metadata'] == manifest['seeds'][str(seed)]['models'],
                'Fixed-test backbone/checkpoint identities differ during aggregation')
        records = frozen.joint.legacy.predictions(folder / 'test/predictions.jsonl.gz')
        require(len(records) == settings['test_samples'], 'Fixed-test export is incomplete')
        identities = [(r['sample_id'], r['label']) for r in records]
        if cohort is None:
            cohort = identities
        require(cohort == identities and locked.archive.canonical_sha(identities) == manifest['test_cohort_sha256'],
                'Fixed-test cohorts differ')
        locked.results.independently_verify(records, s['metrics'], s['comparisons'])
        for name, predictions in locked.results.diagnostics.exports(records).items():
            require(frozen.joint.prediction_digest(predictions) == s['prediction_digests'][name], 'Prediction digest differs')
        summaries[seed] = s
    negatives = {name: {**spec, 'source_result_commit': locked.SNAPSHOT,
        'archived_summary': inputs.json(spec['summary'])} for name, spec in settings['previous_negative_ablations'].items()}
    result = report.aggregate(summaries, output / 'aggregate', protocol, negatives)
    raw.write(output / 'summary.json', result)
    for filename in ('report.md', 'paper_tables.md'):
        locked.atomic_bytes(output / filename, (output / 'aggregate' / filename).read_bytes())
    raw.write(output / 'final_verification.json', {'passed': True, 'models_evaluated': 12,
        'samples_per_model': settings['test_samples'], 'same_test_cohort': True,
        'all_validation_and_historical_C_replays_passed': True, 'independent_prediction_audits_passed': True,
        'new_training_runs': 0, 'new_optimizer_updates': 0, 'test_reselection_permitted': False})
    return {'status': 'completed', 'test_evaluated': True, 'models_evaluated': 12,
            'samples_per_model': settings['test_samples'], 'new_training_runs': 0, 'new_optimizer_updates': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--phase', choices=('lock', 'seed42', 'seed43', 'seed44', 'aggregate'), required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--probe-only', action='store_true')
    parser.add_argument('--download-inputs', action='store_true')
    args = parser.parse_args()
    repo, output = args.repo.resolve(), args.repo.resolve() / 'outputs' / OUTPUT_NAME
    output.mkdir(parents=True, exist_ok=True)
    (repo / 'tmp').mkdir(exist_ok=True)
    with FileLock(str(repo / 'tmp' / (OUTPUT_NAME + '.run.lock')), timeout=0):
        try:
            raw.write(output / 'status.json', {'status': 'running', 'phase': args.phase,
                'queue_completed': False, 'new_optimizer_updates': 0})
            result = run_phase(repo, args.data_root, output, raw.read(CONFIG), args.phase,
                               args.resume, args.probe_only, args.download_inputs)
        except BaseException as exc:
            failure = {'status': 'failed', 'phase': args.phase, 'exception': repr(exc),
                'traceback': traceback.format_exc(), 'new_optimizer_updates': 0,
                'test_evaluation_started': any(output.glob('seed*/test_started.json'))}
            raw.write(output / 'status.json', failure)
            failed = output / 'failures'
            failed.mkdir(exist_ok=True)
            raw.write(failed / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json'), failure)
            raise
        raw.write(output / 'status.json', {**result, 'phase': args.phase,
            'queue_completed': args.phase == 'aggregate'})
        print(json.dumps({**result, 'phase': args.phase, 'output': str(output)}), flush=True)


if __name__ == '__main__':
    main()
