"""Seed42 J0/J1/J2/J3: original-C-bound, validation-only joint fusion training."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import time
import traceback

from filelock import FileLock
import torch
import torch_geometric

import joint_transition_training as training
import joint_transition_diagnostics as diagnostics
import diagnose_joint_transition_fusion as diagnosis

frozen, original, raw, require = training.frozen, training.original, training.raw, training.require
CONFIG = raw.ROOT / 'configs/joint_transition_fusion_s42_v1.json'
OUTPUT = 'outputs/joint_transition_fusion_s42_v1'
ARMS = training.ARMS
SOURCE_FILES = (
    'scripts/run_joint_transition_fusion.py', 'scripts/joint_transition_training.py',
    'scripts/joint_transition_diagnostics.py', 'scripts/diagnose_joint_transition_fusion.py',
    'scripts/run_ordered_transition_readout.py', 'scripts/run_packet_transition_readout.py',
    'scripts/run_packet_transition_diagnosis.py', 'scripts/replay_flow_query_selective.py',
    'scripts/run_independent_packet_transition.py', 'scripts/run_joint_cross_layer_query.py',
    'scripts/evaluate_joint_cross_layer_query.py', 'scripts/run_attention_operator_pair.py',
    'scripts/audit_packet_transitions.py', 'scripts/grouped_graphs.py',
    'src/xgnid/joint_transition_fusion.py', 'src/xgnid/ordered_transition_readout.py',
    'src/xgnid/packet_transition_readout.py', 'src/xgnid/packet_pooling.py',
    'src/xgnid/joint_cross_layer_query.py', 'src/xgnid/fusion_dimension.py', 'src/xgnid/model.py')


def validate_settings(settings):
    fixed = {'backbone_seed': 42, 'readout_seed': 42, 'epochs': 200, 'batch_size': 512,
             'eval_batch_size': 16, 'training_samples': 107621, 'validation_samples': 13446,
             'test_samples': 13448, 'updates_per_epoch': 211, 'updates_per_arm': 42200,
             'fusion_learning_rate': 0.0001, 'readout_learning_rate': 0.001,
             'partitions': ['train', 'val'], 'test_evaluated': False, 'adam_betas': [0.9, 0.999],
             'adam_eps': 1e-8, 'weight_decay': 0.0, 'checkpoint_every_paired_update': True}
    require(all(settings[k] == v for k, v in fixed.items()) and settings['arms'] == list(ARMS),
            'Registered seed42 protocol changed')
    require(len(training.PairedSampler(107621, 512, 42)) == 211, 'Wrong sampler budget')


def history_inputs(repo, settings):
    folder = repo / settings['historical_directory']
    inputs = {folder / name: digest for name, digest in settings['historical_sha256'].items()}
    frozen.evaluation.verify_inputs(inputs)
    provenance, selection = raw.read(folder / 'provenance.json'), raw.read(folder / 'selection.json')
    require(provenance['code_commit'] == settings['historical_code_commit']
            and selection['binding'] == provenance['source'] and not selection['test_evaluated'],
            'Historical Ordered lineage differs')
    records = frozen.joint.legacy.predictions(folder / 'validation/validation_predictions.jsonl.gz')
    require(len(records) == settings['validation_samples'], 'Historical validation count differs')
    return folder, inputs, provenance, selection, records


def reference_arms(source, history, binding, settings, device):
    arms = {'original_C': training.JointTransitionFusion(source.fusion, 'joint_base', 42).to(device)}
    for mode in ('single', 'pointwise', 'ordered'):
        saved = torch.load(history / mode / 'readout.pt', map_location='cpu', weights_only=True)
        require(saved['binding'] == binding and saved['mode'] == mode, 'Frozen readout source binding differs')
        arm = training.JointTransitionFusion(source.fusion, 'joint_' + mode, 42).to(device)
        arm.readout.load_state_dict(saved['best']['B']['state'], strict=True)
        arms['frozen_' + mode] = arm
    for arm in arms.values():
        arm.requires_grad_(False).eval()
    return arms


def cached_split(source, data, rows, part, config, device, output, binding):
    path = output / ('frozen_' + part + '_features.pt')
    if path.exists():
        saved = torch.load(path, map_location='cpu', weights_only=True)
        require(saved['binding'] == binding and saved['ids'] == [r['sample_id'] for r in rows]
                and saved['labels'] == [r['label'] for r in rows], 'Frozen cache source/split/config mismatch')
        features = original.Features(*(saved['features'][key] for key in original.Features._fields))
        cache = original.FrozenCache([features], saved['labels'])
        return cache, saved['reference'], saved['coverage']
    cache, reference, coverage = frozen.cache_split(source, data, rows, part, config, device, output)
    training.atomic_save(path, {'binding': binding, 'ids': [r['sample_id'] for r in rows],
        'labels': [r['label'] for r in rows], 'features': cache.features._asdict(),
        'reference': reference, 'coverage': coverage})
    return cache, reference, coverage


def verify_source_replay(reference, gold, selected, rows, output):
    """Compare prediction identity and metrics; archived gate vectors are diagnostic metadata."""
    fields = ('sample_id', 'label', 'prediction')
    receipt = {'passed': False, 'samples': len(reference), 'comparison_fields': list(fields),
               'archive_extra_fields': sorted({key for row in gold for key in row if key not in fields})}
    try:
        measured = frozen.joint.study.validate_predictions(reference, rows, probabilities=False)
        archived = frozen.joint.study.validate_predictions(gold, rows, probabilities=False)
        receipt.update(frozen.evaluation.replay_receipt(measured, reference, selected, gold))
        receipt['archive_metrics_match'] = archived == selected['validation']
        receipt['passed'] = receipt['passed'] and receipt['archive_metrics_match']
    except (ValueError, KeyError, TypeError) as exc:
        receipt.update(passed=False, error=str(exc))
    # Persist failed comparisons as well, so genuine prediction differences remain inspectable.
    raw.write(output / 'baseline_validation_replay.json', receipt)
    require(receipt['passed'], 'Original-C raw validation reference differs; see baseline_validation_replay.json')
    return receipt


@torch.no_grad()
def initial_replay(source, arms, val, rows, gold, selected, device, settings, output):
    initial_fusion = raw.state_digest(source.fusion.state_dict())
    max_errors, predictions = {name: 0.0 for name in arms}, {name: [] for name in arms}
    for start in range(0, len(rows), settings['eval_batch_size']):
        features, _ = val.batch(range(start, min(start + settings['eval_batch_size'], len(rows))), device)
        base = original.baseline(source, features)
        for name, arm in arms.items():
            scores, residual, _, _ = arm(features)
            require(torch.equal(scores, base) and not bool(residual.any()), f'{name} initial scores differ')
            max_errors[name] = max(max_errors[name], float((scores - base).abs().max()))
            predictions[name].extend(scores.argmax(-1).tolist())
    records = {name: original.prediction_records(rows, values) for name, values in predictions.items()}
    receipts = {name: frozen.evaluation.replay_receipt(
        frozen.joint.study.validate_predictions(items, rows, probabilities=False), items, selected, gold)
        for name, items in records.items()}
    require(all(item['passed'] for item in receipts.values()), 'Epoch-zero replay differs from original C')
    result = {'passed': True, 'samples': len(rows), 'source_C_receipts': receipts,
              'fusion_initial_sha256': initial_fusion, 'max_abs_log_probability_errors': max_errors,
              'common_readout_initialization_verified': True, 'context_initialization_verified': True,
              'independent_parameters_and_optimizers': True}
    raw.write(output / 'epoch_zero_replay.json', result)
    return result


@torch.no_grad()
def cached_diagnosis(arms, val, rows, settings, device):
    records = []
    for start in range(0, len(rows), settings['eval_batch_size']):
        end = min(start + settings['eval_batch_size'], len(rows))
        features, _ = val.batch(range(start, end), device)
        records.extend(diagnostics.observe(arms, features, rows[start:end]))
    return records


def historical_diagnosis(refs, val, rows, archived, settings, device, coverage, output):
    records = cached_diagnosis(refs, val, rows, settings, device)
    predicted = diagnostics.exports(records)
    for name, items in predicted.items():
        old_name = 'A' if name == 'original_C' else name.removeprefix('frozen_')
        expected = original.prediction_records(archived, [r['predictions'][old_name] for r in archived])
        require(items == expected, f'{name} historical validation replay differs')
    folder = output / 'historical_diagnosis'
    folder.mkdir(exist_ok=True)
    # Keep the complete replay, including per-sample continuity and both attention scales.
    raw.save_predictions(folder / 'validation_replay_diagnostics.jsonl.gz', records)
    renamed = [{**r, 'predictions': {k.removeprefix('frozen_'): v for k, v in r['predictions'].items()}}
               for r in records]
    diagnosis.write_report(folder, renamed, coverage['context_coverage_by_class'], {
        'archive_commit': settings['historical_commit'], 'raw_graph_features_replayed': True,
        'all_historical_predictions_identical': True, 'per_sample_continuity_available': True,
        'attention_residual_and_gate_available': True})
    raw.write(folder / 'diagnostic_groups.json', diagnostics.aggregate(records, 'frozen_single', 'frozen_ordered'))
    raw.write(folder / 'replay_receipt.json', {'passed': True, 'samples': len(rows),
        'prediction_ids_sha256': {k: frozen.joint.prediction_digest(v) for k, v in predicted.items()},
        'continuity_rule': 'two_valid_forward_links_sharing_the_middle_packet',
        'not_inferred_from_edge_counts': True})


@torch.no_grad()
def selected_raw_replay(source, arms, refs, data, rows, config, device, output, state):
    for name, arm in arms.items():
        arm.load_state_dict(state['best'][name]['state'], strict=True)
        arm.requires_grad_(False).eval()
    all_arms, records = {**refs, **arms}, []
    for step, indices, batch, expected in original.diagnosis.aligned_batches(data, rows, 'val', config):
        features, _ = frozen.encode(source, batch, device)
        records.extend(diagnostics.observe(all_arms, features, expected))
        if step == 1 or step % 200 == 0 or indices.stop == len(rows):
            raw.progress(output, 'best_raw_validation_replay', samples=indices.stop, total=len(rows))
    predicted = diagnostics.exports(records)
    receipts = {}
    for name in ARMS:
        metric = frozen.joint.study.validate_predictions(predicted[name], rows, probabilities=False)
        require(metric == state['best'][name]['validation'] and
                frozen.joint.prediction_digest(predicted[name]) == state['best'][name]['prediction_ids_sha256'],
                f'{name} selected raw-graph replay differs from cached selection')
        receipts[name] = {'passed': True, 'samples': len(rows), 'selected_epoch': state['best'][name]['epoch'],
                          'prediction_ids_sha256': frozen.joint.prediction_digest(predicted[name])}
    raw.write(output / 'selected_validation_replay.json', receipts)
    return records


def run(repo, graph_root, output, settings, resume=False, probe_only=False):
    started = time.monotonic()
    validate_settings(settings)
    history, inputs, historical, selection, archived = history_inputs(repo, settings)
    source, data, rows, device, config, selected, gold, source_inputs, binding = frozen.load_source(
        repo, 42, graph_root, output)
    inputs.update(source_inputs)
    require(binding == historical['source'] == selection['binding'], 'Original C is not the historical experiment source')
    require(len(rows['train']) == settings['training_samples'] and len(rows['val']) == settings['validation_samples']
            and data.metadata['partition_sizes']['test'] == settings['test_samples'], 'Fixed split counts differ')
    config.update(batch_size=settings['batch_size'], eval_batch_size=settings['eval_batch_size'])
    source_digest = raw.state_digest(source.state_dict())
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=raw.ROOT, text=True).strip()
    sources = {name: raw.sha(raw.ROOT / name) for name in SOURCE_FILES}
    environment = {'python': platform.python_version(), 'torch': str(torch.__version__),
                   'pyg': torch_geometric.__version__, 'cuda': torch.version.cuda,
                   'device': torch.cuda.get_device_name(device), 'platform': platform.platform()}
    run_binding = {'source': binding, 'settings_sha256': raw.sha(CONFIG), 'code_commit': commit,
                   'source_files_sha256': sources, 'environment': environment,
                   'input_sha256': {str(path.relative_to(repo)): value for path, value in inputs.items()},
                   'graph_root': str(data.root), 'workers': config['workers']}
    if (output / 'provenance.json').exists():
        require(resume and raw.read(output / 'provenance.json')['binding'] == run_binding,
                'Existing output provenance/config/code/source differs; use a new directory')
    raw.write(output / 'config.json', settings)
    raw.write(output / 'provenance.json', {'binding': run_binding, 'test_graph_loads': 0,
              'code_root': str(raw.ROOT), 'data_repo': str(repo), 'environment': environment})
    probe_rows = raw.choose_rows(data, 'train', per_class=2)
    probe_config = {**config, 'eval_batch_size': len(probe_rows), 'workers': 0}
    _, _, batch, _ = next(original.diagnosis.aligned_batches(data, probe_rows, 'train', probe_config))
    features, _ = frozen.encode(source, batch, device)
    raw.seed_all(42)
    probe = training.gradient_probe(source, features, batch.y.view(-1).to(device), settings, device)
    probe.update(mode='server_raw_graph_cuda', formal_training_started=False,
                 sample_ids=[r['sample_id'] for r in probe_rows])
    raw.write(output / 'raw_cuda_probe.json', probe)
    training.verify_source(source, source_digest)
    val, reference, coverage_val = cached_split(source, data, rows['val'], 'val', config, device, output, run_binding)
    verify_source_replay(reference, gold, selected, rows['val'], output)
    require(coverage_val == raw.read(history / 'coverage.json')['val'], 'Historical true continuity coverage differs')
    arms = training.make_arms(source, settings, device)
    for name, arm in arms.items():
        require(sum(p.numel() for p in arm.fusion.parameters()) == settings['fusion_parameters']
                and (0 if arm.readout is None else sum(p.numel() for p in arm.readout.parameters()))
                == settings['readout_parameters'][name], f'{name} parameter budget differs')
    initial_replay(source, arms, val, rows['val'], gold, selected, device, settings, output)
    refs = reference_arms(source, history, binding, settings, device)
    historical_diagnosis(refs, val, rows['val'], archived, settings, device, coverage_val, output)
    training.verify_source(source, source_digest)
    if probe_only:
        raw.write(output / 'server_probe.json', {'passed': True, 'formal_training_started': False,
                  'raw_cuda_probe': True, 'full_epoch_zero_replay': True, 'historical_diagnosis': True})
        return {'status': 'server_probe_passed', 'formal_training_started': False, 'test_evaluated': False}
    train, _, coverage_train = cached_split(source, data, rows['train'], 'train', config, device, output, run_binding)
    require(coverage_train == raw.read(history / 'coverage.json')['train'], 'Historical training continuity coverage differs')
    raw.write(output / 'coverage.json', {'train': coverage_train, 'val': coverage_val})
    training.verify_source(source, source_digest)
    run_binding = {**run_binding, 'frozen_cache_sha256': {
        part: raw.sha(output / ('frozen_' + part + '_features.pt')) for part in ('train', 'val')}}
    raw.seed_all(42)
    state = training.fit(arms, train, val, rows['val'], reference, settings, run_binding, device, output, resume=resume)
    del train, val
    records = selected_raw_replay(source, arms, refs, data, rows['val'], config, device, output, state)
    for name in refs:
        expected = original.prediction_records(archived, [r['predictions']['A' if name == 'original_C' else
                                                               name.removeprefix('frozen_')] for r in archived])
        require(diagnostics.exports(records)[name] == expected, 'Historical prediction changed during final replay')
    diagnostics.write_results(output, records, rows['val'], state, settings)
    training.verify_source(source, source_digest)
    frozen.evaluation.verify_inputs(inputs)
    raw.write(output / 'final_verification.json', {'passed': True, 'source_C_parameters_buffers_modes_unchanged': True,
        'source_inputs_unchanged': True, 'test_graph_loads': 0, 'updates_per_arm': state['updates'],
        'epochs_per_arm': state['completed_epoch'], 'same_sample_order_all_arms': True,
        'best_raw_graph_replays_passed': True, 'seconds': time.monotonic() - started})
    return {'status': 'completed', 'formal_training_started': True, 'test_evaluated': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output', type=Path, default=Path(OUTPUT))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = (repo / args.output).resolve()
    require(output == repo / OUTPUT, 'Use the registered independent output directory')
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / '.run.lock'), timeout=0):
        try:
            raw.write(output / 'status.json', {'status': 'preparing', 'test_evaluated': False,
                      'formal_training_started': (output / 'recovery.pt').exists()})
            result = run(repo, args.data_root, output, raw.read(CONFIG), args.resume, args.probe_only)
        except BaseException as exc:
            raw.write(output / 'status.json', {'status': 'failed', 'exception': repr(exc), 'test_evaluated': False,
                      'formal_training_started': (output / 'recovery.pt').exists(),
                      'traceback': traceback.format_exc()})
            raise
        else:
            raw.write(output / 'status.json', result)
            print(json.dumps({**result, 'output': str(output)}), flush=True)


if __name__ == '__main__':
    main()
