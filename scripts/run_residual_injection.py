"""Seed42 R0/R1/R2 matched training from original C, with real-graph checks and recovery."""
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

import run_joint_transition_fusion as parent
import residual_injection_training as training
from run_transition_pair_cross_replay import Archive
from xgnid.residual_injection import ARMS

raw, frozen, original, require = parent.raw, parent.frozen, parent.original, parent.require
OUTPUT_NAME = 'residual_injection_s42_v1'
OUTPUT = 'outputs/' + OUTPUT_NAME
CONFIG = raw.ROOT / ('configs/' + OUTPUT_NAME + '.json')
SOURCE_FILES = (*parent.SOURCE_FILES, 'src/xgnid/residual_injection.py',
                'scripts/run_residual_injection.py', 'scripts/residual_injection_training.py',
                'scripts/residual_injection_results.py', 'scripts/launch_residual_injection.py',
                'scripts/launch_joint_transition_fusion.py', 'scripts/run_transition_pair_cross_replay.py')


def validate_settings(settings):
    fixed = {'backbone_seed': 42, 'readout_seed': 42, 'injection_gate_seed': 20261123,
             'injection_epsilon': 1e-8, 'injection_input_width': 97, 'injection_hidden_width': 16,
             'injection_activation': 'GELU', 'layer_norm_epsilon': 1e-5,
             'layer_norm_affine': False, 'initial_beta': 1.0, 'pairing': 'true_continuous',
             'injection_formula': 'packet + (2 * sigmoid(g)) * residual',
             'injection_input': 'LN(zf_pre);LN(zp_pre);abs(LN(zf_pre)-LN(zp_pre));log1p(norm(r)/(norm(p)+epsilon))',
             'global_input': 'fixed_zero_vector_97', 'epochs': 200, 'batch_size': 512,
             'eval_batch_size': 16, 'training_samples': 107621, 'validation_samples': 13446,
             'test_samples': 13448, 'updates_per_epoch': 211, 'updates_per_arm': 42200,
             'readout_learning_rate': 0.001, 'injection_learning_rate': 0.001,
             'fusion_learning_rate': 0.0001, 'optimizer': 'Adam', 'adam_betas': [0.9, 0.999], 'adam_eps': 1e-8,
             'weight_decay': 0.0, 'checkpoint_every_paired_update': True,
             'loss': 'plain_final_eight_class_NLL',
             'selection': 'best_validation_macro_f1_including_zero_earliest_tie',
             'partitions': ['train', 'val'], 'test_evaluated': False,
             'historical_true_macro_f1': 0.9791403291717622,
             'investment_reference_pp': {'inj_dynamic-inj_fixed': 0.10, 'inj_dynamic-inj_global': 0.05},
             'total_parameters': {'inj_fixed': 10872, 'inj_global': 12457, 'inj_dynamic': 12457}}
    require(all(settings[k] == value for k, value in fixed.items()) and settings['arms'] == list(ARMS),
            'Registered residual-injection protocol changed')
    require(len(training.parent.PairedSampler(107621, 512, 42)) == settings['updates_per_epoch'], 'Wrong update budget')


def history_inputs(repo, settings):
    spec = settings['historical_archive']
    archive = Archive(repo, spec, snapshot=not (repo / spec['source_directory']).is_dir())
    archive.verify_all()
    history = {name: archive.json(name + '.json') for name in ('summary', 'provenance', 'config', 'status', 'final_verification')}
    predictions = archive.rows('pair_true/best_validation_predictions.jsonl.gz')
    metric = frozen.joint.study.validate_predictions(predictions, probabilities=False)
    require(history['status']['status'] == 'completed' and history['final_verification']['passed']
            and not history['summary']['test_evaluated']
            and metric == history['summary']['metrics']['pair_true']
            and metric['macro_f1'] == settings['historical_true_macro_f1']
            and history['summary']['selected_epochs']['pair_true'] == 191,
            'Archived True reference changed')
    for key in ('epochs', 'batch_size', 'eval_batch_size', 'readout_seed', 'readout_learning_rate',
                'fusion_learning_rate', 'loss', 'selection'):
        require(history['config'][key] == settings[key], 'Previous True protocol differs: ' + key)
    return history, predictions, archive


def raw_probe(source, data, config, settings, device, output):
    candidates = raw.choose_rows(data, 'train', per_class=16)
    for _, _, batch, rows in original.diagnosis.aligned_batches(data, candidates, 'train', config):
        features, _ = frozen.encode(source, batch, device)
        if not bool(features.transitions[:, 70].any()):
            continue
        result = training.gradient_probe(source, features, batch.y.view(-1).to(device), settings, device)
        result.update(mode='raw_graph_preflight', device=str(device), formal_training_started=False,
                      sample_ids=[r['sample_id'] for r in rows])
        raw.write(output / 'raw_gradient_probe.json', result)
        return result
    raise ValueError('No real continuous context in gradient preflight')


@torch.no_grad()
def selected_raw_replay(source, arms, data, rows, config, device, output, state, historical_predictions):
    from residual_injection_results import observe, exports
    for name, arm in arms.items():
        arm.load_state_dict(state['best'][name]['state'], strict=True)
        arm.requires_grad_(False).eval()
    reference = training.parent.JointTransitionFusion(source.fusion, 'joint_base', 42).to(device).requires_grad_(False).eval()
    require([(r['sample_id'], r['label']) for r in historical_predictions] ==
            [(r['sample_id'], r['label']) for r in rows], 'Historical validation membership/order differs')
    records = []
    for step, indices, batch, expected in original.diagnosis.aligned_batches(data, rows, 'val', config):
        features, _ = frozen.encode(source, batch, device)
        current = observe(arms, features, expected, reference)
        for row, previous in zip(current, historical_predictions[indices.start:indices.stop]):
            row['predictions']['historical_True'] = previous['prediction']
        records.extend(current)
        if step == 1 or step % 200 == 0 or indices.stop == len(rows):
            raw.progress(output, 'best_injection_raw_validation_replay', samples=indices.stop, total=len(rows))
    exported, receipts = exports(records), {}
    for name in ARMS:
        metric = frozen.joint.study.validate_predictions(exported[name], rows, probabilities=False)
        digest = frozen.joint.prediction_digest(exported[name])
        require(metric == state['best'][name]['validation'] and digest == state['best'][name]['prediction_ids_sha256'],
                name + ' selected raw-graph replay differs')
        receipts[name] = {'passed': True, 'samples': len(rows), 'epoch': state['best'][name]['epoch'],
                          'prediction_ids_sha256': digest, 'pairing': 'true_continuous'}
    raw.write(output / 'selected_validation_replay.json', receipts)
    return records


def run(repo, graph_root, output, settings, resume=False, probe_only=False):
    started = time.monotonic()
    validate_settings(settings)
    raw.write(output / 'preparation_environment.json', {'python': platform.python_version(),
        'torch': str(torch.__version__), 'pyg': torch_geometric.__version__,
        'cuda_available': torch.cuda.is_available(), 'formal_training_started': False})
    history, historical_predictions, archive = history_inputs(repo, settings)
    source, data, rows, device, config, selected, gold, inputs, binding = frozen.load_source(repo, 42, graph_root, output)
    require(binding == history['provenance']['binding']['source'], 'Wrong original C for injection experiment')
    require(len(rows['train']) == settings['training_samples'] and len(rows['val']) == settings['validation_samples']
            and data.metadata['partition_sizes']['test'] == settings['test_samples'], 'Fixed split sizes differ')
    config.update(batch_size=settings['batch_size'], eval_batch_size=settings['eval_batch_size'])
    digest = raw.state_digest(source.state_dict())
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=raw.ROOT, text=True).strip()
    environment = {'python': platform.python_version(), 'torch': str(torch.__version__),
                   'pyg': torch_geometric.__version__, 'cuda': torch.version.cuda,
                   'device': torch.cuda.get_device_name(device) if device.type == 'cuda' else str(device),
                   'platform': platform.platform()}
    run_binding = {'source': binding, 'code_commit': commit, 'settings_sha256': raw.sha(CONFIG),
        'source_files_sha256': {name: raw.sha(raw.ROOT / name) for name in SOURCE_FILES},
        'environment': environment, 'graph_root': str(data.root), 'workers': config['workers'],
        'input_sha256': {str(p.relative_to(repo)): sha for p, sha in inputs.items()},
        'historical_inputs': archive.verified}
    if (output / 'provenance.json').exists():
        require(resume and raw.read(output / 'provenance.json')['binding'] == run_binding,
                'Existing injection source/config/code/environment differs')
    raw.write(output / 'config.json', settings)
    raw.write(output / 'provenance.json', {'binding': run_binding, 'environment': environment,
        'test_graph_loads': 0, 'historical_True_is_external_reference_only': True})
    raw_probe(source, data, config, settings, device, output)
    training.parent.verify_source(source, digest)
    val, reference, coverage_val = parent.cached_split(source, data, rows['val'], 'val', config, device, output, run_binding)
    parent.verify_source_replay(reference, gold, selected, rows['val'], output)
    arms = training.make_arms(source, settings, device)
    parent.initial_replay(source, arms, val, rows['val'], gold, selected, device, settings, output)
    raw.write(output / 'server_probe.json', {'passed': True, 'formal_training_started': False,
        'real_graph_gradients': True, 'five_staged_probe_updates_discarded': True,
        'full_epoch_zero_replay': True, 'initial_beta': 1.0, 'independent_rng_stream': True})
    if probe_only:
        training.parent.verify_source(source, digest)
        archive.verify_all(); frozen.evaluation.verify_inputs(inputs)
        return {'status': 'server_probe_passed', 'formal_training_started': False, 'test_evaluated': False}
    train, _, coverage_train = parent.cached_split(source, data, rows['train'], 'train', config, device, output, run_binding)
    raw.write(output / 'coverage.json', {'train': coverage_train, 'val': coverage_val})
    training.parent.verify_source(source, digest)
    run_binding = {**run_binding, 'frozen_cache_sha256': {
        part: raw.sha(output / ('frozen_' + part + '_features.pt')) for part in ('train', 'val')}}
    raw.seed_all(42)
    state = training.fit(arms, train, val, rows['val'], reference, settings, run_binding, device, output, resume=resume)
    del train
    from residual_injection_results import cached_beta_summary, write_results
    raw.write(output / 'last_beta_distributions.json', cached_beta_summary(arms, val, rows['val'], settings, device))
    del val
    records = selected_raw_replay(source, arms, data, rows['val'], config, device, output, state, historical_predictions)
    parent.verify_source_replay(parent.diagnostics.exports(records)['original_C'], gold, selected, rows['val'], output)
    write_results(output, records, rows['val'], state, settings)
    training.parent.verify_source(source, digest)
    archive.verify_all(); frozen.evaluation.verify_inputs(inputs)
    raw.write(output / 'final_verification.json', {'passed': True, 'source_C_unchanged': True,
        'source_inputs_unchanged': True, 'same_sample_order_all_three_arms': True,
        'updates_per_arm': state['updates'], 'epochs_per_arm': state['completed_epoch'],
        'best_raw_graph_replays_passed': True, 'test_graph_loads': 0, 'seconds': time.monotonic() - started})
    return {'status': 'completed', 'formal_training_started': True, 'test_evaluated': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    repo, output = args.repo.resolve(), args.repo.resolve() / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    (repo / 'tmp').mkdir(exist_ok=True)
    with FileLock(str(repo / 'tmp' / (OUTPUT_NAME + '.run.lock')), timeout=0):
        try:
            result = run(repo, args.data_root, output, raw.read(CONFIG), args.resume, args.probe_only)
        except BaseException as exc:
            raw.write(output / 'status.json', {'status': 'failed', 'exception': repr(exc),
                'traceback': traceback.format_exc(), 'test_evaluated': False,
                'formal_training_started': (output / 'recovery.pt').exists()})
            raise
        else:
            raw.write(output / 'status.json', result)
            print(json.dumps({**result, 'output': str(output)}), flush=True)


if __name__ == '__main__':
    main()
