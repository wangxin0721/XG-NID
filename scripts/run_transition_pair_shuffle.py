"""Seed42 True vs fixed within-flow second-candidate derangements; train/val only."""
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
import joint_transition_training as training
import joint_transition_diagnostics as diagnostics
from transition_pair_cache import PairCache, mapping
from xgnid.transition_pair_shuffle import ARMS, VERSION, PairFusion

raw, frozen, original, require = parent.raw, parent.frozen, parent.original, parent.require
OUTPUT_NAME = 'transition_pair_shuffle_s42_v1'
OUTPUT = 'outputs/' + OUTPUT_NAME
CONFIG = raw.ROOT / ('configs/' + OUTPUT_NAME + '.json')
SOURCE_FILES = (*parent.SOURCE_FILES, 'scripts/run_transition_pair_shuffle.py',
                'scripts/transition_pair_cache.py', 'scripts/transition_pair_results.py',
                'scripts/launch_transition_pair_shuffle.py', 'scripts/launch_joint_transition_fusion.py',
                'src/xgnid/transition_pair_shuffle.py')


def validate_settings(settings):
    fixed = {'backbone_seed': 42, 'readout_seed': 42, 'pairing_seed': 20261121,
             'pairing_version': VERSION, 'epochs': 200, 'batch_size': 512, 'eval_batch_size': 16,
             'training_samples': 107621, 'validation_samples': 13446, 'test_samples': 13448,
             'updates_per_epoch': 211, 'updates_per_arm': 42200, 'readout_learning_rate': 0.001,
             'fusion_learning_rate': 0.0001, 'readout_parameters_per_arm': 3728,
             'fusion_parameters': 7144, 'total_parameters_per_arm': 10872,
             'investment_reference_pp': 0.05, 'partitions': ['train', 'val'], 'test_evaluated': False,
             'adam_betas': [0.9, 0.999], 'adam_eps': 1e-8, 'weight_decay': 0.0,
             'checkpoint_every_paired_update': True}
    require(all(settings[k] == v for k, v in fixed.items()) and settings['arms'] == list(ARMS),
            'Registered True/Shuffle protocol changed')
    require(len(training.PairedSampler(107621, 512, 42)) == 211, 'Wrong paired update budget')


def history_inputs(repo, settings):
    folder = repo / settings['historical_directory']
    inputs = {folder / name: digest for name, digest in settings['historical_sha256'].items()}
    frozen.evaluation.verify_inputs(inputs)
    history = {key: raw.read(folder / (key + '.json')) for key in
               ('provenance', 'summary', 'config', 'status', 'final_verification')}
    require(history['status']['status'] == 'completed' and history['final_verification']['passed']
            and history['provenance']['binding']['code_commit'] == settings['historical_code_commit']
            and not history['summary']['test_evaluated'], 'Historical J3 completion/lineage differs')
    for key in ('epochs', 'batch_size', 'eval_batch_size', 'readout_seed',
                'readout_learning_rate', 'fusion_learning_rate', 'loss', 'selection'):
        require(history['config'][key] == settings[key], 'J3 protocol differs: ' + key)
    predictions = frozen.joint.legacy.predictions(folder / 'joint_ordered/best_validation_predictions.jsonl.gz')
    metric = frozen.joint.study.validate_predictions(predictions, probabilities=False)
    require(metric == history['summary']['metrics']['joint_ordered']
            and history['summary']['selected_epochs']['joint_ordered'] == 84, 'Historical J3 prediction/selection differs')
    return history, predictions, inputs


def make_arms(source, settings, device):
    arms = {name: PairFusion(source.fusion, name, settings['readout_seed']).to(device) for name in ARMS}
    reference = raw.state_digest(arms[ARMS[0]].state_dict())
    require(all(raw.state_digest(arm.state_dict()) == reference for arm in arms.values()),
            'True/Shuffle initialization differs')
    pointers = [{p.data_ptr() for p in arm.parameters()} for arm in arms.values()]
    require(not pointers[0].intersection(pointers[1]) and not set.union(*pointers).intersection(
            p.data_ptr() for p in source.parameters()), 'Trainable parameter storage is shared')
    for name, arm in arms.items():
        require(raw.state_digest(arm.fusion.state_dict()) == raw.state_digest(source.fusion.state_dict())
                and sum(p.numel() for p in arm.readout.parameters()) == settings['readout_parameters_per_arm']
                and sum(p.numel() for p in arm.parameters()) == settings['total_parameters_per_arm']
                and not bool(arm.readout.output.weight.any()), f'{name} parameter/zero initialization differs')
    return arms


@torch.no_grad()
def structure_probe(source, features, settings, device):
    reference = training.JointTransitionFusion(source.fusion, 'joint_ordered', settings['readout_seed']).to(device).eval()
    generator = torch.Generator().manual_seed(915)
    reference.readout.output.weight.copy_(torch.randn(reference.readout.output.weight.shape, generator=generator).to(device) * 0.1)
    arms = make_arms(source, settings, device)
    for arm in arms.values():
        arm.load_state_dict(reference.state_dict(), strict=True)
        arm.eval()
    expected = reference(features)
    actual = arms['pair_true'](features)
    shuffled = arms['pair_shuffle'](features)
    require(all(torch.equal(a, b) for a, b in zip(expected, actual)), 'True differs from existing nonzero J3')
    require(torch.equal(actual[3], shuffled[3]), 'Pairing changed the single-scale attention')
    difference = float((actual[1] - shuffled[1]).abs().max())
    require(difference > 0 and bool(torch.isfinite(shuffled[1]).all()),
            'Nonzero-output check did not change the readout under second-candidate derangement')
    return {'passed': True, 'nonzero_output_weights': True, 'same_weights_both_pairings': True,
            'true_exactly_matches_J3': True, 'single_scale_attention_unchanged': True,
            'max_abs_residual_difference': difference}


def raw_probe(source, data, config, settings, device, output):
    candidates = raw.choose_rows(data, 'train', per_class=16)
    for _, _, batch, rows in original.diagnosis.aligned_batches(
            data, candidates, 'train', {**config, 'eval_batch_size': 16, 'workers': 0}):
        features, _ = frozen.encode(source, batch, device)
        paired, records = mapping(features, [r['sample_id'] for r in rows], 'train', settings['pairing_seed'])
        if not any(r['features_changed'] > 0 for r in records):
            continue
        probe = training.gradient_probe(source, paired, batch.y.view(-1).to(device), settings, device,
                                        arms=make_arms(source, settings, device))
        probe.update(mode='server_raw_graph_cuda', sample_ids=[r['sample_id'] for r in rows],
                     formal_training_started=False, structure=structure_probe(source, paired, settings, device))
        raw.write(output / 'raw_cuda_probe.json', probe)
        return probe
    raise ValueError('Raw training probe found no context whose right features change under derangement')


def prepare_pair_cache(base, rows, part, settings, output):
    state = raw.rng_hash(raw.rng_state())
    raw.progress(output, 'prepare_pair_mapping', partition=part, samples=len(rows))
    cache = PairCache(base, rows, part, settings['pairing_seed'])
    require(raw.rng_hash(raw.rng_state()) == state, 'Pair mapping consumed a global RNG stream')
    cache.save_mapping(output)
    require(cache.coverage['shuffleable_flows'] > 0 and cache.coverage['changed_right_features'] > 0,
            part + ' has no effective shuffle coverage')
    return cache


def observe(arms, features, rows, mapping_records):
    records = diagnostics.observe(arms, features, rows)
    for row, pair in zip(records, mapping_records):
        require((row['sample_id'], row['valid_pairs'], row['ordered_contexts']) ==
                (pair['sample_id'], pair['valid_pairs'], pair['ordered_contexts']), 'Replay pairing geometry differs')
        row.update(pairing=pair, shuffleable=pair['shuffleable'])
    return records


@torch.no_grad()
def selected_raw_replay(source, arms, data, rows, config, settings, device, output, state, mappings, history_predictions):
    for name, arm in arms.items():
        arm.load_state_dict(state['best'][name]['state'], strict=True)
        arm.requires_grad_(False).eval()
    reference = training.JointTransitionFusion(source.fusion, 'joint_base', 42).to(device).requires_grad_(False).eval()
    records = []
    for step, indices, batch, expected in original.diagnosis.aligned_batches(data, rows, 'val', config):
        features, _ = frozen.encode(source, batch, device)
        paired, pairs = mapping(features, [r['sample_id'] for r in expected], 'val', settings['pairing_seed'])
        for actual, saved in zip(pairs, mappings[indices.start:indices.stop]):
            require({k: v for k, v in actual.items() if k != 'features_changed'} ==
                    {k: v for k, v in saved.items() if k != 'features_changed'}, 'Raw replay permutation differs')
        records.extend(observe({'original_C': reference, **arms}, paired, expected, mappings[indices.start:indices.stop]))
        if step == 1 or step % 200 == 0 or indices.stop == len(rows):
            raw.progress(output, 'selected_pair_raw_replay', samples=indices.stop, total=len(rows))
    exported = diagnostics.exports(records)
    receipts = {}
    for name in ARMS:
        metric = frozen.joint.study.validate_predictions(exported[name], rows, probabilities=False)
        digest = frozen.joint.prediction_digest(exported[name])
        require(metric == state['best'][name]['validation'] and digest == state['best'][name]['prediction_ids_sha256'],
                name + ' best raw-graph replay differs')
        receipts[name] = {'passed': True, 'samples': len(rows), 'epoch': state['best'][name]['epoch'],
                          'prediction_ids_sha256': digest, 'fixed_mapping_replayed': True}
    require([(r['sample_id'], r['label']) for r in history_predictions] ==
            [(r['sample_id'], r['label']) for r in records], 'Historical J3 identities differ')
    for row, old in zip(records, history_predictions):
        row['predictions']['historical_J3'] = old['prediction']
    raw.write(output / 'selected_validation_replay.json', receipts)
    return records


def run(repo, graph_root, output, settings, resume=False, probe_only=False):
    started = time.monotonic()
    validate_settings(settings)
    history, history_predictions, inputs = history_inputs(repo, settings)
    source, data, rows, device, config, selected, gold, source_inputs, binding = frozen.load_source(repo, 42, graph_root, output)
    inputs.update(source_inputs)
    require(binding == history['provenance']['binding']['source'], 'Wrong original C for True/Shuffle')
    require(len(rows['train']) == settings['training_samples'] and len(rows['val']) == settings['validation_samples']
            and data.metadata['partition_sizes']['test'] == settings['test_samples'], 'Fixed split sizes differ')
    config.update(batch_size=settings['batch_size'], eval_batch_size=settings['eval_batch_size'])
    digest = raw.state_digest(source.state_dict())
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=raw.ROOT, text=True).strip()
    environment = {'python': platform.python_version(), 'torch': str(torch.__version__),
                   'pyg': torch_geometric.__version__, 'cuda': torch.version.cuda,
                   'device': torch.cuda.get_device_name(device), 'platform': platform.platform()}
    run_binding = {'source': binding, 'code_commit': commit, 'settings_sha256': raw.sha(CONFIG),
                   'source_files_sha256': {name: raw.sha(raw.ROOT / name) for name in SOURCE_FILES},
                   'environment': environment, 'graph_root': str(data.root), 'workers': config['workers'],
                   'input_sha256': {str(p.relative_to(repo)): sha for p, sha in inputs.items()}}
    if (output / 'provenance.json').exists():
        require(resume and raw.read(output / 'provenance.json')['binding'] == run_binding,
                'Existing output source/config/code/environment differs')
    raw.write(output / 'config.json', settings)
    raw.write(output / 'provenance.json', {'binding': run_binding, 'environment': environment,
              'test_graph_loads': 0, 'historical_J3_is_external_reference_only': True})
    raw_probe(source, data, config, settings, device, output)
    training.verify_source(source, digest)
    base_val, reference, coverage_val = parent.cached_split(source, data, rows['val'], 'val', config, device, output, run_binding)
    parent.verify_source_replay(reference, gold, selected, rows['val'], output)
    val = prepare_pair_cache(base_val, rows['val'], 'val', settings, output)
    arms = make_arms(source, settings, device)
    parent.initial_replay(source, arms, val, rows['val'], gold, selected, device, settings, output)
    raw.write(output / 'server_probe.json', {'passed': True, 'formal_training_started': False,
              'real_graph_gradients': True, 'full_epoch_zero_replay': True,
              'mapping_rng_isolated': True, 'val_mapping_sha256': val.mapping_sha256})
    training.verify_source(source, digest)
    if probe_only:
        return {'status': 'server_probe_passed', 'formal_training_started': False, 'test_evaluated': False}
    base_train, _, coverage_train = parent.cached_split(source, data, rows['train'], 'train', config, device, output, run_binding)
    train = prepare_pair_cache(base_train, rows['train'], 'train', settings, output)
    raw.write(output / 'coverage.json', {'train': train.coverage, 'val': val.coverage,
              'original_structure_coverage': {'train': coverage_train, 'val': coverage_val}})
    run_binding = {**run_binding, 'mapping_sha256': {'train': train.mapping_sha256, 'val': val.mapping_sha256},
                   'frozen_cache_sha256': {p: raw.sha(output / ('frozen_' + p + '_features.pt')) for p in ('train', 'val')}}
    raw.seed_all(42)
    state = training.fit(arms, train, val, rows['val'], reference, settings, run_binding, device, output, resume=resume)
    mappings = val.records
    del train, val, base_train, base_val
    records = selected_raw_replay(source, arms, data, rows['val'], config, settings, device, output, state, mappings, history_predictions)
    parent.verify_source_replay(diagnostics.exports(records)['original_C'], gold, selected, rows['val'], output)
    from transition_pair_results import write_results
    write_results(output, records, rows['val'], state, settings, history)
    training.verify_source(source, digest)
    frozen.evaluation.verify_inputs(inputs)
    raw.write(output / 'final_verification.json', {'passed': True, 'source_C_unchanged': True,
              'source_inputs_unchanged': True, 'same_sample_order_both_arms': True,
              'updates_per_arm': state['updates'], 'epochs_per_arm': state['completed_epoch'],
              'fixed_mapping_raw_replay_passed': True, 'test_graph_loads': 0, 'seconds': time.monotonic() - started})
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
    with FileLock(str(output / '.run.lock'), timeout=0):
        try:
            raw.write(output / 'status.json', {'status': 'preparing', 'test_evaluated': False,
                      'formal_training_started': (output / 'recovery.pt').exists()})
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
