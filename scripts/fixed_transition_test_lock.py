"""Immutable validation selections for zero-training fixed-test evaluation."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess

import torch

import fixed_transition_archive as archive
import fixed_transition_results as results
import run_fixed_transition_stability as stability
from xgnid.fixed_transition_stability import ARMS, MODELS, StabilityFusion

raw, frozen, original, require = stability.raw, stability.frozen, stability.original, stability.require
SOURCE = 'fixed_transition_stability_v1/'
SNAPSHOT = '5f3975aeecd41c8c1150b97f88813b7ea703e54d'
LABELS = tuple(results.LABELS)
EPOCHS = {42: (48, 94, 59, 191), 43: (188, 50, 25, 97), 44: (167, 150, 131, 123)}
COMPUTE_FILES = (*stability.parent.SOURCE_FILES, 'src/xgnid/fixed_transition_stability.py',
                 'scripts/grouped_graphs.py', 'scripts/audit_graph_split.py',
                 'scripts/build_grouped_split.py', 'scripts/e3_gate_study.py')
NEW_FILES = ('scripts/fixed_transition_test_lock.py', 'scripts/fixed_transition_test_results.py',
             'scripts/run_fixed_transition_final_test.py', 'scripts/launch_fixed_transition_final_test.py',
             'configs/fixed_transition_final_test_v1.json')


def validate_settings(settings):
    expected = {'schema': 'xgnid.fixed-transition-final-test.v1', 'backbone_seeds': [42, 43, 44],
        'models': list(MODELS), 'readout_seed': 42, 'sampling_seed': 42, 'new_training_runs': 0,
        'new_training_epochs': 0, 'new_optimizer_updates': 0, 'eval_batch_size': 16, 'workers': 2,
        'validation_samples': 13446, 'test_samples': 13448, 'labels': list(LABELS),
        'selection': 'locked_best_validation_macro_f1_including_zero_earliest_tie',
        'test_reselection_permitted': False, 'automatic_training_or_tuning': False,
        'fixed_injection_strength': 1.0, 'pairing': 'true_continuous', 'historical_test_split_used': True,
        'split_directory': 'data/processed/ciciot2023_grouped_input_v1',
        'split_json_sha256': '2f6a7c307121454b881520379f0a298b7e2a179b1db4a7cb03741b7f648a774a',
        'source_training_code_commit': '7c9249eafcea21ff40fe1811afcc01414e00acfb'}
    require(all(settings.get(k) == v for k, v in expected.items()), 'Fixed-test registered protocol differs')
    spec = settings['archive']
    require(spec['source_result_commit'] == SNAPSHOT and spec['source_directory'] == 'outputs',
            'Fixed-test immutable source differs')
    require(set(settings['seeds']) == {'42', '43', '44'}, 'Exactly three locked backbones required')
    for seed, epochs in EPOCHS.items():
        selected = settings['seeds'][str(seed)]
        folder = 'seed42_import' if seed == 42 else 'seed' + str(seed)
        require(selected['folder'] == folder and list(selected['checkpoints']) == list(MODELS),
                'Locked model list/folder differs')
        for name, epoch in zip(MODELS, epochs):
            model = selected['checkpoints'][name]
            path = SOURCE + folder + '/' + name + '/best.pt'
            require(model['path'] == path and model['epoch'] == epoch
                    and spec['inputs'][path] == {k: model[k] for k in ('sha256', 'bytes')},
                    'A locked checkpoint was replaced: ' + str(seed) + '/' + name)


def atomic_bytes(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(value)
    os.replace(temporary, path)


def stripped(rows, key=None):
    values = [{'sample_id': r['sample_id'], 'label': r['label'],
               'prediction': r['prediction'] if key is None else r['predictions'][key]} for r in rows]
    require(len({r['sample_id'] for r in values}) == len(values), 'Duplicate prediction identities')
    require(all(r['label'] in range(8) and r['prediction'] in range(8) for r in values),
            'Prediction labels out of range')
    return values


def historical_predictions(inputs, settings, seed):
    spec = settings['historical_base_C_test'][str(seed)]
    return stripped(inputs.rows(spec['predictions']), spec['prediction_key'])


def verify_historical_identity(inputs, settings, seed):
    selected = settings['seeds'][str(seed)]
    spec = settings['historical_base_C_test'][str(seed)]
    proof = inputs.json(spec['proof'])
    weight = selected['checkpoints']['base_C']['sha256']
    expected = selected['source_binding']['frozen_c_sha256']
    if seed == 42:
        require(proof['model_state_sha256_before']['A'] == proof['model_state_sha256_after']['A'] == expected,
                'Historical seed42 C state differs')
        candidates = [v for p, v in proof['input_sha256'].items()
                      if p.endswith('/joint_cross_layer_query_s42_v1/C/best.pt')]
    else:
        require(proof['state_sha256'][str(seed)]['C'] == expected, 'Historical independent C state differs')
        candidates = [proof['input_sha256']['outputs/packet_transition_independent_s' + str(seed) + '_v1/C/best.pt']]
    require(candidates == [weight], 'Historical C checkpoint bytes differ')
    return {'passed': True, 'proof': spec['proof'], 'checkpoint_sha256': weight,
            'state_sha256': expected, 'historical_test_split_used': True}


def restore_models(inputs, settings, seed, device):
    """Restore exact archived tensors; constructors add no learned state or optimizer."""
    spec = settings['seeds'][str(seed)]
    binding = spec['source_binding']
    # The full C snapshots are trusted, hash-verified legacy archives with NumPy RNG state.
    saved = torch.load(io.BytesIO(inputs.data(spec['checkpoints']['base_C']['path'])),
                       map_location='cpu', weights_only=False)
    config = saved['config']
    require(config['arm'] == 'C' and config['query_source'] == 'flow' and tuple(config['labels']) == LABELS,
            'Original C architecture/label mapping differs')
    require(saved['epoch'] == spec['checkpoints']['base_C']['epoch'], 'Original C selected epoch differs')
    if seed == 42:
        require(config['protocol']['seed'] == 42, 'Seed42 original C protocol differs')
    else:
        source = inputs.json(SOURCE + 'protocol_lock.json')['protocol']['independent_sources'][str(seed)]
        require(config['binding']['seed'] == seed
                and config['binding']['n0_best_sha256'] == source['n0']['sha256']['best.pt'],
                'Independent N0/C lineage differs')
    with torch.random.fork_rng(devices=[]):
        base = raw.OperatorClassifier('GAT', seed=seed)
        source = frozen.joint.JointCrossLayerQueryClassifier(base.flow_encoder, base.packet_encoder,
                                                            base.fusion, 'flow', seed)
        source.load_state_dict(saved['model_state'], strict=True)
        frozen.independent.freeze_c(source)
        require(raw.state_digest(source.state_dict()) == binding['frozen_c_sha256'], 'Original C state differs')
        models = {'base_C': stability.training.parent.JointTransitionFusion(source.fusion, 'joint_base', 42)}
        for name in ARMS:
            checkpoint = inputs.tensor(spec['checkpoints'][name]['path'])
            lineage = checkpoint.get('source_binding', checkpoint.get('binding', {}).get('source'))
            require(lineage == binding and checkpoint['epoch'] == spec['checkpoints'][name]['epoch'],
                    'Readout checkpoint source/epoch differs: ' + name)
            arm = StabilityFusion(source.fusion, name, 42)
            arm.load_state_dict(checkpoint['state'], strict=True)
            if name == 'frozen_single':
                require(raw.state_digest(arm.fusion.state_dict()) == raw.state_digest(source.fusion.state_dict()),
                        'Frozen Single fusion is not the original C fusion')
            models[name] = arm
    for model in (source, *models.values()):
        model.zero_grad(set_to_none=True)
        model.requires_grad_(False).to(device).eval()
    return source, models, config


def frozen_digests(source, models):
    return {name: raw.state_digest(model.state_dict()) for name, model in {'source_C': source, **models}.items()}


def verify_frozen(source, models, before):
    require(frozen_digests(source, models) == before, 'Fixed evaluation changed model parameters/buffers')
    require(all(not m.training for model in (source, *models.values()) for m in model.modules()),
            'Fixed evaluation left an active training module')
    require(all(not p.requires_grad and p.grad is None for model in (source, *models.values())
                for p in model.parameters()), 'Fixed evaluation created a gradient')


def checkpoint_manifest(inputs, settings):
    """No graph access. Verify all validation selections before authorizing test access."""
    validate_settings(settings)
    protocol = inputs.json(SOURCE + 'protocol_lock.json')
    require(protocol['code_commit'] == settings['source_training_code_commit'], 'Training code source differs')
    require(all(settings[k] == protocol['protocol'][k] for k in ('split_id', 'manifest_sha256')),
            'Fixed-test split/manifest differs from training')
    require(inputs.json(SOURCE + 'status.json')['queue_completed']
            and inputs.json(SOURCE + 'final_verification.json')['passed'], 'Stability archive is incomplete')
    # Inference definitions must be byte-equivalent to the registered training definitions.
    for name in COMPUTE_FILES:
        if name in protocol['source_files_canonical_sha256']:
            require(hashlib.sha256((raw.ROOT / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                    == protocol['source_files_canonical_sha256'][name], 'Inference definition changed: ' + name)
    manifest, cohort, historical_cohort = {}, None, None
    for seed in (42, 43, 44):
        selected = settings['seeds'][str(seed)]
        prefix = SOURCE + selected['folder'] + '/'
        summary = inputs.json(prefix + 'summary.json')
        require(summary['source_binding'] == selected['source_binding'] and not summary['test_evaluated'],
                'Validation source binding/scope differs')
        require(inputs.json(prefix + 'final_verification.json')['passed'], 'Missing validation verification')
        entries = {}
        sealed = inputs.json(prefix + 'artifact_manifest.json')['files_sha256']
        for name in MODELS:
            item = selected['checkpoints'][name]
            require(sealed[name + '/best.pt'] == item['sha256'], 'Checkpoint seal differs')
            predictions_path = prefix + name + '/best_validation_predictions.jsonl.gz'
            predictions = stripped(inputs.rows(predictions_path))
            identities = [(r['sample_id'], r['label']) for r in predictions]
            require(len(predictions) == settings['validation_samples'], 'Validation count differs')
            if cohort is None:
                cohort = identities
            require(cohort == identities, 'Validation cohorts differ')
            metric = frozen.joint.study.validate_predictions(predictions, probabilities=False)
            meta = summary['model_metadata'][name]
            require(metric == summary['metrics'][name] == meta['best_validation']
                    and meta['best_epoch'] == item['epoch'] and metric['macro_f1'] == item['validation_macro_f1'],
                    'Locked validation selection differs')
            if name != 'base_C':
                history = [json.loads(line) for line in inputs.data(prefix + name + '/epochs.jsonl').splitlines()]
                require([r['epoch'] for r in history] == list(range(201))
                        and max(history, key=lambda r: r['validation']['macro_f1'])['epoch'] == item['epoch'],
                        'Locked selection violates complete budget/earliest-tie rule')
                ck = inputs.tensor(item['path'])
                require(ck['validation'] == metric, 'Checkpoint validation metric differs')
            entries[name] = {**item, 'validation': metric,
                'validation_prediction_digest': frozen.joint.prediction_digest(predictions),
                'validation_predictions_path': predictions_path,
                'source_result_commit': SNAPSHOT, 'labels': list(LABELS),
                'selection': settings['selection'], 'new_optimizer_updates': 0}
        history_identity = verify_historical_identity(inputs, settings, seed)
        historical = historical_predictions(inputs, settings, seed)
        require(len(historical) == settings['test_samples'], 'Historical C test count differs')
        identities = [(r['sample_id'], r['label']) for r in historical]
        if historical_cohort is None:
            historical_cohort = identities
        require(historical_cohort == identities, 'Historical test cohorts differ')
        manifest[str(seed)] = {'models': entries, 'source_binding': summary['source_binding'],
            'role': selected['role'], 'historical_C_identity': history_identity,
            'historical_C_test_prediction_digest': frozen.joint.prediction_digest(historical)}
    require(len({v['source_binding']['frozen_c_sha256'] for v in manifest.values()}) == 3,
            'Three independent original C states required')
    return {'schema': 'xgnid.fixed-transition-checkpoint-lock.v1', 'source_result_commit': SNAPSHOT,
        'seeds': manifest, 'labels': list(LABELS), 'label_to_id': {label: i for i, label in enumerate(LABELS)},
        'source_training_protocol': protocol['protocol'], 'new_training_runs': 0, 'new_optimizer_updates': 0,
        'test_reselection_permitted': False, 'historical_test_split_used': True,
        'validation_cohort_sha256': archive.canonical_sha(cohort),
        'test_cohort_sha256': archive.canonical_sha(historical_cohort)}


def lock_inputs(repo, output, settings, download=False):
    validate_settings(settings)
    inputs = archive.open_archive(repo, settings['archive'], download)
    candidates = checkpoint_manifest(inputs, settings)
    code = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=raw.ROOT, text=True).strip()
    protocol = {'schema': 'xgnid.fixed-transition-test-protocol-lock.v1', 'code_commit': code,
        'settings': settings, 'checkpoint_manifest_sha256': archive.canonical_sha(candidates),
        'source_files_canonical_sha256': {name: hashlib.sha256((raw.ROOT / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            for name in dict.fromkeys((*COMPUTE_FILES, *NEW_FILES))}}
    output.mkdir(parents=True, exist_ok=True)
    for filename, value in (('protocol_lock.json', protocol), ('checkpoint_lock.json', candidates)):
        path = output / filename
        if path.exists():
            require(raw.read(path) == value, 'Fixed-test lock changed; preserve existing output: ' + filename)
        else:
            raw.write(path, value)
    # Preserve the exact twelve selected weight files in this result's own publication.
    for seed, spec in settings['seeds'].items():
        for name, item in spec['checkpoints'].items():
            path = output / 'locked_checkpoints' / ('seed' + seed) / (name + '.pt')
            if path.exists():
                require(raw.sha(path) == item['sha256'], 'Preserved checkpoint changed')
            else:
                atomic_bytes(path, inputs.data(item['path']))
    raw.write(output / 'checkpoint_lock_receipt.json', {'passed': True, 'models': 12,
        'validation_predictions_verified_per_model': settings['validation_samples'],
        'new_optimizer_updates': 0, 'test_graph_loads': 0,
        'checkpoint_manifest_sha256': archive.canonical_sha(candidates)})
    return inputs, protocol, candidates
