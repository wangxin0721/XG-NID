"""Inventory independent backbones and train D32-Gate from a locked N0 seed."""
import argparse
import copy
import json
from pathlib import Path
import platform
import re
from types import SimpleNamespace

import torch
import torch_geometric

import evaluate_early_baselines_final_test as baseline
from e3_gate_study import guard_graph_loads, identity
import run_fusion_dimension_pair as dim
import run_fusion_gate_pair as gate
from run_fusion_seed_repeats import SeedCore
from xgnid.cross_layer import CrossLayerClassifier
from xgnid.fusion_dimension import FrozenEncoderClassifier, FusionHead
from xgnid.joint_cross_layer import JointCrossLayerClassifier
from xgnid.joint_cross_layer_query import JointCrossLayerQueryClassifier


raw, joint, evaluation, require = baseline.raw, baseline.joint, baseline.evaluation, baseline.require
core = baseline.grouped
CONFIG = raw.ROOT / 'configs/fusion_gate_pair_s42_v1.json'
STAGES = ('D32-Gate', 'CL-Last', 'JCL-Multi', 'C')
N0_EPOCHS = {43: 27, 44: 19}
PACKET_LAYER_TWO = ('packet_encoder.convs2.convs.<flow___contain___packet>.',
                    'packet_encoder.convs2.convs.<packet___link___packet>.')
STAGE_KEYS = {
    'D32-Gate': ('width', 'epochs', 'patience', 'lr', 'weight_decay', 'batch_size', 'eval_batch_size',
                 'include_epoch_zero', 'encoder_mode', 'optimizer', 'loss'),
    'CL-Last': ('epochs', 'head_lr', 'cross_layer_lr', 'weight_decay', 'batch_size', 'eval_batch_size',
                'include_epoch_zero', 'early_stop_enabled', 'encoder_mode', 'optimizer', 'loss'),
    'JCL-Multi': ('epochs', 'encoder_lr', 'projection_lr', 'head_lr', 'weight_decay', 'batch_size',
                  'eval_batch_size', 'include_epoch_zero', 'early_stop_enabled'),
    'C': ('epochs', 'encoder_lr', 'projection_lr', 'head_lr', 'pool_lr', 'weight_decay', 'batch_size',
          'eval_batch_size'),
}


class D32Core(SeedCore):
    """Bind the historical trainer's literal seed to this independent run."""
    def seed_all(self, seed):
        require(seed == 42, 'Unexpected seed usage in the historical D32 trainer')
        return self.core.seed_all(self.seed)


def streams_for_seed(seed):
    return {name: int(identity({'seed': seed, 'component': name})[:15], 16) for name in dim.COMPONENTS}


def stage_protocols(d32):
    return {'D32-Gate': d32,
            'CL-Last': raw.read(raw.ROOT / 'configs/cross_layer_pair_s42_v1.json'),
            'JCL-Multi': raw.read(raw.ROOT / 'configs/joint_cross_layer_packet2_s42_v1.json'),
            'C': raw.read(joint.CONFIG)}


def protocol_matches(stage, protocol, reference):
    return all(protocol.get(key) == reference[key] for key in STAGE_KEYS[stage])


def n0_lineage(binding):
    return binding.get('n0_best_sha256', binding.get('encoder_source', {}).get('N0_best_sha256'))


def preserves_n0_encoders(stage, model, n0):
    original = dict(dim.encoder_values(n0))
    retained = {name: value for name, value in dim.encoder_values(model)
                if stage in ('D32-Gate', 'CL-Last') or not name.startswith(PACKET_LAYER_TWO)}
    return bool(retained) and all(torch.equal(value, original[name]) for name, value in retained.items())


def inventory(repo, n0_settings, protocols, excluded=None):
    candidates = {}
    for path in sorted((repo / 'outputs').rglob('best.pt')):
        directory = path.parent
        stage = directory.name
        if stage not in STAGES or (excluded is not None and excluded in directory.parents):
            continue
        # Some training stages store their run configuration at the parent.
        cfg_path = directory / 'config.json'
        if not cfg_path.is_file():
            cfg_path = directory.parent / 'config.json'
        if not cfg_path.is_file():
            candidates.setdefault('unknown', []).append({'stage': stage, 'path': str(path.relative_to(repo)),
                                                          'status': 'missing_config'})
            continue
        try:
            metadata = raw.read(cfg_path)
            protocol = metadata.get('protocol', metadata)
            seed = int(protocol.get('seed', -1))
        except (OSError, ValueError, TypeError):
            candidates.setdefault('unknown', []).append({'stage': stage, 'path': str(path.relative_to(repo)),
                                                          'status': 'invalid_config'})
            continue
        if seed not in N0_EPOCHS:
            continue
        binding = metadata.get('binding', {})
        n0_sha = n0_settings[f'N0{seed}']['sha256']['best.pt']
        bound_n0 = n0_lineage(binding)
        registered = protocol_matches(stage, protocol, protocols[stage])
        required_files = (directory / 'metrics.json', directory / 'validation_predictions.jsonl.gz')
        status = ('protocol_differs' if not registered else 'n0_lineage_unverified' if bound_n0 != n0_sha
                  else 'missing_replay_inputs' if not all(file.is_file() for file in required_files)
                  else 'ready_for_replay')
        candidates.setdefault(str(seed), []).append({'stage': stage, 'path': str(path.relative_to(repo)),
                                                      'status': status})
    return {'seeds': {str(seed): {
                'n0': {'directory': n0_settings[f'N0{seed}']['directory'], 'epoch': epoch,
                       'present': (repo / n0_settings[f'N0{seed}']['directory'] / 'best.pt').is_file()},
                'candidates': candidates.get(str(seed), [])} for seed, epoch in N0_EPOCHS.items()},
            'unknown_seed_candidates': candidates.get('unknown', [])}


def model_from_stage(stage, n0, device, seed):
    fusion = FusionHead(32, streams_for_seed(seed))
    encoders = (copy.deepcopy(n0.flow_encoder), copy.deepcopy(n0.packet_encoder), fusion)
    if stage == 'C':
        model = JointCrossLayerQueryClassifier(*encoders, 'flow', seed)
    elif stage == 'JCL-Multi':
        model = JointCrossLayerClassifier(*encoders, 'multi')
    elif stage == 'CL-Last':
        model = CrossLayerClassifier(*encoders, 'last')
    else:
        model = FrozenEncoderClassifier(*encoders)
    return model.to(device).eval()


def validate_candidate(repo, candidate, n0, data, rows, config, device, output, seed, n0_sha, protocols):
    path = repo / candidate['path']
    stage = candidate['stage']
    folder = path.parent
    metrics = raw.read(folder / 'metrics.json')
    # The query stage uses the selected.best protocol. Earlier stages use best_epoch.
    if stage == 'C':
        best = metrics['selected']['best']
        digest, epoch, expected = best['checkpoint_sha256'], best['epoch'], best['validation']
    else:
        digest, epoch, expected = metrics['checkpoint_sha256']['best.pt'], metrics['best_epoch'], metrics['best_validation']
    gold_path = folder / 'validation_predictions.jsonl.gz'
    expected_prediction_file = metrics['predictions_sha256']
    evaluation.verify_inputs({path: digest, gold_path: expected_prediction_file})
    saved = torch.load(path, map_location='cpu', weights_only=False)
    metadata = saved['config']
    protocol = metadata['protocol']
    binding = metadata['binding']
    require(saved['epoch'] == epoch and protocol['seed'] == seed
            and protocol_matches(stage, protocol, protocols[stage]) and n0_lineage(binding) == n0_sha
            and protocol.get('split_id', binding.get('split_id')) == config['split_id']
            and binding.get('manifest_sha256') == data.metadata['sample_manifest_sha256']
            and (metadata.get('variant', metadata.get('arm'))) == stage, 'Intermediate metadata differs')
    require(stage != 'C' or metadata.get('query_source') == 'flow', 'C must use Flow node queries')
    require(stage != 'D32-Gate' or metadata.get('architecture') == gate.architecture('dynamic'),
            'D32 candidate does not use the registered dynamic gate')
    gold = joint.legacy.predictions(gold_path)
    measured = joint.study.validate_predictions(gold, rows, probabilities=False)
    require(baseline.matches_historical(measured, expected) and baseline.matches_historical(measured, saved['validation']),
            'Intermediate archive metrics differ')
    model = model_from_stage(stage, n0, device, seed)
    model.load_state_dict(saved['model_state'], strict=True)
    require(preserves_n0_encoders(stage, model, n0),
            'Intermediate checkpoint did not retain the expected N0 encoders')
    loader = joint.legacy.loader(data, rows, 'val', config)
    replayed, records = evaluation.evaluate(model, loader, rows, device, output, 'candidate_validation')
    receipt = evaluation.replay_receipt(replayed, records,
                {'validation': measured, 'prediction_ids_sha256': joint.prediction_digest(gold)}, gold)
    require(receipt['passed'], 'Intermediate candidate validation replay differs')
    return {**candidate, 'status': 'verified', 'epoch': epoch, 'checkpoint_sha256': digest,
            'validation': measured, 'replay': receipt}


def run(repo, output, graph_root, seed, code_commit, mode):
    settings = raw.read(baseline.CONFIG)
    n0_settings = settings['models']
    old = raw.read(CONFIG)
    gate.check_config(old)
    d32 = {**old, 'seed': seed, 'training_jobs': 1,
           'initialization': 'fresh_seed_specific_d32_on_frozen_n0'}
    split_config = raw.read(evaluation.CONFIG)
    protocols = stage_protocols(d32)
    scan = inventory(repo, n0_settings, protocols, output)
    raw.write(output / 'inventory.json', scan)
    if mode == 'inventory':
        return 'inventory_completed'
    device = baseline.runtime(core)
    core.torch.set_num_threads(2)
    split = repo / split_config['split_directory']
    evaluation.verify_inputs({split / 'split.json': split_config['split_json_sha256']})
    data = core.FrozenGraphs(split, split_config['split_id'], data_root=graph_root)
    guard_graph_loads(data, {'train', 'val'})
    manifest_sha = settings['sample_manifest_sha256']
    require(data.metadata['sample_manifest_sha256'] == manifest_sha, 'Manifest differs from locked N0')
    rows = evaluation.evaluation_rows(data, 'val', split_config)
    n0_name = f'N0{seed}'
    spec = n0_settings[n0_name]
    require(spec['epoch'] == N0_EPOCHS[seed], 'Registered N0 epoch differs')
    saved, selected, gold, inputs = baseline.load_archive(repo, n0_name, spec, rows, split_config, manifest_sha)
    inputs.update({split / 'split.json': split_config['split_json_sha256'],
                   split / data.metadata['sample_manifest']: manifest_sha})
    source = saved['config']['binding']
    model_sources = {raw.ROOT / path: sha for path, sha in source['model_sources'].items()}
    evaluation.verify_inputs(model_sources)
    core.seed_all(seed)
    warm = core.Batch.from_data_list([data.load(row) for row in rows[:split_config['eval_batch_size']]]).to(device)
    n0 = baseline.load_model(n0_name, saved, warm, device)
    n0_state = raw.state_digest(n0.state_dict())
    batches = core.DataLoader(core.GraphRows(data, rows), batch_size=split_config['eval_batch_size'],
                              shuffle=False, num_workers=split_config['workers'])
    observed = baseline.evaluate_models({n0_name: n0}, batches, rows, output, 'n0_validation', device)[n0_name]
    n0_metrics = joint.study.validate_predictions(observed, rows, probabilities=False)
    replay = evaluation.replay_receipt(n0_metrics, observed, selected, gold)
    raw.write(output / 'n0_validation_replay.json', {**replay, 'seed': seed, 'epoch': selected['epoch'],
              'validation': n0_metrics, 'checkpoint_sha256': spec['sha256']['best.pt']})
    require(replay['passed'] and raw.state_digest(n0.state_dict()) == n0_state, 'N0 original-graph replay differs')
    raw.save_predictions(output / 'n0_validation_predictions.jsonl.gz', observed)
    raw.progress(output, 'n0_validation_passed', seed=seed, epoch=selected['epoch'], macro_f1=n0_metrics['macro_f1'])
    if mode == 'replay':
        return 'n0_replay_completed'

    binding = {'seed': seed, 'n0_best_sha256': spec['sha256']['best.pt'],
               'split_id': split_config['split_id'], 'manifest_sha256': manifest_sha,
               'n0_validation_prediction_ids_sha256': replay['prediction_ids_sha256'],
               'code_commit': code_commit, 'graph_root': str(data.root), 'python': platform.python_version(),
               'torch': str(torch.__version__), 'cuda': torch.version.cuda, 'pyg': torch_geometric.__version__,
               'device': torch.cuda.get_device_name(device),
               'inputs': {str(path.relative_to(repo)): digest for path, digest in inputs.items()}}
    raw.write(output / 'binding.json', binding)
    d32.update(split_id=split_config['split_id'])
    raw.write(output / 'config.json', d32)
    candidates = scan['seeds'][str(seed)]['candidates']
    available = sorted((item for item in candidates if item['status'] == 'ready_for_replay'),
                       key=lambda item: STAGES.index(item['stage']), reverse=True)
    for candidate in available:
        try:
            verified = validate_candidate(repo, candidate, n0, data, rows, split_config, device, output, seed,
                                          binding['n0_best_sha256'], protocols)
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            candidate.update(status='replay_rejected', reason=str(error))
            raw.write(output / 'inventory.json', scan)
        else:
            raw.write(output / 'reused_checkpoint.json', verified)
            evaluation.verify_inputs({**inputs, **model_sources})
            return 'reused_verified_checkpoint'

    graph = model_from_stage('D32-Gate', n0, device, seed)
    graph.fusion.fusion_mode = 'dynamic'
    encoder_hash = core.state_hash(dim.encoder_values(graph))
    require(preserves_n0_encoders('D32-Gate', graph, n0), 'D32 did not inherit N0 encoders')
    dim.assert_frozen(core, graph, encoder_hash)
    initial = raw.state_digest(graph.fusion.state_dict())
    binding.update(encoder_sha256=encoder_hash, initial_fusion_sha256=initial)
    args = SimpleNamespace(output=output, probes={})
    adapted_core = D32Core(core, seed, output)
    encoder_source = {'encoder_sha256': encoder_hash, 'N0_best_sha256': binding['n0_best_sha256'],
                      'split_id': binding['split_id'], 'manifest_sha256': manifest_sha}
    caches, receipts = {}, {}
    for part in ('val', 'train'):
        cache, receipt = dim.extract_cache(
            core, args, n0, data, device, part, encoder_source,
            gold if part == 'val' else None, selected['validation'] if part == 'val' else None)
        caches[part] = {key: value.to(device) if torch.is_tensor(value) else value for key, value in cache.items()}
        receipts[part] = receipt
        # These features are held in memory, and the original graph loader is the cache source.
        (output / f'{part}_encoder64.pt').unlink()
        receipts[part]['storage'] = 'in_memory'
    raw.write(output / 'cache_receipts.json', receipts)
    args.probes['D32-Gate'] = gate.probe(adapted_core, graph, data, caches['train'], device, d32, encoder_hash)
    require(raw.state_digest(graph.fusion.state_dict()) == initial, 'D32 probe changed initial fusion')
    raw.progress(output, 'd32_training', seed=seed)
    result, exports = gate.adapted(dim.train, 'dynamic')(
        adapted_core, args, d32, graph, caches, 'D32-Gate', binding, encoder_hash)
    folder = output / 'D32-Gate'
    checkpoint = folder / 'best.pt'
    require(raw.sha(checkpoint) == result['checkpoint_sha256']['best.pt'], 'D32 best checkpoint hash differs')
    best = torch.load(checkpoint, map_location='cpu', weights_only=False)
    audit, records = gate.adapted(dim.graph_replay, 'dynamic')(
        adapted_core, args, (FusionHead, FrozenEncoderClassifier), data, device, saved, best, caches['val'],
        streams_for_seed(seed), encoder_hash)
    require(joint.legacy.identities(exports) == joint.legacy.identities(records)
            and raw.state_digest(n0.state_dict()) == n0_state, 'D32 raw replay or frozen N0 differs')
    audit.update(prediction_ids_sha256=joint.prediction_digest(records), checkpoint_sha256=raw.sha(checkpoint))
    raw.write(folder / 'graph_replay.json', audit)
    evaluation.verify_inputs({**inputs, **model_sources})
    raw.write(output / 'binding.json', binding)
    raw.write(output / 'summary.json', {'seed': seed, 'n0': {'epoch': selected['epoch'], 'validation': n0_metrics},
              'd32_gate': result, 'graph_replay': audit, 'test_evaluated': False})
    return 'd32_gate_completed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--seed', type=int, choices=tuple(N0_EPOCHS), default=43)
    parser.add_argument('--mode', choices=('inventory', 'replay', 'train'), default='train')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--code-commit', required=True)
    args = parser.parse_args()
    require(re.fullmatch(r'[0-9a-f]{40}', args.code_commit), 'Expected full code SHA')
    repo = args.repo.resolve()
    default = f'packet_transition_independent_d32_s{args.seed}_v1'
    if args.mode != 'train':
        default += '_' + args.mode
    output = (repo / (args.output or f'outputs/{default}')).resolve()
    require(output.parent == (repo / 'outputs').resolve(), 'Use a new directory directly under outputs')
    output.mkdir(parents=True)
    raw.write(output / 'status.json', {'status': 'running', 'seed': args.seed, 'mode': args.mode})
    try:
        outcome = run(repo, output, args.data_root, args.seed, args.code_commit, args.mode)
    except BaseException as error:
        raw.write(output / 'status.json', {'status': 'failed', 'seed': args.seed, 'mode': args.mode, 'error': str(error)})
        raise
    status = {'status': 'completed', 'seed': args.seed, 'mode': args.mode, 'outcome': outcome}
    raw.write(output / 'status.json', status)
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
