"""Continue each independent N0-derived backbone, then evaluate the locked pairs."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import re
from statistics import mean
from types import SimpleNamespace
from uuid import uuid4

import torch
import torch_geometric
from torch_geometric.loader import DataLoader

import evaluate_packet_transition_final_test as fixed
import run_cross_layer_pair as cross
import run_independent_d32_gate as d32
import run_packet_transition_attention_diagnosis as attention
import run_packet_transition_readout as transition
from filelock import FileLock
from xgnid.joint_cross_layer import reparameterize_projection


raw, joint, evaluation, require = d32.raw, d32.joint, d32.evaluation, d32.require
core, legacy = d32.core, joint.legacy
CONFIG = raw.ROOT / 'configs/independent_packet_transition_v1.json'
OUTPUT = 'packet_transition_independent_s{seed}_v1'
TEST_OUTPUT = 'packet_transition_independent_final_test_v1'
STAGES = ('D32-Gate', 'CL-Last', 'JCL-Multi', 'C', 'B')


def model_state(model):
    return raw.state_digest(model.state_dict())


def protocol(stage, seed, parent, split):
    config = {**d32.stage_protocols(raw.read(d32.CONFIG))[stage], 'seed': seed}
    if stage == 'CL-Last':
        del config['source_config_sha256']
        config.update(source_directory=str(parent['directory']), source_epoch=parent['epoch'],
                      source_macro_f1=parent['validation']['macro_f1'],
                      source_best_sha256=parent['checkpoint_sha256'], split_id=split)
    elif stage == 'JCL-Multi':
        config['references'] = {'CL-Last': {'directory': str(parent['directory']), 'epoch': parent['epoch'],
                                            'sha256': parent['checkpoint_sha256'],
                                            'macro_f1': parent['validation']['macro_f1']}}
    else:
        config.update(source_directory=str(parent['directory']), source_epoch=parent['epoch'],
                      source_macro_f1=parent['validation']['macro_f1'],
                      source_sha256=parent['checkpoint_sha256'],
                      source_predictions_sha256=parent['predictions_sha256'])
    return config


def selection(directory, stage, rows, registration=None, repo=None):
    best_path = directory / 'best.pt'
    gold_path = directory / 'validation_predictions.jsonl.gz'
    report = raw.read(directory / 'metrics.json')
    if stage == 'C':
        best = report['selected']['best']
        epoch, expected, checkpoint_hash = best['epoch'], best['validation'], best['checkpoint_sha256']
    else:
        epoch, expected, checkpoint_hash = report['best_epoch'], report['best_validation'], report['checkpoint_sha256']['best.pt']
    prediction_hash = report['predictions_sha256']
    evaluation.verify_inputs({best_path: checkpoint_hash, gold_path: prediction_hash})
    records = legacy.predictions(gold_path)
    metrics = joint.study.validate_predictions(records, rows, probabilities=False)
    digest = joint.prediction_digest(records)
    require(d32.baseline.matches_historical(metrics, expected), f'{stage} metrics differ from archived predictions')
    record = {'epoch': epoch, 'validation': metrics, 'checkpoint_sha256': checkpoint_hash,
              'predictions_sha256': prediction_hash, 'prediction_ids_sha256': digest,
              'directory': directory.relative_to(repo) if repo else directory}
    if registration is not None:
        require(all(record[key] == value for key, value in registration.items()), 'Registered D32 source differs')
    # Validator calls us with the file hash checked before loading runtime state.
    saved = torch.load(best_path, map_location='cpu', weights_only=False)
    require(saved['epoch'] == epoch and d32.baseline.matches_historical(metrics, saved['validation']),
            f'{stage} checkpoint selection differs from metrics')
    if stage == 'C':
        require(saved['prediction_ids_sha256'] == best['prediction_ids_sha256'] == digest,
                'C checkpoint prediction digest differs')
    return saved, record, records


def build_child(stage, parent_state, n0, device, seed):
    model = d32.model_from_stage(stage, n0, device, seed)
    initialized = model.state_dict()
    added = {'CL-Last': ('cross_layer.',), 'JCL-Multi': (), 'C': ('packet_pool.',)}[stage]
    remaining = {name for name in initialized if not name.startswith(added)}
    require(set(parent_state) == remaining, f'{stage} parent state layout differs')
    model.load_state_dict({**initialized, **parent_state}, strict=True)
    require(raw.state_digest({key: model.state_dict()[key] for key in parent_state})
            == raw.state_digest(parent_state), f'{stage} parent tensors differ')
    if stage == 'JCL-Multi':
        reparameterize_projection(model)
    return model


def raw_replay(model, data, rows, config, device, directory, expected, gold, phase):
    batches = legacy.loader(data, rows, 'val', config)
    metrics, records = evaluation.evaluate(model, batches, rows, device, directory, phase)
    result = evaluation.replay_receipt(metrics, records, expected, gold)
    raw.write(directory / f'{phase}.json', result)
    require(result['passed'], f'{phase} differs from locked predictions')
    return result


def train_cross(model, data, rows, config, device, output, binding, parent, gold):
    args = SimpleNamespace(output=output)
    before = d32.core.state_hash(d32.dim.encoder_values(model))
    caches = {}
    for part in ('val', 'train'):
        retained = sum(v[k].numel() * v[k].element_size() for v in caches.values() for k in cross.TENSORS)
        cache, _ = cross.extract(core, args, model, data, device, part, config, before, binding, retained)
        cross.validate_cache(core, cache, rows[part], part, before, binding)
        caches[part] = cache
        (output / f'{part}_cross_layer.pt').unlink()
    _, cached = cross.evaluate(core, model, caches['val'], device, export=True)
    require(legacy.identities(cached) == legacy.identities(gold), 'CL-Last cached epoch zero differs from D32')
    raw.write(output / 'CL-Last_probe.json', cross.probe(core, model, data, caches['train'], device, config, before))
    result, records = cross.train(d32.D32Core(core, config['seed'], output), args, model, caches, device, config,
                                  binding, before, 'CL-Last')
    receipt, replayed = cross.replay(core, args, model, data, caches['val'], device, config, result['best_validation'],
                                     before, 'CL-Last_selected_raw_replay')
    require(legacy.identities(records) == legacy.identities(replayed)
            and result['epoch_zero']['confusion_matrix'] == parent['validation']['confusion_matrix'],
            'CL-Last raw replay or epoch zero differs from its registered selection')
    raw.write(output / 'CL-Last' / 'graph_replay.json', receipt)


def train_jcl(model, data, rows, config, device, output, binding, parent):
    sampler = raw.TrainBatches(len(rows['train']), config['batch_size'], config['seed'])
    dataset = raw.GraphRows(data, rows['train'], 'train')
    batches = []
    for indices in sampler:
        batches.append(core.Batch.from_data_list([dataset[index] for index in indices]))
        if len(batches) == config['probe_batches']:
            break
    receipt = legacy.probe(model, batches, device, config)
    del batches
    raw.write(output / 'JCL-Multi_probe.json', receipt)
    result, _ = legacy.fit('JCL-Multi', model, data, rows['train'], rows['val'], config, binding, device, output)
    require(result['epoch_zero'] == parent['validation'], 'JCL-Multi formal epoch zero differs from CL-Last')


def train_c(model, data, rows, config, device, output, binding, parent, gold):
    config['source_frozen_state_sha256'] = legacy.frozen_hash(model)
    probe_rows = raw.choose_rows(data, 'train', config['probe_per_class'])
    warm = core.Batch.from_data_list([data.load(row) for row in probe_rows]).to(device)
    receipt = joint.probe(model, warm, config)
    raw.write(output / 'C_probe.json', receipt)
    orders = joint.study.epoch_orders(len(rows['train']), config['batch_size'], config['seed'], config['epochs'])
    joint.fit('C', model, data, rows, gold, parent['validation'], orders, config, binding, device, output)


def train_b(model, data, rows, device, output, binding, selected, gold, settings):
    base = raw.read(evaluation.CONFIG)
    readout_seed = settings['readout_seed']
    config = {**raw.read(transition.CONFIG), 'seed': binding['seed'], 'readout_seed': readout_seed,
              'split_id': binding['split_id'], 'workers': base['workers'],
              'source_checkpoint_sha256': selected['checkpoint_sha256'],
              'baseline_prediction_ids_sha256': selected['prediction_ids_sha256']}
    folder = output / 'B'
    folder.mkdir()
    model.requires_grad_(False).eval()
    frozen = model_state(model)
    heads = transition.make_heads(config, device, ('B',))
    require(sum(p.numel() for p in heads['B'].parameters() if p.requires_grad) == 3200,
            'Unexpected B parameter count')
    initial_hash = model_state(heads['B'])
    require(initial_hash == settings['readout_initial_state_sha256'], 'B initial state differs from seed42 archive')
    attention_probe = transition.check_attention(transition.make_heads(config, device, ('B',))['B'], device)
    raw.write(folder / 'probe.json', {'attention': attention_probe,
              'gradients': transition.probe(model, data, config, device, ('B',))})
    val, val_coverage, baseline = transition.cache_split(model, data, rows['val'], 'val', config, device, output, heads)
    zero_replay = evaluation.replay_receipt(joint.study.validate_predictions(baseline, rows['val'], probabilities=False),
                                           baseline, selected, gold)
    raw.write(folder / 'epoch_zero_replay.json', zero_replay)
    require(zero_replay['passed'], 'B raw epoch zero differs from C')
    train, train_coverage, _ = transition.cache_split(model, data, rows['train'], 'train', config, device, output)
    raw.write(folder / 'edge_coverage.json', {'val': val_coverage, 'train': train_coverage})
    expected_orders = joint.study.epoch_orders(len(rows['train']), config['batch_size'], readout_seed, config['epochs'])
    best, last, updates = transition.fit(model, heads, train, val, rows['val'], baseline, config, device, folder,
                                        order_reference=dict(enumerate(expected_orders, 1)))
    del train, val
    require(updates == 6330, 'B training budget differs')
    path = folder / 'transition_readout.pt'
    torch.save({'best': best, 'last': last, 'config': config, 'query_sources': {'B': 'packet_mean'},
                'binding': {**binding, 'parent_sha256': selected['checkpoint_sha256']},
                'frozen_c_sha256': frozen, 'initial_transition_sha256': initial_hash}, path)
    loaded = torch.load(path, map_location='cpu', weights_only=True)
    exports = transition.replay_selected(model, loaded, data, rows['val'], config, device, output)
    receipts = {
        'C': evaluation.replay_receipt(joint.study.validate_predictions(exports['A'], rows['val'], probabilities=False),
                                       exports['A'], selected, gold),
        'B': evaluation.replay_receipt(joint.study.validate_predictions(exports['B'], rows['val'], probabilities=False),
                                       exports['B'], best['B'], None),
    }
    raw.write(folder / 'validation_replay.json', receipts)
    require(all(receipt['passed'] for receipt in receipts.values()) and model_state(model) == frozen
            and all(not p.requires_grad and p.grad is None for p in model.parameters()),
            'Selected B replay or frozen C changed')
    pred_path = folder / 'validation_predictions.jsonl.gz'
    raw.save_predictions(pred_path, exports['B'])
    result = {key: best['B'][key] for key in ('epoch', 'validation', 'prediction_ids_sha256')}
    result.update(updates=updates, checkpoint_sha256=raw.sha(path), predictions_sha256=raw.sha(pred_path),
                  last_validation=last['B']['validation'], initial_transition_sha256=initial_hash)
    raw.write(folder / 'metrics.json', result)
    return result


def load_data(repo, graph_root):
    config = raw.read(evaluation.CONFIG)
    split = repo / config['split_directory']
    evaluation.verify_inputs({split / 'split.json': config['split_json_sha256']})
    data = core.FrozenGraphs(split, config['split_id'], data_root=graph_root)
    require(data.metadata['sample_manifest_sha256'] == raw.read(d32.baseline.CONFIG)['sample_manifest_sha256'],
            'Fixed manifest differs')
    rows = {'val': evaluation.evaluation_rows(data, 'val', config), 'train': raw.choose_rows(data, 'train')}
    require(len(rows['train']) == 107621, 'Training sample count differs')
    return config, data, rows


def load_n0(repo, seed, data, rows, config, device):
    spec = raw.read(d32.baseline.CONFIG)['models'][f'N0{seed}']
    saved, _, _, inputs = d32.baseline.load_archive(repo, f'N0{seed}', spec, rows['val'], config,
                                                    data.metadata['sample_manifest_sha256'])
    warm = core.Batch.from_data_list([data.load(row) for row in rows['val'][:config['eval_batch_size']]]).to(device)
    model = d32.baseline.load_model(f'N0{seed}', saved, warm, device)
    return model, spec['sha256']['best.pt'], inputs


def parent_binding(saved, model, stage, n0, binding):
    settings = saved['config']
    recipe = d32.stage_protocols(raw.read(d32.CONFIG))[stage]
    require(settings['protocol']['seed'] == binding['seed']
            and d32.protocol_matches(stage, settings['protocol'], recipe)
            and settings.get('variant', settings.get('arm')) == stage
            and all(settings['binding'].get(key) == binding[key] for key in ('split_id', 'manifest_sha256'))
            and d32.n0_lineage(settings['binding']) == binding['n0_best_sha256']
            and (stage != 'C' or settings.get('query_source') == 'flow')
            and (stage != 'D32-Gate' or settings.get('architecture') == d32.gate.architecture('dynamic')),
            f'{stage} protocol or N0 lineage differs')
    require(d32.preserves_n0_encoders(stage, model, n0), f'{stage} did not preserve the required N0 state')


def preserve_incomplete(directory):
    if directory.exists():
        backup = directory.parent.parent / f'.{directory.parent.name}-interrupted'
        backup.mkdir(exist_ok=True)
        directory.rename(backup / f'{directory.name}-{uuid4().hex[:8]}')


def train_chain(repo, output, seed, graph_root, commit, settings):
    device = d32.baseline.runtime(core)
    torch.set_num_threads(2)
    config, data, rows = load_data(repo, graph_root)
    d32.guard_graph_loads(data, {'train', 'val'})
    n0, n0_sha, inputs = load_n0(repo, seed, data, rows, config, device)
    binding = {'seed': seed, 'n0_best_sha256': n0_sha, 'split_id': config['split_id'],
               'manifest_sha256': data.metadata['sample_manifest_sha256'], 'code_commit': commit,
               'graph_root': str(data.root), 'torch': str(torch.__version__),
               'pyg': torch_geometric.__version__, 'device': torch.cuda.get_device_name(device)}
    d32_folder = repo / 'outputs' / f'packet_transition_independent_d32_s{seed}_v1'
    complete = {stage: (output / stage / 'stage_complete.json').is_file() for stage in STAGES[1:]}
    available = [stage for stage in STAGES[1:-1] if complete[stage]]
    prior = available[-1] if available else 'D32-Gate'
    if prior == 'D32-Gate':
        d32_status = d32_folder / 'status.json'
        if not d32_status.is_file() or raw.read(d32_status)['status'] != 'completed':
            require(seed != 43, 'Missing archived D32-Gate43 on server')
            # The standalone D32 entry point and this continuation share the run folder.
            require(not d32_folder.exists(), 'Seed44 D32 already has partial output. Inspect its status and log.')
            with run_status(d32_folder, seed) as status:
                status['outcome'] = d32.run(repo, d32_folder, graph_root, seed, commit, 'train')
        status = raw.read(d32_status)
        require(status['status'] == 'completed', f'D32 or its reuse check is incomplete for seed{seed}')
        if status['outcome'] == 'reused_verified_checkpoint':
            reused = raw.read(d32_folder / 'reused_checkpoint.json')
            prior, directory = reused['stage'], (repo / reused['path']).parent
        else:
            directory = d32_folder / 'D32-Gate'
        if prior == 'D32-Gate':
            protocols = d32.stage_protocols(raw.read(d32.CONFIG))
            candidates = d32.inventory(repo, raw.read(d32.baseline.CONFIG)['models'], protocols, excluded=output)
            for item in sorted(candidates['seeds'][str(seed)]['candidates'],
                               key=lambda value: d32.STAGES.index(value['stage']), reverse=True):
                if item['stage'] == 'D32-Gate' or item['status'] != 'ready_for_replay':
                    continue
                try:
                    verified = d32.validate_candidate(repo, item, n0, data, rows['val'], config, device, output,
                                                       seed, n0_sha, protocols)
                except (OSError, ValueError, KeyError, RuntimeError) as error:
                    item.update(status='replay_rejected', reason=str(error))
                else:
                    prior, directory = verified['stage'], (repo / verified['path']).parent
                    raw.write(output / 'reused_checkpoint.json', verified)
                    break
            raw.write(output / 'inventory.json', candidates)
    else:
        directory = output / prior
    registration = settings['d32_seed43'] if seed == 43 and prior == 'D32-Gate' else None
    saved, parent, gold = selection(directory, prior, rows['val'], registration, repo)
    if prior in available:
        stage_record = raw.read(directory / 'stage_complete.json')
        require(all(stage_record[key] == parent[key] for key in ('epoch', 'checkpoint_sha256', 'prediction_ids_sha256')),
                f'{prior} completion receipt differs')
    model = d32.model_from_stage(prior, n0, device, seed)
    model.load_state_dict(saved['model_state'], strict=True)
    parent_binding(saved, model, prior, n0, binding)
    raw_replay(model, data, rows['val'], config, device, output, parent, gold, 'source_validation_replay')

    for stage in STAGES[STAGES.index(prior) + 1:-1]:
        raw.progress(output, 'stage_started', seed=seed, stage=stage)
        child_protocol = protocol(stage, seed, parent, config['split_id'])
        child_binding = {**binding, 'parent_stage': prior, 'parent_sha256': parent['checkpoint_sha256'],
                         'parent_prediction_ids_sha256': parent['prediction_ids_sha256']}
        model = build_child(stage, saved['model_state'], n0, device, seed)
        raw_replay(model, data, rows['val'], config, device, output, parent, gold, f'{stage}_epoch_zero_replay')
        stage_dir = output / stage
        if stage != 'C' or not (stage_dir / 'epoch_zero.json').is_file():
            preserve_incomplete(stage_dir)
        if stage == 'CL-Last':
            train_cross(model, data, rows, child_protocol, device, output, child_binding, parent, gold)
        elif stage == 'JCL-Multi':
            train_jcl(model, data, rows, child_protocol, device, output, child_binding, parent)
        else:
            train_c(model, data, rows, child_protocol, device, output, child_binding, parent, gold)
        saved, parent, gold = selection(output / stage, stage, rows['val'], repo=repo)
        parent_binding(saved, model, stage, n0, binding)
        require(saved['config']['binding'] == child_binding and raw.state_digest(saved['model_state']) == model_state(model),
                f'{stage} saved checkpoint or parent binding differs')
        raw.write(output / stage / 'config.json', saved['config'])
        raw.write(output / stage / 'stage_complete.json', {'checkpoint_sha256': parent['checkpoint_sha256'],
                  'epoch': parent['epoch'], 'prediction_ids_sha256': parent['prediction_ids_sha256']})
        raw.progress(output, 'stage_completed', seed=seed, stage=stage, epoch=parent['epoch'],
                     macro_f1=parent['validation']['macro_f1'])
        prior = stage
    require(prior == 'C', 'An independently selected original C is required')
    preserve_incomplete(output / 'B')
    pair = train_b(model, data, rows, device, output, binding, parent, gold, settings)
    evaluation.verify_inputs(inputs)
    raw.write(output / 'selection.json', {**binding, 'C': {**parent, 'directory': parent['directory'].as_posix()},
              'B': pair, 'test_evaluated': False})


def load_locked_pair(repo, seed, data, rows, config, device, output, readout_seed):
    folder = repo / 'outputs' / OUTPUT.format(seed=seed)
    require(raw.read(folder / 'status.json')['status'] == 'completed', f'Validation selection for seed{seed} is incomplete')
    locked = raw.read(folder / 'selection.json')
    n0, n0_sha, inputs = load_n0(repo, seed, data, rows, config, device)
    require(locked['seed'] == seed and locked['n0_best_sha256'] == n0_sha
            and locked['split_id'] == config['split_id']
            and locked['manifest_sha256'] == data.metadata['sample_manifest_sha256'], f'Seed{seed} pair lineage differs')
    directory = (repo / locked['C']['directory']).resolve()
    require(directory.is_relative_to((repo / 'outputs').resolve()), 'Locked C checkpoint must be below outputs')
    c_saved, c_selected, c_gold = selection(directory, 'C', rows['val'],
                                             {k: v for k, v in locked['C'].items() if k != 'directory'})
    model = d32.model_from_stage('C', n0, device, seed)
    model.load_state_dict(c_saved['model_state'], strict=True)
    parent_binding(c_saved, model, 'C', n0, locked)
    model.requires_grad_(False).eval()
    c_state = model_state(model)
    b_path = folder / 'B' / 'transition_readout.pt'
    b_gold_path = folder / 'B' / 'validation_predictions.jsonl.gz'
    b_metrics = locked['B']
    inputs.update({directory / 'best.pt': c_selected['checkpoint_sha256'],
                   directory / 'validation_predictions.jsonl.gz': c_selected['predictions_sha256'],
                   b_path: b_metrics['checkpoint_sha256'], b_gold_path: b_metrics['predictions_sha256']})
    evaluation.verify_inputs(inputs)
    saved = torch.load(b_path, map_location='cpu', weights_only=True)
    chosen = saved['best']['B']
    require(all(chosen[key] == b_metrics[key] for key in ('epoch', 'validation', 'prediction_ids_sha256'))
            and b_metrics['updates'] == 6330 and saved['config']['seed'] == seed
            and saved['config']['readout_seed'] == readout_seed and saved['query_sources'] == {'B': 'packet_mean'}
            and saved['config']['split_id'] == config['split_id'] and saved['config']['epochs'] == 30
            and saved['binding']['parent_sha256'] == c_selected['checkpoint_sha256']
            and saved['frozen_c_sha256'] == c_state, f'Seed{seed} B checkpoint selection differs')
    head = transition.make_heads(saved['config'], device, ('B',))['B']
    require(model_state(head) == saved['initial_transition_sha256'] == b_metrics['initial_transition_sha256'],
            'B initialization differs from the common readout seed')
    head.load_state_dict(chosen['state'], strict=True)
    head.requires_grad_(False).eval()
    b_gold = legacy.predictions(b_gold_path)
    expected = {'A': c_selected, 'B': chosen}
    batches = legacy.loader(data, rows['val'], 'val', config)
    val, _ = fixed.evaluate_partition(model, {'B': ('B', head)}, batches, rows['val'], device,
                                      output, f'seed{seed}_locked_validation')
    metrics = {arm: joint.study.validate_predictions(records, rows['val'], probabilities=False)
               for arm, records in val.items()}
    receipts = {key: evaluation.replay_receipt(metrics[key], val[key], expected[key], gold)
                for key, gold in (('A', c_gold), ('B', b_gold))}
    raw.write(output / f'seed{seed}_validation_replay.json', receipts)
    require(all(result['passed'] for result in receipts.values()), f'Seed{seed} locked pair validation differs')
    return model, head, metrics, inputs


def result_row(seed, c, b, comparison):
    return {'seed': seed, 'c_macro_f1': c['macro_f1'], 'b_macro_f1': b['macro_f1'],
            'macro_f1_delta_pp': comparison['delta_macro_f1_pp'],
            'webbased_f1_delta_pp': 100 * (b['per_class']['WebBased']['f1-score'] - c['per_class']['WebBased']['f1-score']),
            'directions': {key: (value['original_errors'], value['total_errors'])
                           for key, value in comparison['directions'].items() if key in (
                               'Recon -> WebBased', 'WebBased -> Recon')}}


def test_pairs(repo, output, graph_root, settings, commit):
    device = d32.baseline.runtime(core)
    torch.set_num_threads(2)
    config, data, rows = load_data(repo, graph_root)
    split = repo / config['split_directory']
    inputs = {split / 'split.json': config['split_json_sha256'],
              split / data.metadata['sample_manifest']: data.metadata['sample_manifest_sha256']}
    pairs = {}
    for seed in settings['seeds']:
        model, head, metrics, paths = load_locked_pair(repo, seed, data, rows, config, device, output,
                                                        settings['readout_seed'])
        pairs[seed] = (model, head, metrics)
        inputs.update(paths)
    historical_state = raw.read(raw.ROOT / 'configs/packet_transition_paired_seeds43_44_v1.json')['backbone_state_sha256']
    require(len({historical_state, *(model_state(model) for model, _, _ in pairs.values())}) == len(pairs) + 1,
            'Independent C states match')
    historic_path = repo / 'outputs/packet_transition_final_test_v1/summary.json'
    inputs[historic_path] = settings['seed42_fixed_test_summary_sha256']
    evaluation.verify_inputs(inputs)
    historic = raw.read(historic_path)['partitions']['test']
    summary = [result_row(42, historic['metrics']['A'], historic['metrics']['B42'],
                          historic['comparisons']['B42-A'])]
    test_rows = evaluation.evaluation_rows(data, 'test', config)
    require(all(historic['metrics'][name]['samples'] == len(test_rows) for name in ('A', 'B42')),
            'Archived seed42 test size differs')
    loader = DataLoader(evaluation.TestGraphs(data, test_rows), batch_size=config['eval_batch_size'], shuffle=False,
                        num_workers=config['workers'], generator=torch.Generator().manual_seed(242))
    hashes = {}
    for seed, (model, head, validation) in pairs.items():
        folder = output / f'seed{seed}'
        folder.mkdir()
        before = {'C': model_state(model), 'B': model_state(head)}
        exports, lengths = fixed.evaluate_partition(model, {'B': ('B', head)}, loader, test_rows, device,
                                                     output, f'seed{seed}_fixed_test')
        metrics, comparisons = attention.comparison_outputs(folder, test_rows, exports, lengths, [('B', 'A')])
        attention.write_predictions(folder / 'test_predictions.jsonl.gz', exports, lengths)
        require(before == {'C': model_state(model), 'B': model_state(head)}, 'Model changed during fixed test')
        summary.append(result_row(seed, metrics['A'], metrics['B'], comparisons['B-A']))
        raw.write(folder / 'metrics.json', {'validation': validation, 'test': metrics, 'comparison': comparisons['B-A'],
                  'states_sha256': before, 'prediction_ids_sha256': {
                      arm: joint.prediction_digest(records) for arm, records in exports.items()}})
        hashes[seed] = before
    evaluation.verify_inputs(inputs)
    average = mean(row['macro_f1_delta_pp'] for row in summary)
    raw.write(output / 'summary.json', {'pairs': summary, 'mean_delta_pp': average,
              'code_commit': commit, 'seed42_archive': settings['seed42_fixed_test_archive'], 'split_id': config['split_id'],
              'state_sha256': hashes, 'input_sha256': {str(path.relative_to(repo)): sha for path, sha in inputs.items()}})
    lines = ['# 独立主干上的报文转移读出', '',
             '测试集含 13,448 张原图。seed42 沿用原实验，seed43、44 从各自的 N0 主干训练。',
             '新增两组 B 都使用 seed42 的读出初始化和采样顺序。', '',
             '| 主干 seed | 原 C 测试 Macro-F1 | B 测试 Macro-F1 | 差值（百分点） | WebBased F1 变化（百分点） | Recon→WebBased | WebBased→Recon |',
             '| ---: | ---: | ---: | ---: | ---: | --- | --- |']
    for row in summary:
        rw, wr = (row['directions'][key] for key in ('Recon -> WebBased', 'WebBased -> Recon'))
        lines.append(f"| {row['seed']} | {row['c_macro_f1']:.6%} | {row['b_macro_f1']:.6%} | "
                     f"{row['macro_f1_delta_pp']:+.6f} | {row['webbased_f1_delta_pp']:+.6f} | "
                     f"{rw[0]} → {rw[1]} | {wr[0]} → {wr[1]} |")
    lines += ['', f'三组测试 Macro-F1 平均差为 {average:+.6f} 个百分点。']
    if all(row['macro_f1_delta_pp'] > 0 for row in summary):
        lines.append('报文转移读出的测试收益在三个独立主干上重复出现。')
    else:
        lines.append('报文转移读出的测试增益随主干变化。')
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


@contextmanager
def run_status(output, seed=None, resume=False):
    output.mkdir(parents=True, exist_ok=resume)
    with FileLock(str(output / '.run.lock'), timeout=0):
        status_path = output / 'status.json'
        if resume and status_path.exists() and raw.read(status_path)['status'] == 'completed':
            require((output / 'selection.json').is_file(), 'Completed training is missing its C/B selection')
            yield False
            return
        status = {'status': 'running', 'seed': seed} if seed else {'status': 'running'}
        raw.write(status_path, status)
        try:
            yield status
        except BaseException as error:
            raw.write(status_path, {**status, 'status': 'failed', 'error': str(error)})
            raise
        raw.write(status_path, {**status, 'status': 'completed'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--mode', choices=('train', 'test'), default='train')
    parser.add_argument('--seed', type=int, choices=(43, 44))
    parser.add_argument('--code-commit', required=True)
    args = parser.parse_args()
    require(re.fullmatch(r'[0-9a-f]{40}', args.code_commit), 'Expected full code SHA')
    require((args.seed is not None) == (args.mode == 'train'), 'Choose a seed for training or use --mode test')
    repo = args.repo.resolve()
    settings = raw.read(CONFIG)
    require(settings['seeds'] == [43, 44] and settings['readout_seed'] == 42, 'Unexpected registered seed controls')
    if args.mode == 'test':
        output = repo / 'outputs' / TEST_OUTPUT
        with run_status(output):
            test_pairs(repo, output, args.data_root, settings, args.code_commit)
    else:
        output = repo / 'outputs' / OUTPUT.format(seed=args.seed)
        with run_status(output, args.seed, resume=True) as active:
            if active:
                train_chain(repo, output, args.seed, args.data_root, args.code_commit, settings)
    print(json.dumps({'status': 'completed', 'output': str(output)}), flush=True)


if __name__ == '__main__':
    main()
