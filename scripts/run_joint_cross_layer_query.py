"""Preflight, train, or resume the three seed42 cross-layer query arms."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
from shutil import copyfile

from filelock import FileLock
import torch
import torch_geometric
from torch_geometric.data import Batch

import run_flow_query_joint_packet2 as query
import run_flow_query_joint_seeds as study
import run_joint_cross_layer_packet2 as legacy
from xgnid.joint_cross_layer_query import JointCrossLayerQueryClassifier

raw, require, LABELS = legacy.raw, legacy.require, legacy.LABELS
CONFIG = raw.ROOT / 'configs/joint_cross_layer_query_s42_v1.json'
DEFAULT_OUTPUT = 'outputs/joint_cross_layer_query_s42_v1'
ARMS = {'A': None, 'B': 'packet_mean', 'C': 'flow'}
NAMES = {'A': 'JCL-Multi', 'B': 'Mean 查询', 'C': 'Flow 查询'}


def make_models(state, device, seed):
    require(not any(name.startswith('packet_pool.') for name in state), 'Unexpected query parameters in JCL-Multi')
    source_digest = raw.state_digest(state)
    models = {}
    for arm, source in ARMS.items():
        base = raw.OperatorClassifier('GAT', seed=seed)
        args = (base.flow_encoder, base.packet_encoder, base.fusion)
        if source is None:
            model = legacy.JointCrossLayerClassifier(*args, 'multi')
            new_state = {}
        else:
            model = JointCrossLayerQueryClassifier(*args, source, seed)
            new_state = {f'packet_pool.{name}': value for name, value in model.packet_pool.state_dict().items()}
        model.load_state_dict({**state, **new_state}, strict=True)
        loaded = model.state_dict()
        require(raw.state_digest({name: loaded[name] for name in state}) == source_digest,
                f'{arm} inherited parameters or BN differ')
        models[arm] = model.to(device).eval()
    require(raw.state_digest(models['B'].state_dict()) == raw.state_digest(models['C'].state_dict()),
            'Mean and Flow must start with identical weights')
    return models


def optimizer(model, config):
    flow_to_packet, packet_to_packet = model.packet_relations()
    modules = [('flow_to_packet', flow_to_packet, 'encoder_lr', 8704),
               ('packet_to_packet', packet_to_packet, 'encoder_lr', 8512),
               ('projection', model.cross_layer, 'projection_lr', 8256),
               ('fusion', model.fusion, 'head_lr', 7144)]
    if isinstance(model, JointCrossLayerQueryClassifier):
        modules.append(('query_key', model.packet_pool, 'pool_lr', 1024))
    groups = []
    for name, module, lr, count in modules:
        params = list(module.parameters())
        require(sum(p.numel() for p in params) == count, f'{name} parameter count differs')
        groups.append({'name': name, 'params': params, 'lr': config[lr]})
    ids = [id(p) for group in groups for p in group['params']]
    require(len(set(ids)) == len(ids) and set(ids) == {id(p) for p in model.parameters() if p.requires_grad},
            'Optimizer parameter scope differs')
    return torch.optim.Adam(groups, weight_decay=config['weight_decay'])


@torch.no_grad()
def check_initial(models, batches, rows, gold, metrics, device, output):
    errors = {arm: {'scores': 0.0, 'gate': 0.0, 'archived_gate': 0.0} for arm in ARMS}
    failures = set()
    offset = 0
    for step, batch in enumerate(batches, 1):
        batch = batch.to(device)
        count = batch.num_graphs
        expected = gold[offset:offset + count]
        labels = batch.y.view(-1).tolist()
        reference = torch.tensor([r['flow_weights'] for r in expected], device=device)
        forwards = {arm: (model(batch), model.last_flow_weights) for arm, model in models.items()}
        for arm, (scores, gate) in forwards.items():
            require(scores.shape == (count, 8) and gate.shape == (count, 32), 'Invalid initial output shape')
            for key, candidate, base, tol in (('scores', scores, forwards['A'][0], 1e-6),
                                              ('gate', gate, forwards['A'][1], 1e-6),
                                              ('archived_gate', gate, reference, 1e-5)):
                difference = (candidate - base).abs()
                require(bool(torch.isfinite(difference).all()), 'Nonfinite initial output')
                errors[arm][key] = max(errors[arm][key], float(difference.max()))
                if not torch.allclose(candidate, base, atol=tol, rtol=tol):
                    failures.add(f'{arm}.{key}')
            actual = list(zip(batch.sample_id, labels, scores.argmax(-1).tolist()))
            if actual != legacy.identities(expected):
                failures.add(f'{arm}.predictions')
        offset += count
        if step == 1 or step % 200 == 0 or offset == len(rows):
            raw.progress(output, 'epoch_zero_equivalence', samples=offset, total=len(rows))
    require(offset == len(rows) and study.validate_predictions(gold, rows, probabilities=False) == metrics,
            'Incomplete source replay or source metrics differ')
    receipt = {'passed': not failures, 'samples': offset, 'failures': sorted(failures), 'validation': metrics,
               'max_abs_errors': errors, 'atol_rtol_paired': 1e-6, 'atol_rtol_archived_gate': 1e-5,
               'initial_state_sha256': {arm: raw.state_digest(model.state_dict()) for arm, model in models.items()}}
    raw.write(output / 'initialization.json', receipt)
    require(receipt['passed'], 'Epoch-zero equivalence failed; see initialization.json')
    return receipt


def probe(model, batch, config):
    rng = raw.rng_state()
    initial = raw.state_digest(model.state_dict())
    candidate = copy.deepcopy(model).train()
    frozen = legacy.frozen_hash(candidate)
    opt = optimizer(candidate, config)
    group_counts = {g['name']: sum(p.numel() for p in g['params']) for g in opt.param_groups}
    projections = ({'query': candidate.packet_pool.query.weight, 'key': candidate.packet_pool.key.weight}
                   if isinstance(candidate, JointCrossLayerQueryClassifier) else {})
    seen_groups, seen_projections, steps = set(), set(), []
    try:
        for index in range(config['probe_steps']):
            before = {id(p): p.detach().clone() for p in candidate.parameters() if p.requires_grad}
            loss = legacy.step(candidate, batch, opt)
            require(all((p.grad is not None and bool(torch.isfinite(p.grad).all())) if p.requires_grad
                        else p.grad is None for p in candidate.parameters()), 'Probe gradient scope differs')
            changed = {id(p) for p in candidate.parameters() if p.requires_grad and
                       bool(p.grad.ne(0).any()) and not torch.equal(before[id(p)], p)}
            updated = {g['name'] for g in opt.param_groups if any(id(p) in changed for p in g['params'])}
            norms = {name: float(p.grad.norm()) for name, p in projections.items()}
            updated_projections = {name for name, p in projections.items() if id(p) in changed}
            if index == 0 and projections:
                require(norms['key'] == 0 and updated_projections == {'query'}, 'Zero-query first step differs')
            seen_groups.update(updated)
            seen_projections.update(updated_projections)
            steps.append({'step': index + 1, 'nll': loss, 'updated_groups': sorted(updated),
                          'query_key_gradient_norms': norms, 'updated_query_key': sorted(updated_projections),
                          'updated_parameter_tensors': [name for name, p in candidate.named_parameters()
                                                        if id(p) in changed]})
        require(seen_groups == set(group_counts) and seen_projections == set(projections), 'Probe missed an update')
        require(legacy.frozen_hash(candidate) == frozen and raw.state_digest(model.state_dict()) == initial,
                'Probe changed the formal model or frozen parameters')
    finally:
        query.restore_rng(rng)
    return {'passed': True, 'parameter_groups': group_counts, 'trainable_parameters': sum(group_counts.values()),
            'frozen_state_sha256': frozen, 'initial_state_sha256': initial, 'steps': steps}


def prediction_digest(rows):
    return hashlib.sha256(json.dumps(legacy.identities(rows), separators=(',', ':')).encode()).hexdigest()


def save_snapshot(path, model, opt, sampler, loaders, cfg, epoch, metrics, predictions_sha256):
    temp = path.with_suffix('.tmp')
    torch.save({'model_state': {n: value.detach().cpu() for n, value in model.state_dict().items()},
                'optimizer_state': opt.state_dict(), 'config': cfg, 'epoch': epoch, 'validation': metrics,
                'prediction_ids_sha256': predictions_sha256, 'rng': raw.rng_state(),
                'sampler_rng': sampler.generator.get_state(),
                'loader_rngs': [loader.generator.get_state() for loader in loaders]}, temp)
    temp.replace(path)
    return raw.sha(path)


def restore_snapshot(path, model, opt, sampler, loaders, cfg, entry):
    require(raw.sha(path) == entry['checkpoint_sha256'], f'Checkpoint hash differs: {path}')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    require(saved['config'] == cfg and all(saved[key] == entry[key] for key in
            ('epoch', 'validation', 'prediction_ids_sha256')), 'Checkpoint metadata differs from the ledger')
    model.load_state_dict(saved['model_state'], strict=True)
    query.assert_tree_equal(opt.state_dict()['param_groups'], saved['optimizer_state']['param_groups'], 'Adam groups')
    opt.load_state_dict(copy.deepcopy(saved['optimizer_state']))
    query.assert_tree_equal(opt.state_dict(), saved['optimizer_state'], 'Adam state')
    updates = saved['epoch'] * len(sampler)
    expected = sum(len(group['params']) for group in opt.param_groups) if updates else 0
    require(len(opt.state) == expected and all(int(value['step']) == updates for value in opt.state.values()),
            'Checkpoint Adam update counts differ')
    require(len(saved['loader_rngs']) == len(loaders), 'Checkpoint loader state count differs')
    sampler.generator.set_state(saved['sampler_rng'])
    for loader, state in zip(loaders, saved['loader_rngs']):
        loader.generator.set_state(state)
    query.restore_rng(saved['rng'])


def selections(zero, history, prefix_epochs):
    key = lambda record: record['validation']['macro_f1']
    return {'epoch_zero': zero, 'best150': max([zero, *history[:prefix_epochs]], key=key),
            'best': max([zero, *history], key=key), 'trained_best': max(history, key=key, default=zero),
            'last': history[-1] if history else zero}


def snapshot_path(folder, epoch):
    return folder / f'epoch_{epoch:03d}.pt'


def commit_history(path, history):
    temp = path.with_suffix('.tmp')
    temp.write_text(''.join(json.dumps(row) + '\n' for row in history), encoding='utf-8', newline='\n')
    temp.replace(path)


def prune_snapshots(folder, chosen):
    retained = {snapshot_path(folder, row['epoch']) for row in chosen.values()}
    for path in folder.glob('epoch_*.pt'):
        if path not in retained:
            path.unlink()


def verify_history(history, zero, orders, config):
    chosen = selections(zero, history, config['prefix_epochs'])
    require(len(history) <= config['epochs'] and [r['epoch'] for r in history] == list(range(1, len(history) + 1))
            and [r['train_order_sha256'] for r in history] == orders[:len(history)], 'Epoch ledger or orders differ')
    if history:
        last = chosen['last']
        best, trained = chosen['best'], chosen['trained_best']
        result = {'epoch_zero': zero['validation'], 'best_epoch': best['epoch'], 'best_validation': best['validation'],
                  'best_trained_epoch': trained['epoch'], 'trained_best_validation': trained['validation'],
                  'optimizer_updates': len(history) * config['updates_per_epoch'], 'epochs': len(history),
                  'last_validation': last['validation'], 'orders': orders[:len(history)],
                  'training_samples_per_epoch': config['training_samples']}
        legacy.check_history(history, result, len(history), config['updates_per_epoch'], config['validation_samples'])
    return chosen


def fit(arm, model, data, rows, gold, baseline, orders, config, binding, device, output):
    folder = output / arm
    snapshots = folder / 'checkpoints'
    snapshots.mkdir(parents=True, exist_ok=True)
    ledger = folder / 'epochs.jsonl'
    zero_path = folder / 'epoch_zero.json'
    cfg = {'arm': arm, 'query_source': ARMS[arm], 'protocol': config, 'binding': binding, 'labels': LABELS,
           'initial_state_sha256': raw.state_digest(model.state_dict())}
    frozen = legacy.frozen_hash(model)
    sampler = raw.TrainBatches(len(rows['train']), config['batch_size'], config['seed'])
    loaders = (legacy.loader(data, rows['train'], 'train', config, sampler),
               legacy.loader(data, rows['val'], 'val', config))
    opt = optimizer(model, config)
    raw.seed_all(config['seed'])
    initial_rng = raw.rng_hash(raw.rng_state())
    history = query.read_history(ledger) if ledger.exists() else []
    if zero_path.exists():
        zero = raw.read(zero_path)
        require(zero['epoch'] == 0 and zero['validation'] == baseline
                and zero['prediction_ids_sha256'] == prediction_digest(gold)
                and zero['training_start_rng_sha256'] == initial_rng, 'Saved epoch-zero receipt differs')
        chosen = verify_history(history, zero, orders, config)
        for row in chosen.values():
            require(raw.sha(snapshot_path(snapshots, row['epoch'])) == row['checkpoint_sha256'],
                    'Retained epoch snapshot differs')
        start = chosen['last']
        restore_snapshot(snapshot_path(snapshots, start['epoch']), model, opt, sampler, loaders, cfg, start)
        require(legacy.frozen_hash(model) == frozen, 'Restored frozen state differs')
    else:
        require(not history, 'Cannot resume a ledger without the epoch-zero receipt')
        metrics, records = raw.evaluate(model, loaders[1], rows['val'], device, output, f'{arm}_epoch_zero', True)
        require(metrics == baseline and legacy.identities(records) == legacy.identities(gold)
                and raw.rng_hash(raw.rng_state()) == initial_rng, 'Formal epoch zero differs from source')
        digest = prediction_digest(records)
        sha = save_snapshot(snapshot_path(snapshots, 0), model, opt, sampler, loaders, cfg, 0, metrics, digest)
        zero = {'epoch': 0, 'validation': metrics, 'prediction_ids_sha256': digest, 'checkpoint_sha256': sha,
                'training_start_rng_sha256': initial_rng}
        raw.write(zero_path, zero)

    for epoch in range(len(history) + 1, config['epochs'] + 1):
        model.train()
        count = steps = 0
        total_loss = 0.0
        for batch in loaders[0]:
            if not steps:
                require(sampler.order_sha256 == orders[epoch - 1], f'{arm} epoch {epoch} order differs')
            expected = [rows['train'][i] for i in sampler.order[count:count + batch.num_graphs]]
            require([(r['sample_id'], r['label']) for r in expected] ==
                    list(zip(batch.sample_id, batch.y.view(-1).tolist())), 'Training IDs or labels differ')
            total_loss += legacy.step(model, batch.to(device), opt) * batch.num_graphs
            count += batch.num_graphs
            steps += 1
            if steps == 1 or steps % 50 == 0:
                raw.progress(output, f'{arm}_training', epoch=epoch, samples=count, total=len(rows['train']))
        require(count == len(rows['train']) and steps == config['updates_per_epoch']
                and legacy.frozen_hash(model) == frozen, 'Training coverage, budget or frozen state differs')
        metrics, records = raw.evaluate(model, loaders[1], rows['val'], device, output, f'{arm}_validation', True)
        digest = prediction_digest(records)
        sha = save_snapshot(snapshot_path(snapshots, epoch), model, opt, sampler, loaders, cfg, epoch, metrics, digest)
        entry = {'epoch': epoch, 'validation': metrics, 'samples': count, 'optimizer_updates': steps,
                 'train_nll': total_loss / count, 'train_order_sha256': sampler.order_sha256,
                 'prediction_ids_sha256': digest, 'checkpoint_sha256': sha}
        chosen = selections(zero, [*history, entry], config['prefix_epochs'])
        entry.update(best_epoch=chosen['best']['epoch'], best_trained_epoch=chosen['trained_best']['epoch'])
        history.append(entry)
        # Commit the ledger only after its snapshot is complete. Older snapshots
        # stay available until the new ledger has atomically replaced the old one.
        commit_history(ledger, history)
        prune_snapshots(snapshots, chosen)
        raw.progress(output, f'{arm}_epoch_complete', epoch=epoch, macro_f1=metrics['macro_f1'])

    chosen = verify_history(history, zero, orders, config)
    for label, row in chosen.items():
        path = snapshot_path(snapshots, row['epoch'])
        require(raw.sha(path) == row['checkpoint_sha256'], 'Selected checkpoint hash differs')
        destination = folder / f'{label}.pt'
        temp = destination.with_suffix('.copying')
        copyfile(path, temp)
        temp.replace(destination)
    best = chosen['best']
    raw.strict_reload(model, folder / 'best.pt', 'cpu', cfg, best['epoch'], best['validation'])
    metrics, exports = raw.evaluate(model, loaders[1], rows['val'], device, output, f'{arm}_best_replay', True)
    require(metrics == best['validation'] and prediction_digest(exports) == best['prediction_ids_sha256']
            and legacy.frozen_hash(model) == frozen, 'Best checkpoint raw replay differs')
    raw.save_predictions(folder / 'validation_predictions.jsonl.gz', exports)
    result = {'selected': chosen, 'frozen_state_sha256': frozen, 'raw_replay_passed': True,
              'updates': len(history) * config['updates_per_epoch'], 'training_start_rng_sha256': initial_rng,
              'predictions_sha256': raw.sha(folder / 'validation_predictions.jsonl.gz')}
    raw.write(folder / 'metrics.json', result)
    return result, exports


def prepare(repo):
    config = raw.read(CONFIG)
    source = repo / config['source_directory']
    checkpoint, predictions = source / 'best.pt', source / 'validation_predictions.jsonl.gz'
    split = repo / config['split_directory']
    inputs = {checkpoint: config['source_sha256'], predictions: config['source_predictions_sha256'],
              split / 'split.json': config['split_json_sha256']}
    missing = [str(path) for path in inputs if not path.is_file()]
    require(not missing, 'Missing inputs: ' + ', '.join(missing))
    for path, digest in inputs.items():
        require(raw.sha(path) == digest, f'Source SHA256 differs: {path}')
    require(torch.cuda.is_available() and torch.__version__.split('+')[0] == '2.3.1'
            and torch_geometric.__version__ == '2.5.3', 'Requires CUDA, Torch 2.3.1 and PyG 2.5.3')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    data = raw.FrozenGraphs(split, config['split_id'])
    rows = {part: raw.choose_rows(data, part) for part in ('train', 'val')}
    require(len(rows['train']) == config['training_samples'] and len(rows['val']) == config['validation_samples'],
            'Registered split sizes differ')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    metadata = payload['config']
    baseline = study.classification_metrics(payload['validation'])
    require(payload['epoch'] == config['source_epoch'] and baseline['macro_f1'] == config['source_macro_f1']
            and metadata['variant'] == 'JCL-Multi' and metadata['labels'] == LABELS
            and metadata['protocol']['seed'] == config['seed'] and
            metadata['protocol']['split_id'] == config['split_id'], 'Wrong JCL-Multi source checkpoint')
    gold = legacy.predictions(predictions)
    require(study.validate_predictions(gold, rows['val'], probabilities=False) == baseline,
            'Source validation predictions differ')
    models = make_models(payload['model_state'], device, config['seed'])
    require(all(legacy.frozen_hash(model) == config['source_frozen_state_sha256'] for model in models.values()),
            'Frozen parameters or BN buffers differ from JCL-Multi')
    binding = {'config_sha256': raw.sha(CONFIG), 'manifest_sha256': data.metadata['sample_manifest_sha256'],
               'inputs': {str(path.relative_to(repo)): digest for path, digest in inputs.items()},
               'source_code': {path: raw.sha(raw.ROOT / path) for path in (
                   'scripts/run_joint_cross_layer_query.py', 'scripts/run_joint_cross_layer_packet2.py',
                   'scripts/run_flow_query_joint_packet2.py', 'scripts/run_flow_query_joint_seeds.py',
                   'scripts/run_attention_operator_pair.py', 'scripts/summarize_attention_operator_pair.py',
                   'scripts/grouped_graphs.py', 'scripts/audit_graph_split.py', 'scripts/build_grouped_split.py',
                   'src/xgnid/joint_cross_layer_query.py', 'src/xgnid/joint_cross_layer.py',
                   'src/xgnid/cross_layer.py', 'src/xgnid/flow_query_pooling.py', 'src/xgnid/packet_pooling.py',
                   'src/xgnid/fusion_dimension.py', 'src/xgnid/model.py', 'src/xgnid/attention_operator_pair.py')},
               'torch': torch.__version__, 'pyg': torch_geometric.__version__,
               'cuda_devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    return config, binding, data, rows, gold, baseline, models, device


def preflight(output, config, binding, data, rows, gold, baseline, models, device):
    raw.write(output / 'config.json', config)
    raw.write(output / 'binding.json', binding)
    check_initial(models, legacy.loader(data, rows['val'], 'val', config), rows['val'], gold, baseline, device, output)
    probe_rows = raw.choose_rows(data, 'train', config['probe_per_class'])
    batch = Batch.from_data_list([data.load(row) for row in probe_rows]).to(device)
    require(len(probe_rows) == len(LABELS) * config['probe_per_class'], 'Incomplete stratified probe')
    raw.seed_all(config['seed'])
    probes = {arm: probe(model, batch, config) for arm, model in models.items()}
    raw.write(output / 'probes.json', {'sample_ids': [row['sample_id'] for row in probe_rows],
                                     'arms': probes, 'formal_optimizer_updates': 0})
    raw.write(output / 'preflight.json', {'passed': True, 'files': {
        name: raw.sha(output / name) for name in ('config.json', 'binding.json', 'initialization.json', 'probes.json')}})


def verify_preflight(output, config, binding, models):
    receipt = raw.read(output / 'preflight.json')
    names = ('config.json', 'binding.json', 'initialization.json', 'probes.json')
    require(receipt['passed'] and receipt['files'] == {name: raw.sha(output / name) for name in names},
            'Run a passing preflight before training')
    require(raw.read(output / 'config.json') == config and raw.read(output / 'binding.json') == binding,
            'Training inputs or environment differ from preflight')
    init = raw.read(output / 'initialization.json')
    probes = raw.read(output / 'probes.json')['arms']
    require(init['passed'] and init['samples'] == config['validation_samples'] and
            all(probes[arm]['passed'] and init['initial_state_sha256'][arm] ==
                probes[arm]['initial_state_sha256'] == raw.state_digest(model.state_dict())
                for arm, model in models.items()), 'Formal models differ from verified preflight')


def comparison_evidence(before, after, before_metrics, after_metrics, output, name):
    counts, changed = raw.compare(before, after)
    raw.save_predictions(output / f'{name}_changed_samples.jsonl.gz', changed)
    directions = {}
    for actual, wrong in (('WebBased', 'Recon'), ('Recon', 'WebBased')):
        y, p = LABELS.index(actual), LABELS.index(wrong)
        repairs = sum(r['label'] == y and r['old_prediction'] == p and r['new_prediction'] == y for r in changed)
        regressions = sum(r['label'] == y and r['old_prediction'] == y and r['new_prediction'] == p for r in changed)
        directions[f'{actual} -> {wrong}'] = {
            'before_errors': before_metrics['confusion_matrix'][y][p],
            'after_errors': after_metrics['confusion_matrix'][y][p],
            'repaired': repairs, 'regressed': regressions}
    return {**counts, 'directions': directions,
            'spoofing_recall_delta_pp': 100 * (after_metrics['per_class']['Spoofing']['recall'] -
                                               before_metrics['per_class']['Spoofing']['recall'])}


def report(output, config, results, exports, baseline):
    metrics = {arm: result['selected']['best']['validation'] for arm, result in results.items()}
    scores = {arm: metric['macro_f1'] for arm, metric in metrics.items()}
    control = max(('A', 'B'), key=scores.get)
    comparisons = {}
    for before, after in (('B', 'C'), ('A', 'C'), ('A', 'B')):
        name = f'{after}_minus_{before}'
        comparisons[name] = {'macro_f1_delta_pp': 100 * (scores[after] - scores[before]),
                             **comparison_evidence(exports[before], exports[after], metrics[before], metrics[after],
                                                   output, name)}
    delta = 100 * (scores['C'] - scores[control])
    analysis = {'comparisons': comparisons, 'C_minus_source_pp': 100 * (scores['C'] - baseline['macro_f1']),
                'stronger_control': control, 'C_minus_stronger_control_pp': delta,
                'expansion_threshold_pp': config['budget_threshold_pp'],
                'priority_recheck_candidate': delta >= config['budget_threshold_pp']}
    with (output / 'per_class.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['arm', 'class', 'precision', 'recall', 'f1', 'support'])
        for arm, metric in metrics.items():
            for label, row in metric['per_class'].items():
                writer.writerow([arm, label, row['precision'], row['recall'], row['f1-score'], row['support']])

    lines = ['# Seed42 四项合并实验', '',
             f"共同起点为 JCL-Multi 第 48 轮，验证 Macro-F1 {baseline['macro_f1']:.6%}。",
             '', '| 组 | 0–150 最佳（轮次） | 0–200 最佳（轮次） | 训练后最佳（轮次） | 末轮 | 更新数 |',
             '|---|---:|---:|---:|---:|---:|']
    for arm, result in results.items():
        values = [f"{result['selected'][key]['validation']['macro_f1']:.6%}（{result['selected'][key]['epoch']}）"
                  for key in ('best150', 'best', 'trained_best', 'last')]
        lines.append(f"| {arm} {NAMES[arm]} | " + ' | '.join(values) + f" | {result['updates']} |")
    lines += ['', '| 比较 | Macro-F1 增量（百分点） | 修复 / 退步 | WebBased → Recon 错分 | Recon → WebBased 错分 | Spoofing 召回增量（百分点） |',
              '|---|---:|---:|---:|---:|---:|']
    for name, entry in comparisons.items():
        transitions = [f"{row['before_errors']} → {row['after_errors']}" for row in entry['directions'].values()]
        counts = entry['counts']
        lines.append(f"| {name.replace('_minus_', ' − ')} | {entry['macro_f1_delta_pp']:+.6f} | "
                     f"{counts['repaired']} / {counts['regressed']} | " + ' | '.join(transitions) +
                     f" | {entry['spoofing_recall_delta_pp']:+.6f} |")
    verdict = '达到' if analysis['priority_recheck_candidate'] else '低于'
    lines += ['', f"主比较为 C − B。C 相对历史起点的变化为 {analysis['C_minus_source_pp']:+.6f} 个百分点。",
              f"C 相对较强对照 {control} 的变化为 {delta:+.6f} 个百分点，{verdict} "
              f"+{config['budget_threshold_pp']:.2f} 个百分点的扩展预算线。", '',
              '| 组 | WebBased 召回 | Recon 召回 | Spoofing 召回 |', '|---|---:|---:|---:|']
    for arm, metric in metrics.items():
        values = [f"{metric['per_class'][label]['recall']:.6%}" for label in ('WebBased', 'Recon', 'Spoofing')]
        lines.append(f"| {arm} | " + ' | '.join(values) + ' |')
    lines += ['', '八类 Precision、Recall、F1 见 `per_class.csv`，各比较的翻转样本见 `*_changed_samples.jsonl.gz`。', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    raw.write(output / 'summary.json', {'status': 'completed', 'test_evaluated': False,
                                      'models': results, 'source_validation': baseline, 'analysis': analysis})


def run(repo, output, mode):
    config, binding, data, rows, gold, baseline, models, device = prepare(repo)
    if mode == 'preflight':
        preflight(output, config, binding, data, rows, gold, baseline, models, device)
        return
    verify_preflight(output, config, binding, models)
    sampler = raw.TrainBatches(len(rows['train']), config['batch_size'], config['seed'])
    require(len(sampler) == config['updates_per_epoch'], 'Unexpected configured update count')
    orders = study.epoch_orders(len(rows['train']), config['batch_size'], config['seed'], config['epochs'])
    results, exports = {}, {}
    for arm, model in models.items():
        results[arm], exports[arm] = fit(arm, model, data, rows, gold, baseline, orders, config, binding, device, output)
    require(len({result['training_start_rng_sha256'] for result in results.values()}) == 1,
            'Formal training start RNG differs')
    for path, digest in binding['inputs'].items():
        require(raw.sha(repo / path) == digest, f'Historical input changed: {path}')
    report(output, config, results, exports, baseline)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, default=Path(DEFAULT_OUTPUT))
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ('preflight', 'train', 'resume'):
        modes.add_argument(f'--{mode}', dest='mode', action='store_const', const=mode)
    args = parser.parse_args()
    repo = args.repo.resolve()
    outputs = (repo / 'outputs').resolve()
    output = (repo / args.output).resolve()
    require(output.parent == outputs, 'Use a run directory directly under outputs')
    outputs.mkdir(exist_ok=True)
    with FileLock(str(outputs / f'.{output.name}.lock'), timeout=0):
        if args.mode == 'preflight':
            output.mkdir()
        else:
            previous = raw.read(output / 'status.json')['status']
            allowed = ('preflight_passed', 'preflighting') if args.mode == 'train' else ('training', 'failed')
            require(previous in allowed and (output / 'preflight.json').is_file(),
                    f'Cannot {args.mode} from status {previous}')
        raw.write(output / 'status.json', {'status': 'preflighting' if args.mode == 'preflight' else 'training'})
        try:
            run(repo, output, args.mode)
        except BaseException as error:
            raw.write(output / 'status.json', {'status': 'failed', 'mode': args.mode, 'error': str(error)})
            raise
        status = 'preflight_passed' if args.mode == 'preflight' else 'completed'
        raw.write(output / 'status.json', {'status': status})
        print(json.dumps({'status': status, 'output': str(output)}), flush=True)


if __name__ == '__main__':
    main()
