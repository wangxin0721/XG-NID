"""Train controlled ordered-transition readouts on each validation-selected C."""
import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from subprocess import check_output
from uuid import uuid4

import torch
from torch_geometric.loader import DataLoader

import run_independent_packet_transition as independent
import run_packet_transition_attention_diagnosis as attention
import run_packet_transition_readout as original
from xgnid.ordered_transition_readout import OrderedTransitionReadout, attach_continuity


raw, joint, evaluation, require = original.raw, original.joint, original.evaluation, original.require
CONFIG = raw.ROOT / 'configs/ordered_transition_readout_v1.json'
OUTPUT = 'outputs/ordered_transition_readout_v1'
ARMS = ('single', 'pointwise', 'ordered')
COMPARISONS = (('ordered', 'single'), ('ordered', 'pointwise'), ('pointwise', 'single'))


def load_source(repo, seed, graph_root, output):
    if seed == 42:
        config = original.base_config()
        model, data, rows, device, selected, inputs = original.load_source(repo, config, graph_root)
        directory = repo / config['source_directory'] / 'C'
        gold = joint.legacy.predictions(directory / 'validation_predictions.jsonl.gz')
    else:
        device = independent.d32.baseline.runtime(independent.core)
        config, data, rows = independent.load_data(repo, graph_root)
        folder = repo / 'outputs' / independent.OUTPUT.format(seed=seed)
        lock_path = folder / 'selection.json'
        locked = raw.read(lock_path)
        require(raw.read(folder / 'status.json')['status'] == 'completed' and locked['seed'] == seed,
                'Independent source training has not completed')
        n0, n0_sha, inputs = independent.load_n0(repo, seed, data, rows, config, device)
        require(locked['n0_best_sha256'] == n0_sha and locked['split_id'] == config['split_id']
                and locked['manifest_sha256'] == data.metadata['sample_manifest_sha256'], 'Wrong N0 lineage')
        directory = repo / locked['C']['directory']
        saved, selected, gold = independent.selection(
            directory, 'C', rows['val'], {k: v for k, v in locked['C'].items() if k != 'directory'})
        model = independent.d32.model_from_stage('C', n0, device, seed)
        model.load_state_dict(saved['model_state'], strict=True)
        independent.parent_binding(saved, model, 'C', n0, locked)
        inputs.update({lock_path: raw.sha(lock_path), directory / 'best.pt': selected['checkpoint_sha256'],
                       directory / 'metrics.json': raw.sha(directory / 'metrics.json'),
                       directory / 'validation_predictions.jsonl.gz': selected['predictions_sha256']})
    independent.freeze_c(model)
    inputs.update({data.directory / 'split.json': config['split_json_sha256'],
                   data.directory / data.metadata['sample_manifest']: data.metadata['sample_manifest_sha256']})
    binding = {'backbone_seed': seed, 'frozen_c_sha256': independent.model_state(model),
               'split_id': config['split_id'], 'manifest_sha256': data.metadata['sample_manifest_sha256'],
               'validation_digest': joint.prediction_digest(gold)}
    return model, data, rows, device, config, selected, gold, inputs, binding


def encode(model, batch, device):
    src, dst, _, _, _ = original.chronological_edges(batch)
    features, counts = original.encode(model, batch, device)
    extended = attach_continuity(features.transitions, src, dst, features.lengths)
    return features._replace(transitions=extended), counts


@torch.no_grad()
def cache_split(model, data, rows, part, config, device, output):
    chunks, predictions = [], []
    for step, indices, batch, _ in original.diagnosis.aligned_batches(data, rows, part, config):
        features, _ = encode(model, batch, device)
        chunks.append(original.Features(*(value.cpu() for value in features)))
        predictions.extend(original.baseline(model, features).argmax(-1).tolist())
        if step == 1 or step % 200 == 0 or indices.stop == len(rows):
            raw.progress(output, 'cache_' + part, samples=indices.stop, total=len(rows))
    cache = original.FrozenCache(chunks, [row['label'] for row in rows])
    owners = torch.repeat_interleave(torch.arange(len(rows)), cache.features.lengths)
    contexts = torch.bincount(owners[cache.features.transitions[:, 70].bool()], minlength=len(rows))
    coverage = {'samples': len(rows), 'valid_transitions': int(cache.features.lengths.sum()),
                'ordered_contexts': int(contexts.sum()), 'flows_with_ordered_contexts': int((contexts > 0).sum()),
                'flows_without_transitions': int((cache.features.lengths == 0).sum()),
                'context_coverage_by_class': {
                    label: {'samples': sum(row['label'] == i for row in rows),
                            'with_context': sum(row['label'] == i and count > 0
                                                for row, count in zip(rows, contexts.tolist()))}
                    for i, label in enumerate(original.LABELS)}}
    return cache, original.prediction_records(rows, predictions), coverage


def make_heads(device, settings):
    return {name: OrderedTransitionReadout(name, settings['readout_seed']).to(device) for name in ARMS}


def check_initial(heads, model, cache, device):
    features, _ = cache.batch(range(min(16, len(cache.labels))), device)
    with torch.no_grad():
        base = original.baseline(model, features)
        for name, head in heads.items():
            scores, residual, _ = original.transition_forward(model, head, 'B', features)
            require(torch.equal(scores, base) and not bool(residual.any()), f'{name} epoch zero differs')
    reference = {name: value for name, value in heads['single'].state_dict().items()}
    for name in ARMS:
        require(all(torch.equal(heads[name].state_dict()[key], value) for key, value in reference.items()),
                f'{name} common initialization differs')


def verify_ledger(folder, saved, settings, config, reference, count):
    rows = [json.loads(line) for line in (folder / 'epochs.jsonl').read_text(encoding='utf-8').splitlines()]
    require([row['epoch'] for row in rows] == list(range(settings['epochs'] + 1)), 'Incomplete epoch ledger')
    sampler = raw.TrainBatches(count, config['batch_size'], config['readout_seed'])
    expected_orders = []
    for _ in range(settings['epochs']):
        for _ in sampler:
            pass
        expected_orders.append(sampler.order_sha256)
    require(rows[0]['updates_per_arm'] == rows[0]['samples_per_arm'] == 0 and rows[0]['order_sha256'] is None
            and rows[0]['validation']['B'] == reference, 'Wrong epoch-zero record')
    for row, order in zip(rows[1:], expected_orders):
        require(row['updates_per_arm'] == len(sampler) and row['samples_per_arm'] == count
                and row['order_sha256'] == order, 'Training budget or sampling order differs')
    top = max(rows, key=lambda row: row['validation']['B']['macro_f1'])
    for phase, row in (('best', top), ('last', rows[-1])):
        require(saved[phase]['B']['epoch'] == row['epoch']
                and saved[phase]['B']['validation'] == row['validation']['B'], 'Checkpoint selection differs')
    return expected_orders


@torch.no_grad()
def replay(model, heads, data, rows, config, device, output, part):
    records = {name: [] for name in ('A', *ARMS)}
    lengths = []
    if part == 'test':
        batches = DataLoader(evaluation.TestGraphs(data, rows), batch_size=config['eval_batch_size'],
                             shuffle=False, num_workers=config['workers'],
                             generator=torch.Generator().manual_seed(242))
    else:
        batches = joint.legacy.loader(data, rows, 'val', config)
    for step, batch in enumerate(batches, 1):
        start = len(lengths)
        require(list(zip(batch.sample_id, batch.y.view(-1).tolist())) ==
                [(row['sample_id'], row['label']) for row in rows[start:start + batch.num_graphs]],
                'Replay graph order or labels differ')
        features, _ = encode(model, batch, device)
        lengths.extend(features.lengths.tolist())
        records['A'].extend(original.baseline(model, features).argmax(-1).tolist())
        for name, head in heads.items():
            scores, _, _ = original.transition_forward(model, head, 'B', features)
            require(bool(torch.isfinite(scores).all()), f'{name} nonfinite replay score')
            records[name].extend(scores.argmax(-1).tolist())
        if step == 1 or step % 200 == 0 or len(lengths) == len(rows):
            raw.progress(output, 'replay_' + part, samples=len(lengths), total=len(rows))
    require(len(lengths) == len(rows), 'Incomplete raw graph replay')
    return {name: original.prediction_records(rows, values) for name, values in records.items()}, lengths


def save_results(folder, exports, lengths, rows, part):
    folder.mkdir(exist_ok=True)
    pairs = (*COMPARISONS, *((name, 'A') for name in ARMS))
    metrics, comparisons = attention.comparison_outputs(folder, rows, exports, lengths, pairs)
    attention.write_predictions(folder / (part + '_predictions.jsonl.gz'), exports, lengths)
    result = {'metrics': metrics, 'comparisons': comparisons}
    raw.write(folder / 'metrics.json', result)
    return result


def verify_frozen(model, binding, inputs):
    require(independent.model_state(model) == binding['frozen_c_sha256']
            and all(not param.requires_grad and param.grad is None for param in model.parameters())
            and all(not module.training for module in model.modules()), 'Frozen source C changed')
    evaluation.verify_inputs(inputs)


def train_seed(repo, seed, graph_root, output, settings, code_commit):
    model, data, rows, device, source_config, selected, gold, inputs, binding = load_source(repo, seed, graph_root, output)
    config = {**source_config, **raw.read(original.CONFIG), 'seed': seed,
              'readout_seed': settings['readout_seed'], 'epochs': settings['epochs'],
              'batch_size': settings['batch_size'], 'eval_batch_size': settings['eval_batch_size'],
              'learning_rate': settings['learning_rate'],
              'baseline_prediction_ids_sha256': binding['validation_digest']}
    require(len(rows['train']) == settings['training_samples'] and len(rows['val']) == settings['validation_samples'],
            'Fixed split membership count differs')
    settings_sha = raw.sha(CONFIG)
    raw.write(output / 'config.json', settings)
    raw.write(output / 'provenance.json', {'code_commit': code_commit, 'source': binding, 'settings_sha256': settings_sha,
              'runner_sha256': raw.sha(Path(__file__)), 'torch': str(torch.__version__),
              'device': torch.cuda.get_device_name(device),
              'input_sha256': {str(path.relative_to(repo)): value for path, value in inputs.items()}})
    val, reference, coverage_val = cache_split(model, data, rows['val'], 'val', config, device, output)
    receipt = evaluation.replay_receipt(
        joint.study.validate_predictions(reference, rows['val'], probabilities=False), reference, selected, gold)
    raw.write(output / 'baseline_validation_replay.json', receipt)
    require(receipt['passed'], 'Source C validation replay differs')
    train, _, coverage_train = cache_split(model, data, rows['train'], 'train', config, device, output)
    raw.write(output / 'coverage.json', {'train': coverage_train, 'val': coverage_val})
    heads, selection, orders = make_heads(device, settings), {}, None
    check_initial(heads, model, val, device)
    for name, head in heads.items():
        require(sum(param.numel() for param in head.parameters()) == settings['parameters_per_arm'][name],
                f'{name} parameter count differs')
        folder, path = output / name, output / name / 'readout.pt'
        if path.is_file():
            saved = torch.load(path, map_location='cpu', weights_only=True)
            require(saved['binding'] == binding and saved['settings_sha256'] == settings_sha
                    and saved['mode'] == name and saved['config'] == config, 'Saved arm belongs to another run')
        else:
            if folder.exists():
                folder.rename(output / (name + '_interrupted_' + uuid4().hex[:8]))
            folder.mkdir()
            best, last, updates = original.fit(
                model, {'B': head}, train, val, rows['val'], reference, config, device, folder)
            require(updates == len(raw.TrainBatches(len(train.labels), config['batch_size'],
                    config['readout_seed'])) * settings['epochs'], 'Unexpected full training budget')
            saved = {'mode': name, 'binding': binding, 'settings_sha256': settings_sha, 'config': config,
                     'best': best, 'last': last}
            temporary = path.with_suffix('.tmp')
            torch.save(saved, temporary)
            temporary.replace(path)
        current_orders = verify_ledger(folder, saved, settings, config, selected['validation'], len(train.labels))
        require(orders is None or current_orders == orders, 'Arms used different sampling orders')
        orders = current_orders
        head.load_state_dict(saved['best']['B']['state'], strict=True)
        head.requires_grad_(False).eval()
        selection[name] = {key: value for key, value in saved['best']['B'].items() if key != 'state'}
        selection[name].update(checkpoint_sha256=raw.sha(path))
    del train, val
    exports, lengths = replay(model, heads, data, rows['val'], config, device, output, 'val')
    receipts = {name: evaluation.replay_receipt(joint.study.validate_predictions(exports[name], rows['val'],
                probabilities=False), exports[name], selection[name], None) for name in ARMS}
    require(all(item['passed'] for item in receipts.values()), 'Selected readout raw replay differs')
    raw.write(output / 'selected_validation_replay.json', receipts)
    results = save_results(output / 'validation', exports, lengths, rows['val'], 'validation')
    verify_frozen(model, binding, inputs)
    raw.write(output / 'selection.json', {'binding': binding, 'settings_sha256': settings_sha,
              'selected': selection, 'results': results, 'test_evaluated': False})
    write_report(output, {seed: results}, 'validation')


def load_locks(root, settings):
    locks = {}
    for seed in settings['backbone_seeds']:
        folder = root / ('seed' + str(seed))
        require(raw.read(folder / 'status.json')['status'] == 'completed', f'Seed{seed} selection incomplete')
        lock = raw.read(folder / 'selection.json')
        require(lock['binding']['backbone_seed'] == seed and lock['settings_sha256'] == raw.sha(CONFIG)
                and lock['test_evaluated'] is False, 'Changed validation protocol')
        for name in ARMS:
            evaluation.verify_inputs({folder / name / 'readout.pt': lock['selected'][name]['checkpoint_sha256']})
        locks[seed] = lock
    require(len({lock['binding']['frozen_c_sha256'] for lock in locks.values()}) == len(locks),
            'The registered C backbones are not independent')
    return locks


def test_all(repo, root, graph_root, output, settings):
    locks = load_locks(root, settings)
    results = {}
    for seed, lock in locks.items():
        folder = output / ('seed' + str(seed))
        folder.mkdir()
        model, data, rows, device, config, _, _, inputs, binding = load_source(repo, seed, graph_root, folder)
        require(binding == lock['binding'], 'Source C changed after validation locking')
        heads = make_heads(device, settings)
        for name, head in heads.items():
            saved = torch.load(root / ('seed' + str(seed)) / name / 'readout.pt',
                               map_location='cpu', weights_only=True)
            selected = saved['best']['B']
            require(saved['binding'] == binding and saved['settings_sha256'] == lock['settings_sha256']
                    and saved['mode'] == name and all(selected[key] == lock['selected'][name][key]
                    for key in ('epoch', 'validation', 'prediction_ids_sha256')), 'Locked readout selection changed')
            head.load_state_dict(selected['state'], strict=True)
            head.requires_grad_(False).eval()
        exports, lengths = replay(model, heads, data, rows['val'], config, device, folder, 'val')
        require(all(joint.prediction_digest(exports[name]) == lock['selected'][name]['prediction_ids_sha256']
                    for name in ARMS), 'Validation replay differs before fixed test')
        config['partition_sizes']['test'] = settings['test_samples']
        test_rows = evaluation.evaluation_rows(data, 'test', config)
        exports, lengths = replay(model, heads, data, test_rows, config, device, folder, 'test')
        results[seed] = save_results(folder, exports, lengths, test_rows, 'test')
        verify_frozen(model, binding, inputs)
    write_report(output, results, 'test')


def write_report(output, results, partition):
    lines = ['# 有序双尺度报文转移读出', '',
             'Single 为原单尺度读出，Pointwise 为同参数量逐转移映射，Ordered 编码连续两次转移。', '',
             '| 主干 seed | Single Macro-F1 | Pointwise Macro-F1 | Ordered Macro-F1 | '
             'Ordered−Single（百分点） | Ordered−Pointwise（百分点） |',
             '| ---: | ---: | ---: | ---: | ---: | ---: |']
    differences = {name: [] for name in ('ordered-single', 'ordered-pointwise')}
    for seed, row in results.items():
        scores = [row['metrics'][name]['macro_f1'] for name in ARMS]
        ds = [row['comparisons'][name]['delta_macro_f1_pp'] for name in differences]
        lines.append(f'| {seed} | ' + ' | '.join(f'{v:.6%}' for v in scores) + ' | '
                     + ' | '.join(f'{v:+.6f}' for v in ds) + ' |')
        for name, value in zip(differences, ds):
            differences[name].append(value)
    aggregate = {name: {'mean_pp': mean(values), 'sd_pp': stdev(values) if len(values) > 1 else None}
                 for name, values in differences.items()}
    for name, item in aggregate.items():
        lines += ['', f"{name} 平均差为 {item['mean_pp']:+.6f} 个百分点。"]
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    raw.write(output / 'summary.json', {'partition': partition, 'results': results, 'aggregate': aggregate})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output', type=Path, default=Path(OUTPUT))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--seed', type=int, choices=(42, 43, 44))
    mode.add_argument('--test', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    repo, root = args.repo.resolve(), (args.repo / args.output).resolve()
    require(root.is_relative_to(repo / 'outputs') and root != repo / 'outputs', 'Use a dedicated outputs directory')
    torch.set_num_threads(2)
    settings = raw.read(CONFIG)
    if args.test:
        with independent.run_status(root / 'test'):
            test_all(repo, root, args.data_root, root / 'test', settings)
    else:
        folder = root / ('seed' + str(args.seed))
        with independent.run_status(folder, args.seed, resume=args.resume) as active:
            if active:
                commit = check_output(['git', 'rev-parse', 'HEAD'], cwd=raw.ROOT, text=True).strip()
                train_seed(repo, args.seed, args.data_root, folder, settings, commit)
    print(json.dumps({'status': 'completed', 'output': str(root)}), flush=True)


if __name__ == '__main__':
    main()
