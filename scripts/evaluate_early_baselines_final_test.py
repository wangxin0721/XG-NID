"""Replay six locked E0-FT/N0 checkpoints, evaluate the fixed test, and write paper tables."""
import argparse
import csv
import json
from pathlib import Path
import platform
import re
import shlex
from statistics import mean, stdev
import sys

import torch
import torch_geometric
from torch_geometric.loader import DataLoader

import evaluate_joint_cross_layer_query as evaluation
from e3_gate_study import runtime
import train_grouped as grouped
from run_fusion_decision_diagnosis import direction_stats
from xgnid.edge_experiments import build_experiment


raw, joint, require, LABELS = evaluation.raw, evaluation.joint, evaluation.require, evaluation.LABELS
CONFIG = raw.ROOT / 'configs/early_baselines_final_test_v1.json'
OUTPUT = 'outputs/early_baselines_final_test_v1'
SEEDS = (42, 43, 44)
NAMES = tuple(f'{arm}{seed}' for seed in SEEDS for arm in ('E0FT', 'N0'))
FOCUS = ('Recon', 'WebBased')
CLASSIFICATION = ('accuracy', 'macro_f1', 'weighted_f1')
DIRECTIONS = (('Recon', 'WebBased'), ('WebBased', 'Recon'), ('Benign', 'Spoofing'), ('Spoofing', 'Benign'))
PAPER_ARCHIVE = 'outputs/packet_transition_final_test_v1/summary.json'
FUSION_DIAGNOSIS = 'outputs/fusion_decision_diagnosis_s42_v1/metrics.json'
METRIC_TOLERANCE = 1e-12


def matches_historical(current, archived):
    """Archived scores came from sklearn. Compare their arithmetic to tolerance."""
    return (current['confusion_matrix'] == archived['confusion_matrix'] and current['samples'] == archived['samples']
            and all(abs(current[key] - archived[key]) <= METRIC_TOLERANCE for key in CLASSIFICATION)
            and all(current['per_class'][label]['support'] == archived['per_class'][label]['support']
                    and all(abs(current['per_class'][label][key] - archived['per_class'][label][key]) <= METRIC_TOLERANCE
                            for key in ('precision', 'recall', 'f1-score')) for label in LABELS))


def load_archive(repo, name, spec, val_rows, config, manifest_sha):
    directory = repo / spec['directory']
    paths = {directory / file: sha for file, sha in spec['sha256'].items()}
    evaluation.verify_inputs(paths)
    training = raw.read(directory / 'config.json')
    metrics = raw.read(directory / 'metrics.json')
    saved = torch.load(directory / 'best.pt', weights_only=True, map_location='cpu')
    arm, seed = name[:-2], int(name[-2:])
    require(saved['config'] == training and training['protocol']['seed'] == seed
            and training['protocol']['evaluation_batch_size'] == config['eval_batch_size']
            and training['variant'] == metrics['variant'] == arm and metrics['test_evaluated'] is False
            and saved['epoch'] == metrics['best_epoch'] == spec['epoch']
            and saved['validation'] == metrics['best_validation']
            and metrics['checkpoint_sha256']['best.pt'] == spec['sha256']['best.pt']
            and metrics['predictions_sha256'] == spec['sha256']['validation_predictions.jsonl.gz'],
            f'{name} checkpoint selection or training metadata differs')
    binding = training['binding']
    if arm == 'N0':
        require(binding['split_id'] == config['split_id'] and binding['manifest_sha256'] == manifest_sha,
                f'{name} training split differs')
    golden = joint.legacy.predictions(directory / 'validation_predictions.jsonl.gz')
    measured = joint.study.validate_predictions(golden, val_rows, probabilities=False)
    require(matches_historical(measured, saved['validation'])
            and joint.prediction_digest(golden) == spec['prediction_ids_sha256'],
            f'{name} archived predictions or validation metrics differ')
    selection = {'epoch': spec['epoch'], 'validation': measured,
                 'prediction_ids_sha256': spec['prediction_ids_sha256']}
    return saved, selection, golden, paths


def load_model(name, saved, warm, device):
    architecture = 'E0' if name.startswith('E0FT') else 'E2'
    model = build_experiment(architecture, warm).to(device)
    grouped.initialize(model, architecture, warm)
    model.load_state_dict(saved['model_state'], strict=True)
    frozen = raw.state_digest(model.state_dict())
    require(frozen == raw.state_digest(saved['model_state'])
            and sum(param.numel() for param in model.parameters()) == saved['config']['total_parameters'],
            f'{name} loaded state or parameter count differs')
    return model.requires_grad_(False).eval()


@torch.no_grad()
def evaluate_models(models, batches, rows, output, phase, device):
    records = {name: [] for name in models}
    offset = 0
    for step, batch in enumerate(batches, 1):
        expected = rows[offset:offset + batch.num_graphs]
        actual = list(zip(batch.sample_id, batch.y.view(-1).tolist()))
        require(actual == [(row['sample_id'], row['label']) for row in expected], f'{phase} order or labels differ')
        batch = batch.to(device)
        for name, model in models.items():
            scores = model(batch)
            require(scores.shape == (len(expected), len(LABELS)) and bool(torch.isfinite(scores).all()),
                    f'{phase} {name} invalid class scores')
            if name.startswith('N0'):
                weights = model._last_cache['gate_weights']
                require(bool((weights == 0.5).all()), f'{name} fusion weights differ from fixed equal weights')
            records[name].extend({'sample_id': sid, 'label': label, 'prediction': pred}
                                 for (sid, label), pred in zip(actual, scores.argmax(-1).tolist()))
        offset += len(expected)
        if step == 1 or step % 200 == 0 or offset == len(rows):
            raw.progress(output, phase, samples=offset, total=len(rows))
    require(offset == len(rows), f'{phase} evaluation is incomplete')
    return records


def write_partition(output, rows, predictions):
    output.mkdir()
    metrics, hashes = {}, {}
    for name, records in predictions.items():
        metrics[name] = joint.study.validate_predictions(records, rows, probabilities=False)
        path = output / f'{name}_predictions.jsonl.gz'
        raw.save_predictions(path, records)
        hashes[name] = {'file_sha256': raw.sha(path), 'prediction_ids_sha256': joint.prediction_digest(records)}
    comparisons = {}
    for seed in SEEDS:
        before, after = f'E0FT{seed}', f'N0{seed}'
        ref, candidate = predictions[before], predictions[after]
        paired, changed = raw.compare(ref, candidate)
        subsets = {f'{a} -> {b}': [i for i, r in enumerate(ref)
                   if (r['label'], r['prediction']) == (LABELS.index(a), LABELS.index(b))]
                   for a, b in DIRECTIONS}
        key = f'{after}-{before}'
        comparisons[key] = {**paired, 'changed_predictions': len(changed),
                            'delta_macro_f1_pp': 100 * (metrics[after]['macro_f1'] - metrics[before]['macro_f1']),
                            'directions': direction_stats(ref, candidate, subsets)}
        raw.save_predictions(output / f'{key}_changed_samples.jsonl.gz', changed)
    values = {key: item['delta_macro_f1_pp'] for key, item in comparisons.items()}
    paired_stats = {'by_pair_pp': values, 'mean_pp': mean(values.values()), 'sample_sd_pp': stdev(values.values())}
    with (output / 'per_class.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', 'class', 'precision', 'recall', 'f1', 'support'])
        for model, result in metrics.items():
            for label, item in result['per_class'].items():
                writer.writerow([model, label, item['precision'], item['recall'], item['f1-score'], item['support']])
    raw.write(output / 'metrics.json', metrics)
    raw.write(output / 'prediction_hashes.json', hashes)
    raw.write(output / 'comparisons.json', comparisons)
    return {'metrics': metrics, 'comparisons': comparisons, 'paired_n0_minus_e0ft': paired_stats,
            'prediction_hashes': hashes}


def paper_tables(output, partitions, archive, probes):
    combined = {}
    for part, results in partitions.items():
        added = archive['partitions'][part]['metrics']
        require(all(item['samples'] == results['metrics']['E0FT42']['samples']
                    for item in added.values()), f'Archived paper metrics have a different {part} size')
        combined[part] = {**results['metrics'], **added}
    main_groups = {'E0-FT': [f'E0FT{s}' for s in SEEDS], 'N0': [f'N0{s}' for s in SEEDS],
                   '原 C': ['A'], '最终 B': [f'B{s}' for s in SEEDS]}

    def average(names, part, key):
        values = [100 * combined[part][name][key] for name in names]
        return f'{mean(values):.6f} ± {stdev(values):.6f}' if len(values) > 1 else f'{values[0]:.6f}'

    lines = ['# 论文主结果与消融', '', '同一固定划分含 13,446 张验证图和 13,448 张测试图。'
             '单模型值或三次算术均值 ± 样本标准差以百分数表示，配对差值使用百分点。', '',
             '## 主结果表', '',
             '| 模型 | 运行数 | 验证 Macro-F1 (%) | 测试 Accuracy (%) | 测试 Macro-F1 (%) | 测试 Weighted-F1 (%) |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for label, names in main_groups.items():
        values = [average(names, 'val', 'macro_f1')] + [average(names, 'test', key) for key in CLASSIFICATION]
        lines.append(f'| {label} | {len(names)} | ' + ' | '.join(values) + ' |')
    lines += ['', '| 主比较 | 验证平均差 ± SD（百分点） | 测试平均差 ± SD（百分点） |',
              '| --- | ---: | ---: |']
    for label, items in (('N0−E0-FT', [partitions[p]['paired_n0_minus_e0ft'] for p in ('val', 'test')]),
                         ('B−原 C', [archive['aggregate'][p]['B-A'] for p in ('val', 'test')])):
        lines.append(f'| {label} | ' + ' | '.join(f"{x['mean_pp']:+.6f} ± {x['sample_sd_pp']:.6f}" for x in items) + ' |')
    lines += ['', 'E0-FT 与 N0 分别沿用单编码器和双编码器的原训练配方及验证选模。'
              'B 的三次实验共用 seed42 原 C 冻结主干，分别训练新增读出。', '',
              '## 消融表', '',
              '| 对照 | 组别 | Seed | 验证 Macro-F1 (%) | 固定测试 Macro-F1 (%) |',
              '| --- | --- | --- | ---: | ---: |']
    for head, label in (('Linear', '线性头'), ('Nonlinear', '非线性头')):
        for model, weighting in (('Equal32', '固定等权'), ('D32', 'D32 动态门控')):
            score = probes['metrics'][f'{model}_{head}']['macro_f1']
            lines.append(f'| 同一对冻结投影与{label} | {weighting} | 42 | {100 * score:.6f} | 仅验证 |')
    for name, group in (('U42', '均匀聚合'), ('B42', 'Packet 均值查询'), ('C1_42', 'Flow 查询')):
        lines.append(f"| 报文转移聚合 | {group} | 42 | {average([name], 'val', 'macro_f1')} | "
                     f"{average([name], 'test', 'macro_f1')} |")
    for prefix, group in (('B', 'Packet 均值查询'), ('C1_', 'Flow 查询')):
        names = [f'{prefix}{seed}' for seed in SEEDS]
        lines.append(f"| 报文转移查询来源 | {group} | 42、43、44 | {average(names, 'val', 'macro_f1')} | "
                     f"{average(names, 'test', 'macro_f1')} |")
    lines += ['', 'Equal32 与 D32 使用同一冻结投影、同类分类头及相同训练预算。'
              'U、B、C1 的共同对照是同一冻结原 C。']
    (output / 'paper_tables.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def report(output, partitions, epochs):
    lines = ['# E0-FT 与 N0 固定测试', '',
             '六份权重按各自归档的验证最佳轮次锁定。先通过验证预测回放，再评估同一份固定测试清单。', '',
             '| Seed | E0-FT 轮次 | 验证 Macro-F1 | 测试 Macro-F1 | N0 轮次 | 验证 Macro-F1 | 测试 Macro-F1 | 测试 N0−E0-FT（百分点） |',
             '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for seed in SEEDS:
        e0, n0 = f'E0FT{seed}', f'N0{seed}'
        entry = [str(seed)]
        for name in (e0, n0):
            entry.append(str(epochs[name]))
            entry.extend(f"{partitions[p]['metrics'][name]['macro_f1']:.6%}" for p in ('val', 'test'))
        entry.append(f"{partitions['test']['comparisons'][f'{n0}-{e0}']['delta_macro_f1_pp']:+.6f}")
        lines.append('| ' + ' | '.join(entry) + ' |')
    lines += ['', '| 划分 | Seed | 错分方向 | E0-FT | N0 | 修复 / 退步（全部样本） |',
              '| --- | ---: | --- | ---: | ---: | --- |']
    for part, result in partitions.items():
        for seed in SEEDS:
            paired = result['comparisons'][f'N0{seed}-E0FT{seed}']
            counts = paired['counts']
            for a, b in (FOCUS, FOCUS[::-1]):
                value = paired['directions'][f'{a} -> {b}']
                lines.append(f"| {part} | {seed} | {a}→{b} | {value['original_errors']} | {value['total_errors']} | "
                             f"{counts['repaired']} / {counts['regressed']} |")
    lines += ['', '各模型八类 Precision、Recall、F1 和混淆矩阵见各划分的 CSV 与 JSON。'
              '三次均值、样本标准差和已归档模型的合并表见 [论文主结果与消融](paper_tables.md)。']
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run(repo, output, graph_root, code_commit, mode):
    settings = raw.read(CONFIG)
    config = raw.read(evaluation.CONFIG)
    require(set(settings['models']) == set(NAMES), 'Expected the six registered baseline checkpoints')
    split = repo / config['split_directory']
    inputs = {split / 'split.json': config['split_json_sha256']}
    evaluation.verify_inputs(inputs)
    device = runtime(grouped)
    torch.set_num_threads(2)
    data = grouped.FrozenGraphs(split, config['split_id'], data_root=graph_root)
    require(data.metadata['sample_manifest_sha256'] == settings['sample_manifest_sha256'], 'Sample manifest differs')
    inputs[data.directory / data.metadata['sample_manifest']] = settings['sample_manifest_sha256']
    val_rows = evaluation.evaluation_rows(data, 'val', config)
    val_batches = DataLoader(grouped.GraphRows(data, val_rows), batch_size=config['eval_batch_size'],
                             shuffle=False, num_workers=config['workers'])
    warm = grouped.Batch.from_data_list([data.load(row) for row in val_rows[:config['eval_batch_size']]]).to(device)
    n0_registry = raw.read(raw.ROOT / 'configs/fair_vector_n0_references_20261029.json')
    selected_seeds = SEEDS[:1] if mode == 'seed42-replay' else SEEDS
    models, before, receipts, epochs, val = {}, {}, {}, {}, {}
    provenance = {'code_commit': code_commit, 'archive_commit': settings['archive_commit'], 'mode': mode,
                  'graph_root': str(data.root), 'command': shlex.join([sys.executable, *sys.argv]),
                  'python': platform.python_version(), 'torch': str(torch.__version__),
                  'pyg': torch_geometric.__version__, 'device': torch.cuda.get_device_name(device),
                  'cuda': torch.version.cuda, 'split_id': config['split_id']}
    raw.write(output / 'config.json', {**settings, 'split_id': config['split_id'], 'mode': mode,
              'partition_sizes': config['partition_sizes'], 'eval_batch_size': config['eval_batch_size']})
    raw.write(output / 'provenance.json', provenance)
    for seed in selected_seeds:
        names = (f'E0FT{seed}', f'N0{seed}')
        selections, archived = {}, {}
        for name in names:
            spec = settings['models'][name]
            saved, selections[name], archived[name], paths = load_archive(
                repo, name, spec, val_rows, config, settings['sample_manifest_sha256'])
            inputs.update(paths)
            binding = saved['config']['binding']
            if name.startswith('N0'):
                ref = n0_registry[str(seed)]
                require(ref['directory'] == spec['directory']
                        and ref['metrics']['checkpoint_sha256']['best.pt'] == spec['sha256']['best.pt'],
                        f'{name} differs from the fair comparison N0 registry')
                sources = binding.get('model_sources', binding.get('sources'))
                evaluation.verify_inputs({raw.ROOT / p: sha for p, sha in sources.items()})
            models[name] = load_model(name, saved, warm, device)
            before[name] = raw.state_digest(models[name].state_dict())
            epochs[name] = selections[name]['epoch']
        observed = evaluate_models({name: models[name] for name in names}, val_batches, val_rows, output,
                                   f'validation_seed{seed}', device)
        for name in names:
            measured = joint.study.validate_predictions(observed[name], val_rows, probabilities=False)
            receipts[name] = evaluation.replay_receipt(measured, observed[name], selections[name], archived[name])
        passed = all(item['passed'] for item in receipts.values())
        raw.write(output / 'validation_replay.json', {'passed': passed, 'checkpoints': receipts,
                  'all_six_passed': passed and len(receipts) == len(NAMES)})
        require(passed, f'Seed{seed} validation replay failed')
        raw.progress(output, 'validation_pair_passed', seed=seed, samples=len(val_rows))
        val.update(observed)
    if mode == 'seed42-replay':
        for name, predictions in val.items():
            raw.save_predictions(output / f'{name}_validation_predictions.jsonl.gz', predictions)
    else:
        require(len(receipts) == len(NAMES), 'Complete all six validation replays before testing')
        paper_inputs = {repo / path: sha for path, sha in settings['paper_inputs'].items()}
        evaluation.verify_inputs(paper_inputs)
        inputs.update(paper_inputs)
        archive = raw.read(repo / PAPER_ARCHIVE)
        probes = raw.read(repo / FUSION_DIAGNOSIS)
        partitions = {'val': write_partition(output / 'val', val_rows, val)}
        test_rows = evaluation.evaluation_rows(data, 'test', config)
        test_batches = DataLoader(evaluation.TestGraphs(data, test_rows), batch_size=config['eval_batch_size'],
                                  shuffle=False, num_workers=config['workers'])
        test = evaluate_models(models, test_batches, test_rows, output, 'fixed_test', device)
        partitions['test'] = write_partition(output / 'test', test_rows, test)
        raw.write(output / 'summary.json', {'selected_epochs': epochs, 'partitions': partitions})
        report(output, partitions, epochs)
        paper_tables(output, partitions, archive, probes)
    after = {name: raw.state_digest(model.state_dict()) for name, model in models.items()}
    require(before == after, 'Evaluation changed model state')
    evaluation.verify_inputs(inputs)
    raw.write(output / 'provenance.json', {**provenance, 'input_sha256': {str(p): sha for p, sha in inputs.items()},
              'model_state_sha256_before': before, 'model_state_sha256_after': after})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--mode', choices=('seed42-replay', 'full'), default='full')
    parser.add_argument('--code-commit', required=True)
    args = parser.parse_args()
    require(re.fullmatch(r'[0-9a-f]{40}', args.code_commit), 'Expected a full code commit SHA')
    repo = args.repo.resolve()
    default = OUTPUT if args.mode == 'full' else OUTPUT + '_seed42_replay'
    output = (repo / (args.output or default)).resolve()
    require(output.parent == (repo / 'outputs').resolve(), 'Use a new directory directly under outputs')
    output.mkdir(parents=True)
    raw.write(output / 'status.json', {'status': 'running', 'mode': args.mode})
    try:
        run(repo, output, args.data_root, args.code_commit, args.mode)
    except BaseException as error:
        raw.write(output / 'status.json', {'status': 'failed', 'mode': args.mode, 'error': str(error)})
        raise
    raw.write(output / 'status.json', {'status': 'completed', 'mode': args.mode})
    print(json.dumps({'status': 'completed', 'mode': args.mode, 'output': str(output)}), flush=True)


if __name__ == '__main__':
    main()
