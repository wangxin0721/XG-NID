"""Replay the four locked seed42 checkpoints, then evaluate the fixed test split."""
import argparse
import copy
import csv
import json
from pathlib import Path
import platform
import re

import torch
from torch.utils.data import Dataset
import torch_geometric
from torch_geometric.loader import DataLoader

import run_joint_cross_layer_query as joint


raw, require, LABELS = joint.raw, joint.require, joint.LABELS
CONFIG = raw.ROOT / 'configs/joint_cross_layer_query_final_test_s42_v1.json'
DEFAULT_OUTPUT = 'outputs/joint_cross_layer_query_final_test_s42_v1'
COMPARISONS = (('C-A', 'A'), ('C-B', 'B'), ('C-B37', 'B37'))
FOCUS = ('WebBased', 'Recon', 'Spoofing', 'BruteForce')
NAMES = {'A': 'JCL-Multi', 'B': 'Mean 查询', 'C': 'Flow 查询', 'B37': 'Mean 查询（训练后最佳）'}


class TestGraphs(Dataset):
    def __init__(self, data, rows):
        require(all(row['partition'] == 'test' for row in rows), 'Expected only fixed test rows')
        self.data, self.rows = data, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.data.load(self.rows[index])


def verify_inputs(inputs):
    missing = [str(path) for path in inputs if not path.is_file()]
    require(not missing, 'Missing archived inputs: ' + ', '.join(missing))
    for path, digest in inputs.items():
        require(raw.sha(path) == digest, f'SHA256 mismatch: {path}')


def archive_inputs(source, split, config):
    paths = {split / 'split.json': config['split_json_sha256']}
    paths.update({source / item['path']: item['sha256'] for item in config['checkpoints'].values()})
    paths.update({source / arm / 'metrics.json': digest for arm, digest in config['metrics_sha256'].items()})
    verify_inputs(paths)
    metrics = {arm: raw.read(source / arm / 'metrics.json') for arm in config['metrics_sha256']}
    predictions = {source / arm / 'validation_predictions.jsonl.gz': value['predictions_sha256']
                   for arm, value in metrics.items()}
    verify_inputs(predictions)
    return metrics, {**paths, **predictions}


def load_checkpoint_state(source, config, metrics, protocol, name):
    spec = config['checkpoints'][name]
    arm = spec['arm']
    selected = metrics[arm]['selected'][spec['selection']]
    saved = torch.load(source / spec['path'], map_location='cpu', weights_only=False)
    metadata = saved['config']
    validation = joint.study.classification_metrics(saved['validation'])
    require(metadata['arm'] == arm and metadata['query_source'] == joint.ARMS[arm]
            and metadata['labels'] == LABELS and metadata['protocol'] == protocol
            and protocol['split_id'] == config['split_id'] and protocol['seed'] == config['seed'],
            f'{name} checkpoint structure, protocol or labels differ')
    require(saved['epoch'] == selected['epoch'] == spec['epoch']
            and saved['prediction_ids_sha256'] == selected['prediction_ids_sha256']
            and selected['checkpoint_sha256'] == spec['sha256']
            and validation == selected['validation'] and validation['macro_f1'] == spec['macro_f1'],
            f'{name} checkpoint validation selection differs')
    return saved['model_state']


def load_models(source, config, metrics, device):
    protocol = raw.read(joint.CONFIG)
    payloads = {name: load_checkpoint_state(source, config, metrics, protocol, name)
                for name in config['checkpoints']}
    models = joint.make_models(payloads['A'], torch.device('cpu'), config['seed'])
    models['B37'] = copy.deepcopy(models['B'])
    for name, model in models.items():
        model.load_state_dict(payloads[name], strict=True)
        arm = config['checkpoints'][name]['arm']
        require(joint.legacy.frozen_hash(model) == metrics[arm]['frozen_state_sha256'] ==
                protocol['source_frozen_state_sha256'], f'{name} frozen state differs')
        if name == 'B':
            require(not bool(model.packet_pool.query.weight.ne(0).any()), 'B epoch-zero query is not zero')
        model.requires_grad_(False).to(device).eval()
    return models


def evaluation_rows(data, partition, config):
    require(partition in ('val', 'test'), 'Expected validation or test partition')
    rows = [row for row in data.rows if row['partition'] == partition]
    expected = config['partition_sizes'][partition]
    require(len(rows) == data.metadata['partition_sizes'][partition] == expected
            and len({row['sample_id'] for row in rows}) == expected,
            f'Wrong {partition} sample count or duplicate IDs')
    return rows


def evaluate(model, batches, rows, device, output, phase):
    before = raw.state_digest(model.state_dict())
    metrics, records = raw.evaluate(model, batches, rows, device, output, phase, export=True)
    records = [{'sample_id': sid, 'label': label, 'prediction': pred}
               for sid, label, pred in joint.legacy.identities(records)]
    require(joint.study.validate_predictions(records, rows, probabilities=False) == metrics
            and raw.state_digest(model.state_dict()) == before, f'{phase} metrics or model state differ')
    return metrics, records


def replay_receipt(metrics, records, selected, archived):
    digest = joint.prediction_digest(records)
    counts, differences = raw.compare(archived, records) if archived is not None else (None, [])
    passed = (metrics == selected['validation'] and digest == selected['prediction_ids_sha256']
              and not differences)
    return {'passed': passed, 'samples': len(records), 'metrics_match': metrics == selected['validation'],
            'prediction_ids_sha256': digest, 'expected_prediction_ids_sha256': selected['prediction_ids_sha256'],
            'archive_comparison': counts, 'first_differences': differences[:10]}


def evaluate_partitions(models, data, config, source, archive, device, output):
    val_rows = evaluation_rows(data, 'val', config)
    val_loader = joint.legacy.loader(data, val_rows, 'val', config)
    validation, test, receipts = {}, {}, {}
    (output / 'val').mkdir()
    for name, model in models.items():
        spec = config['checkpoints'][name]
        selected = archive[spec['arm']]['selected'][spec['selection']]
        archived = None
        if spec['selection'] == 'best':
            archived = joint.legacy.predictions(source / name / 'validation_predictions.jsonl.gz')
            require(joint.study.validate_predictions(archived, val_rows, probabilities=False) == selected['validation']
                    and joint.prediction_digest(archived) == selected['prediction_ids_sha256'],
                    f'{name} archived validation predictions differ from selection')
        validation[name] = evaluate(model, val_loader, val_rows, device, output, f'val_{name}')
        metrics, records = validation[name]
        receipts[name] = replay_receipt(metrics, records, selected, archived)
        raw.write(output / 'validation_replay.json', {'passed': False, 'checkpoints': receipts})
        require(receipts[name]['passed'], f'{name} validation replay failed. See validation_replay.json')
        raw.save_predictions(output / 'val' / f'{name}_predictions.jsonl.gz', records)
    identical = joint.legacy.identities(validation['A'][1]) == joint.legacy.identities(validation['B'][1])
    raw.write(output / 'validation_replay.json', {'passed': identical, 'A_B_predictions_identical': identical,
                                                 'checkpoints': receipts})
    require(identical, 'A and B epoch-zero validation predictions differ')

    test_rows = evaluation_rows(data, 'test', config)
    test_loader = DataLoader(TestGraphs(data, test_rows), batch_size=config['eval_batch_size'],
                             shuffle=False, num_workers=config['workers'],
                             generator=torch.Generator().manual_seed(config['seed'] + 200))
    (output / 'test').mkdir()
    for name, model in models.items():
        test[name] = evaluate(model, test_loader, test_rows, device, output, f'test_{name}')
        raw.save_predictions(output / 'test' / f'{name}_predictions.jsonl.gz', test[name][1])
    return {'val': validation, 'test': test}


def write_partition(output, results):
    with (output / 'per_class.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['checkpoint', 'class', 'precision', 'recall', 'f1', 'support'])
        for name, (metrics, _) in results.items():
            for label in LABELS:
                row = metrics['per_class'][label]
                writer.writerow([name, label, row['precision'], row['recall'], row['f1-score'], row['support']])

    c_metrics, c_rows = results['C']
    paired = {}
    for name, control in COMPARISONS:
        before_metrics, before_rows = results[control]
        evidence = joint.comparison_evidence(before_rows, c_rows, before_metrics, c_metrics, output, name)
        by_class = {}
        for label in LABELS:
            index = LABELS.index(label)
            a, b = before_metrics['per_class'][label], c_metrics['per_class'][label]
            by_class[label] = {'recall_delta_pp': 100 * (b['recall'] - a['recall']),
                               'precision_delta_pp': 100 * (b['precision'] - a['precision']),
                               'f1_delta_pp': 100 * (b['f1-score'] - a['f1-score']),
                               'correct_delta': c_metrics['confusion_matrix'][index][index] -
                               before_metrics['confusion_matrix'][index][index]}
        require(sum(row['correct_delta'] for row in by_class.values()) == evidence['net_correct'],
                f'{name} correct-count difference disagrees with matrix')
        paired[name] = {**evidence, 'macro_f1_delta_pp': 100 * (c_metrics['macro_f1'] - before_metrics['macro_f1']),
                        'per_class': by_class}
    counts, changed = raw.compare(results['A'][1], results['B'][1])
    raw.save_predictions(output / 'B-A_changed_samples.jsonl.gz', changed)
    raw.write(output / 'metrics.json', {name: metrics for name, (metrics, _) in results.items()})
    return {'comparisons': paired, 'A_B_prediction_check': {'identical': not changed, 'changed_samples': len(changed),
                                                          **counts}}


def paper_tables(config, metrics, analysis):
    lines = ['# Seed42 固定划分论文表', '',
             'A/B/C 采用验证集选定权重。B37 为训练后最佳的等参 Mean 查询补充对照。', '',
             '## 主结果', '',
             '| 组别 | 轮次 | 验证 Macro-F1 | 测试 Accuracy | 测试 Macro-F1 | 测试 Weighted-F1 |',
             '|---|---:|---:|---:|---:|---:|']
    for name in ('A', 'B', 'C'):
        val, test = metrics['val'][name], metrics['test'][name]
        epoch = config['checkpoints'][name]['epoch']
        lines.append(f"| {name} {NAMES[name]} | {epoch} | {val['macro_f1']:.6%} | {test['accuracy']:.6%} | "
                     f"{test['macro_f1']:.6%} | {test['weighted_f1']:.6%} |")
    lines += ['', '## 跨分支查询消融', '',
              '| 组别 | 轮次 | 验证 Macro-F1 | 测试 Macro-F1 |',
              '|---|---:|---:|---:|']
    for name in ('B', 'C', 'B37'):
        epoch = config['checkpoints'][name]['epoch']
        lines.append(f"| {name} {NAMES[name]} | {epoch} | {metrics['val'][name]['macro_f1']:.6%} | "
                     f"{metrics['test'][name]['macro_f1']:.6%} |")
    lines += ['', '| 比较 | 验证 Δ Macro-F1（百分点） | 测试 Δ Macro-F1（百分点） |',
              '|---|---:|---:|']
    for name, _ in COMPARISONS:
        val, test = (analysis[part]['comparisons'][name]['macro_f1_delta_pp'] for part in ('val', 'test'))
        lines.append(f"| {name} | {val:+.6f} | {test:+.6f} |")
    lines += ['', '验证 13,446 张图，测试 13,448 张图。所有结果来自 seed42 单次配对实验。']
    return lines


def write_report(output, config, metrics, analysis):
    tables = paper_tables(config, metrics, analysis)
    (output / 'paper_tables.md').write_text('\n'.join(tables) + '\n', encoding='utf-8')
    lines = ['# Seed42 四项合并固定测试集评估', '',
             '四份检查点均通过验证回放。A、B 的第 0 轮继承历史 JCL-Multi 第 48 轮，'
             'C 的第 48 轮指四项合并训练阶段。', '',
             '| 组别 | 轮次 | 验证 Accuracy | 验证 Macro-F1 | 验证 Weighted-F1 | 测试 Accuracy | 测试 Macro-F1 | 测试 Weighted-F1 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, item in config['checkpoints'].items():
        scores = [f'{metrics[part][name][key]:.6%}' for part in ('val', 'test')
                  for key in ('accuracy', 'macro_f1', 'weighted_f1')]
        lines.append(f"| {name} {NAMES[name]} | {item['epoch']} | " + ' | '.join(scores) + ' |')
    ab = analysis['test']['A_B_prediction_check']
    lines += ['', f"A/B 测试预测不同的样本数为 {ab['changed_samples']} / {config['partition_sizes']['test']}。", '',
              '| 划分 | 比较 | Δ Macro-F1（百分点） | 修复 / 退步 / 净正确 | WebBased → Recon | Recon → WebBased |',
              '|---|---|---:|---:|---:|---:|']
    for part in ('val', 'test'):
        for name, _ in COMPARISONS:
            item = analysis[part]['comparisons'][name]
            counts = item['counts']
            directions = [f"{d['before_errors']} → {d['after_errors']}" for d in item['directions'].values()]
            lines.append(f"| {part} | {name} | {item['macro_f1_delta_pp']:+.6f} | {counts['repaired']} / "
                         f"{counts['regressed']} / {item['net_correct']:+d} | " + ' | '.join(directions) + ' |')
    lines.append('')
    for name, _ in COMPARISONS:
        val, test = (analysis[part]['comparisons'][name]['macro_f1_delta_pp'] for part in ('val', 'test'))
        if val > 0:
            outcome = '在测试集上同方向复现' if test > 0 else '在测试集上未复现'
            lines.append(f'{name} 的验证 Macro-F1 增量{outcome}。')
    lines += ['', '| 划分 | 组别 | WebBased 召回 | Recon 召回 | Spoofing 召回 | BruteForce 召回 | BruteForce F1 |',
              '|---|---|---:|---:|---:|---:|---:|']
    for part in ('val', 'test'):
        for name in config['checkpoints']:
            by_class = metrics[part][name]['per_class']
            recalls = [f"{by_class[label]['recall']:.6%}" for label in FOCUS]
            lines.append(f"| {part} | {name} | " + ' | '.join(recalls) + f" | {by_class['BruteForce']['f1-score']:.6%} |")
    lines += ['', '验证和测试分别报告。结论限于本次 seed42 配对实验。', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def run(repo, output, graph_root, code_commit):
    config = raw.read(CONFIG)
    source = repo / config['source_directory']
    split = repo / config['split_directory']
    archive, inputs = archive_inputs(source, split, config)
    require(torch.cuda.is_available() and torch.__version__.split('+')[0] == '2.3.1'
            and torch_geometric.__version__ == '2.5.3', 'Expected CUDA, Torch 2.3.1 and PyG 2.5.3')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    data = raw.FrozenGraphs(split, config['split_id'], data_root=graph_root)
    manifest = data.directory / data.metadata['sample_manifest']
    inputs[manifest] = data.metadata['sample_manifest_sha256']
    raw.write(output / 'config.json', config)
    raw.write(output / 'provenance.json', {'archive_commit': config['archive_commit'], 'code_commit': code_commit,
              'config_sha256': raw.sha(CONFIG), 'split_id': config['split_id'], 'graph_root': str(data.root),
              'source_directory': str(source), 'input_sha256': {str(path): digest for path, digest in inputs.items()},
              'runner_sha256': raw.sha(Path(__file__)), 'python': platform.python_version(),
              'torch': torch.__version__, 'pyg': torch_geometric.__version__, 'cuda': torch.version.cuda,
              'device': str(device), 'device_name': torch.cuda.get_device_name(device), 'tf32': False})
    models = load_models(source, config, archive, device)
    results = evaluate_partitions(models, data, config, source, archive, device, output)
    verify_inputs(inputs)
    metrics = {part: {name: value[0] for name, value in arms.items()} for part, arms in results.items()}
    analysis = {part: write_partition(output / part, arms) for part, arms in results.items()}
    write_report(output, config, metrics, analysis)
    raw.write(output / 'summary.json', {'status': 'completed', 'seed': config['seed'], 'test_evaluated': True,
                                       'checkpoints': config['checkpoints'], 'metrics': metrics, 'analysis': analysis})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument('--data-root', type=Path, help='Override the graph root recorded in split.json')
    parser.add_argument('--code-commit', required=True, help='Git commit of the evaluated code bundle')
    args = parser.parse_args()
    require(re.fullmatch(r'[0-9a-f]{40}', args.code_commit), 'Expected a full code commit SHA')
    repo = args.repo.resolve()
    outputs = (repo / 'outputs').resolve()
    output = (repo / args.output).resolve()
    require(output.parent == outputs and output != (repo / raw.read(CONFIG)['source_directory']).resolve(),
            'Use a new directory directly under outputs')
    output.mkdir(parents=True)
    print(f'Evaluation output: {output}', flush=True)
    raw.write(output / 'status.json', {'status': 'running'})
    try:
        run(repo, output, args.data_root, args.code_commit)
    except BaseException as error:
        raw.write(output / 'status.json', {'status': 'failed', 'error': str(error)})
        raise
    raw.write(output / 'status.json', {'status': 'completed'})
    print(json.dumps({'status': 'completed', 'report': str(output / 'report.md')}), flush=True)


if __name__ == '__main__':
    main()
