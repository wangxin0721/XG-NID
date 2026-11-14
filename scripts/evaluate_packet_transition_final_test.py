"""Evaluate the locked A, B, C1 and U transition checkpoints."""
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

import run_packet_transition_attention_diagnosis as attention
import run_packet_transition_readout as transition


evaluation, joint, raw, require, LABELS = (transition.evaluation, transition.joint, transition.raw,
                                          transition.require, transition.LABELS)
CONFIG = raw.ROOT / 'configs/packet_transition_final_test_v1.json'
OUTPUT = 'outputs/packet_transition_final_test_v1'
SEEDS = (42, 43, 44)
PAIRS = tuple((f'B{seed}', 'A') for seed in SEEDS) + tuple(
    (f'C1_{seed}', f'B{seed}') for seed in SEEDS) + (('U42', 'A'), ('B42', 'U42'))
FOCUS = ('WebBased', 'Recon', 'BruteForce')


def checked_paths(directory, settings):
    paths = {directory / name: digest for name, digest in settings['sha256'].items()}
    evaluation.verify_inputs(paths)
    return paths


def readout_selection(name, saved, arm, registration, receipt, exported, rows):
    item = saved['best'][arm]
    predictions = transition.prediction_records(exported, [row['predictions'][arm] for row in exported])
    metrics = joint.study.validate_predictions(predictions, rows, probabilities=False)
    require(item['epoch'] == registration['epoch'] and item['validation'] == metrics
            and item['prediction_ids_sha256'] == registration['prediction_ids_sha256']
            == joint.prediction_digest(predictions), f'{name} archive predictions or checkpoint selection differs')
    if receipt is not None:
        require(receipt['passed'] and receipt['metrics_match'] and receipt['samples'] == len(rows)
                and receipt['prediction_ids_sha256'] == receipt['expected_prediction_ids_sha256']
                == item['prediction_ids_sha256'], f'{name} archived validation replay differs')
    return item, predictions


def load_readouts(repo, settings, config, rows, base, device):
    input_config42 = raw.read(attention.CONFIG)
    saved42, refs42, lengths, _, inputs = attention.archived_inputs(repo, input_config42, config, rows, base)
    frozen_sha = settings['backbone_state_sha256']
    require(saved42['frozen_c_sha256'] == frozen_sha, 'Seed42 archived backbone differs')
    references = {'A': refs42['A']}
    selections = {'A': base}
    heads = {}

    def add(name, arm, seed, item, reference):
        head = transition.make_heads({**config, 'readout_seed': seed}, device, (arm,))[arm]
        head.load_state_dict(item['state'], strict=True)
        heads[name] = (arm, head.requires_grad_(False).eval())
        selections[name] = {key: item[key] for key in ('epoch', 'validation', 'prediction_ids_sha256')}
        references[name] = reference

    for arm, name in (('B', 'B42'), ('C1', 'C1_42')):
        add(name, arm, 42, saved42['best'][arm], refs42[arm])
    paired = settings['paired_archive']
    for seed in (43, 44):
        spec = paired['seeds'][str(seed)]
        directory = repo / paired['directory'] / f'seed{seed}'
        inputs.update(checked_paths(directory, spec))
        path = directory / 'transition_readouts.pt'
        saved = torch.load(path, map_location='cpu', weights_only=True)
        require(saved['config'] == {**config, 'readout_seed': seed} and saved['query_sources'] == transition.SOURCES
                and saved['frozen_c_sha256'] == frozen_sha, f'Seed{seed} checkpoint protocol differs')
        document = raw.read(directory / 'validation_replay.json')
        require(document['checkpoint_sha256'] == spec['sha256'][path.name], f'Seed{seed} replay checkpoint differs')
        exported = joint.legacy.predictions(directory / 'best/validation_predictions.jsonl.gz')
        require([row['valid_pairs'] for row in exported] == lengths, f'Seed{seed} validation edge counts differ')
        for arm, name in (('B', f'B{seed}'), ('C1', f'C1_{seed}')):
            item, reference = readout_selection(name, saved, arm, spec['selections'][arm], document['best_raw'][arm],
                                                exported, rows)
            add(name, arm, seed, item, reference)
    uniform = settings['seed42_attention_archive']
    directory = repo / uniform['directory']
    inputs.update(checked_paths(directory, uniform))
    saved = torch.load(directory / 'uniform_readout.pt', map_location='cpu', weights_only=True)
    require(saved['config'] == config and saved['query_source'] == 'uniform'
            and saved['frozen_c_sha256'] == frozen_sha, 'U42 checkpoint protocol differs')
    exported = joint.legacy.predictions(directory / 'validation_predictions.jsonl.gz')
    require([row['valid_pairs'] for row in exported] == lengths, 'U42 validation edge counts differ')
    item, reference = readout_selection('U42', saved, 'U', uniform,
                                        raw.read(directory / 'validation_replay.json')['U_selected_raw'], exported, rows)
    add('U42', 'U', 42, item, reference)
    return heads, selections, references, lengths, inputs


@torch.no_grad()
def evaluate_partition(model, heads, batches, rows, device, output, phase):
    predictions = {name: [] for name in ('A', *heads)}
    lengths = []
    for step, batch in enumerate(batches, 1):
        start = len(lengths)
        end = start + batch.num_graphs
        require(list(zip(batch.sample_id, batch.y.view(-1).tolist())) ==
                [(row['sample_id'], row['label']) for row in rows[start:end]], f'{phase} sample IDs or labels differ')
        features, _ = transition.encode(model, batch, device)
        empty = features.lengths == 0
        lengths.extend(features.lengths.tolist())
        baseline = transition.baseline(model, features)
        require(bool(torch.isfinite(baseline).all()), f'{phase} nonfinite baseline scores')
        pred_base = baseline.argmax(-1)
        predictions['A'].extend(pred_base.tolist())
        for name, (arm, head) in heads.items():
            scores, residual, _ = transition.transition_forward(model, head, arm, features)
            pred = scores.argmax(-1)
            require(bool(torch.isfinite(scores).all()) and not bool(residual[empty].any())
                    and torch.equal(pred[empty], pred_base[empty]), f'{phase} invalid scores or empty-edge output for {name}')
            predictions[name].extend(pred.tolist())
        if step == 1 or step % 200 == 0 or end == len(rows):
            raw.progress(output, phase, samples=end, total=len(rows))
    return {name: transition.prediction_records(rows, values) for name, values in predictions.items()}, lengths


def write_partition(output, rows, predictions, lengths):
    output.mkdir()
    metrics, comparisons = attention.comparison_outputs(output, rows, predictions, lengths, PAIRS)
    hashes = {}
    for name, records in predictions.items():
        path = output / f'{name}_predictions.jsonl.gz'
        raw.save_predictions(path, ({**row, 'valid_pairs': length} for row, length in zip(records, lengths)))
        hashes[name] = {'file_sha256': raw.sha(path), 'prediction_ids_sha256': joint.prediction_digest(records)}
    with (output / 'per_class.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', 'class', 'precision', 'recall', 'f1', 'support'])
        for name, result in metrics.items():
            for label, item in result['per_class'].items():
                writer.writerow([name, label, item['precision'], item['recall'], item['f1-score'], item['support']])
    raw.write(output / 'metrics.json', metrics)
    raw.write(output / 'comparisons.json', comparisons)
    raw.write(output / 'prediction_hashes.json', hashes)
    return {'metrics': metrics, 'comparisons': comparisons, 'prediction_hashes': hashes,
            'graphs_without_valid_pairs': lengths.count(0)}


def aggregate(partitions):
    results = {}
    for part, result in partitions.items():
        results[part] = {}
        for name, pairs in (('B-A', [(f'B{s}', 'A') for s in SEEDS]),
                            ('C1-B', [(f'C1_{s}', f'B{s}') for s in SEEDS])):
            values = {seed: result['comparisons'][f'{after}-{before}']['delta_macro_f1_pp']
                      for seed, (after, before) in zip(SEEDS, pairs)}
            results[part][name] = {'by_seed_pp': values, 'mean_pp': mean(values.values()),
                                   'sample_sd_pp': stdev(values.values()),
                                   'positive': sum(value > 0 for value in values.values())}
    return results


def write_report(output, partitions, stats, selections):
    lines = ['# 报文转移读出固定测试评估', '',
             'A 为原 C 第 48 轮。主比较为三个 B 检查点对 A，查询来源消融为同 seed 的 C1 对 B，'
             '均匀聚合消融使用 seed42 的 U。全部读出使用各自锁定的验证最佳轮次。', '',
             '| 组别 | 轮次 | 验证 Accuracy | 验证 Macro-F1 | 验证 Weighted-F1 | 测试 Accuracy | 测试 Macro-F1 | 测试 Weighted-F1 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    names = ('A', *(f'B{s}' for s in SEEDS), *(f'C1_{s}' for s in SEEDS), 'U42')
    for name in names:
        values = [partitions[part]['metrics'][name][key] for part in ('val', 'test')
                  for key in ('accuracy', 'macro_f1', 'weighted_f1')]
        lines.append(f"| {name} | {selections[name]['epoch']} | " +
                     ' | '.join(f'{value:.6%}' for value in values) + ' |')
    lines += ['', '| 划分 | 配对 | 三 seed 平均差（百分点） | 样本标准差（百分点） | 正差次数 |',
              '| --- | --- | ---: | ---: | ---: |']
    for part, comparisons in stats.items():
        for name, item in comparisons.items():
            lines.append(f"| {part} | {name} | {item['mean_pp']:+.6f} | {item['sample_sd_pp']:.6f} | {item['positive']} / 3 |")
    lines += ['', '| 比较 | 验证 Macro-F1 差值（百分点） | 测试 Macro-F1 差值（百分点） | 测试修复 / 退步 | 测试净正确 |',
              '| --- | ---: | ---: | --- | ---: |']
    for name, item in partitions['test']['comparisons'].items():
        val_pp = partitions['val']['comparisons'][name]['delta_macro_f1_pp']
        lines.append(f"| {name} | {val_pp:+.6f} | {item['delta_macro_f1_pp']:+.6f} | "
                     f"{item['counts']['repaired']} / {item['counts']['regressed']} | {item['net_correct']:+d} |")
    lines += ['', '| 划分 | 比较 | 错分方向 | 对照错分 → 当前错分 | 原错分判对 | 原判对样本新增错分 |',
              '| --- | --- | --- | --- | ---: | ---: |']
    for part, result in partitions.items():
        for name, comparison in result['comparisons'].items():
            for direction, item in comparison['directions'].items():
                lines.append(f"| {part} | {name} | {direction.replace(' -> ', '→')} | {item['original_errors']} → "
                             f"{item['total_errors']} | {item['repaired']} | {item['new_errors_from_original_correct']} |")
    lines += ['', '| 划分 | 组别 | 类别 | Precision | Recall | F1 | 样本数 |',
              '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for part, result in partitions.items():
        for name in names:
            for label in FOCUS:
                item = result['metrics'][name]['per_class'][label]
                lines.append(f"| {part} | {name} | {label} | {item['precision']:.6%} | {item['recall']:.6%} | "
                             f"{item['f1-score']:.6%} | {item['support']} |")
    lines += ['', '两份 per_class.csv 收录八类指标，metrics.json 收录完整混淆矩阵。'
              '三个 seed 共用冻结主干和数据划分，差值反映新增读出在不同初始化及采样顺序下的结果。',
              '本次评估已使用固定测试集。']
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def model_digests(model, heads):
    return {'A': raw.state_digest(model.state_dict()),
            **{name: raw.state_digest(head.state_dict()) for name, (_, head) in heads.items()}}


def run(repo, output, graph_root, code_commit):
    settings = raw.read(CONFIG)
    config = transition.base_config()
    # The archived heads bind the train/val protocol. Keep the new test size
    # outside that protocol when verifying their metadata.
    model, data, rows, device, selected_base, source_inputs = transition.load_source(repo, config, graph_root)
    require(data.metadata['sample_manifest_sha256'] == settings['sample_manifest_sha256'],
            'Fixed sample manifest differs')
    heads, selections, references, archived_lengths, archive_inputs = load_readouts(
        repo, settings, config, rows['val'], selected_base, device)
    before = model_digests(model, heads)
    require(before['A'] == settings['backbone_state_sha256'], 'Frozen original C state differs')
    inputs = {**source_inputs, **archive_inputs}
    raw.write(output / 'config.json', {'registration': settings, 'backbone_protocol': config,
              'seed42_transition_registration': raw.read(attention.CONFIG),
              'selected_checkpoints': {name: item['epoch'] for name, item in selections.items()},
              'comparisons': PAIRS})
    provenance = {'code_commit': code_commit, 'graph_root': str(data.root), 'python': platform.python_version(),
                  'torch': str(torch.__version__), 'pyg': torch_geometric.__version__, 'cuda': torch.version.cuda,
                  'device': torch.cuda.get_device_name(device),
                  'command': shlex.join([sys.executable, *sys.argv]), 'model_state_sha256_before': before}
    raw.write(output / 'provenance.json', {**provenance, 'input_sha256': {str(p): sha for p, sha in inputs.items()}})
    val_loader = joint.legacy.loader(data, rows['val'], 'val', config)
    val, val_lengths = evaluate_partition(model, heads, val_loader, rows['val'], device, output, 'validation')
    val_receipts = {name: evaluation.replay_receipt(
        joint.study.validate_predictions(records, rows['val'], probabilities=False), records, selections[name],
        references[name]) for name, records in val.items()}
    passed = all(item['passed'] for item in val_receipts.values()) and val_lengths == archived_lengths
    raw.write(output / 'validation_replay.json', {'passed': passed, 'checkpoints': val_receipts,
              'valid_pair_counts_match_archive': val_lengths == archived_lengths})
    require(passed and model_digests(model, heads) == before, 'Validation replay or model state differs')
    partitions = {'val': write_partition(output / 'val', rows['val'], val, val_lengths)}
    raw.progress(output, 'validation_passed', samples=len(rows['val']))

    baseline_test = settings['baseline_test_archive']
    directory = repo / baseline_test['directory']
    inputs.update(checked_paths(directory, baseline_test))
    gold = joint.legacy.predictions(directory / 'C_predictions.jsonl.gz')
    gold_metrics = raw.read(directory / 'metrics.json')['C']
    test_config = {**config, 'partition_sizes': {**config['partition_sizes'], 'test': settings['test_samples']}}
    test_rows = evaluation.evaluation_rows(data, 'test', test_config)
    require(joint.study.validate_predictions(gold, test_rows, probabilities=False) == gold_metrics
            and joint.prediction_digest(gold) == baseline_test['prediction_ids_sha256']
            and gold_metrics['macro_f1'] == baseline_test['macro_f1'], 'Archived baseline test predictions differ')
    test_loader = DataLoader(evaluation.TestGraphs(data, test_rows), batch_size=config['eval_batch_size'],
                             shuffle=False, num_workers=config['workers'],
                             generator=torch.Generator().manual_seed(config['seed'] + 200))
    test, test_lengths = evaluate_partition(model, heads, test_loader, test_rows, device, output, 'fixed_test')
    test_receipt = evaluation.replay_receipt(joint.study.validate_predictions(test['A'], test_rows, probabilities=False),
                                            test['A'], {'validation': gold_metrics,
                                                        'prediction_ids_sha256': baseline_test['prediction_ids_sha256']},
                                            gold)
    raw.write(output / 'baseline_test_replay.json', test_receipt)
    require(test_receipt['passed'], 'Baseline test replay differs from archive')
    after = model_digests(model, heads)
    require(after == before and all(not p.requires_grad and p.grad is None
                                    for module in (model, *(head for _, head in heads.values()))
                                    for p in module.parameters()), 'Model parameters changed during evaluation')
    evaluation.verify_inputs(inputs)
    partitions['test'] = write_partition(output / 'test', test_rows, test, test_lengths)
    stats = aggregate(partitions)
    raw.write(output / 'summary.json', {'aggregate': stats, 'partitions': partitions,
              'selected_epochs': {name: item['epoch'] for name, item in selections.items()}})
    raw.write(output / 'provenance.json', {**provenance, 'input_sha256': {str(p): sha for p, sha in inputs.items()},
              'model_state_sha256_after': after, 'frozen_states_verified': True})
    write_report(output, partitions, stats, selections)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, default=Path(OUTPUT))
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--code-commit', required=True)
    args = parser.parse_args()
    require(re.fullmatch(r'[0-9a-f]{40}', args.code_commit), 'Expected a full code commit SHA')
    repo = args.repo.resolve()
    output = (repo / args.output).resolve()
    require(output.parent == (repo / 'outputs').resolve(), 'Use a new directory directly under outputs')
    output.mkdir(parents=True)
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
