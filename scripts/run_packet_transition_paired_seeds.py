"""Repeat B/C1 transition-readout training on the frozen seed42 backbone."""
import argparse
import csv
import json
from pathlib import Path
import platform
import re
from statistics import mean, stdev

import torch
import torch_geometric

import run_packet_transition_attention_diagnosis as attention
import run_packet_transition_readout as original


raw, joint, evaluation, require, LABELS = (original.raw, original.joint, original.evaluation,
                                          original.require, original.LABELS)
CONFIG = raw.ROOT / 'configs/packet_transition_paired_seeds43_44_v1.json'
OUTPUT = 'outputs/packet_transition_paired_seeds43_44_v1'
PHASES = ('best', 'last')
ARMS = ('B', 'C1')
FOCUS = ('WebBased', 'Recon', 'BruteForce')
DIRECTIONS = ('Recon -> WebBased', 'WebBased -> Recon')


def initial_hashes(heads):
    digests = {arm: raw.state_digest(head.state_dict()) for arm, head in heads.items()}
    require(len(set(digests.values())) == 1 and all(not bool(head.output.weight.any()) for head in heads.values()),
            'Paired readouts must start identically with zero output weights')
    return digests


def check_zero(model, heads, features):
    scores = original.baseline(model, features)
    for arm, head in heads.items():
        replay, residual, _ = original.transition_forward(model, head, arm, features)
        require(torch.equal(replay, scores) and not bool(residual.any()), f'{arm} raw epoch zero differs')


def verify_history(history, config, selections, baseline, updates):
    require([row['epoch'] for row in history] == list(range(config['epochs'] + 1)),
            'Training ledger does not cover epoch zero through the fixed budget')
    require(history[0]['order_sha256'] is None and history[0]['updates_per_arm'] == 0
            and history[0]['samples_per_arm'] == 0, 'Epoch zero must have no training updates')
    orders = []
    for row in history[1:]:
        digest = row['order_sha256']
        require(row['updates_per_arm'] == updates and row['samples_per_arm'] == config['training_samples']
                and re.fullmatch(r'[0-9a-f]{64}', digest), 'Training order or per-arm budget differs')
        orders.append(digest)
    for arm in ARMS:
        best = max(history, key=lambda row: row['validation'][arm]['macro_f1'])
        for phase, row in (('best', best), ('last', history[-1])):
            item = selections[phase][arm]
            require(item['epoch'] == row['epoch'] and item['validation'] == row['validation'][arm],
                    f'Selected {phase} {arm} differs from the ledger')
        require(history[0]['validation'][arm] == baseline, f'{arm} epoch-zero metrics differ')
    return orders


def replay_last(model, checkpoint, val, rows, baseline, config, device):
    heads = original.make_heads(config, device, ARMS)
    for arm, head in heads.items():
        head.load_state_dict(checkpoint['last'][arm]['state'], strict=True)
        head.eval()
    records, metrics = original.measure(model, heads, val, rows, config, device)
    receipts = {arm: evaluation.replay_receipt(metrics[arm], records[arm], checkpoint['last'][arm], None)
                for arm in ARMS}
    require(all(item['passed'] for item in receipts.values()), 'Fixed final epoch cached replay differs')
    return {'A': baseline, **records}, receipts


def write_seed_results(output, rows, lengths, selections, predictions, updates, orders, initial):
    summary = {'updates_per_arm': updates, 'order_sha256_each_epoch': orders, 'initial_state_sha256': initial,
               'selected_epochs': {arm: selections['best'][arm]['epoch'] for arm in ARMS}}
    for phase in PHASES:
        folder = output / phase
        folder.mkdir()
        metrics, comparisons = attention.comparison_outputs(
            folder, rows, predictions[phase], lengths, [('B', 'A'), ('C1', 'A'), ('C1', 'B')])
        require(all(result['by_valid_edges']['0']['changed_predictions'] == 0 for result in comparisons.values()),
                'A prediction changed on a graph with no valid transitions')
        attention.write_predictions(folder / 'validation_predictions.jsonl.gz', predictions[phase], lengths)
        by_class = {label: 100 * (metrics['C1']['per_class'][label]['f1-score']
                                 - metrics['B']['per_class'][label]['f1-score']) / len(LABELS)
                    for label in LABELS}
        summary[phase] = {'metrics': metrics, 'comparisons': comparisons,
                          'class_contributions_to_c1_minus_b_pp': by_class}
    raw.write(output / 'metrics.json', summary)
    return summary


def run_seed(output, model, data, partitions, device, heads, config, train, val,
             baseline, baseline_selection, baseline_receipt, lengths, expected_updates, frozen_sha):
    initial = initial_hashes(heads)
    raw.write(output / 'config.json', {**config, 'query_sources': original.SOURCES,
              'initial_state_sha256': initial, 'frozen_c_sha256': frozen_sha, 'features_shared_between_arms': True,
              'batch_order_shared_between_arms': True})
    raw.write(output / 'validation_replay.json', {'epoch_zero_raw': baseline_receipt})
    attention_checks = {arm: original.check_attention(head, device)
                        for arm, head in original.make_heads(config, device, ARMS).items()}
    probe = original.probe(model, data, config, device, ARMS)
    raw.write(output / 'preflight.json', {**probe, 'attention_checks': attention_checks})
    require(raw.state_digest(model.state_dict()) == frozen_sha, 'Gradient preflight changed frozen C')
    rows = partitions['val']
    best, last, updates = original.fit(model, heads, train, val, rows, baseline, config, device, output)
    require(updates == expected_updates * config['epochs'], 'Optimizer budget differs')
    history = [json.loads(line) for line in (output / 'epochs.jsonl').read_text(encoding='utf-8').splitlines()]
    selections = {'best': best, 'last': last}
    orders = verify_history(history, config, selections, baseline_selection['validation'], expected_updates)
    checkpoint = output / 'transition_readouts.pt'
    torch.save({**selections, 'config': config, 'frozen_c_sha256': frozen_sha,
                'initial_state_sha256': initial, 'query_sources': original.SOURCES}, checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    selected = original.replay_selected(model, saved, data, rows, config, device, output)
    receipts = {arm: evaluation.replay_receipt(joint.study.validate_predictions(items, rows, probabilities=False),
                                              items, baseline_selection if arm == 'A' else best[arm],
                                              baseline if arm == 'A' else None) for arm, items in selected.items()}
    raw.write(output / 'validation_replay.json', {'epoch_zero_raw': baseline_receipt, 'best_raw': receipts,
              'checkpoint_sha256': raw.sha(checkpoint)})
    require(all(item['passed'] for item in receipts.values()), 'Selected checkpoint original-graph replay differs')
    final, final_receipts = replay_last(model, saved, val, rows, baseline, config, device)
    raw.write(output / 'validation_replay.json', {'epoch_zero_raw': baseline_receipt, 'best_raw': receipts,
              'last_cached': final_receipts, 'checkpoint_sha256': raw.sha(checkpoint)})
    require(raw.state_digest(model.state_dict()) == frozen_sha
            and all(not p.requires_grad and p.grad is None for p in model.parameters())
            and all(not module.training for module in model.modules()), 'Frozen C changed')
    return write_seed_results(output, rows, lengths, saved, {'best': selected, 'last': final}, updates, orders, initial)


def paired_stats(results, phase):
    values = {seed: summary[phase]['comparisons']['C1-B']['delta_macro_f1_pp']
              for seed, summary in results.items()}
    return {'by_seed_pp': values, 'mean_pp': mean(values.values()), 'sample_sd_pp': stdev(values.values()),
            'positive': sum(value > 0 for value in values.values()),
            'tied': sum(value == 0 for value in values.values()),
            'negative': sum(value < 0 for value in values.values())}


def write_summary(output, results, backbone_config):
    aggregate = {phase: {'all': paired_stats(results, phase),
                         'new': paired_stats({seed: value for seed, value in results.items() if seed != 42}, phase)}
                 for phase in PHASES}
    budget = backbone_config['budget_reference_pp']
    raw.write(output / 'summary.json', {'status': 'completed', 'seeds': {42: 'archived', 43: 'completed', 44: 'completed'},
              'aggregate': aggregate, 'by_seed': results, 'baseline_reference_pp': budget})
    with (output / 'per_class.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['readout_seed', 'selection', 'arm', 'class', 'precision', 'recall', 'f1', 'support'])
        for seed, summary in results.items():
            for phase in PHASES:
                for arm, metrics in summary[phase]['metrics'].items():
                    for label, item in metrics['per_class'].items():
                        writer.writerow([seed, phase, arm, label, item['precision'], item['recall'],
                                         item['f1-score'], item['support']])
    lines = ['# 报文转移查询来源配对复核', '',
             '三次配对共用 seed42 原 C 第 48 轮和同一验证划分。seed42 沿用已有训练，'
             '新增 seed43、44 用于检验转移读出对初始化和采样顺序的稳定性。',
             '最佳轮次按验证 Macro-F1 选取，包含第 0 轮，同分取最早。', '',
             '| Seed | 轮次规则 | A | B（轮次） | C1（轮次） | B−A（百分点） | C1−A（百分点） | C1−B（百分点） |',
             '| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for seed, summary in results.items():
        for phase in PHASES:
            row = summary[phase]
            epochs = summary['selected_epochs'] if phase == 'best' else dict.fromkeys(ARMS, backbone_config['epochs'])
            name = '最佳' if phase == 'best' else '固定第 30 轮'
            a, b, c = (row['metrics'][arm]['macro_f1'] for arm in ('A', *ARMS))
            diffs = [row['comparisons'][key]['delta_macro_f1_pp'] for key in ('B-A', 'C1-A', 'C1-B')]
            lines.append(f"| {seed} | {name} | {a:.6%} | {b:.6%}（{epochs['B']}） | {c:.6%}（{epochs['C1']}） | "
                         + ' | '.join(f'{value:+.6f}' for value in diffs) + ' |')
    lines += ['', '| 轮次规则 | Seeds | C1−B 均值（百分点） | 样本标准差（百分点） | 正 / 平 / 负 |',
              '| --- | --- | ---: | ---: | --- |']
    for phase, groups in aggregate.items():
        for cohort, stats in groups.items():
            name = '最佳' if phase == 'best' else '固定第 30 轮'
            seeds = '42、43、44' if cohort == 'all' else '43、44'
            lines.append(f"| {name} | {seeds} | {stats['mean_pp']:+.6f} | {stats['sample_sd_pp']:.6f} | "
                         f"{stats['positive']} / {stats['tied']} / {stats['negative']} |")
    lines += ['', '| Seed | 轮次规则 | B 双向错分 | C1 双向错分 | C1 修复 / 退步 | C1 净正确 | 最大类贡献（百分点） |',
              '| ---: | --- | --- | --- | --- | ---: | --- |']
    for seed, summary in results.items():
        for phase in PHASES:
            row = summary[phase]
            comparison = row['comparisons']['C1-B']
            directions = [comparison['directions'][key] for key in DIRECTIONS]
            b = ' / '.join(str(item['original_errors']) for item in directions)
            c = ' / '.join(str(item['total_errors']) for item in directions)
            contribution = row['class_contributions_to_c1_minus_b_pp']
            label = max(contribution, key=lambda name: abs(contribution[name]))
            lines.append(f"| {seed} | {phase} | {b} | {c} | {comparison['counts']['repaired']} / "
                         f"{comparison['counts']['regressed']} | {comparison['net_correct']:+d} | "
                         f"{label} {contribution[label]:+.6f} |")
    lines += ['', '| Seed | 轮次规则 | 有效报文对数 | 预测变化 | C1 修复 / 退步 | C1 净正确 |',
              '| ---: | --- | --- | ---: | --- | ---: |']
    for seed, summary in results.items():
        for phase in PHASES:
            for key, item in summary[phase]['comparisons']['C1-B']['by_valid_edges'].items():
                lines.append(f"| {seed} | {phase} | {key} | {item['changed_predictions']} | "
                             f"{item['counts']['repaired']} / {item['counts']['regressed']} | {item['net_correct']:+d} |")
    lines += ['', '| Seed | 轮次规则 | 组别 | 类别 | Precision | Recall | F1 |',
              '| ---: | --- | --- | --- | ---: | ---: | ---: |']
    for seed, summary in results.items():
        for phase in PHASES:
            for arm, metrics in summary[phase]['metrics'].items():
                for label in FOCUS:
                    item = metrics['per_class'][label]
                    lines.append(f"| {seed} | {phase} | {arm} | {label} | "
                                 f"{item['precision']:.6%} | {item['recall']:.6%} | {item['f1-score']:.6%} |")
    lines += ['', '双向错分按 Recon→WebBased / WebBased→Recon 排列。只有一个有效报文对时权重固定为 1，'
              '此时改判反映训练后的边编码与输出投影差异。',
              f'相对 A 的投入参考线为 {budget:+.2f} 个百分点。C1−B 按逐种子的方向和类别变化解读。']
    if aggregate['best']['all']['positive'] == len(results):
        lines.append('各 seed 的最佳轮次比较均支持 Flow 查询在该冻结基线上的增益。固定轮次和类别取舍见表。')
    else:
        lines.append('各 seed 的最佳轮次比较存在零差或负差。应按表中范围报告 Flow 查询的收益。')
    lines.append('以上结果来自同一验证划分，用于描述配对重复性。')
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run(repo, output, graph_root, code_commit, state):
    settings = raw.read(CONFIG)
    baseline_config = original.base_config()
    require(settings['backbone_seed'] == baseline_config['seed'] == 42
            and settings['readout_seeds'] == [43, 44], 'Unexpected backbone or readout seed')
    model, data, partitions, device, baseline_selection, inputs = original.load_source(repo, baseline_config, graph_root)
    frozen_sha = raw.state_digest(model.state_dict())
    require(frozen_sha == settings['backbone_state_sha256']
            and data.metadata['sample_manifest_sha256'] == settings['sample_manifest_sha256'],
            'Unexpected frozen backbone or sample manifest')
    rows = partitions['val']
    archived_config = raw.read(attention.CONFIG)
    saved42, reference, lengths, history42, archives = attention.archived_inputs(
        repo, archived_config, baseline_config, rows, baseline_selection)
    require(saved42['frozen_c_sha256'] == frozen_sha, 'Seed42 transition archive uses a different backbone')
    inputs.update(archives)
    expected_updates = archived_config['expected_updates_per_epoch']
    raw.write(output / 'config.json', {**settings, 'backbone_protocol': baseline_config,
              'seed42_transition_archive': archived_config, 'partitions': ['train', 'val'],
              'query_sources': original.SOURCES, 'updates_per_epoch_per_arm': expected_updates})
    provenance = {'code_commit': code_commit, 'seed42_transition_commit': archived_config['archive_commit'],
                  'backbone_commit': baseline_config['archive_commit'], 'frozen_c_sha256': frozen_sha,
                  'graph_root': str(data.root), 'python': platform.python_version(),
                  'torch': str(torch.__version__), 'pyg': torch_geometric.__version__,
                  'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(device),
                  'input_sha256': {str(path): sha for path, sha in inputs.items()}}
    raw.write(output / 'provenance.json', provenance)
    configs = {seed: {**baseline_config, 'readout_seed': seed} for seed in settings['readout_seeds']}
    heads = {seed: original.make_heads(config, device, ARMS) for seed, config in configs.items()}
    initials = {42: saved42['initial_transition_sha256']}
    for seed, pair in heads.items():
        initials[seed] = initial_hashes(pair)['B']
        require(all(sum(p.numel() for p in head.parameters() if p.requires_grad) == settings['parameters_per_arm']
                    for head in pair.values()), f'Seed{seed} trainable parameter count differs')
    require(len(set(initials.values())) == len(initials), 'Distinct readout seeds share the same initialization')

    def check_new_heads(features):
        # Seed43 is checked by cache_split while it builds the baseline predictions.
        check_zero(model, heads[44], features)

    val, val_coverage, baseline = original.cache_split(model, data, rows, 'val', baseline_config, device,
                                                       output, heads[43], observe=check_new_heads)
    receipt = evaluation.replay_receipt(joint.study.validate_predictions(baseline, rows, probabilities=False),
                                        baseline, baseline_selection, reference['A'])
    receipt['exact_zero_residual_and_scores'] = {seed: list(pair) for seed, pair in heads.items()}
    raw.write(output / 'epoch_zero_raw.json', receipt)
    require(receipt['passed'] and val.features.lengths.tolist() == lengths, 'Fixed validation baseline or edges differ')
    raw.progress(output, 'epoch_zero_passed', samples=len(rows))

    # The archived selection supplies seed42 best predictions. Its fixed final checkpoints
    # are evaluated on the same validated features so all three seeds have paired counts.
    folder42 = output / 'seed42'
    folder42.mkdir()
    last42, replay42 = replay_last(model, saved42, val, rows, baseline, baseline_config, device)
    orders42 = verify_history(history42, baseline_config, saved42, baseline_selection['validation'], expected_updates)
    raw.write(folder42 / 'validation_replay.json', {'best_archive_prediction_ids_sha256': {
              arm: joint.prediction_digest(reference[arm]) for arm in ('A', *ARMS)}, 'last_cached': replay42})
    results = {42: write_seed_results(folder42, rows, lengths, saved42, {'best': reference, 'last': last42},
                                      expected_updates * baseline_config['epochs'], orders42, dict.fromkeys(ARMS, initials[42]))}
    raw.write(folder42 / 'status.json', {'status': 'archived', 'archive_commit': archived_config['archive_commit']})
    state['seeds'][42] = 'archived'
    raw.write(output / 'status.json', state)
    train, train_coverage, _ = original.cache_split(model, data, partitions['train'], 'train',
                                                   baseline_config, device, output)
    raw.write(output / 'edge_coverage.json', {'train': train_coverage, 'val': val_coverage})
    failures = []
    for seed, config in configs.items():
        folder = output / f'seed{seed}'
        folder.mkdir()
        raw.write(folder / 'status.json', {'status': 'running'})
        state['seeds'][seed] = 'running'
        raw.write(output / 'status.json', state)
        try:
            results[seed] = run_seed(folder, model, data, partitions, device, heads[seed], config, train, val,
                                     baseline, baseline_selection, receipt, lengths, expected_updates, frozen_sha)
        except BaseException as error:
            raw.write(folder / 'status.json', {'status': 'failed', 'error': str(error)})
            state['seeds'][seed] = 'failed'
            if not isinstance(error, Exception) or raw.state_digest(model.state_dict()) != frozen_sha:
                raise
            failures.append(f'seed{seed}: {error}')
            raw.progress(output, 'seed_failed', seed=seed, error=str(error))
        else:
            raw.write(folder / 'status.json', {'status': 'completed'})
            state['seeds'][seed] = 'completed'
            raw.progress(output, 'seed_completed', seed=seed)
        raw.write(output / 'status.json', state)
    if failures:
        raw.write(output / 'summary.json', {'status': 'failed', 'seeds': state['seeds'], 'by_seed': results,
                  'errors': failures})
        raise RuntimeError(' | '.join(failures))
    for seed in settings['readout_seeds']:
        require(all(a != b for a, b in zip(results[seed]['order_sha256_each_epoch'], orders42)),
                f'Seed{seed} unexpectedly shares an epoch order with seed42')
    require(all(a != b for a, b in zip(results[43]['order_sha256_each_epoch'], results[44]['order_sha256_each_epoch'])),
            'Seed43 and seed44 unexpectedly share an epoch order')
    evaluation.verify_inputs(inputs)
    raw.write(output / 'provenance.json', {**provenance, 'frozen_c_verified': True, 'initial_readout_sha256': initials})
    write_summary(output, results, baseline_config)


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
    state = {'status': 'running', 'seeds': {42: 'pending_archive', 43: 'pending', 44: 'pending'}}
    raw.write(output / 'status.json', state)
    try:
        run(repo, output, args.data_root, args.code_commit, state)
    except BaseException as error:
        raw.write(output / 'status.json', {**state, 'status': 'failed', 'error': str(error)})
        raise
    raw.write(output / 'status.json', {**state, 'status': 'completed'})
    print(json.dumps({'status': 'completed', 'report': str(output / 'report.md')}), flush=True)


if __name__ == '__main__':
    main()
