"""Validation-only structure, residual, attention, gate, and paired reports."""
from collections import defaultdict
import csv
import math

import torch
from torch.nn import functional as F

import diagnose_joint_transition_fusion as diagnosis
import joint_transition_training as training
from xgnid.packet_pooling import ordered_softmax

raw, require, LABELS = training.raw, training.require, diagnosis.LABELS
ARMS, ALIASES = training.ARMS, training.ALIASES


def context_counts(features):
    owner = torch.repeat_interleave(torch.arange(len(features.lengths), device=features.lengths.device), features.lengths)
    return torch.bincount(owner[features.transitions[:, 70].bool()], minlength=len(features.lengths))


@torch.no_grad()
def context_attention(head, features, context_right=None):
    counts = context_counts(features)
    if head is None or head.mode == 'single':
        return [], [0] * len(counts)
    left = torch.where(features.transitions[:, 70].bool())[0]
    if not len(left):
        return [], counts.tolist()
    owner = torch.repeat_interleave(torch.arange(len(features.lengths), device=features.lengths.device), features.lengths)
    values = F.gelu(head.edge(features.transitions[:, :70]))
    right = context_right if context_right is not None else (left + 1 if head.mode == 'ordered' else left)
    contexts = F.gelu(head.context(torch.cat((values[left], values[right]), -1)))
    queries = head.query(features.packet_mean)
    scores = (contexts * queries[owner[left]]).sum(-1, keepdim=True) / 4
    weights = ordered_softmax(scores, owner[left], counts).flatten()
    return weights.cpu().tolist(), counts.tolist()


@torch.no_grad()
def observe(arms, features, rows):
    counts = context_counts(features).cpu().tolist()
    lengths = features.lengths.cpu().tolist()
    records = [{**row, 'valid_pairs': length, 'ordered_contexts': count, 'predictions': {}, 'diagnostics': {}}
               for row, length, count in zip(rows, lengths, counts)]
    packet_norm = features.packet.norm(dim=-1).clamp_min(1e-12)
    for name, arm in arms.items():
        arm.eval()
        scores, residual, alpha, edge_weights = arm(features)
        require(bool(torch.isfinite(scores).all() and torch.isfinite(residual).all()
                     and torch.isfinite(alpha).all()), f'{name} nonfinite diagnostic forward')
        logs, gates = scores.cpu().tolist(), alpha.cpu().tolist()
        predictions = scores.argmax(-1).cpu().tolist()
        magnitude = residual.norm(dim=-1)
        norms = magnitude.cpu().tolist()
        relative = (magnitude / packet_norm).cpu().tolist()
        # Linear bias cancels, so this measures the exact projected residual difference.
        projected = F.linear(residual, arm.fusion.packet_projection.weight).norm(dim=-1).cpu().tolist()
        weights = [] if edge_weights is None else edge_weights.cpu().tolist()
        context_weights, context_lengths = context_attention(
            arm.readout, features, arm.context_right(features) if hasattr(arm, 'context_right') else None)
        offset, context_offset = 0, 0
        for i, row in enumerate(records):
            edges = weights[offset:offset + lengths[i]] if arm.readout is not None else []
            contexts = context_weights[context_offset:context_offset + context_lengths[i]]
            row['predictions'][name] = predictions[i]
            row['diagnostics'][name] = {
                'flow_gate_weights': gates[i], 'flow_gate_mean': sum(gates[i]) / len(gates[i]),
                'residual_l2': norms[i], 'residual_packet_ratio': relative[i],
                'projected_residual_l2': projected[i], 'one_step_attention': edges,
                'context_attention': contexts, 'context_mode': None if arm.readout is None else arm.readout.mode,
                'one_step_attention_entropy': -sum(w * math.log(max(w, 1e-30)) for w in edges),
                'context_attention_entropy': -sum(w * math.log(max(w, 1e-30)) for w in contexts),
                'log_probabilities': logs[i]}
            offset += lengths[i]
            context_offset += context_lengths[i]
    return records


def exports(records):
    return {name: [{'sample_id': row['sample_id'], 'label': row['label'], 'prediction': row['predictions'][name]}
                   for row in records] for name in records[0]['predictions']}


def aggregate(records, before, after):
    groups = defaultdict(list)
    for row in records:
        a, b, y = row['predictions'][before], row['predictions'][after], row['label']
        kind = 'repaired' if a != y and b == y else 'regressed' if a == y and b != y else (
            'wrong_changed' if a != b else 'unchanged')
        for key in (kind, f"{kind}|{LABELS[y]}", f"{kind}|context_{row['ordered_contexts'] > 0}",
                    f"{kind}|{LABELS[y]}|context_{row['ordered_contexts'] > 0}"):
            groups[key].append(row)
    summary = {}
    for key, items in groups.items():
        values = {'samples': len(items), 'mean_valid_pairs': sum(r['valid_pairs'] for r in items) / len(items),
                  'mean_ordered_contexts': sum(r['ordered_contexts'] for r in items) / len(items)}
        for arm in (before, after):
            values[arm] = {field: sum(r['diagnostics'][arm][field] for r in items) / len(items)
                          for field in ('flow_gate_mean', 'residual_l2', 'residual_packet_ratio',
                                        'projected_residual_l2', 'one_step_attention_entropy', 'context_attention_entropy')}
            values[arm]['mean_flow_gate_by_dimension'] = [
                sum(r['diagnostics'][arm]['flow_gate_weights'][d] for r in items) / len(items)
                for d in range(len(items[0]['diagnostics'][arm]['flow_gate_weights']))]
        values['flow_gate_mean_delta'] = values[after]['flow_gate_mean'] - values[before]['flow_gate_mean']
        summary[key] = values
    return summary


def write_tables(output, metrics):
    with (output / 'per_class.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(('arm', 'class', 'precision', 'recall', 'f1', 'support'))
        for name, metric in metrics.items():
            for label, row in metric['per_class'].items():
                writer.writerow((name, label, *(row[k] for k in ('precision', 'recall', 'f1-score', 'support'))))
    for name, metric in metrics.items():
        with (output / (name + '_confusion_matrix.csv')).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(('true/predicted', *LABELS))
            writer.writerows((label, *values) for label, values in zip(LABELS, metric['confusion_matrix']))


def write_results(output, records, rows, state, settings):
    folder = output / 'validation'
    folder.mkdir(exist_ok=True)
    predicted = exports(records)
    metrics = {name: training.frozen.joint.study.validate_predictions(items, rows, probabilities=False)
               for name, items in predicted.items()}
    pairs = [('joint_ordered', name) for name in ARMS[:-1]]
    pairs += [(name, 'original_C') for name in ARMS]
    pairs += [(name, 'frozen_' + training.MODES[name]) for name in ARMS[1:]]
    pairs += [('frozen_ordered', 'frozen_single')]
    comparisons, diagnostics = {}, {}
    for after, before in pairs:
        key = after + '-' + before
        paired, _ = raw.compare(predicted[before], predicted[after])
        summary, changed = diagnosis.summarize(records, before, after)
        comparisons[key] = {**paired, **summary,
            'delta_macro_f1_pp': 100 * (metrics[after]['macro_f1'] - metrics[before]['macro_f1'])}
        raw.save_predictions(folder / (key + '_changed_samples.jsonl.gz'), changed)
        diagnostics[key] = aggregate(records, before, after)
    raw.save_predictions(folder / 'validation_predictions.jsonl.gz', records)
    for name in ARMS:
        raw.save_predictions(output / name / 'best_validation_predictions.jsonl.gz', predicted[name])
    raw.write(folder / 'metrics.json', {'metrics': metrics, 'comparisons': comparisons})
    raw.write(folder / 'diagnostic_groups.json', diagnostics)
    write_tables(folder, metrics)
    budgets = []
    for budget in (30, 100, 150, 200):
        if budget > settings['epochs']:
            continue
        selected = {}
        for name in ARMS:
            top = max((row for row in state['history'] if row['epoch'] <= budget),
                      key=lambda row: row['validation'][name]['macro_f1'])
            selected[name] = {'epoch': top['epoch'], 'macro_f1': top['validation'][name]['macro_f1']}
        budgets.append({'epoch_budget': budget, 'selected': selected,
                        'delta_macro_f1_pp': {'joint_ordered-' + name: 100 * (
                            selected['joint_ordered']['macro_f1'] - selected[name]['macro_f1']) for name in ARMS[:-1]}})
    raw.write(output / 'budget_comparison.json', {'partition': 'validation', 'budgets': budgets})
    gates = {key: {'delta_pp': comparisons[key]['delta_macro_f1_pp'], 'reference_pp': threshold,
                   'passed': comparisons[key]['delta_macro_f1_pp'] >= threshold}
             for key, threshold in settings['investment_reference_pp'].items()}
    summary = {'partition': 'validation', 'test_evaluated': False, 'metrics': metrics,
               'comparisons': comparisons, 'investment_reference': gates,
               'exceeds_frozen_ordered': metrics['joint_ordered']['macro_f1'] > metrics['frozen_ordered']['macro_f1'],
               'selected_epochs': {name: item['epoch'] for name, item in state['best'].items()}}
    raw.write(output / 'summary.json', summary)
    lines = ['# 报文转移读出与双分支融合层联合优化：seed42 配对验证', '',
             '四组从同一个原 C 开始，主干与跨层模块冻结，仅优化融合层及新增读出。仅使用 train/val。', '',
             '| 组别 | 最佳轮 | Macro-F1 | 相对原 C（百分点） |', '| --- | ---: | ---: | ---: |']
    for name in ARMS:
        lines.append(f"| {ALIASES[name]} ({name}) | {state['best'][name]['epoch']} | "
                     f"{metrics[name]['macro_f1']:.6%} | {100*(metrics[name]['macro_f1']-metrics['original_C']['macro_f1']):+.6f} |")
    lines += ['', '| 比较 | Macro-F1 差（百分点） | 修复 | 退步 | 错误间变化 |', '| --- | ---: | ---: | ---: | ---: |']
    for key, value in comparisons.items():
        groups = value['groups']
        lines.append(f"| {key} | {value['delta_macro_f1_pp']:+.6f} | " + ' | '.join(
            str(groups[k]['count']) for k in ('repaired', 'regressed', 'wrong_changed')) + ' |')
    lines += ['', '| 同预算轮数 | J0（最佳轮） | J1（最佳轮） | J2（最佳轮） | J3（最佳轮） | J3−J1 pp | J3−J2 pp |',
              '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in budgets:
        lines.append(f"| {row['epoch_budget']} | " + ' | '.join(
            f"{row['selected'][n]['macro_f1']:.6%} ({row['selected'][n]['epoch']})" for n in ARMS) +
            f" | {row['delta_macro_f1_pp']['joint_ordered-joint_single']:+.6f} | "
            f"{row['delta_macro_f1_pp']['joint_ordered-joint_pointwise']:+.6f} |")
    lines += ['', '投入参考按未四舍五入值判断：' + str(gates),
              f"J3 是否超过冻结 Ordered（实测 {metrics['frozen_ordered']['macro_f1']:.6%}）：{summary['exceeds_frozen_ordered']}。",
              '这些门槛仅用于资源决策，不代表统计显著性。相对原 C 的全部提升不能归因于有序结构。']
    if all(metrics[n]['macro_f1'] > metrics['original_C']['macro_f1'] for n in ARMS) and not all(
            item['passed'] for item in gates.values()):
        lines += ['四组共同提高但 J3 未达到拉开差距的投入参考，当前证据应主要解读为融合层优化收益。']
    elif all(item['passed'] for item in gates.values()):
        lines += ['两项投入参考均达到，可进一步安排同预算流内打乱配对对照；本轮不自动追加实验或测试。']
    else:
        lines += ['尚未同时达到两项投入参考；保留类别得失，不能据总体差值预先宣称协同收益。']
    lines += ['', '八类 Precision/Recall/F1 与混淆矩阵见 validation/per_class.csv 及各组 CSV。',
              '所有修复/退步均按真实类别、错分方向、有效一步数量和真实连续两步数量保留。',
              '门控逐维权重、两尺度注意力、残差强度见 validation/validation_predictions.jsonl.gz；',
              '按修复/退步、真实类别和连续片段分组的均值见 validation/diagnostic_groups.json。',
              '门控与残差的相关变化是观察性诊断，不能单独证明冻结门控导致旧错误。',
              '旧 Single 前 30 轮重跑与历史有小幅差异；本轮以同时实施的配对结果为主要依据。']
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return summary
