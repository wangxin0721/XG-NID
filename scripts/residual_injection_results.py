"""Eight-class validation, matched repairs/regressions and scalar-beta distributions."""
import json

import torch
from torch.nn import functional as F

import joint_transition_diagnostics as parent
import diagnose_joint_transition_fusion as changes
import residual_injection_training as training
from xgnid.residual_injection import ARMS, ALIASES

raw, frozen, require = training.raw, training.frozen, training.require
LABELS = parent.LABELS
exports = parent.exports


@torch.no_grad()
def observe(arms, features, rows, reference=None):
    counts, lengths = parent.context_counts(features).tolist(), features.lengths.tolist()
    records = [{**row, 'valid_pairs': n, 'ordered_contexts': c, 'predictions': {}, 'diagnostics': {}}
               for row, n, c in zip(rows, lengths, counts)]
    if reference is not None:
        scores, _, _, _ = reference(features)
        for row, prediction in zip(records, scores.argmax(-1).tolist()):
            row['predictions']['original_C'] = prediction
    packet_norm = features.packet.norm(dim=-1).clamp_min(1e-12)
    for name, arm in arms.items():
        arm.eval()
        result = arm.detailed_forward(features)
        require(bool(torch.isfinite(result.scores).all() and torch.isfinite(result.beta).all()
                     and torch.isfinite(result.injected).all()), name + ' nonfinite diagnostics')
        _, off_alpha = arm.fusion(features.flow, features.packet)
        gates, deltas = result.alpha.tolist(), (result.alpha - off_alpha).tolist()
        prediction, beta = result.scores.argmax(-1).tolist(), result.beta.flatten().tolist()
        norms, injected = result.residual.norm(dim=-1), result.injected.norm(dim=-1)
        scalar = {'beta': beta, 'residual_l2': norms.tolist(), 'injected_residual_l2': injected.tolist(),
                  'residual_packet_ratio': (norms / packet_norm).tolist(),
                  'injected_packet_ratio': (injected / packet_norm).tolist(),
                  'projected_residual_l2': F.linear(result.injected, arm.fusion.packet_projection.weight).norm(dim=-1).tolist()}
        logs, edges = result.scores.tolist(), result.attention.tolist()
        contexts, context_lengths = parent.context_attention(arm.readout, features)
        offset = context_offset = 0
        for i, row in enumerate(records):
            row['predictions'][name] = prediction[i]
            row['diagnostics'][name] = {key: values[i] for key, values in scalar.items()}
            row['diagnostics'][name].update(log_probabilities=logs[i], flow_gate_weights=gates[i],
                flow_gate_mean=sum(gates[i]) / len(gates[i]), flow_gate_delta_from_residual_off=deltas[i],
                one_step_attention=edges[offset:offset + lengths[i]],
                context_attention=contexts[context_offset:context_offset + context_lengths[i]])
            offset += lengths[i]; context_offset += context_lengths[i]
    return records


def distribution(values):
    if not values:
        return {'samples': 0}
    x = torch.tensor(values, dtype=torch.float64)
    quantiles = torch.quantile(x, torch.tensor([0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0], dtype=x.dtype))
    return {'samples': len(values), 'mean': float(x.mean()), 'std': float(x.std(unbiased=False)),
            'min': float(x.min()), 'max': float(x.max()),
            'quantiles': dict(zip(('p0', 'p5', 'p25', 'p50', 'p75', 'p95', 'p100'), quantiles.tolist())),
            'fraction_below_one': float((x < 1).double().mean()),
            'fraction_above_one': float((x > 1).double().mean()),
            'fraction_below_0_1': float((x < 0.1).double().mean()),
            'fraction_above_1_9': float((x > 1.9).double().mean())}


def beta_summary(records):
    result = {}
    for name in ARMS:
        def summarize(items):
            return distribution([r['diagnostics'][name]['beta'] for r in items])
        result[name] = {'overall': summarize(records),
            'by_true_class': {label: summarize([r for r in records if r['label'] == i]) for i, label in enumerate(LABELS)},
            'by_correctness': {kind: summarize([r for r in records if
                (r['predictions'][name] == r['label']) == (kind == 'correct')]) for kind in ('correct', 'wrong')},
            'by_context': {str(present): summarize([r for r in records if bool(r['ordered_contexts']) == present])
                           for present in (False, True)}}
    require(result['inj_fixed']['overall']['min'] == result['inj_fixed']['overall']['max'] == 1.0,
            'Fixed injection beta changed')
    require(result['inj_global']['overall']['max'] - result['inj_global']['overall']['min'] < 1e-6,
            'R1 beta is not globally shared')
    return result


@torch.no_grad()
def cached_beta_summary(arms, cache, rows, settings, device):
    records = []
    for arm in arms.values():
        arm.eval()
    for start in range(0, len(rows), settings['eval_batch_size']):
        end = min(start + settings['eval_batch_size'], len(rows))
        features, _ = cache.batch(range(start, end), device)
        counts = parent.context_counts(features).tolist()
        current = [{**row, 'ordered_contexts': count, 'predictions': {}, 'diagnostics': {}}
                   for row, count in zip(rows[start:end], counts)]
        for name, arm in arms.items():
            result = arm.detailed_forward(features)
            for row, value, prediction in zip(current, result.beta.flatten().tolist(), result.scores.argmax(-1).tolist()):
                row['diagnostics'][name] = {'beta': value}
                row['predictions'][name] = prediction
        records.extend(current)
    return {'checkpoint': 'last_epoch', 'partition': 'validation', 'distributions': beta_summary(records)}


def decisions(metrics, settings):
    comparisons = {key: {'delta_pp': 100 * (metrics[after]['macro_f1'] - metrics[before]['macro_f1']),
        'reference_pp': threshold} for key, threshold in settings['investment_reference_pp'].items()
        for after, before in [key.split('-')]}
    for row in comparisons.values():
        row['passed'] = row['delta_pp'] >= row['reference_pp']
    absolute = metrics['inj_dynamic']['macro_f1'] > settings['historical_true_macro_f1']
    passed = all(row['passed'] for row in comparisons.values()) and absolute
    if passed:
        interpretation = '达到两项配对差值及绝对分数参考，支持进一步验证逐流适应贡献；门槛不代表统计显著性。'
    elif metrics['inj_global']['macro_f1'] > metrics['inj_fixed']['macro_f1'] and not all(
            row['passed'] for row in comparisons.values()):
        interpretation = '未通过逐流适应的继续投入条件；全局强度组的提高应归于全局调节，不能归于逐流适应。'
    else:
        interpretation = '未同时通过继续投入条件；如实保留类别收益与损失，不能宣称逐流门控有效。'
    return {'paired_thresholds': comparisons, 'dynamic_exceeds_historical_True': absolute,
            'historical_True_macro_f1': settings['historical_true_macro_f1'], 'all_passed': passed,
            'interpretation': interpretation}


def write_results(output, records, rows, state, settings):
    folder = output / 'validation'; folder.mkdir(exist_ok=True)
    predicted = exports(records)
    metrics = {name: frozen.joint.study.validate_predictions(items, rows, probabilities=False)
               for name, items in predicted.items()}
    pairs = [('inj_dynamic', 'inj_fixed'), ('inj_dynamic', 'inj_global'), ('inj_global', 'inj_fixed')]
    pairs += [(arm, before) for arm in ARMS for before in ('original_C', 'historical_True')]
    comparisons, grouped_beta = {}, {}
    for after, before in pairs:
        key = after + '-' + before
        paired, _ = raw.compare(predicted[before], predicted[after])
        summary, changed = changes.summarize(records, before, after)
        comparisons[key] = {**paired, **summary,
            'delta_macro_f1_pp': 100 * (metrics[after]['macro_f1'] - metrics[before]['macro_f1'])}
        raw.save_predictions(folder / (key + '_changed_samples.jsonl.gz'), changed)
        grouped_beta[key] = {kind: {arm: distribution([r['diagnostics'][arm]['beta'] for r in changed if r['kind'] == kind])
                            for arm in ARMS} for kind in ('repaired', 'regressed', 'wrong_changed')}
    for name in ARMS:
        require(metrics[name] == state['best'][name]['validation'], 'Report differs from selected checkpoint')
        raw.save_predictions(output / name / 'best_validation_predictions.jsonl.gz', predicted[name])
    raw.save_predictions(folder / 'validation_predictions.jsonl.gz', records)
    raw.write(folder / 'metrics.json', {'metrics': metrics, 'comparisons': comparisons})
    parent.write_tables(folder, metrics)
    per_class_changes = {key: {label: {field: metrics[after]['per_class'][label][field] -
        metrics[before]['per_class'][label][field] for field in ('precision', 'recall', 'f1-score')}
        for label in LABELS} for after, before in pairs for key in [after + '-' + before]}
    raw.write(folder / 'per_class_changes.json', per_class_changes)
    web = LABELS.index('WebBased')
    webbased = {}
    for name, metric in metrics.items():
        cm = metric['confusion_matrix']
        tp = cm[web][web]
        webbased[name] = {'TP': tp, 'FP': sum(row[web] for row in cm) - tp, 'FN': sum(cm[web]) - tp,
            'FP_by_true_class': {label: cm[i][web] for i, label in enumerate(LABELS) if i != web}}
    raw.write(folder / 'webbased_false_positives.json', webbased)
    beta = beta_summary(records)
    raw.write(output / 'beta_distributions.json', {'checkpoint': 'best_overall_validation_per_arm',
        'partition': 'validation', 'distributions': beta, 'by_paired_changes': grouped_beta})
    budgets = []
    for budget in (30, 100, 150, 200):
        if budget > settings['epochs']:
            continue
        selected = {name: max((r for r in state['history'] if r['epoch'] <= budget),
                    key=lambda r: r['validation'][name]['macro_f1']) for name in ARMS}
        budgets.append({'epoch_budget': budget, 'selected': {name: {'epoch': row['epoch'],
            'macro_f1': row['validation'][name]['macro_f1']} for name, row in selected.items()},
            'decisions': decisions({name: row['validation'][name] for name, row in selected.items()}, settings)})
    raw.write(output / 'budget_comparison.json', {'partition': 'validation', 'budgets': budgets})
    decision = decisions(metrics, settings)
    summary = {'partition': 'validation', 'test_evaluated': False, 'metrics': metrics, 'comparisons': comparisons,
        'decisions': decision, 'selected_epochs': {name: best['epoch'] for name, best in state['best'].items()},
        'epochs_per_arm': state['completed_epoch'], 'updates_per_arm': state['updates'],
        'previous_True_Shuffle_threshold_conclusion_changed': False}
    raw.write(output / 'summary.json', summary)
    lines = ['# 残差注入强度：seed42 三组匹配训练', '',
        '三组从同一个原 C 出发，使用真实连续配对；编码器与跨层表示冻结，仅优化读出、融合层与相应标量注入门控。',
        '原 32 维融合门控保留；beta 是每条流一个标量，可抑制、保持或增强残差。R1 输入常量，R2 输入表示统计，不能称为校准置信度。', '',
        '| 组别 | 最佳轮 | Macro-F1 | beta 均值 | beta 最小/最大 |', '| --- | ---: | ---: | ---: | ---: |']
    for name in ARMS:
        b = beta[name]['overall']
        lines.append(f"| {ALIASES[name]} ({name}) | {state['best'][name]['epoch']} | {metrics[name]['macro_f1']:.6%} | "
                     f"{b['mean']:.6f} | {b['min']:.6f}/{b['max']:.6f} |")
    lines += ['', '| 比较 | Macro-F1 差（百分点） | 修复 | 退步 | 错误间变化 |', '| --- | ---: | ---: | ---: | ---: |']
    for key, value in comparisons.items():
        lines.append(f"| {key} | {value['delta_macro_f1_pp']:+.6f} | " + ' | '.join(
            str(value['groups'][kind]['count']) for kind in ('repaired', 'regressed', 'wrong_changed')) + ' |')
    lines += ['', '| 类别 | R0 F1 | R1 F1 | R2 F1 | R2−R0 pp | R2−R1 pp |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for label in LABELS:
        lines.append('| ' + label + ' | ' + ' | '.join(f"{metrics[name]['per_class'][label]['f1-score']:.6%}" for name in ARMS)
            + f" | {100 * per_class_changes['inj_dynamic-inj_fixed'][label]['f1-score']:+.6f}"
            + f" | {100 * per_class_changes['inj_dynamic-inj_global'][label]['f1-score']:+.6f} |")
    lines += ['', decision['interpretation'],
        f"R2 是否超过既有 True {settings['historical_true_macro_f1']:.6%}：{decision['dynamic_exceeds_historical_True']}。",
        '决策使用未四舍五入数值，两个差值均含边界，绝对分数要求严格超过；不作统计显著性声明。',
        '残差总体有益、True 残差对 WebBased 误报净减少 2 个、此前 True−Shuffle 未通过门槛的结论均保留。',
        '本实验检验残差强度调节，不重新宣称真实相邻配对具有已证实的训练优势。', '',
        '八类 P/R/F1 与混淆矩阵、每组相对原 C/历史 True 的变化见 validation；所有修复/退步样本保留真实类别与错分方向。',
        'beta_distributions.json 包含按真实类别、正误、连续片段覆盖及修复/退步分组的 beta 分布；末轮分布单独保存。',
        '30/100/150/200 轮窗口见 budget_comparison.json。最终报告以验证集选出的最佳轮为准，包含第 0 轮，同分取最早。',
        '历史 True 仅作外部参考，主要结论依据同期三组配对训练。未运行测试集或 seed43/44，不自动追加后续实验。']
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return summary
