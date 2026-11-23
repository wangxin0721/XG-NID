"""Fixed-model sensitivity, residual counterfactuals and explicit WebBased FP inventories."""
from collections import Counter
import math

import joint_transition_diagnostics as tables
import joint_transition_training as training
from run_transition_pair_cross_replay import GROUPS, exports

raw, frozen, require, LABELS = training.raw, training.frozen, training.require, tables.LABELS
WEB = LABELS.index('WebBased')


def compare(records, after, before):
    predicted = exports(records)
    before_metric = frozen.joint.study.validate_predictions(predicted[before], probabilities=False)
    after_metric = frozen.joint.study.validate_predictions(predicted[after], probabilities=False)
    paired, _ = raw.compare(predicted[before], predicted[after])
    changed, kinds, buckets = [], Counter(), {}
    score_changes, probability_max = 0, 0.0
    for row in records:
        a, b, y = row['predictions'][before], row['predictions'][after], row['label']
        xlog = row['diagnostics'][before]['log_probabilities']
        ylog = row['diagnostics'][after]['log_probabilities']
        score_changes += xlog != ylog
        probability_max = max(probability_max, max(abs(math.exp(x) - math.exp(y)) for x, y in zip(xlog, ylog)))
        if a == b:
            continue
        kind = 'repaired' if a != y and b == y else 'regressed' if a == y and b != y else 'wrong_changed'
        kinds[kind] += 1
        changed.append({**row, 'change': kind, 'before_group': before, 'after_group': after,
                        'before_prediction': a, 'after_prediction': b, 'true_class': LABELS[y]})
    for kind in ('repaired', 'regressed', 'wrong_changed'):
        items = [r for r in changed if r['change'] == kind]
        bucket = {'samples': len(items), 'by_true_class': dict(Counter(LABELS[r['label']] for r in items)),
                  'directions': dict(Counter(f"{LABELS[r['label']]}: {LABELS[r['before_prediction']]} -> {LABELS[r['after_prediction']]}" for r in items))}
        if items:
            for group in (before, after):
                bucket[group] = {key: sum(r['diagnostics'][group][key] for r in items) / len(items)
                    for key in ('raw_residual_l2', 'injected_residual_l2', 'residual_packet_ratio',
                                'projected_injected_residual_l2', 'flow_gate_mean', 'gate_mean_abs_change_from_off')}
        buckets[kind] = bucket
    result = {**paired, 'samples': len(records), 'changed_predictions': len(changed),
              'changed_log_probability_vectors': score_changes, 'max_abs_probability_difference': probability_max,
              'delta_macro_f1_pp': 100 * (after_metric['macro_f1'] - before_metric['macro_f1']),
              'change_groups': buckets}
    return result, changed


def web_stats(records, group):
    tp = sum(r['label'] == WEB and r['predictions'][group] == WEB for r in records)
    fp = [r for r in records if r['label'] != WEB and r['predictions'][group] == WEB]
    fn = sum(r['label'] == WEB and r['predictions'][group] != WEB for r in records)
    return {'TP': tp, 'FP': len(fp), 'FN': fn,
            'precision': tp / (tp + len(fp)) if tp + len(fp) else 0,
            'recall': tp / (tp + fn) if tp + fn else 0,
            'fp_by_true_class': {label: sum(r['label'] == i for r in fp) for i, label in enumerate(LABELS) if i != WEB}}


def fp_inventory(records, after, before):
    added, removed, retained = [], [], []
    for row in records:
        if row['label'] == WEB:
            continue
        a, b = row['predictions'][before] == WEB, row['predictions'][after] == WEB
        if a or b:
            kind = 'added_FP' if b and not a else 'removed_FP' if a and not b else 'retained_FP'
            item = {**row, 'fp_change': kind, 'true_class': LABELS[row['label']],
                    'before_group': before, 'after_group': after,
                    'T1_false_positive_removed_by_residual_off': row['predictions']['T1'] == WEB and row['predictions']['T0'] != WEB,
                    'S2_false_positive_removed_by_residual_off': row['predictions']['S2'] == WEB and row['predictions']['S0'] != WEB,
                    'T1_false_positive_removed_by_pair_shuffle': row['predictions']['T1'] == WEB and row['predictions']['T2'] != WEB,
                    'S2_false_positive_removed_by_pair_true': row['predictions']['S2'] == WEB and row['predictions']['S1'] != WEB}
            {'added_FP': added, 'removed_FP': removed, 'retained_FP': retained}[kind].append(item)
    return {'added': len(added), 'removed': len(removed), 'retained': len(retained),
            'net_increase': len(added) - len(removed),
            'added_by_true_class': dict(Counter(r['true_class'] for r in added)),
            'removed_by_true_class': dict(Counter(r['true_class'] for r in removed))}, added + removed, retained


def write_results(output, records, settings, native_receipts, consistency):
    require(all(r['passed'] for r in native_receipts.values()), 'Native archive reproduction is mandatory')
    metrics = {g: frozen.joint.study.validate_predictions(items, probabilities=False) for g, items in exports(records).items()}
    tables.write_tables(output, metrics)
    comparisons = {}
    folder = output / 'comparisons'; folder.mkdir(exist_ok=True)
    for after, before in settings['comparisons']:
        key = after + '-' + before
        result, changed = compare(records, after, before)
        comparisons[key] = result
        raw.save_predictions(folder / (key + '_changed_samples.jsonl.gz'), changed)
    strata = {}
    for label, subset in (('features_changed', [r for r in records if r['actual_features_changed']]),
                          ('features_unchanged', [r for r in records if not r['actual_features_changed']])):
        strata[label] = {'samples': len(subset), 'comparisons': {}}
        if subset:
            for after, before in (('T1', 'T2'), ('S1', 'S2')):
                key = after + '-' + before
                result, changed = compare(subset, after, before)
                strata[label]['comparisons'][key] = result
                raw.save_predictions(folder / (label + '_' + key + '_changed_samples.jsonl.gz'), changed)
    raw.save_predictions(output / 'sample_diagnostics.jsonl.gz', records)
    web = {g: web_stats(records, g) for g in GROUPS}
    web_folder = output / 'webbased'; web_folder.mkdir(exist_ok=True)
    for group in GROUPS:
        raw.save_predictions(web_folder / (group + '_false_positives.jsonl.gz'),
            [r for r in records if r['label'] != WEB and r['predictions'][group] == WEB])
    inventories = {}
    for after, before in (('T1', 'S2'), ('T1', 'T0'), ('S2', 'S0'), ('T1', 'T2'), ('S1', 'S2')):
        key = after + '-' + before
        result, changed, retained = fp_inventory(records, after, before)
        inventories[key] = result
        raw.save_predictions(web_folder / (key + '_fp_changed_samples.jsonl.gz'), changed)
        raw.save_predictions(web_folder / (key + '_fp_retained_samples.jsonl.gz'), retained)
    require(inventories['T1-S2']['net_increase'] == settings['expected_native_webbased_fp_net_increase'],
            'Previously reported native WebBased FP gap does not reproduce')
    decomposition = {}
    for label in [name for name in LABELS if name != 'WebBased']:
        n = {g: web[g]['fp_by_true_class'][label] for g in GROUPS}
        decomposition[label] = {'native_T1_minus_S2': n['T1'] - n['S2'],
            'fusion_only_T0_minus_S0': n['T0'] - n['S0'],
            'True_residual_net_T1_minus_T0': n['T1'] - n['T0'],
            'Shuffle_residual_net_S2_minus_S0': n['S2'] - n['S0']}
        d = decomposition[label]
        require(d['native_T1_minus_S2'] == d['fusion_only_T0_minus_S0'] +
                d['True_residual_net_T1_minus_T0'] - d['Shuffle_residual_net_S2_minus_S0'], 'FP decomposition identity failed')
    web_result = {'groups': web, 'inventories': inventories, 'net_gap_decomposition_by_source_class': decomposition,
                  'interpretation': 'Net FP differences are not counts of newly added samples; decomposition is algebraic across fixed checkpoints.'}
    raw.write(output / 'webbased_false_positives.json', web_result)
    summary = {'scope': settings['scope'], 'training_epochs': 0, 'optimizer_updates': 0,
               'validation_groups': 6, 'validation_samples_per_group': len(records), 'test_evaluated': False,
               'metrics': metrics, 'comparisons': comparisons, 'feature_strata': strata,
               'unchanged_feature_consistency': consistency, 'native_archive_replay': native_receipts,
               'webbased': web_result, 'candidate_True_macro_f1_unchanged': settings['candidate_validation_macro_f1'],
               'previous_training_ablation': {'delta_pp': settings['previous_training_ablation_delta_pp'],
                   'reference_pp': settings['previous_training_ablation_reference_pp'],
                   'passed': settings['previous_training_ablation_passed'], 'conclusion_changed_by_replay': False}}
    raw.write(output / 'summary.json', summary)
    raw.write(output / 'metrics.json', metrics)
    lines = ['# Seed42 六组固定检查点交叉回放', '',
             f"0 轮训练，0 次优化器更新；六组各 {len(records)} 条验证流，统一 eval 且全部参数冻结。",
             '复用归档冻结验证特征；本轮未执行图编码器，也未评估测试集。',
             'T0/S0 直接将 packet 输入各自检查点训练后的融合层，不注入残差，未替换融合权重。', '',
             '| 组别 | 固定检查点轮次 | 模式 | Macro-F1 | Accuracy |', '| --- | ---: | --- | ---: | ---: |']
    modes = {'T1': '真实配对', 'T2': '固定打乱', 'T0': '关闭残差', 'S1': '真实配对', 'S2': '固定打乱', 'S0': '关闭残差'}
    for g in GROUPS:
        epoch = settings['checkpoint_epochs']['pair_true' if g.startswith('T') else 'pair_shuffle']
        lines.append(f"| {g} | {epoch} | {modes[g]} | {metrics[g]['macro_f1']:.6%} | {metrics[g]['accuracy']:.6%} |")
    lines += ['', 'T1、S2 的预测、指标及摘要与归档逐样本完全一致后，才生成本报告。', '',
              '| 比较（后−前） | Macro-F1 差 pp | 修复 | 退步 | 错误间切换 |', '| --- | ---: | ---: | ---: | ---: |']
    for key, row in comparisons.items():
        lines.append(f"| {key} | {row['delta_macro_f1_pp']:+.6f} | {row['counts']['repaired']} | "
                     f"{row['counts']['regressed']} | {row['change_groups']['wrong_changed']['samples']} |")
    lines += ['', '| 实际特征分组 | 样本 | T1−T2 pp | S1−S2 pp |', '| --- | ---: | ---: | ---: |']
    for name, row in strata.items():
        values = row['comparisons']
        lines.append(f"| {name} | {row['samples']} | " + ('NA | NA |' if not values else
            f"{values['T1-T2']['delta_macro_f1_pp']:+.6f} | {values['S1-S2']['delta_macro_f1_pp']:+.6f} |"))
    lines += ['', '特征未变组的 log probability、残差和逐维融合权重要求逐位一致；见 unchanged_feature_consistency.json。', '',
              '| 组别 | WebBased TP | FP | FN | Recon→Web | Benign→Web | Spoofing→Web |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for g, row in web.items():
        lines.append(f"| {g} | {row['TP']} | {row['FP']} | {row['FN']} | {row['fp_by_true_class']['Recon']} | "
                     f"{row['fp_by_true_class']['Benign']} | {row['fp_by_true_class']['Spoofing']} |")
    old = inventories['T1-S2']
    lines += ['', f"此前 WebBased 误报净增 {old['net_increase']} 例，由新增 {old['added']} 例、消除 {old['removed']} 例组成；两组共同误报 {old['retained']} 例。",
              '全部新增/消除样本保留六组预测、残差和门控诊断，见 webbased/T1-S2_fp_changed_samples.jsonl.gz。',
              '按来源类的恒等分解：T1−S2 = (T0−S0) + (T1−T0) − (S2−S0)。',
              'T0−S0 是两套训练后融合参数在无残差条件的差异；该分解本身不是训练因果归因。', '',
              '| 来源类 | 原净增 FP | 无残差两模型差异 | True 残差净增 | Shuffle 残差净增 |',
              '| --- | ---: | ---: | ---: | ---: |']
    for label, d in decomposition.items():
        lines.append(f"| {label} | {d['native_T1_minus_S2']:+d} | {d['fusion_only_T0_minus_S0']:+d} | "
                     f"{d['True_residual_net_T1_minus_T0']:+d} | {d['Shuffle_residual_net_S2_minus_S0']:+d} |")
    lines += ['', '| 残差对比及样本组 | 数量 | 注入残差 L2 均值 | 相对 Packet 强度 | 融合门控变化均值（绝对值） |',
              '| --- | ---: | ---: | ---: | ---: |']
    for key, active in (('T1-T0', 'T1'), ('S2-S0', 'S2')):
        for kind in ('repaired', 'regressed'):
            row = comparisons[key]['change_groups'][kind]
            if row['samples']:
                v = row[active]
                lines.append(f"| {key} {kind} | {row['samples']} | {v['injected_residual_l2']:.6f} | "
                             f"{v['residual_packet_ratio']:.6f} | {v['gate_mean_abs_change_from_off']:.6f} |")
    if metrics['T0']['macro_f1'] > metrics['T1']['macro_f1']:
        suggestion = 'True 检查点关闭残差后总体更好，应优先做简化模型的匹配训练验证；不能直接将敏感性高分替换成新候选。'
    elif comparisons['T1-T0']['counts']['repaired'] and comparisons['T1-T0']['counts']['regressed']:
        suggestion = '残差同时修复和破坏样本，可结合误报集中程度评估自适应注入候选；本轮不启动 R0/R1/R2 训练。'
    else:
        suggestion = '结合关闭残差后的预测变化与配对敏感性，先核查新增读出的实际利用程度。'
    lines += ['', suggestion, '',
              '本轮是固定模型敏感性诊断，不能替代训练消融。原 True 候选 97.914033% 保留；',
              f"上一轮 True−Shuffle = {settings['previous_training_ablation_delta_pp']:+.9f} pp 未通过 0.05 pp 参考的结论不变。",
              '门控权重表示原逐维融合层中 Flow 的占比，并非尚未训练的残差注入强度 beta。',
              'T0/S0 的 raw_residual_l2 仅记录该检查点原生配对的未注入候选；injected_residual_l2 恒为 0。',
              '所有逐样本诊断在 sample_diagnostics.jsonl.gz；六组预测、八类指标、混淆矩阵、配对变化与来源哈希均保留。']
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return summary
