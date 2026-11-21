"""Fixed-eight-class validation reports at globally selected True/Shuffle checkpoints."""
import json

import diagnose_joint_transition_fusion as change_diagnosis
import joint_transition_diagnostics as diagnostics
import joint_transition_training as training
from xgnid.transition_pair_shuffle import ARMS

raw, frozen = training.raw, training.frozen


def report_group(records, output):
    output.mkdir(parents=True, exist_ok=True)
    if not records:
        result = {'samples': 0, 'metrics': None, 'comparison': None, 'reason': 'empty_fixed_stratum'}
        raw.write(output / 'metrics.json', result)
        return result
    predicted = diagnostics.exports(records)
    metrics = {name: frozen.joint.study.validate_predictions(items, probabilities=False)
               for name, items in predicted.items()}
    comparisons = {}
    pairs = [(ARMS[0], ARMS[1]), *((arm, before) for arm in ARMS for before in ('original_C', 'historical_J3'))]
    for after, before in pairs:
        key = after + '-' + before
        paired, _ = raw.compare(predicted[before], predicted[after])
        structure, changed = change_diagnosis.summarize(records, before, after)
        comparisons[key] = {**paired, **structure,
                            'delta_macro_f1_pp': 100 * (metrics[after]['macro_f1'] - metrics[before]['macro_f1'])}
        raw.save_predictions(output / (key + '_changed_samples.jsonl.gz'), changed)
    result = {'samples': len(records), 'metrics': metrics, 'comparisons': comparisons,
              'selection_scope': 'best_overall_validation_epoch_including_zero_earliest_tie',
              'macro_f1_scope': 'fixed_eight_classes_including_zero_support_classes'}
    raw.write(output / 'metrics.json', result)
    raw.write(output / 'diagnostic_groups.json', diagnostics.aggregate(records, ARMS[1], ARMS[0]))
    diagnostics.write_tables(output, metrics)
    return result


def write_results(output, records, rows, state, settings, history):
    groups = {}
    validation = output / 'validation'
    validation.mkdir(exist_ok=True)
    for key, subset in (
            ('all', records), ('shuffleable', [r for r in records if r['shuffleable']]),
            ('unshuffleable', [r for r in records if not r['shuffleable']])):
        groups[key] = report_group(subset, validation / key)
    exported = diagnostics.exports(records)
    for name in ARMS:
        raw.save_predictions(output / name / 'best_validation_predictions.jsonl.gz', exported[name])
    raw.save_predictions(validation / 'validation_predictions.jsonl.gz', records)
    coverage = raw.read(output / 'coverage.json')
    all_result = groups['all']
    delta = all_result['comparisons']['pair_true-pair_shuffle']['delta_macro_f1_pp']
    reference = settings['investment_reference_pp']
    budgets = []
    for budget in sorted(set([n for n in (30, 100, 150, 200) if n <= settings['epochs']] + [settings['epochs']])):
        selected = {}
        for name in ARMS:
            best = max((r for r in state['history'] if r['epoch'] <= budget),
                       key=lambda r: r['validation'][name]['macro_f1'])
            selected[name] = {'epoch': best['epoch'], 'macro_f1': best['validation'][name]['macro_f1']}
        budgets.append({'epoch_budget': budget, 'selected': selected, 'true_minus_shuffle_pp':
                        100 * (selected[ARMS[0]]['macro_f1'] - selected[ARMS[1]]['macro_f1'])})
    raw.write(output / 'budget_comparison.json', {'partition': 'validation', 'budgets': budgets})
    summary = {'partition': 'validation', 'test_evaluated': False, 'metrics': all_result['metrics'],
               'comparisons': all_result['comparisons'], 'strata': groups,
               'selected_epochs': {name: best['epoch'] for name, best in state['best'].items()},
               'investment_reference': {'delta_pp': delta, 'reference_pp': reference, 'passed': delta >= reference,
                                        'statistical_significance_claimed': False},
               'coverage': coverage, 'historical_J3_is_external_reference_only': True,
               'scope': 'explicit_pairing_in_the_added_readout_with_original_encoder_temporal_information_retained'}
    raw.write(output / 'summary.json', summary)
    raw.write(output / 'selection.json', {'binding': state['binding'], 'test_evaluated': False,
        'selected': {name: {k: v for k, v in best.items() if k != 'state'} for name, best in state['best'].items()}})
    lines = ['# Seed42 真实连续配对与流内第二候选重新配对', '',
             '两组从相同原 C、相同零输出读出和融合参数开始，同时训练；历史 J3 仅作外部参考。',
             '图编码器及原跨层模块冻结；原图编码器已有的时序信息仍保留。', '',
             '| 组别 | 最佳轮 | 验证 Macro-F1 | 相对原 C（百分点） | 相对历史 J3（百分点） |',
             '| --- | ---: | ---: | ---: | ---: |']
    metrics = all_result['metrics']
    for name, label in zip(ARMS, ('True', 'Shuffle')):
        lines.append(f"| {label} | {state['best'][name]['epoch']} | {metrics[name]['macro_f1']:.6%} | "
                     f"{100*(metrics[name]['macro_f1']-metrics['original_C']['macro_f1']):+.6f} | "
                     f"{100*(metrics[name]['macro_f1']-metrics['historical_J3']['macro_f1']):+.6f} |")
    lines += ['', '| 分组 | 样本 | True F1 | Shuffle F1 | True−Shuffle pp | 修复 | 退步 |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for key, label in (('all', '总体'), ('shuffleable', '可打乱：至少两个真实片段'),
                       ('unshuffleable', '不可打乱：零或一个片段')):
        result = groups[key]
        if result['metrics'] is None:
            lines.append(f'| {label} | 0 | NA | NA | NA | NA | NA |')
            continue
        comparison = result['comparisons']['pair_true-pair_shuffle']
        lines.append(f"| {label} | {result['samples']} | {result['metrics'][ARMS[0]]['macro_f1']:.6%} | "
                     f"{result['metrics'][ARMS[1]]['macro_f1']:.6%} | {comparison['delta_macro_f1_pp']:+.6f} | "
                     f"{comparison['counts']['repaired']} | {comparison['counts']['regressed']} |")
    lines += ['', '所有分组使用总体验证选出的检查点，不分别按分组选最佳；分组 Macro-F1 始终按固定八类计算。',
              '', '| 分区 | 总流数 | 可打乱流 | 覆盖率 | 改变索引的片段 | 第二项特征实际改变 | 特征改变比例 |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for part in ('train', 'val'):
        c = coverage[part]
        ratio = c['feature_change_fraction_of_changed_indices']
        lines.append(f"| {part} | {c['samples']} | {c['shuffleable_flows']} | {c['shuffleable_flow_fraction']:.6%} | "
                     f"{c['changed_pair_indices']} | {c['changed_right_features']} | "
                     + ('NA' if ratio is None else f'{ratio:.6%}') + ' |')
    lines += ['', '| 分区 | 真实类别 | 总流数 | 可打乱流 | 覆盖率 | 第二项特征改变比例 |',
              '| --- | --- | ---: | ---: | ---: | ---: |']
    for part in ('train', 'val'):
        for label, c in coverage[part]['by_class'].items():
            ratio = c['feature_change_fraction_of_changed_indices']
            lines.append(f"| {part} | {label} | {c['samples']} | {c['shuffleable_flows']} | "
                         + ('NA' if c['shuffleable_flow_fraction'] is None else f"{c['shuffleable_flow_fraction']:.6%}")
                         + ' | ' + ('NA' if ratio is None else f'{ratio:.6%}') + ' |')
    lines += ['', '| 预算轮数 | True（最佳轮） | Shuffle（最佳轮） | True−Shuffle pp |',
              '| ---: | ---: | ---: | ---: |']
    for row in budgets:
        lines.append(f"| {row['epoch_budget']} | " + ' | '.join(
            f"{row['selected'][name]['macro_f1']:.6%} ({row['selected'][name]['epoch']})" for name in ARMS)
                     + f" | {row['true_minus_shuffle_pp']:+.6f} |")
    if delta >= reference:
        conclusion = '达到事前资源投入参考，可结合八类得失安排后续独立主干种子验证；本轮不自动启动额外实验。'
    elif delta > 0:
        conclusion = 'True 有小幅优势但未达到事前投入参考，尚不足以继续把收益主要归因于真实相邻配对。'
    else:
        conclusion = 'True 没有优势，不能继续将本轮收益归因于真实相邻配对。'
    lines += ['', f'True−Shuffle = {delta:+.9f} pp；事前参考为 ≥{reference:.2f} pp，按未四舍五入值判断。',
              conclusion, '门槛只用于资源决策，不作统计显著性声明。',
              '本消融仅检验新增读出的显式连续配对；不能据此排除原图编码器保留的时序信息。',
              '零/单片段流在结构层面保持原样，但两组训练后的公共参数不同，最终预测仍可能不同。',
              '逐类 Precision/Recall/F1、混淆矩阵和配对变化样本见 validation/all、shuffleable、unshuffleable。',
              '门控、残差、两尺度注意力及片段两侧索引见 validation/validation_predictions.jsonl.gz。',
              '映射由 partition、sample_id、片段数及固定独立种子生成，不使用标签；所有片段第一项固定，第二项集合与次数不变。']
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return summary
