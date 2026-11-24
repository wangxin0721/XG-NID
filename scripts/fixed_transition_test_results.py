"""Fixed-test exports, direct prediction audits and publication-ready experiment tables."""
import csv
from statistics import mean, stdev

import fixed_transition_test_lock as locked

raw, frozen, require = locked.raw, locked.frozen, locked.require
MODELS, LABELS = locked.MODELS, locked.LABELS
PAIRS = locked.results.PAIRS
METRICS = ('accuracy', 'macro_f1', 'weighted_f1')


def write_seed(output, records, rows, seed, manifest, runtime):
    require(all(r['partition'] == 'test' for r in records), 'Expected only fixed test predictions')
    folder = output / 'test'
    folder.mkdir(exist_ok=True)
    predictions = locked.results.diagnostics.exports(records)
    require(tuple(predictions) == MODELS, 'Fixed-test model ordering differs')
    metrics = {name: frozen.joint.study.validate_predictions(items, rows, probabilities=False)
               for name, items in predictions.items()}
    comparisons = {}
    for after, before in PAIRS:
        key = after + '-' + before
        paired, _ = raw.compare(predictions[before], predictions[after])
        grouped, changes = locked.results.changes.summarize(records, before, after)
        comparisons[key] = {**paired, **grouped,
            'delta_macro_f1_pp': 100 * (metrics[after]['macro_f1'] - metrics[before]['macro_f1'])}
        raw.save_predictions(folder / (key + '_changed_samples.jsonl.gz'), changes)
    verification = locked.results.independently_verify(records, metrics, comparisons)
    raw.save_predictions(folder / 'predictions.jsonl.gz', records)
    for name, items in predictions.items():
        raw.save_predictions(folder / (name + '_predictions.jsonl.gz'), items)
    locked.results.diagnostics.write_tables(folder, metrics)
    raw.write(folder / 'paired_diagnostic_groups.json', {a + '-' + b:
        locked.results.diagnostics.aggregate(records, b, a) for a, b in PAIRS})
    raw.write(folder / 'per_class_changes.json', {a + '-' + b: {label: {
        k: 100 * (metrics[a]['per_class'][label][k] - metrics[b]['per_class'][label][k])
        for k in ('precision', 'recall', 'f1-score')} for label in LABELS} for a, b in PAIRS})
    summary = {'backbone_seed': seed, 'partition': 'test', 'test_evaluated': True,
        'readout_seed': 42, 'sampling_seed': 42, 'role': manifest['role'],
        'new_training_runs': 0, 'new_optimizer_updates': 0, 'source_binding': manifest['source_binding'],
        'model_metadata': manifest['models'], 'metrics': metrics, 'comparisons': comparisons,
        'prediction_digests': {n: frozen.joint.prediction_digest(p) for n, p in predictions.items()},
        'independent_verification': verification, 'runtime': runtime}
    raw.write(output / 'summary.json', summary)
    raw.write(output / 'independent_verification.json', verification)
    lines = ['# 固定测试：主干 seed' + str(seed), '',
        '全部模型均在测试前锁定，不训练、不按测试分数重新选择。', '',
        '| 模型 | 验证最佳轮 | Accuracy | Macro-F1 | Weighted-F1 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name in MODELS:
        lines.append('| ' + name + ' | ' + str(manifest['models'][name]['epoch']) + ' | ' +
                     ' | '.join(f'{metrics[name][k]:.6%}' for k in METRICS) + ' |')
    lines += ['', '| 配对 | Macro-F1差（百分点） | 修复 | 退步 | 错误间变化 |',
              '| --- | ---: | ---: | ---: | ---: |']
    for key, v in comparisons.items():
        lines.append(f"| {key} | {v['delta_macro_f1_pp']:+.6f} | " + ' | '.join(
            str(v['groups'][k]['count']) for k in ('repaired', 'regressed', 'wrong_changed')) + ' |')
    lines += ['', '| 类别 | 原C F1 | Frozen Single F1 | Joint Single F1 | Ordered F1 | Ordered−Frozen pp |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for label in LABELS:
        lines.append('| ' + label + ' | ' + ' | '.join(f"{metrics[n]['per_class'][label]['f1-score']:.6%}"
            for n in MODELS) + f" | {100*(metrics['joint_ordered']['per_class'][label]['f1-score']-metrics['frozen_single']['per_class'][label]['f1-score']):+.6f} |")
    lines += ['', '八类 P/R/F1、混淆矩阵及全部变化样本保存在 test/。负值和类别退步原样保留。']
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return summary


def stats(values):
    return {'values': values, 'mean': mean(values), 'sample_std': stdev(values)}


def aggregate(summaries, output, protocol, negative_ablations):
    require(set(summaries) == {42, 43, 44}, 'All twelve fixed-test results required')
    require(all(s['partition'] == 'test' and s['new_optimizer_updates'] == 0
                and s['independent_verification']['passed'] for s in summaries.values()), 'Invalid fixed-test scope')
    require(len({s['source_binding']['frozen_c_sha256'] for s in summaries.values()}) == 3,
            'Independent backbone identities overlap')
    def scope(seeds):
        pairs = {}
        for a, b in PAIRS:
            values = [100 * (summaries[s]['metrics'][a]['macro_f1'] - summaries[s]['metrics'][b]['macro_f1']) for s in seeds]
            pairs[a + '-' + b] = {**stats(values), 'all_positive': all(v > 0 for v in values),
                'positive_seeds': [s for s, v in zip(seeds, values) if v > 0],
                'negative_seeds': [s for s, v in zip(seeds, values) if v < 0],
                'zero_seeds': [s for s, v in zip(seeds, values) if v == 0]}
        return {'backbone_seeds': list(seeds), 'metrics': {n: {k: stats([summaries[s]['metrics'][n][k] for s in seeds])
            for k in METRICS} for n in MODELS}, 'paired_macro_f1_differences_pp': pairs,
            'per_class': {n: {label: {k: stats([summaries[s]['metrics'][n]['per_class'][label][k] for s in seeds])
                for k in ('precision', 'recall', 'f1-score')} for label in LABELS} for n in MODELS}}
    summary = {'partition': 'test', 'test_evaluated': True, 'models_evaluated': 12,
        'new_training_runs': 0, 'new_training_epochs': 0, 'new_optimizer_updates': 0,
        'readout_seed': 42, 'sampling_seed': 42, 'seed42_role': 'development_stage_locked_archive',
        'scope': '三个独立主干上的固定模型配对测试；读出与采样seed固定42',
        'test_reselection_permitted': False, 'historical_test_split_used': True,
        'source_result_commit': locked.SNAPSHOT, 'evaluation_code_commit': protocol['code_commit'],
        'per_backbone': {str(s): summaries[s] for s in (42, 43, 44)},
        'all_three': scope((42, 43, 44)), 'new_replications_only': scope((43, 44)),
        'previous_negative_ablations': negative_ablations,
        'previous_negative_ablation_conclusions_changed': False}
    output.mkdir(parents=True, exist_ok=True)
    raw.write(output / 'summary.json', summary)
    lines = ['# 固定注入报文转移模型：固定测试与论文实验表', '', summary['scope'] + '。',
        'seed42 为开发阶段锁定归档；seed43、44 为新增独立主干复验。每个模型评估相同13,448条测试流。',
        '该测试划分已有历史评估使用。本轮在首次加载测试图前锁定全部12个验证最佳检查点；测试后不重选检查点、不调参。',
        '本轮新增训练0次、优化器更新0次。98.013342%是seed43历史验证Macro-F1，不能作为测试分数。', '']
    for metric in METRICS:
        lines += [f'## 固定测试 {metric}', '',
            '| 模型 | seed42 开发锁定 | seed43 | seed44 | 三主干均值±样本标准差 | 新增43/44均值±样本标准差 |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
        for name in MODELS:
            lines.append('| ' + name + ' | ' + ' | '.join(f"{summaries[s]['metrics'][name][metric]:.6%}" for s in (42, 43, 44)) +
                ' | ' + ' | '.join(f"{summary[tag]['metrics'][name][metric]['mean']:.6%} ± {summary[tag]['metrics'][name][metric]['sample_std']:.6%}"
                                  for tag in ('all_three', 'new_replications_only')) + ' |')
        lines += ['']
    lines += ['## 配对差与整体方法收益', '', '| 比较（百分点） | seed42 | seed43 | seed44 | 三主干均值±样本标准差 | 新增43/44均值±样本标准差 |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for key, value in summary['all_three']['paired_macro_f1_differences_pp'].items():
        new = summary['new_replications_only']['paired_macro_f1_differences_pp'][key]
        lines.append('| ' + key + ' | ' + ' | '.join(f'{v:+.6f}' for v in value['values']) +
            f" | {value['mean']:+.6f} ± {value['sample_std']:.6f} | {new['mean']:+.6f} ± {new['sample_std']:.6f} |")
    lines += ['']
    for key, value in summary['all_three']['paired_macro_f1_differences_pp'].items():
        lines.append(key + ('：三个主干配对差均为正，方向一致。' if value['all_positive'] else
            '：存在波动；负值主干' + str(value['negative_seeds']) + '，零差主干' + str(value['zero_seeds']) + '。'))
    strong = summary['all_three']['paired_macro_f1_differences_pp']['joint_ordered-frozen_single']
    if any(v <= 0 for v in strong['values']):
        lines += ['相对强对照Frozen Single的优势未在三个主干全部重复，需保留这一限制。']
    lines += ['', '样本标准差使用n−1分母，配对差先在同一主干内计算。整体差值不能单独归因为真实连续配对；不作统计显著性声明。',
              '', '## 类别取舍：Ordered相对Frozen Single的F1差', '',
              '| 类别 | seed42（百分点） | seed43（百分点） | seed44（百分点） |', '| --- | ---: | ---: | ---: |']
    for label in LABELS:
        lines.append('| ' + label + ' | ' + ' | '.join(f"{100*(summaries[s]['metrics']['joint_ordered']['per_class'][label]['f1-score']-summaries[s]['metrics']['frozen_single']['per_class'][label]['f1-score']):+.6f}"
            for s in (42, 43, 44)) + ' |')
    lines += ['', '全部八类P/R/F1、混淆矩阵、修复/退步及错分方向见各seed/test目录；重点检查WebBased、Recon、Spoofing、BruteForce。',
        '', '## 保留的失败消融（seed42验证集，未追加测试）', '',
        '| 消融 | 保留结论 | 证据 |', '| --- | --- | --- |',
        '| True vs Shuffle | 未通过事前继续投入门槛，不能据本次整体收益宣称真实相邻配对已获独立支持 | transition_pair_shuffle_s42_v1 |',
        '| 标量β门控 | 逐流门控候选未满足约定继续条件，保留固定注入 | residual_injection_s42_v1 |',
        '', '相关原始汇总与来源哈希已随协议归档。到此结束固定模型评价，不自动进入新训练、调参或结构探索。']
    text = '\n'.join(lines) + '\n'
    for filename in ('report.md', 'paper_tables.md'):
        (output / filename).write_text(text, encoding='utf-8')
    with (output / 'model_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(('scope', 'model', 'metric', 'mean', 'sample_std'))
        for scope_name in ('all_three', 'new_replications_only'):
            for name in MODELS:
                for metric in METRICS:
                    value = summary[scope_name]['metrics'][name][metric]
                    writer.writerow((scope_name, name, metric, value['mean'], value['sample_std']))
    with (output / 'paired_differences_pp.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(('scope', 'comparison', 'values_pp', 'mean_pp', 'sample_std_pp', 'all_positive'))
        for scope_name in ('all_three', 'new_replications_only'):
            for name, value in summary[scope_name]['paired_macro_f1_differences_pp'].items():
                writer.writerow((scope_name, name, value['values'], value['mean'], value['sample_std'], value['all_positive']))
    return summary
