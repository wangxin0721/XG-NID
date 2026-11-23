"""Per-backbone paired results and sample-standard-deviation stability summaries."""
import csv
import math
from statistics import mean, stdev

import joint_transition_diagnostics as diagnostics
import diagnose_joint_transition_fusion as changes
import fixed_transition_training as training
from xgnid.fixed_transition_stability import ARMS, MODELS

raw, frozen, require = training.raw, training.frozen, training.require
LABELS = diagnostics.LABELS
PAIRS = (('joint_ordered', 'base_C'), ('joint_ordered', 'joint_single'),
         ('joint_ordered', 'frozen_single'), ('joint_single', 'frozen_single'))
METRICS = ('accuracy', 'macro_f1', 'weighted_f1')


def independently_verify(records, metrics, comparisons):
    """Recompute from exported integer predictions, independently of sklearn/study helpers."""
    n = len(records)
    require(n > 0 and len({r['sample_id'] for r in records}) == n, 'Invalid audit sample identities')
    for name in MODELS:
        cm = [[0]*8 for _ in LABELS]
        for row in records:
            cm[row['label']][row['predictions'][name]] += 1
        require(cm == metrics[name]['confusion_matrix'], name + ' confusion audit failed')
        f1s, supports = [], []
        for i, label in enumerate(LABELS):
            tp, support, predicted = cm[i][i], sum(cm[i]), sum(row[i] for row in cm)
            p, r = tp/predicted if predicted else 0, tp/support if support else 0
            f = 2*tp/(support+predicted) if support+predicted else 0
            saved = metrics[name]['per_class'][label]
            require(all(math.isclose(x, saved[k], rel_tol=0, abs_tol=2e-15)
                        for x, k in ((p, 'precision'), (r, 'recall'), (f, 'f1-score')))
                    and support == saved['support'], name + ' per-class audit failed')
            f1s.append(f); supports.append(support)
        expected = (sum(cm[i][i] for i in range(8))/n, mean(f1s), sum(f*s for f, s in zip(f1s, supports))/n)
        require(all(math.isclose(v, metrics[name][k], rel_tol=0, abs_tol=2e-15)
                    for k, v in zip(METRICS, expected)), name + ' aggregate metric audit failed')
    for after, before in PAIRS:
        key, counts = after+'-'+before, dict(repaired=0, regressed=0, wrong_changed=0)
        for row in records:
            a, b, y = row['predictions'][after], row['predictions'][before], row['label']
            if a != b:
                counts['repaired' if a == y else 'regressed' if b == y else 'wrong_changed'] += 1
        require(all(counts[k] == comparisons[key]['groups'][k]['count'] for k in counts), 'Paired audit failed: '+key)
    return {'passed': True, 'samples': n, 'confusion_matrices': len(MODELS), 'paired_comparisons': len(PAIRS),
            'method': 'integer confusion counts and direct precision/recall/harmonic-F1 formulas'}


def budget_rows(histories, settings):
    result = []
    for budget in (30, 100, 150, 200):
        if budget > settings['epochs']:
            continue
        selected = {name: max((r for r in histories[name] if r['epoch'] <= budget),
                    key=lambda r:r['validation']['macro_f1']) for name in ARMS}
        result.append({'epoch_budget': budget, 'selected': {name: {'epoch': row['epoch'],
            'macro_f1': row['validation']['macro_f1']} for name, row in selected.items()},
            'delta_macro_f1_pp': {after+'-'+before: 100*(selected[after]['validation']['macro_f1']-
                selected[before]['validation']['macro_f1']) for after, before in PAIRS if before in selected}})
    return result


def write_seed_results(output, records, rows, metadata, source_binding, settings, histories, stage, seed):
    folder = output / 'validation'; folder.mkdir(exist_ok=True)
    predicted = diagnostics.exports(records)
    require(tuple(predicted) == MODELS, 'Unexpected stability report models/order')
    metrics = {name: frozen.joint.study.validate_predictions(items, rows, probabilities=False)
               for name, items in predicted.items()}
    comparisons = {}
    for after, before in PAIRS:
        key = after+'-'+before
        paired, _ = raw.compare(predicted[before], predicted[after])
        grouped, changed = changes.summarize(records, before, after)
        comparisons[key] = {**paired, **grouped,
            'delta_macro_f1_pp':100*(metrics[after]['macro_f1']-metrics[before]['macro_f1'])}
        raw.save_predictions(folder / (key+'_changed_samples.jsonl.gz'), changed)
    raw.save_predictions(folder / 'validation_predictions.jsonl.gz', records)
    for name in MODELS:
        (output / name).mkdir(exist_ok=True)
        raw.save_predictions(output / name / 'best_validation_predictions.jsonl.gz', predicted[name])
        require(metrics[name] == metadata[name]['best_validation'], 'Selected metrics differ: '+name)
    diagnostics.write_tables(folder, metrics)
    class_changes = {a+'-'+b: {label:{key:metrics[a]['per_class'][label][key]-metrics[b]['per_class'][label][key]
        for key in ('precision','recall','f1-score')} for label in LABELS} for a,b in PAIRS}
    raw.write(folder / 'per_class_changes.json', class_changes)
    raw.write(folder / 'paired_diagnostic_groups.json', {a+'-'+b:diagnostics.aggregate(records,b,a) for a,b in PAIRS})
    raw.write(output / 'budget_comparison.json', {'partition':'validation', 'budgets':budget_rows(histories,settings)})
    receipt = independently_verify(records, metrics, comparisons)
    raw.write(output / 'independent_verification.json', receipt)
    summary = {'backbone_seed':seed, 'readout_seed':42, 'sampling_seed':42, 'partition':'validation',
        'stage':stage, 'test_evaluated':False, 'source_binding':source_binding, 'metrics':metrics,
        'comparisons':comparisons, 'model_metadata':metadata, 'independent_verification':receipt,
        'new_updates_total':sum(v['new_updates'] for v in metadata.values())}
    raw.write(output / 'summary.json', summary)
    lines = [f'# 固定注入稳定性：主干 seed{seed}', '',
        'seed42 是开发阶段锁定归档；seed43/44 是本阶段新增复验。读出及采样 seed 均固定为42。',
        '三个训练组从此主干同一个原 C 出发；不实例化 beta 门控。本轮不单独归因于真实连续配对。', '',
        '| 模型 | 最佳轮 | Accuracy | Macro-F1 | Weighted-F1 | 末轮 Macro-F1 | 本阶段新增更新 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name in MODELS:
        m, meta = metrics[name], metadata[name]
        lines.append(f"| {name} | {meta['best_epoch']} | {m['accuracy']:.6%} | {m['macro_f1']:.6%} | "
            f"{m['weighted_f1']:.6%} | {meta['last_validation']['macro_f1']:.6%} | {meta['new_updates']} |")
    lines += ['', '| 配对 | Macro-F1 差（百分点） | 修复 | 退步 | 错误间变化 |','| --- | ---: | ---: | ---: | ---: |']
    for key,row in comparisons.items():
        lines.append(f"| {key} | {row['delta_macro_f1_pp']:+.6f} | "+' | '.join(
            str(row['groups'][k]['count']) for k in ('repaired','regressed','wrong_changed'))+' |')
    lines += ['', '| 类别 | base_C F1 | frozen_single F1 | joint_single F1 | joint_ordered F1 | Ordered−FrozenSingle pp |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for label in LABELS:
        lines.append('| '+label+' | '+' | '.join(f"{metrics[n]['per_class'][label]['f1-score']:.6%}" for n in MODELS)+
            f" | {100*class_changes['joint_ordered-frozen_single'][label]['f1-score']:+.6f} |")
    lines += ['', '| 模型 | 推断总参数 | 本阶段可训练参数 | 本组优化计算秒 |', '| --- | ---: | ---: | ---: |']
    for name in MODELS:
        meta=metadata[name]; seconds=meta['training_seconds']
        lines.append(f"| {name} | {meta['inference_parameters']} | {meta['trainable_parameters']} | "+
                     ('归档未记录' if seconds is None else f'{seconds:.3f}')+' |')
    strong = comparisons['joint_ordered-frozen_single']['delta_macro_f1_pp']
    if comparisons['joint_ordered-joint_single']['delta_macro_f1_pp'] > 0 and strong <= 0:
        lines += ['', 'Ordered 只优于联合 Single，未优于强 Frozen Single，对强对照的优势不足。']
    elif strong < 0:
        lines += ['', 'Ordered 相对强 Frozen Single 为负，保留本主干的负向结果，不择高替换。']
    lines += ['', '重点类别 WebBased、Recon、Spoofing、BruteForce 的 P/R/F1、错分方向和全部变化样本见 validation。',
        '优化计算秒只计已提交更新的前向、NLL、反向与 Adam；CUDA 前后同步，不含探测、缓存、验证、磁盘发布或失败回滚的额外尝试。历史没有分组计时时保留空值，不按总耗时分摊。',
        '最佳轮包含第0轮，同分取最早；完整200轮，无结果驱动的提前停止。统计显著性与真实相邻配对的额外贡献不由本轮推断。',
        '此前 beta 门控和 True/Shuffle 未通过门槛的结论保持不变；本轮不自动测试或探索新结构。']
    (output / 'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary


def stats(values):
    return {'n':len(values), 'mean':mean(values), 'sample_std':stdev(values) if len(values)>1 else None, 'values':values}


def aggregate(summaries, output):
    require(set(summaries)=={42,43,44}, 'All three backbones are required before final aggregation')
    output.mkdir(exist_ok=True)
    bindings=[summaries[s]['source_binding'] for s in (42,43,44)]
    require(len({b['frozen_c_sha256'] for b in bindings})==3
            and len({b['split_id'] for b in bindings})==1 and len({b['manifest_sha256'] for b in bindings})==1,
            'Backbones are not distinct original C states on the same split')
    def scope(seeds):
        models={name:{metric:stats([summaries[s]['metrics'][name][metric] for s in seeds]) for metric in METRICS}
                for name in MODELS}
        pairs={}
        for a,b in PAIRS:
            values=[100*(summaries[s]['metrics'][a]['macro_f1']-summaries[s]['metrics'][b]['macro_f1']) for s in seeds]
            pairs[a+'-'+b]={**stats(values),'all_positive':all(v>0 for v in values),
                'positive_seeds':[s for s,v in zip(seeds,values) if v>0],
                'negative_seeds':[s for s,v in zip(seeds,values) if v<0], 'zero_seeds':[s for s,v in zip(seeds,values) if v==0]}
        classes={name:{label:{key:stats([summaries[s]['metrics'][name]['per_class'][label][key] for s in seeds])
            for key in ('precision','recall','f1-score')} for label in LABELS} for name in MODELS}
        return {'backbone_seeds':list(seeds),'metrics':models,'paired_macro_f1_differences_pp':pairs,'per_class':classes}
    all_three, new_two=scope((42,43,44)),scope((43,44))
    summary={'scope':'三个独立主干上的配对验证；非所有模块以三个随机种子完整重训',
        'seed42_role':'development_stage_locked_archive', 'new_replication_seeds':[43,44],
        'readout_seed':42,'sampling_seed':42,'test_evaluated':False,
        'per_backbone':{str(s):summaries[s] for s in (42,43,44)},'all_three':all_three,'new_replications_only':new_two,
        'new_training_runs':6,'new_updates_total':sum(summaries[s]['new_updates_total'] for s in (43,44)),
        'previous_negative_ablation_conclusions_changed':False}
    raw.write(output / 'summary.json',summary)
    lines=['# 固定注入报文转移模型：独立主干稳定性验证', '', summary['scope']+'。',
        'seed42 复用事前锁定开发归档，主要新增配对证据来自 seed43、44；不从同架构重跑中重新择高。',
        'seed42 的其他同结构重跑仅保存在 seed42_import/rerun_context.json 作为运行差异背景，不参与选模或三主干统计。', '',
        '| 模型 | seed42 开发锁定 | seed43 新增 | seed44 新增 | 三主干均值±样本标准差 | 新增两主干均值±样本标准差 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name in MODELS:
        both=[sc['metrics'][name]['macro_f1'] for sc in (all_three,new_two)]
        lines.append('| '+name+' | '+' | '.join(f"{summaries[s]['metrics'][name]['macro_f1']:.6%}" for s in (42,43,44))+
            ' | '+' | '.join(f"{v['mean']:.6%} ± {v['sample_std']:.6%}" for v in both)+' |')
    lines += ['', '| 配对差（百分点） | seed42 | seed43 | seed44 | 三主干均值±样本标准差 | 新增两主干均值±样本标准差 |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for a,b in PAIRS:
        key=a+'-'+b; whole=all_three['paired_macro_f1_differences_pp'][key]; new=new_two['paired_macro_f1_differences_pp'][key]
        lines.append('| '+key+' | '+' | '.join(f'{v:+.6f}' for v in whole['values'])+
            f" | {whole['mean']:+.6f} ± {whole['sample_std']:.6f} | {new['mean']:+.6f} ± {new['sample_std']:.6f} |")
        lines += []
    lines += ['']
    for a,b in PAIRS:
        key=a+'-'+b; row=all_three['paired_macro_f1_differences_pp'][key]
        if row['all_positive']:
            lines.append(key+'：三个主干配对差均为正，可描述为方向一致。')
        else:
            lines.append(key+f"：存在波动，负值主干 {row['negative_seeds']}，零差主干 {row['zero_seeds']}，不能描述为三组均正。")
    if all_three['paired_macro_f1_differences_pp']['joint_ordered-joint_single']['all_positive'] and not all_three[
            'paired_macro_f1_differences_pp']['joint_ordered-frozen_single']['all_positive']:
        lines += ['对联合 Single 的优势未扩展为对强 Frozen Single 的一致优势，必须明确报告强对照优势不足。']
    for metric,title in (('accuracy','Accuracy'),('weighted_f1','Weighted-F1')):
        lines += ['', '| '+title+' | seed42 开发锁定 | seed43 新增 | seed44 新增 | 三主干均值±样本标准差 | 新增两主干均值±样本标准差 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |']
        for name in MODELS:
            both=[sc['metrics'][name][metric] for sc in (all_three,new_two)]
            lines.append('| '+name+' | '+' | '.join(f"{summaries[s]['metrics'][name][metric]:.6%}" for s in (42,43,44))+
                ' | '+' | '.join(f"{v['mean']:.6%} ± {v['sample_std']:.6%}" for v in both)+' |')
    lines += ['', 'Accuracy、Weighted-F1 及八类 P/R/F1 的同口径均值和样本标准差保存在 summary.json；逐主干原始结果、全部修复/退步与预算窗口分别保留。',
        '样本标准差使用 n−1 分母；配对差先在同一主干内相减，再求均值与样本标准差。',
        '本轮只验证保留模型的整体收益，不单独证明真实连续配对的贡献；不以最高 seed 代表方法，不作统计显著性声明。',
        '此前 beta 门控及 True/Shuffle 未通过门槛的结论保持不变。到此收束验证集汇总，不自动运行测试集或新结构。']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with (output/'model_metrics.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(('scope','model','metric','mean','sample_std','values'))
        for tag,sc in (('all_three',all_three),('new_replications_only',new_two)):
            for name,metric_map in sc['metrics'].items():
                for metric,value in metric_map.items():
                    writer.writerow((tag,name,metric,value['mean'],value['sample_std'],value['values']))
    with (output/'paired_differences_pp.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(('scope','comparison','mean_pp','sample_std_pp','all_positive','values_pp'))
        for tag,sc in (('all_three',all_three),('new_replications_only',new_two)):
            for name,v in sc['paired_macro_f1_differences_pp'].items():
                writer.writerow((tag,name,v['mean'],v['sample_std'],v['all_positive'],v['values']))
    return summary
