"""Integer-prediction audits, fixed teacher subgroups and preregistered decisions."""
import csv
import math
from statistics import mean, stdev
import torch

import conditional_complement_training as training
from xgnid.conditional_complement import ARMS, ALIASES

raw, require = training.raw, training.require
LABELS = tuple(training.transaction.original.LABELS)
GROUPS = ('only_flow_correct','only_packet_correct','both_correct','both_wrong')
PAIRS = (('cc_t0','source_ordered'),('cc_t1','cc_t0'),('cc_t2','cc_t1'),
         ('cc_t3','cc_t1'),('cc_t4','cc_t2'),('cc_t4','cc_t3'))


def group(y, flow, packet):
    return GROUPS[0 if flow==y and packet!=y else 1 if packet==y and flow!=y else 2 if flow==y else 3]


def metrics(rows, name):
    if not rows: return None
    cm=[[0]*8 for _ in LABELS]
    for r in rows:
        cm[r['label']][r['predictions'][name]]+=1
    classes={}
    for i,label in enumerate(LABELS):
        tp=cm[i][i];support=sum(cm[i]);predicted=sum(row[i] for row in cm)
        classes[label]={'precision':tp/predicted if predicted else 0.,'recall':tp/support if support else 0.,
            'f1-score':2*tp/(support+predicted) if support+predicted else 0.,'support':support}
    n=len(rows)
    return {'accuracy':sum(cm[i][i] for i in range(8))/n,'macro_f1':mean(v['f1-score'] for v in classes.values()),
        'weighted_f1':sum(v['f1-score']*v['support'] for v in classes.values())/n,
        'per_class':classes,'confusion_matrix':cm,'samples':n}


def check_metric(actual, registered):
    require(actual['confusion_matrix']==registered['confusion_matrix'],'Confusion matrix audit differs')
    for key in ('accuracy','macro_f1','weighted_f1'):
        require(math.isclose(actual[key],registered[key],rel_tol=0,abs_tol=2e-15),'Metric audit differs: '+key)
    for label in LABELS:
        for key in ('precision','recall','f1-score','support'):
            require(math.isclose(actual['per_class'][label][key],registered['per_class'][label][key],rel_tol=0,abs_tol=2e-15),
                    'Class audit differs: '+label+'/'+key)


def quantiles(values):
    values=[v for v in values if v is not None]
    if not values: return None
    t=torch.tensor(values,dtype=torch.float64)
    return {'mean':float(t.mean()),'p0_p25_p50_p75_p100':t.quantile(torch.tensor([0.,.25,.5,.75,1.],dtype=t.dtype)).tolist()}


def subgroup(rows, name):
    result={}
    for key in GROUPS:
        subset=[r for r in rows if r['teacher_group']==key]
        correct=sum(r['predictions'][name]==r['label'] for r in subset)
        # The term destruction is reserved for groups with at least one correct teacher.
        result[key]={'count':len(subset),'fusion_correct':correct,
            'fusion_accuracy':correct/len(subset) if subset else None,
            'fusion_destruction_rate':(len(subset)-correct)/len(subset) if subset and key!='both_wrong' else None,
            'class_counts':{label:sum(r['label']==i for r in subset) for i,label in enumerate(LABELS)},
            'metrics':metrics(subset,name),
            'gate':quantiles([r['diagnostics'][name]['alpha_mean'] for r in subset]),
            'delta_norm':quantiles([r['diagnostics'][name]['delta_norm'] for r in subset])}
    return result


def compare(rows, before, after):
    counts=dict(repaired=0,regressed=0,wrong_changed=0);changes=[]
    by_group={g:dict(counts) for g in GROUPS}
    by_class={c:dict(counts) for c in LABELS}
    for r in rows:
        y=r['label'];a=r['predictions'][after];b=r['predictions'][before]
        if a==b:continue
        kind='repaired' if a==y else 'regressed' if b==y else 'wrong_changed'
        counts[kind]+=1;by_group[r['teacher_group']][kind]+=1;by_class[LABELS[y]][kind]+=1
        changes.append({**r,'before':b,'after':a,'kind':kind})
    for g in GROUPS:
        denominator=sum(r['teacher_group']==g and r['predictions'][before]!=r['label'] for r in rows)
        by_group[g]['repair_opportunities']=denominator
        by_group[g]['repair_rate']=by_group[g]['repaired']/denominator if denominator else None
    return {'counts':counts,'teacher_groups':by_group,'per_class':by_class},changes


def qualify(summary, candidate, settings):
    m=summary['metrics'];control='cc_t1' if candidate=='cc_t2' else 'cc_t3'
    d0=100*(m[candidate]['macro_f1']-m['cc_t0']['macro_f1'])
    dm=100*(m[candidate]['macro_f1']-m[control]['macro_f1'])
    drops={c:100*(m[candidate]['per_class'][c]['f1-score']-m['cc_t0']['per_class'][c]['f1-score'])
           for c in settings['focus_classes']}
    group_checks={}
    for g in GROUPS[:2]:
        a=summary['groups'][candidate][g]['fusion_destruction_rate']
        b=summary['groups']['cc_t0'][g]['fusion_destruction_rate']
        group_checks[g]={'candidate':a,'T0':b,'passed':a<=b if a is not None and b is not None else None}
    checks={'delta_T0':d0>=settings['candidate_delta_t0_pp'],'delta_MLP':dm>=settings['candidate_delta_mlp_pp'],
        'above_source_ordered':m[candidate]['macro_f1']>m['source_ordered']['macro_f1'],
        'above_frozen_single':m[candidate]['macro_f1']>m['frozen_single']['macro_f1'],
        'focus_classes':all(v>=-settings['focus_max_drop_pp'] for v in drops.values()),
        'destruction':all(v['passed'] is not False for v in group_checks.values())}
    return {'candidate':candidate,'matched_control':control,'passed':all(checks.values()),'checks':checks,
        'delta_T0_pp':d0,'delta_MLP_pp':dm,'focus_F1_changes_pp':drops,'destruction_checks':group_checks,
        'empty_group_policy':'N/A; no percentage fabricated; nonempty groups must pass'}


def decision(summary,settings):
    winner=max(('cc_t2','cc_t4'),key=lambda n:summary['metrics'][n]['macro_f1'])
    result=qualify(summary,winner,settings)
    result['next_arms']=list(ARMS if winner=='cc_t4' else ARMS[:3]) if result['passed'] else []
    result['reason']='registered_gate_passed' if result['passed'] else 'registered_gate_failed_stop_v1'
    result['T3_leads']=summary['metrics']['cc_t3']['macro_f1']>max(summary['metrics'][n]['macro_f1'] for n in ('cc_t2','cc_t4'))
    result['mlp_only_route']='retain evidence; no automatic promotion without a preregistered numeric rule'
    return result


def write_result(output,rows,metadata,settings,seed,history=None,partition='val'):
    require(rows and len({r['sample_id'] for r in rows})==len(rows),'Duplicate/empty prediction cohort')
    names=list(rows[0]['predictions'])
    require(all(list(r['predictions'])==names for r in rows),'Prediction arms differ')
    folder=output/('validation' if partition=='val' else 'test');folder.mkdir(parents=True,exist_ok=True)
    raw.save_predictions(folder/'predictions.jsonl.gz',rows)
    measured={n:metrics(rows,n) for n in names}
    comparisons={}
    pairs=list(dict.fromkeys([p for p in PAIRS if all(n in names for n in p)]+
        [(a,b) for a in names if a.startswith('cc_') for b in ('source_ordered','frozen_single','base_C','cc_t0') if b in names and a!=b]))
    for after,before in pairs:
        result,changes=compare(rows,before,after)
        result['delta_macro_f1_pp']=100*(measured[after]['macro_f1']-measured[before]['macro_f1'])
        comparisons[after+'-'+before]=result
        raw.save_predictions(folder/(after+'-'+before+'_changed.jsonl.gz'),changes)
    for n in names:
        records=[{'sample_id':r['sample_id'],'label':r['label'],'prediction':r['predictions'][n]} for r in rows]
        raw.save_predictions(folder/(n+'_predictions.jsonl.gz'),records)
        independently=training.transaction.frozen.joint.study.validate_predictions(records,probabilities=False)
        check_metric(measured[n],independently)
        with (folder/(n+'_confusion.csv')).open('w',newline='',encoding='utf8') as f:
            w=csv.writer(f);w.writerow(('true/predicted',*LABELS))
            w.writerows((label,*row) for label,row in zip(LABELS,measured[n]['confusion_matrix']))
    groups={n:subgroup(rows,n) for n in names}
    summary={'schema':'xgnid.conditional-complement-results.v1','backbone_seed':seed,'partition':partition,
        'test_evaluated':partition=='test','samples':len(rows),'metrics':measured,'groups':groups,
        'comparisons':comparisons,'metadata':metadata,'independent_metrics_verified':True,
        'previous_negative_ablation_conclusions_changed':False}
    if all(n in names for n in ARMS):
        summary['structure_loss_interaction_pp']=100*((measured['cc_t4']['macro_f1']-measured['cc_t2']['macro_f1'])-
                                                   (measured['cc_t3']['macro_f1']-measured['cc_t1']['macro_f1']))
    if history:
        summary['budget_windows']={str(b):{n:{'epoch':(top:=max((r for r in history if r['epoch']<=b),
             key=lambda r:r['validation'][n]['macro_f1']))['epoch'],'validation':top['validation'][n]}
             for n in history[0]['validation']} for b in (30,100,150,200)}
    if seed==42 and partition=='val': summary['decision']=decision(summary,settings)
    raw.write(output/'summary.json',summary)
    lines=[f'# 双向条件互补学习：seed{seed}，{partition}', '',
        '这是固定主干与固定表示上的配对实验。条件残差是线性预测器未拟合部分，不代表已证明的独有或因果互补信息。',
        '教师用于监督及诊断；推理不使用标签选路。此前连续配对和 beta 门控的负向消融结论不变。','',
        '| 模型 | 最佳轮 | Accuracy | Macro-F1 | Weighted-F1 |','| --- | ---: | ---: | ---: | ---: |']
    for n,m in measured.items():
        lines.append(f"| {ALIASES.get(n,n)} | {metadata.get(n,{}).get('best_epoch','固定')} | {m['accuracy']:.6%} | {m['macro_f1']:.6%} | {m['weighted_f1']:.6%} |")
    lines+=['','| 配对差 | Macro-F1 pp | 修复 | 退步 | 错误间变化 |','| --- | ---: | ---: | ---: | ---: |']
    for n,c in comparisons.items():
        lines.append(f"| {n} | {c['delta_macro_f1_pp']:+.6f} | "+' | '.join(str(c['counts'][k]) for k in ('repaired','regressed','wrong_changed'))+' |')
    if 'structure_loss_interaction_pp' in summary:lines+=['',f"结构与损失交互项：{summary['structure_loss_interaction_pp']:+.6f} pp。"]
    lines+=['','| 模型 | WebBased F1 | Recon F1 | Spoofing F1 | BruteForce F1 |','| --- | ---: | ---: | ---: | ---: |']
    for n,m in measured.items():lines.append('| '+ALIASES.get(n,n)+' | '+' | '.join(f"{m['per_class'][c]['f1-score']:.6%}" for c in settings['focus_classes'])+' |')
    lines+=['','| 模型 | 仅Flow正确：破坏/数量 | 仅Packet正确：破坏/数量 | 均正确：融合正确率 | 均错误：融合正确率 |',
            '| --- | --- | --- | --- | --- |']
    for n,g in groups.items():
        items=[f"{g[k]['count']-g[k]['fusion_correct']}/{g[k]['count']}" if g[k]['count'] else 'N/A' for k in GROUPS[:2]]
        items += [f"{g[k]['fusion_accuracy']:.6%}" if g[k]['count'] else 'N/A' for k in GROUPS[2:]]
        lines.append('| '+ALIASES.get(n,n)+' | '+' | '.join(items)+' |')
    if 'decision' in summary:
        d=summary['decision'];lines+=['',f"候选 {ALIASES[d['candidate']]}；晋级：{'通过' if d['passed'] else '未通过，停止本版扩展'}。",
            '逐项判断（使用未舍入值）：'+str(d['checks']),
            '若仅 T3 占优，仅支持普通MLP路线的训练目标收益；不把它报告为条件交互成功。']
    if partition=='test':lines+=['','测试划分有历史使用，本轮是验证锁定后的固定评价。测试结果不用于重新选模、调参或停止训练。']
    lines+=['','完整八类指标、混淆矩阵、全部变化样本、固定教师分组和门控/修正统计见 summary.json 与预测导出。',
            '这些门槛是计算资源决策规则，不是统计显著性结论。保留全部负值及类别取舍。']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    (output/'paper_tables.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    with (output/'paper_tables.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f);w.writerow(('backbone_seed','partition','model','accuracy','macro_f1','weighted_f1'))
        w.writerows((seed,partition,n,*(m[k] for k in ('accuracy','macro_f1','weighted_f1'))) for n,m in measured.items())
    return summary


def aggregate(summaries,settings,candidate,output,partition='val'):
    require(set(summaries)=={42,43,44},'Three backbones required')
    control='cc_t1' if candidate=='cc_t2' else 'cc_t3'
    pairs=((candidate,'cc_t0'),(candidate,control),(candidate,'source_ordered'),(candidate,'frozen_single'))
    def stats(v):return {'values':v,'mean':mean(v),'sample_std':stdev(v)}
    def scope(seeds):
        return {'backbone_seeds':list(seeds),
            'models':{n:{k:stats([summaries[s]['metrics'][n][k] for s in seeds]) for k in ('accuracy','macro_f1','weighted_f1')}
                for n in summaries[43]['metrics']},
            'paired_macro_f1_pp':{a+'-'+b:stats([100*(summaries[s]['metrics'][a]['macro_f1']-summaries[s]['metrics'][b]['macro_f1']) for s in seeds]) for a,b in pairs},
            'per_class':{n:{c:{k:stats([summaries[s]['metrics'][n]['per_class'][c][k] for s in seeds])
                 for k in ('precision','recall','f1-score')} for c in LABELS} for n in summaries[43]['metrics']}}
    all3,new2=scope((42,43,44)),scope((43,44))
    d0=all3['paired_macro_f1_pp'][candidate+'-cc_t0'];dm=all3['paired_macro_f1_pp'][candidate+'-'+control]
    checks={'candidate_T0_all_positive':all(v>0 for v in d0['values']),
        'candidate_T0_mean':d0['mean']>=.10,'candidate_MLP_mean':dm['mean']>=.05,
        'candidate_MLP_new_both_positive':all(v>0 for v in new2['paired_macro_f1_pp'][candidate+'-'+control]['values'])}
    per_seed={s:qualify(summaries[s],candidate,settings) for s in summaries}
    checks.update(focus_classes=all(q['checks']['focus_classes'] for q in per_seed.values()),
                  destruction=all(q['checks']['destruction'] for q in per_seed.values()))
    result={'partition':partition,'test_evaluated':partition=='test','candidate':candidate,'matched_control':control,
        'all_three':all3,'new_replications_only':new2,'per_seed_checks':per_seed,
        'replication_checks':checks if partition=='val' else None,
        'passed':all(checks.values()) if partition=='val' else None,
        'scope':'三个独立主干上的配对验证；seed42为开发主干，新模块及采样seed固定42',
        'previous_negative_ablation_conclusions_changed':False}
    output.mkdir(parents=True,exist_ok=True);raw.write(output/'summary.json',result)
    lines=['# 双向条件互补：独立主干汇总','',result['scope'],'',
        '| 配对差 pp | seed42 | seed43 | seed44 | 三主干均值±样本标准差 | 新增43/44均值±样本标准差 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for key,v in all3['paired_macro_f1_pp'].items():
        n=new2['paired_macro_f1_pp'][key]
        lines.append('| '+key+' | '+' | '.join(f'{x:+.6f}' for x in v['values'])+f" | {v['mean']:+.6f}±{v['sample_std']:.6f} | {n['mean']:+.6f}±{n['sample_std']:.6f} |")
    lines+=['',('复验晋级：'+str(result['passed'])+'；'+str(checks)) if partition=='val' else
        '测试结果原样保留，不重新挑选检查点或调参。该测试划分有历史使用。',
        '均值和样本标准差使用同一主干配对差，n−1分母；保留所有负值。并非所有模块以三个随机种子完整重训。',
        '此前 True/Shuffle 与 beta 门控未通过门槛的结论不变。']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    (output/'paper_tables.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    return result
