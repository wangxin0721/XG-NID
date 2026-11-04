"""Validation-only metrics and selection checks for the fresh operator pair."""
import csv
import json
import math
from pathlib import Path

LABELS = ['Benign','WebBased','Spoofing','Recon','Mirai','Dos','DDos','BruteForce']


def require(ok, message):
    if not ok:
        raise ValueError(message)


def matrix_report(cm):
    require(len(cm)==8 and all(len(row)==8 for row in cm), 'Expected eight-class matrix')
    require(all(type(v)==int and v>=0 for row in cm for v in row), 'Invalid matrix count')
    count=sum(map(sum,cm));require(count>0,'Empty matrix');classes={}
    for i,label in enumerate(LABELS):
        support,predicted,tp=sum(cm[i]),sum(row[i] for row in cm),cm[i][i]
        classes[label]={'precision':tp/predicted if predicted else 0.,'recall':tp/support if support else 0.,
            'f1-score':2*tp/(support+predicted) if support+predicted else 0.,'support':support}
    return {'accuracy':sum(cm[i][i] for i in range(8))/count,'macro_f1':sum(v['f1-score'] for v in classes.values())/8,
        'weighted_f1':sum(v['f1-score']*v['support'] for v in classes.values())/count,
        'per_class':classes,'confusion_matrix':cm,'samples':count}


def check_history(history, result, epochs, steps, samples):
    require(len(history)==epochs and [r['epoch'] for r in history]==list(range(1,epochs+1)), 'Incomplete epoch ledger')
    best=result['epoch_zero']['macro_f1'];selected=0;trained=-math.inf;trained_epoch=None
    for row in history:
        measured=matrix_report(row['validation']['confusion_matrix'])
        require(measured==row['validation'] and measured['samples']==samples,'Epoch matrix/metrics mismatch')
        score=measured['macro_f1']
        if score>trained:
            trained,trained_epoch=score,row['epoch']
        if score>best:
            best,selected=score,row['epoch']
        require(row['best_epoch']==selected and row['best_trained_epoch']==trained_epoch
                and row['samples']==result['training_samples_per_epoch'] and row['optimizer_updates']==steps
                and math.isfinite(row['train_nll']) and row['train_nll']>=0,'Invalid selection/update ledger')
    require(result['best_epoch']==selected and result['best_trained_epoch']==trained_epoch
            and result['best_validation']['macro_f1']==best and result['trained_best_validation']['macro_f1']==trained
            and result['optimizer_updates']==epochs*steps and result['epochs']==epochs
            and result['last_validation']==history[-1]['validation']
            and result['orders']==[r['train_order_sha256'] for r in history], 'Model result differs from ledger')


def analyze(s):
    require(s['status']=='completed' and s['exit_code']==0 and not s['test_evaluated']
            and not s['automatic_followup'] and s['source_artifacts_unchanged'],'Incomplete operator experiment')
    require(s['protocol']['epochs']==150 and not s['protocol']['early_stop_enabled'],'Wrong fixed budget')
    a,b=[s['models'][n] for n in ('GAT','GATv2')]
    require(a['epochs']==b['epochs']==150 and a['orders']==b['orders']
            and a['training_start_rng_sha256']==b['training_start_rng_sha256'],'Unpaired training order/RNG')
    require(s['pairing']['all_common_tensors_identical'] and s['probe_validated'],'Initialization/probe evidence missing')
    for name,m in [('GAT',a),('GATv2',b)]:
        require(m['optimizer_updates']==150*211 and m['training_samples_per_epoch']==107621
                and m['stop_reason']=='fixed_150_epochs' and m['early_stop_enabled'] is False
                and m['training_evidence']['encoders_and_bn_updated']
                and m['parameter_count']==(851688 if name=='GAT' else 850152)
                and m['selected_epoch_zero']==(m['best_epoch']==0),'Wrong training budget/range')
        for phase in ('epoch_zero','best_validation','trained_best_validation','last_validation'):
            require(matrix_report(m[phase]['confusion_matrix'])==m[phase] and m[phase]['samples']==13446,'Metrics differ')
        require(all(m['replays'][k]['predictions_identical'] and m['replays'][k]['strict_state_load']
                    and m['replays'][k]['metrics_identical'] and m['replays'][k]['samples']==13446
                    for k in ('best','trained_best')),'Full raw-graph replay missing')
    counts=s['comparison']['counts']
    require(set(counts)=={'repaired','regressed','both_correct','both_wrong'}
            and all(type(v)==int and v>=0 for v in counts.values()) and sum(counts.values())==13446
            and s['comparison']['net_correct']==counts['repaired']-counts['regressed']
            ==sum(b['best_validation']['confusion_matrix'][i][i]-a['best_validation']['confusion_matrix'][i][i] for i in range(8)),
            'Changed-sample counts disagree with matrices')
    baseline=s['preparation_protocol']['D32_reference_macro_f1'];current=s['preparation_protocol']['CL_Multi_reference_macro_f1']
    score=b['best_validation']['macro_f1'];stronger=max(a['best_validation']['macro_f1'],baseline)
    threshold=s['preparation_protocol']['budget_threshold_pp']
    return {'GATv2_minus_GAT_pp':100*(score-a['best_validation']['macro_f1']),
        'GATv2_minus_D32_pp':100*(score-baseline),'GATv2_minus_CL_Multi_pp':100*(score-current),
        'stronger_reference_macro_f1':stronger,'delta_pp':100*(score-stronger),'budget_threshold_pp':threshold,
        'required_GATv2_macro_f1':stronger+threshold/100,'budget_screen_passed':100*(score-stronger)>=threshold,
        'exceeds_CL_Multi':score>current,'historical_scores_are_references_only':True,
        'statistical_significance_claimed':False,'class_tradeoff_review_required':True,'automatic_followup':False,
        'class_changes':{label:{'GATv2_minus_GAT_recall_pp':100*(b['best_validation']['per_class'][label]['recall']-a['best_validation']['per_class'][label]['recall']),
            'GATv2_minus_GAT_F1_pp':100*(b['best_validation']['per_class'][label]['f1-score']-a['best_validation']['per_class'][label]['f1-score'])} for label in LABELS}}


def report(root, s):
    a=analyze(s)
    for name,m in s['models'].items():
        history=[json.loads(x) for x in (root/name/'epochs.jsonl').read_text(encoding='utf-8').splitlines()]
        check_history(history,m,150,211,13446)
    (root/'analysis.json').write_text(json.dumps(a,indent=2)+'\n',encoding='utf-8')
    lines=['| model | epoch0 F1 | selected best F1 | best epoch | trained best F1 | trained epoch | last F1 | stop epoch |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,m in s['models'].items():
        lines.append(f"| {name} | {100*m['epoch_zero']['macro_f1']:.6f}% | {100*m['best_validation']['macro_f1']:.6f}% | {m['best_epoch']} | {100*m['trained_best_validation']['macro_f1']:.6f}% | {m['best_trained_epoch']} | {100*m['last_validation']['macro_f1']:.6f}% | {m['epochs']} |")
    lines+=['',f"GATv2-GAT={a['GATv2_minus_GAT_pp']:+.6f} pp; GATv2-D32={a['GATv2_minus_D32_pp']:+.6f} pp; GATv2-CL-Multi={a['GATv2_minus_CL_Multi_pp']:+.6f} pp.",
            f"Budget above max(GAT,D32)={a['delta_pp']:+.6f} pp; passed={a['budget_screen_passed']}.",
            'Both arms start fresh and train all encoders/BN/head for150 epochs; no early stopping, frozen feature cache, mature encoder, or previous structural additions.',
            'Epoch0 participates in final selection; trained best is selected separately from epochs1..150; ties retain the earliest epoch.',
            'Historical D32/CL-Multi scores are reference numbers; their checkpoints are not initialized or replayed in this run.',
            'Paired order/RNG, full validation exports, strict checkpoint reload and raw-graph best/trained-best replay passed. No test graph evaluation or automatic follow-up.',
            'Sequential runs include graph I/O and checkpoint writing; wall times and three-step probe timings are not controlled operator benchmarks.',
            '', '| class | GATv2-GAT recall pp | GATv2-GAT F1 pp |','|---|---:|---:|']
    for label,v in a['class_changes'].items():
        lines.append(f"| {label} | {v['GATv2_minus_GAT_recall_pp']:+.6f} | {v['GATv2_minus_GAT_F1_pp']:+.6f} |")
    lines+=['', 'GATv2 versus GAT changed samples: '+json.dumps(s['comparison'],ensure_ascii=False)]
    for name,m in s['models'].items():
        lines+=['',name+' cost: '+json.dumps(m['cost'])]
        for phase in ('best_validation','trained_best_validation','last_validation'):
            lines+=['',name+' '+phase+' confusion matrix (rows=true):','```',json.dumps(m[phase]['confusion_matrix']),'```']
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with (root/'per_class.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['model','phase','class','precision','recall','f1','support'])
        for name,m in s['models'].items():
            for phase in ('epoch_zero','best_validation','trained_best_validation','last_validation'):
                for label,v in m[phase]['per_class'].items():
                    w.writerow([name,phase,label,v['precision'],v['recall'],v['f1-score'],v['support']])
    return a


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    args=parser.parse_args();s=json.loads((args.root/'summary.json').read_text(encoding='utf-8'))
    print(json.dumps(report(args.root,s),indent=2))
