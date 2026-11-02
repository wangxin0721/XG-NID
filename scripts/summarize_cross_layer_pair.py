"""Validation-only cross-layer-pair analysis and report; no torch dependency."""
import argparse
import csv
import json
from pathlib import Path
from e3_gate_study import read,require,write
from run_fusion_seed_repeats import verify_metrics


def budget(mean,gate,baseline,j32,threshold=.1):
    stronger=max(mean,baseline);delta=100*(gate-stronger)
    return {'CL_Multi_minus_CL_Last_pp':100*(gate-mean),'CL_Multi_minus_D32_pp':100*(gate-baseline),
        'CL_Multi_minus_P1_reported_pp':100*(gate-j32),'stronger_reference_macro_f1':stronger,
        'delta_pp':delta,'budget_threshold_pp':threshold,'required_Multi_macro_f1':stronger+threshold/100,
        'budget_screen_passed':delta>=threshold,'P1_reference_macro_f1':j32,
        'P1_reference_provenance':'user_reported_epoch146_matrix_verified_not_checkpoint_rehashed'}


def analyze(s):
    import run_cross_layer_pair as run
    run.check_config(s['protocol'])
    require(s['status']=='completed' and s['exit_code']==0 and s['test_evaluated'] is False
            and s['automatic_followup'] is False and all(s[k] is True for k in
                ('tensor_preflight_passed','encoder_parameters_and_BN_unchanged','source_artifacts_unchanged','cross_layer_caches_unchanged')),
            'Incomplete cross-layer pair')
    require(s['cuda_cross_layer_preflight']['status']=='passed' and s['cuda_cross_layer_preflight']['model_and_rng_unchanged'], 'Device check missing')
    source=s['source_model']['best_validation'];verify_metrics(source)
    require(s['source_model']['best_epoch']==62 and abs(source['macro_f1']-s['protocol']['source_macro_f1'])<1e-12,'Wrong mature baseline')
    mean,gate=[s['models'][n] for n in run.VARIANTS];common=min(mean['epochs'],gate['epochs'])
    require(common==s['common_epochs_order_checked'] and mean['orders'][:common]==gate['orders'][:common]
            and mean['training_start_rng_sha256']==gate['training_start_rng_sha256'],'Pair order/RNG differs')
    for name,mode in run.VARIANTS.items():
        m=s['models'][name]
        for key in ('epoch_zero','best_validation','trained_best_validation','last_validation'):verify_metrics(m[key])
        require(m['architecture']==run.architecture(mode) and m['epoch_zero']['confusion_matrix']==source['confusion_matrix']
                and m['epoch_zero_eligible'] is True and m['selected_epoch_zero']==(m['best_epoch']==0)
                and m['best_validation']['macro_f1']>=source['macro_f1'],'Wrong selection/initialization')
        require(m['early_stop_reference']=='trained_epochs_only' and m['epochs']==150
                and 1<=m['best_trained_epoch']<=m['epochs']
                and m['best_validation']['macro_f1']==max(m['trained_best_validation']['macro_f1'],m['epoch_zero']['macro_f1']),
                'Trained-only early-stop/final selection differs')
        require(s['probes'][name]['clean_model_and_rng_restored'] and s['probes'][name]['optimizer_discarded']
                and s['graph_replays'][name]['strict_state_load'] and s['graph_replays'][name]['confusion_matrix_reproduced'], 'Probe/replay missing')
        require(s['trained_best_graph_replays'][name]['strict_state_load']
                and s['trained_best_graph_replays'][name]['confusion_matrix_reproduced'],'Trained-best raw replay missing')
    require(s['initialization']['pair_allclose_1e6'] and s['initialization']['last_layer_identity']
            and s['initialization']['source_predictions_and_CM_identical'],'Initial identity missing')
    for name,ch in s['comparisons'].items():
        for kind in ('repaired','regressed'):
            require(sum(t['count'] for t in s['prediction_transitions'][name] if t['kind']==kind)==ch['counts'][kind], 'Transitions disagree')
    import run_packet_pool_pair as packet
    acc=s['graph_load_accounting']
    require(acc==packet.graph_load_accounting(acc['main_process_counts'],acc['loader_stages']), 'Load accounting differs')
    expected=[packet.graph_load_stage('extract_train','train',107621,2),packet.graph_load_stage('extract_val','val',13446,2)]
    expected +=[packet.graph_load_stage(phase+'_'+name,'val',13446,2) for phase in ('initial','best') for name in run.VARIANTS]
    expected +=[packet.graph_load_stage('trained_best_'+name,'val',13446,2) for name in run.VARIANTS
                if not s['trained_best_graph_replays'][name]['shared_with_selected_best']]
    require(acc['loader_stages']==expected,'Incomplete graph-load stages')
    a=budget(mean['best_validation']['macro_f1'],gate['best_validation']['macro_f1'],source['macro_f1'],
             s['protocol']['P1_reference_macro_f1'],s['protocol']['budget_above_stronger_reference_pp'])
    a['trained_best_changes']={n:{'minus_D32_pp':100*(m['trained_best_validation']['macro_f1']-source['macro_f1']),
        'minus_P1_reported_pp':100*(m['trained_best_validation']['macro_f1']-s['protocol']['P1_reference_macro_f1'])}
        for n,m in s['models'].items()}
    a.update(selected_epoch_zero={n:m['selected_epoch_zero'] for n,m in s['models'].items()},
        class_tradeoff_review_required=True,statistical_significance_claimed=False,automatic_followup=False,
        source_comparison_policy='Matched cross-layer readout on identical frozen graph means uses 1e-6; independent legacy CUDA probabilities recorded separately; all source predictions/CM must match.',
        class_changes={})
    for label,v in source['per_class'].items():
        x,y=[m['best_validation']['per_class'][label] for m in (mean,gate)]
        a['class_changes'][label]={'Multi_minus_Last_recall_pp':100*(y['recall']-x['recall']),
            'Multi_minus_D32_recall_pp':100*(y['recall']-v['recall']),
            'Multi_minus_Last_F1_pp':100*(y['f1-score']-x['f1-score']),'support':v['support']}
    return a


def summarize(root):
    import run_cross_layer_pair as run
    s=read(root/'summary.json');a=analyze(s)
    for name,m in s['models'].items():
        run.check_history([json.loads(x) for x in (root/name/'epochs.jsonl').read_text(encoding='utf-8').splitlines()],m,s['protocol'])
    write(root/'analysis.json',a)
    lines=['| model | start F1 | selected best F1 | best epoch | trained best F1 | trained best epoch | last F1 | stop epoch | updates | seconds |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,m in s['models'].items():
        t=m['training_evidence']
        lines.append(f"| {name} | {100*m['epoch_zero']['macro_f1']:.6f}% | {100*m['best_validation']['macro_f1']:.6f}% | {m['best_epoch']} | "
            f"{100*t['best_trained_macro_f1']:.6f}% | {t['best_trained_epoch']} | {100*m['last_validation']['macro_f1']:.6f}% | "
            f"{m['epochs']} | {m['optimizer_updates']} | {m['training_wall_seconds']:.3f} |")
    lines += ['',f"Multi-Last={a['CL_Multi_minus_CL_Last_pp']:+.6f} pp; Multi-D32={a['CL_Multi_minus_D32_pp']:+.6f} pp.",
              f"Delta above max(Last,D32)={a['delta_pp']:+.6f} pp; >=+0.10 pp budget passed={a['budget_screen_passed']}; required F1={100*a['required_Multi_macro_f1']:.6f}%.",
              f"Multi versus reported P1 97.427191%={a['CL_Multi_minus_P1_reported_pp']:+.6f} pp. P1 score is a reference only; its checkpoint is not loaded.",
              'Both begin at mature D32-Gate epoch62. Epoch0 is saved separately; final selection compares it with the best trained epoch, preferring epoch0 on ties.',
              'Only Packet cross-layer readout changes. All encoder and BN parameters/buffers remain frozen/eval.',
              'Both use a128-to64 linear readout. Last inputs[h2,h2]; Multi inputs[h1,h2]. Initialization[0,I],bias0 reproduces h2.',
              'Adam WD0, existing head1e-5, cross-layer projection1e-4, batch512/16, seed42, registered min50/max150; both arms complete150 epochs.',
              'Early stopping is disabled. Stale counters describe trained-epoch progress only; patience20 is recorded but never truncates training.',
              'epoch_zero.pt, trained_best.pt, best.pt and last.pt are independent artifacts. Trained-best predictions are exported separately.',
              'Active parameter counts: Last15400; Multi15400, each including8256 new projection parameters.',
              '', '| class | Multi-Last recall pp | Multi-D32 recall pp | Multi-Last F1 pp | support |','|---|---:|---:|---:|---:|']
    for label,v in a['class_changes'].items():lines.append(f"| {label} | {v['Multi_minus_Last_recall_pp']:+.6f} | {v['Multi_minus_D32_recall_pp']:+.6f} | {v['Multi_minus_Last_F1_pp']:+.6f} | {v['support']} |")
    lines += ['', 'Review WebBased, Recon, Spoofing and BruteForce with all eight classes.']
    for name,ch in s['comparisons'].items():lines.append(f"{name}: repaired={ch['counts']['repaired']}, regressed={ch['counts']['regressed']}, net={ch['net_correct']}.")
    lines += ['', '| partition | class | graphs | varying depths | layer1 norm | layer2 norm | L2 difference | cosine |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for part,groups in s['cross_layer_diagnostics'].items():
        for label,v in groups.items():
            lines.append(f"| {part} | {label} | {v['graphs']} | {v['varying_graphs']} | {v['mean_layer1_norm']:.6g} | {v['mean_layer2_norm']:.6g} | {v['mean_difference']:.6g} | {v['mean_cosine']:.6g} |")
    for name,m in s['models'].items():
        lines += ['',f"{name} trained evidence: {m['training_evidence']}; groups: {m['parameter_groups']}."]
        for phase in ('best_validation','trained_best_validation','last_validation'):
            labels=list(m[phase]['per_class']);lines += ['',f'{name} {phase} CM (rows=true):','','| true / predicted | '+' | '.join(labels)+' |','|---|'+'---:|'*8]
            lines += ['| '+label+' | '+' | '.join(map(str,row))+' |' for label,row in zip(labels,m[phase]['confusion_matrix'])]
    lines += ['', 'cross_layer_diagnostics.json compares post-BN/activation graph mean representations at both depths.',
              'Cache stores flow64, Packet first64 and second64 graph means; old last-layer-only caches are not reused.',
              'Per-sample selected predictions retain all32 original fusion weights.',
              'Changed sample IDs: *_changed_samples.jsonl.gz. Best/last/source class P/R/F1: per_class.csv.',
              'GPU backward/initial identity, raw gradient probes, clean-state restoration and full best-checkpoint raw validation replays passed.',
              a['source_comparison_policy'], 'Budget threshold is not statistical significance. No residual connection, pooled attention, adaptive relations, new seed, test evaluation or automatic follow-up.']
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with (root/'per_class.csv').open('w',encoding='utf-8',newline='') as stream:
        w=csv.writer(stream);w.writerow(['model','checkpoint','class','precision','recall','f1','support'])
        for name,m in [('D32-Gate',s['source_model']),*s['models'].items()]:
            for phase in (('best_validation',) if name=='D32-Gate' else ('best_validation','trained_best_validation','last_validation')):
                for label,v in m[phase]['per_class'].items():w.writerow([name,phase,label,*(v[k] for k in ('precision','recall','f1-score','support'))])
    with (root/'prediction_transitions.csv').open('w',encoding='utf-8',newline='') as stream:
        w=csv.writer(stream);w.writerow(['comparison','kind','true_class','old_prediction','new_prediction','count'])
        for name,ts in s['prediction_transitions'].items():
            for t in ts:w.writerow([name,t['kind'],t['true_class'],t['equal_prediction'],t['gate_prediction'],t['count']])
    return '\n'.join(lines)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);print(summarize(p.parse_args().root))
