"""Validation-only residual-pair analysis and report; no torch dependency."""
import argparse
import csv
import json
from pathlib import Path
from e3_gate_study import read,require,write
from run_fusion_seed_repeats import verify_metrics


def budget(mean,gate,baseline,cl_multi,threshold=.1):
    stronger=max(mean,baseline);delta=100*(gate-stronger)
    return {'Res_Skip_minus_Res_Base_pp':100*(gate-mean),'Res_Skip_minus_D32_pp':100*(gate-baseline),
        'Res_Skip_minus_CL_Multi_reported_pp':100*(gate-cl_multi),'stronger_reference_macro_f1':stronger,
        'delta_pp':delta,'budget_threshold_pp':threshold,'required_Skip_macro_f1':stronger+threshold/100,
        'budget_screen_passed':delta>=threshold,'CL_Multi_reference_macro_f1':cl_multi,
        'CL_Multi_reference_provenance':'verified_cross_layer_epoch126_reference_only'}


def analyze(s):
    import run_residual_pair as run
    run.check_config(s['protocol'])
    run.validate_prerequisite(s['prior_cross_layer_audit']['receipt'])
    require(s['status']=='completed' and s['exit_code']==0 and s['test_evaluated'] is False
            and s['automatic_followup'] is False and all(s[k] is True for k in
                ('tensor_preflight_passed','encoder_parameters_and_BN_unchanged','source_artifacts_unchanged','residual_caches_unchanged')),
            'Incomplete residual pair')
    require(s['cuda_residual_preflight']['status']=='passed' and s['cuda_residual_preflight']['model_and_rng_unchanged'], 'Device check missing')
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
    require(s['initialization']['pair_allclose_1e6'] and s['initialization']['zero_residual_identity']
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
             s['protocol']['CL_Multi_reference_macro_f1'],s['protocol']['budget_above_stronger_reference_pp'])
    a['Res_Skip_minus_CL_Last_reference_pp']=100*(gate['best_validation']['macro_f1']-s['protocol']['CL_Last_reference_macro_f1'])
    a['prior_cross_layer_audit']=s['prior_cross_layer_audit']
    a['trained_best_changes']={n:{'minus_D32_pp':100*(m['trained_best_validation']['macro_f1']-source['macro_f1']),
        'minus_CL_Multi_reported_pp':100*(m['trained_best_validation']['macro_f1']-s['protocol']['CL_Multi_reference_macro_f1'])}
        for n,m in s['models'].items()}
    a.update(selected_epoch_zero={n:m['selected_epoch_zero'] for n,m in s['models'].items()},
        class_tradeoff_review_required=True,statistical_significance_claimed=False,automatic_followup=False,
        source_comparison_policy='Matched zero-residual/BN/readout on identical frozen messages uses 1e-6; independent legacy CUDA probabilities recorded separately; all source predictions/CM must match.',
        class_changes={})
    for label,v in source['per_class'].items():
        x,y=[m['best_validation']['per_class'][label] for m in (mean,gate)]
        a['class_changes'][label]={'Skip_minus_Base_recall_pp':100*(y['recall']-x['recall']),
            'Skip_minus_D32_recall_pp':100*(y['recall']-v['recall']),
            'Skip_minus_Base_F1_pp':100*(y['f1-score']-x['f1-score']),'support':v['support']}
    return a


def summarize(root):
    import run_residual_pair as run
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
    lines += ['',f"Skip-Base={a['Res_Skip_minus_Res_Base_pp']:+.6f} pp; Skip-D32={a['Res_Skip_minus_D32_pp']:+.6f} pp.",
              f"Skip versus CL-Last97.467535%={a['Res_Skip_minus_CL_Last_reference_pp']:+.6f} pp.",
              f"Delta above max(Base,D32)={a['delta_pp']:+.6f} pp; >=+0.10 pp budget passed={a['budget_screen_passed']}; required F1={100*a['required_Skip_macro_f1']:.6f}%.",
              f"Skip versus verified CL-Multi 97.509986%={a['Res_Skip_minus_CL_Multi_reported_pp']:+.6f} pp. CL-Multi weights are not used to initialize this experiment.",
              'Both begin at mature D32-Gate epoch62. Epoch0 is saved separately; final selection compares it with the best trained epoch, preferring epoch0 on ties.',
              'Only Packet second-layer pre-BN residual differs. Both share the same CL-Last-style128-to64 projection and training budget; old GAT/BN remain frozen/eval.',
              'Residual R=Linear64/64,biasFalse initialized zero; add after the original relation mean and before BN/activation. Empty-relation outputs unchanged.',
              'Adam WD0, existing head1e-5, common projection/residual1e-4, batch512/16, seed42, registered min50/max150; both arms complete150 epochs.',
              'Early stopping is disabled. Stale counters describe trained-epoch progress only; patience20 is recorded but never truncates training.',
              'epoch_zero.pt, trained_best.pt, best.pt and last.pt are independent artifacts. Trained-best predictions are exported separately.',
              'Active parameter counts: Base15400; Skip19496; common projection8256 in both; residual4096 only in Skip.',
              '', '| class | Skip-Base recall pp | Skip-D32 recall pp | Skip-Base F1 pp | support |','|---|---:|---:|---:|---:|']
    for label,v in a['class_changes'].items():lines.append(f"| {label} | {v['Skip_minus_Base_recall_pp']:+.6f} | {v['Skip_minus_D32_recall_pp']:+.6f} | {v['Skip_minus_Base_F1_pp']:+.6f} | {v['support']} |")
    lines += ['', 'Review WebBased, Recon, Spoofing and BruteForce with all eight classes.']
    for name,ch in s['comparisons'].items():lines.append(f"{name}: repaired={ch['counts']['repaired']}, regressed={ch['counts']['regressed']}, net={ch['net_correct']}.")
    lines += ['', '| partition | class | nodes | empty F relation graphs | empty P relation graphs | mean F norm | mean P norm | mean message L2 difference | mean cosine |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for part,groups in s['residual_diagnostics'].items():
        for label,v in groups.items():lines.append(f"| {part} | {label} | {v['packet_nodes']} | {v['empty_flow_relation_graphs']} | {v['empty_packet_relation_graphs']} | "
            f"{v['flow_norm_node_mean']:.6g} | {v['packet_norm_node_mean']:.6g} | {v['difference_node_mean']:.6g} | {v['cosine_node_mean']:.6g} |")
    for name,m in s['models'].items():
        lines += ['',f"{name} trained evidence: {m['training_evidence']}; groups: {m['parameter_groups']}."]
        for phase in ('best_validation','trained_best_validation','last_validation'):
            labels=list(m[phase]['per_class']);lines += ['',f'{name} {phase} CM (rows=true):','','| true / predicted | '+' | '.join(labels)+' |','|---|'+'---:|'*8]
            lines += ['| '+label+' | '+' | '.join(map(str,row))+' |' for label,row in zip(labels,m[phase]['confusion_matrix'])]
    lines += ['', 'residual_diagnostics.json includes empty relations, per-node missing incoming messages, zero norms and relative differences.',
              'Fresh cache contains pre-BN first-layer/Flow-message/Packet-message tensors. CL graph-level caches cannot recover the required nodes.',
              'Per-sample selected predictions retain all32 fusion weights and residual norm summaries for every graph.',
              'Changed sample IDs: *_changed_samples.jsonl.gz. Best/last/source class P/R/F1: per_class.csv.',
              'Prior checkpoint/fold audit, GPU backward/initial identity, raw gradient probes, clean-state restoration and full best-checkpoint raw validation replays passed.',
              a['source_comparison_policy'], 'Budget threshold is not statistical significance. No adaptive relations, attention pooling, cross-layer shallow readout, new seed, test evaluation or automatic follow-up.']
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
