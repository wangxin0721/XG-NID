"""Report the one-round training-budget confirmation without changing models."""
import argparse
import csv
import json
from pathlib import Path

from e3_gate_study import read, require, write
from run_fusion_seed_repeats import verify_metrics
from summarize_packet_pool_pair import budget


def analyze(s):
    import run_packet_pool_confirmation as run
    packet = run.packet
    run.check_config(s['protocol'])
    require(s['status'] == 'completed' and s['exit_code'] == 0 and s['test_evaluated'] is False
            and s['automatic_followup'] is False and s['training_jobs'] == 2
            and s['experiment_kind'] == 'one_round_training_budget_confirmation_not_new_model'
            and s['cache_reused'] is True and all(s[k] is True for k in
                ('tensor_preflight_passed', 'encoder_parameters_and_BN_unchanged',
                 'source_artifacts_unchanged', 'node_caches_unchanged')), 'Incomplete fixed-budget confirmation')
    require(s['cuda_readout_preflight']['status'] == 'passed'
            and s['cuda_readout_preflight']['optimizer_updates'] == 0
            and s['cuda_readout_preflight']['model_and_rng_unchanged'], 'Device readout preflight missing')
    require(all(s['initialization'][k] is True for k in ('P0_P1_outputs_allclose',
        'source_predictions_identical', 'uniform_packet_attention', 'epoch_zero_eligible')), 'Initial identity missing')
    source = s['source_model']['best_validation'];verify_metrics(source)
    require(s['source_model']['best_epoch'] == 62
            and abs(source['macro_f1']-s['protocol']['source_macro_f1']) < 1e-12, 'Wrong mature start')
    p0,p1 = [s['models'][n] for n in ('P0','P1')]
    require(s['common_epochs_order_checked'] == 150 and p0['orders'] == p1['orders']
            and len(p0['orders']) == 150
            and p0['training_start_rng_sha256'] == p1['training_start_rng_sha256'], 'Paired order/RNG differs')
    for name,mode in run.VARIANTS.items():
        m = s['models'][name]
        for phase in ('epoch_zero','best_validation','trained_best_validation','last_validation'):verify_metrics(m[phase])
        require(m['architecture'] == run.architecture(mode) and m['epochs'] == 150
                and m['optimizer_updates'] == 31650 and m['stop_reason'] == 'max_epochs'
                and m['early_stop_reference'] == 'trained_epochs_only'
                and m['epoch_zero_eligible'] is True and m['selected_epoch_zero'] == (m['best_epoch'] == 0)
                and 1 <= m['best_trained_epoch'] <= 150
                and m['epoch_zero']['confusion_matrix'] == source['confusion_matrix']
                and m['best_validation']['macro_f1'] == max(m['epoch_zero']['macro_f1'], m['trained_best_validation']['macro_f1']),
                'Training budget, initialization or final selection differs')
        require(s['probes'][name]['clean_model_and_rng_restored'] and s['probes'][name]['optimizer_discarded'], 'Probe restore missing')
        for key in ('graph_replays','trained_best_graph_replays'):
            require(all(s[key][name][k] for k in ('strict_state_load','predictions_identical','confusion_matrix_reproduced')), 'Raw replay missing')
        require(s['completed_cache_source']['models'][name]['epochs'] == 20
                and s['completed_cache_source']['models'][name]['best_epoch'] == 0, 'Wrong previous screen')
        prefix = s['budget_confirmation'][name]
        require(prefix['full150_selected_macro_f1'] == m['best_validation']['macro_f1'], 'Budget confirmation score differs')
    acc = s['graph_load_accounting']
    expected = [packet.graph_load_stage(phase+'_'+name,'val',13446,2)
                for phase in ('initial','best') for name in run.VARIANTS]
    expected += [packet.graph_load_stage('trained_best_'+name,'val',13446,2) for name in run.VARIANTS
                 if not s['trained_best_graph_replays'][name]['shared_with_selected_best']]
    require(acc == packet.graph_load_accounting(acc['main_process_counts'],acc['loader_stages'])
            and acc['loader_stages'] == expected and acc['totals'] == s['graph_load_counts']
            and acc['totals']['test'] == 0, 'Load accounting differs')
    for name,ch in s['comparisons'].items():
        for kind in ('repaired','regressed'):
            require(sum(t['count'] for t in s['prediction_transitions'][name] if t['kind'] == kind) == ch['counts'][kind], 'Transition counts differ')
    a = budget(p0['best_validation']['macro_f1'],p1['best_validation']['macro_f1'],source['macro_f1'],s['protocol']['budget_above_stronger_reference_pp'])
    prefix_budget = budget(*(s['budget_confirmation'][n]['first20_selected_macro_f1'] for n in ('P0','P1')),
        source['macro_f1'],s['protocol']['budget_above_stronger_reference_pp'])
    a.update(experiment_kind=s['experiment_kind'], confirmation_rounds=1, epochs_per_arm=150,
        early_stop_enabled=False, budget_confirmation=s['budget_confirmation'], first20_budget=prefix_budget,
        first20_reproduced=all(p['first20_reproduced'] for p in s['budget_confirmation'].values()),
        budget_verdict_changed_after20=a['budget_screen_passed'] != prefix_budget['budget_screen_passed'],
        selected_epoch_zero={n:m['selected_epoch_zero'] for n,m in s['models'].items()},
        class_tradeoff_review_required=True, statistical_significance_claimed=False, automatic_followup=False,
        attribution='Same P0/P1 models and hyperparameters; only the fixed training budget changes. '
            'If the first20 trajectory differs, do not attribute differences solely to removing early stopping.', class_changes={})
    for label,b in source['per_class'].items():
        x,y = [m['best_validation']['per_class'][label] for m in (p0,p1)]
        a['class_changes'][label] = {'P1_minus_P0_recall_pp':100*(y['recall']-x['recall']),
            'P1_minus_D32_recall_pp':100*(y['recall']-b['recall']),
            'P1_minus_P0_F1_pp':100*(y['f1-score']-x['f1-score']), 'support':b['support']}
    return a


def summarize(root):
    import run_packet_pool_confirmation as run
    s = read(root/'summary.json')
    for name,m in s['models'].items():
        history = [json.loads(x) for x in (root/name/'epochs.jsonl').read_text(encoding='utf-8').splitlines()]
        run.check_history(history,m,s['protocol'])
        require(s['budget_confirmation'][name] == run.prefix_comparison(history,
            s['completed_cache_source']['histories'][name],m['epoch_zero']['macro_f1']), 'Prefix ledger differs')
    a = analyze(s);write(root/'analysis.json',a)
    lines = ['| model | epoch0 F1 | trained best F1 | trained epoch | selected F1 | selected epoch | last F1 | epochs | updates |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,m in s['models'].items():
        lines.append(f"| {name} | {100*m['epoch_zero']['macro_f1']:.6f}% | {100*m['trained_best_validation']['macro_f1']:.6f}% | "
            f"{m['best_trained_epoch']} | {100*m['best_validation']['macro_f1']:.6f}% | {m['best_epoch']} | "
            f"{100*m['last_validation']['macro_f1']:.6f}% | {m['epochs']} | {m['optimizer_updates']} |")
    lines += ['', 'One additional paired confirmation only; this is a training-budget check, not a new model improvement.',
        'Both restart from mature D32-Gate epoch62 with fresh Adam/RNG; no epoch20 model or optimizer is resumed.',
        'Both complete all150 epochs; early stopping disabled. Registered min50 does not trigger stopping.',
        'Epoch0 participates in final selection. trained_best.pt selects epochs1..150 independently; ties retain epoch0.',
        'Frozen encoders and BN; same ordered packet mean/attention; same seed42, split, LR1e-5/1e-4, WD0, NLL, batch512/16.',
        'Trainable parameters unchanged: P0=7144, P1=8200; added attention network=1056.',
        '', f"P1-P0={a['P1_minus_P0_pp']:+.6f} pp; P1-D32={a['P1_minus_D32_pp']:+.6f} pp.",
        f"P1-max(P0,D32)={a['P1_minus_stronger_reference_pp']:+.6f} pp; +0.10 pp budget passed={a['budget_screen_passed']}.",
        f"Both first20 trajectories reproduced={a['first20_reproduced']}; budget verdict changed after20={a['budget_verdict_changed_after20']}.",
        '', '| model | first20 order same | first20 scores same | exact train NLL same | NLL max abs error | first20 selected F1 | full150 selected F1 | gain after20 pp |',
        '|---|---|---|---|---:|---:|---:|---:|']
    for name,p in a['budget_confirmation'].items():
        lines.append(f"| {name} | {p['first20_orders_identical']} | {p['first20_validation_scores_identical']} | {p['first20_exact_train_nll']} | "
            f"{p['first20_train_nll_max_abs_error']:.6g} | {100*p['first20_selected_macro_f1']:.6f}% | "
            f"{100*p['full150_selected_macro_f1']:.6f}% | {p['gain_after20_pp']:+.6f} |")
    lines += ['', a['attribution'], '', '| class | P1-P0 recall pp | P1-D32 recall pp | P1-P0 F1 pp | support |', '|---|---:|---:|---:|---:|']
    for label,v in a['class_changes'].items():
        lines.append(f"| {label} | {v['P1_minus_P0_recall_pp']:+.6f} | {v['P1_minus_D32_recall_pp']:+.6f} | {v['P1_minus_P0_F1_pp']:+.6f} | {v['support']} |")
    lines += ['', 'Review WebBased/Recon recall and Spoofing/BruteForce tradeoffs.']
    for name,ch in s['comparisons'].items():
        lines.append(f"{name}: repaired={ch['counts']['repaired']}, regressed={ch['counts']['regressed']}, net={ch['net_correct']}.")
    for name,m in s['models'].items():
        lines += ['',f"{name}: trained evidence={m['training_evidence']}; parameter groups={m['parameter_groups']}."]
        for phase in ('best_validation','trained_best_validation','last_validation'):
            labels = list(m[phase]['per_class'])
            lines += ['',f'{name} {phase} confusion matrix (rows=true):','',
                '| true / predicted | '+' | '.join(labels)+' |','|---|'+'---:|'*8]
            lines += ['| '+label+' | '+' | '.join(map(str,row))+' |' for label,row in zip(labels,m[phase]['confusion_matrix'])]
    lines += ['', 'The completed20-epoch screen and all its artifacts remain unchanged; only its frozen node caches are copied and verified.',
        'Cache identities, split, sample order, file/tensor hashes and encoder hashes are checked again.',
        'Cache memory figures describe original extraction, not current copying or whole-process RSS.',
        f"Current graph sample loads={s['graph_load_counts']}; original cache-extraction reads are excluded.",
        'Initial, selected-best and trained-best checkpoints pass original-graph validation replay (shared when identical).',
        'Matched readout identity retains atol/rtol1e-6; raw/cache1e-5. Independent legacy predictions/CM must match exactly.',
        'Full P/R/F1/support: per_class.csv; changed samples: *_changed_samples.jsonl.gz; destinations: prediction_transitions.csv.',
        'Packet diagnostics: packet_diagnostics.json; load receipts: graph_load_accounting.json.',
        'No test evaluation, extra seed, new hyperparameter search, additional confirmation round or automatic experiment. Budget threshold is not significance.']
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with (root/'per_class.csv').open('w',encoding='utf-8',newline='') as stream:
        writer = csv.writer(stream);writer.writerow(['model','checkpoint','class','precision','recall','f1','support'])
        for name,m in [('D32-Gate',s['source_model']),*s['models'].items()]:
            for phase in (('best_validation',) if name == 'D32-Gate' else ('best_validation','trained_best_validation','last_validation')):
                for label,v in m[phase]['per_class'].items():
                    writer.writerow([name,phase,label,*(v[k] for k in ('precision','recall','f1-score','support'))])
    with (root/'prediction_transitions.csv').open('w',encoding='utf-8',newline='') as stream:
        writer = csv.writer(stream);writer.writerow(['comparison','kind','true_class','old_prediction','new_prediction','count'])
        for name,transitions in s['prediction_transitions'].items():
            for t in transitions:writer.writerow([name,t['kind'],t['true_class'],t['equal_prediction'],t['gate_prediction'],t['count']])
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    print(summarize(parser.parse_args().root))
