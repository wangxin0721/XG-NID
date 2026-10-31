"""Report own-start gains separately from the inherited 32D gate advantage."""
import argparse
import csv
from pathlib import Path

from e3_gate_study import read, require, write
from run_fusion_seed_repeats import verify_metrics


def changed_samples(reference, candidate, labels):
    require(len(reference) == len(candidate), 'Sample counts differ')
    require(len({r['sample_id'] for r in reference}) == len(reference), 'Duplicate reference IDs')
    result = []
    for a, b in zip(reference, candidate):
        require(a['sample_id'] == b['sample_id'] and a['label'] == b['label'], 'Sample IDs/order/labels differ')
        y, old, new = a['label'], a['prediction'], b['prediction']
        require(all(isinstance(x, int) and 0 <= x < len(labels) for x in (y, old, new)), 'Invalid class label')
        if old != new:
            result.append({'sample_id': a['sample_id'], 'label': y, 'true_class': labels[y],
                'old_prediction': old, 'new_prediction': new, 'old_class': labels[old], 'new_class': labels[new],
                'change': 'repaired' if new == y else ('regressed' if old == y else 'both_wrong_changed')})
    return result


def analyze(r):
    require(r['status'] == 'completed' and r['exit_code'] == 0 and r['tensor_preflight_passed']
            and r['encoder_BN_buffers_unchanged'] and r['encoder_parameters_trainable'] and r['raw_graph_training']
            and not r['frozen_cache_used_for_training'] and r['source_artifacts_unchanged']
            and r['test_evaluated'] is False and r['automatic_followup'] is False, 'Incomplete raw-graph J32 experiment')
    e, g = r['models']['J32-Equal'], r['models']['J32-Gate']
    se, sg = r['source_models']['D32-Equal'], r['source_models']['D32-Gate']
    eligible = r['protocol']['include_epoch_zero']
    require(type(eligible) is bool, 'Explicit epoch0 selection policy required')
    common = min(e['epochs'], g['epochs'])
    require(e['orders'][:common] == g['orders'][:common] and r['common_epochs_order_checked'] == common
            and e['training_start_rng_sha256'] == g['training_start_rng_sha256'], 'Paired order/RNG differs')
    for name, start in (('J32-Equal', se), ('J32-Gate', sg)):
        m = r['models'][name]
        for phase in ('epoch_zero', 'best_validation', 'last_validation'):
            verify_metrics(m[phase])
        verify_metrics(start['best_validation'])
        require(m['epoch_zero']['confusion_matrix'] == start['best_validation']['confusion_matrix']
                and m['epoch_zero_eligible'] == eligible and m['selected_epoch_zero'] == (m['best_epoch'] == 0)
                and m['best_epoch'] >= (0 if eligible else 1), 'Own-start selection differs')
        if eligible:
            require(m['best_validation']['macro_f1'] >= m['epoch_zero']['macro_f1'], 'Eligible starting model was lost')
        for kind in ('repaired', 'regressed'):
            require(sum(x['count'] for x in r['own_start_transitions'][name] if x['kind'] == kind)
                    == r['own_start_changes'][name]['counts'][kind], 'Own-start transitions disagree')
    start_e, start_g = se['best_validation']['macro_f1'], sg['best_validation']['macro_f1']
    best_e, best_g = e['best_validation']['macro_f1'], g['best_validation']['macro_f1']
    delta_e, delta_g = 100*(best_e-start_e), 100*(best_g-start_g)
    before, after = 100*(start_g-start_e), 100*(best_g-best_e)
    stronger = max(start_g, best_e)
    margin = 100*(best_g-stronger)
    threshold = r['protocol']['budget_gate_above_stronger_reference_pp']
    require(abs((after-before)-(delta_g-delta_e)) < 1e-10, 'Gain accounting differs')
    return {'Equal_gain_from_own_start_pp': delta_e, 'Gate_gain_from_own_start_pp': delta_g,
        'initial_Gate_minus_Equal_pp': before, 'final_Gate_minus_Equal_pp': after,
        'gap_change_from_joint_training_pp': after-before,
        'stronger_reference_macro_f1': stronger, 'Gate_minus_stronger_reference_pp': margin,
        'required_Gate_macro_f1_for_budget': stronger+threshold/100,
        'budget_threshold_pp': threshold, 'budget_screen_passed': margin >= threshold,
        'epoch_zero_eligible': eligible,
        'Gate_selected_epoch_zero': g['selected_epoch_zero'], 'Equal_selected_epoch_zero': e['selected_epoch_zero'],
        'Gate_improved_over_own_start': delta_g > 0, 'Equal_improved_over_own_start': delta_e > 0,
        'Equal_overtook_Gate': best_e > best_g, 'class_tradeoff_review_required': True,
        'statistical_significance_claimed': False, 'automatic_followup': False,
        'attribution': 'The two arms start from their own different mature checkpoints. Own-start deltas measure '
                       'additional gains from this joint fine-tuning protocol; the inherited gate advantage is not '
                       'a new joint-training gain. Encoder eval mode fixes BN buffers while encoder affine/other '
                       'parameters remain trainable. This pair does not isolate gate capacity from adaptation.',
        'models': r['models'], 'source_models': r['source_models'],
        'own_start_changes': r['own_start_changes'], 'own_start_transitions': r['own_start_transitions'],
        'Gate_versus_Equal': r['Gate_versus_Equal'], 'Gate_versus_Equal_transitions': r['Gate_versus_Equal_transitions'],
        'common_epochs_order_checked': common, 'graph_replays': r['graph_replays'],
        'encoder_BN_buffers_unchanged': True, 'raw_graph_training': True,
        'discarded_probe_updates': r['discarded_probe_updates'], 'preflight_wall_seconds': r['preflight_wall_seconds']}


def summarize(root):
    r = read(root/'summary.json')
    a = analyze(r)
    write(root/'analysis.json', a)
    lines = ['| model | start Macro-F1 | best Macro-F1 | gain pp | last Macro-F1 | best epoch | stop epoch | updates | joint seconds |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, m in r['models'].items():
        zero, best, last = [m[k]['macro_f1'] for k in ('epoch_zero', 'best_validation', 'last_validation')]
        lines.append(f"| {name} | {100*zero:.6f}% | {100*best:.6f}% | {100*(best-zero):+.6f} | {100*last:.6f}% | "
                     f"{m['best_epoch']} | {m['epochs']} | {m['optimizer_updates']} | {m['training_wall_seconds']:.3f} |")
    lines += ['', f"Inherited Gate-Equal gap: {a['initial_Gate_minus_Equal_pp']:+.6f} pp.",
              f"Final Gate-Equal gap: {a['final_Gate_minus_Equal_pp']:+.6f} pp.",
              f"Gap change from joint training: {a['gap_change_from_joint_training_pp']:+.6f} pp.",
              f"Gate above stronger original-Gate/new-Equal reference: {a['Gate_minus_stronger_reference_pp']:+.6f} pp.",
              f"Budget screen >= +{a['budget_threshold_pp']:.2f} pp: {a['budget_screen_passed']}; "
              f"required score={100*a['required_Gate_macro_f1_for_budget']:.6f}%.",
              f"Selected epoch0: Equal={a['Equal_selected_epoch_zero']}, Gate={a['Gate_selected_epoch_zero']}.",
              ('Epoch0 is eligible. Matching the starting score alone is not evidence of benefit.'
               if a['epoch_zero_eligible'] else
               'Epoch0 is recorded but excluded. Select trained epochs only, even when all are below the starting score.'),
              'No head reinitialization or warmup. Fresh Adam: encoders1e-6, projections/head1e-5, gate1e-5; WD0.',
              f"Raw graphs, batch512/16, max{r['protocol']['epochs']}/patience{r['protocol']['patience']}. "
              'Encoder modules remain eval but allow gradients; BN buffers fixed.',
              '', '| class | J32-Equal P/R/F1 (%) | J32-Gate P/R/F1 (%) | Gate F1 gain from own start pp |', '|---|---:|---:|---:|']
    e, g = r['models']['J32-Equal'], r['models']['J32-Gate']
    start_g = r['source_models']['D32-Gate']['best_validation']['per_class']
    fields = ('precision', 'recall', 'f1-score')
    for label, v in g['best_validation']['per_class'].items():
        values = [' / '.join(f"{100*m['best_validation']['per_class'][label][k]:.4f}" for k in fields) for m in (e, g)]
        lines.append(f"| {label} | {values[0]} | {values[1]} | {100*(v['f1-score']-start_g[label]['f1-score']):+.6f} |")
    lines += ['', 'Review WebBased FN/FP separately, Spoofing changes and the small BruteForce sample count.']
    for name, m in r['models'].items():
        ch = r['own_start_changes'][name]
        lines += [f"{name} versus own start: repaired={ch['counts']['repaired']}, regressed={ch['counts']['regressed']}, net_correct={ch['net_correct']}.",
                  f"{name} best-trained evidence: {m['training_evidence']}.",
                  f"{name} trainable parameters={m['joint_trainable_parameters']}; groups={m['parameter_groups']}."]
    ch = r['Gate_versus_Equal']
    lines += [f"Final Gate versus Equal: repaired={ch['counts']['repaired']}, regressed={ch['counts']['regressed']}, net_correct={ch['net_correct']}."]
    for name, m in r['models'].items():
        for phase in ('best_validation', 'last_validation'):
            labels = list(m[phase]['per_class'])
            lines += ['', f'{name} {phase} confusion matrix (rows=true):', '',
                      '| true / predicted | '+' | '.join(labels)+' |', '|---|'+'---:|'*8]
            lines += ['| '+label+' | '+' | '.join(str(n) for n in row)+' |'
                      for label, row in zip(labels, m[phase]['confusion_matrix'])]
    lines += ['', 'Per-class best/last/source metrics: per_class.csv. Prediction destinations: prediction_transitions.csv.',
              'Changed samples: each arm changes_from_own_start.jsonl.gz; root gate_vs_equal_changes.jsonl.gz.',
              'Prediction files retain all 32 gate weights for every validation sample.',
              f"Common epoch order/RNG verified: {a['common_epochs_order_checked']}; encoder BN buffers unchanged.",
              'Best checkpoints independently reconstructed, strictly loaded and replayed on all raw validation graphs.',
              f"Preflight seconds={a['preflight_wall_seconds']:.3f}; discarded probe updates={a['discarded_probe_updates']}.",
              'Joint seconds include training, epoch validation and checkpoint writes; final replay is separately recorded.',
              a['attribution'], 'No test evaluation, significance claim, extra seed or automatic next training.']
    (root/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    with (root/'per_class.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', 'checkpoint', 'class', 'precision', 'recall', 'f1', 'support'])
        for name, m in [*r['source_models'].items(), *r['models'].items()]:
            for phase in ('best_validation', 'last_validation'):
                for label, v in m[phase]['per_class'].items():
                    writer.writerow([name, phase, label, *(v[k] for k in fields), v['support']])
    with (root/'prediction_transitions.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['comparison', 'kind', 'true_class', 'old_prediction', 'new_prediction', 'count'])
        comparisons = {**r['own_start_transitions'], 'final_Gate_versus_Equal': r['Gate_versus_Equal_transitions']}
        for name, transitions in comparisons.items():
            for row in transitions:
                writer.writerow([name, row['kind'], row['true_class'], row['equal_prediction'], row['gate_prediction'], row['count']])
    return '\n'.join(lines)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    print(summarize(p.parse_args().root))
