"""Validation-only matched D32 equal/gate report; no torch dependency."""
import argparse
from collections import Counter
import csv
from pathlib import Path

from e3_gate_study import read, require, write
from run_fusion_seed_repeats import verify_metrics


def prediction_transitions(equal, gate, labels):
    require(len(equal) == len(gate), 'Prediction counts differ')
    counts = Counter()
    for a, b in zip(equal, gate):
        require(a['sample_id'] == b['sample_id'] and a['label'] == b['label'], 'Prediction IDs/labels differ')
        truth, old, new = a['label'], a['prediction'], b['prediction']
        require(all(isinstance(x, int) and 0 <= x < len(labels) for x in (truth, old, new)), 'Invalid prediction label')
        if old == new:
            continue
        kind = 'repaired' if new == truth else ('regressed' if old == truth else 'both_wrong_changed')
        counts[(kind, truth, old, new)] += 1
    return [{'kind': kind, 'true_class': labels[y], 'equal_prediction': labels[a],
             'gate_prediction': labels[b], 'count': n} for (kind, y, a, b), n in sorted(counts.items())]


def analyze(r):
    require(r['status'] == 'completed' and r['exit_code'] == 0 and r['test_evaluated'] is False
            and r['automatic_followup'] is False and r['tensor_preflight_passed'], 'Incomplete validation pair')
    equal, gate, old = [r['models'][k] if k in r['models'] else r['historical_D32']
                        for k in ('D32-Equal', 'D32-Gate', 'historical')]
    for m in (equal, gate, old):
        for key in ('epoch_zero', 'best_validation', 'last_validation'):
            verify_metrics(m[key])
    require(equal['architecture']['trainable_parameters'] == 5480
            and gate['architecture']['trainable_parameters'] == 7144
            and equal['epoch_zero_eligible'] is False and gate['epoch_zero_eligible'] is False
            and equal['best_epoch'] > 0 and gate['best_epoch'] > 0, 'Incorrect control/selection')
    common = min(equal['epochs'], gate['epochs'])
    require(r['common_epochs_order_checked'] == common and equal['orders'][:common] == gate['orders'][:common]
            and equal['training_start_rng_sha256'] == gate['training_start_rng_sha256'] == old['training_start_rng_sha256'],
            'Paired training RNG/order differs')
    delta = 100 * (gate['best_validation']['macro_f1'] - equal['best_validation']['macro_f1'])
    old_delta = 100 * (gate['best_validation']['macro_f1'] - old['best_validation']['macro_f1'])
    changed = abs(old_delta) >= r['protocol']['historical_rerun_material_difference_pp']
    transitions = r['prediction_transitions']
    for name in ('repaired', 'regressed'):
        require(sum(x['count'] for x in transitions if x['kind'] == name)
                == r['Gate_versus_Equal']['counts'][name], 'Transition counts disagree')
    return {'Gate_minus_Equal_pp': delta, 'Gate_minus_historical_D32_pp': old_delta,
        'budget_threshold_pp': r['protocol']['budget_gate_minus_equal_pp'],
        'budget_screen_passed': delta >= r['protocol']['budget_gate_minus_equal_pp'],
        'historical_rerun_materially_different': changed,
        'historical_rerun_review_required': changed,
        'gate_contribution_interpretation_pending_execution_review': changed,
        'class_tradeoff_review_required': True, 'statistical_significance_claimed': False,
        'primary_comparison': 'newly_trained_D32-Gate_minus_newly_trained_D32-Equal',
        'attribution': 'The matched pair measures the effect of adding this trainable dynamic gate, including its '
                       '1,664 parameters and optimization. It does not isolate adaptation from gate capacity. '
                       'The historical D32 is a rerun reference, never a training start. No fraction of the '
                       'absolute 97.32% score can be assigned to the gate.',
        'epoch_zero_eligible': False, 'automatic_followup': False,
        'prediction_changes': r['Gate_versus_Equal'], 'prediction_transitions': transitions,
        'Gate_versus_historical_D32': r['Gate_versus_historical_D32'],
        'models': r['models'], 'graph_replays': r['graph_replays'], 'initialization': r['initialization'],
        'encoder_parameters_and_BN_unchanged': r['encoder_parameters_and_BN_unchanged'],
        'source_artifacts_unchanged': r['source_artifacts_unchanged'],
        'shared_cache_unchanged': r['shared_cache_unchanged'], 'common_epochs_order_checked': common,
        'shared_cache_reuse_wall_seconds': r['shared_cache_reuse_wall_seconds'],
        'preflight_including_cache_wall_seconds': r['preflight_including_cache_wall_seconds'],
        'discarded_probe_updates': r['discarded_probe_updates'],
        'execution_comparison': r['execution_comparison']}


def summarize(root):
    r = read(root / 'summary.json')
    a = analyze(r)
    write(root / 'analysis.json', a)
    lines = ['| model | best validation Macro-F1 | last Macro-F1 | epoch0 Macro-F1 | best epoch | stop epoch | updates | train seconds | active parameters |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, m in r['models'].items():
        lines.append(f"| {name} | {100*m['best_validation']['macro_f1']:.6f}% | {100*m['last_validation']['macro_f1']:.6f}% | "
                     f"{100*m['epoch_zero']['macro_f1']:.6f}% | {m['best_epoch']} | {m['epochs']} | "
                     f"{m['optimizer_updates']} | {m['training_wall_seconds']:.3f} | {m['architecture']['trainable_parameters']} |")
    h = r['historical_D32']
    lines += [f"| historical D32 | {100*h['best_validation']['macro_f1']:.6f}% | {100*h['last_validation']['macro_f1']:.6f}% | "
              f"{100*h['epoch_zero']['macro_f1']:.6f}% | {h['best_epoch']} | {h['epochs']} | {h['optimizer_updates']} | "
              f"{h['training_wall_seconds']:.3f} | 7144 |", '',
              f"Gate-Equal: {a['Gate_minus_Equal_pp']:+.6f} pp.",
              f"Gate-historical D32: {a['Gate_minus_historical_D32_pp']:+.6f} pp.",
              f"Budget screen (>= +{a['budget_threshold_pp']:.2f} pp): {a['budget_screen_passed']}; class review still required.",
              'Both arms reuse the same clean D32 snapshot, NOT the epoch62 checkpoint. Epoch0 is recorded but excluded.',
              'Adam lr=1e-3, weight_decay=0, train/eval batch512/16, max100/patience20; encoders/BN fixed in eval.',
              'Equal stores 1,664 unused frozen gate parameters for strict checkpoint compatibility; its active count is 5,480.',
              '', '| class | Equal best P/R/F1 (%) | Gate best P/R/F1 (%) | F1 delta pp |', '|---|---:|---:|---:|']
    fields = ('precision', 'recall', 'f1-score')
    x, y = [r['models'][name]['best_validation']['per_class'] for name in ('D32-Equal', 'D32-Gate')]
    for label in x:
        values = [' / '.join(f'{100*z[label][key]:.4f}' for key in fields) for z in (x, y)]
        lines.append(f"| {label} | {values[0]} | {values[1]} | {100*(y[label]['f1-score']-x[label]['f1-score']):+.6f} |")
    changes = a['prediction_changes']
    lines += ['', f"Gate versus Equal: repaired={changes['counts']['repaired']}, regressed={changes['counts']['regressed']}, net_correct={changes['net_correct']}.",
              'Inspect WebBased missed attacks, Spoofing losses, and BruteForce gains with the complete eight-class results.',
              'Prediction destinations (all changed samples):', '',
              '| change | true class | Equal prediction | Gate prediction | samples |', '|---|---|---|---|---:|']
    for t in a['prediction_transitions']:
        lines.append(f"| {t['kind']} | {t['true_class']} | {t['equal_prediction']} | {t['gate_prediction']} | {t['count']} |")
    for name, m in r['models'].items():
        for phase in ('best_validation', 'last_validation'):
            lines += ['', f'{name} {phase} confusion matrix (rows=true, columns=prediction):', '',
                      '| true / predicted | ' + ' | '.join(x) + ' |', '|---|' + '---:|' * len(x)]
            for label, row in zip(x, m[phase]['confusion_matrix']):
                lines.append('| ' + label + ' | ' + ' | '.join(str(n) for n in row) + ' |')
    lines += ['', f"Common paired epochs verified: {a['common_epochs_order_checked']}.",
              f"Public initialization identical: {a['initialization']['public_initialization_identical']}; full initial validation equality passed.",
              f"Encoder parameters/BN unchanged: {a['encoder_parameters_and_BN_unchanged']}; shared caches unchanged: {a['shared_cache_unchanged']}.",
              'Best checkpoints strictly reloaded and replayed through raw validation graphs; all predictions/CM reproduced.',
              f"Shared cache load/validation seconds (charged once): {a['shared_cache_reuse_wall_seconds']:.3f}.",
              f"Preflight including cache seconds: {a['preflight_including_cache_wall_seconds']:.3f}; discarded probe updates: {a['discarded_probe_updates']}.",
              f"Historical rerun materially different (>= {r['protocol']['historical_rerun_material_difference_pp']:.2f} pp): {a['historical_rerun_materially_different']}.",
              'Execution comparison: ' + str(a['execution_comparison']),
              'If the historical rerun differs materially, explain the execution differences before interpreting gate contribution.',
              a['attribution'], 'No significance claim. No test evaluation, extra seeds or automatic joint training.']
    (root / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    with (root / 'per_class.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', 'checkpoint', 'class', 'precision', 'recall', 'f1', 'support'])
        for name, m in r['models'].items():
            for phase in ('best_validation', 'last_validation'):
                for label, v in m[phase]['per_class'].items():
                    writer.writerow([name, phase, label, *(v[k] for k in fields), v['support']])
    with (root / 'prediction_transitions.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['kind', 'true_class', 'equal_prediction', 'gate_prediction', 'count'])
        writer.writeheader()
        writer.writerows(a['prediction_transitions'])
    return '\n'.join(lines)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    print(summarize(p.parse_args().root))
