"""Validation-only P0/P1 readout report, with matched and mature references."""
import argparse
import csv
import json
from pathlib import Path

from e3_gate_study import read, require, write
from run_fusion_seed_repeats import verify_metrics


def budget(p0, p1, baseline, threshold):
    stronger = max(p0, baseline)
    delta = 100 * (p1 - stronger)
    return {'P1_minus_P0_pp': 100 * (p1 - p0), 'P1_minus_D32_pp': 100 * (p1 - baseline),
        'stronger_reference_macro_f1': stronger, 'P1_minus_stronger_reference_pp': delta,
        'budget_threshold_pp': threshold, 'required_P1_macro_f1': stronger + threshold / 100,
        'budget_screen_passed': delta >= threshold}


def analyze(r):
    from run_packet_pool_pair import architecture, check_config
    check_config(r['protocol'])
    require(r['status'] == 'completed' and r['exit_code'] == 0
            and all(r[k] is True for k in ('tensor_preflight_passed', 'encoder_parameters_and_BN_unchanged',
                'node_caches_unchanged', 'source_artifacts_unchanged'))
            and r['test_evaluated'] is False and r['automatic_followup'] is False
            and r['graph_load_counts']['test'] == 0, 'Incomplete P0/P1 validation experiment')
    baseline = r['source_model']['best_validation']
    require(all(r['initialization'][k] is True for k in ('P0_P1_outputs_allclose',
                'source_predictions_identical', 'uniform_packet_attention', 'epoch_zero_eligible')),
            'Initialization verification missing')
    verify_metrics(baseline)
    require(r['source_model']['best_epoch'] == 62
            and abs(baseline['macro_f1'] - r['protocol']['source_macro_f1']) < 1e-12, 'Wrong mature baseline')
    p0, p1 = [r['models'][n] for n in ('P0', 'P1')]
    common = min(p0['epochs'], p1['epochs'])
    require(common == r['common_epochs_order_checked'] and p0['orders'][:common] == p1['orders'][:common]
            and p0['training_start_rng_sha256'] == p1['training_start_rng_sha256'], 'Matched order/RNG differs')
    for name, mode in (('P0', 'mean'), ('P1', 'attention')):
        m = r['models'][name]
        for phase in ('epoch_zero', 'best_validation', 'last_validation'):
            verify_metrics(m[phase])
        require(m['architecture'] == architecture(mode) and m['epoch_zero_eligible'] is True
                and m['selected_epoch_zero'] == (m['best_epoch'] == 0)
                and 0 <= m['best_epoch'] <= m['epochs'] <= r['protocol']['epochs']
                and m['epoch_zero']['confusion_matrix'] == baseline['confusion_matrix']
                and m['best_validation']['macro_f1'] >= baseline['macro_f1'], 'Epoch0/selection/architecture differs')
        require(r['probes'][name]['clean_model_and_rng_restored'] and r['probes'][name]['optimizer_discarded']
                and r['graph_replays'][name]['strict_state_load'] and r['graph_replays'][name]['predictions_identical']
                and r['graph_replays'][name]['confusion_matrix_reproduced'], 'Probe/replay verification missing')
    for name, changes in r['comparisons'].items():
        for kind in ('repaired', 'regressed'):
            require(sum(t['count'] for t in r['prediction_transitions'][name] if t['kind'] == kind)
                    == changes['counts'][kind], 'Prediction transition counts differ')
    a = budget(p0['best_validation']['macro_f1'], p1['best_validation']['macro_f1'], baseline['macro_f1'],
               r['protocol']['budget_above_stronger_reference_pp'])
    a.update(P0_gain_from_D32_pp=100 * (p0['best_validation']['macro_f1'] - baseline['macro_f1']),
        selected_epoch_zero={n: m['selected_epoch_zero'] for n, m in r['models'].items()},
        class_tradeoff_review_required=True, statistical_significance_claimed=False, automatic_followup=False,
        attribution='P1-P0 measures the packet readout change including its 1,056 added parameters and training. '
                    'P0 measures the same mature checkpoint with continued downstream training. '
                    'The budget threshold is not a statistical significance criterion.')
    a['class_changes'] = {}
    for label in baseline['per_class']:
        b, x, y = [m['per_class'][label] for m in (baseline, p0['best_validation'], p1['best_validation'])]
        a['class_changes'][label] = {'P1_minus_P0_recall_pp': 100 * (y['recall'] - x['recall']),
            'P1_minus_D32_recall_pp': 100 * (y['recall'] - b['recall']),
            'P1_minus_P0_F1_pp': 100 * (y['f1-score'] - x['f1-score']),
            'P1_minus_D32_F1_pp': 100 * (y['f1-score'] - b['f1-score']), 'support': b['support']}
    return a


def summarize(root):
    r = read(root / 'summary.json')
    a = analyze(r)
    from run_packet_pool_pair import check_history
    for name, m in r['models'].items():
        history = [json.loads(s) for s in (root / name / 'epochs.jsonl').read_text(encoding='utf-8').splitlines()]
        check_history(history, m, r['protocol'])
    write(root / 'analysis.json', a)
    lines = ['| model | start Macro-F1 | best Macro-F1 | last Macro-F1 | best epoch | stop epoch | updates | seconds |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, m in r['models'].items():
        lines.append(f"| {name} | {100*m['epoch_zero']['macro_f1']:.6f}% | {100*m['best_validation']['macro_f1']:.6f}% | "
                     f"{100*m['last_validation']['macro_f1']:.6f}% | {m['best_epoch']} | {m['epochs']} | "
                     f"{m['optimizer_updates']} | {m['training_wall_seconds']:.3f} |")
    lines += ['', f"P1-P0={a['P1_minus_P0_pp']:+.6f} pp; P1-D32={a['P1_minus_D32_pp']:+.6f} pp; "
              f"P0-D32={a['P0_gain_from_D32_pp']:+.6f} pp.",
              f"Delta=P1-max(P0,D32)={a['P1_minus_stronger_reference_pp']:+.6f} pp; "
              f"budget >= {a['budget_threshold_pp']:.2f} pp: {a['budget_screen_passed']}; "
              f"required P1={100*a['required_P1_macro_f1']:.6f}%.",
              'Both start from D32-Gate epoch62. Epoch0 is eligible; retaining it is not evidence of a gain.',
              'Frozen encoders/BN; existing projection/gate/head lr1e-5; new pool lr1e-4; fresh Adam WD0.',
              'Final NLL, seed42, batch512/16, max100/patience20; active parameters P0=7144, P1=8200.',
              '', '| class | D32 recall | P0 recall | P1 recall | P1-P0 recall pp | P1-D32 recall pp | P1-P0 F1 pp |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for label, v in a['class_changes'].items():
        recalls = [r['source_model']['best_validation']['per_class'][label]['recall']]
        recalls += [r['models'][n]['best_validation']['per_class'][label]['recall'] for n in ('P0', 'P1')]
        lines.append('| ' + label + ' | ' + ' | '.join(f'{100*x:.6f}%' for x in recalls) +
            f" | {v['P1_minus_P0_recall_pp']:+.6f} | {v['P1_minus_D32_recall_pp']:+.6f} | {v['P1_minus_P0_F1_pp']:+.6f} |")
    lines += ['', 'Review WebBased/Recon recall and Spoofing/BruteForce tradeoffs with support counts in per_class.csv.']
    for name, ch in r['comparisons'].items():
        lines += [f"{name}: repaired={ch['counts']['repaired']}, regressed={ch['counts']['regressed']}, net_correct={ch['net_correct']}."]
    lines += ['', '| partition | class | graphs | multi-packet graphs | varying multi-packet graphs | mean packets | mean multi-packet RMS distance |',
              '|---|---|---:|---:|---:|---:|---:|']
    for part, groups in r['packet_diagnostics'].items():
        for label, v in groups.items():
            lines.append(f"| {part} | {label} | {v['graphs']} | {v['multi_packet_graphs']} | "
                         f"{v['varying_multi_packet_graphs']} | {v['mean_packets']:.4f} | {v['multi_packet_rms_mean']} |")
    for name, m in r['models'].items():
        lines += ['', f"{name} trained-epoch evidence: {m['training_evidence']}; parameter groups: {m['parameter_groups']}."]
        for phase in ('best_validation', 'last_validation'):
            labels = list(m[phase]['per_class'])
            lines += ['', f'{name} {phase} confusion matrix (rows=true):', '',
                      '| true / predicted | ' + ' | '.join(labels) + ' |', '|---|' + '---:|' * 8]
            lines += ['| ' + label + ' | ' + ' | '.join(str(x) for x in row) + ' |'
                      for label, row in zip(labels, m[phase]['confusion_matrix'])]
    lines += ['', 'Full class P/R/F1/support: per_class.csv; prediction destinations: prediction_transitions.csv.',
              'Changed sample IDs: *_changed_samples.jsonl.gz, for both arms versus D32 and P1 versus P0.',
              'Each validation prediction retains all 32 fusion weights and per-packet attention in original node order.',
              'Packet counts and within-graph representation distances: packet_diagnostics.json.',
              f"Common epochs with identical order/RNG: {r['common_epochs_order_checked']}; discarded probe updates=6.",
              'Both best checkpoints strictly reconstructed and replayed on every original validation graph.',
              a['attribution'], 'No test evaluation or automatic follow-up experiment.']
    (root / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    with (root / 'per_class.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', 'checkpoint', 'class', 'precision', 'recall', 'f1', 'support'])
        for name, m in [('D32-Gate', r['source_model']), *r['models'].items()]:
            phases = ('best_validation',) if name == 'D32-Gate' else ('best_validation', 'last_validation')
            for phase in phases:
                for label, v in m[phase]['per_class'].items():
                    writer.writerow([name, phase, label, *(v[k] for k in ('precision', 'recall', 'f1-score', 'support'))])
    with (root / 'prediction_transitions.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['comparison', 'kind', 'true_class', 'old_prediction', 'new_prediction', 'count'])
        for name, transitions in r['prediction_transitions'].items():
            for t in transitions:
                writer.writerow([name, t['kind'], t['true_class'], t['equal_prediction'], t['gate_prediction'], t['count']])
    return '\n'.join(lines)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    print(summarize(p.parse_args().root))
