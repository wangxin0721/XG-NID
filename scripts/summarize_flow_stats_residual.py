"""Read-only validation analysis; no torch, checkpoint or test data imports."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from e3_gate_study import read, require, write


def analyze(r):
    require(r['status'] == 'completed' and r['test_evaluated'] is False, 'Incomplete validation screen')
    models = [('N0', r['N0']), ('frozen_V2', r['frozen_V2']),
              ('lower_lr_J0', r['reference_joint']['J0']),
              ('lower_lr_J2', r['reference_joint']['J2']), ('J2_flow_stats', r['candidate'])]
    rows = [{'model': name, 'macro_f1': m['best_validation']['macro_f1'],
        'per_class': m['best_validation']['per_class'], 'best_epoch': m['best_epoch'],
        'stop_epoch': m.get('epochs', m.get('stopped_epoch')),
        'updates': m.get('updates', m.get('optimizer_updates')),
        'joint_wall_seconds': m.get('joint_wall_seconds'),
        'selected_epoch_zero': m['best_epoch'] == 0, 'training_evidence': m.get('training_evidence')}
        for name, m in models]
    s = {row['model']: row['macro_f1'] for row in rows}
    strong = max(s['frozen_V2'], s['lower_lr_J2'])
    gain = 100 * (s['J2_flow_stats'] - strong)
    threshold = r['protocol']['screening']['minimum_gain_over_stronger_reference_pp']
    return {'rows': rows,
        'candidate_minus_N0_pp': 100 * (s['J2_flow_stats'] - s['N0']),
        'candidate_minus_frozen_V2_pp': 100 * (s['J2_flow_stats'] - s['frozen_V2']),
        'candidate_minus_lower_lr_J2_pp': 100 * (s['J2_flow_stats'] - s['lower_lr_J2']),
        'candidate_minus_stronger_reference_pp': gain,
        'budget_screening': {'threshold_pp': threshold, 'score_threshold_passed': gain >= threshold,
            'class_tradeoff_review_required': True, 'statistical_significance_claimed': False,
            'automatic_followup': False},
        'stats_on_minus_off_pp': r['stats_on_minus_off_pp'], 'comparisons': r['comparisons'],
        'stage_A': r['stage_A'], 'attribution': r['attribution']}


def summarize(root):
    r = read(root / 'summary.json')
    a = analyze(r)
    write(root / 'analysis.json', a)
    lines = ['| model | validation Macro-F1 | best epoch | stop epoch | updates | joint seconds |',
             '|---|---:|---:|---:|---:|---:|']
    for row in a['rows']:
        lines.append(f"| {row['model']} | {100*row['macro_f1']:.6f}% | {row['best_epoch']} | "
                     f"{row['stop_epoch']} | {row['updates']} | {row['joint_wall_seconds']} |")
    for key in ('candidate_minus_N0_pp', 'candidate_minus_frozen_V2_pp',
                'candidate_minus_lower_lr_J2_pp', 'candidate_minus_stronger_reference_pp'):
        lines.append(f"\n{key}: {a[key]:+.6f}")
    lines += ['', 'Budget screen (>= 0.10 pp above the stronger frozen-V2/lower-LR-J2 reference): '
              + str(a['budget_screening']['score_threshold_passed']) + '. Class review is still required.',
              '', f"Same-checkpoint path-on minus path-off: {a['stats_on_minus_off_pp']:+.6f} pp.",
              'Path-off is a post-selection dependency diagnostic, not a separately trained capacity ablation.']
    for name, counts in a['comparisons'].items():
        lines.append(f"\nAgainst {name}: repaired={counts['counts']['repaired']}, "
                     f"regressed={counts['counts']['regressed']}, net_correct={counts['net_correct']}.")
    lines += ['', 'All eight classes (P / R / F1 in percent):', '',
              '| class | lower-LR J2 P/R/F1 | residual P/R/F1 | F1 difference pp |',
              '|---|---:|---:|---:|']
    old = r['reference_joint']['J2']['best_validation']['per_class']
    new = r['candidate']['best_validation']['per_class']
    fields = ('precision', 'recall', 'f1-score')
    labels = [k for k, v in new.items() if isinstance(v, dict) and k not in ('macro avg', 'weighted avg', 'micro avg')]
    for name in labels:
        left = ' / '.join(f'{100*old[name][k]:.4f}' for k in fields)
        right = ' / '.join(f'{100*new[name][k]:.4f}' for k in fields)
        lines.append(f"| {name} | {left} | {right} | {100*(new[name]['f1-score']-old[name]['f1-score']):+.6f} |")
    lines += ['', 'Candidate selected epoch0: ' + str(r['candidate']['selected_epoch_zero']),
              'Candidate training evidence: ' + str(r['candidate']['training_evidence']),
              'Stage A is reused, its historical missing duration remains unavailable: ' + str(r['stage_A']),
              'New preflight seconds: ' + str(r['preflight_wall_seconds']),
              'New post-training diagnostic seconds: ' + str(r['post_training_diagnostics_wall_seconds']),
              r['attribution'], 'No test evaluation, no automatic next run, no change to old V2 screening.']
    (root / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    with (root / 'per_class.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', 'class', 'precision', 'recall', 'f1', 'support'])
        for row in a['rows']:
            for label in labels:
                m = row['per_class'][label]
                writer.writerow([row['model'], label, *(m[k] for k in fields), m['support']])
        for label in labels:
            m = r['stats_off']['per_class'][label]
            writer.writerow(['same_checkpoint_stats_off', label, *(m[k] for k in fields), m['support']])
    return '\n'.join(lines)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    print(summarize(p.parse_args().root))
