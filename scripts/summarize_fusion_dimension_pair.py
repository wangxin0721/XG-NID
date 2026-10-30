"""Summarize the fresh-head 8/32 pair without torch or graph reads."""
import argparse
import csv
from pathlib import Path

from e3_gate_study import read, require, write


def analyze(r):
    require(r['status'] == 'completed' and r['test_evaluated'] is False, 'Incomplete validation pair')
    a, b = r['models']['D8'], r['models']['D32']
    delta = 100 * (b['best_validation']['macro_f1'] - a['best_validation']['macro_f1'])
    historical_delta = 100 * (b['best_validation']['macro_f1'] - r['historical_J2']['best_validation']['macro_f1'])
    require(abs(delta - r['D32_minus_D8_pp']) < 1e-12
            and abs(historical_delta - r['D32_minus_historical_J2_pp']) < 1e-12, 'Difference report disagrees')
    return {'D32_minus_D8_pp': delta, 'D32_minus_historical_J2_pp': historical_delta,
        'D32_above_paired_D8': delta > 0, 'D32_above_historical_J2': historical_delta > 0,
        'models': {k: {name: m[name] for name in ('architecture', 'epoch_zero', 'best_validation',
            'last_validation', 'best_epoch', 'epochs', 'optimizer_updates', 'training_wall_seconds', 'training_evidence')}
            for k, m in r['models'].items()},
        'paired_prediction_changes': r['D32_versus_D8'], 'D32_versus_N0': r['D32_versus_N0'],
        'class_tradeoff_review_required': True, 'statistical_significance_claimed': False,
        'automatic_followup': False, 'epoch_zero_eligible': False,
        'only_this_pair_budget': {'epochs': r['protocol']['epochs'], 'patience': r['protocol']['patience']},
        'shared_cache_wall_seconds': r['shared_cache_wall_seconds'],
        'common_epochs_order_checked': r['common_epochs_order_checked'], 'attribution': r['attribution']}


def summarize(root):
    r = read(root / 'summary.json')
    a = analyze(r)
    write(root / 'analysis.json', a)
    lines = ['| model | validation Macro-F1 | epoch0 Macro-F1 | best epoch | stop epoch | updates | training seconds |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, m in r['models'].items():
        lines.append(f"| {name} | {100*m['best_validation']['macro_f1']:.6f}% | {100*m['epoch_zero']['macro_f1']:.6f}% | "
                     f"{m['best_epoch']} | {m['epochs']} | {m['optimizer_updates']} | {m['training_wall_seconds']:.3f} |")
    h = r['historical_J2']
    lines += [f"| historical lower-LR J2 | {100*h['best_validation']['macro_f1']:.6f}% | — | {h['best_epoch']} | "
              f"{h['epochs']} | {h['updates']} | {h['joint_wall_seconds']:.3f} |", '',
              f"D32-D8: {a['D32_minus_D8_pp']:+.6f} pp.",
              f"D32-historical J2: {a['D32_minus_historical_J2_pp']:+.6f} pp.",
              'Both new heads are freshly initialized. Epoch0 is recorded and excluded from selection.',
              f"Only this pair uses max {r['protocol']['epochs']} epochs / patience {r['protocol']['patience']}.",
              'Historical J2 is a score reference with a different training protocol, not a matched control.',
              '', '| class | D8 P/R/F1 (%) | D32 P/R/F1 (%) | D32-D8 F1 pp |',
              '|---|---:|---:|---:|']
    fields = ('precision', 'recall', 'f1-score')
    x, y = (r['models'][name]['best_validation']['per_class'] for name in ('D8', 'D32'))
    for label in x:
        left = ' / '.join(f'{100*x[label][key]:.4f}' for key in fields)
        right = ' / '.join(f'{100*y[label][key]:.4f}' for key in fields)
        lines.append(f"| {label} | {left} | {right} | {100*(y[label]['f1-score']-x[label]['f1-score']):+.6f} |")
    paired = r['D32_versus_D8']
    lines += ['', f"D32 versus D8: repaired={paired['counts']['repaired']}, regressed={paired['counts']['regressed']}, "
              f"net_correct={paired['net_correct']}.",
              f"Shared 64D train/val cache build seconds: {r['shared_cache_wall_seconds']:.3f} (charged once).",
              f"Encoder/BN state unchanged: {r['frozen_encoder_unchanged']}; raw graph checkpoint replays passed."]
    for name, m in r['models'].items():
        lines += [f"{name}: trainable parameters={m['architecture']['trainable_parameters']}; "
                  f"gate={m['architecture']['gate']}; discarded probe updates=3; evidence={m['training_evidence']}."]
    lines += [r['attribution'], 'No statistical significance claim. Review class tradeoffs before retaining D32.',
              'Dynamic-versus-equal 32D fusion and other seeds are not automatically launched. No test evaluation.']
    (root / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    with (root / 'per_class.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', 'class', 'precision', 'recall', 'f1', 'support'])
        for name, m in [('D8', r['models']['D8']), ('D32', r['models']['D32']),
                        ('historical_J2', h), ('source_N0', r['N0_source'])]:
            for label in x:
                v = m['best_validation']['per_class'][label]
                writer.writerow([name, label, *(v[k] for k in fields), v['support']])
    return '\n'.join(lines)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    print(summarize(p.parse_args().root))
