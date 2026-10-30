"""Label the independent LN candidate and include stability evidence."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from e3_gate_study import read, require, write
import summarize_flow_stats_residual as parent


def summarize(root):
    r = read(root / 'summary.json')
    variant = r['protocol']['variant']
    require(variant == 'J2_flow_stats_ln', 'Wrong normalized candidate')
    # Preserve the existing metric/paired-count analysis, with a distinct name.
    text = parent.summarize(root)
    a = read(root / 'analysis.json')
    a['rows'][-1]['model'] = variant
    probe = read(root / 'probe.json')
    a['stability_probe'] = probe
    a['last_validation_scale'] = r['candidate']['last_validation']['flow_stats_diagnostics']
    a['validation_scale_history'] = 'epoch_scale_diagnostics.jsonl'
    write(root / 'analysis.json', a)
    with (root / 'per_class.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.reader(stream))
    for row in rows:
        if row and row[0] == 'J2_flow_stats':
            row[0] = variant
    with (root / 'per_class.csv').open('w', encoding='utf-8', newline='') as stream:
        csv.writer(stream).writerows(rows)
    text = text.replace('| J2_flow_stats |', '| J2_flow_stats_ln |')
    text += '\nLayerNorm is the only model change from the unnormalized candidate; +128 affine parameters.\n'
    text += 'Stability probe: ' + probe['status'] + '; discarded updates: ' + str(len(probe['updates'])) + '.\n'
    text += 'Pre-registered probe bounds: ' + str(probe['bounds']) + '.\n'
    text += 'Per-epoch validation norm/gate distributions: epoch_scale_diagnostics.jsonl.\n'
    text += 'Last trained ratio distribution: ' + str(a['last_validation_scale']['residual_to_graph_norm']) + '\n'
    text += 'Probe bounds are stability screening rules, not validated optima or evidence of generalization.\n'
    (root / 'report.md').write_text(text, encoding='utf-8')
    return text


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    print(summarize(p.parse_args().root))
