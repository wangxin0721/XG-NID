"""Archive-only diagnosis; raw replay enriches the same records on the server."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import subprocess


LABELS = ('Benign', 'WebBased', 'Spoofing', 'Recon', 'Mirai', 'Dos', 'DDos', 'BruteForce')
ARCHIVE = '944ba61faca474d02c409c2038e02f33e5da3d50'
PREFIX = 'outputs/ordered_transition_readout_200/seed42/'


def changes(records, before, after):
    result = []
    for row in records:
        a, b, y = row['predictions'][before], row['predictions'][after], row['label']
        if a == b:
            continue
        kind = 'repaired' if a != y and b == y else 'regressed' if a == y and b != y else 'wrong_changed'
        result.append({**row, 'kind': kind, 'before': a, 'after': b})
    return result


def summarize(records, before='single', after='ordered'):
    changed = changes(records, before, after)
    groups = {}
    for kind in ('repaired', 'regressed', 'wrong_changed'):
        selected = [r for r in changed if r['kind'] == kind]
        groups[kind] = {
            'count': len(selected),
            'true_class': dict(Counter(LABELS[r['label']] for r in selected)),
            'directions': dict(Counter(f"{LABELS[r['label']]}: {LABELS[r['before']]} -> {LABELS[r['after']]}"
                                       for r in selected)),
            'valid_one_step_counts': dict(Counter(str(r['valid_pairs']) for r in selected)),
            'valid_two_step_counts': dict(Counter(str(r.get('ordered_contexts', 'unknown')) for r in selected)),
            'by_class_and_context': dict(Counter(
                f"{LABELS[r['label']]}|" + ('unknown' if 'ordered_contexts' not in r else
                                          'present' if r['ordered_contexts'] else 'absent') for r in selected))}
    return {'samples': len(records), 'changed': len(changed), 'groups': groups}, changed


def write_report(output, records, coverage, evidence, before='single', after='ordered'):
    output.mkdir(parents=True, exist_ok=True)
    summary, changed = summarize(records, before, after)
    summary.update(evidence=evidence, context_coverage=coverage)
    (output / 'diagnosis.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with gzip.open(output / 'changed_samples.jsonl.gz', 'wt', encoding='utf-8') as stream:
        for row in changed:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    lines = ['# Ordered 相对 Single 的定向误差诊断', '',
             f"验证样本 {len(records)}，预测变化 {len(changed)}。", '',
             '| 真实类别 | 修复 | 退步 | 错误间变化 |', '| --- | ---: | ---: | ---: |']
    for label in LABELS:
        lines.append('| ' + label + ' | ' + ' | '.join(str(summary['groups'][k]['true_class'].get(label, 0))
                     for k in ('repaired', 'regressed', 'wrong_changed')) + ' |')
    lines += ['', '| 类别 | 有效连续两步覆盖 |', '| --- | ---: |']
    for label, item in coverage.items():
        lines.append(f"| {label} | {item['with_context']}/{item['samples']} ({item['with_context']/item['samples']:.4%}) |")
    for kind, group in summary['groups'].items():
        lines += ['', f'## {kind}', '', '| 真实类别: 原预测 → 新预测 | 数量 |', '| --- | ---: |']
        lines += [f'| {key} | {value} |' for key, value in group['directions'].items()]
        lines += ['', '有效一步转移数量分组：' + json.dumps(group['valid_one_step_counts'], ensure_ascii=False),
                  '', '有效连续两步片段数量分组：' + json.dumps(group['valid_two_step_counts'], ensure_ascii=False)]
    lines += ['', '观察与推测：归档中有 7 例修复、5 例退步只有一条有效转移，因此这些样本没有连续两步片段。',
              'WebBased 与 Recon 的连续片段覆盖明显不均衡；BruteForce 的一例变化为错误类别之间的切换。',
              '转移数大于等于 2 不能推定存在连续片段。未知项需原图连续性标记回放，不能填为 0。',
              '冻结门控限制新增表示利用率属于待检验假设；本诊断不作因果归因。',
              '', '证据：' + json.dumps(evidence, ensure_ascii=False)]
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    def blob(name):
        return subprocess.check_output(['git', 'show', ARCHIVE + ':' + PREFIX + name], cwd=args.repo)
    payload = blob('validation/validation_predictions.jsonl.gz')
    records = [json.loads(line) for line in gzip.decompress(payload).splitlines()]
    coverage = json.loads(blob('coverage.json'))['val']['context_coverage_by_class']
    # A single valid link is sufficient to establish absence; other missing counts remain unknown.
    records = [{**r, **({'ordered_contexts': 0} if r['valid_pairs'] < 2 else {})} for r in records]
    summary = write_report(args.output, records, coverage, {
        'archive_commit': ARCHIVE, 'prediction_file_sha256': hashlib.sha256(payload).hexdigest(),
        'raw_graph_replay_completed': False,
        'pending': ['per_sample_continuity_for_ge2', 'attention', 'residual', 'gate_weights']})
    assert summary['samples'] == 13446 and summary['groups']['repaired']['count'] == 22
    assert summary['groups']['regressed']['count'] == 22 and summary['groups']['wrong_changed']['count'] == 6
    print(json.dumps({'archive_diagnosis': 'passed', 'samples': summary['samples'], 'changed': summary['changed']}))


if __name__ == '__main__':
    main()
