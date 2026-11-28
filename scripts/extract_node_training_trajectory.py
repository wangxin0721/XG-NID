"""Extract validation trajectories from the exact completed S0-S3 archive; no training."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

COMMIT = '1469f7190e7077db7afb639d2efa1347d9e261f1'
PATH = 'outputs/flow_packet_node_modulation_v1/history.json'
EPOCHS = (0, 1, 2, 5, 10, 20, 50, 100, 200)


def extract(repo, destination):
    payload = subprocess.check_output(['git', 'show', COMMIT + ':' + PATH], cwd=repo)
    history = json.loads(payload)
    assert [r['epoch'] for r in history] == list(range(201))
    selected = [{'epoch': r['epoch'], 'train_online_nll': r.get('train_nll'),
                 'validation_macro_f1': {n: m['macro_f1'] for n, m in r['validation'].items()}}
                for r in history if r['epoch'] in EPOCHS]
    result = {'source_result_commit': COMMIT, 'source_path': PATH,
              'source_bytes_sha256': hashlib.sha256(payload).hexdigest(), 'selected': selected,
              'limitations': ['epoch0 training NLL was not recorded',
                  'historical validation NLL and fixed training-subset NLL were not recorded',
                  'online epoch-mean training NLL uses changing weights and is not an eval-mode fixed-subset loss'],
              'interpretation': 'Shared degradation is a hypothesis; these data alone do not identify learning rate, overfitting or module ineffectiveness.'}
    destination.mkdir(parents=True, exist_ok=True)
    (destination/'trajectory.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    lines = ['# 已有 S1–S3 训练轨迹（不重训）', '', '来源：`'+COMMIT+'`。', '',
             '| epoch | S1 训练NLL | S1 验证F1% | S2 训练NLL | S2 验证F1% | S3 训练NLL | S3 验证F1% |',
             '|---:|---:|---:|---:|---:|---:|---:|']
    for r in selected:
        cells = [str(r['epoch'])]
        for n in ('S1', 'S2', 'S3'):
            cells += [f"{r['train_online_nll'][n]:.9f}" if r['train_online_nll'] else '未记录',
                      f"{100*r['validation_macro_f1'][n]:.6f}"]
        lines.append('| '+' | '.join(cells)+' |')
    lines += ['', 'epoch0训练NLL、历史验证NLL和固定训练子集NLL均未记录，不能补造。训练NLL是更新中在线均值，与固定权重eval损失不能直接等同。',
              '三组同步退化与共同训练设置主导的假设相容，尚不能直接判定学习率太大、过拟合或调制无效。新诊断补齐相同样本的训练/验证损失与参数变化。']
    (destination/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path.cwd())
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(extract(a.repo, a.output)))
