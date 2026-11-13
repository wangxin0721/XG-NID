"""Replay C's original and mean Packet readouts on the frozen validation split."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import platform
import re
import time

import torch
import torch_geometric

import evaluate_joint_cross_layer_query as evaluation
from xgnid.packet_pooling import ordered_mean_pool


joint, raw, require, LABELS = evaluation.joint, evaluation.raw, evaluation.require, evaluation.LABELS
CONFIG = raw.ROOT / 'configs/flow_query_selective_replay_s42_v1.json'
DEFAULT_OUTPUT = 'outputs/flow_query_selective_replay_s42_v1'
MODES = ('on', 'off', 'oracle')
CATEGORIES = ('both_correct', 'only_on_correct', 'only_off_correct', 'both_wrong')
OUTCOME_BY_CORRECTNESS = dict(zip(((True, True), (True, False), (False, True), (False, False)), CATEGORIES))
FOCUS = ('WebBased', 'Recon', 'Spoofing')


def shared_readouts(model, batch):
    means = []

    def capture_mean(module, args):
        _, nodes, index, size = args
        means.append(ordered_mean_pool(nodes, index, size))

    hook = model.packet_pool.register_forward_pre_hook(capture_mean)
    try:
        (flow, first, query), _ = model.encode(batch)
    finally:
        hook.remove()
    require(len(means) == 1, 'Expected one query pool call per batch')
    mean = means[0]
    require(all(t.shape == (batch.num_graphs, 64) and bool(torch.isfinite(t).all())
                for t in (flow, first, mean, query)), 'Invalid shared encoding or second-layer readout')
    return flow, first, mean, query


def paired_forward(model, batch):
    flow, first, mean, query = shared_readouts(model, batch)
    scores_on, gate_on = model.forward_features(flow, first, query)
    scores_off, gate_off = model.forward_features(flow, first, mean)
    require(all(t.shape == (batch.num_graphs, width) and bool(torch.isfinite(t).all())
                for t, width in ((scores_on, 8), (scores_off, 8), (gate_on, 32), (gate_off, 32))),
            'Invalid paired scores or D32 gates')
    return scores_on, scores_off, gate_on


@torch.no_grad()
def replay(model, batches, rows, archived, device, output):
    model.eval()
    records, offset = [], 0
    for step, batch in enumerate(batches, 1):
        size = batch.num_graphs
        labels = batch.y.view(-1).tolist()
        identities = list(zip(batch.sample_id, labels))
        require(identities == [(row['sample_id'], row['label']) for row in rows[offset:offset + size]],
                'Validation IDs, labels or order differ')
        batch = batch.to(device)
        scores_on, scores_off, gate_on = paired_forward(model, batch)
        on, off = scores_on.argmax(-1).tolist(), scores_off.argmax(-1).tolist()
        if step == 1:
            original = model(batch)
            scores_match = torch.allclose(scores_on, original, atol=1e-5, rtol=1e-5)
            gates_match = torch.allclose(gate_on, model.last_flow_weights, atol=1e-5, rtol=1e-5)
            predictions_match = on == original.argmax(-1).tolist() == [row['prediction'] for row in archived[:size]]
            passed = scores_match and gates_match and predictions_match
            raw.write(output / 'small_sample_check.json', {'passed': passed, 'samples': size,
                      'sample_ids': list(batch.sample_id), 'on_predictions_match': predictions_match,
                      'max_abs_log_probability_error': float((scores_on - original).abs().max()),
                      'max_abs_d32_error': float((gate_on - model.last_flow_weights).abs().max()),
                      'atol': 1e-5, 'rtol': 1e-5})
            require(passed, 'Small-sample replay differs from original C')
        for (sid, label), p, q, on_log, off_log in zip(identities, on, off, scores_on.tolist(), scores_off.tolist()):
            records.append({'sample_id': sid, 'label': label, 'on_prediction': p, 'off_prediction': q,
                            'on_log_probabilities': on_log, 'off_log_probabilities': off_log})
        offset += size
        if step == 1 or step % 200 == 0 or offset == len(rows):
            raw.progress(output, 'paired_validation', samples=offset, total=len(rows))
    require(offset == len(rows), 'Incomplete paired validation')
    return records


def predictions(records, mode):
    return [{'sample_id': row['sample_id'], 'label': row['label'], 'prediction': row[f'{mode}_prediction']}
            for row in records]


def analyze(records, rows):
    total = dict.fromkeys(CATEGORIES, 0)
    by_class = {label: dict.fromkeys(CATEGORIES, 0) for label in LABELS}
    origins = {label: Counter() for label in LABELS}
    for row in records:
        label = row['label']
        on, off = row['on_prediction'] == label, row['off_prediction'] == label
        category = OUTCOME_BY_CORRECTNESS[on, off]
        row['outcome'] = category
        row['oracle_prediction'] = row['off_prediction'] if category == 'only_off_correct' else row['on_prediction']
        total[category] += 1
        by_class[LABELS[label]][category] += 1
        if category == 'only_off_correct':
            origins[LABELS[label]][LABELS[row['on_prediction']]] += 1
    metrics = {mode: joint.study.validate_predictions(predictions(records, mode), rows, probabilities=False)
               for mode in MODES}
    base = metrics['on']['confusion_matrix']
    gained = sum(metrics['oracle']['confusion_matrix'][i][i] - base[i][i] for i in range(len(LABELS)))
    require(gained == total['only_off_correct'], 'Oracle repair count disagrees with confusion matrix')
    directions = {}
    for true_class, wrong_class in (('WebBased', 'Recon'), ('Recon', 'WebBased')):
        y, p = LABELS.index(true_class), LABELS.index(wrong_class)
        relevant = [row for row in records if row['label'] == y and row['on_prediction'] == p]
        directions[f'{true_class} -> {wrong_class}'] = {
            **{mode: metrics[mode]['confusion_matrix'][y][p] for mode in MODES},
            'original_errors_repaired': sum(row['off_prediction'] == y for row in relevant),
            'original_errors_still_wrong': sum(row['off_prediction'] != y for row in relevant),
            'new_errors_from_correct': sum(row['label'] == y and row['on_prediction'] == y
                                           and row['off_prediction'] == p for row in records)}
    deltas = {mode: 100 * (metrics[mode]['macro_f1'] - metrics['on']['macro_f1']) for mode in ('off', 'oracle')}
    return metrics, {'outcomes': total, 'by_class': by_class, 'only_off_correct_origins': origins,
                     'directions': directions, 'oracle_correct_gain': gained, 'macro_f1_delta_pp': deltas}


def diagnostic_priority(analysis, threshold):
    if analysis['oracle_correct_gain'] == 0:
        return 'close_no_repairs', '验证集没有仅关闭才判对的样本，本轮收束。'
    focus_repairs = sum(analysis['by_class'][label]['only_off_correct'] for label in FOCUS)
    if analysis['macro_f1_delta_pp']['oracle'] < threshold and focus_repairs == 0:
        return 'close_low_priority', '理想切换低于预算参考线，重点类别也没有可修复样本，本轮收束。'
    if analysis['macro_f1_delta_pp']['oracle'] < threshold:
        return 'review_focus_repairs', '理想切换低于预算参考线。先核对重点类别可修复数，再决定是否补做训练集配对诊断。'
    return 'check_training_pairs', '验证集有可修复空间。下一步核对训练集两类互补样本的数量，再决定是否训练门控。'


def write_report(output, metrics, analysis, threshold, seconds):
    priority, judgment = diagnostic_priority(analysis, threshold)
    analysis['next_step'] = priority
    analysis['budget_reference_pp'] = threshold
    raw.write(output / 'metrics.json', {'samples': metrics['on']['samples'], 'metrics': metrics,
                                       'paired_analysis': analysis, 'wall_seconds': seconds})
    raw.write(output / 'confusion_matrices.json', {'labels': LABELS, **{
        mode: metrics[mode]['confusion_matrix'] for mode in MODES}})
    with (output / 'per_class.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['mode', 'class', 'precision', 'recall', 'f1', 'support'])
        for mode in MODES:
            for label, values in metrics[mode]['per_class'].items():
                writer.writerow([mode, label, values['precision'], values['recall'], values['f1-score'], values['support']])
    lines = ['# Seed42 选择性 Flow 查询验证回放', '',
             f"同一 C 第 48 轮检查点，完整验证回放 {metrics['on']['samples']:,} 张图，耗时 {seconds:.1f} 秒。",
             '开启路径已逐样本复现 C 的归档预测及指标。', '',
             '| 读出 | Accuracy | Macro-F1 | Weighted-F1 | Δ Macro-F1（百分点） |',
             '|---|---:|---:|---:|---:|']
    for mode, name in zip(MODES, ('开启查询', '关闭查询', '按真标签理想切换')):
        score = metrics[mode]
        delta = analysis['macro_f1_delta_pp'].get(mode, 0)
        lines.append(f"| {name} | {score['accuracy']:.6%} | {score['macro_f1']:.6%} | "
                     f"{score['weighted_f1']:.6%} | {delta:+.6f} |")
    lines += ['', '| 真实类别 | 两种都对 | 仅开启对 | 仅关闭对 | 两种都错 |',
              '|---|---:|---:|---:|---:|']
    for label, counts in {'全部': analysis['outcomes'], **analysis['by_class']}.items():
        lines.append(f"| {label} | " + ' | '.join(str(counts[key]) for key in CATEGORIES) + ' |')
    lines += ['', '| 错分方向 | 开启 | 关闭 | 理想切换 | 原错分关闭后判对 | 原错分关闭后仍错 |',
              '|---|---:|---:|---:|---:|---:|']
    for direction, stats in analysis['directions'].items():
        values = [str(stats[key]) for key in (*MODES, 'original_errors_repaired', 'original_errors_still_wrong')]
        lines.append(f"| {direction} | " + ' | '.join(values) + ' |')
    lines += ['', '| 类别 | 开启召回 | 关闭召回 | 理想切换召回 |', '|---|---:|---:|---:|']
    for label in FOCUS:
        values = [f"{metrics[mode]['per_class'][label]['recall']:.6%}" for mode in MODES]
        lines.append(f"| {label} | " + ' | '.join(values) + ' |')
    lines += ['', f"两个固定读出之间按真标签切换，最多可多判对 {analysis['oracle_correct_gain']} 张图。"
              '对应 Macro-F1 是用真标签构造的诊断参考值。', '',
              f'预算参考线为 +{threshold:.2f} 个百分点。{judgment}', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def run(repo, output, graph_root, code_commit):
    started = time.perf_counter()
    settings = raw.read(CONFIG)
    name = settings['checkpoint']
    config = raw.read(evaluation.CONFIG)
    config['checkpoints'] = {name: config['checkpoints'][name]}
    config['metrics_sha256'] = {name: config['metrics_sha256'][name]}
    source = repo / config['source_directory']
    split = repo / config['split_directory']
    archive, inputs = evaluation.archive_inputs(source, split, config)
    selected = archive[name]['selected']['best']
    require(selected['prediction_ids_sha256'] == settings['prediction_ids_sha256'], 'Wrong archived C prediction digest')
    require(torch.cuda.is_available() and torch.__version__.split('+')[0] == '2.3.1'
            and torch_geometric.__version__ == '2.5.3', 'Expected CUDA, Torch 2.3.1 and PyG 2.5.3')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    data = raw.FrozenGraphs(split, config['split_id'], data_root=graph_root)
    inputs[data.directory / data.metadata['sample_manifest']] = data.metadata['sample_manifest_sha256']
    rows = evaluation.evaluation_rows(data, 'val', config)
    archived = joint.legacy.predictions(source / name / 'validation_predictions.jsonl.gz')
    require(joint.study.validate_predictions(archived, rows, probabilities=False) == selected['validation']
            and joint.prediction_digest(archived) == selected['prediction_ids_sha256'],
            'Archived C predictions differ from selection')

    protocol = raw.read(joint.CONFIG)
    state = evaluation.load_checkpoint_state(source, config, archive, protocol, name)
    base = raw.OperatorClassifier('GAT', seed=config['seed'])
    model = joint.JointCrossLayerQueryClassifier(base.flow_encoder, base.packet_encoder, base.fusion,
                                                 joint.ARMS[name], config['seed'])
    model.load_state_dict(state, strict=True)
    require(joint.legacy.frozen_hash(model) == protocol['source_frozen_state_sha256'] ==
            archive[name]['frozen_state_sha256'], 'Archived C frozen state differs')
    model.requires_grad_(False).to(device).eval()
    state_digest = raw.state_digest(model.state_dict())
    raw.write(output / 'config.json', {**config, 'replay': settings, 'partitions_evaluated': ['val']})
    raw.write(output / 'provenance.json', {'code_commit': code_commit, 'archive_commit': config['archive_commit'],
              'input_sha256': {str(path): digest for path, digest in inputs.items()}, 'graph_root': str(data.root),
              'config_sha256': raw.sha(CONFIG), 'fixed_evaluation_config_sha256': raw.sha(evaluation.CONFIG),
              'runner_sha256': raw.sha(Path(__file__)), 'model_state_sha256': state_digest,
              'python': platform.python_version(), 'torch': torch.__version__, 'pyg': torch_geometric.__version__,
              'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(device), 'tf32': False})

    loader = joint.legacy.loader(data, rows, 'val', config)
    records = replay(model, loader, rows, archived, device, output)
    on = predictions(records, 'on')
    baseline = joint.study.validate_predictions(on, rows, probabilities=False)
    receipt = evaluation.replay_receipt(baseline, on, selected, archived)
    raw.write(output / 'validation_replay.json', receipt)
    require(receipt['passed'], 'Full query-on replay differs from archived C. See validation_replay.json')
    require(raw.state_digest(model.state_dict()) == state_digest, 'C model state changed during replay')
    evaluation.verify_inputs(inputs)
    raw.write(output / 'input_check.json', {'verified_before_and_after': True, 'model_state_unchanged': True,
                                           'checkpoint_sha256': config['checkpoints'][name]['sha256']})
    metrics, analysis = analyze(records, rows)
    raw.save_predictions(output / 'paired_predictions.jsonl.gz', records)
    raw.save_predictions(output / 'changed_samples.jsonl.gz',
                         (row for row in records if row['on_prediction'] != row['off_prediction']))
    write_report(output, metrics, analysis, settings['budget_reference_pp'], time.perf_counter() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--code-commit', required=True)
    args = parser.parse_args()
    require(re.fullmatch(r'[0-9a-f]{40}', args.code_commit), 'Expected a full code commit SHA')
    repo = args.repo.resolve()
    output = (repo / args.output).resolve()
    require(output.parent == (repo / 'outputs').resolve(), 'Use a new directory directly under outputs')
    output.mkdir(parents=True)
    print(f'Replay output: {output}', flush=True)
    raw.write(output / 'status.json', {'status': 'running'})
    try:
        run(repo, output, args.data_root, args.code_commit)
    except BaseException as error:
        raw.write(output / 'status.json', {'status': 'failed', 'error': str(error)})
        raise
    raw.write(output / 'status.json', {'status': 'completed'})
    print(json.dumps({'status': 'completed', 'report': str(output / 'report.md')}), flush=True)


if __name__ == '__main__':
    main()
