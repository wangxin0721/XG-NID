"""Replay the three best Joint-Flow checkpoints at fixed validation strengths."""
import argparse
import json
from pathlib import Path
import tempfile

import torch
import torch_geometric

import run_flow_query_joint_seeds as study
from xgnid.packet_pooling import ordered_lengths, ordered_mean_pool, uniform_packet_weights

core, raw, require, LABELS = study.core, study.raw, study.require, study.LABELS
CONFIG = raw.ROOT / 'configs/flow_query_strength_replay_v1.json'
STRENGTHS = ('1', '0.5', '0')
REDUCED = STRENGTHS[1:]


def packet_readouts(pool, flow, nodes, index, size):
    full, beta = pool(flow, nodes, index, size)
    lengths = ordered_lengths(index, size)
    mean = ordered_mean_pool(nodes, index, size, lengths)
    uniform = uniform_packet_weights(nodes, index, size, lengths)
    # Scale the centered residual before it is added to the mean. Use the
    # historical pool output directly for the reproduction control.
    correction = (beta - uniform)[:, None] * (nodes - mean[index])
    delta = torch.segment_reduce(correction, 'sum', lengths=lengths, axis=0)
    return {'1': full, '0.5': mean + 0.5 * delta, '0': mean}


def verify_source(path, digest):
    require(path.is_file(), f'Missing replay input: {path}')
    require(raw.sha(path) == digest, f'Replay input SHA256 differs: {path}')
    return path


def load_source(repo, seed, source, config, rows):
    folder = repo / source['directory']
    checkpoint = verify_source(folder / 'Joint-Flow/best.pt', source['checkpoint_sha256'])
    flow_path = verify_source(folder / 'Joint-Flow/validation_predictions.jsonl.gz',
                              source['joint_flow_predictions_sha256'])
    mean_path = verify_source(folder / 'Joint-Mean/validation_predictions.jsonl.gz',
                              source['joint_mean_predictions_sha256'])
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    metadata = saved['config']
    protocol = metadata['protocol']
    require(saved['epoch'] == source['best_epoch'] and metadata['variant'] == 'Joint-Flow'
            and metadata['labels'] == LABELS and protocol['seed'] == seed
            and all(protocol[key] == config[key] for key in ('split_id', 'eval_batch_size', 'workers')),
            f'Joint-Flow checkpoint metadata differs for seed{seed}')
    gold = core.read_predictions(flow_path)
    original_metrics = study.classification_metrics(saved['validation'])
    require(study.validate_predictions(gold, rows, probabilities=False) == original_metrics,
            f'Original Joint-Flow predictions/metrics differ for seed{seed}')
    reference = core.read_predictions(mean_path)
    mean_metrics = study.validate_predictions(reference, rows, probabilities=False)
    base = raw.OperatorClassifier('GAT', seed=42)
    model = core.JointQueryPacketClassifier(base.flow_encoder, base.packet_encoder, base.fusion, 'flow', seed)
    model.load_state_dict(saved['model_state'], strict=True)
    return model.requires_grad_(False).eval(), gold, reference, original_metrics, mean_metrics


@torch.no_grad()
def replay(model, batches, rows, device, output, seed):
    predictions = {strength: [] for strength in STRENGTHS}
    offset = 0
    for step, batch in enumerate(batches, 1):
        batch = batch.to(device)
        size = batch.num_graphs
        ids = list(zip(batch.sample_id, batch.y.view(-1).tolist()))
        expected = rows[offset:offset + size]
        require(ids == [(r['sample_id'], r['label']) for r in expected], 'Validation IDs/labels/order differ')
        flow, nodes, index, size = model.encode(batch)
        for strength, packet in packet_readouts(model.packet_pool, flow, nodes, index, size).items():
            scores, gate = model.fusion(flow, packet)
            require(scores.shape == (size, len(LABELS)) and gate.shape == (size, 32)
                    and bool(torch.isfinite(scores).all()) and bool(torch.isfinite(gate).all()),
                    f'Invalid fusion output at seed{seed}, lambda={strength}')
            predictions[strength].extend(
                {'sample_id': sid, 'label': label, 'prediction': pred}
                for (sid, label), pred in zip(ids, scores.argmax(-1).tolist())
            )
        offset += size
        if step == 1 or step % 200 == 0 or offset == len(rows):
            raw.progress(output, 'validation_replay', seed=seed, samples=offset, total=len(rows))
    require(offset == len(rows), 'Incomplete validation replay')
    return predictions


def verify_baseline(replayed, original, seed, output):
    require(len(replayed) == len(original), f'Incomplete lambda=1 control for seed{seed}')
    differences = []
    for actual, expected in zip(replayed, original):
        require(actual['sample_id'] == expected['sample_id'] and actual['label'] == expected['label'],
                f'Lambda=1 validation IDs/labels/order differ for seed{seed}')
        if actual['prediction'] != expected['prediction']:
            differences.append({**actual, 'original_prediction': expected['prediction']})
    raw.write(output / f'seed{seed}_lambda1_check.json',
              {'samples': len(original), 'mismatches': len(differences), 'examples': differences[:10]})
    require(not differences, f'Lambda=1 differs on {len(differences)} predictions for seed{seed}')


def paired_rows(predictions, reference):
    for full, half, zero, mean in zip(*(predictions[s] for s in STRENGTHS), reference):
        yield {'sample_id': full['sample_id'], 'label': full['label'],
               'true_class': LABELS[full['label']], 'joint_mean_prediction': mean['prediction'],
               'predictions': {'1': full['prediction'], '0.5': half['prediction'], '0': zero['prediction']}}


def analyze_focus(records):
    counts = {
        strength: {
            name: {'repaired': 0, 'regressed': 0, 'reversed_joint_mean_difference': 0}
            for name in ('WebBased', 'Recon')
        }
        for strength in REDUCED
    }
    focus_samples = []
    for row in records:
        label, name = row['label'], row['true_class']
        if name not in ('WebBased', 'Recon'):
            continue
        original = row['predictions']['1'] == label
        joint_mean = row['joint_mean_prediction'] == label
        changes = {}
        for strength in REDUCED:
            current = row['predictions'][strength] == label
            if original == current:
                continue
            change = 'repaired' if current else 'regressed'
            changes[strength] = change
            counts[strength][name][change] += 1
            if joint_mean != original:
                counts[strength][name]['reversed_joint_mean_difference'] += 1
        if changes:
            focus_samples.append({**row, 'correctness_changes_vs_lambda1': changes})
    return counts, focus_samples


def write_report(output, results, evidence):
    lines = [
        '# Joint-Flow 固定强度验证集回放', '',
        '三个 seed 的 λ=1 均逐样本复现原预测。每个检查点回放同一份验证集的 13,446 张原图。',
        'λ=0.5 和 λ=0 分别保留一半和零倍的 Flow 池化修正。全程仅推理。', '',
        '| Seed | 最佳轮次 | 设置 | Macro-F1 | Δ Macro-F1（百分点） | WebBased 召回 | Recon 召回 |',
        '|---|---:|---|---:|---:|---:|---:|',
    ]
    for seed, result in results.items():
        baseline = result['metrics']['1']['macro_f1']
        comparisons = {'Joint-Mean': result['joint_mean_metrics'],
                       **{f'λ={s}': result['metrics'][s] for s in STRENGTHS}}
        for name, metric in comparisons.items():
            web = metric['per_class']['WebBased']['recall']
            recon = metric['per_class']['Recon']['recall']
            delta = 100 * (metric['macro_f1'] - baseline)
            lines.append(f"| {seed} | {result['best_epoch']} | {name} | {metric['macro_f1']:.6%} | "
                         f"{delta:+.6f} | {web:.6%} | {recon:.6%} |")
    lines += [
        '', '下表以同 seed 的 λ=1 为基准。原 WebBased 收益指 Joint-Mean 判错而 λ=1 判对的样本，'
        '原 Recon 漏检指 Joint-Mean 判对而 λ=1 判错的样本。', '',
        '| Seed | λ | WebBased 找回 / 新增漏检 | 其中丢失的原 WebBased 收益 | Recon 找回 / λ=1 漏检总数 | Recon 新增漏检 | 其中找回的原 Recon 漏检 |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for seed, result in results.items():
        recon_index = LABELS.index('Recon')
        recon_row = result['metrics']['1']['confusion_matrix'][recon_index]
        recon_missed = sum(recon_row) - recon_row[recon_index]
        for strength, by_class in result['focus_counts'].items():
            web, recon = by_class['WebBased'], by_class['Recon']
            lines.append(f"| {seed} | {strength} | {web['repaired']} / {web['regressed']} | "
                         f"{web['reversed_joint_mean_difference']} | {recon['repaired']} / {recon_missed} | {recon['regressed']} | "
                         f"{recon['reversed_joint_mean_difference']} |")
    lines += [
        '',
        f"任一较弱设置使 {evidence['WebBased_requires_stronger_unique_ids']} 个不同 WebBased 样本从对变错，"
        f"使 {evidence['Recon_recovered_unique_ids']} 个不同 Recon 样本从错变对。按验证集 sample_id 跨 seed 去重。",
        '',
        '每个 seed 的 `focus_samples.jsonl.gz` 给出 WebBased 和 Recon 正误翻转的样本 ID 及三档预测，'
        '`paired_validation.jsonl.gz` 包含全部样本。', '',
    ]
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def run(repo, output):
    require(torch.cuda.is_available() and torch.__version__.split('+')[0] == '2.3.1'
            and torch_geometric.__version__ == '2.5.3', 'Requires CUDA, Torch 2.3.1 and PyG 2.5.3')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    config = raw.read(CONFIG)
    split = repo / config['split_directory']
    verify_source(split / 'split.json', config['split_json_sha256'])
    data = raw.FrozenGraphs(split, config['split_id'])
    rows = raw.choose_rows(data, 'val')
    require(len(rows) == config['validation_samples'], 'Wrong validation set size')
    raw.write(output / 'config.json', config)

    verified = {}
    for seed, source in config['seeds'].items():
        model, gold, reference, original_metrics, mean_metrics = load_source(repo, int(seed), source, config, rows)
        model.to(device)
        state_before = raw.state_digest(model.state_dict())
        batches = core.graph_loader(data, rows, 'val', {**config, 'seed': int(seed)})
        predictions = replay(model, batches, rows, device, output, seed)
        verify_baseline(predictions['1'], gold, seed, output)
        metrics = {s: study.validate_predictions(records, rows, probabilities=False)
                   for s, records in predictions.items()}
        require(metrics['1'] == original_metrics and raw.state_digest(model.state_dict()) == state_before,
                f'Baseline metrics or model state differs for seed{seed}')
        verified[seed] = (predictions, reference, metrics, mean_metrics)

    results = {}
    evidence_ids = {'WebBased': set(), 'Recon': set()}
    for seed, (predictions, reference, metrics, mean_metrics) in verified.items():
        folder = output / f'seed{seed}'
        folder.mkdir()
        records = list(paired_rows(predictions, reference))
        counts, focus = analyze_focus(records)
        for row in focus:
            name = row['true_class']
            target = 'regressed' if name == 'WebBased' else 'repaired'
            if target in row['correctness_changes_vs_lambda1'].values():
                evidence_ids[name].add(row['sample_id'])
        raw.save_predictions(folder / 'paired_validation.jsonl.gz', records)
        raw.save_predictions(folder / 'focus_samples.jsonl.gz', focus)
        results[seed] = {'best_epoch': config['seeds'][seed]['best_epoch'], 'lambda1_mismatches': 0,
                         'metrics': metrics, 'joint_mean_metrics': mean_metrics, 'focus_counts': counts}
    evidence = {'WebBased_requires_stronger_unique_ids': len(evidence_ids['WebBased']),
                'Recon_recovered_unique_ids': len(evidence_ids['Recon'])}
    write_report(output, results, evidence)
    raw.write(output / 'summary.json', {'validation_samples': len(rows), 'optimizer_updates': 0,
              'torch': torch.__version__, 'pyg': torch_geometric.__version__, 'device': torch.cuda.get_device_name(0),
              'results': results, 'evidence': evidence})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo = args.repo.resolve()
    (repo / 'outputs').mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='flow_query_strength_replay_', dir=repo / 'outputs'))
    print(f'Replay output: {output}', flush=True)
    raw.write(output / 'status.json', {'status': 'running'})
    try:
        run(repo, output)
    except BaseException as error:
        raw.write(output / 'status.json', {'status': 'failed', 'error': str(error)})
        raise
    raw.write(output / 'status.json', {'status': 'completed'})
    print(json.dumps({'status': 'completed', 'report': str(output / 'report.md')}), flush=True)


if __name__ == '__main__':
    main()
