"""Train paired Packet-mean and Flow transition readouts on frozen C."""
import argparse
import csv
import json
from pathlib import Path
import platform
import re
import time
from typing import NamedTuple

import numpy as np
import torch
from torch.nn import functional as F
import torch_geometric

import run_fusion_decision_diagnosis as fusion_diagnosis
import run_packet_transition_diagnosis as diagnosis
from audit_packet_transitions import transition_fields
from xgnid.packet_transition_readout import PacketTransitionReadout


evaluation, joint, raw, require, LABELS = (diagnosis.evaluation, diagnosis.joint, diagnosis.raw,
                                          diagnosis.require, diagnosis.LABELS)
CONFIG = raw.ROOT / 'configs/packet_transition_readout_s42_v1.json'
OUTPUT = 'outputs/packet_transition_readout_s42_v1'
SOURCES = {'B': 'packet_mean', 'C1': 'flow'}


class Features(NamedTuple):
    flow: torch.Tensor
    packet_mean: torch.Tensor
    packet: torch.Tensor
    transitions: torch.Tensor
    lengths: torch.Tensor


def chronological_edges(batch):
    fields = transition_fields(batch)
    keep = np.flatnonzero(fields.valid)
    keep = keep[np.argsort(fields.src[keep], kind='stable')]
    src, dst = fields.src[keep], fields.dst[keep]
    groups = (2 * fields.direction[src] + fields.direction[dst]).astype(np.int64)
    times = np.log1p(fields.dt[keep].astype(np.float64))
    changes = np.log1p(fields.payload[dst]) - np.log1p(fields.payload[src])
    metadata = np.column_stack((np.eye(4)[groups], times, changes)).astype(np.float32)
    lengths = np.bincount(fields.owner[src], minlength=batch.num_graphs)
    counts = {'expected_pairs': int(np.maximum(fields.sizes - 1, 0).sum()),
              'forward_links': int(fields.forward.sum()), 'reverse_links': int(fields.reverse.sum()),
              'other_links': int((~(fields.forward | fields.reverse)).sum()),
              'cross_flow_links': int((~fields.same_flow).sum()),
              'duplicate_forward_links': int((fields.forward & ~fields.unique).sum()),
              'invalid_time_forward_links': int((fields.forward & ~fields.time_ok).sum()),
              'invalid_direction_forward_links': int((fields.forward & ~fields.edge_direction_ok).sum()),
              'invalid_payload_forward_links': int((fields.forward & ~fields.edge_payload_ok).sum()),
              'invalid_forward_links': int((fields.forward & ~fields.valid).sum()),
              'valid_pairs': len(keep)}
    return src, dst, metadata, lengths, counts


@torch.no_grad()
def encode(model, batch, device):
    src, dst, metadata, lengths, counts = chronological_edges(batch)
    captured = []
    hook = model.packet_pool.register_forward_pre_hook(lambda module, args: captured.append(args[1]))
    try:
        flow, first, mean, query = diagnosis.replay.shared_readouts(model, batch.to(device))
    finally:
        hook.remove()
    require(len(captured) == 1 and captured[0].shape == (batch['packet'].num_nodes, 64),
            'Unexpected second-layer Packet tensor')
    nodes = captured[0]
    src, dst = (torch.from_numpy(index).to(device) for index in (src, dst))
    edges = torch.cat((nodes[dst] - nodes[src], torch.from_numpy(metadata).to(device)), dim=-1)
    packet = model.cross_layer(torch.cat((first, query), dim=-1))
    require(bool(torch.isfinite(edges).all()) and bool(torch.isfinite(packet).all()), 'Nonfinite frozen features')
    return Features(flow, mean, packet, edges, torch.from_numpy(lengths).to(device)), counts


class FrozenCache:
    """Keep one packed CPU copy per split and gather whole graphs in sampler order."""
    def __init__(self, chunks, labels):
        self.features = Features(*(torch.cat(parts) for parts in zip(*chunks)))
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.offsets = np.concatenate(([0], np.cumsum(self.features.lengths.numpy())))
        require(len(self.labels) == len(self.offsets) - 1 and self.offsets[-1] == len(self.features.transitions),
                'Frozen cache graph or transition count differs')

    def batch(self, indices, device):
        graphs = np.asarray(indices, dtype=np.int64)
        starts = self.offsets[graphs]
        lengths = self.offsets[graphs + 1] - starts
        ends = np.cumsum(lengths)
        transition_indices = np.arange(ends[-1]) + np.repeat(starts - (ends - lengths), lengths)
        index = torch.from_numpy(graphs)
        flow, mean, packet, edges, _ = self.features
        features = Features(flow[index], mean[index], packet[index],
                            edges[torch.from_numpy(transition_indices)], torch.from_numpy(lengths))
        return Features(*(tensor.to(device) for tensor in features)), self.labels[index].to(device)

    def bytes(self):
        return sum(t.numel() * t.element_size() for t in (*self.features, self.labels)) + self.offsets.nbytes


def transition_forward(model, head, arm, features):
    query = getattr(features, SOURCES[arm])
    residual, attention = head(query, features.transitions, features.lengths)
    scores, _ = model.fusion(features.flow, features.packet + residual)
    return scores, residual, attention


def baseline(model, features):
    return model.fusion(features.flow, features.packet)[0]


def make_heads(config, device):
    heads = {arm: PacketTransitionReadout(config['seed']).to(device) for arm in SOURCES}
    require(len({raw.state_digest(head.state_dict()) for head in heads.values()}) == 1,
            'Transition readouts must start with identical weights')
    return heads


def optimizer(head, config):
    return torch.optim.Adam(head.parameters(), lr=config['learning_rate'])


def probe(model, data, config, device):
    rows = raw.choose_rows(data, 'train', per_class=2)
    probe_config = {**config, 'eval_batch_size': len(rows), 'workers': 0}
    _, _, batch, _ = next(diagnosis.aligned_batches(data, rows, 'train', probe_config))
    features, counts = encode(model, batch, device)
    labels = batch.y.view(-1)
    require(bool((features.lengths >= 2).any()), 'Gradient preflight needs multi-transition graphs')
    checks = {}
    for arm, head in make_heads(config, device).items():
        opt = optimizer(head, config)
        steps = []
        for step in range(2):
            opt.zero_grad(set_to_none=True)
            scores, residual, _ = transition_forward(model, head, arm, features)
            if step == 0:
                require(torch.equal(scores, baseline(model, features)) and not bool(residual.any()),
                        f'{arm} preflight zero residual changed original scores')
            loss = F.nll_loss(scores, labels)
            loss.backward()
            gradients = {name: float(param.grad.norm()) for name, param in head.named_parameters()}
            require(all(np.isfinite(value) for value in gradients.values()) and bool(torch.isfinite(loss)),
                    f'{arm} preflight has nonfinite loss or gradient')
            require(gradients['output.weight'] > 0, f'{arm} W gradient is zero')
            if step == 0:
                require(all(value == 0 for name, value in gradients.items() if name != 'output.weight'),
                        f'{arm} E or Q received a gradient before W updated')
            else:
                require(gradients['edge.weight'] > 0 and gradients['query.weight'] > 0,
                        f'{arm} E or Q has no gradient after W updated')
            opt.step()
            steps.append({'step': step + 1, 'loss': float(loss.detach()), 'gradient_norms': gradients})
        checks[arm] = steps
    return {'passed': True, 'sample_ids': [row['sample_id'] for row in rows], 'edge_counts': counts,
            'graphs_with_at_least_two_edges': int((features.lengths >= 2).sum()), 'gradient_steps': checks}


@torch.no_grad()
def check_attention(head, device):
    generator = torch.Generator().manual_seed(42)
    query = torch.randn(4, 64, generator=generator).to(device)
    edges = torch.randn(3, 70, generator=generator).to(device)
    lengths = torch.tensor([0, 2, 1, 0], device=device)
    # Use nonzero W so empty segments and cross-graph isolation are observable.
    head.output.weight.fill_(0.125)
    residual, attention = head(query, edges, lengths)
    changed = edges.clone()
    changed[2] *= -5
    other, other_attention = head(query, changed, lengths)
    isolated, isolated_attention = head(query[1:2], edges[:2], lengths[1:2])
    empty, empty_attention = head(query, edges[:0], torch.zeros_like(lengths))
    require(not bool(residual[[0, 3]].any()) and not bool(empty.any()) and empty_attention.numel() == 0
            and float(attention[2]) == 1 and torch.allclose(attention[:2].sum(), attention.new_tensor(1.0))
            and torch.equal(residual[1], other[1]) and torch.equal(attention[:2], other_attention[:2])
            and torch.allclose(residual[1:2], isolated, atol=1e-6, rtol=1e-5)
            and torch.allclose(attention[:2], isolated_attention, atol=1e-6, rtol=1e-5),
            'Empty, single-edge or isolated attention preflight failed')
    return {'empty_graph_zero_residual': True, 'empty_batch_zero_residual': True,
            'single_edge_attention': float(attention[2]), 'per_graph_attention_isolated': True}


def prediction_records(rows, predictions):
    require(len(rows) == len(predictions), 'Incomplete validation predictions')
    return [{'sample_id': row['sample_id'], 'label': row['label'], 'prediction': prediction}
            for row, prediction in zip(rows, predictions)]


@torch.no_grad()
def cache_split(model, data, rows, part, config, device, output, heads=None):
    chunks, predictions = [], []
    field_counts = {}
    for step, indices, batch, _ in diagnosis.aligned_batches(data, rows, part, config):
        features, counts = encode(model, batch, device)
        if heads is not None:
            scores = baseline(model, features)
            require(bool(torch.isfinite(scores).all()), 'Nonfinite original C scores')
            predictions.extend(scores.argmax(dim=-1).tolist())
            for arm, head in heads.items():
                adjusted, residual, _ = transition_forward(model, head, arm, features)
                require(torch.equal(scores, adjusted) and not bool(residual.any()),
                        f'{arm} epoch-zero logits differ from original C')
        chunks.append(Features(*(value.cpu() for value in features)))
        for key, value in counts.items():
            field_counts[key] = field_counts.get(key, 0) + value
        if step == 1 or step % 200 == 0 or indices.stop == len(rows):
            raw.progress(output, f'cache_{part}', samples=indices.stop, total=len(rows))
    cache = FrozenCache(chunks, [row['label'] for row in rows])
    lengths = cache.features.lengths
    multi = int((lengths >= 2).sum())
    coverage = {**field_counts, 'samples': len(rows), 'graphs_without_valid_edges': int((lengths == 0).sum()),
                'graphs_with_at_least_two_edges': multi, 'coverage_ge2': multi / len(rows),
                'cache_bytes': cache.bytes()}
    return cache, coverage, prediction_records(rows, predictions) if heads is not None else None


@torch.no_grad()
def measure(model, heads, val, rows, config, device):
    predictions = {arm: [] for arm in heads}
    size = config['eval_batch_size']
    for start in range(0, len(rows), size):
        features, _ = val.batch(range(start, min(start + size, len(rows))), device)
        for arm, head in heads.items():
            scores, _, _ = transition_forward(model, head, arm, features)
            require(bool(torch.isfinite(scores).all()), f'{arm} nonfinite validation scores')
            predictions[arm].extend(scores.argmax(dim=-1).tolist())
    records = {arm: prediction_records(rows, values) for arm, values in predictions.items()}
    return records, {arm: joint.study.validate_predictions(items, rows, probabilities=False)
                     for arm, items in records.items()}


def fit(model, heads, train, val, rows, reference, config, device, output):
    sampler = raw.TrainBatches(len(train.labels), config['batch_size'], config['seed'])
    optimizers = {arm: optimizer(head, config) for arm, head in heads.items()}
    best, last = {}, {}
    with (output / 'epochs.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
        for epoch in range(config['epochs'] + 1):
            losses = {arm: torch.zeros((), device=device) for arm in heads}
            updates, samples = 0, 0
            if epoch:
                for indices in sampler:
                    features, labels = train.batch(indices, device)
                    for arm, head in heads.items():
                        opt = optimizers[arm]
                        opt.zero_grad(set_to_none=True)
                        scores, _, _ = transition_forward(model, head, arm, features)
                        loss = F.nll_loss(scores, labels)  # Eight-class CE on C's log-softmax output.
                        loss.backward()
                        opt.step()
                        losses[arm] += loss.detach() * len(indices)
                    updates += 1
                    samples += len(indices)
                require(updates == len(sampler) and samples == len(train.labels)
                        and all(bool(torch.isfinite(loss)) for loss in losses.values()), 'Incomplete or invalid epoch')
            records, metrics = measure(model, heads, val, rows, config, device)
            if epoch == 0:
                require(all(items == reference for items in records.values()), 'Cached epoch zero differs from original C')
            for arm, head in heads.items():
                improved = epoch == 0 or metrics[arm]['macro_f1'] > best[arm]['validation']['macro_f1']
                if improved or epoch == config['epochs']:
                    item = {'epoch': epoch, 'validation': metrics[arm],
                            'prediction_ids_sha256': joint.prediction_digest(records[arm]),
                            'state': {key: tensor.detach().cpu().clone() for key, tensor in head.state_dict().items()}}
                    if improved:
                        best[arm] = item
                    if epoch == config['epochs']:
                        last[arm] = item
            stream.write(json.dumps({'epoch': epoch, 'updates_per_arm': updates, 'samples_per_arm': samples,
                                     'order_sha256': sampler.order_sha256, 'validation': metrics,
                                     'train_cross_entropy': {arm: float(loss / samples) for arm, loss in losses.items()}
                                     if samples else None}) + '\n')
            stream.flush()
            raw.progress(output, 'paired_training', epoch=epoch, total_epochs=config['epochs'],
                         macro_f1={arm: value['macro_f1'] for arm, value in metrics.items()})
    return best, last, len(sampler) * config['epochs']


@torch.no_grad()
def replay_selected(model, checkpoint, data, rows, config, device, output):
    heads = make_heads(config, device)
    for arm, head in heads.items():
        head.load_state_dict(checkpoint['best'][arm]['state'], strict=True)
        head.eval()
    predictions = {arm: [] for arm in ('A', *heads)}
    for step, indices, batch, _ in diagnosis.aligned_batches(data, rows, 'val', config):
        features, _ = encode(model, batch, device)
        predictions['A'].extend(baseline(model, features).argmax(dim=-1).tolist())
        for arm, head in heads.items():
            scores, _, _ = transition_forward(model, head, arm, features)
            require(bool(torch.isfinite(scores).all()), f'{arm} invalid raw replay scores')
            predictions[arm].extend(scores.argmax(dim=-1).tolist())
        if step == 1 or step % 200 == 0 or indices.stop == len(rows):
            raw.progress(output, 'selected_raw_replay', samples=indices.stop, total=len(rows))
    return {arm: prediction_records(rows, values) for arm, values in predictions.items()}


def summarize(output, rows, predictions, lengths, selections, config, updates, coverage, seconds):
    metrics = {arm: joint.study.validate_predictions(items, rows, probabilities=False)
               for arm, items in predictions.items()}
    comparisons = {}
    for after, before in (('B', 'A'), ('C1', 'A'), ('C1', 'B')):
        name = f'{after}-{before}'
        reference, candidate = predictions[before], predictions[after]
        paired, changed = raw.compare(reference, candidate)
        subsets = {f'{true} -> {wrong}': [i for i, item in enumerate(reference)
                   if item['label'] == LABELS.index(true) and item['prediction'] == LABELS.index(wrong)]
                   for true, wrong in diagnosis.DIRECTIONS}
        directions = fusion_diagnosis.direction_stats(reference, candidate, subsets)
        for key, subset in subsets.items():
            directions[key]['original_errors_with_at_least_two_edges'] = sum(lengths[i] >= 2 for i in subset)
            directions[key]['net_change'] = directions[key]['total_errors'] - directions[key]['original_errors']
        comparisons[name] = {**paired, 'changed_predictions': len(changed), 'directions': directions,
                             'delta_macro_f1_pp': 100 * (metrics[after]['macro_f1'] - metrics[before]['macro_f1'])}
        raw.save_predictions(output / f'{name}_changed_samples.jsonl.gz', changed)
    raw.save_predictions(output / 'validation_predictions.jsonl.gz', (
        {**row, 'valid_pairs': lengths[i], 'predictions': {arm: items[i]['prediction']
                                                         for arm, items in predictions.items()}}
        for i, row in enumerate(predictions['A'])))
    with (output / 'per_class.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['arm', 'class', 'precision', 'recall', 'f1', 'support'])
        for arm, result in metrics.items():
            for label, item in result['per_class'].items():
                writer.writerow([arm, label, item['precision'], item['recall'], item['f1-score'], item['support']])
    against_a = comparisons['C1-A']
    decision = {'meets_reference': against_a['delta_macro_f1_pp'] >= config['budget_reference_pp'],
                'beats_packet_mean_query': comparisons['C1-B']['delta_macro_f1_pp'] > 0,
                'recon_to_webbased_change': against_a['directions']['Recon -> WebBased']['net_change'],
                'webbased_to_recon_change': against_a['directions']['WebBased -> Recon']['net_change']}
    epochs = {'A': 48, **{arm: item['epoch'] for arm, item in selections['best'].items()}}
    last = {arm: item['validation'] for arm, item in selections['last'].items()}
    raw.write(output / 'metrics.json', {'metrics': metrics, 'comparisons': comparisons, 'selected_epochs': epochs,
              'last_epoch_metrics': last, 'updates_per_arm': updates, 'decision': decision,
              'budget_reference_pp': config['budget_reference_pp'], 'wall_seconds': seconds})
    lines = ['# Seed42 报文转移读出', '',
             'A 为归档 C 第 48 轮。B 使用 Packet 第二层均值查询，C1 使用 Flow 查询。',
             f"B 和 C1 各训练 {config['epochs']} 轮，按全验证集 Macro-F1 选模，包含第 0 轮，同分取较早轮次。", '',
             '| 组别 | 选中轮次 | Accuracy | Macro-F1 | Weighted-F1 | WebBased 召回 | Recon 召回 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm, result in metrics.items():
        values = [result[key] for key in ('accuracy', 'macro_f1', 'weighted_f1')]
        values += [result['per_class'][label]['recall'] for label in ('WebBased', 'Recon')]
        lines.append(f'| {arm} | {epochs[arm]} | ' + ' | '.join(f'{value:.6%}' for value in values) + ' |')
    lines += ['', '| 比较 | Macro-F1 差值（百分点） | 修复 | 退步 | 净正确 |',
              '| --- | ---: | ---: | ---: | ---: |']
    for name, result in comparisons.items():
        counts = result['counts']
        lines.append(f"| {name} | {result['delta_macro_f1_pp']:+.6f} | {counts['repaired']} | "
                     f"{counts['regressed']} | {result['net_correct']:+d} |")
    lines += ['', '| 比较 | 错分方向 | 原错分 | 当前错分 | 判对 | 从原判对样本新增 | 从其他错分新增 |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for name, result in comparisons.items():
        for direction, item in result['directions'].items():
            lines.append(f"| {name} | {direction} | {item['original_errors']} | {item['total_errors']} | "
                         f"{item['repaired']} | {item['new_errors_from_original_correct']} | "
                         f"{item['new_errors_from_other_wrong_class']} |")
    lines += ['', '| 组别 | 类别 | Precision | Recall | F1 | 样本数 |',
              '| --- | --- | ---: | ---: | ---: | ---: |']
    for arm, result in metrics.items():
        for label, item in result['per_class'].items():
            values = [f'{item[key]:.6%}' for key in ('precision', 'recall', 'f1-score')]
            lines.append(f"| {arm} | {label} | " + ' | '.join(values) + f" | {item['support']} |")
    target = against_a['directions']['Recon -> WebBased']
    meets = '达到' if decision['meets_reference'] else '低于'
    beats = '高于' if decision['beats_packet_mean_query'] else '未超过'
    lines += ['', f"C1 相对 A 的增量{meets} {config['budget_reference_pp']:+.2f} 个百分点的参考线，"
              f"C1 的 Macro-F1 {beats} B。",
              f"原 C 的 {target['original_errors']} 条 Recon→WebBased 错分中，"
              f"{target['original_errors_with_at_least_two_edges']} 条有至少两个有效报文对。",
              f"训练集与验证集至少两个有效报文对的覆盖率分别为 "
              f"{coverage['train']['coverage_ge2']:.2%} 和 {coverage['val']['coverage_ge2']:.2%}。",
              f"每组完成 {updates:,} 次更新。第 0 轮和所选检查点均通过原图回放。结果限于 seed42 验证集。"]
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run(repo, output, graph_root, code_commit):
    started = time.perf_counter()
    config = {**raw.read(evaluation.CONFIG), **raw.read(CONFIG)}
    config['checkpoints'] = {'C': config['checkpoints']['C']}
    config['metrics_sha256'] = {'C': config['metrics_sha256']['C']}
    config['partition_sizes'] = {'val': config['partition_sizes']['val'], 'train': config['training_samples']}
    source, split = (repo / config[key] for key in ('source_directory', 'split_directory'))
    archive, inputs = evaluation.archive_inputs(source, split, config)
    require(torch.cuda.is_available() and torch.__version__.split('+')[0] == '2.3.1'
            and torch_geometric.__version__ == '2.5.3', 'Expected CUDA, Torch 2.3.1 and PyG 2.5.3')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    data = raw.FrozenGraphs(split, config['split_id'], data_root=graph_root)
    manifest = data.directory / data.metadata['sample_manifest']
    inputs[manifest] = data.metadata['sample_manifest_sha256']
    rows = {'val': evaluation.evaluation_rows(data, 'val', config), 'train': raw.choose_rows(data, 'train')}
    require(len(rows['train']) == data.metadata['partition_sizes']['train'] == config['training_samples']
            and len({row['sample_id'] for row in rows['train']}) == config['training_samples']
            and not {row['sample_id'] for row in rows['val']}.intersection(row['sample_id'] for row in rows['train']),
            'Train or validation split membership differs')
    protocol = raw.read(joint.CONFIG)
    state = evaluation.load_checkpoint_state(source, config, archive, protocol, 'C')
    base = raw.OperatorClassifier('GAT', seed=config['seed'])
    model = joint.JointCrossLayerQueryClassifier(base.flow_encoder, base.packet_encoder, base.fusion,
                                                 'flow', config['seed'])
    model.load_state_dict(state, strict=True)
    require(joint.legacy.frozen_hash(model) == archive['C']['frozen_state_sha256']
            == protocol['source_frozen_state_sha256'], 'Archived C frozen state differs')
    model.requires_grad_(False).to(device).eval()
    before = raw.state_digest(model.state_dict())
    heads = make_heads(config, device)
    initial = raw.state_digest(heads['B'].state_dict())
    parameter_count = sum(p.numel() for p in heads['B'].parameters())
    raw.write(output / 'config.json', {**config, 'query_sources': SOURCES, 'parameters_per_arm': parameter_count,
              'selection': 'best_validation_macro_f1_including_zero_earliest_tie',
              'feature_cache': 'packed_float32_cpu', 'time_unit': 'ms', 'payload_unit': 'bytes'})
    provenance = {'code_commit': code_commit, 'archive_commit': config['archive_commit'],
                  'input_sha256': {str(path): sha for path, sha in inputs.items()}, 'graph_root': str(data.root),
                  'python': platform.python_version(), 'torch': str(torch.__version__),
                  'pyg': torch_geometric.__version__, 'cuda': torch.version.cuda,
                  'device': torch.cuda.get_device_name(device), 'frozen_c_sha256': before,
                  'initial_transition_sha256': initial}
    raw.write(output / 'provenance.json', provenance)
    raw.progress(output, 'gradient_preflight')
    attention = check_attention(PacketTransitionReadout(config['seed']).to(device), device)
    checks = probe(model, data, config, device)
    require(raw.state_digest(model.state_dict()) == before, 'Preflight mutated frozen C')
    raw.write(output / 'preflight.json', {**checks, 'attention_checks': attention})

    val, val_coverage, reference = cache_split(model, data, rows['val'], 'val', config, device, output, heads)
    selected = archive['C']['selected']['best']
    gold = joint.legacy.predictions(source / 'C' / 'validation_predictions.jsonl.gz')
    baseline_metrics = joint.study.validate_predictions(reference, rows['val'], probabilities=False)
    receipt = evaluation.replay_receipt(baseline_metrics, reference, selected, gold)
    receipt = {**receipt, 'exact_zero_residual_and_scores': list(heads)}
    raw.write(output / 'validation_replay.json', {'baseline_and_epoch_zero': receipt})
    require(receipt['passed'] and receipt['prediction_ids_sha256'] == config['baseline_prediction_ids_sha256'],
            'Original C validation predictions or metrics differ')
    raw.progress(output, 'preflight_passed', validation_samples=len(reference), macro_f1=baseline_metrics['macro_f1'])

    train, train_coverage, _ = cache_split(model, data, rows['train'], 'train', config, device, output)
    coverage = {'train': train_coverage, 'val': val_coverage}
    raw.write(output / 'edge_coverage.json', coverage)
    best, last, updates = fit(model, heads, train, val, rows['val'], reference, config, device, output)
    lengths = val.features.lengths.tolist()
    del train, val
    checkpoint = output / 'transition_readouts.pt'
    torch.save({'best': best, 'last': last, 'query_sources': SOURCES, 'config': config,
                'frozen_c_sha256': before, 'initial_transition_sha256': initial}, checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    predictions = replay_selected(model, saved, data, rows['val'], config, device, output)
    replays = {arm: evaluation.replay_receipt(joint.study.validate_predictions(items, rows['val'], probabilities=False),
              items, selected if arm == 'A' else best[arm], reference if arm == 'A' else None)
               for arm, items in predictions.items()}
    raw.write(output / 'validation_replay.json', {'baseline_and_epoch_zero': receipt, 'selected_raw': replays})
    require(all(item['passed'] for item in replays.values()), 'Selected checkpoint raw replay differs')
    require(raw.state_digest(model.state_dict()) == before
            and all(p.grad is None and not p.requires_grad for p in model.parameters())
            and all(not module.training for module in model.modules()), 'Frozen C changed')
    evaluation.verify_inputs(inputs)
    raw.write(output / 'provenance.json', {**provenance, 'frozen_c_verified': True,
              'transition_checkpoint_sha256': raw.sha(checkpoint)})
    summarize(output, rows['val'], predictions, lengths, saved, config, updates, coverage, time.perf_counter() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, default=Path(OUTPUT))
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--code-commit', required=True)
    args = parser.parse_args()
    require(re.fullmatch(r'[0-9a-f]{40}', args.code_commit), 'Expected a full code commit SHA')
    repo = args.repo.resolve()
    output = (repo / args.output).resolve()
    require(output.parent == (repo / 'outputs').resolve(), 'Use a new directory directly under outputs')
    output.mkdir(parents=True)
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
