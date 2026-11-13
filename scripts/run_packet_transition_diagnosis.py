"""Diagnose frozen C features and chronological Packet fields on train/validation."""
import argparse
import csv
import json
from pathlib import Path
import platform
import re
import time

import torch
from torch import nn
from torch.nn import functional as F
import torch_geometric

import replay_flow_query_selective as replay
from audit_packet_transitions import STAT_NAMES, aggregate, audit_batch


evaluation, joint, raw, require, LABELS = replay.evaluation, replay.joint, replay.raw, replay.require, replay.LABELS
CONFIG = raw.ROOT / 'configs/packet_transition_diagnosis_s42_v1.json'
OUTPUT = 'outputs/packet_transition_diagnosis_s42_v1'
DIRECTIONS = (('Recon', 'WebBased'), ('WebBased', 'Recon'),
              ('Benign', 'Spoofing'), ('Spoofing', 'Benign'))
FEATURES = {'Flow': 64, 'PacketMean': 64, 'PacketQuery': 64, 'FlowPacketMean': 128,
            'D32Flow': 32, 'D32Packet': 32, 'D32Fused': 32, 'PacketMeanTransition': 64 + len(STAT_NAMES)}


@torch.no_grad()
def frozen_query_features(model, batch):
    flow, first, mean, query = replay.shared_readouts(model, batch)
    packet_query = model.cross_layer(torch.cat((first, query), dim=-1))
    fusion = model.fusion
    flow32, packet32, fused32 = [], [], []
    hooks = [
        fusion.flow_projection.register_forward_hook(lambda module, args, result: flow32.append(result)),
        fusion.packet_projection.register_forward_hook(lambda module, args, result: packet32.append(result)),
        fusion.shared_head.register_forward_pre_hook(lambda module, args: fused32.append(args[0])),
    ]
    try:
        scores, alpha = fusion(flow, packet_query)
    finally:
        for hook in hooks:
            hook.remove()
    require(len(flow32) == len(packet32) == len(fused32) == 1 and bool(torch.isfinite(scores).all()),
            'Invalid C fusion output')
    return {'Flow': flow, 'PacketQuery': packet_query, 'D32Flow': flow32[0],
            'D32Packet': packet32[0], 'D32Fused': fused32[0], 'Alpha': alpha,
            'LogProbabilities': scores}, first, mean


@torch.no_grad()
def frozen_features(model, batch):
    features, first, mean = frozen_query_features(model, batch)
    scores = features.pop('LogProbabilities')
    del features['Alpha']
    packet_mean = model.cross_layer(torch.cat((first, mean), dim=-1))
    features['PacketMean'] = packet_mean
    features['FlowPacketMean'] = torch.cat((features['Flow'], packet_mean), dim=-1)
    return features, scores.argmax(dim=-1).tolist()


def aligned_batches(data, rows, part, config):
    offset = 0
    for step, batch in enumerate(joint.legacy.loader(data, rows, part, config), 1):
        end = offset + batch.num_graphs
        expected = rows[offset:end]
        actual = list(zip(batch.sample_id, batch.y.view(-1).tolist()))
        require(actual == [(r['sample_id'], r['label']) for r in expected], f'{part} IDs, labels or order differ')
        yield step, slice(offset, end), batch, expected
        offset = end
    require(offset == len(rows), f'Incomplete {part} extraction')


@torch.no_grad()
def extract(model, data, rows, part, config, device, output):
    values = {name: torch.empty((len(rows), size), dtype=torch.float32) for name, size in FEATURES.items()}
    audits, predictions = [], []
    for step, indices, batch, expected in aligned_batches(data, rows, part, config):
        checks, stats = audit_batch(batch)
        features, preds = frozen_features(model, batch.to(device))
        features['PacketMeanTransition'] = torch.cat((features['PacketMean'], torch.from_numpy(stats).to(device)), -1)
        for name, tensor in features.items():
            require(tensor.shape == (len(expected), FEATURES[name]) and bool(torch.isfinite(tensor).all()),
                    f'Invalid frozen feature tensor {name}')
            values[name][indices] = tensor.cpu()
        audits.extend({'sample_id': row['sample_id'], 'label': row['label'], **check}
                      for row, check in zip(expected, checks))
        predictions.extend({'sample_id': row['sample_id'], 'label': row['label'], 'prediction': pred}
                           for row, pred in zip(expected, preds))
        if step == 1 or step % 200 == 0 or indices.stop == len(rows):
            raw.progress(output, f'extract_{part}', samples=indices.stop, total=len(rows))
    return values, audits, predictions


def train_probes(train, val, train_rows, val_rows, config, device, output):
    heads = {}
    for name, width in FEATURES.items():
        raw.seed_all(config['seed'])
        heads[name] = nn.Linear(width, len(LABELS))
    exports, checkpoints, updates = fit_probes(heads, train, val, train_rows, val_rows, config, device, output)
    torch.save({'epoch': config['epochs'], 'seed': config['seed'], 'probes': checkpoints}, output / 'linear_probes.pt')
    return exports, updates


def fit_probes(heads, train, val, train_rows, val_rows, config, device, output):
    labels = torch.tensor([r['label'] for r in train_rows], device=device)
    sampler = raw.TrainBatches(len(train_rows), config['batch_size'], config['seed'])
    train, val = dict(train), dict(val)
    optimizers, normalization = {}, {}
    for name, head in heads.items():
        train[name] = train[name].to(device)
        val[name] = val[name].to(device)
        mean = train[name].mean(dim=0)
        scale = train[name].std(dim=0, unbiased=False)
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        normalization[name] = {'mean': mean.cpu(), 'scale': scale.cpu()}
        train[name] = (train[name] - mean) / scale
        val[name] = (val[name] - mean) / scale
        head.to(device)
        optimizers[name] = torch.optim.Adam(head.parameters(), lr=config['learning_rate'],
                                             weight_decay=config['weight_decay'])
    with (output / 'epochs.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
        for epoch in range(1, config['epochs'] + 1):
            losses = {name: torch.zeros((), device=device) for name in heads}
            samples, updates = 0, 0
            for indices in sampler:
                idx = torch.as_tensor(indices, device=device)
                y = labels[idx]
                for name, head in heads.items():
                    optimizer = optimizers[name]
                    optimizer.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(head(train[name][idx]), y)
                    loss.backward()
                    optimizer.step()
                    losses[name] += loss.detach() * len(indices)
                samples += len(indices)
                updates += 1
            require(samples == len(train_rows) and updates == len(sampler)
                    and all(bool(torch.isfinite(loss)) for loss in losses.values()), 'Incomplete or nonfinite probe epoch')
            stream.write(json.dumps({'epoch': epoch, 'samples_per_probe': samples, 'updates_per_probe': updates,
                                     'order_sha256': sampler.order_sha256,
                                     'train_ce': {name: float(loss / samples) for name, loss in losses.items()}}) + '\n')
            stream.flush()
            raw.progress(output, 'probes', epoch=epoch, total_epochs=config['epochs'])
    exports, checkpoints = {}, {}
    with torch.no_grad():
        for name, head in heads.items():
            head.eval()
            scores = head(val[name])
            require(bool(torch.isfinite(scores).all()), f'Invalid probe scores {name}')
            exports[name] = [{**{key: row[key] for key in ('sample_id', 'label')}, 'prediction': pred}
                             for row, pred in zip(val_rows, scores.argmax(-1).tolist())]
            checkpoints[name] = {**normalization[name], 'state': {k: v.cpu() for k, v in head.state_dict().items()}}
    return exports, checkpoints, config['epochs'] * len(sampler)


def diagnostics(reference, exports, val_audits):
    by_model = {**reference, **exports}
    metrics = {name: joint.study.validate_predictions(records, reference['C'], probabilities=False)
               for name, records in by_model.items()}
    c, a, b = (reference[name] for name in ('C', 'A', 'B37'))
    common_recon = [i for i, (rc, ra, rb) in enumerate(zip(c, a, b))
                    if (rc['label'], rc['prediction'], ra['prediction'], rb['prediction']) == (3, 1, 1, 1)]
    b37_repaired = [i for i, (rc, rb) in enumerate(zip(c, b))
                    if (rc['label'], rc['prediction'], rb['prediction']) == (3, 1, 3)]
    targets = {f'{true} -> {wrong}': [i for i, r in enumerate(c)
               if (r['label'], r['prediction']) == (LABELS.index(true), LABELS.index(wrong))]
               for true, wrong in DIRECTIONS}
    subsets = {**targets, 'common_Recon -> WebBased_C_A_B37': common_recon, 'B37_repaired_Recon': b37_repaired}
    probes = {}
    for name, records in exports.items():
        changes, _ = raw.compare(c, records)
        directions = {}
        for true, wrong in DIRECTIONS:
            key = f'{true} -> {wrong}'
            y, p = LABELS.index(true), LABELS.index(wrong)
            indices = targets[key]
            directions[key] = {'original_errors': len(indices),
                               'repaired': sum(records[i]['prediction'] == y for i in indices),
                               'new_errors': sum(r['label'] == y and r['prediction'] != p
                                                 and n['prediction'] == p for r, n in zip(c, records)),
                               'new_errors_from_original_correct': sum(r['label'] == y and r['prediction'] == y
                                                                     and n['prediction'] == p for r, n in zip(c, records)),
                               'probe_errors': metrics[name]['confusion_matrix'][y][p]}
        probes[name] = {**changes, 'directions': directions,
                        'common_38_repaired': sum(records[i]['prediction'] == 3 for i in common_recon),
                        'delta_macro_f1_pp': 100 * (metrics[name]['macro_f1'] - metrics['C']['macro_f1'])}
    matched = [{'sample_id': row['sample_id'], 'label': row['label'],
                'predictions': {name: records[i]['prediction'] for name, records in by_model.items()},
                'transition_fields': {k: v for k, v in val_audits[i].items() if k not in ('sample_id', 'label')}}
               for i, row in enumerate(c)]
    return metrics, probes, subsets, matched


def write_report(output, metrics, probes, fields, config, updates, seconds):
    lines = ['# Seed42 冻结特征诊断与报文转移审计', '',
             f"原 C 已逐样本复现，验证 Macro-F1 为 {metrics['C']['macro_f1']:.6%}。"
             f"每个线性头使用同一训练顺序训练 {config['epochs']} 轮，共 {updates} 次更新，"
             '表中使用固定末轮。标准化统计来自训练集。', '',
             '| 表示 | 维度 / 参数量 | 验证 Macro-F1 | 相对 C（百分点） | 修复 / 退步 |',
             '|---|---:|---:|---:|---:|']
    for name, size in FEATURES.items():
        p, m = probes[name], metrics[name]
        lines.append(f"| {name} | {size} / {(size + 1) * len(LABELS)} | {m['macro_f1']:.6%} | "
                     f"{p['delta_macro_f1_pp']:+.6f} | {p['counts']['repaired']} / {p['counts']['regressed']} |")
    for true, wrong in DIRECTIONS:
        key = f'{true} -> {wrong}'
        lines += ['', f'## {true} → {wrong}', '',
                  '| 表示 | C 原错分 | 原错分判对 | 新增该方向错分 | 探针该方向错分 |',
                  '|---|---:|---:|---:|---:|']
        for name, item in probes.items():
            d = item['directions'][key]
            lines.append(f"| {name} | {d['original_errors']} | {d['repaired']} | {d['new_errors']} | {d['probe_errors']} |")
    lines += ['', '## C、A、B37 共同的 Recon → WebBased 难例', '',
              '| 表示 | 找回数 |', '|---|---:|']
    for name, item in probes.items():
        lines.append(f"| {name} | {item['common_38_repaired']} |")
    lines += ['', '## 有效相邻报文覆盖', '',
              '| 范围 | 图数 | 无有效转移 | 至少两条有效转移 | 有效 / 预期转移 | 同向 / 换向 |',
              '|---|---:|---:|---:|---:|---:|']
    for name, item in {'train': fields['train']['all'], 'val': fields['val']['all'], **fields['hard_cases']}.items():
        lines.append(f"| {name} | {item.get('graphs', 0)} | {item.get('graphs_without_valid_pairs', 0)} | "
                     f"{item.get('graphs_with_at_least_two_valid_pairs', 0)} | "
                     f"{item.get('valid_pairs', 0)} / {item.get('expected_pairs', 0)} | "
                     f"{item.get('same_direction_pairs', 0)} / {item.get('direction_change_pairs', 0)} |")
    stat_delta = 100 * (metrics['PacketMeanTransition']['macro_f1'] - metrics['PacketMean']['macro_f1'])
    raw_probe = probes['FlowPacketMean']['directions']['Recon -> WebBased']
    fused_probe = probes['D32Fused']['directions']['Recon -> WebBased']
    plain_probe = probes['PacketMean']['directions']['Recon -> WebBased']
    stat_probe = probes['PacketMeanTransition']['directions']['Recon -> WebBased']
    lines += ['', f"FlowPacketMean 找回 {raw_probe['repaired']} 个原 Recon → WebBased，新增 {raw_probe['new_errors']} 个。"
              f"D32Fused 找回 {fused_probe['repaired']} 个，新增 {fused_probe['new_errors']} 个。",
              f'PacketMeanTransition 相对 PacketMean 的验证 Macro-F1 变化为 {stat_delta:+.6f} 个百分点。'
              f"原 Recon → WebBased 找回数从 {plain_probe['repaired']} 到 {stat_probe['repaired']}，"
              f"新增该方向错分从 {plain_probe['new_errors']} 到 {stat_probe['new_errors']}。", '',
              '有效边按时间顺序只计同流 i→i+1。NFStream 方向 0/1 合法，时间单位为毫秒，载荷为字节。'
              '0 毫秒和 0 字节保留。缺失或重复的 contain 映射、越界字段和非相邻边单列审计。'
              '归档图中没有用于区分零值填补与合法零值的掩码。', '',
              f'本次耗时 {seconds:.1f} 秒。八类指标和混淆矩阵见 `metrics.json`，分真实类别和难例的字段审计见 `field_audit.json`。', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def run(repo, output, graph_root, code_commit):
    started = time.perf_counter()
    settings = raw.read(CONFIG)
    config = {**raw.read(evaluation.CONFIG), **settings}
    config['checkpoints'] = {'C': config['checkpoints']['C']}
    config['metrics_sha256'] = {'C': config['metrics_sha256']['C']}
    source = repo / config['source_directory']
    split = repo / config['split_directory']
    archive, inputs = evaluation.archive_inputs(source, split, config)
    compare_paths = {name: repo / settings['comparison_validation_directory'] / f'{name}_predictions.jsonl.gz'
                     for name in settings['comparison_sha256']}
    compare_inputs = {path: settings['comparison_sha256'][name] for name, path in compare_paths.items()}
    evaluation.verify_inputs(compare_inputs)
    inputs.update(compare_inputs)
    require(torch.cuda.is_available() and torch.__version__.split('+')[0] == '2.3.1'
            and torch_geometric.__version__ == '2.5.3', 'Expected CUDA, Torch 2.3.1 and PyG 2.5.3')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    data = raw.FrozenGraphs(split, config['split_id'], data_root=graph_root)
    inputs[data.directory / data.metadata['sample_manifest']] = data.metadata['sample_manifest_sha256']
    rows = {'val': evaluation.evaluation_rows(data, 'val', config), 'train': raw.choose_rows(data, 'train')}
    require(len(rows['train']) == config['training_samples'], 'Wrong training set size')
    protocol = raw.read(joint.CONFIG)
    state = evaluation.load_checkpoint_state(source, config, archive, protocol, 'C')
    base = raw.OperatorClassifier('GAT', seed=config['seed'])
    model = joint.JointCrossLayerQueryClassifier(base.flow_encoder, base.packet_encoder, base.fusion,
                                                 'flow', config['seed'])
    model.load_state_dict(state, strict=True)
    require(joint.legacy.frozen_hash(model) == protocol['source_frozen_state_sha256'], 'Archived C frozen state differs')
    model.requires_grad_(False).to(device).eval()
    before = raw.state_digest(model.state_dict())
    raw.write(output / 'config.json', {**config, 'features': FEATURES, 'transition_stat_columns': STAT_NAMES,
                                       'partitions_evaluated': ['train', 'val'], 'probe_selection': 'fixed_last_epoch'})
    raw.write(output / 'provenance.json', {'code_commit': code_commit, 'archive_commit': config['archive_commit'],
              'comparison_archive_commit': settings['comparison_archive_commit'], 'graph_root': str(data.root),
              'input_sha256': {str(path): digest for path, digest in inputs.items()}, 'model_state_sha256': before,
              'python': platform.python_version(), 'torch': torch.__version__, 'pyg': torch_geometric.__version__,
              'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(device)})
    val, val_audits, c = extract(model, data, rows['val'], 'val', config, device, output)
    gold = joint.legacy.predictions(source / 'C' / 'validation_predictions.jsonl.gz')
    val_metrics = joint.study.validate_predictions(c, rows['val'], probabilities=False)
    receipt = evaluation.replay_receipt(val_metrics, c, archive['C']['selected']['best'], gold)
    raw.write(output / 'validation_replay.json', receipt)
    require(receipt['passed'], 'Frozen-feature validation replay differs from original C')
    reference = {'C': c, **{name: joint.legacy.predictions(path) for name, path in compare_paths.items()}}
    for records in reference.values():
        joint.study.validate_predictions(records, rows['val'], probabilities=False)
    train, train_audits, _ = extract(model, data, rows['train'], 'train', config, device, output)
    require(raw.state_digest(model.state_dict()) == before, 'Frozen C changed during extraction')
    exports, updates = train_probes(train, val, rows['train'], rows['val'], config, device, output)
    metrics, probes, subsets, matched = diagnostics(reference, exports, val_audits)
    require(len(subsets['Recon -> WebBased']) == 43 and len(subsets['WebBased -> Recon']) == 15
            and len(subsets['common_Recon -> WebBased_C_A_B37']) == 38 and len(subsets['B37_repaired_Recon']) == 5,
            'Archived hard-case cohorts differ')
    fields = {}
    for part, records in (('train', train_audits), ('val', val_audits)):
        fields[part] = {'all': aggregate(records), 'by_class': {
            label: aggregate(r for r in records if r['label'] == i) for i, label in enumerate(LABELS)}}
    fields['hard_cases'] = {name: aggregate(val_audits[i] for i in indices) for name, indices in subsets.items()}
    raw.write(output / 'field_audit.json', fields)
    raw.write(output / 'metrics.json', {'metrics': metrics, 'probe_comparisons': probes, 'updates_per_probe': updates})
    raw.save_predictions(output / 'validation_predictions.jsonl.gz', matched)
    raw.save_predictions(output / 'hard_cases.jsonl.gz',
                         ({**matched[i], 'cohort': name} for name, indices in subsets.items() for i in indices))
    with (output / 'per_class.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', 'class', 'precision', 'recall', 'f1', 'support'])
        for name, result in metrics.items():
            for label, item in result['per_class'].items():
                writer.writerow([name, label, item['precision'], item['recall'], item['f1-score'], item['support']])
    evaluation.verify_inputs(inputs)
    write_report(output, metrics, probes, fields, config, updates, time.perf_counter() - started)


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
    print(f'Diagnosis output: {output}', flush=True)
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
