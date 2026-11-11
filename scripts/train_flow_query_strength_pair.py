"""Train the two seed42 Flow strength parameters on a shared frozen representation."""
import argparse
import json
from pathlib import Path
import tempfile

import torch
from torch import nn
from torch.nn import functional as F
import torch_geometric

import replay_flow_query_strength as replay

core, raw, require, LABELS = replay.core, replay.raw, replay.require, replay.LABELS
CONFIG = raw.ROOT / 'configs/flow_query_strength_pair_s42_v1.json'
ARMS = ('global', 'adaptive')


class GlobalStrength(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit = nn.Parameter(torch.zeros(()))

    def forward(self, flow, mean):
        return self.logit.sigmoid().expand(flow.size(0), 1)


def adaptive_strength():
    # Constructing the linear layer should leave the sampling RNG untouched.
    with torch.random.fork_rng(devices=[]):
        layer = nn.Linear(128, 1)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return nn.Sequential(layer, nn.Sigmoid())


def classify(fusion, arm, strength, features):
    flow, mean, delta = features.unbind(dim=1)
    if arm == 'global':
        lam = strength(flow, mean)
    else:
        lam = strength(torch.cat((flow, mean), dim=-1))
    scores, _ = fusion(flow, mean + lam * delta)
    return scores, lam.squeeze(-1)


@torch.no_grad()
def extract_features(model, batches, rows, device, output, partition):
    features, labels, original = [], [], []
    offset = 0
    for step, batch in enumerate(batches, 1):
        expected = rows[offset:offset + batch.num_graphs]
        identities = list(zip(batch.sample_id, batch.y.view(-1).tolist()))
        require(identities == [(r['sample_id'], r['label']) for r in expected], 'Feature IDs/labels/order differ')
        batch = batch.to(device)
        flow, nodes, index, size = model.encode(batch)
        mean, delta, full = replay.packet_components(model.packet_pool, flow, nodes, index, size)
        features.append(torch.stack((flow, mean, delta), dim=1))
        labels.append(batch.y.view(-1))
        if partition == 'val':
            require(torch.equal(mean + delta, full), 'Cached components differ from original Packet pool')
            scores, _ = model.fusion(flow, full)
            original.extend({'sample_id': sid, 'label': label, 'prediction': pred}
                            for (sid, label), pred in zip(identities, scores.argmax(-1).tolist()))
        offset += size
        if step == 1 or step % 200 == 0 or offset == len(rows):
            raw.progress(output, f'extract_{partition}', samples=offset, total=len(rows))
    require(offset == len(rows), 'Incomplete feature extraction')
    values = torch.cat(features)
    require(bool(torch.isfinite(values).all()), 'Nonfinite frozen features')
    return values, torch.cat(labels), original


@torch.no_grad()
def evaluate(fusion, arm, strength, features, rows, batch_size):
    predictions, lambdas, finite = [], [], []
    for batch in features.split(batch_size):
        scores, lam = classify(fusion, arm, strength, batch)
        predictions.append(scores.argmax(-1))
        lambdas.append(lam)
        finite.append(torch.isfinite(scores).all())
    lambdas = torch.cat(lambdas)
    require(bool(torch.stack(finite).all() & ((lambdas >= 0) & (lambdas <= 1)).all()),
            'Invalid validation scores or strengths')
    records = [{'sample_id': row['sample_id'], 'label': row['label'], 'prediction': pred, 'lambda': lam}
               for row, pred, lam in zip(rows, torch.cat(predictions).tolist(), lambdas.tolist())]
    return replay.study.validate_predictions(records, rows, probabilities=False), records


def fit_pair(fusion, train_features, train_labels, val_features, val_rows, config, output):
    device = train_features.device
    arms = {'global': GlobalStrength().to(device), 'adaptive': adaptive_strength().to(device)}
    optimizers = {arm: torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
                  for arm, model in arms.items()}
    parameter_counts = {arm: sum(p.numel() for p in model.parameters()) for arm, model in arms.items()}
    require(parameter_counts == {'global': 1, 'adaptive': 129}
            and all(not p.requires_grad for p in fusion.parameters()) and not fusion.training,
            'Only the strength parameters can be trained')
    sampler = raw.TrainBatches(len(train_labels), config['batch_size'], config['seed'])
    initial = {arm: evaluate(fusion, arm, model, val_features, val_rows, config['eval_batch_size'])
               for arm, model in arms.items()}
    require(initial['global'] == initial['adaptive']
            and all(row['lambda'] == 0.5 for row in initial['global'][1]), 'Paired lambda=0.5 initialization differs')
    best = {}
    with (output / 'epochs.jsonl').open('x', encoding='utf-8') as stream:
        stream.write(json.dumps({'epoch': 0, 'validation': {arm: value[0] for arm, value in initial.items()}}) + '\n')
        for epoch in range(1, config['epochs'] + 1):
            losses = {arm: torch.zeros((), device=device) for arm in arms}
            samples, steps = 0, 0
            for indices in sampler:
                index = torch.as_tensor(indices, device=device)
                features, labels = train_features[index], train_labels[index]
                for arm, strength in arms.items():
                    optimizer = optimizers[arm]
                    optimizer.zero_grad(set_to_none=True)
                    scores, _ = classify(fusion, arm, strength, features)
                    loss = F.nll_loss(scores, labels)
                    loss.backward()
                    optimizer.step()
                    losses[arm] += loss.detach() * len(indices)
                samples += len(indices)
                steps += 1
            require(samples == len(train_labels) and steps == len(sampler), 'Training epoch budget differs')
            validation = {}
            for arm, strength in arms.items():
                require(bool(torch.isfinite(losses[arm])) and
                        all(bool(torch.isfinite(p).all()) and p.grad is not None for p in strength.parameters()),
                        f'Invalid strength optimization for {arm}')
                metrics, _ = evaluate(fusion, arm, strength, val_features, val_rows, config['eval_batch_size'])
                validation[arm] = metrics
                if arm not in best or metrics['macro_f1'] > best[arm]['validation']['macro_f1']:
                    best[arm] = {'epoch': epoch, 'validation': metrics,
                                 'state': {name: tensor.detach().cpu().clone()
                                           for name, tensor in strength.state_dict().items()}}
            stream.write(json.dumps({'epoch': epoch, 'samples_per_arm': samples, 'updates_per_arm': steps,
                                     'order_sha256': sampler.order_sha256, 'validation': validation,
                                     'train_nll': {arm: float(loss / samples) for arm, loss in losses.items()}}) + '\n')
            stream.flush()
            raw.progress(output, 'paired_training', epoch=epoch, epochs=config['epochs'],
                         macro_f1={arm: value['macro_f1'] for arm, value in validation.items()})
    require(all(any(bool(p.detach().ne(0).any()) for p in model.parameters()) for model in arms.values()),
            'A strength model did not learn from its initialization')
    exports = {}
    for arm, strength in arms.items():
        strength.load_state_dict(best[arm]['state'], strict=True)
        metrics, exports[arm] = evaluate(fusion, arm, strength, val_features, val_rows, config['eval_batch_size'])
        require(metrics == best[arm]['validation'], f'Selected validation metrics differ for {arm}')
    return best, exports, config['epochs'] * len(sampler), parameter_counts


def compare(original, current):
    classes = {name: {'original_correct': 0, 'original_missed': 0, 'retained': 0, 'repaired': 0, 'regressed': 0}
               for name in ('WebBased', 'Recon')}
    for old, new in zip(original, current):
        label = old['label']
        if LABELS[label] not in classes:
            continue
        counts = classes[LABELS[label]]
        if old['prediction'] == label:
            counts['original_correct'] += 1
            key = 'retained' if new['prediction'] == label else 'regressed'
            counts[key] += 1
        else:
            counts['original_missed'] += 1
            counts['repaired'] += int(new['prediction'] == label)
    return classes


def summarize(original, trained, threshold):
    adaptive = trained['adaptive']['validation']
    global_metrics = trained['global']['validation']
    control = 'global' if global_metrics['macro_f1'] > original['macro_f1'] else 'original'
    versus = {'original': original, 'global': global_metrics}
    deltas = {name: 100 * (adaptive['macro_f1'] - metrics['macro_f1']) for name, metrics in versus.items()}
    web, recon = (adaptive['per_class'][name]['recall'] - original['per_class'][name]['recall']
                  for name in ('WebBased', 'Recon'))
    return {'stronger_control': control, 'adaptive_delta_macro_f1_pp': deltas,
            'adaptive_beats_global': deltas['global'] > 0, 'threshold_pp': threshold,
            'macro_f1_budget_met': deltas[control] >= threshold,
            'adaptive_recall_delta_vs_original_pp': {'WebBased': 100 * web, 'Recon': 100 * recon},
            'webbased_recall_preserved_and_recon_improved': web >= 0 and recon > 0}


def write_report(output, original, trained, focus, comparison, updates, config):
    lines = [
        '# Seed42 Flow 引导强度配对训练', '',
        f"同一 Joint-Flow 第 95 轮起点，原模型全程冻结。两组从 λ=0.5 起步，各训练 {config['epochs']} 轮"
        f"（{updates} 次更新），按第 1 至 {config['epochs']} 轮验证 Macro-F1 选最佳，同分取最早轮。",
        f"固定强度回放见[修正报告]({config['replay_report']})。", '',
        '| 模型 | 轮次 | Macro-F1 | WebBased 召回 | Recon 召回 |',
        '|---|---:|---:|---:|---:|',
    ]
    results = {'原版 Joint-Flow': (95, original), '全局强度': (trained['global']['epoch'], trained['global']['validation']),
               '样本自适应强度': (trained['adaptive']['epoch'], trained['adaptive']['validation'])}
    for name, (epoch, metric) in results.items():
        web, recon = (metric['per_class'][label]['recall'] for label in ('WebBased', 'Recon'))
        lines.append(f"| {name} | {epoch} | {metric['macro_f1']:.6%} | {web:.6%} | {recon:.6%} |")
    lines += ['', '| 模型 | 原版判对的 WebBased 保留数 | 原版漏检的 Recon 找回数 | Recon 新增漏检 |',
              '|---|---:|---:|---:|']
    for arm, name in zip(ARMS, ('全局强度', '样本自适应强度')):
        web, recon = (focus[arm][label] for label in ('WebBased', 'Recon'))
        lines.append(f"| {name} | {web['retained']} / {web['original_correct']} | "
                     f"{recon['repaired']} / {recon['original_missed']} | {recon['regressed']} |")
    delta = comparison['adaptive_delta_macro_f1_pp']
    stronger = {'original': '原版 Joint-Flow', 'global': '全局强度'}[comparison['stronger_control']]
    verdict = '达到' if comparison['macro_f1_budget_met'] else '低于'
    lines += ['', f"自适应相对原版为 {delta['original']:+.6f} 个百分点，相对全局强度为 {delta['global']:+.6f} 个百分点。",
              f"以 {stronger} 为较强对照，Macro-F1 增量{verdict} +{comparison['threshold_pp']:.2f} 个百分点预算线。",
              '', '逐样本预测和 λ 见 `validation_predictions.jsonl.gz`。', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def run(repo, output):
    require(torch.cuda.is_available() and torch.__version__.split('+')[0] == '2.3.1'
            and torch_geometric.__version__ == '2.5.3', 'Requires CUDA, Torch 2.3.1 and PyG 2.5.3')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    replay_config = raw.read(replay.CONFIG)
    config = {**replay_config, **raw.read(CONFIG)}
    source = config.pop('seeds')[str(config['seed'])]
    raw.write(output / 'config.json', {**config, 'source': source, 'optimizer': 'Adam', 'loss': 'NLL',
                                       'selection_epochs': [1, config['epochs']], 'tie_policy': 'earliest'})
    split = repo / config['split_directory']
    replay.verify_source(split / 'split.json', config['split_json_sha256'])
    data = raw.FrozenGraphs(split, config['split_id'])
    val_rows = raw.choose_rows(data, 'val')
    train_rows = raw.choose_rows(data, 'train')
    require(len(train_rows) == config['train_samples'] and len(val_rows) == config['validation_samples'],
            'Unexpected training or validation size')
    raw.seed_all(config['seed'])
    model, original, _, baseline, _ = replay.load_source(repo, config['seed'], source, config, val_rows)
    model.to(device)
    frozen_state = raw.state_digest(model.state_dict())

    val_loader = core.graph_loader(data, val_rows, 'val', config)
    val_features, _, reproduced = extract_features(model, val_loader, val_rows, device, output, 'val')
    replay.verify_baseline(reproduced, original, config['seed'], output)
    train_loader = core.graph_loader(data, train_rows, 'train', {**config, 'eval_batch_size': config['feature_batch_size']})
    train_features, labels, _ = extract_features(model, train_loader, train_rows, device, output, 'train')
    best, exports, updates, counts = fit_pair(model.fusion, train_features, labels, val_features, val_rows, config, output)
    require(raw.state_digest(model.state_dict()) == frozen_state and
            all(p.grad is None and not p.requires_grad for p in model.parameters()), 'Original Joint-Flow state changed')

    torch.save({'source_checkpoint_sha256': source['checkpoint_sha256'], 'arms': best}, output / 'strength_best.pt')
    predictions = [{'sample_id': row['sample_id'], 'label': row['label'],
                    'original': {'prediction': row['prediction'], 'lambda': 1.0},
                    **{arm: {key: exports[arm][i][key] for key in ('prediction', 'lambda')} for arm in ARMS}}
                   for i, row in enumerate(original)]
    raw.save_predictions(output / 'validation_predictions.jsonl.gz', predictions)
    trained = {arm: {'epoch': result['epoch'], 'validation': result['validation'], 'parameter_count': counts[arm]}
               for arm, result in best.items()}
    focus = {arm: compare(original, exports[arm]) for arm in ARMS}
    comparison = summarize(baseline, trained, config['expansion_threshold_pp'])
    raw.write(output / 'summary.json', {'original': baseline, 'trained': trained, 'focus': focus,
              'comparison': comparison, 'updates_per_arm': updates, 'validation_samples': len(val_rows),
              'frozen_state_sha256': frozen_state, 'torch': torch.__version__, 'pyg': torch_geometric.__version__,
              'device': torch.cuda.get_device_name(0)})
    write_report(output, baseline, trained, focus, comparison, updates, config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo = args.repo.resolve()
    (repo / 'outputs').mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='flow_query_strength_pair_s42_', dir=repo / 'outputs'))
    print(f'Training output: {output}', flush=True)
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
