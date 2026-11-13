"""Train matched local rank corrections on frozen C features, using train/val only."""
import argparse
from collections import Counter
import copy
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

import run_fusion_decision_diagnosis as diagnosis


evaluation, joint, raw, require, LABELS = (
    diagnosis.evaluation, diagnosis.joint, diagnosis.raw, diagnosis.require, diagnosis.LABELS)
RECON, WEB = diagnosis.RECON, diagnosis.WEB
HEADS = ('Fused', 'Difference')
CONFIG = raw.ROOT / 'configs/recon_webbased_rerank_s42_v1.json'
OUTPUT = 'outputs/recon_webbased_rerank_s42_v1'


def make_heads(seed):
    raw.seed_all(seed)
    template = nn.Sequential(nn.Linear(32, 16), nn.LeakyReLU(), nn.Linear(16, 1))
    nn.init.zeros_(template[-1].weight)
    nn.init.zeros_(template[-1].bias)
    require(sum(p.numel() for p in template.parameters()) == 545, 'Unexpected correction head size')
    return {name: copy.deepcopy(template) for name in HEADS}


def fit_bias(margin, labels):
    # BCE is strictly convex here, with an interior minimum when both labels occur.
    margin, labels = margin.detach().cpu().double(), labels.detach().cpu().double()
    require(set(labels.tolist()) == {0., 1.}, 'Bias fit requires both training labels')
    bound = float(margin.abs().max()) + 50
    lo, hi = -bound, bound
    for _ in range(80):
        midpoint = (lo + hi) / 2
        if float(((margin + midpoint).sigmoid() - labels).mean()) > 0:
            hi = midpoint
        else:
            lo = midpoint
    bias = (lo + hi) / 2
    require(abs(float(((margin + bias).sigmoid() - labels).mean())) < 1e-12, 'Bias fit did not converge')
    return {'bias': bias, 'train_bce': float(F.binary_cross_entropy_with_logits(margin + bias, labels)),
            'fit_samples': len(labels)}


def rerank(records, delta):
    require(len(delta) == len(records) and bool(torch.isfinite(delta).all()), 'Invalid correction vector')
    exports = []
    for row, adjustment in zip(records, delta.tolist()):
        adjustment = adjustment if row['eligible'] else 0.
        corrected = row['recon_minus_webbased'] + adjustment
        pred = row['prediction']
        if row['eligible'] and corrected != 0:
            pred = RECON if corrected > 0 else WEB
        exports.append({'sample_id': row['sample_id'], 'label': row['label'], 'prediction': pred,
                        'delta': adjustment, 'corrected_margin': corrected})
    return exports


def eligible_training_rows(ranking):
    return [i for i, row in enumerate(ranking) if row['eligible'] and row['label'] in (RECON, WEB)]


def prepare_features(train, val, indices, device):
    fitted, train_views, val_views = {}, {}, {}
    for name in HEADS:
        if name == 'Fused':
            a, b = train['D32Fused'][indices], val['D32Fused']
        else:
            a = train['D32Flow'][indices] - train['D32Packet'][indices]
            b = val['D32Flow'] - val['D32Packet']
        a, b = a.to(device), b.to(device)
        mean = a.mean(0)
        scale = a.std(0, unbiased=False)
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        fitted[name] = {'mean': mean.cpu(), 'scale': scale.cpu()}
        train_views[name], val_views[name] = (a - mean) / scale, (b - mean) / scale
        require(bool(torch.isfinite(train_views[name]).all() & torch.isfinite(val_views[name]).all()),
                f'Invalid standardized features for {name}')
    return train_views, val_views, fitted


@torch.no_grad()
def evaluate(head, features, ranking):
    exports = rerank(ranking, head(features).flatten().cpu())
    metrics = joint.study.validate_predictions(exports, ranking, probabilities=False)
    return metrics, exports


def preflight(heads, train, val, margin, labels, ranking, config, output):
    initial = {name: raw.state_digest(head.state_dict()) for name, head in heads.items()}
    require(len(set(initial.values())) == 1, 'Correction heads start from different parameters')
    expected = joint.legacy.identities(ranking)
    for name, head in heads.items():
        _, zero = evaluate(head, val[name], ranking)
        require(joint.legacy.identities(zero) == expected and all(row['delta'] == 0 for row in zero),
                f'{name} epoch zero differs from C')
        probe = copy.deepcopy(head)
        opt = torch.optim.Adam(probe.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        original = raw.state_digest(probe.state_dict())
        opt.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(margin[:16] + probe(train[name][:16]).flatten(), labels[:16]).backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in probe.parameters())
                and all(p.grad is None for p in head.parameters()), f'Unexpected gradient scope for {name}')
        opt.step()
        require(raw.state_digest(probe.state_dict()) != original and raw.state_digest(head.state_dict()) == initial[name],
                f'{name} isolated optimizer step failed')
    offsets = torch.tensor([r['recon_minus_webbased'] for r in ranking], dtype=torch.float64)
    require(joint.legacy.identities(rerank(ranking, -offsets)) == expected, 'Zero margin must preserve C')
    for target, sign in ((RECON, 1), (WEB, -1)):
        extreme = -offsets + sign * 100
        want = [target if r['eligible'] else r['prediction'] for r in ranking]
        require([r['prediction'] for r in rerank(ranking, extreme)] == want, 'Correction escaped eligible pair')
    raw.write(output / 'preflight.json', {'passed': True, 'epoch_zero_samples': len(ranking),
                                        'parameter_count_per_head': 545, 'initial_state_sha256': initial})


def fit_heads(train, val, margin, labels, ranking, config, output):
    heads = {name: head.to(margin.device) for name, head in make_heads(config['seed']).items()}
    preflight(heads, train, val, margin, labels, ranking, config, output)
    optimizers = {name: torch.optim.Adam(head.parameters(), lr=config['learning_rate'],
                                       weight_decay=config['weight_decay']) for name, head in heads.items()}
    sampler = raw.TrainBatches(len(labels), config['batch_size'], config['seed'])
    best, last = {}, {}
    with (output / 'epochs.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
        for epoch in range(config['epochs'] + 1):
            losses = {name: torch.zeros((), device=margin.device) for name in HEADS}
            steps, samples = 0, 0
            if epoch:
                for indices in sampler:
                    index = torch.as_tensor(indices, device=margin.device)
                    for name, head in heads.items():
                        optimizer = optimizers[name]
                        optimizer.zero_grad(set_to_none=True)
                        loss = F.binary_cross_entropy_with_logits(
                            margin[index] + head(train[name][index]).flatten(), labels[index])
                        loss.backward()
                        optimizer.step()
                        losses[name] += loss.detach() * len(indices)
                    steps += 1
                    samples += len(indices)
                require(samples == len(labels) and steps == len(sampler)
                        and all(bool(torch.isfinite(loss)) for loss in losses.values()), 'Incomplete or invalid epoch')
            validation = {}
            for name, head in heads.items():
                metrics, predictions = evaluate(head, val[name], ranking)
                validation[name] = metrics
                if epoch in (0, config['epochs']) or metrics['macro_f1'] > best[name]['validation']['macro_f1']:
                    candidate = {'epoch': epoch, 'validation': metrics,
                                 'prediction_ids_sha256': joint.prediction_digest(predictions),
                                 'state': {key: tensor.detach().cpu().clone() for key, tensor in head.state_dict().items()}}
                    if epoch == 0 or metrics['macro_f1'] > best[name]['validation']['macro_f1']:
                        best[name] = candidate
                    if epoch == config['epochs']:
                        last[name] = candidate
            stream.write(json.dumps({'epoch': epoch, 'updates_per_head': steps, 'samples_per_head': samples,
                                     'order_sha256': sampler.order_sha256, 'validation': validation,
                                     'train_bce': {name: float(loss / samples) for name, loss in losses.items()}
                                     if samples else None}) + '\n')
            stream.flush()
            raw.progress(output, 'rerank_heads', epoch=epoch, total_epochs=config['epochs'],
                         macro_f1={name: metrics['macro_f1'] for name, metrics in validation.items()})
    exports = {}
    for name, head in heads.items():
        for tag, item in (('best', best[name]), ('last', last[name])):
            head.load_state_dict(item['state'], strict=True)
            metrics, records = evaluate(head, val[name], ranking)
            require(metrics == item['validation'] and joint.prediction_digest(records) == item['prediction_ids_sha256'],
                    f'{name} {tag} strict reload differs')
            exports[name if tag == 'best' else name + 'Last'] = records
    return best, last, exports, config['epochs'] * len(sampler)


def load_features(repo, output, provenance, binding, current_binding, model, data, rows, part, config, device):
    entry = provenance['feature_caches'][part]
    recorded = Path(entry['path'])
    # The archive may come from a relocated checkout. Use its cache key within
    # the current repo's known cache directory, never an arbitrary source path.
    require(recorded.name == part + '.pt' and recorded.parent.parent.name == '.fusion_decision_feature_cache'
            and re.fullmatch(r'[0-9a-f]{64}', recorded.parent.name), 'Unexpected archived cache path')
    prior = repo / 'outputs' / '.fusion_decision_feature_cache' / recorded.parent.name / recorded.name
    if prior.is_file() and raw.sha(prior) == entry['sha256']:
        # The archived file is byte-pinned above. Its metadata includes the
        # TorchVersion string subclass written by the original diagnostic.
        cache = torch.load(prior, map_location='cpu', weights_only=False)
        ids = [r['sample_id'] for r in rows]
        labels = [r['label'] for r in rows]
        values = cache['features']
        if (cache['binding'] == {**binding, 'partition': part} and cache['sample_ids'] == ids
                and cache['labels'] == labels and raw.state_digest(values) == cache['feature_sha256']
                and set(values) == set(diagnosis.FEATURES)
                and all(v.dtype == torch.float32 and v.shape == (len(rows), diagnosis.FEATURES[name])
                        and bool(torch.isfinite(v).all()) for name, v in values.items())):
            return values, {'path': str(prior), 'sha256': entry['sha256'], 'reused_diagnosis_cache': True}
    new_cache = output.parent / ('.' + output.name + '_feature_cache') / (part + '.pt')
    values, receipt = diagnosis.cached_features(new_cache, current_binding, model, data, rows, part, config, device, output)
    return values, {**receipt, 'reused_diagnosis_cache': False}


def summarize(reference, results, output):
    base = reference['C']
    metrics = {name: joint.study.validate_predictions(rows, base, probabilities=False)
               for name, rows in {**reference, **results}.items()}
    subsets = {f'{true} -> {wrong}': [i for i, row in enumerate(base)
               if (row['label'], row['prediction']) == (LABELS.index(true), LABELS.index(wrong))]
               for true, wrong in diagnosis.transition.DIRECTIONS}
    subsets['common_Recon'] = [i for i in subsets['Recon -> WebBased']
                               if reference['A'][i]['prediction'] == reference['B37'][i]['prediction'] == WEB]
    effects = {}
    for name, rows in results.items():
        paired, _ = raw.compare(base, rows)
        dirs = diagnosis.direction_stats(base, rows, subsets)
        effects[name] = {**paired, 'directions': dirs,
                         'mutual_errors': sum(dirs[key]['total_errors'] for key in
                                              ('Recon -> WebBased', 'WebBased -> Recon')),
                         'common_recon_repaired': sum(rows[i]['prediction'] == RECON for i in subsets['common_Recon']),
                         'eligible_other_classes_changed': sum(r['eligible'] and r['label'] not in (RECON, WEB)
                                                               and r['prediction'] != p['prediction']
                                                               for r, p in zip(base, rows)),
                         'delta_macro_f1_pp': 100 * (metrics[name]['macro_f1'] - metrics['C']['macro_f1'])}
    matched = [{**r, 'reference_predictions': {name: rows[i]['prediction'] for name, rows in reference.items()},
                'corrections': {name: {key: rows[i][key] for key in ('prediction', 'delta', 'corrected_margin')}
                                for name, rows in results.items()}} for i, r in enumerate(base)]
    raw.save_predictions(output / 'validation_predictions.jsonl.gz', matched)
    raw.save_predictions(output / 'hard_cases.jsonl.gz',
                         ({**matched[i], 'cohort': name} for name, indices in subsets.items() for i in indices))
    return metrics, effects, subsets


def write_report(output, metrics, effects, best, last, bias, settings, updates):
    lines = ['# Seed42 Recon / WebBased 局部重排序', '',
             f"原 C 第 48 轮逐样本复现。仅原前两名为 Recon 和 WebBased 的样本允许改判。"
             f"两头各训练 {settings['epochs']} 轮、{updates} 次更新，按八类验证 Macro-F1 选模，含第 0 轮。",
             f"常数偏置 {bias['bias']:+.9f} 来自 {bias['fit_samples']} 条训练样本，训练 BCE {bias['train_bce']:.9f}。",
             '', '| 模型 | 轮次 | Accuracy | Macro-F1 | Weighted-F1 | Δ Macro-F1（百分点） | 双向错分 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    epochs = {'C': 48, 'Bias': '—', **{k: v['epoch'] for k, v in best.items()},
              **{k + 'Last': v['epoch'] for k, v in last.items()}}
    for name, epoch in epochs.items():
        m = metrics[name]
        matrix = m['confusion_matrix']
        delta = effects.get(name, {}).get('delta_macro_f1_pp', 0.)
        lines.append(f"| {name} | {epoch} | {m['accuracy']:.6%} | {m['macro_f1']:.6%} | "
                     f"{m['weighted_f1']:.6%} | {delta:+.6f} | "
                     f"{matrix[RECON][WEB]} + {matrix[WEB][RECON]} = {matrix[RECON][WEB] + matrix[WEB][RECON]} |")
    lines += ['', '| 模型 | Recon→WebBased 修复 / 新增 | 反向修复 / 新增 | 38 条共同难例找回 | 其他真实类别改判 |',
              '|---|---:|---:|---:|---:|']
    for name, e in effects.items():
        directions = [e['directions'][key] for key in ('Recon -> WebBased', 'WebBased -> Recon')]
        cells = [f"{d['repaired']} / {d['new_errors_from_original_correct']}" for d in directions]
        lines.append(f"| {name} | " + ' | '.join(cells) + f" | {e['common_recon_repaired']} | {e['eligible_other_classes_changed']} |")
    lines += ['', '| 模型 | WebBased P / R / F1 | Recon P / R / F1 |',
              '|---|---:|---:|']
    for name in epochs:
        values = [' / '.join(f"{metrics[name]['per_class'][label][k]:.6%}"
                             for k in ('precision', 'recall', 'f1-score')) for label in ('WebBased', 'Recon')]
        lines.append(f"| {name} | " + ' | '.join(values) + ' |')
    deltas = {name: 100 * (metrics['Difference']['macro_f1'] - metrics[name]['macro_f1'])
              for name in ('C', 'Bias', 'Fused')}
    lines += ['', f"分支差异头相对 C、常数偏置、融合修正头分别变化 "
              f"{deltas['C']:+.6f}、{deltas['Bias']:+.6f}、{deltas['Fused']:+.6f} 个百分点。",
              f"扩大投入的参考线为相对原 C +{settings['budget_reference_pp']:.2f} 个百分点。",
              '', '本次证据来自 seed42 训练和验证。完整八类指标、混淆矩阵和逐样本分差见归档。', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return deltas


def run(repo, output, graph_root, code_commit):
    started = time.perf_counter()
    settings = raw.read(CONFIG)
    config = {**raw.read(evaluation.CONFIG), **settings}
    config['checkpoints'] = {'C': config['checkpoints']['C']}
    config['metrics_sha256'] = {'C': config['metrics_sha256']['C']}
    source, split = (repo / config[key] for key in ('source_directory', 'split_directory'))
    archive, inputs = evaluation.archive_inputs(source, split, config)
    previous = repo / settings['diagnosis_provenance']
    prior = raw.read(diagnosis.transition.CONFIG)
    reference_paths = {name: repo / prior['comparison_validation_directory'] / f'{name}_predictions.jsonl.gz'
                       for name in prior['comparison_sha256']}
    additional = {path: prior['comparison_sha256'][name] for name, path in reference_paths.items()}
    additional[previous] = settings['diagnosis_provenance_sha256']
    evaluation.verify_inputs(additional)
    inputs.update(additional)
    require(torch.cuda.is_available() and torch.__version__.split('+')[0] == '2.3.1'
            and torch_geometric.__version__ == '2.5.3', 'Expected CUDA, Torch 2.3.1 and PyG 2.5.3')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    data = raw.FrozenGraphs(split, config['split_id'], data_root=graph_root)
    manifest = data.metadata['sample_manifest_sha256']
    inputs[data.directory / data.metadata['sample_manifest']] = manifest
    rows = {'val': evaluation.evaluation_rows(data, 'val', config), 'train': raw.choose_rows(data, 'train')}
    require(len(rows['train']) == prior['training_samples'], 'Wrong training set size')
    provenance = raw.read(previous)
    protocol = raw.read(joint.CONFIG)
    state = evaluation.load_checkpoint_state(source, config, archive, protocol, 'C')
    base = raw.OperatorClassifier('GAT', seed=config['seed'])
    model = joint.JointCrossLayerQueryClassifier(base.flow_encoder, base.packet_encoder, base.fusion,
                                                 'flow', config['seed'])
    model.load_state_dict(state, strict=True)
    require(joint.legacy.frozen_hash(model) == protocol['source_frozen_state_sha256'], 'C frozen state differs')
    model.requires_grad_(False).to(device).eval()
    before = raw.state_digest(model.state_dict())
    binding = provenance['cache_binding']
    require(before == provenance['model_state_sha256'] and binding['checkpoint_sha256'] == config['checkpoints']['C']['sha256']
            and binding['split_id'] == config['split_id'] and binding['manifest_sha256'] == manifest,
            'Archived cache provenance differs from locked C or split')
    # A cache made by the previous diagnosis is valid even when a different GPU
    # fits the new heads. Any re-extraction uses the current environment binding.
    current_binding = {**binding, 'torch': str(torch.__version__), 'pyg': torch_geometric.__version__,
                       'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(device),
                       'source_sha256': {name: raw.sha(raw.ROOT / name) for name in binding['source_sha256']}}
    raw.write(output / 'config.json', {**config, 'partitions_evaluated': ['train', 'val'],
                                       'selection': 'best_validation_macro_f1_including_zero_earliest_tie'})
    features, cache_receipts, rankings = {}, {}, {}
    for part in ('val', 'train'):
        features[part], cache_receipts[part] = load_features(
            repo, output, provenance, binding, current_binding, model, data, rows[part], part, config, device)
        rankings[part] = diagnosis.ranking_records(features[part]['LogProbabilities'], rows[part])
        if part == 'val':
            c = rankings['val']
            gold = joint.legacy.predictions(source / 'C' / 'validation_predictions.jsonl.gz')
            measured = joint.study.validate_predictions(c, rows['val'], probabilities=False)
            receipt = evaluation.replay_receipt(measured, c, archive['C']['selected']['best'], gold)
            raw.write(output / 'validation_replay.json', receipt)
            require(receipt['passed'] and receipt['prediction_ids_sha256'] == settings['prediction_ids_sha256'],
                    'Original C validation replay differs')
    indices = eligible_training_rows(rankings['train'])
    train_counts = Counter(LABELS[rankings['train'][i]['label']] for i in indices)
    require(dict(train_counts) == settings['training_counts']
            and sum(r['eligible'] for r in c) == settings['validation_eligible'], 'Archived eligibility counts differ')
    train, val, normalization = prepare_features(features['train'], features['val'], indices, device)
    del features
    margin = torch.tensor([rankings['train'][i]['recon_minus_webbased'] for i in indices], device=device)
    labels = torch.tensor([float(rankings['train'][i]['label'] == RECON) for i in indices], device=device)
    bias = fit_bias(margin, labels)
    best, last, exports, updates = fit_heads(train, val, margin, labels, c, config, output)
    bias_predictions = rerank(c, torch.full((len(c),), bias['bias'], dtype=torch.float64))
    reference = {'C': c, **{name: joint.legacy.predictions(path) for name, path in reference_paths.items()}}
    metrics, effects, subsets = summarize(reference, {'Bias': bias_predictions, **exports}, output)
    require(tuple(len(subsets[key]) for key in ('Recon -> WebBased', 'WebBased -> Recon', 'common_Recon'))
            == (43, 15, 38), 'Archived hard-case cohorts differ')
    require(raw.state_digest(model.state_dict()) == before and
            all(p.grad is None and not p.requires_grad for p in model.parameters()), 'Frozen C changed')
    evaluation.verify_inputs(inputs)
    for item in cache_receipts.values():
        require(raw.sha(Path(item['path'])) == item['sha256'], 'Frozen feature cache changed')
    torch.save({'best': best, 'last': last, 'bias': bias, 'normalization': normalization,
                'checkpoint_sha256': config['checkpoints']['C']['sha256']}, output / 'rerank_heads.pt')
    deltas = write_report(output, metrics, effects, best, last, bias, settings, updates)
    directions = effects['Difference']['directions']
    decision = {'meets_reference': deltas['C'] >= settings['budget_reference_pp'],
                'beats_both_matched_controls': min(deltas['Bias'], deltas['Fused']) > 0,
                'reduces_recon_to_webbased': directions['Recon -> WebBased']['total_errors'] < 43,
                'webbased_to_recon_change': directions['WebBased -> Recon']['total_errors'] - 15,
                'mutual_errors_change': effects['Difference']['mutual_errors'] - 58}
    raw.write(output / 'metrics.json', {'metrics': metrics, 'effects': effects, 'bias': bias,
              'best_epochs': {name: item['epoch'] for name, item in best.items()},
              'training_counts': train_counts, 'updates_per_head': updates,
              'difference_delta_pp': deltas, 'reference_threshold_pp': settings['budget_reference_pp'], 'decision': decision,
              'wall_seconds': time.perf_counter() - started})
    with (output / 'per_class.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', 'class', 'precision', 'recall', 'f1', 'support'])
        for name, result in metrics.items():
            for label, values in result['per_class'].items():
                writer.writerow([name, label, values['precision'], values['recall'], values['f1-score'], values['support']])
    raw.write(output / 'provenance.json', {'code_commit': code_commit, 'archive_commit': config['archive_commit'],
              'diagnosis_commit': settings['diagnosis_commit'], 'model_state_sha256': before, 'frozen_c_verified': True,
              'input_sha256': {str(path): digest for path, digest in inputs.items()}, 'feature_caches': cache_receipts,
              'execution_binding': current_binding, 'python': platform.python_version(), 'graph_root': str(data.root)})


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
    print(f'Reranking output: {output}', flush=True)
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
