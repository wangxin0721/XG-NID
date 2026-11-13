"""Compare matched frozen fusion readouts and audit C's two-class ranking."""
import argparse
from collections import Counter
import copy
import csv
import hashlib
import json
from pathlib import Path
import platform
import re
import time

import torch
from torch import nn
import torch_geometric

import run_packet_transition_diagnosis as transition


evaluation, joint, raw, require, LABELS = (
    transition.evaluation, transition.joint, transition.raw, transition.require, transition.LABELS)
CONFIG = raw.ROOT / 'configs/fusion_decision_diagnosis_s42_v1.json'
OUTPUT = 'outputs/fusion_decision_diagnosis_s42_v1'
RECON, WEB = LABELS.index('Recon'), LABELS.index('WebBased')
READOUTS = ('D32', 'Equal32', 'Concat32')
FEATURES = {'Flow': 64, 'PacketQuery': 64, 'D32Flow': 32, 'D32Packet': 32,
            'D32Fused': 32, 'Alpha': 32, 'LogProbabilities': len(LABELS)}


@torch.no_grad()
def extract(model, data, rows, part, config, device, output):
    values = {name: torch.empty((len(rows), width)) for name, width in FEATURES.items()}
    for step, indices, batch, expected in transition.aligned_batches(data, rows, part, config):
        features, _, _ = transition.frozen_query_features(model, batch.to(device))
        for name, tensor in features.items():
            require(tensor.shape == (len(expected), FEATURES[name]), f'Invalid shape for {name}')
            values[name][indices] = tensor.cpu()
        if step == 1 or step % 200 == 0 or indices.stop == len(rows):
            raw.progress(output, f'extract_{part}', samples=indices.stop, total=len(rows))
    return values


def cached_features(path, binding, model, data, rows, part, config, device, output):
    ids = [r['sample_id'] for r in rows]
    labels = [r['label'] for r in rows]
    metadata = {**binding, 'partition': part}
    reused = False
    if path.is_file():
        cache = torch.load(path, map_location='cpu', weights_only=True)
        reused = cache['binding'] == metadata and cache['sample_ids'] == ids and cache['labels'] == labels
        if reused:
            values = cache['features']
            reused = raw.state_digest(values) == cache['feature_sha256']
    if not reused:
        values = extract(model, data, rows, part, config, device, output)
    require(set(values) == set(FEATURES) and all(values[name].dtype == torch.float32 and
            values[name].shape == (len(rows), width) and bool(torch.isfinite(values[name]).all())
            for name, width in FEATURES.items()), f'Invalid frozen features for {part}')
    if not reused:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        torch.save({'binding': metadata, 'sample_ids': ids, 'labels': labels, 'features': values,
                    'feature_sha256': raw.state_digest(values)}, temporary)
        temporary.replace(path)
    return values, {'path': str(path), 'reused': reused, 'sha256': raw.sha(path)}


def fit_pca(train, dimensions):
    require(len(train) >= 2 and 0 < dimensions <= train.size(1), 'Invalid PCA fit dimensions')
    train = train.double()
    mean = train.mean(dim=0)
    centered = train - mean
    eigenvalues, vectors = torch.linalg.eigh(centered.T @ centered / (len(train) - 1))
    basis = vectors[:, -dimensions:].flip(dims=(1,))
    pivot = basis.abs().argmax(dim=0, keepdim=True)
    basis = basis * basis.gather(0, pivot).sign()
    explained = eigenvalues.clamp_min(0)
    total = float(explained.sum())
    return {'mean': mean, 'basis': basis, 'explained_variance_fraction':
            float(explained[-dimensions:].sum()) / total if total else 0.0}


def readouts(train, val, dimensions):
    pca = {name: fit_pca(train[name], dimensions) for name in ('D32Flow', 'D32Packet')}
    outputs = []
    for features in (train, val):
        flow, packet = (features[name] for name in pca)
        concat = torch.cat([(features[name].double() - spec['mean']) @ spec['basis']
                            for name, spec in pca.items()], dim=-1).float()
        outputs.append({'D32': features['D32Fused'], 'Equal32': 0.5 * flow + 0.5 * packet, 'Concat32': concat})
    return outputs, pca


def matched_heads(seed):
    raw.seed_all(seed)
    linear = nn.Linear(32, 8)
    nonlinear = nn.Sequential(nn.Linear(32, 32), nn.LeakyReLU(), nn.Linear(32, 8))
    return {f'{name}_{kind}': copy.deepcopy(template) for kind, template in
            (('Linear', linear), ('Nonlinear', nonlinear)) for name in READOUTS}


def ranking_records(scores, rows):
    # Preserve argmax's lower-index tie rule for first place and use the same
    # deterministic ordering for second place.
    top2 = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :2].tolist()
    margins = (scores[:, RECON] - scores[:, WEB]).tolist()
    return [{'sample_id': row['sample_id'], 'label': row['label'], 'prediction': pair[0], 'top_two': pair,
             'recon_minus_webbased': margin, 'eligible': set(pair) == {RECON, WEB}}
            for row, pair, margin in zip(rows, top2, margins)]


def ranking_summary(records):
    pairs = [r for r in records if r['eligible']]
    return {'samples': len(records), 'eligible': len(pairs),
            'eligible_correct': dict(Counter(LABELS[r['label']] for r in pairs if r['label'] == r['prediction'])),
            'eligible_repairable': dict(Counter(LABELS[r['label']] for r in pairs
                                                if r['label'] in (RECON, WEB) and r['label'] != r['prediction'])),
            'eligible_other_classes': dict(Counter(LABELS[r['label']] for r in pairs if r['label'] not in (RECON, WEB)))}


def direction_stats(base, records, subsets):
    results = {}
    for true, wrong in transition.DIRECTIONS:
        y, p = LABELS.index(true), LABELS.index(wrong)
        key = f'{true} -> {wrong}'
        targeted = [records[i]['prediction'] for i in subsets[key]]
        results[key] = {'original_errors': len(targeted), 'repaired': targeted.count(y),
                        'changed_to_other_wrong_class': sum(pred not in (y, p) for pred in targeted),
                        'still_same_error': targeted.count(p),
                        'new_errors_from_original_correct': sum(b['label'] == b['prediction'] == y and r['prediction'] == p
                                                                 for b, r in zip(base, records)),
                        'new_errors_from_other_wrong_class': sum(b['label'] == y and b['prediction'] not in (y, p)
                                                                  and r['prediction'] == p for b, r in zip(base, records)),
                        'total_errors': sum(r['label'] == y and r['prediction'] == p for r in records)}
    return results


def summarize(reference, exports, ranking, output):
    val = reference['C']
    oracle = [{**r, 'prediction': r['label']} if r['eligible'] and r['label'] in (RECON, WEB) else r
              for r in ranking['val']]
    combined = {**reference, **exports, 'Oracle': oracle}
    metrics = {name: joint.study.validate_predictions(records, val, probabilities=False)
               for name, records in combined.items()}
    subsets = {f'{true} -> {wrong}': [i for i, row in enumerate(val) if
               (row['label'], row['prediction']) == (LABELS.index(true), LABELS.index(wrong))]
               for true, wrong in transition.DIRECTIONS}
    subsets['common_Recon'] = [i for i in subsets['Recon -> WebBased']
                               if reference['A'][i]['prediction'] == reference['B37'][i]['prediction'] == WEB]
    effects = {}
    for name, records in {**exports, 'Oracle': oracle}.items():
        changes, _ = raw.compare(val, records)
        effects[name] = {**changes, 'directions': direction_stats(val, records, subsets),
                         'common_recon_repaired': sum(records[i]['prediction'] == RECON for i in subsets['common_Recon']),
                         'delta_macro_f1_pp': 100 * (metrics[name]['macro_f1'] - metrics['C']['macro_f1'])}
    paired = [{**row, 'predictions': {name: records[i]['prediction'] for name, records in combined.items()}}
              for i, row in enumerate(ranking['val'])]
    raw.save_predictions(output / 'validation_predictions.jsonl.gz', paired)
    raw.save_predictions(output / 'hard_cases.jsonl.gz',
                         ({**paired[i], 'cohort': group} for group, indices in subsets.items() for i in indices))
    raw.save_predictions(output / 'train_ranking.jsonl.gz', ranking['train'])
    coverage = {part: ranking_summary(rows) for part, rows in ranking.items()}
    coverage['val']['hard_cases_eligible'] = {name: sum(ranking['val'][i]['eligible'] for i in indices)
                                             for name, indices in subsets.items()}
    return metrics, effects, coverage, subsets


def write_report(output, config, metrics, effects, coverage, pca, updates, seconds):
    lines = ['# Seed42 匹配融合与分类头诊断', '',
             f"原 C 第 48 轮通过完整回放。六个探针共用 C 的 PacketQuery 读出与冻结投影，"
             f"各训练 {config['epochs']} 轮和 {updates} 次更新，比较固定末轮。",
             '原 C 分类头来自历史 200 轮合并训练的第 48 轮，另作基线。', '',
             '| 表示与分类头 | 参数量 | Accuracy | Macro-F1 | 相对 C（百分点） | 修复 / 退步 | Recon→WebBased / WebBased→Recon |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name in ('C', *effects):
        m, e = metrics[name], effects.get(name)
        counts = '—' if e is None else f"{e['counts']['repaired']} / {e['counts']['regressed']}"
        params = '—' if name in ('C', 'Oracle') else '264' if name.endswith('_Linear') else '1,320'
        cm = m['confusion_matrix']
        lines.append(f"| {name} | {params} | {m['accuracy']:.6%} | {m['macro_f1']:.6%} | "
                     f"{e['delta_macro_f1_pp'] if e else 0:+.6f} | {counts} | {cm[RECON][WEB]} / {cm[WEB][RECON]} |")
    lines += ['', '| 表示与分类头 | 原 Recon→WebBased 判对 / 转其他错类 / 原判对新增 | 原 WebBased→Recon 判对 / 转其他错类 / 原判对新增 | 38 条共同难例判对 |',
              '|---|---:|---:|---:|']
    for name, item in effects.items():
        columns = [' / '.join(str(item['directions'][key][field]) for field in
                              ('repaired', 'changed_to_other_wrong_class', 'new_errors_from_original_correct'))
                   for key in ('Recon -> WebBased', 'WebBased -> Recon')]
        lines.append(f"| {name} | " + ' | '.join(columns) + f" | {item['common_recon_repaired']} |")
    lines += ['', '| 模型 | WebBased 精确率 | WebBased 召回 | Recon 召回 | Benign→Spoofing / Spoofing→Benign |',
              '|---|---:|---:|---:|---:|']
    for name in ('C', *effects):
        m = metrics[name]
        w, r = m['per_class']['WebBased'], m['per_class']['Recon']
        cm = m['confusion_matrix']
        lines.append(f"| {name} | {w['precision']:.6%} | {w['recall']:.6%} | {r['recall']:.6%} | "
                     f"{cm[0][2]} / {cm[2][0]} |")
    lines += ['', '## Recon 与 WebBased 同占前两名', '',
              '| 划分 | 覆盖 | Recon 原判对 | WebBased 原判对 | Recon 原错分 | WebBased 原错分 | 其他真实类别 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for part, item in coverage.items():
        ok, fix = item['eligible_correct'], item['eligible_repairable']
        lines.append(f"| {part} | {item['eligible']} / {item['samples']} | {ok.get('Recon', 0)} | "
                     f"{ok.get('WebBased', 0)} | {fix.get('Recon', 0)} | {fix.get('WebBased', 0)} | "
                     f"{sum(item['eligible_other_classes'].values())} |")
    hard = coverage['val']['hard_cases_eligible']
    lines += ['', f"验证集覆盖原 43 条 Recon→WebBased 中的 {hard['Recon -> WebBased']} 条、原 15 条反向错分中的 "
              f"{hard['WebBased -> Recon']} 条，以及 38 条共同 Recon 难例中的 {hard['common_Recon']} 条。"
              'Oracle 使用真标签仅修正覆盖范围内的两类错分。',
              '', 'D32 与 Equal32 的差值用于比较同一对投影上的加权方式。D32 的两种新头用于比较分类头结构。'
              'Concat32 的结果包含分别降维的影响。']
    lines += ['', '| 匹配比较 | Δ Macro-F1（百分点） | Δ Recon→WebBased | Δ WebBased→Recon |',
              '|---|---:|---:|---:|']
    for before, after in (('Equal32_Linear', 'D32_Linear'), ('Equal32_Nonlinear', 'D32_Nonlinear'),
                          ('D32_Linear', 'D32_Nonlinear')):
        old, new = metrics[before], metrics[after]
        a, b = old['confusion_matrix'], new['confusion_matrix']
        lines.append(f"| {after} − {before} | {100 * (new['macro_f1'] - old['macro_f1']):+.6f} | "
                     f"{b[RECON][WEB] - a[RECON][WEB]:+d} | {b[WEB][RECON] - a[WEB][RECON]:+d} |")
    lines.append('两路训练集 PCA 保留方差比例分别为 ' +
                 '、'.join(f"{spec['explained_variance_fraction']:.4%}" for spec in pca.values()) + '。')
    lines += ['', f'耗时 {seconds:.1f} 秒。完整八类指标、各错分方向和覆盖细项见 `metrics.json`。', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def run(repo, output, graph_root, code_commit):
    started = time.perf_counter()
    config = {**raw.read(evaluation.CONFIG), **raw.read(CONFIG)}
    prior = raw.read(transition.CONFIG)
    config['checkpoints'] = {'C': config['checkpoints']['C']}
    config['metrics_sha256'] = {'C': config['metrics_sha256']['C']}
    source, split = (repo / config[key] for key in ('source_directory', 'split_directory'))
    archive, inputs = evaluation.archive_inputs(source, split, config)
    reference_paths = {name: repo / prior['comparison_validation_directory'] / f'{name}_predictions.jsonl.gz'
                       for name in prior['comparison_sha256']}
    reference_inputs = {path: prior['comparison_sha256'][name] for name, path in reference_paths.items()}
    evaluation.verify_inputs(reference_inputs)
    inputs.update(reference_inputs)
    require(torch.cuda.is_available() and torch.__version__.split('+')[0] == '2.3.1'
            and torch_geometric.__version__ == '2.5.3', 'Expected CUDA, Torch 2.3.1 and PyG 2.5.3')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    data = raw.FrozenGraphs(split, config['split_id'], data_root=graph_root)
    manifest_hash = data.metadata['sample_manifest_sha256']
    inputs[data.directory / data.metadata['sample_manifest']] = manifest_hash
    rows = {'val': evaluation.evaluation_rows(data, 'val', config), 'train': raw.choose_rows(data, 'train')}
    require(len(rows['train']) == prior['training_samples'], 'Wrong training set size')
    protocol = raw.read(joint.CONFIG)
    state = evaluation.load_checkpoint_state(source, config, archive, protocol, 'C')
    base = raw.OperatorClassifier('GAT', seed=config['seed'])
    model = joint.JointCrossLayerQueryClassifier(base.flow_encoder, base.packet_encoder, base.fusion,
                                                 'flow', config['seed'])
    model.load_state_dict(state, strict=True)
    require(joint.legacy.frozen_hash(model) == protocol['source_frozen_state_sha256'], 'Archived C frozen state differs')
    model.requires_grad_(False).to(device).eval()
    before = raw.state_digest(model.state_dict())
    binding = {'schema': 'xgnid.fusion-decision-cache.v1', 'checkpoint_sha256': config['checkpoints']['C']['sha256'],
               'split_id': config['split_id'], 'manifest_sha256': manifest_hash, 'torch': torch.__version__,
               'pyg': torch_geometric.__version__, 'cuda': torch.version.cuda,
               'device': torch.cuda.get_device_name(device), 'eval_batch_size': config['eval_batch_size'],
               'tf32': False,
               'source_sha256': {name: raw.sha(raw.ROOT / name) for name in (
                   'scripts/run_fusion_decision_diagnosis.py', 'scripts/run_packet_transition_diagnosis.py',
                   'scripts/replay_flow_query_selective.py', 'scripts/grouped_graphs.py',
                   'scripts/audit_graph_split.py', 'src/xgnid/joint_cross_layer_query.py',
                   'src/xgnid/joint_cross_layer.py', 'src/xgnid/cross_layer.py', 'src/xgnid/flow_query_pooling.py',
                   'src/xgnid/packet_pooling.py', 'src/xgnid/fusion_dimension.py', 'src/xgnid/model.py')}}
    raw.write(output / 'config.json', {**config, 'partitions_evaluated': ['train', 'val'],
                                       'selection': 'fixed_last_epoch', 'features': FEATURES})
    cache_key = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    cache_dir = output.parent / '.fusion_decision_feature_cache' / cache_key
    val, val_cache = cached_features(cache_dir / 'val.pt', binding, model, data, rows['val'],
                                     'val', config, device, output)
    ranking = {'val': ranking_records(val['LogProbabilities'], rows['val'])}
    c = ranking['val']
    gold = joint.legacy.predictions(source / 'C' / 'validation_predictions.jsonl.gz')
    measured = joint.study.validate_predictions(c, rows['val'], probabilities=False)
    receipt = evaluation.replay_receipt(measured, c, archive['C']['selected']['best'], gold)
    raw.write(output / 'validation_replay.json', receipt)
    require(receipt['passed'], 'Frozen features do not reproduce archived C')
    reference = {'C': c, **{name: joint.legacy.predictions(path) for name, path in reference_paths.items()}}
    train, train_cache = cached_features(cache_dir / 'train.pt', binding, model, data, rows['train'],
                                         'train', config, device, output)
    ranking['train'] = ranking_records(train['LogProbabilities'], rows['train'])
    require(raw.state_digest(model.state_dict()) == before, 'Frozen C changed during diagnosis')
    raw.write(output / 'provenance.json', {'code_commit': code_commit, 'archive_commit': config['archive_commit'],
              'comparison_archive_commit': prior['comparison_archive_commit'], 'graph_root': str(data.root),
              'input_sha256': {str(path): digest for path, digest in inputs.items()},
              'model_state_sha256': before, 'python': platform.python_version(), 'cache_binding': binding,
              'feature_caches': {'train': train_cache, 'val': val_cache}})
    (train_views, val_views), pca = readouts(train, val, config['pca_dimensions_per_branch'])
    del train, val
    heads = matched_heads(config['seed'])
    train_heads = {name: train_views[name.split('_')[0]] for name in heads}
    val_heads = {name: val_views[name.split('_')[0]] for name in heads}
    exports, checkpoints, updates = transition.fit_probes(
        heads, train_heads, val_heads, rows['train'], rows['val'], config, device, output)
    metrics, effects, coverage, subsets = summarize(reference, exports, ranking, output)
    require(tuple(len(subsets[key]) for key in ('Recon -> WebBased', 'WebBased -> Recon', 'common_Recon'))
            == (43, 15, 38), 'Archived hard-case cohorts differ')
    torch.save({'epoch': config['epochs'], 'seed': config['seed'], 'pca': pca, 'probes': checkpoints,
                'input_binding': binding}, output / 'matched_probes.pt')
    raw.write(output / 'metrics.json', {'metrics': metrics, 'effects': effects, 'ranking_coverage': coverage,
                                       'updates_per_probe': updates, 'wall_seconds': time.perf_counter() - started})
    with (output / 'per_class.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', 'class', 'precision', 'recall', 'f1', 'support'])
        for name, result in metrics.items():
            for label, item in result['per_class'].items():
                writer.writerow([name, label, item['precision'], item['recall'], item['f1-score'], item['support']])
    evaluation.verify_inputs(inputs)
    write_report(output, config, metrics, effects, coverage, pca, updates, time.perf_counter() - started)


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
