"""Compare global and per-flow D32 weights on each frozen final B backbone."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from statistics import mean, stdev
from subprocess import check_output
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.loader import DataLoader

import run_independent_packet_transition as independent
import run_packet_transition_attention_diagnosis as attention
import run_packet_transition_readout as transition


raw, joint, evaluation, require = independent.raw, independent.joint, independent.evaluation, independent.require
SEEDS = (42, 43, 44)
ARMS = ('static', 'dynamic')
PAIRS = (('dynamic', 'static'),)
OUTPUT = 'final_d32_gate_ablation_v1'
SCHEDULE = {'epochs': 30, 'batch_size': 512, 'learning_rate': 0.001, 'training_seed': 42}
SELECTION = 'max_validation_macro_f1_including_epoch_zero_earliest_tie'


class FusionArm(nn.Module):
    def __init__(self, fusion, fixed_input):
        super().__init__()
        self.gate = deepcopy(fusion.gate).requires_grad_(True)
        self.classifier = deepcopy(fusion.shared_head).requires_grad_(True)
        self.register_buffer('fixed_input', fixed_input)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, features):
        flow, packet = features[:, :32], features[:, 32:64]
        gate_input = features if self.fixed_input is None else self.fixed_input.expand_as(features)
        weights = self.gate(gate_input).sigmoid()
        scores = self.classifier(weights * flow + (1 - weights) * packet)
        return F.log_softmax(scores, dim=-1), weights


@torch.no_grad()
def collect(model, readout, batches, rows, device, output, phase):
    features = torch.empty((len(rows), 69), dtype=torch.float32)
    lengths, original = [], []
    captured = []
    hook = model.fusion.gate.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    try:
        for step, batch in enumerate(batches, 1):
            start, end = len(lengths), len(lengths) + batch.num_graphs
            require(list(zip(batch.sample_id, batch.y.view(-1).tolist())) ==
                    [(r['sample_id'], r['label']) for r in rows[start:end]], f'{phase} sample IDs or labels differ')
            encoded, _ = transition.encode(model, batch, device)
            scores, _, _ = transition.transition_forward(model, readout, 'B', encoded)
            require(len(captured) == 1 and captured[0].shape == (batch.num_graphs, 69)
                    and bool(torch.isfinite(captured[0]).all() and torch.isfinite(scores).all()),
                    f'{phase} invalid frozen D32 inputs')
            features[start:end] = captured.pop().cpu()
            lengths.extend(encoded.lengths.tolist())
            original.extend(scores.argmax(-1).tolist())
            if step == 1 or step % 200 == 0 or end == len(rows):
                raw.progress(output, phase, samples=end, total=len(rows))
    finally:
        hook.remove()
    require(len(original) == len(rows), f'{phase} extraction is incomplete')
    return SimpleNamespace(features=features, lengths=lengths,
                           original=transition.prediction_records(rows, original))


def make_arms(fusion, reference):
    return {name: FusionArm(fusion, reference if name == 'static' else None) for name in ARMS}


@torch.no_grad()
def evaluate_arms(arms, values, rows, batch_size):
    predictions, weights = {}, {}
    for name, arm in arms.items():
        scores = [arm(batch) for batch in values.split(batch_size)]
        predictions[name] = transition.prediction_records(
            rows, torch.cat([score.argmax(-1) for score, _ in scores]).tolist())
        weights[name] = torch.cat([weight for _, weight in scores])
        require(bool(torch.isfinite(weights[name]).all() and
                     torch.stack([torch.isfinite(score).all() for score, _ in scores]).all()),
                f'{name} nonfinite validation output')
    # Batched GEMM kernels can differ slightly across batch sizes.
    require(torch.allclose(weights['static'], weights['static'][:1].expand_as(weights['static']),
                           rtol=1e-6, atol=1e-7), 'Static gate varies across flows')
    metrics = {name: joint.study.validate_predictions(items, rows, probabilities=False)
               for name, items in predictions.items()}
    variation = {name: float(weight.std(dim=0, unbiased=False).mean()) for name, weight in weights.items()}
    return predictions, metrics, variation


def fit(arms, train, labels, val, rows, eval_batch, output, schedule):
    sampler = raw.TrainBatches(len(labels), schedule['batch_size'], schedule['training_seed'])
    best, last = {}, {}
    optimizers = {name: torch.optim.Adam(arm.parameters(), lr=schedule['learning_rate'])
                  for name, arm in arms.items()}
    with (output / 'epochs.jsonl').open('x', encoding='utf-8') as ledger:
        for epoch in range(schedule['epochs'] + 1):
            updates, samples = 0, 0
            losses = {name: train.new_zeros(()) for name in ARMS}
            if epoch:
                for arm in arms.values():
                    arm.train()
                for indices in sampler:
                    batch, targets = train[indices], labels[indices]
                    for name, arm in arms.items():
                        optimizer = optimizers[name]
                        optimizer.zero_grad(set_to_none=True)
                        loss = F.nll_loss(arm(batch)[0], targets)
                        loss.backward()
                        optimizer.step()
                        losses[name] += loss.detach() * len(indices)
                    updates += 1
                    samples += len(indices)
                require(updates == len(sampler) and samples == len(labels)
                        and all(bool(torch.isfinite(value)) for value in losses.values()), 'Incomplete training epoch')
            for arm in arms.values():
                arm.eval()
            predictions, metrics, variation = evaluate_arms(arms, val, rows, eval_batch)
            if not epoch:
                require(predictions['static'] == predictions['dynamic'] and variation == dict.fromkeys(ARMS, 0.0),
                        'Paired initialization differs')
            for name, arm in arms.items():
                improved = not epoch or metrics[name]['macro_f1'] > best[name]['validation']['macro_f1']
                if improved or epoch == schedule['epochs']:
                    item = {'epoch': epoch, 'validation': metrics[name],
                            'prediction_ids_sha256': joint.prediction_digest(predictions[name]),
                            'state': {key: value.detach().cpu().clone() for key, value in arm.state_dict().items()}}
                    if improved:
                        best[name] = item
                    if epoch == schedule['epochs']:
                        last[name] = item
            ledger.write(json.dumps({'epoch': epoch, 'updates_per_arm': updates, 'samples_per_arm': samples,
                                     'order_sha256': sampler.order_sha256, 'validation': metrics,
                                     'weight_variation': variation, 'train_nll': {
                                         name: float(loss / samples) for name, loss in losses.items()
                                     } if samples else None}) + '\n')
            ledger.flush()
            raw.progress(output, 'fit_fusion', epoch=epoch, total_epochs=schedule['epochs'],
                         macro_f1={name: item['macro_f1'] for name, item in metrics.items()})
    return best, last


def source(repo, seed, graph_root, output):
    if seed == 42:
        config = transition.base_config()
        model, data, rows, device, selected, paths = transition.load_source(repo, config, graph_root)
        saved, references, _, _, archive_paths = attention.archived_inputs(
            repo, raw.read(attention.CONFIG), config, rows['val'], selected)
        require(independent.model_state(model) == saved['frozen_c_sha256'], 'Archived seed42 C state differs')
        head = transition.make_heads(config, device, ('B',))['B']
        head.load_state_dict(saved['best']['B']['state'], strict=True)
        paths.update(archive_paths)
        source_prediction_digest = joint.prediction_digest(references['B'])
    else:
        device = independent.d32.baseline.runtime(independent.core)
        config, data, rows = independent.load_data(repo, graph_root)
        readout_seed = raw.read(independent.CONFIG)['readout_seed']
        model, head, _, paths = independent.load_locked_pair(repo, seed, data, rows, config, device, output,
                                                             readout_seed)
        receipt = raw.read(output / f'seed{seed}_validation_replay.json')['B']
        source_prediction_digest = receipt['prediction_ids_sha256'] if receipt['exact_raw_match'] else None
    split = data.directory / 'split.json'
    manifest = data.directory / data.metadata['sample_manifest']
    paths.update({split: config['split_json_sha256'], manifest: data.metadata['sample_manifest_sha256']})
    head.requires_grad_(False).eval()
    binding = {'seed': seed, 'split_id': config['split_id'], 'manifest_sha256': paths[manifest],
               'frozen_c_sha256': independent.model_state(model), 'frozen_readout_sha256': independent.model_state(head)}
    return model, head, data, rows, device, config, binding, paths, source_prediction_digest


def validation_cache(model, head, data, rows, config, device, output, digest):
    batches = joint.legacy.loader(data, rows, 'val', config)
    val = collect(model, head, batches, rows, device, output, 'extract_val')
    require(digest is None or joint.prediction_digest(val.original) == digest,
            'Original final B predictions differ from the validation archive')
    return val


def selected_predictions(arms, saved, cache, rows, config, device):
    for name, arm in arms.items():
        arm.load_state_dict(saved[name]['state'], strict=True)
        arm.eval()
    exports, metrics, variation = evaluate_arms(arms, cache.features.to(device), rows, config['eval_batch_size'])
    require(all(joint.prediction_digest(exports[name]) == saved[name]['prediction_ids_sha256']
                and metrics[name] == saved[name]['validation'] for name in ARMS), 'Selected validation replay differs')
    return exports, variation


def check_frozen(model, head, binding, paths):
    require(independent.model_state(model) == binding['frozen_c_sha256']
            and independent.model_state(head) == binding['frozen_readout_sha256']
            and all(not module.training for parent in (model, head) for module in parent.modules())
            and all(not p.requires_grad and p.grad is None for parent in (model, head) for p in parent.parameters()),
            'Frozen C or B readout changed')
    evaluation.verify_inputs(paths)


def save_partition(output, part, rows, cache, exports):
    exports = {'A': cache.original, **exports}
    metrics, comparisons = attention.comparison_outputs(output, rows, exports, cache.lengths, PAIRS)
    attention.write_predictions(output / f'{part}_predictions.jsonl.gz', exports, cache.lengths)
    return {'original_B': metrics['A'], **{name: metrics[name] for name in ARMS},
            'comparison': comparisons['dynamic-static']}


def train_seed(repo, seed, graph_root, output, code_commit):
    model, head, data, rows, device, config, binding, paths, digest = source(repo, seed, graph_root, output)
    require(len(rows['train']) == 107621 and len(rows['val']) == 13446, 'Split counts differ')
    raw.write(output / 'provenance.json', {'code_commit': code_commit, 'runner_sha256': raw.sha(Path(__file__)),
              'binding': binding, 'schedule': SCHEDULE,
              'selection': SELECTION, 'graph_root': str(data.root), 'torch': str(torch.__version__),
              'device': torch.cuda.get_device_name(device),
              'input_sha256': {str(path.relative_to(repo)): sha for path, sha in paths.items()}})
    val = validation_cache(model, head, data, rows['val'], config, device, output, digest)
    batches = joint.legacy.loader(data, rows['train'], 'train', config)
    train = collect(model, head, batches, rows['train'], device, output, 'extract_train').features.to(device)
    # Both arms receive the same projected vectors as values. Only the conditioning input differs.
    reference = train.double().mean(dim=0, keepdim=True).float()
    arms = make_arms(model.fusion, reference)
    require(sum(p.numel() for p in arms['static'].parameters())
            == sum(p.numel() for p in arms['dynamic'].parameters()) == 2984, 'Fusion parameter count differs')
    labels = torch.tensor([row['label'] for row in rows['train']], dtype=torch.long, device=device)
    best, last = fit(arms, train, labels, val.features.to(device), rows['val'], config['eval_batch_size'],
                     output, SCHEDULE)
    del train, labels
    checkpoint = output / 'fusion_arms.pt'
    torch.save({'binding': binding, 'schedule': SCHEDULE, 'reference': reference.cpu(),
                'best': best, 'last': last}, checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    replay = validation_cache(model, head, data, rows['val'], config, device, output, digest)
    exports, variation = selected_predictions(arms, saved['best'], replay, rows['val'], config, device)
    check_frozen(model, head, binding, paths)
    result = save_partition(output, 'validation', rows['val'], replay, exports)
    raw.write(output / 'selection.json', {'binding': binding, 'schedule': SCHEDULE, 'checkpoint_sha256': raw.sha(checkpoint),
              'epochs': {name: item['epoch'] for name, item in best.items()}, 'updates_per_arm': 6330,
              'validation': result, 'weight_variation': variation, 'test_evaluated': False})
    return result


def test_all(repo, graph_root, output):
    locked = {}
    for seed in SEEDS:
        folder = output.parent / f'seed{seed}'
        require(raw.read(folder / 'status.json')['status'] == 'completed', f'Seed{seed} validation run is incomplete')
        selection = raw.read(folder / 'selection.json')
        checkpoint = folder / 'fusion_arms.pt'
        require(raw.sha(checkpoint) == selection['checkpoint_sha256'] and selection['test_evaluated'] is False,
                f'Seed{seed} validation checkpoint changed')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        require(saved['binding'] == selection['binding'] and saved['schedule'] == selection['schedule'] == SCHEDULE
                and selection['updates_per_arm'] == 6330
                and all(saved['best'][name]['epoch'] == selection['epochs'][name]
                        and saved['best'][name]['validation'] == selection['validation'][name] for name in ARMS),
                f'Seed{seed} selection differs')
        locked[seed] = (saved, selection)
    require(len({saved['binding']['frozen_c_sha256'] for saved, _ in locked.values()}) == len(SEEDS),
            'Expected three independent C main models')
    results = {}
    for seed, (saved, selection) in locked.items():
        folder = output / f'seed{seed}'
        folder.mkdir()
        model, head, data, rows, device, config, binding, paths, digest = source(repo, seed, graph_root, folder)
        require(binding == saved['binding'], f'Seed{seed} frozen source changed')
        arms = make_arms(model.fusion, saved['reference'].to(device))
        val = validation_cache(model, head, data, rows['val'], config, device, folder, digest)
        selected_predictions(arms, saved['best'], val, rows['val'], config, device)
        config['partition_sizes']['test'] = 13448
        test_rows = evaluation.evaluation_rows(data, 'test', config)
        batches = DataLoader(evaluation.TestGraphs(data, test_rows), batch_size=config['eval_batch_size'],
                             shuffle=False, num_workers=config['workers'],
                             generator=torch.Generator().manual_seed(242))
        cache = collect(model, head, batches, test_rows, device, folder, 'extract_test')
        exports, _, variation = evaluate_arms(arms, cache.features.to(device), test_rows, config['eval_batch_size'])
        check_frozen(model, head, binding, paths)
        results[seed] = {'epochs': selection['epochs'], 'validation': selection['validation'],
                         'test': save_partition(folder, 'test', test_rows, cache, exports),
                         'test_weight_variation': variation}
        raw.write(folder / 'metrics.json', results[seed])
    write_report(output, results)


def write_report(output, results):
    lines = ['# 最终双分支模型的逐维门控消融', '',
             '两组共享每个种子的冻结主干、报文读出、投影和训练预算。静态组的门控输入为训练集的平均特征。',
             'Macro-F1 差值以百分点表示。', '',
             '| Seed | Static 验证 | Dynamic 验证 | Static 测试 | Dynamic 测试 | 测试差值 | '
             'WebBased F1 差值 | Recon→WebBased | WebBased→Recon | 修复／退步 |',
             '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |']
    differences = []
    for seed, result in results.items():
        val, test = result['validation'], result['test']
        comparison = test['comparison']
        differences.append(comparison['delta_macro_f1_pp'])
        rw, wr = (comparison['directions'][name] for name in ('Recon -> WebBased', 'WebBased -> Recon'))
        web = 100 * (test['dynamic']['per_class']['WebBased']['f1-score']
                     - test['static']['per_class']['WebBased']['f1-score'])
        lines.append(f"| {seed} | {val['static']['macro_f1']:.6%} | {val['dynamic']['macro_f1']:.6%} | "
                     f"{test['static']['macro_f1']:.6%} | {test['dynamic']['macro_f1']:.6%} | "
                     f"{differences[-1]:+.6f} | {web:+.6f} | {rw['original_errors']} → {rw['total_errors']} | "
                     f"{wr['original_errors']} → {wr['total_errors']} | {comparison['counts']['repaired']} / "
                     f"{comparison['counts']['regressed']} |")
    lines += ['', f'测试 Macro-F1 配对平均差为 {mean(differences):+.6f} 个百分点，'
                   f'样本标准差为 {stdev(differences):.6f} 个百分点。']
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    raw.write(output / 'summary.json', {'results': results, 'mean_test_delta_pp': mean(differences),
              'sd_test_delta_pp': stdev(differences)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output', type=Path, default=Path('outputs') / OUTPUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--seed', type=int, choices=SEEDS)
    mode.add_argument('--test', action='store_true')
    args = parser.parse_args()
    repo = args.repo.resolve()
    root = (repo / args.output).resolve()
    require(root.is_relative_to((repo / 'outputs').resolve()) and root != (repo / 'outputs').resolve(),
            'Use a dedicated directory below outputs')
    torch.set_num_threads(2)
    if args.test:
        with independent.run_status(root / 'test'):
            test_all(repo, args.data_root, root / 'test')
    else:
        commit = check_output(['git', 'rev-parse', 'HEAD'], cwd=raw.ROOT, text=True).strip()
        output = root / f'seed{args.seed}'
        with independent.run_status(output, args.seed):
            train_seed(repo, args.seed, args.data_root, output, commit)
    print(json.dumps({'status': 'completed', 'output': str(root)}), flush=True)


if __name__ == '__main__':
    main()
