"""Ablate C1 attention and train a matched uniform transition readout."""
import argparse
import csv
import json
from pathlib import Path
import platform
import re

import torch
import torch_geometric

import run_packet_transition_readout as original


raw, joint, evaluation, require, LABELS = (original.raw, original.joint, original.evaluation,
                                          original.require, original.LABELS)
CONFIG = raw.ROOT / 'configs/packet_transition_attention_diagnosis_s42_v1.json'
OUTPUT = 'outputs/packet_transition_attention_diagnosis_s42_v1'
ABLATION = 'C1_equal'


def archived_inputs(repo, settings, config, rows, baseline_selection):
    directory = repo / settings['directory']
    inputs = {directory / name: digest for name, digest in settings['sha256'].items()}
    evaluation.verify_inputs(inputs)
    provenance = raw.read(directory / 'provenance.json')
    require(provenance['code_commit'] == settings['archive_code_commit']
            and provenance['frozen_c_verified']
            and provenance['transition_checkpoint_sha256'] == settings['sha256']['transition_readouts.pt'],
            'Archived transition provenance differs')
    saved = torch.load(directory / 'transition_readouts.pt', map_location='cpu', weights_only=True)
    require(saved['config'] == config and saved['query_sources'] == original.SOURCES
            and all(saved[key] == provenance[key] for key in ('frozen_c_sha256', 'initial_transition_sha256')),
            'Archived transition checkpoint protocol or state identifiers differ')
    exports = joint.legacy.predictions(directory / 'validation_predictions.jsonl.gz')
    references = {arm: original.prediction_records(exports, [row['predictions'][arm] for row in exports])
                  for arm in ('A', 'B', 'C1')}
    metrics = raw.read(directory / 'metrics.json')
    measured = {arm: joint.study.validate_predictions(items, rows, probabilities=False)
                for arm, items in references.items()}
    require(measured == metrics['metrics']
            and measured['A'] == baseline_selection['validation']
            and joint.prediction_digest(references['A']) == config['baseline_prediction_ids_sha256'],
            'Archived validation metrics or baseline IDs differ')
    history = [json.loads(line) for line in (directory / 'epochs.jsonl').read_text(encoding='utf-8').splitlines()]
    updates = settings['expected_updates_per_epoch']
    require([row['epoch'] for row in history] == list(range(config['epochs'] + 1))
            and all(row['updates_per_arm'] == updates and row['samples_per_arm'] == config['training_samples']
                    and re.fullmatch(r'[0-9a-f]{64}', row['order_sha256']) for row in history[1:])
            and metrics['updates_per_arm'] == updates * config['epochs'], 'Invalid archived training budget or order')
    for arm, selection in settings['selections'].items():
        best = saved['best'][arm]
        top = max(history, key=lambda row: row['validation'][arm]['macro_f1'])
        last = saved['last'][arm]
        require(best['epoch'] == selection['epoch'] == top['epoch'] == metrics['selected_epochs'][arm]
                and best['prediction_ids_sha256'] == selection['prediction_ids_sha256']
                == joint.prediction_digest(references[arm])
                and best['validation'] == measured[arm] == top['validation'][arm]
                and history[0]['validation'][arm] == measured['A']
                and last['epoch'] == config['epochs']
                and last['validation'] == history[-1]['validation'][arm] == metrics['last_epoch_metrics'][arm],
                f'Archived {arm} selection or last epoch differs')
    return saved, references, [row['valid_pairs'] for row in exports], history, inputs


class AttentionReplay:
    """Compare two weighting rules on the same frozen features and C1 parameters."""
    def __init__(self, model, head):
        self.model = model
        self.head = head
        self.predictions = {'C1': [], ABLATION: []}
        self.max_mass_error = 0.0
        self.nonuniform_graphs = 0

    @torch.no_grad()
    def __call__(self, features):
        lengths = features.lengths
        flow, flow_residual, flow_weights = original.transition_forward(self.model, self.head, 'C1', features)
        equal, equal_residual, equal_weights = original.transition_forward(self.model, self.head, 'U', features)
        nonempty = lengths > 0
        if len(equal_weights):
            index = torch.repeat_interleave(torch.arange(len(lengths), device=lengths.device), lengths,
                                            output_size=len(equal_weights))
            expected = lengths[index].to(equal_weights.dtype).reciprocal()
            require(torch.allclose(equal_weights, expected, rtol=1e-6, atol=0),
                    'Equal attention must give 1/n')
            for weights in (flow_weights, equal_weights):
                mass = torch.segment_reduce(weights[:, None], 'sum', lengths=lengths, axis=0).flatten()
                error = float((mass - nonempty.to(mass.dtype)).abs().max())
                require(error <= 1e-5 and bool((weights >= 0).all()), 'Invalid per-graph attention mass')
                self.max_mass_error = max(self.max_mass_error, error)
            deviation = (flow_weights - equal_weights).abs()
            max_deviation = torch.segment_reduce(deviation[:, None], 'max', lengths=lengths, axis=0).flatten()
            self.nonuniform_graphs += int((max_deviation > 1e-6).sum())
        pred_flow, pred_equal = flow.argmax(-1), equal.argmax(-1)
        require(bool(torch.isfinite(flow).all() and torch.isfinite(equal).all())
                and not bool(flow_residual[~nonempty].any() or equal_residual[~nonempty].any())
                and torch.equal(flow_residual[lengths <= 1], equal_residual[lengths <= 1])
                and torch.equal(pred_flow[lengths <= 1], pred_equal[lengths <= 1]),
                'Invalid scores or attention effect on a graph with fewer than two valid edges')
        self.predictions['C1'].extend(pred_flow.tolist())
        self.predictions[ABLATION].extend(pred_equal.tolist())


def compare(reference, candidate, lengths, before, after):
    paired, changed = raw.compare(reference, candidate)
    subsets = {f'{true} -> {wrong}': [i for i, item in enumerate(reference)
               if (item['label'], item['prediction']) == (LABELS.index(true), LABELS.index(wrong))]
               for true, wrong in original.diagnosis.DIRECTIONS}
    directions = original.fusion_diagnosis.direction_stats(reference, candidate, subsets)
    for item in directions.values():
        item['net_change'] = item['total_errors'] - item['original_errors']
    strata = {}
    for name, indices in (('0', [i for i, n in enumerate(lengths) if n == 0]),
                          ('1', [i for i, n in enumerate(lengths) if n == 1]),
                          ('2+', [i for i, n in enumerate(lengths) if n >= 2])):
        stats, changes = raw.compare([reference[i] for i in indices], [candidate[i] for i in indices])
        strata[name] = {'samples': len(indices), 'changed_predictions': len(changes), **stats}
    return {**paired, 'changed_predictions': len(changed), 'by_valid_edges': strata, 'directions': directions,
            'delta_macro_f1_pp': 100 * (after['macro_f1'] - before['macro_f1'])}, changed


def comparison_outputs(output, rows, exports, lengths, pairs):
    metrics = {arm: joint.study.validate_predictions(items, rows, probabilities=False)
               for arm, items in exports.items()}
    comparisons = {}
    for after, before in pairs:
        name = f'{after}-{before}'
        comparisons[name], changed = compare(exports[before], exports[after], lengths, metrics[before], metrics[after])
        raw.save_predictions(output / f'{name}_changed_samples.jsonl.gz', changed)
    return metrics, comparisons


def write_predictions(path, exports, lengths, hard_cases=False):
    target = {(LABELS.index('Recon'), LABELS.index('WebBased')), (LABELS.index('WebBased'), LABELS.index('Recon'))}
    raw.save_predictions(path, ({**item, 'valid_pairs': lengths[i],
         'predictions': {arm: records[i]['prediction'] for arm, records in exports.items()}}
         for i, item in enumerate(exports['A'])
         if not hard_cases or (item['label'], item['prediction']) in target))


def write_report(output, metrics, comparisons, epochs, updates=None):
    lines = ['# Seed42 报文转移注意力复核', '',
             'C1_equal 使用第 28 轮 C1 的原权重，在同一流的有效报文对内等权聚合。',
             'U 从原 C 独立训练，边编码器和输出投影沿用归档初态。', '',
             '| 组别 | 轮次 | Accuracy | Macro-F1 | Weighted-F1 | BruteForce F1 |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, item in metrics.items():
        values = [item[key] for key in ('accuracy', 'macro_f1', 'weighted_f1')]
        values.append(item['per_class']['BruteForce']['f1-score'])
        lines.append(f'| {name} | {epochs[name]} | ' + ' | '.join(f'{value:.6%}' for value in values) + ' |')
    lines += ['', '| 比较 | Macro-F1 差值（百分点） | 修复 | 退步 | 净正确 |',
              '| --- | ---: | ---: | ---: | ---: |']
    for name, item in comparisons.items():
        lines.append(f"| {name} | {item['delta_macro_f1_pp']:+.6f} | {item['counts']['repaired']} | "
                     f"{item['counts']['regressed']} | {item['net_correct']:+d} |")
    ablation = comparisons['C1-C1_equal']
    lines += ['', '| 有效报文对数 | 样本 | C1 相对同权重等权修复 | 退步 |',
              '| --- | ---: | ---: | ---: |']
    for key, item in ablation['by_valid_edges'].items():
        lines.append(f"| {key} | {item['samples']} | {item['counts']['repaired']} | {item['counts']['regressed']} |")
    lines += ['', '| 比较 | 错分方向 | 参考错分 | 当前错分 | 净变化 | 从该方向判对 | 从原判对样本新增 |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for name, item in comparisons.items():
        for direction, values in item['directions'].items():
            lines.append(f"| {name} | {direction.replace(' -> ', '→')} | {values['original_errors']} | "
                         f"{values['total_errors']} | {values['net_change']:+d} | {values['repaired']} | "
                         f"{values['new_errors_from_original_correct']} |")
    lines += ['', '| 组别 | 类别 | Precision | Recall | F1 |',
              '| --- | --- | ---: | ---: | ---: |']
    for name, item in metrics.items():
        for label in ('WebBased', 'Recon'):
            values = item['per_class'][label]
            lines.append(f'| {name} | {label} | ' +
                         ' | '.join(f'{values[k]:.6%}' for k in ('precision', 'recall', 'f1-score')) + ' |')
    lines += ['', '同权重回放和独立均匀训练均使用固定验证集评估。']
    if updates is not None:
        lines.append(f'U 完成 {updates:,} 次训练更新，采样顺序逐轮与归档 B/C1 匹配。')
        deltas = [comparisons[key]['delta_macro_f1_pp'] for key in ('C1-C1_equal', 'C1-U', 'C1-B')]
        if min(deltas) > 0:
            lines.append('同权重选权和两项训练对照的 Macro-F1 差值均为正，可结合错分取舍评估是否复核其稳定性。')
        else:
            lines.append('至少一项 Flow 查询对照的 Macro-F1 差值为零或为负，现有证据支持按各对照实测差值描述。')
    lines.append('本轮结果限于 seed42 验证集。')
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run(repo, output, graph_root, code_commit):
    config, settings = original.base_config(), raw.read(CONFIG)
    model, data, partitions, device, baseline_selection, inputs = original.load_source(repo, config, graph_root)
    rows = partitions['val']
    saved, reference, archived_lengths, history, archive_inputs = archived_inputs(
        repo, settings, config, rows, baseline_selection)
    inputs.update(archive_inputs)
    frozen_sha = raw.state_digest(model.state_dict())
    require(frozen_sha == saved['frozen_c_sha256'], 'Frozen C differs from archived training backbone')
    uniform = original.make_heads(config, device, ('U',))
    head = uniform['U']
    initial_sha = raw.state_digest(head.state_dict())
    query_sha = raw.state_digest(head.query.state_dict())
    parameter_count = sum(p.numel() for p in head.parameters() if p.requires_grad)
    require(initial_sha == saved['initial_transition_sha256'] and parameter_count == 2160,
            'Uniform initialization or trainable parameter count differs')
    candidate = original.make_heads(config, device, ('C1',))['C1']
    candidate.load_state_dict(saved['best']['C1']['state'], strict=True)
    candidate.requires_grad_(False).eval()
    candidate_sha = raw.state_digest(candidate.state_dict())
    raw.write(output / 'config.json', {'training_protocol': config, 'archive': settings, 'trainable_parameters': {
              'U': parameter_count, 'B': 3200, 'C1': 3200}, 'uniform_query_bypassed': True,
              'partitions': ['train', 'val']})
    provenance = {'code_commit': code_commit, 'source_archive_commit': settings['archive_commit'],
                  'source_code_commit': settings['archive_code_commit'], 'frozen_c_sha256': frozen_sha,
                  'c1_state_sha256': candidate_sha, 'initial_transition_sha256': initial_sha,
                  'input_sha256': {str(path): digest for path, digest in inputs.items()},
                  'graph_root': str(data.root), 'device': torch.cuda.get_device_name(device),
                  'torch': str(torch.__version__), 'pyg': torch_geometric.__version__,
                  'cuda': torch.version.cuda, 'python': platform.python_version()}
    raw.write(output / 'provenance.json', provenance)

    observer = AttentionReplay(model, candidate)
    val, val_coverage, baseline = original.cache_split(
        model, data, rows, 'val', config, device, output, uniform, observe=observer)
    lengths = val.features.lengths.tolist()
    require(lengths == archived_lengths, 'Valid edge counts differ from archived validation graphs')
    replay = {'A': baseline, **{arm: original.prediction_records(rows, preds)
                                for arm, preds in observer.predictions.items()}}
    receipts = {arm: evaluation.replay_receipt(joint.study.validate_predictions(replay[arm], rows, probabilities=False),
               replay[arm], baseline_selection if arm == 'A' else saved['best'][arm], reference[arm])
               for arm in ('A', 'C1')}
    receipts['attention'] = {'max_graph_mass_error': observer.max_mass_error,
                             'minimum_difference_for_nonuniform': 1e-6,
                             'graphs_with_nonuniform_weights': observer.nonuniform_graphs,
                             'graphs_with_fewer_than_two_edges_changed': 0}
    receipts['U_epoch_zero_exact_logits'] = True
    raw.write(output / 'validation_replay.json', receipts)
    require(all(receipts[arm]['passed'] for arm in ('A', 'C1'))
            and raw.state_digest(model.state_dict()) == frozen_sha
            and raw.state_digest(candidate.state_dict()) == candidate_sha, 'Archived replay or frozen state differs')
    exports = {**replay, 'B': reference['B']}
    pairs = [('C1', ABLATION), (ABLATION, 'A'), ('B', 'A'), ('C1', 'A'), ('C1', 'B')]
    metrics, comparisons = comparison_outputs(output, rows, exports, lengths, pairs)
    epochs = {'A': baseline_selection['epoch'], ABLATION: saved['best']['C1']['epoch'],
              **{arm: item['epoch'] for arm, item in saved['best'].items()}}
    write_predictions(output / 'validation_predictions.jsonl.gz', exports, lengths)
    write_predictions(output / 'baseline_hard_cases.jsonl.gz', exports, lengths, hard_cases=True)
    raw.write(output / 'replay_metrics.json', {'metrics': metrics, 'comparisons': comparisons})
    write_report(output, metrics, comparisons, epochs)
    raw.progress(output, 'attention_replay_passed', samples=len(rows),
                 flow_minus_equal_pp=comparisons['C1-C1_equal']['delta_macro_f1_pp'])

    attention = original.check_attention(original.make_heads(config, device, ('U',))['U'], device, uniform=True)
    preflight = original.probe(model, data, config, device, ('U',))
    raw.write(output / 'preflight.json', {**preflight, 'uniform_attention_checks': attention})
    train, train_coverage, _ = original.cache_split(model, data, partitions['train'], 'train', config, device, output)
    raw.write(output / 'edge_coverage.json', {'train': train_coverage, 'val': val_coverage})
    orders = {row['epoch']: row['order_sha256'] for row in history[1:]}
    best, last, updates = original.fit(model, uniform, train, val, rows, baseline, config, device, output,
                                      order_reference=orders)
    require(updates == settings['expected_updates_per_epoch'] * config['epochs']
            and raw.state_digest(head.query.state_dict()) == query_sha, 'U budget or frozen query differs')
    del train, val
    checkpoint = output / 'uniform_readout.pt'
    torch.save({'best': best, 'last': last, 'config': config, 'query_source': 'uniform',
                'initial_transition_sha256': initial_sha, 'frozen_c_sha256': frozen_sha}, checkpoint)
    loaded = torch.load(checkpoint, map_location='cpu', weights_only=True)
    selected = original.replay_selected(model, loaded, data, rows, config, device, output)
    exports['U'] = selected['U']
    receipt = evaluation.replay_receipt(joint.study.validate_predictions(exports['U'], rows, probabilities=False),
                                         exports['U'], best['U'], None)
    receipts['U_selected_raw'] = receipt
    raw.write(output / 'validation_replay.json', receipts)
    require(receipt['passed'] and selected['A'] == baseline, 'U selected checkpoint differs on original validation graphs')
    require(raw.state_digest(model.state_dict()) == frozen_sha
            and raw.state_digest(candidate.state_dict()) == candidate_sha
            and all(not param.requires_grad and param.grad is None for param in model.parameters()),
            'Frozen C or archived C1 changed')
    evaluation.verify_inputs(inputs)
    raw.write(output / 'provenance.json', {**provenance, 'frozen_models_verified': True,
              'uniform_checkpoint_sha256': raw.sha(checkpoint)})

    pairs += [('U', 'A'), ('C1', 'U'), ('B', 'U')]
    metrics, comparisons = comparison_outputs(output, rows, exports, lengths, pairs)
    epochs['U'] = best['U']['epoch']
    final_epochs = {arm: item['validation'] for arm, item in {**saved['last'], **last}.items()}
    with (output / 'per_class.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['selection', 'model', 'class', 'precision', 'recall', 'f1', 'support'])
        for selection, table in (('best', metrics), ('last', final_epochs)):
            for arm, result in table.items():
                for label, item in result['per_class'].items():
                    writer.writerow([selection, arm, label, item['precision'], item['recall'],
                                     item['f1-score'], item['support']])
    write_predictions(output / 'validation_predictions.jsonl.gz', exports, lengths)
    write_predictions(output / 'baseline_hard_cases.jsonl.gz', exports, lengths, hard_cases=True)
    raw.write(output / 'metrics.json', {'metrics': metrics, 'comparisons': comparisons,
              'selected_epochs': epochs, 'last_epoch_metrics': final_epochs, 'updates_uniform': updates,
              'archived_order_matched_each_epoch': True,
              'decision': {key: comparisons[key]['delta_macro_f1_pp'] > 0
                           for key in ('C1-C1_equal', 'C1-U', 'C1-B')}})
    write_report(output, metrics, comparisons, epochs, updates)
    with (output / 'report.md').open('a', encoding='utf-8', newline='\n') as stream:
        stream.write('\n| 固定第 30 轮 | 验证 Macro-F1 |\n| --- | ---: |\n')
        for arm, item in final_epochs.items():
            stream.write(f"| {arm} | {item['macro_f1']:.6%} |\n")


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
