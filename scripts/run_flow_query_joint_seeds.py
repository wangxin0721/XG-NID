"""Registered43/44 Joint query pairs, seed42 artifact audit, and probability evidence."""
import argparse
import csv
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path

import torch
import torch_geometric
from torch_geometric.data import Batch

import run_flow_query_joint_packet2 as core

raw, require, LABELS = core.raw, core.require, core.LABELS
CONFIG = raw.ROOT/'configs/flow_query_joint_seeds43_44_v1.json'
NAMES = ('Joint-Mean', 'Joint-Flow')
CLASSIFICATION_FIELDS = ('accuracy', 'macro_f1', 'weighted_f1', 'per_class', 'confusion_matrix', 'samples')


def stage_config(previous, study):
    # Carry common architecture/data settings, without advertising the old
    # seed42 frozen-resume stage as part of these fresh43/44 joint trainings.
    stale = {'seed', 'resume_seed42_from_epoch', 'resume_directory', 'additional_seeds_deferred', 'variants'}
    return {**{key: value for key, value in previous.items() if key not in stale}, **study}


def classification_metrics(report):
    """Validate the complete classification result; preserve extra diagnostics."""
    require(all(key in report for key in CLASSIFICATION_FIELDS), 'Missing classification metric fields')
    classification = {key: report[key] for key in CLASSIFICATION_FIELDS}
    require(classification == core.parent.matrix_report(report['confusion_matrix']),
            'Classification metrics disagree with their confusion matrix')
    return classification


def load_source(repo, config, expected_rows=None):
    root = repo/config['source_directory']
    paths = {root/'best.pt': config['source_sha256'],
             root/'validation_predictions.jsonl.gz': config['source_predictions_sha256']}
    for path, digest in paths.items():
        require(raw.sha(path) == digest, 'Mature source hash differs: '+str(path))
    saved = torch.load(root/'best.pt', map_location='cpu', weights_only=False)
    require(saved['epoch'] == config['source_epoch'] == 62
            and saved['validation']['macro_f1'] == config['source_macro_f1'], 'Wrong mature D32 source')
    measured = classification_metrics(saved['validation'])
    gold = core.read_predictions(root/'validation_predictions.jsonl.gz')
    require(validate_predictions(gold, expected_rows, False) == measured, 'Source predictions/metrics differ')
    receipt = {'status': 'passed', 'source_sha256': config['source_sha256'],
               'predictions_sha256': config['source_predictions_sha256'], 'samples': measured['samples'],
               'classification_fields_checked': list(CLASSIFICATION_FIELDS),
               'preserved_diagnostic_fields': sorted(set(saved['validation'])-set(CLASSIFICATION_FIELDS)),
               'original_source_report_unchanged': True}
    return saved, gold, receipt


def validate_predictions(records, expected=None, probabilities=True):
    require(bool(records) and len({r['sample_id'] for r in records}) == len(records), 'Empty/duplicate prediction IDs')
    if expected is not None:
        require([(r['sample_id'], r['label']) for r in records] ==
                [(r['sample_id'], r['label']) for r in expected], 'Prediction IDs/labels/order differ')
    cm = [[0]*8 for _ in LABELS]
    for row in records:
        y, p = row['label'], row['prediction']
        require(type(y) == type(p) == int and y in range(8) and p in range(8), 'Invalid prediction label')
        if probabilities:
            scores, probs = row['log_probabilities'], row['probabilities']
            require(len(scores) == len(probs) == 8 and all(math.isfinite(v) for v in scores+probs)
                    and all(0 <= v <= 1 for v in probs) and abs(sum(probs)-1) < 1e-5
                    and max(range(8), key=lambda i: scores[i]) == p
                    and max(range(8), key=lambda i: probs[i]) == p, 'Invalid class probability vector')
            require(all(math.isclose(math.exp(s), q, rel_tol=1e-9, abs_tol=1e-15) for s, q in zip(scores, probs)),
                    'Probabilities differ from saved log-softmax output')
        cm[y][p] += 1
    return core.parent.matrix_report(cm)


@torch.no_grad()
def probability_replay(model, batches, rows, device, output=None, phase='probability_replay'):
    model.eval()
    records, offset = [], 0
    for step, batch in enumerate(batches, 1):
        batch = batch.to(device)
        expected = rows[offset:offset+batch.num_graphs]
        require(list(zip(batch.sample_id, batch.y.view(-1).tolist())) ==
                [(r['sample_id'], r['label']) for r in expected], 'Raw probability replay order differs')
        scores, gate, beta = model.forward_with_attention(*model.encode(batch))
        require(scores.shape == (batch.num_graphs, 8) and gate.shape == (batch.num_graphs, 32)
                and bool(torch.isfinite(scores).all()) and bool(torch.isfinite(gate).all())
                and bool(torch.isfinite(beta).all()), 'Invalid replay tensor')
        logs = scores.double().cpu().tolist()
        probs = scores.double().exp().cpu().tolist()
        weights = beta.cpu().split(torch.bincount(batch['packet'].batch, minlength=batch.num_graphs).tolist())
        for i, (sid, y, p) in enumerate(zip(batch.sample_id, batch.y.view(-1).tolist(), scores.argmax(-1).tolist())):
            require(abs(float(weights[i].sum())-1) < 1e-5 and bool((weights[i] >= 0).all()), 'Invalid Packet weights')
            records.append({'sample_id': sid, 'label': y, 'prediction': p, 'log_probabilities': logs[i],
                            'probabilities': probs[i], 'flow_weights': gate[i].tolist(), 'packet_attention': weights[i].tolist()})
        offset += batch.num_graphs
        if output is not None and (step == 1 or step % 200 == 0):
            raw.progress(output, phase, samples=offset, total=len(rows))
    require(offset == len(rows), 'Incomplete probability replay')
    return validate_predictions(records, rows), records


def build_models(source_state, seed, device):
    models = {}
    for name, mode in core.JOINT.items():
        base = raw.OperatorClassifier('GAT', seed=42)
        base.load_state_dict(source_state, strict=True)
        models[name] = core.JointQueryPacketClassifier(base.flow_encoder, base.packet_encoder, base.fusion,
                                                      mode, seed).to(device).eval()
    require(len({raw.state_digest(model.state_dict()) for model in models.values()}) == 1, 'Paired initial states differ')
    return models


def epoch_orders(size, batch_size, seed, epochs):
    sampler, hashes = raw.TrainBatches(size, batch_size, seed), []
    for _ in range(epochs):
        list(sampler)
        hashes.append(sampler.order_sha256)
    return hashes


def paired_evidence(mean, flow, folder, seed):
    a, b = validate_predictions(mean), validate_predictions(flow, mean)
    comparison, legacy_changes = raw.compare(mean, flow)
    records, transitions, kinds = [], Counter(), Counter()
    for old, new in zip(mean, flow):
        y, p, q = old['label'], old['prediction'], new['prediction']
        kind = 'both_correct' if p == q == y else 'regressed' if p == y else 'repaired' if q == y else 'both_wrong'
        kinds[kind] += 1
        transitions[(y, p, q)] += 1
        records.append({'seed': seed, 'sample_id': old['sample_id'], 'label': y, 'true_class': LABELS[y],
                        'old_prediction': p, 'new_prediction': q, 'mean_prediction': p, 'flow_prediction': q,
                        'mean_probabilities': old['probabilities'], 'flow_probabilities': new['probabilities'],
                        'mean_log_probabilities': old['log_probabilities'], 'flow_log_probabilities': new['log_probabilities'],
                        'change': 'both_wrong_changed' if kind == 'both_wrong' and p != q else kind})
    require(dict(kinds) == {key: value for key, value in comparison['counts'].items() if value}, 'Pair counts differ')
    raw.save_predictions(folder/'paired_validation.jsonl.gz', records)
    changed = [r for r in records if r['old_prediction'] != r['new_prediction']]
    raw.save_predictions(folder/'changed_samples.jsonl.gz', changed)
    require([{key: r[key] for key in ('sample_id', 'label', 'true_class', 'old_prediction', 'new_prediction', 'change')}
             for r in changed] == legacy_changes, 'Detailed sample changes differ')
    by_class = {}
    for y, label in enumerate(LABELS):
        counts, _ = raw.compare([r for r in mean if r['label'] == y], [r for r in flow if r['label'] == y])
        by_class[label] = {**counts, 'Joint-Mean': a['per_class'][label], 'Joint-Flow': b['per_class'][label]}
    web, recon = LABELS.index('WebBased'), LABELS.index('Recon')
    mutual = ((web, recon), (recon, web))
    relevant_repairs = sum(transitions[(y, wrong, y)] for y, wrong in mutual)
    relevant_regressions = sum(transitions[(y, y, wrong)] for y, wrong in mutual)
    directions = [{'true_class': LABELS[y], 'wrong_prediction': LABELS[p],
                   'mean_errors': a['confusion_matrix'][y][p], 'flow_errors': b['confusion_matrix'][y][p],
                   'error_reduction': a['confusion_matrix'][y][p]-b['confusion_matrix'][y][p]}
                  for y in range(8) for p in range(8) if y != p]
    error_types = []
    for y in range(8):
        for wrong in range(8):
            if y != wrong:
                repaired, regressed = transitions[(y, wrong, y)], transitions[(y, y, wrong)]
                if repaired or regressed:
                    error_types.append({'true_class': LABELS[y], 'wrong_prediction': LABELS[wrong],
                                        'repaired': repaired, 'regressed': regressed})
    with (folder/'prediction_transitions.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['true_class', 'mean_prediction', 'flow_prediction', 'samples'])
        for (y, p, q), count in sorted(transitions.items()):
            writer.writerow([LABELS[y], LABELS[p], LABELS[q], count])
    result = {'seed': seed, 'Joint_Flow_minus_Joint_Mean_pp': 100*(b['macro_f1']-a['macro_f1']),
              'comparison': comparison, 'per_class': by_class, 'confusion_directions': directions, 'error_types': error_types,
              'WebBased_Recon_repaired': relevant_repairs, 'WebBased_Recon_regressed': relevant_regressions,
              'WebBased_Recon_share_of_all_repairs': relevant_repairs/comparison['counts']['repaired']
                  if comparison['counts']['repaired'] else None,
              'prediction_files_sha256': {name: raw.sha(folder/name) for name in (
                  'paired_validation.jsonl.gz', 'changed_samples.jsonl.gz', 'prediction_transitions.csv')}}
    raw.write(folder/'analysis.json', result)
    return result


def checkpoint_check(saved, model, result, history, filename, name, steps):
    epochs = {'epoch_zero.pt': 0, 'best.pt': result['best_epoch'], 'trained_best.pt': result['best_trained_epoch'],
              'last.pt': result['epochs'], 'epoch150.pt': 150, 'best150.pt': result['best150_epoch']}
    epoch = epochs[filename]
    expected = result['epoch_zero'] if epoch == 0 else history[epoch-1]['validation']
    require(saved['epoch'] == epoch and saved['validation'] == expected, 'Seed42 checkpoint epoch/metrics differ: '+name+'/'+filename)
    require(saved['config']['labels'] == LABELS and saved['config']['protocol']['source_sha256'] ==
            '7815875a111d3e94ceefd0e9ef6cb64dedaf6ff80723645f1ec035b376af31ff',
            'Seed42 checkpoint source/label mapping differs')
    variant = core.FROZEN[name] if filename == 'epoch150.pt' and name in core.FROZEN else name
    require(saved['config']['variant'] == variant, 'Seed42 checkpoint variant differs')
    model.load_state_dict(saved['model_state'], strict=True)
    measured_frozen = core.frozen_hash(model) if name in core.JOINT else core.parent.encoder_hash(model)
    require(measured_frozen == result['frozen_state_sha256'], 'Seed42 frozen state differs')
    optimizer = saved['optimizer_state']
    parameters = [p for group in optimizer['param_groups'] for p in group['params']]
    require((not optimizer['state']) if epoch == 0 else set(optimizer['state']) == set(parameters), 'Seed42 Adam state incomplete')
    require(all(int(state['step']) == epoch*steps for state in optimizer['state'].values()), 'Seed42 Adam steps differ')


def audit_seed42(repo, output, data, rows, device, archive, source_state, config, inputs):
    source_root = repo/config['seed42_directory']
    require(raw.read(source_root/'summary.json') == archive, 'Server seed42 summary differs from registered archive')
    require(raw.read(source_root/'status.json')['status'] == 'completed', 'Seed42 output is incomplete')
    inputs[source_root/'summary.json'] = raw.sha(source_root/'summary.json')
    output.mkdir()
    exports, receipt = {}, {}
    for name, result in archive['models'].items():
        original, folder = source_root/name, output/name
        folder.mkdir()
        expected = {**result['checkpoint_sha256'], 'epochs.jsonl': result['ledger_sha256'],
                    'validation_predictions.jsonl.gz': result['predictions_sha256']}
        for filename, digest in expected.items():
            path = original/filename
            require(raw.sha(path) == digest, 'Seed42 artifact hash differs: '+str(path))
            inputs[path] = digest
        history = core.read_history(original/'epochs.jsonl')
        core.parent.check_history(history, result, 200, 211, 13446)
        old = core.read_predictions(original/'validation_predictions.jsonl.gz')
        require(validate_predictions(old, rows['val'], False) == result['best_validation'], 'Seed42 original predictions/metrics differ')
        base = raw.OperatorClassifier('GAT', seed=42)
        base.load_state_dict(source_state, strict=True)
        mode = 'packet_mean' if name.endswith('Mean') else 'flow'
        cls = core.JointQueryPacketClassifier if name in core.JOINT else core.parent.QueryPacketClassifier
        model = cls(base.flow_encoder, base.packet_encoder, base.fusion, mode, 42).eval()
        for filename in result['checkpoint_sha256']:
            saved = torch.load(original/filename, map_location='cpu', weights_only=False)
            checkpoint_check(saved, model, result, history, filename, name, 211)
        selected = torch.load(original/'best.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(selected['model_state'], strict=True)
        before = raw.state_digest(model.state_dict())
        model.to(device)
        batches = core.graph_loader(data, rows['val'], 'val', archive['protocol'])
        measured, replay = probability_replay(model, batches, rows['val'], device, output, name+'_seed42_probability_replay')
        require(measured == result['best_validation'] and core.parent.identities(replay) == core.parent.identities(old)
                and raw.state_digest(model.state_dict()) == before, 'Seed42 independent full raw replay differs')
        raw.save_predictions(folder/'validation_probabilities.jsonl.gz', replay)
        exports[name] = replay
        receipt[name] = {'checkpoint_hashes_and_strict_loads': len(result['checkpoint_sha256']), 'ledger_epochs': len(history),
                         'ledger_sha256': result['ledger_sha256'], 'full_raw_predictions_identical': True,
                         'probabilities_sha256': raw.sha(folder/'validation_probabilities.jsonl.gz')}
        del model, base, saved, selected
    # Reconstruct the old claims from actual per-sample files, rather than from
    # confusion matrices or precomputed counts. Preserve old evidence unchanged.
    old_analysis = core.analyze_four(archive['models'], exports, archive['protocol'], output)
    require(old_analysis == archive['analysis'], 'Seed42 four-arm sample evidence differs from reported summary')
    for key in archive['analysis']['comparisons']:
        path = source_root/(key+'_changed_samples.jsonl.gz')
        expected = core.read_predictions(output/path.name)
        require(core.read_predictions(path) == expected, 'Seed42 changed sample IDs differ: '+key)
        inputs[path] = raw.sha(path)
    evidence = paired_evidence(exports[NAMES[0]], exports[NAMES[1]], output, 42)
    raw.write(output/'audit_receipt.json', {'status': 'passed', 'models': receipt,
              'four_original_changed_sample_files_verified': True, 'new_optimizer_updates': 0,
              'seed42_threshold_reclassified': False})
    return {'models': {name: archive['models'][name] for name in NAMES}, 'analysis': evidence}


def summarize(pairs, config):
    require(set(pairs) == {'42', '43', '44'}, 'All three seeds are required')
    deltas = {seed: pair['analysis']['Joint_Flow_minus_Joint_Mean_pp'] for seed, pair in pairs.items()}
    for seed, pair in pairs.items():
        scores = [pair['models'][name]['best_validation']['macro_f1'] for name in NAMES]
        require(math.isclose(deltas[seed], 100*(scores[1]-scores[0]), rel_tol=0, abs_tol=1e-12), 'Paired delta/metrics differ')
    require(deltas['42'] < config['budget_threshold_pp'], 'Seed42 original failed verdict must remain unchanged')
    per_model = {}
    for name in NAMES:
        values = [100*pairs[str(seed)]['models'][name]['best_validation']['macro_f1'] for seed in config['aggregate_seeds']]
        per_model[name] = {'mean_macro_f1_percent': statistics.mean(values), 'sample_std_pp': statistics.stdev(values)}
    types = {}
    for seed, pair in pairs.items():
        for row in pair['analysis']['error_types']:
            key = row['true_class']+'->'+row['wrong_prediction']
            types.setdefault(key, {})[seed] = {'repaired': row['repaired'], 'regressed': row['regressed']}
    new = [deltas[str(seed)] for seed in config['new_seeds']]
    return {'per_seed_paired_delta_pp': deltas, 'paired_delta_mean_pp': statistics.mean(deltas.values()),
            'paired_delta_sample_std_pp': statistics.stdev(deltas.values()), 'std_ddof': 1, 'n': 3,
            'new_seeds43_44': {'paired_delta_pp': {str(seed): deltas[str(seed)] for seed in config['new_seeds']},
                             'mean_pp': statistics.mean(new), 'sample_std_pp': statistics.stdev(new)},
            'per_model': per_model, 'all_three_positive': all(v > 0 for v in deltas.values()),
            'seed42_delta_pp': deltas['42'], 'seed42_budget_passed': False,
            'original_threshold_pp': config['budget_threshold_pp'], 'exploratory_exception': True,
            'conditional_on_fixed_pretrained_checkpoint': True, 'full_training_multiseed_claimed': False,
            'error_type_recurrence': types, 'statistical_significance_claimed': False, 'automatic_followup': False}


def write_report(output, pairs, aggregate, config):
    lines = ['| Seed | Model | Epoch0 F1 | Best0-150 F1 | Best0-200 F1 | Selected epoch | Trained best F1 | Trained epoch | Last F1 | Updates |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for seed in config['aggregate_seeds']:
        for name, r in pairs[str(seed)]['models'].items():
            lines.append(f"| {seed} | {name} | {100*r['epoch_zero']['macro_f1']:.6f}% | {100*r['best150_validation']['macro_f1']:.6f}% | "
                         f"{100*r['best_validation']['macro_f1']:.6f}% | {r['best_epoch']} | "
                         f"{100*r['trained_best_validation']['macro_f1']:.6f}% | {r['best_trained_epoch']} | "
                         f"{100*r['last_validation']['macro_f1']:.6f}% | {r['optimizer_updates']} |")
    lines += ['', '| Seed | Joint-Flow minus Joint-Mean pp | Joint-Flow minus historical JCL-Multi pp |', '|---|---:|---:|']
    for seed in config['aggregate_seeds']:
        pair = pairs[str(seed)]
        historical = 100*(pair['models']['Joint-Flow']['best_validation']['macro_f1']-config['JCL_Multi_reference_macro_f1'])
        lines.append(f"| {seed} | {pair['analysis']['Joint_Flow_minus_Joint_Mean_pp']:+.6f} | {historical:+.6f} |")
    lines += ['', f"Three-seed paired mean {aggregate['paired_delta_mean_pp']:+.6f} pp; sample SD "
              f"{aggregate['paired_delta_sample_std_pp']:.6f} pp (n=3, ddof=1).",
              f"New43/44 paired mean {aggregate['new_seeds43_44']['mean_pp']:+.6f} pp; sample SD "
              f"{aggregate['new_seeds43_44']['sample_std_pp']:.6f} pp (n=2, ddof=1).",
              f"All three positive: {aggregate['all_three_positive']}.", '',
              'Exploratory exception registered before new training; seed42 remains below0.10pp and is not retroactively passed.',
              'All seeds share the same mature D32 checkpoint. Variability covers query/key initialization and joint-stage order/RNG; this is not a full-pretraining multiseed experiment.',
              'Each arm selects independently over epochs0..200; ties keep the earliest epoch. Epoch95 is not forced. Constant learning rates, no scheduler or early stopping.',
              'The experiment tests the query source under joint training. Frozen43/44 are absent; a multiseed enhancement from unfreezing cannot yet be claimed.',
              'Probability vectors are normalized classifier outputs, not additional calibration. Error transitions give behavioral evidence; attention alone does not establish complementarity.', '',
              '| Model | Three-seed mean Macro-F1 percent | Sample SD pp (n=3, ddof=1) |', '|---|---:|---:|']
    for name, stats in aggregate['per_model'].items():
        lines.append(f"| {name} | {stats['mean_macro_f1_percent']:.6f} | {stats['sample_std_pp']:.6f} |")
    lines += ['', '| Seed | True class | Mean P/R/F1 | Flow P/R/F1 | Repaired | Regressed |', '|---|---|---|---|---:|---:|']
    for seed in config['aggregate_seeds']:
        for label in config['focus_classes']:
            row = pairs[str(seed)]['analysis']['per_class'][label]
            values = [' / '.join(f"{100*row[name][key]:.6f}%" for key in ('precision', 'recall', 'f1-score')) for name in NAMES]
            lines.append(f"| {seed} | {label} | {values[0]} | {values[1]} | {row['counts']['repaired']} | {row['counts']['regressed']} |")
    lines += ['', '| Seed | All repairs | All regressions | WebBased/Recon repairs | WebBased/Recon regressions |',
              '|---|---:|---:|---:|---:|']
    for seed in config['aggregate_seeds']:
        evidence = pairs[str(seed)]['analysis']
        counts = evidence['comparison']['counts']
        lines.append(f"| {seed} | {counts['repaired']} | {counts['regressed']} | "
                     f"{evidence['WebBased_Recon_repaired']} | {evidence['WebBased_Recon_regressed']} |")
    lines += ['', 'Error-type recurrence (true class -> wrong prediction):',
              '| Error type | Seed42 repaired/regressed | Seed43 repaired/regressed | Seed44 repaired/regressed |',
              '|---|---:|---:|---:|']
    for key, recurrence in sorted(aggregate['error_type_recurrence'].items()):
        values = [recurrence.get(str(seed), {'repaired': 0, 'regressed': 0}) for seed in config['aggregate_seeds']]
        lines.append('| '+key+' | '+' | '.join(f"{v['repaired']}/{v['regressed']}" for v in values)+' |')
    (output/'report.md').write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))
    with (output/'per_class.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['seed', 'model', 'class', 'precision', 'recall', 'f1', 'support'])
        for seed, pair in pairs.items():
            for name in NAMES:
                for label, row in pair['models'][name]['best_validation']['per_class'].items():
                    writer.writerow([seed, name, label, row['precision'], row['recall'], row['f1-score'], row['support']])


def prepare(repo, output):
    study = raw.read(CONFIG)
    require(study['new_seeds'] == [43, 44] and study['aggregate_seeds'] == [42, 43, 44]
            and study['variants'] == core.JOINT
            and study['epochs'] == 200 and study['sample_std_ddof'] == 1
            and study['seed42_budget_passed'] is False and study['exploratory_exception_authorized_before_new_training']
            and study['new_frozen_training_runs'] == 0 and not study['automatic_followup']
            and not study['test_evaluated'] and not study['statistical_significance_claimed']
            and study['scheduler'] == 'none' and not study['fixed_epoch95_selection']
            and not study['early_stop_enabled'] and study['include_epoch_zero'] and study['prefix_epochs'] == 150
            and study['tie_policy'] == 'earliest_epoch' and study['fixed_pretrained_start']
            and not study['end_to_end_training_multiseed_claimed'], 'Unregistered exploratory protocol')
    base_path = CONFIG.parent/study['base_protocol']
    archive_path = raw.ROOT/'docs'/study['seed42_archive']
    require(raw.sha(base_path) == study['base_protocol_sha256'] and raw.sha(archive_path) == study['seed42_archive_sha256'],
            'Registered protocol/archive bytes changed')
    archive = raw.read(archive_path)
    delta = archive['analysis']['Joint_Flow_minus_Joint_Mean_pp']
    require(0 < delta < study['budget_threshold_pp'] == .1 and archive['status'] == 'completed', 'Original seed42 verdict differs')
    for key in ('epochs', 'batch_size', 'eval_batch_size', 'workers', 'encoder_lr', 'head_lr', 'pool_lr', 'weight_decay'):
        require(study[key] == archive['protocol'][key], 'Joint training setting changed: '+key)
    for name, digest in archive['binding']['sources'].items():
        require(raw.sha(raw.ROOT/name) == digest, 'Seed42 implementation changed: '+name)
    config = stage_config(archive['protocol'], study)
    data = raw.FrozenGraphs(repo/config['split_directory'], config['split_id'])
    rows = {part: raw.choose_rows(data, part) for part in ('train', 'val')}
    require((len(rows['train']), len(rows['val'])) == (107621, 13446), 'Wrong fixed split sizes')
    root = repo/config['source_directory']
    inputs = {root/'best.pt': config['source_sha256'], root/'validation_predictions.jsonl.gz': config['source_predictions_sha256']}
    saved, gold, source_check = load_source(repo, config, rows['val'])
    sources = set(archive['binding']['sources']) | {
        'scripts/run_flow_query_joint_seeds.py', 'scripts/test_flow_query_joint_seeds.py',
        'scripts/launch_flow_query_joint_seeds.sh', 'configs/flow_query_joint_seeds43_44_v1.json',
        'docs/flow_query_joint_seeds43_44_v1.md', archive_path.relative_to(raw.ROOT).as_posix()}
    binding = {'protocol_sha256': raw.sha(CONFIG), 'seed42_archive_sha256': raw.sha(archive_path),
               'sources': {name: raw.sha(raw.ROOT/name) for name in sorted(sources)},
               'manifest_sha256': data.metadata['sample_manifest_sha256'], 'split_json_sha256': raw.sha(data.directory/'split.json'),
               'torch': torch.__version__,
               'pyg': torch_geometric.__version__, 'cuda_device': torch.cuda.get_device_name(0),
               'source_classification_check': source_check}
    raw.write(output/'config.json', config)
    raw.write(output/'binding.json', binding)
    raw.write(output/'source_metrics_check.json', source_check)
    return config, archive, data, rows, saved, gold, inputs, binding


def run(repo, output, train):
    require(torch.cuda.is_available() and torch.__version__.split('+')[0] == '2.3.1'
            and torch_geometric.__version__ == '2.5.3', 'Requires CUDA/Torch2.3.1/PyG2.5.3')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    config, archive, data, rows, source, gold, inputs, binding = prepare(repo, output)
    source_metrics = classification_metrics(source['validation'])
    pairs = {'42': audit_seed42(repo, output/'seed42_audit', data, rows, device, archive,
                              source['model_state'], config, inputs)}
    # Only16 real training graphs form this temporary probe snapshot. All formal
    # joint updates below recompute representations from the original graphs.
    probe_rows = data.training_candidates(per_class=config['raw_probe_per_class'])
    batch = Batch.from_data_list([data.load(row) for row in probe_rows]).to(device)
    probe_source = build_models(source['model_state'], 42, device)['Joint-Mean']
    with torch.no_grad():
        flow, nodes, index, size = probe_source.encode(batch)
        lengths = torch.bincount(index, minlength=size).cpu()
        cache = {'ids': list(batch.sample_id), 'labels': batch.y.cpu(), 'flow64': flow.cpu(), 'packet64': nodes.cpu(),
                 'ptr': torch.cat((torch.zeros(1, dtype=torch.long), lengths.cumsum(0)))}
    del probe_source
    models, preflights = {}, {}
    for seed in config['new_seeds']:
        c = {**config, 'seed': seed}
        models[seed] = build_models(source['model_state'], seed, device)
        preflights[str(seed)] = {}
        for name, model in models[seed].items():
            require(core.parent.encoder_hash(model) == config['encoder_sha256'], 'Initial encoder differs from mature source')
            measured, predictions = probability_replay(model, core.graph_loader(data, rows['val'], 'val', c),
                                                       rows['val'], device, output, f'seed{seed}_{name}_epoch0')
            require(measured == source_metrics and core.parent.identities(predictions) == core.parent.identities(gold),
                    'New paired epoch0 predictions differ from D32')
            preflights[str(seed)][name] = {'initial_state_sha256': raw.state_digest(model.state_dict()),
                'epoch_zero': measured, 'probe': core.joint_probe(model, cache, list(range(size)), batch, c, device)}
    raw.write(output/'preflight.json', {'status': 'passed', 'pairs': preflights,
              'formal_training_started': False, 'seed42_independent_audit_passed': True, 'test_graph_loads': 0})
    raw.write(output/'input_receipts.json', {str(path.relative_to(repo)): digest for path, digest in inputs.items()})
    if not train:
        return
    for seed in config['new_seeds']:
        c, folder = {**config, 'seed': seed}, output/('seed'+str(seed))
        folder.mkdir()
        orders = epoch_orders(len(rows['train']), c['batch_size'], seed, c['epochs'])
        results, exports = {}, {}
        for name, model in models[seed].items():
            result, selected = core.fit_joint(name, model, data, rows, device, c, binding, folder, orders)
            require(result['optimizer_updates'] == 42200 and result['epochs'] == 200
                    and result['trainable_parameters'] == 25384, 'New joint budget/scope differs')
            measured, replay = probability_replay(model, core.graph_loader(data, rows['val'], 'val', c),
                                                  rows['val'], device, output, f'seed{seed}_{name}_selected_probabilities')
            require(measured == result['best_validation'] and core.parent.identities(replay) == core.parent.identities(selected),
                    'Selected probability replay differs')
            path = folder/name/'validation_probabilities.jsonl.gz'
            raw.save_predictions(path, replay)
            result['probability_predictions_sha256'] = raw.sha(path)
            result['probability_replay_passed'] = True
            raw.write(folder/name/'metrics.json', result)
            results[name], exports[name] = result, replay
        require(results[NAMES[0]]['orders'] == results[NAMES[1]]['orders'] == orders
                and results[NAMES[0]]['initial_state_sha256'] == results[NAMES[1]]['initial_state_sha256']
                and results[NAMES[0]]['training_start_rng_sha256'] == results[NAMES[1]]['training_start_rng_sha256'],
                'New seed pair is not matched')
        pairs[str(seed)] = {'models': results, 'analysis': paired_evidence(exports[NAMES[0]], exports[NAMES[1]], folder, seed)}
        raw.write(folder/'summary.json', {'status': 'completed', 'protocol': c, **pairs[str(seed)]})
    aggregate = summarize(pairs, config)
    raw.write(output/'analysis.json', aggregate)
    write_report(output, pairs, aggregate, config)
    raw.write(output/'input_receipts.json', {str(path.relative_to(repo)): digest for path, digest in inputs.items()})
    for path, digest in inputs.items():
        require(raw.sha(path) == digest, 'Read-only original input changed: '+str(path))
    for name, digest in binding['sources'].items():
        require(raw.sha(raw.ROOT/name) == digest, 'Registered source changed during study: '+name)
    new_updates = sum(m['optimizer_updates'] for seed in config['new_seeds'] for m in pairs[str(seed)]['models'].values())
    require(new_updates == 168800, 'Four-run update budget differs')
    raw.write(output/'summary.json', {'status': 'completed', 'protocol': config, 'binding': binding,
              'pairs': pairs, 'analysis': aggregate, 'new_training_runs': 4, 'new_optimizer_updates': new_updates,
              'seed42_retrained': False, 'test_evaluated': False, 'original_inputs_unchanged': True,
              'automatic_followup': False})


def archive_failed_source_check(repo):
    """Preserve the known setup-only failure; never move an audit/training run."""
    root = (repo/'outputs').resolve()
    folder = root/'flow_query_joint_seeds43_44_v1'
    require(folder.is_dir() and not folder.is_symlink() and folder.resolve().parent == root,
            'Retry requires the named failed output directory')
    status = raw.read(folder/'status.json')
    require(status.get('status') == 'failed' and status.get('error') == 'Source predictions/metrics differ',
            'Retry accepts only the original source-report comparison failure')
    require({path.name for path in folder.iterdir()} == {'status.json'},
            'Refuse to move a run containing setup/audit/training artifacts')
    backup = root/(folder.name+'_source_check_failed_'+str(time.time_ns()))
    require(backup.resolve().parent == root and not backup.exists(), 'Invalid source-check backup path')
    folder.rename(backup)
    print(json.dumps({'preserved_failed_source_check': str(backup), 'new_formal_updates_in_failed_run': 0}), flush=True)
    return backup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--check-source-only', action='store_true', help='Read-only CPU source SHA/prediction/classification check')
    parser.add_argument('--retry-source-check-failure', action='store_true', help='Preserve the known setup-only failed directory')
    args = parser.parse_args()
    repo = args.repo.resolve()
    if args.check_source_only:
        require(not args.train and not args.retry_source_check_failure, 'Source-only check cannot start/retry training')
        study = raw.read(CONFIG)
        archive_path = raw.ROOT/'docs'/study['seed42_archive']
        require(raw.sha(archive_path) == study['seed42_archive_sha256'], 'Registered seed42 archive changed')
        archive = raw.read(archive_path)
        _, _, receipt = load_source(repo, stage_config(archive['protocol'], study))
        print(json.dumps(receipt), flush=True)
        return
    if args.retry_source_check_failure:
        require(args.train, 'Retry flag requires --train')
        if (repo/'outputs/flow_query_joint_seeds43_44_v1').exists():
            archive_failed_source_check(repo)
    name = 'flow_query_joint_seeds43_44_v1' if args.train else 'flow_query_joint_seeds43_44_preflight_v1'
    output = repo/'outputs'/name
    output.mkdir()
    raw.write(output/'status.json', {'status': 'running'})
    try:
        run(repo, output, args.train)
    except BaseException as error:
        raw.write(output/'status.json', {'status': 'failed', 'error': str(error)})
        raise
    raw.write(output/'status.json', {'status': 'completed', 'exit_code': 0, 'training_started': args.train})
    print(output, flush=True)


if __name__ == '__main__':
    main()
