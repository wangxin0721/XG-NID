"""Verify synced seed42 A/C/B artifacts and paired predictions, using stdlib only.

No torch import, checkpoint deserialization, graph loading or test inference.
Outputs only aggregates; private sample identities remain in local outputs.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
import sys

from e3_gate_study import identity, matrix_metrics, read, require, sha, write
from fusion_acb import VARIANTS, validate_probe
from diagnose_gate_candidate import class_metrics, finish_groups, group_name, groups

LABELS = ('Benign', 'WebBased', 'Spoofing', 'Recon', 'Mirai', 'Dos', 'DDos', 'BruteForce')


def complete(path, mode):
    r = read(path)
    require((r.get('status'), r.get('exit_code'), r.get('mode')) == ('completed', 0, mode),
            'Incomplete A/C/B artifact: ' + str(path))
    return r


def predictions(path):
    result = {}
    matrices = {k: [[0]*8 for _ in range(8)] for k in ('fusion', 'flow', 'packet')}
    grouped, by_class = groups(), {c: groups() for c in LABELS}
    flow_weights = []
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            r = json.loads(line)
            if 'predictions' in r:
                p = r['predictions']
            else:
                aliases = {'fusion': 'dynamic', 'flow': 'flow_only', 'packet': 'packet_only'}
                p = {}
                for k in matrices:
                    values = r['probabilities'][k if k in r['probabilities'] else aliases[k]]
                    p[k] = max(range(8), key=lambda i: values[i])
            label, sample = r['label'], r['sample_id']
            require(sample not in result and label in range(8) and all(p[k] in range(8) for k in matrices), 'Duplicate or invalid row')
            g = group_name(p['flow'] == label, p['packet'] == label)
            require(r.get('group', g) == g, 'Branch group mismatch')
            weights = r['final_weights']
            require(len(weights) == 2 and all(math.isfinite(v) and 0 <= v <= 1 for v in weights)
                    and abs(sum(weights)-1) < 1e-5, 'Invalid fusion weights')
            flow_weights.append(weights[0])
            for k in matrices:
                matrices[k][label][p[k]] += 1
            for t in (grouped[g], by_class[LABELS[label]][g]):
                t['samples'] += 1
                t['fusion_correct'] += int(p['fusion'] == label)
                t['fusion_wrong'] += int(p['fusion'] != label)
            result[sample] = {'label': label, 'node_hash': r['node_hash'], 'predictions': p, 'group': g}
    require(len(result) == 13446, 'Incomplete validation cache')
    return result, {'metrics': {k: matrix_metrics(v) for k, v in matrices.items()},
                    'per_class': {k: class_metrics(v, LABELS) for k, v in matrices.items()},
                    'groups': finish_groups(grouped), 'groups_by_class': {k: finish_groups(v) for k, v in by_class.items()},
                    'ordered_validation_identity_sha256': identity([(s, r['label']) for s, r in result.items()]),
                    'flow_weight_min': min(flow_weights), 'flow_weight_max': max(flow_weights),
                    'flow_weight_mean': sum(flow_weights)/len(flow_weights)}


def paired(old, new):
    require(set(old) == set(new), 'Paired sample sets differ')
    counts = {'repaired': 0, 'regressed': 0, 'both_correct': 0, 'both_wrong': 0}
    classes = {c: dict(counts) for c in LABELS}
    rescue = {'samples': 0, 'retained_correct': 0, 'lost_correct': 0}
    rescue_classes = {c: dict(rescue) for c in LABELS}
    for sample, a in old.items():
        b = new[sample]
        require((a['label'], a['node_hash']) == (b['label'], b['node_hash']), 'Paired labels or graph identity differ')
        label = a['label']
        ac, bc = a['predictions']['fusion'] == label, b['predictions']['fusion'] == label
        k = 'both_correct' if ac and bc else 'regressed' if ac else 'repaired' if bc else 'both_wrong'
        counts[k] += 1
        classes[LABELS[label]][k] += 1
        if a['group'] == 'both_wrong' and ac:
            for t in (rescue, rescue_classes[LABELS[label]]):
                t['samples'] += 1
                t['retained_correct' if bc else 'lost_correct'] += 1
    return {'total': counts, 'by_class': classes, 'original_fusion_rescue_subset': rescue,
            'original_fusion_rescue_subset_by_class': rescue_classes}


def compare_metrics(rebuilt, expected):
    require(rebuilt['confusion_matrix'] == expected['confusion_matrix'], 'Prediction matrix differs')
    for k in ('macro_f1', 'accuracy', 'weighted_f1'):
        require(abs(rebuilt[k]-expected[k]) < 1e-10, 'Recomputed score differs: ' + k)


def analyze(repo):
    root = repo / 'outputs/fusion_acb_b512_s42_v1'
    complete(root/'status.json', 'queue')
    queue = complete(root/'summary.json', 'queue')
    binding = queue['binding']
    require(queue['test_evaluated'] is False, 'Screening evaluated test')
    for filename, digest in binding['sources'].items():
        require(sha(repo/'scripts'/filename) == digest, 'Bound source changed: ' + filename)
    for filename, field in [('fusion_acb_execution_20261026.json','config_sha256'),
                            ('fusion_acb_study_20261023.json','design_sha256'), ('e3_gate_study.json','reference_config_sha256')]:
        require(sha(repo/'configs'/filename) == binding[field], 'Bound study config differs')
    probe = read(root/'probe/summary.json')
    validate_probe(probe, read(root/'probe/status.json'), binding)
    require(probe['hardware']['physical_index'] == 1, 'This run was assigned physical GPU1')
    hashes, notes, models, data, orders = {}, [], {}, {}, {}
    def checked(path, expected=None):
        digest = sha(path)
        require(expected is None or digest == expected, 'File bytes differ: ' + str(path))
        hashes[path.relative_to(repo).as_posix()] = digest
        return digest
    def historical_copy(path, expected):
        actual = checked(path)
        if actual != expected:
            notes.append({'path': path.relative_to(repo).as_posix(), 'expected_sha256': expected,
                          'actual_sha256': actual, 'note': 'Historical local copy differs from the completed server artifact; not used as a verified final checkpoint/log.'})
        return actual == expected
    for v in VARIANTS:
        folder = root/v
        complete(folder/'status.json', 'screen')
        summary = complete(folder/'summary.json', 'screen')
        metrics, cfg, diag = [read(folder/f) for f in ('metrics.json', 'config.json', 'diagnostics.json')]
        require(summary['binding'] == cfg['binding'] == binding and summary['model'] == queue['models'][v] == metrics,
                'Run documents disagree')
        require(metrics['variant'] == cfg['variant'] == v and metrics['test_evaluated'] is cfg['test_evaluated'] is diag['test_evaluated'] is False,
                'Wrong variant or test boundary')
        require(cfg['lambda'] == .05 and cfg['batch_size'] == 512 and cfg['lr'] == .001
                and (cfg['seed'], cfg['epochs'], cfg['patience'], cfg['workers']) == (42,100,12,2), 'Training controls differ')
        require(cfg['initialization'] == metrics['initialization'] == probe['variants'][v]['initialization'], 'Initialization differs')
        checked(folder/'config.json', metrics['config_sha256'])
        checked(folder/'diagnostics.json', metrics['diagnostics_sha256'])
        checked(root/'probe/summary.json', cfg['probe_sha256'])
        for f, digest in metrics['checkpoint_sha256'].items():
            checked(folder/f, digest)  # bytes only; never torch.load
        path = folder/'validation_predictions.jsonl.gz'
        checked(path, diag['predictions_sha256'])
        rows, rebuilt = predictions(path)
        for k, score in rebuilt['metrics'].items():
            compare_metrics(score, diag['metrics'][k])
        compare_metrics(rebuilt['metrics']['fusion'], metrics['best_validation'])
        for k in ('per_class', 'groups', 'groups_by_class', 'ordered_validation_identity_sha256'):
            require(rebuilt[k] == diag[k], 'Rebuilt diagnostic differs: ' + k)
        if v == 'A_equal':
            require(rebuilt['flow_weight_min'] == rebuilt['flow_weight_max'] == .5, 'A weights vary')
        elif v == 'C_global':
            require(rebuilt['flow_weight_min'] == rebuilt['flow_weight_max'], 'C weights vary by sample')
        epochs = [json.loads(line) for line in (folder/'epochs.jsonl').read_text(encoding='utf-8').splitlines()]
        require([r['epoch'] for r in epochs] == list(range(1,metrics['stopped_epoch']+1)), 'Epoch log incomplete')
        require(all(r['train_samples'] == 107621 and r['optimizer_updates'] == 211 for r in epochs), 'Epoch exposure differs')
        require(metrics['optimizer_updates'] == 211*len(epochs) and metrics['training_samples_seen'] == 107621*len(epochs), 'Total exposure mismatch')
        chosen = max(epochs, key=lambda r: r['val_macro_f1'])
        require(chosen['epoch'] == metrics['best_epoch'] and abs(chosen['val_macro_f1']-metrics['best_validation']['macro_f1']) < 1e-10,
                'Strict best validation selection differs')
        require(metrics['stopped_epoch'] == 100 or metrics['stopped_epoch']-metrics['best_epoch'] == 12, 'Stopping rule differs')
        orders[v] = [r['train_order_sha256'] for r in epochs]
        data[v] = rows
        models[v] = {'best_epoch': metrics['best_epoch'], 'stopped_epoch': metrics['stopped_epoch'],
                     'parameters': metrics['parameters'], 'wall_seconds': metrics['wall_seconds'],
                     'selected_forward_seconds': metrics['best_validation']['forward_seconds'],
                     'selected_weight_summary': metrics['best_validation']['gate_weights'], **rebuilt}
        checked(folder/'epochs.jsonl')
        checked(folder/'metrics.json')
    for i in range(max(map(len, orders.values()))):
        require(len({o[i] for o in orders.values() if i < len(o)}) == 1, 'A/C/B epoch orders differ')
    references = {}
    config = read(repo/'configs/fusion_acb_execution_20261026.json')
    for v, paths in config['completed_candidates'].items():
        for key, field in [('screen','screen_sha256'), ('diagnostic','diagnostic_sha256'), ('cache_analysis','cache_analysis_sha256')]:
            checked(repo/paths[key], binding['candidate_evidence'][v][field])
        folder = (repo/paths['screen']).parent/v
        checkpoint_matches = historical_copy(folder/'best.pt', binding['candidate_evidence'][v]['best_sha256'])
        epoch_log_matches = historical_copy(folder/'epochs.jsonl', binding['candidate_evidence'][v]['epochs_sha256'])
        status = read(folder.parent/'status.json')
        if (status.get('status'),status.get('exit_code')) != ('completed',0):
            notes.append({'path': (folder.parent/'status.json').relative_to(repo).as_posix(),
                          'local_status': status, 'note': 'Historical status copy is stale. Completed summary and validation cache match their receipts; mismatched historical checkpoint/log are not accepted as final. No source artifact modified.'})
        diag = read(repo/paths['diagnostic'])
        checked((repo/paths['diagnostic']).parent/'validation_cache.jsonl.gz', diag['cache_sha256'])
        reference_rows, rebuilt = predictions((repo/paths['diagnostic']).parent/'validation_cache.jsonl.gz')
        compare_metrics(rebuilt['metrics']['fusion'], diag['metrics']['fusion'])
        references[v] = rebuilt
        references[v]['historical_local_checkpoint_matches_completed_receipt'] = checkpoint_matches
        references[v]['historical_local_epoch_log_matches_completed_receipt'] = epoch_log_matches
        data[v] = reference_rows
        if v == 'E3a_equal_init':
            z = read(folder/'metrics.json')
            require(z == read(repo/paths['screen'])['model'], 'Z metrics disagree with bound completed summary')
            require(z['initialization']['variant_initial_state_sha256'] ==
                    probe['variants']['B_dynamic']['initialization']['full_initial_state_sha256'], 'B/Z initial state differs')
            z_orders = [json.loads(line)['train_order_sha256'] for line in (folder/'epochs.jsonl').read_text(encoding='utf-8').splitlines()]
            n = min(len(z_orders),len(orders['B_dynamic']))
            require(z_orders[:n] == orders['B_dynamic'][:n], 'B/Z overlapping epoch orders differ')
            references[v]['locally_verified_overlapping_epochs_with_B'] = n
            references[v]['server_reported_overlapping_epochs_with_B'] = queue['B_lambda_zero_reference']['overlapping_epochs']
    original = read(repo/config['original_diagnostic'])
    checked(repo/config['original_diagnostic'], binding['original_diagnostic']['report_sha256'])
    for key, rel in [('E0','outputs/e0_batch_full_b512_s42_shared_v1/E0/metrics.json'),
                     ('E2','outputs/e2_batch_full_b512_s42_shared_v1/E2/metrics.json')]:
        m = read(repo/rel)
        references[key] = {'metrics': {'fusion': matrix_metrics(m['best_validation']['confusion_matrix'])}}
        compare_metrics(references[key]['metrics']['fusion'],m['best_validation'])
        checked(repo/rel)
        if key == 'E2':
            path = (repo/config['original_diagnostic']).parent/'E2/validation_cache.jsonl.gz'
            checked(path,original['models']['E2']['cache_sha256'])
            data[key], rebuilt = predictions(path)
            compare_metrics(rebuilt['metrics']['fusion'],m['best_validation'])
    pairs = {}
    for old,new in [('A_equal','B_dynamic'),('C_global','B_dynamic'),('E3a_equal_init','B_dynamic'),
                    ('E2','A_equal'),('E2','C_global'),('E2','B_dynamic')]:
        pairs[old+' -> '+new] = paired(data[old],data[new])
    return {'schema': 'xgnid.fusion-acb-results-analysis.v1', 'date': '2026-10-27',
            'partition': 'val', 'seed': 42, 'samples': 13446, 'models': models, 'references': references,
            'paired_outcomes': pairs, 'checks': {'completed_acb_artifacts': True, 'probe_binding_and_gradients': True,
            'checkpoint_file_hashes_only_no_deserialization': True, 'prediction_matrices_classes_and_groups_recomputed': True,
            'common_initial_states_and_epoch_orders_match': True, 'B_Z_initial_state_and_available_local_epoch_orders_match': True,
            'paired_sample_labels_and_node_hashes_match': True, 'no_test_inference_or_ml_import': True},
            'hardware': probe['hardware'], 'local_sync_notes': notes, 'artifact_sha256': hashes,
            'scope': 'One exploratory seed; aggregate and gzip-prediction verification, no model/graph execution. Group members differ across models; counts alone are not a causal attribution.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path.cwd())
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = analyze(args.repo.resolve())
    require('torch' not in sys.modules and 'torch_geometric' not in sys.modules, 'Offline analysis imported an ML runtime')
    write(args.output,result)
    print(json.dumps({'status':'verified','output':str(args.output),
                     'scores':{v:r['metrics']['fusion']['macro_f1'] for v,r in result['models'].items()},
                     'local_sync_notes':result['local_sync_notes']}))


if __name__ == '__main__':
    main()
