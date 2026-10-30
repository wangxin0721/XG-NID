"""One seed42 structure screen; reuse completed lower-LR J2 as the control.

No torch import in metadata/check mode. Tensor probes must finish before the
unchanged joint trainer is invoked. Historical sources and outputs stay intact.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from itertools import islice
from pathlib import Path
import sys
import time
import traceback
from types import FunctionType, SimpleNamespace

from e3_gate_study import (activate_gpu, checked_gpu, guard_graph_loads, inside,
                           read, require, runtime, sha, write)
import run_e2_concat_pair as pair
import run_fair_vector as vector
import run_frozen_fusion as frozen
import run_joint_gate as joint
from run_fusion_seed_repeats import predictions

HERE = Path(__file__).resolve().parent
MODULE = HERE.parent / 'src/xgnid/flow_stats_residual.py'
VARIANT = 'J2_flow_stats'
COMMON = ('seed', 'encoder_lr', 'head_lr', 'gate_lr', 'weight_decay',
          'epochs', 'patience', 'batch_size', 'eval_batch_size',
          'test_evaluated', 'automatic_followup')


def check_config(c, reference):
    joint.check_config(reference)
    require(c['schema'] == 'xgnid.flow-stats-residual.v1' and c['variant'] == VARIANT,
            'Wrong residual experiment')
    require(all(c[k] == reference[k] for k in COMMON), 'Public training protocol differs from J2')
    require(c['training_jobs'] == 1 and c['encoder_mode'] == 'train'
            and c['optimizer'] == 'fresh_Adam' and c['loss'] == 'plain_NLL'
            and c['start'] == 'same_N0_plus_frozen_V2_not_joint_best', 'Wrong start/training scope')
    require(0 < c['stats_lr'] < 1 and c['stats_path'] == {
        'hidden_dim': 64, 'activation': 'LeakyReLU_default', 'normalization': 'none',
        'dropout': 0., 'pooling': 'flow_global_mean', 'last_layer_zero_init': True,
        'coefficient': 'fixed_one'}, 'Unregistered residual architecture')
    require(c['screening'] == {'minimum_gain_over_stronger_reference_pp': .10,
        'class_tradeoff_review_required': True, 'statistical_significance_claimed': False},
        'Screening rule changed')


def history_evidence(root, m):
    """Validate selected epoch, full update budget, scores, and recorded order."""
    history = [json.loads(s) for s in (root / 'epochs.jsonl').read_text(encoding='utf-8').splitlines()]
    evidence = joint.trained_evidence(m['epoch_zero']['macro_f1'], history)
    require(history and len(history) == m['epochs'] <= 30, 'Incomplete joint history')
    best, best_epoch, stale = m['epoch_zero']['macro_f1'], 0, 0
    for epoch, row in enumerate(history, 1):
        require(row['optimizer_updates'] == 211, 'Unexpected epoch update budget')
        best, stale, improved = pair.selection_state(row['val_macro_f1'], best, stale)
        if improved:
            best_epoch = epoch
        require(row['best_epoch'] == best_epoch and row['stale_epochs'] == stale,
                'Selection log disagrees')
        require(epoch == len(history) or stale < 12, 'Training continued after early stop')
    require(m['updates'] == 211 * len(history)
            and (stale >= 12 or len(history) == 30)
            and m['orders'] == [r['train_order_sha256'] for r in history]
            and m['best_epoch'] == best_epoch
            and math.isclose(best, m['best_validation']['macro_f1'], abs_tol=1e-12, rel_tol=0.)
            and m['training_evidence'] == evidence
            and m['selected_epoch_zero'] == (best_epoch == 0)
            and math.isclose(history[-1]['val_macro_f1'], m['last_validation']['macro_f1'],
                             abs_tol=1e-12, rel_tol=0.), 'Joint selection/budget evidence differs')
    return evidence


def reference_receipt(repo, c, reference, parent, refs, report, receipt, config_path):
    """Read exactly the completed lower-LR pair, with no fallback/retraining."""
    root = inside(repo / 'outputs', inside(repo, c['reference_directory']))
    require(read(root / 'status.json') == {'status': 'completed', 'exit_code': 0},
            'Lower-LR reference pair is incomplete')
    summary = read(root / 'summary.json')
    require(sha(config_path) == c['reference_config_sha256']
            and sha(HERE / 'run_joint_gate.py') == c['reference_source_sha256'],
            'Historical config/trainer bytes changed')
    binding = summary['binding']
    require(summary['status'] == 'completed' and summary['test_evaluated'] is False
            and binding['joint_config_sha256'] == c['reference_config_sha256']
            and binding['joint_source_sha256'] == c['reference_source_sha256']
            and binding['parent'] == parent
            and binding['stage_A_best_sha256'] == receipt['best_sha256']
            and binding['stage_A_receipt'] == receipt
            and summary['N0'] == refs['42']['metrics']
            and summary['frozen_V2'] == report['metrics'], 'Reference start/config/binding differs')
    for name in ('J0', 'J2'):
        folder = root / name
        m = summary['joint'][name]
        require(read(folder / 'metrics.json') == m, 'Reference summary/metrics disagree')
        vector.verify_metrics(m['best_validation'])
        vector.verify_metrics(m['last_validation'])
        require(sha(folder / 'best.pt') == m['checkpoint_sha256'], 'Reference weight changed')
        history_evidence(folder, m)
    m = summary['joint']['J2']
    require(m['checkpoint_sha256'] == c['reference_J2_best_sha256']
            and m['best_epoch'] == c['reference_J2_best_epoch'], 'Wrong completed J2 control')
    require(m['epoch_zero']['confusion_matrix'] == report['metrics']['best_validation']['confusion_matrix'],
            'Reference J2 did not start from selected V2')
    common = min(len(x['orders']) for x in summary['joint'].values())
    require(summary['joint']['J0']['orders'][:common] == m['orders'][:common], 'Reference pair order differs')
    return root, summary, {'directory': str(root), 'summary_sha256': sha(root / 'summary.json'),
        'J2_best_sha256': m['checkpoint_sha256'],
        'J2_predictions_sha256': sha(root / 'J2/validation_predictions.jsonl.gz'),
        'public_training_settings': {k: reference[k] for k in COMMON}, 'reused': True}


def metadata(args):
    c = read(args.config)
    refpath = inside(args.config.parent, c['reference_config'])
    reference = read(refpath)
    check_config(c, reference)
    joint.validate_scaled_config(refpath, reference)
    _, registry, _, fusion, refs, parent = vector.metadata(
        SimpleNamespace(repo=args.repo, config=inside(refpath.parent, reference['fair_config'])))
    stage, report, receipt = joint.stage_receipt(args.repo, reference, parent, fusion)
    root, summary, control_receipt = reference_receipt(
        args.repo, c, reference, parent, refs, report, receipt, refpath)
    binding = {'config_sha256': sha(args.config), 'parent': parent, 'stage_A_receipt': receipt,
        'reference_receipt': control_receipt, 'sources': {
            'scripts/run_flow_stats_residual.py': sha(Path(__file__)),
            'src/xgnid/flow_stats_residual.py': sha(MODULE),
            'scripts/summarize_flow_stats_residual.py': sha(HERE / 'summarize_flow_stats_residual.py'),
            'scripts/run_joint_gate.py': sha(HERE / 'run_joint_gate.py')}}
    return c, registry, refs, stage, report, root, summary, binding


def groups(model, variant, c):
    require(variant == VARIANT, 'Unexpected candidate variant')
    buckets = {k: [] for k in ('encoder', 'head', 'gate', 'stats')}
    parameters = {}
    for name, p in model.named_parameters(remove_duplicate=False):
        kind = ('stats' if name.startswith('flow_stats_mlp.') else joint.group_name(name, 'J2'))
        item = parameters.setdefault(id(p), {'parameter': p, 'kinds': set()})
        if kind is not None:
            item['kinds'].add(kind)
    for item in parameters.values():
        require(len(item['kinds']) <= 1, 'Shared parameter crosses learning-rate groups')
        p = item['parameter']
        p.requires_grad_(bool(item['kinds']))
        if item['kinds']:
            buckets[next(iter(item['kinds']))].append(p)
    require(all(buckets.values()), 'An active parameter group is missing')
    result = [{'params': ps, 'lr': c[k + '_lr'], 'name': k} for k, ps in buckets.items()]
    ids = [id(p) for g in result for p in g['params']]
    require(len(ids) == len(set(ids)) == sum(p.requires_grad for p in model.parameters()),
            'Optimizer parameter omission/duplication')
    return result


def candidate_class():
    # An archive contains ONLY the new src file; the frozen xgnid.model package
    # continues to come from the verified server worktree. Never overwrite it.
    name = 'xgnid.flow_stats_residual'
    spec = importlib.util.spec_from_file_location(name, MODULE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.FlowStatsResidualModel


def build(core, data, device, n0, gate, model_class):
    core.seed_all(42)
    warm = core.Batch.from_data_list([data.load(r) for r in data.training_candidates()[:16]]).to(device)
    m = model_class(hidden_dim=64, aux_loss_weight=0, webbased_aux_weight=0,
        webbased_recon_aux_weight=0, benign_spoofing_aux_weight=0, webbased_recon_hard_weight=0).to(device)
    m.use_gate = False
    core.initialize(m, 'E2', warm)
    m.load_state_dict(n0['model_state'], strict=True)
    m.joint_gate = vector.make_vector(core, m, 'V2', device)
    m.joint_gate.load_state_dict(gate, strict=True)
    before = pair.rng_state(core)
    m.add_flow_stats_path(warm)
    pair.restore_rng(core, before)
    require(pair.rng_hash(core, before) == pair.rng_hash(core, pair.rng_state(core)),
            'New module initialization perturbed public RNG')
    expected = 64 * m.flow_stats_input_dim + 4224
    require(sum(p.numel() for p in m.flow_stats_mlp.parameters()) == expected, 'Wrong MLP capacity')
    return m.eval()


def checked_sampler(core, expected_orders):
    """Check each common epoch's order BEFORE its first optimizer update."""
    class Ordered(core.TrainBatches):
        def __init__(self, size, batch_size, seed):
            super().__init__(size, batch_size, seed)
            self.epoch_index = 0

        def __iter__(self):
            iterator = super().__iter__()
            first = next(iterator)
            if self.epoch_index < len(expected_orders):
                require(self.last_order_sha256 == expected_orders[self.epoch_index],
                        'Training order differs from completed J2 control')
            self.epoch_index += 1
            yield first
            yield from iterator
    return Ordered


def probe(core, m, loader, device, c):
    clean = {n: t.detach().clone() for n, t in m.state_dict().items()}
    initial_hash = core.state_hash(clean.items())
    rng = pair.rng_state(core)
    opt = core.torch.optim.Adam(groups(m, VARIANT, c), weight_decay=c['weight_decay'])
    active = ('flow_encoder.', 'packet_encoder.', 'flow_classifier.', 'packet_classifier.',
              'shared_head.', 'joint_gate.', 'flow_stats_mlp.0.', 'flow_stats_mlp.2.')
    records = []
    inactive = {n: p.detach().clone() for n, p in m.named_parameters() if not p.requires_grad}
    for batch in islice(loader, 3):
        m.train()
        require(all(module.training for module in m.modules()), 'Joint train/eval mode differs')
        before = {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad}
        opt.zero_grad(set_to_none=True)
        batch = batch.to(device)
        loss = core.F.nll_loss(m(batch), batch.y.view(-1))
        require(bool(core.torch.isfinite(loss)), 'Nonfinite probe loss')
        loss.backward()
        require(all(p.grad is None or bool(core.torch.isfinite(p.grad).all()) for p in m.parameters()),
                'Nonfinite probe gradient')
        norms = {prefix: sum(float(p.grad.abs().sum()) for n, p in m.named_parameters()
                 if n.startswith(prefix) and p.grad is not None) for prefix in active}
        opt.step()
        changed = {prefix: any(not core.torch.equal(before[n], p) for n, p in m.named_parameters()
                    if n.startswith(prefix) and n in before) for prefix in active}
        for prefix in active[:-2]:
            require(norms[prefix] > 0 and changed[prefix], 'Public gradient/update missing: ' + prefix)
        m.eval()
        with core.torch.no_grad():
            m(batch)
            norm = float(m._flow_stats_diagnostics['flow_residual_norm'].max())
        records.append({'loss': float(loss.detach()), 'gradient_l1': norms,
                        'parameters_changed': changed, 'max_residual_norm_after_update': norm,
                        'stats_parameter_gradients_l1': {n: float(p.grad.abs().sum()) if p.grad is not None else 0.
                            for n, p in m.named_parameters() if n.startswith('flow_stats_mlp.')},
                        'stats_parameters_changed': {n: not core.torch.equal(before[n], p)
                            for n, p in m.named_parameters() if n.startswith('flow_stats_mlp.')}})
    require(len(records) == 3, 'Incomplete probe')
    for prefix in active[-2:]:
        require(any(r['gradient_l1'][prefix] > 0 and r['parameters_changed'][prefix] for r in records[1:]),
                'New MLP never received later gradient/update: ' + prefix)
    for name, _ in m.flow_stats_mlp.named_parameters():
        full = 'flow_stats_mlp.' + name
        require(any(r['stats_parameter_gradients_l1'][full] > 0 and r['stats_parameters_changed'][full]
                    for r in records[1:]), 'MLP parameter never received later gradient/update: ' + full)
    require(any(r['max_residual_norm_after_update'] > 0 for r in records), 'Residual remains identically zero')
    require(all(core.torch.equal(p, inactive[n]) for n, p in m.named_parameters() if n in inactive),
            'Inactive historical parameters changed')
    counters = [n for n in clean if n.endswith('num_batches_tracked')]
    require(counters and all(int(m.state_dict()[n] - clean[n]) == 3 for n in counters),
            'BatchNorm joint-training behavior differs from J2')
    m.load_state_dict(clean, strict=True)
    m.zero_grad(set_to_none=True)
    m.eval()
    pair.restore_rng(core, rng)
    require(core.state_hash(m.state_dict().items()) == initial_hash
            and pair.rng_hash(core, pair.rng_state(core)) == pair.rng_hash(core, rng)
            and not any(module.training for module in m.modules()), 'Probe leaked state/RNG/mode')
    return {'updates': records, 'discarded': True, 'clean_start_restored': True,
            'BN_train_updates_verified': True, 'initial_state_sha256': initial_hash}


def roundtrip_probe(core, args, data, device, model, model_class, loader):
    """Serialize the clean NEW architecture and strictly reload before training."""
    rng = pair.rng_state(core)
    digest = core.state_hash(model.state_dict().items())
    path = args.output / 'preflight_start.pt'
    core.checkpoint(path, {'model_state': model.state_dict(), 'flow_input_dim': model.flow_stats_input_dim,
                           'variant': VARIANT, 'epoch': 0, 'probe_updates_retained': False})
    saved = core.torch.load(path, map_location=device, weights_only=False)
    fresh = build(core, data, device, args.n0, args.gate, model_class)
    fresh.load_state_dict(saved['model_state'], strict=True)
    require(core.state_hash(fresh.state_dict().items()) == digest, 'Roundtrip state differs')
    model.eval()
    with core.torch.no_grad():
        checked = 0
        for batch in islice(loader, 2):
            batch = batch.to(device)
            left, right = model(batch), fresh(batch)
            require(core.torch.allclose(left, right, atol=1e-5, rtol=1e-5)
                    and bool((left.argmax(-1) == right.argmax(-1)).all()), 'Roundtrip output differs')
            checked += batch.num_graphs
    require(checked == 32, 'Incomplete roundtrip probe')
    pair.restore_rng(core, rng)
    return {'strict_state_reload': True, 'samples': checked, 'sha256': sha(path)}


def tensor_summary(core, tensors):
    x = core.torch.cat(tensors).float().cpu()
    return {'mean': x.mean(0).tolist(), 'std': x.std(0, unbiased=False).tolist(),
            'min': x.min(0).values.tolist(), 'max': x.max(0).values.tolist(),
            'quantiles': {str(q): x.quantile(q, dim=0).tolist() for q in (0., .25, .5, .75, .95, .99, 1.)}}


def diagnostics(core, m, loader, device, rows, equivalent=None):
    """Export eight weights and 64D residual/graph norms for EVERY val sample."""
    cm = [[0] * 8 for _ in range(8)]
    exports = []
    values = {k: [] for k in ('flow_weights', 'flow_graph_norm', 'flow_residual_norm', 'residual_to_graph_norm')}
    max_error = 0.
    m.eval()
    if equivalent is not None:
        equivalent.eval()
    with core.torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            scores = m(batch)
            require(bool(core.torch.isfinite(scores).all()), 'Nonfinite diagnostic prediction')
            if equivalent is not None:
                reference = equivalent(batch)
                require(core.torch.allclose(scores, reference, atol=1e-5, rtol=1e-5)
                        and bool((scores.argmax(-1) == reference.argmax(-1)).all()), 'Epoch0 outputs differ from V2')
                max_error = max(max_error, float((scores - reference).abs().max()))
            d = m._flow_stats_diagnostics
            require(d['flow_weights'].shape == (batch.num_graphs, 8)
                    and bool(((d['flow_weights'] >= 0) & (d['flow_weights'] <= 1)).all()), 'Invalid eight-dimensional weights')
            for key in values:
                require(bool(core.torch.isfinite(d[key]).all()), 'Nonfinite residual diagnostics')
                values[key].append(d[key].cpu())
            truth, predicted = batch.y.view(-1).tolist(), scores.argmax(-1).tolist()
            alpha = d['flow_weights'].tolist()
            for i, sample_id in enumerate(batch.sample_id):
                cm[truth[i]][predicted[i]] += 1
                exports.append({'sample_id': sample_id, 'label': truth[i], 'prediction': predicted[i],
                    'flow_weights': alpha[i], 'packet_weights': [1 - a for a in alpha[i]],
                    **{k: float(d[k][i]) for k in values if k != 'flow_weights'}})
    frozen.verify_order(exports, rows)
    return frozen.matrix_report(cm, core.LABELS), exports, {
        'samples': len(exports), 'max_abs_error_vs_V2': max_error if equivalent is not None else None,
        'distributions': {k: tensor_summary(core, v) for k, v in values.items()},
        'norm_ratio_epsilon': 1e-12}


def validate_predictions(rows, expected, metrics):
    frozen.verify_order(rows, expected)
    cm = [[0] * 8 for _ in range(8)]
    for r in rows:
        cm[r['label']][r['prediction']] += 1
    require(cm == metrics['confusion_matrix'], 'Predictions disagree with recorded confusion matrix')


def run(args, meta):
    c, registry, refs, stage, report, refroot, reference, binding = meta
    gpu = checked_gpu(args.physical_gpu, args.allow_shared_gpu)
    activate_gpu(gpu)
    # Import the frozen core only AFTER selecting the physical GPU.
    server_scripts = str(args.repo / 'scripts')
    sys.path.insert(0, server_scripts)
    try:
        import train_grouped as core
    finally:
        sys.path.remove(server_scripts)
    device = runtime(core)
    write(args.output / 'gpu.json', gpu)
    data = core.FrozenGraphs(inside(args.repo, registry['split_directory']), registry['split_id'])
    guard_graph_loads(data, {'train', 'val'})
    valrows = core.select_rows(data, 'val')
    loader = core.DataLoader(core.GraphRows(data, valrows), batch_size=16, shuffle=False, num_workers=2)
    args.n0 = core.torch.load(inside(args.repo, refs['42']['directory']) / 'best.pt', map_location=device, weights_only=False)
    require(args.n0['epoch'] == refs['42']['metrics']['best_epoch']
            and args.n0['validation']['confusion_matrix'] == refs['42']['metrics']['best_validation']['confusion_matrix'],
            'N0 checkpoint payload differs')
    saved_gate = core.torch.load(stage / 'V2/best.pt', map_location=device, weights_only=False)
    vm = report['metrics']
    require(saved_gate['epoch'] == vm['best_epoch']
            and saved_gate['config'] == read(stage / 'V2/config.json')
            and saved_gate['validation']['confusion_matrix'] == vm['best_validation']['confusion_matrix'],
            'V2 checkpoint payload differs')
    args.gate = saved_gate['fusion_state']
    model_class = candidate_class()
    preflight_started = time.perf_counter()
    base = joint.build(core, data, device, args.n0, 'J2', args.gate).eval()
    m = build(core, data, device, args.n0, args.gate, model_class)
    base_state = base.state_dict()
    require(all(core.torch.equal(base_state[n], t) for n, t in m.state_dict().items() if n in base_state),
            'Candidate altered public parameters or buffers')
    require({n for n in m.state_dict() if n not in base_state} ==
            {n for n in m.state_dict() if n.startswith('flow_stats_mlp.')}, 'Unexpected additional state')
    base_groups = joint.groups(base, 'J2', c)
    gs = groups(m, VARIANT, c)
    require([(g['name'], g['lr'], sum(p.numel() for p in g['params'])) for g in gs[:-1]] ==
            [(g['name'], g['lr'], sum(p.numel() for p in g['params'])) for g in base_groups],
            'Public optimizer groups differ from J2')
    initial, initial_rows, initial_diag = diagnostics(core, m, loader, device, valrows, equivalent=base)
    require(initial['confusion_matrix'] == vm['best_validation']['confusion_matrix']
            and abs(initial['macro_f1'] - vm['best_validation']['macro_f1']) < 1e-12
            and all(r['flow_residual_norm'] == 0. for r in initial_rows), 'Zero residual did not reproduce V2')
    v2rows = predictions(stage / 'V2/validation_predictions.jsonl.gz')
    validate_predictions(v2rows, valrows, vm['best_validation'])
    require(all(a['prediction'] == b['prediction'] for a, b in zip(initial_rows, v2rows)), 'Epoch0 prediction differs from V2')
    # Verify the actual historical control, but NEVER use its weights as a start.
    control = core.torch.load(refroot / 'J2/best.pt', map_location=device, weights_only=False)
    require(control['epoch'] == reference['joint']['J2']['best_epoch']
            and control['config']['binding'] == reference['binding']
            and control['config']['protocol'] == read(inside(args.config.parent, c['reference_config']))
            and control['config']['variant'] == 'J2', 'Control checkpoint config differs')
    base.load_state_dict(control['model_state'], strict=True)
    control_measured, control_rows = core.evaluate(base, loader, device, export_rows=True)
    frozen.verify_order(control_rows, valrows)
    old_rows = predictions(refroot / 'J2/validation_predictions.jsonl.gz')
    validate_predictions(old_rows, valrows, reference['joint']['J2']['best_validation'])
    require(control_measured['confusion_matrix'] == reference['joint']['J2']['best_validation']['confusion_matrix']
            and all(a['prediction'] == b['prediction'] for a, b in zip(control_rows, old_rows)), 'Historical J2 replay failed')
    del base, control
    write(args.output / 'initial_checks.json', {'V2_epoch_zero_reproduced': True,
        'public_state_identical': True, 'public_optimizer_groups_identical': True,
        'historical_J2_replayed': True, 'initial_diagnostics': initial_diag,
        'flow_input_dim': m.flow_stats_input_dim,
        'additional_parameters': sum(p.numel() for p in m.flow_stats_mlp.parameters()),
        'initialization_rng_preserved': True})
    pair.save_predictions(args.output / 'epoch_zero_predictions.jsonl.gz', initial_rows)
    trainrows = core.select_rows(data, 'train')
    batches = list(islice(core.TrainBatches(len(trainrows), 512, 42), 3))
    pl = core.DataLoader(core.GraphRows(data, trainrows), batch_sampler=batches, num_workers=0)
    args.probes = {VARIANT: probe(core, m, pl, device, c)}
    args.probes[VARIANT]['checkpoint_roundtrip'] = roundtrip_probe(
        core, args, data, device, m, model_class, loader)
    write(args.output / 'probe.json', args.probes[VARIANT])
    preflight_seconds = time.perf_counter() - preflight_started
    write(args.output / 'status.json', {'status': 'running', 'phase': 'training', 'tensor_preflight_passed': True})
    # Reuse the historical trainer's exact body without mutating its globals.
    # Only construction and the explicit extra optimizer group are adapted.
    def reconstruct(core, data, device, n0, variant, gate=None):
        require(variant == VARIANT, 'Wrong replay architecture')
        return build(core, data, device, n0, args.gate, model_class)
    ns = dict(joint.train.__globals__, groups=groups, build=reconstruct)
    trainer = FunctionType(joint.train.__code__, ns, 'train_flow_stats')
    class CoreProxy:
        TrainBatches = staticmethod(checked_sampler(core, reference['joint']['J2']['orders']))
        def __getattr__(self, name):
            return getattr(core, name)
    result = trainer(CoreProxy(), args, c, data, device, m, VARIANT, binding)
    folder = args.output / VARIANT
    common = min(len(result['orders']), len(reference['joint']['J2']['orders']))
    require(result['orders'][:common] == reference['joint']['J2']['orders'][:common], 'Common epoch orders differ')
    history_evidence(folder, result)
    write(args.output / 'status.json', {'status': 'running', 'phase': 'best_checkpoint_diagnostics'})
    diagnostic_started = time.perf_counter()
    selected = core.torch.load(folder / 'best.pt', map_location=device, weights_only=False)
    fresh = reconstruct(core, data, device, args.n0, VARIANT)
    fresh.load_state_dict(selected['model_state'], strict=True)
    best, best_rows, best_diag = diagnostics(core, fresh, loader, device, valrows)
    require(best['confusion_matrix'] == result['best_validation']['confusion_matrix'], 'Diagnostic replay differs')
    recorded_rows = predictions(folder / 'validation_predictions.jsonl.gz')
    validate_predictions(recorded_rows, valrows, result['best_validation'])
    require(all(a['prediction'] == b['prediction'] for a, b in zip(
        best_rows, recorded_rows)), 'Prediction replay differs')
    pair.save_predictions(folder / 'validation_diagnostics.jsonl.gz', best_rows)
    write(folder / 'diagnostics.json', best_diag)
    best_hash = core.state_hash(fresh.state_dict().items())
    fresh.flow_stats_enabled = False
    off, off_rows, off_diag = diagnostics(core, fresh, loader, device, valrows)
    require(core.state_hash(fresh.state_dict().items()) == best_hash, 'Path-off evaluation changed state')
    fresh.flow_stats_enabled = True
    pair.save_predictions(folder / 'validation_stats_off.jsonl.gz', off_rows)
    write(folder / 'stats_off_metrics.json', off)
    summary = {'status': 'completed', 'exit_code': 0, 'binding': binding, 'protocol': c,
        'N0': refs['42']['metrics'], 'frozen_V2': vm,
        'reference_joint': reference['joint'], 'candidate': result,
        'comparisons': {name: pair.compare_predictions(rows, best_rows) for name, rows in (
            ('frozen_V2', v2rows), ('lower_lr_J2', old_rows), ('stats_off_same_checkpoint', off_rows))},
        'stats_off': off, 'stats_on_minus_off_pp': 100 * (best['macro_f1'] - off['macro_f1']),
        'diagnostics': best_diag, 'stats_off_diagnostics': off_diag,
        'common_epochs_order_checked': common, 'tensor_preflight_passed': True,
        'preflight_wall_seconds': preflight_seconds,
        'post_training_diagnostics_wall_seconds': time.perf_counter() - diagnostic_started,
        'stage_A': reference['stage_A'], 'training_jobs': 1, 'test_evaluated': False,
        'automatic_followup': False,
        'attribution': 'Candidate changes the statistics path and adds its parameters/learning-rate group; '
                       'overall N0 gains are not an isolated gate effect. Path-off is a post-selection '
                       'dependency diagnostic at the same checkpoint, not a separately trained ablation.'}
    summary['artifacts'] = {p.name: sha(p) for p in (folder / 'best.pt', folder / 'last.pt',
        folder / 'validation_predictions.jsonl.gz', folder / 'validation_diagnostics.jsonl.gz',
        folder / 'validation_stats_off.jsonl.gz')}
    write(args.output / 'summary.json', summary)
    from summarize_flow_stats_residual import summarize
    summarize(args.output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path.cwd())
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=('check', 'run'), default='check')
    p.add_argument('--physical-gpu', type=int, default=1)
    p.add_argument('--allow-shared-gpu', action='store_true')
    args = p.parse_args()
    args.repo = args.repo.resolve()
    args.config = (args.repo / args.config).resolve()
    meta = metadata(args)
    if args.mode == 'check':
        print(json.dumps({'status': 'metadata_checks_passed', 'binding': meta[-1],
                          'tensor_checks_executed': False, 'training_started': False}), flush=True)
        return
    require(os.name == 'posix', 'Formal training runs on Linux server only')
    args.output = inside(args.repo / 'outputs', inside(args.repo, args.output))
    require(not args.output.exists(), 'Refuse overwrite/resume')
    args.output.mkdir(parents=True)
    write(args.output / 'config.json', read(args.config))
    write(args.output / 'binding.json', meta[-1])
    write(args.output / 'status.json', {'status': 'running', 'phase': 'tensor_preflight'})
    try:
        run(args, meta)
        write(args.output / 'status.json', {'status': 'completed', 'exit_code': 0})
        write(args.output / 'progress.json', {'status': 'completed', 'training_jobs': 1})
    except BaseException:
        write(args.output / 'status.json', {'status': 'failed', 'exit_code': 1})
        (args.output / 'error.log').write_text(traceback.format_exc(), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
