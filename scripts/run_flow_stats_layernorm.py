"""One LN-only seed42 candidate with a mandatory, restorable stability probe."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import traceback
from types import FunctionType, SimpleNamespace

from e3_gate_study import identity, inside, read, require, sha, write
import run_flow_stats_residual as parent

HERE = Path(__file__).resolve().parent
MODULE = HERE.parent / 'src/xgnid/flow_stats_layernorm.py'
VARIANT = 'J2_flow_stats_ln'
PROBE = {'updates': 3, 'ratio_median_ceiling': .10, 'ratio_p95_ceiling': .25,
         'ratio_max_ceiling': 1., 'eval_nll_increase_ceiling': .10,
         'prediction_change_fraction_ceiling': .05}


class ProbeRejected(ValueError):
    pass


def check_config(c, base):
    require(c['schema'] == 'xgnid.flow-stats-layernorm.v1' and c['variant'] == VARIANT,
            'Wrong LayerNorm candidate')
    require(c['probe'] == PROBE, 'Unregistered stability probe thresholds')
    actual = {k: v for k, v in c.items() if k not in ('parent_config', 'parent_config_sha256', 'probe')}
    actual['schema'], actual['variant'] = base['schema'], base['variant']
    actual['stats_path'] = dict(actual['stats_path'])
    require(actual['stats_path']['normalization'] == 'LayerNorm64_eps1e-5_affine', 'Wrong LayerNorm definition')
    actual['stats_path']['normalization'] = base['stats_path']['normalization']
    require(actual == base, 'Only hidden LayerNorm may change the registered candidate protocol')


def metadata(args):
    c = read(args.config)
    path = inside(args.config.parent, c['parent_config'])
    require(sha(path) == c['parent_config_sha256'], 'Original residual config changed')
    check_config(c, read(path))
    meta = parent.metadata(SimpleNamespace(repo=args.repo, config=path))
    binding = {'config_sha256': sha(args.config), 'parent_residual_binding': meta[-1],
        'sources': {'scripts/run_flow_stats_layernorm.py': sha(Path(__file__)),
                    'src/xgnid/flow_stats_layernorm.py': sha(MODULE),
                    'scripts/summarize_flow_stats_layernorm.py': sha(HERE / 'summarize_flow_stats_layernorm.py')}}
    return (c, *meta[1:-1], binding)


def candidate_class():
    parent.candidate_class()  # Load the exact archived parent, not a stale server copy.
    name = 'xgnid.flow_stats_layernorm'
    spec = importlib.util.spec_from_file_location(name, MODULE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.FlowStatsLayerNormModel


def build(core, data, device, n0, gate, model_class):
    # Parent construction has exactly the desired strict load, RNG restore and
    # eval behavior. Only its parameter-count expectation needs the extra 128.
    core.seed_all(42)
    warm = core.Batch.from_data_list([data.load(r) for r in data.training_candidates()[:16]]).to(device)
    m = model_class(hidden_dim=64, aux_loss_weight=0, webbased_aux_weight=0,
        webbased_recon_aux_weight=0, benign_spoofing_aux_weight=0, webbased_recon_hard_weight=0).to(device)
    m.use_gate = False
    core.initialize(m, 'E2', warm)
    m.load_state_dict(n0['model_state'], strict=True)
    m.joint_gate = parent.vector.make_vector(core, m, 'V2', device)
    m.joint_gate.load_state_dict(gate, strict=True)
    rng = parent.pair.rng_state(core)
    m.add_flow_stats_path(warm)
    parent.pair.restore_rng(core, rng)
    require(parent.pair.rng_hash(core, rng) == parent.pair.rng_hash(core, parent.pair.rng_state(core)),
            'LayerNorm construction changed public RNG')
    require(sum(p.numel() for p in m.flow_stats_mlp.parameters()) == 64 * m.flow_stats_input_dim + 4352,
            'Wrong normalized MLP capacity')
    require(isinstance(m.flow_stats_mlp[1], core.torch.nn.LayerNorm)
            and isinstance(m.flow_stats_mlp[3], core.torch.nn.Linear)
            and bool((m.flow_stats_mlp[3].weight == 0).all())
            and bool((m.flow_stats_mlp[3].bias == 0).all()), 'Wrong LayerNorm/zero-last-layer structure')
    return m.eval()


# Separate namespaces: historical functions and module globals stay unchanged.
groups = FunctionType(parent.groups.__code__, dict(parent.groups.__globals__, VARIANT=VARIANT), 'groups')


def scalar_distribution(x):
    return {'median': float(x.quantile(.5)), 'p95': float(x.quantile(.95)),
            'max': float(x.max()), 'mean': float(x.mean())}


def snapshot(core, model, raw, device):
    model.eval()
    require(not any(m.training for m in model.modules()), 'Probe evaluation mode is incorrect')
    before = core.state_hash(model.state_dict().items())
    with core.torch.no_grad():
        batch = raw.clone().to(device)
        scores = model(batch)
        d = model._flow_stats_diagnostics
        require(bool(core.torch.isfinite(scores).all())
                and all(bool(core.torch.isfinite(t).all()) for t in d.values()), 'Nonfinite probe snapshot')
        result = {'sample_ids': list(batch.sample_id), 'labels': batch.y.view(-1).cpu(),
            'scores': scores.cpu(), 'flow_weights': d['flow_weights'].cpu(),
            'ratio': d['residual_to_graph_norm'].cpu(), 'residual_norm': d['flow_residual_norm'].cpu(),
            'graph_norm': d['flow_graph_norm'].cpu(), 'eval_nll': float(core.F.nll_loss(scores, batch.y.view(-1)))}
    require(core.state_hash(model.state_dict().items()) == before, 'Eval probe modified parameters/BN')
    return result


def compare(core, a, b):
    require(a['sample_ids'] == b['sample_ids'] and bool(core.torch.equal(a['labels'], b['labels'])),
            'Probe comparison must use the SAME samples')
    gate = (b['flow_weights'] - a['flow_weights']).abs()
    return {'eval_nll_before': a['eval_nll'], 'eval_nll_after': b['eval_nll'],
        'eval_nll_increase': b['eval_nll'] - a['eval_nll'],
        'prediction_change_fraction': float((a['scores'].argmax(-1) != b['scores'].argmax(-1)).float().mean()),
        'log_probability_max_abs_change': float((b['scores'] - a['scores']).abs().max()),
        'flow_weight_mean_abs_change_per_dim': gate.mean(0).tolist(),
        'flow_weight_max_abs_change': float(gate.max()),
        'ratio_after': scalar_distribution(b['ratio'])}


def stability_violations(change, bounds):
    observed = {'ratio_median': change['ratio_after']['median'], 'ratio_p95': change['ratio_after']['p95'],
        'ratio_max': change['ratio_after']['max'], 'eval_nll_increase': change['eval_nll_increase'],
        'prediction_change_fraction': change['prediction_change_fraction']}
    return [{'measure': k, 'value': v, 'ceiling': bounds[k + '_ceiling']}
            for k, v in observed.items() if not math.isfinite(v) or v > bounds[k + '_ceiling']]


def probe(core, m, loader, device, c, output):
    clean = {n: t.detach().clone() for n, t in m.state_dict().items()}
    digest = core.state_hash(clean.items())
    rng = parent.pair.rng_state(core)
    trace, records = {}, []
    result = {'status': 'running', 'bounds': c['probe'], 'updates': records,
              'discarded': True, 'formal_training_updates': 0}
    try:
        raw = list(loader)
        require(len(raw) == c['probe']['updates'] == 3 and all(b.num_graphs == 512 for b in raw),
                'Expected exactly three fixed seed42 training batches')
        initial = [snapshot(core, m, batch, device) for batch in raw]
        trace['initial'] = initial
        trace['updates'] = []
        require(all(bool((x['ratio'] == 0).all()) for x in initial), 'Initial residual is nonzero')
        opt = core.torch.optim.Adam(groups(m, VARIANT, c), weight_decay=c['weight_decay'])
        inactive = {n: p.detach().clone() for n, p in m.named_parameters() if not p.requires_grad}
        public = ('flow_encoder.', 'packet_encoder.', 'flow_classifier.', 'packet_classifier.', 'shared_head.', 'joint_gate.')
        for step, batch in enumerate(raw, 1):
            before_eval = snapshot(core, m, batch, device)
            before_parameters = {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad}
            m.train()
            require(all(module.training for module in m.modules()), 'Joint train mode differs from J2')
            opt.zero_grad(set_to_none=True)
            work = batch.clone().to(device)
            loss = core.F.nll_loss(m(work), work.y.view(-1))
            require(bool(core.torch.isfinite(loss)), 'Nonfinite probe train loss')
            loss.backward()
            require(all(p.grad is None or bool(core.torch.isfinite(p.grad).all()) for p in m.parameters()),
                    'Nonfinite probe gradients')
            norms = {n: float(p.grad.abs().sum()) if p.grad is not None else 0.
                     for n, p in m.named_parameters() if p.requires_grad}
            opt.step()
            changed = {n: not core.torch.equal(before_parameters[n], p)
                       for n, p in m.named_parameters() if n in before_parameters}
            for prefix in public:
                require(any(v > 0 for n, v in norms.items() if n.startswith(prefix))
                        and any(v for n, v in changed.items() if n.startswith(prefix)),
                        'Public module did not receive gradient/update: ' + prefix)
            after_eval = snapshot(core, m, batch, device)
            anchor = after_eval if step == 1 else snapshot(core, m, raw[0], device)
            cumulative = compare(core, initial[step - 1], after_eval)
            anchor_change = compare(core, initial[0], anchor)
            reasons = ([{'view': 'current_batch', **v} for v in stability_violations(cumulative, c['probe'])]
                       + [{'view': 'fixed_first_batch', **v} for v in stability_violations(anchor_change, c['probe'])])
            row = {'step': step, 'sample_ids_sha256': identity(after_eval['sample_ids']),
                'train_nll': float(loss.detach()), 'before_after_same_batch': compare(core, before_eval, after_eval),
                'from_clean_start_same_batch': cumulative, 'fixed_first_batch_from_start': anchor_change,
                'stats_gradients_l1': {n: v for n, v in norms.items() if n.startswith('flow_stats_mlp.')},
                'stats_parameters_changed': {n: v for n, v in changed.items() if n.startswith('flow_stats_mlp.')},
                'public_gradients_and_updates_passed': True, 'stability_violations': reasons}
            records.append(row)
            trace['updates'].append({'before': before_eval, 'after': after_eval, 'fixed_first_batch': anchor})
            write(output / 'progress.json', {'status': 'running', 'phase': 'LayerNorm_stability_probe', 'step': step})
            write(output / 'probe.json', result)
            print(json.dumps({'phase': 'LayerNorm_stability_probe', **row}), flush=True)
            if reasons:
                raise ProbeRejected('Probe stability bounds exceeded at update ' + str(step))
        # Record every tensor, but require updates by layer: a tiny first-layer
        # bias update can round away in FP32 after LayerNorm rescales its input.
        for prefix in ('flow_stats_mlp.0.', 'flow_stats_mlp.1.', 'flow_stats_mlp.3.'):
            require(any(any(v > 0 for n, v in r['stats_gradients_l1'].items() if n.startswith(prefix))
                        and any(v for n, v in r['stats_parameters_changed'].items() if n.startswith(prefix))
                        for r in records[1:]), 'No later gradient/update for layer ' + prefix)
        require(all(core.torch.equal(p, inactive[n]) for n, p in m.named_parameters() if n in inactive),
                'Inactive historical parameters changed')
        counters = [n for n in clean if n.endswith('num_batches_tracked')]
        require(counters and all(int(m.state_dict()[n] - clean[n]) == 3 for n in counters),
                'Encoder BN behavior differs from original J2')
        require(any(r['before_after_same_batch']['ratio_after']['max'] > 0 for r in records), 'Path never became active')
        result['status'] = 'passed'
    except ProbeRejected as exc:
        result.update(status='rejected', error=str(exc))
        raise
    except BaseException as exc:
        result.update(status='failed', error=str(exc))
        raise
    finally:
        m.load_state_dict(clean, strict=True)
        m.zero_grad(set_to_none=True)
        m.eval()
        parent.pair.restore_rng(core, rng)
        restored = (core.state_hash(m.state_dict().items()) == digest
                    and parent.pair.rng_hash(core, parent.pair.rng_state(core)) == parent.pair.rng_hash(core, rng)
                    and not any(module.training for module in m.modules()))
        result['clean_start_restored'] = restored
        core.checkpoint(output / 'probe_snapshots.pt', trace)
        result['snapshot_sha256'] = sha(output / 'probe_snapshots.pt')
        write(output / 'probe.json', result)
        require(restored, 'Probe leaked model/BN/RNG/mode state')
    return result


def monitored_train(core, args, c, data, device, m, variant, binding):
    """Observe all validation forwards; leave loss, updates and selection intact."""
    path = args.output / 'epoch_scale_diagnostics.jsonl'
    with path.open('x', encoding='utf-8') as log:
        class ObservedCore:
            index = 0
            def __getattr__(self, name):
                return getattr(core, name)

            def evaluate(self, model, loader, device, export_rows=False):
                samples = {k: [] for k in ('flow_weights', 'flow_graph_norm', 'flow_residual_norm', 'residual_to_graph_norm')}
                def collect(module, inputs, outputs):
                    for key in samples:
                        samples[key].append(module._flow_stats_diagnostics[key].detach().cpu())
                handle = model.register_forward_hook(collect)
                try:
                    measured, rows = core.evaluate(model, loader, device, export_rows=export_rows)
                finally:
                    handle.remove()
                diag = {k: parent.tensor_summary(core, values) for k, values in samples.items()}
                measured['flow_stats_diagnostics'] = diag
                row = {'phase': 'selected_checkpoint_replay' if export_rows else 'epoch_validation',
                       'epoch': None if export_rows else self.index, 'macro_f1': measured['macro_f1'],
                       'distributions': diag}
                if not export_rows:
                    self.index += 1
                log.write(json.dumps(row) + '\n')
                log.flush()
                return measured, rows
        ns = dict(parent.joint.train.__globals__, groups=groups, build=build)
        trainer = FunctionType(parent.joint.train.__code__, ns, 'unchanged_joint_train')
        return trainer(ObservedCore(), args, c, data, device, m, variant, binding)


def runtime_entry(output):
    """Adapt construction/probing/observation without changing frozen sources."""
    class JointProxy:
        train = staticmethod(monitored_train)
        def __getattr__(self, name):
            return getattr(parent.joint, name)
    ns = dict(parent.run.__globals__, VARIANT=VARIANT, build=build, groups=groups,
              candidate_class=candidate_class, joint=JointProxy(),
              probe=lambda core, m, loader, device, c: probe(core, m, loader, device, c, output))
    ns['roundtrip_probe'] = FunctionType(parent.roundtrip_probe.__code__, ns, 'roundtrip_probe')
    return FunctionType(parent.run.__code__, ns, 'run_layernorm')


def execute(args, meta):
    require(not args.output.exists(), 'Refuse overwrite/resume')
    args.output.mkdir(parents=True)
    write(args.output / 'config.json', meta[0])
    write(args.output / 'binding.json', meta[-1])
    write(args.output / 'status.json', {'status': 'running', 'phase': 'tensor_preflight'})
    write(args.output / 'progress.json', {'status': 'running', 'phase': 'tensor_preflight'})
    print(json.dumps({'phase': 'initial_V2_equivalence_and_replay', 'variant': VARIANT}), flush=True)
    try:
        runtime_entry(args.output)(args, meta)
        from summarize_flow_stats_layernorm import summarize
        summarize(args.output)
        write(args.output / 'status.json', {'status': 'completed', 'exit_code': 0})
        write(args.output / 'progress.json', {'status': 'completed', 'training_jobs': 1})
    except ProbeRejected as exc:
        status = {'status': 'preflight_rejected', 'exit_code': 2, 'formal_training_started': False,
                  'formal_training_updates': 0, 'reason': str(exc)}
        write(args.output / 'status.json', status)
        write(args.output / 'progress.json', status)
        print(json.dumps(status), flush=True)
        raise SystemExit(2)
    except BaseException:
        write(args.output / 'status.json', {'status': 'failed', 'exit_code': 1})
        (args.output / 'error.log').write_text(traceback.format_exc(), encoding='utf-8')
        raise


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
        print(json.dumps({'status': 'metadata_checks_passed', 'binding': meta[-1], 'probe_bounds': PROBE,
                          'tensor_checks_executed': False, 'training_started': False}), flush=True)
        return
    require(os.name == 'posix', 'Formal training runs on the original Linux wx server')
    args.output = inside(args.repo / 'outputs', inside(args.repo, args.output))
    execute(args, meta)


if __name__ == '__main__':
    main()
