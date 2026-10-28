"""Validation-only warm-start E2/concat pair; tensor execution on Linux wx only."""
from __future__ import annotations

import argparse
import copy
import gc
import gzip
import json
import os
from pathlib import Path
import random
import sys
import traceback
from types import MethodType
from itertools import islice

from e3_gate_study import (activate_gpu, checked_gpu, guard_graph_loads, identity,
                           inside, read, require, runtime, sha, write)

VARIANTS = ('N0', 'N1')


def rng_state(core):
    return {'python': random.getstate(), 'torch': core.torch.get_rng_state().clone(),
            'cuda': [s.clone() for s in core.torch.cuda.get_rng_state_all()]}


def restore_rng(core, state):
    random.setstate(state['python'])
    core.torch.set_rng_state(state['torch'])
    core.torch.cuda.set_rng_state_all(state['cuda'])


def rng_hash(core, state):
    return identity({'python': state['python'], 'tensors': core.state_hash(
        [('cpu', state['torch'])]+[(str(i), s) for i, s in enumerate(state['cuda'])])})


def metadata(args):
    spec = read(args.config)
    require(spec['schema'] == 'xgnid.e2-concat-pair.v1', 'Wrong pair protocol')
    require(spec['variants'] == list(VARIANTS) and spec['epochs'] == 30
            and spec['seed'] == 42 and spec['batch_size'] == 512
            and spec['lr'] == .0001 and spec['weight_decay'] == 0.
            and spec['optimizer'] == 'Adam' and spec['workers'] == 2
            and spec['evaluation_batch_size'] == 16 and spec['early_stop'] is True
            and spec['patience'] == 12 and spec['probe_batches'] == 3
            and spec['include_initial_checkpoint'] is True
            and spec['test_evaluated'] is False, 'Frozen pair settings changed')
    registry_path = inside(args.config.parent, spec['reference_registry'])
    registry = read(registry_path)
    ref = registry['references']['E2']
    folder = inside(args.repo, ref['directory'])
    for name, key in [('summary.json', 'summary_sha256'), ('E2/config.json', 'config_sha256'),
                      ('E2/metrics.json', 'metrics_sha256')]:
        require(sha(folder / name) == ref[key], 'E2 reference changed: ' + name)
    status = read(folder / 'status.json')
    require((status['status'], status['exit_code']) == ('completed', 0), 'Incomplete E2')
    require(sha(folder / 'E2/best.pt') == ref['checkpoint_sha256']['best.pt'], 'E2 best changed')
    for name, digest in ref['sources'].items():
        require(sha(inside(args.repo, name)) == digest, 'Historical source changed: ' + name)
    split_dir = inside(args.repo, registry['split_directory'])
    split = read(split_dir / 'split.json')
    require(split['split_id'] == registry['split_id'] and split['partition_sizes'] ==
            {'train': 107621, 'val': 13446, 'test': 13448}, 'Wrong split')
    manifest = sha(inside(split_dir, split['sample_manifest']))
    require(manifest == split['sample_manifest_sha256'] ==
            read(folder / 'E2/config.json')['binding']['samples_sha256'], 'Manifest changed')
    metrics = read(folder / 'E2/metrics.json')
    require(read(folder / 'summary.json')['models']['E2'] == metrics, 'E2 reports disagree')
    return spec, registry, ref, folder, {'config_sha256': sha(args.config),
        'registry_sha256': sha(registry_path), 'checkpoint_sha256': ref['checkpoint_sha256']['best.pt'],
        'sources': ref['sources'], 'split_id': registry['split_id'], 'manifest_sha256': manifest,
        'entry_sha256': sha(Path(__file__)), 'wrappers': {v: sha(Path(__file__).with_name(
            'run_e2_' + ('n0_continue.py' if v == 'N0' else 'n1_concat.py'))) for v in VARIANTS},
        'helper_sha256': sha(Path(__file__).with_name('e3_gate_study.py'))}


def concat_logits(self, data):
    _, _, flow, packet = self._branch_outputs(data)
    self._last_cache = {}
    return self.shared_head(self._pair_torch.cat((flow, packet), dim=-1))


def convert(core, model):
    """Only widen H's first layer; retain every other parameter/buffer."""
    before = {n: t.detach().clone() for n, t in model.state_dict().items()}
    old = model.shared_head[0]
    require((old.in_features, old.out_features) == (8, 32), 'Unexpected E2 head')
    layer = core.torch.nn.Linear(16, 32).to(device=old.weight.device, dtype=old.weight.dtype)
    with core.torch.no_grad():
        layer.weight.copy_(core.torch.cat((old.weight / 2, old.weight / 2), dim=1))
        layer.bias.copy_(old.bias)
    model.shared_head[0] = layer
    model._pair_torch = core.torch
    model._graph_logits = MethodType(concat_logits, model)
    for name, value in model.state_dict().items():
        if name != 'shared_head.0.weight':
            require(core.torch.equal(value, before[name]), 'Conversion changed ' + name)
    require(sum(p.numel() for p in model.parameters()) == 847056, 'Wrong concat capacity')
    return model


def verify_initial(core, base, candidate, loader, device, expected):
    base.eval(); candidate.eval()
    before = [core.state_hash(m.state_dict().items()) for m in (base, candidate)]
    count, changed, max_error = 0, 0, 0.
    with core.torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            left, right = base(batch), candidate(batch)
            require(core.torch.isfinite(left).all() and core.torch.isfinite(right).all(), 'Nonfinite initial outputs')
            require(core.torch.allclose(left, right, atol=1e-5, rtol=1e-5), 'Initial outputs differ beyond tolerance')
            max_error = max(max_error, float((left-right).abs().max()))
            changed += int((left.argmax(-1) != right.argmax(-1)).sum())
            count += batch.num_graphs
    require(count == 13446 and changed == 0, 'Initial validation labels differ')
    measured, exports = core.evaluate(base, loader, device, export_rows=True)
    require(measured['confusion_matrix'] == expected['confusion_matrix'], 'Original E2 validation not reproduced')
    require(abs(measured['macro_f1']-expected['macro_f1']) < 1e-10
            and abs(100*measured['macro_f1']-96.5994) < .00005, 'Original E2 Macro-F1 differs')
    require(before == [core.state_hash(m.state_dict().items()) for m in (base, candidate)], 'Verification mutated model')
    return measured, exports, {'samples': count, 'changed_predictions': changed,
                      'max_abs_log_probability_error': max_error, 'atol': 1e-5, 'rtol': 1e-5,
                      'saved_validation_reproduced': True, 'model_state_unchanged': True}


def save_predictions(path, rows):
    with gzip.open(path, 'xt', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row)+'\n')


def selection_state(score, best, stale):
    improved = score > best
    return (score if improved else best), (0 if improved else stale+1), improved


def compare_predictions(reference, candidate):
    require(len(reference) == len(candidate) == 13446, 'Incomplete paired predictions')
    indexed = {r['sample_id']: r for r in reference}
    require(len(indexed) == 13446 and len({r['sample_id'] for r in candidate}) == 13446,
            'Duplicate validation IDs')
    counts = {k: 0 for k in ('repaired', 'regressed', 'both_correct', 'both_wrong')}
    by_class = {str(i): dict(counts) for i in range(8)}
    for row in candidate:
        old = indexed[row['sample_id']]
        require(old['label'] == row['label'], 'Paired label differs')
        a, b = old['prediction'] == row['label'], row['prediction'] == row['label']
        key = 'both_correct' if a and b else 'regressed' if a else 'repaired' if b else 'both_wrong'
        counts[key] += 1
        by_class[str(row['label'])][key] += 1
    return {'counts': counts, 'by_class': by_class,
            'net_correct': counts['repaired']-counts['regressed']}


def probe(core, model, frozen, device, start_rng, lr):
    clean = {n: t.detach().clone() for n, t in model.state_dict().items()}
    clean_hash = core.state_hash(clean.items())
    restore_rng(core, start_rng)
    optimizer = core.torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.)
    rows = core.select_rows(frozen, 'train')
    batches = list(islice(core.TrainBatches(len(rows), 512, 42), 3))
    loader = core.DataLoader(core.GraphRows(frozen, rows), batch_sampler=batches, num_workers=0)
    observations = []
    model.train()
    for batch in loader:
        batch = batch.to(device)
        before = model.shared_head[0].weight.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        loss = core.F.nll_loss(model(batch), batch.y.view(-1))
        require(bool(core.torch.isfinite(loss)), 'Nonfinite probe loss')
        loss.backward()
        require(all(p.grad is None or bool(core.torch.isfinite(p.grad).all())
                    for p in model.parameters()), 'Nonfinite probe gradients')
        norms = {}
        for prefix in ('flow_encoder.', 'packet_encoder.', 'flow_classifier.',
                       'packet_classifier.', 'shared_head.'):
            norms[prefix] = sum(float(p.grad.detach().abs().sum()) for n, p in model.named_parameters()
                                if n.startswith(prefix) and p.grad is not None)
            require(norms[prefix] > 0, 'Missing probe gradient: ' + prefix)
        require(all(p.grad is None for n, p in model.named_parameters()
                    if n.startswith(('gate.', 'webbased_head.', 'webbased_recon_head.',
                                     'benign_spoofing_head.'))), 'Unexpected auxiliary gradient')
        optimizer.step()
        require(not core.torch.equal(before, model.shared_head[0].weight), 'Head failed to update')
        observations.append({'loss': float(loss.detach()), 'gradient_l1': norms, 'head_updated': True})
    require(len(observations) == 3, 'Incomplete probe')
    model.load_state_dict(clean, strict=True)
    model.zero_grad(set_to_none=True)
    model._last_cache = {}
    model.eval()
    restore_rng(core, start_rng)
    require(core.state_hash(model.state_dict().items()) == clean_hash, 'Probe state was not discarded')
    return {'batches': observations, 'clean_start_restored': True, 'initial_state_sha256': clean_hash}


def train(core, args, spec, frozen, device, model, start_rng, initial, binding, variant, expected_orders,
          original_predictions):
    output = args.output / variant
    output.mkdir()
    restore_rng(core, start_rng)
    start_hash = rng_hash(core, rng_state(core))
    # Both arms discard the old Adam moments. This is a new matched fine-tune,
    # not an exact continuation of the historical optimizer trajectory.
    optimizer = core.torch.optim.Adam(model.parameters(), lr=spec['lr'], weight_decay=0.)
    rows = {p: core.select_rows(frozen, p) for p in ('train', 'val')}
    sampler = core.TrainBatches(len(rows['train']), 512, 42)
    kwargs = dict(num_workers=2, pin_memory=True, persistent_workers=True, multiprocessing_context='fork')
    loader = core.DataLoader(core.GraphRows(frozen, rows['train']), batch_sampler=sampler, **kwargs)
    val_loader = core.DataLoader(core.GraphRows(frozen, rows['val']), batch_size=16, shuffle=False, **kwargs)
    cfg = {'binding': binding, 'protocol': spec, 'variant': variant,
           'training_start_rng_sha256': start_hash, 'optimizer_state': 'reset in both arms',
           'total_parameters': sum(p.numel() for p in model.parameters()),
           'inactive_historical_gate_and_heads_retained': True}
    write(output / 'config.json', cfg)
    best, best_epoch, best_metrics = initial['macro_f1'], 0, initial
    core.checkpoint(output / 'best.pt', {'model_state': model.state_dict(), 'epoch': 0,
                    'validation': initial, 'config': cfg})
    orders, updates, stale = [], 0, 0
    with (output / 'epochs.jsonl').open('x', encoding='utf-8') as log:
        for epoch in range(1, spec['epochs']+1):
            model.train()
            count, loss_sum, steps = 0, 0., 0
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = core.F.nll_loss(model(batch), batch.y.view(-1))
                require(bool(core.torch.isfinite(loss)), 'Nonfinite loss')
                loss.backward()
                require(all(p.grad is None or bool(core.torch.isfinite(p.grad).all())
                            for p in model.parameters()), 'Nonfinite gradients')
                if epoch == 1 and steps == 0:
                    gradient_groups = {}
                    for prefix in ('flow_encoder.', 'packet_encoder.', 'flow_classifier.',
                                   'packet_classifier.', 'shared_head.'):
                        norm = sum(float(p.grad.detach().abs().sum()) for n, p in model.named_parameters()
                                   if n.startswith(prefix) and p.grad is not None)
                        require(norm > 0, 'Missing initial gradient: ' + prefix)
                        gradient_groups[prefix] = norm
                    require(all(p.grad is None for n, p in model.named_parameters()
                                if n.startswith(('gate.', 'webbased_head.', 'webbased_recon_head.',
                                                 'benign_spoofing_head.'))), 'Inactive head received gradient')
                    write(output/'first_update_gradients.json', gradient_groups)
                optimizer.step()
                count += batch.num_graphs; steps += 1
                loss_sum += float(loss.detach()) * batch.num_graphs
            require(count == 107621 and steps == 211, 'Incomplete epoch')
            updates += steps
            orders.append(sampler.last_order_sha256)
            if expected_orders is not None:
                require(orders[-1] == expected_orders[epoch-1], 'Paired sample order differs')
            validation, _ = core.evaluate(model, val_loader, device)
            next_best, stale, improved = selection_state(validation['macro_f1'], best, stale)
            if improved:
                best, best_epoch, best_metrics = validation['macro_f1'], epoch, validation
                core.checkpoint(output / 'best.pt', {'model_state': model.state_dict(), 'epoch': epoch,
                                'validation': validation, 'config': cfg})
            require(best == next_best, 'Selection state mismatch')
            payload = {'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
                       'epoch': epoch, 'best_epoch': best_epoch, 'best_val_macro_f1': best,
                       'validation': validation, 'config': cfg, 'rng': rng_state(core),
                       'sampler_rng_state': sampler.generator.get_state(), 'optimizer_updates': updates,
                       'stale_epochs': stale}
            core.checkpoint(output / 'last.pt', payload)
            entry = {'epoch': epoch, 'train_loss': loss_sum/count, 'train_samples': count,
                     'optimizer_updates': steps, 'cumulative_optimizer_updates': updates,
                     'train_order_sha256': orders[-1], 'val_macro_f1': validation['macro_f1'],
                     'val_accuracy': validation['accuracy'], 'best_epoch': best_epoch, 'stale_epochs': stale}
            log.write(json.dumps(entry)+'\n'); log.flush()
            print(json.dumps({'variant': variant, **entry}), flush=True)
            if stale >= spec['patience']:
                break
    chosen = core.torch.load(output / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(chosen['model_state'], strict=True)
    reproduced, exports = core.evaluate(model, val_loader, device, export_rows=True)
    require(reproduced['confusion_matrix'] == best_metrics['confusion_matrix'], 'Best validation not reproduced')
    save_predictions(output / 'validation_predictions.jsonl.gz', exports)
    result = {'variant': variant, 'best_epoch': best_epoch, 'best_validation': best_metrics,
              'last_validation': validation,
              'epochs': epoch, 'optimizer_updates': updates, 'training_samples_seen': 107621*epoch,
              'versus_original': compare_predictions(original_predictions, exports),
              'macro_f1_minus_original_pp': 100*(best_metrics['macro_f1']-initial['macro_f1']),
              'training_start_rng_sha256': start_hash,
              'orders': orders, 'test_evaluated': False,
              'predictions_sha256': sha(output/'validation_predictions.jsonl.gz'),
              'checkpoint_sha256': {n: sha(output/n) for n in ('best.pt', 'last.pt')}}
    write(output / 'metrics.json', result)
    write(output / 'status.json', {'status': 'completed', 'exit_code': 0})
    del loader, val_loader, optimizer
    gc.collect()
    return result


def analyze(output, arms=None):
    arms = arms or {v: output/v for v in VARIANTS}
    results = {v: read(arms[v]/'metrics.json') for v in VARIANTS}
    configs = {v: read(arms[v]/'config.json') for v in VARIANTS}
    require(configs['N0']['binding'] == configs['N1']['binding'], 'Pair bindings differ')
    for v in VARIANTS:
        require(read(arms[v]/'status.json') == {'status': 'completed', 'exit_code': 0}, 'Incomplete arm')
        require(configs[v]['variant'] == results[v]['variant'] == v, 'Wrong arm identity')
        require(sha(arms[v]/'validation_predictions.jsonl.gz') == results[v]['predictions_sha256'], 'Predictions changed')
        for name, digest in results[v]['checkpoint_sha256'].items():
            require(sha(arms[v]/name) == digest, 'Checkpoint changed')
    common = min(len(results[v]['orders']) for v in VARIANTS)
    require(results['N0']['orders'][:common] == results['N1']['orders'][:common], 'Sample orders differ')
    require(results['N0']['training_start_rng_sha256'] == results['N1']['training_start_rng_sha256'],
            'Paired start RNG differs')
    def load_predictions(v):
        with gzip.open(arms[v]/'validation_predictions.jsonl.gz', 'rt', encoding='utf-8') as stream:
            return [json.loads(line) for line in stream]
    delta = 100*(results['N1']['best_validation']['macro_f1']-results['N0']['best_validation']['macro_f1'])
    summary = {'status': 'completed', 'exit_code': 0, 'binding': configs['N0']['binding'], 'models': results,
        'N1_minus_N0_macro_f1_pp': delta, 'N1_versus_N0': compare_predictions(load_predictions('N0'),load_predictions('N1')),
        'test_evaluated': False, 'paired_common_epochs': common, 'paired_sample_order_verified': True,
        'automatic_followup': False}
    write(output/'summary.json', summary)
    write(output/'status.json', {'status': 'completed', 'exit_code': 0})
    return summary


def main(default_variant=None, default_mode='check'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--config', type=Path, default=Path('configs/e2_concat_pair_20261028.json'))
    parser.add_argument('--mode', choices=('check', 'probe', 'train', 'queue', 'analyze'), default=default_mode)
    parser.add_argument('--variant', choices=VARIANTS, default=default_variant)
    parser.add_argument('--physical-gpu', type=int)
    parser.add_argument('--allow-shared-gpu', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--n0-output', type=Path, help='N0 arm directory for independent-run analysis')
    parser.add_argument('--n1-output', type=Path, help='N1 arm directory for independent-run analysis')
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    args.config = (args.repo / args.config).resolve()
    spec, registry, ref, folder, binding = metadata(args)
    if args.mode == 'analyze':
        require(args.output is not None, 'Output required')
        output = inside(args.repo/'outputs', inside(args.repo, args.output))
        arms = None
        if args.n0_output or args.n1_output:
            require(args.n0_output and args.n1_output, 'Both independent arm directories required')
            arms = {v: inside(args.repo/'outputs', inside(args.repo, p)) for v, p in
                    zip(VARIANTS, (args.n0_output, args.n1_output))}
            require(not output.exists(), 'Independent analysis requires a new output')
            output.mkdir(parents=True)
        analyze(output, arms)
        return
    if args.mode == 'check':
        print(json.dumps({'status': 'metadata_checks_passed', 'binding': binding, 'training_started': False}))
        return
    require(os.name == 'posix', 'Use Linux wx training server')
    require(args.mode != 'train' or args.variant in VARIANTS, 'Independent training needs --variant')
    require(args.physical_gpu is not None and args.output is not None, 'Explicit GPU and output required')
    args.output = inside(args.repo / 'outputs', inside(args.repo, args.output))
    require(not args.output.exists(), 'Refuse to overwrite or resume an existing run')
    gpu = checked_gpu(args.physical_gpu, args.allow_shared_gpu)
    activate_gpu(gpu)
    args.output.mkdir(parents=True)
    write(args.output / 'status.json', {'status': 'running', 'pid': os.getpid()})
    try:
        sys.path.insert(0, str(args.repo/'scripts'))
        import train_grouped as core
        device = runtime(core)
        frozen = core.FrozenGraphs(inside(args.repo, registry['split_directory']), registry['split_id'])
        guard_graph_loads(frozen, {'train', 'val'})
        write(args.output/'status.json', {'status': 'running', 'phase': 'initial_equivalence', 'pid': os.getpid()})
        core.seed_all(42)
        warm = core.Batch.from_data_list([frozen.load(r) for r in
                core.select_rows(frozen, 'train')[:16]]).to(device)
        base = core.build_experiment('E2', warm).to(device)
        core.initialize(base, 'E2', warm)
        checkpoint = core.torch.load(folder/'E2/best.pt', map_location=device, weights_only=False)
        base.load_state_dict(checkpoint['model_state'], strict=True)
        require(sum(p.numel() for p in base.parameters()) == 846800 and not base.use_gate, 'Wrong E2 model')
        start_rng = rng_state(core)
        # Build candidate before the verification pass; restore shared RNG for training.
        candidate = convert(core, copy.deepcopy(base))
        validation_loader = core.DataLoader(core.GraphRows(frozen, core.select_rows(frozen, 'val')),
                                            batch_size=16, shuffle=False, num_workers=2)
        initial, original_predictions, receipt = verify_initial(core, base, candidate, validation_loader, device,
                                         read(folder/'E2/metrics.json')['best_validation'])
        write(args.output/'initial_equivalence.json', receipt)
        print(json.dumps({'phase': 'initial_equivalence_passed', **receipt}), flush=True)
        save_predictions(args.output/'original_validation_predictions.jsonl.gz', original_predictions)
        models = dict(zip(VARIANTS, (base, candidate)))
        write(args.output/'status.json', {'status': 'running', 'phase': 'disposable_probe', 'pid': os.getpid()})
        probes = {v: probe(core, m, frozen, device, start_rng, spec['lr']) for v, m in models.items()}
        write(args.output/'probe.json', {'status': 'completed', 'variants': probes})
        print(json.dumps({'phase': 'disposable_probe_passed', 'clean_starts_restored': True}), flush=True)
        if args.mode == 'probe':
            write(args.output/'status.json', {'status': 'completed', 'exit_code': 0, 'mode': 'probe'})
            return
        results = {}
        rows = core.select_rows(frozen, 'train')
        schedule = core.TrainBatches(len(rows),512,42)
        orders = []
        for _ in range(spec['epochs']):
            for _batch in schedule:
                pass
            orders.append(schedule.last_order_sha256)
        write(args.output/'sample_order_schedule.json', orders)
        for variant in VARIANTS if args.mode == 'queue' else (args.variant,):
            model = models[variant]
            write(args.output/'status.json', {'status': 'running', 'phase': 'training',
                                            'variant': variant, 'pid': os.getpid()})
            results[variant] = train(core, args, spec, frozen, device, model, start_rng,
                                     initial, binding, variant, orders, original_predictions)
        write(args.output/'hardware.json', gpu)
        if args.mode == 'queue':
            analyze(args.output)
        else:
            write(args.output/'status.json', {'status': 'completed', 'exit_code': 0, 'variant': args.variant})
    except BaseException:
        write(args.output/'status.json', {'status': 'failed', 'exit_code': 1})
        (args.output/'error.log').write_text(traceback.format_exc(), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
