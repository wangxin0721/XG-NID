"""One fixed-150 P0/P1 confirmation of the earlier patience20 screen."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from types import FunctionType

from e3_gate_study import (activate_gpu, checked_gpu, guard_graph_loads, inside,
                           read, require, runtime, sha, write)
import run_packet_pool_pair as packet
import run_relation_pair as fixed
from summarize_packet_pool_pair import analyze as analyze_screen

HERE = Path(__file__).resolve().parent
VARIANTS = packet.VARIANTS
dim, pair = packet.dim, packet.pair
architecture = packet.architecture
selection_step, check_history = fixed.selection_step, fixed.check_history


def check_config(c):
    expected = read(HERE.parent / 'configs/packet_pool_pair_s42_v1.json')
    expected.update(schema='xgnid.packet-pool-confirmation.v1', epochs=150,
        min_epochs=50, early_stop_enabled=False, early_stop_reference='trained_epochs_only',
        final_selection='epoch_zero_or_best_trained',
        purpose='confirm_effect_of_patience20_only', confirmation_rounds=1,
        previous_screen='outputs/packet_pool_pair_s42_recovery_v2',
        restart='mature_D32_epoch62_fresh_Adam_not_epoch20_resume')
    require(c == expected, 'Unregistered single-round packet confirmation protocol')


# Reuse the fixed-budget loop with packet-specific operators in its own
# namespace. Neither historical runner is mutated, including its train globals.
train = FunctionType(fixed.train.__code__, dict(fixed.train.__globals__,
    architecture=architecture, evaluate=packet.evaluate, optimizer=packet.optimizer,
    minibatch=packet.minibatch), 'train_packet_confirmation', fixed.train.__defaults__)


def assert_previous_unchanged(receipt):
    root = Path(receipt['directory'])
    names = sorted(p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file())
    require(names == sorted(receipt['file_sha256']), 'Previous completed screen inventory changed')
    for name, digest in receipt['file_sha256'].items():
        require(sha(root / name) == digest, 'Previous completed screen file changed: ' + name)


def previous_receipt(root, meta):
    c, previous, _, _, binding = meta
    require(read(root/'status.json') == {'status': 'completed', 'exit_code': 0}, 'Previous P0/P1 screen incomplete')
    summary = read(root/'summary.json')
    analyze_screen(summary)
    require(summary['protocol'] == read(root/'config.json') == read(HERE.parent/'configs/packet_pool_pair_s42_v1.json')
            and summary['binding'] == read(root/'binding.json'), 'Previous screen protocol/binding changed')
    require(summary['binding']['source'] == binding['source']
            and summary['binding']['source_pair_binding'] == binding['source_pair_binding'], 'Previous mature source differs')
    for name, digest in summary['binding']['sources'].items():
        require(sha(inside(HERE.parent, name)) == digest, 'Previous implementation differs: ' + name)
    histories = {}
    for name in VARIANTS:
        model = summary['models'][name]
        history = [json.loads(x) for x in (root/name/'epochs.jsonl').read_text(encoding='utf-8').splitlines()]
        packet.check_history(history, model, summary['protocol'])
        require(model == read(root/name/'metrics.json') and model['epochs'] == 20 and model['best_epoch'] == 0,
                'Confirmation requires the recorded epoch20/epoch0-selected screen')
        for file, digest in model['checkpoint_sha256'].items():
            require(sha(root/name/file) == digest, 'Previous checkpoint changed')
        require(sha(root/name/'validation_predictions.jsonl.gz') == model['predictions_sha256'], 'Previous predictions changed')
        histories[name] = history
    receipts = read(root/'cache_receipts.json')
    require(receipts == summary['cache_receipts'] and set(receipts) == {'train', 'val'}, 'Previous cache receipts differ')
    encoder_hash = previous[3]['binding']['encoder_source']['encoder_sha256']
    split = previous[1][2]['binding']
    for part, count in (('train', 107621), ('val', 13446)):
        r = receipts[part]
        h = r['metadata']
        require(h['schema'] == 'xgnid.packet-node64-cache.v1' and h['partition'] == part
                and h['samples'] == count and h['encoder_sha256'] == encoder_hash and h['split_binding'] == split
                and bool(h['sample_order_sha256']) and bool(r['tensor_sha256'])
                and sha(root/(part+'_packet_nodes.pt')) == r['sha256'], 'Previous frozen node cache changed')
    return {'directory': str(root), 'status': 'completed', 'models': summary['models'],
        'histories': histories, 'cache_receipts': receipts,
        'file_sha256': {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob('*')) if p.is_file()},
        'reuse_scope': 'frozen_node_cache_only_no_model_or_optimizer_resume'}


def metadata(args):
    env = dict(packet.metadata.__globals__, check_config=check_config)
    meta = list(FunctionType(packet.metadata.__code__, env, 'confirmation_metadata', packet.metadata.__defaults__)(args))
    root = inside(args.repo/'outputs', inside(args.repo, meta[0]['previous_screen']))
    require(args.output != root and root not in args.output.parents and args.output not in root.parents,
            'Refuse nested/overwritten historical output')
    receipt = previous_receipt(root, meta)
    meta[-1]['completed_cache_source'] = receipt
    names = ('scripts/run_packet_pool_confirmation.py', 'scripts/summarize_packet_pool_confirmation.py',
        'scripts/run_relation_pair.py', 'scripts/run_packet_pool_pair.py',
        'scripts/summarize_packet_pool_pair.py', 'src/xgnid/packet_pooling.py',
        'scripts/test_packet_pooling_tensor.py', 'scripts/test_packet_pool_confirmation.py',
        'scripts/test_packet_pool_confirmation_tensor.py', 'scripts/launch_packet_pool_confirmation.sh',
        'docs/packet_pool_results_20261102.json')
    meta[-1]['sources'] = {name: sha(HERE.parent/name) for name in names}
    return tuple(meta)


def cached_nodes(receipt):
    def reuse(core, args, model, data, device, part, c, encoder_hash, split_binding, retained_bytes=0):
        started = time.perf_counter()
        source = Path(receipt['directory'])/(part+'_packet_nodes.pt')
        target = args.output/source.name
        old = receipt['cache_receipts'][part]
        require(not target.exists() and sha(source) == old['sha256'], 'Refuse changed cache or overwrite')
        dim.progress(args, 'copy_completed_frozen_node_cache', partition=part)
        shutil.copyfile(source, target)
        require(sha(target) == old['sha256'], 'Copied cache differs')
        cache = core.torch.load(target, map_location='cpu', weights_only=False)
        packet.validate_cache(core, cache, core.select_rows(data, part), part, encoder_hash, split_binding)
        require(packet.tensor_hash(core, cache) == old['tensor_sha256'] and cache['metadata'] == old['metadata'], 'Cache tensors differ')
        dim.assert_frozen(core, model, encoder_hash)
        return cache, {**copy.deepcopy(old), 'reused': True, 'reused_from': str(source),
            'wall_seconds': time.perf_counter()-started,
            'original_extraction_wall_seconds': old.get('original_extraction_wall_seconds', old['wall_seconds']),
            'memory_estimate_scope': 'original_extraction_not_current_copy_or_whole_process_RSS'}
    return reuse


def prefix_comparison(history, old, baseline):
    require(len(history) == 150 and len(old) == 20, 'Incomplete confirmation/previous histories')
    order = all(a['train_order_sha256'] == b['train_order_sha256'] for a,b in zip(history,old))
    scores = all(a['val_macro_f1'] == b['val_macro_f1'] for a,b in zip(history,old))
    nll = all(a['train_nll'] == b['train_nll'] for a,b in zip(history,old))
    reproduced = order and scores and nll
    prefix = max([baseline] + [r['val_macro_f1'] for r in history[:20]])
    full = max([baseline] + [r['val_macro_f1'] for r in history])
    return {'first20_orders_identical': order, 'first20_validation_scores_identical': scores,
        'first20_train_nll_max_abs_error': max(abs(a['train_nll']-b['train_nll']) for a,b in zip(history,old)),
        'first20_exact_train_nll': nll,
        'first20_selected_macro_f1': prefix, 'full150_selected_macro_f1': full,
        'gain_after20_pp': 100*(full-prefix), 'first20_reproduced': reproduced,
        'best_trained_epoch': max(history, key=lambda r:r['val_macro_f1'])['epoch'],
        'interpretation': 'matched_prefix_budget_confirmation' if reproduced else 'prefix_differs_do_not_attribute_solely_to_early_stopping'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path.cwd())
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=('check', 'run'), default='check')
    p.add_argument('--physical-gpu', type=int, default=1)
    p.add_argument('--allow-shared-gpu', action='store_true')
    args = p.parse_args()
    args.repo, args.config = args.repo.resolve(), args.config.resolve()
    args.output = inside(args.repo/'outputs', inside(args.repo, args.output))
    require(not args.output.exists(), 'Refuse overwrite/resume/additional confirmation round')
    meta = metadata(args)
    if args.mode == 'check':
        print(json.dumps({'status': 'confirmation_metadata_checks_passed', 'epochs_per_arm': 150,
            'early_stop_enabled': False, 'previous_screen': meta[-1]['completed_cache_source']['directory'],
            'training_started': False}), flush=True)
        return
    require(os.name == 'posix', 'Formal confirmation runs on Linux wx server only')
    args.output.mkdir(parents=True)
    write(args.output/'config.json', meta[0]);write(args.output/'binding.json', meta[-1])
    write(args.output/'status.json', {'status': 'running', 'phase': 'preflight'})
    try:
        run(args, meta)
        write(args.output/'status.json', {'status': 'completed', 'exit_code': 0})
        write(args.output/'progress.json', {'status': 'completed', 'training_jobs': 2, 'confirmation_rounds': 1})
    except BaseException:
        write(args.output/'status.json', {'status': 'failed', 'exit_code': 1})
        write(args.output/'progress.json', {'status': 'failed', 'exit_code': 1})
        (args.output/'error.log').write_text(traceback.format_exc(), encoding='utf-8')
        raise


def run(args, meta):
    receipt = meta[-1]["completed_cache_source"]
    cache_provider = cached_nodes(receipt)
    verify_cache_source = lambda: assert_previous_unchanged(receipt)
    started = time.perf_counter()
    c, previous, root, source, binding = meta
    _, spec, ref, nroot, _, _ = previous[1]
    gpu = checked_gpu(args.physical_gpu, args.allow_shared_gpu)
    activate_gpu(gpu)
    # The archived entry uses the server checkout only for frozen core helpers.
    scripts = str(args.repo / 'scripts')
    sys.path.insert(0, scripts)
    try:
        import train_grouped as core
    finally:
        sys.path.remove(scripts)
    device = runtime(core)
    write(args.output / 'gpu.json', gpu)
    data = core.FrozenGraphs(inside(args.repo, spec['split_directory']), ref['binding']['split_id'])
    main_load_counts = guard_graph_loads(data, {'train', 'val'})
    n0 = core.torch.load(nroot / 'N0/best.pt', map_location=device, weights_only=False)
    mature = core.torch.load(root / 'D32-Gate/best.pt', map_location=device, weights_only=False)
    expected = source['models']['D32-Gate']['best_validation']
    require(mature['epoch'] == c['source_epoch'] and mature['validation']['confusion_matrix'] == expected['confusion_matrix']
            and mature['config']['binding'] == source['binding'] and mature['config']['variant'] == 'D32-Gate'
            and mature['config']['labels'] == list(core.LABELS) and mature['config']['protocol'] == previous[0]
            and mature['config']['architecture'] == packet.source_pair.architecture('dynamic'),
            'Mature checkpoint payload differs')
    classes = packet.load_classes()
    models = {name: packet.build(core, classes, data, device, n0, mature['model_state'], mode)
              for name, mode in VARIANTS.items()}
    encoder_hash = core.state_hash(dim.encoder_values(models['P0']))
    require(encoder_hash == previous[3]['binding']['encoder_source']['encoder_sha256'], 'Source encoders changed')
    for model in models.values():
        dim.assert_frozen(core, model, encoder_hash)
        packet.optimizer(core, model, c)
    device_preflight = packet.cuda_readout_preflight(core, models, device, args.output)
    caches, receipts = {}, {}
    prepare_cache = cache_provider
    for part in ('train', 'val'):
        retained_bytes = sum(cache[k].numel() * cache[k].element_size() for cache in caches.values() for k in packet.TENSORS)
        caches[part], receipts[part] = prepare_cache(core, args, models['P0'], data, device, part, c, encoder_hash,
                                                    ref['binding'], retained_bytes)
        packet.validate_cache(core, caches[part], core.select_rows(data, part), part, encoder_hash, ref['binding'])
    write(args.output / 'cache_receipts.json', receipts)
    write(args.output / 'cache_memory_estimates.json', {part: {**r['memory_estimate'],
        'measurement_scope': 'original_extraction_not_current_copy_or_process_RSS'} for part,r in receipts.items()})
    write(args.output / 'packet_diagnostics.json', {p: packet.diagnostics(core, x) for p, x in caches.items()})
    gold = dim.predictions(root / 'D32-Gate/validation_predictions.jsonl.gz')
    initialization = packet.initial_equivalence(core, models, caches['val'], device, gold, expected, args.output)
    # Use the unchanged historical class as an independent raw-graph reference.
    _, Original = dim.load_classes()
    original = Original(models['P0'].flow_encoder, models['P0'].packet_encoder, models['P0'].fusion).eval()
    initial_replays = {}
    for name, model in models.items():
        initial_replays[name], _ = packet.graph_replay(core, args, model, data, caches['val'], device, c,
                                               expected, encoder_hash, original, phase='initial_' + name)
    del original
    initialization['raw_graph_replays'] = initial_replays
    write(args.output / 'initialization.json', initialization)
    probes = {}
    for name, model in models.items():
        dim.progress(args, 'raw_training_gradient_probe', variant=name)
        probes[name] = packet.probe(core, args, model, data, caches['train'], device, c, encoder_hash)
        path = args.output / (name + '_preflight.pt')
        core.checkpoint(path, {'model_state': model.state_dict(), 'architecture': architecture(model.pooling)})
        fresh = packet.build(core, classes, data, device, n0, mature['model_state'], model.pooling)
        fresh.load_state_dict(core.torch.load(path, map_location=device, weights_only=False)['model_state'], strict=True)
        features, _ = packet.minibatch(core, caches['val'], list(range(16)), device)
        with core.torch.no_grad():
            require(core.torch.allclose(model.forward_features(*features)[0], fresh.forward_features(*features)[0],
                                       atol=1e-6, rtol=1e-6), 'Preflight checkpoint roundtrip failed')
        dim.assert_frozen(core, fresh, encoder_hash)
        del fresh
    write(args.output / 'probes.json', probes)
    packet.source_pair.assert_source_unchanged(binding)
    verify_cache_source()
    preflight_seconds = time.perf_counter() - started
    write(args.output / 'status.json', {'status': 'running', 'phase': 'training', 'tensor_preflight_passed': True})
    results, exports, replays, trained_replays = {}, {}, {}, {}
    for name, model in models.items():
        orders = results['P0']['orders'] if name == 'P1' else None
        results[name], exports[name] = train(core, args, model, caches, device, c, binding, encoder_hash, name, orders)
        saved = core.torch.load(args.output / name / 'best.pt', map_location=device, weights_only=False)
        require(saved['config']['binding'] == binding and saved['config']['variant'] == name
                and saved['config']['architecture'] == architecture(model.pooling), 'Best checkpoint identity differs')
        replay = packet.build(core, classes, data, device, n0, mature['model_state'], model.pooling)
        replay.load_state_dict(saved['model_state'], strict=True)
        replays[name], raw = packet.graph_replay(core, args, replay, data, caches['val'], device, c,
            saved['validation'], encoder_hash, phase='best_' + name)
        require([r['prediction'] for r in raw] == [r['prediction'] for r in exports[name]], 'Best raw predictions differ')
        write(args.output / name / 'graph_replay.json', replays[name])
        del replay
        trained = core.torch.load(args.output / name / 'trained_best.pt', map_location=device, weights_only=False)
        require(trained['epoch'] == results[name]['best_trained_epoch']
                and sha(args.output/name/'trained_best.pt') == results[name]['checkpoint_sha256']['trained_best.pt'],
                'Trained-best checkpoint identity differs')
        if trained['epoch'] == saved['epoch']:
            require(core.state_hash(trained['model_state'].items()) == core.state_hash(saved['model_state'].items()),
                    'Same selected/trained epoch has different model state')
            trained_replays[name] = {**replays[name], 'shared_with_selected_best': True}
        else:
            replay = packet.build(core, classes, data, device, n0, mature['model_state'], model.pooling)
            replay.load_state_dict(trained['model_state'], strict=True)
            trained_replays[name], raw = packet.graph_replay(core, args, replay, data, caches['val'], device, c,
                trained['validation'], encoder_hash, phase='trained_best_' + name)
            trained_exports = dim.predictions(args.output / name / 'trained_best_validation_predictions.jsonl.gz')
            require([r['prediction'] for r in raw] == [r['prediction'] for r in trained_exports], 'Trained-best raw predictions differ')
            trained_replays[name]['shared_with_selected_best'] = False
            del replay
        write(args.output / name / 'trained_best_graph_replay.json', trained_replays[name])
    common = min(len(m['orders']) for m in results.values())
    require(results['P0']['orders'][:common] == results['P1']['orders'][:common]
            and results['P0']['training_start_rng_sha256'] == results['P1']['training_start_rng_sha256'], 'Pair RNG differs')
    for part, cache in caches.items():
        require(packet.tensor_hash(core, cache) == receipts[part]['tensor_sha256']
                and sha(args.output / (part + '_packet_nodes.pt')) == receipts[part]['sha256'], 'Node cache changed')
    for model in models.values():
        dim.assert_frozen(core, model, encoder_hash)
    packet.source_pair.assert_source_unchanged(binding)
    verify_cache_source()
    from summarize_joint_fusion32 import changed_samples
    from summarize_fusion_gate_pair import prediction_transitions
    from summarize_packet_pool_confirmation import summarize
    comparisons = {'P1_vs_P0': (exports['P0'], exports['P1']),
                   'P0_vs_D32': (gold, exports['P0']), 'P1_vs_D32': (gold, exports['P1'])}
    changes, transitions = {}, {}
    for name, (a, b) in comparisons.items():
        changes[name] = pair.compare_predictions(a, b)
        transitions[name] = prediction_transitions(a, b, core.LABELS)
        pair.save_predictions(args.output / (name + '_changed_samples.jsonl.gz'), changed_samples(a, b, core.LABELS))
    stages = [r['graph_load_stage'] for group in (receipts, initial_replays, replays)
              for r in group.values() if not r.get('reused', False)]
    stages += [r['graph_load_stage'] for r in trained_replays.values() if not r['shared_with_selected_best']]
    accounting = packet.graph_load_accounting(main_load_counts, stages)
    write(args.output / 'graph_load_accounting.json', accounting)
    summary = {'status': 'completed', 'exit_code': 0, 'protocol': c, 'binding': binding,
        'source_model': source['models']['D32-Gate'], 'models': results, 'initialization': initialization,
        'cuda_readout_preflight': device_preflight,
        'probes': probes, 'packet_diagnostics': read(args.output / 'packet_diagnostics.json'),
        'comparisons': changes, 'prediction_transitions': transitions, 'graph_replays': replays,
        'common_epochs_order_checked': common, 'encoder_parameters_and_BN_unchanged': True,
        'node_caches_unchanged': True, 'source_artifacts_unchanged': True, 'tensor_preflight_passed': True,
        'preflight_wall_seconds': preflight_seconds, 'cache_receipts': receipts,
        'graph_load_counts': accounting['totals'], 'graph_load_accounting': accounting,
        'cache_memory_estimates': {part: r['memory_estimate'] for part, r in receipts.items()},
        'cache_reused': cache_provider is not None,
        'completed_cache_source': receipt,
        'trained_best_graph_replays': trained_replays,
        'budget_confirmation': {name: prefix_comparison([json.loads(x) for x in
            (args.output/name/'epochs.jsonl').read_text(encoding='utf-8').splitlines()],
            receipt['histories'][name], results[name]['epoch_zero']['macro_f1']) for name in VARIANTS},
        'experiment_kind': 'one_round_training_budget_confirmation_not_new_model',
        'discarded_probe_updates': 6, 'training_jobs': 2, 'test_evaluated': False, 'automatic_followup': False}
    write(args.output / 'summary.json', summary)
    print(summarize(args.output), flush=True)


if __name__ == '__main__':
    main()
