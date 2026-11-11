"""Finish the interrupted seed44 Joint-Flow run and reuse the three finished arms."""
import argparse
import copy
import json
import shutil
import time
from pathlib import Path

import torch
import torch_geometric

import run_flow_query_joint_seeds as study

core, raw, require = study.core, study.raw, study.require
ARM = 'Joint-Flow'
ORIGINAL_RUN = 'outputs/flow_query_joint_seeds43_44_v1'
SNAPSHOTS = ('epoch_zero.pt', 'trained_best.pt', 'best150.pt', 'epoch150.pt', 'resume.pt', 'last.pt')


def complete_history(path):
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines(keepends=True)
    if lines and not lines[-1].endswith(b'\n'):
        lines.pop()
    return [json.loads(line) for line in lines]


def choose_checkpoint(folder, config, orders, snapshots):
    history = complete_history(folder/'epochs.jsonl')
    require(len(history) <= config['epochs'] and [r['epoch'] for r in history] == list(range(1, len(history)+1)),
            'Interrupted ledger has invalid epoch sequence')
    require([r['train_order_sha256'] for r in history] == orders[:len(history)], 'Interrupted ledger order differs')
    zero = snapshots['epoch_zero.pt']['validation']
    # An older trainer could write a new best checkpoint just before its ledger
    # row. A checkpoint needs a complete matching row to reuse that epoch.
    usable = {}
    for name, saved in snapshots.items():
        epoch = saved['epoch']
        if 0 <= epoch <= len(history):
            expected = zero if epoch == 0 else history[epoch-1]['validation']
            if saved['validation'] == expected:
                usable[epoch] = name
    for epoch, name in sorted(usable.items(), reverse=True):
        prefix = history[:epoch]
        best = max(prefix, key=lambda row: row['validation']['macro_f1'], default=None)
        if best and best['epoch'] not in usable:
            continue
        keep = {'epoch_zero.pt': 'epoch_zero.pt'}
        if best:
            keep['trained_best.pt'] = usable[best['epoch']]
        limit = config['prefix_epochs']
        if epoch >= limit:
            _, best_prefix_epoch = core.prefix_best(zero, prefix, limit)
            if limit not in usable or best_prefix_epoch not in usable:
                continue
            keep['epoch150.pt'] = usable[limit]
            keep['best150.pt'] = usable[best_prefix_epoch]
        return name, prefix, keep, len(history)
    raise ValueError('No checkpoint with a consistent ledger and selection state')


def atomic_copy(source, destination):
    temp = destination.with_suffix('.copying')
    shutil.copyfile(source, temp)
    temp.replace(destination)


def checkpoint_config(saved, config, initial):
    cfg = saved['config']
    require(cfg['variant'] == ARM and cfg['protocol'] == config and cfg['labels'] == study.LABELS
            and cfg['initial_state_sha256'] == initial, 'Interrupted checkpoint configuration differs')
    return cfg


def restore(model, opt, sampler, saved, train_loader, val_loader):
    epoch = saved['epoch']
    model.load_state_dict(saved['model_state'], strict=True)
    state = saved['optimizer_state']
    core.assert_tree_equal(opt.state_dict()['param_groups'], state['param_groups'], 'Adam settings')
    opt.load_state_dict(copy.deepcopy(state))
    core.assert_tree_equal(opt.state_dict(), state, 'restored Adam')
    count = sum(len(group['params']) for group in opt.param_groups) if epoch else 0
    require(len(opt.state) == count and all(int(value['step']) == epoch*len(sampler) for value in opt.state.values()),
            'Interrupted Adam update counters differ')
    sampler.generator.set_state(saved['sampler_rng'].cpu())
    # The pinned PyTorch loader consumes one worker seed per iterator. The
    # validation loader has also run at epoch zero.
    for loader, iterations in ((train_loader, epoch), (val_loader, epoch+1)):
        for _ in range(iterations):
            torch.empty((), dtype=torch.int64).random_(generator=loader.generator)
    core.restore_rng(saved['rng'])


def recover_arm(previous, folder, model, data, rows, device, config, binding, orders):
    source = previous/'seed44'/ARM
    initial = raw.state_digest(model.state_dict())
    frozen = core.frozen_hash(model)
    snapshots = {}
    for name in SNAPSHOTS:
        path = source/name
        if path.is_file():
            saved = torch.load(path, map_location='cpu', weights_only=False)
            checkpoint_config(saved, config, initial)
            snapshots[name] = saved
    require('epoch_zero.pt' in snapshots, 'Interrupted arm has no epoch-zero checkpoint')
    name, history, keep, logged_epoch = choose_checkpoint(source, config, orders, snapshots)
    saved = snapshots[name]
    start_epoch = saved['epoch']
    folder.mkdir()
    for destination, original in keep.items():
        shutil.copyfile(source/original, folder/destination)
    zero = snapshots['epoch_zero.pt']['validation']
    rng = raw.rng_hash(snapshots['epoch_zero.pt']['rng'])
    cfg = {**saved['config'], 'binding': binding}
    train_rows, val_rows = rows['train'], rows['val']
    sampler = raw.TrainBatches(len(train_rows), config['batch_size'], config['seed'])
    train_loader = core.graph_loader(data, train_rows, 'train', config, sampler)
    val_loader = core.graph_loader(data, val_rows, 'val', config)
    opt = core.joint_optimizer(model, config)
    restore(model, opt, sampler, saved, train_loader, val_loader)
    require(core.frozen_hash(model) == frozen, 'Checkpoint frozen parameters/BN differ from D32')
    validation_loader = core.graph_loader(data, val_rows, 'val', config)
    print(json.dumps({'last_complete_ledger_epoch': logged_epoch, 'checkpoint_epoch': start_epoch,
                      'checkpoint': str(source/name)}), flush=True)
    restored, _ = core.joint_evaluate(model, validation_loader, val_rows, device, folder.parent, ARM+'_resume_validation')
    require(restored == saved['validation'], 'Recovered checkpoint validation differs')
    core.restore_rng(saved['rng'])
    selected, selected_epoch = core.prefix_best(zero, history, start_epoch)
    trained_row = max(history, key=lambda row: row['validation']['macro_f1'], default=None)
    trained = trained_row['validation'] if trained_row else None
    trained_epoch = trained_row['epoch'] if trained_row else None
    metrics = restored
    if history:
        partial = {'epoch_zero': zero, 'best_epoch': selected_epoch, 'best_validation': selected,
                   'best_trained_epoch': trained_epoch, 'trained_best_validation': trained,
                   'last_validation': metrics, 'epochs': start_epoch,
                   'optimizer_updates': start_epoch*len(sampler), 'training_samples_per_epoch': len(train_rows),
                   'orders': orders[:start_epoch]}
        core.parent.check_history(history, partial, start_epoch, len(sampler), len(val_rows))
    receipt = {'source_checkpoint': str(source/name), 'source_checkpoint_sha256': raw.sha(source/name),
               'last_complete_ledger_epoch': logged_epoch, 'resumed_from_epoch': start_epoch,
               'optimizer_updates_performed_after_recovery': (config['epochs']-start_epoch)*len(sampler)}
    raw.write(folder/'resume_receipt.json', receipt)
    print(json.dumps(receipt), flush=True)
    started = time.perf_counter()
    with (folder/'epochs.jsonl').open('x', encoding='utf-8', newline='\n') as ledger:
        for row in history:
            ledger.write(json.dumps(row)+'\n')
        ledger.flush()
        atomic_copy(source/name, folder/'resume.pt')
        for epoch in range(start_epoch+1, config['epochs']+1):
            model.train()
            total, count, steps = 0., 0, 0
            for batch in train_loader:
                if steps == 0:
                    require(sampler.order_sha256 == orders[epoch-1], 'Recovered training order differs')
                expected = [train_rows[i] for i in sampler.order[count:count+batch.num_graphs]]
                require(list(zip(batch.sample_id, batch.y.view(-1).tolist())) ==
                        [(row['sample_id'], row['label']) for row in expected], 'Recovered training batch IDs/labels differ')
                total += core.joint_step(model, batch.to(device), opt)*batch.num_graphs
                count += batch.num_graphs
                steps += 1
                if steps == 1 or steps % 50 == 0:
                    raw.progress(folder.parent, ARM+'_train', epoch=epoch, samples=count, total=len(train_rows))
            require(count == len(train_rows) and steps == len(sampler) and core.frozen_hash(model) == frozen,
                    'Recovered training coverage or frozen state differs')
            metrics, _ = core.joint_evaluate(model, val_loader, val_rows, device, folder.parent, ARM+'_val')
            improved = trained is None or metrics['macro_f1'] > trained['macro_f1']
            if improved:
                trained, trained_epoch = metrics, epoch
            if metrics['macro_f1'] > selected['macro_f1']:
                selected, selected_epoch = metrics, epoch
            row = {'epoch': epoch, 'validation': metrics, 'best_epoch': selected_epoch,
                   'best_trained_epoch': trained_epoch, 'samples': count, 'optimizer_updates': steps,
                   'train_nll': total/count, 'train_order_sha256': sampler.order_sha256}
            history.append(row)
            ledger.write(json.dumps(row)+'\n')
            ledger.flush()
            # Save current state before replacing best. If interrupted between
            # these writes, resume.pt itself supplies the improved epoch.
            latest = folder/'resume.pt'
            raw.save_checkpoint(latest, model, opt, epoch, metrics, cfg, sampler)
            if improved:
                atomic_copy(latest, folder/'trained_best.pt')
            if epoch == config['prefix_epochs']:
                atomic_copy(latest, folder/'epoch150.pt')
                selected_file = 'epoch_zero.pt' if selected_epoch == 0 else 'trained_best.pt'
                atomic_copy(folder/selected_file, folder/'best150.pt')
            print(json.dumps({'model': ARM, 'epoch': epoch, 'val_macro_f1': metrics['macro_f1'],
                              'best_epoch': selected_epoch, 'checkpoint': str(latest)}), flush=True)
    updates = config['epochs']*len(sampler)
    require(all(int(state['step']) == updates for state in opt.state.values()), 'Final recovered Adam steps differ')
    raw.save_checkpoint(folder/'last.pt', model, opt, config['epochs'], metrics, cfg, sampler)
    selected_file = 'epoch_zero.pt' if selected_epoch == 0 else 'trained_best.pt'
    atomic_copy(folder/selected_file, folder/'best.pt')
    best = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
    checkpoint_config(best, config, initial)
    require(best['epoch'] == selected_epoch and best['validation'] == selected, 'Recovered best checkpoint differs')
    model.load_state_dict(best['model_state'], strict=True)
    replay, selected_rows = core.joint_evaluate(model, val_loader, val_rows, device, folder.parent, ARM+'_best_raw_replay', True)
    measured, probability_rows = study.probability_replay(model, validation_loader, val_rows, device, folder.parent)
    require(replay == measured == selected and core.parent.identities(selected_rows) == core.parent.identities(probability_rows)
            and core.frozen_hash(model) == frozen, 'Recovered best raw/probability replay differs')
    raw.save_predictions(folder/'validation_predictions.jsonl.gz', selected_rows)
    raw.save_predictions(folder/'validation_probabilities.jsonl.gz', probability_rows)
    prefix, prefix_epoch = core.prefix_best(zero, history, config['prefix_epochs'])
    result = {'epoch_zero': zero, 'best_validation': selected, 'best_epoch': selected_epoch,
              'trained_best_validation': trained, 'best_trained_epoch': trained_epoch, 'last_validation': metrics,
              'best150_validation': prefix, 'best150_epoch': prefix_epoch,
              'gain_after150_pp': 100*(selected['macro_f1']-prefix['macro_f1']),
              'epochs': config['epochs'], 'optimizer_updates': updates, 'new_epochs': config['epochs'],
              'new_optimizer_updates': updates, 'training_samples_per_epoch': len(train_rows),
              'orders': [row['train_order_sha256'] for row in history], 'training_start_rng_sha256': rng,
              'initial_state_sha256': initial, 'frozen_state_sha256': frozen, 'raw_replay_passed': True,
              'historical_node_cache_used_for_training': False, 'probability_replay_passed': True,
              'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'wall_seconds_new_work': time.perf_counter()-started, 'recovery': receipt,
              'checkpoint_sha256': {file: raw.sha(folder/file) for file in (
                  'epoch_zero.pt', 'trained_best.pt', 'best.pt', 'last.pt', 'epoch150.pt', 'best150.pt')},
              'ledger_sha256': raw.sha(folder/'epochs.jsonl'),
              'predictions_sha256': raw.sha(folder/'validation_predictions.jsonl.gz'),
              'probability_predictions_sha256': raw.sha(folder/'validation_probabilities.jsonl.gz')}
    core.parent.check_history(history, result, config['epochs'], len(sampler), len(val_rows))
    raw.write(folder/'metrics.json', result)
    return result, probability_rows


def copy_completed(source, destination, config, rows, orders):
    result = raw.read(source/'metrics.json')
    history = core.read_history(source/'epochs.jsonl')
    require(result['raw_replay_passed'] and result['probability_replay_passed'] and result['orders'] == orders
            and result['training_samples_per_epoch'] == len(rows['train']) and result['trainable_parameters'] == 25384,
            'Completed arm metadata differs: '+str(source))
    steps = len(raw.TrainBatches(len(rows['train']), config['batch_size'], config['seed']))
    core.parent.check_history(history, result, config['epochs'], steps, len(rows['val']))
    expected_files = {**result['checkpoint_sha256'], 'epochs.jsonl': result['ledger_sha256'],
                      'validation_predictions.jsonl.gz': result['predictions_sha256'],
                      'validation_probabilities.jsonl.gz': result['probability_predictions_sha256']}
    for name, digest in expected_files.items():
        require(raw.sha(source/name) == digest, 'Completed artifact hash differs: '+str(source/name))
    predictions = core.read_predictions(source/'validation_predictions.jsonl.gz')
    probabilities = core.read_predictions(source/'validation_probabilities.jsonl.gz')
    require(study.validate_predictions(probabilities, rows['val']) == result['best_validation']
            and core.parent.identities(predictions) == core.parent.identities(probabilities),
            'Completed validation predictions differ: '+str(source))
    shutil.copytree(source, destination)
    return result, probabilities


def copy_seed42(previous, output, archive, val_rows):
    folder = previous/'seed42_audit'
    receipt = raw.read(folder/'audit_receipt.json')
    require(receipt['status'] == 'passed' and receipt['four_original_changed_sample_files_verified']
            and receipt['new_optimizer_updates'] == 0, 'Previous seed42 audit did not complete')
    exports = {}
    for name in study.NAMES:
        path = folder/name/'validation_probabilities.jsonl.gz'
        require(raw.sha(path) == receipt['models'][name]['probabilities_sha256'], 'Seed42 probability hash differs')
        exports[name] = core.read_predictions(path)
        require(study.validate_predictions(exports[name], val_rows) == archive['models'][name]['best_validation'],
                'Seed42 probability classification differs')
    target = output/'seed42_audit'
    shutil.copytree(folder, target)
    return {'models': {name: archive['models'][name] for name in study.NAMES},
            'analysis': study.paired_evidence(exports['Joint-Mean'], exports[ARM], target, 42)}


def run(repo, previous, output):
    require(torch.cuda.is_available() and torch.__version__.split('+')[0] == '2.3.1'
            and torch_geometric.__version__ == '2.5.3', 'Requires CUDA/Torch2.3.1/PyG2.5.3')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    config, archive, data, rows, source, _, inputs, binding = study.prepare(repo, output)
    require(raw.read(previous/'config.json') == config, 'Previous run protocol differs')
    old_binding = raw.read(previous/'binding.json')
    for key in ('protocol_sha256', 'seed42_archive_sha256', 'manifest_sha256', 'split_json_sha256', 'torch', 'pyg'):
        require(old_binding[key] == binding[key], 'Previous run binding differs: '+key)
    for name, digest in binding['sources'].items():
        require(old_binding['sources'].get(name) == digest, 'Previous training source differs: '+name)
    binding['sources']['scripts/resume_flow_query_joint_seeds.py'] = raw.sha(Path(__file__))
    raw.write(output/'binding.json', binding)
    preflight = raw.read(previous/'preflight.json')
    require(preflight['status'] == 'passed' and preflight['seed42_independent_audit_passed'], 'Previous preflight incomplete')
    shutil.copyfile(previous/'preflight.json', output/'preflight.json')
    old_inputs = {repo/name: digest for name, digest in raw.read(previous/'input_receipts.json').items()}
    require(all(old_inputs.get(path) == digest for path, digest in inputs.items()), 'Previous source receipts differ')
    for path, digest in old_inputs.items():
        require(raw.sha(path) == digest, 'Previous experiment input changed: '+str(path))
    shutil.copyfile(previous/'input_receipts.json', output/'input_receipts.json')
    pairs = {'42': copy_seed42(previous, output, archive, rows['val'])}
    recovery = {'previous_run': str(previous), 'previous_binding_sha256': raw.sha(previous/'binding.json'),
                'reused_completed_arms': [], 'optimizer_updates_performed_after_recovery': 0}
    for seed in config['new_seeds']:
        c = {**config, 'seed': seed}
        stage_name = 'seed'+str(seed)
        folder = output/stage_name
        folder.mkdir()
        orders = study.epoch_orders(len(rows['train']), c['batch_size'], seed, c['epochs'])
        results, exports = {}, {}
        for name in study.NAMES:
            arm_source = previous/stage_name/name
            result_path = arm_source/'metrics.json'
            complete = result_path.is_file() and raw.read(result_path).get('probability_replay_passed') is True
            if complete:
                result, predictions = copy_completed(arm_source, folder/name, c, rows, orders)
                recovery['reused_completed_arms'].append(stage_name+'/'+name)
            else:
                require((seed, name) == (44, ARM), 'Expected a completed arm at '+str(arm_source))
                model = study.build_models(source['model_state'], seed, device)[name]
                require(raw.state_digest(model.state_dict()) == preflight[str(seed)][name]['initial_state_sha256'],
                        'Previous preflight initial state differs')
                result, predictions = recover_arm(previous, folder/name, model, data, rows, device, c, binding, orders)
                recovery['optimizer_updates_performed_after_recovery'] = result['recovery']['optimizer_updates_performed_after_recovery']
            results[name], exports[name] = result, predictions
        require(results['Joint-Mean']['orders'] == results[ARM]['orders'] == orders
                and results['Joint-Mean']['initial_state_sha256'] == results[ARM]['initial_state_sha256']
                and results['Joint-Mean']['training_start_rng_sha256'] == results[ARM]['training_start_rng_sha256'],
                'Recovered seed pair differs')
        pairs[str(seed)] = {'models': results, 'analysis': study.paired_evidence(exports['Joint-Mean'], exports[ARM], folder, seed)}
        raw.write(folder/'summary.json', {'status': 'completed', 'protocol': c, **pairs[str(seed)]})
    aggregate = study.summarize(pairs, config)
    raw.write(output/'analysis.json', aggregate)
    study.write_report(output, pairs, aggregate, config)
    require(sum(result['optimizer_updates'] for seed in config['new_seeds'] for result in pairs[str(seed)]['models'].values())
            == 168800, 'Recovered four-run budget differs')
    for path, digest in old_inputs.items():
        require(raw.sha(path) == digest, 'Original input changed during recovery: '+str(path))
    raw.write(output/'summary.json', {'status': 'completed', 'protocol': config, 'binding': binding,
              'pairs': pairs, 'analysis': aggregate, 'new_training_runs': 4, 'new_optimizer_updates': 168800,
              'recovery': recovery, 'seed42_retrained': False, 'test_evaluated': False,
              'original_inputs_unchanged': True, 'automatic_followup': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--from-run', type=Path, default=Path(ORIGINAL_RUN))
    args = parser.parse_args()
    repo = args.repo.resolve()
    previous = (repo/args.from_run).resolve()
    require(previous.is_dir() and previous.parent == (repo/'outputs').resolve(), 'Expected an existing run directly under outputs')
    output = repo/'outputs'/('flow_query_joint_seeds43_44_resume_'+str(time.time_ns()))
    output.mkdir()
    raw.write(output/'status.json', {'status': 'running', 'resumed_from': str(previous)})
    print(f'Recovery output: {output}', flush=True)
    try:
        run(repo, previous, output)
    except BaseException as error:
        raw.write(output/'status.json', {'status': 'failed', 'error': str(error), 'resumed_from': str(previous)})
        raise
    raw.write(output/'status.json', {'status': 'completed', 'exit_code': 0, 'resumed_from': str(previous)})


if __name__ == '__main__':
    main()
