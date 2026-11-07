"""Seed42 frozen continuations and Packet-layer-two joint query optimization."""
import argparse
import copy
import csv
import gzip
import json
import random
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

import run_flow_query_pooling as parent
from xgnid.joint_flow_query import JointQueryPacketClassifier

raw, packet = parent.raw, parent.packet
require, LABELS = parent.require, parent.LABELS
CONFIG = raw.ROOT/'configs/flow_query_joint_packet2_s42_v1.json'
ARCHIVE = raw.ROOT/'docs/flow_query_pooling_server_summary_20261107.json'
ARCHIVE_SHA256 = 'fb3df1f629f2ed064d39c45c1c4e7c5419dd4d420cef83e5a2a76e6d9233a4b7'
NAMES = ('Mean-Query', 'Flow-Query')
FROZEN = {'Frozen-Mean': 'Mean-Query', 'Frozen-Flow': 'Flow-Query'}
JOINT = {'Joint-Mean': 'packet_mean', 'Joint-Flow': 'flow'}


def restore_rng(state):
    require(len(state['cuda']) == torch.cuda.device_count(), 'Saved CUDA RNG device count differs')
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])
    require(raw.rng_hash(raw.rng_state()) == raw.rng_hash(state), 'RNG restoration differs')


def assert_tree_equal(a, b, location='state'):
    """Compare Adam tensors on CPU without moving its step counter to CUDA."""
    if isinstance(a, torch.Tensor):
        require(isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape
                and torch.equal(a.cpu(), b.cpu()), 'Tensor differs: '+location)
    elif isinstance(a, dict):
        require(isinstance(b, dict) and a.keys() == b.keys(), 'State keys differ: '+location)
        for key in a:
            assert_tree_equal(a[key], b[key], location+'.'+str(key))
    elif isinstance(a, (tuple, list)):
        require(type(a) == type(b) and len(a) == len(b), 'State sequence differs: '+location)
        for i, (x, y) in enumerate(zip(a, b)):
            assert_tree_equal(x, y, location+'.'+str(i))
    else:
        require(a == b, 'State value differs: '+location)


def restore_training(model, opt, sampler, saved, config, expected_updates):
    model.load_state_dict(saved['model_state'], strict=True)
    # saved was loaded onto CPU. Adam.load_state_dict casts moments to the parameter
    # device while retaining the ordinary (non-capturable) CPU step counter.
    # load_state_dict may reuse same-device tensors (notably Adam's CPU step
    # counter even when parameters are CUDA). Own the state before any update.
    opt.load_state_dict(copy.deepcopy(saved['optimizer_state']))
    assert_tree_equal(saved['optimizer_state'], opt.state_dict(), 'Adam')
    groups = opt.param_groups
    require([g['lr'] for g in groups] == [config['head_lr'], config['pool_lr']]
            and all(g['weight_decay'] == config['weight_decay'] for g in groups), 'Restored Adam settings differ')
    parameters = [p for group in groups for p in group['params']]
    require(set(opt.state) == set(parameters), 'Missing Adam parameter state')
    counters = dict(sorted(Counter(int(opt.state[p]['step']) for p in parameters).items()))
    require(set(counters) == {expected_updates},
            f'Adam update counter differs: expected={expected_updates}, observed={counters}')
    sampler.generator.set_state(saved['sampler_rng'].cpu())
    require(torch.equal(sampler.generator.get_state(), saved['sampler_rng'].cpu()), 'Sampler restoration differs')
    restore_rng(saved['rng'])
    return {'strict_model_load': True, 'adam_values_identical': True, 'adam_updates': expected_updates,
            'adam_parameter_step_distribution': counters,
            'rng_sha256': raw.rng_hash(raw.rng_state()),
            'sampler_sha256': raw.state_digest({'sampler': sampler.generator.get_state()})}


def read_predictions(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def read_history(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def prefix_best(zero, history, limit):
    selected, epoch = zero, 0
    for row in history[:limit]:
        if row['validation']['macro_f1'] > selected['macro_f1']:
            selected, epoch = row['validation'], row['epoch']
    return selected, epoch


def validate_resume(repo, config, archive, inputs, size, batch_size):
    old_root = repo/config['resume_directory']
    summary = raw.read(old_root/'summary.json')
    require(summary == archive and summary['status'] == 'completed', 'Original150 summary differs from archive')
    require(raw.read(old_root/'status.json')['status'] == 'completed', 'Original150 run incomplete')
    inputs[old_root/'summary.json'] = raw.sha(old_root/'summary.json')
    results = {}
    for name in NAMES:
        result, folder = summary['models'][name], old_root/name
        require(result['epochs'] == config['resume_seed42_from_epoch'] == 150
                and result['raw_replay_passed'], 'Wrong resume epoch/replay receipt')
        for filename, expected in {**result['checkpoint_sha256'],
                                   'validation_predictions.jsonl.gz': result['predictions_sha256']}.items():
            path = folder/filename
            require(raw.sha(path) == expected, 'Original150 artifact differs: '+str(path))
            inputs[path] = expected
        history = read_history(folder/'epochs.jsonl')
        parent.check_history(history, result, 150, 211, 13446)
        require(history[result['best_epoch']-1]['validation'] == result['best_validation'], 'Old best matrix differs')
        sampler = raw.TrainBatches(size, batch_size, 42)
        for row in history:
            list(sampler)
            require(row['train_order_sha256'] == sampler.order_sha256, 'Original150 sampler order differs')
        saved = torch.load(folder/'last.pt', map_location='cpu', weights_only=False)
        require(saved['epoch'] == 150 and saved['validation'] == result['last_validation']
                and saved['config']['protocol'] == archive['protocol']
                and saved['config']['variant'] == name and saved['config']['binding'] == archive['binding'],
                'Last checkpoint metadata differs')
        require(torch.equal(saved['sampler_rng'].cpu(), sampler.generator.get_state()), 'Epoch150 sampler boundary differs')
        encoder = {key: value for key, value in saved['model_state'].items()
                   if key.startswith(('flow_encoder.', 'packet_encoder.'))}
        require(raw.state_digest(encoder) == config['encoder_sha256'], 'Saved encoders differ')
        rows = read_predictions(folder/'validation_predictions.jsonl.gz')
        require(len(rows) == 13446, 'Incomplete original predictions')
        # Full prediction identities and matrices were independently checked on the
        # server in the archived run; hashes bind this continuation to those bytes.
        inputs[folder/'epochs.jsonl'] = raw.sha(folder/'epochs.jsonl')
        results[name] = {'result': result, 'saved': saved, 'history': history, 'folder': folder}
    require(results[NAMES[0]]['result']['orders'] == results[NAMES[1]]['result']['orders']
            and raw.rng_hash(results[NAMES[0]]['saved']['rng']) == raw.rng_hash(results[NAMES[1]]['saved']['rng']),
            'Original paired sampler/RNG boundary differs')
    return results


@torch.no_grad()
def compare_raw_cache(model, batch, cache, numbers, device):
    model.eval()
    features, labels = packet.minibatch(parent.CORE, cache, numbers, device)
    require(list(batch.sample_id) == [cache['ids'][i] for i in numbers]
            and batch.y.view(-1).tolist() == labels.tolist(), 'Raw/cache probe sample order differs')
    raw_features = model.encode(batch)
    require(torch.equal(raw_features[2], features[2]) and raw_features[3] == features[3], 'Raw/cache Packet layout differs')
    a = model.forward_with_attention(*raw_features)
    b = model.forward_with_attention(*features)
    errors = {}
    for key, x, y in zip(('scores', 'gate', 'packet_attention'), a, b):
        require(x.shape == y.shape and bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all())
                and torch.allclose(x, y, atol=1e-5, rtol=1e-5), 'Raw/cache '+key+' differs')
        errors[key] = float((x-y).abs().max())
    require(torch.equal(a[0].argmax(-1), b[0].argmax(-1)), 'Raw/cache probe predictions differ')
    return {'max_abs_errors': errors, 'atol': 1e-5, 'rtol': 1e-5, 'predictions_identical': True}


def raw_update_probe(model, saved, cache, numbers, batch, config, device):
    rng = raw.rng_state()
    digest, frozen = raw.state_digest(model.state_dict()), parent.encoder_hash(model)
    checkpoint_adam = copy.deepcopy(saved['optimizer_state']) if saved is not None else None
    try:
        candidate = copy.deepcopy(model)
        opt = parent.optimizer(candidate, config)
        if saved is not None:
            opt.load_state_dict(copy.deepcopy(saved['optimizer_state']))
            assert_tree_equal(saved['optimizer_state'], opt.state_dict())
        before = compare_raw_cache(candidate, batch, cache, numbers, device)
        records, changed = [], set()
        for index in range(config['raw_probe_updates']):
            candidate.train()
            weights = {'query': candidate.packet_pool.query.weight, 'key': candidate.packet_pool.key.weight}
            old = {key: p.detach().clone() for key, p in weights.items()}
            loss = parent.step(candidate, cache, numbers, device, opt)
            grads = {key: float(p.grad.norm()) for key, p in weights.items()}
            updated = {key for key, p in weights.items() if not torch.equal(p, old[key])}
            if saved is None and index == 0:
                require(updated == {'query'} and grads['query'] > 0 and grads['key'] == 0, 'Zero-query first gradient differs')
            changed.update(updated)
            records.append({'step': index+1, 'nll': loss, 'gradient_norms': grads, 'updated': sorted(updated)})
        after = compare_raw_cache(candidate, batch, cache, numbers, device)
        require(changed == {'query', 'key'} and parent.encoder_hash(candidate) == frozen, 'Probe update/frozen state differs')
        require(raw.state_digest(model.state_dict()) == digest, 'Probe changed formal training start')
        return {'before': before, 'updates': records, 'after': after, 'formal_state_unchanged': True,
                'restored_adam_used': saved is not None,
                'checkpoint_optimizer_unchanged': saved is not None, 'samples': len(numbers)}
    finally:
        restore_rng(rng)
        if saved is not None:
            assert_tree_equal(checkpoint_adam, saved['optimizer_state'], 'Probe source checkpoint Adam')


def inherit_checkpoint(source, destination, cfg):
    saved = torch.load(source, map_location='cpu', weights_only=False)
    # Only the metadata envelope changes. Tensor/Adam/RNG/sampler values remain
    # inherited, including when the old150 best remains the selected200 best.
    saved['inherited_from'] = {'path': str(source), 'sha256': raw.sha(source), 'config': saved['config']}
    saved['config'] = cfg
    torch.save(saved, destination)


def fit(name, model, caches, device, config, binding, output, resume=None, expected_orders=None):
    folder = output/name
    folder.mkdir()
    opt = parent.optimizer(model, config)
    sampler = raw.TrainBatches(len(caches['train']['ids']), config['batch_size'], config['seed'])
    frozen = parent.encoder_hash(model)
    cfg = {'variant': name, 'protocol': config, 'binding': binding, 'labels': LABELS,
           'initial_state_sha256': (resume['saved']['config']['initial_state_sha256'] if resume else
                                    raw.state_digest(model.state_dict()))}
    if resume:
        old, history = resume['result'], copy.deepcopy(resume['history'])
        zero, selected, trained = old['epoch_zero'], old['best_validation'], old['trained_best_validation']
        selected_epoch, trained_epoch = old['best_epoch'], old['best_trained_epoch']
        for filename in ('epoch_zero.pt', 'trained_best.pt'):
            inherit_checkpoint(resume['folder']/filename, folder/filename, cfg)
        inherit_checkpoint(resume['folder']/'best.pt', folder/'best150.pt', cfg)
        shutil.copyfile(resume['folder']/'last.pt', folder/'epoch150.pt')
        shutil.copyfile(resume['folder']/'epochs.jsonl', folder/'epochs.jsonl')
        require((folder/'epochs.jsonl').read_bytes().endswith(b'\n'), 'Old ledger lacks final newline')
        receipt = restore_training(model, opt, sampler, resume['saved'], config, len(history)*len(sampler))
        boundary_metrics, _ = parent.evaluate(model, caches['val'], device, config['eval_batch_size'])
        require(boundary_metrics == old['last_validation'] and parent.encoder_hash(model) == frozen,
                'Restored150 model/metrics differ')
        # Evaluate/serialization must never change the continuation RNG boundary.
        restore_rng(resume['saved']['rng'])
        rng = old['training_start_rng_sha256']
    else:
        raw.seed_all(config['seed'])
        history, receipt = [], None
        zero, _ = parent.evaluate(model, caches['val'], device, config['eval_batch_size'])
        raw.save_checkpoint(folder/'epoch_zero.pt', model, opt, 0, zero, cfg, sampler)
        selected, selected_epoch, trained, trained_epoch = zero, 0, None, None
        (folder/'epochs.jsonl').touch()
        rng = raw.rng_hash(raw.rng_state())
    start_epoch = len(history)
    start_rng = raw.rng_hash(raw.rng_state())
    start_state = raw.state_digest(model.state_dict())
    started = time.perf_counter()
    if receipt:
        raw.write(folder/'resume_receipt.json', {**receipt, 'from_epoch': start_epoch,
                  'next_epoch': start_epoch+1, 'last_pt_sha256': raw.sha(resume['folder']/'last.pt'),
                  'prefix_ledger_sha256': raw.sha(resume['folder']/'epochs.jsonl'),
                  'training_start_state_sha256': start_state, 'boundary_metrics_identical': True})
    with (folder/'epochs.jsonl').open('a', encoding='utf-8', newline='\n') as ledger:
        for epoch in range(start_epoch+1, config['epochs']+1):
            model.train()
            count = steps = 0
            total = 0.0
            for numbers in sampler:
                if steps == 0 and expected_orders is not None:
                    require(sampler.order_sha256 == expected_orders[epoch-1], 'Paired order differs')
                total += parent.step(model, caches['train'], numbers, device, opt)*len(numbers)
                count += len(numbers)
                steps += 1
            require(count == len(caches['train']['ids']) and steps == len(sampler)
                    and parent.encoder_hash(model) == frozen, 'Incomplete epoch or frozen encoder/BN changed')
            metrics, _ = parent.evaluate(model, caches['val'], device, config['eval_batch_size'])
            if trained is None or metrics['macro_f1'] > trained['macro_f1']:
                trained, trained_epoch = metrics, epoch
                raw.save_checkpoint(folder/'trained_best.pt', model, opt, epoch, metrics, cfg, sampler)
            if metrics['macro_f1'] > selected['macro_f1']:
                selected, selected_epoch = metrics, epoch
            row = {'epoch': epoch, 'validation': metrics, 'best_epoch': selected_epoch, 'best_trained_epoch': trained_epoch,
                   'samples': count, 'optimizer_updates': steps, 'train_nll': total/count,
                   'train_order_sha256': sampler.order_sha256}
            history.append(row)
            ledger.write(json.dumps(row)+'\n')
            ledger.flush()
            if epoch == config['prefix_epochs']:
                raw.save_checkpoint(folder/'epoch150.pt', model, opt, epoch, metrics, cfg, sampler)
                shutil.copyfile(folder/('epoch_zero.pt' if selected_epoch == 0 else 'trained_best.pt'), folder/'best150.pt')
            raw.progress(output, 'training', seed=config['seed'], model=name, epoch=epoch,
                         val_macro_f1=metrics['macro_f1'], selected_epoch=selected_epoch)
    metrics = history[-1]['validation']
    require(all(int(state['step']) == config['epochs']*len(sampler) for state in opt.state.values()),
            'Frozen cumulative Adam step count differs')
    raw.save_checkpoint(folder/'last.pt', model, opt, config['epochs'], metrics, cfg, sampler)
    shutil.copyfile(folder/('epoch_zero.pt' if selected_epoch == 0 else 'trained_best.pt'), folder/'best.pt')
    raw.strict_reload(model, folder/'best.pt', device, cfg, selected_epoch, selected)
    replay, rows = parent.evaluate(model, caches['val'], device, config['eval_batch_size'], True)
    require(replay == selected and parent.encoder_hash(model) == frozen, 'Best cached replay/frozen state differs')
    raw.save_predictions(folder/'validation_predictions.jsonl.gz', rows)
    prefix_metrics, prefix_epoch = prefix_best(zero, history, config['prefix_epochs'])
    prefix = torch.load(folder/'best150.pt', map_location='cpu', weights_only=False)
    require(prefix['epoch'] == prefix_epoch and prefix['validation'] == prefix_metrics, 'Best150 inheritance differs')
    result = {'epoch_zero': zero, 'best_validation': selected, 'best_epoch': selected_epoch,
              'trained_best_validation': trained, 'best_trained_epoch': trained_epoch, 'last_validation': metrics,
              'best150_validation': prefix_metrics, 'best150_epoch': prefix_epoch,
              'gain_after150_pp': 100*(selected['macro_f1']-prefix_metrics['macro_f1']),
              'epochs': config['epochs'], 'optimizer_updates': sum(r['optimizer_updates'] for r in history),
              'new_epochs': config['epochs']-start_epoch, 'new_optimizer_updates': (config['epochs']-start_epoch)*len(sampler),
              'training_samples_per_epoch': len(caches['train']['ids']), 'orders': [r['train_order_sha256'] for r in history],
              'training_start_rng_sha256': rng, 'continuation_start_rng_sha256': start_rng,
              'continuation_start_state_sha256': start_state, 'resume': receipt,
              'frozen_state_sha256': frozen, 'wall_seconds_new_work': time.perf_counter()-started,
              'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'checkpoint_sha256': {n: raw.sha(folder/n) for n in (
                  'epoch_zero.pt', 'trained_best.pt', 'best.pt', 'last.pt', 'epoch150.pt', 'best150.pt')},
              'ledger_sha256': raw.sha(folder/'epochs.jsonl'),
              'predictions_sha256': raw.sha(folder/'validation_predictions.jsonl.gz')}
    parent.check_history(history, result, config['epochs'], len(sampler), len(caches['val']['ids']))
    if resume:
        require(history[:start_epoch] == resume['history'] and prefix_metrics == resume['result']['best_validation']
                and prefix_epoch == resume['result']['best_epoch'], 'Original150 history/selection changed')
    return result, rows


def frozen_hash(model):
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    return raw.state_digest({name: value for name, value in model.state_dict().items() if name not in trainable})


def joint_optimizer(model, config):
    groups = [
        {'name': 'fusion', 'params': list(model.fusion.parameters()), 'lr': config['head_lr']},
        {'name': 'readout', 'params': list(model.packet_pool.parameters()), 'lr': config['pool_lr']},
        {'name': 'flow_to_packet', 'params': list(model.packet_relations()[0].parameters()), 'lr': config['encoder_lr']},
        {'name': 'packet_to_packet', 'params': list(model.packet_relations()[1].parameters()), 'lr': config['encoder_lr']},
    ]
    ids = [id(p) for group in groups for p in group['params']]
    require(len(ids) == len(set(ids)) and set(ids) == {id(p) for p in model.parameters() if p.requires_grad}
            and [sum(p.numel() for p in group['params']) for group in groups] == [7144, 1024, 8704, 8512],
            'Joint optimizer scope differs')
    return torch.optim.Adam(groups, weight_decay=config['weight_decay'])


def joint_step(model, batch, opt):
    opt.zero_grad(set_to_none=True)
    scores = model(batch)
    loss = torch.nn.functional.nll_loss(scores, batch.y.view(-1))
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    require(bool(torch.isfinite(loss)) and all(g is not None for g in gradients)
            and bool(torch.isfinite(torch.stack([g.norm() for g in gradients]).norm())), 'Joint loss/gradient path differs')
    opt.step()
    return float(loss.detach())


def graph_loader(data, rows, part, config, sampler=None):
    generator = torch.Generator().manual_seed(config['seed']+(100 if part == 'train' else 200))
    dataset = raw.GraphRows(data, rows, part)
    if sampler is not None:
        return DataLoader(dataset, batch_sampler=sampler, num_workers=config['workers'], generator=generator)
    return DataLoader(dataset, batch_size=config['eval_batch_size'], shuffle=False,
                      num_workers=config['workers'], generator=generator)


@torch.no_grad()
def joint_evaluate(model, loader, rows, device, output=None, phase='joint_val', export=False):
    model.eval()
    matrix, records, offset = [[0]*8 for _ in range(8)], [], 0
    for step, batch in enumerate(loader, 1):
        batch = batch.to(device)
        expected = rows[offset:offset+batch.num_graphs]
        require(list(zip(batch.sample_id, batch.y.view(-1).tolist())) ==
                [(r['sample_id'], r['label']) for r in expected], 'Joint raw sample order differs')
        scores, alpha, beta = model.forward_with_attention(*model.encode(batch))
        require(bool(torch.isfinite(scores).all()) and bool(torch.isfinite(alpha).all())
                and bool(torch.isfinite(beta).all()), 'Nonfinite joint validation output')
        attentions = beta.cpu().split(torch.bincount(batch['packet'].batch, minlength=batch.num_graphs).tolist()) if export else None
        for i, (sid, y, pred) in enumerate(zip(batch.sample_id, batch.y.view(-1).tolist(), scores.argmax(-1).tolist())):
            matrix[y][pred] += 1
            row = {'sample_id': sid, 'label': y, 'prediction': pred}
            if export:
                b = attentions[i]
                require(abs(float(b.sum())-1) < 1e-5, 'Invalid joint Packet attention sum')
                row.update(flow_weights=alpha[i].tolist(), packet_attention=b.tolist())
            records.append(row)
        offset += batch.num_graphs
        if output is not None and (step == 1 or step % 200 == 0):
            raw.progress(output, phase, samples=offset, total=len(rows))
    require(offset == len(rows), 'Incomplete joint validation')
    return parent.matrix_report(matrix), records


def joint_probe(model, cache, numbers, batch, config, device):
    rng = raw.rng_state()
    initial, frozen = raw.state_digest(model.state_dict()), frozen_hash(model)
    try:
        candidate = copy.deepcopy(model)
        opt = joint_optimizer(candidate, config)
        before = compare_raw_cache(candidate, batch, cache, numbers, device)
        with torch.no_grad():
            outputs_before = tuple(x.detach().clone() for x in candidate.forward_with_attention(*candidate.encode(batch)))
        updated_groups, updated_projections, history = set(), set(), []
        for index in range(config['raw_probe_updates']):
            candidate.train()
            old = {group['name']: [p.detach().clone() for p in group['params']] for group in opt.param_groups}
            projections = {'query': candidate.packet_pool.query.weight, 'key': candidate.packet_pool.key.weight}
            old_projections = {key: p.detach().clone() for key, p in projections.items()}
            loss = joint_step(candidate, batch, opt)
            norms = {key: float(p.grad.norm()) for key, p in projections.items()}
            changed = {key for key, p in projections.items() if not torch.equal(p, old_projections[key])}
            if index == 0:
                require(changed == {'query'} and norms['query'] > 0 and norms['key'] == 0, 'Joint zero-query gradient differs')
            updated_projections.update(changed)
            changed_groups = {group['name'] for group in opt.param_groups if any(
                p.grad is not None and bool(p.grad.ne(0).any()) and not torch.equal(p, previous)
                for p, previous in zip(group['params'], old[group['name']]))}
            updated_groups.update(changed_groups)
            history.append({'step': index+1, 'nll': loss, 'projection_gradient_norms': norms,
                            'updated_groups': sorted(changed_groups), 'updated_projections': sorted(changed)})
        candidate.eval()
        with torch.no_grad():
            # A fresh, detached snapshot from the UPDATED encoder is used only
            # to test the readout path. Historical frozen nodes cannot match it.
            flow, nodes, index, size = candidate.encode(batch)
            lengths = torch.bincount(index, minlength=size).cpu()
            current = {'ids': list(batch.sample_id), 'flow64': flow.detach().cpu(), 'packet64': nodes.detach().cpu(),
                       'ptr': torch.cat((torch.zeros(1, dtype=torch.long), lengths.cumsum(0))),
                       'labels': batch.y.detach().cpu()}
            after = compare_raw_cache(candidate, batch, current, list(range(size)), device)
            outputs_after = candidate.forward_with_attention(*candidate.encode(batch))
            changes = {key: float((a-b).abs().max()) for key, a, b in
                       zip(('scores', 'gate', 'packet_attention'), outputs_before, outputs_after)}
        require(updated_groups == {g['name'] for g in opt.param_groups} and updated_projections == {'query', 'key'},
                'Joint probe did not update every registered group')
        require(frozen_hash(candidate) == frozen and raw.state_digest(model.state_dict()) == initial,
                'Joint probe changed frozen state or formal model')
        require(all(value > 0 for value in changes.values()), 'Joint output/gate/Packet attention did not change')
        return {'before_old_cache': before, 'updates': history, 'after_fresh_raw_feature_snapshot': after,
                'max_abs_changes': changes, 'frozen_state_unchanged': True, 'formal_state_unchanged': True,
                'historical_node_cache_used_for_joint_updates': False}
    finally:
        restore_rng(rng)


def fit_joint(name, model, data, rows, device, config, binding, output, expected_orders):
    raw.seed_all(config['seed'])
    folder = output/name
    folder.mkdir()
    sampler = raw.TrainBatches(len(rows['train']), config['batch_size'], config['seed'])
    train_batches = graph_loader(data, rows['train'], 'train', config, sampler)
    val_batches = graph_loader(data, rows['val'], 'val', config)
    opt = joint_optimizer(model, config)
    frozen = frozen_hash(model)
    cfg = {'variant': name, 'protocol': config, 'binding': binding, 'labels': LABELS,
           'initial_state_sha256': raw.state_digest(model.state_dict())}
    rng = raw.rng_hash(raw.rng_state())
    zero, initial_rows = joint_evaluate(model, val_batches, rows['val'], device)
    require(raw.rng_hash(raw.rng_state()) == rng, 'Joint epoch0 evaluation advanced training RNG')
    raw.save_checkpoint(folder/'epoch_zero.pt', model, opt, 0, zero, cfg, sampler)
    selected, selected_epoch, selected_rows = zero, 0, initial_rows
    trained, trained_epoch, history = None, None, []
    started = time.perf_counter()
    with (folder/'epochs.jsonl').open('x', encoding='utf-8', newline='\n') as ledger:
        for epoch in range(1, config['epochs']+1):
            model.train()
            total, count, steps = 0., 0, 0
            for batch in train_batches:
                if steps == 0:
                    require(sampler.order_sha256 == expected_orders[epoch-1], 'Joint/frozen sample orders differ')
                expected = [rows['train'][i] for i in sampler.order[count:count+batch.num_graphs]]
                require(list(zip(batch.sample_id, batch.y.view(-1).tolist())) ==
                        [(r['sample_id'], r['label']) for r in expected], 'Joint training batch IDs/labels differ')
                total += joint_step(model, batch.to(device), opt)*batch.num_graphs
                count += batch.num_graphs
                steps += 1
                if steps == 1 or steps % 50 == 0:
                    raw.progress(output, name+'_train', epoch=epoch, samples=count, total=len(rows['train']))
            require(count == len(rows['train']) and steps == len(sampler) and frozen_hash(model) == frozen,
                    'Joint training coverage or frozen parameters/BN differ')
            metrics, predictions = joint_evaluate(model, val_batches, rows['val'], device, output, name+'_val')
            if trained is None or metrics['macro_f1'] > trained['macro_f1']:
                trained, trained_epoch = metrics, epoch
                raw.save_checkpoint(folder/'trained_best.pt', model, opt, epoch, metrics, cfg, sampler)
            if metrics['macro_f1'] > selected['macro_f1']:
                selected, selected_epoch, selected_rows = metrics, epoch, predictions
            row = {'epoch': epoch, 'validation': metrics, 'best_epoch': selected_epoch, 'best_trained_epoch': trained_epoch,
                   'samples': count, 'optimizer_updates': steps, 'train_nll': total/count,
                   'train_order_sha256': sampler.order_sha256}
            history.append(row)
            ledger.write(json.dumps(row)+'\n')
            ledger.flush()
            if epoch == config['prefix_epochs']:
                raw.save_checkpoint(folder/'epoch150.pt', model, opt, epoch, metrics, cfg, sampler)
                shutil.copyfile(folder/('epoch_zero.pt' if selected_epoch == 0 else 'trained_best.pt'), folder/'best150.pt')
            print(json.dumps({'model': name, 'epoch': epoch, 'val_macro_f1': metrics['macro_f1'],
                              'best_epoch': selected_epoch}), flush=True)
    require(all(int(state['step']) == config['epochs']*len(sampler) for state in opt.state.values()), 'Joint Adam step count differs')
    raw.save_checkpoint(folder/'last.pt', model, opt, config['epochs'], metrics, cfg, sampler)
    shutil.copyfile(folder/('epoch_zero.pt' if selected_epoch == 0 else 'trained_best.pt'), folder/'best.pt')
    raw.strict_reload(model, folder/'best.pt', device, cfg, selected_epoch, selected)
    replay, exports = joint_evaluate(model, val_batches, rows['val'], device, output, name+'_best_raw_replay', True)
    require(replay == selected and parent.identities(exports) == parent.identities(selected_rows)
            and frozen_hash(model) == frozen, 'Joint best strict reload/raw replay differs')
    raw.save_predictions(folder/'validation_predictions.jsonl.gz', exports)
    prefix, prefix_epoch = prefix_best(zero, history, config['prefix_epochs'])
    result = {'epoch_zero': zero, 'best_validation': selected, 'best_epoch': selected_epoch,
              'trained_best_validation': trained, 'best_trained_epoch': trained_epoch, 'last_validation': metrics,
              'best150_validation': prefix, 'best150_epoch': prefix_epoch,
              'gain_after150_pp': 100*(selected['macro_f1']-prefix['macro_f1']),
              'epochs': config['epochs'], 'optimizer_updates': config['epochs']*len(sampler),
              'new_epochs': config['epochs'], 'new_optimizer_updates': config['epochs']*len(sampler),
              'training_samples_per_epoch': len(rows['train']), 'orders': [r['train_order_sha256'] for r in history],
              'training_start_rng_sha256': rng, 'initial_state_sha256': cfg['initial_state_sha256'],
              'frozen_state_sha256': frozen, 'raw_replay_passed': True, 'historical_node_cache_used_for_training': False,
              'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'wall_seconds_new_work': time.perf_counter()-started,
              'checkpoint_sha256': {n: raw.sha(folder/n) for n in (
                  'epoch_zero.pt', 'trained_best.pt', 'best.pt', 'last.pt', 'epoch150.pt', 'best150.pt')},
              'ledger_sha256': raw.sha(folder/'epochs.jsonl'),
              'predictions_sha256': raw.sha(folder/'validation_predictions.jsonl.gz')}
    parent.check_history(history, result, config['epochs'], len(sampler), len(rows['val']))
    raw.write(folder/'metrics.json', result)
    return result, exports


def analyze_four(results, exports, config, output):
    comparisons = {}
    for new, old in (('Frozen-Flow', 'Frozen-Mean'), ('Joint-Flow', 'Joint-Mean'),
                     ('Joint-Flow', 'Frozen-Flow'), ('Joint-Mean', 'Frozen-Mean')):
        a, b = exports[old], exports[new]
        counts, changed = raw.compare(a, b)
        key = new+'_vs_'+old
        raw.save_predictions(output/(key+'_changed_samples.jsonl.gz'), changed)
        focus = {}
        for label in config['focus_classes']:
            y = LABELS.index(label)
            focus[label], _ = raw.compare([r for r in a if r['label'] == y], [r for r in b if r['label'] == y])
        per_class = {label: {
            'recall_delta_pp': 100*(results[new]['best_validation']['per_class'][label]['recall']-
                                    results[old]['best_validation']['per_class'][label]['recall']),
            'F1_delta_pp': 100*(results[new]['best_validation']['per_class'][label]['f1-score']-
                                results[old]['best_validation']['per_class'][label]['f1-score'])} for label in LABELS}
        comparisons[key] = {'macro_f1_delta_pp': 100*(results[new]['best_validation']['macro_f1']-
                                                      results[old]['best_validation']['macro_f1']),
                            'comparison': counts, 'focus_comparisons': focus, 'class_changes': per_class}
    frozen = comparisons['Frozen-Flow_vs_Frozen-Mean']['macro_f1_delta_pp']
    joint = comparisons['Joint-Flow_vs_Joint-Mean']['macro_f1_delta_pp']
    return {'Joint_Flow_minus_Joint_Mean_pp': joint, 'Frozen_Flow_minus_Frozen_Mean_pp': frozen,
            'query_advantage_change_with_unfreezing_pp': joint-frozen,
            'Joint_Flow_minus_Frozen_Flow_pp': comparisons['Joint-Flow_vs_Frozen-Flow']['macro_f1_delta_pp'],
            'Joint_Mean_minus_Frozen_Mean_pp': comparisons['Joint-Mean_vs_Frozen-Mean']['macro_f1_delta_pp'],
            'Joint_Flow_minus_JCL_Multi_pp': 100*(results['Joint-Flow']['best_validation']['macro_f1']-
                                                 config['JCL_Multi_reference_macro_f1']),
            'original_seed42_150_paired_delta_pp': 100*(results['Frozen-Flow']['best150_validation']['macro_f1']-
                                                       results['Frozen-Mean']['best150_validation']['macro_f1']),
            'comparisons': comparisons, 'additional_seeds_deferred': [43, 44],
            'statistical_significance_claimed': False, 'automatic_followup': False}


def write_report(output, results, analysis):
    lines = ['| Model | Epoch0 F1 | Best0-150 F1 | Epoch | Best0-200 F1 | Epoch | Trained best F1 | Epoch | Gain after150 pp | Last F1 | Updates |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, r in results.items():
        lines.append(f"| {name} | {100*r['epoch_zero']['macro_f1']:.6f}% | {100*r['best150_validation']['macro_f1']:.6f}% | "
                     f"{r['best150_epoch']} | {100*r['best_validation']['macro_f1']:.6f}% | {r['best_epoch']} | "
                     f"{100*r['trained_best_validation']['macro_f1']:.6f}% | {r['best_trained_epoch']} | "
                     f"{r['gain_after150_pp']:+.6f} | {100*r['last_validation']['macro_f1']:.6f}% | {r['optimizer_updates']} |")
    lines.extend(['', '| Comparison | Macro-F1 delta pp |', '|---|---:|'])
    for key, value in analysis['comparisons'].items():
        lines.append(f"| {key} | {value['macro_f1_delta_pp']:+.6f} |")
    lines.extend(['', f"Change in Flow−Mean advantage with unfreezing: {analysis['query_advantage_change_with_unfreezing_pp']:+.6f} pp.",
                  f"Joint-Flow minus historical JCL-Multi: {analysis['Joint_Flow_minus_JCL_Multi_pp']:+.6f} pp.", '',
                  'Frozen arms resume each original last.pt at epoch151 with model, Adam, all RNG, sampler and existing best history.',
                  'Joint arms start at the same D32 source for200 raw-graph epochs. Only the two incoming Packet layer-two GATs, query/key and D32 fusion head train; BN stays eval.',
                  'All four select from epochs0..200; ties retain the earliest epoch. Seeds43/44 are deferred; no significance claim or automatic follow-up.', '',
                  '| Comparison | True class | Repaired | Regressed | Recall delta pp | F1 delta pp |', '|---|---|---:|---:|---:|---:|'])
    for key, value in analysis['comparisons'].items():
        for label, counts in value['focus_comparisons'].items():
            delta = value['class_changes'][label]
            lines.append(f"| {key} | {label} | {counts['counts']['repaired']} | {counts['counts']['regressed']} | "
                         f"{delta['recall_delta_pp']:+.6f} | {delta['F1_delta_pp']:+.6f} |")
    (output/'report.md').write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))
    with (output/'per_class.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', 'phase', 'class', 'precision', 'recall', 'f1', 'support'])
        for name, result in results.items():
            for phase in ('epoch_zero', 'best150_validation', 'best_validation', 'trained_best_validation', 'last_validation'):
                for label, row in result[phase]['per_class'].items():
                    writer.writerow([name, phase, label, row['precision'], row['recall'], row['f1-score'], row['support']])


def prepare(repo, output):
    study = raw.read(CONFIG)
    require(study['seed'] == 42 and study['epochs'] == 200 and study['additional_seeds_deferred'] == [43, 44]
            and study['resume_seed42_from_epoch'] == 150 and not study['early_stop_enabled']
            and study['include_epoch_zero'] and study['test_evaluated'] is False, 'Unregistered study budget')
    base_path = CONFIG.parent/study['base_protocol']
    require(raw.sha(base_path) == study['base_protocol_sha256'] and raw.sha(ARCHIVE) == ARCHIVE_SHA256,
            'Base protocol/archive hash differs')
    base, archive = raw.read(base_path), raw.read(ARCHIVE)
    require(base == archive['protocol'], 'Original protocol differs from archive')
    for key in ('head_lr', 'pool_lr', 'weight_decay', 'batch_size', 'eval_batch_size', 'workers'):
        require(study[key] == base[key], 'Training setting changed: '+key)
    require(study['encoder_lr'] == 1e-6, 'Unregistered Packet2 LR')
    for path, digest in archive['binding']['sources'].items():
        require(raw.sha(raw.ROOT/path) == digest, 'Historical source changed: '+path)
    data = raw.FrozenGraphs(repo/base['split_directory'], base['split_id'])
    rows = {part: raw.choose_rows(data, part) for part in ('train', 'val')}
    require((len(rows['train']), len(rows['val'])) == (107621, 13446), 'Wrong split sizes')
    inputs = {repo/path: digest for path, digest in archive['binding']['input_sha256'].items()}
    for path, digest in inputs.items():
        require(raw.sha(path) == digest, 'Pinned input hash differs: '+str(path))
    source = torch.load(repo/base['source_directory']/'best.pt', map_location='cpu', weights_only=False)
    require(source['epoch'] == 62 and source['validation']['macro_f1'] == base['source_macro_f1'], 'Wrong D32 starting checkpoint')
    gold = read_predictions(repo/base['source_directory']/'validation_predictions.jsonl.gz')
    caches = {}
    for part in rows:
        cache = torch.load(repo/base['cache_directory']/(part+'_packet_nodes.pt'), map_location='cpu', weights_only=False)
        split = cache['metadata']['split_binding']
        require(split['split_id'] == base['split_id'] and split['manifest_sha256'] == data.metadata['sample_manifest_sha256'],
                'Cache partition/manifest differs')
        packet.validate_cache(parent.CORE, cache, rows[part], part, base['encoder_sha256'], split)
        caches[part] = cache
    config = {**base, **study, 'prefix_epochs': 150}
    resumed = validate_resume(repo, config, archive, inputs, len(rows['train']), base['batch_size'])
    binding = {'study_sha256': raw.sha(CONFIG), 'archive_sha256': raw.sha(ARCHIVE),
               'manifest_sha256': data.metadata['sample_manifest_sha256'], 'torch': torch.__version__,
               'pyg': torch_geometric.__version__, 'cuda_device': torch.cuda.get_device_name(0),
               'input_sha256': {str(path.relative_to(repo)): digest for path, digest in inputs.items()},
               'sources': {path: raw.sha(raw.ROOT/path) for path in (
                   'scripts/run_flow_query_joint_packet2.py', 'scripts/test_flow_query_joint_packet2.py',
                   'scripts/launch_flow_query_joint_packet2.sh',
                   'src/xgnid/joint_flow_query.py', 'configs/flow_query_joint_packet2_s42_v1.json',
                   'docs/flow_query_joint_packet2_s42_v1.md', *archive['binding']['sources'].keys())}}
    raw.write(output/'config.json', config)
    raw.write(output/'binding.json', binding)
    return config, source, gold, data, rows, caches, resumed, binding, inputs


def run(repo, output, train):
    require(torch.cuda.is_available() and torch.__version__.split('+')[0] == '2.3.1'
            and torch_geometric.__version__ == '2.5.3', 'Requires CUDA, Torch2.3.1, PyG2.5.3')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    config, source, gold, data, rows, caches, resumed, binding, inputs = prepare(repo, output)
    raw.progress(output, 'preflight')
    probe_rows = data.training_candidates(per_class=config['raw_probe_per_class'])
    lookup = {sid: i for i, sid in enumerate(caches['train']['ids'])}
    numbers = [lookup[row['sample_id']] for row in probe_rows]
    batch = Batch.from_data_list([data.load(row) for row in probe_rows]).to(device)
    frozen_models = parent.build_models(source['model_state'], config, device)
    initial = parent.initial_check(frozen_models, caches['val'], device, config, gold, source['validation'])
    probes, resumes = {}, {}
    for name, old_name in FROZEN.items():
        model, saved = frozen_models[old_name], resumed[old_name]['saved']
        opt = parent.optimizer(model, config)
        sampler = raw.TrainBatches(len(rows['train']), config['batch_size'], config['seed'])
        resumes[name] = restore_training(model, opt, sampler, saved, config, 31650)
        require(parent.encoder_hash(model) == config['encoder_sha256'], 'Restored encoder hash differs')
        probes[name] = raw_update_probe(model, saved, caches['train'], numbers, batch, config, device)
    joint_models = {}
    for name, mode in JOINT.items():
        base = raw.OperatorClassifier('GAT', seed=42)
        base.load_state_dict(source['model_state'], strict=True)
        model = JointQueryPacketClassifier(base.flow_encoder, base.packet_encoder, base.fusion, mode, 42).to(device).eval()
        require(parent.encoder_hash(model) == config['encoder_sha256'], 'Joint source encoder hash differs')
        loader = graph_loader(data, rows['val'], 'val', config)
        metrics, predictions = joint_evaluate(model, loader, rows['val'], device, output, name+'_initial_raw')
        require(metrics == initial['epoch_zero'] and parent.identities(predictions) == parent.identities(gold),
                'Joint raw epoch0 differs from D32')
        probes[name] = joint_probe(model, caches['train'], numbers, batch, config, device)
        joint_models[name] = model
    states = {raw.state_digest(model.state_dict()) for model in joint_models.values()}
    original_states = {resume['saved']['config']['initial_state_sha256'] for resume in resumed.values()}
    require(states == original_states and len(states) == 1, 'Four-arm original query/D32 states differ')
    raw.write(output/'preflight.json', {'initialization': initial, 'resumes': resumes, 'probes': probes,
              'common_epoch_zero_state_sha256': next(iter(states)),
              'raw_train_sample_ids': [r['sample_id'] for r in probe_rows],
              'test_graph_loads': 0, 'formal_training_started': False})
    if not train:
        return
    results, exports = {}, {}
    for name, old_name in FROZEN.items():
        model = frozen_models[old_name]
        result, cached = fit(name, model, caches, device, config, binding, output,
                             resumed[old_name], results['Frozen-Mean']['orders'] if results else None)
        require(result['epoch_zero'] == initial['epoch_zero'], 'Frozen epoch0 differs from preflight')
        loader = graph_loader(data, rows['val'], 'val', config)
        measured, replay = joint_evaluate(model, loader, rows['val'], device, output, name+'_best_raw_replay', True)
        require(measured == result['best_validation'] and parent.identities(replay) == parent.identities(cached)
                and parent.encoder_hash(model) == config['encoder_sha256'], 'Frozen full selected raw replay differs')
        errors = {}
        for key in ('flow_weights', 'packet_attention'):
            a = torch.tensor([v for row in cached for v in row[key]])
            b = torch.tensor([v for row in replay for v in row[key]])
            require(a.shape == b.shape and torch.allclose(a, b, atol=1e-5, rtol=1e-5), 'Full frozen raw readout differs: '+key)
            errors[key] = float((a-b).abs().max())
        result['raw_readout_max_abs_errors'] = errors
        raw.save_predictions(output/name/'raw_validation_predictions.jsonl.gz', replay)
        result['raw_replay_passed'] = True
        result['raw_predictions_sha256'] = raw.sha(output/name/'raw_validation_predictions.jsonl.gz')
        require(result['optimizer_updates'] == 42200 and result['trainable_parameters'] == 8168, 'Frozen budget/parameters differ')
        raw.write(output/name/'metrics.json', result)
        results[name], exports[name] = result, cached
    for name, model in joint_models.items():
        result, predictions = fit_joint(name, model, data, rows, device, config, binding, output, results['Frozen-Mean']['orders'])
        require(result['epoch_zero'] == initial['epoch_zero'] and result['optimizer_updates'] == 42200
                and result['trainable_parameters'] == 25384, 'Joint initialization/budget/parameters differ')
        results[name], exports[name] = result, predictions
    require(len({tuple(result['orders']) for result in results.values()}) == 1
            and len({result['training_start_rng_sha256'] for result in results.values()}) == 1, 'Four-arm order/RNG differs')
    require(results['Joint-Mean']['initial_state_sha256'] == results['Joint-Flow']['initial_state_sha256']
            and results['Frozen-Mean']['continuation_start_rng_sha256'] == results['Frozen-Flow']['continuation_start_rng_sha256'],
            'Paired initial state/continuation RNG differs')
    analysis = analyze_four(results, exports, config, output)
    raw.write(output/'analysis.json', analysis)
    write_report(output, results, analysis)
    for path, digest in inputs.items():
        require(raw.sha(path) == digest, 'Pinned input changed during training: '+str(path))
    for path, digest in binding['sources'].items():
        require(raw.sha(raw.ROOT/path) == digest, 'Bound implementation changed during training: '+path)
    new_updates = sum(m['new_optimizer_updates'] for m in results.values())
    require(new_updates == 105500, 'Four-arm new update budget differs')
    raw.write(output/'summary.json', {'status': 'completed', 'protocol': config, 'binding': binding,
              'models': results, 'analysis': analysis, 'new_optimizer_updates': new_updates,
              'test_evaluated': False, 'source_artifacts_unchanged': True, 'automatic_followup': False})


def archive_failed_preflight(repo):
    """Preserve only the known pre-training failure; reject runs with updates."""
    root = (repo/'outputs').resolve()
    folder = root/'flow_query_joint_packet2_s42_v1'
    require(folder.is_dir() and not folder.is_symlink() and folder.resolve().parent == root,
            'Retry requires the named original failed output directory')
    status = raw.read(folder/'status.json')
    require(status.get('status') == 'failed' and status.get('error') == 'Adam update counter differs',
            'Retry only accepts the known original preflight Adam counter failure')
    preflight = raw.read(folder/'preflight.json')
    require(preflight.get('formal_training_started') is False, 'Original preflight receipt missing')
    require(not any((folder/n).exists() for n in ('summary.json', 'analysis.json', 'report.md')),
            'Refuse to archive a run with final results')
    for name, old_name in FROZEN.items():
        arm = folder/name
        require(not any((arm/n).exists() for n in (
            'last.pt', 'best.pt', 'metrics.json', 'resume_receipt.json', 'validation_predictions.jsonl.gz')),
            'Refuse to archive a run that reached formal continuation: '+name)
        ledger = arm/'epochs.jsonl'
        if ledger.exists():
            original = repo/'outputs/flow_query_pooling_s42_v1'/old_name/'epochs.jsonl'
            require(ledger.read_bytes() == original.read_bytes(), 'Failed run contains new epoch records: '+name)
    require(all(not (folder/name).exists() for name in JOINT), 'Joint training output already exists')
    backup = root/(folder.name+'_preflight_failed_'+str(time.time_ns()))
    # Both resolved endpoints stay in the explicitly named outputs directory.
    require(backup.resolve().parent == root and not backup.exists(), 'Invalid retry backup path')
    folder.rename(backup)
    print(json.dumps({'preserved_failed_preflight': str(backup), 'new_formal_updates_in_failed_run': 0}), flush=True)
    return backup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--retry-preflight-failure', action='store_true',
                        help='Preserve the known Adam preflight failure, then start from clean original last.pt files')
    args = parser.parse_args()
    repo = args.repo.resolve()
    if args.retry_preflight_failure:
        require(args.train, 'Retry flag requires --train')
        if (repo/'outputs/flow_query_joint_packet2_s42_v1').exists():
            archive_failed_preflight(repo)
    name = 'flow_query_joint_packet2_s42_v1' if args.train else 'flow_query_joint_packet2_preflight_s42_v1'
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
