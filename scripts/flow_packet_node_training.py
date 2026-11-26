"""Raw-graph paired NLL training with an all-arm transactional checkpoint."""
import json
import time

import torch
from torch.nn import functional as F
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

import joint_transition_training as transaction
import summarize_conditional_complement as metrics

raw, frozen, require = transaction.raw, transaction.frozen, transaction.require


def topology(batch, device):
    src, dst, metadata, lengths, _ = transaction.original.chronological_edges(batch)
    return tuple(torch.as_tensor(x).to(device) for x in (src, dst, metadata, lengths))


def frozen_digest(model):
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    return raw.state_digest({n: p for n, p in model.state_dict().items() if n not in trainable})


def optimizer(model, settings):
    groups = []
    scopes = [('packet2', list(model.packet_relations()), settings['encoder_learning_rate']),
              ('cross_layer', [model.cross_layer], settings['readout_learning_rate']),
              ('query_pool', [model.packet_pool], settings['readout_learning_rate']),
              ('fusion', [model.fusion], settings['fusion_learning_rate']),
              ('ordered', [model.readout], settings['ordered_learning_rate'])]
    if model.adjustment is not None:
        scopes.append(('adjustment', [model.adjustment], settings['new_learning_rate']))
    for name, modules, lr in scopes:
        params = [p for module in modules for p in module.parameters() if p.requires_grad]
        groups.append({'params': params, 'lr': lr, 'name': name})
    ids = [id(p) for g in groups for p in g['params']]
    require(len(ids) == len(set(ids)) and set(ids) == {id(p) for p in model.parameters() if p.requires_grad},
            'Optimizer scope differs')
    return torch.optim.Adam(groups, betas=tuple(settings['adam_betas']), eps=settings['adam_eps'],
                            weight_decay=settings['weight_decay'])


def identity(batch, expected):
    require(list(zip(batch.sample_id, batch.y.view(-1).tolist())) ==
            [(r['sample_id'], r['label']) for r in expected], 'Graph identities/labels differ')


@torch.no_grad()
def evaluate(models, data, rows, settings, device, output=None, phase='validation'):
    for m in models.values():
        m.eval()
    batches = DataLoader(raw.GraphRows(data, rows, 'val'), batch_size=settings['eval_batch_size'],
                         shuffle=False, num_workers=settings['workers'],
                         generator=torch.Generator().manual_seed(242))
    records = []
    for step, batch in enumerate(batches, 1):
        expected = rows[len(records):len(records) + batch.num_graphs]
        identity(batch, expected)
        links = topology(batch, device)
        batch = batch.to(device)
        outputs = {n: m(batch, links) for n, m in models.items()}
        require(all(bool(torch.isfinite(x).all()) for x in outputs.values()), 'Nonfinite validation')
        predictions = {n: x.argmax(-1).tolist() for n, x in outputs.items()}
        for i, row in enumerate(expected):
            records.append({'sample_id': row['sample_id'], 'label': row['label'],
                            'predictions': {n: p[i] for n, p in predictions.items()}})
        if output and (step == 1 or step % 200 == 0 or len(records) == len(rows)):
            raw.progress(output, phase, samples=len(records), total=len(rows))
    require(len(records) == len(rows), 'Incomplete validation cohort')
    return records, {n: metrics.metrics(records, n) for n in models}


def save_views(output, state):
    for name, saved in state['best'].items():
        transaction.atomic_save(output / name / 'best.pt', {**saved, 'binding': state['binding']})
        transaction.atomic_save(output / name / 'last.pt', {'state': state['models'][name],
            'epoch': state['completed_epoch'], 'updates': state['updates'], 'binding': state['binding']})
    raw.write(output / 'history.json', state['history'])
    raw.write(output / 'training_progress.json', {'completed_epoch': state['completed_epoch'],
              'updates_per_arm': state['updates'], 'cursor': state['sampler']['cursor'],
              'best_epochs': {n: s['epoch'] for n, s in state['best'].items()}})


def fit(models, data, rows, settings, binding, device, output, resume=True,
        fault=None, stop_after_updates=None):
    """No partial group update commits; restarting rolls back to the previous batch."""
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'recovery.pt'
    opts = {n: optimizer(m, settings) for n, m in models.items()}
    sampler = transaction.PairedSampler(len(rows['train']), settings['batch_size'], settings['sampling_seed'])
    fixed = {n: frozen_digest(m) for n, m in models.items()}
    if path.exists():
        require(resume, 'Recovery exists; use --resume auto')
        state = torch.load(path, map_location='cpu', weights_only=True)
        require(state['schema'] == 'xgnid.node-transaction.v1' and state['binding'] == binding
                and state['settings'] == settings and list(state['models']) == list(models),
                'Recovery code/source/config/arm mismatch')
        for n, m in models.items():
            m.load_state_dict(state['models'][n], strict=True)
            opts[n].load_state_dict(state['optimizers'][n])
            require(frozen_digest(m) == fixed[n], 'Frozen weights/BN changed')
            require(all(int(v['step']) == state['updates'] for v in opts[n].state.values()), 'Adam cursor differs')
        sampler.load_state_dict(state['sampler'])
        require(state['updates'] == state['completed_epoch'] * len(sampler) + state['epoch_updates']
                and state['epoch_samples'] == sampler.cursor, 'Recovery cursor differs')
        transaction.restore_rng(state['rng'])
        raw.write(output / 'resume_receipt.json', {'passed': True, 'updates_per_arm': state['updates'],
                  'restored': ['models', 'Adam', 'RNG', 'sampler', 'loss_sums']})
    else:
        require(not (output / 'history.json').exists(), 'History exists without recovery')
        records, zero = evaluate(models, data, rows['val'], settings, device, output, 'epoch_zero')
        state = {'schema': 'xgnid.node-transaction.v1', 'binding': binding, 'settings': settings,
                 'completed_epoch': 0, 'updates': 0, 'epoch_updates': 0, 'epoch_samples': 0,
                 'loss_sums': {n: 0. for n in models}, 'training_seconds': {n: 0. for n in models},
                 'best': {n: {'epoch': 0, 'validation': zero[n],
                            'predictions': [r['predictions'][n] for r in records],
                            'state': transaction.cpu_state(m)} for n, m in models.items()},
                 'history': [{'epoch': 0, 'validation': zero, 'order_sha256': None,
                              'samples_per_arm': 0, 'updates_per_arm': 0}]}

    def persist():
        state.update(models={n: transaction.cpu_state(m) for n, m in models.items()},
                     optimizers={n: o.state_dict() for n, o in opts.items()},
                     sampler=sampler.state_dict(), rng=transaction.rng_state())
        transaction.atomic_save(path, state)

    if not path.exists():
        persist()
        for n in models:
            transaction.atomic_save(output / n / 'epoch0.pt', {**state['best'][n], 'binding': binding})
    save_views(output, state)
    while state['completed_epoch'] < settings['epochs']:
        epoch = state['completed_epoch'] + 1
        if sampler.epoch != epoch:
            sampler.start(epoch)
            state.update(epoch_samples=0, epoch_updates=0, loss_sums={n: 0. for n in models})
        for m in models.values():
            m.train()
        while sampler.cursor < sampler.size:
            indices = sampler.peek()
            expected = [rows['train'][i] for i in indices]
            require(all(r['partition'] == 'train' for r in expected), 'Forbidden training partition')
            batch = Batch.from_data_list([data.load(r) for r in expected])
            identity(batch, expected)
            links = topology(batch, device)
            batch = batch.to(device)
            for n, model in models.items():
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                opts[n].zero_grad(set_to_none=True)
                loss = F.nll_loss(model(batch, links), batch.y.view(-1))
                require(bool(torch.isfinite(loss)), 'Nonfinite NLL: ' + n)
                loss.backward()
                require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
                        'Nonfinite gradient: ' + n)
                opts[n].step()
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                state['training_seconds'][n] += time.perf_counter() - started
                state['loss_sums'][n] += float(loss.detach()) * len(indices)
                if fault:
                    fault(n, state['updates'] + 1)
            sampler.cursor += len(indices)
            state['epoch_samples'] += len(indices)
            state['epoch_updates'] += 1
            state['updates'] += 1
            persist()
            if state['epoch_updates'] == 1 or state['epoch_updates'] % 50 == 0:
                raw.progress(output, 'paired_raw_training', epoch=epoch, samples=sampler.cursor,
                             total=sampler.size, updates_per_arm=state['updates'])
            if stop_after_updates == state['updates']:
                return state
        require(state['epoch_samples'] == sampler.size and state['epoch_updates'] == len(sampler),
                'Incomplete paired epoch')
        require(all(frozen_digest(m) == fixed[n] for n, m in models.items()), 'Frozen encoder/BN changed')
        records, measured = evaluate(models, data, rows['val'], settings, device, output, 'epoch_validation')
        for n, m in models.items():
            if measured[n]['macro_f1'] > state['best'][n]['validation']['macro_f1']:
                state['best'][n] = {'epoch': epoch, 'validation': measured[n],
                    'predictions': [r['predictions'][n] for r in records], 'state': transaction.cpu_state(m)}
        state['history'].append({'epoch': epoch, 'validation': measured,
            'samples_per_arm': state['epoch_samples'], 'updates_per_arm': state['epoch_updates'],
            'cumulative_updates_per_arm': state['updates'], 'order_sha256': sampler.order_sha256,
            'train_nll': {n: v / sampler.size for n, v in state['loss_sums'].items()}})
        state['completed_epoch'] = epoch
        state['epoch_updates'] = 0
        persist()
        save_views(output, state)
        raw.progress(output, 'epoch_completed', epoch=epoch, total_epochs=settings['epochs'],
                     macro_f1={n: m['macro_f1'] for n, m in measured.items()})
    require(state['updates'] == settings['epochs'] * len(sampler), 'Final update budget differs')
    return state
