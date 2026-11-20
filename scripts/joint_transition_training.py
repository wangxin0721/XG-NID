"""Transactional paired Adam training, including mid-epoch recovery."""
import hashlib
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

import run_ordered_transition_readout as frozen
from xgnid.joint_transition_fusion import JointTransitionFusion, MODES, ALIASES

raw, original, require = frozen.raw, frozen.original, frozen.require
ARMS = tuple(MODES)


def atomic_save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def cpu_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def rng_state():
    numpy = np.random.get_state()
    return {'python': random.getstate(), 'numpy': (numpy[0], numpy[1].tolist(), *numpy[2:]),
            'cpu': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(saved):
    require(len(saved['cuda']) == torch.cuda.device_count(), 'Recovery CUDA device count differs')
    random.setstate(saved['python'])
    numpy = saved['numpy']
    np.random.set_state((numpy[0], np.array(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(saved['cpu'].cpu())
    if saved['cuda']:
        torch.cuda.set_rng_state_all([value.cpu() for value in saved['cuda']])


class PairedSampler:
    """Same randperm and singleton-tail merge as raw.TrainBatches; explicit commit cursor."""
    def __init__(self, size, batch_size, seed):
        require(size >= 2 and batch_size >= 2, 'Singleton sampler')
        self.size, self.batch_size, self.seed = size, batch_size, seed
        self.generator = torch.Generator().manual_seed(seed)
        self.epoch, self.cursor, self.order = 0, 0, []

    def __len__(self):
        return math.ceil(self.size / self.batch_size) - int(self.size % self.batch_size == 1)

    @property
    def order_sha256(self):
        return hashlib.sha256(json.dumps(self.order).encode()).hexdigest() if self.order else None

    def start(self, epoch):
        if epoch == self.epoch:
            return
        require(epoch == self.epoch + 1 and (not self.epoch or self.cursor == self.size),
                'Sampler would repeat or skip an epoch')
        self.order = torch.randperm(self.size, generator=self.generator).tolist()
        self.cursor, self.epoch = 0, epoch

    def peek(self):
        require(self.epoch > 0 and self.cursor < self.size, 'No pending sampler batch')
        stop = self.size if self.size - self.cursor == self.batch_size + 1 else min(
            self.cursor + self.batch_size, self.size)
        return self.order[self.cursor:stop]

    def state_dict(self):
        return {'size': self.size, 'batch_size': self.batch_size, 'seed': self.seed,
                'epoch': self.epoch, 'cursor': self.cursor, 'order': self.order,
                'order_sha256': self.order_sha256, 'generator': self.generator.get_state()}

    def load_state_dict(self, saved):
        require((self.size, self.batch_size, self.seed) == tuple(saved[k] for k in ('size', 'batch_size', 'seed')),
                'Sampler configuration mismatch')
        self.epoch, self.cursor, self.order = saved['epoch'], saved['cursor'], saved['order']
        require((not self.epoch and not self.order and not self.cursor) or
                (sorted(self.order) == list(range(self.size)) and 0 <= self.cursor <= self.size),
                'Corrupt sampler order or cursor')
        require(self.order_sha256 == saved['order_sha256'], 'Sampler order digest mismatch')
        self.generator.set_state(saved['generator'].cpu())


def make_arms(source, settings, device):
    arms = {name: JointTransitionFusion(source.fusion, name, settings['readout_seed']).to(device) for name in ARMS}
    initial = raw.state_digest(source.fusion.state_dict())
    require(all(raw.state_digest(arm.fusion.state_dict()) == initial for arm in arms.values()),
            'Fusion initial parameters differ from the same original C')
    pointers = [{p.data_ptr() for p in arm.parameters()} for arm in arms.values()]
    require(all(not a.intersection(b) for i, a in enumerate(pointers) for b in pointers[i + 1:])
            and not set.union(*pointers).intersection(p.data_ptr() for p in source.parameters()),
            'Arms share parameter storage with each other or the reference C')
    common = arms['joint_single'].readout.state_dict()
    require(all(torch.equal(arms[name].readout.state_dict()[key], value)
                for name in ARMS[1:] for key, value in common.items()), 'Common readout initialization differs')
    require(raw.state_digest(arms['joint_pointwise'].readout.state_dict()) ==
            raw.state_digest(arms['joint_ordered'].readout.state_dict()), 'Context initialization differs')
    return arms


def optimizer(arm, settings):
    groups = [{'params': list(arm.fusion.parameters()), 'lr': settings['fusion_learning_rate'], 'name': 'fusion'}]
    if arm.readout is not None:
        groups.append({'params': list(arm.readout.parameters()), 'lr': settings['readout_learning_rate'],
                       'name': 'readout'})
    return torch.optim.Adam(groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)


def verify_source(source, digest):
    require(raw.state_digest(source.state_dict()) == digest
            and all(not p.requires_grad and p.grad is None for p in source.parameters())
            and all(not module.training for module in source.modules()),
            'Reference C parameters, buffers, gradients, or eval modes changed')


def gradient_probe(source, features, labels, settings, device):
    digest = raw.state_digest(source.state_dict())
    arms, receipt = make_arms(source, settings, device), {}
    baseline = original.baseline(source, features)
    require(bool(features.transitions[:, 70].any()), 'Gradient probe requires a real continuous context')
    for name, arm in arms.items():
        opt, steps = optimizer(arm, settings), []
        for step in range(3):
            arm.train()
            opt.zero_grad(set_to_none=True)
            scores, residual, _, _ = arm(features)
            if step == 0:
                require(torch.equal(scores, baseline) and not bool(residual.any()), f'{name} epoch zero differs')
            loss = F.nll_loss(scores, labels)
            loss.backward()
            gradients = {key: float(p.grad.abs().sum()) for key, p in arm.named_parameters() if p.grad is not None}
            require(set(gradients) == {key for key, p in arm.named_parameters() if p.requires_grad}
                    and all(math.isfinite(v) for v in gradients.values()), f'{name} missing/nonfinite gradient')
            for module in ('flow_projection', 'packet_projection', 'gate', 'shared_head'):
                require(sum(v for k, v in gradients.items() if k.startswith('fusion.' + module + '.')) > 0,
                        f'{name} no gradient reaches {module}')
            if arm.readout is not None:
                require(gradients['readout.output.weight'] > 0, f'{name} zero output gradient')
                if not step:
                    require(all(v == 0 for k, v in gradients.items() if k.startswith('readout.')
                                and k != 'readout.output.weight'), 'Internal readout gradient before W update')
                else:
                    for module in ('edge', 'query', *(('context',) if hasattr(arm.readout, 'context') else ())):
                        require(gradients['readout.' + module + '.weight'] > 0,
                                f'{name} missing staged {module} gradient')
            opt.step()
            steps.append({'step': step + 1, 'nll': float(loss.detach()), 'gradient_l1': gradients})
            verify_source(source, digest)
        receipt[name] = steps
    return {'passed': True, 'source_unchanged': True, 'independent_parameter_storage': True, 'steps': receipt}


@torch.no_grad()
def measure(arms, cache, rows, settings, device):
    predictions = {name: [] for name in arms}
    for arm in arms.values():
        arm.eval()
    for start in range(0, len(rows), settings['eval_batch_size']):
        features, _ = cache.batch(range(start, min(start + settings['eval_batch_size'], len(rows))), device)
        for name, arm in arms.items():
            scores, _, _, _ = arm(features)
            require(bool(torch.isfinite(scores).all()), f'{name} nonfinite validation score')
            predictions[name].extend(scores.argmax(-1).tolist())
    records = {name: original.prediction_records(rows, values) for name, values in predictions.items()}
    metrics = {name: frozen.joint.study.validate_predictions(items, rows, probabilities=False)
               for name, items in records.items()}
    return records, metrics


def save_views(output, state):
    """Recovery is authoritative; derived ledgers and best/last files can always be regenerated."""
    for name in ARMS:
        folder = output / name
        folder.mkdir(exist_ok=True)
        ledger = [{**{k: v for k, v in row.items() if k not in ('validation', 'train_nll')},
                   'validation': row['validation'][name],
                   'train_nll': row['train_nll'][name] if row['train_nll'] is not None else None}
                  for row in state['history']]
        temporary = folder / 'epochs.jsonl.tmp'
        temporary.write_text(''.join(json.dumps(row) + '\n' for row in ledger), encoding='utf-8')
        os.replace(temporary, folder / 'epochs.jsonl')
        atomic_save(folder / 'best.pt', {'binding': state['binding'], **state['best'][name]})
        atomic_save(folder / 'last.pt', {'binding': state['binding'], 'completed_epoch': state['completed_epoch'],
                                       'updates': state['updates'], 'state': state['models'][name]})
    raw.write(output / 'training_progress.json', {'completed_epoch': state['completed_epoch'],
              'current_epoch': state['sampler']['epoch'], 'cursor': state['sampler']['cursor'],
              'updates_per_arm': state['updates'], 'selected_epochs': {k: v['epoch'] for k, v in state['best'].items()}})


def fit(arms, train, val, rows, reference, settings, binding, device, output, resume=False, stop_after_updates=None):
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'recovery.pt'
    optimizers = {name: optimizer(arm, settings) for name, arm in arms.items()}
    sampler = PairedSampler(len(train.labels), settings['batch_size'], settings['readout_seed'])
    if path.exists():
        require(resume, 'Recovery already exists; use --resume')
        state = torch.load(path, map_location='cpu', weights_only=True)
        require(state['schema'] == 'xgnid.joint-transition-recovery.v1'
                and state['binding'] == binding and state['settings'] == settings, 'Recovery source/config/code mismatch')
        for name, arm in arms.items():
            arm.load_state_dict(state['models'][name], strict=True)
            optimizers[name].load_state_dict(state['optimizers'][name])
        sampler.load_state_dict(state['sampler'])
        require(state['epoch_samples'] == sampler.cursor
                and state['updates'] == state['completed_epoch'] * len(sampler) + state['epoch_updates'],
                'Recovery update or sample cursor mismatch')
        restore_rng(state['rng'])
        raw.write(output / 'resume_receipt.json', {'passed': True, 'completed_epoch': state['completed_epoch'],
                  'current_epoch': sampler.epoch, 'sampler_cursor': sampler.cursor,
                  'updates_per_arm': state['updates'], 'adam_restored': True, 'rng_restored': True})
        save_views(output, state)
    else:
        require(not (output / ARMS[0] / 'epochs.jsonl').exists(), 'Ledger exists without recovery; preserve and investigate')
        records, metrics = measure(arms, val, rows, settings, device)
        require(all(items == reference for items in records.values()), 'Full validation epoch zero differs from original C')
        state = {'schema': 'xgnid.joint-transition-recovery.v1', 'binding': binding, 'settings': settings,
                 'completed_epoch': 0, 'updates': 0, 'epoch_updates': 0, 'epoch_samples': 0,
                 'epoch_loss': {name: 0.0 for name in ARMS}, 'best': {}, 'history': []}
        for name, arm in arms.items():
            state['best'][name] = {'epoch': 0, 'validation': metrics[name], 'state': cpu_state(arm),
                                  'prediction_ids_sha256': frozen.joint.prediction_digest(records[name])}
        state['history'].append({'epoch': 0, 'updates_per_arm': 0, 'cumulative_updates_per_arm': 0,
                                 'samples_per_arm': 0, 'order_sha256': None, 'validation': metrics, 'train_nll': None})

    def persist():
        state.update(models={name: cpu_state(arm) for name, arm in arms.items()},
                     optimizers={name: opt.state_dict() for name, opt in optimizers.items()},
                     sampler=sampler.state_dict(), rng=rng_state())
        atomic_save(path, state)

    if not path.exists():
        persist()
        save_views(output, state)
    while state['completed_epoch'] < settings['epochs']:
        epoch = state['completed_epoch'] + 1
        if sampler.epoch != epoch:
            sampler.start(epoch)
            state.update(epoch_updates=0, epoch_samples=0, epoch_loss={name: 0.0 for name in ARMS})
        for arm in arms.values():
            arm.train()
        while sampler.cursor < sampler.size:
            indices = sampler.peek()
            features, labels = train.batch(indices, device)
            for name, arm in arms.items():
                opt = optimizers[name]
                opt.zero_grad(set_to_none=True)
                scores, _, _, _ = arm(features)
                loss = F.nll_loss(scores, labels)
                require(bool(torch.isfinite(loss)), f'{name} nonfinite training loss')
                loss.backward()
                require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in arm.parameters()),
                        f'{name} nonfinite training gradient')
                opt.step()
                state['epoch_loss'][name] += float(loss.detach()) * len(indices)
            sampler.cursor += len(indices)
            state['epoch_samples'] += len(indices)
            state['epoch_updates'] += 1
            state['updates'] += 1
            # One transaction includes all four Adam steps on exactly the same batch.
            persist()
            if stop_after_updates is not None and state['updates'] == stop_after_updates:
                return state
        require(state['epoch_samples'] == len(train.labels) and state['epoch_updates'] == len(sampler),
                'Incomplete epoch budget')
        records, metrics = measure(arms, val, rows, settings, device)
        for name, arm in arms.items():
            if metrics[name]['macro_f1'] > state['best'][name]['validation']['macro_f1']:
                state['best'][name] = {'epoch': epoch, 'validation': metrics[name], 'state': cpu_state(arm),
                                      'prediction_ids_sha256': frozen.joint.prediction_digest(records[name])}
        state['history'].append({'epoch': epoch, 'updates_per_arm': state['epoch_updates'],
            'cumulative_updates_per_arm': state['updates'], 'samples_per_arm': state['epoch_samples'],
            'order_sha256': sampler.order_sha256, 'validation': metrics,
            'train_nll': {name: loss / state['epoch_samples'] for name, loss in state['epoch_loss'].items()}})
        state['completed_epoch'] = epoch
        # Completed-epoch recovery has no pending updates; retain the exhausted sampler as evidence.
        state['epoch_updates'] = 0
        persist()
        save_views(output, state)
        raw.progress(output, 'joint_paired_training', epoch=epoch, total_epochs=settings['epochs'],
                     updates_per_arm=state['updates'], macro_f1={k: v['macro_f1'] for k, v in metrics.items()})
    require(state['updates'] == len(sampler) * settings['epochs'], 'Final update budget differs')
    return state
