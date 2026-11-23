"""Matched initialization, frozen-range probes and resumable three-arm optimization."""
import math

import torch
from torch.nn import functional as F

import joint_transition_training as parent
from xgnid.fixed_transition_stability import ARMS, StabilityFusion

raw, frozen, original, require = parent.raw, parent.frozen, parent.original, parent.require


def make_arms(source, settings, device):
    rng = parent.rng_state()
    arms = {name: StabilityFusion(source.fusion, name, settings['readout_seed']).to(device) for name in ARMS}
    require(torch.equal(rng['cpu'], torch.get_rng_state()), 'Initialization consumed global CPU RNG')
    fusion = raw.state_digest(source.fusion.state_dict())
    common = arms['frozen_single'].readout.state_dict()
    pointers = [{p.data_ptr() for p in arm.parameters()} for arm in arms.values()]
    require(all(not a.intersection(b) for i, a in enumerate(pointers) for b in pointers[i+1:])
            and not set.union(*pointers).intersection(p.data_ptr() for p in source.parameters()),
            'Model parameter storage is shared')
    for name, arm in arms.items():
        require(raw.state_digest(arm.fusion.state_dict()) == fusion
                and all(torch.equal(arm.readout.state_dict()[k], v) for k, v in common.items())
                and not bool(arm.readout.output.weight.any()), 'Common model initialization differs')
        require(sum(p.numel() for p in arm.parameters() if p.requires_grad) ==
                settings['trainable_parameters'][name], 'Unexpected trainable parameter count')
        require(not hasattr(arm, 'injection_gate'), 'Stability study must not instantiate beta gates')
    return arms


def optimizer(arm, settings):
    groups = []
    fusion = [p for p in arm.fusion.parameters() if p.requires_grad]
    if fusion:
        groups.append({'name': 'fusion', 'params': fusion, 'lr': settings['fusion_learning_rate']})
    groups.append({'name': 'readout', 'params': list(arm.readout.parameters()), 'lr': settings['readout_learning_rate']})
    return torch.optim.Adam(groups, betas=tuple(settings['adam_betas']), eps=settings['adam_eps'],
                            weight_decay=settings['weight_decay'], amsgrad=False)


def fit(*args, **kwargs):
    return parent.fit(*args, **kwargs, optimizer_factory=optimizer,
                      recovery_schema='xgnid.fixed-transition-stability-recovery.v1')


def gradient_probe(source, features, labels, settings, device):
    digest, rng = raw.state_digest(source.state_dict()), parent.rng_state()
    arms, receipt = make_arms(source, settings, device), {}
    baseline = original.baseline(source, features)
    require(bool(features.transitions[:, 70].any()), 'Need true continuous context in probe')
    for name, arm in arms.items():
        opt, steps = optimizer(arm, settings), []
        fusion_hash = raw.state_digest(arm.fusion.state_dict())
        for step in range(3):
            arm.train(); opt.zero_grad(set_to_none=True)
            scores, residual, _, _ = arm(features)
            if step == 0:
                require(torch.equal(scores, baseline) and not bool(residual.any()), name + ' initial outputs differ')
            loss = F.nll_loss(scores, labels); loss.backward()
            grads = {k: float(p.grad.abs().sum()) for k, p in arm.named_parameters() if p.grad is not None}
            require(set(grads) == {k for k, p in arm.named_parameters() if p.requires_grad}
                    and all(math.isfinite(v) for v in grads.values()), name + ' bad gradient paths')
            require(grads['readout.output.weight'] > 0, name + ' no output gradient')
            if step == 0:
                require(all(v == 0 for k, v in grads.items() if k.startswith('readout.')
                            and k != 'readout.output.weight'), 'Hidden gradients before zero output update')
            else:
                for module in ('edge', 'query', *(('context',) if name == 'joint_ordered' else ())):
                    require(grads['readout.' + module + '.weight'] > 0, name + ' staged gradient missing')
            if name == 'frozen_single':
                require(all(not p.requires_grad and p.grad is None for p in arm.fusion.parameters())
                        and all(not m.training for m in arm.fusion.modules()), 'Frozen Single fusion changed mode/grad')
            else:
                for module in ('flow_projection', 'packet_projection', 'gate', 'shared_head'):
                    require(sum(v for k, v in grads.items() if k.startswith('fusion.' + module + '.')) > 0,
                            name + ' joint fusion path missing')
            opt.step()
            if name == 'frozen_single':
                require(raw.state_digest(arm.fusion.state_dict()) == fusion_hash, 'Frozen fusion parameters/buffers changed')
            parent.verify_source(source, digest)
            steps.append({'step': step+1, 'nll': float(loss.detach()), 'gradient_l1': grads})
        receipt[name] = steps
    parent.restore_rng(rng)
    return {'passed': True, 'steps': receipt, 'probe_updates_discarded': True, 'rng_restored': True,
            'source_encoders_cross_layer_and_frozen_single_fusion_unchanged': True}
