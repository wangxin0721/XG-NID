"""Initialization, staged gradients and transactional recovery for three injection arms."""
import math

import torch
from torch.nn import functional as F

import joint_transition_training as parent
from xgnid.residual_injection import ARMS, ResidualInjectionFusion

raw, original, frozen, require = parent.raw, parent.original, parent.frozen, parent.require


def make_arms(source, settings, device):
    before_rng = parent.rng_state()
    arms = {name: ResidualInjectionFusion(source.fusion, name, settings['readout_seed'],
            settings['injection_gate_seed'], settings['injection_epsilon']).to(device) for name in ARMS}
    require(torch.equal(before_rng['cpu'], torch.get_rng_state()), 'Arm initialization consumed training RNG')
    initial = raw.state_digest(source.fusion.state_dict())
    readout = raw.state_digest(arms[ARMS[0]].readout.state_dict())
    pointers = [{p.data_ptr() for p in arm.parameters()} for arm in arms.values()]
    require(all(not a.intersection(b) for i, a in enumerate(pointers) for b in pointers[i + 1:])
            and not set.union(*pointers).intersection(p.data_ptr() for p in source.parameters()),
            'Injection arms share parameter storage')
    for name, arm in arms.items():
        require(raw.state_digest(arm.fusion.state_dict()) == initial
                and raw.state_digest(arm.readout.state_dict()) == readout
                and not bool(arm.readout.output.weight.any()), 'Common initial state differs')
        require(sum(p.numel() for p in arm.fusion.parameters()) == 7144
                and sum(p.numel() for p in arm.readout.parameters()) == 3728,
                'Historical J3 fusion/readout parameter count differs')
    require(raw.state_digest(arms['inj_global'].injection_gate.state_dict()) ==
            raw.state_digest(arms['inj_dynamic'].injection_gate.state_dict()), 'Injection gate initialization differs')
    return arms


def optimizer(arm, settings):
    groups = [{'params': list(arm.fusion.parameters()), 'lr': settings['fusion_learning_rate'], 'name': 'fusion'},
              {'params': list(arm.readout.parameters()), 'lr': settings['readout_learning_rate'], 'name': 'readout'}]
    if arm.injection_gate is not None:
        groups.append({'params': list(arm.injection_gate.parameters()),
                       'lr': settings['injection_learning_rate'], 'name': 'injection_gate'})
    return torch.optim.Adam(groups, betas=tuple(settings['adam_betas']), eps=settings['adam_eps'],
                            weight_decay=settings['weight_decay'], amsgrad=False)


def fit(*args, **kwargs):
    return parent.fit(*args, **kwargs, optimizer_factory=optimizer,
                      recovery_schema='xgnid.residual-injection-recovery.v1')


def gradient_probe(source, features, labels, settings, device):
    digest, rng = raw.state_digest(source.state_dict()), parent.rng_state()
    arms, receipts = make_arms(source, settings, device), {}
    baseline = original.baseline(source, features)
    require(bool(features.transitions[:, 70].any()), 'Injection gradient probe needs true contexts')
    for name, arm in arms.items():
        opt, steps = optimizer(arm, settings), []
        for step in range(5):
            arm.train(); opt.zero_grad(set_to_none=True)
            result = arm.detailed_forward(features)
            if step == 0:
                require(torch.equal(result.scores, baseline) and not bool(result.residual.any())
                        and bool((result.beta == 1).all()), name + ' epoch zero differs')
            loss = F.nll_loss(result.scores, labels)
            loss.backward()
            gradients = {key: float(p.grad.abs().sum()) for key, p in arm.named_parameters() if p.grad is not None}
            require(set(gradients) == {key for key, p in arm.named_parameters() if p.requires_grad}
                    and all(math.isfinite(v) for v in gradients.values()), name + ' missing/nonfinite gradient')
            for module in ('flow_projection', 'packet_projection', 'gate', 'shared_head'):
                require(sum(v for k, v in gradients.items() if k.startswith('fusion.' + module + '.')) > 0,
                        name + ' gradient misses fusion.' + module)
            require(gradients['readout.output.weight'] > 0, name + ' missing output gradient')
            if step == 0:
                require(all(v == 0 for k, v in gradients.items() if k.startswith('readout.')
                            and k != 'readout.output.weight'), 'Readout hidden gradient before output update')
                require(all(v == 0 for k, v in gradients.items() if k.startswith('injection_gate.')),
                        'Injection gate gradient with zero residual')
            if step >= 1:
                for module in ('edge', 'query', 'context'):
                    require(gradients['readout.' + module + '.weight'] > 0, name + ' staged readout gradient missing')
            if arm.injection_gate is not None:
                if step >= 1:
                    require(gradients['injection_gate.network.2.bias'] > 0
                            and gradients['injection_gate.network.2.weight'] > 0, name + ' gate output disconnected')
                if step >= 2:
                    require(gradients['injection_gate.network.0.bias'] > 0, name + ' gate hidden bias disconnected')
                    require((gradients['injection_gate.network.0.weight'] > 0) == (name == 'inj_dynamic'),
                            name + ' gate input path differs from registered design')
                if name == 'inj_global':
                    require(bool((result.beta == result.beta[:1]).all()), 'R1 beta is not global')
            opt.step(); parent.verify_source(source, digest)
            steps.append({'step': step + 1, 'nll': float(loss.detach()), 'gradient_l1': gradients,
                          'beta_min': float(result.beta.detach().min()), 'beta_max': float(result.beta.detach().max())})
        if name == 'inj_dynamic':
            require(float(result.beta.detach().std()) > 0, 'R2 failed to develop per-flow beta variation')
        receipts[name] = steps
    parent.restore_rng(rng)
    return {'passed': True, 'steps': receipts, 'probe_updates_discarded': True, 'rng_restored': True,
            'source_parameters_buffers_modes_unchanged': True,
            'R1_hidden_input_weight_zero_gradient_expected': True,
            'R1_hidden_bias_and_output_learnable': True}
