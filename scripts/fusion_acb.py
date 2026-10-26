"""A/C/B controls. Import is standard-library only; tensor work is server-only."""
from __future__ import annotations

import random

from e3_gate_study import identity, require
from run_e3_gate_ablation_server import new_model as original_equal_init

VARIANTS = ('A_equal', 'C_global', 'B_dynamic')
DESIGN_ID = '3729044857e0fe39dacfbc2894f5157404351173d86d4a22bfaad4e1e5373cd7'
INACTIVE_HEADS = ('webbased_head.', 'webbased_recon_head.', 'benign_spoofing_head.')


def check_config(config, design):
    require(identity(design) == DESIGN_ID, 'Registered A/C/B design changed; do not silently alter this round')
    require(config['schema'] == 'xgnid.fusion-acb-execution.v1', 'Wrong A/C/B protocol')
    require(config['design'] == design, 'Execution differs from registered A/C/B design')
    require(config['design']['loss']['lambda'] == .05, 'This round fixes lambda=.05')
    require(config['design']['scope'] == list(VARIANTS), 'Unregistered variant')
    require(config['design']['gpu_hour_budget_enforced'] is False, 'GPU-hour limits were cancelled')
    require(config['seed'] == 42 and config['test_evaluated'] is False, 'Seed42 validation screening only')
    require(config['probe_updates'] == 3, 'Use three training batches per probe')


def validate_probe(report, status, binding):
    require((report.get('status'), report.get('exit_code'), report.get('mode')) ==
            ('completed', 0, 'probe'), 'Completed joint A/C/B probe required')
    require((status.get('status'), status.get('exit_code'), status.get('mode')) ==
            ('completed', 0, 'probe'), 'Probe process did not finish')
    require(report['binding'] == binding and report['test_evaluated'] is False,
            'Probe configuration/source/evidence binding differs')
    require(set(report['variants']) == set(VARIANTS), 'All three probes required')
    certs = [report['variants'][v]['initialization'] for v in VARIANTS]
    for key in ('common_initial_state_sha256', 'training_start_rng_sha256'):
        require(len({c[key] for c in certs}) == 1, 'Common initialization/RNG differs')
    for v in VARIANTS:
        result = report['variants'][v]
        require(result['optimizer_update_verified'] is True and len(result['updates']) == 3,
                'Incomplete probe updates')
        require(result['auxiliary_gradient_paths_verified'] is True, 'Branch supervision gradient missing')
        if v != 'A_equal':
            require(result['fusion_parameter_update_verified'] is True, 'Fusion weights did not learn')


def rng_state(core):
    return {'python': random.getstate(), 'torch': core.torch.get_rng_state().clone(),
            'cuda': [s.clone() for s in core.torch.cuda.get_rng_state_all()]}


def restore_rng(core, state):
    random.setstate(state['python'])
    core.torch.set_rng_state(state['torch'])
    core.torch.cuda.set_rng_state_all(state['cuda'])


def rng_hash(core, state):
    return identity({'python': state['python'], 'tensor_states': core.state_hash(
        [('cpu', state['torch'])] + [('cuda_%d' % i, s) for i, s in enumerate(state['cuda'])])})


def build_model(core, frozen, device, variant, expected_z_initialization):
    require(variant in VARIANTS, 'Unknown fusion variant')
    # The exact frozen E3a-Z constructor, lazy materialization and zero-init.
    model, original_certificate = original_equal_init(core, frozen, device, 'E3a_equal_init')
    for key in ('original_initial_state_sha256', 'variant_initial_state_sha256', 'forward_ast_sha256'):
        require(original_certificate[key] == expected_z_initialization[key], 'E3a-Z initial identity differs: ' + key)
    saved_rng = rng_state(core)
    common = {n: t.detach().clone() for n, t in model.state_dict().items() if not n.startswith('gate.')}
    if variant == 'A_equal':
        model.use_gate = False
    elif variant == 'C_global':
        class GlobalGate(core.torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logit = core.torch.nn.Parameter(core.torch.zeros((), device=device))

            def forward(self, flow, *unused):
                weight = self.logit.sigmoid()
                return core.torch.stack((weight, 1 - weight)).expand(flow.size(0), 2)
        model.gate = GlobalGate()
    # A keeps its unused original gate for compatibility; counts disclose it.
    # Inactive historical heads are retained exactly as in E3a-Z, never supervised.
    model.load_state_dict(common, strict=False)
    require(core.state_hash(common.items()) == core.state_hash(
        (n, t) for n, t in model.state_dict().items() if not n.startswith('gate.')),
        'Common encoder/projection/head state changed')
    certificate = {'variant': variant, 'original_equal_init': original_certificate,
                   'common_initial_state_sha256': core.state_hash(common.items()),
                   'full_initial_state_sha256': core.state_hash(model.state_dict().items()),
                   'training_start_rng_sha256': rng_hash(core, saved_rng)}
    restore_rng(core, saved_rng)
    # GPU means are recorded in the legacy certificate, not compared bitwise.
    certificate['original_equal_init'].pop('actual_initial_weight_mean')
    return model, certificate


def parameter_counts(model, variant):
    active = [(n, p) for n, p in model.named_parameters()
              if not n.startswith(INACTIVE_HEADS) and not (variant == 'A_equal' and n.startswith('gate.'))]
    return {'total': sum(p.numel() for p in model.parameters()),
            'requires_grad': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'active_in_training_loss': sum(p.numel() for _, p in active),
            'fusion_active': 0 if variant == 'A_equal' else sum(p.numel() for p in model.gate.parameters()),
            'inactive_historical_heads_retained': True,
            'unused_original_gate_retained': variant == 'A_equal'}


def losses(core, model, batch):
    prediction = model(batch)
    y = batch.y.view(-1)
    cache = model._last_cache
    # Explicitly supervise H(z); NEVER use CE(z,y) or model.loss().
    main = core.F.nll_loss(prediction, y)
    flow = core.F.cross_entropy(model.shared_head(cache['flow_logits']), y)
    packet = core.F.cross_entropy(model.shared_head(cache['packet_logits']), y)
    total = main + .05 / 2 * (flow + packet)
    return total, main, flow, packet


def auxiliary_gradient_check(core, model, parts):
    evidence = {}
    for branch, loss in zip(('flow', 'packet'), parts[2:]):
        z = model._last_cache[branch + '_logits']
        projection = getattr(model, branch + '_classifier').weight
        targets = [z, projection, model.shared_head[0].weight] + list(getattr(model, branch + '_encoder').parameters())
        grads = core.torch.autograd.grad(loss, targets, retain_graph=True, allow_unused=True)
        values = [float(g.detach().abs().max()) if g is not None else 0. for g in grads]
        # Some GAT attention tensors legitimately have zero gradients on
        # single-neighbor relations. Check the encoder as a whole.
        require(all(bool(core.torch.isfinite(g).all()) for g in grads if g is not None)
                and all(v > 0 for v in values[:3]) and max(values[3:], default=0.) > 0,
                'Auxiliary loss must reach representation, projection, shared head and encoder')
        evidence[branch] = dict(zip(('representation', 'projection', 'shared_head', 'encoder_max'),
                                    values[:3] + [max(values[3:])]))
    return evidence


def update(core, model, optimizer, batch, variant, probe=False, initial=False):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    parts = losses(core, model, batch)
    require(all(bool(core.torch.isfinite(loss)) for loss in parts), 'Nonfinite loss')
    evidence = auxiliary_gradient_check(core, model, parts) if probe else None
    weights = model._last_cache['gate_weights'].detach()
    if initial:
        require(core.torch.equal(weights, core.torch.full_like(weights, .5)), 'Initial weights are not exactly .5/.5')
    require(bool(core.torch.isfinite(weights).all()) and bool((weights >= 0).all())
            and bool((weights <= 1).all()) and bool(core.torch.allclose(weights.sum(-1), weights.new_ones(weights.size(0)), atol=1e-6)),
            'Invalid final fusion weights')
    if variant == 'A_equal':
        require(core.torch.equal(weights, core.torch.full_like(weights, .5)), 'A is not equal fusion')
    if variant == 'C_global':
        require(core.torch.equal(weights, weights[:1].expand_as(weights)), 'C weights vary across samples')
    parts[0].backward()
    core.torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
    gate_grads = {n: float(p.grad.abs().max()) if p.grad is not None else None
                  for n, p in model.gate.named_parameters()} if probe else None
    optimizer.step()
    return {'loss_total': float(parts[0].detach()), 'loss_fusion': float(parts[1].detach()),
            'loss_flow': float(parts[2].detach()), 'loss_packet': float(parts[3].detach()),
            'fusion_weight_mean': weights.mean(0).cpu().tolist() if probe else None,
            'gate_gradients': gate_grads, 'auxiliary_gradient_paths': evidence}
