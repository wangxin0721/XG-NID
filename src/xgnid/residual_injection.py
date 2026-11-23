"""Matched fixed, global and per-flow scalar injection above frozen graph features."""
from copy import deepcopy
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from .ordered_transition_readout import OrderedTransitionReadout


ARMS = ('inj_fixed', 'inj_global', 'inj_dynamic')
ALIASES = dict(zip(ARMS, ('R0', 'R1', 'R2')))


class InjectionOutput(NamedTuple):
    scores: torch.Tensor
    residual: torch.Tensor
    injected: torch.Tensor
    alpha: torch.Tensor
    attention: torch.Tensor
    beta: torch.Tensor


class InjectionGate(nn.Module):
    """Identically initialized 97->16->1 MLP; no global RNG consumption."""
    def __init__(self, seed=20261123):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
            self.network = nn.Sequential(nn.Linear(97, 16), nn.GELU(), nn.Linear(16, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, inputs):
        return 2 * self.network(inputs).sigmoid()


class ResidualInjectionFusion(nn.Module):
    def __init__(self, source_fusion, mode, readout_seed=42, gate_seed=20261123, epsilon=1e-8):
        super().__init__()
        if mode not in ARMS or source_fusion.width != 32:
            raise ValueError('Expected a registered residual-injection arm and D32 fusion')
        self.mode, self.epsilon = mode, epsilon
        self.fusion = deepcopy(source_fusion).requires_grad_(True)
        self.readout = OrderedTransitionReadout('ordered', readout_seed)
        self.injection_gate = None if mode == 'inj_fixed' else InjectionGate(gate_seed)

    def gate_inputs(self, features, residual):
        if self.mode == 'inj_global':
            return features.packet.new_zeros((len(features.packet), 97))
        # Both projections are live, BEFORE injection. No detach or cached projection.
        zf = self.fusion.flow_projection(features.flow)
        zp = self.fusion.packet_projection(features.packet)
        lf, lp = F.layer_norm(zf, (32,), eps=1e-5), F.layer_norm(zp, (32,), eps=1e-5)
        relative = torch.log1p(residual.norm(dim=-1) / (features.packet.norm(dim=-1) + self.epsilon))
        return torch.cat((lf, lp, (lf - lp).abs(), relative[:, None]), dim=-1)

    def detailed_forward(self, features):
        residual, attention = self.readout(features.packet_mean, features.transitions, features.lengths)
        beta = (features.packet.new_ones((len(features.packet), 1)) if self.injection_gate is None
                else self.injection_gate(self.gate_inputs(features, residual)))
        injected = beta * residual
        scores, alpha = self.fusion(features.flow, features.packet + injected)
        return InjectionOutput(scores, residual, injected, alpha, attention, beta)

    def forward(self, features):
        result = self.detailed_forward(features)
        return result.scores, result.injected, result.alpha, result.attention
