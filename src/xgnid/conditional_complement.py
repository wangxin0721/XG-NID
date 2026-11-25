"""Fixed-coordinate conditional corrections; inference never accepts labels."""
from copy import deepcopy
from collections import namedtuple
import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from .fusion_dimension import FusionHead

ARMS = ('cc_t0', 'cc_t1', 'cc_t2', 'cc_t3', 'cc_t4')
ALIASES = dict(zip(ARMS, ('T0', 'T1', 'T2', 'T3', 'T4')))
Projected = namedtuple('Projected', 'flow packet statistics reference')


def stream(seed, name):
    return int.from_bytes(hashlib.sha256(f'{seed}:{name}'.encode()).digest()[:8], 'big') % (2**63-1)


def normalized(z):
    return F.layer_norm(z, (32,))


def gate_input(zf, zp):
    return torch.cat((zf, zp, FusionHead.concentration(zf)[:, None],
        FusionHead.concentration(zp)[:, None], FusionHead.margin(zf)[:, None],
        FusionHead.margin(zp)[:, None], F.cosine_similarity(zf, zp, dim=-1)[:, None]), -1)


@torch.no_grad()
def project(ordered, features):
    residual, _ = ordered.readout(features.packet_mean, features.transitions, features.lengths)
    zf = ordered.fusion.flow_projection(features.flow)
    zp = ordered.fusion.packet_projection(features.packet + residual)
    stats = gate_input(zf, zp)
    alpha = ordered.fusion.gate(stats).sigmoid()
    scores = F.log_softmax(ordered.fusion.shared_head(alpha*zf + (1-alpha)*zp), -1)
    return Projected(zf.detach(), zp.detach(), stats.detach(), scores.detach())


class Preparation(nn.Module):
    """Train-only linear teachers plus two conditional affine predictors."""
    def __init__(self, seed=42):
        super().__init__()
        for name in ('teacher_flow', 'teacher_packet', 'flow_to_packet', 'packet_to_flow'):
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(stream(seed, name))
                setattr(self, name, nn.Linear(32, 8 if name.startswith('teacher') else 32))

    def losses(self, features, labels):
        zf, zp = features.flow.detach(), features.packet.detach()
        vf, vp = normalized(zf), normalized(zp)
        return {'flow_ce': F.cross_entropy(self.teacher_flow(zf), labels),
                'packet_ce': F.cross_entropy(self.teacher_packet(zp), labels),
                'flow_to_packet_mse': F.mse_loss(self.flow_to_packet(vf), vp),
                'packet_to_flow_mse': F.mse_loss(self.packet_to_flow(vp), vf)}

    @torch.no_grad()
    def teachers(self, features):
        return self.teacher_flow(features.flow), self.teacher_packet(features.packet)


class ConditionalComplement(nn.Module):
    def __init__(self, fusion, preparation, arm, seed=42):
        super().__init__()
        if arm not in ARMS:
            raise ValueError('Unregistered conditional complement arm')
        self.arm = arm
        self.gate = deepcopy(fusion.gate).requires_grad_(True)
        self.shared_head = deepcopy(fusion.shared_head).requires_grad_(True)
        self.correction = None
        with torch.random.fork_rng(devices=[]):
            if arm in ('cc_t1', 'cc_t3'):
                torch.random.default_generator.manual_seed(stream(seed, 'ordinary_mlp'))
                self.correction = nn.Sequential(nn.Linear(64, 40), nn.LeakyReLU(), nn.Linear(40, 32))
            elif arm in ('cc_t2', 'cc_t4'):
                torch.random.default_generator.manual_seed(stream(seed, 'conditional_interaction'))
                self.flow_to_packet = deepcopy(preparation.flow_to_packet).requires_grad_(False).eval()
                self.packet_to_flow = deepcopy(preparation.packet_to_flow).requires_grad_(False).eval()
                self.left = nn.Linear(32, 8, bias=False)
                self.right = nn.Linear(32, 8, bias=False)
                self.correction = nn.Sequential(nn.Linear(72, 32), nn.LeakyReLU(), nn.Linear(32, 32))
        if self.correction is not None:
            nn.init.zeros_(self.correction[-1].weight)
            nn.init.zeros_(self.correction[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        for name in ('flow_to_packet', 'packet_to_flow'):
            if hasattr(self, name):
                getattr(self, name).eval()
        return self

    def forward(self, features):
        # Explicitly detach stable coordinates and recompute the trainable gate/head.
        zf, zp, stats = (x.detach() for x in features[:3])
        alpha = self.gate(stats).sigmoid()
        hidden = alpha*zf + (1-alpha)*zp
        if self.arm in ('cc_t2', 'cc_t4'):
            vf, vp = normalized(zf), normalized(zp)
            uf, up = vf-self.packet_to_flow(vp), vp-self.flow_to_packet(vf)
            cross = self.left(uf)*self.right(up)
            delta = self.correction(torch.cat((uf, up, cross), -1))
        elif self.correction is not None:
            delta = self.correction(torch.cat((zf, zp), -1))
        else:
            delta = torch.zeros_like(hidden)
        return F.log_softmax(self.shared_head(hidden+delta), -1), delta, alpha


def collaborative_loss(scores, labels, reference, teacher_flow, teacher_packet, tau=2.):
    """Labels are used exclusively here, for detached training supervision."""
    y = labels.detach()
    qf, qp, p0 = teacher_flow.detach(), teacher_packet.detach(), reference.detach()
    cf, cp = qf.argmax(-1).eq(y), qp.argmax(-1).eq(y)
    mask = cf ^ cp
    teacher = torch.where(cf[:, None], qf, qp)
    q = F.softmax(teacher/tau, -1)
    sd = (F.kl_div(F.log_softmax(scores/tau, -1), q, reduction='none').sum(-1)
          * mask.to(scores.dtype)).sum()*tau**2/scores.size(0)
    correct = p0.argmax(-1).eq(y)
    keep = (F.relu(p0.gather(1, y[:, None]).squeeze(1)-scores.gather(1, y[:, None]).squeeze(1))
            * correct.to(scores.dtype)).sum()/scores.size(0)
    return {'nll': F.nll_loss(scores, y), 'sd': sd, 'keep': keep}


def candidate_loss(model, features, labels, preparation):
    scores, _, _ = model(features)
    if model.arm in ('cc_t3', 'cc_t4'):
        qf, qp = preparation.teachers(features)
        terms = collaborative_loss(scores, labels, features.reference, qf, qp)
    else:
        terms = {'nll': F.nll_loss(scores, labels), 'sd': scores.new_zeros(()), 'keep': scores.new_zeros(())}
    return terms['nll'] + .1*terms['sd'] + .1*terms['keep'], terms
