"""Matched additive and multiplicative residuals for mature D32-Gate."""
import torch
from torch import nn
from torch.nn import functional as F

from .fusion_dimension import FusionHead


class InteractionFusionHead(FusionHead):
    def __init__(self, streams, mode, seed=42):
        super().__init__(32, streams)
        if mode not in ('add', 'multiply'):
            raise ValueError('Unknown interaction mode')
        self.mode = mode
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
            self.interaction_flow = nn.Linear(32, 8, bias=False)
            self.interaction_packet = nn.Linear(32, 8, bias=False)
            self.interaction_output = nn.Linear(8, 32, bias=False)
        nn.init.zeros_(self.interaction_output.weight)

    def interaction_parameters(self):
        return [self.interaction_flow.weight, self.interaction_packet.weight, self.interaction_output.weight]

    def forward(self, flow, packet):
        if flow.ndim != 2 or flow.shape != packet.shape or flow.size(-1) != 64:
            raise ValueError('Expected paired [batch,64] encoder representations')
        zf, zp = self.flow_projection(flow), self.packet_projection(packet)
        stats = torch.cat((zf, zp, self.concentration(zf)[:, None], self.concentration(zp)[:, None],
                           self.margin(zf)[:, None], self.margin(zp)[:, None],
                           F.cosine_similarity(zf, zp, dim=-1)[:, None]), dim=-1)
        alpha = self.gate(stats).sigmoid()
        base = alpha * zf + (1 - alpha) * zp
        u, v = self.interaction_flow(zf), self.interaction_packet(zp)
        combined = u + v if self.mode == 'add' else u * v
        enhanced = base + (4 * alpha * (1 - alpha)) * self.interaction_output(combined)
        return F.log_softmax(self.shared_head(enhanced), dim=-1), alpha
