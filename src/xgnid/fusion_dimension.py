"""Fresh fusion heads on frozen N0 encoders; independent of historical models."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import global_mean_pool


def fresh_linear(inputs, outputs, stream):
    """Default Linear initialization on a named CPU RNG stream; no RNG leak."""
    state = torch.get_rng_state()
    try:
        generator = torch.Generator(device='cpu').manual_seed(stream)
        torch.set_rng_state(generator.get_state())
        return nn.Linear(inputs, outputs)
    finally:
        torch.set_rng_state(state)


class FusionHead(nn.Module):
    def __init__(self, width, streams):
        super().__init__()
        if width not in (8, 32):
            raise ValueError('Only the registered 8/32 pair is supported')
        self.width = width
        self.flow_projection = fresh_linear(64, width, streams['flow_projection'])
        self.packet_projection = fresh_linear(64, width, streams['packet_projection'])
        self.gate = nn.Sequential(
            fresh_linear(2 * width + 5, 16, streams['gate_hidden']), nn.LeakyReLU(),
            fresh_linear(16, width, streams['gate_output']))
        self.shared_head = nn.Sequential(
            fresh_linear(width, 32, streams['head_hidden']), nn.LeakyReLU(),
            fresh_linear(32, 8, streams['head_output']))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    @staticmethod
    def concentration(z):
        # Same representation-statistic formula as V2, generalized to d dims.
        # Components are NOT calibrated class probabilities, particularly at32.
        p = F.softmax(z, dim=-1)
        entropy = -(p * p.clamp_min(1e-8).log()).sum(-1)
        return 1 - entropy / math.log(z.size(-1))

    @staticmethod
    def margin(z):
        values = F.softmax(z, dim=-1).topk(2, dim=-1).values
        return values[:, 0] - values[:, 1]

    def forward(self, flow, packet):
        if flow.ndim != 2 or flow.shape != packet.shape or flow.size(-1) != 64:
            raise ValueError('Expected paired [batch,64] encoder representations')
        zf, zp = self.flow_projection(flow), self.packet_projection(packet)
        stats = torch.cat((zf, zp, self.concentration(zf)[:, None], self.concentration(zp)[:, None],
                           self.margin(zf)[:, None], self.margin(zp)[:, None],
                           F.cosine_similarity(zf, zp, dim=-1)[:, None]), dim=-1)
        alpha = self.gate(stats).sigmoid()
        logits = self.shared_head(alpha * zf + (1 - alpha) * zp)
        return F.log_softmax(logits, dim=-1), alpha


class FrozenEncoderClassifier(nn.Module):
    """Self-contained graph model. train() always leaves both encoders in eval."""
    def __init__(self, flow_encoder, packet_encoder, fusion):
        super().__init__()
        self.flow_encoder, self.packet_encoder, self.fusion = flow_encoder, packet_encoder, fusion
        for encoder in (self.flow_encoder, self.packet_encoder):
            encoder.requires_grad_(False)
            encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        self.flow_encoder.eval()
        self.packet_encoder.eval()
        return self

    def forward(self, data):
        if self.flow_encoder.training or self.packet_encoder.training:
            raise ValueError('Frozen encoders must stay in eval, including BatchNorm')
        with torch.no_grad():
            flow = global_mean_pool(self.flow_encoder(data)['flow'], data['flow'].batch)
            packet = global_mean_pool(self.packet_encoder(data)['packet'], data['packet'].batch)
        scores, alpha = self.fusion(flow, packet)
        self.last_flow_weights = alpha.detach()
        return scores
