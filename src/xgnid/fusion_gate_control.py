"""Fixed equal fusion with the exact original D32 checkpoint layout."""
import torch
from torch.nn import functional as F

from .fusion_dimension import FusionHead


class EqualFusionHead(FusionHead):
    def __init__(self, width, streams):
        if width != 32:
            raise ValueError('This control is registered only for 32D fusion')
        super().__init__(width, streams)
        # Retain the unused gate state for strict loading of the clean D32
        # snapshot. Its 1,664 parameters are frozen and never evaluated.
        self.gate.requires_grad_(False)

    def forward(self, flow, packet):
        if flow.ndim != 2 or flow.shape != packet.shape or flow.size(-1) != 64:
            raise ValueError('Expected paired [batch,64] encoder representations')
        zf, zp = self.flow_projection(flow), self.packet_projection(packet)
        alpha = torch.full_like(zf, .5)
        logits = self.shared_head(alpha * zf + (1 - alpha) * zp)
        return F.log_softmax(logits, dim=-1), alpha
