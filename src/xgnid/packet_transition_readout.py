"""Pool chronological Packet transitions within each flow."""
import torch
from torch import nn
from torch.nn import functional as F

from .packet_pooling import ordered_softmax


class PacketTransitionReadout(nn.Module):
    def __init__(self, seed=42):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
            self.edge = nn.Linear(70, 16)
            self.query = nn.Linear(64, 16)
            self.output = nn.Linear(16, 64, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, query, transitions, lengths):
        """Return grouped attention and residual. A missing query gives equal weights."""
        if not len(transitions):
            pooled = transitions.new_zeros((len(lengths), self.output.in_features))
            return self.output(pooled), transitions.new_empty((0,))
        index = torch.repeat_interleave(torch.arange(len(lengths), device=lengths.device),
                                        lengths, output_size=len(transitions))
        values = F.gelu(self.edge(transitions))
        if query is None:
            scores = values.new_zeros((len(values), 1))
        else:
            scores = (values * self.query(query)[index]).sum(dim=-1, keepdim=True) / 4
        attention = ordered_softmax(scores, index, lengths)
        pooled = torch.segment_reduce(values * attention, 'sum', lengths=lengths, axis=0)
        return self.output(pooled), attention.flatten()
