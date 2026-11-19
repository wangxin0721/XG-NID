"""Read adjacent transitions and ordered pairs of consecutive transitions."""
import torch
from torch import nn
from torch.nn import functional as F

from .packet_pooling import ordered_softmax
from .packet_transition_readout import PacketTransitionReadout


MODES = ('single', 'pointwise', 'ordered')


def attach_continuity(transitions, source, destination, lengths):
    """Append a flag only when two valid links share their middle Packet node."""
    if len(transitions) != len(source) or len(source) != len(destination) or int(lengths.sum()) != len(source):
        raise ValueError('Transition endpoint counts differ')
    source = torch.as_tensor(source, device=transitions.device)
    destination = torch.as_tensor(destination, device=transitions.device)
    owner = torch.repeat_interleave(torch.arange(len(lengths), device=lengths.device), lengths)
    connected = torch.zeros(len(source), dtype=torch.bool, device=transitions.device)
    connected[:-1] = (destination[:-1] == source[1:]) & (owner[:-1] == owner[1:])
    return torch.cat((transitions, connected[:, None].to(transitions.dtype)), dim=-1)


class OrderedTransitionReadout(PacketTransitionReadout):
    """Three arms share their original edge, query, and zero-output initialization."""
    def __init__(self, mode, seed=42):
        if mode not in MODES:
            raise ValueError('Unknown transition readout mode')
        super().__init__(seed)
        self.mode = mode
        if mode != 'single':
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(torch.Generator().manual_seed(seed + 1000).get_state())
                self.context = nn.Linear(32, 16)

    def forward(self, query, transitions, lengths):
        if transitions.ndim != 2 or transitions.size(1) != 71:
            raise ValueError('Expected 70 transition fields and a continuity flag')
        if self.mode == 'single' or not len(transitions):
            return super().forward(query, transitions[:, :70], lengths)
        owner = torch.repeat_interleave(torch.arange(len(lengths), device=lengths.device), lengths)
        flags = transitions[:, 70]
        if len(owner) != len(transitions) or not bool(((flags == 0) | (flags == 1)).all()):
            raise ValueError('Invalid grouped transition lengths or flags')
        left = torch.where(flags.bool())[0]
        if len(left) and (int(left[-1]) >= len(owner) - 1 or bool((owner[left] != owner[left + 1]).any())):
            raise ValueError('A context cannot cross graph boundaries')
        values = F.gelu(self.edge(transitions[:, :70]))
        queries = self.query(query)
        scores = (values * queries[owner]).sum(dim=-1, keepdim=True) / 4
        attention = ordered_softmax(scores, owner, lengths)
        pooled = torch.segment_reduce(values * attention, 'sum', lengths=lengths, axis=0)
        if len(left):
            right = left + 1 if self.mode == 'ordered' else left
            context = F.gelu(self.context(torch.cat((values[left], values[right]), dim=-1)))
            groups = owner[left]
            counts = torch.bincount(groups, minlength=len(lengths))
            context_scores = (context * queries[groups]).sum(dim=-1, keepdim=True) / 4
            context_weights = ordered_softmax(context_scores, groups, counts)
            context_pool = torch.segment_reduce(context * context_weights, 'sum', lengths=counts, axis=0)
            # Each available scale receives equal mass. Empty contexts retain the single-scale readout.
            pooled = (pooled + context_pool) / (1 + (counts > 0).to(pooled.dtype))[:, None]
        return self.output(pooled), attention.flatten()
