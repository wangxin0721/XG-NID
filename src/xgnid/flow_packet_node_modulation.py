"""Flow-conditioned Packet nodes between graph layers, with matched controls."""
from copy import deepcopy

import torch
from torch import nn
from torch_geometric.nn import global_mean_pool

from .model import FLOW_TO_PACKET, PACKET_TO_PACKET
from .ordered_transition_readout import attach_continuity
from .packet_pooling import ordered_mean_pool

ARMS = ('S0', 'S1', 'S2', 'S3')


class NodeAdjustment(nn.Module):
    """Both active controls have 3,008 effective trainable parameters."""
    def __init__(self, arm, seed=42):
        super().__init__()
        if arm not in ('S2', 'S3'):
            raise ValueError('Expected S2 or S3')
        self.arm = arm
        hidden, output = (23, 64) if arm == 'S2' else (15, 128)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.net = nn.Sequential(nn.Linear(64, hidden, bias=False), nn.LeakyReLU(),
                                     nn.Linear(hidden, output))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, nodes, context, index):
        if self.arm == 'S2':
            return nodes + self.net(nodes)
        gamma, beta = self.net(context.detach()).chunk(2, dim=-1)
        return (1 + gamma[index]) * nodes + beta[index]


class FlowPacketNodeClassifier(nn.Module):
    """Keep the unmodified first-depth skip; modulate messages entering layer two.

    topology contains chronological src/dst, six metadata fields and per-flow
    valid-link counts. Inference accepts graph structure, never supervision.
    """
    relations = (FLOW_TO_PACKET, PACKET_TO_PACKET)

    def __init__(self, source, ordered, arm, seed=42):
        super().__init__()
        if arm not in ARMS:
            raise ValueError('Unknown experiment arm')
        self.arm = arm
        for name in ('flow_encoder', 'packet_encoder', 'cross_layer', 'packet_pool'):
            setattr(self, name, deepcopy(getattr(source, name)))
        self.fusion = deepcopy(ordered.fusion)
        self.readout = deepcopy(ordered.readout)
        self.requires_grad_(False)
        if arm != 'S0':
            for module in (self.cross_layer, self.packet_pool, self.fusion, self.readout,
                           *self.packet_relations()):
                module.requires_grad_(True)
        self.adjustment = NodeAdjustment(arm, seed) if arm in ('S2', 'S3') else None
        self.zero_grad(set_to_none=True)
        self.train(False)

    def packet_relations(self):
        return tuple(self.packet_encoder.convs2.convs[r] for r in self.relations)

    def train(self, mode=True):
        super().train(mode)
        self.flow_encoder.eval()
        self.packet_encoder.eval()  # Trainable GAT weights, frozen BN buffers/dropout.
        return self

    def forward(self, batch, topology):
        if any(m.training for e in (self.flow_encoder, self.packet_encoder) for m in e.modules()):
            raise ValueError('Encoders and BN must remain eval')
        src, dst, metadata, lengths = topology
        index, size = batch['packet'].batch, batch.num_graphs
        flow_first = []
        hook = self.flow_encoder.relus1['flow'].register_forward_hook(
            lambda module, args, out: flow_first.append(out))
        try:
            with torch.no_grad():
                flow = global_mean_pool(self.flow_encoder(batch)['flow'], batch['flow'].batch, size=size)
                if len(flow_first) != 1:
                    raise ValueError('Expected one Flow layer-one output')
                context = global_mean_pool(flow_first[0], batch['flow'].batch, size=size)
        finally:
            hook.remove()
        first = []

        def adjust(module, args, out):
            first.append(out)
            return out if self.adjustment is None else self.adjustment(out, context, index)

        hook = self.packet_encoder.relus1['packet'].register_forward_hook(adjust)
        try:
            nodes = self.packet_encoder(batch)['packet']
        finally:
            hook.remove()
        if len(first) != 1 or first[0].shape != nodes.shape:
            raise ValueError('Packet layer layouts differ')
        first_mean = ordered_mean_pool(first[0], index, size)
        mean = ordered_mean_pool(nodes, index, size)
        query, _ = self.packet_pool(flow, nodes, index, size)
        packet = self.cross_layer(torch.cat((first_mean, query), dim=-1))
        transitions = torch.cat((nodes[dst] - nodes[src], metadata), dim=-1)
        transitions = attach_continuity(transitions, src, dst, lengths)
        residual, _ = self.readout(mean, transitions, lengths)
        return self.fusion(flow, packet + residual)[0]
