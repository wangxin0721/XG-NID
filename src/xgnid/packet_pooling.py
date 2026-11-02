"""Packet readout experiments on the unchanged, frozen D32 encoders."""
import torch
from torch import nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.aggr import AttentionalAggregation

from .fusion_dimension import FrozenEncoderClassifier


def ordered_lengths(index, size):
    """PyG batches and the ragged cache group each graph's nodes contiguously."""
    if index.ndim != 1 or size < 0:
        raise ValueError('Expected a one-dimensional graph index')
    invalid = (index < 0).any() | (index >= size).any() | (index[1:] < index[:-1]).any()
    if bool(invalid):
        raise ValueError('Packet nodes must be grouped in graph order')
    return torch.bincount(index, minlength=size)


def ordered_mean_pool(nodes, index, size, lengths=None):
    lengths = ordered_lengths(index, size) if lengths is None else lengths
    total = torch.segment_reduce(nodes, 'sum', lengths=lengths, axis=0)
    return total / lengths.clamp_min(1).to(nodes.dtype)[:, None]


def ordered_softmax(scores, index, lengths):
    # Keep a two-dimensional score tensor: the pinned PyTorch 2.3.1 CUDA
    # kernel then walks each segment in node order, with no scatter_add.
    maximum = torch.segment_reduce(scores.detach(), 'max', lengths=lengths, axis=0)
    numerator = (scores - maximum[index]).exp()
    denominator = torch.segment_reduce(numerator, 'sum', lengths=lengths, axis=0)
    return numerator / (denominator[index] + 1e-16)


def uniform_packet_weights(nodes, index, size, lengths=None):
    lengths = ordered_lengths(index, size) if lengths is None else lengths
    # Use exactly the same operations as the zero-initialized scorer, rather
    # than a separately rounded reciprocal of the packet count.
    return ordered_softmax(nodes.new_zeros((len(nodes), 1)), index, lengths).view(-1)


class PacketAttentionPool(nn.Module):
    def __init__(self, seed=42):
        super().__init__()
        # Construction must not change either arm's training RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator(device='cpu').manual_seed(seed).get_state())
            score = nn.Sequential(nn.Linear(64, 16), nn.LeakyReLU(),
                                  nn.Linear(16, 1, bias=False))
        nn.init.zeros_(score[-1].weight)
        self.aggregation = AttentionalAggregation(gate_nn=score)

    def forward(self, nodes, index, size):
        # Equivalent attention in real arithmetic, with an exact zero correction
        # at uniform initialization. sum(x)/n and sum(x/n) differ in float32.
        lengths = ordered_lengths(index, size)
        mean = ordered_mean_pool(nodes, index, size, lengths)
        beta = ordered_softmax(self.aggregation.gate_nn(nodes), index, lengths).view(-1)
        uniform = uniform_packet_weights(nodes, index, size, lengths)
        correction = (beta - uniform)[:, None] * (nodes - mean[index])
        return mean + torch.segment_reduce(correction, 'sum', lengths=lengths, axis=0)

    def weights(self, nodes, index, size):
        lengths = ordered_lengths(index, size)
        return ordered_softmax(self.aggregation.gate_nn(nodes), index, lengths).view(-1)


class PacketReadoutClassifier(FrozenEncoderClassifier):
    packet_mean = staticmethod(ordered_mean_pool)
    uniform_packet_weights = staticmethod(uniform_packet_weights)

    def __init__(self, flow_encoder, packet_encoder, fusion, pooling='mean'):
        super().__init__(flow_encoder, packet_encoder, fusion)
        if pooling not in ('mean', 'attention'):
            raise ValueError('Unknown packet readout')
        self.pooling = pooling
        self.packet_pool = PacketAttentionPool() if pooling == 'attention' else None

    def encode(self, data):
        if any(m.training for e in (self.flow_encoder, self.packet_encoder) for m in e.modules()):
            raise ValueError('Frozen encoders and BatchNorm must stay in eval')
        # Only feature extraction is frozen. The trainable readout is outside.
        with torch.no_grad():
            flow = global_mean_pool(self.flow_encoder(data)['flow'], data['flow'].batch,
                                    size=data.num_graphs)
            nodes = self.packet_encoder(data)['packet']
        return flow, nodes, data['packet'].batch, data.num_graphs

    def forward_features(self, flow, nodes, index, size):
        packet = (ordered_mean_pool(nodes, index, size) if self.packet_pool is None
                  else self.packet_pool(nodes, index, size))
        return self.fusion(flow, packet)

    def packet_weights(self, nodes, index, size):
        if self.packet_pool is not None:
            return self.packet_pool.weights(nodes, index, size)
        return uniform_packet_weights(nodes, index, size)

    def forward(self, data):
        features = self.encode(data)
        scores, alpha = self.forward_features(*features)
        self.last_flow_weights = alpha.detach()
        return scores
