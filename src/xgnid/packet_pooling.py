"""Packet readout experiments on the unchanged, frozen D32 encoders."""
import torch
from torch import nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.utils import softmax

from .fusion_dimension import FrozenEncoderClassifier


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
        return self.aggregation(nodes, index=index, dim_size=size)

    def weights(self, nodes, index, size):
        return softmax(self.aggregation.gate_nn(nodes), index=index, num_nodes=size).view(-1)


class PacketReadoutClassifier(FrozenEncoderClassifier):
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
        packet = (global_mean_pool(nodes, index, size=size) if self.packet_pool is None
                  else self.packet_pool(nodes, index, size))
        return self.fusion(flow, packet)

    def packet_weights(self, nodes, index, size):
        if self.packet_pool is not None:
            return self.packet_pool.weights(nodes, index, size)
        counts = torch.bincount(index, minlength=size).clamp_min(1)
        return counts[index].to(nodes.dtype).reciprocal()

    def forward(self, data):
        features = self.encode(data)
        scores, alpha = self.forward_features(*features)
        self.last_flow_weights = alpha.detach()
        return scores
