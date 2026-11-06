"""Matched mean-query and Flow-query attention over frozen Packet nodes."""
import math

import torch
from torch import nn

from .fusion_dimension import FrozenEncoderClassifier
from .packet_pooling import (PacketReadoutClassifier, ordered_lengths,
                             ordered_mean_pool, ordered_softmax, uniform_packet_weights)


class QueryPacketPool(nn.Module):
    def __init__(self, query_source, seed=42):
        super().__init__()
        if query_source not in ('packet_mean', 'flow'):
            raise ValueError('Unknown query source')
        self.query_source = query_source
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator(device='cpu').manual_seed(seed).get_state())
            self.query = nn.Linear(64, 8, bias=False)
            self.key = nn.Linear(64, 8, bias=False)
        nn.init.zeros_(self.query.weight)

    def forward(self, flow, nodes, index, size):
        lengths = ordered_lengths(index, size)
        mean = ordered_mean_pool(nodes, index, size, lengths)
        query_input = mean if self.query_source == 'packet_mean' else flow
        q = self.query(query_input)
        k = self.key(nodes)
        scores = (q[index] * k).sum(dim=-1, keepdim=True) / math.sqrt(self.query.out_features)
        beta = ordered_softmax(scores, index, lengths).view(-1)
        uniform = uniform_packet_weights(nodes, index, size, lengths)
        # The centered correction gives exact ordered-mean output at uniform attention.
        correction = (beta - uniform)[:, None] * (nodes - mean[index])
        pooled = mean + torch.segment_reduce(correction, 'sum', lengths=lengths, axis=0)
        return pooled, beta


class QueryPacketClassifier(FrozenEncoderClassifier):
    encode = PacketReadoutClassifier.encode

    def __init__(self, flow_encoder, packet_encoder, fusion, query_source, seed=42):
        super().__init__(flow_encoder, packet_encoder, fusion)
        self.packet_pool = QueryPacketPool(query_source, seed)

    def forward_with_attention(self, flow, nodes, index, size):
        packet, beta = self.packet_pool(flow, nodes, index, size)
        scores, alpha = self.fusion(flow, packet)
        return scores, alpha, beta

    def forward_features(self, flow, nodes, index, size):
        scores, alpha, _ = self.forward_with_attention(flow, nodes, index, size)
        return scores, alpha

    def forward(self, data):
        scores, alpha = self.forward_features(*self.encode(data))
        self.last_flow_weights = alpha.detach()
        return scores
