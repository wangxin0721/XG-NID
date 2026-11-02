"""Frozen Packet depths with a trainable matched 128-to-64 readout."""
import torch
from torch import nn
from torch_geometric.nn import global_mean_pool
from .fusion_dimension import FrozenEncoderClassifier
from .packet_pooling import ordered_mean_pool


class CrossLayerClassifier(FrozenEncoderClassifier):
    def __init__(self, flow_encoder, packet_encoder, fusion, mode='last'):
        super().__init__(flow_encoder, packet_encoder, fusion)
        if mode not in ('last', 'multi'):
            raise ValueError('Unknown cross-layer mode')
        self.pooling = mode
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator().manual_seed(42).get_state())
            self.cross_layer = nn.Linear(128, 64)
        with torch.no_grad():
            self.cross_layer.weight.zero_()
            self.cross_layer.weight[:, 64:].copy_(torch.eye(64))
            self.cross_layer.bias.zero_()

    def encode(self, data):
        if any(m.training for e in (self.flow_encoder, self.packet_encoder) for m in e.modules()):
            raise ValueError('Frozen encoders/BN must remain eval')
        captured = []
        def capture(module, args, out):
            captured.append(out.detach())
        handle = self.packet_encoder.relus1['packet'].register_forward_hook(capture)
        try:
            with torch.no_grad():
                flow = global_mean_pool(self.flow_encoder(data)['flow'], data['flow'].batch, size=data.num_graphs)
                last = self.packet_encoder(data)['packet']
                if len(captured) != 1 or captured[0].shape != last.shape or last.size(-1) != 64:
                    raise ValueError('Expected one post-BN/activation first-layer Packet output')
                first = ordered_mean_pool(captured[0], data['packet'].batch, data.num_graphs)
                second = ordered_mean_pool(last, data['packet'].batch, data.num_graphs)
                legacy = global_mean_pool(last, data['packet'].batch, size=data.num_graphs)
            return (flow, first, second), legacy
        finally:
            handle.remove()

    def forward_features(self, flow, first, second):
        left = first if self.pooling == 'multi' else second
        packet = self.cross_layer(torch.cat((left, second), dim=-1))
        return self.fusion(flow, packet)

    def forward(self, data):
        features, _ = self.encode(data)
        scores, alpha = self.forward_features(*features)
        self.last_flow_weights = alpha.detach()
        return scores
