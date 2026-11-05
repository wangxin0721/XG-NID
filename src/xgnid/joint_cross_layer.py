"""Cross-layer readout with two trainable Packet layer-two GAT relations."""
import torch
from torch_geometric.nn import global_mean_pool

from .cross_layer import CrossLayerClassifier
from .model import FLOW_TO_PACKET, PACKET_TO_PACKET
from .packet_pooling import ordered_mean_pool


class JointCrossLayerClassifier(CrossLayerClassifier):
    relations = (FLOW_TO_PACKET, PACKET_TO_PACKET)

    def __init__(self, flow_encoder, packet_encoder, fusion, mode):
        super().__init__(flow_encoder, packet_encoder, fusion, mode)
        for module in self.packet_relations():
            module.requires_grad_(True)

    def packet_relations(self):
        return tuple(self.packet_encoder.convs2.convs[relation] for relation in self.relations)

    def encode(self, data):
        if any(m.training for encoder in (self.flow_encoder, self.packet_encoder) for m in encoder.modules()):
            raise ValueError('Encoders and BN must remain in eval mode')
        first_layer = []
        handle = self.packet_encoder.relus1['packet'].register_forward_hook(
            lambda module, args, output: first_layer.append(output.detach()))
        try:
            with torch.no_grad():
                flow = global_mean_pool(self.flow_encoder(data)['flow'], data['flow'].batch, size=data.num_graphs)
            last = self.packet_encoder(data)['packet']
        finally:
            handle.remove()
        if len(first_layer) != 1 or first_layer[0].shape != last.shape:
            raise ValueError('Expected both Packet layers with the same node layout')
        index = data['packet'].batch
        return (flow, ordered_mean_pool(first_layer[0], index, data.num_graphs),
                ordered_mean_pool(last, index, data.num_graphs)), None


def reparameterize_projection(model):
    with torch.no_grad():
        left, right = model.cross_layer.weight.split(64, dim=1)
        combined = (left.double() + right.double()).to(right.dtype)
        left.zero_()
        right.copy_(combined)
