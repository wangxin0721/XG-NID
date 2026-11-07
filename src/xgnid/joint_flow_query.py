"""Flow/mean query pooling with only the two incoming Packet layer-two GATs trainable."""
import torch
from torch_geometric.nn import global_mean_pool

from .flow_query_pooling import QueryPacketClassifier
from .model import FLOW_TO_PACKET, PACKET_TO_PACKET


class JointQueryPacketClassifier(QueryPacketClassifier):
    relations = (FLOW_TO_PACKET, PACKET_TO_PACKET)

    def __init__(self, flow_encoder, packet_encoder, fusion, query_source, seed=42):
        super().__init__(flow_encoder, packet_encoder, fusion, query_source, seed)
        for module in self.packet_relations():
            module.requires_grad_(True)

    def packet_relations(self):
        return tuple(self.packet_encoder.convs2.convs[relation] for relation in self.relations)

    def encode(self, data):
        if any(module.training for encoder in (self.flow_encoder, self.packet_encoder) for module in encoder.modules()):
            raise ValueError('Encoders and BN must stay eval during joint query training')
        with torch.no_grad():
            flow = global_mean_pool(self.flow_encoder(data)['flow'], data['flow'].batch, size=data.num_graphs)
        # Frozen first-layer parameters and fixed input tensors do not build a
        # gradient graph. Layer-two messages, BN(eval), activation and pooling
        # remain in autograd, allowing the Packet representation to adapt.
        nodes = self.packet_encoder(data)['packet']
        return flow, nodes, data['packet'].batch, data.num_graphs
