"""Keep the mature cross-layer projection and replace its second-layer readout."""
from .flow_query_pooling import QueryPacketPool
from .joint_cross_layer import JointCrossLayerClassifier


class JointCrossLayerQueryClassifier(JointCrossLayerClassifier):
    def __init__(self, flow_encoder, packet_encoder, fusion, query_source, seed=42):
        super().__init__(flow_encoder, packet_encoder, fusion, 'multi')
        self.packet_pool = QueryPacketPool(query_source, seed)

    def encode(self, data):
        captured = []
        handle = self.packet_encoder.relus2['packet'].register_forward_hook(
            lambda module, args, output: captured.append(output))
        try:
            (flow, first, _), _ = super().encode(data)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise ValueError('Expected one post-BN/activation second-layer Packet output')
        second, _ = self.packet_pool(flow, captured[0], data['packet'].batch, data.num_graphs)
        return (flow, first, second), None
