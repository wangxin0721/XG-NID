"""Second-layer Packet relation selection; original convolutions stay intact."""
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import global_mean_pool

from .fusion_dimension import FrozenEncoderClassifier
from .model import FLOW_TO_PACKET, PACKET_TO_PACKET
from .packet_pooling import ordered_mean_pool


class RelationClassifier(FrozenEncoderClassifier):
    def __init__(self, flow_encoder, packet_encoder, fusion, mode='mean'):
        super().__init__(flow_encoder, packet_encoder, fusion)
        if mode not in ('mean', 'relation_gate'):
            raise ValueError('Unknown relation mode')
        self.pooling = mode
        self.relation_gate = None
        if mode == 'relation_gate':
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(torch.Generator().manual_seed(42).get_state())
                self.relation_gate = nn.Sequential(nn.Linear(192, 16), nn.LeakyReLU(), nn.Linear(16, 2))
            nn.init.zeros_(self.relation_gate[-1].weight)
            nn.init.zeros_(self.relation_gate[-1].bias)

    def encode(self, data):
        if any(m.training for e in (self.flow_encoder, self.packet_encoder) for m in e.modules()):
            raise ValueError('Frozen encoders/BN must remain eval')
        captured, handles = {}, []
        def before(module, args):
            captured['first'] = args[0]['packet'].detach()
        def capture(name):
            def hook(module, args, out):
                if name in captured:
                    raise ValueError('Relation convolution called twice')
                captured[name] = out.detach()
            return hook
        try:
            handles.append(self.packet_encoder.convs2.register_forward_pre_hook(before))
            for name, relation in (('flow_message', FLOW_TO_PACKET), ('packet_message', PACKET_TO_PACKET)):
                handles.append(self.packet_encoder.convs2.convs[relation].register_forward_hook(capture(name)))
            with torch.no_grad():
                flow = global_mean_pool(self.flow_encoder(data)['flow'], data['flow'].batch, size=data.num_graphs)
                legacy = self.packet_encoder(data)['packet']
            if set(captured) != {'first', 'flow_message', 'packet_message'}:
                raise ValueError('Missing original relation invocation; do not invent a replacement/mask')
            if any(x.shape != legacy.shape or x.size(-1) != 64 for x in captured.values()):
                raise ValueError('Expected unchanged 64D relation outputs')
            features = (flow, captured['first'], captured['flow_message'], captured['packet_message'],
                        data['packet'].batch, data.num_graphs)
            return features, legacy
        finally:
            for handle in handles:
                handle.remove()

    def packet_nodes(self, first, flow_message, packet_message, force_mean=False):
        # Match HeteroConv.group(..., 'mean'), including bias-only outputs of
        # zero-edge relations. No incoming-edge mask is applied.
        mean = torch.stack((flow_message, packet_message), dim=0).mean(0)
        if self.relation_gate is None or force_mean:
            gamma = mean.new_full((len(mean), 2), .5)
            mixed = mean
        else:
            gamma = self.relation_gate(torch.cat((first, flow_message, packet_message), dim=-1)).softmax(-1)
            mixed = mean + ((gamma[:, :1] - .5) * (flow_message - mean)
                            + (gamma[:, 1:] - .5) * (packet_message - mean))
        # Frozen BN affine/statistics still permit derivatives w.r.t. mixed.
        nodes = self.packet_encoder.relus2['packet'](self.packet_encoder.bns2['packet'](mixed))
        return nodes, gamma

    def forward_features(self, flow, first, flow_message, packet_message, index, size):
        nodes, gamma = self.packet_nodes(first, flow_message, packet_message)
        self.last_relation_weights = gamma.detach()
        return self.fusion(flow, ordered_mean_pool(nodes, index, size))

    def forward(self, data):
        features, _ = self.encode(data)
        scores, alpha = self.forward_features(*features)
        self.last_flow_weights = alpha.detach()
        return scores
