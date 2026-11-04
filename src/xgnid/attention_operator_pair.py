"""Fresh, paired GAT/GATv2 encoders; historical model definitions stay intact."""
import hashlib

import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, GATv2Conv, global_mean_pool

from .model import (_PaperStyleHeteroEncoder, FLOW_TO_PACKET,
                    PACKET_TO_FLOW, PACKET_TO_PACKET)
from .fusion_dimension import FusionHead

RELATIONS = (FLOW_TO_PACKET, PACKET_TO_FLOW, PACKET_TO_PACKET)


class MatchedGATv2Conv(GATv2Conv):
    """GAT-compatible bias placement: no inner bias, one output bias."""
    def __init__(self, *args, **kwargs):
        kwargs['bias'] = False
        super().__init__(*args, **kwargs)
        size = self.heads * self.out_channels if self.concat else self.out_channels
        self.bias = nn.Parameter(torch.zeros(size))


def stream(seed, name):
    return int.from_bytes(hashlib.sha256((str(seed) + ':' + name).encode()).digest()[:8], 'big') % (2**63-1)


def warm_graph(flow_dim, packet_dim):
    # Every relation is nonempty, including lazy edge projections. BN stays eval.
    g = HeteroData()
    g['flow'].x = torch.zeros(2, flow_dim)
    g['packet'].x = torch.zeros(3, packet_dim)
    g[FLOW_TO_PACKET].edge_index = torch.tensor([[0,1,0,1,0,1], [0,0,1,1,2,2]])
    g[PACKET_TO_FLOW].edge_index = g[FLOW_TO_PACKET].edge_index.flip(0)
    g[PACKET_TO_PACKET].edge_index = torch.tensor([[0,1,2,0,1,2], [1,2,0,2,0,1]])
    for relation, dim in zip(RELATIONS, (4,4,1)):
        g[relation].edge_attr = torch.zeros(g[relation].edge_index.shape[1], dim)
    return g


def canonical_name(name):
    # PyG uses lin_src/lin_dst in GAT and lin_l/lin_r in GATv2.
    return name.replace('.lin_l.', '.lin_src.').replace('.lin_r.', '.lin_dst.')


class OperatorClassifier(nn.Module):
    def __init__(self, operator, flow_dim=82, packet_dim=1500, seed=42):
        super().__init__()
        if operator not in ('GAT', 'GATv2'):
            raise ValueError('Only GAT and GATv2 are registered')
        self.operator = operator
        conv = GATConv if operator == 'GAT' else MatchedGATv2Conv
        # Constructor/lazy initialization must not advance the caller RNG.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator().manual_seed(stream(seed, 'construction')).get_state())
            self.flow_encoder = _PaperStyleHeteroEncoder(conv_cls=conv, edge_aware=True,
                flow_input_dim=flow_dim, packet_input_dim=packet_dim)
            self.packet_encoder = _PaperStyleHeteroEncoder(conv_cls=conv, edge_aware=True,
                flow_input_dim=flow_dim, packet_input_dim=packet_dim)
            self.eval()
            with torch.no_grad():
                warm = warm_graph(flow_dim, packet_dim)
                self.flow_encoder(warm)
                self.packet_encoder(warm)
            names = ('flow_projection','packet_projection','gate_hidden','gate_output','head_hidden','head_output')
            self.fusion = FusionHead(32, {k: stream(seed, 'fusion.'+k) for k in names})
            with torch.no_grad():
                for name, p in self.named_parameters():
                    if name.startswith('fusion.'):
                        continue
                    if '.bns' in name:
                        p.fill_(1 if name.endswith('.weight') else 0)
                    elif name.endswith('.bias'):
                        p.zero_()
                    else:
                        generator = torch.Generator().manual_seed(stream(seed, canonical_name(name)))
                        nn.init.xavier_uniform_(p, generator=generator)
        self.eval()

    def forward(self, data):
        flow = global_mean_pool(self.flow_encoder(data)['flow'], data['flow'].batch)
        packet = global_mean_pool(self.packet_encoder(data)['packet'], data['packet'].batch)
        scores, weights = self.fusion(flow, packet)
        self.last_flow_weights = weights.detach()
        return scores


def common_state(model):
    return {canonical_name(k): v for k, v in model.state_dict().items()
            if not any(part.startswith('att') for part in k.split('.'))}


def assert_paired(models):
    a, b = [common_state(models[k]) for k in ('GAT', 'GATv2')]
    if set(a) != set(b) or any(not torch.equal(a[k], b[k]) for k in a):
        raise ValueError('Corresponding fresh parameters/buffers differ')
    return {'common_tensors': len(a), 'all_common_tensors_identical': True,
            'operator_attention_parameters': 'independent_named_streams; shapes differ',
            'initial_predictions_required_identical': False}
