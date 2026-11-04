"""Packet-only second-layer residual before frozen BN, with a shared last readout."""
import torch
from torch import nn
from .relation_selection import RelationClassifier
from .packet_pooling import ordered_mean_pool


class ResidualClassifier(RelationClassifier):
    def __init__(self, flow_encoder, packet_encoder, fusion, mode='base'):
        super().__init__(flow_encoder, packet_encoder, fusion, 'mean')
        if mode not in ('base','skip'):
            raise ValueError('Unknown residual mode')
        self.pooling=mode
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator().manual_seed(42).get_state())
            self.cross_layer=nn.Linear(128,64)
            self.residual=nn.Linear(64,64,bias=False) if mode=='skip' else None
        with torch.no_grad():
            self.cross_layer.weight.zero_();self.cross_layer.weight[:,64:].copy_(torch.eye(64))
            self.cross_layer.bias.zero_()
            if self.residual is not None:self.residual.weight.zero_()

    def packet_nodes(self,first,flow_message,packet_message,force_mean=False):
        mean=torch.stack((flow_message,packet_message),dim=0).mean(0)
        delta=self.residual(first) if self.residual is not None and not force_mean else torch.zeros_like(mean)
        # This computation must retain gradients through frozen BN and activation.
        nodes=self.packet_encoder.relus2['packet'](self.packet_encoder.bns2['packet'](mean+delta))
        self.last_residual_norm=delta.detach().norm(dim=-1)
        return nodes,mean.new_full((len(mean),2),.5)

    def forward_features(self,flow,first,flow_message,packet_message,index,size):
        nodes,_=self.packet_nodes(first,flow_message,packet_message)
        last=ordered_mean_pool(nodes,index,size)
        packet=self.cross_layer(torch.cat((last,last),dim=-1))
        return self.fusion(flow,packet)
