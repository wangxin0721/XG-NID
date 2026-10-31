"""Raw-graph 32D fine-tuning: encoder gradients enabled, encoder modes fixed."""
from torch import nn
from torch_geometric.nn import global_mean_pool


class JointEncoderClassifier(nn.Module):
    def __init__(self, flow_encoder, packet_encoder, fusion):
        super().__init__()
        # Keep exactly the source checkpoint's names and shapes.
        self.flow_encoder, self.packet_encoder, self.fusion = flow_encoder, packet_encoder, fusion
        self.flow_encoder.requires_grad_(True)
        self.packet_encoder.requires_grad_(True)
        self.flow_encoder.eval()
        self.packet_encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        self.flow_encoder.eval()
        self.packet_encoder.eval()
        return self

    def forward(self, data):
        if any(m.training for e in (self.flow_encoder, self.packet_encoder) for m in e.modules()):
            raise ValueError('Encoder modules, including BN, must remain in eval mode')
        # No cached representations or no_grad here: eval does not disable autograd.
        flow = global_mean_pool(self.flow_encoder(data)['flow'], data['flow'].batch)
        packet = global_mean_pool(self.packet_encoder(data)['packet'], data['packet'].batch)
        scores, alpha = self.fusion(flow, packet)
        self.last_flow_weights = alpha.detach()
        return scores
