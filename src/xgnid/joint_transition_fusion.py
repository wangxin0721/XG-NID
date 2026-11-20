"""Independent trainable fusion copies above a shared, frozen feature cache."""
from copy import deepcopy

from torch import nn

from .ordered_transition_readout import OrderedTransitionReadout


MODES = {'joint_base': None, 'joint_single': 'single',
         'joint_pointwise': 'pointwise', 'joint_ordered': 'ordered'}
ALIASES = dict(zip(MODES, ('J0', 'J1', 'J2', 'J3')))


class JointTransitionFusion(nn.Module):
    def __init__(self, source_fusion, mode, seed=42):
        super().__init__()
        if mode not in MODES:
            raise ValueError('Unknown joint transition arm')
        if set(dict(source_fusion.named_children())) != {
                'flow_projection', 'packet_projection', 'gate', 'shared_head'}:
            raise ValueError('Unexpected source fusion modules')
        self.fusion = deepcopy(source_fusion).requires_grad_(True)
        self.readout = None if MODES[mode] is None else OrderedTransitionReadout(MODES[mode], seed)

    def forward(self, features):
        if self.readout is None:
            residual, attention = features.packet.new_zeros(features.packet.shape), None
        else:
            residual, attention = self.readout(features.packet_mean, features.transitions, features.lengths)
        # Always recompute all trainable projections, dimensional gates, and the final head.
        scores, alpha = self.fusion(features.flow, features.packet + residual)
        return scores, residual, alpha, attention
