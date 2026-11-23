"""Three matched readouts on independent original-C backbones; fixed residual strength."""
from .joint_transition_fusion import JointTransitionFusion


ARMS = ('frozen_single', 'joint_single', 'joint_ordered')
MODELS = ('base_C', *ARMS)


class StabilityFusion(JointTransitionFusion):
    def __init__(self, source_fusion, arm, readout_seed=42):
        if arm not in ARMS:
            raise ValueError('Unknown stability arm')
        super().__init__(source_fusion, 'joint_single' if arm == 'frozen_single' else arm, readout_seed)
        self.arm = arm
        if arm == 'frozen_single':
            self.fusion.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        if self.arm == 'frozen_single':
            self.fusion.eval()
        return self
