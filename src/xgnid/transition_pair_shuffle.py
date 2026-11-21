"""Label-free, sample-stable derangements of second-transition candidates."""
import hashlib
import json
import random

from .joint_transition_fusion import JointTransitionFusion


VERSION = 'sha256-local-random-rejection-derangement-v1'
ARMS = ('pair_true', 'pair_shuffle')


def derangement(sample_id, partition, count, seed):
    if partition not in ('train', 'val') or not isinstance(sample_id, str) or count < 0:
        raise ValueError('Expected train/val sample identity and nonnegative context count')
    if count < 2:
        return list(range(count))
    payload = json.dumps([VERSION, seed, partition, sample_id, count], separators=(',', ':')).encode()
    generator = random.Random(int.from_bytes(hashlib.sha256(payload).digest(), 'big'))
    # Rejection sampling is uniform over derangements; no process-global RNG is touched.
    while True:
        result = list(range(count))
        generator.shuffle(result)
        if all(i != value for i, value in enumerate(result)):
            return result


class PairFusion(JointTransitionFusion):
    def __init__(self, source_fusion, arm, seed=42):
        if arm not in ARMS:
            raise ValueError('Unknown True/Shuffle arm')
        super().__init__(source_fusion, 'joint_ordered', seed)
        self.shuffle = arm == 'pair_shuffle'

    def context_right(self, features):
        return features.shuffled_right if self.shuffle else None

    def forward(self, features):
        residual, attention = self.readout(features.packet_mean, features.transitions, features.lengths,
                                          context_right=self.context_right(features))
        scores, alpha = self.fusion(features.flow, features.packet + residual)
        return scores, residual, alpha, attention
