"""The only model change from the original residual: hidden LayerNorm(64)."""
from torch import nn

from .flow_stats_residual import FlowStatsResidualModel


class FlowStatsLayerNormModel(FlowStatsResidualModel):
    def add_flow_stats_path(self, graph):
        super().add_flow_stats_path(graph)
        first, activation, last = self.flow_stats_mlp
        norm = nn.LayerNorm(64, eps=1e-5, elementwise_affine=True).to(
            device=first.weight.device, dtype=first.weight.dtype)
        # Reuse the same initialized Linear layers; last remains exactly zero.
        self.flow_stats_mlp = nn.Sequential(first, norm, activation, last)
