"""An opt-in Flow-statistics residual for the seed42 structure screen.

Historical model definitions and state keys are unchanged. Load the original
N0 strictly, attach its V2 gate, then call add_flow_stats_path().
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import global_mean_pool

from .model import DualBranchLogitGatedHeteroGNN_Edge, _align_feature_dim


class FlowStatsResidualModel(DualBranchLogitGatedHeteroGNN_Edge):
    """Keep the 21->16->8 gate; add statistics before Flow's 64->8 layer."""

    def add_flow_stats_path(self, graph):
        if hasattr(self, "flow_stats_mlp"):
            raise ValueError("Flow statistics path is already attached")
        raw_dim = int(graph["flow"].x.size(-1))
        self.flow_stats_input_dim = self.flow_encoder.flow_input_dim or raw_dim
        if self.hidden_dim != 64 or self.num_classes != 8:
            raise ValueError("This screen requires hidden_dim=64 and eight outputs")
        self.flow_stats_mlp = nn.Sequential(
            nn.Linear(self.flow_stats_input_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
        ).to(device=self.flow_classifier.weight.device,
             dtype=self.flow_classifier.weight.dtype)
        nn.init.zeros_(self.flow_stats_mlp[-1].weight)
        nn.init.zeros_(self.flow_stats_mlp[-1].bias)
        self.flow_stats_enabled = True

    def _branch_outputs(self, data):
        flow_graph = self._branch_embedding(self.flow_encoder, data, "flow")
        packet_emb = self._branch_embedding(self.packet_encoder, data, "packet")
        residual = torch.zeros_like(flow_graph)
        if hasattr(self, "flow_stats_mlp") and self.flow_stats_enabled:
            # Apply the SAME explicit pad/truncate policy as the graph encoder.
            # When the encoder has no fixed input_dim, require the observed width.
            features = _align_feature_dim(data["flow"].x.float(), self.flow_encoder.flow_input_dim)
            if features.size(-1) != self.flow_stats_input_dim:
                raise ValueError("Flow feature width changed after initialization")
            residual = global_mean_pool(self.flow_stats_mlp(features), data["flow"].batch)
        flow_emb = flow_graph + residual
        # Detach diagnostics only; the path used for prediction retains autograd.
        graph_norm = flow_graph.detach().norm(dim=-1)
        residual_norm = residual.detach().norm(dim=-1)
        self._flow_stats_diagnostics = {
            "flow_graph_norm": graph_norm,
            "flow_residual_norm": residual_norm,
            "residual_to_graph_norm": residual_norm / graph_norm.clamp_min(1e-12),
        }
        return flow_emb, packet_emb, self.flow_classifier(flow_emb), self.packet_classifier(packet_emb)

    def _graph_logits(self, data):
        # During lazy initialization / strict N0 loading, retain original E2.
        if not hasattr(self, "joint_gate"):
            return super()._graph_logits(data)
        _, _, zf, zp = self._branch_outputs(data)
        features = torch.cat((
            zf, zp,
            self._confidence_from_logits(zf)[:, None],
            self._confidence_from_logits(zp)[:, None],
            self._margin_from_logits(zf)[:, None],
            self._margin_from_logits(zp)[:, None],
            F.cosine_similarity(zf, zp, dim=-1)[:, None],
        ), dim=-1)
        alpha = self.joint_gate(features).sigmoid()
        self._flow_stats_diagnostics["flow_weights"] = alpha.detach()
        # train_grouped.evaluate assumes a two-column scalar gate. Keep that
        # legacy cache empty; the new entry exports all eight weights explicitly.
        self._last_cache = {}
        return self.shared_head(alpha * zf + (1 - alpha) * zp)
