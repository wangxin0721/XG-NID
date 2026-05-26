from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, global_mean_pool


def _pad_edge_attr(edge_attr: torch.Tensor | None, target_dim: int, device: torch.device) -> torch.Tensor:
    if edge_attr is None:
        return torch.zeros((0, target_dim), device=device)
    if edge_attr.dim() == 1:
        edge_attr = edge_attr.unsqueeze(-1)
    edge_attr = edge_attr.to(device=device, dtype=torch.float32)
    if edge_attr.size(-1) > target_dim:
        edge_attr = edge_attr[..., :target_dim]
    if edge_attr.size(-1) < target_dim:
        pad = torch.zeros((edge_attr.size(0), target_dim - edge_attr.size(-1)), device=device)
        edge_attr = torch.cat([edge_attr, pad], dim=-1)
    return edge_attr


class XGNIDClassifier(nn.Module):
    def __init__(
        self,
        flow_dim: int = 82,
        packet_dim: int = 1500,
        hidden_dim: int = 64,
        num_classes: int = 8,
        heads: int = 2,
        dropout: float = 0.2,
        edge_dim: int = 5,
    ) -> None:
        super().__init__()
        self.flow_encoder = nn.Linear(flow_dim, hidden_dim)
        self.packet_encoder = nn.Linear(packet_dim, hidden_dim)
        self.conv1 = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False, edge_dim=edge_dim, add_self_loops=False)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False, edge_dim=edge_dim, add_self_loops=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _pack(self, data: HeteroData) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        flow_x = data["flow"].x.to(device=device, dtype=torch.float32)
        packet_x = data["packet"].x.to(device=device, dtype=torch.float32)
        flow_h = self.flow_encoder(flow_x)
        packet_h = self.packet_encoder(packet_x)
        x = torch.cat([flow_h, packet_h], dim=0)

        flow_batch = getattr(data["flow"], "batch", None)
        packet_batch = getattr(data["packet"], "batch", None)
        if flow_batch is None:
            flow_batch = torch.zeros(flow_h.size(0), dtype=torch.long, device=device)
        else:
            flow_batch = flow_batch.to(device=device)
        if packet_batch is None:
            packet_batch = torch.zeros(packet_h.size(0), dtype=torch.long, device=device)
        else:
            packet_batch = packet_batch.to(device=device)
        batch = torch.cat([flow_batch, packet_batch], dim=0)

        num_flow = flow_h.size(0)
        edge_index_parts: list[torch.Tensor] = []
        edge_attr_parts: list[torch.Tensor] = []

        def add_edges(
            edge_type: tuple[str, str, str],
            relation_id: float,
            reverse: bool = False,
        ) -> None:
            if edge_type not in data.edge_types:
                return
            edge_store = data[edge_type]
            edge_index = edge_store.edge_index.to(device=device)
            edge_attr = _pad_edge_attr(getattr(edge_store, "edge_attr", None), 4, device=device)
            if edge_type == ("flow", "contain", "packet"):
                src = edge_index[0]
                dst = edge_index[1] + num_flow
                if reverse:
                    src, dst = dst, src
            elif edge_type == ("packet", "link", "packet"):
                src = edge_index[0] + num_flow
                dst = edge_index[1] + num_flow
                if reverse:
                    src, dst = dst, src
            else:
                return
            relation = torch.full((edge_attr.size(0), 1), relation_id, dtype=torch.float32, device=device)
            edge_index_parts.append(torch.stack([src, dst], dim=0))
            edge_attr_parts.append(torch.cat([edge_attr, relation], dim=-1))

        add_edges(("flow", "contain", "packet"), relation_id=0.0, reverse=False)
        add_edges(("flow", "contain", "packet"), relation_id=1.0, reverse=True)
        add_edges(("packet", "link", "packet"), relation_id=2.0, reverse=False)

        if edge_index_parts:
            edge_index = torch.cat(edge_index_parts, dim=1)
            edge_attr = torch.cat(edge_attr_parts, dim=0)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.empty((0, 5), dtype=torch.float32, device=device)

        return x, edge_index, edge_attr, batch

    def forward(self, data: HeteroData) -> torch.Tensor:
        x, edge_index, edge_attr, batch = self._pack(data)
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = global_mean_pool(x, batch)
        return self.classifier(x)
