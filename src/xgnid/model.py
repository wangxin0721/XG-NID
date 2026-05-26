from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv, SAGEConv, global_mean_pool


FLOW_TO_PACKET: Tuple[str, str, str] = ("flow", "contain", "packet")
PACKET_TO_FLOW: Tuple[str, str, str] = ("packet", "rev_contain", "flow")
PACKET_TO_PACKET: Tuple[str, str, str] = ("packet", "link", "packet")


class AdaptivePacketPool(nn.Module):
    def __init__(self, hidden_dim: int, selection_ratio: float = 0.35) -> None:
        super().__init__()
        self.selection_ratio = float(selection_ratio)
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, packet_x: torch.Tensor, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if packet_x.numel() == 0:
            empty = packet_x.new_zeros((0, packet_x.size(-1)))
            return empty, empty

        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        mean_pools: list[torch.Tensor] = []
        selected_pools: list[torch.Tensor] = []
        hidden_dim = int(packet_x.size(-1))

        for graph_idx in range(num_graphs):
            mask = batch == graph_idx
            graph_packets = packet_x[mask]
            if graph_packets.numel() == 0:
                zero = packet_x.new_zeros(hidden_dim)
                mean_pools.append(zero)
                selected_pools.append(zero)
                continue

            graph_scores = self.score(graph_packets).squeeze(-1)
            target_k = int(round(graph_packets.size(0) * self.selection_ratio))
            k = max(1, min(graph_packets.size(0), target_k))
            selected_scores, selected_idx = torch.topk(graph_scores, k=k, largest=True)
            selected_packets = graph_packets[selected_idx]
            weights = torch.softmax(selected_scores, dim=0).unsqueeze(-1)

            mean_pools.append(graph_packets.mean(dim=0))
            selected_pools.append((weights * selected_packets).sum(dim=0))

        return torch.stack(mean_pools, dim=0), torch.stack(selected_pools, dim=0)


class _PaperStyleHeteroBase(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        num_classes: int = 8,
        eps: float = 1.0,
        conv_cls = SAGEConv,
        edge_aware: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.edge_aware = edge_aware

        conv_kwargs = {"edge_dim": -1, "add_self_loops": False} if edge_aware else {}

        self.convs1 = HeteroConv(
            {
                FLOW_TO_PACKET: conv_cls((-1, -1), hidden_dim, **conv_kwargs),
                PACKET_TO_FLOW: conv_cls((-1, -1), hidden_dim, **conv_kwargs),
                PACKET_TO_PACKET: conv_cls((-1, -1), hidden_dim, **conv_kwargs),
            },
            aggr="mean",
        )
        self.convs2 = HeteroConv(
            {
                FLOW_TO_PACKET: conv_cls((hidden_dim, hidden_dim), hidden_dim, **conv_kwargs),
                PACKET_TO_FLOW: conv_cls((hidden_dim, hidden_dim), hidden_dim, **conv_kwargs),
                PACKET_TO_PACKET: conv_cls((hidden_dim, hidden_dim), hidden_dim, **conv_kwargs),
            },
            aggr="mean",
        )

        self.bns1 = nn.ModuleDict(
            {
                "flow": nn.BatchNorm1d(hidden_dim, eps=eps),
                "packet": nn.BatchNorm1d(hidden_dim, eps=eps),
            }
        )
        self.bns2 = nn.ModuleDict(
            {
                "flow": nn.BatchNorm1d(hidden_dim, eps=eps),
                "packet": nn.BatchNorm1d(hidden_dim, eps=eps),
            }
        )
        self.relus1 = nn.ModuleDict(
            {
                "flow": nn.LeakyReLU(),
                "packet": nn.LeakyReLU(),
            }
        )
        self.relus2 = nn.ModuleDict(
            {
                "flow": nn.LeakyReLU(),
                "packet": nn.LeakyReLU(),
            }
        )

        self.graph_prediction = nn.Linear(hidden_dim * 2, hidden_dim)
        self.graph_prediction_1 = nn.Linear(hidden_dim, max(hidden_dim // 4, 16))
        self.graph_prediction_2 = nn.Linear(max(hidden_dim // 4, 16), num_classes)

    def _edge_index_dict(self, data: HeteroData) -> dict[tuple[str, str, str], torch.Tensor]:
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor] = {
            FLOW_TO_PACKET: data[FLOW_TO_PACKET].edge_index,
            PACKET_TO_PACKET: data[PACKET_TO_PACKET].edge_index,
        }
        edge_index_dict[PACKET_TO_FLOW] = data[FLOW_TO_PACKET].edge_index.flip(0)
        return edge_index_dict

    def _edge_attr_dict(self, data: HeteroData) -> dict[tuple[str, str, str], torch.Tensor]:
        if not self.edge_aware:
            return {}
        edge_attr_dict: dict[tuple[str, str, str], torch.Tensor] = {}
        contain_attr = getattr(data[FLOW_TO_PACKET], "edge_attr", None)
        link_attr = getattr(data[PACKET_TO_PACKET], "edge_attr", None)
        if contain_attr is not None:
            edge_attr_dict[FLOW_TO_PACKET] = contain_attr
            edge_attr_dict[PACKET_TO_FLOW] = contain_attr
        if link_attr is not None:
            edge_attr_dict[PACKET_TO_PACKET] = link_attr
        return edge_attr_dict

    @staticmethod
    def _apply_nodewise_ops(
        x_dict: dict[str, torch.Tensor],
        bns: nn.ModuleDict,
        relus: nn.ModuleDict,
    ) -> dict[str, torch.Tensor]:
        return {
            node_type: relus[node_type](bns[node_type](x))
            for node_type, x in x_dict.items()
            if node_type in bns
        }

    def _ensure_all_node_types(
        self,
        x_dict: dict[str, torch.Tensor],
        residual: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        for node_type, x in residual.items():
            if node_type not in x_dict:
                x_dict[node_type] = x
        return x_dict

    def _forward_convs(self, data: HeteroData) -> dict[str, torch.Tensor]:
        x_dict = {
            "flow": data["flow"].x.float(),
            "packet": data["packet"].x.float(),
        }
        edge_index_dict = self._edge_index_dict(data)
        edge_attr_dict = self._edge_attr_dict(data)

        if self.edge_aware and edge_attr_dict:
            x_dict = self.convs1(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
        else:
            x_dict = self.convs1(x_dict, edge_index_dict)
        x_dict = self._ensure_all_node_types(x_dict, {
            "flow": data["flow"].x.float(),
            "packet": data["packet"].x.float(),
        })
        x_dict = self._apply_nodewise_ops(x_dict, self.bns1, self.relus1)

        if self.edge_aware and edge_attr_dict:
            x_dict = self.convs2(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
        else:
            x_dict = self.convs2(x_dict, edge_index_dict)
        x_dict = self._ensure_all_node_types(x_dict, {
            "flow": data["flow"].x.float(),
            "packet": data["packet"].x.float(),
        })
        x_dict = self._apply_nodewise_ops(x_dict, self.bns2, self.relus2)
        return x_dict

    def _graph_logits(self, data: HeteroData) -> torch.Tensor:
        x_dict = self._forward_convs(data)
        flow_emb = global_mean_pool(x_dict["flow"], data["flow"].batch)
        packet_emb = global_mean_pool(x_dict["packet"], data["packet"].batch)
        graph_emb = torch.cat([flow_emb, packet_emb], dim=1)
        out = self.graph_prediction(graph_emb)
        out = F.leaky_relu(out)
        out = self.graph_prediction_1(out)
        out = F.leaky_relu(out)
        out = self.graph_prediction_2(out)
        return out

    def forward(self, data: HeteroData) -> torch.Tensor:
        return F.log_softmax(self._graph_logits(data), dim=1)

    def loss(self, preds: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return F.nll_loss(preds, label)


class HeteroGNN(_PaperStyleHeteroBase):
    def __init__(
        self,
        hidden_dim: int = 64,
        num_classes: int = 8,
        eps: float = 1.0,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            eps=eps,
            conv_cls=SAGEConv,
            edge_aware=False,
        )


class HeteroGNN_Edge(_PaperStyleHeteroBase):
    def __init__(
        self,
        hidden_dim: int = 64,
        num_classes: int = 8,
        eps: float = 1.0,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            eps=eps,
            conv_cls=GATConv,
            edge_aware=True,
        )


class AdaptivePacketHeteroGNN(_PaperStyleHeteroBase):
    def __init__(
        self,
        hidden_dim: int = 64,
        num_classes: int = 8,
        eps: float = 1.0,
        selection_ratio: float = 0.35,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            eps=eps,
            conv_cls=SAGEConv,
            edge_aware=False,
        )
        self.selection_ratio = float(selection_ratio)
        self.packet_pool = AdaptivePacketPool(hidden_dim, selection_ratio=selection_ratio)
        self.graph_prediction = nn.Linear(hidden_dim * 3, hidden_dim)
        self.graph_prediction_1 = nn.Linear(hidden_dim, max(hidden_dim // 4, 16))
        self.graph_prediction_2 = nn.Linear(max(hidden_dim // 4, 16), num_classes)

    def _graph_logits(self, data: HeteroData) -> torch.Tensor:
        x_dict = self._forward_convs(data)
        flow_emb = global_mean_pool(x_dict["flow"], data["flow"].batch)
        packet_mean, packet_selected = self.packet_pool(x_dict["packet"], data["packet"].batch)
        graph_emb = torch.cat([flow_emb, packet_mean, packet_selected], dim=1)
        out = self.graph_prediction(graph_emb)
        out = F.leaky_relu(out)
        out = self.graph_prediction_1(out)
        out = F.leaky_relu(out)
        out = self.graph_prediction_2(out)
        return out


def build_model(
    model_name: str = "baseline",
    hidden_dim: int = 64,
    num_classes: int = 8,
    *,
    selection_ratio: float = 0.35,
) -> nn.Module:
    name = model_name.lower()
    if name in {"baseline", "paper", "heterognn", "hetero"}:
        return HeteroGNN(hidden_dim=hidden_dim, num_classes=num_classes)
    if name in {"innov1", "adaptive_packet", "hierarchical_packet"}:
        return AdaptivePacketHeteroGNN(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            selection_ratio=selection_ratio,
        )
    if name in {"edge", "hetero_edge", "heterognn_edge"}:
        return HeteroGNN_Edge(hidden_dim=hidden_dim, num_classes=num_classes)
    raise ValueError(f"Unknown model_name: {model_name!r}")


XGNIDClassifier = HeteroGNN
