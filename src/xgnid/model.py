from __future__ import annotations

import math
from typing import Literal, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv, SAGEConv, global_mean_pool


FLOW_TO_PACKET: Tuple[str, str, str] = ("flow", "contain", "packet")
PACKET_TO_FLOW: Tuple[str, str, str] = ("packet", "rev_contain", "flow")
PACKET_TO_PACKET: Tuple[str, str, str] = ("packet", "link", "packet")
ModelName = Literal[
    "paper",
    "edge",
    "dual",
    "dual_edge",
    "dual_gate",
    "dual_gate_edge",
    "dual_gate_logit",
    "dual_gate_logit_edge",
]
BranchMode = Literal["flow", "packet", "dual"]


def _align_feature_dim(x: torch.Tensor, target_dim: int | None) -> torch.Tensor:
    if target_dim is None:
        return x
    current_dim = int(x.size(-1))
    if current_dim == target_dim:
        return x
    if current_dim < target_dim:
        return F.pad(x, (0, target_dim - current_dim))
    return x[..., :target_dim]


class _PaperStyleHeteroEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        eps: float = 1.0,
        conv_cls=SAGEConv,
        edge_aware: bool = False,
        flow_input_dim: int | None = None,
        packet_input_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_aware = edge_aware
        self.flow_input_dim = flow_input_dim
        self.packet_input_dim = packet_input_dim

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

    @staticmethod
    def _ensure_all_node_types(
        x_dict: dict[str, torch.Tensor],
        residual: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        for node_type, x in residual.items():
            if node_type not in x_dict:
                x_dict[node_type] = x
        return x_dict

    def forward(self, data: HeteroData) -> dict[str, torch.Tensor]:
        x_dict = {
            "flow": _align_feature_dim(data["flow"].x.float(), self.flow_input_dim),
            "packet": _align_feature_dim(data["packet"].x.float(), self.packet_input_dim),
        }
        edge_index_dict = self._edge_index_dict(data)
        edge_attr_dict = self._edge_attr_dict(data)

        if self.edge_aware and edge_attr_dict:
            x_dict = self.convs1(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
        else:
            x_dict = self.convs1(x_dict, edge_index_dict)
        x_dict = self._ensure_all_node_types(
            x_dict,
            {
                "flow": _align_feature_dim(data["flow"].x.float(), self.flow_input_dim),
                "packet": _align_feature_dim(data["packet"].x.float(), self.packet_input_dim),
            },
        )
        x_dict = self._apply_nodewise_ops(x_dict, self.bns1, self.relus1)

        if self.edge_aware and edge_attr_dict:
            x_dict = self.convs2(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
        else:
            x_dict = self.convs2(x_dict, edge_index_dict)
        x_dict = self._ensure_all_node_types(
            x_dict,
            {
                "flow": _align_feature_dim(data["flow"].x.float(), self.flow_input_dim),
                "packet": _align_feature_dim(data["packet"].x.float(), self.packet_input_dim),
            },
        )
        x_dict = self._apply_nodewise_ops(x_dict, self.bns2, self.relus2)
        return x_dict


class _PaperStyleHeteroBase(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        num_classes: int = 8,
        eps: float = 1.0,
        conv_cls = SAGEConv,
        edge_aware: bool = False,
        flow_input_dim: int | None = None,
        packet_input_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.encoder = _PaperStyleHeteroEncoder(
            hidden_dim=hidden_dim,
            eps=eps,
            conv_cls=conv_cls,
            edge_aware=edge_aware,
            flow_input_dim=flow_input_dim,
            packet_input_dim=packet_input_dim,
        )
        self.edge_aware = edge_aware

        self.graph_prediction = nn.Linear(hidden_dim * 2, hidden_dim)
        self.graph_prediction_1 = nn.Linear(hidden_dim, max(hidden_dim // 4, 16))
        self.graph_prediction_2 = nn.Linear(max(hidden_dim // 4, 16), num_classes)

    def _graph_logits(self, data: HeteroData) -> torch.Tensor:
        x_dict = self.encoder(data)
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

    def loss(
        self,
        preds: torch.Tensor,
        label: torch.Tensor,
        class_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return F.nll_loss(preds, label, weight=class_weight)


class HeteroGNN(_PaperStyleHeteroBase):
    def __init__(
        self,
        hidden_dim: int = 64,
        num_classes: int = 8,
        eps: float = 1.0,
        flow_input_dim: int | None = None,
        packet_input_dim: int | None = None,
        **_: object,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            eps=eps,
            conv_cls=SAGEConv,
            edge_aware=False,
            flow_input_dim=flow_input_dim,
            packet_input_dim=packet_input_dim,
        )


class HeteroGNN_Edge(_PaperStyleHeteroBase):
    def __init__(
        self,
        hidden_dim: int = 64,
        num_classes: int = 8,
        eps: float = 1.0,
        flow_input_dim: int | None = None,
        packet_input_dim: int | None = None,
        **_: object,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            eps=eps,
            conv_cls=GATConv,
            edge_aware=True,
            flow_input_dim=flow_input_dim,
            packet_input_dim=packet_input_dim,
        )


class DualBranchHeteroGNN(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        num_classes: int = 8,
        eps: float = 1.0,
        branch_mode: BranchMode = "dual",
        conv_cls=SAGEConv,
        edge_aware: bool = False,
        flow_input_dim: int | None = None,
        packet_input_dim: int | None = None,
    ) -> None:
        super().__init__()
        if branch_mode not in {"flow", "packet", "dual"}:
            raise ValueError(f"Unsupported branch_mode: {branch_mode}")
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.branch_mode = branch_mode
        self.flow_encoder = _PaperStyleHeteroEncoder(
            hidden_dim=hidden_dim,
            eps=eps,
            conv_cls=conv_cls,
            edge_aware=edge_aware,
            flow_input_dim=flow_input_dim,
            packet_input_dim=packet_input_dim,
        )
        self.packet_encoder = _PaperStyleHeteroEncoder(
            hidden_dim=hidden_dim,
            eps=eps,
            conv_cls=conv_cls,
            edge_aware=edge_aware,
            flow_input_dim=flow_input_dim,
            packet_input_dim=packet_input_dim,
        )
        self.graph_prediction = nn.Linear(hidden_dim * 2, hidden_dim)
        self.graph_prediction_1 = nn.Linear(hidden_dim, max(hidden_dim // 4, 16))
        self.graph_prediction_2 = nn.Linear(max(hidden_dim // 4, 16), num_classes)

    def _branch_embedding(
        self,
        encoder: _PaperStyleHeteroEncoder,
        data: HeteroData,
        node_type: str,
    ) -> torch.Tensor:
        x_dict = encoder(data)
        return global_mean_pool(x_dict[node_type], data[node_type].batch)

    def _fuse_embeddings(self, flow_emb: torch.Tensor, packet_emb: torch.Tensor) -> torch.Tensor:
        if self.branch_mode == "flow":
            packet_emb = torch.zeros_like(packet_emb)
        elif self.branch_mode == "packet":
            flow_emb = torch.zeros_like(flow_emb)
        return torch.cat([flow_emb, packet_emb], dim=1)

    def _graph_logits(self, data: HeteroData) -> torch.Tensor:
        flow_emb = self._branch_embedding(self.flow_encoder, data, "flow")
        packet_emb = self._branch_embedding(self.packet_encoder, data, "packet")
        graph_emb = self._fuse_embeddings(flow_emb, packet_emb)
        out = self.graph_prediction(graph_emb)
        out = F.leaky_relu(out)
        out = self.graph_prediction_1(out)
        out = F.leaky_relu(out)
        out = self.graph_prediction_2(out)
        return out

    def forward(self, data: HeteroData) -> torch.Tensor:
        return F.log_softmax(self._graph_logits(data), dim=1)

    def loss(
        self,
        preds: torch.Tensor,
        label: torch.Tensor,
        class_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return F.nll_loss(preds, label, weight=class_weight)


class DualBranchHeteroGNN_Edge(DualBranchHeteroGNN):
    def __init__(
        self,
        hidden_dim: int = 64,
        num_classes: int = 8,
        eps: float = 1.0,
        branch_mode: BranchMode = "dual",
        flow_input_dim: int | None = None,
        packet_input_dim: int | None = None,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            eps=eps,
            branch_mode=branch_mode,
            conv_cls=GATConv,
            edge_aware=True,
            flow_input_dim=flow_input_dim,
            packet_input_dim=packet_input_dim,
        )


class _ConfidenceGate(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 4, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        flow_emb: torch.Tensor,
        packet_emb: torch.Tensor,
        flow_conf: torch.Tensor,
        packet_conf: torch.Tensor,
        flow_margin: torch.Tensor,
        packet_margin: torch.Tensor,
    ) -> torch.Tensor:
        gate_input = torch.cat(
            [
                flow_emb,
                packet_emb,
                flow_conf.unsqueeze(-1),
                packet_conf.unsqueeze(-1),
                flow_margin.unsqueeze(-1),
                packet_margin.unsqueeze(-1),
            ],
            dim=-1,
        )
        return torch.softmax(self.net(gate_input), dim=-1)


class _LogitConfidenceGate(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_classes * 2 + 5, max(num_classes, 16)),
            nn.LeakyReLU(),
            nn.Linear(max(num_classes, 16), 2),
        )

    def forward(
        self,
        flow_logits: torch.Tensor,
        packet_logits: torch.Tensor,
        flow_conf: torch.Tensor,
        packet_conf: torch.Tensor,
        flow_margin: torch.Tensor,
        packet_margin: torch.Tensor,
        branch_agreement: torch.Tensor,
    ) -> torch.Tensor:
        gate_input = torch.cat(
            [
                flow_logits,
                packet_logits,
                flow_conf.unsqueeze(-1),
                packet_conf.unsqueeze(-1),
                flow_margin.unsqueeze(-1),
                packet_margin.unsqueeze(-1),
                branch_agreement.unsqueeze(-1),
            ],
            dim=-1,
        )
        return torch.softmax(self.net(gate_input), dim=-1)


class DualBranchGatedHeteroGNN(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        num_classes: int = 8,
        eps: float = 1.0,
        branch_mode: BranchMode = "dual",
        conv_cls=SAGEConv,
        edge_aware: bool = False,
        aux_loss_weight: float = 0.05,
        flow_input_dim: int | None = None,
        packet_input_dim: int | None = None,
    ) -> None:
        super().__init__()
        if branch_mode not in {"flow", "packet", "dual"}:
            raise ValueError(f"Unsupported branch_mode: {branch_mode}")
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.branch_mode = branch_mode
        self.aux_loss_weight = aux_loss_weight

        self.flow_encoder = _PaperStyleHeteroEncoder(
            hidden_dim=hidden_dim,
            eps=eps,
            conv_cls=conv_cls,
            edge_aware=edge_aware,
            flow_input_dim=flow_input_dim,
            packet_input_dim=packet_input_dim,
        )
        self.packet_encoder = _PaperStyleHeteroEncoder(
            hidden_dim=hidden_dim,
            eps=eps,
            conv_cls=conv_cls,
            edge_aware=edge_aware,
            flow_input_dim=flow_input_dim,
            packet_input_dim=packet_input_dim,
        )
        self.flow_classifier = nn.Linear(hidden_dim, num_classes)
        self.packet_classifier = nn.Linear(hidden_dim, num_classes)
        self.gate = _ConfidenceGate(hidden_dim)
        self.shared_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self._last_cache: dict[str, torch.Tensor] = {}

    @staticmethod
    def _branch_embedding(
        encoder: _PaperStyleHeteroEncoder,
        data: HeteroData,
        node_type: str,
    ) -> torch.Tensor:
        x_dict = encoder(data)
        return global_mean_pool(x_dict[node_type], data[node_type].batch)

    @staticmethod
    def _confidence_from_logits(logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
        denom = math.log(float(max(probs.size(-1), 2)))
        return 1.0 - entropy / denom

    @staticmethod
    def _margin_from_logits(logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        top2 = torch.topk(probs, k=min(2, probs.size(-1)), dim=-1).values
        if top2.size(-1) == 1:
            return top2[:, 0]
        return top2[:, 0] - top2[:, 1]

    def _branch_outputs(self, data: HeteroData) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        flow_emb = self._branch_embedding(self.flow_encoder, data, "flow")
        packet_emb = self._branch_embedding(self.packet_encoder, data, "packet")
        flow_logits = self.flow_classifier(flow_emb)
        packet_logits = self.packet_classifier(packet_emb)
        return flow_emb, packet_emb, flow_logits, packet_logits

    def _graph_logits(self, data: HeteroData) -> torch.Tensor:
        flow_emb, packet_emb, flow_logits, packet_logits = self._branch_outputs(data)
        flow_conf = self._confidence_from_logits(flow_logits)
        packet_conf = self._confidence_from_logits(packet_logits)
        flow_margin = self._margin_from_logits(flow_logits)
        packet_margin = self._margin_from_logits(packet_logits)
        branch_agreement = F.cosine_similarity(flow_logits, packet_logits, dim=-1)

        if self.branch_mode == "flow":
            gate_weights = torch.zeros(flow_emb.size(0), 2, device=flow_emb.device, dtype=flow_emb.dtype)
            gate_weights[:, 0] = 1.0
            packet_emb = torch.zeros_like(packet_emb)
        elif self.branch_mode == "packet":
            gate_weights = torch.zeros(flow_emb.size(0), 2, device=flow_emb.device, dtype=flow_emb.dtype)
            gate_weights[:, 1] = 1.0
            flow_emb = torch.zeros_like(flow_emb)
        else:
            gate_weights = self.gate(flow_emb, packet_emb, flow_conf, packet_conf, flow_margin, packet_margin)
            gate_weights = 0.85 * gate_weights + 0.15 * torch.stack([flow_conf, packet_conf], dim=-1)
            gate_weights = gate_weights / gate_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        fused = gate_weights[:, :1] * flow_emb + gate_weights[:, 1:] * packet_emb
        logits = self.shared_head(fused)
        self._last_cache = {
            "flow_logits": flow_logits,
            "packet_logits": packet_logits,
            "gate_weights": gate_weights,
            "flow_conf": flow_conf,
            "packet_conf": packet_conf,
            "flow_margin": flow_margin,
            "packet_margin": packet_margin,
            "branch_agreement": branch_agreement,
        }
        return logits

    def forward(self, data: HeteroData) -> torch.Tensor:
        return F.log_softmax(self._graph_logits(data), dim=1)

    def loss(
        self,
        preds: torch.Tensor,
        label: torch.Tensor,
        class_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        main_loss = F.nll_loss(preds, label, weight=class_weight)
        cache = getattr(self, "_last_cache", {})
        if not cache:
            return main_loss
        flow_loss = F.nll_loss(F.log_softmax(cache["flow_logits"], dim=1), label, weight=class_weight)
        packet_loss = F.nll_loss(F.log_softmax(cache["packet_logits"], dim=1), label, weight=class_weight)
        aux_loss = 0.5 * (flow_loss + packet_loss)
        agreement_penalty = (1.0 - cache["branch_agreement"].mean()).clamp_min(0.0)
        aux_loss = aux_loss + 0.1 * agreement_penalty
        return main_loss + self.aux_loss_weight * aux_loss


class DualBranchGatedHeteroGNN_Edge(DualBranchGatedHeteroGNN):
    def __init__(
        self,
        hidden_dim: int = 64,
        num_classes: int = 8,
        eps: float = 1.0,
        branch_mode: BranchMode = "dual",
        aux_loss_weight: float = 0.2,
        flow_input_dim: int | None = None,
        packet_input_dim: int | None = None,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            eps=eps,
            branch_mode=branch_mode,
            conv_cls=GATConv,
            edge_aware=True,
            aux_loss_weight=aux_loss_weight,
            flow_input_dim=flow_input_dim,
            packet_input_dim=packet_input_dim,
        )


class DualBranchLogitGatedHeteroGNN(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        num_classes: int = 8,
        eps: float = 1.0,
        branch_mode: BranchMode = "dual",
        conv_cls=SAGEConv,
        edge_aware: bool = False,
        aux_loss_weight: float = 0.02,
        webbased_aux_weight: float = 0.25,
        webbased_recon_aux_weight: float = 0.25,
        webbased_recon_hard_weight: float = 2.0,
        flow_input_dim: int | None = None,
        packet_input_dim: int | None = None,
    ) -> None:
        super().__init__()
        if branch_mode not in {"flow", "packet", "dual"}:
            raise ValueError(f"Unsupported branch_mode: {branch_mode}")
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.branch_mode = branch_mode
        self.aux_loss_weight = aux_loss_weight
        self.webbased_aux_weight = webbased_aux_weight
        self.webbased_recon_aux_weight = webbased_recon_aux_weight
        self.webbased_recon_hard_weight = webbased_recon_hard_weight

        self.flow_encoder = _PaperStyleHeteroEncoder(
            hidden_dim=hidden_dim,
            eps=eps,
            conv_cls=conv_cls,
            edge_aware=edge_aware,
            flow_input_dim=flow_input_dim,
            packet_input_dim=packet_input_dim,
        )
        self.packet_encoder = _PaperStyleHeteroEncoder(
            hidden_dim=hidden_dim,
            eps=eps,
            conv_cls=conv_cls,
            edge_aware=edge_aware,
            flow_input_dim=flow_input_dim,
            packet_input_dim=packet_input_dim,
        )
        self.flow_classifier = nn.Linear(hidden_dim, num_classes)
        self.packet_classifier = nn.Linear(hidden_dim, num_classes)
        self.gate = _LogitConfidenceGate(num_classes)
        self.webbased_head = nn.Linear(num_classes, 2)
        self.webbased_recon_head = nn.Linear(hidden_dim, 2)
        self.shared_head = nn.Sequential(
            nn.Linear(num_classes, max(hidden_dim // 2, 16)),
            nn.LeakyReLU(),
            nn.Linear(max(hidden_dim // 2, 16), num_classes),
        )
        self._last_cache: dict[str, torch.Tensor] = {}

    @staticmethod
    def _branch_embedding(
        encoder: _PaperStyleHeteroEncoder,
        data: HeteroData,
        node_type: str,
    ) -> torch.Tensor:
        x_dict = encoder(data)
        return global_mean_pool(x_dict[node_type], data[node_type].batch)

    @staticmethod
    def _confidence_from_logits(logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
        denom = math.log(float(max(probs.size(-1), 2)))
        return 1.0 - entropy / denom

    @staticmethod
    def _margin_from_logits(logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        top2 = torch.topk(probs, k=min(2, probs.size(-1)), dim=-1).values
        if top2.size(-1) == 1:
            return top2[:, 0]
        return top2[:, 0] - top2[:, 1]

    def _branch_outputs(self, data: HeteroData) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        flow_emb = self._branch_embedding(self.flow_encoder, data, "flow")
        packet_emb = self._branch_embedding(self.packet_encoder, data, "packet")
        flow_logits = self.flow_classifier(flow_emb)
        packet_logits = self.packet_classifier(packet_emb)
        return flow_emb, packet_emb, flow_logits, packet_logits

    @staticmethod
    def _rerank_recon_webbased(
        logits: torch.Tensor,
        recon_webbased_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Re-score Recon/WebBased only when the 8-class head is already in that pair."""
        base_pred = logits.argmax(dim=-1)
        rerank_mask = (base_pred == 1) | (base_pred == 3)
        if not rerank_mask.any():
            return logits

        reranked = logits.clone()
        pair_logits = recon_webbased_logits[rerank_mask]
        # Binary head order matches the training target: 0=WebBased, 1=Recon.
        reranked[rerank_mask, 1] = pair_logits[:, 0]
        reranked[rerank_mask, 3] = pair_logits[:, 1]
        return reranked

    def _graph_logits(self, data: HeteroData) -> torch.Tensor:
        flow_emb, packet_emb, flow_logits, packet_logits = self._branch_outputs(data)
        flow_conf = self._confidence_from_logits(flow_logits)
        packet_conf = self._confidence_from_logits(packet_logits)
        flow_margin = self._margin_from_logits(flow_logits)
        packet_margin = self._margin_from_logits(packet_logits)
        branch_agreement = F.cosine_similarity(flow_logits, packet_logits, dim=-1)

        if self.branch_mode == "flow":
            gate_weights = torch.zeros(flow_logits.size(0), 2, device=flow_logits.device, dtype=flow_logits.dtype)
            gate_weights[:, 0] = 1.0
        elif self.branch_mode == "packet":
            gate_weights = torch.zeros(flow_logits.size(0), 2, device=flow_logits.device, dtype=flow_logits.dtype)
            gate_weights[:, 1] = 1.0
        else:
            gate_weights = self.gate(
                flow_logits,
                packet_logits,
                flow_conf,
                packet_conf,
                flow_margin,
                packet_margin,
                branch_agreement,
            )
            gate_weights = 0.9 * gate_weights + 0.1 * torch.stack([flow_conf, packet_conf], dim=-1)
            gate_weights = gate_weights / gate_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        fused_logits = gate_weights[:, :1] * flow_logits + gate_weights[:, 1:] * packet_logits
        fused_emb = gate_weights[:, :1] * flow_emb + gate_weights[:, 1:] * packet_emb
        webbased_logits = self.webbased_head(fused_logits)
        webbased_recon_logits = self.webbased_recon_head(fused_emb)
        logits = self.shared_head(fused_logits)
        self._last_cache = {
            "flow_logits": flow_logits,
            "packet_logits": packet_logits,
            "gate_weights": gate_weights,
            "flow_conf": flow_conf,
            "packet_conf": packet_conf,
            "flow_margin": flow_margin,
            "packet_margin": packet_margin,
            "branch_agreement": branch_agreement,
            "fused_emb": fused_emb,
            "webbased_logits": webbased_logits,
            "webbased_recon_logits": webbased_recon_logits,
        }
        if self.training:
            return logits
        return self._rerank_recon_webbased(logits, webbased_recon_logits)

    @staticmethod
    def _recon_binary_aux_loss(
        label: torch.Tensor,
        sep_logits: torch.Tensor,
    ) -> torch.Tensor:
        pair_mask = (label == 1) | (label == 3)
        if not pair_mask.any():
            return sep_logits.new_zeros(())
        pair_logits = sep_logits[pair_mask]
        pair_target = (label[pair_mask] == 3).long()
        return F.cross_entropy(pair_logits, pair_target)

    @staticmethod
    def _recon_hard_negative_loss(
        fused_emb: torch.Tensor,
        label: torch.Tensor,
        *,
        topk: int,
        sim_margin: float,
    ) -> torch.Tensor:
        wb_mask = label == 1
        recon_mask = label == 3
        if not wb_mask.any() or not recon_mask.any():
            return fused_emb.new_zeros(())

        wb_emb = F.normalize(fused_emb[wb_mask], dim=-1)
        recon_emb = F.normalize(fused_emb[recon_mask], dim=-1)
        sim = wb_emb @ recon_emb.transpose(0, 1)
        if sim.numel() == 0:
            return fused_emb.new_zeros(())

        k_wb = min(topk, sim.size(1))
        k_recon = min(topk, sim.size(0))
        wb_hard = sim.topk(k=k_wb, dim=1).values
        recon_hard = sim.topk(k=k_recon, dim=0).values
        hard_loss = F.relu(wb_hard - sim_margin).mean() + F.relu(recon_hard - sim_margin).mean()

        wb_center = wb_emb.mean(dim=0, keepdim=True)
        recon_center = recon_emb.mean(dim=0, keepdim=True)
        center_sim = F.cosine_similarity(wb_center, recon_center, dim=-1)
        center_loss = F.relu(center_sim - (sim_margin - 0.05)).mean()
        return hard_loss + 0.5 * center_loss

    def _recon_confusion_penalty(
        self,
        fused_emb: torch.Tensor,
        label: torch.Tensor,
        webbased_logits: torch.Tensor,
    ) -> torch.Tensor:
        recon_mask = label == 3
        webbased_mask = label == 1
        if not recon_mask.any() or not webbased_mask.any():
            return fused_emb.new_zeros(())

        recon_emb = F.normalize(fused_emb[recon_mask], dim=-1)
        recon_scores = webbased_logits[recon_mask, 1] - webbased_logits[recon_mask, 0]
        confusion_mask = recon_scores > 0
        if not confusion_mask.any():
            return fused_emb.new_zeros(())

        confused_recon_emb = recon_emb[confusion_mask]
        webbased_center = F.normalize(fused_emb[webbased_mask], dim=-1).mean(dim=0, keepdim=True)
        mean_sim = torch.matmul(confused_recon_emb, webbased_center.t()).mean()
        if not torch.isfinite(mean_sim):
            return fused_emb.new_zeros(())
        return F.relu(mean_sim - 0.1)

    def forward(self, data: HeteroData) -> torch.Tensor:
        return F.log_softmax(self._graph_logits(data), dim=1)

    def loss(
        self,
        preds: torch.Tensor,
        label: torch.Tensor,
        class_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        main_loss = F.nll_loss(preds, label, weight=class_weight)
        cache = getattr(self, "_last_cache", {})
        if not cache:
            return main_loss
        recon_binary_loss = self._recon_binary_aux_loss(label, cache["webbased_recon_logits"])
        return main_loss + self.webbased_recon_aux_weight * recon_binary_loss


class DualBranchLogitGatedHeteroGNN_Edge(DualBranchLogitGatedHeteroGNN):
    def __init__(
        self,
        hidden_dim: int = 64,
        num_classes: int = 8,
        eps: float = 1.0,
        branch_mode: BranchMode = "dual",
        aux_loss_weight: float = 0.02,
        webbased_aux_weight: float = 0.25,
        webbased_recon_aux_weight: float = 0.25,
        webbased_recon_hard_weight: float = 2.0,
        flow_input_dim: int | None = None,
        packet_input_dim: int | None = None,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            eps=eps,
            branch_mode=branch_mode,
            conv_cls=GATConv,
            edge_aware=True,
            aux_loss_weight=aux_loss_weight,
            webbased_aux_weight=webbased_aux_weight,
            webbased_recon_aux_weight=webbased_recon_aux_weight,
            webbased_recon_hard_weight=webbased_recon_hard_weight,
            flow_input_dim=flow_input_dim,
            packet_input_dim=packet_input_dim,
        )


MODEL_REGISTRY = {
    "paper": HeteroGNN,
    "edge": HeteroGNN_Edge,
    "dual": DualBranchHeteroGNN,
    "dual_edge": DualBranchHeteroGNN_Edge,
    "dual_gate": DualBranchGatedHeteroGNN,
    "dual_gate_edge": DualBranchGatedHeteroGNN_Edge,
    "dual_gate_logit": DualBranchLogitGatedHeteroGNN,
    "dual_gate_logit_edge": DualBranchLogitGatedHeteroGNN_Edge,
}


def build_model(model_name: str, **kwargs) -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model: {model_name}. Supported: {', '.join(sorted(MODEL_REGISTRY))}")
    return MODEL_REGISTRY[model_name](**kwargs)


def build_model_from_checkpoint(checkpoint: dict[str, object]) -> nn.Module:
    model_name = str(checkpoint.get("model_name", "paper"))
    model_kwargs = dict(checkpoint.get("model_kwargs", {}))
    return build_model(model_name, **model_kwargs)


XGNIDClassifier = HeteroGNN
