from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, global_mean_pool


class HeteroGNN(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        num_classes: int = 8,
        eps: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.convs1 = HeteroConv(
            {
                ("flow", "contain", "packet"): SAGEConv((-1, -1), hidden_dim),
                ("packet", "link", "packet"): SAGEConv((-1, -1), hidden_dim),
            },
            aggr="mean",
        )
        self.convs2 = HeteroConv(
            {
                ("flow", "contain", "packet"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                ("packet", "link", "packet"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
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

    def forward(self, data: HeteroData) -> torch.Tensor:
        x_dict = {
            "flow": data["flow"].x.float(),
            "packet": data["packet"].x.float(),
        }
        edge_index_dict = {
            etype: data[etype].edge_index
            for etype in data.edge_types
            if etype in self.convs1.convs
        }

        x_dict = self.convs1(x_dict, edge_index_dict)
        x_dict = {k: self.bns1[k](v) for k, v in x_dict.items()}
        x_dict = {k: self.relus1[k](v) for k, v in x_dict.items()}

        x_dict = self.convs2(x_dict, edge_index_dict)
        x_dict = {k: self.bns2[k](v) for k, v in x_dict.items()}
        x_dict = {k: self.relus2[k](v) for k, v in x_dict.items()}

        graph_emb = {
            node_type: global_mean_pool(x_dict[node_type], data[node_type].batch)
            for node_type in data.node_types
        }
        graph_emb = torch.cat([graph_emb["flow"], graph_emb["packet"]], dim=1)

        out = self.graph_prediction(graph_emb)
        out = F.leaky_relu(out)
        out = self.graph_prediction_1(out)
        out = F.leaky_relu(out)
        out = self.graph_prediction_2(out)
        return F.log_softmax(out, dim=1)

    def loss(self, preds: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return F.nll_loss(preds, label)


class HeteroGNN_Edge(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_classes: int = 8, eps: float = 1.0) -> None:
        super().__init__()
        self.base = HeteroGNN(hidden_dim=hidden_dim, num_classes=num_classes, eps=eps)

    def forward(self, data: HeteroData) -> torch.Tensor:
        return self.base(data)

    def loss(self, preds: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return self.base.loss(preds, label)


XGNIDClassifier = HeteroGNN
