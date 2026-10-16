"""Controlled experiment adapters. Historical model names retain their meaning."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch import nn

from .model import (DualBranchLogitGatedHeteroGNN_Edge, _PaperStyleHeteroEncoder,
                    FLOW_TO_PACKET, PACKET_TO_FLOW, PACKET_TO_PACKET)

RELATIONS = (FLOW_TO_PACKET, PACKET_TO_FLOW, PACKET_TO_PACKET)


class ReferenceEdge(nn.Module):
    """Run the vendored reference model, with a same-GAT attribute ablation."""

    def __init__(self, graph, use_edge_attr=True):
        super().__init__()
        path = Path(__file__).resolve().parents[2] / "third_party/GNN4ID/Utility/Model.py"
        spec = importlib.util.spec_from_file_location("gnn4id_reference", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.reference = module.HeteroGNN_Edge(graph, {"hidden_size": 64, "eps": 1.0})
        self.use_edge_attr = use_edge_attr

    def forward(self, data):
        validate_graph(data)
        # The reference accepts a positional attribute dictionary. None disables
        # GAT's edge term while preserving every edge and every layer.
        attrs = {r: data[r].edge_attr if self.use_edge_attr else None for r in RELATIONS}
        return self.reference(data.x_dict, data.edge_index_dict, attrs, data)


def validate_graph(data):
    for node in ("flow", "packet"):
        if data[node].x.size(0) == 0 or not torch.isfinite(data[node].x).all():
            raise ValueError(f"{node}: empty or nonfinite node features")
    for relation, dim in zip(RELATIONS, (4, 4, 1)):
        index, attr = data[relation].edge_index, data[relation].edge_attr
        if index.dtype != torch.long or index.ndim != 2 or index.size(0) != 2:
            raise ValueError(f"{relation}: invalid edge_index")
        if attr.shape != (index.size(1), dim) or not torch.isfinite(attr).all():
            raise ValueError(f"{relation}: invalid edge_attr")
        for axis, node in ((0, relation[0]), (1, relation[2])):
            if index.numel() and (index[axis].min() < 0 or index[axis].max() >= data[node].x.size(0)):
                raise ValueError(f"{relation}: node index out of bounds")
    if not torch.equal(data[PACKET_TO_FLOW].edge_index, data[FLOW_TO_PACKET].edge_index.flip(0)):
        raise ValueError("reverse contain order differs")
    if not torch.equal(data[PACKET_TO_FLOW].edge_attr, data[FLOW_TO_PACKET].edge_attr):
        raise ValueError("reverse contain attributes differ from packet metadata")


def build_experiment(name, graph):
    if name in ("E0", "E1"):
        return ReferenceEdge(graph, use_edge_attr=name == "E0")
    if name not in ("E2", "E3", "E4"):
        raise ValueError(name)
    model = DualBranchLogitGatedHeteroGNN_Edge(
        hidden_dim=64, aux_loss_weight=0, webbased_aux_weight=0,
        webbased_recon_aux_weight=0, benign_spoofing_aux_weight=0,
        webbased_recon_hard_weight=0)
    # E2 is fixed equal-weight logits fusion using the E3 architecture, rather
    # than the historical concatenation model with a different prediction head.
    model.use_gate = name != "E2"
    for encoder in model.modules():
        if isinstance(encoder, _PaperStyleHeteroEncoder):
            encoder.use_edge_attr = name != "E4"
    return model
