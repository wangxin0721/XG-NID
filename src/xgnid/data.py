from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch.utils.data import Dataset, random_split
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader


def _as_graph_list(obj) -> list[HeteroData]:
    if isinstance(obj, HeteroData):
        return [obj]
    if isinstance(obj, list):
        graphs = obj
    elif isinstance(obj, dict) and "graphs" in obj:
        graphs = obj["graphs"]
    else:
        raise TypeError(f"Unsupported graph container: {type(obj)!r}")
    if not all(isinstance(graph, HeteroData) for graph in graphs):
        bad = {type(graph) for graph in graphs if not isinstance(graph, HeteroData)}
        raise TypeError(f"Expected HeteroData graphs, got: {bad}")
    return graphs


def load_graphs(path: str | Path) -> list[HeteroData]:
    path = Path(path)
    if path.is_dir():
        graphs: list[HeteroData] = []
        for file in sorted(path.glob("*.pt")):
            graphs.extend(_as_graph_list(torch.load(file, map_location="cpu")))
        return graphs
    return _as_graph_list(torch.load(path, map_location="cpu"))


def summarize_graphs(graphs: Sequence[HeteroData]) -> dict[str, int]:
    labels = []
    flow_nodes = 0
    packet_nodes = 0
    for graph in graphs:
        if "flow" in graph.node_types:
            flow_nodes += int(graph["flow"].num_nodes)
        if "packet" in graph.node_types:
            packet_nodes += int(graph["packet"].num_nodes)
        if hasattr(graph, "y") and graph.y is not None:
            labels.append(int(graph.y.item()))
    return {
        "graphs": len(graphs),
        "flow_nodes": flow_nodes,
        "packet_nodes": packet_nodes,
        "labels": len(set(labels)) if labels else 0,
    }


def split_graphs(
    graphs: Sequence[HeteroData],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[HeteroData], list[HeteroData], list[HeteroData]]:
    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1:
        raise ValueError("Ratios must be in (0, 1).")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1.")
    total = len(graphs)
    test_ratio = 1.0 - train_ratio - val_ratio
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(graphs, [train_size, val_size, test_size], generator=generator)
    return list(train_set), list(val_set), list(test_set)


def make_loaders(
    graphs: Sequence[HeteroData],
    batch_size: int = 16,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_graphs, val_graphs, test_graphs = split_graphs(graphs, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    return (
        DataLoader(train_graphs, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_graphs, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_graphs, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )

