from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader


def graph_label(graph: HeteroData) -> int | None:
    if hasattr(graph, "y") and graph.y is not None:
        return int(graph.y.view(-1)[0].item())
    return None


def label_counts(graphs: Sequence[HeteroData]) -> dict[int, int]:
    counts: Counter[int] = Counter()
    for graph in graphs:
        label = graph_label(graph)
        if label is not None:
            counts[label] += 1
    return dict(sorted(counts.items()))


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


def _resolve_reference_path(raw: str | Path, bases: Sequence[Path]) -> Path | None:
    raw_path = Path(raw)
    candidates = [raw_path]
    if not raw_path.is_absolute():
        for base in bases:
            candidates.append((base / raw_path).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_graph_container(path: Path, max_depth: int = 3):
    obj = torch.load(path, map_location="cpu")
    depth = 0
    current_path = path
    while depth < max_depth and isinstance(obj, (str, Path)):
        ref = _resolve_reference_path(obj, bases=[current_path.parent, current_path.parent.parent, Path.cwd()])
        if ref is None:
            break
        current_path = ref
        obj = torch.load(ref, map_location="cpu")
        depth += 1
    return obj


def load_graphs(path: str | Path) -> list[HeteroData]:
    path = Path(path)
    if path.is_dir():
        graphs: list[HeteroData] = []
        for file in sorted(path.glob("*.pt")):
            graphs.extend(_as_graph_list(_load_graph_container(file)))
        return graphs
    return _as_graph_list(_load_graph_container(path))


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


def _can_stratify(labels: Sequence[int]) -> bool:
    counts = Counter(labels)
    return len(counts) > 1 and min(counts.values()) >= 2


def _subset_graphs(graphs: Sequence[HeteroData], indices: Sequence[int]) -> list[HeteroData]:
    return [graphs[i] for i in indices]


def _balance_train_graphs(
    graphs: Sequence[HeteroData],
    seed: int = 42,
    train_samples_per_class: int | None = None,
) -> list[HeteroData]:
    labels = [graph_label(graph) for graph in graphs]
    grouped: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        if label is None:
            continue
        grouped.setdefault(label, []).append(idx)
    if not grouped:
        return list(graphs)

    target = train_samples_per_class
    if target is None:
        target = min(len(indices) for indices in grouped.values())
    if target <= 0:
        raise ValueError("train_samples_per_class must be positive.")

    rng = torch.Generator().manual_seed(seed)
    balanced_indices: list[int] = []
    for label in sorted(grouped):
        indices = grouped[label]
        if len(indices) >= target:
            perm = torch.randperm(len(indices), generator=rng).tolist()
            balanced_indices.extend(indices[i] for i in perm[:target])
        else:
            balanced_indices.extend(indices)
            extra = target - len(indices)
            if extra > 0:
                pick = torch.randint(low=0, high=len(indices), size=(extra,), generator=rng).tolist()
                balanced_indices.extend(indices[i] for i in pick)

    perm = torch.randperm(len(balanced_indices), generator=rng).tolist()
    balanced_indices = [balanced_indices[i] for i in perm]
    return _subset_graphs(graphs, balanced_indices)


def split_graphs(
    graphs: Sequence[HeteroData],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    stratify: bool = True,
    balance_train: bool = False,
    train_samples_per_class: int | None = None,
) -> tuple[list[HeteroData], list[HeteroData], list[HeteroData]]:
    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1:
        raise ValueError("Ratios must be in (0, 1).")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1.")
    total = len(graphs)
    test_ratio = 1.0 - train_ratio - val_ratio
    indices = list(range(total))
    labels = [graph_label(graph) for graph in graphs]

    train_idx: list[int]
    val_idx: list[int]
    test_idx: list[int]

    if stratify and _can_stratify([label for label in labels if label is not None]):
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=val_ratio + test_ratio,
            random_state=seed,
            stratify=labels,
        )
        temp_labels = [labels[i] for i in temp_idx]
        if _can_stratify([label for label in temp_labels if label is not None]):
            val_idx, test_idx = train_test_split(
                temp_idx,
                test_size=test_ratio / (val_ratio + test_ratio),
                random_state=seed,
                stratify=temp_labels,
            )
        else:
            temp_idx = list(temp_idx)
            rng = torch.Generator().manual_seed(seed)
            perm = torch.randperm(len(temp_idx), generator=rng).tolist()
            temp_idx = [temp_idx[i] for i in perm]
            split = int(len(temp_idx) * val_ratio / (val_ratio + test_ratio))
            val_idx = temp_idx[:split]
            test_idx = temp_idx[split:]
    else:
        rng = torch.Generator().manual_seed(seed)
        perm = torch.randperm(total, generator=rng).tolist()
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        train_idx = perm[:train_size]
        val_idx = perm[train_size : train_size + val_size]
        test_idx = perm[train_size + val_size :]

    train_graphs = _subset_graphs(graphs, train_idx)
    if balance_train:
        train_graphs = _balance_train_graphs(train_graphs, seed=seed, train_samples_per_class=train_samples_per_class)
    return train_graphs, _subset_graphs(graphs, val_idx), _subset_graphs(graphs, test_idx)


def make_loaders(
    graphs: Sequence[HeteroData],
    batch_size: int = 16,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
    stratify: bool = True,
    balance_train: bool = False,
    train_samples_per_class: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_graphs, val_graphs, test_graphs = split_graphs(
        graphs,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        stratify=stratify,
        balance_train=balance_train,
        train_samples_per_class=train_samples_per_class,
    )
    return (
        DataLoader(train_graphs, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_graphs, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_graphs, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )
