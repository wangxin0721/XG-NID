from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence
import warnings

import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader


TABLE4_TEST_TARGETS: dict[int, int] = {
    0: 4000,  # Benign
    1: 1090,  # WebBased
    2: 4000,  # Spoofing
    3: 4000,  # Recon
    4: 4000,  # Mirai
    5: 4000,  # DoS
    6: 4000,  # DDoS
    7: 467,   # BruteForce
}


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


def _attach_graph_indices(graphs: Sequence[HeteroData]) -> list[HeteroData]:
    indexed_graphs: list[HeteroData] = []
    for idx, graph in enumerate(graphs):
        graph.graph_index = torch.tensor([idx], dtype=torch.long)
        indexed_graphs.append(graph)
    return indexed_graphs


def load_graphs(path: str | Path) -> list[HeteroData]:
    path = Path(path)
    if path.is_dir():
        graphs: list[HeteroData] = []
        for file in sorted(path.rglob("*.pt")):
            try:
                graphs.extend(_as_graph_list(_load_graph_container(file)))
            except TypeError as exc:
                warnings.warn(f"Skipping unsupported graph file {file.name}: {exc}")
        return _attach_graph_indices(graphs)
    return _attach_graph_indices(_as_graph_list(_load_graph_container(path)))


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


def subset_graphs(graphs: Sequence[HeteroData], indices: Sequence[int]) -> list[HeteroData]:
    return _subset_graphs(graphs, indices)


def _group_indices_by_label(graphs: Sequence[HeteroData]) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = {}
    for idx, graph in enumerate(graphs):
        label = graph_label(graph)
        if label is None:
            continue
        grouped.setdefault(label, []).append(idx)
    return grouped


def _shuffle_indices(indices: Sequence[int], seed: int) -> list[int]:
    if not indices:
        return []
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(indices), generator=rng).tolist()
    return [indices[i] for i in perm]


def _sample_indices(indices: Sequence[int], n: int, seed: int) -> list[int]:
    if n <= 0 or not indices:
        return []
    rng = torch.Generator().manual_seed(seed)
    if n <= len(indices):
        perm = torch.randperm(len(indices), generator=rng).tolist()
        return [indices[i] for i in perm[:n]]
    pick = torch.randint(low=0, high=len(indices), size=(n,), generator=rng).tolist()
    return [indices[i] for i in pick]


def split_graph_indices_paper_table4(
    graphs: Sequence[HeteroData],
    test_ratio: float = 0.2,
    test_cap: int = 4000,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[int]]:
    if not 0 < test_ratio <= 1:
        raise ValueError("test_ratio must be in (0, 1].")
    if test_cap <= 0:
        raise ValueError("test_cap must be positive.")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1).")

    grouped = _group_indices_by_label(graphs)
    required_labels = set(TABLE4_TEST_TARGETS)
    missing = sorted(required_labels.difference(grouped))
    if missing:
        raise ValueError(f"Missing labels for paper-aligned split: {missing}")
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for label in sorted(grouped):
        indices = _shuffle_indices(grouped[label], seed + label)
        target_test = TABLE4_TEST_TARGETS.get(label)
        if target_test is None:
            raise ValueError(f"Unexpected label in dataset: {label}")
        n_test = min(target_test, test_cap) if test_cap > 0 else target_test
        n_test = min(n_test, len(indices))
        if len(indices) < target_test:
            raise ValueError(
                f"Label {label} has only {len(indices)} samples, "
                f"but paper Table 4 requires {target_test} test samples."
            )
        label_test = indices[:n_test]
        label_remain = indices[n_test:]
        n_val = int(round(len(label_remain) * val_ratio))
        n_val = min(n_val, len(label_remain))
        label_val = label_remain[:n_val]
        label_train = label_remain[n_val:]
        test_idx.extend(label_test)
        val_idx.extend(label_val)
        train_idx.extend(label_train)

    train_idx = _shuffle_indices(train_idx, seed)
    val_idx = _shuffle_indices(val_idx, seed + 999)
    test_idx = _shuffle_indices(test_idx, seed + 1999)
    return {"train": train_idx, "val": val_idx, "test": test_idx}


split_graph_indices = split_graph_indices_paper_table4


def build_split_record(
    graphs: Sequence[HeteroData],
    split_indices: dict[str, Sequence[int]],
    *,
    data_path: str | Path,
    split_strategy: str,
    test_ratio: float,
    test_cap: int,
    val_ratio: float,
    seed: int,
    target_per_class: int | None,
    test_targets: dict[int, int] | None = None,
) -> dict[str, object]:
    train_graphs = _subset_graphs(graphs, split_indices["train"])
    val_graphs = _subset_graphs(graphs, split_indices["val"])
    test_graphs = _subset_graphs(graphs, split_indices["test"])
    return {
        "schema_version": 1,
        "split_strategy": split_strategy,
        "data_path": str(Path(data_path)),
        "data_path_resolved": str(Path(data_path).resolve()),
        "test_ratio": float(test_ratio),
        "test_cap": int(test_cap),
        "val_ratio": float(val_ratio),
        "seed": int(seed),
        "target_per_class": target_per_class,
        "test_targets": {str(k): int(v) for k, v in (test_targets or TABLE4_TEST_TARGETS).items()},
        "total_graphs": len(graphs),
        "label_counts": label_counts(graphs),
        "split_indices": {
            "train": [int(i) for i in split_indices["train"]],
            "val": [int(i) for i in split_indices["val"]],
            "test": [int(i) for i in split_indices["test"]],
        },
        "split_label_counts": {
            "train": label_counts(train_graphs),
            "val": label_counts(val_graphs),
            "test": label_counts(test_graphs),
        },
    }


def save_split_record(record: dict[str, object], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_split_record(path: str | Path) -> dict[str, object]:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _balance_train_graphs(
    graphs: Sequence[HeteroData],
    seed: int = 42,
    train_samples_per_class: int | None = None,
) -> list[HeteroData]:
    target = train_samples_per_class if train_samples_per_class is not None else 20000
    labels = [graph_label(graph) for graph in graphs]
    grouped: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        if label is None:
            continue
        grouped.setdefault(label, []).append(idx)
    if not grouped:
        return list(graphs)
    balanced_indices: list[int] = []
    for label in sorted(grouped):
        indices = grouped[label]
        if len(indices) >= target:
            balanced_indices.extend(_sample_indices(indices, target, seed + label))
        else:
            balanced_indices.extend(indices)
            balanced_indices.extend(_sample_indices(indices, target - len(indices), seed + label + 10_000))
    balanced_indices = _shuffle_indices(balanced_indices, seed)
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
    if stratify is False:
        warnings.warn("stratify=False is ignored for the paper-aligned split; using class-wise sampling.")
    split_indices = split_graph_indices_paper_table4(
        graphs,
        test_ratio=1.0 - train_ratio - val_ratio,
        test_cap=4000,
        val_ratio=val_ratio,
        seed=seed,
    )
    train_graphs = _subset_graphs(graphs, split_indices["train"])
    if balance_train:
        train_graphs = _balance_train_graphs(train_graphs, seed=seed, train_samples_per_class=train_samples_per_class)
    return train_graphs, _subset_graphs(graphs, split_indices["val"]), _subset_graphs(graphs, split_indices["test"])


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
    split_indices: dict[str, Sequence[int]] | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    if split_indices is None:
        train_graphs, val_graphs, test_graphs = split_graphs(
            graphs,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
            stratify=stratify,
            balance_train=balance_train,
            train_samples_per_class=train_samples_per_class,
        )
    else:
        train_graphs = _subset_graphs(graphs, split_indices["train"])
        if balance_train:
            train_graphs = _balance_train_graphs(train_graphs, seed=seed, train_samples_per_class=train_samples_per_class)
        val_graphs = _subset_graphs(graphs, split_indices["val"])
        test_graphs = _subset_graphs(graphs, split_indices["test"])
    return (
        DataLoader(train_graphs, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_graphs, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_graphs, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )
