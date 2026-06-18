from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence
import warnings

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

from .data import (
    TABLE4_TEST_TARGETS,
    build_split_record,
    load_graphs,
    make_loaders,
    save_split_record,
    split_graph_indices_paper_table4,
)
from .data import label_counts
from .model import build_model, build_model_from_checkpoint


@dataclass
class TrainResult:
    best_path: Path
    metrics: dict[str, float | str]


def _move_batch(batch, device: torch.device):
    return batch.to(device)


def _graph_index_list(graphs_or_loader) -> list[int]:
    indices: list[int] = []
    dataset = getattr(graphs_or_loader, "dataset", graphs_or_loader)
    for graph in dataset:
        graph_idx = getattr(graph, "graph_index", None)
        if graph_idx is None:
            continue
        indices.append(int(graph_idx.view(-1)[0].item()))
    return indices


def _node_input_dims(graphs) -> tuple[int | None, int | None]:
    flow_dim: int | None = None
    packet_dim: int | None = None
    for graph in graphs:
        flow_x = getattr(graph["flow"], "x", None)
        packet_x = getattr(graph["packet"], "x", None)
        if flow_x is not None:
            flow_dim = max(int(flow_x.size(-1)), flow_dim or 0)
        if packet_x is not None:
            packet_dim = max(int(packet_x.size(-1)), packet_dim or 0)
    return flow_dim, packet_dim


def _model_kwargs_for_training(
    *,
    model_name: str,
    hidden_dim: int,
    num_classes: int,
    branch_mode: str,
    webbased_aux_weight: float,
    webbased_recon_aux_weight: float,
    webbased_recon_hard_weight: float,
    webbased_recon_sep_weight: float,
    webbased_recon_sep_margin: float,
    flow_input_dim: int | None,
    packet_input_dim: int | None,
) -> dict[str, object]:
    base_kwargs: dict[str, object] = {
        "hidden_dim": hidden_dim,
        "num_classes": num_classes,
        "branch_mode": branch_mode,
        "flow_input_dim": flow_input_dim,
        "packet_input_dim": packet_input_dim,
    }
    if model_name in {"paper", "edge"}:
        return base_kwargs
    if model_name in {"dual", "dual_edge"}:
        return base_kwargs
    if model_name in {"dual_gate", "dual_gate_edge"}:
        base_kwargs["aux_loss_weight"] = webbased_aux_weight
        return base_kwargs
    if model_name in {"dual_gate_logit", "dual_gate_logit_edge"}:
        base_kwargs.update(
            aux_loss_weight=webbased_aux_weight,
            webbased_aux_weight=webbased_aux_weight,
            webbased_recon_aux_weight=webbased_recon_aux_weight,
            webbased_recon_hard_weight=webbased_recon_hard_weight,
            webbased_recon_sep_weight=webbased_recon_sep_weight,
            webbased_recon_sep_margin=webbased_recon_sep_margin,
        )
        return base_kwargs
    raise ValueError(f"Unsupported model for training: {model_name}")


def _model_kwargs_for_checkpoint(
    *,
    model_name: str,
    hidden_dim: int,
    num_classes: int,
    branch_mode: str,
    flow_input_dim: int | None,
    packet_input_dim: int | None,
    webbased_aux_weight: float,
    webbased_recon_aux_weight: float,
    webbased_recon_hard_weight: float,
    webbased_recon_sep_weight: float,
    webbased_recon_sep_margin: float,
) -> dict[str, object]:
    checkpoint_kwargs: dict[str, object] = {
        "hidden_dim": hidden_dim,
        "num_classes": num_classes,
        "branch_mode": branch_mode,
        "flow_input_dim": flow_input_dim,
        "packet_input_dim": packet_input_dim,
    }
    if model_name in {"dual_gate", "dual_gate_edge", "dual_gate_logit", "dual_gate_logit_edge"}:
        checkpoint_kwargs["aux_loss_weight"] = webbased_aux_weight
    if model_name in {"dual_gate_logit", "dual_gate_logit_edge"}:
        checkpoint_kwargs.update(
            webbased_aux_weight=webbased_aux_weight,
            webbased_recon_aux_weight=webbased_recon_aux_weight,
            webbased_recon_hard_weight=webbased_recon_hard_weight,
            webbased_recon_sep_weight=webbased_recon_sep_weight,
            webbased_recon_sep_margin=webbased_recon_sep_margin,
        )
    return checkpoint_kwargs


@torch.no_grad()
def predict(model: nn.Module, loader, device: torch.device) -> tuple[list[int], list[int]]:
    model.eval()
    all_preds: list[int] = []
    all_targets: list[int] = []
    for batch in loader:
        batch = _move_batch(batch, device)
        logits = model(batch)
        preds = logits.argmax(dim=-1).detach().cpu().tolist()
        targets = batch.y.view(-1).detach().cpu().tolist()
        all_preds.extend(preds)
        all_targets.extend(targets)
    return all_preds, all_targets


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> dict[str, float]:
    all_preds, all_targets = predict(model, loader, device)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)
    acc = accuracy_score(all_targets, all_preds)
    return {
        "accuracy": float(acc),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


def train(
    data_path: str | Path,
    output_dir: str | Path,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    heads: int = 2,
    num_classes: int = 8,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    device: str = "cuda",
    stratify: bool = True,
    balance_train: bool = False,
    train_samples_per_class: int | None = None,
    early_stop_patience: int = 12,
    early_stop_min_delta: float = 0.0,
    model_name: str = "paper",
    branch_mode: str = "dual",
    webbased_weight: float = 1.0,
    webbased_aux_weight: float = 0.25,
    webbased_recon_aux_weight: float = 0.25,
    webbased_recon_hard_weight: float = 2.0,
    webbased_recon_sep_weight: float = 0.2,
    webbased_recon_sep_margin: float = 0.15,
    hard_negative_weight: float = 1.5,
    hard_negative_warmup_epoch: int = 1,
) -> TrainResult:
    if abs(train_ratio - 0.8) > 1e-9:
        warnings.warn(
            "train_ratio is ignored by the paper-aligned split; using the fixed Table 4 class-wise test targets.",
            stacklevel=2,
        )
    device_t = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    graphs = load_graphs(data_path)
    flow_input_dim, packet_input_dim = _node_input_dims(graphs)
    split_indices = split_graph_indices_paper_table4(
        graphs,
        test_ratio=0.2,
        test_cap=4000,
        val_ratio=val_ratio,
        seed=seed,
    )
    loaders = make_loaders(
        graphs,
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        stratify=stratify,
        balance_train=balance_train,
        train_samples_per_class=train_samples_per_class,
        split_indices=split_indices,
    )
    train_loader, val_loader, test_loader = loaders
    train_graph_indices = _graph_index_list(train_loader)
    model = build_model(
        model_name,
        **_model_kwargs_for_training(
            model_name=model_name,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            branch_mode=branch_mode,
            webbased_aux_weight=webbased_aux_weight,
            webbased_recon_aux_weight=webbased_recon_aux_weight,
            webbased_recon_hard_weight=webbased_recon_hard_weight,
            webbased_recon_sep_weight=webbased_recon_sep_weight,
            webbased_recon_sep_margin=webbased_recon_sep_margin,
            flow_input_dim=flow_input_dim,
            packet_input_dim=packet_input_dim,
        ),
    ).to(device_t)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    class_weight = torch.ones(num_classes, dtype=torch.float32, device=device_t)
    if webbased_weight != 1.0 and num_classes > 1:
        class_weight[1] = float(webbased_weight)
    criterion = nn.NLLLoss(weight=class_weight)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    split_path = output_dir / "split.json"
    best_f1 = -1.0
    best_epoch = 0
    patience_left = max(int(early_stop_patience), 0)
    best_metrics: dict[str, float] = {}
    hard_negative_counts: Counter[int] = Counter()
    hard_negative_indices: set[int] = set()

    split_record = build_split_record(
        graphs,
        split_indices,
        data_path=data_path,
        split_strategy="paper_table4_fixed_counts",
        test_ratio=0.2,
        test_cap=4000,
        val_ratio=val_ratio,
        seed=seed,
        target_per_class=train_samples_per_class if balance_train else None,
        test_targets=TABLE4_TEST_TARGETS,
    )
    save_split_record(split_record, split_path)

    print("dataset label counts:", label_counts(graphs))
    print("train label counts:", label_counts(train_loader.dataset))
    print("val label counts:", label_counts(val_loader.dataset))
    print("test label counts:", label_counts(test_loader.dataset))

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        hard_negative_indices = set(hard_negative_counts.keys())
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False):
            batch = _move_batch(batch, device_t)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            if hasattr(model, "loss"):
                sample_weight = None
                if hard_negative_indices:
                    batch_indices = getattr(batch, "graph_index", None)
                    if batch_indices is not None:
                        batch_indices = batch_indices.view(-1).detach().cpu().tolist()
                        weights = torch.ones_like(batch.y.view(-1), dtype=torch.float32, device=device_t)
                        for i, graph_idx in enumerate(batch_indices):
                            if int(graph_idx) in hard_negative_indices:
                                weights[i] = float(hard_negative_weight)
                        sample_weight = weights
                loss = model.loss(logits, batch.y.view(-1), class_weight=class_weight)
                if sample_weight is not None:
                    loss = loss * sample_weight.mean()
            else:
                loss = criterion(logits, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        val_metrics = evaluate(model, val_loader, device_t)
        improved = val_metrics["f1_macro"] > (best_f1 + early_stop_min_delta)
        if improved:
            best_f1 = val_metrics["f1_macro"]
            best_epoch = epoch
            best_metrics = val_metrics
            patience_left = max(int(early_stop_patience), 0)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": model_name,
                    "model_kwargs": _model_kwargs_for_checkpoint(
                        model_name=model_name,
                        hidden_dim=hidden_dim,
                        num_classes=num_classes,
                        branch_mode=branch_mode,
                        flow_input_dim=flow_input_dim,
                        packet_input_dim=packet_input_dim,
                        webbased_aux_weight=webbased_aux_weight,
                        webbased_recon_aux_weight=webbased_recon_aux_weight,
                        webbased_recon_hard_weight=webbased_recon_hard_weight,
                        webbased_recon_sep_weight=webbased_recon_sep_weight,
                        webbased_recon_sep_margin=webbased_recon_sep_margin,
                    ),
                    "metrics": val_metrics,
                },
                best_path,
            )
        if epoch >= hard_negative_warmup_epoch:
            model.eval()
            with torch.no_grad():
                for batch in train_loader:
                    batch = _move_batch(batch, device_t)
                    preds = model(batch).argmax(dim=-1).detach().cpu().tolist()
                    targets = batch.y.view(-1).detach().cpu().tolist()
                    batch_indices = getattr(batch, "graph_index", None)
                    if batch_indices is None:
                        continue
                    batch_indices = batch_indices.view(-1).detach().cpu().tolist()
                    for pred, target, graph_idx in zip(preds, targets, batch_indices):
                        if int(target) == 3 and int(pred) == 1:
                            hard_negative_counts[int(graph_idx)] += 1
                        elif int(graph_idx) in hard_negative_counts and int(pred) == int(target):
                            hard_negative_counts[int(graph_idx)] = max(hard_negative_counts[int(graph_idx)] - 1, 0)
                            if hard_negative_counts[int(graph_idx)] <= 0:
                                del hard_negative_counts[int(graph_idx)]
        print(
            f"epoch={epoch} loss={total_loss / max(len(train_loader), 1):.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1_macro']:.4f}"
        )
        if not improved and early_stop_patience >= 0:
            patience_left -= 1
            if patience_left <= 0:
                print(
                    f"early stopping at epoch {epoch} "
                    f"(best_epoch={best_epoch}, best_val_f1={best_f1:.4f})"
                )
                break

    best_checkpoint = torch.load(best_path, map_location=device_t)
    best_model = build_model_from_checkpoint(best_checkpoint).to(device_t)
    best_model.load_state_dict(best_checkpoint["model_state"])
    test_metrics = evaluate(best_model, test_loader, device_t)
    return TrainResult(
        best_path=best_path,
        metrics={
            "best_val_f1": best_f1,
            "best_epoch": best_epoch,
            **best_metrics,
            **{f"test_{k}": v for k, v in test_metrics.items()},
            "split_path": str(split_path),
            "stopped_epoch": epoch,
        },
    )
