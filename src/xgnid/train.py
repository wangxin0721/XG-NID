from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import warnings

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.optim import Adam
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
from .model import XGNIDClassifier


@dataclass
class TrainResult:
    best_path: Path
    metrics: dict[str, float | str]


def _move_batch(batch, device: torch.device):
    return batch.to(device)


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
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-2,
    weight_decay: float = 1e-5,
    hidden_dim: int = 64,
    heads: int = 2,
    num_classes: int = 8,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    device: str = "cuda",
    stratify: bool = True,
    balance_train: bool = False,
    train_samples_per_class: int | None = None,
) -> TrainResult:
    if abs(train_ratio - 0.8) > 1e-9:
        warnings.warn(
            "train_ratio is ignored by the paper-aligned split; using the fixed Table 4 class-wise test targets.",
            stacklevel=2,
        )
    device_t = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    graphs = load_graphs(data_path)
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
    model = XGNIDClassifier(hidden_dim=hidden_dim, heads=heads, num_classes=num_classes).to(device_t)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    split_path = output_dir / "split.json"
    best_f1 = -1.0
    best_metrics: dict[str, float] = {}

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
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False):
            batch = _move_batch(batch, device_t)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        val_metrics = evaluate(model, val_loader, device_t)
        if val_metrics["f1_macro"] >= best_f1:
            best_f1 = val_metrics["f1_macro"]
            best_metrics = val_metrics
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_kwargs": {
                        "hidden_dim": hidden_dim,
                        "heads": heads,
                        "num_classes": num_classes,
                    },
                    "metrics": val_metrics,
                },
                best_path,
            )
        print(
            f"epoch={epoch} loss={total_loss / max(len(train_loader), 1):.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1_macro']:.4f}"
        )

    best_checkpoint = torch.load(best_path, map_location=device_t)
    best_model = XGNIDClassifier(**best_checkpoint["model_kwargs"]).to(device_t)
    best_model.load_state_dict(best_checkpoint["model_state"])
    test_metrics = evaluate(best_model, test_loader, device_t)
    return TrainResult(
        best_path=best_path,
        metrics={
            "best_val_f1": best_f1,
            **best_metrics,
            **{f"test_{k}": v for k, v in test_metrics.items()},
            "split_path": str(split_path),
        },
    )
