from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from torch_geometric.loader import DataLoader

from .data import load_graphs, load_split_record, summarize_graphs, subset_graphs
from .data import label_counts
from .export_test_results import save_test_outputs
from .model import XGNIDClassifier
from .train import predict, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xgnid")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect_p = sub.add_parser("inspect", help="Inspect a graph dataset file or directory")
    inspect_p.add_argument("--data", required=True)

    train_p = sub.add_parser("train", help="Train the graph classifier")
    train_p.add_argument("--data", required=True)
    train_p.add_argument("--output-dir", default="outputs/xgnid")
    train_p.add_argument("--epochs", type=int, default=30)
    train_p.add_argument("--batch-size", type=int, default=64)
    train_p.add_argument("--lr", type=float, default=1e-2)
    train_p.add_argument("--weight-decay", type=float, default=1e-5)
    train_p.add_argument("--hidden-dim", type=int, default=64)
    train_p.add_argument("--heads", type=int, default=2)
    train_p.add_argument("--num-classes", type=int, default=8)
    train_p.add_argument("--train-ratio", type=float, default=0.8)
    train_p.add_argument("--val-ratio", type=float, default=0.1)
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--device", default="cuda")
    train_p.add_argument("--no-stratify", action="store_false", dest="stratify", default=True)
    train_p.add_argument("--balance-train", action="store_true")
    train_p.add_argument("--train-samples-per-class", type=int, default=None)

    eval_p = sub.add_parser("eval", help="Evaluate a checkpoint")
    eval_p.add_argument("--data", required=True)
    eval_p.add_argument("--checkpoint", required=True)
    eval_p.add_argument("--split", default=None, help="Path to split.json saved by train")
    eval_p.add_argument("--batch-size", type=int, default=64)
    eval_p.add_argument("--device", default="cuda")
    eval_p.add_argument("--output-dir", default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "inspect":
        graphs = load_graphs(args.data)
        print(summarize_graphs(graphs))
        print("label_counts:", label_counts(graphs))
        if graphs:
            g = graphs[0]
            print("node_types:", g.node_types)
            print("edge_types:", g.edge_types)
            for node_type in g.node_types:
                x = g[node_type].x
                print(f"{node_type}.x:", tuple(x.shape))
            for edge_type in g.edge_types:
                edge_store = g[edge_type]
                print(f"{edge_type}.edge_index:", tuple(edge_store.edge_index.shape))
                if getattr(edge_store, "edge_attr", None) is not None:
                    print(f"{edge_type}.edge_attr:", tuple(edge_store.edge_attr.shape))
        return 0

    if args.command == "train":
        result = train(
            data_path=args.data,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            hidden_dim=args.hidden_dim,
            heads=args.heads,
            num_classes=args.num_classes,
            train_ratio=0.8,
            val_ratio=args.val_ratio,
            seed=args.seed,
            device=args.device,
            stratify=args.stratify,
            balance_train=args.balance_train,
            train_samples_per_class=args.train_samples_per_class,
        )
        print(result.best_path)
        print(result.metrics)
        return 0

    if args.command == "eval":
        graphs = load_graphs(args.data)
        split_path = args.split
        if split_path is None:
            candidate = Path(args.checkpoint).resolve().parent / "split.json"
            if candidate.exists():
                split_path = str(candidate)
        if split_path is not None:
            split_record = load_split_record(split_path)
            test_indices = split_record["split_indices"]["test"]
            graphs = subset_graphs(graphs, test_indices)
        loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=False)
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model = XGNIDClassifier(**checkpoint["model_kwargs"])
        model.load_state_dict(checkpoint["model_state"])
        device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
        model = model.to(device)
        preds, labels = predict(model, loader, device)
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
        metrics = {
            "accuracy": float(accuracy),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "f1_macro": float(f1),
        }
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = str(Path(args.checkpoint).resolve().parent / "test_exports")
        save_test_outputs(accuracy, preds, labels, output_dir=output_dir)
        print(metrics)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
