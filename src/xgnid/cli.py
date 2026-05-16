from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from .data import load_graphs, summarize_graphs
from .model import XGNIDClassifier
from .train import evaluate, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xgnid")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect_p = sub.add_parser("inspect", help="Inspect a graph dataset file or directory")
    inspect_p.add_argument("--data", required=True)

    train_p = sub.add_parser("train", help="Train the graph classifier")
    train_p.add_argument("--data", required=True)
    train_p.add_argument("--output-dir", default="outputs/xgnid")
    train_p.add_argument("--epochs", type=int, default=30)
    train_p.add_argument("--batch-size", type=int, default=16)
    train_p.add_argument("--lr", type=float, default=1e-3)
    train_p.add_argument("--hidden-dim", type=int, default=128)
    train_p.add_argument("--heads", type=int, default=2)
    train_p.add_argument("--num-classes", type=int, default=8)
    train_p.add_argument("--train-ratio", type=float, default=0.8)
    train_p.add_argument("--val-ratio", type=float, default=0.1)
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--device", default="cuda")

    eval_p = sub.add_parser("eval", help="Evaluate a checkpoint")
    eval_p.add_argument("--data", required=True)
    eval_p.add_argument("--checkpoint", required=True)
    eval_p.add_argument("--batch-size", type=int, default=16)
    eval_p.add_argument("--device", default="cuda")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "inspect":
        graphs = load_graphs(args.data)
        print(summarize_graphs(graphs))
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
            hidden_dim=args.hidden_dim,
            heads=args.heads,
            num_classes=args.num_classes,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            device=args.device,
        )
        print(result.best_path)
        print(result.metrics)
        return 0

    if args.command == "eval":
        graphs = load_graphs(args.data)
        loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=False)
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model = XGNIDClassifier(**checkpoint["model_kwargs"])
        model.load_state_dict(checkpoint["model_state"])
        device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
        model = model.to(device)
        metrics = evaluate(model, loader, device)
        print(metrics)
        return 0

    return 1
