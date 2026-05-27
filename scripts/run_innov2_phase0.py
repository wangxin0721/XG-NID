from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

LOCKED_GRAPH_DIR = ROOT / "data" / "processed" / "CICIoT2023_processed" / "processed_innov1_r25_graphs"
LOCKED_OUTPUT_DIR = ROOT / "outputs" / "innov2"
LOCKED_BEST_PATH = LOCKED_OUTPUT_DIR / "best.pt"
LOCKED_SPLIT_PATH = LOCKED_OUTPUT_DIR / "split.json"
LOCKED_TEST_EXPORTS = LOCKED_OUTPUT_DIR / "test_exports"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run-innov2-phase0")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train innov2 with locked input/output paths")
    train_p.add_argument("--epochs", type=int, default=100)
    train_p.add_argument("--batch-size", type=int, default=64)
    train_p.add_argument("--lr", type=float, default=0.01)
    train_p.add_argument("--hidden-dim", type=int, default=64)
    train_p.add_argument("--heads", type=int, default=2)
    train_p.add_argument("--num-classes", type=int, default=8)
    train_p.add_argument("--val-ratio", type=float, default=0.1)
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--device", default="cuda")
    train_p.add_argument("--no-stratify", action="store_false", dest="stratify", default=True)
    train_p.add_argument("--balance-train", action="store_true")
    train_p.add_argument("--train-samples-per-class", type=int, default=20000)
    train_p.add_argument("--early-stop-patience", type=int, default=12)
    train_p.add_argument("--early-stop-min-delta", type=float, default=0.0)
    train_p.add_argument("--model", default="dual", choices=["paper", "edge", "dual", "dual_edge"])
    train_p.add_argument("--branch-mode", default="dual", choices=["flow", "packet", "dual"])

    eval_p = sub.add_parser("eval", help="Evaluate innov2 with locked input/output paths")
    eval_p.add_argument("--batch-size", type=int, default=64)
    eval_p.add_argument("--device", default="cuda")

    paths_p = sub.add_parser("paths", help="Print the locked innov2 paths")

    return parser


def _run_train(args: argparse.Namespace) -> int:
    from xgnid.cli import main as xgnid_main

    print(f"[innov2] locked input : {LOCKED_GRAPH_DIR}")
    print(f"[innov2] locked output: {LOCKED_OUTPUT_DIR}")
    cli_args = [
        "train",
        "--data",
        str(LOCKED_GRAPH_DIR),
        "--output-dir",
        str(LOCKED_OUTPUT_DIR),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--hidden-dim",
        str(args.hidden_dim),
        "--heads",
        str(args.heads),
        "--num-classes",
        str(args.num_classes),
        "--val-ratio",
        str(args.val_ratio),
        "--seed",
        str(args.seed),
        "--device",
        str(args.device),
        "--train-samples-per-class",
        str(args.train_samples_per_class),
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--early-stop-min-delta",
        str(args.early_stop_min_delta),
        "--model",
        str(args.model),
        "--branch-mode",
        str(args.branch_mode),
    ]
    if args.stratify is False:
        cli_args.append("--no-stratify")
    if args.balance_train:
        cli_args.append("--balance-train")
    return xgnid_main(cli_args)


def _run_eval(args: argparse.Namespace) -> int:
    from xgnid.cli import main as xgnid_main

    print(f"[innov2] locked input : {LOCKED_GRAPH_DIR}")
    print(f"[innov2] locked output: {LOCKED_OUTPUT_DIR}")
    cli_args = [
        "eval",
        "--data",
        str(LOCKED_GRAPH_DIR),
        "--checkpoint",
        str(LOCKED_BEST_PATH),
        "--split",
        str(LOCKED_SPLIT_PATH),
        "--batch-size",
        str(args.batch_size),
        "--device",
        str(args.device),
        "--output-dir",
        str(LOCKED_TEST_EXPORTS),
    ]
    return xgnid_main(cli_args)


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "train":
        return _run_train(args)
    if args.command == "eval":
        return _run_eval(args)
    if args.command == "paths":
        print(f"input={LOCKED_GRAPH_DIR}")
        print(f"output={LOCKED_OUTPUT_DIR}")
        print(f"best={LOCKED_BEST_PATH}")
        print(f"split={LOCKED_SPLIT_PATH}")
        print(f"exports={LOCKED_TEST_EXPORTS}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
