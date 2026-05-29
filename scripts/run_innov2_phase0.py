from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

LOCKED_GRAPH_DIR = ROOT / "data" / "processed" / "CICIoT2023_processed" / "processed_innov1_r25_graphs"
LOCKED_OUTPUT_ROOT = ROOT / "outputs" / "innov2"


def _resolve_run_dir(run_name: str) -> Path:
    return LOCKED_OUTPUT_ROOT / run_name


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
    train_p.add_argument(
        "--model",
        default="dual",
        choices=[
            "paper",
            "edge",
            "dual",
            "dual_edge",
            "dual_gate",
            "dual_gate_edge",
            "dual_gate_logit",
            "dual_gate_logit_edge",
        ],
    )
    train_p.add_argument("--branch-mode", default="dual", choices=["flow", "packet", "dual"])
    train_p.add_argument("--webbased-weight", type=float, default=1.0)
    train_p.add_argument("--run-name", default="dual_v1")

    eval_p = sub.add_parser("eval", help="Evaluate innov2 with locked input/output paths")
    eval_p.add_argument("--batch-size", type=int, default=64)
    eval_p.add_argument("--device", default="cuda")
    eval_p.add_argument("--run-name", default="dual_v1")

    paths_p = sub.add_parser("paths", help="Print the locked innov2 paths")
    paths_p.add_argument("--run-name", default="dual_v1")

    return parser


def _run_train(args: argparse.Namespace) -> int:
    from xgnid.cli import main as xgnid_main

    output_dir = _resolve_run_dir(args.run_name)
    best_path = output_dir / "best.pt"
    split_path = output_dir / "split.json"

    print(f"[innov2] locked input : {LOCKED_GRAPH_DIR}")
    print(f"[innov2] locked output: {output_dir}")
    cli_args = [
        "train",
        "--data",
        str(LOCKED_GRAPH_DIR),
        "--output-dir",
        str(output_dir),
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
        "--webbased-weight",
        str(args.webbased_weight),
    ]
    if args.stratify is False:
        cli_args.append("--no-stratify")
    if args.balance_train:
        cli_args.append("--balance-train")
    return xgnid_main(cli_args)


def _run_eval(args: argparse.Namespace) -> int:
    from xgnid.cli import main as xgnid_main

    output_dir = _resolve_run_dir(args.run_name)
    best_path = output_dir / "best.pt"
    split_path = output_dir / "split.json"
    test_exports = output_dir / "test_exports"

    print(f"[innov2] locked input : {LOCKED_GRAPH_DIR}")
    print(f"[innov2] locked output: {output_dir}")
    cli_args = [
        "eval",
        "--data",
        str(LOCKED_GRAPH_DIR),
        "--checkpoint",
        str(best_path),
        "--split",
        str(split_path),
        "--batch-size",
        str(args.batch_size),
        "--device",
        str(args.device),
        "--output-dir",
        str(test_exports),
    ]
    return xgnid_main(cli_args)


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "train":
        return _run_train(args)
    if args.command == "eval":
        return _run_eval(args)
    if args.command == "paths":
        output_dir = _resolve_run_dir(args.run_name)
        print(f"input={LOCKED_GRAPH_DIR}")
        print(f"output={output_dir}")
        print(f"best={output_dir / 'best.pt'}")
        print(f"split={output_dir / 'split.json'}")
        print(f"exports={output_dir / 'test_exports'}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
