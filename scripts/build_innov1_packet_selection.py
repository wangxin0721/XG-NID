from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from xgnid.packet_selection import PacketSelectionConfig, process_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="build-innov1-packet-selection")
    parser.add_argument(
        "--train-csv",
        default=str(ROOT / "data" / "processed" / "CICIoT2023_processed" / "df_class_8_train.csv"),
    )
    parser.add_argument(
        "--test-csv",
        default=str(ROOT / "data" / "processed" / "CICIoT2023_processed" / "df_class_8_test.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "data" / "processed" / "CICIoT2023_processed" / "processed_innov1"),
    )
    parser.add_argument("--selection-ratio", type=float, default=0.35)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--strategy", choices=["topk", "threshold"], default="topk")
    parser.add_argument("--min-packets", type=int, default=1)
    parser.add_argument("--max-packets", type=int, default=20)
    parser.add_argument("--no-keep-original-order", action="store_false", dest="keep_original_order", default=True)
    parser.add_argument("--chunksize", type=int, default=2048)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    config = PacketSelectionConfig(
        strategy=args.strategy,
        selection_ratio=args.selection_ratio,
        score_threshold=args.score_threshold,
        min_packets=args.min_packets,
        max_packets=args.max_packets,
        keep_original_order=args.keep_original_order,
    )

    train_out = out_root / "df_class_8_train_innov1.csv"
    test_out = out_root / "df_class_8_test_innov1.csv"

    train_summary = process_csv(args.train_csv, train_out, config, chunksize=args.chunksize)
    test_summary = process_csv(args.test_csv, test_out, config, chunksize=args.chunksize)

    summary = {
        "config": train_summary["config"],
        "train": train_summary,
        "test": test_summary,
    }
    (out_root / "packet_selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
