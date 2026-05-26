from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from xgnid.graph_building import build_graphs_from_csv, save_graphs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="build-innov1-graphs")
    parser.add_argument(
        "--input-dir",
        default=str(ROOT / "data" / "processed" / "CICIoT2023_processed" / "processed_innov1"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "data" / "processed" / "CICIoT2023_processed" / "processed_innov1_graphs"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_csv = input_dir / "df_class_8_train_innov1.csv"
    test_csv = input_dir / "df_class_8_test_innov1.csv"

    train_graphs = build_graphs_from_csv(train_csv)
    test_graphs = build_graphs_from_csv(test_csv)

    train_saved = save_graphs(train_graphs, output_dir / "train", "data")
    test_saved = save_graphs(test_graphs, output_dir / "test", "data_test")

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "train_graphs": len(train_saved),
        "test_graphs": len(test_saved),
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
    }
    (output_dir / "graph_build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
