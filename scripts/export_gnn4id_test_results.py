from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
from torch_geometric.loader import DataLoader

from xgnid.export_test_results import save_test_outputs
from xgnid.data import load_graphs
from xgnid.model import XGNIDClassifier
from xgnid.train import predict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/test_exports")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    graphs = load_graphs(args.data)
    loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = XGNIDClassifier(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = model.to(device)
    preds, labels = predict(model, loader, device)
    save_path = save_test_outputs(
        acc=float((torch.tensor(preds) == torch.tensor(labels)).float().mean().item()),
        prediction=preds,
        label=labels,
        output_dir=args.output_dir,
    )
    print(save_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
