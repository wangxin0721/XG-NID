from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
THIRD_PARTY = ROOT / "third_party" / "GNN4ID"
for path in (SRC, THIRD_PARTY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch
from sklearn.metrics import accuracy_score
from torch_geometric.loader import DataLoader

from xgnid.data import load_graphs
from xgnid.export_test_results import save_test_outputs
from xgnid.model import XGNIDClassifier


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "processed" / "CICIoT2023_processed" / "processed"))
    parser.add_argument("--checkpoint", default=str(ROOT / "outputs" / "GNN4ID" / "model.pth"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "test_exports"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    return parser


def load_checkpoint(path: str | Path, device: torch.device):
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")

    if isinstance(obj, dict) and "model_state" in obj and "model_kwargs" in obj:
        model = XGNIDClassifier(**obj["model_kwargs"])
        model.load_state_dict(obj["model_state"])
        return model.to(device)

    if hasattr(obj, "to") and hasattr(obj, "eval"):
        return obj.to(device)

    raise TypeError(f"Unsupported checkpoint type: {type(obj)!r}")


def load_test_graphs(path: str | Path):
    path = Path(path)
    if path.is_dir():
        test_files = sorted(path.glob("data_test_*.pt"))
        if test_files:
            graphs = []
            for file in test_files:
                graphs.extend(load_graphs(file))
            return graphs
    return load_graphs(path)


def predict(model, loader, device: torch.device):
    model.eval()
    preds: list[int] = []
    labels: list[int] = []
    forward_params = len(inspect.signature(model.forward).parameters)

    for batch in loader:
        batch = batch.to(device)
        with torch.no_grad():
            if forward_params == 1:
                logits = model(batch)
            elif forward_params == 3:
                logits = model(batch.x_dict, batch.edge_index_dict, batch)
            elif forward_params == 4:
                logits = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch)
            else:
                raise RuntimeError(f"Unsupported forward signature with {forward_params} parameters.")

        preds.extend(logits.argmax(dim=-1).detach().cpu().tolist())
        labels.extend(batch.y.view(-1).detach().cpu().tolist())

    return preds, labels


def main() -> int:
    args = build_parser().parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    graphs = load_test_graphs(args.data)
    loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=False)
    model = load_checkpoint(args.checkpoint, device)
    preds, labels = predict(model, loader, device)
    accuracy = accuracy_score(labels, preds)
    save_path = save_test_outputs(accuracy, preds, labels, output_dir=args.output_dir)
    print(f"saved: {save_path}")
    print(f"accuracy: {accuracy:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
