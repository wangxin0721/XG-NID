from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from xgnid.export_test_results import DEFAULT_CLASS_NAMES, save_test_outputs


PACKET_COLUMNS = [
    "udps.payload_data",
    "udps.delta_time",
    "udps.packet_direction",
    "udps.ip_size",
    "udps.transport_size",
    "udps.payload_size",
    "udps.syn",
    "udps.cwr",
    "udps.ece",
    "udps.urg",
    "udps.ack",
    "udps.psh",
    "udps.rst",
    "udps.fin",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="table6-packet-only")
    parser.add_argument(
        "--train-csv",
        default=str(ROOT / "data" / "processed" / "CICIoT2023_processed" / "df_class_8_train.csv"),
    )
    parser.add_argument(
        "--test-csv",
        default=str(ROOT / "data" / "processed" / "CICIoT2023_processed" / "df_class_8_test.csv"),
    )
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "table6_packet_only"))
    parser.add_argument(
        "--models",
        nargs="+",
        default=["RF", "LR", "AdaBoost", "DNN", "KNN"],
        help="Supported: RF, LR, AdaBoost, DNN, KNN",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-packets", type=int, default=20)
    parser.add_argument("--no-export-features", action="store_false", dest="export_features", default=True)
    return parser


def _parse_list_cell(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        text = text.strip("[]")
        if not text:
            return []
        return [item.strip().strip("'\"") for item in text.split(",")]
    if isinstance(parsed, (list, tuple)):
        return [str(item) for item in parsed]
    return [str(parsed)]


def _packet_summary_from_payload(payload_hex: str) -> tuple[float, float, float, float, float]:
    if not payload_hex or payload_hex == "00":
        return 0.0, 0.0, 0.0, 0.0, 0.0

    try:
        payload_bytes = bytes.fromhex(payload_hex)
    except ValueError:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    byte_arr = np.frombuffer(payload_bytes, dtype=np.uint8).astype(np.float32)
    if byte_arr.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    return (
        float(byte_arr.mean()),
        float(byte_arr.std(ddof=0)),
        float(byte_arr.min()),
        float(byte_arr.max()),
        float(np.count_nonzero(byte_arr) / byte_arr.size),
    )


def _safe_float_list(values: list[str]) -> list[float]:
    out: list[float] = []
    for value in values:
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            out.append(0.0)
    return out


def _packet_features_for_row(row: pd.Series, max_packets: int) -> np.ndarray:
    payloads = _parse_list_cell(row["udps.payload_data"])
    delta_time = _parse_list_cell(row["udps.delta_time"])
    direction = _parse_list_cell(row["udps.packet_direction"])
    ip_size = _parse_list_cell(row["udps.ip_size"])
    transport_size = _parse_list_cell(row["udps.transport_size"])
    payload_size = _parse_list_cell(row["udps.payload_size"])
    syn = _parse_list_cell(row["udps.syn"])
    cwr = _parse_list_cell(row["udps.cwr"])
    ece = _parse_list_cell(row["udps.ece"])
    urg = _parse_list_cell(row["udps.urg"])
    ack = _parse_list_cell(row["udps.ack"])
    psh = _parse_list_cell(row["udps.psh"])
    rst = _parse_list_cell(row["udps.rst"])
    fin = _parse_list_cell(row["udps.fin"])

    per_packet = []
    packet_count = min(len(payloads), max_packets)
    for idx in range(packet_count):
        payload_mean, payload_std, payload_min, payload_max, payload_nz = _packet_summary_from_payload(payloads[idx])
        packet_vector = [
            float(direction[idx]) if idx < len(direction) and direction[idx] != "" else 0.0,
            float(ip_size[idx]) if idx < len(ip_size) and ip_size[idx] != "" else 0.0,
            float(transport_size[idx]) if idx < len(transport_size) and transport_size[idx] != "" else 0.0,
            float(payload_size[idx]) if idx < len(payload_size) and payload_size[idx] != "" else 0.0,
            float(delta_time[idx]) if idx < len(delta_time) and delta_time[idx] != "" else 0.0,
            float(syn[idx]) if idx < len(syn) and syn[idx] != "" else 0.0,
            float(cwr[idx]) if idx < len(cwr) and cwr[idx] != "" else 0.0,
            float(ece[idx]) if idx < len(ece) and ece[idx] != "" else 0.0,
            float(urg[idx]) if idx < len(urg) and urg[idx] != "" else 0.0,
            float(ack[idx]) if idx < len(ack) and ack[idx] != "" else 0.0,
            float(psh[idx]) if idx < len(psh) and psh[idx] != "" else 0.0,
            float(rst[idx]) if idx < len(rst) and rst[idx] != "" else 0.0,
            float(fin[idx]) if idx < len(fin) and fin[idx] != "" else 0.0,
            payload_mean,
            payload_std,
            payload_min,
            payload_max,
            payload_nz,
        ]
        per_packet.extend(packet_vector)

    feature_dim = 19
    expected_len = max_packets * feature_dim
    if len(per_packet) < expected_len:
        per_packet.extend([0.0] * (expected_len - len(per_packet)))

    payload_size_values = _safe_float_list(payload_size)
    per_packet.extend(
        [
            float(len(payloads)),
            float(max(0, len(payloads) - max_packets)),
            float(np.mean(payload_size_values)) if payload_size_values else 0.0,
        ]
    )
    return np.asarray(per_packet, dtype=np.float32)


def load_packet_only(csv_path: str | Path, max_packets: int) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    if "Label" not in df.columns:
        raise ValueError(f"Missing Label column in {csv_path}")
    missing = [col for col in PACKET_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing packet columns in {csv_path}: {missing}")

    features = np.vstack([_packet_features_for_row(row, max_packets) for _, row in df.iterrows()])
    feature_cols = [f"pkt_{i:03d}" for i in range(features.shape[1])]
    X = pd.DataFrame(features, columns=feature_cols)
    y = df["Label"].astype(int)
    return X, y


def build_models(seed: int):
    return {
        "RF": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
        "LR": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, n_jobs=-1, random_state=seed),
        ),
        "AdaBoost": AdaBoostClassifier(random_state=seed),
        "DNN": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size="auto",
                learning_rate_init=1e-3,
                max_iter=200,
                random_state=seed,
                early_stopping=True,
                n_iter_no_change=10,
            ),
        ),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    }


def main() -> int:
    args = build_parser().parse_args()
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_packet_only(args.train_csv, args.max_packets)
    X_test, y_test = load_packet_only(args.test_csv, args.max_packets)

    (out_root / "features").mkdir(parents=True, exist_ok=True)
    if args.export_features:
        X_train.assign(Label=y_train).to_csv(out_root / "features" / "train_packet_features.csv", index=False)
        X_test.assign(Label=y_test).to_csv(out_root / "features" / "test_packet_features.csv", index=False)

    available = build_models(args.seed)
    selected = []
    for name in args.models:
        if name not in available:
            raise ValueError(f"Unsupported model: {name}. Supported: {', '.join(available)}")
        selected.append((name, available[name]))

    print(f"[packet-only] max_packets={args.max_packets} num_features={X_train.shape[1]}")

    summary_rows = []
    for name, model in selected:
        print(f"[packet-only] training {name} ...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test,
            preds,
            average="macro",
            zero_division=0,
        )

        model_dir = out_root / name
        model_dir.mkdir(parents=True, exist_ok=True)
        save_test_outputs(acc, preds, y_test, output_dir=model_dir, class_names=DEFAULT_CLASS_NAMES)

        (model_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "accuracy": float(acc),
                    "precision_macro": float(precision),
                    "recall_macro": float(recall),
                    "f1_macro": float(f1),
                    "num_train_rows": int(len(X_train)),
                    "num_test_rows": int(len(X_test)),
                    "num_features": int(X_train.shape[1]),
                    "max_packets": int(args.max_packets),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        summary_rows.append(
            {
                "model": name,
                "accuracy": float(acc),
                "precision_macro": float(precision),
                "recall_macro": float(recall),
                "f1_macro": float(f1),
                "num_features": int(X_train.shape[1]),
                "max_packets": int(args.max_packets),
            }
        )
        print(f"[packet-only] {name}: acc={acc:.4f} f1={f1:.4f}")

    summary = pd.DataFrame(summary_rows).sort_values(by="f1_macro", ascending=False)
    summary_path = out_root / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"saved summary: {summary_path}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
