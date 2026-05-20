from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from xgnid.export_test_results import DEFAULT_CLASS_NAMES, save_test_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="table5-flow-only")
    parser.add_argument(
        "--train-csv",
        default=str(ROOT / "data" / "processed" / "CICIoT2023_processed" / "df_class_8_train.csv"),
    )
    parser.add_argument(
        "--test-csv",
        default=str(ROOT / "data" / "processed" / "CICIoT2023_processed" / "df_class_8_test.csv"),
    )
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "table5_flow_only"))
    parser.add_argument(
        "--models",
        nargs="+",
        default=["RF", "LR", "AdaBoost", "DNN", "KNN"],
        help="Supported: RF, LR, AdaBoost, DNN, KNN",
    )
    parser.add_argument(
        "--feature-mode",
        choices=["paper_flow", "flow82"],
        default="paper_flow",
        help=(
            "paper_flow keeps only the base flow statistics before packet/temporal columns; "
            "flow82 keeps the current 82-dim non-packet feature set used by the graph pipeline."
        ),
    )
    parser.add_argument("--rf-trees", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def load_flow_only(csv_path: str | Path, feature_mode: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    if "Label" not in df.columns:
        raise ValueError(f"Missing Label column in {csv_path}")

    if feature_mode == "paper_flow":
        cutoff = df.columns.get_loc("udps.payload_data")
        feature_cols = list(df.columns[:cutoff])
    elif feature_mode == "flow82":
        feature_cols = [c for c in df.columns if not c.startswith("udps.") and c != "Label"]
    else:
        raise ValueError(f"Unsupported feature mode: {feature_mode}")

    X = df[feature_cols].fillna(0)
    y = df["Label"].astype(int)
    return X, y


def build_models(seed: int, rf_trees: int):
    return {
        "RF": RandomForestClassifier(n_estimators=rf_trees, random_state=seed, n_jobs=-1),
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

    X_train, y_train = load_flow_only(args.train_csv, args.feature_mode)
    X_test, y_test = load_flow_only(args.test_csv, args.feature_mode)

    available = build_models(args.seed, args.rf_trees)
    selected = []
    for name in args.models:
        if name not in available:
            raise ValueError(f"Unsupported model: {name}. Supported: {', '.join(available)}")
        selected.append((name, available[name]))

    print(f"[flow-only] feature_mode={args.feature_mode} num_features={X_train.shape[1]}")
    if args.feature_mode == "paper_flow":
        print("[flow-only] using base flow statistics only; packet and temporal add-on columns are excluded.")

    summary_rows = []
    for name, model in selected:
        print(f"[flow-only] training {name} ...")
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
                    "feature_mode": args.feature_mode,
                    "num_train_rows": int(len(X_train)),
                    "num_test_rows": int(len(X_test)),
                    "num_features": int(X_train.shape[1]),
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
            }
        )
        print(f"[flow-only] {name}: acc={acc:.4f} f1={f1:.4f}")

    summary = pd.DataFrame(summary_rows).sort_values(by="f1_macro", ascending=False)
    summary_path = out_root / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"saved summary: {summary_path}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
