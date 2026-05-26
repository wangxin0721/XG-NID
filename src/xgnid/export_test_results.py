from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


DEFAULT_CLASS_NAMES = [
    "Benign",
    "WebBased",
    "Spoofing",
    "Recon",
    "Mirai",
    "Dos",
    "DDos",
    "BruteForce",
]


def save_test_outputs(
    acc: float,
    prediction: Sequence[int] | np.ndarray,
    label: Sequence[int] | np.ndarray,
    output_dir: str | Path = "outputs/GNN4ID/test_exports",
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction = np.asarray(prediction).reshape(-1)
    label = np.asarray(label).reshape(-1)
    cm = confusion_matrix(label, prediction)
    report = classification_report(
        label,
        prediction,
        labels=list(range(len(class_names))),
        target_names=list(class_names),
        digits=4,
        zero_division=0,
    )

    pd.DataFrame(
        {
            "label": label,
            "prediction": prediction,
            "label_name": [class_names[int(i)] for i in label],
            "prediction_name": [class_names[int(i)] for i in prediction],
        }
    ).to_csv(output_dir / "predictions.csv", index=False)

    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(output_dir / "confusion_matrix.csv")
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    (output_dir / "metrics.json").write_text(
        json.dumps({"accuracy": float(acc)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    return output_dir


def save_test_summary(
    metrics: dict[str, float],
    prediction: Sequence[int] | np.ndarray,
    label: Sequence[int] | np.ndarray,
    output_dir: str | Path,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    acc = float(metrics.get("accuracy", 0.0))
    save_test_outputs(
        acc,
        prediction,
        label,
        output_dir=output_dir,
        class_names=class_names,
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_dir
