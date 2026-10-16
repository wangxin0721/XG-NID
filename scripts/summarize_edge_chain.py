"""Check paired configs and summarize actual completed pilot results."""
import argparse
import json
from pathlib import Path


def summarize(root):
    configs, metrics = {}, {}
    for name in ("E0", "E1", "E2", "E3"):
        folder = root / name
        status = json.loads((folder / "status.json").read_text())
        if status != {"status": "completed", "exit_code": 0, "scope": "pilot"}:
            raise ValueError(f"Incomplete experiment: {name}")
        configs[name] = json.loads((folder / "config.json").read_text())
        metrics[name] = json.loads((folder / "metrics.json").read_text())
    comparisons = {}
    for left, right, allowed in (("E0", "E1", {"experiment", "use_edge_attr"}),
                                  ("E2", "E3", {"experiment", "fusion"})):
        differences = {key for key in configs[left].keys() | configs[right].keys()
                       if configs[left].get(key) != configs[right].get(key)}
        if differences != allowed:
            raise ValueError(f"Uncontrolled config difference: {left}/{right}: {differences}")
        comparisons[f"{left}_minus_{right}"] = {
            "config_differences": sorted(differences),
            "macro_f1_delta": metrics[left]["macro_f1"] - metrics[right]["macro_f1"],
            "per_class_f1_delta": {label: values["f1-score"] - metrics[right]["per_class"][label]["f1-score"]
                                   for label, values in metrics[left]["per_class"].items()
                                   if isinstance(values, dict) and label not in ("macro avg", "weighted avg")}}
    with (root / "comparison.json").open("x", encoding="utf-8") as stream:
        json.dump(comparisons, stream, indent=2, ensure_ascii=False)
    lines = ["仅为256图、单种子、3 epoch试运行；不能据此判断模型优劣。", "",
             "| 模式 | Accuracy | Macro-F1 | Weighted-F1 | 最佳epoch | 训练秒 | 测试推理秒 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, m in metrics.items():
        lines.append(f"| {name} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['weighted_f1']:.4f} | {m['best_epoch']} | {m['training_seconds']:.3f} | {m['inference_seconds']:.3f} |")
    with (root / "comparison.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    summarize(parser.parse_args().root)
