"""Analyze existing validation JSONL caches using only the standard library.

No ML imports, model/checkpoint/graph loads, new inference, training or test data.
Separates wrong branch preference from correct preference with wrong fused
classification. Row-level exports remain in ignored outputs, never in Git.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
from itertools import zip_longest
import json
import math
from pathlib import Path
import sys
import traceback

from e3_gate_study import check_diagnostic, identity, inside, matrix_metrics, read, require, sha, write

LABELS = ["Benign", "WebBased", "Spoofing", "Recon", "Mirai", "Dos", "DDos", "BruteForce"]


def prediction(probabilities):
    require(len(probabilities) == 8 and all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
            and abs(sum(probabilities) - 1) < 1e-5, "Invalid cached probability vector")
    return max(range(8), key=lambda k: probabilities[k])


def failure_category(dynamic_correct, flow_correct, packet_correct, weights):
    if dynamic_correct:
        return "recovered_when_both_single_interventions_wrong" if not (flow_correct or packet_correct) else "fusion_correct"
    if flow_correct and packet_correct:
        return "both_single_interventions_correct_but_fusion_wrong"
    if not (flow_correct or packet_correct):
        return "both_single_interventions_wrong_and_fusion_wrong"
    if weights[0] == weights[1]:
        return "tied_weights_with_one_correct_branch_and_fusion_wrong"
    selected_correct = flow_correct if weights[0] > weights[1] else packet_correct
    return "correct_branch_favored_but_fusion_wrong" if selected_correct else "wrong_branch_favored_and_fusion_wrong"


def failure_bounds(total, favored_correct, fused_correct):
    require(0 <= favored_correct <= total and 0 <= fused_correct <= total, "Invalid group counts")
    return {"minimum": max(0, favored_correct - fused_correct),
            "maximum": min(favored_correct, total - fused_correct)}


def norm(vector):
    require(all(math.isfinite(x) for x in vector), "Nonfinite cached representation")
    return math.sqrt(sum(x * x for x in vector))


def measurements(row):
    wf, wp = row["final_weights"]
    zf, zp = norm(row["z_flow"]), norm(row["z_packet"])
    flow_term, packet_term = wf * zf, wp * zp
    denominator = flow_term + packet_term
    values = {"final_flow_weight": wf, "raw_flow_weight": row["raw_gate"][0],
              "z_flow_norm": zf, "z_packet_norm": zp, "h_flow_norm": norm(row["h_flow"]),
              "h_packet_norm": norm(row["h_packet"]), "weighted_z_flow_norm": flow_term,
              "weighted_z_packet_norm": packet_term,
              "flow_fraction_of_weighted_norm_sum": flow_term / denominator if denominator else 0.,
              "z_cosine": row["z_cosine"], "flow_proxy": row["flow_proxy"], "packet_proxy": row["packet_proxy"],
              "flow_margin": row["flow_margin"], "packet_margin": row["packet_margin"]}
    label = row["label"]
    for key in ("flow_only", "packet_only", "equal", "dynamic", "pure_gate"):
        values[key + "_true_class_probability"] = row["probabilities"][key][label]
    return values


def summarize_values(rows):
    if not rows:
        return {"samples": 0}
    result = {"samples": len(rows), "statistics": {}}
    for key in rows[0]:
        values = sorted(row[key] for row in rows)
        result["statistics"][key] = {"mean": sum(values) / len(values),
            "quantiles": {str(q): values[round(q * (len(values) - 1))] for q in (0, .25, .5, .75, 1)}}
    return result


def analyze(report_path, output):
    report = read(report_path)
    require(report.get("schema") == "xgnid.e3-gate-diagnostic.v1", "Use the original server report.json")
    check_diagnostic(report, report["reference"])
    require(report["models"]["E2"]["ordered_validation_identity_sha256"] ==
            report["models"]["E3"]["ordered_validation_identity_sha256"], "Paired identities differ")
    caches = {name: report_path.parent / name / "validation_cache.jsonl.gz" for name in ("E2", "E3")}
    for name, path in caches.items():
        require(path.is_file() and sha(path) == report["models"][name]["cache_sha256"], f"Missing/changed {name} cache")
    require(not output.exists(), "Output already exists; inspect it, do not overwrite")
    output.mkdir()
    write(output / "status.json", {"status": "running", "mode": "cache_analysis"})
    try:
        matrices = {name: {key: [[0] * 8 for _ in range(8)] for key in
                          ("dynamic", "flow_only", "packet_only", "equal") + (("pure_gate",) if name == "E3" else ())}
                    for name in ("E2", "E3")}
        categories = {label: Counter() for label in LABELS}
        paired = {label: Counter() for label in LABELS}
        web_rows = {"outcome_categories": {}, "branch_correctness_groups": {}}
        identities = []
        csv_path = output / "validation_fusion_failures.csv"
        fields = ["sample_id", "label", "category", "E2_prediction", "E3_prediction", "flow_prediction", "packet_prediction",
                  "final_flow_weight", "raw_flow_weight", "z_flow_norm", "z_packet_norm", "h_flow_norm", "h_packet_norm",
                  "weighted_z_flow_norm", "weighted_z_packet_norm", "flow_fraction_of_weighted_norm_sum", "z_cosine",
                  "flow_proxy", "packet_proxy", "flow_margin", "packet_margin", "flow_only_true_class_probability",
                  "packet_only_true_class_probability", "equal_true_class_probability", "dynamic_true_class_probability",
                  "pure_gate_true_class_probability"]
        with gzip.open(caches["E2"], "rt", encoding="utf-8") as left, gzip.open(caches["E3"], "rt", encoding="utf-8") as right, \
                csv_path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for line2, line3 in zip_longest(left, right):
                require(line2 is not None and line3 is not None, "Cache row counts differ")
                rows = {"E2": json.loads(line2), "E3": json.loads(line3)}
                row2, row3 = rows["E2"], rows["E3"]
                require((row2["sample_id"], row2["label"], row2["node_hash"]) ==
                        (row3["sample_id"], row3["label"], row3["node_hash"]), "Cache identity/label/group order differs")
                label = row3["label"]
                require(isinstance(label, int) and 0 <= label < 8, "Invalid validation label")
                identities.append((row3["sample_id"], label))
                predicted = {}
                for name in ("E2", "E3"):
                    predicted[name] = {}
                    for key, cm in matrices[name].items():
                        pred = prediction(rows[name]["probabilities"][key])
                        predicted[name][key] = pred
                        cm[label][pred] += 1
                weights = row3["final_weights"]
                require(len(weights) == 2 and all(math.isfinite(x) and 0 <= x <= 1 for x in weights)
                        and abs(sum(weights) - 1) < 1e-5, "Invalid cached gate weights")
                correct3, correct2 = predicted["E3"]["dynamic"] == label, predicted["E2"]["dynamic"] == label
                fc, pc = predicted["E3"]["flow_only"] == label, predicted["E3"]["packet_only"] == label
                category = failure_category(correct3, fc, pc, weights)
                categories[LABELS[label]][category] += 1
                pair = "both_correct" if correct2 and correct3 else "E2_only_correct" if correct2 else "E3_only_correct" if correct3 else "both_wrong"
                paired[LABELS[label]][pair] += 1
                values = measurements(row3)
                if label == 1:
                    web_rows["outcome_categories"].setdefault(category, []).append(values)
                    subgroup = "only_packet_correct" if pc and not fc else "only_flow_correct" if fc and not pc else "both_correct" if fc and pc else "both_wrong"
                    web_rows["branch_correctness_groups"].setdefault(subgroup, []).append(values)
                if not correct3:
                    writer.writerow({"sample_id": row3["sample_id"], "label": LABELS[label], "category": category,
                        "E2_prediction": LABELS[predicted["E2"]["dynamic"]], "E3_prediction": LABELS[predicted["E3"]["dynamic"]],
                        "flow_prediction": LABELS[predicted["E3"]["flow_only"]], "packet_prediction": LABELS[predicted["E3"]["packet_only"]], **values})
        require(len(identities) == 13446 and len(set(i[0] for i in identities)) == 13446, "Incomplete/duplicate validation cache")
        require(identity(identities) == report["models"]["E3"]["ordered_validation_identity_sha256"], "Cached validation identity hash differs")
        for name, methods in matrices.items():
            for key, cm in methods.items():
                require(cm == report["models"][name]["metrics"][key]["confusion_matrix"], f"{name}/{key} cached predictions differ from report")
        web_group = report["models"]["E3"]["branch_correctness_groups_by_class"]["WebBased"]["packet_only_correct"]
        bounds = failure_bounds(web_group["samples"], web_group["gate_picks_correct_branch"], web_group["dynamic_correct"])
        result = {"schema": "xgnid.e3-gate-cache-analysis.v1", "status": "completed", "exit_code": 0,
                  "mode": "cache_analysis", "partition": "val", "samples": len(identities),
                  "source_report_sha256": sha(report_path), "source_reference": report["reference"],
                  "source_caches_sha256": {name: sha(path) for name, path in caches.items()},
                  "cache_predictions_reproduce_report": True, "categories_by_true_class": categories,
                  "paired_E2_E3_outcomes_by_true_class": paired,
                  "WebBased_scale_and_probability_summaries": {axis: {key: summarize_values(value) for key, value in entries.items()}
                                                               for axis, entries in web_rows.items()},
                  "WebBased_packet_only_correct_report_bound_for_correct_favored_fusion_failure": bounds,
                  "row_export": csv_path.name, "row_export_sha256": sha(csv_path),
                  "entry_sha256": sha(Path(__file__)), "test_evaluated": False, "model_loaded": False,
                  "limitations": ["H(z) single-branch interventions are not separately trained branch classifiers.",
                    "Weighted representation norms describe scale; they are not causal attribution or a new class probability.",
                    "Cached output analysis cannot determine whether a new scale normalization or auxiliary loss will improve retraining.",
                    "Paired outcomes compare separately trained checkpoints; they do not isolate one training variable."]}
        write(output / "report.json", result)
        write(output / "status.json", {"status": "completed", "exit_code": 0, "mode": "cache_analysis"})
        print(json.dumps({"status": "completed", "report": str(output / "report.json"),
            "WebBased_failure_categories": categories["WebBased"], "WebBased_paired_outcomes": paired["WebBased"],
            "test_evaluated": False, "model_loaded": False}, indent=2), flush=True)
        return result
    except BaseException as exc:
        write(output / "status.json", {"status": "failed", "exit_code": 130 if isinstance(exc, KeyboardInterrupt) else 1, "mode": "cache_analysis"})
        (output / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = inside(repo / "outputs", args.output.resolve())
    require(output.name.startswith("e3_gate_cache_analysis_"), "Use independent outputs/e3_gate_cache_analysis_* directory")
    analyze(args.report.resolve(), output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
