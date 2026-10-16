"""Audit source CSVs without torch/pandas; never modifies the source data."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path


PACKET_FIELDS = ["payload_data", "delta_time", "packet_direction", "ip_size",
                 "transport_size", "payload_size", "syn", "cwr", "ece", "urg",
                 "ack", "psh", "rst", "fin"]


def audit(paths, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    csv.field_size_limit(2**31 - 1)
    seen = {}
    feature_seen = {}
    report = {"sources": [], "cross_file_duplicate_rows": 0,
              "cross_file_duplicate_features": 0, "conflicting_label_rows": 0,
              "identity_limitation": "Row/feature hashes detect exact duplicates only; flow/capture independence requires provenance and group IDs.",
              "duplicate_definition": "SHA256 of parsed CSV string cells; feature hash excludes Label; no numeric normalization"}
    for path in map(Path, paths):
        counts, errors = Counter(), Counter()
        local_seen = set()
        within_file_duplicates = 0
        invalid_rows = 0
        examples = []
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames
            required = {"Label"} | {"udps." + k for k in PACKET_FIELDS}
            missing = sorted(required - set(columns or []))
            if missing:
                raise ValueError(f"{path}: missing required columns {missing}")
            for index, row in enumerate(reader):
                counts[row["Label"]] += 1
                fingerprint = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
                if fingerprint in local_seen:
                    within_file_duplicates += 1
                local_seen.add(fingerprint)
                feature_hash = hashlib.sha256(json.dumps(
                    {k: v for k, v in row.items() if k != "Label"}, sort_keys=True).encode()).hexdigest()
                if feature_hash in feature_seen:
                    source, first_label = feature_seen[feature_hash]
                    if source != str(path):
                        report["cross_file_duplicate_features"] += 1
                    if first_label != row["Label"]:
                        report["conflicting_label_rows"] += 1
                else:
                    feature_seen[feature_hash] = (str(path), row["Label"])
                if fingerprint in seen:
                    errors["duplicate_rows"] += 1
                    if seen[fingerprint] != str(path):
                        report["cross_file_duplicate_rows"] += 1
                else:
                    seen[fingerprint] = str(path)
                try:
                    values = {k: ast.literal_eval(row["udps." + k]) for k in PACKET_FIELDS}
                    n = len(values["payload_data"])
                    if n == 0:
                        raise ValueError("empty packets")
                    if any(not isinstance(v, list) or len(v) != n for v in values.values()):
                        raise ValueError("packet field lengths")
                    for k, vals in values.items():
                        if k != "payload_data" and any(not math.isfinite(float(v)) for v in vals):
                            raise ValueError("nonfinite " + k)
                    for payload in values["payload_data"]:
                        bytes.fromhex(str(payload))
                    for k, v in row.items():
                        if not k.startswith("udps.") and not math.isfinite(float(v)):
                            raise ValueError("nonfinite " + k)
                except (ValueError, TypeError, SyntaxError, KeyError) as exc:
                    invalid_rows += 1
                    errors[str(exc)] += 1
                    if len(examples) < 5:
                        examples.append({"row_index": index, "error": str(exc)})
        report["sources"].append({"path": str(path.resolve()), "bytes": path.stat().st_size,
                                  "sha256": digest.hexdigest(), "columns": columns,
                                  "rows": sum(counts.values()), "label_counts": dict(counts),
                                  "within_file_duplicate_rows": within_file_duplicates,
                                  "invalid_rows": invalid_rows,
                                  "errors": dict(errors), "examples": examples})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audit(args.paths, args.output)
