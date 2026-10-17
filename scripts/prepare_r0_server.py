"""Server-only CPU preparation: freeze CSV evidence and R0 graph manifest.

No CUDA checks, model construction, graph rewrite, or training. Use a separate
Git worktree so an active E1 keeps its original code and working directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source_repo.resolve(), args.output.resolve()
    if not source.is_dir():
        parser.error("Source repository directory is missing")
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    status = {"status": "running", "phase": "source_evidence", "pid": os.getpid(),
              "started_utc": started, "python": sys.executable,
              "cuda_used": False, "training_started": False}
    write(output / "status.json", status)
    try:
        root = Path(__file__).resolve().parents[1]
        graph_root = source / "data/processed/CICIoT2023_processed/processed"
        csv_root = graph_root.parent
        if not graph_root.is_dir():
            raise ValueError(f"Missing graph directory: {graph_root}")
        evidence = {"schema": "xgnid.r0.source-evidence.v1", "source_repo": str(source),
                    "source_graph_root": str(graph_root), "code_root": str(root),
                    "code_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                    "claim": "protocol_reimplementation", "membership_basis": "existing official-style graph filenames",
                    "historical_author_sample_identity_verified": False,
                    "csv_to_graph_row_mapping_verified": False,
                    "limitations": ["CSV hashes identify current candidate files, not proof that graphs were built from them.",
                                    "Original author row/capture identities and training RNG state are unavailable.",
                                    "Preserve all source train/test members and any observed overlap; never select by test scores."],
                    "csv_sources": {}}
        for part in ("train", "test"):
            path = csv_root / f"df_class_8_{part}.csv"
            before = path.stat()
            print(f"Hashing source candidate {part}: {path}", flush=True)
            digest = sha(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("Source CSV changed while computing its hash")
            evidence["csv_sources"][part] = {"path": str(path), "bytes": after.st_size, "sha256": digest}
        write(output / "source_evidence.json", evidence)
        write(output / "status.json", {**status, "phase": "graph_manifest"})
        print("Freezing source graph membership on CPU; original files remain unchanged.", flush=True)
        from r0_data import prepare
        metadata = prepare(argparse.Namespace(trusted_data=graph_root,
            membership_evidence=output / "source_evidence.json", locations=None, output=output / "manifest"))
        result = {**status, "status": "completed", "phase": "manifest_preparation_only", "exit_code": 0,
                  "finished_utc": datetime.now(timezone.utc).isoformat(), "manifest_id": metadata["manifest_id"],
                  "table4_counts_match": metadata["table4_counts_match"],
                  "partition_label_counts": metadata["partition_label_counts"],
                  "edge_overlap": metadata["edge_overlap"], "node_overlap": metadata["node_overlap"],
                  "historical_author_sample_identity_verified": False,
                  "server_gpu_smoke_completed": False, "formal_training_completed": False}
        write(output / "summary.json", result)
        write(output / "status.json", result)
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        return 0
    except BaseException as exc:
        (output / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        write(output / "status.json", {**status, "status": "failed", "exit_code": 1, "error": str(exc)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
