"""Create a source-only handoff archive and an explicit Git path list."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "src/xgnid/__init__.py", "src/xgnid/model.py", "src/xgnid/graph_building.py",
    "src/xgnid/edge_experiments.py", "third_party/GNN4ID/Utility/Model.py",
    "scripts/audit_edge_data.py", "scripts/check_edge_server.py",
    "scripts/test_server_audit.py",
    "scripts/audit_graph_split.py", "scripts/test_graph_split_audit.py",
    "scripts/build_grouped_split.py", "scripts/test_grouped_split.py",
    "scripts/grouped_graphs.py", "scripts/smoke_grouped_gpu.py", "scripts/test_grouped_gpu_smoke.py",
    "scripts/train_grouped.py", "scripts/test_grouped_training.py",
    "scripts/run_edge_chain.py", "scripts/summarize_edge_chain.py",
    "scripts/package_edge_delivery.py", "docs/edge_P0_20261016.md",
    "docs/edge_stage_report_20261016.md", "docs/edge_server_handoff.md",
]


def package(output):
    # Read all selected files first; never collect directories or Git state files.
    sources = {path: (ROOT / path).read_bytes() for path in FILES}
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"scope": "source-only; no datasets, checkpoints or experiment outputs",
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "files": {path: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                          for path, data in sources.items()}}
    payload = json.dumps(manifest, ensure_ascii=False, indent=2)
    (output / "manifest.json").write_text(payload, encoding="utf-8")
    (output / "git-files.txt").write_text("\n".join(FILES) + "\n", encoding="utf-8")
    with zipfile.ZipFile(output / "edge_source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path, data in sources.items():
            archive.writestr(path, data)
        archive.writestr("edge_delivery_manifest.json", payload)
    print(f"Packaged {len(FILES)} files: {(output / 'edge_source.zip').resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    package(parser.parse_args().output)
