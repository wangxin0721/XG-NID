"""Create a grouped VS Code results workspace without moving experiment files.

Standard library only. Reads directory names, never models or datasets. Existing
project .vscode/settings.json and experiment files are left untouched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def workspace(repo):
    outputs = repo / "outputs"
    if not outputs.is_dir():
        raise ValueError("Repository outputs directory is missing")
    folders = [{"name": "00 代码、配置与文档", "path": "."}]

    def add(name, directory):
        if directory.is_dir():
            folders.append({"name": name, "path": directory.relative_to(repo).as_posix()})

    add("01 R0-A 官方流程对照 · 正式训练", outputs / "r0_aligned_full_s42_v1")
    add("02 R0-A CUDA 冒烟", outputs / "r0_aligned_smoke_s42_v1")
    # List every matching formal run; never silently select one as the best.
    for directory in sorted(outputs.glob("r0_full_s42_*")):
        add("03 原 R0 · " + directory.name, directory)
    for group in ("E0", "E1", "E2", "E3"):
        for directory in sorted(outputs.glob("grouped_full_" + group + "_*")):
            add("04 分组实验 " + group + " · " + directory.name, directory)
    add("90 全部产物与日志（按需展开）", outputs)
    return {
        "folders": folders,
        "settings": {
            # Excludes the child outputs folder in the code root. The separate
            # workspace root pointing to outputs remains available above.
            "files.exclude": {"outputs": True},
            "explorer.compactFolders": False,
            "explorer.sortOrder": "default",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo = args.repo.resolve()
    value = workspace(repo)
    destination = repo / "XG-NID-results.code-workspace"
    encoded = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    # Refuse to overwrite an existing customized workspace. Repeated execution
    # with exactly the same inventory is harmless; directory changes are explicit.
    if destination.exists():
        if destination.read_text(encoding="utf-8") != encoded:
            raise FileExistsError("Workspace already exists with different content; retained: " + str(destination))
        action = "existing_workspace_unchanged"
    else:
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
        action = "workspace_created"
    print(json.dumps({"status": action, "workspace": str(destination),
                      "visible_roots": [item["name"] for item in value["folders"]],
                      "experiment_files_moved": 0, "experiment_files_deleted": 0,
                      "project_vscode_settings_modified": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
