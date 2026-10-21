"""Group server results through links and a VS Code workspace; never move runs.

Standard library only. Default: preview. --apply: Linux server only.
Reads directory names and small status JSONs, never graphs or checkpoints.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


CATEGORIES = {
    "01_batch_comparisons": "01 正式实验 · Batch 对照",
    "02_grouped_E0_E4": "02 正式实验 · 原 E0–E4",
    "03_gate_studies": "03 门控改进 · E3a 与后续版本",
    "04_gate_diagnostics": "04 门控诊断与缓存分析",
    "05_R0_references": "05 R0 系列对照",
    "06_probes_and_checks": "06 探测、冒烟与前置核对",
    "07_restart_snapshots": "07 重启恢复快照",
    "08_legacy": "08 历史输出与归档",
    "99_other": "99 其他待核实文件",
}
MARKER = {"schema": "xgnid.outputs-organized-view.v1", "original_files_moved": False}
VIEW = "_organized"
WORKSPACE = "XG-NID-organized.code-workspace"
SUFFIXES = (".supervisor.pid", ".restart.json", ".exitcode", ".log", ".pid")


def classify(name):
    n = name.lower()
    if n.startswith("e3_restart_snapshot_"):
        return "07_restart_snapshots"
    if n.startswith(("e3_gate_diagnostic", "e3_gate_cache")):
        return "04_gate_diagnostics"
    if n.startswith("e3_gate_"):
        return "03_gate_studies"
    if n.startswith(("e0_batch_full_", "e2_batch_full_", "e3_batch_full_", "e4_batch_full_")):
        return "01_batch_comparisons"
    if n.startswith("grouped_full_"):
        return "02_grouped_E0_E4"
    if n.startswith("r0"):
        return "05_R0_references"
    if ("_probe_" in n or "preflight" in n or "smoke" in n or
            "pilot" in n or n.startswith("server_graph_split_audit")):
        return "06_probes_and_checks"
    if n in ("_archive", "gnn4id", "gnn4id_edge", "good", "best", "baselines") or n.startswith(("baseline_", "innov", "edge_")):
        return "08_legacy"
    return "99_other"


def inventory(outputs):
    """One level only; existing symlinks are listed, never traversed."""
    entries = sorted((p for p in outputs.iterdir() if p.name != VIEW), key=lambda p: p.name)
    # Group sidecars even if the matching run directory was never created.
    groups = {}
    for path in entries:
        suffix = next((s for s in SUFFIXES if path.name.endswith(s) and not path.is_dir()), None)
        stem = path.name[:-len(suffix)] if suffix else path.name
        group = groups.setdefault(stem, {"name": stem, "category": classify(stem), "files": []})
        group["files"].append(path)
    return sorted(groups.values(), key=lambda g: (g["category"], g["name"]))


def status_for(group):
    # This is an on-disk status, not evidence that a PID is currently alive.
    run = next((p for p in group["files"] if p.name == group["name"] and not p.is_symlink() and p.is_dir()), None)
    if run is None:
        return "未核实（仅旁文件或未知类型）"
    path = run / "status.json"
    try:
        if path.is_symlink() or path.stat().st_size > 65536:
            return "状态文件未读取"
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            return "状态格式未识别"
        return "文件记录：{}；exit_code={}（非进程检查）".format(value.get("status", "unknown"), value.get("exit_code", "未记录"))
    except (OSError, ValueError):
        return "无可读根级 status.json；待核实"


def workspace():
    return {
        "folders": [{"name": "00 代码与配置", "path": "."}] + [
            {"name": label, "path": "outputs/{}/{}".format(VIEW, category)}
            for category, label in CATEGORIES.items()
        ],
        "settings": {
            "files.exclude": {"outputs": True},
            "explorer.compactFolders": False,
            "explorer.sortOrder": "default",
        },
    }


def links_for(outputs, groups):
    for group in groups:
        parent = outputs / VIEW / group["category"] / group["name"]
        for source in group["files"]:
            yield parent / source.name, source


def lexists(path):
    return os.path.lexists(str(path))


def preflight(repo, groups):
    outputs, view = repo / "outputs", repo / "outputs" / VIEW
    # Refuse pre-existing custom directories and symlink parents.
    if lexists(view):
        if view.is_symlink() or not view.is_dir():
            raise ValueError("View path must be a real managed directory")
        marker = view / "managed.json"
        if marker.is_symlink() or json.loads(marker.read_text(encoding="utf-8")) != MARKER:
            raise ValueError("Existing view is not managed by this tool")
    for category in CATEGORIES:
        path = view / category
        if lexists(path) and (path.is_symlink() or not path.is_dir()):
            raise ValueError("Unsafe category: " + str(path))
    for destination, source in links_for(outputs, groups):
        for parent in (destination.parent.parent, destination.parent):
            if lexists(parent) and (parent.is_symlink() or not parent.is_dir()):
                raise ValueError("Unsafe view parent: " + str(parent))
        if lexists(destination) and (not destination.is_symlink() or os.readlink(destination) != os.path.relpath(source, destination.parent)):
            raise ValueError("Existing view entry differs; retained: " + str(destination))
    path = repo / WORKSPACE
    encoded = json.dumps(workspace(), ensure_ascii=False, indent=2) + "\n"
    if lexists(path) and (path.is_symlink() or path.read_text(encoding="utf-8") != encoded):
        raise ValueError("Existing customized workspace retained: " + str(path))
    for name in ("README.md", "inventory.json"):
        if (view / name).is_symlink():
            raise ValueError("Managed index must not be a symlink")
    return encoded


def apply(repo, groups):
    encoded = preflight(repo, groups)
    outputs, view = repo / "outputs", repo / "outputs" / VIEW
    view.mkdir(exist_ok=True)
    marker = view / "managed.json"
    if not marker.exists():
        with marker.open("x", encoding="utf-8") as stream:
            json.dump(MARKER, stream, indent=2)
    for category in CATEGORIES:
        path = view / category
        if path.is_symlink() or (lexists(path) and not path.is_dir()):
            raise ValueError("Unsafe category: " + str(path))
        path.mkdir(exist_ok=True)
    created = 0
    for destination, source in links_for(outputs, groups):
        destination.parent.mkdir(exist_ok=True)
        if not lexists(destination):
            destination.symlink_to(os.path.relpath(source, destination.parent), target_is_directory=source.is_dir())
            created += 1
    records = [{"name": g["name"], "category": g["category"], "recorded_status": status_for(g),
                "original_paths": [p.relative_to(repo).as_posix() for p in g["files"]]} for g in groups]
    index = ["# 服务器结果分类入口", "", "原文件均在原位，分类目录中的条目是链接。训练可以继续写入，链接内容同步可见。",
             "", "请通过仓库根目录 XG-NID-organized.code-workspace 打开分类工作区。",
             "状态仅来自磁盘 status.json，不能据此断定进程是否存活。类别由名称确定，不代表成功与否或最佳结果。",
             "新实验结束或启动后可重新执行本工具，补充新增链接、刷新状态。不会自动删除失效链接。",
             "不要在链接目录内删除或修改文件，这会作用于原始实验。", ""]
    for category, label in CATEGORIES.items():
        index.extend(["## " + label, ""])
        index.extend("- `{}` — {}".format(r["name"], r["recorded_status"]) for r in records if r["category"] == category)
        index.append("")
    (view / "inventory.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (view / "README.md").write_text("\n".join(index), encoding="utf-8")
    if not (repo / WORKSPACE).exists():
        with (repo / WORKSPACE).open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    return created


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--apply", action="store_true", help="Create managed links/workspace on Linux; default preview only")
    args = parser.parse_args()
    repo = args.repo.resolve()
    outputs = repo / "outputs"
    if outputs.is_symlink() or not outputs.is_dir():
        raise ValueError("outputs must be an existing real directory")
    groups = inventory(outputs)
    created = 0
    if args.apply:
        if not sys.platform.startswith("linux"):
            raise ValueError("--apply runs only on the Linux server")
        created = apply(repo, groups)
    print(json.dumps({"status": "organized" if args.apply else "preview_only", "groups": len(groups),
                      "categories": {label: sum(g["category"] == key for g in groups) for key, label in CATEGORIES.items()},
                      "links_created": created, "original_files_moved": 0, "original_files_deleted": 0,
                      "workspace": str(repo / WORKSPACE)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
