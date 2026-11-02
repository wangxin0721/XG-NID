"""Download author-published CSVs and rebuild all graphs in an isolated server job.

No GPU, training, resampling, replacement of existing data, or fabricated payload.
Default command submits a durable background job. --status only reads progress.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, deque
import csv
import hashlib
from http.cookiejar import CookieJar
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
from types import MethodType, SimpleNamespace
from urllib.parse import urlencode, urlparse
from urllib.request import Request, build_opener, HTTPCookieProcessor


SOURCES = {
    "train": {"id": "1IClFYOup0U58WKj2R0p1IJ_ZMc1Os2FD", "bytes": 1027673808, "rows": 160000,
              "observed_sha256": "1b546e6c69faad482b98b4c95219ef9b13aba0c13afcfe0b4abe69c1e7f38b02"},
    "test": {"id": "1Yxpd-ruxC8B1zz3e8oLOcZcfrahIfhJc", "bytes": 175063419, "rows": 25557,
             "observed_sha256": "08b256997d0368dde6c85a3b43ef3d56b4ce336939fcdce41596725bbdee42ed"},
}
REPOSITORY_COMMIT = "551d1f10189e5259f8f34e32f918d557a2b3a64e"
TOTAL = 185557


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def within(root, value):
    path = (root / value).resolve()
    path.relative_to(root)
    return path


class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.forms, self.current = [], None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "form":
            self.current = {"action": attrs.get("action", ""), "fields": {}}
            self.forms.append(self.current)
        elif tag == "input" and self.current is not None and attrs.get("name"):
            self.current["fields"][attrs["name"]] = attrs.get("value", "")

    def handle_endtag(self, tag):
        if tag == "form":
            self.current = None


def download(part, destination, progress, resume=False):
    source = SOURCES[part]
    url = "https://drive.usercontent.google.com/download?" + urlencode(
        {"id": source["id"], "export": "download", "confirm": "t"})
    opener = build_opener(HTTPCookieProcessor(CookieJar()))
    digest, count, last, failures, headers = hashlib.sha256(), 0, 0, 0, {}
    if destination.exists():
        require(resume, "Download destination exists; no implicit overwrite")
        with destination.open("rb") as prior:
            for block in iter(lambda: prior.read(4 * 1024 * 1024), b""):
                digest.update(block)
                count += len(block)
        require(count <= source["bytes"], "Existing partial download exceeds published size")
        progress(part, count)
    with destination.open("ab" if resume and destination.exists() else "xb") as stream:
        while count < source["bytes"]:
            end = min(count + 16 * 1024 * 1024, source["bytes"]) - 1
            try:
                for confirmation in range(3):
                    response = opener.open(Request(url, headers={"User-Agent": "Mozilla/5.0",
                        "Range": f"bytes={count}-{end}"}), timeout=30)
                    if "text/html" not in response.headers.get("Content-Type", "").lower():
                        break
                    with response:
                        page = response.read(2 * 1024 * 1024).decode("utf-8", "replace")
                    parser = DownloadForm()
                    parser.feed(page)
                    forms = [f for f in parser.forms if urlparse(f["action"]).scheme == "https"
                             and urlparse(f["action"]).hostname in ("drive.google.com", "drive.usercontent.google.com")
                             and f["fields"].get("id") == source["id"]]
                    require(len(forms) == 1, "Author download unavailable/permission or quota page: " + part)
                    form = forms[0]
                    url = form["action"] + ("&" if "?" in form["action"] else "?") + urlencode(form["fields"])
                else:
                    raise ValueError("Download confirmation did not return a CSV")
                headers = {k: response.headers.get(k) for k in ("Content-Type", "Content-Length", "ETag", "Last-Modified", "Content-Range")}
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", headers["Content-Range"] or "")
                if response.status == 206:
                    require(match is not None and int(match[1]) == count and int(match[3]) == source["bytes"]
                            and count <= int(match[2]) <= end, "Range response does not match requested source offset/size")
                    response_end = int(match[2])
                else:
                    require(response.status == 200 and count == 0, "Server cannot safely resume requested byte offset")
                    response_end = source["bytes"] - 1
                with response:
                    while count <= response_end:
                        block = response.read(min(1024 * 1024, response_end - count + 1))
                        require(bool(block), "Download stream ended before its declared range")
                        stream.write(block)
                        digest.update(block)
                        count += len(block)
                failures = 0
                stream.flush()
                if count - last >= 16 * 1024 * 1024:
                    progress(part, count)
                    last = count
            except (OSError, TimeoutError) as exc:
                stream.flush()
                failures += 1
                print(f"download {part} interrupted at byte {count}; retry {failures}/4: {exc}", flush=True)
                if failures >= 4:
                    raise
    require(count == source["bytes"], "Published file size differs/truncated download: " + part)
    if source.get("observed_sha256"):
        require(digest.hexdigest() == source["observed_sha256"], "Author file content changed from observed download: " + part)
    progress(part, count)
    return {"file_id": source["id"], "url": url, "bytes": count, "sha256": digest.hexdigest(), "headers": headers}


def verified_copy(part, source, destination, progress):
    """Accept existing server bytes only after matching the full author SHA."""
    expected = SOURCES[part]
    before = source.stat()
    require(source.is_file() and before.st_size == expected["bytes"], "Existing CSV size differs: " + part)
    digest = sha(source)
    require(digest == expected["observed_sha256"], "Existing CSV does not match verified author SHA: " + part)
    after_hash = source.stat()
    require((before.st_size, before.st_mtime_ns) == (after_hash.st_size, after_hash.st_mtime_ns), "Source changed during verification")
    copied, h = 0, hashlib.sha256()
    with source.open("rb") as original, destination.open("xb") as target:
        for block in iter(lambda: original.read(4 * 1024 * 1024), b""):
            target.write(block)
            h.update(block)
            copied += len(block)
            if copied % (64 * 1024 * 1024) == 0:
                progress(part, copied)
    after_copy = source.stat()
    require((before.st_size, before.st_mtime_ns) == (after_copy.st_size, after_copy.st_mtime_ns)
            and copied == expected["bytes"] and h.hexdigest() == digest,
            "Existing CSV changed while copying")
    require(sha(destination) == digest, "Copied CSV verification failed")
    progress(part, copied)
    return {"file_id": expected["id"], "bytes": copied, "sha256": digest,
            "acquisition": "verified_existing_server_csv_copy", "copy_source": str(source),
            "author_url": "https://drive.google.com/file/d/" + expected["id"] + "/view",
            "network_used": False}


def dataset_path(repo, offline):
    suffix = "offline_v2" if offline else "v1"
    return repo / ("data/processed/xgnid_author_rebuild_" + suffix)


def job_path(repo, offline):
    suffix = "offline_v2" if offline else "v1"
    return repo / ("outputs/r0_graph_rebuild_author_" + suffix)


def worker(repo, job, offline=False):
    require(sys.platform.startswith("linux"), "Run data processing only on the Linux server")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Rebuild must hide all GPUs")
    started = time.time()
    status = {"status": "running", "phase": "source_checks", "pid": os.getpid(),
              "gpu_used": False, "training_started": False, "graphs_completed": 0, "offline": offline}
    write(job / "status.json", status)
    try:
        launch = read(repo / "outputs/r0_full_launch.json")
        code, manifest, original = [within(repo, launch[k]) for k in ("code_root", "manifest_dir", "output")]
        summary = read(original / "summary.json")
        require(summary["status"] == "completed" and summary["exit_code"] == 0, "Original R0 completion missing")
        require(sha(original / "last.pt") == summary["last_checkpoint_sha256"], "Original checkpoint changed")
        for name, digest in summary["binding"]["code"].items():
            require(identity(within(code, name).read_text(encoding="utf-8-sig")) == digest, "Frozen source changed: " + name)
        lock = read(code / "docs/r0_source_lock.json")
        require(lock["commit"] == REPOSITORY_COMMIT, "Unexpected author commit")
        functions = (code / "third_party/GNN4ID/Utility/Functions.py").read_text(encoding="utf-8-sig")
        require(identity(functions) == lock["files"]["Utility/Functions.py"]["normalized_text_identity"], "Author graph source differs")
        meta = read(manifest / "manifest.json")
        require(meta["manifest_id"] == summary["binding"]["manifest_id"] and sha(manifest / "samples.jsonl")
                == meta["sample_manifest_sha256"], "Frozen old manifest differs")
        old, old_counts = {p: {} for p in SOURCES}, {p: Counter() for p in SOURCES}
        with (manifest / "samples.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                part = row["partition"]
                pattern = r"data_(\d+)\.pt" if part == "train" else r"data_test_(\d+)\.pt"
                match = re.fullmatch(pattern, Path(row["path"]).name)
                require(match is not None and row["container_offset"] == 0, "Old source is not official numbered graphs")
                index = int(match[1])
                require(index not in old[part], "Duplicate old numeric index")
                old[part][index] = row
                old_counts[part][str(row["label"])] += 1
        for part in SOURCES:
            require(set(old[part]) == set(range(SOURCES[part]["rows"])), "Old numeric membership is incomplete")
        evidence = read(manifest.parent / "source_evidence.json")
        data = dataset_path(repo, offline)
        require(not data.exists(), "New rebuild directory already exists; no overwrite or automatic resume")
        # Conservative allowance for max20 packets x1500 float32 plus metadata,
        # new CSVs, manifests and filesystem overhead. Not a measured ETA/size.
        required_bytes = TOTAL * 192 * 1024 + sum(s["bytes"] for s in SOURCES.values()) + 2 * 1024**3
        space = shutil.disk_usage(data.parent)
        require(space.free >= required_bytes, f"Insufficient free space: {space.free} available, {required_bytes} required")
        fs = os.statvfs(data.parent)
        require(not fs.f_files or fs.f_favail > TOTAL + 128, "Insufficient free inodes for separate graph files")
        (data / "raw").mkdir(parents=True, exist_ok=False)
        (data / "processed").mkdir()
        sources = {}

        def download_progress(part, count):
            status.update(phase="verify_and_copy_existing_csv" if offline else "download_author_csv", partition=part, downloaded_bytes=count,
                          download_total_bytes=SOURCES[part]["bytes"])
            write(job / "status.json", status)
            print(f"{'copy' if offline else 'download'} {part}: {count}/{SOURCES[part]['bytes']} bytes", flush=True)

        for part in SOURCES:
            destination = data / "raw" / ("df_class_8_" + part + ".csv")
            if offline:
                source = within(repo, evidence["csv_sources"][part]["path"])
                print(f"Verifying existing {part} against full author SHA: {source}", flush=True)
                status.update(phase="verify_existing_csv_sha256", partition=part)
                write(job / "status.json", status)
                sources[part] = verified_copy(part, source, destination, download_progress)
            else:
                sources[part] = download(part, destination, download_progress)
            sources[part]["path"] = str(destination)
            sources[part]["equal_to_previous_candidate_sha256"] = sources[part]["sha256"] == evidence["csv_sources"][part]["sha256"]
            print(f"author {part} SHA256={sources[part]['sha256']}; same as previous candidate="
                  f"{sources[part]['equal_to_previous_candidate_sha256']}", flush=True)
        write(job / "download_sources.json", sources)
        sys.path.insert(0, str(code / "scripts"))
        import r0_data
        import pandas as pd
        import numpy as np
        import torch
        import torch_geometric
        from torch_geometric.data import Dataset, HeteroData
        import torch_geometric.transforms as transforms
        require(r0_data.ROOT.resolve() == code, "Wrong frozen graph validator loaded")
        runtime = {"python": sys.version.split()[0], "torch": torch.__version__, "pyg": torch_geometric.__version__,
                   "pandas": pd.__version__, "numpy": np.__version__}
        for name in ("python", "torch", "pyg"):
            require(runtime[name] == summary["binding"]["runtime"][name], "Original server environment differs: " + name)
        torch.set_num_threads(1)
        require(old_counts == {p: Counter(v) for p, v in r0_data.table4_counts().items()}, "Old class counts differ")
        node = next(n for n in ast.parse(functions).body if isinstance(n, ast.ClassDef) and n.name == "NIDSDataset")
        (job / "executed_author_class.py").write_text(ast.unparse(node) + "\n", encoding="utf-8")
        namespace = {"__name__": "isolated_author_graphs", "Dataset": Dataset, "HeteroData": HeteroData,
                     "np": np, "T": transforms, "os": os, "sys": sys, "tqdm": lambda rows, **kw: rows}
        counts, differences = {p: Counter() for p in SOURCES}, Counter()
        graph_bytes, completed, current_part, next_index = 0, 0, "", 0
        generation_started = time.time()
        writer = (job / "graphs.jsonl").open("x", encoding="utf-8")

        def save_checked(graph, filename):
            nonlocal graph_bytes, completed, next_index
            path = Path(filename).resolve()
            expected_name = f"data_{next_index}.pt" if current_part == "train" else f"data_test_{next_index}.pt"
            require(path.parent == data / "processed" and path.name == expected_name, "Unexpected author output filename")
            r0_data.check(graph)
            edge, node_hash = r0_data.fingerprint(graph, True), r0_data.fingerprint(graph, False)
            with path.open("xb") as stream:
                torch.save(graph, stream)
            restored = torch.load(path, map_location="cpu", weights_only=False)
            r0_data.check(restored)
            require(r0_data.fingerprint(restored, True) == edge and r0_data.fingerprint(restored, False) == node_hash
                    and torch.equal(restored.y, graph.y), "Graph serialization verification failed")
            previous = old[current_part][next_index]
            matches = {"edge_input_equal": edge == previous["edge_hash"], "node_input_equal": node_hash == previous["node_hash"],
                       "label_equal": int(graph.y.item()) == previous["label"]}
            for name, equal in matches.items():
                if not equal:
                    differences[name] += 1
            row = {"partition": current_part, "csv_row_index": next_index,
                   "source_csv_sha256": sources[current_part]["sha256"], "path": "processed/" + path.name,
                   "label": int(graph.y.item()), "edge_hash": edge, "node_hash": node_hash,
                   "serialized_sha256": sha(path), "old_sample_id": previous["sample_id"], **matches}
            writer.write(json.dumps(row) + "\n")
            counts[current_part][str(row["label"])] += 1
            graph_bytes += path.stat().st_size
            next_index += 1
            completed += 1
            if completed % 1000 == 0 or completed == TOTAL:
                writer.flush()
                elapsed = time.time() - generation_started
                status.update(phase="generate_and_compare_graphs", partition=current_part, graphs_completed=completed,
                              total_graphs=TOTAL, graph_bytes=graph_bytes,
                              generation_seconds=elapsed, estimated_remaining_seconds=(TOTAL - completed) * elapsed / completed,
                              differences=dict(differences))
                write(job / "status.json", status)
                print(f"graphs={completed}/{TOTAL} rate={completed/elapsed:.2f}/s "
                      f"remaining~{status['estimated_remaining_seconds']/60:.1f}min differences={dict(differences)}", flush=True)
                require(shutil.disk_usage(data).free > 512 * 1024**2, "Less than 512MiB remaining; stop without deleting partial data")

        class TorchWithSave:
            def __getattr__(self, name):
                return save_checked if name == "save" else getattr(torch, name)

        namespace["torch"] = TorchWithSave()
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "<locked-author-NIDSDataset>", "exec"), namespace)
        cls = namespace["NIDSDataset"]
        packet = ["udps." + c for c in ("payload_data", "delta_time", "packet_direction", "ip_size", "transport_size",
                                       "payload_size", "syn", "cwr", "ece", "urg", "ack", "psh", "rst", "fin")]
        flow_order = None
        try:
            for current_part in SOURCES:
                next_index = 0
                source = Path(sources[current_part]["path"])
                with source.open(newline="", encoding="utf-8-sig") as stream:
                    header = next(csv.reader(stream))
                require(len(header) == len(set(header)) and {"Label", *packet}.issubset(header), "Source is not author flow+packet CSV")
                columns = [c for c in header if c not in ["Label", *packet]]
                require(len(columns) == 82, "Expected exactly 82 ordered Flow columns")
                if flow_order is None:
                    flow_order = columns
                require(columns == flow_order, "Train/test Flow feature order differs")
                context = SimpleNamespace(raw_paths=[str(source)], processed_dir=str(data / "processed"),
                    index=0, single_file=True, include_packetflag=False, include_packetpayload=True,
                    test=current_part == "test", skip_processing=True)
                for name, method in cls.__dict__.items():
                    if name.startswith("_get_"):
                        setattr(context, name, MethodType(method, context))
                for frame in pd.read_csv(source, chunksize=1024):
                    require(list(frame.index) == list(range(next_index, next_index + len(frame))), "CSV row order changed")
                    # Only the CSV reader is supplied an ordered independent-row
                    # chunk; all parsing, tensor/edge construction and undirected
                    # transformation are the unchanged official process method.
                    namespace["pd"] = SimpleNamespace(read_csv=lambda *a, **kw: frame)
                    cls.process(context)
                    require(context.index == next_index, "Official filename counter differs")
                require(next_index == SOURCES[current_part]["rows"], "Downloaded CSV row count differs")
                require(sha(source) == sources[current_part]["sha256"], "Downloaded input changed during generation")
        finally:
            writer.close()
        require(counts == {p: Counter(v) for p, v in r0_data.table4_counts().items()} and completed == TOTAL, "Final per-class counts differ")
        require(sum(1 for p in (data / "processed").iterdir() if p.suffix == ".pt") == TOTAL, "Final graph file count differs")
        require(sha(original / "last.pt") == summary["last_checkpoint_sha256"], "Original R0 checkpoint changed")
        result = {"status": "completed", "exit_code": 0, "graphs_completed": completed, "dataset": str(data),
                  "processed_dir": str(data / "processed"), "sources": sources, "class_counts": {p: dict(c) for p, c in counts.items()},
                  "flow_feature_order": flow_order, "graph_bytes": graph_bytes, "wall_seconds": time.time() - started,
                  "all_old_inputs_and_labels_equal": not differences, "differences": dict(differences),
                  "graphs_manifest_sha256": sha(job / "graphs.jsonl"), "author_commit": REPOSITORY_COMMIT,
                  "author_functions_identity": identity(functions), "old_manifest_id": meta["manifest_id"],
                  "runtime": runtime,
                  "gpu_used": False, "training_started": False, "old_data_replaced": False, "offline": offline,
                  "historical_paper_sample_identity_verified": False,
                  "processing": "Fresh published processed CSVs; unchanged row-wise official process/helpers, streamed chunks, observed saves",
                  "limitations": ["No PCAP re-extraction or upstream rolling/filter/sampling reconstruction",
                                  "New file SHA differs from graph tensor equivalence; test graphs validated without model inference",
                                  "New dataset is not automatically attached to frozen R0/E0-E3"]}
        write(job / "summary.json", result)
        write(job / "status.json", {"status": "completed", "exit_code": 0, "graphs_completed": completed,
                                     "all_old_inputs_and_labels_equal": not differences})
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    except BaseException as exc:
        (job / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        write(job / "status.json", {**status, "status": "failed", "exit_code": 1, "error": str(exc)})
        traceback.print_exc()
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--offline", action="store_true",
                        help="Use existing server CSVs only after verifying full author SHA; isolated v2 job")
    parser.add_argument("--download-only", action="store_true",
                        help="Download published CSVs only; allowed locally, never loads graphs or models")
    parser.add_argument("--resume-download", action="store_true",
                        help="Explicitly resume this tool's local download-only directory")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--supervise", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    repo = args.repo.resolve()
    if args.download_only:
        require(not (args.status or args.worker or args.supervise or args.offline), "Download-only cannot be combined with job modes")
        destination = repo / "data/raw/xgnid_author_published_v1"
        destination.parent.mkdir(parents=True, exist_ok=True)
        require(shutil.disk_usage(destination.parent).free > sum(s["bytes"] for s in SOURCES.values()) + 512 * 1024**2,
                "Insufficient free space to download author CSVs")
        if args.resume_download:
            previous = read(destination / "download.json")
            require(previous.get("repository_commit") == REPOSITORY_COMMIT
                    and previous.get("source") == "README-linked author Google Drive folder", "Cannot resume unrelated download directory")
        else:
            destination.mkdir(exist_ok=False)
        result = {"status": "downloading", "source": "README-linked author Google Drive folder",
                  "folder_url": "https://drive.google.com/drive/folders/1FiZh87vvCZF3gX1Fnj9iTB4j74u-nuR6",
                  "repository_commit": REPOSITORY_COMMIT, "sources": {}, "models_or_graphs_executed": False}
        write(destination / "download.json", result)
        try:
            def show(part, count):
                print(f"download {part}: {count}/{SOURCES[part]['bytes']} bytes", flush=True)
            for part in SOURCES:
                path = destination / ("df_class_8_" + part + ".csv")
                item = download(part, path, show, resume=args.resume_download)
                with path.open(newline="", encoding="utf-8-sig") as stream:
                    columns = next(csv.reader(stream))
                require({"Label", "udps.payload_data", "udps.delta_time", "udps.packet_direction",
                         "udps.ip_size", "udps.transport_size", "udps.payload_size"}.issubset(columns),
                        "Downloaded file lacks required author graph fields")
                item.update(path=str(path), columns=columns)
                result["sources"][part] = item
                write(destination / "download.json", result)
            result["status"] = "completed"
            write(destination / "download.json", result)
            print(json.dumps({"status": "completed", "directory": str(destination),
                              "files": {p: {k: s[k] for k in ("bytes", "sha256")} for p, s in result["sources"].items()},
                              "models_or_graphs_executed": False}, ensure_ascii=False, indent=2), flush=True)
            return 0
        except BaseException as exc:
            result.update(status="failed", error=str(exc))
            write(destination / "download.json", result)
            raise
    require(not args.resume_download, "--resume-download requires --download-only")
    require(sys.platform.startswith("linux"), "Run graph processing only on Linux server")
    job = job_path(repo, args.offline)
    if args.status:
        value = {"job": str(job), "status": read(job / "status.json") if (job / "status.json").is_file() else "not_submitted"}
        for name in ("worker.pid", "supervisor.pid", "exitcode"):
            value[name] = (job / name).read_text().strip() if (job / name).is_file() else None
        if (job / "run.log").is_file():
            with (job / "run.log").open(encoding="utf-8", errors="replace") as stream:
                value["log_tail"] = list(deque(stream, maxlen=8))
        if (job / "summary.json").is_file():
            result = read(job / "summary.json")
            value["summary"] = {k: result[k] for k in ("graphs_completed", "dataset", "all_old_inputs_and_labels_equal", "differences", "wall_seconds")}
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return 0
    if args.worker:
        return worker(repo, job, args.offline)
    mode_flags = ["--offline"] if args.offline else []
    if args.supervise:
        rc = 1
        try:
            process = subprocess.Popen([sys.executable, "-B", "-u", str(job / "entry.py"), "--repo", str(repo), "--worker", *mode_flags],
                                       cwd=repo, stdin=subprocess.DEVNULL)
            (job / "worker.pid").write_text(str(process.pid) + "\n")
            rc = process.wait()
        finally:
            (job / "exitcode").write_text(str(rc) + "\n")
        return rc
    require((repo / "outputs/r0_full_launch.json").is_file(), "Run in the original server repository")
    require(not dataset_path(repo, args.offline).exists(), "Rebuild dataset already exists; retained")
    job.mkdir(exist_ok=False)
    shutil.copyfile(Path(__file__), job / "entry.py")
    write(job / "status.json", {"status": "submitted", "training_started": False, "gpu_used": False, "offline": args.offline})
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    with (job / "run.log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-B", "-u", str(job / "entry.py"), "--repo", str(repo), "--supervise", *mode_flags],
                                   cwd=repo, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   env=environment, start_new_session=True)
    (job / "supervisor.pid").write_text(str(process.pid) + "\n")
    print(json.dumps({"submission": "recorded; check actual status", "supervisor_pid": process.pid,
                      "job": str(job), "log": str(job / "run.log"), "training_started": False, "offline": args.offline}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
