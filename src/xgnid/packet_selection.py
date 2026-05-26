from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


PACKET_COLUMNS = [
    "udps.payload_data",
    "udps.delta_time",
    "udps.packet_direction",
    "udps.ip_size",
    "udps.transport_size",
    "udps.payload_size",
    "udps.syn",
    "udps.cwr",
    "udps.ece",
    "udps.urg",
    "udps.ack",
    "udps.psh",
    "udps.rst",
    "udps.fin",
]

PACKET_FLAG_COLUMNS = [
    "udps.syn",
    "udps.cwr",
    "udps.ece",
    "udps.urg",
    "udps.ack",
    "udps.psh",
    "udps.rst",
    "udps.fin",
]


@dataclass(frozen=True)
class PacketSelectionConfig:
    strategy: str = "topk"
    selection_ratio: float = 0.35
    score_threshold: float = 0.5
    min_packets: int = 1
    max_packets: int | None = 20
    keep_original_order: bool = True


def _parse_list_cell(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        text = text.strip("[]")
        if not text:
            return []
        return [item.strip().strip("'\"") for item in text.split(",")]
    if isinstance(parsed, (list, tuple)):
        return [str(item) for item in parsed]
    return [str(parsed)]


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _numeric_array(values: Sequence[str], length: int) -> np.ndarray:
    out = np.zeros(length, dtype=np.float32)
    for idx in range(min(length, len(values))):
        out[idx] = _safe_float(values[idx])
    return out


def _normalize(values: np.ndarray, invert: bool = False) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)
    min_v = float(np.min(values))
    max_v = float(np.max(values))
    if abs(max_v - min_v) < 1e-12:
        normalized = np.zeros_like(values, dtype=np.float32)
    else:
        normalized = (values - min_v) / (max_v - min_v)
    if invert:
        normalized = 1.0 - normalized
    return np.clip(normalized.astype(np.float32), 0.0, 1.0)


def _payload_stats(payload_hex: str) -> tuple[float, float, float, float, float]:
    if not payload_hex or payload_hex == "0":
        return 0.0, 0.0, 0.0, 0.0, 0.0
    try:
        payload_bytes = bytes.fromhex(payload_hex)
    except ValueError:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    byte_arr = np.frombuffer(payload_bytes, dtype=np.uint8).astype(np.float32)
    if byte_arr.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    return (
        float(byte_arr.mean()),
        float(byte_arr.std(ddof=0)),
        float(byte_arr.min()),
        float(byte_arr.max()),
        float(np.count_nonzero(byte_arr) / byte_arr.size),
    )


def score_packets_heuristic(record: dict[str, object]) -> list[float]:
    payloads = _parse_list_cell(record.get("udps.payload_data"))
    packet_count = len(payloads)
    if packet_count == 0:
        return []

    delta_time = _numeric_array(_parse_list_cell(record.get("udps.delta_time")), packet_count)
    packet_direction = _numeric_array(_parse_list_cell(record.get("udps.packet_direction")), packet_count)
    ip_size = _numeric_array(_parse_list_cell(record.get("udps.ip_size")), packet_count)
    transport_size = _numeric_array(_parse_list_cell(record.get("udps.transport_size")), packet_count)
    payload_size = _numeric_array(_parse_list_cell(record.get("udps.payload_size")), packet_count)
    flag_arrays = [
        _numeric_array(_parse_list_cell(record.get(column)), packet_count)
        for column in PACKET_FLAG_COLUMNS
    ]

    payload_mean = np.zeros(packet_count, dtype=np.float32)
    payload_std = np.zeros(packet_count, dtype=np.float32)
    payload_nonzero = np.zeros(packet_count, dtype=np.float32)
    for idx, payload_hex in enumerate(payloads):
        mean, std, _min_v, _max_v, nonzero = _payload_stats(payload_hex)
        payload_mean[idx] = mean / 255.0
        payload_std[idx] = std / 128.0
        payload_nonzero[idx] = nonzero

    flag_sum = np.zeros(packet_count, dtype=np.float32)
    for arr in flag_arrays:
        flag_sum += arr
    flag_sum = np.clip(flag_sum / max(len(PACKET_FLAG_COLUMNS), 1), 0.0, 1.0)

    payload_size_norm = _normalize(payload_size)
    transport_size_norm = _normalize(transport_size)
    ip_size_norm = _normalize(ip_size)
    delta_time_norm = _normalize(delta_time, invert=True)
    direction_norm = _normalize(packet_direction)

    score = (
        0.24 * payload_size_norm
        + 0.14 * transport_size_norm
        + 0.10 * ip_size_norm
        + 0.16 * delta_time_norm
        + 0.14 * flag_sum
        + 0.14 * payload_nonzero
        + 0.06 * payload_std
        + 0.02 * direction_norm
    )
    score = np.clip(score + 0.02 * payload_mean, 0.0, 1.0)
    return score.astype(float).tolist()


def select_packet_indices(scores: Sequence[float], config: PacketSelectionConfig) -> list[int]:
    if not scores:
        return []

    score_arr = np.asarray(scores, dtype=np.float32)
    packet_count = int(score_arr.size)

    if config.strategy == "threshold":
        selected = np.flatnonzero(score_arr >= float(config.score_threshold)).tolist()
        if not selected:
            selected = [int(score_arr.argmax())]
    else:
        target_k = int(round(packet_count * float(config.selection_ratio)))
        k = max(int(config.min_packets), target_k)
        if config.max_packets is not None:
            k = min(k, int(config.max_packets))
        k = max(1, min(k, packet_count))
        selected = np.argsort(-score_arr)[:k].tolist()

    if config.keep_original_order:
        selected = sorted(set(int(i) for i in selected))
    else:
        selected = [int(i) for i in dict.fromkeys(int(i) for i in selected)]
    return selected


def _select_values(values: Sequence[str], indices: Sequence[int], fallback: str = "0") -> list[str]:
    out: list[str] = []
    if not values:
        return out
    for idx in indices:
        if 0 <= idx < len(values):
            out.append(str(values[idx]))
        else:
            out.append(fallback)
    return out


def select_packets_for_record(
    record: dict[str, object],
    config: PacketSelectionConfig,
) -> tuple[dict[str, object], dict[str, object]]:
    scores = score_packets_heuristic(record)
    selected_indices = select_packet_indices(scores, config)
    payloads = _parse_list_cell(record.get("udps.payload_data"))
    selected_scores = [scores[i] for i in selected_indices if 0 <= i < len(scores)]

    updated = dict(record)
    for column in PACKET_COLUMNS:
        values = _parse_list_cell(record.get(column))
        fallback = "" if column == "udps.payload_data" else "0"
        updated[column] = json.dumps(_select_values(values, selected_indices, fallback=fallback), ensure_ascii=False)

    updated["innov1.packet_selection_strategy"] = config.strategy
    updated["innov1.packet_selection_ratio"] = float(config.selection_ratio)
    updated["innov1.packet_score_threshold"] = float(config.score_threshold)
    updated["innov1.original_packet_count"] = int(len(payloads))
    updated["innov1.selected_packet_count"] = int(len(selected_indices))
    updated["innov1.selected_packet_indices"] = json.dumps([int(i) for i in selected_indices], ensure_ascii=False)
    updated["innov1.selected_packet_scores"] = json.dumps(
        [float(score) for score in selected_scores],
        ensure_ascii=False,
    )
    updated["innov1.packet_scores"] = json.dumps([float(score) for score in scores], ensure_ascii=False)

    meta = {
        "original_packet_count": int(len(payloads)),
        "selected_packet_count": int(len(selected_indices)),
        "selection_ratio": float(len(selected_indices) / max(len(payloads), 1)),
    }
    return updated, meta


def select_packets_frame(
    frame: pd.DataFrame,
    config: PacketSelectionConfig,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, object]] = []
    total_original = 0
    total_selected = 0

    for record in frame.to_dict(orient="records"):
        updated, meta = select_packets_for_record(record, config)
        rows.append(updated)
        total_original += int(meta["original_packet_count"])
        total_selected += int(meta["selected_packet_count"])

    out = pd.DataFrame(rows, columns=list(rows[0].keys()) if rows else list(frame.columns))
    summary = {
        "rows": float(len(out)),
        "original_packets": float(total_original),
        "selected_packets": float(total_selected),
        "average_selected_per_row": float(total_selected / max(len(out), 1)),
        "average_selection_ratio": float(total_selected / max(total_original, 1)),
    }
    return out, summary


def process_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    config: PacketSelectionConfig,
    *,
    chunksize: int = 2048,
) -> dict[str, float]:
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    first_chunk = True
    summary = {
        "rows": 0.0,
        "original_packets": 0.0,
        "selected_packets": 0.0,
    }

    for chunk in pd.read_csv(input_csv, chunksize=chunksize):
        transformed, chunk_summary = select_packets_frame(chunk, config)
        transformed.to_csv(output_csv, index=False, mode="w" if first_chunk else "a", header=first_chunk)
        first_chunk = False
        summary["rows"] += chunk_summary["rows"]
        summary["original_packets"] += chunk_summary["original_packets"]
        summary["selected_packets"] += chunk_summary["selected_packets"]

    summary["average_selected_per_row"] = summary["selected_packets"] / max(summary["rows"], 1.0)
    summary["average_selection_ratio"] = summary["selected_packets"] / max(summary["original_packets"], 1.0)
    summary["config"] = asdict(config)
    summary["input_csv"] = str(input_csv)
    summary["output_csv"] = str(output_csv)
    return summary
