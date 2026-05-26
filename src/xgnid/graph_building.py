from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData


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


def _parse_list_cell(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        text = text.strip("[]")
        if not text:
            return []
        return [item.strip().strip("'\"") for item in text.split(",")]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [str(parsed)]


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _flow_features(row: pd.Series) -> torch.Tensor:
    drop_cols = set(PACKET_COLUMNS) | {"Label"}
    values = [float(_safe_float(v)) for col, v in row.items() if col not in drop_cols]
    feats = torch.tensor(values, dtype=torch.float32).view(1, -1)
    return feats


def _packet_payload_to_bytes(payload_hex: str, dims: int = 1500) -> np.ndarray:
    if not payload_hex or payload_hex in {"0", "00"}:
        return np.zeros(dims, dtype=np.float32)
    try:
        payload_bytes = bytes.fromhex(payload_hex)
    except ValueError:
        return np.zeros(dims, dtype=np.float32)
    byte_list = list(payload_bytes)
    if len(byte_list) < dims:
        byte_list = byte_list + [0] * (dims - len(byte_list))
    else:
        byte_list = byte_list[:dims]
    return np.asarray(byte_list, dtype=np.float32)


def _packet_features(row: pd.Series) -> torch.Tensor:
    payloads = _parse_list_cell(row.get("udps.payload_data"))
    packet_count = len(payloads)
    if packet_count == 0:
        return torch.zeros((0, 1508), dtype=torch.float32)

    delta_time = _parse_list_cell(row.get("udps.delta_time"))
    direction = _parse_list_cell(row.get("udps.packet_direction"))
    ip_size = _parse_list_cell(row.get("udps.ip_size"))
    transport_size = _parse_list_cell(row.get("udps.transport_size"))
    payload_size = _parse_list_cell(row.get("udps.payload_size"))
    syn = _parse_list_cell(row.get("udps.syn"))
    cwr = _parse_list_cell(row.get("udps.cwr"))
    ece = _parse_list_cell(row.get("udps.ece"))
    urg = _parse_list_cell(row.get("udps.urg"))
    ack = _parse_list_cell(row.get("udps.ack"))
    psh = _parse_list_cell(row.get("udps.psh"))
    rst = _parse_list_cell(row.get("udps.rst"))
    fin = _parse_list_cell(row.get("udps.fin"))

    packet_rows: list[np.ndarray] = []
    for idx, payload_hex in enumerate(payloads):
        byte_feats = _packet_payload_to_bytes(payload_hex)
        packet_row = np.concatenate(
            [
                np.asarray(
                    [
                        _safe_float(direction[idx]) if idx < len(direction) else 0.0,
                        _safe_float(ip_size[idx]) if idx < len(ip_size) else 0.0,
                        _safe_float(transport_size[idx]) if idx < len(transport_size) else 0.0,
                        _safe_float(payload_size[idx]) if idx < len(payload_size) else 0.0,
                        _safe_float(delta_time[idx]) if idx < len(delta_time) else 0.0,
                        _safe_float(syn[idx]) if idx < len(syn) else 0.0,
                        _safe_float(cwr[idx]) if idx < len(cwr) else 0.0,
                        _safe_float(ece[idx]) if idx < len(ece) else 0.0,
                        _safe_float(urg[idx]) if idx < len(urg) else 0.0,
                        _safe_float(ack[idx]) if idx < len(ack) else 0.0,
                        _safe_float(psh[idx]) if idx < len(psh) else 0.0,
                        _safe_float(rst[idx]) if idx < len(rst) else 0.0,
                        _safe_float(fin[idx]) if idx < len(fin) else 0.0,
                    ],
                    dtype=np.float32,
                ),
                byte_feats,
            ]
        )
        packet_rows.append(packet_row)
    return torch.tensor(np.asarray(packet_rows, dtype=np.float32), dtype=torch.float32)


def _contain_edge_attr(row: pd.Series) -> torch.Tensor:
    payloads = _parse_list_cell(row.get("udps.payload_data"))
    edge_rows = []
    for idx in range(len(payloads)):
        edge_rows.append(
            [
                _safe_float(_parse_list_cell(row.get("udps.packet_direction"))[idx]) if idx < len(_parse_list_cell(row.get("udps.packet_direction"))) else 0.0,
                _safe_float(_parse_list_cell(row.get("udps.ip_size"))[idx]) if idx < len(_parse_list_cell(row.get("udps.ip_size"))) else 0.0,
                _safe_float(_parse_list_cell(row.get("udps.transport_size"))[idx]) if idx < len(_parse_list_cell(row.get("udps.transport_size"))) else 0.0,
                _safe_float(_parse_list_cell(row.get("udps.payload_size"))[idx]) if idx < len(_parse_list_cell(row.get("udps.payload_size"))) else 0.0,
            ]
        )
    return torch.tensor(np.asarray(edge_rows, dtype=np.float32), dtype=torch.float32)


def _link_edge_attr(row: pd.Series) -> torch.Tensor:
    deltas = _parse_list_cell(row.get("udps.delta_time"))
    if len(deltas) <= 1:
        return torch.zeros((0, 1), dtype=torch.float32)
    vals = np.asarray([_safe_float(v) for v in deltas[1:]], dtype=np.float32).reshape(-1, 1)
    return torch.tensor(vals, dtype=torch.float32)


def _edge_index(num_packets: int) -> torch.Tensor:
    if num_packets <= 0:
        return torch.zeros((2, 0), dtype=torch.long)
    src = torch.zeros(num_packets, dtype=torch.long)
    dst = torch.arange(num_packets, dtype=torch.long)
    return torch.stack([src, dst], dim=0)


def _link_edge_index(num_packets: int) -> torch.Tensor:
    if num_packets <= 1:
        return torch.zeros((2, 0), dtype=torch.long)
    src = torch.arange(0, num_packets - 1, dtype=torch.long)
    dst = torch.arange(1, num_packets, dtype=torch.long)
    return torch.stack([src, dst], dim=0)


def build_graph_from_row(row: pd.Series) -> HeteroData:
    data = HeteroData()
    flow_x = _flow_features(row)
    packet_x = _packet_features(row)
    num_packets = int(packet_x.size(0))

    data["flow"].x = flow_x
    data["packet"].x = packet_x
    data["flow", "contain", "packet"].edge_index = _edge_index(num_packets)
    data["packet", "rev_contain", "flow"].edge_index = data["flow", "contain", "packet"].edge_index.flip(0)
    data["packet", "link", "packet"].edge_index = _link_edge_index(num_packets)
    data["flow", "contain", "packet"].edge_attr = _contain_edge_attr(row)
    data["packet", "rev_contain", "flow"].edge_attr = _contain_edge_attr(row)
    data["packet", "link", "packet"].edge_attr = _link_edge_attr(row)
    data.y = torch.tensor([int(_safe_float(row["Label"]))], dtype=torch.long)
    return data


def build_graphs_from_csv(csv_path: str | Path) -> list[HeteroData]:
    frame = pd.read_csv(csv_path)
    graphs = [build_graph_from_row(row) for _, row in frame.iterrows()]
    return graphs


def save_graphs(graphs: Sequence[HeteroData], output_dir: str | Path, prefix: str) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for idx, graph in enumerate(graphs):
        file_path = output_dir / f"{prefix}_{idx}.pt"
        torch.save([graph], file_path)
        saved.append(file_path)
    return saved
