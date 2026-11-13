"""Audit the chronological Packet chain and summarize observed edge fields."""
from collections import Counter

import numpy as np

from xgnid.model import FLOW_TO_PACKET, PACKET_TO_PACKET


DIRECTION_GROUPS = ('0_to_0', '0_to_1', '1_to_0', '1_to_1')
STAT_NAMES = [f'{group}_{field}' for group in DIRECTION_GROUPS
              for field in ('count', 'log_dt_mean', 'log_dt_std', 'signed_log_payload_change_mean')]


def audit_batch(batch):
    """Return one audit record and one feature vector per graph, in batch order."""
    packet = batch['packet']
    owner = packet.batch.numpy()
    sizes = np.diff(packet.ptr.numpy())
    contain = batch[FLOW_TO_PACKET]
    links = batch[PACKET_TO_PACKET]
    flows, packets = contain.edge_index.numpy()
    src, dst = links.edge_index.numpy()
    attrs = contain.edge_attr.numpy()
    dt = links.edge_attr.numpy().reshape(-1)
    if attrs.shape != (len(packets), 4) or len(dt) != len(src):
        raise ValueError('Invalid contain or link field layout')
    if (np.any(packets < 0) or np.any(packets >= len(owner)) or np.any(src < 0)
            or np.any(src >= len(owner)) or np.any(dst < 0) or np.any(dst >= len(owner))
            or np.any(flows < 0) or np.any(flows >= batch['flow'].num_nodes)):
        raise ValueError('Packet transition endpoint outside batch')
    same_owner = batch['flow'].batch.numpy()[flows] == owner[packets]
    contained = np.bincount(packets[same_owner], minlength=len(owner))
    direction = np.full(len(owner), np.nan)
    payload = np.full(len(owner), np.nan)
    good_rows = same_owner & (contained[packets] == 1)
    direction[packets[good_rows]] = attrs[good_rows, 0]
    payload[packets[good_rows]] = attrs[good_rows, 3]
    directions_ok = (direction == 0) | (direction == 1)
    payloads_ok = np.isfinite(payload) & (payload >= 0) & (payload == np.floor(payload))
    edge_owner = owner[src]
    same_flow = edge_owner == owner[dst]
    forward = same_flow & (dst == src + 1)
    reverse = same_flow & (dst == src - 1)
    forward_counts = np.bincount(src[forward], minlength=len(owner))
    unique = forward & (forward_counts[src] == 1)
    time_ok = np.isfinite(dt) & (dt >= 0) & (dt == np.floor(dt))
    edge_direction_ok = directions_ok[src] & directions_ok[dst]
    edge_payload_ok = payloads_ok[src] & payloads_ok[dst]
    valid = unique & time_ok & edge_direction_ok & edge_payload_ok

    audits, features = [], []
    for i, size in enumerate(sizes):
        belongs = edge_owner == i
        keep = belongs & valid
        times = dt[keep].astype(np.float64)
        before, after = direction[src[keep]], direction[dst[keep]]
        payload_from, payload_to = payload[src[keep]], payload[dst[keep]]
        changes = np.log1p(payload_to) - np.log1p(payload_from)
        pair_type = 2 * before + after
        groups = [pair_type == group for group in range(4)]
        counts = [int(group.sum()) for group in groups]
        stats = []
        log_times = np.log1p(times)
        for group, count in zip(groups, counts):
            stats.extend((count, float(log_times[group].mean()) if count else 0.0,
                          float(log_times[group].std()) if count else 0.0,
                          float(changes[group].mean()) if count else 0.0))
        nodes = owner == i
        audits.append({
            'packets': int(size), 'expected_pairs': max(int(size) - 1, 0), 'links': int(belongs.sum()),
            'cross_flow_links': int((belongs & ~same_flow).sum()),
            'nonadjacent_links': int((belongs & same_flow & ~(forward | reverse)).sum()),
            'reverse_adjacent_links': int((belongs & reverse).sum()),
            'forward_adjacent_links': int((belongs & forward).sum()),
            'missing_forward_pairs': max(int(size) - 1, 0) - int((nodes & (forward_counts > 0)).sum()),
            'duplicate_forward_links': int((belongs & forward & ~unique).sum()),
            'invalid_contain_ownership_edges': int((~same_owner & (owner[packets] == i)).sum()),
            'missing_or_duplicate_contain_packets': int((nodes & (contained != 1)).sum()),
            'invalid_direction_packets': int((nodes & ~directions_ok).sum()),
            'invalid_payload_packets': int((nodes & ~payloads_ok).sum()),
            'invalid_time_forward_links': int((belongs & forward & ~time_ok).sum()),
            'invalid_direction_forward_links': int((belongs & forward & ~edge_direction_ok).sum()),
            'invalid_payload_forward_links': int((belongs & forward & ~edge_payload_ok).sum()),
            'valid_pairs': int(keep.sum()), 'same_direction_pairs': counts[0] + counts[3],
            'direction_change_pairs': counts[1] + counts[2], 'direction_groups': counts,
            'zero_dt_pairs': int((times == 0).sum()),
            'zero_payload_endpoints': int((payload_from == 0).sum() + (payload_to == 0).sum()),
            'valid_dt_min_ms': float(times.min()) if len(times) else None,
            'valid_dt_max_ms': float(times.max()) if len(times) else None,
            'valid_payload_min_bytes': float(min(payload_from.min(), payload_to.min())) if len(times) else None,
            'valid_payload_max_bytes': float(max(payload_from.max(), payload_to.max())) if len(times) else None,
        })
        features.append(stats)
    return audits, np.asarray(features, dtype=np.float32)


def aggregate(records):
    counts = Counter()
    groups = np.zeros(4, dtype=np.int64)
    ranges = {}
    for row in records:
        counts.update({key: value for key, value in row.items() if type(value) is int and key != 'label'})
        counts['graphs'] += 1
        counts['graphs_without_valid_pairs'] += int(row['valid_pairs'] == 0)
        counts['graphs_with_at_least_two_valid_pairs'] += int(row['valid_pairs'] >= 2)
        counts['graphs_with_direction_change'] += int(row['direction_change_pairs'] > 0)
        groups += row['direction_groups']
        for key in ('valid_dt_min_ms', 'valid_dt_max_ms', 'valid_payload_min_bytes', 'valid_payload_max_bytes'):
            if row[key] is not None:
                bound = min if '_min_' in key else max
                ranges[key] = bound(ranges.get(key, row[key]), row[key])
    graphs = counts['graphs']
    return {**counts, **ranges, 'direction_groups': dict(zip(DIRECTION_GROUPS, groups.tolist())),
            'valid_pair_ratio': counts['valid_pairs'] / counts['expected_pairs'] if counts['expected_pairs'] else None,
            'graph_coverage_ge1': (graphs - counts['graphs_without_valid_pairs']) / graphs if graphs else None,
            'graph_coverage_ge2': counts['graphs_with_at_least_two_valid_pairs'] / graphs if graphs else None}
