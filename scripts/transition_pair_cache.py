"""Frozen graph features plus reproducible local pairing indices, never fusion outputs."""
from collections import Counter
import hashlib
import json
from typing import NamedTuple

import torch

import joint_transition_training as training
from xgnid.transition_pair_shuffle import VERSION, derangement

original, raw, require = training.original, training.raw, training.require


class PairFeatures(NamedTuple):
    flow: torch.Tensor
    packet_mean: torch.Tensor
    packet: torch.Tensor
    transitions: torch.Tensor
    lengths: torch.Tensor
    shuffled_right: torch.Tensor


def mapping(features, sample_ids, partition, seed):
    require(len(sample_ids) == len(features.lengths), 'Pair mapping identity count differs')
    records, rights, offset = [], [], 0
    for sample_id, length in zip(sample_ids, features.lengths.cpu().tolist()):
        transitions = features.transitions[offset:offset + length]
        left = torch.where(transitions[:, 70].bool())[0]
        true_right = left + 1
        require(not len(left) or int(true_right[-1]) < length, 'Context crosses a flow boundary')
        permutation = derangement(sample_id, partition, len(left), seed)
        shuffled = true_right[torch.tensor(permutation, dtype=torch.long, device=true_right.device)]
        changed = (transitions[true_right, :70] != transitions[shuffled, :70]).any(-1)
        records.append({'sample_id': sample_id, 'partition': partition, 'valid_pairs': length,
                        'ordered_contexts': len(left), 'left_local': left.cpu().tolist(),
                        'true_right_local': true_right.cpu().tolist(), 'permutation': permutation,
                        'shuffled_right_local': shuffled.cpu().tolist(), 'shuffleable': len(left) >= 2,
                        'indices_changed': sum(i != value for i, value in enumerate(permutation)),
                        'features_changed': int(changed.sum())})
        rights.append(shuffled + offset)
        offset += length
    require(offset == len(features.transitions), 'Pair mapping transition count differs')
    right = torch.cat(rights) if rights else torch.empty(0, dtype=torch.long, device=features.transitions.device)
    return PairFeatures(*features, right), records


def coverage(records, rows):
    def counts(items):
        n = len(items)
        eligible = [r for r in items if r['shuffleable']]
        contexts = sum(r['ordered_contexts'] for r in items)
        indices = sum(r['indices_changed'] for r in items)
        changed = sum(r['features_changed'] for r in items)
        effective = sum(r['features_changed'] > 0 for r in items)
        return {'samples': n, 'shuffleable_flows': len(eligible),
                'shuffleable_flow_fraction': len(eligible) / n if n else None,
                'zero_context_flows': sum(r['ordered_contexts'] == 0 for r in items),
                'single_context_flows': sum(r['ordered_contexts'] == 1 for r in items),
                'contexts': contexts, 'changed_pair_indices': indices, 'changed_right_features': changed,
                'feature_change_fraction_of_changed_indices': changed / indices if indices else None,
                'flows_with_changed_features': effective,
                'effective_flow_fraction_of_shuffleable': effective / len(eligible) if eligible else None}
    result = counts(records)
    result['by_class'] = {label: counts([r for r, row in zip(records, rows) if row['label'] == i])
                          for i, label in enumerate(original.LABELS)}
    return result


class PairCache:
    def __init__(self, base, rows, partition, seed):
        self.base, self.labels, self.rows = base, base.labels, rows
        self.partition, self.seed = partition, seed
        self.features = base.features  # Only the frozen five-feature representation.
        _, self.records = mapping(self.features, [r['sample_id'] for r in rows], partition, seed)
        self.mapping_sha256 = hashlib.sha256(json.dumps(
            self.records, separators=(',', ':'), sort_keys=True).encode()).hexdigest()
        self.coverage = coverage(self.records, rows)

    def batch(self, indices, device):
        indices = list(indices)
        features, labels = self.base.batch(indices, device)
        rights, offset = [], 0
        for index in indices:
            record = self.records[index]
            rights.extend(value + offset for value in record['shuffled_right_local'])
            offset += record['valid_pairs']
        return PairFeatures(*features, torch.tensor(rights, dtype=torch.long, device=device)), labels

    def save_mapping(self, output):
        raw.save_predictions(output / ('pairing_' + self.partition + '.jsonl.gz'), self.records)
        raw.write(output / ('pairing_' + self.partition + '_receipt.json'), {
            'version': VERSION, 'seed': self.seed, 'partition': self.partition,
            'mapping_sha256': self.mapping_sha256, 'samples': len(self.records),
            'label_used_for_mapping': False, 'coverage': self.coverage})
