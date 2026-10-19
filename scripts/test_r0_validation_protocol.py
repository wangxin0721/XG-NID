"""Stdlib-only protocol tests. No torch, model execution or real graph reads."""
from collections import Counter
from pathlib import Path
import tempfile
import unittest

import run_r0_validation_server as protocol


def fixture():
    rows = []
    for label in range(8):
        for group in range(15):
            for copy in range(1 + group % 4):
                index = len(rows)
                rows.append({"sample_id": str(index), "partition": "train", "label": label,
                             "node_hash": f"node:{label}:{group}", "edge_hash": f"edge:{label}:{group}:{copy % 2}",
                             "path": f"data_{index}.pt", "container_offset": 0})
    return rows


class ValidationProtocolTests(unittest.TestCase):
    def test_holdout_preserves_all_members_and_keeps_duplicates_and_variants_together(self):
        rows = fixture()
        selected, details = protocol.grouped_holdout(rows)
        again, _ = protocol.grouped_holdout(list(reversed(rows)))
        self.assertEqual({r['sample_id']: r['partition'] for r in selected},
                         {r['sample_id']: r['partition'] for r in again})
        self.assertEqual(Counter(r['sample_id'] for r in selected), Counter(r['sample_id'] for r in rows))
        parts = {}
        for row in selected:
            self.assertEqual(parts.setdefault(row['node_hash'], row['partition']), row['partition'])
        for part in ('train', 'val'):
            self.assertEqual({r['label'] for r in selected if r['partition'] == part}, set(range(8)))
        for label, detail in details.items():
            actual = sum(r['label'] == int(label) and r['partition'] == 'val' for r in selected)
            self.assertEqual(actual, detail['actual_val_samples'])
            self.assertLessEqual(abs(actual - detail['target_val_samples']), 4)
        self.assertTrue(all(r['partition'] == 'train' for r in rows))

    def test_rejects_test_members_label_conflicts_duplicate_ids_and_missing_classes(self):
        rows = fixture()
        with self.assertRaisesRegex(ValueError, 'Only source TRAIN'):
            protocol.grouped_holdout([dict(r, partition='test') for r in rows])
        with self.assertRaisesRegex(ValueError, 'Conflicting labels'):
            protocol.grouped_holdout(rows + [dict(rows[0], sample_id='conflict', label=1)])
        with self.assertRaisesRegex(ValueError, 'Repeated sample ID'):
            protocol.grouped_holdout(rows + [rows[0]])
        with self.assertRaisesRegex(ValueError, 'misses a class'):
            protocol.grouped_holdout([r for r in rows if r['label'] != 7])

    def test_early_stop_ties_earliest_best_reset_and_exact_patience(self):
        selector = protocol.Selection(12)
        self.assertEqual(selector.observe(1, .6), (True, False))
        for epoch in range(2, 13):
            self.assertEqual(selector.observe(epoch, .6), (False, False))
        self.assertEqual(selector.observe(13, .59), (False, True))
        self.assertEqual(selector.best_epoch, 1)
        self.assertEqual(selector.observe(14, .61), (True, False))
        self.assertEqual((selector.best_epoch, selector.stale), (14, 0))
        for invalid in (float('nan'), float('inf'), -.1, 1.1):
            with self.assertRaises(ValueError):
                selector.observe(15, invalid)

    def test_reviewed_budget_stops_at_patience_and_keeps_best(self):
        config = protocol.configuration(Path(__file__).resolve().parents[1] / 'configs/r0_validation.json')
        self.assertEqual((config['max_epochs'], config['early_stopping'], config['patience']), (100, True, 12))
        selector = protocol.Selection(config['patience'])
        for epoch in range(1, config['max_epochs'] + 1):
            _, stop = selector.observe(epoch, .9 if epoch == 2 else .7)
            if stop:
                break
        self.assertEqual(selector.best_epoch, 2)
        self.assertEqual((epoch, selector.stale), (14, 12))

    def test_singleton_merge_keeps_every_sample(self):
        for count in (2, 63, 64, 65, 128, 129, 144001):
            order = list(range(count))
            result = protocol.batches(order, 64)
            self.assertEqual([value for batch in result for value in batch], order)
            self.assertTrue(all(2 <= len(batch) <= 65 for batch in result))

    def test_full_run_requires_matching_cuda_smoke_and_both_checkpoint_hashes(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'tmp') as folder:
            root = Path(folder)
            for name in ('best', 'last'):
                (root / (name + '.pt')).write_bytes(name.encode())
            binding = {'runtime': {'device_type': 'cuda'}, 'split_id': 'fixture'}
            report = {'status': 'completed', 'exit_code': 0, 'mode': 'smoke', 'epochs_completed': 2,
                      'binding': binding, 'optimizer_step_passed': True, 'test': None, 'test_passes': 0,
                      'best_checkpoint_sha256': protocol.sha(root / 'best.pt'),
                      'last_checkpoint_sha256': protocol.sha(root / 'last.pt')}
            path = root / 'summary.json'
            protocol.write(path, report)
            protocol.verify_smoke(path, binding)
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                protocol.verify_smoke(path, dict(binding, split_id='different'))
            report['test_passes'] = 1
            protocol.write(path, report)
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                protocol.verify_smoke(path, binding)
            report['test_passes'] = 0
            protocol.write(path, report)
            (root / 'best.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checkpoint changed'):
                protocol.verify_smoke(path, binding)


if __name__ == '__main__':
    unittest.main()
