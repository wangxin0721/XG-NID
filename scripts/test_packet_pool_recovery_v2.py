"""No-torch provenance and progress protection for the second numeric recovery."""
import copy
from pathlib import Path
import tempfile
import unittest

import recover_packet_pool_pair as first
import recover_packet_pool_pair_v2 as second
import run_packet_pool_pair as run
import test_packet_pool_recovery as original_tests


class RecoveryV2(unittest.TestCase):
    def fixture(self, directory):
        original, previous = Path(directory) / 'original', Path(directory) / 'previous'
        original.mkdir()
        previous.mkdir()
        _, meta = original_tests.Recovery().fixture(original)
        receipt = first.source_receipt(original, meta)
        for name in first.REQUIRED:
            (previous / name).write_bytes((original / name).read_bytes())
        expected = copy.deepcopy(meta[-1])
        expected['sources'] = second.PREVIOUS_SOURCES
        expected['cache_recovery'] = receipt
        run.write(previous / 'binding.json', expected)
        run.write(previous / 'recovery_receipt.json', receipt)
        run.write(previous / 'initialization_comparison.json', {'samples': 13446, 'atol': 1e-6, 'rtol': 1e-6,
            'new_failed_batches': 1, 'new_prediction_mismatches': 0,
            'max_abs_new_vs_P0_log_probability_error': 2.288818359375e-5})
        caches = run.read(previous / 'cache_receipts.json')
        for cache in caches.values():
            cache['reused'] = True
        run.write(previous / 'cache_receipts.json', caches)
        return original, previous, meta, receipt

    def test_two_failed_runs_are_read_only_and_both_snapshots_are_audited(self):
        with tempfile.TemporaryDirectory() as directory:
            original, previous, meta, receipt = self.fixture(directory)
            snapshots = {p: {f.name: f.read_bytes() for f in p.iterdir()} for p in (original, previous)}
            prior = second.previous_failure_receipt(previous, meta, receipt)
            first.assert_source_unchanged(receipt)
            first.assert_source_unchanged(prior)
            self.assertEqual(prior['formal_training_updates_before_failure'], 0)
            for p in snapshots:
                self.assertEqual(snapshots[p], {f.name: f.read_bytes() for f in p.iterdir()})

    def test_rejects_wrong_binding_changed_copied_cache_or_non_numeric_failure(self):
        for kind in ('binding', 'cache', 'error', 'predictions', 'complete'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                original, previous, meta, receipt = self.fixture(directory)
                if kind == 'binding':
                    binding = run.read(previous / 'binding.json')
                    binding['sources'] = {}
                    run.write(previous / 'binding.json', binding)
                elif kind == 'cache':
                    (previous / 'train_packet_nodes.pt').write_bytes(b'changed')
                elif kind == 'error':
                    (previous / 'error.log').write_text('unrelated failure', encoding='utf-8')
                elif kind == 'complete':
                    run.write(previous / 'status.json', {'status': 'completed', 'exit_code': 0})
                else:
                    d = run.read(previous / 'initialization_comparison.json')
                    d['new_prediction_mismatches'] = 1
                    run.write(previous / 'initialization_comparison.json', d)
                with self.assertRaises(ValueError):
                    second.previous_failure_receipt(previous, meta, receipt)

    def test_rejects_any_formal_or_probe_progress(self):
        for marker in ('P0', 'P1', 'summary.json', 'initialization.json', 'probes.json', 'P1_preflight.pt'):
            with self.subTest(marker=marker), tempfile.TemporaryDirectory() as directory:
                original, previous, meta, receipt = self.fixture(directory)
                if marker in ('P0', 'P1'):
                    (previous / marker).mkdir()
                else:
                    (previous / marker).write_bytes(b'progress')
                with self.assertRaises(ValueError):
                    second.previous_failure_receipt(previous, meta, receipt)

    def test_subsequent_change_to_either_failed_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            original, previous, meta, receipt = self.fixture(directory)
            prior = second.previous_failure_receipt(previous, meta, receipt)
            (previous / 'error.log').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                first.assert_source_unchanged(prior)
            (original / 'status.json').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                first.assert_source_unchanged(receipt)


if __name__ == '__main__':
    unittest.main()
