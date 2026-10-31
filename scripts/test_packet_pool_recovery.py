"""No-torch checks for the exact pre-training failure and cache-only recovery."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import recover_packet_pool_pair as recovery
import run_packet_pool_pair as run
from summarize_packet_pool_pair import analyze
import test_packet_pool_pair as contract_tests


class Recovery(unittest.TestCase):
    def fixture(self, directory):
        source = Path(directory)
        c = run.read(Path(__file__).resolve().parents[1] / 'configs/packet_pool_pair_s42_v1.json')
        ref = {'binding': {'split_id': 'fixed'}}
        previous = [None, (None, None, ref), None,
                    {'binding': {'encoder_source': {'encoder_sha256': 'enc'}}}]
        binding = {'config_sha256': recovery.ORIGINAL_CONFIG, 'source_pair_binding': {'fixed': True},
                   'source': {'directory': '/mature_D32'}, 'sources': {'new': 'implementation'}}
        meta = (c, previous, None, None, binding)
        for name in recovery.REQUIRED:
            (source / name).write_bytes(b'prepared artifact')
        run.write(source / 'status.json', {'status': 'failed', 'exit_code': 1})
        run.write(source / 'config.json', c)
        run.write(source / 'binding.json', {**binding, 'sources': recovery.ORIGINAL_SOURCES})
        (source / 'error.log').write_text('in initial_equivalence\n' + recovery.FAILURE + '\n', encoding='utf-8')
        receipts, memory = {}, {}
        for part, size in (('train', 107621), ('val', 13446)):
            memory[part] = {'graph_samples': size}
            receipts[part] = {'metadata': {'schema': 'xgnid.packet-node64-cache.v1', 'partition': part,
                'samples': size, 'encoder_sha256': 'enc', 'split_binding': ref['binding'],
                'sample_order_sha256': 'ordered'}, 'sha256': run.sha(source / (part + '_packet_nodes.pt')),
                'tensor_sha256': 'content', 'graph_load_stage': run.graph_load_stage('extract_' + part, part, size, 2),
                'memory_estimate': memory[part], 'wall_seconds': 1.}
        run.write(source / 'cache_receipts.json', receipts)
        run.write(source / 'cache_memory_estimates.json', memory)
        return source, meta

    def test_known_failure_reuses_only_prepared_cache_and_preserves_every_source_file(self):
        with tempfile.TemporaryDirectory() as directory:
            source, meta = self.fixture(directory)
            before = {p.name: p.read_bytes() for p in source.iterdir()}
            receipt = recovery.source_receipt(source, meta)
            recovery.assert_source_unchanged(receipt)
            self.assertEqual(receipt['formal_training_updates_before_failure'], 0)
            self.assertEqual(receipt['probe_updates_before_failure'], 0)
            self.assertEqual(receipt['reused_graph_samples'], {'train': 107621, 'val': 13446})
            self.assertEqual(before, {p.name: p.read_bytes() for p in source.iterdir()})

    def test_different_failure_and_any_later_phase_are_rejected(self):
        for marker in ('P0', 'P1', 'summary.json', 'initialization.json', 'probes.json', 'P0_preflight.pt'):
            with self.subTest(marker=marker), tempfile.TemporaryDirectory() as directory:
                source, meta = self.fixture(directory)
                if marker in ('P0', 'P1'):
                    (source / marker).mkdir()
                else:
                    (source / marker).write_bytes(b'possible training progress')
                with self.assertRaises(ValueError):
                    recovery.source_receipt(source, meta)
        with tempfile.TemporaryDirectory() as directory:
            source, meta = self.fixture(directory)
            (source / 'error.log').write_text('CUDA out of memory', encoding='utf-8')
            with self.assertRaises(ValueError):
                recovery.source_receipt(source, meta)

    def test_changed_cache_config_and_encoder_identity_are_rejected(self):
        for change in ('cache', 'config', 'binding', 'encoder'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                source, meta = self.fixture(directory)
                if change == 'cache':
                    (source / 'train_packet_nodes.pt').write_bytes(b'changed')
                elif change == 'config':
                    run.write(source / 'config.json', {**meta[0], 'head_lr': .001})
                elif change == 'binding':
                    run.write(source / 'binding.json', {**meta[-1], 'sources': {}})
                else:
                    receipts = run.read(source / 'cache_receipts.json')
                    receipts['val']['metadata']['encoder_sha256'] = 'wrong'
                    run.write(source / 'cache_receipts.json', receipts)
                with self.assertRaises(ValueError):
                    recovery.source_receipt(source, meta)

    def test_source_mutation_after_receipt_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source, meta = self.fixture(directory)
            receipt = recovery.source_receipt(source, meta)
            (source / 'status.json').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                recovery.assert_source_unchanged(receipt)

    def test_reuse_provider_copies_and_checks_caches_without_loading_a_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'failed', Path(directory) / 'recovery'
            source.mkdir()
            output.mkdir()
            source, meta = self.fixture(source)
            receipt = recovery.source_receipt(source, meta)
            cached = {'metadata': receipt['cache_receipts']['train']['metadata']}
            core = SimpleNamespace(torch=SimpleNamespace(load=lambda *a, **k: cached), select_rows=lambda *a: ['row'])
            with patch.object(run, 'validate_cache') as validate, patch.object(run, 'tensor_hash', return_value='content'), \
                    patch.object(run.dim, 'assert_frozen'), patch.object(run.dim, 'progress'):
                cache, r = recovery.cache_provider(receipt)(core, SimpleNamespace(output=output), None,
                    SimpleNamespace(load=lambda *a: self.fail('Graph re-encoding is forbidden')), None,
                    'train', meta[0], 'enc', {'split_id': 'fixed'})
                self.assertIs(cache, cached)
                self.assertTrue(r['reused'])
                self.assertEqual(run.sha(output / 'train_packet_nodes.pt'), r['sha256'])
                validate.assert_called_once()
            recovery.assert_source_unchanged(receipt)

    def test_recovery_load_accounting_does_not_count_old_extraction_again(self):
        r, _ = contract_tests.Contracts().fixture()
        r['cache_reused'] = True
        r['cache_recovery'] = {'formal_training_updates_before_failure': 0, 'probe_updates_before_failure': 0}
        stages = r['graph_load_accounting']['loader_stages'][2:]
        r['graph_load_accounting'] = run.graph_load_accounting({'train': 96, 'val': 0, 'test': 0}, stages)
        r['graph_load_counts'] = r['graph_load_accounting']['totals']
        with patch.object(run, 'check_config'):
            a = analyze(r)
            self.assertTrue(a['cache_reused'])
            self.assertEqual(a['graph_load_accounting']['totals'], {'train': 96, 'val': 53784, 'test': 0})
            bad = copy.deepcopy(r)
            bad['cache_recovery']['formal_training_updates_before_failure'] = 1
            with self.assertRaises(ValueError):
                analyze(bad)


if __name__ == '__main__':
    unittest.main()
