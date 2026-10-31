"""No-torch source-recovery integrity, namespace and preservation checks."""
import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import recover_fusion_dimension_pair as recovery

ROOT = Path(__file__).resolve().parents[1]


class Checks(unittest.TestCase):
    def fixture(self, directory):
        source = Path(directory) / 'source'
        (source / 'D8').mkdir(parents=True)
        c = recovery.read(ROOT / 'configs/fusion_dimension_pair_s42_v1.json')
        ref = recovery.read(ROOT / 'configs/n0_seed42_reference_20261029.json')
        current = {'config_sha256': recovery.ORIGINAL_CONFIG, 'parent': 'same',
                   'sources': {k: 'new-code' for k in recovery.ORIGINAL_SOURCES}}
        enc = {'encoder_sha256': 'same-encoder', 'N0_best_sha256': c['n0_best_sha256'],
               'split_id': ref['binding']['split_id'], 'manifest_sha256': ref['binding']['manifest_sha256']}
        receipts = {}
        for part in ('train', 'val'):
            filename = part + '_encoder64.pt'
            # File-integrity fixtures only, not loadable tensors or checkpoints.
            (source / filename).write_bytes(('fixture:' + part).encode())
            receipts[part] = {'sha256': recovery.sha(source / filename), 'wall_seconds': 1.,
                'metadata': {'schema': 'xgnid.encoder64-cache.v1', 'embedding_dim': 64,
                             'partition': part, 'source': enc, 'samples': 107621 if part == 'train' else 13446},
                'N0_replayed_validation': {'confusion_matrix': ref['model']['best_validation']['confusion_matrix']}}
        init = {'architecture': {k: recovery.parent.architecture(v) for k, v in recovery.parent.VARIANTS.items()},
                'streams': recovery.parent.initialization_streams(42), 'all_heads_fresh': True,
                'shared_output_layer_initialization_identical': True, 'encoder_sha256': 'same-encoder',
                'head_state_sha256': {'D8': 'initial8', 'D32': 'initial32'}}
        probes, rounds = {}, {}
        for name in recovery.parent.VARIANTS:
            (source / (name + '_preflight.pt')).write_bytes(('fixture:' + name).encode())
            probes[name] = {'discarded': True, 'clean_state_and_rng_restored': True,
                'encoder_parameters_and_BN_unchanged': True, 'updates': [{'step': i} for i in (1, 2, 3)],
                'initial_head_sha256': init['head_state_sha256'][name]}
            rounds[name] = {'strict_reload': True, 'samples': 16,
                            'checkpoint_sha256': recovery.sha(source / (name + '_preflight.pt'))}
        binding = {**copy.deepcopy(current), 'sources': dict(recovery.ORIGINAL_SOURCES),
                   'encoder_source': enc, 'caches': receipts, 'cache_tensor_sha256': {'train': 't', 'val': 'v'}}
        for name, value in [('status.json', {'status': 'failed', 'exit_code': 1}), ('config.json', c),
                            ('progress.json', {'status': 'failed'}), ('gpu.json', {'fixture': True}),
                            ('binding.json', binding), ('cache_receipts.json', receipts),
                            ('initialization.json', init), ('probes.json', probes), ('preflight_roundtrips.json', rounds)]:
            recovery.write(source / name, value)
        (source / 'error.log').write_text('Traceback\n' + recovery.FAILURE + '\n', encoding='utf-8')
        meta = (c, {}, ref, None, None, current)
        return source, meta

    def test_matching_pretraining_failure_and_cache_proofs_are_reusable(self):
        with tempfile.TemporaryDirectory() as directory:
            source, meta = self.fixture(directory)
            r = recovery.source_receipt(source, meta)
            self.assertEqual(r['formal_training_updates_before_failure'], 0)
            self.assertEqual(r['gradient_probe_updates_reused'], 6)
            self.assertEqual(r['original_cache_extraction_wall_seconds'], 2.)

    def test_rejects_modified_source_or_any_training_progress(self):
        for failure in ('cache', 'checkpoint', 'config', 'runner', 'probe', 'training', 'other_error', 'running'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                source, meta = self.fixture(directory)
                if failure == 'cache': (source / 'train_encoder64.pt').write_bytes(b'changed')
                if failure == 'checkpoint': (source / 'D8_preflight.pt').write_bytes(b'changed')
                if failure == 'config':
                    c = recovery.read(source / 'config.json'); c['epochs'] = 30; recovery.write(source / 'config.json', c)
                if failure == 'runner':
                    b = recovery.read(source / 'binding.json'); b['sources']['scripts/run_fusion_dimension_pair.py'] = 'wrong'
                    recovery.write(source / 'binding.json', b)
                if failure == 'probe':
                    p = recovery.read(source / 'probes.json'); p['D8']['clean_state_and_rng_restored'] = False
                    recovery.write(source / 'probes.json', p)
                if failure == 'training': (source / 'D8/last.pt').write_bytes(b'must preserve')
                if failure == 'other_error': (source / 'error.log').write_text('Other exception')
                if failure == 'running': recovery.write(source / 'status.json', {'status': 'running'})
                with self.assertRaises(ValueError): recovery.source_receipt(source, meta)

    def test_recovery_callbacks_do_not_mutate_parent_and_mark_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            source, meta = self.fixture(directory)
            r = recovery.source_receipt(source, meta)
            output = Path(directory) / 'new'; output.mkdir()
            fn = recovery.recovered_runtime(SimpleNamespace(output=output), r)
            self.assertIsNot(fn.__globals__['probe'], recovery.parent.probe)
            self.assertIsNot(fn.__globals__['extract_cache'], recovery.parent.extract_cache)
            self.assertIsNot(fn.__globals__['make_head'], recovery.parent.make_head)
            self.assertIs(recovery.parent.run.__globals__['probe'], recovery.parent.probe)
            fn.__globals__['write'](output / 'summary.json', {'status': 'completed'})
            s = recovery.read(output / 'summary.json')
            self.assertTrue(s['cache_reused'])
            self.assertEqual(s['new_probe_optimizer_updates'], 0)

    def test_execute_copies_cache_and_preserves_all_source_files(self):
        with tempfile.TemporaryDirectory() as directory:
            source, meta = self.fixture(directory)
            r = recovery.source_receipt(source, meta)
            args = SimpleNamespace(source=source, output=Path(directory) / 'new')
            def fake_run(work, current):
                self.assertEqual(current[0]['epochs'], 100)
                self.assertEqual(current[0]['patience'], 20)
                self.assertFalse(current[0]['include_epoch_zero'])
                recovery.write(work.output / 'summary.json', {'fixture': True})
            with patch.object(recovery, 'recovered_runtime', return_value=fake_run):
                recovery.execute(args, meta, r)
            self.assertEqual(recovery.read(args.output / 'status.json')['status'], 'completed')
            for part in ('train', 'val'):
                self.assertEqual((args.output / (part + '_encoder64.pt')).read_bytes(),
                                 (source / (part + '_encoder64.pt')).read_bytes())
            self.assertEqual(recovery.read(source / 'status.json')['status'], 'failed')
            self.assertFalse((source / 'summary.json').exists())
            self.assertFalse(list((source / 'D8').iterdir()))


if __name__ == '__main__':
    unittest.main()
