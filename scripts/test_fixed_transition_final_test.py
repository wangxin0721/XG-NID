"""CPU verification of frozen evaluation, cache recovery, paired reports and publication exits."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import Batch

import run_fixed_transition_final_test as run
import launch_fixed_transition_final_test as launcher
from test_packet_transition_readout import fixture

locked = run.locked


class MemoryArchive:
    def __init__(self, values):
        self.values = values

    def data(self, name):
        value = self.values[name]
        return value if isinstance(value, bytes) else json.dumps(value).encode()

    def rows(self, name): return self.values[name]
    def json(self, name): return self.values[name]
    def tensor(self, name): return torch.load(io.BytesIO(self.data(name)), weights_only=True)
    def verify_all(self): pass


def tensor_bytes(value):
    stream = io.BytesIO()
    torch.save(value, stream)
    return stream.getvalue()


class TestGraphs:
    def __init__(self, graphs):
        self.root = Path('/synthetic-graphs').resolve()
        self.graphs, self.rows = {}, []
        for part in ('val', 'test'):
            for i, graph in enumerate(graphs):
                g = graph.clone()
                g.sample_id = part + ':' + str(i)
                self.graphs[g.sample_id] = g
                self.rows.append({'sample_id': g.sample_id, 'label': i, 'partition': part})

    def load(self, row): return self.graphs[row['sample_id']].clone()


class FixedTestChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.device = torch.device('cpu')
        self.settings = deepcopy(run.raw.read(run.CONFIG))
        self.settings.update(validation_samples=8, test_samples=8, eval_batch_size=4, workers=0)
        self.source, graphs = fixture()
        self.data = TestGraphs(graphs)
        self.rows = {p: [r for r in self.data.rows if r['partition'] == p] for p in ('val', 'test')}
        self.binding = {'backbone_seed': 42, 'frozen_c_sha256': run.raw.state_digest(self.source.state_dict()),
                        'split_id': 'split', 'manifest_sha256': 'manifest', 'validation_digest': 'validation'}
        self.settings['seeds']['42']['source_binding'] = self.binding
        self.config = {'seed': 42, 'workers': 0, 'eval_batch_size': 4, 'partition_sizes': {'val': 8, 'test': 8}}
        self.values = {}
        self.models = {'base_C': locked.stability.training.parent.JointTransitionFusion(self.source.fusion, 'joint_base', 42)}
        for i, name in enumerate(locked.ARMS):
            model = locked.StabilityFusion(self.source.fusion, name, 42)
            with torch.no_grad():
                model.readout.output.weight.copy_(torch.randn(model.readout.output.weight.shape,
                    generator=torch.Generator().manual_seed(101 + i)) * .04)
            self.models[name] = model
        for model in self.models.values(): model.requires_grad_(False).eval()
        features, _ = run.frozen.encode(self.source, Batch.from_data_list([self.data.load(r) for r in self.rows['val']]), self.device)
        self.cache = run.original.FrozenCache([features], list(range(8)))
        self.records = run.predict(self.models, self.cache, self.rows['val'], 4, self.device)
        predictions = locked.results.diagnostics.exports(self.records)
        entries = {}
        for name in locked.MODELS:
            item = self.settings['seeds']['42']['checkpoints'][name]
            metric = run.frozen.joint.study.validate_predictions(predictions[name], probabilities=False)
            if name == 'base_C':
                ck = {'model_state': self.source.state_dict(), 'epoch': item['epoch'], 'config': {
                    'arm': 'C', 'query_source': 'flow', 'labels': list(locked.LABELS), 'protocol': {'seed': 42}}}
            else:
                ck = {'state': self.models[name].state_dict(), 'source_binding': self.binding,
                      'epoch': item['epoch'], 'validation': metric}
            self.values[item['path']] = tensor_bytes(ck)
            path = 'selected_' + name
            self.values[path] = predictions[name]
            entries[name] = {**item, 'validation': metric, 'validation_predictions_path': path}
        self.values['historical_test'] = [{'sample_id': 'test:' + str(i), 'label': i,
            'prediction': predictions['base_C'][i]['prediction']} for i in range(8)]
        self.settings['historical_base_C_test']['42'] = {'predictions': 'historical_test', 'prediction_key': None}
        self.inputs = MemoryArchive(self.values)
        self.manifest = {'seeds': {'42': {'models': entries, 'source_binding': self.binding,
            'role': 'development_stage_locked_archive'}},
            'validation_cohort_sha256': locked.archive.canonical_sha([(r['sample_id'], r['label']) for r in self.rows['val']]),
            'test_cohort_sha256': locked.archive.canonical_sha([(r['sample_id'], r['label']) for r in self.rows['test']])}
        self.protocol = {'code_commit': 'CPU-check', 'checkpoint_manifest_sha256': 'locked-before-test'}

    def seed(self, root, resume=False, probe=False):
        with patch.object(run.frozen.independent.d32.baseline, 'runtime', return_value=self.device), \
             patch.object(run, 'load_data', return_value=(self.data, self.rows, self.config, {})), \
             patch.object(torch.optim, 'Adam', side_effect=AssertionError('Optimizer forbidden')), \
             patch.object(torch.Tensor, 'backward', side_effect=AssertionError('Backward forbidden')):
            return run.run_seed(Path(root), None, Path(root), self.settings, self.protocol,
                               self.manifest, self.inputs, 42, resume, probe)

    def prepare(self, root):
        run.raw.write(root / 'checkpoint_lock_receipt.json', {'passed': True})

    def test_registered_protocol_prevents_training_reselection_or_different_checkpoints(self):
        original = run.raw.read(run.CONFIG)
        locked.validate_settings(original)
        for key, value in (('new_optimizer_updates', 1), ('test_reselection_permitted', True),
                           ('pairing', 'shuffle'), ('eval_batch_size', 512), ('readout_seed', 43)):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'protocol differs'):
                locked.validate_settings({**original, key: value})
        altered = deepcopy(original)
        altered['seeds']['42']['checkpoints']['joint_ordered']['epoch'] = 66
        with self.assertRaisesRegex(ValueError, 'checkpoint was replaced'): locked.validate_settings(altered)

    def test_restore_matches_nonzero_archived_models_without_rng_or_gradient_changes(self):
        rng = torch.get_rng_state().clone()
        source, models, _ = locked.restore_models(self.inputs, self.settings, 42, self.device)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(locked.frozen_digests(self.source, self.models), locked.frozen_digests(source, models))
        before = locked.frozen_digests(source, models)
        self.assertEqual(self.records, run.predict(models, self.cache, self.rows['val'], 4, self.device))
        locked.verify_frozen(source, models, before)
        self.assertFalse(any('injection_gate' in k for m in models.values() for k in m.state_dict()))

    def test_wrong_source_or_frozen_fusion_or_label_mapping_is_rejected(self):
        for key in ('source', 'fusion', 'labels'):
            values = deepcopy(self.values)
            if key == 'source':
                name = self.settings['seeds']['42']['checkpoints']['joint_single']['path']
                ck = self.inputs.tensor(name); ck['source_binding']['frozen_c_sha256'] = 'other'
            elif key == 'fusion':
                name = self.settings['seeds']['42']['checkpoints']['frozen_single']['path']
                ck = self.inputs.tensor(name); ck['state']['fusion.flow_projection.weight'] += 1
            else:
                name = self.settings['seeds']['42']['checkpoints']['base_C']['path']
                ck = self.inputs.tensor(name); ck['config']['labels'] = list(reversed(locked.LABELS))
            values[name] = tensor_bytes(ck)
            with self.subTest(key=key), self.assertRaises(ValueError):
                locked.restore_models(MemoryArchive(values), self.settings, 42, self.device)

    def test_frozen_state_verifier_detects_parameter_and_mode_changes(self):
        before = locked.frozen_digests(self.source, self.models)
        self.models['joint_ordered'].train()
        with self.assertRaisesRegex(ValueError, 'training module'):
            locked.verify_frozen(self.source, self.models, before)
        self.models['joint_ordered'].eval()
        with torch.no_grad(): self.models['joint_ordered'].readout.output.weight.add_(1)
        with self.assertRaisesRegex(ValueError, 'parameters/buffers'):
            locked.verify_frozen(self.source, self.models, before)

    def test_cache_reuse_integrity_and_missing_receipt_reextracts(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp); binding = {'source': self.binding}
            run.bound_cache(self.source, self.data, self.rows['val'], 'val', self.config, self.device, root, binding)
            with patch.object(run.frozen, 'cache_split', side_effect=AssertionError('Cache should be reused')):
                run.bound_cache(self.source, self.data, self.rows['val'], 'val', self.config, self.device, root, binding)
                with self.assertRaisesRegex(ValueError, 'source/environment/hash'):
                    run.bound_cache(self.source, self.data, self.rows['val'], 'val', self.config, self.device, root, {'other': 1})
            receipt = root / 'cache_val_receipt.json'
            receipt.unlink()
            with patch.object(run.frozen, 'cache_split', wraps=run.frozen.cache_split) as extract:
                run.bound_cache(self.source, self.data, self.rows['val'], 'val', self.config, self.device, root, binding)
                extract.assert_called_once()
            with (root / 'frozen_val_features.pt').open('ab') as stream: stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash differs'):
                run.bound_cache(self.source, self.data, self.rows['val'], 'val', self.config, self.device, root, binding)

    def test_complete_fixed_test_has_zero_optimizer_updates_and_completed_resume_skips_graphs(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp); self.prepare(root)
            result = self.seed(root)
            self.assertEqual(result['new_optimizer_updates'], 0)
            verified = run.raw.read(root / 'seed42/final_verification.json')
            self.assertTrue(verified['same_weight_historical_C_predictions_identical'])
            self.assertEqual(verified['train_graph_loads'], 0)
            with patch.object(run, 'load_data', side_effect=AssertionError('Completed results must be reused')):
                # seed() patches load_data itself, so check the no-runtime early-return path directly.
                result = run.run_seed(root, None, root, self.settings, self.protocol, self.manifest,
                                      self.inputs, 42, resume=True)
            self.assertTrue(result['reused_completed_test'])

    def test_probe_never_accesses_test_graphs_or_produces_test_metrics(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp); self.prepare(root); self.seed(root, probe=True)
            self.assertFalse((root / 'seed42/test_started.json').exists())
            self.assertFalse((root / 'seed42/summary.json').exists())
            self.assertEqual(run.raw.read(root / 'seed42/server_probe.json')['test_graph_loads'], 0)

    def test_completed_backbone_cannot_be_relabelled_or_resumed_with_another_protocol(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp); self.prepare(root); self.seed(root)
            changed = deepcopy(self.manifest)
            changed['seeds']['42']['source_binding']['frozen_c_sha256'] = 'another-backbone'
            with self.assertRaisesRegex(ValueError, 'identities differ'):
                run.run_seed(root, None, root, self.settings, self.protocol, changed, self.inputs, 42, resume=True)
            with self.assertRaisesRegex(ValueError, 'protocol differs'):
                run.run_seed(root, None, root, self.settings, {**self.protocol, 'code_commit': 'different'},
                             self.manifest, self.inputs, 42, resume=True)

    def test_test_dataset_rejects_non_test_rows_and_cache_rejects_training(self):
        with self.assertRaisesRegex(ValueError, 'only fixed test'):
            run.frozen.evaluation.TestGraphs(self.data, self.rows['val'])
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'cannot extract training'):
                run.bound_cache(self.source, self.data, self.rows['val'], 'train', self.config,
                                self.device, Path(tmp), {})

    def test_partial_prediction_failure_resumes_cache_without_training(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp); self.prepare(root)
            predict = run.predict; calls = []
            def interrupt(*args):
                calls.append(1)
                if len(calls) == 2: raise RuntimeError('interrupted fixed test')
                return predict(*args)
            with patch.object(run, 'predict', side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, 'interrupted'): self.seed(root)
            with patch.object(run.frozen, 'cache_split', side_effect=AssertionError('Both caches should be reused')):
                self.seed(root, resume=True)
            self.assertTrue(run.raw.read(root / 'seed42/final_verification.json')['passed'])

    def test_historical_C_mismatch_persists_failed_receipt_and_stops(self):
        self.values['historical_test'][0]['prediction'] = (self.values['historical_test'][0]['prediction'] + 1) % 8
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp); self.prepare(root)
            with self.assertRaisesRegex(ValueError, 'replay differs'): self.seed(root)
            self.assertFalse(run.raw.read(root / 'seed42/historical_base_C_test_replay.json')['passed'])
            self.assertFalse((root / 'seed42/summary.json').exists())

    def test_aggregate_retains_negative_and_zero_differences_and_historical_split_disclosure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); summaries = {}
            for seed, correct in ((42, 7), (43, 4), (44, 6)):
                folder = root / str(seed); folder.mkdir()
                records = deepcopy(self.records)
                for i, row in enumerate(records):
                    row['partition'] = 'test'
                    row['predictions'] = {n: i if i < limit else 0 for n, limit in
                        (('base_C', 4), ('frozen_single', 6), ('joint_single', 5), ('joint_ordered', correct))}
                manifest = deepcopy(self.manifest['seeds']['42'])
                manifest['source_binding']['frozen_c_sha256'] = 'independent-' + str(seed)
                summaries[seed] = run.report.write_seed(folder, records,
                    [{**r, 'partition': 'test'} for r in self.rows['val']], seed, manifest, {})
            summary = run.report.aggregate(summaries, root / 'aggregate', self.protocol, {'failures': 'retained'})
            pair = summary['all_three']['paired_macro_f1_differences_pp']['joint_ordered-frozen_single']
            self.assertEqual(pair['negative_seeds'], [43]); self.assertEqual(pair['zero_seeds'], [44])
            self.assertFalse(pair['all_positive']); self.assertTrue(summary['historical_test_split_used'])
            self.assertEqual(summary['new_optimizer_updates'], 0)
            self.assertTrue((root / 'aggregate/paper_tables.md').is_file())
            text = (root / 'aggregate/paper_tables.md').read_text(encoding='utf-8')
            self.assertIn('已有历史评估使用', text); self.assertIn('未通过事前', text)


class LauncherChecks(unittest.TestCase):
    def args(self, root, **kw):
        return SimpleNamespace(repo=root, data_root=None, lock_only=False, probe_only=False,
                               publish_remote=None, **kw)

    def test_full_queue_publishes_lock_before_test_then_all_backbones_and_final(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp); events = []
            def popen(command, **kwargs):
                events.append(command[command.index('--phase') + 1]); return SimpleNamespace(wait=lambda: 0)
            def publish(*args, **kwargs): events.append('upload'); return [0, 0]
            with patch.object(launcher, 'lfs_environment', return_value=dict(os.environ)), \
                 patch.object(launcher.subprocess, 'Popen', side_effect=popen), \
                 patch.object(launcher.publisher, 'publish_results', side_effect=publish):
                self.assertEqual(launcher.launch(self.args(root)), 0)
            self.assertEqual(events, ['lock', 'upload', 'seed42', 'upload', 'seed43', 'upload',
                                      'seed44', 'upload', 'aggregate', 'upload'])

    def test_failure_still_uploads_and_preserves_evaluation_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            with patch.object(launcher, 'lfs_environment', return_value=dict(os.environ)), \
                 patch.object(launcher.subprocess, 'Popen', return_value=SimpleNamespace(wait=lambda: 17)), \
                 patch.object(launcher.publisher, 'publish_results', return_value=[0, 0]) as upload:
                self.assertEqual(launcher.launch(self.args(root)), 17)
            self.assertEqual(upload.call_count, 1)
            self.assertEqual(upload.call_args.kwargs['exit_key'], 'evaluation_exit_code')
            self.assertEqual(json.loads((root / 'outputs' / launcher.OUTPUT_NAME / 'evaluation_exit.json').read_text())['exit_code'], 17)
            self.assertEqual(json.loads((root / 'outputs' / launcher.OUTPUT_NAME / 'status.json').read_text())['status'], 'failed')

    def test_lock_probe_queues_and_missing_lfs_fail_with_evidence(self):
        self.assertEqual(launcher.phases(lock_only=True), ('lock',))
        self.assertNotIn('aggregate', launcher.phases(probe_only=True))
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            with patch.object(launcher, 'lfs_environment', side_effect=RuntimeError('bootstrap failure')), \
                 patch.object(launcher.publisher, 'publish_results', return_value=[0, 0]) as pub:
                self.assertEqual(launcher.launch(self.args(root)), 1)
            self.assertTrue((root / 'outputs' / launcher.OUTPUT_NAME / 'launcher_failure.json').is_file())
            pub.assert_called_once()


if __name__ == '__main__': unittest.main()
