"""CPU regression checks for actual graph encoding, joint gradients and exact Adam resume."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch_geometric.data import Batch

import run_joint_transition_fusion as run
import launch_joint_transition_fusion as launcher
from test_packet_transition_readout import fixture
from test_flow_query_joint_packet2 import TinyGraphs


class JointFusionChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.device = torch.device('cpu')
        self.settings = {**run.raw.read(run.CONFIG), 'epochs': 3, 'batch_size': 3, 'eval_batch_size': 4}
        self.source, self.graphs = fixture()
        run.frozen.independent.freeze_c(self.source)
        self.rows = [{'sample_id': g.sample_id, 'label': g.y.item(), 'partition': 'val'} for g in self.graphs]
        self.features, _ = run.frozen.encode(self.source, Batch.from_data_list(self.graphs), self.device)
        self.cache = run.original.FrozenCache([self.features], [r['label'] for r in self.rows])
        self.reference = run.original.prediction_records(
            self.rows, run.original.baseline(self.source, self.features).argmax(-1).tolist())
        self.binding = {'source_C': run.raw.state_digest(self.source.state_dict()), 'test_evaluated': False}

    def assert_tree_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertEqual(left.dtype, right.dtype)
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(type(left), type(right))
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_joint_initialization_gradients_and_frozen_raw_graph_buffers(self):
        before = run.raw.state_digest(self.source.state_dict())
        arms = run.training.make_arms(self.source, self.settings, self.device)
        for name, arm in arms.items():
            scores, residual, _, _ = arm(self.features)
            self.assertTrue(torch.equal(scores, run.original.baseline(self.source, self.features)))
            self.assertFalse(bool(residual.any()))
            self.assertEqual(sum(p.numel() for p in arm.fusion.parameters()), 7144)
            opt = run.training.optimizer(arm, self.settings)
            self.assertEqual([g['lr'] for g in opt.param_groups], [1e-4] + ([] if arm.readout is None else [1e-3]))
        probe = run.training.gradient_probe(self.source, self.features, self.cache.labels, self.settings, self.device)
        self.assertTrue(probe['passed'])
        run.training.verify_source(self.source, before)
        # The actual frozen graph forward still equals the cached reference after all probe updates.
        with torch.no_grad():
            raw_scores = self.source(Batch.from_data_list(self.graphs))
        self.assertEqual(raw_scores.argmax(-1).tolist(), [r['prediction'] for r in self.reference])

    def test_source_replay_accepts_archived_gate_metadata(self):
        gold = [{**r, 'flow_weights': [0.5] * 32} for r in self.reference]
        selected = {'validation': run.frozen.joint.study.validate_predictions(self.reference, self.rows, False),
                    'prediction_ids_sha256': run.frozen.joint.prediction_digest(self.reference)}
        self.assertNotEqual(self.reference, gold)  # Exact shape of the server failure.
        with tempfile.TemporaryDirectory() as temp:
            receipt = run.verify_source_replay(self.reference, gold, selected, self.rows, Path(temp))
            self.assertTrue(receipt['passed'])
            self.assertEqual(receipt['archive_extra_fields'], ['flow_weights'])
            self.assertTrue(receipt['metrics_match'] and receipt['archive_metrics_match'])

    def test_source_replay_rejects_real_prediction_identity_metric_and_digest_changes(self):
        gold = [{**r, 'flow_weights': [0.5] * 32} for r in self.reference]
        selected = {'validation': run.frozen.joint.study.validate_predictions(self.reference, self.rows, False),
                    'prediction_ids_sha256': run.frozen.joint.prediction_digest(self.reference)}
        for field, value in (('prediction', (gold[0]['prediction'] + 1) % 8),
                             ('sample_id', 'wrong-sample'), ('label', (gold[0]['label'] + 1) % 8)):
            changed = deepcopy(gold)
            changed[0][field] = value
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                output = Path(temp)
                with self.assertRaisesRegex(ValueError, 'baseline_validation_replay.json'):
                    run.verify_source_replay(self.reference, changed, selected, self.rows, output)
                receipt = run.raw.read(output / 'baseline_validation_replay.json')
                self.assertFalse(receipt['passed'])
                if field == 'prediction':
                    self.assertEqual(len(receipt['first_differences']), 1)
        for field in ('validation', 'prediction_ids_sha256'):
            changed = deepcopy(selected)
            if field == 'validation':
                changed[field]['macro_f1'] += 1e-4
            else:
                changed[field] = '0' * 64
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                with self.assertRaisesRegex(ValueError, 'baseline_validation_replay.json'):
                    run.verify_source_replay(self.reference, gold, changed, self.rows, Path(temp))

    def test_sampler_matches_historical_order_including_singleton_tail(self):
        for size, batch in ((8, 3), (7, 3), (107621, 512)):
            sampler = run.training.PairedSampler(size, batch, 42)
            old = run.raw.TrainBatches(size, batch, 42)
            for epoch in range(1, 4):
                sampler.start(epoch)
                actual = []
                while sampler.cursor < size:
                    indices = sampler.peek()
                    actual.append(indices)
                    sampler.cursor += len(indices)
                expected = list(old)
                self.assertEqual(actual, expected)
                self.assertEqual(sampler.order_sha256, old.order_sha256)
                self.assertEqual(len(actual), len(sampler))
                self.assertEqual(sorted(sum(actual, [])), list(range(size)))
        self.assertEqual(len(run.training.PairedSampler(107621, 512, 42)) * 200, 42200)

    def test_interrupted_mid_epoch_and_epoch_boundary_equal_uninterrupted(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            def fit(path, resume=False, stop=None):
                arms = run.training.make_arms(self.source, self.settings, self.device)
                return run.training.fit(arms, self.cache, self.cache, self.rows, self.reference,
                    self.settings, self.binding, self.device, path, resume=resume, stop_after_updates=stop)
            run.raw.seed_all(42)
            uninterrupted = fit(root / 'continuous')
            for stop in (1, 3, 5):
                folder = root / ('stop_' + str(stop))
                run.raw.seed_all(42)
                partial = fit(folder, stop=stop)
                self.assertEqual(partial['updates'], stop)
                # Perturb every RNG and parameters before reloading; restored results must stay exact.
                random.random(); np.random.rand(10); torch.rand(10)
                resumed = fit(folder, resume=True)
                self.assert_tree_equal(uninterrupted, resumed)
                self.assertTrue(run.raw.read(folder / 'resume_receipt.json')['adam_restored'])
                self.assertEqual(resumed['updates'], 9)
                for name in run.ARMS:
                    ledger = [json.loads(line) for line in (folder / name / 'epochs.jsonl').read_text().splitlines()]
                    self.assertEqual([r['epoch'] for r in ledger], [0, 1, 2, 3])
                    selected = max(ledger, key=lambda r: r['validation']['macro_f1'])
                    self.assertEqual(resumed['best'][name]['epoch'], selected['epoch'])
                    self.assertEqual(ledger[-1]['cumulative_updates_per_arm'], 9)
                    steps = [v['step'].item() for v in resumed['optimizers'][name]['state'].values()]
                    self.assertTrue(steps and all(step == 9 for step in steps))

    def test_resume_rejects_changed_configuration_and_source_binding(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            output = Path(temp)
            arms = run.training.make_arms(self.source, self.settings, self.device)
            run.training.fit(arms, self.cache, self.cache, self.rows, self.reference, self.settings,
                             self.binding, self.device, output, stop_after_updates=1)
            for settings, binding in (({**self.settings, 'fusion_learning_rate': 9e-4}, self.binding),
                                      (self.settings, {**self.binding, 'source_C': 'different'})):
                with self.assertRaisesRegex(ValueError, 'mismatch'):
                    run.training.fit(arms, self.cache, self.cache, self.rows, self.reference, settings,
                                     binding, self.device, output, resume=True)

    def test_selected_checkpoint_replays_on_raw_graphs_and_reports_all_comparisons(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            output = Path(temp)
            arms = run.training.make_arms(self.source, self.settings, self.device)
            state = run.training.fit(arms, self.cache, self.cache, self.rows, self.reference, self.settings,
                                     self.binding, self.device, output)
            refs = {'original_C': run.training.JointTransitionFusion(self.source.fusion, 'joint_base', 42)}
            refs.update({'frozen_' + mode: run.training.JointTransitionFusion(self.source.fusion, 'joint_' + mode, 42)
                         for mode in ('single', 'pointwise', 'ordered')})
            for arm in refs.values():
                arm.requires_grad_(False).eval()
            data = TinyGraphs(self.graphs)
            records = run.selected_raw_replay(self.source, arms, refs, data, self.rows,
                     {'seed': 42, 'workers': 0, 'eval_batch_size': 4}, self.device, output, state)
            summary = run.diagnostics.write_results(output, records, self.rows, state,
                                                     {**self.settings, 'epochs': 200})
            self.assertFalse(summary['test_evaluated'])
            self.assertIn('joint_ordered-joint_single', summary['comparisons'])
            self.assertEqual(len(summary['metrics']), 8)
            self.assertEqual(len(run.raw.read(output / 'budget_comparison.json')['budgets']), 4)
            self.assertEqual(len((output / 'validation/per_class.csv').read_text().splitlines()), 65)
            for name in run.ARMS:
                self.assertTrue(run.raw.read(output / 'selected_validation_replay.json')[name]['passed'])
            self.assertTrue(all('ordered_contexts' in r for r in records))

    def test_continuity_diagnosis_does_not_infer_context_from_two_edges(self):
        features = self.features._replace(transitions=self.features.transitions.clone())
        features.transitions[:, 70] = 0
        self.assertTrue(bool((features.lengths >= 2).any()))
        self.assertFalse(bool(run.diagnostics.context_counts(features).any()))
        arms = run.training.make_arms(self.source, self.settings, self.device)
        records = run.diagnostics.observe(arms, features, self.rows)
        self.assertTrue(all(r['ordered_contexts'] == 0 for r in records))
        for row in records:
            for detail in row['diagnostics'].values():
                self.assertEqual(detail['context_attention'], [])

    def test_failure_inside_four_arm_transaction_recovers_previous_committed_update(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            def fit(folder, resume=False, stop=None):
                arms = run.training.make_arms(self.source, self.settings, self.device)
                return run.training.fit(arms, self.cache, self.cache, self.rows, self.reference,
                    self.settings, self.binding, self.device, folder, resume=resume, stop_after_updates=stop)
            run.raw.seed_all(42)
            expected = fit(root / 'continuous')
            run.raw.seed_all(42)
            fit(root / 'failure', stop=1)
            digest = run.raw.sha(root / 'failure/recovery.pt')
            factory = run.training.optimizer
            created = []
            def failing_optimizer(arm, settings):
                opt = factory(arm, settings)
                created.append(opt)
                if len(created) == 4:
                    def fail(*args, **kwargs):
                        raise RuntimeError('simulated failure after three arm steps')
                    opt.step = fail
                return opt
            with patch.object(run.training, 'optimizer', side_effect=failing_optimizer):
                with self.assertRaisesRegex(RuntimeError, 'simulated failure'):
                    fit(root / 'failure', resume=True)
            self.assertEqual(run.raw.sha(root / 'failure/recovery.pt'), digest)
            resumed = fit(root / 'failure', resume=True)
            self.assert_tree_equal(expected, resumed)

    def test_publisher_uploads_failure_evidence_and_preserves_training_exit(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            repo = Path(temp)
            output = repo / 'outputs' / launcher.OUTPUT
            output.mkdir(parents=True)
            statuses = [1, 0]
            def publish(command, **kwargs):
                self.assertEqual(command[-2:], ['--only', launcher.OUTPUT])
                kwargs['stdout'].write(b'fake publication log\n')
                return type('Result', (), {'returncode': statuses.pop(0)})()
            with patch.object(launcher.subprocess, 'run', side_effect=publish):
                attempts = launcher.publish_results(run.raw.ROOT, repo, output, 7, {})
            self.assertEqual(attempts, [1, 0])
            self.assertEqual(run.raw.read(output / 'exit_codes.json')['training_exit_code'], 7)
            self.assertEqual(run.raw.read(output / 'exit_codes.json')['primary_upload_exit_code'], 1)
            self.assertIn('publication log', (output / 'upload.log').read_text())

    def test_complete_runner_probe_cache_training_and_completed_resume_with_synthetic_graphs(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            repo = Path(temp)
            history, output = repo / 'history', repo / run.OUTPUT
            history.mkdir(); output.mkdir(parents=True)
            settings = {**self.settings, 'epochs': 2, 'training_samples': 8, 'validation_samples': 8, 'test_samples': 8}
            data = TinyGraphs(self.graphs)
            data.root = repo / 'graphs'
            data.root.mkdir()
            rows = {'val': self.rows, 'train': [{**r, 'partition': 'train'} for r in self.rows]}
            data.rows = rows['train'] + rows['val']
            data.metadata = {'partition_sizes': {'train': 8, 'val': 8, 'test': 8}}
            config = {'seed': 42, 'workers': 0, 'batch_size': 3, 'eval_batch_size': 4}
            selected = {'epoch': 48, 'validation': run.frozen.joint.study.validate_predictions(
                self.reference, self.rows, probabilities=False),
                'prediction_ids_sha256': run.frozen.joint.prediction_digest(self.reference)}
            coverage = {}
            for part in ('train', 'val'):
                _, _, coverage[part] = run.frozen.cache_split(self.source, data, rows[part], part,
                                                             config, self.device, output)
            run.raw.write(history / 'coverage.json', coverage)
            for mode in ('single', 'pointwise', 'ordered'):
                (history / mode).mkdir()
                head = run.frozen.OrderedTransitionReadout(mode, 42)
                run.training.atomic_save(history / mode / 'readout.pt', {
                    'binding': self.binding, 'mode': mode, 'best': {'B': {'state': head.state_dict()}}})
            archived = [{**r, 'predictions': {name: r['prediction'] for name in ('A', 'single', 'pointwise', 'ordered')}}
                        for r in self.reference]
            # Real original-C archives carry dimensional gate vectors beside the core prediction fields.
            gold = [{**r, 'flow_weights': [0.5] * 32} for r in self.reference]
            source_result = (self.source, data, rows, self.device, config, selected, gold, {}, self.binding)
            history_result = (history, {}, {'source': self.binding}, {'binding': self.binding}, archived)
            with patch.object(run, 'validate_settings'), patch.object(run, 'history_inputs', return_value=history_result), \
                 patch.object(run.frozen, 'load_source', return_value=source_result), \
                 patch.object(run.torch.cuda, 'get_device_name', return_value='synthetic CPU fixture'):
                result = run.run(repo, None, output, settings, probe_only=True)
                self.assertFalse(result['formal_training_started'])
                self.assertFalse((output / 'recovery.pt').exists())
                result = run.run(repo, None, output, settings, resume=True)
                self.assertEqual(result['status'], 'completed')
                state = torch.load(output / 'recovery.pt', map_location='cpu', weights_only=True)
                self.assertEqual(state['updates'], 6)
                # Running the completed recovery again consumes no updates and preserves Adam.
                run.run(repo, None, output, settings, resume=True)
                resumed = torch.load(output / 'recovery.pt', map_location='cpu', weights_only=True)
                self.assert_tree_equal(state, resumed)
            self.assertTrue(run.raw.read(output / 'epoch_zero_replay.json')['passed'])
            self.assertTrue(run.raw.read(output / 'historical_diagnosis/replay_receipt.json')['passed'])
            self.assertEqual(run.raw.read(output / 'final_verification.json')['test_graph_loads'], 0)


if __name__ == '__main__':
    unittest.main()
