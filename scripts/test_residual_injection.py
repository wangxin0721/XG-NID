"""CPU checks for scalar injection, delayed gradients, exact Adam recovery and raw replay."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.data import Batch

import run_residual_injection as run
import residual_injection_results as results
import launch_residual_injection as launcher
from test_packet_transition_readout import fixture
from test_flow_query_joint_packet2 import TinyGraphs
from test_joint_transition_fusion import JointFusionChecks
from xgnid.residual_injection import ARMS


class InjectionChecks(unittest.TestCase):
    assert_tree_equal = JointFusionChecks.assert_tree_equal

    def setUp(self):
        torch.set_num_threads(2)
        self.device = torch.device('cpu')
        self.settings = {**run.raw.read(run.CONFIG), 'epochs': 3, 'batch_size': 3, 'eval_batch_size': 4}
        self.source, self.graphs = fixture()
        run.frozen.independent.freeze_c(self.source)
        self.rows = [{'sample_id': g.sample_id, 'label': g.y.item(), 'partition': 'val'} for g in self.graphs]
        self.features, _ = run.frozen.encode(self.source, Batch.from_data_list(self.graphs), self.device)
        self.cache = run.original.FrozenCache([self.features], [r['label'] for r in self.rows])
        self.reference = run.original.prediction_records(self.rows,
            run.original.baseline(self.source, self.features).argmax(-1).tolist())
        self.binding = {'source_C': run.raw.state_digest(self.source.state_dict()), 'test_evaluated': False}

    def arms(self):
        return run.training.make_arms(self.source, self.settings, self.device)

    def fit(self, folder, resume=False, stop=None):
        return run.training.fit(self.arms(), self.cache, self.cache, self.rows, self.reference,
            self.settings, self.binding, self.device, folder, resume=resume, stop_after_updates=stop)

    def test_registered_protocol_rejects_design_or_budget_changes(self):
        settings = run.raw.read(run.CONFIG)
        run.validate_settings(settings)
        for key, value in (('injection_gate_seed', 42), ('injection_input_width', 96), ('initial_beta', 0),
                           ('epochs', 100), ('injection_learning_rate', 0.0001), ('pairing', 'shuffle'),
                           ('layer_norm_affine', True), ('loss', 'weighted_NLL')):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'protocol changed'):
                run.validate_settings({**settings, key: value})

    def test_common_initialization_rng_parameter_counts_and_optimizer_groups(self):
        rng = run.training.parent.rng_state()
        arms = self.arms()
        self.assert_tree_equal(rng, run.training.parent.rng_state())
        for name, arm in arms.items():
            out = arm.detailed_forward(self.features)
            self.assertTrue(torch.equal(out.scores, run.original.baseline(self.source, self.features)))
            self.assertTrue(bool((out.beta == 1).all()))
            self.assertFalse(bool(out.residual.any()))
            self.assertEqual(sum(p.numel() for p in arm.parameters()), self.settings['total_parameters'][name])
            opt = run.training.optimizer(arm, self.settings)
            self.assertEqual([g['lr'] for g in opt.param_groups], [1e-4, 1e-3] + ([] if name == 'inj_fixed' else [1e-3]))
            self.assertEqual(sum(len(g['params']) for g in opt.param_groups), len(list(arm.parameters())))
        different = run.training.make_arms(self.source, {**self.settings, 'injection_gate_seed': 13}, self.device)
        for name in ARMS:
            self.assert_tree_equal(arms[name].readout.state_dict(), different[name].readout.state_dict())
            self.assert_tree_equal(arms[name].fusion.state_dict(), different[name].fusion.state_dict())
        self.assertFalse(torch.equal(arms['inj_dynamic'].injection_gate.network[0].weight,
                                     different['inj_dynamic'].injection_gate.network[0].weight))

    def test_exact_97_features_use_pre_injection_projections_and_global_zeros(self):
        arms = self.arms()
        arm = arms['inj_dynamic']
        residual = torch.randn_like(self.features.packet)
        zf = F.layer_norm(arm.fusion.flow_projection(self.features.flow), (32,))
        zp = F.layer_norm(arm.fusion.packet_projection(self.features.packet), (32,))
        expected = torch.cat((zf, zp, (zf-zp).abs(), torch.log1p(residual.norm(dim=-1) /
                    (self.features.packet.norm(dim=-1) + 1e-8))[:, None]), dim=-1)
        self.assertTrue(torch.equal(arm.gate_inputs(self.features, residual), expected))
        self.assertEqual(expected.shape, (8, 97))
        self.assertFalse(bool(arms['inj_global'].gate_inputs(self.features, residual).any()))
        self.assertFalse(any('norm' in name for name, _ in arm.named_parameters()))

    def test_nonzero_fixed_matches_previous_ordered_and_beta_can_enhance_or_suppress(self):
        arms = self.arms()
        reference = run.training.parent.JointTransitionFusion(self.source.fusion, 'joint_ordered', 42)
        with torch.no_grad():
            reference.readout.output.weight.normal_(std=0.1)
            for arm in arms.values():
                arm.readout.load_state_dict(reference.readout.state_dict())
        expected = reference(self.features)
        for arm in arms.values():
            actual = arm(self.features)
            for a, b in zip(actual, expected):
                self.assertTrue(torch.equal(a, b))
        dynamic = arms['inj_dynamic']
        for bias in (-3.0, 3.0):
            with torch.no_grad():
                dynamic.injection_gate.network[-1].bias.fill_(bias)
            out = dynamic.detailed_forward(self.features)
            self.assertTrue(bool(((out.beta > 0) & (out.beta < 2)).all()))
            self.assertEqual(bool((out.beta > 1).all()), bias > 0)
            self.assertTrue(torch.equal(out.injected, out.beta * out.residual))
            manual = dynamic.fusion(self.features.flow, self.features.packet + out.beta * out.residual)
            self.assertTrue(torch.equal(manual[0], out.scores))
            self.assertEqual(out.alpha.shape, (8, 32))

    def test_staged_gate_gradients_and_frozen_encoders_cross_layer_buffers(self):
        before = run.raw.state_digest(self.source.state_dict())
        rng = run.training.parent.rng_state()
        receipt = run.training.gradient_probe(self.source, self.features, self.cache.labels, self.settings, self.device)
        self.assertTrue(receipt['passed'])
        self.assert_tree_equal(rng, run.training.parent.rng_state())
        run.training.parent.verify_source(self.source, before)
        with torch.no_grad():
            raw_scores = self.source(Batch.from_data_list(self.graphs))
        self.assertEqual(raw_scores.argmax(-1).tolist(), [r['prediction'] for r in self.reference])

    def test_dynamic_input_gradient_reaches_live_projections_and_readout(self):
        arm = self.arms()['inj_dynamic']
        with torch.no_grad():
            arm.readout.output.weight.normal_(std=0.1)
            arm.injection_gate.network[-1].weight.normal_(std=0.1)
        beta = arm.detailed_forward(self.features).beta
        gradients = torch.autograd.grad(beta.sum(), (arm.fusion.flow_projection.weight,
                            arm.fusion.packet_projection.weight, arm.readout.output.weight))
        self.assertTrue(all(bool(torch.isfinite(g).all()) and float(g.abs().sum()) > 0 for g in gradients))

    def test_true_resume_mid_epoch_boundary_and_completed_preserves_all_adam_rng_and_order(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            run.raw.seed_all(42)
            expected = self.fit(root / 'continuous')
            self.assertEqual(expected['schema'], 'xgnid.residual-injection-recovery.v1')
            for stop in (1, 3, 5):
                folder = root / str(stop)
                run.raw.seed_all(42)
                self.fit(folder, stop=stop)
                random.random(); np.random.rand(7); torch.rand(10)
                resumed = self.fit(folder, resume=True)
                self.assert_tree_equal(expected, resumed)
                self.assert_tree_equal(resumed, self.fit(folder, resume=True))
                for name, opt in resumed['optimizers'].items():
                    for value in opt['state'].values():
                        self.assertEqual(value['step'].item(), 9)
                self.assertEqual(resumed['updates'], 9)
                self.assertEqual([r['epoch'] for r in resumed['history']], [0, 1, 2, 3])

    def test_failure_during_three_arm_transaction_rolls_back_to_last_committed_batch(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            run.raw.seed_all(42); expected = self.fit(root / 'continuous')
            run.raw.seed_all(42); self.fit(root / 'failure', stop=1)
            digest = run.raw.sha(root / 'failure/recovery.pt')
            factory, created = run.training.optimizer, []
            def failing(arm, settings):
                opt = factory(arm, settings); created.append(opt)
                if len(created) == 3:
                    def fail(*args, **kwargs):
                        raise RuntimeError('failure after two arm steps')
                    opt.step = fail
                return opt
            with patch.object(run.training, 'optimizer', side_effect=failing):
                with self.assertRaisesRegex(RuntimeError, 'after two arm steps'):
                    self.fit(root / 'failure', resume=True)
            self.assertEqual(run.raw.sha(root / 'failure/recovery.pt'), digest)
            self.assert_tree_equal(expected, self.fit(root / 'failure', resume=True))

    def test_resume_rejects_source_design_and_gate_state_mismatch(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            output = Path(temp)
            self.fit(output, stop=1)
            for settings, binding in (({**self.settings, 'injection_gate_seed': 7}, self.binding),
                                      (self.settings, {**self.binding, 'source_C': 'other'})):
                with self.assertRaisesRegex(ValueError, 'mismatch'):
                    run.training.fit(self.arms(), self.cache, self.cache, self.rows, self.reference,
                        settings, binding, self.device, output, resume=True)

    def test_selected_raw_replay_predictions_class_tables_repairs_beta_and_budget_windows(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            output = Path(temp)
            state = self.fit(output)
            records = run.selected_raw_replay(self.source, self.arms(), TinyGraphs(self.graphs), self.rows,
                {'seed': 42, 'workers': 0, 'eval_batch_size': 4}, self.device, output, state, self.reference)
            settings = {**self.settings, 'epochs': 200}
            summary = results.write_results(output, records, self.rows, state, settings)
            self.assertFalse(summary['test_evaluated'])
            self.assertEqual(len(summary['metrics']), 5)
            self.assertIn('inj_dynamic-inj_global', summary['comparisons'])
            self.assertIn('inj_dynamic-historical_True', summary['comparisons'])
            self.assertEqual(len(run.raw.read(output / 'budget_comparison.json')['budgets']), 4)
            self.assertEqual(len((output / 'validation/per_class.csv').read_text().splitlines()), 41)
            self.assertEqual(results.beta_summary(records)['inj_fixed']['overall']['mean'], 1.0)
            self.assertTrue(run.raw.read(output / 'selected_validation_replay.json')['inj_dynamic']['passed'])
            self.assertTrue((output / 'validation/inj_dynamic-inj_fixed_changed_samples.jsonl.gz').exists())
            arms = self.arms()
            for name in ARMS:
                arms[name].load_state_dict(state['models'][name])
            last = results.cached_beta_summary(arms, self.cache, self.rows, self.settings, self.device)
            self.assertEqual(last['distributions']['inj_global']['overall']['samples'], 8)

    def test_decision_uses_unrounded_differences_and_strict_absolute_requirement(self):
        settings = {**self.settings, 'historical_true_macro_f1': 0.969}
        def metric(dynamic):
            return {'inj_fixed': {'macro_f1': 0.970}, 'inj_global': {'macro_f1': 0.9705},
                    'inj_dynamic': {'macro_f1': dynamic}}
        self.assertFalse(results.decisions(metric(0.9709999999), settings)['all_passed'])
        self.assertTrue(results.decisions(metric(0.9710000001), settings)['all_passed'])
        self.assertFalse(results.decisions(metric(0.9710000001), self.settings)['all_passed'])
        exact = {**settings, 'historical_true_macro_f1': 0.9711}
        self.assertFalse(results.decisions(metric(0.9711), exact)['dynamic_exceeds_historical_True'])

    def test_launcher_caught_failure_still_publishes_all_files_and_preserves_exit(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            args = SimpleNamespace(repo=repo, data_root=None, probe_only=False, publish_remote=None)
            def publish(code, repo, output, status, env, **kwargs):
                self.assertEqual(kwargs['output_name'], launcher.OUTPUT_NAME)
                self.assertEqual(status, 7)
                self.assertEqual(run.raw.read(output / 'training_exit.json')['exit_code'], 7)
                # Read every byte while the live launcher lock is held (Windows regression).
                for path in output.rglob('*'):
                    if path.is_file():
                        path.read_bytes()
                return [0, 0]
            with patch.object(launcher.subprocess, 'Popen', return_value=SimpleNamespace(wait=lambda: 7)), \
                 patch.object(launcher.publisher, 'publish_results', side_effect=publish):
                self.assertEqual(launcher.launch(args), 7)

    def test_unexpected_upload_failure_preserves_training_exit_and_retry_command(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            repo = Path(temp)
            args = SimpleNamespace(repo=repo, data_root=None, probe_only=False, publish_remote=None)
            with patch.object(launcher.subprocess, 'Popen', return_value=SimpleNamespace(wait=lambda: 7)), \
                 patch.object(launcher.publisher, 'publish_results', side_effect=PermissionError('publisher IO failure')), \
                 patch.object(launcher.sys, 'stderr', io.StringIO()):
                self.assertEqual(launcher.launch(args), 7)
            receipt = run.raw.read(repo / 'outputs' / launcher.OUTPUT_NAME / 'upload_failure.json')
            self.assertEqual(receipt['training_exit_code'], 7)
            self.assertIn('--only residual_injection_s42_v1', receipt['retry_command'])

    def test_complete_runner_probe_train_and_completed_resume_on_synthetic_raw_graphs(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            repo = Path(temp); output = repo / run.OUTPUT; output.mkdir(parents=True)
            data = TinyGraphs(self.graphs); data.root = repo / 'graphs'; data.root.mkdir()
            rows = {'val': self.rows, 'train': [{**r, 'partition': 'train'} for r in self.rows]}
            data.rows = rows['train'] + rows['val']
            data.metadata = {'partition_sizes': {'train': 8, 'val': 8, 'test': 8}}
            settings = {**self.settings, 'epochs': 2, 'training_samples': 8, 'validation_samples': 8, 'test_samples': 8}
            config = {'seed': 42, 'workers': 0, 'batch_size': 3, 'eval_batch_size': 4}
            selected = {'epoch': 48, 'validation': run.frozen.joint.study.validate_predictions(self.reference, self.rows, False),
                        'prediction_ids_sha256': run.frozen.joint.prediction_digest(self.reference)}
            gold = [{**r, 'flow_weights': [0.5] * 32} for r in self.reference]
            source = (self.source, data, rows, self.device, config, selected, gold, {}, self.binding)
            archive = SimpleNamespace(verified={'fixed': 'hash'}, verify_all=lambda: None)
            history = ({'provenance': {'binding': {'source': self.binding}}}, self.reference, archive)
            with patch.object(run, 'validate_settings'), patch.object(run, 'history_inputs', return_value=history), \
                 patch.object(run.frozen, 'load_source', return_value=source):
                probe = run.run(repo, None, output, settings, probe_only=True)
                self.assertFalse(probe['formal_training_started'])
                self.assertFalse((output / 'recovery.pt').exists())
                complete = run.run(repo, None, output, settings, resume=True)
                self.assertEqual(complete['status'], 'completed')
                state = torch.load(output / 'recovery.pt', map_location='cpu', weights_only=True)
                self.assertEqual(state['updates'], 6)
                run.run(repo, None, output, settings, resume=True)
                restored = torch.load(output / 'recovery.pt', map_location='cpu', weights_only=True)
                self.assert_tree_equal(state, restored)
            self.assertTrue(run.raw.read(output / 'epoch_zero_replay.json')['passed'])
            self.assertTrue(run.raw.read(output / 'final_verification.json')['best_raw_graph_replays_passed'])
            self.assertEqual(run.raw.read(output / 'final_verification.json')['test_graph_loads'], 0)


if __name__ == '__main__':
    unittest.main()
