"""Exercise the combined raw-graph path and its epoch recovery on CPU."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import run_joint_cross_layer_query as run
from test_flow_query_joint_packet2 import TinyGraphs, joint_fixture
from xgnid.packet_pooling import ordered_mean_pool


class CombinedChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.device = torch.device('cpu')
        self.config = {**run.raw.read(run.CONFIG), 'epochs': 3, 'prefix_epochs': 2, 'batch_size': 4,
                       'eval_batch_size': 4, 'workers': 0, 'training_samples': 8,
                       'validation_samples': 8, 'updates_per_epoch': 2}
        backbone, _, self.batch, self.graphs = joint_fixture()
        source = run.legacy.JointCrossLayerClassifier(backbone.flow_encoder, backbone.packet_encoder,
                                                       backbone.fusion, 'multi').eval()
        with torch.no_grad():
            source.cross_layer.weight.normal_(std=0.08)
            source.cross_layer.bias.normal_(std=0.05)
        self.models = run.make_models(source.state_dict(), self.device, 42)
        self.rows = {part: [{'sample_id': g.sample_id, 'label': g.y.item(), 'partition': part} for g in self.graphs]
                     for part in ('train', 'val')}
        self.data = TinyGraphs(self.graphs)

    def gold(self):
        batches = run.legacy.loader(self.data, self.rows['val'], 'val', self.config)
        return run.raw.evaluate(self.models['A'], batches, self.rows['val'], self.device, export=True)

    def test_epoch_zero_and_stratified_gradient_checks(self):
        baseline, gold = self.gold()
        before = run.raw.rng_hash(run.raw.rng_state())
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            output = Path(tmp)
            batches = run.legacy.loader(self.data, self.rows['val'], 'val', self.config)
            receipt = run.check_initial(self.models, batches, self.rows['val'], gold, baseline, self.device, output)
        self.assertTrue(receipt['passed'])
        self.assertEqual(receipt['samples'], 8)
        self.assertTrue(all(v == 0 for errors in receipt['max_abs_errors'].values() for v in errors.values()))
        self.assertEqual(len({run.legacy.frozen_hash(m) for m in self.models.values()}), 1)
        for arm, model in self.models.items():
            result = run.probe(model, self.batch, self.config)
            self.assertEqual(result['trainable_parameters'], 32616 if arm == 'A' else 33640)
            if arm != 'A':
                self.assertEqual(result['steps'][0]['query_key_gradient_norms']['key'], 0)
                self.assertIn('key', result['steps'][1]['updated_query_key'])
        self.assertEqual(run.raw.rng_hash(run.raw.rng_state()), before)

    def test_query_readout_handles_empty_and_single_packet_graphs(self):
        flow = torch.randn(3, 64)
        nodes = torch.randn(3, 64)
        index = torch.tensor([0, 2, 2])
        expected = ordered_mean_pool(nodes, index, 3)
        for arm in ('B', 'C'):
            pooled, weights = self.models[arm].packet_pool(flow, nodes, index, 3)
            torch.testing.assert_close(pooled, expected, rtol=0, atol=0)
            torch.testing.assert_close(weights, torch.tensor([1., 0.5, 0.5]), rtol=0, atol=0)

    def test_recovery_repeats_only_uncommitted_epoch_and_matches_full_run(self):
        baseline, gold = self.gold()
        orders = run.study.epoch_orders(8, 4, 42, 3)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            full, interrupted = Path(tmp) / 'full', Path(tmp) / 'interrupted'

            def fit(output):
                return run.fit('C', copy.deepcopy(self.models['C']), self.data, self.rows,
                               gold, baseline, orders, self.config, {}, self.device, output)

            expected, expected_rows = fit(full)
            commit = run.commit_history

            def interrupt_before_commit(path, history):
                if len(history) == 2:
                    raise RuntimeError('synthetic interruption before ledger commit')
                commit(path, history)

            with patch.object(run, 'commit_history', side_effect=interrupt_before_commit):
                with self.assertRaisesRegex(RuntimeError, 'synthetic interruption'):
                    fit(interrupted)
            arm = interrupted / 'C'
            self.assertEqual(len(run.query.read_history(arm / 'epochs.jsonl')), 1)
            self.assertTrue((arm / 'checkpoints' / 'epoch_002.pt').is_file())
            actual, actual_rows = fit(interrupted)
            self.assertEqual(expected_rows, actual_rows)
            self.assertEqual(actual['updates'], 6)
            for key in expected['selected']:
                self.assertEqual(expected['selected'][key]['epoch'], actual['selected'][key]['epoch'])
                self.assertEqual(expected['selected'][key]['validation'], actual['selected'][key]['validation'])
            saved = [torch.load(output / 'C' / 'last.pt', map_location='cpu', weights_only=False)
                     for output in (full, interrupted)]
            for key in ('model_state', 'optimizer_state', 'sampler_rng', 'loader_rngs'):
                run.query.assert_tree_equal(saved[0][key], saved[1][key], key)
            self.assertEqual(run.raw.rng_hash(saved[0]['rng']), run.raw.rng_hash(saved[1]['rng']))
            run.report(interrupted, self.config, dict.fromkeys(run.ARMS, actual),
                       dict.fromkeys(run.ARMS, actual_rows), baseline)
            summary = run.raw.read(interrupted / 'summary.json')
            self.assertEqual(summary['analysis']['comparisons']['C_minus_B']['macro_f1_delta_pp'], 0)
            self.assertFalse(summary['analysis']['priority_recheck_candidate'])
            self.assertIn('Spoofing', (interrupted / 'per_class.csv').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
