"""Check checkpoint inheritance, isolated readout seeds, and the fixed-test gate."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

import run_independent_packet_transition as run
from test_packet_transition_readout import fixture


class IndependentTransitionChecks(unittest.TestCase):
    def test_freezing_reloaded_c_clears_last_step_gradients_without_changing_weights(self):
        torch.set_num_threads(2)
        _, graphs = fixture()
        batch = run.core.Batch.from_data_list(graphs)
        n0 = run.d32.baseline.build_experiment('E2', batch)
        run.core.initialize(n0, 'E2', batch)
        model = run.d32.model_from_stage('C', n0, torch.device('cpu'), 44)
        config = run.d32.stage_protocols(run.raw.read(run.d32.CONFIG))['C']
        optimizer = run.joint.optimizer(model, config)
        model.train()
        selected = {name: value.clone() for name, value in model.state_dict().items()}
        run.legacy.step(model, batch, optimizer)
        model.load_state_dict(selected, strict=True)
        model.requires_grad_(False).eval()
        self.assertTrue(any(param.grad is not None for param in model.parameters()))
        with torch.no_grad():
            expected = model(batch)

        self.assertEqual(run.freeze_c(model), run.raw.state_digest(selected))
        self.assertTrue(all(not param.requires_grad and param.grad is None for param in model.parameters()))
        self.assertTrue(all(not module.training for module in model.modules()))
        with torch.no_grad():
            torch.testing.assert_close(model(batch), expected, rtol=0, atol=0)

    def test_recovery_matches_training_digest_even_if_confusion_matrix_is_unchanged(self):
        original = [{'sample_id': 'x', 'label': 2, 'prediction': 0},
                    {'sample_id': 'y', 'label': 2, 'prediction': 1}]
        replay = [{**row, 'prediction': 1 - row['prediction']} for row in original]
        metrics = run.joint.study.validate_predictions(replay, probabilities=False)
        selected = {'validation': metrics, 'prediction_ids_sha256': run.joint.prediction_digest(original)}
        candidates = {0: {0: 5e-6}, 1: {1: 6e-6}}
        for archive in (None, original):
            receipt, result = run.near_tie_receipt(replay, metrics, selected, archive, candidates)
            self.assertTrue(receipt['passed'])
            self.assertFalse(receipt['exact_raw_match'])
            self.assertEqual(result, original)
            self.assertEqual(len(receipt['first_differences']), 2)
            receipt, result = run.near_tie_receipt(replay, metrics, selected, archive, {0: candidates[0]})
            self.assertFalse(receipt['passed'])
            self.assertIsNone(result)

    def test_completed_b_requires_full_budget_order_and_earliest_best(self):
        config = {'epochs': 30, 'batch_size': 512, 'readout_seed': 42}
        orders = [str(i) for i in range(30)]
        binding = {'seed': 43, 'split_id': 'split', 'manifest_sha256': 'manifest', 'n0_best_sha256': 'n0'}
        selected = {'checkpoint_sha256': 'parent', 'validation': {'macro_f1': 0.5}}
        ledger = [
            {'epoch': epoch, 'order_sha256': order, 'updates_per_arm': 211 if epoch else 0,
             'samples_per_arm': 107621 if epoch else 0, 'validation': {'B': {'macro_f1': 0.5 if not epoch else 0.6}}}
            for epoch, order in enumerate((None, *orders))
        ]
        saved = {'config': config, 'query_sources': {'B': 'packet_mean'},
                 'binding': {**binding, 'parent_sha256': 'parent'}, 'frozen_c_sha256': 'frozen',
                 'initial_transition_sha256': 'initial', 'best': {'B': {'epoch': 1, 'validation': {'macro_f1': 0.6}}},
                 'last': {'B': {'epoch': 30, 'validation': {'macro_f1': 0.6}}}}
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            run.raw.write(folder / 'epoch_zero_replay.json', {'passed': True})
            (folder / 'epochs.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in ledger), encoding='utf-8')
            run.verify_b_training(folder, saved, binding, selected, config, 'frozen', 'initial', orders, 107621)
            saved['best']['B']['epoch'] = 2
            with self.assertRaisesRegex(ValueError, 'best selection differs'):
                run.verify_b_training(folder, saved, binding, selected, config, 'frozen', 'initial', orders, 107621)
            orders[0] = 'changed'
            with self.assertRaisesRegex(ValueError, 'sampling or training budget differs'):
                run.verify_b_training(folder, saved, binding, selected, config, 'frozen', 'initial', orders, 107621)

    def test_child_stages_preserve_epoch_zero_and_optimizer_groups(self):
        torch.set_num_threads(2)
        _, graphs = fixture()
        batch = run.core.Batch.from_data_list(graphs)
        device = torch.device('cpu')
        n0 = run.d32.baseline.build_experiment('E2', batch)
        run.core.initialize(n0, 'E2', batch)
        parent = run.d32.model_from_stage('D32-Gate', n0, device, 43)
        for stage, params in (('CL-Last', 15400), ('JCL-Multi', 32616), ('C', 33640)):
            metadata = {'directory': Path('parent'), 'epoch': 1,
                        'validation': {'macro_f1': 0.9}, 'checkpoint_sha256': 'ckpt',
                        'predictions_sha256': 'pred'}
            config = run.protocol(stage, 43, metadata, 'split')
            child = run.build_child(stage, parent.state_dict(), n0, device, 43)
            with torch.no_grad():
                torch.testing.assert_close(child(batch), parent(batch), atol=1e-5, rtol=1e-5)
            opt = (run.cross.optimizer(run.core, child, config) if stage == 'CL-Last' else
                   (run.legacy.optimizer if stage == 'JCL-Multi' else run.joint.optimizer)(child, config))
            self.assertEqual(sum(p.numel() for group in opt.param_groups for p in group['params']), params)
            self.assertEqual(config['seed'], 43)
            parent = child
        settings = run.raw.read(run.CONFIG)
        heads = [run.transition.make_heads({'seed': seed, 'readout_seed': 42}, device, ('B',))['B']
                 for seed in settings['seeds']]
        self.assertEqual([run.model_state(head) for head in heads], [settings['readout_initial_state_sha256']] * 2)

    def test_fixed_test_loading_waits_for_both_pairs(self):
        settings = run.raw.read(run.CONFIG)
        with TemporaryDirectory() as temp:
            output = Path(temp)
            with patch.object(run.d32.baseline, 'runtime', return_value=torch.device('cpu')), \
                 patch.object(run, 'load_data', return_value=({'split_directory': 'split', 'split_json_sha256': 'x'},
                     type('Data', (), {'metadata': {'sample_manifest': 'manifest', 'sample_manifest_sha256': 'y'}})(),
                     {'val': []})), \
                 patch.object(run, 'load_locked_pair', side_effect=[(None, None, {}, {}),
                                                                    ValueError('Seed44 incomplete')]) as loads, \
                 patch.object(run.evaluation, 'TestGraphs') as test_graphs:
                with self.assertRaisesRegex(ValueError, 'Seed44 incomplete'):
                    run.test_pairs(output, output, None, settings, 'commit')
                self.assertEqual([call.args[1] for call in loads.call_args_list], [43, 44])
                test_graphs.assert_not_called()


if __name__ == '__main__':
    unittest.main()
