"""Exercise the seed-ordered validation gate and paper tables."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import evaluate_early_baselines_final_test as run
from test_flow_query_joint_packet2 import TinyGraphs
from test_packet_transition_readout import fixture


class EarlyBaselinesFinalTestChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        _, graphs = fixture()
        held_out = [graph.clone() for graph in graphs]
        for graph in held_out:
            graph.sample_id = 'test-' + graph.sample_id
        self.data = TinyGraphs(graphs + held_out)
        self.data.root = Path('.')
        self.data.directory = Path('.')
        self.data.metadata = {'partition_sizes': {'val': 8, 'test': 8}, 'sample_manifest_sha256': 'fixed',
                              'sample_manifest': 'samples.jsonl'}
        self.data.rows = [{'partition': part, 'sample_id': graph.sample_id, 'label': graph.y.item()}
                          for part, batch in (('val', graphs), ('test', held_out)) for graph in batch]
        self.config = {**run.raw.read(run.evaluation.CONFIG), 'split_directory': 'split',
                       'partition_sizes': {'val': 8, 'test': 8}, 'eval_batch_size': 4, 'workers': 0}
        self.device = torch.device('cpu')
        self.val_rows = run.evaluation.evaluation_rows(self.data, 'val', self.config)
        self.models = {}
        warm = run.grouped.Batch.from_data_list(graphs)
        for arm, architecture in (('E0FT', 'E0'), ('N0', 'E2')):
            initial = run.build_experiment(architecture, warm)
            run.grouped.initialize(initial, architecture, warm)
            saved = {'model_state': initial.state_dict(),
                     'config': {'total_parameters': sum(p.numel() for p in initial.parameters())}}
            model = run.load_model(f'{arm}42', saved, warm, self.device)
            self.assertTrue(all(not param.requires_grad for param in model.parameters()))
            self.models.update({f'{arm}{seed}': model for seed in run.SEEDS})
        loader = run.DataLoader(run.grouped.GraphRows(self.data, self.val_rows), batch_size=4, shuffle=False)
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            self.val = run.evaluate_models(self.models, loader, self.val_rows, Path(tmp), 'fixture', self.device)
        self.metrics = {name: run.joint.study.validate_predictions(rows, self.val_rows, probabilities=False)
                        for name, rows in self.val.items()}

    def exercise(self, output, mode, mismatch=None):
        order = []
        fixture_metric = self.metrics['N042']
        main_metrics = dict.fromkeys(('A', 'B42', 'B43', 'B44', 'C1_42', 'C1_43', 'C1_44', 'U42'), fixture_metric)
        stats = {'B-A': {'mean_pp': 0, 'sample_sd_pp': 0}}
        main = {'partitions': {part: {'metrics': main_metrics} for part in ('val', 'test')},
                'aggregate': dict.fromkeys(('val', 'test'), stats)}
        probes = {'metrics': dict.fromkeys(('D32_Linear', 'D32_Nonlinear', 'Equal32_Linear',
                                           'Equal32_Nonlinear'), fixture_metric)}
        registered = {name: {'directory': name, 'sha256': {'best.pt': name}} for name in run.NAMES}
        settings = {'models': registered, 'archive_commit': 'fixture', 'sample_manifest_sha256': 'fixed',
                    'paper_inputs': dict.fromkeys((run.PAPER_ARCHIVE, run.FUSION_DIAGNOSIS), 'fixture')}
        n0_registry = {str(seed): {'directory': f'N0{seed}',
                                  'metrics': {'checkpoint_sha256': {'best.pt': f'N0{seed}'}}} for seed in run.SEEDS}
        inputs = {run.CONFIG: settings, run.evaluation.CONFIG: self.config,
                  run.raw.ROOT / 'configs/fair_vector_n0_references_20261029.json': n0_registry,
                  output / run.PAPER_ARCHIVE: main, output / run.FUSION_DIAGNOSIS: probes}

        def archive(repo, name, *args):
            order.append(name)
            digest = run.joint.prediction_digest(self.val[name]) if name != mismatch else 'wrong'
            return ({'config': {'binding': {'sources': {}}}},
                    {'epoch': 1, 'validation': self.metrics[name], 'prediction_ids_sha256': digest},
                    self.val[name], {})

        with patch.object(run.raw, 'read', side_effect=inputs.__getitem__), \
             patch.object(run.evaluation, 'verify_inputs'), \
             patch.object(run.grouped, 'FrozenGraphs', return_value=self.data), \
             patch.object(run, 'runtime', return_value=self.device), \
             patch.object(torch.cuda, 'get_device_name', return_value='CPU fixture'), \
             patch.object(run, 'load_archive', side_effect=archive), \
             patch.object(run, 'load_model', side_effect=lambda name, *args: self.models[name]), \
             patch.object(self.data, 'load', wraps=self.data.load) as load:
            if mismatch:
                with self.assertRaisesRegex(ValueError, 'Seed42 validation replay failed'):
                    run.run(output, output, None, 'a' * 40, mode)
            else:
                run.run(output, output, None, 'a' * 40, mode)
            partitions = [args[0]['partition'] for args, _ in load.call_args_list]
            only_seed42 = mode == 'seed42-replay' or mismatch
            self.assertEqual(order, list(run.NAMES[:2] if only_seed42 else run.NAMES))
            self.assertEqual(partitions, ['val'] * 12 if only_seed42 else ['val'] * 28 + ['test'] * 8)

    def test_seed42_failure_blocks_later_replay_and_test(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            self.exercise(root, 'full', mismatch='N042')
            receipt = run.raw.read(root / 'validation_replay.json')
            self.assertFalse(receipt['passed'])
            self.assertFalse((root / 'test').exists())

    def test_seed42_replay_can_run_independently(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            self.exercise(root, 'seed42-replay')
            receipt = run.raw.read(root / 'validation_replay.json')
            self.assertTrue(receipt['passed'])
            self.assertFalse(receipt['all_six_passed'])
            self.assertFalse((root / 'test').exists())

    def test_full_test_exports_models_and_paper_tables(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            self.exercise(root, 'full')
            receipt = run.raw.read(root / 'validation_replay.json')
            self.assertTrue(receipt['all_six_passed'])
            table = (root / 'paper_tables.md').read_text(encoding='utf-8')
            self.assertIn('N0−E0-FT', table)
            self.assertIn('D32 动态门控', table)
            self.assertIn('均匀聚合', table)
            self.assertEqual(len(run.raw.read(root / 'test/metrics.json')), 6)


if __name__ == '__main__':
    unittest.main()
