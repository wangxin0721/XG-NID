"""CPU checks for strict loading, the validation gate, and fixed-split reports."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest

import torch

import evaluate_joint_cross_layer_query as run
from test_flow_query_joint_packet2 import TinyGraphs, joint_fixture


class FixedEvaluationChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        backbone, _, _, graphs = joint_fixture()
        base = run.joint.legacy.JointCrossLayerClassifier(backbone.flow_encoder, backbone.packet_encoder,
                                                        backbone.fusion, 'multi')
        self.device = torch.device('cpu')
        self.models = run.joint.make_models(base.state_dict(), self.device, 42)
        self.models['B37'] = copy.deepcopy(self.models['B'])
        with torch.no_grad():
            self.models['B37'].packet_pool.query.weight.fill_(0.1)
        test_graphs = [g.clone() for g in graphs]
        for graph in test_graphs:
            graph.sample_id = f'test-{graph.sample_id}'
        self.data = TinyGraphs(graphs + test_graphs)
        self.data.metadata = {'partition_sizes': {'val': 8, 'test': 8}}
        self.data.rows = [{'partition': part, 'sample_id': g.sample_id, 'label': int(g.y.item())}
                          for part, batch in (('val', graphs), ('test', test_graphs)) for g in batch]
        self.config = {**run.raw.read(run.CONFIG), 'partition_sizes': self.data.metadata['partition_sizes'],
                       'eval_batch_size': 4, 'workers': 0}

    def archive(self, source):
        archive = {arm: {'selected': {}} for arm in run.joint.ARMS}
        for name, spec in self.config['checkpoints'].items():
            arm, selection = spec['arm'], spec['selection']
            rows = run.evaluation_rows(self.data, 'val', self.config)
            batches = run.joint.legacy.loader(self.data, rows, 'val', self.config)
            metrics, records = run.raw.evaluate(self.models[name], batches, rows, self.device, export=True)
            folder = source / arm
            folder.mkdir(exist_ok=True)
            if selection == 'best':
                run.raw.save_predictions(folder / 'validation_predictions.jsonl.gz', records)
            archive[arm]['selected'][selection] = {
                'validation': metrics, 'prediction_ids_sha256': run.joint.prediction_digest(records)}
        return archive

    def test_validation_digest_failure_prevents_any_test_graph_load(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            archive = self.archive(root)
            archive['B']['selected']['trained_best']['prediction_ids_sha256'] = 'wrong'
            load = self.data.load

            def validation_only(row):
                self.assertEqual(row['partition'], 'val')
                return load(row)

            self.data.load = validation_only
            with self.assertRaisesRegex(ValueError, 'B37 validation replay failed'):
                run.evaluate_partitions(self.models, self.data, self.config, root, archive, self.device, root)
            self.assertFalse((root / 'test').exists())
            self.assertFalse(run.raw.read(root / 'validation_replay.json')['checkpoints']['B37']['passed'])

    def test_four_validation_and_test_passes_and_paper_tables(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            archive = self.archive(root)
            loaded = []
            load = self.data.load

            def track(row):
                loaded.append(row['partition'])
                return load(row)

            self.data.load = track
            results = run.evaluate_partitions(self.models, self.data, self.config, root, archive, self.device, root)
            self.assertEqual(loaded, ['val'] * 32 + ['test'] * 32)
            self.assertTrue(run.raw.read(root / 'validation_replay.json')['passed'])
            metrics = {part: {name: values[0] for name, values in arms.items()} for part, arms in results.items()}
            evidence = {part: run.write_partition(root / part, arms) for part, arms in results.items()}
            self.assertTrue(evidence['test']['A_B_prediction_check']['identical'])
            self.assertEqual(metrics['test']['B37']['samples'], 8)
            run.write_report(root, self.config, metrics, evidence)
            self.assertIn('C-B37', (root / 'report.md').read_text(encoding='utf-8'))
            self.assertIn('跨分支查询消融', (root / 'paper_tables.md').read_text(encoding='utf-8'))
            self.assertEqual(len((root / 'test' / 'per_class.csv').read_text().splitlines()), 33)

    def test_trained_mean_weights_load_into_the_mean_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = self.archive(root)
            protocol = run.raw.read(run.joint.CONFIG)
            for name, spec in self.config['checkpoints'].items():
                arm, selected = spec['arm'], archive[spec['arm']]['selected'][spec['selection']]
                state = self.models[name].state_dict()
                archive[arm]['frozen_state_sha256'] = run.joint.legacy.frozen_hash(self.models[name])
                protocol['source_frozen_state_sha256'] = archive[arm]['frozen_state_sha256']
                spec['macro_f1'] = selected['validation']['macro_f1']
                selected.update(epoch=spec['epoch'], checkpoint_sha256=spec['sha256'])
                torch.save({'model_state': state, 'epoch': spec['epoch'], 'validation': selected['validation'],
                            'prediction_ids_sha256': selected['prediction_ids_sha256'],
                            'config': {'arm': arm, 'query_source': run.joint.ARMS[arm],
                                       'labels': run.LABELS, 'protocol': protocol}}, root / spec['path'])
            config_file = root / 'train_config.json'
            run.raw.write(config_file, protocol)
            from unittest.mock import patch
            with patch.object(run.joint, 'CONFIG', config_file):
                models = run.load_models(root, self.config, archive, self.device)
            self.assertTrue(torch.equal(models['B37'].packet_pool.query.weight,
                                        self.models['B37'].packet_pool.query.weight))
            self.assertFalse(bool(models['B'].packet_pool.query.weight.ne(0).any()))
            self.assertTrue(all(not p.requires_grad for model in models.values() for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
