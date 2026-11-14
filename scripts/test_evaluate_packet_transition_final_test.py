"""Check single encoding per batch and the validation gate before fixed testing."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import evaluate_packet_transition_final_test as run
from test_flow_query_joint_packet2 import TinyGraphs
from test_packet_transition_readout import fixture


class TransitionFinalTestChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.model, graphs = fixture()
        held_out = [graph.clone() for graph in graphs]
        for graph in held_out:
            graph.sample_id = 'test-' + graph.sample_id
        self.data = TinyGraphs(graphs + held_out)
        self.data.root = Path('.')
        self.data.metadata = {'partition_sizes': {'val': 8, 'test': 8}, 'sample_manifest_sha256': 'fixed'}
        self.data.rows = [{'partition': part, 'sample_id': graph.sample_id, 'label': graph.y.item()}
                          for part, batch in (('val', graphs), ('test', held_out)) for graph in batch]
        self.rows = {part: [row for row in self.data.rows if row['partition'] == part] for part in ('val', 'test')}
        self.config = {'seed': 42, 'eval_batch_size': 4, 'workers': 0, 'partition_sizes': {'val': 8}}
        self.device = torch.device('cpu')
        specs = [(f'B{s}', 'B') for s in run.SEEDS] + [(f'C1_{s}', 'C1') for s in run.SEEDS] + [('U42', 'U')]
        self.heads = {name: (arm, run.transition.PacketTransitionReadout().requires_grad_(False).eval())
                      for name, arm in specs}

    def exercise(self, output, mismatched):
        val_loader = run.joint.legacy.loader(self.data, self.rows['val'], 'val', self.config)
        val, lengths = run.evaluate_partition(self.model, self.heads, val_loader, self.rows['val'], self.device,
                                              output, 'fixture')
        selected = {name: {'epoch': 48 if name == 'A' else 18,
                          'validation': run.joint.study.validate_predictions(items, self.rows['val'], probabilities=False),
                          'prediction_ids_sha256': run.joint.prediction_digest(items)} for name, items in val.items()}
        if mismatched:
            selected['U42']['prediction_ids_sha256'] = 'wrong'
        # Validation and test fixtures have identical graphs and distinct IDs.
        test_gold = run.transition.prediction_records(self.rows['test'], [row['prediction'] for row in val['A']])
        test_metrics = run.joint.study.validate_predictions(test_gold, self.rows['test'], probabilities=False)
        archive = output / 'history'
        archive.mkdir()
        run.raw.save_predictions(archive / 'C_predictions.jsonl.gz', test_gold)
        run.raw.write(archive / 'metrics.json', {'C': test_metrics})
        settings = {'sample_manifest_sha256': 'fixed', 'backbone_state_sha256': run.model_digests(self.model, {})['A'],
                    'test_samples': 8, 'baseline_test_archive': {'directory': 'history',
                    'sha256': {path.name: run.raw.sha(path) for path in archive.iterdir()},
                    'macro_f1': test_metrics['macro_f1'], 'prediction_ids_sha256': run.joint.prediction_digest(test_gold)}}
        actual_read = run.raw.read
        with patch.object(run.raw, 'read', side_effect=lambda path: settings if path == run.CONFIG else actual_read(path)), \
             patch.object(run.transition, 'base_config', return_value=self.config), \
             patch.object(run.transition, 'load_source', return_value=(self.model, self.data, self.rows, self.device,
                                                                       selected['A'], {})), \
             patch.object(run, 'load_readouts', return_value=(self.heads, selected, val, lengths, {})), \
             patch.object(torch.cuda, 'get_device_name', return_value='CPU fixture'), \
             patch.object(self.data, 'load', wraps=self.data.load) as load, \
             patch.object(run.transition, 'encode', wraps=run.transition.encode) as encode:
            if mismatched:
                with self.assertRaisesRegex(ValueError, 'Validation replay'):
                    run.run(output, output, None, 'a' * 40)
            else:
                run.run(output, output, None, 'a' * 40)
            graph_reads = [args[0]['partition'] for args, _ in load.call_args_list]
            self.assertEqual(graph_reads, ['val'] * 8 if mismatched else ['val'] * 8 + ['test'] * 8)
            self.assertEqual(encode.call_count, 2 if mismatched else 4)

    def test_validation_mismatch_blocks_test_graphs(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            output = Path(tmp)
            self.exercise(output, mismatched=True)
            self.assertFalse(run.raw.read(output / 'validation_replay.json')['passed'])
            self.assertFalse((output / 'test').exists())

    def test_locked_heads_share_encoding_and_export_fixed_test(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            output = Path(tmp)
            self.exercise(output, mismatched=False)
            self.assertTrue(run.raw.read(output / 'baseline_test_replay.json')['passed'])
            self.assertTrue(run.raw.read(output / 'provenance.json')['frozen_states_verified'])
            summary = run.raw.read(output / 'summary.json')
            self.assertEqual(len(summary['partitions']['test']['metrics']), 8)
            self.assertEqual(len(summary['partitions']['test']['comparisons']), 8)
            self.assertEqual(summary['aggregate']['test']['B-A']['mean_pp'], 0)
            self.assertEqual(summary['partitions']['test']['graphs_without_valid_pairs'], 1)


if __name__ == '__main__':
    unittest.main()
