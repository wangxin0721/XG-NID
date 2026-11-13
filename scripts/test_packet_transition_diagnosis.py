"""Exercise chronological edge accounting, frozen readouts and linear probes."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

import torch
from torch_geometric.data import Batch

import run_packet_transition_diagnosis as run
from test_flow_query_joint_packet2 import TinyGraphs, joint_fixture
from xgnid.model import FLOW_TO_PACKET, PACKET_TO_PACKET


class TransitionDiagnosisChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_packet_fields_are_mapped_by_node_and_reverse_edges_are_not_counted_twice(self):
        _, _, _, graphs = joint_fixture()
        graph = graphs[1]
        graph[FLOW_TO_PACKET].edge_index = torch.tensor([[0, 0, 0], [2, 0, 1]])
        graph[FLOW_TO_PACKET].edge_attr = torch.tensor([[1., 0., 0., 100.],
                                                        [0., 0., 0., 0.],
                                                        [1., 0., 0., 50.]])
        graph[PACKET_TO_PACKET].edge_index = torch.tensor([[1, 0, 1, 2, 0], [0, 1, 2, 1, 2]])
        graph[PACKET_TO_PACKET].edge_attr = torch.tensor([[0.], [0.], [10.], [10.], [100.]])
        batch = Batch.from_data_list([graph])
        records, stats = run.audit_batch(batch)
        record = records[0]
        self.assertEqual(record['direction_groups'], [0, 1, 0, 1])
        self.assertEqual(record['valid_pairs'], 2)
        self.assertEqual(record['zero_dt_pairs'], 1)
        self.assertEqual(record['zero_payload_endpoints'], 1)
        self.assertEqual(record['reverse_adjacent_links'], 2)
        self.assertEqual(record['nonadjacent_links'], 1)
        self.assertEqual(run.aggregate(records)['valid_pair_ratio'], 1)
        self.assertAlmostEqual(float(stats[0, 7]), float(torch.log1p(torch.tensor(50.))), places=6)
        self.assertAlmostEqual(float(stats[0, 13]), float(torch.log1p(torch.tensor(10.))), places=6)

        graph[FLOW_TO_PACKET].edge_attr[2, 0] = 9
        records, stats = run.audit_batch(Batch.from_data_list([graph]))
        self.assertEqual(records[0]['invalid_direction_forward_links'], 2)
        self.assertEqual(records[0]['valid_pairs'], 0)
        self.assertFalse(stats.any())

    def test_frozen_features_reproduce_c_and_probes_share_the_same_batches(self):
        backbone, _, batch, graphs = joint_fixture()
        model = run.joint.JointCrossLayerQueryClassifier(
            backbone.flow_encoder, backbone.packet_encoder, backbone.fusion, 'flow').requires_grad_(False).eval()
        before = run.raw.state_digest(model.state_dict())
        with torch.no_grad():
            expected = model(batch).argmax(-1).tolist()
        features, actual = run.frozen_features(model, batch)
        self.assertEqual(actual, expected)
        self.assertEqual(features['D32Fused'].shape, (8, 32))
        config = {'seed': 42, 'workers': 0, 'eval_batch_size': 4, 'batch_size': 4,
                  'epochs': 2, 'learning_rate': 0.001, 'weight_decay': 0}
        rows = {part: [{'sample_id': graph.sample_id, 'label': graph.y.item(), 'partition': part}
                       for graph in graphs] for part in ('train', 'val')}
        data = TinyGraphs(graphs)
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            train, _, _ = run.extract(model, data, rows['train'], 'train', config, torch.device('cpu'), root)
            val, audit, predictions = run.extract(model, data, rows['val'], 'val', config, torch.device('cpu'), root)
            exports, updates = run.train_probes(train, val, rows['train'], rows['val'], config, torch.device('cpu'), root)
            metrics, probes, subsets, matched = run.diagnostics(dict.fromkeys(('C', 'A', 'B37'), predictions),
                                                                 exports, audit)
            fields = {'train': {'all': run.aggregate(audit)}, 'val': {'all': run.aggregate(audit)},
                      'hard_cases': {name: run.aggregate(audit[i] for i in ids) for name, ids in subsets.items()}}
            run.write_report(root, metrics, probes, fields, config, updates, 0)
            ledger = run.joint.query.read_history(root / 'epochs.jsonl')
            self.assertEqual([r['updates_per_probe'] for r in ledger], [2, 2])
            self.assertEqual(set(ledger[0]['train_ce']), set(run.FEATURES))
            self.assertIn('PacketMeanTransition', (root / 'report.md').read_text(encoding='utf-8'))
        self.assertEqual(updates, 4)
        self.assertEqual(len(matched), 8)
        self.assertEqual(run.raw.state_digest(model.state_dict()), before)


if __name__ == '__main__':
    unittest.main()
