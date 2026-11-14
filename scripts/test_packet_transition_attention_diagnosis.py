"""Check uniform by-pass, replay strata, and archived order enforcement."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import Batch

import run_packet_transition_attention_diagnosis as run
from test_flow_query_joint_packet2 import TinyGraphs
from test_packet_transition_readout import fixture


class AttentionDiagnosisChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.device = torch.device('cpu')

    def test_same_weights_keep_empty_and_single_edge_predictions(self):
        model, graphs = fixture()
        batch = Batch.from_data_list(graphs[:3])
        features, _ = run.original.encode(model, batch, self.device)
        head = run.original.PacketTransitionReadout()
        with torch.no_grad():
            head.output.weight.fill_(0.125)
        observer = run.AttentionReplay(model, head)
        observer(features)
        self.assertEqual(observer.nonuniform_graphs, 1)
        reference, equal = observer.predictions['C1'], observer.predictions[run.ABLATION]
        self.assertEqual((reference[0], reference[2]), (equal[0], equal[2]))
        with patch.object(head.query, 'forward', side_effect=AssertionError('Unexpected query')):
            residual, weights = head(None, features.transitions, features.lengths)
        self.assertEqual(weights.tolist(), [0.5, 0.5, 1])
        self.assertFalse(residual[0].any())

    def test_archived_order_is_checked_and_uniform_checkpoint_replays(self):
        model, graphs = fixture()
        data = TinyGraphs(graphs)
        rows = {part: [{'sample_id': g.sample_id, 'label': g.y.item(), 'partition': part} for g in graphs]
                for part in ('train', 'val')}
        data.rows = rows['train']
        config = {'seed': 42, 'epochs': 2, 'batch_size': 4, 'eval_batch_size': 4, 'workers': 0,
                  'learning_rate': 0.001}
        heads = run.original.make_heads(config, self.device, ('U',))
        expected_params = sum(p.numel() for p in heads['U'].parameters() if p.requires_grad)
        self.assertEqual(expected_params, 2160)
        checks = run.original.probe(model, data, config, self.device, ('U',))
        self.assertEqual(set(checks['gradient_steps']['U'][0]['gradient_norms']),
                         {'edge.weight', 'edge.bias', 'output.weight'})
        self.assertTrue(run.original.check_attention(heads['U'], self.device, uniform=True)['empty_graph_zero_residual'])
        heads = run.original.make_heads(config, self.device, ('U',))
        sampler = run.raw.TrainBatches(len(rows['train']), config['batch_size'], config['seed'])
        orders = {}
        for epoch in range(1, config['epochs'] + 1):
            list(sampler)
            orders[epoch] = sampler.order_sha256
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            output = Path(tmp)
            edges, _, _ = run.original.cache_split(model, data, rows['train'], 'train', config, self.device, output)
            val, _, reference = run.original.cache_split(model, data, rows['val'], 'val', config,
                                                         self.device, output, heads)
            bad = output / 'bad_order'
            bad.mkdir()
            with self.assertRaisesRegex(ValueError, 'training order differs'):
                run.original.fit(model, heads, edges, val, rows['val'], reference, config,
                                 self.device, bad, order_reference={1: 'wrong'})
            best, last, updates = run.original.fit(model, heads, edges, val, rows['val'], reference, config,
                                                   self.device, output, order_reference=orders)
            self.assertEqual(updates, 4)
            self.assertEqual(last['U']['epoch'], 2)
            ledger = [json.loads(line) for line in (output / 'epochs.jsonl').read_text().splitlines()]
            self.assertEqual([item['order_sha256'] for item in ledger], [None, *orders.values()])
            replay = run.original.replay_selected(model, {'best': best}, data, rows['val'], config,
                                                  self.device, output)
            self.assertEqual(replay['A'], reference)
            self.assertEqual(run.joint.prediction_digest(replay['U']), best['U']['prediction_ids_sha256'])
            metrics, comparisons = run.comparison_outputs(output, rows['val'], replay,
                                                           val.features.lengths.tolist(), [('U', 'A')])
            strata = comparisons['U-A']['by_valid_edges']
            self.assertEqual(sum(item['samples'] for item in strata.values()), len(graphs))
            self.assertEqual(strata['0']['changed_predictions'], 0)
            self.assertEqual(metrics['U'], best['U']['validation'])


if __name__ == '__main__':
    unittest.main()
