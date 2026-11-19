"""Check continuity, ordered context effects, staged gradients, and the existing trainer."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
from torch_geometric.data import Batch

import run_ordered_transition_readout as run
from test_packet_transition_readout import fixture
from xgnid.packet_transition_readout import PacketTransitionReadout


class OrderedReadoutChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.device = torch.device('cpu')

    def inputs(self):
        generator = torch.Generator().manual_seed(8)
        transitions = torch.randn(7, 70, generator=generator)
        lengths = torch.tensor([0, 3, 1, 3])
        # The last graph has a missing middle edge, so only its first pair is continuous.
        src, dst = [1, 2, 3, 5, 8, 9, 12], [2, 3, 4, 6, 9, 10, 13]
        packed = run.attach_continuity(transitions, src, dst, lengths)
        query = torch.randn(4, 64, generator=generator)
        return transitions, packed, lengths, query

    def test_continuity_respects_missing_links_and_graph_boundaries(self):
        original, packed, lengths, _ = self.inputs()
        self.assertEqual(packed[:, 70].tolist(), [1, 1, 0, 0, 1, 0, 0])
        torch.testing.assert_close(packed[:, :70], original, rtol=0, atol=0)
        crossed = run.attach_continuity(torch.zeros(2, 70), [0, 1], [1, 2], torch.tensor([1, 1]))
        self.assertEqual(crossed[:, 70].tolist(), [0, 0])

    def test_original_mode_is_exact_and_all_modes_preserve_empty_or_single_edge(self):
        values, packed, lengths, query = self.inputs()
        baseline = PacketTransitionReadout(42)
        single = run.OrderedTransitionReadout('single', 42)
        self.assertEqual(run.independent.model_state(single), run.independent.model_state(baseline))
        with torch.no_grad():
            baseline.output.weight.fill_(0.1)
            single.output.weight.fill_(0.1)
        expected, weights = baseline(query, values, lengths)
        actual, new_weights = single(query, packed, lengths)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        torch.testing.assert_close(weights, new_weights, rtol=0, atol=0)
        for mode, count in (('single', 3200), ('pointwise', 3728), ('ordered', 3728)):
            head = run.OrderedTransitionReadout(mode, 42)
            self.assertEqual(sum(p.numel() for p in head.parameters()), count)
            with torch.no_grad():
                head.output.weight.fill_(0.1)
            result, _ = head(query, packed, lengths)
            torch.testing.assert_close(result[2], expected[2], rtol=0, atol=0)
            self.assertFalse(bool(result[0].any()))
            empty, attention = head(query, packed[:0], torch.zeros_like(lengths))
            self.assertFalse(bool(empty.any()))
            self.assertEqual(attention.numel(), 0)

    def test_context_observes_order_and_cannot_read_another_graph(self):
        _, packed, lengths, query = self.inputs()
        point = run.OrderedTransitionReadout('pointwise', 42)
        ordered = run.OrderedTransitionReadout('ordered', 42)
        self.assertEqual(run.independent.model_state(point), run.independent.model_state(ordered))
        with torch.no_grad():
            point.output.weight.fill_(0.1)
            ordered.output.weight.fill_(0.1)
        before = ordered(query, packed, lengths)[0]
        self.assertGreater(float((before[1] - point(query, packed, lengths)[0][1]).abs().max()), 1e-6)
        changed = packed.clone()
        changed[4:, :70] += 50
        after = ordered(query, changed, lengths)[0]
        torch.testing.assert_close(before[:3], after[:3], rtol=0, atol=0)
        backwards = packed.clone()
        backwards[:3, :70] = packed[:3, :70].flip(0)
        self.assertGreater(float((ordered(query, backwards, lengths)[0][1] - before[1]).abs().max()), 1e-6)
        single = run.OrderedTransitionReadout('single', 42)
        with torch.no_grad():
            single.output.weight.fill_(0.1)
        torch.testing.assert_close(single(query, packed, lengths)[0],
                                   single(query, backwards, lengths)[0], rtol=1e-5, atol=1e-6)
        invalid = packed.clone()
        invalid[2, 70] = 1
        with self.assertRaisesRegex(ValueError, 'cross graph boundaries'):
            ordered(query, invalid, lengths)

    def test_zero_initialization_and_context_gradient_after_first_update(self):
        _, packed, lengths, query = self.inputs()
        head = run.OrderedTransitionReadout('ordered', 42)
        optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
        target = torch.ones(4, 64)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            residual, _ = head(query, packed, lengths)
            if not step:
                self.assertFalse(bool(residual.any()))
            ((residual - target)**2).mean().backward()
            self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters()))
            self.assertGreater(float(head.output.weight.grad.abs().sum()), 0)
            if step:
                self.assertGreater(float(head.context.weight.grad.abs().sum()), 0)
                self.assertGreater(float(head.query.weight.grad.abs().sum()), 0)
            else:
                self.assertEqual(float(head.context.weight.grad.abs().sum()), 0)
            optimizer.step()

    def test_old_training_entry_replays_new_heads_on_graphs(self):
        model, graphs = fixture()
        rows = [{'sample_id': g.sample_id, 'label': g.y.item()} for g in graphs]
        batch = Batch.from_data_list(graphs)
        features, _ = run.encode(model, batch, self.device)
        cache = run.original.FrozenCache([features], [r['label'] for r in rows])
        reference = run.original.prediction_records(rows, run.original.baseline(model, features).argmax(-1).tolist())
        settings = {**run.raw.read(run.CONFIG), 'epochs': 2}
        config = {'seed': 42, 'readout_seed': 42, 'batch_size': 4, 'eval_batch_size': 4,
                  'epochs': 2, 'learning_rate': 1e-3}
        heads = run.make_heads(self.device, settings)
        run.check_initial(heads, model, cache, self.device)
        with TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            for name, head in heads.items():
                folder = Path(temp) / name
                folder.mkdir()
                best, last, updates = run.original.fit(model, {'B': head}, cache, cache, rows,
                                                        reference, config, self.device, folder)
                self.assertEqual(updates, 4)
                orders = run.verify_ledger(folder, {'best': best, 'last': last}, settings, config,
                                          best['B']['validation'] if best['B']['epoch'] == 0 else
                                          run.joint.study.validate_predictions(reference, rows, probabilities=False),
                                          len(rows))
                self.assertEqual(len(orders), 2)
                head.load_state_dict(best['B']['state'], strict=True)
                predictions, _ = run.original.measure(model, {'B': head}, cache, rows, config, self.device)
                self.assertEqual(run.joint.prediction_digest(predictions['B']), best['B']['prediction_ids_sha256'])
                ledger = [json.loads(line) for line in (folder / 'epochs.jsonl').read_text().splitlines()]
                self.assertEqual(best['B']['epoch'], max(ledger, key=lambda x: x['validation']['B']['macro_f1'])['epoch'])
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
