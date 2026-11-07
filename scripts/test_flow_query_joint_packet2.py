"""Resume continuity, inherited selection and raw/cache probe checks on CPU."""
import contextlib
import copy
import io
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch_geometric.data import Batch, HeteroData

import run_flow_query_joint_packet2 as run
from test_packet_pooling_tensor import Encoder
from xgnid.flow_query_pooling import QueryPacketClassifier
from xgnid.fusion_dimension import FusionHead
from xgnid.attention_operator_pair import stream
from xgnid.model import FLOW_TO_PACKET, PACKET_TO_FLOW, PACKET_TO_PACKET


def fixture(mode='flow', seed=42):
    run.raw.seed_all(19)
    names = ('flow_projection', 'packet_projection', 'gate_hidden', 'gate_output', 'head_hidden', 'head_output')
    fusion = FusionHead(32, {name: stream(42, name) for name in names})
    model = QueryPacketClassifier(Encoder(), Encoder(), fusion, mode, seed).eval()
    graphs, ptr = [], [0]
    for i, count in enumerate((1, 3, 2, 4, 2, 3, 1, 2, 3)):
        graph = HeteroData()
        graph['flow'].x, graph['packet'].x = torch.randn(1, 64), torch.randn(count, 64)
        graph.y, graph.sample_id = torch.tensor([i % 8]), str(i)
        graphs.append(graph)
        ptr.append(ptr[-1]+count)
    batch = Batch.from_data_list(graphs)
    flow, nodes, _, _ = model.encode(batch)
    cache = {'ids': [g.sample_id for g in graphs], 'flow64': flow, 'packet64': nodes,
             'ptr': torch.tensor(ptr), 'labels': batch.y}
    return model, cache, batch


def config(seed=42, epochs=4):
    return {'seed': seed, 'epochs': epochs, 'prefix_epochs': 2, 'batch_size': 4,
            'eval_batch_size': 4, 'head_lr': 1e-5, 'pool_lr': 1e-4, 'weight_decay': 0., 'raw_probe_updates': 3}


def joint_fixture(mode='flow'):
    run.raw.seed_all(23)
    base = run.raw.OperatorClassifier('GAT', seed=42)
    model = run.JointQueryPacketClassifier(base.flow_encoder, base.packet_encoder, base.fusion, mode, 42).eval()
    graphs = []
    for i in range(8):
        graph, count = HeteroData(), 2+i % 2
        graph['flow'].x = torch.randn(1, 82)
        graph['packet'].x = torch.randn(count, 1500)
        graph[FLOW_TO_PACKET].edge_index = torch.stack((torch.zeros(count, dtype=torch.long), torch.arange(count)))
        graph[PACKET_TO_FLOW].edge_index = graph[FLOW_TO_PACKET].edge_index.flip(0)
        pairs = [(a, b) for a in range(count) for b in range(count) if a != b]
        # Include the original empty-relation case without adding a mask rule.
        graph[PACKET_TO_PACKET].edge_index = (torch.tensor(pairs).t().contiguous() if i else torch.empty(2, 0, dtype=torch.long))
        for relation, width in ((FLOW_TO_PACKET, 4), (PACKET_TO_FLOW, 4), (PACKET_TO_PACKET, 1)):
            graph[relation].edge_attr = torch.randn(graph[relation].edge_index.shape[1], width)
        graph.y, graph.sample_id = torch.tensor([i]), str(i)
        graphs.append(graph)
    batch = Batch.from_data_list(graphs)
    with torch.no_grad():
        flow, nodes, index, _ = model.encode(batch)
    lengths = torch.bincount(index)
    cache = {'ids': list(batch.sample_id), 'flow64': flow, 'packet64': nodes,
             'ptr': torch.cat((torch.zeros(1, dtype=torch.long), lengths.cumsum(0))), 'labels': batch.y}
    return model, cache, batch, graphs


class TinyGraphs:
    def __init__(self, graphs):
        self.graphs = {graph.sample_id: graph for graph in graphs}

    def load(self, row):
        return self.graphs[row['sample_id']].clone()


class ResumeChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.device = torch.device('cpu')
        self.temporary = run.raw.ROOT/'tmp'/'flow_query_joint_checks'
        self.temporary.mkdir(parents=True, exist_ok=True)

    def test_uninterrupted_equals_resumed_for_both_query_sources(self):
        # Exercises the actual fit/save/strict-reload path, including inherited
        # best checkpoints, copied prefix ledger and all subsequent Adam updates.
        for name, mode in zip(run.NAMES, ('packet_mean', 'flow')):
            with self.subTest(name=name), tempfile.TemporaryDirectory(dir=self.temporary) as tmp:
                root = Path(tmp)
                for folder in ('full', 'first', 'resumed'):
                    (root/folder).mkdir()
                initial, cache, _ = fixture(mode)
                caches, c = {'train': cache, 'val': cache}, config()
                with contextlib.redirect_stdout(io.StringIO()):
                    full, _ = run.fit(name, copy.deepcopy(initial), caches, self.device, c, {}, root/'full')
                    first, _ = run.fit(name, copy.deepcopy(initial), caches, self.device,
                                       config(epochs=2), {}, root/'first')
                    source = root/'first'/name
                    saved = torch.load(source/'last.pt', map_location='cpu', weights_only=False)
                    history = run.read_history(source/'epochs.jsonl')
                    resume = {'saved': saved, 'history': history, 'result': first, 'folder': source}
                    candidate = copy.deepcopy(initial)
                    candidate.load_state_dict(saved['model_state'])
                    # Advance every RNG before restoring, as a real restarted
                    # process and the preflight checks do.
                    random.random(); np.random.rand(3); torch.rand(11)
                    resumed, _ = run.fit(name, candidate, caches, self.device, c, {}, root/'resumed', resume,
                                         full['orders'])
                for filename in ('last.pt', 'trained_best.pt', 'best.pt', 'best150.pt'):
                    a = torch.load(root/'full'/name/filename, map_location='cpu', weights_only=False)
                    b = torch.load(root/'resumed'/name/filename, map_location='cpu', weights_only=False)
                    run.assert_tree_equal(a['model_state'], b['model_state'])
                    run.assert_tree_equal(a['optimizer_state'], b['optimizer_state'])
                    self.assertTrue(torch.equal(a['sampler_rng'], b['sampler_rng']))
                    self.assertEqual(run.raw.rng_hash(a['rng']), run.raw.rng_hash(b['rng']))
                    self.assertEqual((a['epoch'], a['validation']), (b['epoch'], b['validation']))
                self.assertEqual(run.read_history(root/'full'/name/'epochs.jsonl'),
                                 run.read_history(root/'resumed'/name/'epochs.jsonl'))
                self.assertTrue((root/'resumed'/name/'epochs.jsonl').read_bytes().startswith(
                    (source/'epochs.jsonl').read_bytes()))
                self.assertEqual(full['best_validation'], resumed['best_validation'])
                self.assertEqual(full['best_epoch'], resumed['best_epoch'])
                self.assertEqual(resumed['optimizer_updates'], 8)
                self.assertEqual(resumed['new_optimizer_updates'], 4)
                self.assertEqual(resumed['best150_validation'], first['best_validation'])

    def test_next_update_and_rng_draws_identical_after_restore(self):
        model, cache, _ = fixture()
        c = config()
        opt = run.parent.optimizer(model, c)
        sampler = run.raw.TrainBatches(9, 4, 42)
        for numbers in sampler:
            model.train()
            run.parent.step(model, cache, numbers, self.device, opt)
        with tempfile.TemporaryDirectory(dir=self.temporary) as tmp:
            path = Path(tmp)/'last.pt'
            run.raw.save_checkpoint(path, model, opt, 1, {}, {}, sampler)
            saved = torch.load(path, map_location='cpu', weights_only=False)
        next_batch = next(iter(sampler))
        expected_draws = (random.random(), np.random.rand(), torch.rand(4))
        loss = run.parent.step(model, cache, next_batch, self.device, opt)
        expected = copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict())
        fresh, _, _ = fixture()
        restored_opt = run.parent.optimizer(fresh, c)
        restored_sampler = run.raw.TrainBatches(9, 4, 999)
        run.restore_training(fresh, restored_opt, restored_sampler, saved, c, 2)
        self.assertEqual(next(iter(restored_sampler)), next_batch)
        draws = (random.random(), np.random.rand(), torch.rand(4))
        self.assertEqual(draws[:2], expected_draws[:2])
        self.assertTrue(torch.equal(draws[2], expected_draws[2]))
        self.assertEqual(run.parent.step(fresh, cache, next_batch, self.device, restored_opt), loss)
        run.assert_tree_equal(fresh.state_dict(), expected[0])
        run.assert_tree_equal(restored_opt.state_dict(), expected[1])
        with self.assertRaisesRegex(ValueError, 'counter'):
            run.restore_training(fresh, restored_opt, restored_sampler, saved, c, 99)

    def test_raw_probe_checks_updated_output_gate_attention_and_restores_state(self):
        model, cache, batch = fixture()
        config_value = config()
        digest, rng = run.raw.state_digest(model.state_dict()), run.raw.rng_hash(run.raw.rng_state())
        receipt = run.raw_update_probe(model, None, cache, list(range(9)), batch, config_value, self.device)
        self.assertTrue(receipt['after']['predictions_identical'])
        self.assertEqual(set(receipt['after']['max_abs_errors']), {'scores', 'gate', 'packet_attention'})
        self.assertEqual(run.raw.state_digest(model.state_dict()), digest)
        self.assertEqual(run.raw.rng_hash(run.raw.rng_state()), rng)
        with patch.object(run, 'compare_raw_cache', side_effect=ValueError('probe failure')):
            with self.assertRaises(ValueError):
                run.raw_update_probe(model, None, cache, list(range(9)), batch, config_value, self.device)
        self.assertEqual(run.raw.rng_hash(run.raw.rng_state()), rng)
        corrupted = copy.deepcopy(cache)
        corrupted['packet64'] += 100
        with self.assertRaisesRegex(ValueError, 'Raw/cache'):
            run.compare_raw_cache(model, batch, corrupted, list(range(9)), self.device)

    def test_seed_pair_initial_weights_match_but_new_seeds_change_keys(self):
        keys = []
        for seed in (42, 43, 44):
            a, _, _ = fixture('packet_mean', seed)
            b, _, _ = fixture('flow', seed)
            run.assert_tree_equal(a.state_dict(), b.state_dict())
            self.assertEqual(sum(p.numel() for p in a.parameters() if p.requires_grad), 8168)
            keys.append(a.packet_pool.key.weight.detach())
        self.assertFalse(torch.equal(keys[0], keys[1]))
        self.assertFalse(torch.equal(keys[1], keys[2]))

    def test_prefix_selection_retains_earliest_tie(self):
        history = [{'epoch': i+1, 'validation': {'macro_f1': score}} for i, score in enumerate((.5, .7, .7, .8))]
        self.assertEqual(run.prefix_best({'macro_f1': .6}, history, 3), ({'macro_f1': .7}, 2))
        self.assertEqual(run.prefix_best({'macro_f1': .7}, history, 3), ({'macro_f1': .7}, 0))

    def test_joint_gat_probe_preserves_every_frozen_parameter_and_bn(self):
        for mode in ('packet_mean', 'flow'):
            with self.subTest(mode=mode):
                model, cache, batch, _ = joint_fixture(mode)
                c = {**config(), 'encoder_lr': 1e-6}
                before, rng = run.frozen_hash(model), run.raw.rng_hash(run.raw.rng_state())
                receipt = run.joint_probe(model, cache, list(range(8)), batch, c, self.device)
                self.assertTrue(receipt['frozen_state_unchanged'])
                self.assertEqual(run.raw.rng_hash(run.raw.rng_state()), rng)
                opt = run.joint_optimizer(model, c)
                self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 25384)
                # The raw and frozen paths must agree before learning, including
                # the first graph's empty Packet->Packet relation.
                frozen = QueryPacketClassifier(copy.deepcopy(model.flow_encoder), copy.deepcopy(model.packet_encoder),
                                               copy.deepcopy(model.fusion), mode).eval()
                torch.testing.assert_close(model(batch), frozen(batch), atol=0, rtol=0)
                original = {n: p.detach().clone() for n, p in model.named_parameters()}
                for _ in range(3):
                    model.train()
                    run.joint_step(model, batch, opt)
                changed = {n for n, p in model.named_parameters() if not torch.equal(p, original[n])}
                expected = {n for n, p in model.named_parameters() if p.requires_grad}
                self.assertTrue(changed <= expected)
                self.assertTrue(any(n.startswith('packet_encoder.convs2.') for n in changed))
                self.assertEqual(run.frozen_hash(model), before)
                self.assertTrue(all(not module.training for encoder in (model.flow_encoder, model.packet_encoder)
                                    for module in encoder.modules()))
                self.assertTrue(all(p.grad is None for p in model.flow_encoder.parameters()))
                self.assertTrue(all(p.grad is None for p in model.packet_encoder.convs1.parameters()))

    def test_joint_raw_training_uses_paired_sampler_and_reloads_best(self):
        model, _, _, graphs = joint_fixture()
        c = {**config(epochs=3), 'encoder_lr': 1e-6, 'workers': 0}
        rows = {part: [{'sample_id': graph.sample_id, 'label': int(graph.y), 'partition': part} for graph in graphs]
                for part in ('train', 'val')}
        sampler, orders = run.raw.TrainBatches(8, 4, 42), []
        for _ in range(3):
            list(sampler)
            orders.append(sampler.order_sha256)
        with tempfile.TemporaryDirectory(dir=self.temporary) as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            result, exports = run.fit_joint('Joint-Flow', model, TinyGraphs(graphs), rows, self.device, c, {}, root, orders)
            self.assertEqual(result['orders'], orders)
            self.assertEqual(result['optimizer_updates'], 6)
            self.assertTrue(result['raw_replay_passed'])
            self.assertFalse(result['historical_node_cache_used_for_training'])
            self.assertEqual(len(exports), 8)
            self.assertTrue(all('flow_weights' in r and 'packet_attention' in r for r in exports))
            saved = torch.load(root/'Joint-Flow'/'last.pt', map_location='cpu', weights_only=False)
            self.assertTrue(all(int(state['step']) == 6 for state in saved['optimizer_state']['state'].values()))

    def test_four_arm_analysis_separates_query_and_unfreeze_effects(self):
        labels = list(range(8))
        exports, results = {}, {}
        for name, wrong in (('Frozen-Mean', (0, 1, 2)), ('Frozen-Flow', (0, 1)),
                            ('Joint-Mean', (0,)), ('Joint-Flow', ())):
            rows = [{'sample_id': str(y), 'label': y, 'prediction': (y+1) % 8 if y in wrong else y} for y in labels]
            cm = [[0]*8 for _ in labels]
            for row in rows:
                cm[row['label']][row['prediction']] += 1
            metric = run.parent.matrix_report(cm)
            results[name], exports[name] = {'best_validation': metric, 'best150_validation': metric}, rows
        c = {'focus_classes': ['WebBased', 'Spoofing', 'Recon'], 'JCL_Multi_reference_macro_f1': .9752460910749855}
        with tempfile.TemporaryDirectory(dir=self.temporary) as tmp:
            analysis = run.analyze_four(results, exports, c, Path(tmp))
            self.assertAlmostEqual(analysis['query_advantage_change_with_unfreezing_pp'],
                                   analysis['Joint_Flow_minus_Joint_Mean_pp']-analysis['Frozen_Flow_minus_Frozen_Mean_pp'])
            self.assertEqual(analysis['comparisons']['Joint-Flow_vs_Joint-Mean']['comparison']['net_correct'], 1)
            self.assertEqual(analysis['additional_seeds_deferred'], [43, 44])


if __name__ == '__main__':
    unittest.main()
