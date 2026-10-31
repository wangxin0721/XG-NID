"""Small CPU tensor checks; torch/PyG are required, never silently skipped."""
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import global_mean_pool

import run_packet_pool_pair as run

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from xgnid.fusion_dimension import FusionHead, FrozenEncoderClassifier
from xgnid.packet_pooling import PacketAttentionPool, PacketReadoutClassifier


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(64)
        self.scale = nn.Parameter(torch.ones(64))
    def forward(self, data):
        return {name: self.bn(data[name].x) * self.scale for name in ('flow', 'packet')}


def state_hash(values):
    h = hashlib.sha256()
    for n, t in sorted(values):
        h.update(n.encode())
        h.update(t.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


class Tensors(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(2)

    def graph(self, mode='attention'):
        return PacketReadoutClassifier(Encoder(), Encoder(), FusionHead(32, run.dim.initialization_streams(42)), mode)

    def data(self):
        graphs = []
        for i, count in enumerate((1, 3, 2)):
            g = HeteroData()
            g['flow'].x, g['packet'].x = torch.randn(1, 64), torch.randn(count, 64)
            g.y, g.sample_id = torch.tensor([i]), str(i)
            graphs.append(g)
        return graphs

    def test_uniform_mean_and_rng_preservation(self):
        cpu = torch.get_rng_state().clone()
        cuda = [s.clone() for s in torch.cuda.get_rng_state_all()]
        pool = PacketAttentionPool()
        self.assertTrue(torch.equal(cpu, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(cuda, torch.cuda.get_rng_state_all())))
        self.assertEqual(sum(p.numel() for p in pool.parameters()), 1056)
        nodes, index = torch.randn(6, 64), torch.tensor([0, 1, 1, 1, 2, 2])
        torch.testing.assert_close(pool(nodes, index, 3), global_mean_pool(nodes, index, size=3))
        beta = pool.weights(nodes, index, 3)
        for i, n in enumerate((1, 3, 2)):
            torch.testing.assert_close(beta[index == i], torch.full((n,), 1 / n))

    def test_nonuniform_attention_is_local_and_matches_manual_weighted_sum(self):
        pool = PacketAttentionPool()
        with torch.no_grad():
            pool.aggregation.gate_nn[-1].weight.fill_(.2)
        nodes, index = torch.randn(5, 64), torch.tensor([0, 0, 1, 1, 1])
        out, beta = pool(nodes, index, 2), pool.weights(nodes, index, 2)
        for i in range(2):
            x = nodes[index == i]
            weights = pool.aggregation.gate_nn(x).view(-1).softmax(0)
            torch.testing.assert_close(out[i], (x * weights[:, None]).sum(0))
            torch.testing.assert_close(beta[index == i].sum(), torch.tensor(1.))
        other = nodes.clone()
        other[index == 1] *= 100
        torch.testing.assert_close(pool(other, index, 2)[0], out[0])
        torch.testing.assert_close(pool(nodes[:1], torch.tensor([0]), 1), nodes[:1])

    def test_uniform_start_matches_mean_with_cancellation_and_keeps_gradient(self):
        pool = PacketAttentionPool()
        nodes = torch.ones(3, 64)
        nodes[:, 0] = torch.tensor([1e6, 1., -1e6])
        index = torch.zeros(3, dtype=torch.long)
        mean = global_mean_pool(nodes, index, size=1)
        legacy = pool.aggregation(nodes, index=index, dim_size=1)
        result = pool(nodes, index, 1)
        self.assertTrue(torch.equal(result, mean))
        self.assertGreater(float((legacy - mean).abs().max()), 1e-6)
        result[:, 0].sum().backward()
        self.assertGreater(float(pool.aggregation.gate_nn[-1].weight.grad.abs().sum()), 0.)
        self.assertEqual(float(pool.aggregation.gate_nn[0].weight.grad.abs().sum()), 0.)

    def test_zero_output_initialization_delays_hidden_gradient_then_learns(self):
        pool = PacketAttentionPool()
        nodes, index = torch.randn(6, 64), torch.tensor([0, 0, 0, 1, 1, 1])
        opt = torch.optim.Adam(pool.parameters(), lr=1e-4)
        hidden_before = pool.aggregation.gate_nn[0].weight.detach().clone()
        for step in range(3):
            opt.zero_grad(set_to_none=True)
            pool(nodes, index, 2).square().sum().backward()
            if step == 0:
                self.assertEqual(float(pool.aggregation.gate_nn[0].weight.grad.abs().sum()), 0.)
                self.assertGreater(float(pool.aggregation.gate_nn[-1].weight.grad.abs().sum()), 0.)
            opt.step()
        self.assertFalse(torch.equal(hidden_before, pool.aggregation.gate_nn[0].weight))
        self.assertGreater(float((pool.weights(nodes, index, 2) - 1 / 3).abs().max()), 0.)

    def test_raw_model_freezes_bn_and_encoders_but_pool_receives_gradients(self):
        model = self.graph()
        data = Batch.from_data_list(self.data())
        model.eval()
        original = FrozenEncoderClassifier(model.flow_encoder, model.packet_encoder, model.fusion).eval()
        torch.testing.assert_close(model(data), original(data), atol=1e-6, rtol=1e-6)
        encoder_hash = state_hash(run.dim.encoder_values(model))
        model.train()
        self.assertTrue(model.packet_pool.training)
        self.assertTrue(all(not m.training for e in (model.flow_encoder, model.packet_encoder) for m in e.modules()))
        F.nll_loss(model(data), data.y).backward()
        self.assertGreater(float(model.packet_pool.aggregation.gate_nn[-1].weight.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for e in (model.flow_encoder, model.packet_encoder) for p in e.parameters()))
        self.assertEqual(encoder_hash, state_hash(run.dim.encoder_values(model)))

    def cache(self, model, graphs):
        data = Batch.from_data_list(graphs)
        model.eval()
        flow, nodes, _, _ = model.encode(data)
        return {'ids': [g.sample_id for g in graphs], 'flow64': flow, 'packet64': nodes,
                'ptr': torch.tensor([0, 1, 4, 6]), 'labels': data.y,
                'packet_rms_distance': [0., 1., 1.]}

    def test_ragged_cache_preserves_reordered_graphs_and_strict_checkpoint(self):
        model, graphs = self.graph(), self.data()
        cache = self.cache(model, graphs)
        core = SimpleNamespace(torch=torch)
        features, labels = run.minibatch(core, cache, [2, 0, 1], torch.device('cpu'))
        raw = Batch.from_data_list([graphs[i] for i in (2, 0, 1)])
        torch.testing.assert_close(model.forward_features(*features)[0], model(raw))
        self.assertEqual(labels.tolist(), [2, 0, 1])
        self.assertEqual(features[2].tolist(), [0, 0, 1, 2, 2, 2])
        fresh = self.graph()
        fresh.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(fresh(raw), model(raw))
        self.assertEqual([g['lr'] for g in run.optimizer(core, model, run.read(ROOT / 'configs/packet_pool_pair_s42_v1.json')).param_groups],
                         [1e-5, 1e-4])

    def test_probe_restores_full_state_gradients_and_rng(self):
        model, graphs = self.graph(), self.data()
        cache = self.cache(model, graphs)
        rows = [{'sample_id': str(i), 'label': i} for i in range(3)]
        core = SimpleNamespace(torch=torch, F=F, Batch=Batch, state_hash=state_hash,
                               select_rows=lambda data, part: rows)
        data = SimpleNamespace(load=lambda row: graphs[int(row['sample_id'])])
        clean, rng = state_hash(model.state_dict().items()), run.pair.rng_state(core)
        receipt = run.probe(core, None, model, data, cache, torch.device('cpu'),
                            run.read(ROOT / 'configs/packet_pool_pair_s42_v1.json'),
                            state_hash(run.dim.encoder_values(model)))
        self.assertTrue(receipt['clean_model_and_rng_restored'])
        self.assertEqual(len(receipt['updates']), 3)
        self.assertEqual(state_hash(model.state_dict().items()), clean)
        self.assertEqual(run.pair.rng_hash(core, run.pair.rng_state(core)), run.pair.rng_hash(core, rng))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_failed_probe_also_restores_state(self):
        model, graphs = self.graph(), self.data()
        cache = self.cache(model, graphs)
        rows = [{'sample_id': str(i), 'label': i} for i in range(3)]
        core = SimpleNamespace(torch=torch, F=F, Batch=Batch, state_hash=state_hash,
                               select_rows=lambda data, part: rows)
        data = SimpleNamespace(load=lambda row: graphs[int(row['sample_id'])])
        clean, rng = state_hash(model.state_dict().items()), run.pair.rng_state(core)
        opt = run.optimizer(core, model, run.read(ROOT / 'configs/packet_pool_pair_s42_v1.json'))
        original_step = opt.step
        def fail():
            original_step()
            raise RuntimeError('injected failure after updating parameters')
        opt.step = fail
        with patch.object(run, 'optimizer', return_value=opt), self.assertRaisesRegex(RuntimeError, 'injected failure'):
            run.probe(core, None, model, data, cache, torch.device('cpu'),
                      run.read(ROOT / 'configs/packet_pool_pair_s42_v1.json'), state_hash(run.dim.encoder_values(model)))
        self.assertEqual(state_hash(model.state_dict().items()), clean)
        self.assertEqual(run.pair.rng_hash(core, run.pair.rng_state(core)), run.pair.rng_hash(core, rng))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_node_cache_validation_rejects_bad_boundaries_and_identity(self):
        model, graphs = self.graph(), self.data()
        cache = self.cache(model, graphs)
        rows = [{'sample_id': str(i), 'label': i} for i in range(3)]
        source = {'split_id': 'fixed'}
        cache['metadata'] = {'schema': 'xgnid.packet-node64-cache.v1', 'partition': 'train',
            'encoder_sha256': 'encoder', 'split_binding': source,
            'sample_order_sha256': run.identity(cache['ids']), 'samples': 3}
        core = SimpleNamespace(torch=torch)
        run.validate_cache(core, cache, rows, 'train', 'encoder', source)
        for ptr in (torch.tensor([0, 1, 4, 5]), torch.tensor([0, 4, 1, 6])):
            with self.assertRaises(ValueError):
                run.validate_cache(core, {**cache, 'ptr': ptr}, rows, 'train', 'encoder', source)
        with self.assertRaises(ValueError):
            run.validate_cache(core, {**cache, 'ids': ['2', '1', '0']}, rows, 'train', 'encoder', source)


if __name__ == '__main__':
    unittest.main()
