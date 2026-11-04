"""Real torch/PyG checks for the next operator experiment, not score claims."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from torch_geometric.data import Batch
from torch_geometric.nn import GATv2Conv
from xgnid.attention_operator_pair import (OperatorClassifier, MatchedGATv2Conv,
    assert_paired, RELATIONS)
from probe_attention_operator_pair import synthetic_graphs, probe_one, state_digest


class OperatorChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(42)

    def models(self):
        return {n:OperatorClassifier(n,3,5) for n in ('GAT','GATv2')}

    def test_common_initialization_and_caller_rng_are_matched(self):
        before=torch.get_rng_state().clone();models=self.models()
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertTrue(assert_paired(models)['all_common_tensors_identical'])
        counts={n:sum(p.numel() for p in m.parameters()) for n,m in models.items()}
        self.assertEqual(counts['GAT']-counts['GATv2'],12*128)
        for m in models.values():
            self.assertTrue(all(p.requires_grad for p in m.parameters()))
            self.assertTrue(all(int(v)==0 for n,v in m.named_buffers() if n.endswith('num_batches_tracked')))

    def test_gat_control_preserves_original_encoder_forward(self):
        from xgnid.model import _PaperStyleHeteroEncoder
        from torch_geometric.nn import GATConv,global_mean_pool
        model=self.models()['GAT'];batch=Batch.from_data_list(synthetic_graphs())
        old=[]
        for encoder in (model.flow_encoder,model.packet_encoder):
            e=_PaperStyleHeteroEncoder(conv_cls=GATConv,edge_aware=True,flow_input_dim=3,packet_input_dim=5)
            e.load_state_dict(encoder.state_dict(),strict=True);old.append(e.eval())
        with torch.no_grad():
            expected,_=model.fusion(global_mean_pool(old[0](batch)['flow'],batch['flow'].batch),
                                    global_mean_pool(old[1](batch)['packet'],batch['packet'].batch))
            torch.testing.assert_close(model(batch),expected,atol=1e-6,rtol=1e-6)

    def test_v2_matches_stock_operator_with_zero_internal_biases(self):
        ours=MatchedGATv2Conv((3,5),64,edge_dim=4,add_self_loops=False)
        stock=GATv2Conv((3,5),64,edge_dim=4,add_self_loops=False)
        state=ours.state_dict();state.update({'lin_l.bias':torch.zeros(64),'lin_r.bias':torch.zeros(64)})
        stock.load_state_dict(state,strict=True)
        x=(torch.randn(2,3),torch.randn(3,5));ei=torch.tensor([[0,1,0,1,0,1],[0,0,1,1,2,2]]);ea=torch.randn(6,4)
        torch.testing.assert_close(ours(x,ei,ea),stock(x,ei,ea))
        self.assertIsNone(ours.lin_l.bias);self.assertIsNone(ours.lin_r.bias)

    def test_all_relation_edge_features_affect_both_operators(self):
        g=synthetic_graphs()[0]
        for name,model in self.models().items():
            for relation in RELATIONS:
                conv=model.packet_encoder.convs1.convs[relation]
                x=(g[relation[0]].x,g[relation[2]].x)
                edge=g[relation].edge_attr.detach().clone().requires_grad_()
                out=conv(x,g[relation].edge_index,edge)
                out.square().sum().backward()
                self.assertGreater(float(edge.grad.abs().sum()),0,(name,relation))
                self.assertFalse(torch.allclose(out,conv(x,g[relation].edge_index,edge.detach()*-20)))

    def test_empty_relation_single_packet_and_strict_reload(self):
        graphs=synthetic_graphs()
        for g in graphs:
            g['flow'].x=g['flow'].x[:1];g['packet'].x=g['packet'].x[:1]
            for relation in RELATIONS[:2]:
                g[relation].edge_index=torch.tensor([[0],[0]])
                g[relation].edge_attr=g[relation].edge_attr[:1]
            rel=RELATIONS[2];g[rel].edge_index=torch.empty(2,0,dtype=torch.long);g[rel].edge_attr=torch.empty(0,1)
        batch=Batch.from_data_list(graphs)
        for model in self.models().values():
            out=model(batch);self.assertEqual(out.shape,(8,8));self.assertTrue(bool(torch.isfinite(out).all()))
            other=OperatorClassifier(model.operator,3,5)
            other.load_state_dict(model.state_dict(),strict=True)
            torch.testing.assert_close(out,other(batch))

    def test_trainable_encoders_bn_and_probe_restore(self):
        c=json.loads((Path(__file__).resolve().parents[1]/'configs/attention_operator_pair_s42_v1.json').read_text())
        batch=Batch.from_data_list(synthetic_graphs())
        for model in self.models().values():
            clean=state_digest(model.state_dict());rng=torch.get_rng_state().clone()
            report=probe_one(model,batch,batch,torch.device('cpu'),c)
            self.assertTrue(report['clean_state_restored']);self.assertTrue(report['optimizer_discarded'])
            self.assertEqual(clean,state_digest(model.state_dict()));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual(len(report['updates']),3)

    def test_probe_failure_restores_parameters_bn_gradients_and_rng(self):
        c=json.loads((Path(__file__).resolve().parents[1]/'configs/attention_operator_pair_s42_v1.json').read_text())
        model=self.models()['GATv2'];batch=Batch.from_data_list(synthetic_graphs())
        clean=state_digest(model.state_dict());rng=torch.get_rng_state().clone()
        step=torch.optim.Adam.step
        def fail(opt,*args,**kwargs):
            step(opt,*args,**kwargs)
            raise RuntimeError('injected update failure')
        with patch.object(torch.optim.Adam,'step',fail),self.assertRaisesRegex(RuntimeError,'injected'):
            probe_one(model,batch,batch,torch.device('cpu'),c)
        self.assertEqual(clean,state_digest(model.state_dict()))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__=='__main__':
    unittest.main()
