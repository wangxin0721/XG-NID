"""CPU tensor checks before server launch; missing torch/PyG is a failure."""
import copy
import hashlib
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch.nn import functional as F
from torch.nn.parameter import UninitializedParameter
from torch_geometric.data import HeteroData,Batch
from torch_geometric.nn import GATConv,global_mean_pool
import run_relation_pair as run
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from xgnid.model import _PaperStyleHeteroEncoder,FLOW_TO_PACKET,PACKET_TO_FLOW,PACKET_TO_PACKET
from xgnid.fusion_dimension import FusionHead,FrozenEncoderClassifier
from xgnid.relation_selection import RelationClassifier


def state_hash(values):
    h=hashlib.sha256()
    for n,t in sorted(values):h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class TensorChecks(unittest.TestCase):
    def setUp(self):torch.manual_seed(7);torch.set_num_threads(2)

    def graphs(self,empty=False):
        graphs=[]
        for i,n in enumerate((1,3,2)):
            g=HeteroData();g['flow'].x=torch.randn(1,3);g['packet'].x=torch.randn(n,5)
            ei=torch.stack((torch.zeros(n,dtype=torch.long),torch.arange(n)))
            g[FLOW_TO_PACKET].edge_index=ei;g[FLOW_TO_PACKET].edge_attr=torch.randn(n,4)
            g[PACKET_TO_FLOW].edge_index=ei.flip(0);g[PACKET_TO_FLOW].edge_attr=g[FLOW_TO_PACKET].edge_attr.clone()
            pp=torch.empty(2,0,dtype=torch.long) if empty or n==1 else torch.stack((torch.arange(n-1),torch.arange(1,n)))
            g[PACKET_TO_PACKET].edge_index=pp;g[PACKET_TO_PACKET].edge_attr=torch.randn(pp.size(1),1)
            g.sample_id=str(i);g.y=torch.tensor([i]);graphs.append(g)
        return graphs

    def models(self,graphs):
        batch=Batch.from_data_list(graphs)
        # PyG skips lazy edge projections for an empty relation. Materialize
        # every projection on a separate nonempty fixture before freezing;
        # the batch returned to the empty-relation test remains untouched.
        warm=Batch.from_data_list(self.graphs()) if batch[PACKET_TO_PACKET].edge_index.numel()==0 else batch
        def encoder():
            e=_PaperStyleHeteroEncoder(conv_cls=GATConv,edge_aware=True,flow_input_dim=3,packet_input_dim=5).eval()
            with torch.no_grad():e(warm)
            self.assertFalse(any(isinstance(p,UninitializedParameter) for p in e.parameters()))
            return e
        mean=RelationClassifier(encoder(),encoder(),FusionHead(32,run.dim.initialization_streams(42)),'mean').eval()
        gate=RelationClassifier(copy.deepcopy(mean.flow_encoder),copy.deepcopy(mean.packet_encoder),copy.deepcopy(mean.fusion),'relation_gate').eval()
        return mean,gate,batch

    def test_capture_matches_original_and_only_adds3122_parameters(self):
        mean,gate,batch=self.models(self.graphs())
        f,legacy=mean.encode(batch);nodes,gamma=mean.packet_nodes(*f[1:4])
        torch.testing.assert_close(nodes,legacy,atol=1e-6,rtol=1e-6)
        original=FrozenEncoderClassifier(mean.flow_encoder,mean.packet_encoder,mean.fusion).eval()
        torch.testing.assert_close(mean(batch),original(batch),atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(mean(batch),gate(batch),atol=1e-6,rtol=1e-6)
        self.assertTrue(bool((gamma==.5).all()))
        self.assertEqual(sum(p.numel() for p in gate.relation_gate.parameters()),3122)
        self.assertEqual(sum(p.numel() for p in gate.parameters() if p.requires_grad),10266)

    def test_empty_packet_relation_keeps_original_bias_and_no_new_mask(self):
        mean,gate,batch=self.models(self.graphs(empty=True))
        self.assertEqual(batch[PACKET_TO_PACKET].edge_index.numel(),0)
        self.assertEqual(batch[PACKET_TO_PACKET].edge_attr.shape,(0,1))
        for m in (mean,gate):
            with torch.no_grad():m.packet_encoder.convs2.convs[PACKET_TO_PACKET].bias.fill_(.37)
        f,legacy=gate.encode(batch)
        torch.testing.assert_close(f[3],torch.full_like(f[3],.37))
        nodes,gamma=gate.packet_nodes(*f[1:4]);torch.testing.assert_close(nodes,legacy)
        self.assertTrue(bool((gamma==.5).all()))
        with torch.no_grad():gate.relation_gate[-1].bias.copy_(torch.tensor([-2.,2.]))
        _,gamma=gate.packet_nodes(*f[1:4]);self.assertTrue(bool((gamma[:,1]>.9).all()))

    def test_hooks_removed_on_success_and_exception(self):
        mean,gate,batch=self.models(self.graphs())
        def count():return sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in gate.packet_encoder.modules())
        before=count();gate.encode(batch);self.assertEqual(count(),before)
        with patch.object(gate.packet_encoder,'forward',side_effect=RuntimeError('injected')):
            with self.assertRaises(RuntimeError):gate.encode(batch)
        self.assertEqual(count(),before)

    def test_bn_transmits_gradient_while_old_parameters_and_statistics_stay_frozen(self):
        mean,gate,batch=self.models(self.graphs());old=state_hash(run.dim.encoder_values(gate))
        gate.train();f,_=gate.encode(batch);before=gate.relation_gate[0].weight.detach().clone()
        opt=run.optimizer(SimpleNamespace(torch=torch),gate,run.read(Path(__file__).resolve().parents[1]/'configs/relation_pair_s42_v1.json'))
        for step in range(3):
            opt.zero_grad(set_to_none=True);scores,_=gate.forward_features(*f);F.nll_loss(scores,batch.y).backward()
            if step==0:
                self.assertEqual(float(gate.relation_gate[0].weight.grad.abs().sum()),0.)
                self.assertGreater(float(gate.relation_gate[-1].weight.grad.abs().sum()),0.)
            opt.step()
        self.assertFalse(torch.equal(before,gate.relation_gate[0].weight))
        run.dim.assert_frozen(SimpleNamespace(state_hash=state_hash),gate,old)

    def test_nonuniform_result_matches_direct_weighted_relation_sum(self):
        _,gate,batch=self.models(self.graphs());f,_=gate.encode(batch)
        with torch.no_grad():gate.relation_gate[-1].weight.normal_(0,.05)
        nodes,gamma=gate.packet_nodes(*f[1:4]);mixed=gamma[:,:1]*f[2]+gamma[:,1:]*f[3]
        expected=gate.packet_encoder.relus2['packet'](gate.packet_encoder.bns2['packet'](mixed))
        torch.testing.assert_close(nodes,expected,atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(gamma.sum(-1),torch.ones(len(gamma)))

    def test_ragged_cached_messages_reorder_like_raw_graphs_and_strict_reload(self):
        graphs=self.graphs();_,gate,batch=self.models(graphs);f,legacy=gate.encode(batch)
        cache={'flow64':f[0],'first64':f[1],'flow_message64':f[2],'packet_message64':f[3],
               'ptr':torch.tensor([0,1,4,6]),'labels':batch.y}
        features,y=run.minibatch(SimpleNamespace(torch=torch),cache,[2,0,1],torch.device('cpu'))
        torch.testing.assert_close(gate.forward_features(*features)[0],gate(Batch.from_data_list([graphs[i] for i in (2,0,1)])),atol=1e-5,rtol=1e-5)
        self.assertEqual(y.tolist(),[2,0,1])
        # Reconstruct and explicitly enter eval, as the production replay
        # does. Deep-copying PyG modules need not preserve all mode flags.
        _,fresh,_=self.models(graphs)
        fresh.load_state_dict(gate.state_dict(),strict=True);fresh.eval()
        self.assertTrue(all(not m.training for m in fresh.modules()))
        torch.testing.assert_close(fresh(batch),gate(batch))

    def test_probe_restores_clean_weights_rng_and_gradients_even_after_failure(self):
        graphs=self.graphs();_,gate,batch=self.models(graphs);f,legacy=gate.encode(batch)
        rows=[{'sample_id':str(i),'label':i} for i in range(3)]
        cache={'ids':[str(i) for i in range(3)],'flow64':f[0],'first64':f[1],'flow_message64':f[2],
            'packet_message64':f[3],'ptr':torch.tensor([0,1,4,6]),'labels':batch.y,
            'graph_diagnostics':[{'max_difference':1.} for _ in range(3)]}
        core=SimpleNamespace(torch=torch,F=F,Batch=Batch,state_hash=state_hash,select_rows=lambda *args:rows)
        data=SimpleNamespace(load=lambda row:graphs[int(row['sample_id'])]);c=run.read(Path(__file__).resolve().parents[1]/'configs/relation_pair_s42_v1.json')
        clean=state_hash(gate.state_dict().items());rng=run.pair.rng_state(core);enc=state_hash(run.dim.encoder_values(gate))
        receipt=run.probe(core,gate,data,cache,torch.device('cpu'),c,enc)
        self.assertTrue(receipt['clean_model_and_rng_restored']);self.assertEqual(state_hash(gate.state_dict().items()),clean)
        self.assertEqual(run.pair.rng_hash(core,rng),run.pair.rng_hash(core,run.pair.rng_state(core)))
        opt=run.optimizer(core,gate,c);step=opt.step
        def fail():step();raise RuntimeError('injected update failure')
        opt.step=fail
        with patch.object(run,'optimizer',return_value=opt),self.assertRaisesRegex(RuntimeError,'injected'):
            run.probe(core,gate,data,cache,torch.device('cpu'),c,enc)
        self.assertEqual(state_hash(gate.state_dict().items()),clean)
        self.assertTrue(all(p.grad is None for p in gate.parameters()))


if __name__=='__main__':unittest.main()
