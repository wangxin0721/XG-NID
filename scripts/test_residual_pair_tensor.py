"""Real PyG residual placement, common readout, gradients and folding checks."""
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
from torch_geometric.data import Batch
import run_residual_pair as run
import test_relation_pair_tensor as fixtures
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from xgnid.residual_readout import ResidualClassifier
from xgnid.cross_layer import CrossLayerClassifier
from xgnid.fold_readout import fold_last


class Tensors(unittest.TestCase):
    def setUp(self):torch.manual_seed(7);torch.set_num_threads(2)

    def models(self,empty=False):
        fixture=fixtures.TensorChecks();graphs=fixture.graphs(empty);original,_,batch=fixture.models(graphs)
        prepared=[(copy.deepcopy(original.flow_encoder),copy.deepcopy(original.packet_encoder),copy.deepcopy(original.fusion)) for _ in range(3)]
        models=[]
        for mode,inputs in zip(('base','skip'),prepared):
            rng=torch.get_rng_state().clone();m=ResidualClassifier(*inputs,mode).eval()
            self.assertTrue(torch.equal(rng,torch.get_rng_state()));models.append(m)
        last=CrossLayerClassifier(*prepared[2],'last').eval()
        return models,graphs,batch,original,last

    def core(self):
        return SimpleNamespace(torch=torch,F=F,Batch=Batch,state_hash=fixtures.state_hash,
            select_rows=lambda *args:[{'sample_id':str(i),'label':i} for i in range(3)])

    def cache(self,m,batch):
        f,legacy=m.encode(batch)
        return {'ids':[str(i) for i in range(3)],'flow64':f[0],'first64':f[1],
            'flow_message64':f[2],'packet_message64':f[3],'ptr':torch.tensor([0,1,4,6]),'labels':batch.y}

    def test_initial_identity_common_last_readout_capacity_and_empty_relations(self):
        for empty in (False,True):
            models,_,batch,original,last=self.models(empty)
            for m in models:
                torch.testing.assert_close(m(batch),original(batch),atol=1e-6,rtol=1e-6)
                torch.testing.assert_close(m(batch),last(batch),atol=1e-6,rtol=1e-6)
                self.assertTrue(bool((m.last_residual_norm==0).all()))
                self.assertEqual(sum(p.numel() for p in m.cross_layer.parameters()),8256)
                self.assertEqual(sum(p.numel() for p in m.parameters() if p.requires_grad),run.architecture(m.pooling)['trainable_parameters'])
            self.assertIsNone(models[1].residual.bias)

    def test_residual_is_before_bn_activation_and_keeps_empty_convolution_outputs(self):
        models,_,batch,_,_=self.models(empty=True);m=models[1];f,legacy=m.encode(batch)
        with torch.no_grad():
            m.residual.weight.copy_(torch.eye(64)*.2)
            m.packet_encoder.bns2['packet'].running_var.fill_(4.)
            m.packet_encoder.bns2['packet'].running_mean.fill_(.3)
        mean=torch.stack((f[2],f[3])).mean(0);delta=m.residual(f[1]);bn=m.packet_encoder.bns2['packet'];act=m.packet_encoder.relus2['packet']
        nodes,_=m.packet_nodes(*f[1:4]);torch.testing.assert_close(nodes,act(bn(mean+delta)))
        self.assertFalse(torch.allclose(nodes,act(bn(mean))+delta))
        zero,_=m.packet_nodes(*f[1:4],force_mean=True);torch.testing.assert_close(zero,act(bn(mean)))

    def test_raw_probe_restores_clean_state_on_success_and_update_failure(self):
        models,graphs,batch,_,_=self.models();core=self.core();c=run.read(run.HERE.parent/'configs/residual_pair_s42_v1.json')
        data=SimpleNamespace(load=lambda row:graphs[int(row['sample_id'])])
        for m in models:
            cache=self.cache(m,batch);clean=fixtures.state_hash(m.state_dict().items());rng=run.pair.rng_state(core)
            enc=fixtures.state_hash(run.dim.encoder_values(m));r=run.probe(core,m,data,cache,torch.device('cpu'),c,enc)
            self.assertTrue(r['clean_model_and_rng_restored']);self.assertEqual(clean,fixtures.state_hash(m.state_dict().items()))
            self.assertEqual(run.pair.rng_hash(core,rng),run.pair.rng_hash(core,run.pair.rng_state(core)))
            f,y=run.minibatch(core,cache,[0,1,2],torch.device('cpu'));out,_=m.forward_features(*f);F.nll_loss(out,y).backward()
            torch.testing.assert_close(m.cross_layer.weight.grad[:,:64],m.cross_layer.weight.grad[:,64:])
            self.assertGreater(float(m.cross_layer.weight.grad[:,:64].abs().sum()),0)
            if m.residual is not None:self.assertGreater(float(m.residual.weight.grad.abs().sum()),0)
            m.zero_grad(set_to_none=True)
            opt=run.optimizer(core,m,c);step=opt.step
            def fail():step();raise RuntimeError('injected')
            opt.step=fail
            with patch.object(run,'optimizer',return_value=opt),self.assertRaisesRegex(RuntimeError,'injected'):
                run.probe(core,m,data,cache,torch.device('cpu'),c,enc)
            self.assertEqual(clean,fixtures.state_hash(m.state_dict().items()));self.assertTrue(all(p.grad is None for p in m.parameters()))

    def test_ragged_cache_reorder_and_strict_reload_after_residual_changes(self):
        models,graphs,batch,_,_=self.models();m=models[1]
        with torch.no_grad():m.residual.weight.normal_(0,.01)
        cache=self.cache(m,batch);f,y=run.minibatch(self.core(),cache,[2,0,1],torch.device('cpu'))
        raw=Batch.from_data_list([graphs[i] for i in (2,0,1)])
        torch.testing.assert_close(m.forward_features(*f)[0],m(raw),atol=1e-5,rtol=1e-5);self.assertEqual(y.tolist(),[2,0,1])
        fresh,_,_,_,_=self.models();fresh[1].load_state_dict(m.state_dict(),strict=True);fresh[1].eval()
        torch.testing.assert_close(fresh[1](batch),m(batch))

    def test_device_backward_preflight_and_hook_cleanup(self):
        models,_,batch,_,_=self.models();m=models[1]
        before=sum(len(x._forward_hooks)+len(x._forward_pre_hooks) for x in m.packet_encoder.modules())
        with patch.object(m.packet_encoder,'forward',side_effect=RuntimeError('injected')),self.assertRaises(RuntimeError):m.encode(batch)
        after=sum(len(x._forward_hooks)+len(x._forward_pre_hooks) for x in m.packet_encoder.modules());self.assertEqual(before,after)
        with tempfile.TemporaryDirectory() as folder:
            r=run.device_preflight(self.core(),dict(zip(run.VARIANTS,models)),torch.device('cpu'),SimpleNamespace(output=Path(folder)))
            self.assertEqual(r['status'],'passed');self.assertTrue(r['model_and_rng_unchanged'])

    def test_folding_trained_affines_preserves_outputs_and_source_parameters(self):
        _,_,batch,_,last=self.models()
        with torch.no_grad():last.cross_layer.weight.normal_(0,.05);last.cross_layer.bias.normal_(0,.05)
        f,_=last.encode(batch);before=fixtures.state_hash(last.state_dict().items());rng=torch.get_rng_state().clone()
        folded=fold_last(last.cross_layer,last.fusion)
        torch.testing.assert_close(last.forward_features(*f)[0],folded(f[0],f[2])[0],atol=1e-5,rtol=1e-5)
        self.assertEqual(before,fixtures.state_hash(last.state_dict().items()));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        with self.assertRaises(ValueError):fold_last(torch.nn.Linear(64,64),last.fusion)


if __name__=='__main__':unittest.main()
