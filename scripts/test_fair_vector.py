"""Protocol/adapter/numerical checks, plus optional real torch autograd."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import run_fair_vector as fv

ROOT=Path(__file__).resolve().parents[1]
HAS_TORCH=importlib.util.find_spec('torch') is not None


class Checks(unittest.TestCase):
    def test_import_and_protocol(self):
        self.assertNotIn('torch',sys.modules)
        ast.parse((ROOT/'scripts/run_fair_vector.py').read_text(encoding='utf-8'),feature_version=(3,9))
        spec=json.loads((ROOT/'configs/fair_vector_20261029.json').read_text());fv.check_spec(spec)
        for k,v in [('jobs',['V2']),('V2_minus_N0_min_pp',.05),('V2_minus_V1_min_pp',0),('automatic_followup',True)]:
            bad=copy.deepcopy(spec);bad[k]=v
            with self.assertRaises(ValueError):fv.check_spec(bad)

    def test_adapter_changes_diagnostic_only(self):
        original=fv.pair.train
        adapted=fv.e0_trainer()
        self.assertEqual(adapted.__code__.co_argcount,original.__code__.co_argcount)
        self.assertIs(fv.pair.train,original)
        self.assertNotIn('first_update_gradients.json',adapted.__code__.co_consts)
        self.assertIn('best.pt',adapted.__code__.co_consts)

    def test_function_namespaces_are_isolated(self):
        original=fv.ff.predict
        functions=fv.vector_functions()
        self.assertIs(fv.ff.predict,original)
        self.assertIs(functions['train'].__globals__['evaluate'],fv.vector_evaluate)
        self.assertIs(functions['update'].__globals__['predict'],fv.vector_predict)

    def test_thresholds_require_both_and_no_auto_followup(self):
        self.assertTrue(fv.screening(.96,.9604,.962)['budget_threshold_passed'])
        self.assertFalse(fv.screening(.96,.9609,.9611)['budget_threshold_passed'])
        self.assertFalse(fv.screening(.96,.9601,.9609)['budget_threshold_passed'])
        self.assertFalse(fv.screening(.96,.9604,.962)['automatic_followup'])

    def test_seed42_sampler_and_N0_references(self):
        calls=[]
        core=fv.Core(SimpleNamespace(TrainBatches=lambda *a:calls.append(a)),42,Path('.'))
        core.TrainBatches(100,512,42);self.assertEqual(calls,[(100,512,42)])
        refs=json.loads((ROOT/'configs/fair_vector_n0_references_20261029.json').read_text())
        self.assertEqual(set(refs),{'42','43','44'})
        for r in refs.values():fv.verify_metrics(r['metrics']['best_validation'])

    def test_vector_prediction_with_numpy(self):
        import numpy as np
        class T:
            def __init__(self,x):self.x=np.asarray(x)
            @property
            def shape(self):return self.x.shape
            def new_full(self,shape,value):return T(np.full(shape,value))
            def sigmoid(self):return T(1/(1+np.exp(-self.x)))
            def expand_as(self,other):return T(np.broadcast_to(self.x,other.shape))
            def __getitem__(self,ix):return T(self.x[ix])
            def __mul__(self,other):return T(self.x*other.x)
            def __rsub__(self,other):return T(other-self.x)
            def __add__(self,other):return T(self.x+other.x)
        rng=np.random.default_rng(42)
        cache={'flow':T(rng.normal(size=(9,8))),'packet':T(rng.normal(size=(9,8))),'features':T(rng.normal(size=(9,21)))}
        core=SimpleNamespace(F=SimpleNamespace(log_softmax=lambda x,dim:x))
        head=lambda x:x
        _,alpha,mixed=fv.vector_predict(core,head,SimpleNamespace(logit=T(np.zeros(8))),'V1',cache,slice(None))
        np.testing.assert_allclose(mixed.x,.5*(cache['flow'].x+cache['packet'].x))
        self.assertEqual(alpha.shape,(9,8))
        fusion=SimpleNamespace(logit=T(np.arange(8)-4))
        _,a,z=fv.vector_predict(core,head,fusion,'V1',cache,slice(None))
        np.testing.assert_allclose(z.x,a.x*cache['flow'].x+(1-a.x)*cache['packet'].x)


@unittest.skipUnless(HAS_TORCH,'torch unavailable locally; real server probes required')
class TensorCheck(unittest.TestCase):
    def test_actual_gradient_through_frozen_head(self):
        import torch
        from torch.nn import functional as F
        torch.manual_seed(42)
        core=SimpleNamespace(torch=torch,F=F,state_hash=lambda xs:repr([(n,t.detach().tolist()) for n,t in xs]))
        model=torch.nn.Module();model.shared_head=torch.nn.Sequential(torch.nn.Linear(8,32),torch.nn.LeakyReLU(),torch.nn.Linear(32,8))
        model.gate=torch.nn.Module();model.gate.net=torch.nn.Sequential(torch.nn.Linear(21,16),torch.nn.LeakyReLU(),torch.nn.Linear(16,2))
        model.requires_grad_(False);model.eval()
        cache={'flow':torch.randn(32,8),'packet':torch.randn(32,8),'features':torch.randn(32,21),'y':torch.arange(32)%8}
        functions=fv.vector_functions()
        for variant in ('V1','V2'):
            fusion=fv.make_vector(core,model,variant,torch.device('cpu'));opt=torch.optim.Adam(fusion.parameters(),lr=.001)
            for i in range(3):
                _,receipt=functions['update'](core,model,fusion,variant,cache,slice(None),opt,probe=True)
                self.assertGreater(receipt['head_input_gradient_l1'],0)
                if variant=='V2' and i>0:self.assertGreater(receipt['fusion_gradients_l1']['0.weight'],0)
            self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__=='__main__':unittest.main()
