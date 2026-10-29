"""Local control/identity checks, plus optional real torch autograd tests."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import run_frozen_fusion as ff

ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec('torch') is not None


class FrozenChecks(unittest.TestCase):
    def test_py39_stdlib_import(self):
        self.assertNotIn('torch',sys.modules)
        ast.parse((ROOT/'scripts/run_frozen_fusion.py').read_text(encoding='utf-8'),feature_version=(3,9))

    def test_controls_reject_budget_and_input_changes(self):
        spec=json.loads((ROOT/'configs/frozen_fusion_20261029.json').read_text())
        ff.check_spec(spec)
        for key,value in [('lr',.0001),('epochs',100),('patience',20),('dynamic_input','prediction stats'),
                          ('include_epoch_zero',False),('proxy_weight_mixture',True),('test_evaluated',True)]:
            altered=copy.deepcopy(spec);altered[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):ff.check_spec(altered)

    def test_n0_registered_metrics(self):
        ref=json.loads((ROOT/'configs/n0_seed42_reference_20261029.json').read_text())
        val=ref['model']['best_validation']
        names=list(val['per_class'])[:8]
        measured=ff.matrix_report(val['confusion_matrix'],names)
        for k in ('accuracy','macro_f1','weighted_f1'):
            self.assertAlmostEqual(measured[k],val[k],places=12)
        for name in names:
            for k in ('precision','recall','f1-score','support'):
                self.assertAlmostEqual(measured['per_class'][name][k],val['per_class'][name][k],places=12)
        self.assertEqual(ref['model']['best_epoch'],5)

    def test_f0_rejects_reordered_labels_and_duplicate_ids(self):
        rows=[{'sample_id':str(i),'label':i%8} for i in range(16)]
        ff.verify_order(copy.deepcopy(rows),rows)
        for changed in (list(reversed(rows)),rows[:-1],rows[:-1]+rows[:1]):
            with self.assertRaises(ValueError):ff.verify_order(changed,rows)
        wrong=copy.deepcopy(rows);wrong[0]['label']=7
        with self.assertRaises(ValueError):ff.verify_order(wrong,rows)

    def test_zero_support_and_epoch_zero_selection(self):
        cm=[[0]*8 for _ in range(8)];cm[0][0]=3
        val=ff.matrix_report(cm,list(map(str,range(8))))
        self.assertEqual((val['accuracy'],val['macro_f1'],val['weighted_f1']),(1.,.125,1.))
        best,stale= .9699623354279772,0
        for _ in range(12):
            best,stale,improved=ff.selection_state(.96,best,stale)
            self.assertFalse(improved)
        self.assertEqual(stale,12)
        self.assertEqual(best,.9699623354279772)

    def test_metadata_binds_server_n0_and_rejects_changed_weights(self):
        # Synthetic server fixture exercises the actual metadata checker without
        # opening tensor files. Production reference still pins the real N0 SHA.
        real=json.loads((ROOT/'configs/n0_seed42_reference_20261029.json').read_text())
        with tempfile.TemporaryDirectory(prefix='frozen-fusion-fixture-') as directory:
            repo=Path(directory); cfgdir=repo/'configs';cfgdir.mkdir()
            root=repo/real['directory'];(root/'N0').mkdir(parents=True)
            ref=copy.deepcopy(real)
            for n in ref['binding']['sources']:
                p=repo/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'frozen source')
                ref['binding']['sources'][n]=ff.sha(p)
            (root/'N0/best.pt').write_bytes(b'trusted fixture checkpoint')
            (root/'N0/validation_predictions.jsonl.gz').write_bytes(b'fixture predictions')
            ref['model']['checkpoint_sha256']['best.pt']=ff.sha(root/'N0/best.pt')
            ref['model']['predictions_sha256']=ff.sha(root/'N0/validation_predictions.jsonl.gz')
            splitdir=repo/'data/processed/ciciot2023_grouped_input_v1';splitdir.mkdir(parents=True)
            (splitdir/'samples.jsonl').write_bytes(b'fixture manifest')
            ref['binding']['manifest_sha256']=ff.sha(splitdir/'samples.jsonl')
            ff.write(splitdir/'split.json',{'split_id':ref['binding']['split_id'],
                'partition_sizes':{'train':107621,'val':13446,'test':13448},
                'sample_manifest':'samples.jsonl','sample_manifest_sha256':ref['binding']['manifest_sha256']})
            for folder in (root,root/'N0'):ff.write(folder/'status.json',{'status':'completed','exit_code':0})
            ff.write(root/'N0/metrics.json',ref['model'])
            ff.write(root/'N0/config.json',{'binding':ref['binding']})
            ff.write(root/'summary.json',{'models':{'N0':ref['model']},'binding':ref['binding']})
            ff.write(cfgdir/'n0_seed42_reference_20261029.json',ref)
            spec=json.loads((ROOT/'configs/frozen_fusion_20261029.json').read_text())
            ff.write(cfgdir/'frozen_fusion_20261029.json',spec)
            args=SimpleNamespace(repo=repo,config=cfgdir/'frozen_fusion_20261029.json')
            ff.metadata(args)
            (root/'N0/best.pt').write_bytes(b'other checkpoint')
            with self.assertRaisesRegex(ValueError,'checkpoint changed'):ff.metadata(args)


@unittest.skipUnless(HAS_TORCH,'torch not installed locally; server probe verifies gradients')
class TensorChecks(unittest.TestCase):
    def test_frozen_head_keeps_input_gradients_and_only_fusion_updates(self):
        import torch
        from torch.nn import functional as F
        torch.manual_seed(42)
        core=SimpleNamespace(torch=torch,F=F,state_hash=lambda items:ff.identity(
            [(n,t.detach().tolist()) for n,t in items]))
        model=torch.nn.Module()
        model.shared_head=torch.nn.Sequential(torch.nn.Linear(8,32),torch.nn.LeakyReLU(),torch.nn.Linear(32,8))
        model.gate=torch.nn.Module()
        model.gate.net=torch.nn.Sequential(torch.nn.Linear(21,16),torch.nn.LeakyReLU(),torch.nn.Linear(16,2))
        model.requires_grad_(False);model.eval()
        cache={'flow':torch.randn(32,8),'packet':torch.randn(32,8),'features':torch.randn(32,21),'y':torch.arange(32)%8}
        original=core.state_hash(model.state_dict().items())
        for variant in ff.VARIANTS:
            fusion=ff.make_fusion(core,model,variant,torch.device('cpu'))
            opt=torch.optim.Adam(fusion.parameters(),lr=.001)
            _,w,_=ff.predict(core,model.shared_head,fusion,variant,cache,slice(None))
            self.assertTrue(bool((w==.5).all()))
            observations=[]
            for _ in range(3):
                _,r=ff.update(core,model,fusion,variant,cache,slice(None),opt,probe=True)
                self.assertGreater(r['head_input_gradient_l1'],0)
                observations.append(r)
            if variant=='F2':self.assertGreater(observations[1]['fusion_gradients_l1']['0.weight'],0)
            self.assertEqual(original,core.state_hash(model.state_dict().items()))


if __name__ == '__main__':
    unittest.main()
