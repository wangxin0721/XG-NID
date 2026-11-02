"""Standard-library cross-layer protocol, ledger and report checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import run_cross_layer_pair as run
import test_relation_pair as previous
from summarize_cross_layer_pair import analyze,budget,summarize
ROOT=Path(__file__).resolve().parents[1]


class Contracts(unittest.TestCase):
    def config(self):return run.read(ROOT/'configs/cross_layer_pair_s42_v1.json')

    def test_exact_protocol(self):
        c=self.config();run.check_config(c)
        for k,v in [('epochs',100),('early_stop_enabled',True),('cross_layer_lr',1e-5),
                    ('cross_layer_initialization','random'),('include_epoch_zero',False),('encoder_mode','train')]:
            with self.assertRaises(ValueError):run.check_config({**c,k:v})
        self.assertEqual(run.architecture('last')['trainable_parameters'],15400)
        self.assertEqual(run.architecture('multi')['cross_layer_parameters'],128*64+64)

    def history(self,scores):
        h,m=previous.Contracts().history(scores)
        return h,m

    def test_full_budget_and_late_rebound(self):
        for scores in ([.96]*150,[.96]*149+[.975],[.9732070542814468]*150):
            h,m=self.history(scores);run.check_history(h,m,self.config())
        self.assertEqual(m['best_epoch'],0)
        h,m=self.history([.96]*149+[.975]);self.assertEqual(m['best_epoch'],150)
        for n in (20,50,149):
            h,m=self.history([.96]*n)
            with self.assertRaises(ValueError):run.check_history(h,m,self.config())

    def test_budget_uses_matched_last_and_p1_is_only_reference(self):
        a=budget(.974,.9745,.9732070542814468,.9742719120553645)
        self.assertFalse(a['budget_screen_passed']);self.assertGreater(a['CL_Multi_minus_P1_reported_pp'],0)

    def test_optimizer_two_groups_and_frozen_exclusion(self):
        class Param:
            def __init__(self,n,active=True):self.n,self.requires_grad=n,active
            def numel(self):return self.n
        h,p,e=Param(7144),Param(8256),Param(7,False);captured=[]
        m=SimpleNamespace(fusion=SimpleNamespace(parameters=lambda:iter([h])),cross_layer=SimpleNamespace(parameters=lambda:iter([p])),parameters=lambda:iter([h,p,e]))
        def Adam(groups,**kw):captured.append((groups,kw));return SimpleNamespace(state={})
        run.optimizer(SimpleNamespace(torch=SimpleNamespace(optim=SimpleNamespace(Adam=Adam))),m,self.config())
        self.assertEqual([g['lr'] for g in captured[0][0]],[1e-5,1e-4])
        e.requires_grad=True
        with self.assertRaises(ValueError):run.optimizer(SimpleNamespace(torch=SimpleNamespace(optim=SimpleNamespace(Adam=Adam))),m,self.config())

    def fixture(self):
        s=previous.Contracts().fixture();s['protocol']=self.config()
        for key in ('models','probes','graph_replays','trained_best_graph_replays'):
            s[key]={n:s[key][old] for n,old in [('CL-Last','Rel-Mean'),('CL-Multi','Rel-Gate')]}
        for n,mode in run.VARIANTS.items():s['models'][n]['architecture']=run.architecture(mode)
        s['cuda_cross_layer_preflight']=s.pop('cuda_relation_preflight')
        s['cross_layer_caches_unchanged']=s.pop('relation_caches_unchanged')
        s['cross_layer_diagnostics']={};s.pop('relation_diagnostics')
        s['initialization']['last_layer_identity']=s['initialization'].pop('uniform_relations')
        stages=[{**r,'phase':r['phase'].replace('Rel-Mean','CL-Last').replace('Rel-Gate','CL-Multi')} for r in s['graph_load_accounting']['loader_stages']]
        s['graph_load_accounting']=run.packet.graph_load_accounting(s['graph_load_accounting']['main_process_counts'],stages)
        return s

    def test_report_all_phases_and_eight_classes(self):
        s=self.fixture();analyze(s)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run.write(root/'summary.json',s)
            for name in run.VARIANTS:
                (root/name).mkdir();h,_=self.history([s['models'][name]['epoch_zero']['macro_f1']]*150)
                for row in h:row['train_order_sha256']='order'
                (root/name/'epochs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in h))
            text=summarize(root)
            self.assertIn('128-to64',text);self.assertIn('97.427191',text)
            self.assertEqual(len((root/'per_class.csv').read_text().splitlines()),57)

    def test_report_rejects_missing_replay_or_different_order(self):
        s=self.fixture()
        for change in (lambda x:x['models']['CL-Multi']['orders'].__setitem__(0,'wrong'),
                       lambda x:x['trained_best_graph_replays']['CL-Last'].update(confusion_matrix_reproduced=False),
                       lambda x:x.update(cross_layer_caches_unchanged=False)):
            bad=copy.deepcopy(s);change(bad)
            with self.assertRaises(ValueError):analyze(bad)


if __name__=='__main__':unittest.main()
