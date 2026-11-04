"""No-torch residual protocol, matched common readout and report checks."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import run_residual_pair as run
import test_relation_pair as prior
from summarize_residual_pair import analyze,budget,summarize
ROOT=Path(__file__).resolve().parents[1]


class Contracts(unittest.TestCase):
    def config(self):return run.read(ROOT/'configs/residual_pair_s42_v1.json')
    def audit(self):
        return {'status':'passed','optimizer_updates':0,'source_files_unchanged':True,'rng_restored':True,
            'folding':{'status':'passed','predictions_identical':True,'metrics_identical':True,'samples':13446,'atol':1e-5,'rtol':1e-5},
            'checkpoint_verification':{n:{'epoch':e,'cached_prediction_replay_identical':True,'strict_head_and_projection_load':True} for n,e in (('CL-Last',113),('CL-Multi',126))}}

    def test_prerequisite_rejects_wrong_epoch_changed_prediction_or_relaxed_tolerance(self):
        run.validate_prerequisite(self.audit())
        for mutate in (lambda a:a['folding'].update(predictions_identical=False),lambda a:a['folding'].update(atol=1e-3),
                       lambda a:a['checkpoint_verification']['CL-Last'].update(epoch=0),lambda a:a.update(optimizer_updates=1)):
            a=self.audit();mutate(a)
            with self.assertRaises(ValueError):run.validate_prerequisite(a)
    def test_registered_residual_location_and_matched_readout(self):
        c=self.config();run.check_config(c)
        for k,v in [('epochs',100),('early_stop_enabled',True),('common_last_projection',None),
                    ('residual_initialization','random'),('residual_bias',True),('cross_layer_lr',1e-5),
                    ('changed_location','post_BN'),('source_epoch',113)]:
            with self.subTest(k=k),self.assertRaises(ValueError):run.check_config({**c,k:v})
        self.assertEqual(run.architecture('base')['trainable_parameters'],15400)
        self.assertEqual(run.architecture('skip')['trainable_parameters'],19496)
        self.assertEqual(run.architecture('base')['common_last_parameters'],8256)

    def test_all150_epochs_late_recovery_and_epoch_zero_ties(self):
        for scores in ([.96]*150,[.96]*149+[.975],[.9732070542814468]*150):
            h,m=prior.Contracts().history(scores);run.check_history(h,m,self.config())
        self.assertEqual(m['best_epoch'],0)
        h,m=prior.Contracts().history([.96]*149+[.975]);self.assertEqual(m['best_epoch'],150)
        for n in (20,50,149):
            h,m=prior.Contracts().history([.96]*n)
            with self.assertRaises(ValueError):run.check_history(h,m,self.config())

    def test_budget_uses_matched_control(self):
        a=budget(.976,.9765,.9732070542814468,.9750998613318087)
        self.assertFalse(a['budget_screen_passed']);self.assertGreater(a['Res_Skip_minus_D32_pp'],.1)

    def test_optimizer_common_components_receive_identical_training_opportunity(self):
        class P:
            def __init__(self,n,active=True):self.n,self.requires_grad=n,active
            def numel(self):return self.n
        captured=[]
        def Adam(groups,**kw):captured.append((groups,kw));return SimpleNamespace(state={})
        core=SimpleNamespace(torch=SimpleNamespace(optim=SimpleNamespace(Adam=Adam)))
        for mode in ('base','skip'):
            h,p,r,e=P(7144),P(8256),P(4096),P(7,False)
            m=SimpleNamespace(pooling=mode,fusion=SimpleNamespace(parameters=lambda:iter([h])),
                cross_layer=SimpleNamespace(parameters=lambda:iter([p])),residual=SimpleNamespace(parameters=lambda:iter([r])) if mode=='skip' else None,
                parameters=lambda:iter([h,p,e]+([r] if mode=='skip' else [])))
            run.optimizer(core,m,self.config())
            self.assertEqual([g['lr'] for g in captured[-1][0]],[1e-5,1e-4]+([1e-4] if mode=='skip' else []))
            e.requires_grad=True
            with self.assertRaises(ValueError):run.optimizer(core,m,self.config())

    def fixture(self):
        s=prior.Contracts().fixture();s['protocol']=self.config()
        for key in ('models','probes','graph_replays','trained_best_graph_replays'):
            s[key]={new:s[key][old] for new,old in (('Res-Base','Rel-Mean'),('Res-Skip','Rel-Gate'))}
        for n,mode in run.VARIANTS.items():s['models'][n]['architecture']=run.architecture(mode)
        for old,new in (('cuda_relation_preflight','cuda_residual_preflight'),('relation_caches_unchanged','residual_caches_unchanged'),('relation_diagnostics','residual_diagnostics')):s[new]=s.pop(old)
        s['initialization']['zero_residual_identity']=s['initialization'].pop('uniform_relations')
        acc=s['graph_load_accounting'];s['graph_load_accounting']=run.packet.graph_load_accounting(acc['main_process_counts'],
            [{**r,'phase':r['phase'].replace('Rel-Mean','Res-Base').replace('Rel-Gate','Res-Skip')} for r in acc['loader_stages']])
        s['prior_cross_layer_audit']={'receipt':self.audit(),'sha256':'audit'}
        return s

    def test_report_and_failed_acceptance_rejection(self):
        s=self.fixture();analyze(s)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run.write(root/'summary.json',s)
            for n in run.VARIANTS:
                (root/n).mkdir();h,_=prior.Contracts().history([s['models'][n]['epoch_zero']['macro_f1']]*150)
                for row in h:row['train_order_sha256']='order'
                (root/n/'epochs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in h))
            text=summarize(root);self.assertIn('15400',text);self.assertIn('19496',text)
            self.assertEqual(len((root/'per_class.csv').read_text().splitlines()),57)
        for mutate in (lambda x:x['models']['Res-Skip']['orders'].__setitem__(0,'wrong'),
                       lambda x:x['trained_best_graph_replays']['Res-Skip'].update(confusion_matrix_reproduced=False)):
            bad=copy.deepcopy(s);mutate(bad)
            with self.assertRaises(ValueError):analyze(bad)


if __name__=='__main__':unittest.main()
