"""No-torch protocol, optimizer, selection and report regression checks."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import run_relation_pair as run
from summarize_relation_pair import analyze,budget,summarize

ROOT=Path(__file__).resolve().parents[1]


class Contracts(unittest.TestCase):
    def config(self):return run.read(ROOT/'configs/relation_pair_s42_v1.json')

    def test_registered_protocol_and_source(self):
        self.assertNotIn('torch',sys.modules)
        c=self.config();run.check_config(c)
        self.assertEqual(run.sha(ROOT/'configs'/c['source_protocol']),c['source_config_sha256'])
        for k,v in [('variants',{'R0':'mean'}),('score_network',[128,16,2]),('relation_lr',1e-3),
                    ('score_output_bias',False),('include_epoch_zero',False),('encoder_mode','train'),
                    ('empty_relation_policy','mask_zero_edges'),('automatic_followup',True)]:
            with self.subTest(k=k),self.assertRaises(ValueError):run.check_config({**c,k:v})
        for k,v in [('min_epochs',40),('early_stop_reference','epoch_zero'),('final_selection','trained_only'),('epochs',100),('early_stop_enabled',True)]:
            with self.assertRaises(ValueError):run.check_config({**c,k:v})

    def history(self,scores):
        c=self.config();initial=.9732070542814468;best=-float('inf');selected=None;stale=0;rows=[]
        for epoch,score in enumerate(scores,1):
            st=run.selection_step(initial,best,selected,stale,score,epoch,c)
            best,selected,stale=st['trained_best'],st['trained_epoch'],st['stale']
            rows.append({'epoch':epoch,'val_macro_f1':score,'optimizer_updates':211,'train_order_sha256':str(epoch),
                'best_epoch':st['final_epoch'],'best_trained_epoch':selected,'stale_epochs':stale})
        return rows,{'epochs':len(rows),'epoch_zero':{'macro_f1':initial},'best_epoch':st['final_epoch'],
            'best_trained_epoch':selected,'trained_best_validation':{'macro_f1':best},
            'best_validation':{'macro_f1':max(initial,best)},'last_validation':{'macro_f1':scores[-1]},
            'epoch_zero_eligible':True,'selected_epoch_zero':st['final_epoch']==0,'orders':[str(i) for i in range(1,len(rows)+1)],
            'optimizer_updates':211*len(rows)}

    def test_trained_recovery_below_initial_resets_counter_and_runs150(self):
        h,m=self.history([.96+i*.00005 for i in range(150)])
        run.check_history(h,m,self.config());self.assertEqual(m['best_epoch'],0);self.assertEqual(m['best_trained_epoch'],150)
        self.assertTrue(all(r['stale_epochs']==0 for r in h))

    def test_full150_epochs_are_required_even_with_long_stale_counters(self):
        h,m=self.history([.96]*150);run.check_history(h,m,self.config())
        self.assertEqual(m['best_trained_epoch'],1);self.assertEqual(h[-1]['stale_epochs'],149)
        for n in (20,40,49,50,100,149):
            h,m=self.history([.96]*n)
            with self.assertRaises(ValueError):run.check_history(h,m,self.config())
        h,m=self.history([.96+i*.0001 for i in range(30)]+[.9629]*120)
        run.check_history(h,m,self.config());self.assertEqual(m['best_trained_epoch'],30)

    def test_final_choice_is_independent_of_trained_best(self):
        h,m=self.history([.974]*150);run.check_history(h,m,self.config())
        self.assertEqual(m['best_epoch'],1);self.assertFalse(m['selected_epoch_zero'])
        h,m=self.history([.9732070542814468]*150);run.check_history(h,m,self.config());self.assertEqual(m['best_epoch'],0)

    def test_capacities_and_independent_training_leave_historical_runner_unchanged(self):
        self.assertEqual(run.architecture('relation_gate')['relation_parameters'],192*16+16+16*2+2)
        self.assertEqual(run.architecture('relation_gate')['trainable_parameters'],10266)
        self.assertIs(run.train.__globals__['evaluate'],run.evaluate)
        self.assertIs(run.packet.train.__globals__['evaluate'],run.packet.evaluate)
        self.assertIsNot(run.train,run.packet.train)

    def test_budget_uses_stronger_mean_and_j32_is_separate_reference(self):
        base=.9732070542814468;j=.97368275
        b=budget(base,base,base,j)
        self.assertFalse(b['budget_screen_passed']);self.assertAlmostEqual(b['required_Gate_macro_f1']*100,97.42070542814468)
        self.assertLess(b['Rel_Gate_minus_J32_reported_pp'],0)
        b=budget(base+.002,base+.0025,base,j)
        self.assertFalse(b['budget_screen_passed']);self.assertGreater(b['Rel_Gate_minus_D32_pp'],.1)

    def test_optimizer_two_rates_and_frozen_exclusion(self):
        class P:
            def __init__(self,n,active=True):self.n,self.requires_grad=n,active
            def numel(self):return self.n
        enc,head,gate=P(7,False),P(7144),P(3122)
        model=SimpleNamespace(pooling='relation_gate',fusion=SimpleNamespace(parameters=lambda:iter([head])),
            relation_gate=SimpleNamespace(parameters=lambda:iter([gate])),parameters=lambda:iter([enc,head,gate]))
        captured=[]
        def Adam(groups,**kw):captured.append((groups,kw));return SimpleNamespace(state={})
        core=SimpleNamespace(torch=SimpleNamespace(optim=SimpleNamespace(Adam=Adam)))
        run.optimizer(core,model,self.config())
        self.assertEqual([g['lr'] for g in captured[0][0]],[1e-5,1e-4]);self.assertEqual(captured[0][1],{'weight_decay':0.})
        enc.requires_grad=True
        with self.assertRaises(ValueError):run.optimizer(core,model,self.config())

    def fixture(self):
        base=run.read(ROOT/'docs/packet_pool_results_20261102.json')['models']['P0']['epoch_zero']
        models={}
        for name,mode in run.VARIANTS.items():
            models[name]={'architecture':run.architecture(mode),'epoch_zero':base,'best_validation':base,'last_validation':base,'trained_best_validation':base,
                'best_epoch':0,'best_trained_epoch':1,'early_stop_reference':'trained_epochs_only','epochs':150,'optimizer_updates':31650,'orders':['order']*150,'training_start_rng_sha256':'rng',
                'epoch_zero_eligible':True,'selected_epoch_zero':True,'training_wall_seconds':1.,'parameter_groups':[],
                'training_evidence':{'best_trained_epoch':1,'best_trained_macro_f1':base['macro_f1']}}
        stages=[run.packet.graph_load_stage('extract_train','train',107621,2),run.packet.graph_load_stage('extract_val','val',13446,2)]
        stages +=[run.packet.graph_load_stage(p+'_'+n,'val',13446,2) for p in ('initial','best') for n in run.VARIANTS]
        return {'status':'completed','exit_code':0,'protocol':self.config(),'models':models,
            'source_model':{'best_epoch':62,'best_validation':base},'test_evaluated':False,'automatic_followup':False,
            'tensor_preflight_passed':True,'encoder_parameters_and_BN_unchanged':True,'source_artifacts_unchanged':True,'relation_caches_unchanged':True,
            'cuda_relation_preflight':{'status':'passed','model_and_rng_unchanged':True},'common_epochs_order_checked':150,
            'probes':{n:{'clean_model_and_rng_restored':True,'optimizer_discarded':True} for n in models},
            'graph_replays':{n:{'strict_state_load':True,'confusion_matrix_reproduced':True} for n in models},
            'trained_best_graph_replays':{n:{'strict_state_load':True,'confusion_matrix_reproduced':True,'shared_with_selected_best':False} for n in models},
            'initialization':{'pair_allclose_1e6':True,'uniform_relations':True,'source_predictions_and_CM_identical':True},
            'comparisons':{n:{'counts':{'repaired':0,'regressed':0},'net_correct':0} for n in ('Gate_vs_Mean','Mean_vs_D32','Gate_vs_D32')},
            'prediction_transitions':{n:[] for n in ('Gate_vs_Mean','Mean_vs_D32','Gate_vs_D32')},
            'relation_diagnostics':{},'graph_load_accounting':run.packet.graph_load_accounting({'train':1120,'val':0,'test':0},stages+
                [run.packet.graph_load_stage('trained_best_'+n,'val',13446,2) for n in models])}

    def test_report_records_zero_epoch_training_evidence_and_all_classes(self):
        s=self.fixture();a=analyze(s);self.assertFalse(a['budget_screen_passed'])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run.write(root/'summary.json',s)
            for name,m in s['models'].items():
                (root/name).mkdir()
                history=[{'epoch':i,'val_macro_f1':m['epoch_zero']['macro_f1'],'optimizer_updates':211,
                    'train_order_sha256':'order','best_epoch':0,'best_trained_epoch':1,'stale_epochs':i-1} for i in range(1,151)]
                (root/name/'epochs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history),encoding='utf-8')
            report=summarize(root)
            self.assertIn('97.368275',report);self.assertIn('WebBased, Recon, Spoofing and BruteForce',report)
            self.assertEqual(len((root/'per_class.csv').read_text().splitlines()),57)

    def test_report_rejects_missing_preflights_or_different_training_orders(self):
        s=self.fixture()
        for change in (lambda x:x.update(relation_caches_unchanged=False),
                       lambda x:x['cuda_relation_preflight'].update(status='failed'),
                       lambda x:x['models']['Rel-Gate']['orders'].__setitem__(0,'wrong'),
                       lambda x:x['graph_load_accounting']['loader_stages'].pop(),
                       lambda x:x['initialization'].update(uniform_relations=False)):
            bad=copy.deepcopy(s);change(bad)
            with self.assertRaises(ValueError):analyze(bad)


if __name__=='__main__':unittest.main()
