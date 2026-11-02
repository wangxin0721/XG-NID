"""No-torch checks for a single fixed-budget confirmation and its attribution."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import run_packet_pool_confirmation as run
from summarize_packet_pool_confirmation import analyze, summarize

ROOT = Path(__file__).resolve().parents[1]


class Contracts(unittest.TestCase):
    def config(self):return run.read(ROOT/'configs/packet_pool_confirmation_s42_v1.json')

    def history(self,scores):
        initial = .9732070542814468
        c = self.config();best,selected,stale = -float('inf'),None,0;rows=[]
        for epoch,score in enumerate(scores,1):
            state = run.selection_step(initial,best,selected,stale,score,epoch,c)
            best,selected,stale = state['trained_best'],state['trained_epoch'],state['stale']
            rows.append({'epoch':epoch,'val_macro_f1':score,'train_nll':.01,
                'optimizer_updates':211,'train_order_sha256':str(epoch),'best_epoch':state['final_epoch'],
                'best_trained_epoch':selected,'best_trained_macro_f1':best,'stale_epochs':stale})
        m = {'epochs':len(rows),'epoch_zero':{'macro_f1':initial},'best_epoch':state['final_epoch'],
            'best_trained_epoch':selected,'selected_epoch_zero':state['final_epoch']==0,'epoch_zero_eligible':True,
            'orders':[str(i) for i in range(1,len(rows)+1)],'optimizer_updates':211*len(rows),
            'best_validation':{'macro_f1':max(initial,best)},'trained_best_validation':{'macro_f1':best},
            'last_validation':{'macro_f1':scores[-1]}}
        return rows,m

    def test_exact_protocol_changes_only_budget_and_confirmation_registration(self):
        self.assertNotIn('torch',sys.modules)
        c = self.config();run.check_config(c)
        old = run.read(ROOT/'configs/packet_pool_pair_s42_v1.json')
        self.assertEqual({k:v for k,v in old.items() if k not in ('schema','epochs')},
                         {k:c[k] for k in old if k not in ('schema','epochs')})
        for key,value in [('epochs',100),('early_stop_enabled',True),('confirmation_rounds',2),
                ('pool_lr',1e-5),('seed',43),('restart','resume_epoch20'),('automatic_followup',True)]:
            with self.subTest(key=key),self.assertRaises(ValueError):run.check_config({**c,key:value})

    def test_full150_ledger_requires_all_updates_even_after149_stale_epochs(self):
        h,m = self.history([.96]*150);run.check_history(h,m,self.config())
        self.assertEqual(h[-1]['stale_epochs'],149);self.assertEqual(m['best_epoch'],0)
        self.assertEqual(m['optimizer_updates'],31650)
        for count in (20,50,100,149):
            h,m = self.history([.96]*count)
            with self.assertRaises(ValueError):run.check_history(h,m,self.config())

    def test_late_rebound_is_selected_and_epoch_zero_wins_ties(self):
        h,m = self.history([.96]*149+[.9745]);run.check_history(h,m,self.config())
        self.assertEqual(m['best_epoch'],150);self.assertEqual(m['best_trained_epoch'],150)
        h,m = self.history([.9732070542814468]*150);run.check_history(h,m,self.config())
        self.assertEqual(m['best_epoch'],0);self.assertEqual(m['best_trained_epoch'],1)

    def test_packet_specific_train_bindings_do_not_mutate_historical_trainers(self):
        self.assertIs(run.train.__globals__['evaluate'],run.packet.evaluate)
        self.assertIs(run.train.__globals__['minibatch'],run.packet.minibatch)
        self.assertIs(run.train.__globals__['optimizer'],run.packet.optimizer)
        self.assertIs(run.fixed.train.__globals__['evaluate'],run.fixed.evaluate)
        self.assertIs(run.packet.train.__globals__['evaluate'],run.packet.evaluate)
        self.assertEqual(run.architecture('attention')['trainable_parameters'],8200)

    def test_prefix_comparison_separates_late_gain_from_different_trajectories(self):
        h,m = self.history([.96]*20+[.9745]*130);old=copy.deepcopy(h[:20]);base=m['epoch_zero']['macro_f1']
        result=run.prefix_comparison(h,old,base)
        self.assertTrue(result['first20_reproduced']);self.assertGreater(result['gain_after20_pp'],.1)
        self.assertEqual(result['best_trained_epoch'],21)
        for key,value in [('train_order_sha256','different'),('val_macro_f1',.961),('train_nll',.010001)]:
            changed=copy.deepcopy(old);changed[0][key]=value
            r=run.prefix_comparison(h,changed,base)
            self.assertFalse(r['first20_reproduced']);self.assertIn('do_not_attribute',r['interpretation'])

    def test_previous_artifacts_are_read_only_and_inventory_is_verified(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'P0').mkdir();file=root/'P0/best.pt';file.write_bytes(b'previous weights')
            receipt={'directory':str(root),'file_sha256':{'P0/best.pt':run.sha(file)}}
            run.assert_previous_unchanged(receipt)
            file.write_bytes(b'changed')
            with self.assertRaises(ValueError):run.assert_previous_unchanged(receipt)
            file.write_bytes(b'previous weights');(root/'unexpected.json').write_text('{}')
            with self.assertRaises(ValueError):run.assert_previous_unchanged(receipt)

    def test_completed_screen_receipt_checks_histories_hashes_and_split(self):
        _,s=self.fixture()
        archived=run.read(ROOT/'docs/packet_pool_results_20261102.json')
        old_config=run.read(ROOT/'configs/packet_pool_pair_s42_v1.json')
        binding={'source':{'directory':'mature','file_sha256':{}},'source_pair_binding':{'bound':'source'},
            'sources':{'scripts/run_packet_pool_pair.py':run.sha(ROOT/'scripts/run_packet_pool_pair.py')}}
        previous=[None,(None,None,{'binding':{'split_id':'split'}}),None,
            {'binding':{'encoder_source':{'encoder_sha256':'encoder'}}}]
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);receipts={}
            for part,count in (('train',107621),('val',13446)):
                path=root/(part+'_packet_nodes.pt');path.write_bytes(('frozen '+part).encode())
                receipts[part]={'sha256':run.sha(path),'tensor_sha256':'tensor',
                    'metadata':{'schema':'xgnid.packet-node64-cache.v1','partition':part,'samples':count,
                        'encoder_sha256':'encoder','split_binding':{'split_id':'split'},'sample_order_sha256':'ids'}}
            s.update(protocol=old_config,binding=binding,models=copy.deepcopy(archived['models']),
                cache_receipts=receipts,cache_memory_estimates={},
                cache_recovery={'formal_training_updates_before_failure':0,'probe_updates_before_failure':0})
            stages=[run.packet.graph_load_stage(phase+'_'+n,'val',13446,2) for phase in ('initial','best') for n in run.VARIANTS]
            s['graph_load_accounting']=run.packet.graph_load_accounting({'train':1120,'val':0,'test':0},stages)
            s['graph_load_counts']=s['graph_load_accounting']['totals'];s['common_epochs_order_checked']=20
            for name,m in s['models'].items():
                arm=root/name;arm.mkdir()
                for file in ('best.pt','last.pt','validation_predictions.jsonl.gz'):(arm/file).write_bytes(file.encode())
                m['checkpoint_sha256']={file:run.sha(arm/file) for file in ('best.pt','last.pt')}
                m['predictions_sha256']=run.sha(arm/'validation_predictions.jsonl.gz')
                scores=[.95]*20;t=m['training_evidence']
                scores[t['best_trained_epoch']-1]=t['best_trained_macro_f1'];scores[-1]=m['last_validation']['macro_f1']
                history=[{'epoch':i+1,'val_macro_f1':score,'train_nll':.01,'optimizer_updates':211,
                    'train_order_sha256':m['orders'][i],'best_epoch':0,'stale_epochs':i+1} for i,score in enumerate(scores)]
                (arm/'epochs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history))
                run.write(arm/'metrics.json',m)
            for file,value in [('status.json',{'status':'completed','exit_code':0}),('summary.json',s),
                    ('binding.json',binding),('config.json',old_config),('cache_receipts.json',receipts)]:run.write(root/file,value)
            meta=(self.config(),previous,None,None,binding)
            receipt=run.previous_receipt(root,meta);run.assert_previous_unchanged(receipt)
            self.assertEqual(receipt['reuse_scope'],'frozen_node_cache_only_no_model_or_optimizer_resume')
            (root/'train_packet_nodes.pt').write_bytes(b'changed frozen cache')
            with self.assertRaises(ValueError):run.previous_receipt(root,meta)

    def fixture(self):
        base=run.read(ROOT/'docs/packet_pool_results_20261102.json')['models']['P0']['epoch_zero']
        h,m=self.history([base['macro_f1']]*150)
        models={n:{**copy.deepcopy(m),'architecture':run.architecture(mode),'epoch_zero':base,
            'best_validation':base,'trained_best_validation':base,'last_validation':base,
            'training_start_rng_sha256':'rng','stop_reason':'max_epochs','early_stop_reference':'trained_epochs_only',
            'training_evidence':run.dim.trained_evidence(base['macro_f1'],h),'parameter_groups':[]}
            for n,mode in run.VARIANTS.items()}
        stages=[run.packet.graph_load_stage(phase+'_'+n,'val',13446,2) for phase in ('initial','best') for n in models]
        stages +=[run.packet.graph_load_stage('trained_best_'+n,'val',13446,2) for n in models]
        acc=run.packet.graph_load_accounting({'train':1120,'val':0,'test':0},stages)
        replay={'strict_state_load':True,'predictions_identical':True,'confusion_matrix_reproduced':True}
        old={n:copy.deepcopy(h[:20]) for n in models}
        return h,{'status':'completed','exit_code':0,'protocol':self.config(),'models':models,
            'source_model':{'best_epoch':62,'best_validation':base},'test_evaluated':False,'automatic_followup':False,
            'training_jobs':2,'experiment_kind':'one_round_training_budget_confirmation_not_new_model','cache_reused':True,
            'tensor_preflight_passed':True,'encoder_parameters_and_BN_unchanged':True,'source_artifacts_unchanged':True,'node_caches_unchanged':True,
            'cuda_readout_preflight':{'status':'passed','optimizer_updates':0,'model_and_rng_unchanged':True},
            'initialization':dict.fromkeys(('P0_P1_outputs_allclose','source_predictions_identical','uniform_packet_attention','epoch_zero_eligible'),True),
            'common_epochs_order_checked':150,'graph_load_counts':acc['totals'],'graph_load_accounting':acc,
            'probes':{n:{'clean_model_and_rng_restored':True,'optimizer_discarded':True} for n in models},
            'graph_replays':{n:dict(replay) for n in models},
            'trained_best_graph_replays':{n:{**replay,'shared_with_selected_best':False} for n in models},
            'completed_cache_source':{'models':{n:{'epochs':20,'best_epoch':0} for n in models},'histories':old},
            'budget_confirmation':{n:run.prefix_comparison(h,old[n],base['macro_f1']) for n in models},
            'comparisons':{n:{'counts':{'repaired':0,'regressed':0},'net_correct':0} for n in ('P1_vs_P0','P0_vs_D32','P1_vs_D32')},
            'prediction_transitions':{n:[] for n in ('P1_vs_P0','P0_vs_D32','P1_vs_D32')}}

    def test_report_includes_trained_best_late_gain_and_all_class_metrics(self):
        h,s=self.fixture();a=analyze(s)
        self.assertFalse(a['budget_screen_passed']);self.assertTrue(a['first20_reproduced'])
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run.write(root/'summary.json',s)
            for name in s['models']:
                (root/name).mkdir();(root/name/'epochs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in h))
            text=summarize(root)
            self.assertIn('not a new model',text);self.assertIn('all150 epochs',text)
            self.assertIn('trained best F1',text);self.assertIn('gain after20',text)
            self.assertEqual(len((root/'per_class.csv').read_text().splitlines()),57)
            broken=copy.deepcopy(h);broken.pop()
            (root/'P0/epochs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in broken))
            with self.assertRaises(ValueError):summarize(root)

    def test_report_rejects_incomplete_budget_or_missing_raw_replay(self):
        _,s=self.fixture()
        for change in (lambda x:x['models']['P1'].update(epochs=20),
            lambda x:x['models']['P1'].update(optimizer_updates=4220),
            lambda x:x['trained_best_graph_replays']['P0'].update(predictions_identical=False),
            lambda x:x['graph_load_accounting']['loader_stages'].pop(),
            lambda x:x['models']['P1']['orders'].__setitem__(0,'wrong')):
            bad=copy.deepcopy(s);change(bad)
            with self.assertRaises(ValueError):analyze(bad)


if __name__ == '__main__':unittest.main()
