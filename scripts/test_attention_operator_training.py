"""Training lifecycle checks on tiny raw graphs; no dataset or score claims."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import run_attention_operator_pair as run
from probe_attention_operator_pair import synthetic_graphs
from summarize_attention_operator_pair import matrix_report,check_history,report


class TinyGraphs:
    def __init__(self,graphs):
        self.graphs={g.sample_id:g for g in graphs}
        self.loads=[]

    def load(self,row):
        if row['partition'] not in ('train','val'):
            raise AssertionError('test graph loaded')
        self.loads.append(row['partition'])
        return self.graphs[row['sample_id']].clone()


class TrainingChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);run.seed_all(42)

    def test_independent_sampler_reproducibility_and_singleton_merge(self):
        a=run.TrainBatches(9,4,42);b=run.TrainBatches(9,4,42)
        previous=None
        for _ in range(4):
            first=list(a);torch.rand(100);second=list(b)
            self.assertEqual(first,second);self.assertEqual(a.order_sha256,b.order_sha256)
            self.assertEqual(sorted(sum(first,[])),list(range(9)))
            self.assertEqual([len(v) for v in first],[4,5])
            if previous is not None:self.assertNotEqual(previous,a.order_sha256)
            previous=a.order_sha256
        self.assertEqual(len(a),2)

    def test_graph_dataset_rejects_test_and_mixed_partitions(self):
        with self.assertRaises(ValueError):run.GraphRows(None,[{'partition':'test'}],'test')
        with self.assertRaises(ValueError):run.GraphRows(None,[{'partition':'val'}],'train')

    def test_full_budget_ledger_ties_keep_epoch_zero_and_earliest_trained(self):
        metrics=matrix_report([[int(i==j) for j in range(8)] for i in range(8)])
        history=[{'epoch':i,'validation':metrics,'best_epoch':0,'best_trained_epoch':1,
            'samples':32,'optimizer_updates':211,'train_nll':1.,'train_order_sha256':str(i)} for i in range(1,151)]
        m={'epoch_zero':metrics,'best_epoch':0,'best_trained_epoch':1,'best_validation':metrics,
            'trained_best_validation':metrics,'optimizer_updates':31650,'epochs':150,'last_validation':metrics,
            'orders':[str(i) for i in range(1,151)],'training_samples_per_epoch':32}
        check_history(history,m,150,211,8)
        broken=copy.deepcopy(history);broken[50]['best_trained_epoch']=51
        with self.assertRaises(ValueError):check_history(broken,m,150,211,8)
        with self.assertRaises(ValueError):check_history(history[:149],m,150,211,8)

    def test_comparison_counts_identity_and_both_wrong_changes(self):
        a=[{'sample_id':str(i),'label':0,'prediction':p} for i,p in enumerate((1,0,0,1))]
        b=[{'sample_id':str(i),'label':0,'prediction':p} for i,p in enumerate((0,1,0,2))]
        counts,changed=run.compare(a,b)
        self.assertEqual(counts['counts'],{'repaired':1,'regressed':1,'both_correct':1,'both_wrong':1})
        self.assertEqual(len(changed),3);self.assertEqual(changed[-1]['change'],'both_wrong_changed')
        with self.assertRaises(ValueError):run.compare(a,list(reversed(b)))

    def test_cuda_probe_validator_rejects_dirty_and_changed_receipts(self):
        prep=run.read(run.ROOT/'configs/attention_operator_pair_s42_v1.json')
        # Fixture only: construct validator inputs, not a claim of executing CUDA.
        sources=['src/xgnid/attention_operator_pair.py','src/xgnid/model.py','src/xgnid/fusion_dimension.py',
            'scripts/probe_attention_operator_pair.py','scripts/grouped_graphs.py','scripts/audit_graph_split.py',
            'scripts/build_grouped_split.py','scripts/e3_gate_study.py','configs/attention_operator_pair_s42_v1.json']
        update={'nll':2.,'seconds':.1,'gradient_l1':{'flow_encoder.a':1.,'packet_encoder.a':1.,'fusion.a':1.}}
        p={'status':'passed','protocol':prep,'provenance':{'mode':'raw_graphs','test_graph_loads':0},
            'device':'cuda:0','torch':'2.3.1+cu118','pyg':'2.5.3','train_batch_graphs':512,'val_batch_graphs':16,
            'formal_training_started':False,'full_validation_evaluated':False,'pairing':{'all_common_tensors_identical':True,'common_tensors':100},
            'sources':{n:run.sha(run.ROOT/n) for n in sources},'models':{}}
        for name,count in [('GAT',851688),('GATv2',850152)]:
            p['models'][name]={'total_parameters':count,'trainable_parameters':count,'clean_state_restored':True,
                'optimizer_discarded':True,'rng_restored':True,'updates':[{**update,'step':i} for i in (1,2,3)],
                'changed_tensors':['flow_encoder.convs1.weight','packet_encoder.bns2.packet.running_mean']}
        run.validate_probe(p,prep,run.ROOT)
        for kind in ('dirty','synthetic','changed_source','zero_gradient'):
            broken=copy.deepcopy(p)
            if kind=='dirty':broken['models']['GATv2']['optimizer_discarded']=False
            elif kind=='synthetic':broken['provenance']['mode']='synthetic'
            elif kind=='changed_source':broken['sources'][sources[0]]='0'*64
            else:broken['models']['GAT']['updates'][0]['gradient_l1']['flow_encoder.a']=0.
            with self.assertRaises(ValueError):run.validate_probe(broken,prep,run.ROOT)

    def test_tiny_paired_training_full_checkpoint_export_and_replay_lifecycle(self):
        graphs=synthetic_graphs();data=TinyGraphs(graphs)
        tr=[{'partition':'train','sample_id':g.sample_id,'label':int(g.y)} for g in graphs]
        va=[{**r,'partition':'val'} for r in tr]
        c={'seed':42,'batch_size':4,'eval_batch_size':4,'workers':0,'encoder_lr':.001,'head_lr':.001,'weight_decay':0.,'epochs':2}
        models={name:run.OperatorClassifier(name,3,5) for name in run.VARIANTS}
        run.assert_paired(models)
        scratch=run.ROOT/'tmp';scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch,prefix='operator_training_test_') as folder,patch.object(run,'progress'):
            root=Path(folder);results={};exports={}
            for name,model in models.items():
                run.seed_all(42)
                results[name],exports[name]=run.fit(name,model,data,tr,va,torch.device('cpu'),c,{'fixture':True},{},root,
                    results['GAT']['orders'] if name=='GATv2' else None)
                m=results[name]
                self.assertEqual(m['optimizer_updates'],4);self.assertEqual(m['epochs'],2)
                self.assertTrue(m['training_evidence']['encoders_and_bn_updated'])
                self.assertTrue(m['replays']['best']['predictions_identical'])
                for file,digest in m['checkpoint_sha256'].items():self.assertEqual(run.sha(root/name/file),digest)
                saved=torch.load(root/name/'best.pt',map_location='cpu',weights_only=False)
                self.assertEqual(saved['epoch'],m['best_epoch'])
                with self.assertRaises(ValueError):
                    run.strict_reload(model,root/name/'best.pt',torch.device('cpu'),{'wrong':'config'},m['best_epoch'],m['best_validation'])
            self.assertEqual(results['GAT']['orders'],results['GATv2']['orders'])
            self.assertEqual(results['GAT']['training_start_rng_sha256'],results['GATv2']['training_start_rng_sha256'])
            run.compare(exports['GAT'],exports['GATv2'])
        self.assertNotIn('test',data.loads)

    def test_complete_report_recomputes_metrics_orders_and_comparison(self):
        support=[2389,544,1739,2278,2400,1475,2388,233]
        metrics=matrix_report([[support[i] if i==j else 0 for j in range(8)] for i in range(8)])
        history=[{'epoch':i,'validation':metrics,'best_epoch':0,'best_trained_epoch':1,
            'samples':107621,'optimizer_updates':211,'train_nll':1.,'train_order_sha256':str(i)} for i in range(1,151)]
        m={'epoch_zero':metrics,'best_epoch':0,'best_trained_epoch':1,'best_validation':metrics,
            'trained_best_validation':metrics,'optimizer_updates':31650,'epochs':150,'last_validation':metrics,
            'orders':[str(i) for i in range(1,151)],'training_samples_per_epoch':107621,
            'training_start_rng_sha256':'fixture','stop_reason':'fixed_150_epochs','early_stop_enabled':False,
            'training_evidence':{'encoders_and_bn_updated':True},'selected_epoch_zero':True,
            'replays':{k:{'strict_state_load':True,'predictions_identical':True,'metrics_identical':True,'samples':13446} for k in ('best','trained_best')},
            'cost':{}}
        s={'status':'completed','exit_code':0,'test_evaluated':False,'automatic_followup':False,'source_artifacts_unchanged':True,
            'protocol':{'epochs':150,'early_stop_enabled':False},'models':{n:{**copy.deepcopy(m),'parameter_count':count}
                for n,count in [('GAT',851688),('GATv2',850152)]},'pairing':{'all_common_tensors_identical':True},'probe_validated':True,
            'preparation_protocol':{'D32_reference_macro_f1':.9732070542814468,'CL_Multi_reference_macro_f1':.9750998613318087,'budget_threshold_pp':.1},
            'comparison':{'counts':{'repaired':0,'regressed':0,'both_correct':13446,'both_wrong':0},'net_correct':0}}
        scratch=run.ROOT/'tmp';scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch,prefix='operator_report_test_') as folder:
            root=Path(folder)
            for name in run.VARIANTS:
                (root/name).mkdir();(root/name/'epochs.jsonl').write_text('\n'.join(json.dumps(v) for v in history)+'\n')
            analysis=report(root,s)
            self.assertFalse(analysis['budget_screen_passed']);self.assertTrue(analysis['exceeds_CL_Multi'])
            self.assertEqual(analysis['GATv2_minus_GAT_pp'],0.)
            self.assertTrue((root/'per_class.csv').exists())
            broken=copy.deepcopy(s);broken['comparison']['counts']['regressed']=1
            with self.assertRaises(ValueError):report(root,broken)


if __name__=='__main__':
    unittest.main()
