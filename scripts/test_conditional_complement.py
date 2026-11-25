"""CPU checks for zero initialization, supervised masks, frozen paths and transactions."""
from contextlib import redirect_stdout
from copy import deepcopy
import inspect
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from torch_geometric.data import Batch

import prepare_conditional_complement as prep
import conditional_complement_training as train
import run_conditional_complement as run
import summarize_conditional_complement as report
import launch_conditional_complement as launch
import audit_conditional_complement_archive as audit
from xgnid.conditional_complement import *
from xgnid.joint_transition_fusion import JointTransitionFusion
from test_packet_transition_readout import fixture
from test_flow_query_joint_packet2 import TinyGraphs
from test_fixed_transition_final_test import TestGraphs as FixedTestGraphs


class Checks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.device=torch.device('cpu')
        self.settings={**prep.raw.read(prep.CONFIG),'epochs':2,'preparation_epochs':2,'batch_size':3,'eval_batch_size':4}
        self.source,self.graphs=fixture();prep.frozen.independent.freeze_c(self.source)
        self.ordered=JointTransitionFusion(self.source.fusion,'joint_ordered',42)
        with torch.no_grad():self.ordered.readout.output.weight.fill_(.01)
        self.ordered.requires_grad_(False).eval()
        self.features,_=prep.frozen.encode(self.source,Batch.from_data_list(self.graphs),self.device)
        self.coords=project(self.ordered,self.features)
        self.labels=torch.tensor([g.y.item() for g in self.graphs])
        self.cache=train.Cache(self.coords,self.labels)
        self.preparation=Preparation().requires_grad_(False).eval()
        self.binding={'source':prep.raw.state_digest(self.ordered.state_dict())}

    def arms(self):return prep.make_arms(self.ordered.fusion,self.preparation,ARMS,self.settings,self.device)
    def assert_tree(self,a,b):
        if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
        elif isinstance(a,dict):
            self.assertEqual(a.keys(),b.keys())
            for k in a:self.assert_tree(a[k],b[k])
        elif isinstance(a,(tuple,list)):
            self.assertEqual(len(a),len(b))
            for x,y in zip(a,b):self.assert_tree(x,y)
        else:self.assertEqual(a,b)

    def test_source_projection_epoch_zero_exact_nonzero_ordered_residual(self):
        before=prep.raw.state_digest(self.source.state_dict());ob=prep.raw.state_digest(self.ordered.state_dict())
        reference=self.ordered(self.features)[0]
        self.assertTrue(torch.equal(reference,self.coords.reference))
        for m in self.arms().values():
            scores,delta,_=m(self.coords)
            self.assertTrue(torch.equal(scores,reference));self.assertFalse(bool(delta.any()))
        self.assertEqual(before,prep.raw.state_digest(self.source.state_dict()))
        self.assertEqual(ob,prep.raw.state_digest(self.ordered.state_dict()))
        self.assertTrue(all(p.grad is None for p in self.source.parameters()))

    def test_parameter_counts_named_rng_and_no_dummy_parameters(self):
        state=torch.get_rng_state().clone();p=Preparation();arms=self.arms()
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertEqual(sum(x.numel() for x in p.parameters()),2640)
        for n,m in arms.items():
            new=sum(p.numel() for k,p in m.named_parameters() if p.requires_grad and not k.startswith(('gate.','shared_head.')))
            self.assertEqual(new,0 if n=='cc_t0' else 3912 if n in ('cc_t1','cc_t3') else 3904)
            self.assertEqual(sum(p.numel() for p in m.parameters() if not p.requires_grad),2112 if n in ('cc_t2','cc_t4') else 0)
        self.assert_tree(arms['cc_t1'].state_dict(),arms['cc_t3'].state_dict())
        self.assert_tree(arms['cc_t2'].state_dict(),arms['cc_t4'].state_dict())
        self.assertEqual(list(inspect.signature(ConditionalComplement.forward).parameters),['self','features'])

    def test_staged_gradients_freeze_and_stable_coordinate_detach(self):
        f=Projected(*(v.detach().clone().requires_grad_(True) for v in self.coords))
        before=prep.raw.state_digest(self.preparation.state_dict())
        with tempfile.TemporaryDirectory() as temp:
            prep.gradient_probe(self.ordered.fusion,self.preparation,self.cache,self.settings,self.device,Path(temp))
        m=self.arms()['cc_t4'];candidate_loss(m,f,self.labels,self.preparation)[0].backward()
        self.assertTrue(all(v.grad is None for v in f))
        self.assertTrue(all(p.grad is None for p in self.preparation.parameters()))
        self.assertEqual(before,prep.raw.state_digest(self.preparation.state_dict()))
        m.train();self.assertFalse(m.flow_to_packet.training);self.assertFalse(m.packet_to_flow.training)

    def test_selective_kl_temperature_batch_denominator_and_detach(self):
        y=torch.tensor([0,1,2,3])
        qf=torch.tensor([[8.,0,0,0],[8.,0,0,0],[0,0,8.,0],[8.,0,0,0]],requires_grad=True)
        qp=torch.tensor([[0.,8,0,0],[0,8.,0,0],[0,0,8.,0],[0,8.,0,0]],requires_grad=True)
        logits=torch.tensor([[.2,.4,.1,.3]]*4,requires_grad=True)
        p=F.log_softmax(logits,-1);p0=p.detach().clone().requires_grad_(True)
        terms=collaborative_loss(p,y,p0,qf,qp)
        expected=4/4*sum(F.kl_div(F.log_softmax(p[i]/2,-1),F.softmax(q/2,-1),reduction='sum')
                         for i,q in ((0,qf[0].detach()),(1,qp[1].detach())))
        torch.testing.assert_close(terms['sd'],expected)
        sum(terms.values()).backward()
        self.assertIsNone(qf.grad);self.assertIsNone(qp.grad);self.assertIsNone(p0.grad)
        self.assertGreater(float(logits.grad.abs().sum()),0)

    def test_empty_disagreement_and_keep_exact(self):
        y=torch.tensor([0,1]);teacher=torch.tensor([[3.,0],[0,3.]])
        ref=F.log_softmax(torch.tensor([[4.,0.],[4.,0.]]),-1)
        scores=F.log_softmax(torch.tensor([[2.,0.],[0.,4.]],requires_grad=True),-1)
        terms=collaborative_loss(scores,y,ref,teacher,teacher)
        self.assertEqual(terms['sd'].item(),0.);self.assertTrue(torch.isfinite(terms['sd']))
        torch.testing.assert_close(terms['keep'],(ref[0,0]-scores[0,0])/2)
        self.assertEqual(terms['sd'].dtype,scores.dtype)

    def test_predictor_training_detaches_inputs_and_final_prep_state(self):
        model=Preparation();f=Projected(*(v.detach().clone().requires_grad_(True) for v in self.coords))
        sum(model.losses(f,self.labels).values()).backward()
        self.assertTrue(all(v.grad is None for v in f))
        self.assertTrue(all(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters()))
        with tempfile.TemporaryDirectory() as temp,redirect_stdout(io.StringIO()):
            state=train.fit({'common':Preparation()},self.cache,self.settings,self.binding,self.device,Path(temp),
                lambda n,m,f,y:(sum((t:=m.losses(f,y)).values()),t),preparation=True)
            self.assertEqual(state['best'],{});self.assertEqual(state['completed_epoch'],2)
            self.assertEqual(state['updates'],2*len(train.transaction.PairedSampler(len(self.labels),3,42020)))

    def fit(self,folder,**kw):
        arms=self.arms()
        return train.fit(arms,self.cache,self.settings,self.binding,self.device,folder,
            lambda n,m,f,y:candidate_loss(m,f,y,self.preparation),
            lambda models:train.measure(models,self.cache,self.settings,self.device),**kw)

    def test_midbatch_failure_rolls_back_all_adam_rng_sampler_and_resumes_exactly(self):
        rng=train.transaction.rng_state()
        with tempfile.TemporaryDirectory() as temp,redirect_stdout(io.StringIO()):
            base=Path(temp);full=self.fit(base/'full')
            train.transaction.restore_rng(rng)
            def fail(n,update):
                if n=='cc_t2' and update==2:raise RuntimeError('injected partial batch')
            with self.assertRaisesRegex(RuntimeError,'partial batch'):self.fit(base/'interrupted',fault=fail)
            saved=torch.load(base/'interrupted/recovery.pt',weights_only=True)
            self.assertEqual(saved['updates'],1)
            self.assertTrue(all(int(v['step'])==1 for o in saved['optimizers'].values() for v in o['state'].values()))
            resumed=self.fit(base/'interrupted')
            for key in ('models','optimizers','sampler','rng','history','best','updates','completed_epoch'):
                self.assert_tree(full[key],resumed[key])

    def test_recovery_source_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp,redirect_stdout(io.StringIO()):
            self.fit(Path(temp),stop_after_updates=1)
            self.binding={'source':'wrong'}
            with self.assertRaisesRegex(ValueError,'source/config/code'):self.fit(Path(temp))

    def test_budget_order_211_and_independent_optimizer_storage(self):
        a=train.transaction.PairedSampler(107621,512,42);b=train.transaction.PairedSampler(107621,512,42)
        self.assertEqual(len(a),211);self.assertEqual(len(a)*200*5,211000)
        a.start(1);_ = self.arms();b.start(1);self.assertEqual(a.order,b.order)
        arms=self.arms();opts={n:train.optimizer(m,self.settings) for n,m in arms.items()}
        for n,m in arms.items():candidate_loss(m,self.coords,self.labels,self.preparation)[0].backward();opts[n].step()
        pointers=[{t.data_ptr() for v in o.state.values() for t in v.values() if isinstance(t,torch.Tensor)} for o in opts.values()]
        self.assertTrue(all(not a.intersection(b) for i,a in enumerate(pointers) for b in pointers[i+1:]))

    def test_no_development_test_cache_access(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError,'cannot load test'):
                prep.projected_cache(None,None,None,[],'test',{},self.device,Path(temp),{})

    def test_raw_replay_cache_selection_identical(self):
        rows=[{'sample_id':g.sample_id,'label':g.y.item(),'partition':'val'} for g in self.graphs]
        refs={n:self.ordered for n in ('joint_ordered','base_C','frozen_single')}
        data=TinyGraphs(self.graphs)
        with tempfile.TemporaryDirectory() as temp,redirect_stdout(io.StringIO()):
            records=run.raw_replay(self.source,refs,self.arms(),self.preparation,data,rows,
                {'seed':42,'eval_batch_size':16,'workers':0},self.device,Path(temp))
            for r in records:self.assertTrue(all(r['predictions'][n]==r['predictions']['source_ordered'] for n in ARMS))
            for n in ARMS:report.check_metric(report.metrics(records,n),train.measure(self.arms(),self.cache,self.settings,self.device)[1][n])
            summary=report.write_result(Path(temp),records,{},self.settings,42)
            self.assertFalse(summary['decision']['passed'])
            self.assertEqual(summary['comparisons']['cc_t4-cc_t3']['counts'],dict(repaired=0,regressed=0,wrong_changed=0))

    def test_projection_cache_integrity_and_id_binding(self):
        rows=[{'sample_id':g.sample_id,'label':g.y.item(),'partition':'val'} for g in self.graphs]
        refs={n:self.ordered for n in ('joint_ordered','base_C','frozen_single')}
        config={'seed':42,'eval_batch_size':16,'workers':0};data=TinyGraphs(self.graphs)
        with tempfile.TemporaryDirectory() as temp,redirect_stdout(io.StringIO()):
            folder=Path(temp)
            a,_=prep.projected_cache(self.source,refs,data,rows,'val',config,self.device,folder,self.binding)
            with patch.object(prep.frozen,'encode',side_effect=AssertionError('should reuse')):
                b,_=prep.projected_cache(self.source,refs,data,rows,'val',config,self.device,folder,self.binding)
            self.assert_tree(a.features,b.features)
            with self.assertRaisesRegex(ValueError,'binding/hash'):
                prep.projected_cache(self.source,refs,data,rows,'val',config,self.device,folder,{'changed':True})
            payload=folder/'projected_val.pt';payload.write_bytes(payload.read_bytes()+b'corrupt')
            with self.assertRaisesRegex(ValueError,'binding/hash'):
                prep.projected_cache(self.source,refs,data,rows,'val',config,self.device,folder,self.binding)

    def test_complete_candidate_pipeline_and_sealed_resume_on_synthetic_graphs(self):
        rows={part:[{'sample_id':g.sample_id,'label':g.y.item(),'partition':part} for g in self.graphs] for part in ('train','val')}
        refs={n:self.ordered for n in ('joint_ordered','base_C','frozen_single')}
        config={'seed':42,'eval_batch_size':16,'workers':0};data=TinyGraphs(self.graphs)
        before=prep.locked.frozen_digests(self.source,refs)
        reference=prep.frozen.original.prediction_records(rows['val'],self.coords.reference.argmax(-1).tolist())
        manifest={'seeds':{'42':{'models':{n:{'validation_predictions_path':n} for n in refs}}}}
        protocol={'synthetic':True};bind={**self.binding,'protocol_sha256':prep.archive.canonical_sha(protocol)}
        with tempfile.TemporaryDirectory() as temp,redirect_stdout(io.StringIO()):
            root=Path(temp);folder=root/'seed42';folder.mkdir()
            prep.raw.write(folder/'provenance.json',bind)
            def context(*args,**kwargs):return folder,self.source,refs,data,rows,self.device,config,{},bind,before,{'test':0}
            inputs=SimpleNamespace(rows=lambda path:reference)
            def prepared(*args,**kwargs):return self.preparation,{'val':{'repairable_disagreement_samples':1}}
            with patch.object(prep,'context',side_effect=context),patch.object(prep,'prepare',side_effect=prepared),\
                 patch.object(run,'verify_budget'):
                result=run.run_seed(root,None,root,self.settings,protocol,inputs,manifest,42)
            self.assertEqual(result['status'],'completed')
            self.assertTrue((folder/'selected_validation_replay.json').is_file())
            with patch.object(prep,'context',side_effect=AssertionError('completed phase must not reload graphs')):
                result=run.run_seed(root,None,root,self.settings,protocol,inputs,manifest,42)
            self.assertTrue(result['reused_completed_phase'])

    def test_promoted_fixed_test_replays_locked_models_without_optimizers(self):
        data=FixedTestGraphs(self.graphs);rows={p:[r for r in data.rows if r['partition']==p] for p in ('val','test')}
        refs={n:self.ordered for n in ('joint_ordered','base_C','frozen_single')}
        before=prep.locked.frozen_digests(self.source,refs);protocol={'synthetic':True}
        config={'seed':42,'eval_batch_size':16,'workers':0}
        settings={**self.settings,'test_samples':len(self.graphs)}
        names=['cc_t0','cc_t1','cc_t2'];arms={n:self.arms()[n] for n in names}
        with tempfile.TemporaryDirectory() as temp,redirect_stdout(io.StringIO()):
            root=Path(temp);folder=root/'seed42';folder.mkdir();(root/'aggregate').mkdir()
            prep.raw.write(root/'aggregate/summary.json',{'passed':True})
            val=run.raw_replay(self.source,refs,arms,self.preparation,data,rows['val'],config,self.device,folder)
            (folder/'validation').mkdir();prep.raw.save_predictions(folder/'validation/predictions.jsonl.gz',val)
            spec={}
            for n,m in arms.items():
                file=folder/n/'best.pt';train.transaction.atomic_save(file,{'state':train.transaction.cpu_state(m)})
                spec[n]={'file':str(file.relative_to(root)),'sha256':prep.raw.sha(file),'epoch':0,'validation':report.metrics(val,n)}
            preppath=folder/'preparation/common/last.pt';train.transaction.atomic_save(preppath,{'state':train.transaction.cpu_state(self.preparation)})
            prep.raw.write(root/'final_checkpoint_lock.json',{'protocol_sha256':prep.archive.canonical_sha(protocol),
                'seeds':{'42':{'models':spec,'preparation_sha256':prep.raw.sha(preppath)}}})
            expected_test=run.raw_replay(self.source,refs,arms,self.preparation,data,rows['test'],config,self.device,folder,'test')
            def historical(path):
                n=Path(path).name.replace('_predictions.jsonl.gz','');key='source_ordered' if n=='joint_ordered' else n
                return [{'sample_id':r['sample_id'],'label':r['label'],'prediction':r['predictions'][key]} for r in expected_test]
            manifest={'test_cohort_sha256':prep.archive.canonical_sha([(r['sample_id'],r['label']) for r in rows['test']])}
            context=(folder,self.source,refs,data,rows,self.device,config,{},self.binding,before,{'train':0})
            with patch.object(prep,'context',return_value=context),\
                 patch.object(run.archive,'open_archive',return_value=SimpleNamespace(rows=historical)),\
                 patch.object(run.frozen.evaluation,'evaluation_rows',return_value=rows['test']),\
                 patch.object(torch.optim,'Adam',side_effect=AssertionError('No optimizer in fixed test')),\
                 patch.object(torch.Tensor,'backward',side_effect=AssertionError('No backward in fixed test')):
                result=run.fixed_test_seed(root,None,root,settings,protocol,None,manifest,42)
                self.assertTrue(result['test_evaluated']);self.assertEqual(result['new_optimizer_updates'],0)
                with patch.object(prep,'context',side_effect=AssertionError('No repeated evaluation')):
                    result=run.fixed_test_seed(root,None,root,settings,protocol,None,manifest,42)
                    self.assertTrue(result['reused_completed_test'])

    def test_group_empty_negative_and_promotion_tie(self):
        self.assertEqual([report.group(0,f,p) for f,p in ((0,1),(1,0),(0,0),(1,1))],list(report.GROUPS))
        s={'metrics':{},'groups':{}}
        for n in (*ARMS,'source_ordered','frozen_single'):
            s['metrics'][n]={'macro_f1':.97,'per_class':{c:{'f1-score':.9} for c in self.settings['focus_classes']}}
            s['groups'][n]={g:{'fusion_destruction_rate':None if g==report.GROUPS[1] else .1} for g in report.GROUPS}
        s['metrics']['cc_t2']['macro_f1']=s['metrics']['cc_t4']['macro_f1']=.972
        d=report.decision(s,self.settings);self.assertEqual(d['candidate'],'cc_t2');self.assertTrue(d['passed'])
        s['metrics']['cc_t2']['per_class']['WebBased']['f1-score']=.896
        self.assertFalse(report.decision(s,self.settings)['passed'])
        s['metrics']['cc_t2']['per_class']['WebBased']['f1-score']=.9
        s['groups']['cc_t2'][report.GROUPS[0]]['fusion_destruction_rate']=.11
        self.assertFalse(report.decision(s,self.settings)['passed'])

    def test_queue_stops_failed_gate_and_replicates_both_before_test(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'seed42').mkdir();(root/'aggregate').mkdir()
            prep.raw.write(root/'seed42/summary.json',{'decision':{'passed':False}})
            self.assertIsNone(launch.next_phase(root,['seed42']))
            prep.raw.write(root/'seed42/summary.json',{'decision':{'passed':True}})
            self.assertEqual(launch.next_phase(root,['seed42']),'prepare43')
            self.assertEqual(launch.next_phase(root,['seed43']),'prepare44')
            prep.raw.write(root/'aggregate/summary.json',{'passed':False})
            self.assertIsNone(launch.next_phase(root,['aggregate']))
            prep.raw.write(root/'aggregate/summary.json',{'passed':True})
            self.assertEqual(launch.next_phase(root,['aggregate']),'lock_test')
            with self.assertRaisesRegex(ValueError,'Replication gate failed'):
                with patch.object(run,'decision',return_value={'candidate':'cc_t2'}),patch.object(run.raw,'read',return_value={'passed':False}):
                    run.test_lock(root,self.settings,{})

    def test_launcher_publishes_failure_and_preserves_child_exit(self):
        with tempfile.TemporaryDirectory() as temp,redirect_stdout(io.StringIO()):
            repo=Path(temp);args=SimpleNamespace(repo=repo,data_root=None,publish_remote=None,
                lock_only=False,probe_only=False,resume='auto')
            child=SimpleNamespace(wait=lambda:7,poll=lambda:7)
            def uploaded(*a,**kw):
                folder=a[2];(folder/'upload.log').write_text(json.dumps({'status':'uploaded','commit':'a'*40})+'\n')
                return [0,0]
            with patch.object(launch,'lfs_environment',return_value={'PATH':'test'}),\
                 patch.object(launch.subprocess,'Popen',return_value=child),\
                 patch.object(launch.publisher,'publish_results',side_effect=uploaded) as publish:
                self.assertEqual(launch.launch(args),7)
                self.assertEqual(publish.call_count,1)
                self.assertEqual(publish.call_args.args[3],7)
                self.assertEqual(json.loads((repo/'outputs/conditional_complement_v1/training_exit.json').read_text())['exit_code'],7)

    def test_upload_failure_does_not_relabel_successful_training_exit(self):
        with tempfile.TemporaryDirectory() as temp,redirect_stdout(io.StringIO()):
            repo=Path(temp);args=SimpleNamespace(repo=repo,data_root=None,publish_remote=None,
                lock_only=True,probe_only=False,resume='auto')
            child=SimpleNamespace(wait=lambda:0,poll=lambda:0)
            with patch.object(launch,'lfs_environment',return_value={'PATH':'test'}),\
                 patch.object(launch.subprocess,'Popen',return_value=child),\
                 patch.object(launch.publisher,'publish_results',return_value=[1,1]):
                self.assertEqual(launch.launch(args),1)
                receipt=json.loads((repo/'outputs/conditional_complement_v1/training_exit.json').read_text())
                self.assertEqual(receipt['exit_code'],0)
                self.assertEqual(receipt['queue_exit_code'],1)

    def test_aggregate_uses_paired_sample_std_and_retains_negative_replicate(self):
        with tempfile.TemporaryDirectory() as temp,redirect_stdout(io.StringIO()):
            folder=Path(temp)
            rows=[{'sample_id':g.sample_id,'label':g.y.item(),'partition':'val'} for g in self.graphs]
            records=prep.observe({'source_ordered':self.arms()['cc_t0'],**self.arms()},self.preparation,self.coords,rows,
                {n:self.coords.reference.argmax(-1).tolist() for n in ('joint_ordered','base_C','frozen_single')})
            summary=report.write_result(folder,records,{},self.settings,42)
            summaries={s:deepcopy(summary) for s in (42,43,44)}
            for s,delta in ((42,.002),(43,.001),(44,-.001)):
                summaries[s]['metrics']['cc_t2']['macro_f1']+=delta
            result=report.aggregate(summaries,self.settings,'cc_t2',folder/'aggregate')
            self.assertFalse(result['passed'])
            values=result['all_three']['paired_macro_f1_pp']['cc_t2-cc_t0']['values']
            self.assertLess(values[-1],0)
            self.assertAlmostEqual(result['new_replications_only']['paired_macro_f1_pp']['cc_t2-cc_t0']['mean'],0.)

    def test_immutable_archive_audit_recomputes_all_groups_pairs_and_aggregates(self):
        with tempfile.TemporaryDirectory() as temp,redirect_stdout(io.StringIO()):
            root=Path(temp);prep.raw.write(root/'config.json',self.settings)
            rows=[{'sample_id':g.sample_id,'label':g.y.item(),'partition':'val'} for g in self.graphs]
            records=prep.observe({'source_ordered':self.arms()['cc_t0'],**self.arms()},self.preparation,self.coords,rows,
                {n:self.coords.reference.argmax(-1).tolist() for n in ('joint_ordered','base_C','frozen_single')})
            summaries={}
            for seed in (42,43,44):
                folder=root/('seed'+str(seed));folder.mkdir()
                summaries[seed]=report.write_result(folder,records,{},self.settings,seed)
            report.aggregate(summaries,self.settings,'cc_t2',root/'aggregate')
            prefix='outputs/conditional_complement_v1/'
            payloads={prefix+p.relative_to(root).as_posix():p.read_bytes() for p in root.rglob('*') if p.is_file()}
            def git(args,**kw):
                if args[1]=='ls-tree':return '\n'.join(payloads)+'\n'
                if args[1]=='show':return payloads[args[2].split(':',1)[1]]
                raise AssertionError(args)
            with patch.object(audit.subprocess,'run'),patch.object(audit.subprocess,'check_output',side_effect=git),\
                 patch.object(audit.archive,'lfs_environment',return_value={}):
                receipt=audit.audit(root,'a'*40)
                self.assertEqual(len(receipt['published_prediction_audits']),3)
                self.assertEqual(len(receipt['published_aggregate_audits']),1)
                name=prefix+'seed43/summary.json';corrupt=json.loads(payloads[name])
                corrupt['comparisons']['cc_t2-cc_t1']['counts']['repaired']+=1
                payloads[name]=json.dumps(corrupt).encode()
                with self.assertRaisesRegex(ValueError,'pair counts differ'):audit.audit(root,'a'*40)


if __name__=='__main__':unittest.main()
