"""CPU checks for independent source locks, matched frozen/joint arms and queue recovery."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import math
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch_geometric.data import Batch

import run_fixed_transition_stability as run
import launch_fixed_transition_stability as launcher
from test_packet_transition_readout import fixture
from test_flow_query_joint_packet2 import TinyGraphs
from test_joint_transition_fusion import JointFusionChecks


class StabilityChecks(unittest.TestCase):
    assert_tree_equal=JointFusionChecks.assert_tree_equal

    def setUp(self):
        torch.set_num_threads(2);self.device=torch.device('cpu')
        self.settings={**run.raw.read(run.CONFIG),'epochs':3,'batch_size':3,'eval_batch_size':4,
            'training_samples':8,'validation_samples':8,'test_samples':8,'updates_per_epoch':3,'updates_per_arm':9}
        self.source,self.graphs=fixture();run.frozen.independent.freeze_c(self.source)
        self.rows=[{'sample_id':g.sample_id,'label':g.y.item(),'partition':'val'} for g in self.graphs]
        self.features,_=run.frozen.encode(self.source,Batch.from_data_list(self.graphs),self.device)
        self.cache=run.original.FrozenCache([self.features],[r['label'] for r in self.rows])
        self.reference=run.original.prediction_records(self.rows,run.original.baseline(self.source,self.features).argmax(-1).tolist())
        self.metric=run.frozen.joint.study.validate_predictions(self.reference,self.rows,False)
        self.binding={'source_C':run.raw.state_digest(self.source.state_dict())}

    def arms(self): return run.training.make_arms(self.source,self.settings,self.device)

    def fit(self,path,resume=False,stop=None):
        return run.training.fit(self.arms(),self.cache,self.cache,self.rows,self.reference,self.settings,
                                self.binding,self.device,path,resume=resume,stop_after_updates=stop)

    def equal_recovery(self,a,b):
        self.assert_tree_equal({k:v for k,v in a.items() if k!='training_seconds'},
                               {k:v for k,v in b.items() if k!='training_seconds'})

    def source_binding(self,seed):
        return {'backbone_seed':seed,'frozen_c_sha256':'C'+str(seed),'split_id':'split','manifest_sha256':'manifest',
                'validation_digest':'digest'+str(seed)}

    def test_protocol_locks_six_runs_and_prevents_seed42_reselection_or_seed_drift(self):
        settings=run.raw.read(run.CONFIG);run.validate_settings(settings)
        for key,value in (('epochs',100),('readout_seed',43),('sampling_seed',43),('new_training_runs',9),
                          ('fixed_injection_strength',0.5),('total_new_updates',126600)):
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'protocol changed'):
                run.validate_settings({**settings,key:value})
        changed=deepcopy(settings);changed['historical_seed42']['joint_ordered']['selected_epoch']=66
        with self.assertRaisesRegex(ValueError,'seed42 result changed'):run.validate_settings(changed)
        self.assertEqual(6*settings['updates_per_arm'],253200)

    def test_initialization_rng_no_beta_independent_storage_and_optimizer_scope(self):
        rng=run.training.parent.rng_state();arms=self.arms()
        self.assert_tree_equal(rng,run.training.parent.rng_state())
        for name,arm in arms.items():
            out=arm(self.features)
            self.assertTrue(torch.equal(out[0],run.original.baseline(self.source,self.features)))
            self.assertFalse(bool(out[1].any()));self.assertFalse(hasattr(arm,'injection_gate'))
            opt=run.training.optimizer(arm,self.settings)
            self.assertEqual([g['lr'] for g in opt.param_groups],[.001] if name=='frozen_single' else [.0001,.001])
            arm.train()
            if name=='frozen_single':
                self.assertTrue(all(not m.training for m in arm.fusion.modules()))
                self.assertTrue(all(not p.requires_grad for p in arm.fusion.parameters()))
        for name in ('frozen_single','joint_single'):
            for key,value in arms[name].readout.state_dict().items():
                self.assertTrue(torch.equal(value,arms['joint_ordered'].readout.state_dict()[key]))

    def test_nonzero_models_match_existing_single_ordered_implementations(self):
        for name,arm in self.arms().items():
            reference=run.training.parent.JointTransitionFusion(self.source.fusion,
                'joint_ordered' if name=='joint_ordered' else 'joint_single',42)
            with torch.no_grad():reference.readout.output.weight.normal_(std=.1)
            arm.load_state_dict(reference.state_dict(),strict=True)
            for a,b in zip(arm(self.features),reference(self.features)):
                self.assertTrue(torch.equal(a,b))

    def test_staged_gradients_frozen_fusion_and_source_parameters_buffers_remain_unchanged(self):
        before=run.raw.state_digest(self.source.state_dict());rng=run.training.parent.rng_state()
        probe=run.training.gradient_probe(self.source,self.features,self.cache.labels,self.settings,self.device)
        self.assertTrue(probe['passed']);run.training.parent.verify_source(self.source,before)
        self.assert_tree_equal(rng,run.training.parent.rng_state())

    def test_mid_epoch_boundary_and_completed_resume_preserve_weights_adam_rng_and_sampler(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            root=Path(tmp);run.raw.seed_all(42);expected=self.fit(root/'continuous')
            for stop in (1,3,5):
                output=root/str(stop);run.raw.seed_all(42);self.fit(output,stop=stop)
                random.random();np.random.rand(10);torch.rand(9)
                resumed=self.fit(output,resume=True);self.equal_recovery(expected,resumed)
                self.equal_recovery(resumed,self.fit(output,resume=True))
                self.assertTrue(all(v>0 for v in resumed['training_seconds'].values()))
                self.assertTrue(run.verify_training_state(resumed,self.settings,
                    run.raw.state_digest(self.source.fusion.state_dict()))['passed'])
                for arm in resumed['optimizers'].values():
                    self.assertTrue(all(v['step'].item()==9 for v in arm['state'].values()))

    def test_failure_inside_three_arm_transaction_leaves_previous_commit_recoverable(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            root=Path(tmp);run.raw.seed_all(42);expected=self.fit(root/'continuous')
            run.raw.seed_all(42);self.fit(root/'failure',stop=1);sha=run.raw.sha(root/'failure/recovery.pt')
            factory=run.training.optimizer;made=[]
            def failing(arm,settings):
                opt=factory(arm,settings);made.append(opt)
                if len(made)==3:
                    def fail(*args,**kwargs):raise RuntimeError('simulated third arm failure')
                    opt.step=fail
                return opt
            with patch.object(run.training,'optimizer',side_effect=failing):
                with self.assertRaisesRegex(RuntimeError,'third arm'):self.fit(root/'failure',resume=True)
            self.assertEqual(sha,run.raw.sha(root/'failure/recovery.pt'))
            self.equal_recovery(expected,self.fit(root/'failure',resume=True))

    def test_changed_source_sampling_or_configuration_cannot_resume(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            output=Path(tmp);self.fit(output,stop=1)
            for settings,binding in (({**self.settings,'sampling_seed':43},self.binding),
                                     (self.settings,{'source_C':'different'})):
                with self.assertRaisesRegex(ValueError,'mismatch'):
                    run.training.fit(self.arms(),self.cache,self.cache,self.rows,self.reference,settings,
                                     binding,self.device,output,resume=True)

    def test_independent_source_identity_rejects_copied_or_renamed_seed42(self):
        settings=run.raw.read(run.CONFIG);spec=settings['independent_sources']['43']
        expected=spec['C_selection']
        binding={'backbone_seed':43,'frozen_c_sha256':spec['frozen_c_sha256'],'split_id':settings['split_id'],
                 'manifest_sha256':settings['manifest_sha256'],'validation_digest':expected['prediction_ids_sha256']}
        run.check_source_identity(43,binding,expected,spec,settings)
        for change in ({'backbone_seed':42},{'frozen_c_sha256':'seed42_copy'},{'split_id':'other'}):
            with self.assertRaisesRegex(ValueError,'identity mismatch'):
                run.check_source_identity(43,{**binding,**change},expected,spec,settings)

    def test_n0_lineage_mismatch_is_rejected_even_when_files_match_registered_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo=Path(tmp);folder=repo/'source';folder.mkdir();n0=repo/'n0';n0.mkdir()
            run.raw.write(folder/'selection.json',{'seed':43,'n0_best_sha256':'wrong-parent'})
            (n0/'best.pt').write_bytes(b'locked-parent')
            spec={'source_directory':'source','seed':43,
                'inputs':{'selection.json':{'sha256':run.raw.sha(folder/'selection.json')}},
                'n0':{'directory':'n0','sha256':{'best.pt':run.raw.sha(n0/'best.pt')}}}
            with self.assertRaisesRegex(ValueError,'N0/C source chain changed'):run.check_source_files(repo,spec)

    def make_summary(self,output,seed=43):
        state=self.fit(output)
        records=run.selected_raw_replay(self.source,self.arms(),TinyGraphs(self.graphs),self.rows,
            {'seed':42,'workers':0,'eval_batch_size':4},self.device,output,state)
        counts=run.archive.parameter_counts(self.source,self.settings)
        metadata={'base_C':{**counts['base_C'],'best_epoch':48,'last_epoch':48,'best_validation':self.metric,
            'last_validation':self.metric,'new_updates':0,'training_seconds':0}}
        for name in run.ARMS:
            metadata[name]={**counts[name],'best_epoch':state['best'][name]['epoch'],'last_epoch':3,
                'best_validation':state['best'][name]['validation'],'last_validation':state['history'][-1]['validation'][name],
                'new_updates':9,'training_seconds':state['training_seconds'][name]}
        histories={name:[{**r,'validation':r['validation'][name]} for r in state['history']] for name in run.ARMS}
        result=run.results.write_seed_results(output,records,self.rows,metadata,self.source_binding(seed),
            {**self.settings,'epochs':200},histories,'synthetic_check',seed)
        return result,records

    def test_best_raw_replay_independent_prediction_audit_and_class_change_exports(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            output=Path(tmp);summary,records=self.make_summary(output)
            self.assertEqual(len(summary['metrics']),4);self.assertEqual(len(summary['comparisons']),4)
            self.assertTrue(run.raw.read(output/'independent_verification.json')['passed'])
            self.assertEqual(len((output/'validation/per_class.csv').read_text().splitlines()),33)
            self.assertEqual(len(run.raw.read(output/'budget_comparison.json')['budgets']),4)
            changed=deepcopy(summary['metrics']);changed['joint_ordered']['confusion_matrix'][0][0]+=1
            with self.assertRaisesRegex(ValueError,'confusion audit'):
                run.results.independently_verify(records,changed,summary['comparisons'])

    def test_aggregate_reports_new_two_separately_sample_std_and_negative_strong_control(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            root=Path(tmp);base,_=self.make_summary(root/'seed')
            summaries={seed:deepcopy(base) for seed in (42,43,44)}
            for seed,item in summaries.items():
                item['source_binding']=self.source_binding(seed);item['backbone_seed']=seed
                item['metrics']['base_C']['macro_f1']=.95
                item['metrics']['frozen_single']['macro_f1']=.96
                item['metrics']['joint_single']['macro_f1']=.959
                item['metrics']['joint_ordered']['macro_f1']={42:.962,43:.961,44:.9595}[seed]
            result=run.results.aggregate(summaries,root/'aggregate')
            self.assertEqual(result['new_replications_only']['backbone_seeds'],[43,44])
            whole=result['all_three']['paired_macro_f1_differences_pp']['joint_ordered-frozen_single']
            self.assertFalse(whole['all_positive']);self.assertEqual(whole['negative_seeds'],[44])
            new=result['new_replications_only']['paired_macro_f1_differences_pp']['joint_ordered-frozen_single']
            self.assertAlmostEqual(new['sample_std'],abs(new['values'][0]-new['values'][1])/math.sqrt(2),places=14)
            self.assertIn('强 Frozen Single',(root/'aggregate/report.md').read_text(encoding='utf-8'))
            with self.assertRaisesRegex(ValueError,'All three'):
                run.results.aggregate({43:summaries[43],44:summaries[44]},root/'incomplete')

    def test_sealed_completed_results_reject_modified_predictions(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            output=Path(tmp);self.make_summary(output)
            run.raw.write(output/'status.json',{'status':'completed'});run.raw.write(output/'final_verification.json',{'passed':True})
            run.archive.seal(output);run.archive.verify_sealed(output)
            (output/'joint_single/best_validation_predictions.jsonl.gz').write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError,'artifact changed'):run.archive.verify_sealed(output)

    def test_launcher_runs_all_six_without_score_based_stop_and_publishes_each_backbone(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            repo=Path(tmp);args=SimpleNamespace(repo=repo,data_root=None,import_only=False,probe_only=False,publish_remote=None)
            called=[];publications=[]
            def popen(command,**kwargs):
                called.append(command[command.index('--phase')+1]);return SimpleNamespace(wait=lambda:0)
            def publish(code,repo,output,status,env,**kwargs):
                publications.append(called[-1]);self.assertEqual(kwargs['output_name'],launcher.OUTPUT_NAME)
                for file in output.rglob('*'):
                    if file.is_file():file.read_bytes()
                return [0,0]
            with patch.object(launcher.subprocess,'Popen',side_effect=popen),patch.object(launcher.publisher,'publish_results',side_effect=publish):
                self.assertEqual(launcher.launch(args),0)
            self.assertEqual(called,['import','seed43','seed44','aggregate'])
            self.assertEqual(publications,['seed43','seed44','aggregate'])
            self.assertEqual(launcher.phases(import_only=True),('import',))

    def test_launcher_caught_failure_uploads_evidence_and_preserves_training_exit(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            repo=Path(tmp);args=SimpleNamespace(repo=repo,data_root=None,import_only=False,probe_only=False,publish_remote=None)
            codes=iter((0,7));called=[]
            def popen(command,**kwargs):
                called.append(command[command.index('--phase')+1]);code=next(codes)
                return SimpleNamespace(wait=lambda:code)
            with patch.object(launcher.subprocess,'Popen',side_effect=popen),patch.object(launcher.publisher,'publish_results',return_value=[0,0]) as pub:
                self.assertEqual(launcher.launch(args),7)
            self.assertEqual(called,['import','seed43']);pub.assert_called_once()
            self.assertEqual(run.raw.read(repo/'outputs'/launcher.OUTPUT_NAME/'training_exit.json')['exit_code'],7)

    def test_upload_failure_does_not_skip_seed44_and_publication_history_survives_restart(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            repo=Path(tmp);args=SimpleNamespace(repo=repo,data_root=None,import_only=False,probe_only=False,publish_remote=None)
            called=[];statuses=iter(([1,1],[0,0],[0,0]))
            def popen(command,**kwargs):
                called.append(command[command.index('--phase')+1]);return SimpleNamespace(wait=lambda:0)
            def publish(code,repo,output,status,env,**kwargs):
                value=next(statuses);(output/'upload.log').write_text('failure retained' if value[0] else 'uploaded')
                run.raw.write(output/'exit_codes.json',{'training_exit_code':status,'primary_upload_exit_code':value[0]})
                return value
            with patch.object(launcher.subprocess,'Popen',side_effect=popen),patch.object(launcher.publisher,'publish_results',side_effect=publish):
                self.assertEqual(launcher.launch(args),0)
            output=repo/'outputs'/launcher.OUTPUT_NAME
            self.assertEqual(called,['import','seed43','seed44','aggregate'])
            self.assertEqual(run.raw.read(output/'backbone_uploads.json')['uploads'][0]['upload_attempt_exit_codes'],[1,1])
            self.assertTrue(any('failure retained' in p.read_text() for p in (output/'publication_history').glob('*upload.log')))
            args.import_only=True
            with patch.object(launcher.subprocess,'Popen',side_effect=popen),patch.object(launcher.publisher,'publish_results',return_value=[0,0]):
                self.assertEqual(launcher.launch(args),0)
            self.assertEqual(len(run.raw.read(output/'backbone_uploads.json')['uploads']),2)

    def test_aggregation_rejects_a_completed_import_with_another_protocol_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo=Path(tmp);root=repo/'experiment';folder=root/'seed42_import';folder.mkdir(parents=True)
            (folder/'artifact_manifest.json').write_text('{}')
            run.raw.write(folder/'provenance.json',{'protocol_lock_sha256':'different-protocol'})
            settings=run.raw.read(run.CONFIG)
            with patch.object(run.archive,'lock_protocol',return_value={'code_commit':'current'}):
                with self.assertRaisesRegex(ValueError,'Imported protocol differs'):
                    run.run_phase(repo,None,root,settings,'aggregate')

    def test_seed_runner_probe_train_raw_replay_and_completed_skip_on_synthetic_graphs(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            repo=Path(tmp);root=repo/'experiment';root.mkdir()
            data=TinyGraphs(self.graphs);data.root=repo/'graphs';data.root.mkdir()
            rows={'val':self.rows,'train':[{**r,'partition':'train'} for r in self.rows]}
            data.rows=rows['train']+rows['val'];data.metadata={'partition_sizes':{'train':8,'val':8,'test':8}}
            binding=self.source_binding(43);binding['frozen_c_sha256']=run.raw.state_digest(self.source.state_dict())
            selected={'epoch':48,'validation':self.metric,'checkpoint_sha256':'ck','predictions_sha256':'pred',
                      'prediction_ids_sha256':run.frozen.joint.prediction_digest(self.reference)}
            binding['validation_digest']=selected['prediction_ids_sha256']
            settings=deepcopy(self.settings);settings.update(split_id='split',manifest_sha256='manifest')
            spec=settings['independent_sources']['43'];spec.update(frozen_c_sha256=binding['frozen_c_sha256'],C_selection=selected)
            source_folder=repo/spec['source_directory']/'C';source_folder.mkdir(parents=True);(source_folder/'best.pt').write_bytes(b'fixture')
            config={'seed':43,'workers':0,'batch_size':3,'eval_batch_size':4}
            gold=[{**r,'flow_weights':[.5]*32} for r in self.reference]
            source=(self.source,data,rows,self.device,config,selected,gold,{},binding)
            lock={'code_commit':'synthetic'}
            with patch.object(run,'check_source_files',return_value={}),patch.object(run.frozen,'load_source',return_value=source) as loader:
                probe=run.run_seed(repo,None,root,settings,lock,43,probe_only=True)
                self.assertFalse(probe['formal_training_started']);self.assertFalse((root/'seed43/recovery.pt').exists())
                complete=run.run_seed(repo,None,root,settings,lock,43,resume=True)
                self.assertEqual(complete['status'],'completed')
                state=torch.load(root/'seed43/recovery.pt',map_location='cpu',weights_only=True)
                self.assertEqual(state['updates'],9)
                old_calls=loader.call_count
                skipped=run.run_seed(repo,None,root,settings,lock,43,resume=True)
                self.assertTrue(skipped['reused_completed_training']);self.assertEqual(loader.call_count,old_calls)
                with self.assertRaisesRegex(ValueError,'graph root differs'):
                    run.run_seed(repo,repo/'another_graph_root',root,settings,lock,43,resume=True)
            self.assertTrue(run.archive.verify_sealed(root/'seed43')['independent_verification']['passed'])


if __name__=='__main__':unittest.main()
