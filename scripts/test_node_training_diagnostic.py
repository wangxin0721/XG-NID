"""CPU graph checks for frozen-original gradients, matched budgets and diagnostic audits."""
from copy import deepcopy
from contextlib import redirect_stdout
import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import node_training_diagnostic as d
import run_node_training_diagnostic as run
import audit_node_training_diagnostic as audit
import launch_node_training_diagnostic as launcher
from test_flow_packet_node_modulation import Checks as Fixture


class DiagnosticChecks(unittest.TestCase):
    def setUp(self):
        fixture = Fixture()
        fixture.setUp()
        self.source, self.ordered, self.data, self.rows = fixture.source, fixture.ordered, fixture.data, fixture.rows
        self.device = torch.device('cpu')
        self.settings = {**d.raw.read(run.CONFIG), 'workers': 0, 'batch_size': 8, 'eval_batch_size': 4,
                         'training_samples': 8, 'validation_samples': 8, 'diagnostic_samples_per_class': 1,
                         'updates_per_epoch': 1, 'updates_per_arm': 20, 'total_updates': 60}

    def models(self):
        return d.make_models(self.source, self.ordered, self.settings, self.device)

    def test_exact_scopes_rates_and_frozen_GNN_input_gradients(self):
        models, _ = self.models()
        self.assertEqual([g['lr'] for g in d.optimizer(models['R1'], self.settings).param_groups],
                         [1e-7, 1e-5, 1e-5, 1e-5, 1e-4])
        for n in ('R2','R3'):
            trainable = [(k,p) for k,p in models[n].named_parameters() if p.requires_grad]
            self.assertTrue(all(k.startswith('adjustment.') for k,p in trainable))
            self.assertEqual(sum(p.numel() for k,p in trainable), 3008)
            self.assertEqual(d.optimizer(models[n],self.settings).param_groups[0]['lr'],1e-4)
        with tempfile.TemporaryDirectory(dir=d.raw.ROOT/'tmp') as tmp:
            before = {n:d.raw.state_digest(m.state_dict()) for n,m in models.items()}
            d.gradient_probe(models,self.data,self.rows['train'],self.settings,self.device,Path(tmp))
            self.assertEqual(before,{n:d.raw.state_digest(m.state_dict()) for n,m in models.items()})
            self.assertTrue(all(v['passed'] for v in d.raw.read(Path(tmp)/'gradient_probe.json').values()))

    def test_subset_all_classes_reproducible_and_test_prohibited(self):
        a = d.diagnostic_subset(self.rows['train'],self.settings)
        self.assertEqual(a,d.diagnostic_subset(list(reversed(self.rows['train'])),self.settings))
        self.assertEqual(set(r['label'] for r in a),set(range(8)))
        models, reference = self.models()
        with self.assertRaises(ValueError):
            d.evaluate(models,reference,self.data,[{**a[0],'partition':'test'}],self.settings,self.device)

    def test_NLL_counts_and_gate_are_recomputed_from_exports(self):
        records=[{'sample_id':str(i),'label':i,'predictions':{'Ordered':i,'R3':(i+1)%8},
                  'true_log_probability':{'Ordered':-.1,'R3':-.2}} for i in range(8)]
        m=d.record_metrics(records,'R3')
        self.assertEqual(m['nll'],.2)
        self.assertEqual(m['source_comparison']['counts']['regressed'],8)
        best={n:{'macro_f1':.9} for n in ('Ordered','R1','R2','R3')}
        self.assertFalse(d.decision(best,self.settings)['extension_gate_passed'])
        best['R3']['macro_f1']=.9006
        self.assertTrue(d.decision(best,self.settings)['extension_gate_passed'])
        best['R2']['macro_f1']=.9005
        self.assertFalse(d.decision(best,self.settings)['extension_gate_passed'])

    def test_partial_batch_resume_restores_custom_optimizers_and_observations(self):
        models, reference=self.models()
        settings={**self.settings,'epochs':2,'batch_size':3}
        initial={n:d.training.transaction.cpu_state(m) for n,m in models.items()}
        def measure(current,epoch):
            rows,metrics=d.evaluate(current,reference,self.data,self.rows['val'],settings,self.device)
            for n in metrics:
                if n!='Ordered': metrics[n]['parameter_delta']=d.parameter_delta(current[n],initial[n])
            return rows,{n:metrics[n] for n in current}
        with tempfile.TemporaryDirectory(dir=d.raw.ROOT/'tmp') as tmp, redirect_stdout(io.StringIO()):
            root=Path(tmp)
            d.raw.seed_all(99)
            full=d.training.fit(deepcopy(models),self.data,self.rows,settings,{},self.device,root/'full',
                                optimizer_fn=d.optimizer,evaluate_fn=measure)
            d.raw.seed_all(99)
            d.training.fit(deepcopy(models),self.data,self.rows,settings,{},self.device,root/'resume',
                           optimizer_fn=d.optimizer,evaluate_fn=measure,stop_after_updates=1)
            def fault(n,step):
                if n=='R2' and step==2: raise RuntimeError('partial')
            with self.assertRaisesRegex(RuntimeError,'partial'):
                d.training.fit(deepcopy(models),self.data,self.rows,settings,{},self.device,root/'resume',
                               optimizer_fn=d.optimizer,evaluate_fn=measure,fault=fault)
            resumed=d.training.fit(deepcopy(models),self.data,self.rows,settings,{},self.device,root/'resume',
                                   optimizer_fn=d.optimizer,evaluate_fn=measure)
            fixture=Fixture()
            for key in ('models','optimizers','sampler','rng','history','loss_sums','best'):
                fixture.tree_equal(full[key],resumed[key])

    def test_full_twenty_epoch_pipeline_seal_and_export_audit_rejects_tampering(self):
        models,reference=self.models()
        gold,_=d.evaluate({},reference,self.data,self.rows['val'],self.settings,self.device)
        gold=[{'sample_id':r['sample_id'],'label':r['label'],'prediction':r['predictions']['Ordered']} for r in gold]
        class Inputs:
            def rows(self,name): return gold
        refs={'joint_ordered':self.ordered}
        before=run.parent.sources.locked.frozen_digests(self.source,refs)
        manifest={'seeds':{'42':{'models':{'joint_ordered':{'validation_predictions_path':'gold'}}}}}
        with tempfile.TemporaryDirectory(dir=d.raw.ROOT/'tmp') as tmp, redirect_stdout(io.StringIO()):
            root=Path(tmp)
            protocol={'settings':self.settings,'validation_cohort_sha256':d.archive.canonical_sha([(r['sample_id'],r['label']) for r in gold])}
            def lock(*args):
                d.raw.write(root/'config.json',self.settings); d.raw.write(root/'protocol_lock.json',protocol)
                (root/'historical').mkdir(exist_ok=True)
                old=d.raw.read(d.raw.ROOT/'docs/node_training_diagnostic_v1/archive_trajectory/trajectory.json')
                d.raw.write(root/'historical/trajectory.json',old)
                return self.settings,protocol,{}, {},Inputs(),manifest
            context=(root,self.source,refs,self.data,self.rows,self.device,{}, {},{},before,{'test':0})
            with patch.object(run,'lock',lock),patch.object(run.parent.sources,'context',return_value=context):
                run.run(root,None,root)
                d.archive.verify_sealed(root)
                run.run(root,None,root)
            class Archive:
                def json(self,name): return d.raw.read(root/name)
                def rows(self,name):
                    with gzip.open(root/name,'rt',encoding='utf8') as f: return [json.loads(l) for l in f]
            self.assertTrue(audit.verify(Archive())['completed_training_audited'])
            history=d.raw.read(root/'history.json')
            self.assertEqual(len(history),21)
            for row in history:
                for n in ('R2','R3'):
                    self.assertTrue(all(v['absolute_l2']==0 for k,v in row['validation'][n]['parameter_delta'].items() if k!='adjustment'))
            history[1]['validation']['R3']['nll']+=.1
            d.raw.write(root/'history.json',history)
            with self.assertRaisesRegex(ValueError,'NLL'):
                audit.verify(Archive())

    def test_launcher_failure_publishes_recovery_and_keeps_training_exit(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory(dir=d.raw.ROOT/'tmp') as tmp:
            args=SimpleNamespace(repo=Path(tmp),resume='auto',data_root=None,lock_only=False,probe_only=False,publish_remote=None)
            child=SimpleNamespace(wait=lambda:7,poll=lambda:7)
            with patch.object(launcher,'lfs_environment',return_value={'PATH':''}), \
                 patch.object(launcher.subprocess,'Popen',return_value=child), \
                 patch.object(launcher.publisher,'publish_results',return_value=[0,0]) as publish, \
                 patch.object(launcher,'uploaded_commit',return_value='a'*40):
                self.assertEqual(launcher.launch(args),7)
                self.assertEqual(publish.call_args.kwargs['output_name'],run.OUTPUT)


if __name__=='__main__': unittest.main()
