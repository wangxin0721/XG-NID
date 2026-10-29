"""Repeat controls, seed mapping and paired statistics; no torch/graph loading."""
import ast
import copy
import json
from pathlib import Path
import random
import shutil
import statistics
import sys
import tempfile
from types import SimpleNamespace
import unittest

import run_fusion_seed_repeats as repeat

ROOT=Path(__file__).resolve().parents[1]


class RepeatChecks(unittest.TestCase):
    def test_import_py39_and_six_jobs(self):
        self.assertNotIn('torch',sys.modules)
        ast.parse((ROOT/'scripts/run_fusion_seed_repeats.py').read_text(encoding='utf-8'),feature_version=(3,9))
        study,n0,fusion=repeat.protocols(ROOT/'configs/fusion_seed_repeats_20261029.json')
        self.assertEqual(study['training_jobs'],[[43,'N0'],[43,'F1'],[43,'F2'],[44,'N0'],[44,'F1'],[44,'F2']])
        self.assertEqual((n0['lr'],fusion['lr']),(.0001,.001))
        self.assertEqual((n0['epochs'],fusion['epochs'],n0['patience'],fusion['patience']),(30,30,12,12))

    def test_sampler_changes_only_requested_seed(self):
        calls=[]
        def sampler(size,batch,seed):
            calls.append((size,batch,seed))
            result=list(range(size));random.Random(seed).shuffle(result)
            return result
        original=SimpleNamespace(TrainBatches=sampler,marker=object())
        first=repeat.SeedCore(original,43,Path('.'))
        second=repeat.SeedCore(original,44,Path('.'))
        self.assertEqual(first.TrainBatches(100,512,42),first.TrainBatches(100,512,42))
        self.assertNotEqual(first.TrainBatches(100,512,42),second.TrainBatches(100,512,42))
        self.assertIs(first.marker,original.marker)
        self.assertEqual(calls[-1],(100,512,44))
        with self.assertRaises(ValueError):first.TrainBatches(100,512,43)
        with self.assertRaises(ValueError):repeat.SeedCore(original,42,Path('.'))

    def runs(self):
        return {s:{'F0':{'macro_f1':.96},'models':{
            'F1':{'best_validation':{'macro_f1':.9601},'best_epoch':5},
            'F2':{'best_validation':{'macro_f1':x},'best_epoch':1}}}
            for s,x in [(42,.961),(43,.962),(44,.963)]}

    def test_raw_paired_means_sample_std_and_new_seeds(self):
        stats=repeat.paired_statistics(self.runs())
        self.assertAlmostEqual(stats['paired_differences']['all_three']['F2_minus_F0_pp']['mean_pp'],.2)
        self.assertAlmostEqual(stats['paired_differences']['all_three']['F2_minus_F0_pp']['sample_std_pp'],.1)
        self.assertAlmostEqual(stats['paired_differences']['new_confirmation_seeds']['F2_minus_F0_pp']['sample_std_pp'],
                               statistics.stdev([.2,.3]))
        self.assertEqual(stats['new_seed_F2_above_both_count'],2)
        self.assertFalse(stats['statistical_significance_claimed'])

    def test_epoch_zero_floor_not_counted_as_positive(self):
        runs=self.runs()
        for seed in (43,44):
            runs[seed]['models']['F2']={'best_validation':{'macro_f1':.96},'best_epoch':0}
        stats=repeat.paired_statistics(runs)
        self.assertEqual(stats['new_seed_F2_above_both_count'],0)
        self.assertTrue(stats['per_seed']['43']['selected_epoch_zero']['F2'])
        self.assertEqual(stats['numeric_screen'],'no_dynamic_advantage_prefer_N0')
        with self.assertRaises(ValueError):repeat.paired_statistics({43:runs[43],44:runs[44]})
        runs[43]['models']['F2']['best_validation']['macro_f1']=float('nan')
        with self.assertRaises(ValueError):repeat.paired_statistics(runs)

    def test_swapped_E2_seed_or_sources_rejected(self):
        registry=json.loads((ROOT/'configs/baseline_seed_repeats_20261028.json').read_text())
        source=registry['references']['E2']
        cfg={'controls':repeat.study_controls(registry,43),'experiment':'E2','epochs':100,'workers':2,
             'evaluation_batch_size':16,'binding':{'repeat_seed':43,'repeat_model':'E2',
             'split_id':registry['split_id'],'samples_sha256':'manifest','sources':source['sources'],
             'entry_sha256':source['entry_sha256']}}
        metrics={'parameter_count':846800,'test':None,'test_evaluated':False}
        summary={'models':{'E2':metrics},'binding':cfg['binding']}
        repeat.validate_parent(cfg,metrics,summary,43,registry,'manifest')
        with self.assertRaises(ValueError):repeat.validate_parent(cfg,metrics,summary,44,registry,'manifest')
        altered=copy.deepcopy(cfg);altered['binding']['sources']={}
        with self.assertRaises(ValueError):repeat.validate_parent(altered,metrics,summary,43,registry,'manifest')

    def test_seed42_snapshot_exact_matrices_and_delta(self):
        ref=json.loads((ROOT/'configs/frozen_fusion_seed42_result_20261029.json').read_text())
        repeat.verify_metrics(ref['F0'])
        for v in ('F1','F2'):repeat.verify_metrics(ref['models'][v]['best_validation'])
        self.assertAlmostEqual(100*(ref['models']['F2']['best_validation']['macro_f1']-ref['F0']['macro_f1']),
                               .057004419166484244)
        altered=copy.deepcopy(ref['F0']);altered['per_class']['WebBased']['recall']=0
        with self.assertRaises(ValueError):repeat.verify_metrics(altered)

    def test_best_trained_below_epoch_zero_is_exposed(self):
        with tempfile.TemporaryDirectory(prefix='fusion-history-') as directory:
            folder=Path(directory)
            history=[{'epoch':i,'val_macro_f1':.95,'train_samples':107621,'optimizer_updates':211,
                      'cumulative_optimizer_updates':i*211} for i in range(1,13)]
            (folder/'epochs.jsonl').write_text('\n'.join(json.dumps(r) for r in history))
            metrics={'epochs':12,'best_epoch':0,'best_validation':{'macro_f1':.96}}
            evidence=repeat.training_evidence(folder,metrics,.96)
            self.assertTrue(evidence['selected_epoch_zero'])
            self.assertEqual(evidence['best_trained_epoch'],1)
            self.assertAlmostEqual(evidence['best_trained_minus_initial_pp'],-1.)
            metrics['best_epoch']=1
            with self.assertRaises(ValueError):repeat.training_evidence(folder,metrics,.96)

    def test_complete_preflight_and_changed_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix='repeat-preflight-') as directory:
            repo=Path(directory);cfgdir=repo/'configs';cfgdir.mkdir()
            study,n0,fusion=repeat.protocols(ROOT/'configs/fusion_seed_repeats_20261029.json')
            registry=json.loads((ROOT/'configs/baseline_seed_repeats_20261028.json').read_text())
            snapshot=json.loads((ROOT/'configs/frozen_fusion_seed42_result_20261029.json').read_text())
            for name in (study['N0_protocol'],study['fusion_protocol']):shutil.copyfile(ROOT/'configs'/name,cfgdir/name)
            for name in registry['references']['E2']['sources']:
                path=repo/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fixture frozen source')
                registry['references']['E2']['sources'][name]=repeat.sha(path)
            splitdir=repo/registry['split_directory'];splitdir.mkdir(parents=True)
            (splitdir/'samples.jsonl').write_bytes(b'fixture manifest')
            manifest=repeat.sha(splitdir/'samples.jsonl')
            repeat.write(splitdir/'split.json',{'split_id':registry['split_id'],
                'partition_sizes':{'train':107621,'val':13446,'test':13448},
                'sample_manifest':'samples.jsonl','sample_manifest_sha256':manifest})
            root42=repo/study['seed42_directory'];root42.mkdir(parents=True)
            repeat.write(root42/'status.json',{'status':'completed','exit_code':0})
            for v in ('F1','F2'):
                folder=root42/v;folder.mkdir()
                metrics=snapshot['models'][v]
                for filename in ('best.pt','last.pt'):
                    (folder/filename).write_bytes((v+filename).encode())
                    metrics['checkpoint_sha256'][filename]=repeat.sha(folder/filename)
                (folder/'validation_predictions.jsonl.gz').write_bytes(b'fixture predictions')
                metrics['predictions_sha256']=repeat.sha(folder/'validation_predictions.jsonl.gz')
                repeat.write(folder/'metrics.json',metrics)
                history=[{'epoch':i,'val_macro_f1':metrics['best_validation']['macro_f1'] if i==metrics['best_epoch']
                          else snapshot['F0']['macro_f1']-.001,'train_samples':107621,'optimizer_updates':211,
                          'cumulative_optimizer_updates':i*211} for i in range(1,metrics['epochs']+1)]
                (folder/'epochs.jsonl').write_text('\n'.join(json.dumps(r) for r in history))
            repeat.write(root42/'summary.json',snapshot)
            repeat.write(cfgdir/study['seed42_result'],snapshot)
            for seed in (43,44):
                root=repo/study['baseline_directory']/f'E2_s{seed}';folder=root/'E2';folder.mkdir(parents=True)
                for p in (root,folder):repeat.write(p/'status.json',{'status':'completed','exit_code':0})
                binding={'repeat_seed':seed,'repeat_model':'E2','split_id':registry['split_id'],
                    'samples_sha256':manifest,'sources':registry['references']['E2']['sources'],
                    'entry_sha256':registry['references']['E2']['entry_sha256']}
                cfg={'controls':repeat.study_controls(registry,seed),'experiment':'E2','epochs':100,'workers':2,
                     'evaluation_batch_size':16,'binding':binding}
                repeat.write(folder/'config.json',cfg)
                (folder/'best.pt').write_bytes(f'fixture seed{seed}'.encode())
                metrics={'parameter_count':846800,'test_evaluated':False,'test':None,'best_epoch':1,
                    'best_validation':snapshot['F0'],'config_sha256':repeat.sha(folder/'config.json'),
                    'checkpoint_sha256':{'best.pt':repeat.sha(folder/'best.pt')}}
                repeat.write(folder/'metrics.json',metrics)
                repeat.write(root/'summary.json',{'models':{'E2':metrics},'binding':binding})
                study['expected_E2_macro_f1_percent_rounded4'][str(seed)]=100*snapshot['F0']['macro_f1']
            repeat.write(cfgdir/study['baseline_registry'],registry)
            repeat.write(cfgdir/'study.json',study)
            args=SimpleNamespace(repo=repo,config=cfgdir/'study.json')
            repeat.metadata(args)
            (repo/study['baseline_directory']/'E2_s44/E2/best.pt').write_bytes(b'other weights')
            with self.assertRaisesRegex(ValueError,'config/checkpoint changed'):repeat.metadata(args)


if __name__=='__main__':unittest.main()
