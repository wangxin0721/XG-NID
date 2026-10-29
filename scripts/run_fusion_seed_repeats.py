"""Six new trainings only: N0/F1/F2 for seeds43/44; immutable seed42 helpers."""
from __future__ import annotations

import argparse
import gc
import gzip
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

from e3_gate_study import (activate_gpu, checked_gpu, guard_graph_loads, identity,
                           inside, read, require, runtime, sha, write)
import run_e2_concat_pair as pair
import run_frozen_fusion as frozen_helpers
from baseline_seed_repeats import study_controls

HERE = Path(__file__).resolve().parent
SEEDS = (43,44)
JOBS = [[s,v] for s in SEEDS for v in ('N0','F1','F2')]
LABELS = ('Benign','WebBased','Spoofing','Recon','Mirai','Dos','DDos','BruteForce')


def protocols(config):
    study=read(config)
    require(study['schema']=='xgnid.fusion-seed-repeats.v1' and study['seeds']==list(SEEDS)
            and study['training_jobs']==JOBS and study['test_evaluated'] is False
            and study['automatic_followup'] is False,'Only six registered trainings allowed')
    n0=read(inside(config.parent,study['N0_protocol']))
    fusion=read(inside(config.parent,study['fusion_protocol']))
    frozen_helpers.check_spec(fusion)
    expected={'seed':42,'batch_size':512,'evaluation_batch_size':16,'workers':2,'epochs':30,
        'early_stop':True,'patience':12,'probe_batches':3,'optimizer':'Adam','lr':.0001,
        'weight_decay':0.,'include_initial_checkpoint':True,'test_evaluated':False}
    require(all(n0.get(k)==v for k,v in expected.items()),'Seed42 N0 protocol changed')
    return study,n0,fusion


def verify_metrics(validation):
    measured=frozen_helpers.matrix_report(validation['confusion_matrix'],LABELS)
    require(measured['samples']==13446,'Wrong validation count')
    for key in ('accuracy','macro_f1','weighted_f1'):
        require(abs(measured[key]-validation[key])<1e-12,'Matrix/metric mismatch')
    for label in LABELS:
        for key in ('precision','recall','f1-score','support'):
            require(abs(measured['per_class'][label][key]-validation['per_class'][label][key])<1e-10,
                    'Per-class matrix/metric mismatch')
    return measured


def validate_parent(cfg,metrics,summary,seed,registry,manifest):
    source_ref=registry['references']['E2']
    require(cfg['controls']==study_controls(registry,seed) and cfg['experiment']=='E2'
            and (cfg['epochs'],cfg['workers'],cfg['evaluation_batch_size'])==(100,2,16),'Wrong E2 protocol/seed')
    require(cfg['binding']['repeat_seed']==seed and cfg['binding']['repeat_model']=='E2'
            and cfg['binding']['split_id']==registry['split_id'] and cfg['binding']['samples_sha256']==manifest
            and cfg['binding']['sources']==source_ref['sources']
            and cfg['binding']['entry_sha256']==source_ref['entry_sha256'],'Wrong E2 binding')
    require(summary['models']['E2']==metrics and summary['binding']==cfg['binding']
            and metrics['parameter_count']==846800 and metrics['test_evaluated'] is False
            and metrics['test'] is None,'E2 reports disagree/test evaluated')


def metadata(args):
    study,n0,fusion=protocols(args.config)
    registry=read(inside(args.config.parent,study['baseline_registry']))
    source_ref=registry['references']['E2']
    for name,digest in source_ref['sources'].items():
        require(sha(inside(args.repo,name))==digest,'Frozen model/data source changed: '+name)
    splitdir=inside(args.repo,registry['split_directory']); split=read(splitdir/'split.json')
    require(split['split_id']==registry['split_id'] and split['partition_sizes']==
            {'train':107621,'val':13446,'test':13448},'Wrong split')
    manifest=sha(inside(splitdir,split['sample_manifest']))
    require(manifest==split['sample_manifest_sha256'],'Manifest changed')
    parents={}
    for seed in SEEDS:
        root=inside(args.repo,study['baseline_directory'])/f'E2_s{seed}'
        for folder in (root,root/'E2'):
            status=read(folder/'status.json')
            require((status['status'],status['exit_code'])==('completed',0),'E2 incomplete')
        cfg=read(root/'E2/config.json'); metrics=read(root/'E2/metrics.json'); summary=read(root/'summary.json')
        validate_parent(cfg,metrics,summary,seed,registry,manifest)
        require(sha(root/'E2/config.json')==metrics['config_sha256'] and
                sha(root/'E2/best.pt')==metrics['checkpoint_sha256']['best.pt'],'E2 config/checkpoint changed')
        verify_metrics(metrics['best_validation'])
        expected=study['expected_E2_macro_f1_percent_rounded4'][str(seed)]
        require(abs(100*metrics['best_validation']['macro_f1']-expected)<.00005,'Unexpected E2 starting score')
        parents[str(seed)]={'directory':root.relative_to(args.repo).as_posix(),
            'config_sha256':sha(root/'E2/config.json'),'metrics_sha256':sha(root/'E2/metrics.json'),
            'best_sha256':metrics['checkpoint_sha256']['best.pt'],'best_epoch':metrics['best_epoch'],
            'best_validation':metrics['best_validation']}
    snapshot_path=inside(args.config.parent,study['seed42_result']); snapshot=read(snapshot_path)
    require(sha(inside(args.config.parent,study['N0_protocol']))==snapshot['binding']['n0_binding']['config_sha256']
            and sha(inside(args.config.parent,study['fusion_protocol']))==snapshot['binding']['config_sha256'],
            'Protocols differ from completed seed42 screening')
    for name,digest in snapshot['binding']['sources'].items():
        require(sha(HERE/name)==digest,'Seed42 helper changed: '+name)
    root42=inside(args.repo,study['seed42_directory']); actual=read(root42/'summary.json')
    status=read(root42/'status.json')
    require((status['status'],status['exit_code'])==('completed',0) and
            all(actual[k]==v for k,v in snapshot.items()),'Seed42 reference changed')
    require(snapshot['test_evaluated'] is False and snapshot['frozen_model_unchanged'] is True,'Seed42 not accepted')
    for v in ('F1','F2'):
        require(read(root42/v/'metrics.json')==snapshot['models'][v],'Seed42 arm differs')
        for name,digest in snapshot['models'][v]['checkpoint_sha256'].items():
            require(sha(root42/v/name)==digest,'Seed42 checkpoint changed')
        require(sha(root42/v/'validation_predictions.jsonl.gz')==snapshot['models'][v]['predictions_sha256'],
                'Seed42 predictions changed')
        training_evidence(root42/v,snapshot['models'][v],snapshot['F0']['macro_f1'])
    binding={'study_sha256':sha(args.config),'N0_protocol_sha256':sha(inside(args.config.parent,study['N0_protocol'])),
        'fusion_protocol_sha256':sha(inside(args.config.parent,study['fusion_protocol'])),
        'seed42_reference_sha256':sha(snapshot_path),'parents':parents,'split_id':registry['split_id'],
        'manifest_sha256':manifest,'model_sources':source_ref['sources'],
        'helpers':{n:sha(HERE/n) for n in ('run_fusion_seed_repeats.py','run_e2_concat_pair.py',
            'run_frozen_fusion.py','e3_gate_study.py','baseline_seed_repeats.py')}}
    return study,n0,fusion,registry,snapshot,binding


class SeedCore:
    """Only map legacy literal sampler seed42 to the registered new seed.

    Model, loss, optimizer, loader and trainer code are delegated unchanged.
    No global monkeypatching or mutation of historical experiment sources.
    """
    def __init__(self,core,seed,output):
        require(seed in SEEDS,'Only seeds43/44')
        self.core,self.seed,self.output=core,seed,output
        self.phase='setup';self.epoch=0
    def __getattr__(self,name):return getattr(self.core,name)
    def TrainBatches(self,size,batch_size,legacy_seed):
        require(legacy_seed==42 and batch_size==512,'Historical sampler contract changed')
        return self.core.TrainBatches(size,batch_size,self.seed)
    def evaluate(self,*args,**kwargs):
        result=self.core.evaluate(*args,**kwargs)
        if self.phase=='N0_train' and not kwargs.get('export_rows',False):
            self.epoch+=1
            write(self.output/'progress.json',{'phase':'N0_train','seed':self.seed,'epoch':self.epoch})
        return result


def predictions(path):
    with gzip.open(path,'rt',encoding='utf-8') as stream:return [json.loads(line) for line in stream]


def training_evidence(folder,metrics,initial_score):
    history=[json.loads(line) for line in (folder/'epochs.jsonl').read_text(encoding='utf-8').splitlines()]
    require([r['epoch'] for r in history]==list(range(1,metrics['epochs']+1)) and
            1<=len(history)<=30,'Incomplete training history')
    best,best_epoch,stale=initial_score,0,0
    for i,row in enumerate(history):
        require(math.isfinite(row['val_macro_f1']) and 0<=row['val_macro_f1']<=1 and
                row['train_samples']==107621 and row['optimizer_updates']==211 and
                row['cumulative_optimizer_updates']==row['epoch']*211,'Invalid history/update accounting')
        best,stale,improved=pair.selection_state(row['val_macro_f1'],best,stale)
        if improved:best_epoch=row['epoch']
        require(i==len(history)-1 or stale<12,'Training continued beyond patience')
    require((len(history)==30 or stale==12) and best_epoch==metrics['best_epoch'] and
            abs(best-metrics['best_validation']['macro_f1'])<1e-12,'Selection/early stopping differs')
    post=max(history,key=lambda r:r['val_macro_f1'])
    return {'selected_epoch_zero':best_epoch==0,'best_trained_epoch':post['epoch'],
        'best_trained_macro_f1':post['val_macro_f1'],
        'best_trained_minus_initial_pp':100*(post['val_macro_f1']-initial_score),
        'last_trained_minus_initial_pp':100*(history[-1]['val_macro_f1']-initial_score)}


def run_seed(args,study,n0_protocol,fusion_protocol,registry,binding):
    gpu=checked_gpu(args.physical_gpu,args.allow_shared_gpu);activate_gpu(gpu)
    sys.path.insert(0,str(args.repo/'scripts'))
    import train_grouped as original
    device=runtime(original)
    core=SeedCore(original,args.seed,args.output)
    core.seed_all(args.seed)
    data=core.FrozenGraphs(inside(args.repo,registry['split_directory']),registry['split_id'])
    guard_graph_loads(data,{'train','val'})
    parent=binding['parents'][str(args.seed)]; folder=inside(args.repo,parent['directory'])/'E2'
    warm=core.Batch.from_data_list([data.load(r) for r in core.select_rows(data,'train')[:16]]).to(device)
    model=core.build_experiment('E2',warm).to(device);core.initialize(model,'E2',warm)
    saved=core.torch.load(folder/'best.pt',map_location=device,weights_only=False)
    require(saved['epoch']==parent['best_epoch'] and saved['validation']['confusion_matrix']==
            parent['best_validation']['confusion_matrix'],'Wrong E2 best payload')
    model.load_state_dict(saved['model_state'],strict=True)
    require(model.use_gate is False and sum(p.numel() for p in model.parameters())==846800,'Wrong E2 architecture')
    start_rng=pair.rng_state(core)
    loader=core.DataLoader(core.GraphRows(data,core.select_rows(data,'val')),batch_size=16,shuffle=False,
                           num_workers=2,pin_memory=True,multiprocessing_context='fork')
    initial,initial_predictions=core.evaluate(model,loader,device,export_rows=True)
    frozen_helpers.verify_order(initial_predictions,core.select_rows(data,'val'))
    require(initial['confusion_matrix']==parent['best_validation']['confusion_matrix'] and
            abs(initial['macro_f1']-parent['best_validation']['macro_f1'])<1e-10,'E2 validation not reproduced')
    pair.save_predictions(args.output/'E2_initial_predictions.jsonl.gz',initial_predictions)
    write(args.output/'E2_initial.json',initial)
    clean=core.state_hash(model.state_dict().items())
    pair.probe(core,model,data,device,start_rng,.0001)
    require(core.state_hash(model.state_dict().items())==clean,'N0 probe leaked state')
    n0_spec={**n0_protocol,'seed':args.seed}
    n0_binding={**binding,'repeat_seed':args.seed,'parent_E2':parent,'stage':'N0'}
    write(args.output/'progress.json',{'phase':'N0_train','seed':args.seed,'epoch':0})
    core.phase='N0_train'
    n0=pair.train(core,args,n0_spec,data,device,model,start_rng,initial,n0_binding,'N0',None,initial_predictions)
    core.phase='fusion'
    best=core.torch.load(args.output/'N0/best.pt',map_location=device,weights_only=False)
    # Match seed42's separately started frozen stage: seed afresh, materialize
    # the same constructor, and strictly load this seed's selected N0 weights.
    core.seed_all(args.seed)
    warm=core.Batch.from_data_list([data.load(r) for r in data.training_candidates()[:16]]).to(device)
    model=core.build_experiment('E2',warm).to(device);core.initialize(model,'E2',warm)
    model.load_state_dict(best['model_state'],strict=True)
    model.requires_grad_(False);model.eval();model.zero_grad(set_to_none=True)
    frozen_hash=core.state_hash(model.state_dict().items())
    gold=predictions(args.output/'N0/validation_predictions.jsonl.gz')
    fusion_output=args.output/'fusion';fusion_output.mkdir()
    child=SimpleNamespace(output=fusion_output)
    val,vcert=frozen_helpers.extract_cache(core,child,model,data,device,'val',gold)
    val=frozen_helpers.device_cache(core,val,device)
    f0,baseline=frozen_helpers.evaluate(core,model,None,'F0',val)
    require(f0['confusion_matrix']==n0['best_validation']['confusion_matrix'] and
            abs(f0['macro_f1']-n0['best_validation']['macro_f1'])<1e-10 and
            [r['prediction'] for r in baseline]==[r['prediction'] for r in gold],'Own-seed F0 failed')
    write(fusion_output/'F0.json',{'status':'completed','seed':args.seed,'best_validation':f0,
                                'ordered_predictions_reproduced':True,'cache':vcert,'test_evaluated':False})
    pair.save_predictions(fusion_output/'F0_validation_predictions.jsonl.gz',baseline)
    training,tcert=frozen_helpers.extract_cache(core,child,model,data,device,'train')
    caches={'val':val,'train':frozen_helpers.device_cache(core,training,device)}
    require(core.state_hash(model.state_dict().items())==frozen_hash,'Cache mutated N0')
    fusion_spec={**fusion_protocol,'seed':args.seed,
                 'reference':f'generated N0-s{args.seed}; see binding.n0_best_sha256'}
    fusion_binding={**binding,'repeat_seed':args.seed,'n0_best_sha256':n0['checkpoint_sha256']['best.pt'],
        'frozen_state_sha256':frozen_hash,'caches':{'train':tcert,'val':vcert}}
    write(fusion_output/'binding.json',fusion_binding)
    start_rng=pair.rng_state(core)
    probes={v:frozen_helpers.probe_fusion(core,fusion_spec,model,v,caches['train'],device,start_rng) for v in ('F1','F2')}
    write(fusion_output/'probe.json',{'status':'completed','variants':probes})
    results,exports={},{}
    for v in ('F1','F2'):
        results[v],exports[v]=frozen_helpers.train(core,child,fusion_spec,model,v,caches,device,start_rng,
                                                f0,baseline,fusion_binding,probes[v])
        require(core.state_hash(model.state_dict().items())==frozen_hash,'Fusion training mutated N0')
    common=min(len(results[v]['orders']) for v in ('F1','F2'))
    require(results['F1']['orders'][:common]==results['F2']['orders'][:common] and
            results['F1']['training_start_rng_sha256']==results['F2']['training_start_rng_sha256'],'Unpaired fusion')
    summary={'status':'completed','exit_code':0,'seed':args.seed,'binding':binding,'hardware':gpu,
        'N0':n0,'F0':f0,'models':results,'frozen_model_unchanged':True,'paired_common_epochs':common,
        'paired_sample_order_verified':True,'F2_minus_F1_macro_f1_pp':100*(results['F2']['best_validation']['macro_f1']-
        results['F1']['best_validation']['macro_f1']),'F2_versus_F1':pair.compare_predictions(exports['F1'],exports['F2']),
        'test_evaluated':False,'training_jobs':3,'automatic_followup':False}
    summary['training_evidence']={'N0':training_evidence(args.output/'N0',n0,initial['macro_f1']),
        **{v:training_evidence(fusion_output/v,results[v],f0['macro_f1']) for v in ('F1','F2')}}
    write(args.output/'summary.json',summary)
    del model,loader,caches
    gc.collect();core.torch.cuda.empty_cache()
    return summary


def paired_statistics(runs):
    require(set(runs)=={42,43,44},'Need all three seeds')
    rows={}
    for seed,run in runs.items():
        scores={'F0':run['F0']['macro_f1'],**{v:run['models'][v]['best_validation']['macro_f1'] for v in ('F1','F2')}}
        require(all(math.isfinite(x) and 0<=x<=1 for x in scores.values()),'Invalid metrics')
        rows[str(seed)]={'macro_f1_percent':{k:100*x for k,x in scores.items()},
            'F2_minus_F0_pp':100*(scores['F2']-scores['F0']),
            'F2_minus_F1_pp':100*(scores['F2']-scores['F1']),
            'F2_above_both':scores['F2']>scores['F0'] and scores['F2']>scores['F1'],
            'selected_epoch_zero':{v:run['models'][v]['best_epoch']==0 for v in ('F1','F2')}}
    aggregates={}
    for group,seeds in [('all_three',(42,43,44)),('new_confirmation_seeds',(43,44))]:
        aggregates[group]={key:{'mean_pp':statistics.mean(rows[str(s)][key] for s in seeds),
            'sample_std_pp':statistics.stdev(rows[str(s)][key] for s in seeds)}
            for key in ('F2_minus_F0_pp','F2_minus_F1_pp')}
    wins=sum(rows[str(s)]['F2_above_both'] for s in SEEDS)
    model_stats={v:{'mean_percent':statistics.mean(rows[str(s)]['macro_f1_percent'][v] for s in (42,43,44)),
        'sample_std_pp':statistics.stdev(rows[str(s)]['macro_f1_percent'][v] for s in (42,43,44))}
        for v in ('F0','F1','F2')}
    return {'per_seed':rows,'paired_differences':aggregates,'model_statistics':model_stats,
        'new_seed_F2_above_both_count':wins,
        'numeric_screen':('both_positive_pending_magnitude_and_class_review' if wins==2 else
                          'mixed_exploratory_prefer_N0' if wins==1 else 'no_dynamic_advantage_prefer_N0'),
        'epoch_zero_floor_is_not_evidence_of_gain':True,'statistical_significance_claimed':False,
        'seed42_was_used_for_candidate_selection':True}


def analyze(args,snapshot,binding):
    # Snapshot dictionaries remain independent of the immutable reference file.
    snapshot={**snapshot,'training_evidence':{v:training_evidence(
        inside(args.repo,read(args.config)['seed42_directory'])/v,snapshot['models'][v],snapshot['F0']['macro_f1'])
        for v in ('F1','F2')}}
    runs={42:snapshot}
    for seed in SEEDS:
        root=args.output/f'seed{seed}'; status=read(root/'status.json')
        require((status['status'],status['exit_code'])==('completed',0),'Seed not complete')
        run=read(root/'summary.json')
        require(run['binding']==binding and run['seed']==seed and run['test_evaluated'] is False,'Run binding differs')
        require(run['paired_sample_order_verified'] and run['frozen_model_unchanged'],'Seed checks not accepted')
        for name,metrics,folder in [('N0',run['N0'],root/'N0')]+[
            (v,run['models'][v],root/'fusion'/v) for v in ('F1','F2')]:
            require(read(folder/'metrics.json')==metrics,'Metrics/summary disagree')
            for filename,digest in metrics['checkpoint_sha256'].items():require(sha(folder/filename)==digest,'Checkpoint changed')
            require(sha(folder/'validation_predictions.jsonl.gz')==metrics['predictions_sha256'],'Predictions changed')
            require(metrics['optimizer_updates']==metrics['epochs']*211,'Update accounting differs')
            verify_metrics(metrics['best_validation'])
            if name in ('F1','F2'):
                require(training_evidence(folder,metrics,run['F0']['macro_f1'])==run['training_evidence'][name],
                        'Trained-candidate evidence differs')
        runs[seed]=run
    summary={'status':'completed','exit_code':0,'training_jobs':6,'F0_checks':2,'binding':binding,
        'paired_macro_f1':paired_statistics(runs),'per_seed':{str(k):v for k,v in runs.items()},
        'test_evaluated':False,'automatic_followup':False}
    class_stats={}
    for c in LABELS:
        class_stats[c]={v:{metric:{'mean_percent':100*statistics.mean(
            (r['F0'] if v=='F0' else r['models'][v]['best_validation'])['per_class'][c][metric] for r in runs.values()),
            'sample_std_pp':100*statistics.stdev((r['F0'] if v=='F0' else r['models'][v]['best_validation'])[
                'per_class'][c][metric] for r in runs.values())} for metric in ('precision','recall','f1-score')}
            for v in ('F0','F1','F2')}
    summary['per_class_three_seed_statistics']=class_stats
    write(args.output/'summary.json',summary)
    lines=['# Frozen fusion paired seeds42/43/44','',
        '| seed | F0 Macro-F1 | F1 Macro-F1 | F2 Macro-F1 | F2-F0 pp | F2-F1 pp |',
        '|---|---:|---:|---:|---:|---:|']
    for seed,row in summary['paired_macro_f1']['per_seed'].items():
        p=row['macro_f1_percent']
        lines.append(f"| {seed} | {p['F0']:.6f}% | {p['F1']:.6f}% | {p['F2']:.6f}% | {row['F2_minus_F0_pp']:+.6f} | {row['F2_minus_F1_pp']:+.6f} |")
    lines += ['', 'Differences and sample standard deviations use unrounded values.',
        'Seed42 selected the candidate; inspect new seeds43/44 separately. No significance claim.',
        'Epoch0 was eligible: matching F0 alone is not evidence of benefit.',
        'Review class tradeoffs and magnitude before retaining F2. No automatic next training.']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary


def queue(args,snapshot,binding):
    base=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--repo',str(args.repo),
          '--config',str(args.config),'--mode','seed','--physical-gpu',str(args.physical_gpu)]
    if args.allow_shared_gpu:base.append('--allow-shared-gpu')
    for seed in SEEDS:
        root=args.output/f'seed{seed}'
        child=subprocess.Popen(base+['--seed',str(seed),'--output',str(root)])
        try:
            while child.poll() is None:
                candidates=[p for p in (root/'progress.json',root/'fusion/progress.json') if p.exists()]
                try:
                    detail=read(max(candidates,key=lambda p:p.stat().st_mtime)) if candidates else {'phase':'setup'}
                except (json.JSONDecodeError,FileNotFoundError):
                    # Historical progress writers are non-atomic; an in-flight
                    # snapshot must not abort the queue and orphan its child.
                    detail={'phase':'progress_snapshot_pending'}
                write(args.output/'progress.json',{**detail,'seed':seed,'child_pid':child.pid,'training_jobs':6})
                time.sleep(5)
        except BaseException:
            if child.poll() is None:
                child.terminate();child.wait(timeout=30)
            raise
        require(child.returncode==0,f'Seed{seed} failed; no retry or later training')
    return analyze(args,snapshot,binding)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path.cwd())
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--mode',choices=('check','queue','seed'),default='check')
    parser.add_argument('--seed',type=int,choices=SEEDS)
    parser.add_argument('--physical-gpu',type=int,default=1)
    parser.add_argument('--allow-shared-gpu',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();args.repo=args.repo.resolve();args.config=(args.repo/args.config).resolve()
    study,n0,fusion,registry,snapshot,binding=metadata(args)
    if args.mode=='check':
        print(json.dumps({'status':'metadata_checks_passed','binding':binding,'training_started':False}),flush=True)
        return
    require(os.name=='posix' and args.output is not None,'Linux wx and new output required')
    require(args.mode!='seed' or args.seed in SEEDS,'Choose a registered seed')
    args.output=inside(args.repo/'outputs',inside(args.repo,args.output))
    require(not args.output.exists(),'Refuse overwrite/resume')
    args.output.mkdir(parents=True)
    write(args.output/'status.json',{'status':'running','pid':os.getpid()})
    try:
        if args.mode=='queue':queue(args,snapshot,binding)
        else:run_seed(args,study,n0,fusion,registry,binding)
        write(args.output/'status.json',{'status':'completed','exit_code':0})
        write(args.output/'progress.json',{'phase':'completed','status':'completed'})
    except BaseException:
        write(args.output/'status.json',{'status':'failed','exit_code':1})
        (args.output/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
        raise


if __name__=='__main__':main()
