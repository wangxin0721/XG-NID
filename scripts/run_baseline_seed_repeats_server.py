"""Exactly four E0/E2 batch512 repeats; seed43/44, validation selection only."""
import argparse
from datetime import datetime, timezone
import gc
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
from e3_gate_study import (activate_gpu, checked_gpu, guard_graph_loads, inside, read,
                          require, sha, write, identity)
from baseline_seed_repeats import JOBS, metadata, functions

HERE=Path(__file__).resolve().parent

def historical_module(study,model):
    path=HERE.parent/study['references'][model]['entry']
    spec=importlib.util.spec_from_file_location('baseline_seed42_'+model,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def one(args,study,receipt):
    gpu=checked_gpu(args.physical_gpu,args.allow_shared_gpu)
    activate_gpu(gpu)
    sys.path.insert(0,str(args.repo/'scripts'))
    import train_grouped as core
    module=historical_module(study,args.model)
    ref=study['references'][args.model]
    original_config=read(HERE.parent/ref['study_config'])
    module.check_config(original_config,512)
    old=SimpleNamespace(**vars(args))
    old.config=HERE.parent/ref['study_config'];old.batch_size=512
    if args.model=='E0':
        old.reference_metadata=module.metadata_checks(args.repo,original_config)
    frozen,device,binding=module.setup(core,old,original_config,gpu)
    guard_graph_loads(frozen,{'train'} if args.mode=='probe' else {'train','val'})
    # Zero-update audit of the historical seed42 path, before any new seed is used.
    reference=module.initial_model(core,frozen,device,binding)
    require(core.state_hash(reference.state_dict().items())==ref['initial_state_sha256'],'Seed42 initialization did not reproduce')
    del reference
    seeded,probe,train,adapter=functions(module,args.seed)
    candidate=seeded(core,frozen,device,binding,args.seed)
    actual_initial=core.state_hash(candidate.state_dict().items())
    initial_rng=identity({'python':random.getstate(),'torch':core.state_hash(
        [('cpu',core.torch.get_rng_state())]+[('cuda_%d'%i,s) for i,s in enumerate(core.torch.cuda.get_rng_state_all())])})
    del candidate
    gc.collect();core.torch.cuda.empty_cache()
    binding.update(seed42_initial_state_sha256=binding['initial_state_sha256'],initial_state_sha256=actual_initial,
                   repeat_seed=args.seed,repeat_model=args.model,repeat_study=receipt,
                   seed42_initialization_reproduced=True,seeded_initial_rng_sha256=initial_rng,adapter=adapter)
    # Old functions rebuild from the exact seeded constructor before the probe
    # or real training. Temporary audit models never contribute optimizer steps.
    return (probe if args.mode=='probe' else train)(core,old,original_config,frozen,device,binding,gpu)

def queue(args,study,receipt):
    base=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--repo',str(args.repo),
          '--config',str(args.config),'--physical-gpu',str(args.physical_gpu)]
    if args.allow_shared_gpu:base.append('--allow-shared-gpu')
    for index,(model,seed) in enumerate(JOBS,1):
        name=f'{model}_s{seed}'
        for phase,folder in [('probe',args.output/(name+'_probe')),('full',args.output/name)]:
            command=base+['--mode',phase,'--model',model,'--seed',str(seed),'--output',str(folder)]
            if phase=='full':command+=['--probe-report',str(args.output/(name+'_probe')/'summary.json')]
            print(json.dumps({'starting':name,'mode':phase,'job':index,'jobs':4}),flush=True)
            child=subprocess.Popen(command)
            while child.poll() is None:
                item={'model':model,'seed':seed,'phase':phase+'_setup','epoch':None,'batches_done':None}
                if (folder/'progress.json').exists():
                    item.update(read(folder/'progress.json'))
                item.update(job=index,total_jobs=4,child_pid=child.pid,
                            queue_poll_utc=datetime.now(timezone.utc).isoformat())
                write(args.output/'progress.json',item)
                time.sleep(5)
            require(child.returncode==0,f'{name} {phase} failed; queue stopped, no automatic retry')
    from analyze_baseline_seed_repeats import analyze
    analysis=analyze(args.repo,args.output,study,receipt)
    write(args.output/'analysis.json',analysis)
    write(args.output/'progress.json',{'phase':'completed','total_training_runs':4})
    return {'status':'completed','exit_code':0,'mode':'queue','binding':receipt,
            'training_runs':4,'analysis_sha256':sha(args.output/'analysis.json'),
            'paired_macro_f1_percentage_points':analysis['paired_macro_f1_percentage_points'],
            'test_evaluated':False,'frozen_fusion_jobs_started':False}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path.cwd())
    parser.add_argument('--config',type=Path,default=HERE.parent/'configs/baseline_seed_repeats_20261028.json')
    parser.add_argument('--mode',choices=('check','queue','probe','full'),default='check')
    parser.add_argument('--model',choices=('E0','E2'))
    parser.add_argument('--seed',type=int,choices=(43,44))
    parser.add_argument('--physical-gpu',type=int,choices=(0,1),default=1)
    parser.add_argument('--allow-shared-gpu',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--probe-report',type=Path)
    args=parser.parse_args();args.repo=args.repo.resolve();args.config=args.config.resolve()
    study,receipt=metadata(args)
    if args.mode=='check':
        # Compile all historical AST hooks without importing model libraries.
        import inspect
        from baseline_seed_repeats import initial_ast,probe_ast,full_ast
        for model in ('E0','E2'):
            module=historical_module(study,model)
            for name,adapter in [('initial_model',initial_ast),('probe',probe_ast),('full',full_ast)]:
                compile(adapter(inspect.getsource(getattr(module,name))),'<repeat-static-check>','exec')
        print(json.dumps({'status':'metadata_and_adapter_checks_passed','binding':receipt,
                          'gpu_touched':False,'training_started':False}),flush=True)
        return 0
    require(os.name=='posix','Train only on Linux wx')
    require(args.output is not None,'Explicit new output required')
    if args.mode in ('probe','full'):
        require((args.model,args.seed) in JOBS,'Unregistered model/seed')
    require(args.mode!='full' or args.probe_report is not None,'Full needs matching probe')
    args.output=inside(args.repo/'outputs',args.output.resolve())
    require(args.output!=args.repo/'outputs','Use a new child output directory')
    args.output.mkdir(exist_ok=False)
    write(args.output/'status.json',{'status':'running','mode':args.mode,'pid':os.getpid(),
                                    'batch_size':512,'model':args.model,'seed':args.seed})
    try:
        result=queue(args,study,receipt) if args.mode=='queue' else one(args,study,receipt)
        result['test_evaluated']=False
        write(args.output/'summary.json',result)
        write(args.output/'status.json',{'status':'completed','exit_code':0,'mode':args.mode,
                                        'batch_size':512,'model':args.model,'seed':args.seed})
        print(json.dumps({'status':'completed','output':str(args.output),'test_evaluated':False}),flush=True)
    except BaseException as exc:
        (args.output/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
        failure={'status':'failed','exit_code':130 if isinstance(exc,KeyboardInterrupt) else 1,
                 'mode':args.mode,'batch_size':512,'model':args.model,'seed':args.seed}
        write(args.output/'status.json',failure)
        if args.model and (args.output/args.model).exists():write(args.output/args.model/'status.json',failure)
        raise
    return 0

if __name__=='__main__':sys.exit(main())
