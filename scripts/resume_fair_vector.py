"""Recover interrupted E0-FT into a new tree, then finish remaining fixed jobs.

The source tree remains untouched. Vector jobs may be completed or not started;
this entry refuses partially trained vector jobs rather than resetting them.
"""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

import run_fair_vector as fair
from e3_gate_study import (read,write,require,sha,inside,checked_gpu,activate_gpu,runtime,guard_graph_loads,identity)
from run_fusion_seed_repeats import predictions,training_evidence
from run_e2_concat_pair import rng_state,rng_hash,restore_rng,compare_predictions,save_predictions,selection_state


def complete(folder,job,binding):
    if not (folder/'result.json').exists():return False
    result=read(folder/'result.json');require(result['binding']==binding and result['job']==job
        and result['test_evaluated'] is False,'Completed job binding/identity differs')
    model_folder=folder/('E0FT' if job.startswith('E0FT') else job);m=result['metrics']
    require(read(model_folder/'metrics.json')==m,'Completed metric records differ')
    fair.verify_metrics(m['best_validation'])
    for filename,digest in m['checkpoint_sha256'].items():require(sha(model_folder/filename)==digest,'Completed weight changed')
    require(sha(model_folder/'validation_predictions.jsonl.gz')==m['predictions_sha256'],'Completed predictions changed')
    return True


def source_plan(source,binding):
    plan=[];missing_seen=False
    for job in fair.JOBS:
        folder=source/job
        if complete(folder,job,binding):
            require(not missing_seen,'Completed job after unfinished predecessor; inspect source')
            action='skip_completed'
        elif (folder/('E0FT' if job.startswith('E0FT') else job)/'last.pt').exists():
            require(job.startswith('E0FT'),'Partially trained vector job needs a separate recovery implementation')
            require(not missing_seen,'Multiple interrupted jobs; inspect source')
            action='resume';missing_seen=True
        elif folder.exists():
            raise ValueError('Existing job without complete result or last.pt: '+job)
        else:action='new';missing_seen=True
        plan.append({'job':job,'action':action})
    require(any(r['action']=='resume' for r in plan),'No interrupted E0 last.pt to resume')
    return plan


def ensure_inactive(source,output):
    require(Path('/proc').is_dir(),'Process audit requires Linux /proc')
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit() or int(folder.name)==os.getpid():continue
        try:args=(folder/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if not any(Path(a).name in ('run_fair_vector.py','resume_fair_vector.py') for a in args):continue
        require(not any(a==str(r) or a.startswith(str(r)+'/') for a in args for r in (source,output)),
                'An original/recovery process is still active: '+folder.name)


def history(path,saved,initial):
    # Accept a torn final JSON line. A committed checkpoint can precede its log
    # write by one row; reconstruct that row using validated sampler replay.
    rows=[]
    lines=path.read_text(encoding='utf-8').splitlines()
    for i,line in enumerate(lines):
        try:row=json.loads(line)
        except json.JSONDecodeError:
            require(i==len(lines)-1,'Corrupt middle of epoch log');break
        rows.append(row)
    epoch=saved['epoch'];require(1<=epoch<=30,'Invalid saved epoch')
    require(len(rows)<=epoch,'Log is ahead of last.pt; inspect before rollback')
    require(len(rows) in (epoch-1,epoch),'Log/checkpoint gap exceeds one row')
    require([r['epoch'] for r in rows]==list(range(1,len(rows)+1)),'Non-contiguous history')
    best,best_epoch,stale=initial,0,0
    for row in rows:
        require(row['optimizer_updates']==211 and row['train_samples']==107621 and
                row['cumulative_optimizer_updates']==row['epoch']*211,'History update mismatch')
        best,stale,improved=selection_state(row['val_macro_f1'],best,stale)
        if improved:best_epoch=row['epoch']
    if len(rows)==epoch-1:
        best,stale,improved=selection_state(saved['validation']['macro_f1'],best,stale)
        if improved:best_epoch=epoch
    else:require(rows[-1]['val_macro_f1']==saved['validation']['macro_f1'],'Last score mismatch')
    require(best_epoch==saved['best_epoch'] and stale==saved['stale_epochs'] and
            abs(best-saved['best_val_macro_f1'])<1e-12 and saved['optimizer_updates']==epoch*211,
            'Saved best/stale/update counters disagree with history')
    return rows


def optimizer_hash(core,state):
    tensors=[];scalars={}
    for key,values in state['state'].items():
        for name,value in values.items():
            label=str(key)+'/'+name
            if core.torch.is_tensor(value):tensors.append((label,value))
            else:scalars[label]=value
    return identity({'tensors':core.state_hash(tensors),'scalars':scalars,'groups':state['param_groups']})


def resume_e0(args,job,reg,protocol,binding):
    gpu=checked_gpu(args.physical_gpu,args.allow_shared_gpu);activate_gpu(gpu)
    sys.path.insert(0,str(args.repo/'scripts'));import train_grouped as raw
    device=runtime(raw);seed=int(job[-2:]);core=fair.Core(raw,seed,args.output)
    folder=args.output/'E0FT';cfg=read(folder/'config.json');initial=read(args.output/'epoch_zero.json')
    require(cfg['binding']==binding and cfg['protocol']=={**protocol,'seed':seed} and cfg['variant']=='E0FT',
            'Interrupted configuration differs')
    before={n:sha(folder/n) for n in ('last.pt','best.pt','config.json','epochs.jsonl')}
    saved=core.torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
    selected=core.torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
    require(saved['config']==cfg and selected['config']==cfg and selected['epoch']==saved['best_epoch']
            and abs(selected['validation']['macro_f1']-saved['best_val_macro_f1'])<1e-12,'Best/last checkpoint inconsistency')
    rows=history(folder/'epochs.jsonl',saved,initial['macro_f1'])
    core.seed_all(seed)
    data=core.FrozenGraphs(inside(args.repo,reg['split_directory']),reg['split_id']);guard_graph_loads(data,{'train','val'})
    train_rows=core.select_rows(data,'train');val_rows=core.select_rows(data,'val')
    warm=core.Batch.from_data_list([data.load(r) for r in train_rows[:16]]).to(device)
    model=core.build_experiment('E0',warm).to(device);core.initialize(model,'E0',warm)
    model.load_state_dict(saved['model_state'],strict=True)
    require(core.state_hash(model.state_dict().items())==core.state_hash(saved['model_state'].items()),'Model restore mismatch')
    opt=core.torch.optim.Adam(model.parameters(),lr=.0001,weight_decay=0.)
    opt.load_state_dict(saved['optimizer_state'])
    require(optimizer_hash(core,opt.state_dict())==optimizer_hash(core,saved['optimizer_state']),'Adam restore mismatch')
    sampler=core.TrainBatches(len(train_rows),512,42)
    for i in range(saved['epoch']):
        for _ in sampler:pass
        if i<len(rows):require(sampler.last_order_sha256==rows[i]['train_order_sha256'],'Sampler history mismatch')
    require(core.torch.equal(sampler.generator.get_state(),saved['sampler_rng_state']),'Sampler RNG mismatch')
    if len(rows)<saved['epoch']:
        rows.append({'epoch':saved['epoch'],'train_loss':None,'train_samples':107621,'optimizer_updates':211,
            'cumulative_optimizer_updates':saved['optimizer_updates'],'train_order_sha256':sampler.last_order_sha256,
            'val_macro_f1':saved['validation']['macro_f1'],'val_accuracy':saved['validation']['accuracy'],
            'best_epoch':saved['best_epoch'],'stale_epochs':saved['stale_epochs'],'reconstructed_from_checkpoint':True})
    (folder/'epochs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    kwargs={'num_workers':2,'pin_memory':True,'persistent_workers':True,'multiprocessing_context':'fork'}
    loader=core.DataLoader(core.GraphRows(data,train_rows),batch_sampler=sampler,
                          generator=core.torch.Generator().manual_seed(seed+1000),**kwargs)
    val_loader=core.DataLoader(core.GraphRows(data,val_rows),batch_size=16,shuffle=False,
                              generator=core.torch.Generator().manual_seed(seed+1001),**kwargs)
    # Restore all global RNG after constructor/materialization and index replay.
    restore_rng(core,saved['rng'])
    require(rng_hash(core,rng_state(core))==rng_hash(core,saved['rng']),'Global RNG restore mismatch')
    receipt={'job':job,'files_before_recovery':before,'saved_epoch':saved['epoch'],'next_epoch':saved['epoch']+1,
        'best_epoch':saved['best_epoch'],'stale_epochs':saved['stale_epochs'],'updates':saved['optimizer_updates'],
        'model_state_restored':True,'Adam_state_restored':True,'sampler_rng_verified':True,
        'global_rng_restored':True,'worker_rng_not_saved':True,'bitwise_uninterrupted_equivalence_claimed':False,
        'physical_gpu':args.physical_gpu,'gpu':gpu,'recovery_entry_sha256':sha(Path(__file__))}
    write(args.output/'recovery_receipt.json',receipt);print(json.dumps({'resume':receipt}),flush=True)
    best,best_epoch,stale,updates=saved['best_val_macro_f1'],saved['best_epoch'],saved['stale_epochs'],saved['optimizer_updates']
    epoch=saved['epoch'];validation=saved['validation'];orders=[r['train_order_sha256'] for r in rows]
    with (folder/'epochs.jsonl').open('a',encoding='utf-8') as log:
        while epoch<30 and stale<12:
            epoch+=1;model.train();count=steps=0;total=0.
            for batch in loader:
                batch=batch.to(device);opt.zero_grad(set_to_none=True)
                loss=core.F.nll_loss(model(batch),batch.y.view(-1));require(bool(core.torch.isfinite(loss)),'Nonfinite loss')
                loss.backward();require(all(p.grad is None or bool(core.torch.isfinite(p.grad).all()) for p in model.parameters()),'Bad gradients')
                opt.step();count+=batch.num_graphs;steps+=1;total+=float(loss.detach())*batch.num_graphs
            require(count==107621 and steps==211,'Incomplete epoch');updates+=steps
            orders.append(sampler.last_order_sha256);validation,_=core.evaluate(model,val_loader,device)
            best,stale,improved=selection_state(validation['macro_f1'],best,stale)
            if improved:
                best_epoch=epoch
                core.checkpoint(folder/'best.pt',{'model_state':model.state_dict(),'epoch':epoch,'validation':validation,'config':cfg})
            payload={'model_state':model.state_dict(),'optimizer_state':opt.state_dict(),'epoch':epoch,'best_epoch':best_epoch,
                'best_val_macro_f1':best,'stale_epochs':stale,'optimizer_updates':updates,'validation':validation,'config':cfg,
                'sampler_rng_state':sampler.generator.get_state(),'rng':rng_state(core)}
            core.checkpoint(folder/'last.pt',payload)
            entry={'epoch':epoch,'train_loss':total/count,'train_samples':count,'optimizer_updates':steps,
                'cumulative_optimizer_updates':updates,'train_order_sha256':orders[-1],'val_macro_f1':validation['macro_f1'],
                'val_accuracy':validation['accuracy'],'best_epoch':best_epoch,'stale_epochs':stale}
            log.write(json.dumps(entry)+'\n');log.flush();print(json.dumps({'job':job,**entry}),flush=True)
            write(args.output/'progress.json',{'job':job,'epoch':epoch,'stale_epochs':stale,'updates':updates})
    selected=core.torch.load(folder/'best.pt',map_location=device,weights_only=False)
    model.load_state_dict(selected['model_state'],strict=True)
    reproduced,exports=core.evaluate(model,val_loader,device,export_rows=True)
    require(reproduced['confusion_matrix']==selected['validation']['confusion_matrix'],'Best validation not reproduced')
    pred=folder/'validation_predictions.jsonl.gz'
    if pred.exists():pred.rename(folder/'previous_partial_predictions.jsonl.gz')
    save_predictions(pred,exports)
    m={'variant':'E0FT','best_epoch':best_epoch,'best_validation':selected['validation'],'last_validation':validation,
        'epochs':epoch,'optimizer_updates':updates,'training_samples_seen':107621*epoch,
        'versus_original':compare_predictions(predictions(args.output/'epoch_zero_predictions.jsonl.gz'),exports),
        'macro_f1_minus_original_pp':100*(best-initial['macro_f1']),
        'training_start_rng_sha256':cfg['training_start_rng_sha256'],'orders':orders,'test_evaluated':False,
        'predictions_sha256':sha(pred),'checkpoint_sha256':{n:sha(folder/n) for n in ('best.pt','last.pt')}}
    write(folder/'metrics.json',m);write(folder/'status.json',{'status':'completed','exit_code':0})
    write(args.output/'result.json',{'job':job,'binding':binding,'metrics':m,
        'training_evidence':training_evidence(folder,m,initial['macro_f1']),'test_evaluated':False})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path.cwd());p.add_argument('--config',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=('check','queue','job'),default='check');p.add_argument('--job',choices=fair.JOBS)
    p.add_argument('--physical-gpu',type=int,default=0);p.add_argument('--allow-shared-gpu',action='store_true')
    args=p.parse_args();args.repo=args.repo.resolve();args.config=(args.repo/args.config).resolve()
    args.source=inside(args.repo/'outputs',inside(args.repo,args.source));args.output=inside(args.repo/'outputs',inside(args.repo,args.output))
    require(args.source!=args.output and args.source not in args.output.parents,'New independent recovery tree required')
    spec,reg,n0,fusion,refs,binding=fair.metadata(args)
    if args.mode=='job':
        require(args.job.startswith('E0FT'),'Only interrupted E0 can resume')
        resume_e0(args,args.job,reg,n0,binding);write(args.output/'status.json',{'status':'completed','exit_code':0});return
    plan=source_plan(args.source,binding)
    if os.name=='posix':ensure_inactive(args.source,args.output)
    if args.mode=='check':print(json.dumps({'status':'recovery_metadata_passed','plan':plan}));return
    require(os.name=='posix' and not args.output.exists(),'Linux wx/new output required')
    args.output.mkdir(parents=True);write(args.output/'status.json',{'status':'running'})
    write(args.output/'recovery_plan.json',{'source':str(args.source),'plan':plan,'binding':binding})
    try:
        for item in plan:
            job,action=item['job'],item['action'];target=args.output/job
            if action!='new':shutil.copytree(args.source/job,target)
            if action=='skip_completed':
                require(complete(target,job,binding),'Copied completed job failed validation')
                write(target/'status.json',{'status':'completed','exit_code':0});continue
            entry=Path(__file__) if action=='resume' else Path(fair.__file__)
            command=[sys.executable,'-B','-u',str(entry),'--repo',str(args.repo),'--config',str(args.config),
                '--mode','job','--job',job,'--physical-gpu',str(args.physical_gpu),'--output',str(target)]
            if action=='resume':command+=['--source',str(args.source)]
            if args.allow_shared_gpu:command.append('--allow-shared-gpu')
            write(args.output/'progress.json',{'job':job,'action':action,'physical_gpu':args.physical_gpu})
            require(subprocess.run(command).returncode==0,'Recovery job failed; no further jobs started')
        fair.analyze(args,refs,binding);write(args.output/'status.json',{'status':'completed','exit_code':0})
    except BaseException:
        write(args.output/'status.json',{'status':'failed','exit_code':1})
        (args.output/'error.log').write_text(traceback.format_exc(),encoding='utf-8');raise


if __name__=='__main__':main()
