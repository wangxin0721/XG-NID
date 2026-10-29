"""N0 warm-start equal control and pretrained vector gate joint fine-tuning."""
from __future__ import annotations
import argparse,copy,json,os,subprocess,sys,time,traceback
from pathlib import Path
from types import MethodType,SimpleNamespace
from itertools import islice
from e3_gate_study import read,write,require,sha,inside,runtime,activate_gpu,checked_gpu,guard_graph_loads
import run_e2_concat_pair as pair
import run_fair_vector as vector


def check_config(c):
    require(c['schema']=='xgnid.joint-gate.v1' and c['seed']==42 and c['variants']==['J0','J2']
        and (c['epochs'],c['patience'],c['batch_size'],c['eval_batch_size'])==(30,12,512,16)
        and c['test_evaluated'] is False and c['automatic_followup'] is False,'Wrong joint protocol')
    require(all(0<c[k]<1 for k in ('encoder_lr','head_lr','gate_lr')) and c['weight_decay']>=0,'Invalid hyperparameters')


def group_name(name,variant):
    if name.startswith(('flow_encoder.','packet_encoder.')):return 'encoder'
    if name.startswith(('flow_classifier.','packet_classifier.','shared_head.')):return 'head'
    if variant=='J2' and name.startswith('joint_gate.'):return 'gate'
    return None


def groups(model,variant,c):
    buckets={k:[] for k in ('encoder','head','gate')};seen=set()
    for name,p in model.named_parameters():
        kind=group_name(name,variant);p.requires_grad_(kind is not None)
        if kind is not None and id(p) not in seen:buckets[kind].append(p);seen.add(id(p))
    return [{'params':ps,'lr':c[k+'_lr'],'name':k} for k,ps in buckets.items() if ps]


def logits(self,data):
    _,_,zf,zp=self._branch_outputs(data)
    if self.joint_variant=='J0':alpha=zf.new_full(zf.shape,.5)
    else:
        f=self.joint_torch.cat((zf,zp,self._confidence_from_logits(zf)[:,None],self._confidence_from_logits(zp)[:,None],
            self._margin_from_logits(zf)[:,None],self._margin_from_logits(zp)[:,None],
            self.joint_F.cosine_similarity(zf,zp,dim=-1)[:,None]),dim=-1)
        alpha=self.joint_gate(f).sigmoid()
    self._last_cache={}
    return self.shared_head(alpha*zf+(1-alpha)*zp)


def build(core,data,device,n0,variant,gate=None):
    core.seed_all(42)
    warm=core.Batch.from_data_list([data.load(r) for r in data.training_candidates()[:16]]).to(device)
    m=core.build_experiment('E2',warm).to(device);core.initialize(m,'E2',warm)
    m.load_state_dict(n0['model_state'],strict=True)
    m.joint_variant=variant;m.joint_torch=core.torch;m.joint_F=core.F
    if variant=='J2':
        m.joint_gate=vector.make_vector(core,m,'V2',device)
        if gate is not None:m.joint_gate.load_state_dict(gate,strict=True)
    m._graph_logits=MethodType(logits,m)
    return m


def probe(core,m,loader,device,c,variant):
    clean={n:t.detach().clone() for n,t in m.state_dict().items()};rng=pair.rng_state(core)
    opt=core.torch.optim.Adam(groups(m,variant,c),weight_decay=c['weight_decay']);receipts=[]
    for batch in islice(loader,3):
        m.train();opt.zero_grad(set_to_none=True);before={n:p.detach().clone() for n,p in m.named_parameters() if p.requires_grad}
        batch=batch.to(device);loss=core.F.nll_loss(m(batch),batch.y.view(-1));loss.backward()
        require(bool(core.torch.isfinite(loss)) and all(p.grad is None or bool(core.torch.isfinite(p.grad).all()) for p in m.parameters()),'Bad probe gradient')
        norms={}
        for prefix in ('flow_encoder.','packet_encoder.','flow_classifier.','packet_classifier.','shared_head.')+ (('joint_gate.',) if variant=='J2' else ()):
            norm=sum(float(p.grad.abs().sum()) for n,p in m.named_parameters() if n.startswith(prefix) and p.grad is not None)
            require(norm>0,'Missing gradient '+prefix);norms[prefix]=norm
        opt.step()
        require(all(any(not core.torch.equal(before[n],p) for n,p in m.named_parameters() if n.startswith(prefix) and n in before)
                    for prefix in norms),'Module failed to update')
        if variant=='J2':norms['gate_hidden']=float(m.joint_gate[0].weight.grad.abs().sum())
        receipts.append(norms)
    require(len(receipts)==3,'Incomplete probe')
    if variant=='J2':require(any(r['gate_hidden']>0 for r in receipts[1:]),'Gate hidden layer never receives gradient')
    m.load_state_dict(clean,strict=True);m.zero_grad(set_to_none=True);m.eval();pair.restore_rng(core,rng)
    return {'updates':receipts,'discarded':True}


def train(core,args,c,data,device,m,variant,binding):
    folder=args.output/variant;folder.mkdir();rows={p:core.select_rows(data,p) for p in ('train','val')}
    sampler=core.TrainBatches(len(rows['train']),512,42)
    kwargs=dict(num_workers=2,pin_memory=True,persistent_workers=True,multiprocessing_context='fork')
    loader=core.DataLoader(core.GraphRows(data,rows['train']),batch_sampler=sampler,**kwargs)
    val=core.DataLoader(core.GraphRows(data,rows['val']),batch_size=16,shuffle=False,**kwargs)
    gs=groups(m,variant,c);write(folder/'parameter_groups.json',[{'name':g['name'],'lr':g['lr'],'parameters':sum(p.numel() for p in g['params'])} for g in gs])
    zero,_=core.evaluate(m,val,device);write(folder/'epoch_zero.json',zero)
    write(folder/'probe.json',args.probes[variant])
    sampler=core.TrainBatches(len(rows['train']),512,42)
    loader=core.DataLoader(core.GraphRows(data,rows['train']),batch_sampler=sampler,**kwargs)
    core.seed_all(42);opt=core.torch.optim.Adam(groups(m,variant,c),weight_decay=c['weight_decay'])
    best,best_epoch,stale=zero['macro_f1'],0,0;orders=[];started=time.perf_counter()
    cfg={'binding':binding,'protocol':c,'variant':variant,'optimizer_reset':True}
    core.checkpoint(folder/'best.pt',{'model_state':m.state_dict(),'validation':zero,'epoch':0,'config':cfg})
    with (folder/'epochs.jsonl').open('x',encoding='utf-8') as log:
        for epoch in range(1,c['epochs']+1):
            m.train();steps=count=0
            for batch in loader:
                batch=batch.to(device);opt.zero_grad(set_to_none=True)
                loss=core.F.nll_loss(m(batch),batch.y.view(-1));require(bool(core.torch.isfinite(loss)),'Nonfinite loss')
                loss.backward();opt.step();steps+=1;count+=batch.num_graphs
            require(steps==211 and count==107621,'Incomplete epoch');orders.append(sampler.last_order_sha256)
            metrics,_=core.evaluate(m,val,device);best,stale,improved=pair.selection_state(metrics['macro_f1'],best,stale)
            if improved:best_epoch=epoch
            payload={'model_state':m.state_dict(),'optimizer_state':opt.state_dict(),'validation':metrics,'epoch':epoch,
                'config':cfg,'sampler_rng':sampler.generator.get_state(),'rng':pair.rng_state(core)}
            core.checkpoint(folder/'last.pt',payload)
            if improved:core.checkpoint(folder/'best.pt',payload)
            entry={'epoch':epoch,'val_macro_f1':metrics['macro_f1'],'optimizer_updates':steps,'train_order_sha256':orders[-1]}
            log.write(json.dumps(entry)+'\n');log.flush();print(json.dumps({'variant':variant,**entry}),flush=True)
            if stale>=c['patience']:break
    selected=core.torch.load(folder/'best.pt',map_location=device,weights_only=False)
    # Reconstruct the appropriate architecture, not just reload a mutated model.
    fresh=build(core,data,device,args.n0,variant,args.gate if variant=='J2' else None)
    fresh.load_state_dict(selected['model_state'],strict=True)
    measured,exports=core.evaluate(fresh,val,device,export_rows=True)
    require(measured['confusion_matrix']==selected['validation']['confusion_matrix'],'Checkpoint replay failed')
    pair.save_predictions(folder/'validation_predictions.jsonl.gz',exports)
    result={'best_validation':measured,'best_epoch':best_epoch,'epochs':epoch,'updates':epoch*211,'orders':orders,
            'joint_wall_seconds':time.perf_counter()-started,'epoch_zero':zero,'checkpoint_sha256':sha(folder/'best.pt')}
    write(folder/'metrics.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path.cwd());p.add_argument('--config',type=Path,required=True)
    p.add_argument('--physical-gpu',type=int,default=1);p.add_argument('--allow-shared-gpu',action='store_true');p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.repo=args.repo.resolve();args.config=(args.repo/args.config).resolve();c=read(args.config);check_config(c)
    fa=SimpleNamespace(repo=args.repo,config=args.config.parent/c['fair_config'])
    _,reg,_,fusion,refs,parent=vector.metadata(fa)
    args.output=inside(args.repo/'outputs',inside(args.repo,args.output));require(not args.output.exists(),'Refuse overwrite/resume')
    args.output.mkdir(parents=True)
    try:
        fairroot=inside(args.repo,c['stage_A']).parent
        snapshot={}
        for job in ('E0FT42','E0FT43','E0FT44','V1','V2'):
            folder=fairroot/job
            item=read(folder/'status.json') if (folder/'status.json').exists() else {'status':'not_started_or_unavailable'}
            if (folder/'result.json').exists():item={**item,'metrics':read(folder/'result.json')['metrics']}
            snapshot[job]=item
        write(args.output/'fair_queue_snapshot.json',snapshot)
        stage=inside(args.repo,c['stage_A']);queue=stage.parent
        # Never compete with or terminate an already running fair/vector queue.
        while not (stage/'status.json').exists() and (queue/'status.json').exists() and read(queue/'status.json')['status']=='running':
            write(args.output/'status.json',{'status':'waiting_for_existing_V2'});time.sleep(30)
        while (stage/'status.json').exists() and read(stage/'status.json')['status']=='running':
            write(args.output/'status.json',{'status':'waiting_for_existing_V2'});time.sleep(30)
        stage_seconds=None
        if not (stage/'status.json').exists():
            stage=args.output/'stage_A'
            cmd=[sys.executable,'-B','-u',str(Path(__file__).with_name('run_fair_vector.py')),'--repo',str(args.repo),
                 '--config',str(fa.config),'--mode','job','--job','V2','--physical-gpu',str(args.physical_gpu),'--output',str(stage)]
            if args.allow_shared_gpu:cmd.append('--allow-shared-gpu')
            tick=time.perf_counter();require(subprocess.run(cmd).returncode==0,'Existing V2 entry failed');stage_seconds=time.perf_counter()-tick
        require(read(stage/'status.json')['status']=='completed','V2 failed; inspect existing run, no auto retry')
        report=read(stage/'result.json');require(report['binding']==parent and report['frozen_model_unchanged'],'V2 binding/config/cache mismatch')
        vm=report['metrics'];require(sha(stage/'V2/best.pt')==vm['checkpoint_sha256']['best.pt'],'V2 changed')
        gpu=checked_gpu(args.physical_gpu,args.allow_shared_gpu);activate_gpu(gpu)
        sys.path.insert(0,str(args.repo/'scripts'));import train_grouped as core
        device=runtime(core);data=core.FrozenGraphs(inside(args.repo,reg['split_directory']),reg['split_id']);guard_graph_loads(data,{'train','val'})
        args.n0=core.torch.load(inside(args.repo,refs['42']['directory'])/'best.pt',map_location=device,weights_only=False)
        args.gate=core.torch.load(stage/'V2/best.pt',map_location=device,weights_only=False)['fusion_state']
        before=build(core,data,device,args.n0,'J2');base=build(core,data,device,args.n0,'J0')
        loader=core.DataLoader(core.GraphRows(data,core.select_rows(data,'val')),batch_size=16,shuffle=False,num_workers=2)
        with core.torch.no_grad():
            for batch in loader:
                batch=batch.to(device);a,b=base(batch),before(batch)
                require(core.torch.allclose(a,b,atol=1e-5,rtol=1e-5) and bool((a.argmax(-1)==b.argmax(-1)).all()),'Prewarm equivalence failed')
        n0check,_=core.evaluate(base,loader,device);require(n0check['confusion_matrix']==refs['42']['metrics']['best_validation']['confusion_matrix'],'N0 replay failed')
        warmed=build(core,data,device,args.n0,'J2',args.gate)
        vcheck,_=core.evaluate(warmed,loader,device);require(vcheck['confusion_matrix']==vm['best_validation']['confusion_matrix'],'Warm gate graph/cache replay failed')
        write(args.output/'initial_checks.json',{'N0_reproduced':True,'prewarm_equivalent':True,'warm_V2_reproduced_on_graphs':True})
        args.probes={}
        for variant,m in [('J0',base),('J2',warmed)]:
            sampler=core.TrainBatches(len(core.select_rows(data,'train')),512,42)
            batches=list(islice(sampler,3))
            pl=core.DataLoader(core.GraphRows(data,core.select_rows(data,'train')),batch_sampler=batches,num_workers=0)
            args.probes[variant]=probe(core,m,pl,device,c,variant)
        write(args.output/'joint_probes.json',args.probes)
        binding={'joint_config_sha256':sha(args.config),'joint_source_sha256':sha(Path(__file__)),
                 'parent':parent,'stage_A_best_sha256':sha(stage/'V2/best.pt')}
        results={}
        for variant,m in [('J0',base),('J2',warmed)]:results[variant]=train(core,args,c,data,device,m,variant,binding)
        common=min(len(r['orders']) for r in results.values());require(results['J0']['orders'][:common]==results['J2']['orders'][:common],'Joint data order differs')
        write(args.output/'summary.json',{'status':'completed','binding':binding,'N0':refs['42']['metrics'],'frozen_V2':vm,
            'joint':results,'stage_A':{'directory':str(stage),'updates':vm['optimizer_updates'],
            'elapsed_seconds':stage_seconds,'timing_available':stage_seconds is not None,
            'timing_note':'Historical V2 did not record wall time; never infer duration from file timestamps'},
            'test_evaluated':False,'attribution':'J2 includes gate warmup plus joint adaptation; gains are not isolated gate-component effects'})
        write(args.output/'status.json',{'status':'completed','exit_code':0})
        from summarize_joint_gate import summarize
        summarize(args.output)
    except BaseException:
        write(args.output/'status.json',{'status':'failed','exit_code':1});(args.output/'error.log').write_text(traceback.format_exc(),encoding='utf-8');raise


if __name__=='__main__':main()
