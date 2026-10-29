"""Three fair E0 fine-tunes and two frozen feature-wise gates. Stdlib import."""
from __future__ import annotations
import argparse
import ast
import copy
import gc
import inspect
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import textwrap
import time
import traceback
from types import FunctionType, SimpleNamespace
from itertools import islice

from e3_gate_study import (read,write,require,sha,inside,runtime,checked_gpu,activate_gpu,guard_graph_loads)
import run_e2_concat_pair as pair
import run_frozen_fusion as ff
from run_fusion_seed_repeats import SeedCore, study_controls, verify_metrics, training_evidence, predictions

HERE=Path(__file__).resolve().parent
JOBS=('E0FT42','E0FT43','E0FT44','V1','V2')


def check_spec(spec):
    require(spec['schema']=='xgnid.fair-vector.v1' and spec['jobs']==list(JOBS)
        and spec['V2_minus_N0_min_pp']==.10 and spec['V2_minus_V1_min_pp']==.05
        and spec['class_tradeoff_review_required'] is True and spec['test_evaluated'] is False
        and spec['automatic_followup'] is False,'Five fixed jobs/thresholds only')


def metadata(args):
    spec=read(args.config);check_spec(spec)
    registry=read(inside(args.config.parent,spec['baseline_registry']))
    n0spec=read(inside(args.config.parent,spec['N0_protocol']))
    fusion=read(inside(args.config.parent,spec['fusion_protocol']));ff.check_spec(fusion)
    refs=read(inside(args.config.parent,spec['n0_references']))
    snap=read(args.config.parent/'frozen_fusion_seed42_result_20261029.json')
    require(sha(inside(args.config.parent,spec['N0_protocol']))==snap['binding']['n0_binding']['config_sha256']
        and sha(inside(args.config.parent,spec['fusion_protocol']))==snap['binding']['config_sha256'],
        'Must use actual completed N0/frozen protocols')
    for name,digest in snap['binding']['sources'].items():require(sha(HERE/name)==digest,'Historical helper changed')
    parents={}
    for seed in (42,43,44):
        root=(inside(args.repo,registry['references']['E0']['directory']) if seed==42 else
              inside(args.repo,spec['baseline_directory'])/f'E0_s{seed}')
        for p in (root,root/'E0'):
            status=read(p/'status.json');require((status['status'],status['exit_code'])==('completed',0),'E0 incomplete')
        m=read(root/'E0/metrics.json');cfg=read(root/'E0/config.json')
        require(cfg['controls']==study_controls(registry,seed) and cfg['experiment']=='E0'
            and m['parameter_count']==431704 and read(root/'summary.json')['models']['E0']==m,'Wrong E0/seed')
        require(cfg['binding']['split_id']==registry['split_id'] and cfg['binding']['sources']==registry['references']['E0']['sources']
            and cfg['binding']['samples_sha256']==snap['binding']['n0_binding']['manifest_sha256'],'E0/N0 split/source mismatch')
        require(sha(root/'E0/config.json')==m['config_sha256'] and sha(root/'E0/best.pt')==m['checkpoint_sha256']['best.pt'],
            'E0 config/weight changed')
        if seed!=42:require(cfg['binding']['repeat_seed']==seed and m['test_evaluated'] is False,'Wrong repeat seed/test')
        for name,digest in cfg['binding']['sources'].items():require(sha(inside(args.repo,name))==digest,'Model/data source changed')
        verify_metrics(m['best_validation'])
        parents[str(seed)]={'directory':root.relative_to(args.repo).as_posix(),'metrics':m,
                           'config_sha256':sha(root/'E0/config.json')}
        nroot=inside(args.repo,refs[str(seed)]['directory'])
        require(read(nroot/'metrics.json')==refs[str(seed)]['metrics'],'N0 reference differs')
        nm=refs[str(seed)]['metrics'];verify_metrics(nm['best_validation'])
        require(sha(nroot/'best.pt')==nm['checkpoint_sha256']['best.pt'],'N0 weights changed')
    fargs=SimpleNamespace(repo=args.repo,config=inside(args.config.parent,spec['fusion_protocol']))
    ff.metadata(fargs)
    cache_root=inside(args.repo,'outputs/frozen_fusion_n0_b512_s42_v1')
    for part in ('train','val'):
        require(sha(cache_root/(part+'_cache.pt'))==snap['binding']['caches'][part]['file_sha256'],'Cache changed')
    binding={'study_sha256':sha(args.config),'parents':parents,'n0_reference_sha256':sha(inside(args.config.parent,spec['n0_references'])),
        'vector_cache':snap['binding']['caches'],'vector_n0_best_sha256':snap['binding']['n0_best_sha256'],
        'helpers':{n:sha(HERE/n) for n in ('run_fair_vector.py','run_e2_concat_pair.py','run_frozen_fusion.py','run_fusion_seed_repeats.py','e3_gate_study.py')}}
    return spec,registry,n0spec,fusion,refs,binding


class Core(SeedCore):
    def __init__(self,core,seed,output):
        require(seed in (42,43,44),'Registered seeds only')
        self.core,self.seed,self.output=core,seed,output;self.phase='setup';self.epoch=0


def e0_trainer():
    # Remove only the first-update diagnostic specific to dual-branch names.
    # Loss, optimizer, data flow, selection and budget remain the N0 function.
    tree=ast.parse(textwrap.dedent(inspect.getsource(pair.train)))
    removed=[];labels=[]
    class Adapt(ast.NodeTransformer):
        def visit_If(self,node):
            if ast.unparse(node.test)=='epoch == 1 and steps == 0':
                removed.append(node);return None
            return self.generic_visit(node)
        def visit_Dict(self,node):
            for i,key in enumerate(node.keys):
                if isinstance(key,ast.Constant) and key.value=='inactive_historical_gate_and_heads_retained':
                    node.values[i]=ast.Constant(False);labels.append(key)
            return self.generic_visit(node)
    tree=Adapt().visit(tree);require(len(removed)==len(labels)==1,'N0 diagnostic/metadata hook changed')
    ast.fix_missing_locations(tree);ns=dict(pair.train.__globals__)
    exec(compile(tree,'<fair-E0-N0-trainer>','exec'),ns)
    return ns['train']


def make_vector(core,model,variant,device):
    state=pair.rng_state(core)
    if variant=='V1':
        fusion=core.torch.nn.Module();fusion.register_parameter('logit',core.torch.nn.Parameter(core.torch.zeros(8,device=device)))
    else:
        require(variant=='V2','Vector candidates only')
        fusion=copy.deepcopy(model.gate.net)
        fusion[-1]=core.torch.nn.Linear(16,8).to(device)
        with core.torch.no_grad():fusion[-1].weight.zero_();fusion[-1].bias.zero_()
    fusion.requires_grad_(True)
    require(sum(p.numel() for p in fusion.parameters())==(8 if variant=='V1' else 488),'Wrong capacity')
    pair.restore_rng(core,state)
    return fusion


def vector_predict(core,head,fusion,variant,cache,index):
    flow,packet=cache['flow'][index],cache['packet'][index]
    alpha=(flow.new_full(flow.shape,.5) if variant=='F0' else
           fusion.logit.sigmoid().expand_as(flow) if variant=='V1' else fusion(cache['features'][index]).sigmoid())
    mixed=alpha*flow+(1-alpha)*packet
    return core.F.log_softmax(head(mixed),dim=-1),alpha,mixed


def vector_evaluate(core,model,fusion,variant,cache):
    cm=[[0]*8 for _ in range(8)];exports=[];all_weights=[]
    with core.torch.no_grad():
        for offset in range(0,len(cache['ids']),16):
            ix=slice(offset,offset+16);scores,alpha,_=vector_predict(core,model.shared_head,fusion,variant,cache,ix)
            require(bool(core.torch.isfinite(scores).all()),'Nonfinite predictions')
            for j,(y,p,a) in enumerate(zip(cache['y'][ix].tolist(),scores.argmax(-1).tolist(),alpha.tolist())):
                cm[y][p]+=1
                exports.append({'sample_id':cache['ids'][offset+j],'label':y,'prediction':p,
                                'flow_weights':a,'packet_weights':[1-x for x in a]})
            all_weights.append(alpha)
    a=core.torch.cat(all_weights);result=ff.matrix_report(cm,core.LABELS)
    result['weights']={k:v.tolist() for k,v in {'mean':a.mean(0),'std':a.std(0,unbiased=False),
                                              'min':a.min(0).values,'max':a.max(0).values}.items()}
    return result,exports


def vector_functions():
    # Independent function namespaces, not global mutation of historical helpers.
    ns=dict(ff.train.__globals__);ns.update(make_fusion=make_vector,predict=vector_predict,evaluate=vector_evaluate)
    for name in ('update','probe_fusion','train'):
        old=getattr(ff,name);ns[name]=FunctionType(old.__code__,ns,name,old.__defaults__,old.__closure__)
    return ns


def e0_probe(core,model,data,device,start_rng):
    clean={n:t.detach().clone() for n,t in model.state_dict().items()}
    pair.restore_rng(core,start_rng)
    opt=core.torch.optim.Adam(model.parameters(),lr=.0001)
    rows=core.select_rows(data,'train');batches=list(islice(core.TrainBatches(len(rows),512,42),3))
    loader=core.DataLoader(core.GraphRows(data,rows),batch_sampler=batches,num_workers=0)
    receipts=[];model.train()
    for batch in loader:
        opt.zero_grad(set_to_none=True);loss=core.F.nll_loss(model(batch.to(device)),batch.y.view(-1).to(device))
        require(bool(core.torch.isfinite(loss)),'Nonfinite probe loss');loss.backward()
        require(all(p.grad is None or bool(core.torch.isfinite(p.grad).all()) for p in model.parameters()),'Bad probe gradients')
        for prefix in ('reference.convs','reference.graph_prediction'):
            require(sum(float(p.grad.abs().sum()) for n,p in model.named_parameters()
                        if n.startswith(prefix) and p.grad is not None)>0,'Missing encoder/head gradient')
        before=core.state_hash(model.named_parameters());opt.step()
        require(before!=core.state_hash(model.named_parameters()),'No update');receipts.append(float(loss.detach()))
    model.load_state_dict(clean,strict=True);model.zero_grad(set_to_none=True);model.eval()
    pair.restore_rng(core,start_rng)
    return {'losses':receipts,'clean_start_restored':core.state_hash(model.state_dict().items())==core.state_hash(clean.items())}


def one(args,job,registry,n0spec,fusion,refs,binding):
    gpu=checked_gpu(args.physical_gpu,args.allow_shared_gpu);activate_gpu(gpu)
    sys.path.insert(0,str(args.repo/'scripts'));import train_grouped as raw
    device=runtime(raw);seed=int(job[-2:]) if job.startswith('E0FT') else 42
    core=Core(raw,seed,args.output);core.seed_all(seed)
    data=core.FrozenGraphs(inside(args.repo,registry['split_directory']),registry['split_id']);guard_graph_loads(data,{'train','val'})
    warm=core.Batch.from_data_list([data.load(r) for r in core.select_rows(data,'train')[:16]]).to(device)
    if job.startswith('E0FT'):
        parent=binding['parents'][str(seed)];folder=inside(args.repo,parent['directory'])/'E0'
        model=core.build_experiment('E0',warm).to(device);core.initialize(model,'E0',warm)
        saved=core.torch.load(folder/'best.pt',map_location=device,weights_only=False)
        model.load_state_dict(saved['model_state'],strict=True);start=pair.rng_state(core)
        loader=core.DataLoader(core.GraphRows(data,core.select_rows(data,'val')),batch_size=16,shuffle=False,num_workers=2)
        initial,exports=core.evaluate(model,loader,device,export_rows=True)
        ff.verify_order(exports,core.select_rows(data,'val'))
        require(initial['confusion_matrix']==parent['metrics']['best_validation']['confusion_matrix'] and
            abs(initial['macro_f1']-parent['metrics']['best_validation']['macro_f1'])<1e-12,'E0 epoch0 mismatch')
        write(args.output/'epoch_zero.json',initial);pair.save_predictions(args.output/'epoch_zero_predictions.jsonl.gz',exports)
        receipt=e0_probe(core,model,data,device,start);require(receipt['clean_start_restored'],'Probe leaked state')
        write(args.output/'probe.json',receipt);core.phase='N0_train'
        result=e0_trainer()(core,args,{**n0spec,'seed':seed},data,device,model,start,initial,binding,'E0FT',None,exports)
        result={'metrics':result,'training_evidence':training_evidence(args.output/'E0FT',result,initial['macro_f1'])}
    else:
        model=core.build_experiment('E2',warm).to(device);core.initialize(model,'E2',warm)
        nroot=inside(args.repo,refs['42']['directory'])
        saved=core.torch.load(nroot/'best.pt',map_location=device,weights_only=False);model.load_state_dict(saved['model_state'],strict=True)
        model.requires_grad_(False);model.eval();frozen_hash=core.state_hash(model.state_dict().items())
        snap=read(args.config.parent/'frozen_fusion_seed42_result_20261029.json')
        require(frozen_hash==snap['binding']['frozen_state_sha256'],'Cached representation model differs')
        root=inside(args.repo,'outputs/frozen_fusion_n0_b512_s42_v1');caches={}
        for part in ('train','val'):
            cache=core.torch.load(root/(part+'_cache.pt'),map_location='cpu',weights_only=False)
            rows=core.select_rows(data,part)
            require(cache['ids']==[r['sample_id'] for r in rows] and cache['y'].tolist()==[r['label'] for r in rows],'Cache order/labels mismatch')
            caches[part]=ff.device_cache(core,cache,device)
        f0,baseline=vector_evaluate(core,model,None,'F0',caches['val']);gold=predictions(nroot/'validation_predictions.jsonl.gz')
        require(f0['confusion_matrix']==refs['42']['metrics']['best_validation']['confusion_matrix'] and
                [r['prediction'] for r in baseline]==[r['prediction'] for r in gold],'Vector epoch0/F0 mismatch')
        write(args.output/'F0.json',f0);functions=vector_functions();start=pair.rng_state(core)
        candidate=make_vector(core,model,job,device);max_error=0.
        with core.torch.no_grad():
            for offset in range(0,len(baseline),16):
                ix=slice(offset,offset+16)
                left,_,_=vector_predict(core,model.shared_head,None,'F0',caches['val'],ix)
                right,a,_=vector_predict(core,model.shared_head,candidate,job,caches['val'],ix)
                require(bool((a==.5).all()) and core.torch.allclose(left,right,atol=1e-5,rtol=1e-5)
                        and bool((left.argmax(-1)==right.argmax(-1)).all()),'Vector initial outputs differ')
                max_error=max(max_error,float((left-right).abs().max()))
        del candidate
        write(args.output/'initial_equivalence.json',{'samples':13446,'max_abs_error':max_error,'changed_predictions':0})
        probe=functions['probe_fusion'](core,fusion,model,job,caches['train'],device,start)
        if job=='V2':require(any(r['fusion_gradients_l1']['0.weight']>0 for r in probe['updates'][1:]),'V2 hidden layer has no gradient')
        write(args.output/'probe.json',probe)
        metrics,_=functions['train'](core,args,fusion,model,job,caches,device,start,f0,baseline,binding,probe)
        require(core.state_hash(model.state_dict().items())==frozen_hash,'Frozen state changed')
        result={'metrics':metrics,'F0':f0,'training_evidence':training_evidence(args.output/job,metrics,f0['macro_f1']),
                'frozen_model_unchanged':True}
    write(args.output/'result.json',{'job':job,'binding':binding,**result,'test_evaluated':False})


def screening(n0,v1,v2):
    a,b=100*(v2-n0),100*(v2-v1)
    return {'V2_minus_N0_pp':a,'V2_minus_V1_pp':b,'budget_threshold_passed':a>=.10 and b>=.05,
            'class_tradeoff_review_required':True,'statistical_significance_claimed':False,'automatic_followup':False}


def analyze(args,refs,binding):
    results={job:read(args.output/job/'result.json') for job in JOBS}
    require(all(r['binding']==binding for r in results.values()),'Job binding differs')
    for job,result in results.items():
        status=read(args.output/job/'status.json');require((status['status'],status['exit_code'])==('completed',0),'Job incomplete')
        folder=args.output/job/('E0FT' if job.startswith('E0FT') else job);m=result['metrics']
        require(read(folder/'metrics.json')==m,'Metric records differ');verify_metrics(m['best_validation'])
        for file,digest in m['checkpoint_sha256'].items():require(sha(folder/file)==digest,'Checkpoint changed')
        require(sha(folder/'validation_predictions.jsonl.gz')==m['predictions_sha256'],'Predictions changed')
    rows={}
    for seed in (42,43,44):
        e0=results[f'E0FT{seed}']['metrics'];n0=refs[str(seed)]['metrics']
        common=min(len(e0['orders']),len(n0['orders']))
        require(e0['orders'][:common]==n0['orders'][:common],'E0-FT/N0 data order differs')
        rows[str(seed)]={'N0':n0['best_validation'],'E0_FT':e0['best_validation'],
            'N0_minus_E0_FT_pp':100*(n0['best_validation']['macro_f1']-e0['best_validation']['macro_f1']),
            'N0_gain_pp':n0['macro_f1_minus_original_pp'],'E0_FT_gain_pp':e0['macro_f1_minus_original_pp'],
            'N0_updates':n0['optimizer_updates'],'E0_FT_updates':e0['optimizer_updates']}
    first,second=results['V1']['metrics'],results['V2']['metrics'];common=min(len(first['orders']),len(second['orders']))
    require(first['orders'][:common]==second['orders'][:common] and first['training_start_rng_sha256']==second['training_start_rng_sha256'],'Vector pair differs')
    values=[r['N0_minus_E0_FT_pp'] for r in rows.values()]
    out={'status':'completed','exit_code':0,'training_jobs':5,'binding':binding,'results':results,'fair_comparison':rows,
        'paired_N0_minus_E0_FT':{'mean_pp':statistics.mean(values),'sample_std_pp':statistics.stdev(values)},
        'vector_screening':screening(refs['42']['metrics']['best_validation']['macro_f1'],
                                    first['best_validation']['macro_f1'],second['best_validation']['macro_f1']),
        'V2_versus_V1':pair.compare_predictions(predictions(args.output/'V1/V1/validation_predictions.jsonl.gz'),
                                               predictions(args.output/'V2/V2/validation_predictions.jsonl.gz')),
        'test_evaluated':False,'automatic_followup':False}
    write(args.output/'summary.json',out)
    lines=['# Fair fine-tuning and vector fusion','',
        '| seed | E0-FT Macro-F1 | N0 Macro-F1 | N0-E0-FT pp |','|---|---:|---:|---:|']
    for seed,r in rows.items():lines.append(f"| {seed} | {100*r['E0_FT']['macro_f1']:.6f}% | {100*r['N0']['macro_f1']:.6f}% | {r['N0_minus_E0_FT_pp']:+.6f} |")
    lines+=['','| N0-s42 | V1 | V2 | V2-N0 pp | V2-V1 pp |','|---:|---:|---:|---:|---:|',
        f"| {100*refs['42']['metrics']['best_validation']['macro_f1']:.6f}% | {100*first['best_validation']['macro_f1']:.6f}% | {100*second['best_validation']['macro_f1']:.6f}% | {out['vector_screening']['V2_minus_N0_pp']:+.6f} | {out['vector_screening']['V2_minus_V1_pp']:+.6f} |",'',
        'Budget thresholds: V2-N0 >= 0.10 pp and V2-V1 >= 0.05 pp. Class tradeoffs still require review.',
        'No statistical significance claim, no automatic repetitions, no test evaluation.']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path.cwd());parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--mode',choices=('check','queue','job'),default='check');parser.add_argument('--job',choices=JOBS)
    parser.add_argument('--physical-gpu',type=int,default=1);parser.add_argument('--allow-shared-gpu',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();args.repo=args.repo.resolve();args.config=(args.repo/args.config).resolve()
    spec,reg,n0,fusion,refs,binding=metadata(args);e0_trainer()
    if args.mode=='check':print(json.dumps({'status':'metadata_checks_passed','training_started':False,'binding':binding}));return
    require(os.name=='posix' and args.output is not None,'Linux wx/new output required')
    args.output=inside(args.repo/'outputs',inside(args.repo,args.output));require(not args.output.exists(),'Refuse overwrite/resume')
    args.output.mkdir(parents=True);write(args.output/'status.json',{'status':'running'})
    try:
        if args.mode=='job':require(args.job in JOBS,'Choose job');one(args,args.job,reg,n0,fusion,refs,binding)
        else:
            for job in JOBS:
                cmd=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--repo',str(args.repo),'--config',str(args.config),
                     '--mode','job','--job',job,'--physical-gpu',str(args.physical_gpu),'--output',str(args.output/job)]
                if args.allow_shared_gpu:cmd.append('--allow-shared-gpu')
                write(args.output/'progress.json',{'job':job,'training_jobs':5})
                require(subprocess.run(cmd).returncode==0,'Job failed; queue stopped')
            analyze(args,refs,binding)
        write(args.output/'status.json',{'status':'completed','exit_code':0})
        write(args.output/'progress.json',{'status':'completed','training_jobs':5})
    except BaseException:
        write(args.output/'status.json',{'status':'failed','exit_code':1})
        (args.output/'error.log').write_text(traceback.format_exc(),encoding='utf-8');raise


if __name__=='__main__':main()
