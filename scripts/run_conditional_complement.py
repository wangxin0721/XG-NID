"""Preregistered conditional complement queue; no test access before both gates pass."""
import argparse
from datetime import datetime, timezone
import json
import signal
import time
import traceback
from pathlib import Path

from filelock import FileLock
import torch
from torch_geometric.loader import DataLoader

import prepare_conditional_complement as prepare
import conditional_complement_training as training
import summarize_conditional_complement as report
from xgnid.conditional_complement import ARMS, ConditionalComplement, Preparation, project, candidate_loss

raw,frozen,require,archive=prepare.raw,prepare.frozen,prepare.require,prepare.archive


def decision(root):
    result=raw.read(root/'seed42'/'summary.json')['decision']
    require(result['passed'],'Seed42 gate failed; no replication/test permitted')
    return result


def verify_budget(state,settings,names):
    sampler=training.transaction.PairedSampler(settings['training_samples'],settings['batch_size'],settings['sampling_seed'])
    require(state['completed_epoch']==200 and state['updates']==42200,'Incomplete candidate budget')
    require([r['epoch'] for r in state['history']]==list(range(201)),'Incomplete validation ledger')
    for row in state['history'][1:]:
        sampler.start(row['epoch'])
        require(row['order_sha256']==sampler.order_sha256 and row['updates_per_arm']==211
                and row['samples_per_arm']==107621 and row['cumulative_updates_per_arm']==row['epoch']*211,
                'Order/update budget differs')
        sampler.cursor=sampler.size
    for n in names:
        best=max(state['history'],key=lambda r:r['validation'][n]['macro_f1'])
        require(state['best'][n]['epoch']==best['epoch'],'Earliest-best selection differs')
        require(all(int(v['step'])==42200 for v in state['optimizers'][n]['state'].values()),'Adam final cursor differs')


@torch.no_grad()
def raw_replay(source,references,arms,prep,data,rows,config,device,output,part='val'):
    for m in (*references.values(),*arms.values(),prep):m.requires_grad_(False).eval()
    source_arm=ConditionalComplement(references['joint_ordered'].fusion,prep,'cc_t0').to(device).requires_grad_(False).eval()
    if part=='test':
        batches=DataLoader(frozen.evaluation.TestGraphs(data,rows),batch_size=16,shuffle=False,
            num_workers=config['workers'],generator=torch.Generator().manual_seed(242))
    else:batches=frozen.joint.legacy.loader(data,rows,'val',config)
    records=[]
    for step,batch in enumerate(batches,1):
        start=len(records);stop=start+batch.num_graphs;expected=rows[start:stop]
        require(list(zip(batch.sample_id,batch.y.view(-1).tolist()))==[(r['sample_id'],r['label']) for r in expected], 'Raw replay identities differ')
        features,_=frozen.encode(source,batch,device)
        projected=project(references['joint_ordered'],features)
        refs={n:m(features)[0].argmax(-1).tolist() for n,m in references.items()}
        chunk=prepare.observe({'source_ordered':source_arm,**arms},prep,projected,expected,refs)
        for r in chunk:
            base=r['diagnostics']['source_ordered']
            for n in arms:
                d=r['diagnostics'][n]
                d['alpha_delta_mean']=sum(a-b for a,b in zip(d['alpha'],base['alpha']))/32
                d['alpha_delta_l1_mean']=sum(abs(a-b) for a,b in zip(d['alpha'],base['alpha']))/32
        records.extend(chunk)
        if step==1 or step%200==0 or stop==len(rows):raw.progress(output,'best_conditional_raw_'+part+'_replay',samples=stop,total=len(rows))
    require(len(records)==len(rows),'Incomplete raw replay')
    return records


def parameter_costs(source,references,arm):
    source_total=sum(p.numel() for p in source.parameters())+sum(p.numel() for p in references['joint_ordered'].readout.parameters())
    common=sum(p.numel() for m in (arm.gate,arm.shared_head) for p in m.parameters())
    trainable=sum(p.numel() for p in arm.parameters() if p.requires_grad)
    # Called before replay freezes candidates.
    conditional=arm.arm in ('cc_t2','cc_t4')
    return {'full_inference_parameters':source_total+sum(p.numel() for p in arm.parameters())-common,
        'trainable_parameters':trainable,'correction_trainable_parameters':trainable-common,
        'frozen_predictor_parameters':2112 if conditional else 0,'preparation_total_parameters':2640,
        'added_linear_MACs_per_sample':(2048+512+72*32+32*32) if conditional else 64*40+40*32 if arm.correction is not None else 0,
        'MAC_scope':'new affine/multiplicative correction only; excludes normalization/activation and shared source encoder/readout',
        'multiplications_rank_interaction':8 if conditional else 0}


@torch.no_grad()
def inference_benchmark(arms,cache,device):
    f,_=cache.batch(range(min(512,len(cache.labels))),device);result={}
    for n,m in arms.items():
        m.eval()
        for _ in range(10):m(f)
        if device.type=='cuda':torch.cuda.synchronize(device)
        started=time.perf_counter()
        for _ in range(50):m(f)
        if device.type=='cuda':torch.cuda.synchronize(device)
        result[n]={'seconds_per_batch':(time.perf_counter()-started)/50,'batch_size':len(f.flow),
            'scope':'cached-coordinate gate/head/correction only; graph source common and excluded','repeats':50}
    return result


def run_seed(repo,data_root,root,settings,protocol,inputs,manifest,seed,resume=True,prepare_only=False,probe_only=False):
    names=list(ARMS) if seed==42 else decision(root)['next_arms']
    output=root/('seed'+str(seed))
    if (output/'artifact_manifest.json').exists():
        require(resume,'Completed phase exists; use resume')
        summary=archive.verify_sealed(output)
        require(raw.read(output/'provenance.json')['protocol_sha256']==archive.canonical_sha(protocol), 'Completed phase protocol differs')
        return {'status':'completed','reused_completed_phase':True,'backbone_seed':seed,'test_evaluated':False}
    started=time.perf_counter()
    output,source,references,data,rows,device,config,paths,binding,before,guard=prepare.context(repo,data_root,root,settings,protocol,inputs,seed)
    caches={};refs={}
    # Verify source predictions and IDs on all validation flows before preparation.
    caches['val'],refs['val']=prepare.projected_cache(source,references,data,rows['val'],'val',config,device,output,binding)
    receipts={}
    for name,spec in manifest['seeds'][str(seed)]['models'].items():
        expected=prepare.locked.stripped(inputs.rows(spec['validation_predictions_path']))
        actual=frozen.original.prediction_records(rows['val'],refs['val'][name])
        receipts[name]=prepare.fixed_test.compare_replay(actual,expected,output,name+'_source_validation_replay.json')
    raw.write(output/'source_validation_replay.json',receipts)
    caches['train'],refs['train']=prepare.projected_cache(source,references,data,rows['train'],'train',config,device,output,binding)
    binding={**binding,'cache_sha256':{p:raw.sha(output/('projected_'+p+'.pt')) for p in ('train','val')}}
    prep,diagnosis=prepare.prepare(caches,rows,refs,references['joint_ordered'].fusion,output,settings,binding,device,resume)
    prep_digest=raw.state_digest(prep.state_dict())
    prepare.locked.verify_frozen(source,references,before)
    frozen.evaluation.verify_inputs(paths)
    if prepare_only:
        return {'status':'preparation_completed','backbone_seed':seed,'preparation_updates':4220,
            'repairable_validation_disagreements':diagnosis['val']['repairable_disagreement_samples'],
            'formal_candidate_training_started':False,'test_evaluated':False}
    prepare.gradient_probe(references['joint_ordered'].fusion,prep,caches['train'],settings,device,output)
    arms=prepare.make_arms(references['joint_ordered'].fusion,prep,names,settings,device)
    zero=prepare.epoch_zero(arms,caches['val'],settings,device,output)
    raw.write(output/'server_probe.json',{'passed':True,'original_graph_source_replay':True,
        'full_epoch_zero_replay':zero,'real_coordinate_gradient_probe':True,'probe_updates_discarded':True,
        'frozen_parameters_buffers_unchanged':True,'test_graph_loads':guard['test']})
    if probe_only:return {'status':'server_probe_passed','formal_candidate_training_started':False,'test_evaluated':False}
    binding={**binding,'preparation_state_sha256':prep_digest,'arms':names}
    def loss(name,model,features,labels):return candidate_loss(model,features,labels,prep)
    def evaluate(models):return training.measure(models,caches['val'],settings,device)
    state=training.fit(arms,caches['train'],settings,binding,device,output,loss,evaluate,resume=resume)
    verify_budget(state,settings,names)
    metadata={}
    for n,m in arms.items():
        metadata[n]={**parameter_costs(source,references,m),'best_epoch':state['best'][n]['epoch'],
            'last_epoch':state['completed_epoch'],'last_validation':state['history'][-1]['validation'][n],
            'updates':state['updates'],'training_seconds':state['training_seconds'][n]}
        m.load_state_dict(state['best'][n]['state'],strict=True)
    benchmark=inference_benchmark(arms,caches['val'],device)
    for n in metadata:metadata[n]['inference_benchmark']=benchmark[n]
    records=raw_replay(source,references,arms,prep,data,rows['val'],config,device,output)
    replay={}
    for n in names:
        values=[r['predictions'][n] for r in records]
        require(values==state['best'][n]['predictions'],n+' best raw predictions differ')
        report.check_metric(report.metrics(records,n),state['best'][n]['validation'])
        replay[n]={'passed':True,'samples':len(records),'best_epoch':state['best'][n]['epoch'],
                  'prediction_digest':archive.canonical_sha(values)}
    raw.write(output/'selected_validation_replay.json',replay)
    for name,key in (('joint_ordered','source_ordered'),('frozen_single','frozen_single'),('base_C','base_C')):
        require([r['predictions'][key] for r in records]==refs['val'][name],'Reference changed in best replay')
        metadata[key]={'best_epoch':settings['seeds'][str(seed)]['checkpoints'][name]['epoch'],'updates':0}
    require(raw.state_digest(prep.state_dict())==prep_digest,'Frozen preparation changed')
    for m in arms.values():
        if hasattr(m,'flow_to_packet'):
            require(raw.state_digest(m.flow_to_packet.state_dict())==raw.state_digest(prep.flow_to_packet.state_dict())
                and raw.state_digest(m.packet_to_flow.state_dict())==raw.state_digest(prep.packet_to_flow.state_dict()), 'Frozen arm predictor changed')
    prepare.locked.verify_frozen(source,references,before);frozen.evaluation.verify_inputs(paths)
    require(guard['test']==0,'Development loaded test graphs')
    summary=report.write_result(output,records,metadata,settings,seed,state['history'])
    raw.write(output/'final_verification.json',{'passed':True,'epochs':200,'updates_per_arm':42200,
        'total_candidate_updates':len(names)*42200,'preparation_updates':4220,'best_raw_replays_passed':True,
        'frozen_source_and_preparation_unchanged':True,'test_graph_loads':0,
        'wall_seconds_this_invocation':time.perf_counter()-started})
    raw.write(output/'status.json',{'status':'completed','backbone_seed':seed,'test_evaluated':False})
    archive.seal(output)
    return {'status':'completed','backbone_seed':seed,'formal_candidate_training_started':True,
        'test_evaluated':False,'decision':summary.get('decision'),'candidate_updates':len(names)*42200}


def test_lock(root,settings,protocol):
    d=decision(root);rep=raw.read(root/'aggregate'/'summary.json')
    require(rep['passed'] and rep['candidate']==d['candidate'],'Replication gate failed; no fixed test permitted')
    names=['cc_t0',d['matched_control'],d['candidate']];seeds={}
    for seed in (42,43,44):
        folder=root/('seed'+str(seed));summary=archive.verify_sealed(folder)
        seeds[str(seed)]={'models':{n:{'file':str((folder/n/'best.pt').relative_to(root)),
            'sha256':raw.sha(folder/n/'best.pt'),'epoch':summary['metadata'][n]['best_epoch'],
            'validation':summary['metrics'][n]} for n in names},
            'preparation_sha256':raw.sha(folder/'preparation/common/last.pt')}
    lock={'seeds':seeds,'protocol_sha256':archive.canonical_sha(protocol),
        'labels':settings['labels'],'source_manifest_sha256':settings['checkpoint_manifest_sha256'],
        'selection':'locked_validation_best_no_test_reselection','historical_test_split_used':True}
    prepare.immutable(root/'final_checkpoint_lock.json',lock)
    return {'status':'final_checkpoints_locked','test_evaluated':False,'models':9}


def fixed_test_seed(repo,data_root,root,settings,protocol,inputs,manifest,seed):
    require(raw.read(root/'aggregate/summary.json')['passed'],'Replication must pass before test')
    lock=raw.read(root/'final_checkpoint_lock.json')
    require(lock['protocol_sha256']==archive.canonical_sha(protocol),'Final lock protocol differs')
    testout=root/'fixed_test'/('seed'+str(seed));testout.mkdir(parents=True,exist_ok=True)
    if (testout/'artifact_manifest.json').exists():
        archive.verify_sealed(testout)
        return {'status':'completed','test_evaluated':True,'backbone_seed':seed,'reused_completed_test':True}
    folder,source,references,data,rows,device,config,paths,binding,before,guard=prepare.context(repo,data_root,root,settings,protocol,inputs,seed,test=True)
    require(raw.sha(folder/'preparation/common/last.pt')==lock['seeds'][str(seed)]['preparation_sha256'],'Preparation test lock differs')
    prep=Preparation(settings['new_module_seed']).to(device)
    prep.load_state_dict(torch.load(folder/'preparation/common/last.pt',map_location='cpu',weights_only=True)['state'],strict=True)
    prep.requires_grad_(False).eval()
    spec=lock['seeds'][str(seed)]['models'];arms=prepare.make_arms(references['joint_ordered'].fusion,prep,list(spec),settings,device)
    for n,m in arms.items():
        require(raw.sha(root/spec[n]['file'])==spec[n]['sha256'],'Final weight lock differs')
        m.load_state_dict(torch.load(root/spec[n]['file'],map_location='cpu',weights_only=True)['state'],strict=True)
    inference_before={n:raw.state_digest(m.state_dict()) for n,m in {**arms,'preparation':prep}.items()}
    val=raw_replay(source,references,arms,prep,data,rows['val'],config,device,testout)
    old=frozen.joint.legacy.predictions(folder/'validation/predictions.jsonl.gz')
    require([(r['sample_id'],r['label']) for r in val]==[(r['sample_id'],r['label']) for r in old], 'Pretest validation cohort differs')
    for n in arms:
        require([r['predictions'][n] for r in val]==[r['predictions'][n] for r in old], 'Pretest original-graph replay differs')
        report.check_metric(report.metrics(val,n),spec[n]['validation'])
    raw.write(testout/'pretest_validation_replay.json',{'passed':True,'samples':len(val),'models':list(arms)})
    # Test metadata/predictions first opened only here, after the validation gates and lock.
    historical=archive.open_archive(repo,settings['post_promotion_test_archive'],True)
    test_config={**config,'partition_sizes':{'test':settings['test_samples']}}
    test_rows=frozen.evaluation.evaluation_rows(data,'test',test_config)
    require(archive.canonical_sha([(r['sample_id'],r['label']) for r in test_rows])==manifest['test_cohort_sha256'],'Fixed test identities differ')
    raw.write(testout/'test_started.json',{'checkpoint_lock_sha256':archive.canonical_sha(lock),'new_optimizer_updates':0})
    records=raw_replay(source,references,arms,prep,data,test_rows,config,device,testout,'test')
    require(inference_before=={n:raw.state_digest(m.state_dict()) for n,m in {**arms,'preparation':prep}.items()},
            'Fixed evaluation changed candidate/preparation parameters or buffers')
    for n,key in (('base_C','base_C'),('frozen_single','frozen_single'),('joint_ordered','source_ordered')):
        expected=prepare.locked.stripped(historical.rows(f'seed{seed}/test/{n}_predictions.jsonl.gz'))
        actual=[{'sample_id':r['sample_id'],'label':r['label'],'prediction':r['predictions'][key]} for r in records]
        prepare.fixed_test.compare_replay(actual,expected,testout,n+'_historical_test_replay.json')
    summary=report.write_result(testout,records,{n:{'best_epoch':v['epoch'],'updates':0} for n,v in spec.items()},settings,seed,partition='test')
    prepare.locked.verify_frozen(source,references,before);frozen.evaluation.verify_inputs(paths)
    raw.write(testout/'final_verification.json',{'passed':True,'samples':len(records),'new_optimizer_updates':0,
        'pretest_raw_validation_replay_passed':True,'historical_references_identical':True,'train_graph_loads':guard['train']})
    raw.write(testout/'status.json',{'status':'completed','test_evaluated':True});archive.seal(testout)
    return {'status':'completed','test_evaluated':True,'backbone_seed':seed,'new_optimizer_updates':0}


def run(repo,data_root,root,phase,resume=True,download=False,probe_only=False):
    settings=raw.read(prepare.CONFIG)
    inputs,protocol,manifest=prepare.lock(repo,root,settings,download)
    if phase=='lock':return {'status':'source_lock_completed','models':12,'test_evaluated':False}
    if phase.startswith(('prepare','seed')):
        seed=int(phase[-2:]);require(seed in (42,43,44),'Unregistered backbone')
        if seed==44:archive.verify_sealed(root/'seed43')
        return run_seed(repo,data_root,root,settings,protocol,inputs,manifest,seed,resume,phase.startswith('prepare'),probe_only)
    if phase=='aggregate':
        d=decision(root);summaries={s:archive.verify_sealed(root/('seed'+str(s))) for s in (42,43,44)}
        result=report.aggregate(summaries,settings,d['candidate'],root/'aggregate')
        raw.write(root/'summary.json',result)
        (root/'report.md').write_bytes((root/'aggregate/report.md').read_bytes())
        return {'status':'replication_completed','passed':result['passed'],'test_evaluated':False}
    if phase=='lock_test':return test_lock(root,settings,protocol)
    if phase in ('test42','test43','test44'):return fixed_test_seed(repo,data_root,root,settings,protocol,inputs,manifest,int(phase[-2:]))
    require(phase=='test_aggregate','Unexpected phase')
    d=decision(root);summaries={s:archive.verify_sealed(root/'fixed_test'/('seed'+str(s))) for s in (42,43,44)}
    result=report.aggregate(summaries,settings,d['candidate'],root/'fixed_test/aggregate',partition='test')
    raw.write(root/'summary.json',result);(root/'report.md').write_bytes((root/'fixed_test/aggregate/report.md').read_bytes())
    return {'status':'completed','test_evaluated':True,'new_test_optimizer_updates':0}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path.cwd());parser.add_argument('--data-root',type=Path)
    parser.add_argument('--phase',required=True,choices=('lock','prepare42','prepare43','prepare44','seed42','seed43','seed44',
        'aggregate','lock_test','test42','test43','test44','test_aggregate'))
    parser.add_argument('--resume',choices=('auto','never'),default='auto')
    parser.add_argument('--download-inputs',action='store_true');parser.add_argument('--probe-only',action='store_true')
    args=parser.parse_args();repo=args.repo.resolve();root=repo/'outputs'/prepare.OUTPUT_NAME
    root.mkdir(parents=True,exist_ok=True);(repo/'tmp').mkdir(exist_ok=True)
    def interrupted(signum,frame):raise KeyboardInterrupt('Received signal '+str(signum))
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,interrupted)
    with FileLock(str(repo/'tmp'/(prepare.OUTPUT_NAME+'.run.lock')),timeout=0):
        try:
            raw.write(root/'status.json',{'status':'running','phase':args.phase})
            result=run(repo,args.data_root,root,args.phase,args.resume=='auto',args.download_inputs,args.probe_only)
        except BaseException as exc:
            failure={'status':'failed','phase':args.phase,'exception':repr(exc),'traceback':traceback.format_exc(),
                'recovery_policy':'last complete all-arm transaction; partial in-memory updates discarded'}
            raw.write(root/'status.json',failure);folder=root/'failures';folder.mkdir(exist_ok=True)
            raw.write(folder/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'),failure)
            raise
        raw.write(root/'status.json',{**result,'phase':args.phase});print(json.dumps(result),flush=True)


if __name__=='__main__':main()
