"""Import locked seed42, train paired readouts on independent C43/C44, then aggregate."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import time
import traceback

from filelock import FileLock
import torch
import torch_geometric

import run_joint_transition_fusion as parent
import fixed_transition_training as training
import fixed_transition_results as results
import fixed_transition_archive as archive
from xgnid.fixed_transition_stability import ARMS, MODELS

raw, frozen, original, require=parent.raw,parent.frozen,parent.original,parent.require
OUTPUT_NAME='fixed_transition_stability_v1'
OUTPUT='outputs/'+OUTPUT_NAME
CONFIG=raw.ROOT/('configs/'+OUTPUT_NAME+'.json')
SOURCE_FILES=(*parent.SOURCE_FILES,'src/xgnid/fixed_transition_stability.py',
    'scripts/run_fixed_transition_stability.py','scripts/fixed_transition_training.py',
    'scripts/fixed_transition_results.py','scripts/fixed_transition_archive.py',
    'scripts/launch_fixed_transition_stability.py','scripts/launch_joint_transition_fusion.py',
    'scripts/run_transition_pair_cross_replay.py','configs/fixed_transition_stability_v1.json',
    'configs/early_baselines_final_test_v1.json','configs/independent_packet_transition_v1.json',
    'configs/ordered_transition_readout_200.json','configs/packet_transition_readout_s42_v1.json')


def validate_settings(settings):
    fixed={'backbone_seeds':[42,43,44],'new_training_backbone_seeds':[43,44],'readout_seed':42,'sampling_seed':42,
        'epochs':200,'batch_size':512,'eval_batch_size':16,'training_samples':107621,'validation_samples':13446,
        'test_samples':13448,'updates_per_epoch':211,'updates_per_arm':42200,'new_training_runs':6,
        'total_new_updates':253200,'optimizer':'Adam','adam_betas':[0.9,0.999],'adam_eps':1e-8,'weight_decay':0.0,
        'readout_learning_rate':0.001,'fusion_learning_rate':0.0001,'loss':'plain_final_eight_class_NLL',
        'selection':'best_validation_macro_f1_including_zero_earliest_tie','fixed_injection_strength':1.0,
        'pairing':'true_continuous','partitions':['train','val'],'test_evaluated':False,
        'checkpoint_every_paired_update':True,'track_training_seconds':True,
        'trainable_parameters':{'frozen_single':3200,'joint_single':10344,'joint_ordered':10872}}
    require(all(settings[k]==v for k,v in fixed.items()) and settings['arms']==list(ARMS),'Registered stability protocol changed')
    for arm,ref,epoch in (('frozen_single','944ba61faca474d02c409c2038e02f33e5da3d50',94),
                         ('joint_single','ac00f1f319aec1b1079e15ca5609c2ec08c85ba9',59),
                         ('joint_ordered','7c40c4b6772679d51b20a6a268fc8b4ae2f3d518',191)):
        require(settings['historical_seed42'][arm]['source_result_commit']==ref
                and settings['historical_seed42'][arm]['selected_epoch']==epoch,'Locked seed42 result changed')
    require(len(training.parent.PairedSampler(107621,512,42))==211,'Wrong update budget')
    sources=settings['independent_sources']
    require(set(sources)=={'43','44'} and len({v['n0']['sha256']['best.pt'] for v in sources.values()})==2
            and len({v['frozen_c_sha256'] for v in sources.values()})==2,'Independent backbone registrations overlap')


def check_source_identity(seed,binding,selected,spec,settings):
    require(seed in (43,44) and binding['backbone_seed']==seed==spec['seed']
        and binding['frozen_c_sha256']==spec['frozen_c_sha256'] and binding['split_id']==settings['split_id']
        and binding['manifest_sha256']==settings['manifest_sha256'],'Independent source C identity mismatch')
    expected=spec['C_selection']
    require(all(selected[k]==expected[k] for k in ('epoch','validation','checkpoint_sha256','predictions_sha256','prediction_ids_sha256'))
        and binding['validation_digest']==expected['prediction_ids_sha256'],'Independent source selection differs')


def check_source_files(repo,spec):
    folder=repo/spec['source_directory']
    inputs={folder/name:meta['sha256'] for name,meta in spec['inputs'].items()}
    inputs.update({repo/spec['n0']['directory']/name:sha for name,sha in spec['n0']['sha256'].items()})
    frozen.evaluation.verify_inputs(inputs)
    selection=raw.read(folder/'selection.json')
    require(selection['n0_best_sha256']==spec['n0']['sha256']['best.pt'] and selection['seed']==spec['seed'],
            'Registered N0/C source chain changed')
    return inputs


def raw_probe(source,data,rows,config,settings,device,output):
    candidates=raw.choose_rows(data,'train',per_class=16)
    classes=[[r for r in candidates if r['label']==i] for i in range(8)]
    candidates=[group[j] for j in range(max(map(len,classes))) for group in classes if j<len(group)]
    for _,_,batch,expected in original.diagnosis.aligned_batches(data,candidates,'train',config):
        features,_=frozen.encode(source,batch,device)
        if not bool(features.transitions[:,70].any()): continue
        receipt=training.gradient_probe(source,features,batch.y.view(-1).to(device),settings,device)
        receipt.update(sample_ids=[r['sample_id'] for r in expected],mode='real_graph_preflight',device=str(device),
                       formal_updates=0)
        raw.write(output/'raw_gradient_probe.json',receipt)
        return receipt
    raise ValueError('No continuous training context for staged gradient probe')


@torch.no_grad()
def selected_raw_replay(source,arms,data,rows,config,device,output,state):
    for name,arm in arms.items():
        arm.load_state_dict(state['best'][name]['state'],strict=True)
        arm.requires_grad_(False).eval()
    reference=training.parent.JointTransitionFusion(source.fusion,'joint_base',42).to(device).requires_grad_(False).eval()
    records=[]
    for step,indices,batch,expected in original.diagnosis.aligned_batches(data,rows,'val',config):
        features,_=frozen.encode(source,batch,device)
        records.extend(results.diagnostics.observe({'base_C':reference,**arms},features,expected))
        if step==1 or step%200==0 or indices.stop==len(rows):
            raw.progress(output,'best_stability_raw_validation_replay',samples=indices.stop,total=len(rows))
    exported=results.diagnostics.exports(records);receipts={}
    for name in ARMS:
        metric=frozen.joint.study.validate_predictions(exported[name],rows,probabilities=False)
        digest=frozen.joint.prediction_digest(exported[name])
        require(metric==state['best'][name]['validation'] and digest==state['best'][name]['prediction_ids_sha256'],
                name+' best original-graph replay differs')
        receipts[name]={'passed':True,'epoch':state['best'][name]['epoch'],'samples':len(rows),'prediction_ids_sha256':digest}
    raw.write(output/'selected_validation_replay.json',receipts)
    return records


def verify_training_state(state,settings,source_fusion_digest):
    require(state['completed_epoch']==settings['epochs'] and state['updates']==settings['updates_per_arm'],
            'Incomplete registered training budget')
    sampler=training.parent.PairedSampler(settings['training_samples'],settings['batch_size'],settings['sampling_seed'])
    history=state['history']
    require([r['epoch'] for r in history]==list(range(settings['epochs']+1)),'Incomplete validation ledger')
    for epoch,row in enumerate(history[1:],1):
        sampler.start(epoch)
        require(row['order_sha256']==sampler.order_sha256 and row['samples_per_arm']==settings['training_samples']
                and row['updates_per_arm']==len(sampler) and row['cumulative_updates_per_arm']==epoch*len(sampler),
                'Sampling or update budget differs')
        sampler.cursor=sampler.size
    frozen_fusion={k.removeprefix('fusion.'):v for k,v in state['models']['frozen_single'].items() if k.startswith('fusion.')}
    require(raw.state_digest(frozen_fusion)==source_fusion_digest,'Frozen Single fusion changed during training')
    for name in ARMS:
        best=max(history,key=lambda r:r['validation'][name]['macro_f1'])
        require(state['best'][name]['epoch']==best['epoch'],'Best selection or earliest tie rule differs')
        require(all(int(v['step'])==state['updates'] for v in state['optimizers'][name]['state'].values()),
                'Adam update state differs')
    return {'passed':True,'same_sample_order_all_three_arms':True,'epochs_per_arm':state['completed_epoch'],
            'updates_per_arm':state['updates'],'all_sampling_orders_verified':True,'frozen_single_fusion_unchanged':True}


def run_seed(repo,graph_root,root,settings,lock,seed,resume=False,probe_only=False):
    require(seed in settings['new_training_backbone_seeds'],'Seed42 is import-only')
    output=root/('seed'+str(seed));output.mkdir(parents=True,exist_ok=True)
    spec=settings['independent_sources'][str(seed)]
    registered=check_source_files(repo,spec)
    if (output/'artifact_manifest.json').exists():
        completed_binding=raw.read(output/'provenance.json')['binding']
        require(completed_binding['protocol_lock_sha256']==archive.canonical_sha(lock),
                'Completed backbone has another protocol')
        require(graph_root is None or str(graph_root.resolve())==completed_binding['graph_root'],
                'Completed backbone graph root differs')
        frozen.evaluation.verify_inputs({repo/name:sha for name,sha in completed_binding['input_sha256'].items()})
        archive.verify_sealed(output)
        return {'status':'completed','backbone_seed':seed,'reused_completed_training':True,'new_updates_on_this_invocation':0}
    started=time.monotonic()
    source,data,rows,device,config,selected,gold,inputs,binding=frozen.load_source(repo,seed,graph_root,output)
    check_source_identity(seed,binding,selected,spec,settings)
    inputs.update(registered)
    require(len(rows['train'])==settings['training_samples'] and len(rows['val'])==settings['validation_samples']
            and data.metadata['partition_sizes']['test']==settings['test_samples'],'Split counts differ')
    graph_counts=frozen.independent.d32.guard_graph_loads(data,{'train','val'})
    config.update(batch_size=settings['batch_size'],eval_batch_size=settings['eval_batch_size'])
    digest=raw.state_digest(source.state_dict());fusion_digest=raw.state_digest(source.fusion.state_dict())
    environment={'python':platform.python_version(),'torch':str(torch.__version__),'pyg':torch_geometric.__version__,
        'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(device) if device.type=='cuda' else str(device),
        'platform':platform.platform()}
    run_binding={'protocol_lock_sha256':archive.canonical_sha(lock),'source':binding,'code_commit':lock['code_commit'],
        'environment':environment,'graph_root':str(data.root),'workers':config['workers'],
        'input_sha256':{str(p.relative_to(repo)):sha for p,sha in inputs.items()},'backbone_seed':seed}
    path=output/'provenance.json'
    if path.exists():
        require(resume and raw.read(path)['binding']==run_binding,'Existing backbone cache/config/source/environment differs')
    raw.write(path,{'binding':run_binding,'test_graph_loads':0,'readout_seed':42,'sampling_seed':42})
    raw.write(output/'config.json',settings)
    raw_probe(source,data,rows,config,settings,device,output)
    training.parent.verify_source(source,digest)
    val,reference,coverage_val=parent.cached_split(source,data,rows['val'],'val',config,device,output,run_binding)
    parent.verify_source_replay(reference,gold,selected,rows['val'],output)
    arms=training.make_arms(source,settings,device)
    parent.initial_replay(source,arms,val,rows['val'],gold,selected,device,settings,output)
    raw.write(output/'server_probe.json',{'passed':True,'backbone_seed':seed,'full_epoch_zero_replay':True,
        'real_graph_gradient_paths':True,'probe_updates_discarded':True,'formal_training_started':False})
    if probe_only:
        training.parent.verify_source(source,digest)
        frozen.evaluation.verify_inputs(inputs)
        return {'status':'server_probe_passed','backbone_seed':seed,'formal_training_started':False,'test_evaluated':False}
    train,_,coverage_train=parent.cached_split(source,data,rows['train'],'train',config,device,output,run_binding)
    raw.write(output/'coverage.json',{'train':coverage_train,'val':coverage_val})
    run_binding={**run_binding,'frozen_cache_sha256':{p:raw.sha(output/('frozen_'+p+'_features.pt')) for p in ('train','val')}}
    raw.seed_all(settings['sampling_seed'])
    state=training.fit(arms,train,val,rows['val'],reference,settings,run_binding,device,output,resume=resume)
    budget=verify_training_state(state,settings,fusion_digest)
    del train,val
    records=selected_raw_replay(source,arms,data,rows['val'],config,device,output,state)
    parent.verify_source_replay(results.diagnostics.exports(records)['base_C'],gold,selected,rows['val'],output)
    counts=archive.parameter_counts(source,settings)
    metadata={'base_C':{**counts['base_C'],'best_epoch':selected['epoch'],'last_epoch':selected['epoch'],
        'best_validation':selected['validation'],'last_validation':selected['validation'],'new_updates':0,
        'training_seconds':0.0,'role':'fixed original C reference'}}
    (output/'base_C').mkdir(exist_ok=True)
    (output/'base_C'/'best.pt').write_bytes((repo/spec['source_directory']/'C/best.pt').read_bytes())
    for name in ARMS:
        metadata[name]={**counts[name],'best_epoch':state['best'][name]['epoch'],'last_epoch':state['completed_epoch'],
            'best_validation':state['best'][name]['validation'],'last_validation':state['history'][-1]['validation'][name],
            'new_updates':state['updates'],'training_seconds':state['training_seconds'][name],
            'training_seconds_scope':'committed optimization compute; synchronized CUDA; excludes cache/evaluation/I/O and discarded retries',
            'role':'new_independent_backbone_replication'}
    histories={name:[{**r,'validation':r['validation'][name]} for r in state['history']] for name in ARMS}
    results.write_seed_results(output,records,rows['val'],metadata,binding,settings,histories,'new_replication',seed)
    training.parent.verify_source(source,digest);frozen.evaluation.verify_inputs(inputs)
    require(graph_counts['test']==0,'Unexpected test graph load')
    raw.write(output/'final_verification.json',{**budget,'source_C_parameters_buffers_modes_unchanged':True,
        'source_inputs_unchanged':True,'best_raw_graph_replays_passed':True,'test_graph_loads':0,
        'wall_seconds_this_invocation':time.monotonic()-started})
    raw.write(output/'status.json',{'status':'completed','backbone_seed':seed,'formal_training_started':True,'test_evaluated':False})
    archive.seal(output)
    return {'status':'completed','backbone_seed':seed,'formal_training_started':True,'test_evaluated':False,
            'updates_per_arm':state['updates']}


def run_phase(repo,graph_root,root,settings,phase,resume=False,probe_only=False,download=False):
    validate_settings(settings)
    lock=archive.lock_protocol(repo,root,settings,SOURCE_FILES,download)
    if phase=='import':
        archive.import_seed42(repo,root,settings,lock,download)
        completed=[s for s in (43,44) if (root/('seed'+str(s))/'artifact_manifest.json').is_file()]
        return {'status':'import_completed','formal_training_started':any((root/('seed'+str(s))/'recovery.pt').exists() for s in (43,44)),
                'new_training_runs_completed':3*len(completed),'new_backbones_pending':[s for s in (43,44) if s not in completed],
                'test_evaluated':False}
    require((root/'seed42_import/artifact_manifest.json').is_file(),'Import locked seed42 before new training')
    if phase in ('seed43','seed44'):
        if phase=='seed44' and not probe_only:
            archive.verify_sealed(root/'seed43')
        return run_seed(repo,graph_root,root,settings,lock,int(phase[-2:]),resume,probe_only)
    require(phase=='aggregate','Unexpected phase')
    require(raw.read(root/'seed42_import/provenance.json')['protocol_lock_sha256']==archive.canonical_sha(lock),
            'Imported protocol differs from aggregate lock')
    summaries={42:archive.verify_sealed(root/'seed42_import')}
    for seed in (43,44):
        require(raw.read(root/('seed'+str(seed))/'provenance.json')['binding']['protocol_lock_sha256']==archive.canonical_sha(lock),
                'Backbone protocol differs from aggregate lock')
        summaries[seed]=archive.verify_sealed(root/('seed'+str(seed)))
        require(summaries[seed]['new_updates_total']==3*settings['updates_per_arm'],'New training budget incomplete')
    result=results.aggregate(summaries,root/'aggregate')
    require(result['new_updates_total']==settings['total_new_updates'],'Wrong total new update budget')
    raw.write(root/'summary.json',result)
    (root/'report.md').write_text((root/'aggregate/report.md').read_text(encoding='utf-8'),encoding='utf-8')
    raw.write(root/'final_verification.json',{'passed':True,'new_training_runs':6,'new_updates_total':253200,
        'all_best_raw_graph_replays_passed':True,'independent_prediction_audits_passed':True,'test_graph_loads':0})
    return {'status':'completed','formal_training_started':True,'new_training_runs':6,'new_updates_total':253200,'test_evaluated':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path.cwd());parser.add_argument('--data-root',type=Path)
    parser.add_argument('--phase',choices=('import','seed43','seed44','aggregate'),required=True)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--probe-only',action='store_true')
    parser.add_argument('--download-inputs',action='store_true')
    args=parser.parse_args();repo=args.repo.resolve();root=repo/OUTPUT
    root.mkdir(parents=True,exist_ok=True);(repo/'tmp').mkdir(exist_ok=True)
    with FileLock(str(repo/'tmp'/(OUTPUT_NAME+'.run.lock')),timeout=0):
        try:
            raw.write(root/'status.json',{'status':'running','phase':args.phase,'queue_completed':False,'test_evaluated':False})
            result=run_phase(repo,args.data_root,root,raw.read(CONFIG),args.phase,args.resume,args.probe_only,args.download_inputs)
        except BaseException as exc:
            failure={'status':'failed','phase':args.phase,'queue_completed':False,'exception':repr(exc),
                     'traceback':traceback.format_exc(),'test_evaluated':False}
            raw.write(root/'status.json',failure)
            folder=root/'failures';folder.mkdir(exist_ok=True)
            raw.write(folder/(args.phase+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'),failure)
            raise
        else:
            status={**result,'phase':args.phase,'queue_completed':args.phase=='aggregate' and result['status']=='completed'}
            if args.phase in ('seed43','seed44') and result['status']=='completed':
                status.update(status='backbone_completed',backbone_status='completed')
            raw.write(root/'status.json',status)
            print(json.dumps({**result,'phase':args.phase,'output':str(root)}),flush=True)


if __name__=='__main__': main()
