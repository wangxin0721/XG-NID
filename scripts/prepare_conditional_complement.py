"""Hash-bound source lock, frozen projection cache and fixed 20-epoch preparation."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch

import fixed_transition_test_lock as locked
import run_fixed_transition_final_test as fixed_test
import conditional_complement_training as training
import summarize_conditional_complement as report
from xgnid.conditional_complement import (Preparation, ConditionalComplement, Projected,
    ARMS, project, normalized, stream, candidate_loss)

raw, frozen, require, archive = locked.raw, locked.frozen, locked.require, locked.archive
OUTPUT_NAME='conditional_complement_v1'
CONFIG=raw.ROOT/('configs/'+OUTPUT_NAME+'.json')
NEW_FILES=('src/xgnid/conditional_complement.py','scripts/prepare_conditional_complement.py',
    'scripts/conditional_complement_training.py','scripts/run_conditional_complement.py',
    'scripts/summarize_conditional_complement.py','scripts/launch_conditional_complement.py',
    'scripts/audit_conditional_complement_archive.py','configs/conditional_complement_v1.json')
NEW_FILES=(*NEW_FILES,'scripts/launch_joint_transition_fusion.py','scripts/sync_server_outputs.py')


def immutable(path,value):
    if path.exists():require(raw.read(path)==value,'Existing source/config lock differs: '+str(path))
    else:raw.write(path,value)


def lock(repo,root,settings,download=False):
    fixed={'epochs':200,'preparation_epochs':20,'batch_size':512,'eval_batch_size':16,'rank':8,'tau':2.,
        'sd_weight':.1,'keep_weight':.1,'fusion_learning_rate':1e-4,'new_learning_rate':1e-3,
        'preparation_learning_rate':1e-3,'adam_betas':[.9,.999],'adam_eps':1e-8,'weight_decay':0.,
        'new_module_seed':42,'sampling_seed':42,'preparation_sampling_seed':42020,
        'training_samples':107621,'validation_samples':13446,'test_samples':13448,
        'updates_per_epoch':211,'updates_per_arm':42200,'screen_updates':211000,'preparation_updates':4220,
        'candidate_delta_t0_pp':.10,'candidate_delta_mlp_pp':.05,'focus_max_drop_pp':.30,
        'development_partitions':['train','val'],'arms':list(ARMS)}
    require(all(settings.get(k)==v for k,v in fixed.items()), 'Registered first-version protocol changed')
    roots=archive.open_archive(repo,settings['lock_archive'],download)
    manifest=roots.json('checkpoint_lock.json')
    require(archive.canonical_sha(manifest)==settings['checkpoint_manifest_sha256'], 'Source checkpoint lock differs')
    old=roots.json('protocol_lock.json')
    require(old['checkpoint_manifest_sha256']==settings['checkpoint_manifest_sha256']
            and old['code_commit']=='2b6690ac8600732b98365e91f6fb7b464e33912f', 'Source code/protocol differs')
    inputs=archive.open_archive(repo,settings['validation_archive'],download)
    require(tuple(manifest['labels'])==report.LABELS==tuple(settings['labels']),'Label mapping differs')
    require(len({s['source_binding']['frozen_c_sha256'] for s in settings['seeds'].values()})==3,'Backbones are not independent')
    for seed,spec in settings['seeds'].items():
        require(spec['source_binding']==manifest['seeds'][seed]['source_binding'],'Source binding differs')
        for name,item in spec['checkpoints'].items():
            source=manifest['seeds'][seed]['models'][name]
            require(all(source[k]==v for k,v in item.items()),'Wrong source checkpoint: '+seed+'/'+name)
            records=locked.stripped(inputs.rows(source['validation_predictions_path']))
            actual=frozen.joint.study.validate_predictions(records,probabilities=False)
            require(actual==source['validation'] and len(records)==settings['validation_samples'], 'Source validation differs')
            require(archive.canonical_sha([(r['sample_id'],r['label']) for r in records])==manifest['validation_cohort_sha256'],
                    'Source validation sample identity differs')
    files=tuple(dict.fromkeys((*locked.COMPUTE_FILES,*locked.NEW_FILES,*NEW_FILES)))
    hashes={n:hashlib.sha256((raw.ROOT/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for n in files}
    for name,value in old['source_files_canonical_sha256'].items():
        require(hashlib.sha256((raw.ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==value,
                'Locked source inference code differs: '+name)
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=raw.ROOT,text=True).strip()
    protocol={'schema':'xgnid.conditional-protocol.v1','settings':settings,'code_commit':code,
        'source_files_canonical_sha256':hashes,'source_checkpoint_manifest_sha256':settings['checkpoint_manifest_sha256'],
        'no_test_predictions_loaded_for_development':True}
    root.mkdir(parents=True,exist_ok=True)
    immutable(root/'source_lock.json',manifest);immutable(root/'protocol_lock.json',protocol)
    immutable(root/'config.json',settings)
    for seed,spec in settings['seeds'].items():
        for name,item in spec['checkpoints'].items():
            path=root/'source_checkpoints'/('seed'+seed)/(name+'.pt')
            if path.exists():require(raw.sha(path)==item['sha256'],'Preserved source weight changed')
            else:locked.atomic_bytes(path,inputs.data(item['path']))
    raw.write(root/'source_lock_receipt.json',{'passed':True,'models':12,'validation_samples_each':13446,
        'source_result_commit':settings['lock_archive']['source_result_commit'],
        'no_test_graph_or_prediction_access':True,'new_optimizer_updates':0})
    return inputs,protocol,manifest


def context(repo,data_root,root,settings,protocol,inputs,seed,test=False):
    output=root/('seed'+str(seed));output.mkdir(exist_ok=True)
    source,data,rows,device,config,selected,gold,paths,binding=frozen.load_source(repo,seed,data_root,output)
    require(device.type=='cuda','Formal preparation/training requires the existing GPU environment; local CPU checks are separate')
    require(binding==settings['seeds'][str(seed)]['source_binding'],'Original load_source binding differs')
    restored,models,_=locked.restore_models(inputs,settings,seed,device)
    require(raw.state_digest(restored.state_dict())==raw.state_digest(source.state_dict()), 'Archived C and load_source disagree')
    del restored
    before=locked.frozen_digests(source,models)
    config.update(batch_size=settings['batch_size'],eval_batch_size=settings['eval_batch_size'],workers=settings['workers'])
    require(len(rows['train'])==settings['training_samples'] and len(rows['val'])==settings['validation_samples'], 'Cohort counts differ')
    guard=frozen.independent.d32.guard_graph_loads(data,{'val','test'} if test else {'train','val'})
    binding={'protocol_sha256':archive.canonical_sha(protocol),'source':binding,'backbone_seed':seed,
        'source_weights':{n:s['sha256'] for n,s in settings['seeds'][str(seed)]['checkpoints'].items()},
        'graph_root':str(data.root),'environment':fixed_test.environment(device),
        'input_sha256':{str(p.relative_to(repo)):h for p,h in paths.items()},
        'sample_identity_sha256':{part:archive.canonical_sha([(r['sample_id'],r['label']) for r in rows[part]])
                                  for part in ('train','val')},'eval_batch_size':16}
    require(not {r['sample_id'] for r in rows['train']}.intersection(r['sample_id'] for r in rows['val']), 'Train/val overlap')
    immutable(output/'provenance.json',binding)
    return output,source,models,data,rows,device,config,paths,binding,before,guard


@torch.no_grad()
def projected_cache(source,models,data,rows,part,config,device,output,binding):
    require(part in ('train','val'),'Development cache cannot load test')
    path=output/('projected_'+part+'.pt');receipt_path=output/('projected_'+part+'_receipt.json')
    if path.exists() and receipt_path.exists():
        receipt=raw.read(receipt_path)
        require(receipt['binding']==binding and receipt['sha256']==raw.sha(path),'Projection cache binding/hash differs')
        saved=torch.load(path,map_location='cpu',weights_only=True)
        require(saved['binding']==binding and saved['ids']==[r['sample_id'] for r in rows]
                and saved['labels']==[r['label'] for r in rows], 'Projection cache identities differ')
    else:
        chunks=[];refs={n:[] for n in models};max_difference=0.
        for step,indices,batch,expected in locked.original.diagnosis.aligned_batches(data,rows,part,config):
            features,_=frozen.encode(source,batch,device)
            projected=project(models['joint_ordered'],features)
            original=models['joint_ordered'](features)[0]
            max_difference=max(max_difference,float((projected.reference-original).abs().max()))
            require(torch.equal(projected.reference,original),'Projected Ordered source differs')
            chunks.append(Projected(*(v.cpu() for v in projected)))
            for n,m in models.items():refs[n].extend(m(features)[0].argmax(-1).tolist())
            if step==1 or step%200==0 or indices.stop==len(rows):raw.progress(output,'project_'+part,samples=indices.stop,total=len(rows))
        saved={'binding':binding,'ids':[r['sample_id'] for r in rows],'labels':[r['label'] for r in rows],
            'features':{k:torch.cat([getattr(c,k) for c in chunks]) for k in Projected._fields},
            'reference_predictions':refs,'source_projection_max_difference':max_difference}
        training.transaction.atomic_save(path,saved)
        raw.write(receipt_path,{'binding':binding,'sha256':raw.sha(path),'partition':part,'samples':len(rows),
            'only_frozen_coordinates':True,'source_projection_max_difference':max_difference})
    return training.Cache(Projected(*(saved['features'][k] for k in Projected._fields)),saved['labels']),saved['reference_predictions']


@torch.no_grad()
def observe(arms,prep,features,rows,reference_predictions):
    qf,qp=prep.teachers(features)
    names={'joint_ordered':'source_ordered','frozen_single':'frozen_single','base_C':'base_C'}
    records=[{'sample_id':r['sample_id'],'label':r['label'],
        'teacher_group':report.group(r['label'],int(qf[i].argmax()),int(qp[i].argmax())),
        'teacher_predictions':{'flow':int(qf[i].argmax()),'packet':int(qp[i].argmax())},
        'predictions':{names[n]:p[i] for n,p in reference_predictions.items() if n in names},'diagnostics':{}}
        for i,r in enumerate(rows)]
    for name,arm in arms.items():
        scores,delta,alpha=arm(features)
        hidden=alpha*features.flow+(1-alpha)*features.packet
        conditional=None
        if hasattr(arm,'flow_to_packet'):
            vf,vp=normalized(features.flow),normalized(features.packet)
            uf,up=vf-arm.packet_to_flow(vp),vp-arm.flow_to_packet(vf)
            conditional=(uf.norm(dim=-1),up.norm(dim=-1),(arm.left(uf)*arm.right(up)).norm(dim=-1))
        for i,r in enumerate(records):
            r['predictions'][name]=int(scores[i].argmax())
            r['diagnostics'][name]={'alpha_mean':float(alpha[i].mean()),'alpha':alpha[i].tolist(),
                'delta_norm':float(delta[i].norm()),'delta_relative_norm':float(delta[i].norm()/(hidden[i].norm()+1e-8)),
                'conditional_flow_norm':float(conditional[0][i]) if conditional is not None else None,
                'conditional_packet_norm':float(conditional[1][i]) if conditional is not None else None,
                'multiplicative_norm':float(conditional[2][i]) if conditional is not None else None,
                'log_probs':scores[i].tolist()}
    # References use their own predictions; source gate details are provided by the read-only source arm.
    for r in records:
        for n in r['predictions']:
            if n not in r['diagnostics']:r['diagnostics'][n]={'alpha_mean':None,'delta_norm':None}
    return records


def preparation_diagnostics(prep,caches,rows,refs,source_arm,output,settings):
    folder=output/'preparation';folder.mkdir(exist_ok=True)
    device=next(prep.parameters()).device
    train=caches['train'];mean_flow=normalized(train.features.flow).mean(0).to(device)
    mean_packet=normalized(train.features.packet).mean(0).to(device)
    result={}
    for part,cache in caches.items():
        records=[];sums=dict(flow_to_packet=0.,packet_to_flow=0.,mean_packet=0.,mean_flow=0.)
        for start in range(0,len(cache.labels),settings['eval_batch_size']):
            stop=min(start+settings['eval_batch_size'],len(cache.labels))
            f,y=cache.batch(range(start,stop),device)
            with torch.no_grad():
                vf,vp=normalized(f.flow),normalized(f.packet)
                for key,pred,target in (('flow_to_packet',prep.flow_to_packet(vf),vp),('packet_to_flow',prep.packet_to_flow(vp),vf),
                                       ('mean_packet',mean_packet.expand_as(vp),vp),('mean_flow',mean_flow.expand_as(vf),vf)):
                    sums[key]+=float((pred-target).square().sum())
                records.extend(observe({'source_ordered':source_arm},prep,f,rows[part][start:stop],
                                       {n:p[start:stop] for n,p in refs[part].items()}))
        for r in records:r['predictions'].update(teacher_flow=r['teacher_predictions']['flow'],teacher_packet=r['teacher_predictions']['packet'])
        result[part]={'teacher_metrics':{n:report.metrics(records,n) for n in ('teacher_flow','teacher_packet')},
            'conditional_mse':{k:v/(32*len(records)) for k,v in sums.items()},
            'groups':report.subgroup(records,'source_ordered'),
            'repairable_disagreement_samples':sum(r['teacher_group'] in report.GROUPS[:2]
                and r['predictions']['source_ordered']!=r['label'] for r in records),
            'focus':{label:report.subgroup([r for r in records if r['label']==i],'source_ordered')
                     for i,label in enumerate(report.LABELS) if label in settings['focus_classes']},
            'web_recon_directions':{report.LABELS[a]+'->'+report.LABELS[b]:sum(r['label']==a and r['predictions']['source_ordered']==b for r in records)
                                    for a,b in ((1,3),(3,1))}}
        raw.save_predictions(folder/(part+'_diagnosis.jsonl.gz'),records)
    raw.write(folder/'diagnosis.json',result)
    return result


def prepare(caches,rows,refs,source_fusion,output,settings,binding,device,resume=True):
    prep=Preparation(settings['new_module_seed']).to(device)
    def losses(name,model,features,labels):
        terms=model.losses(features,labels)
        return sum(terms.values()),terms
    state=training.fit({'common':prep},caches['train'],settings,{**binding,'stage':'common_preparation'},device,
        output/'preparation',losses,preparation=True,resume=resume)
    require(state['updates']==4220 and state['completed_epoch']==20,'Common preparation budget differs')
    prep.zero_grad(set_to_none=True);prep.requires_grad_(False).eval()
    source_arm=ConditionalComplement(source_fusion,prep,'cc_t0').to(device).requires_grad_(False).eval()
    diagnosis=preparation_diagnostics(prep,caches,rows,refs,source_arm,output,settings)
    raw.write(output/'preparation'/'selection.json',{'selected_epoch':20,'selection':'fixed_final_epoch_no_validation_selection',
        'updates':4220,'total_parameters':sum(p.numel() for p in prep.parameters()),
        'teacher_parameters':528,'conditional_predictor_parameters':2112,
        'predictor_inference_linear_MACs_per_sample':2048,'teacher_inference_required':False,
        'training_seconds':state['training_seconds']['common'],
        'state_sha256':raw.state_digest(prep.state_dict())})
    require(diagnosis['val']['repairable_disagreement_samples']>0,'No repairable validation disagreements; stop before candidate training')
    return prep,diagnosis


def make_arms(fusion,prep,names,settings,device):
    arms={n:ConditionalComplement(fusion,prep,n,settings['new_module_seed']).to(device) for n in names}
    pointers=[{p.data_ptr() for p in m.parameters()} for m in arms.values()]
    require(all(not a.intersection(b) for i,a in enumerate(pointers) for b in pointers[i+1:]),'Arms share parameter storage')
    for a,b in (('cc_t1','cc_t3'),('cc_t2','cc_t4')):
        if a in arms and b in arms:require(raw.state_digest(arms[a].state_dict())==raw.state_digest(arms[b].state_dict()), 'Paired head initialization differs')
    return arms


@torch.no_grad()
def epoch_zero(arms,cache,settings,device,output):
    maxdiff={n:0. for n in arms}
    for start in range(0,len(cache.labels),16):
        f,_=cache.batch(range(start,min(start+16,len(cache.labels))),device)
        for n,m in arms.items():
            m.eval();scores,delta,_=m(f)
            require(not bool(delta.any()) and torch.equal(scores.argmax(-1),f.reference.argmax(-1)), 'Epoch0 predictions differ')
            require(torch.allclose(scores,f.reference,atol=settings['log_probability_atol'],rtol=settings['log_probability_rtol']), 'Epoch0 log probabilities differ')
            maxdiff[n]=max(maxdiff[n],float((scores-f.reference).abs().max()))
    receipt={'passed':True,'samples':len(cache.labels),'max_log_probability_difference':maxdiff,
        'atol':settings['log_probability_atol'],'rtol':settings['log_probability_rtol'], 'batch_size':16}
    raw.write(output/'epoch_zero.json',receipt);return receipt


def gradient_probe(fusion,prep,cache,settings,device,output):
    arms=make_arms(fusion,prep,ARMS,settings,device);receipt={}
    features,labels=cache.batch(range(min(512,len(cache.labels))),device)
    frozen_digest=raw.state_digest(prep.state_dict())
    for n,m in arms.items():
        opt=training.optimizer(m,settings);steps=[]
        for step in range(4):
            opt.zero_grad(set_to_none=True);loss,_=candidate_loss(m,features,labels,prep);loss.backward()
            grads={k:float(p.grad.abs().sum()) for k,p in m.named_parameters() if p.requires_grad and p.grad is not None}
            require(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None), 'Probe nonfinite gradient')
            for prefix in ('gate.','shared_head.'):
                require(sum(v for k,v in grads.items() if k.startswith(prefix))>0,'Missing '+prefix+' gradient')
            if m.correction is not None:
                require(grads['correction.2.weight']>0,'Missing correction output gradient')
                if step==0:require(all(v==0 for k,v in grads.items() if k.startswith(('correction.0.','left.','right.'))), 'Zero initialization mechanism differs')
                if step==3:require(all(v>0 for k,v in grads.items() if k.startswith(('correction.0.','left.','right.'))),'Upstream correction gradients did not establish')
            require(all(p.grad is None for p in m.parameters() if not p.requires_grad),'Frozen predictor received gradient')
            steps.append(grads);opt.step()
        receipt[n]=steps
    require(raw.state_digest(prep.state_dict())==frozen_digest and all(p.grad is None for p in prep.parameters()), 'Frozen preparation changed')
    raw.write(output/'gradient_probe.json',{'passed':True,'probe_updates_discarded':True,'steps':receipt})
