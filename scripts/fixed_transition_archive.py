"""Immutable source locks, seed42 import and completed-artifact verification."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import torch

import run_joint_transition_fusion as parent
import fixed_transition_training as training
import fixed_transition_results as results
from run_transition_pair_cross_replay import Archive
from xgnid.fixed_transition_stability import ARMS, MODELS

raw, frozen, original, require = parent.raw, parent.frozen, parent.original, parent.require


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def open_archive(repo, spec, download=False):
    ref, directory = spec['source_result_commit'], spec['source_directory']
    first = next(iter(spec['inputs']))
    check = subprocess.run(['git','cat-file','-e',ref+':'+directory+'/'+first],cwd=repo,
                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    if check.returncode:
        require(download,'Missing Git source snapshot: '+ref+'; fetch codex/server-outputs')
        subprocess.run(['git','fetch','--no-tags','origin',ref],cwd=repo,check=True)
    archive=Archive(repo,spec,snapshot=True)
    missing=[directory+'/'+name for name in spec['inputs'] if isinstance(archive.locate(name),Path)
             and not archive.locate(name).is_file()]
    if missing:
        require(download,'Missing registered LFS inputs; run launcher with --download-inputs: '+','.join(missing))
        subprocess.run(['git','lfs','fetch','--include='+','.join(missing),'--exclude=','origin',ref],cwd=repo,check=True)
    archive.verify_all()
    return archive


def lock_protocol(repo, output, settings, source_files, download=False):
    for spec in (*settings['historical_seed42'].values(),*settings['independent_sources'].values(),
                 *settings.get('seed42_rerun_context',[])):
        open_archive(repo,spec,download)
    code = subprocess.check_output(['git','rev-parse','HEAD'],cwd=raw.ROOT,text=True).strip()
    lock={'schema':'xgnid.fixed-transition-protocol-lock.v1','code_commit':code,'protocol':settings,
        'source_files_canonical_sha256':{name:hashlib.sha256((raw.ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                                       for name in source_files},
        'seed42_role':'development_stage_locked_archive','new_training_backbones':[43,44],
        'scope':'paired validation on independent backbones, fixed readout and sampling seeds at42'}
    path=output/'protocol_lock.json'; output.mkdir(parents=True,exist_ok=True)
    if path.exists():
        require(raw.read(path)==lock,'Protocol/source/code lock differs; preserve the existing output')
    else:
        raw.write(path,lock)
    return lock


def seal(output):
    files={p.relative_to(output).as_posix():raw.sha(p) for p in sorted(output.rglob('*'))
           if p.is_file() and p.name!='artifact_manifest.json' and not p.name.endswith(('.tmp','.lock'))}
    raw.write(output/'artifact_manifest.json',{'files_sha256':files})


def verify_sealed(output):
    manifest=raw.read(output/'artifact_manifest.json')
    for name,digest in manifest['files_sha256'].items():
        require((output/name).is_file() and raw.sha(output/name)==digest,'Completed artifact changed: '+str(output/name))
    require(raw.read(output/'status.json')['status']=='completed'
            and raw.read(output/'final_verification.json')['passed'],'Completed phase lacks verification')
    return raw.read(output/'summary.json')


def parameter_counts(source, settings):
    total=sum(p.numel() for p in source.parameters())
    return {name:{'inference_parameters':total+(0 if name=='base_C' else 3728 if name=='joint_ordered' else 3200),
                  'trainable_parameters':0 if name=='base_C' else settings['trainable_parameters'][name]}
            for name in MODELS}


def check_historical_protocol(settings, archives):
    checks={}
    for arm in ARMS:
        archived=archives[arm].json('config.json')
        require(archives[arm].json('status.json')['status']=='completed','Historical run not completed')
        selection=archives[arm].json('selection.json' if arm=='frozen_single' else 'summary.json')
        require(not selection['test_evaluated'],'Historical scope differs')
        for key in ('epochs','batch_size','eval_batch_size','readout_seed','training_samples','validation_samples','selection'):
            require(archived[key]==settings[key],arm+' historical protocol differs: '+key)
        require(archived['loss'] in ('plain_final_NLL','plain_final_eight_class_NLL'),
                arm+' historical loss is not the registered plain final NLL')
        lr=archived['learning_rate'] if arm=='frozen_single' else archived['readout_learning_rate']
        require(lr==settings['readout_learning_rate'],'Historical readout LR differs')
        if arm!='frozen_single':
            require(archived['fusion_learning_rate']==settings['fusion_learning_rate']
                    and archived['adam_betas']==settings['adam_betas'] and archived['adam_eps']==settings['adam_eps']
                    and archived['weight_decay']==settings['weight_decay'],'Historical joint Adam differs')
        provenance=archives[arm].json('provenance.json')
        code=provenance['code_commit'] if arm=='frozen_single' else provenance['binding']['code_commit']
        require(code==settings['historical_seed42'][arm]['source_code_commit'],'Historical code commit differs')
        checks[arm]={'code_commit':code,'same_registered_training_budget':True,'archived_loss_label':archived['loss'],
                     'loss_equivalence':'plain final eight-class NLL; legacy label omits class count',
                     'readout_seed':42,'sampling_seed':42,'source_result_commit':archives[arm].settings['source_result_commit']}
    return checks


def version_differences(repo,settings):
    files=('src/xgnid/fusion_dimension.py','src/xgnid/packet_transition_readout.py',
           'src/xgnid/ordered_transition_readout.py','src/xgnid/packet_pooling.py','src/xgnid/joint_transition_fusion.py')
    result={}
    for arm in ARMS:
        ref=settings['historical_seed42'][arm]['source_code_commit'];rows={}
        for name in files:
            command=subprocess.run(['git','show',ref+':'+name],cwd=repo,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
            old=hashlib.sha256(command.stdout.replace(b'\r\n',b'\n')).hexdigest() if command.returncode==0 else None
            current=hashlib.sha256((raw.ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
            rows[name]={'archived_canonical_sha256':old,'current_canonical_sha256':current,'same_file':old==current}
        result[arm]={'code_commit':ref,'files':rows,
            'interpretation':'Historical Single used frozen source fusion directly; wrappers now copy/freeze it. Later explicit context_right interface is unused for true pairs. Functional checks cover matching modules and all archived validation predictions.'}
    return result


def histories_from_archives(archives):
    names={'frozen_single':'single/epochs.jsonl','joint_single':'joint_single/epochs.jsonl','joint_ordered':'pair_true/epochs.jsonl'}
    histories={}
    for arm,path in names.items():
        history=[json.loads(line) for line in archives[arm].data(path).splitlines()]
        if arm=='frozen_single':
            history=[{**r,'validation':r['validation']['B']} for r in history]
        require([r['epoch'] for r in history]==list(range(201)),arm+' historical epoch budget differs')
        require(history[0]['updates_per_arm']==0 and all(r['updates_per_arm']==211 and r['samples_per_arm']==107621
                for r in history[1:]),arm+' historical update budget differs')
        histories[arm]=history
    require(all([r['order_sha256'] for r in h]==[r['order_sha256'] for r in histories['frozen_single']]
                for h in histories.values()),'Historical seed42 sample orders differ')
    return histories


def import_seed42(repo, root, settings, lock, download=False):
    torch.set_num_threads(2)
    output=root/'seed42_import'
    if (output/'artifact_manifest.json').exists():
        require(raw.read(output/'provenance.json')['protocol_lock_sha256']==canonical_sha(lock),'Imported protocol differs')
        return verify_sealed(output)
    output.mkdir(parents=True,exist_ok=True)
    archives={name:open_archive(repo,spec,download) for name,spec in settings['historical_seed42'].items()}
    protocols=check_historical_protocol(settings,archives)
    source_saved=torch.load(archives['base_C'].locate('C/best.pt'),map_location='cpu',weights_only=False)
    base=raw.OperatorClassifier('GAT',seed=42)
    source=frozen.joint.JointCrossLayerQueryClassifier(base.flow_encoder,base.packet_encoder,base.fusion,'flow',42)
    source.load_state_dict(source_saved['model_state'],strict=True)
    digest=frozen.independent.freeze_c(source)
    source_binding=archives['frozen_single'].json('provenance.json')['source']
    require(digest==source_binding['frozen_c_sha256'] and source_binding['backbone_seed']==42
            and source_binding['split_id']==settings['split_id'] and source_binding['manifest_sha256']==settings['manifest_sha256'],
            'Wrong original C for locked seed42 import')
    for arm in ('joint_single','joint_ordered'):
        require(archives[arm].json('provenance.json')['binding']['source']==source_binding,'Historical C sources differ')
    frozen_checkpoint=archives['frozen_single'].tensor('single/readout.pt')
    require(frozen_checkpoint['binding']==source_binding,'Frozen Single C binding differs')
    checkpoints={'frozen_single':frozen_checkpoint['best']['B'],
                 'joint_single':archives['joint_single'].tensor('joint_single/best.pt'),
                 'joint_ordered':archives['joint_ordered'].tensor('pair_true/best.pt')}
    last={'frozen_single':frozen_checkpoint['last']['B'],
          'joint_single':archives['joint_single'].tensor('joint_single/last.pt'),
          'joint_ordered':archives['joint_ordered'].tensor('pair_true/last.pt')}
    names={'joint_single':'joint_single','joint_ordered':'pair_true'}
    for arm,name in names.items():
        require(checkpoints[arm]['binding']['source']==source_binding,'Checkpoint source differs')
        require(archives[arm].json('selected_validation_replay.json')[name]['passed'],'Missing historical raw replay')
    require(archives['frozen_single'].json('selected_validation_replay.json')['single']['passed'],
            'Missing historical Frozen Single raw replay')
    arms=training.make_arms(source,settings,torch.device('cpu'))
    histories=histories_from_archives(archives)
    expected={}
    base_predictions=archives['base_C'].rows('C/validation_predictions.jsonl.gz')
    expected['base_C']=[{k:r[k] for k in ('sample_id','label','prediction')} for r in base_predictions]
    frozen_rows=archives['frozen_single'].rows('validation/validation_predictions.jsonl.gz')
    expected['frozen_single']=original.prediction_records(frozen_rows,[r['predictions']['single'] for r in frozen_rows])
    expected['joint_single']=archives['joint_single'].rows('joint_single/best_validation_predictions.jsonl.gz')
    expected['joint_ordered']=archives['joint_ordered'].rows('pair_true/best_validation_predictions.jsonl.gz')
    counts=parameter_counts(source,settings)
    metadata={}
    base_metrics=frozen.joint.study.validate_predictions(expected['base_C'],probabilities=False)
    metadata['base_C']={**counts['base_C'],'best_epoch':source_saved['epoch'],'last_epoch':source_saved['epoch'],
        'best_validation':base_metrics,'last_validation':base_metrics,'new_updates':0,'historical_readout_updates':0,
        'training_seconds':0.0,'role':'fixed original C reference'}
    for arm,model in arms.items():
        ck=checkpoints[arm]
        require(ck['epoch']==settings['historical_seed42'][arm]['selected_epoch']==max(histories[arm],
                key=lambda r:r['validation']['macro_f1'])['epoch'],'Locked historical best epoch differs')
        if arm=='frozen_single':
            model.readout.load_state_dict(ck['state'],strict=True)
        else:
            model.load_state_dict(ck['state'],strict=True)
        model.requires_grad_(False).eval()
        require(frozen.joint.study.validate_predictions(expected[arm],probabilities=False)==ck['validation'],
                arm+' archive predictions differ')
        metadata[arm]={**counts[arm],'best_epoch':ck['epoch'],'last_epoch':200,'best_validation':ck['validation'],
            'last_validation':histories[arm][-1]['validation'],'new_updates':0,'historical_readout_updates':42200,
            'training_seconds':None,'role':'development_stage_locked_archive',
            'archive':settings['historical_seed42'][arm]['source_result_commit']}
        folder=output/arm;folder.mkdir(exist_ok=True)
        training.parent.atomic_save(folder/'best.pt',{'source_binding':source_binding,'epoch':ck['epoch'],
            'state':training.parent.cpu_state(model),'validation':ck['validation'],'historical_source':metadata[arm]['archive']})
        last_model=deepcopy(model)
        if arm=='frozen_single': last_model.readout.load_state_dict(last[arm]['state'],strict=True)
        else: last_model.load_state_dict(last[arm]['state'],strict=True)
        training.parent.atomic_save(folder/'last.pt',{'source_binding':source_binding,'completed_epoch':200,
            'state':training.parent.cpu_state(last_model),'historical_source':metadata[arm]['archive']})
        (folder/'epochs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in histories[arm]),encoding='utf-8')
    (output/'base_C').mkdir(exist_ok=True)
    (output/'base_C'/'best.pt').write_bytes(archives['base_C'].data('C/best.pt'))
    cached=archives['joint_ordered'].tensor('frozen_val_features.pt')
    require(cached['binding']['source']==source_binding,'Imported feature cache source differs')
    features=original.Features(*(cached['features'][k] for k in original.Features._fields))
    rows=[{'sample_id':sid,'label':label,'partition':'val'} for sid,label in zip(cached['ids'],cached['labels'])]
    require(len(rows)==settings['validation_samples'],'Historical validation size differs')
    cache=original.FrozenCache([features],cached['labels'])
    reference=training.parent.JointTransitionFusion(source.fusion,'joint_base',42).requires_grad_(False).eval()
    records=[]
    with torch.no_grad():
        for start in range(0,len(rows),settings['eval_batch_size']):
            stop=min(start+settings['eval_batch_size'],len(rows));batch,_=cache.batch(range(start,stop),torch.device('cpu'))
            records.extend(results.diagnostics.observe({'base_C':reference,**arms},batch,rows[start:stop]))
    exported=results.diagnostics.exports(records)
    require(all(exported[name]==expected[name] for name in MODELS),'Historical nonzero model replay differs')
    training.parent.verify_source(source,digest)
    raw.write(output/'provenance.json',{'protocol_lock_sha256':canonical_sha(lock),'source_binding':source_binding,
        'historical_protocols':protocols,'new_training_epochs':0,'new_optimizer_updates':0,
        'functional_equivalence':'same fixed-injection modules and shapes; all four selected validation predictions reproduced',
        'historical_log_probabilities_comparison':'not archived for all models; prediction identities verified',
        'source_versions_differ':True,'shared_readout_and_fusion_definitions_retained':True})
    raw.write(output/'code_version_differences.json',version_differences(repo,settings))
    context=[]
    for spec in settings.get('seed42_rerun_context',[]):
        previous=open_archive(repo,spec,download)
        summary=previous.json('summary.json');lineage=previous.json('provenance.json')['binding']['source']
        require(lineage==source_binding and summary['selected_epochs'][spec['arm']]==spec['epoch'],
                'Seed42 context lineage/epoch differs')
        context.append({'source_result_commit':spec['source_result_commit'],'source_directory':spec['source_directory'],
            'arm':spec['arm'],'epoch':spec['epoch'],'metrics':summary['metrics'][spec['arm']],
            'role':'run_variation_context_only; not eligible for reselection'})
    raw.write(output/'rerun_context.json',{'locked_joint_ordered_source':'7c40c4b6772679d51b20a6a268fc8b4ae2f3d518',
        'contexts':context,'reselection_permitted':False})
    raw.write(output/'selected_validation_replay.json',{'passed':True,'samples':len(rows),'models':list(MODELS),
        'mode':'CPU replay of hash-bound frozen validation features','encoders_executed':False,
        'native_raw_graph_replays':'verified inherited archived receipts','new_training_epochs':0})
    summary=results.write_seed_results(output,records,rows,metadata,source_binding,settings,histories,
                                      'development_stage_locked_archive',42)
    raw.write(output/'final_verification.json',{'passed':True,'historical_inputs_unchanged':True,
        'original_C_unchanged':True,'selected_predictions_identical_to_locked_archives':True,
        'new_optimizer_updates':0,'test_graph_loads':0})
    for archive in archives.values(): archive.verify_all()
    raw.write(output/'status.json',{'status':'completed','stage':'import_only','formal_training_started':False,'test_evaluated':False})
    seal(output)
    return summary
