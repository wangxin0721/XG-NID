"""Two fresh raw-graph operator trainings, after a verified CUDA probe."""
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, Sampler
from torch_geometric.loader import DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from xgnid.attention_operator_pair import OperatorClassifier, assert_paired
from grouped_graphs import FrozenGraphs
from probe_attention_operator_pair import state_digest
from summarize_attention_operator_pair import LABELS, matrix_report, require, check_history, report

VARIANTS=('GAT','GATv2')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
    temporary.replace(path)


def save_predictions(path,rows):
    with gzip.open(path,'wt',encoding='utf-8',newline='\n') as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'cpu':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def rng_hash(state):
    h=hashlib.sha256(repr((state['python'],state['numpy'][0],state['numpy'][1].tolist(),state['numpy'][2:])).encode())
    for value in [state['cpu']]+state['cuda']:
        h.update(value.cpu().numpy().tobytes())
    return h.hexdigest()


class GraphRows(Dataset):
    def __init__(self,frozen,rows,partition):
        require(partition in ('train','val') and all(r['partition']==partition for r in rows),'Forbidden graph partition')
        self.frozen,self.rows=frozen,rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,i):
        return self.frozen.load(self.rows[i])


class TrainBatches(Sampler):
    def __init__(self,size,batch_size,seed):
        require(size>=2 and batch_size>=2,'Singleton training dataset/batch')
        self.size,self.batch_size=size,batch_size
        self.generator=torch.Generator().manual_seed(seed)
        self.order=[];self.order_sha256=None

    def __len__(self):
        return math.ceil(self.size/self.batch_size)-int(self.size%self.batch_size==1)

    def __iter__(self):
        self.order=torch.randperm(self.size,generator=self.generator).tolist()
        self.order_sha256=hashlib.sha256(json.dumps(self.order).encode()).hexdigest()
        start=0
        while start<self.size:
            stop=self.size if self.size-start==self.batch_size+1 else min(start+self.batch_size,self.size)
            yield self.order[start:stop]
            start=stop


def choose_rows(frozen,part,per_class=None):
    counts=[0]*8;rows=[]
    require(part in ('train','val'),'Test graph selection forbidden')
    for row in frozen.rows:
        if row['partition']==part and (per_class is None or counts[row['label']]<per_class):
            rows.append(row);counts[row['label']]+=1
    require(all(v>0 for v in counts),'Missing class')
    return rows


def validate_probe(p,prep,source_root):
    require(p['status']=='passed' and p['protocol']==prep and p['provenance']['mode']=='raw_graphs'
            and p['device']=='cuda:0' and p['torch'].startswith('2.3.1+') and p['pyg']=='2.5.3'
            and p['train_batch_graphs']==512 and p['val_batch_graphs']==16
            and p['formal_training_started'] is False and p['full_validation_evaluated'] is False,
            'A completed registered raw CUDA probe is required')
    require(p['pairing']['all_common_tensors_identical'] and p['pairing']['common_tensors']==100
            and p['provenance']['test_graph_loads']==0,'Unpaired probe or test graph access')
    required={'src/xgnid/attention_operator_pair.py','src/xgnid/model.py','src/xgnid/fusion_dimension.py',
        'scripts/probe_attention_operator_pair.py','scripts/grouped_graphs.py','scripts/audit_graph_split.py',
        'scripts/build_grouped_split.py','scripts/e3_gate_study.py','configs/attention_operator_pair_s42_v1.json'}
    require(set(p['sources'])==required,'Incomplete probe source binding')
    for name,h in p['sources'].items():
        require(sha(source_root/name)==h,'Probe source changed: '+name)
    for name,count in [('GAT',851688),('GATv2',850152)]:
        m=p['models'][name]
        require(m['total_parameters']==m['trainable_parameters']==count
                and all(m[k] is True for k in ('clean_state_restored','optimizer_discarded','rng_restored')),
                'Wrong parameters or dirty probe state')
        require(len(m['updates'])==3 and [r['step'] for r in m['updates']]==[1,2,3],'Incomplete three-update probe')
        for row in m['updates']:
            require(math.isfinite(row['nll']) and row['nll']>=0 and row['seconds']>0
                    and all(math.isfinite(v) and v>=0 for v in row['gradient_l1'].values()),'Invalid probe gradient/loss/cost')
            for prefix in ('flow_encoder.','packet_encoder.','fusion.'):
                require(any(n.startswith(prefix) and v>0 for n,v in row['gradient_l1'].items()),'Probe gradient path missing')
        require(any('.convs' in n for n in m['changed_tensors'])
                and any('.bns' in n and 'running_' in n for n in m['changed_tensors']),'Probe did not update encoder/BN')


def metadata(args):
    c=read(args.config)
    expected={'schema':'xgnid.attention-operator-training.v1','workers':2,'epochs':150,'early_stop_enabled':False,
        'include_epoch_zero':True,'batch_size':512,'eval_batch_size':16,'seed':42,'encoder_lr':.001,'head_lr':.001,
        'weight_decay':0.,'test_evaluated':False,'automatic_followup':False,
        'preparation_protocol':'attention_operator_pair_s42_v1.json','probe_directory':'outputs/attention_operator_probe_s42_v1'}
    require(all(c.get(k)==v for k,v in expected.items()),'Unregistered full-training protocol')
    prep_path=args.config.parent/c['preparation_protocol'];prep=read(prep_path)
    require(sha(prep_path)==c['preparation_config_sha256'],'Preparation protocol changed')
    probe_root=args.repo/c['probe_directory'];p=read(probe_root/'probe.json')
    require(read(probe_root/'status.json')=={'status':'completed','exit_code':0},'Server probe incomplete')
    validate_probe(p,prep,ROOT)
    frozen=FrozenGraphs(args.repo/prep['split_directory'],prep['split_id'])
    require(frozen.metadata['partition_sizes']==prep['partition_sizes'],'Split sizes changed')
    require(p['provenance']['split_id']==frozen.metadata['split_id']
            and p['provenance']['sample_manifest_sha256']==frozen.metadata['sample_manifest_sha256']
            and p['provenance']['split_json_sha256']==sha(frozen.directory/'split.json'),'Probe split binding differs')
    for part,per_class in [('train',64),('val',2)]:
        rows=choose_rows(frozen,part,per_class)
        require(p['provenance'][part+'_ids']==[r['sample_id'] for r in rows]
                and p['provenance'][part+'_graph_loads']==len(rows),'Probe subset identity differs')
    sources=set(p['sources'])|{'scripts/run_attention_operator_pair.py','scripts/summarize_attention_operator_pair.py',
                              'scripts/test_attention_operator_training.py','scripts/launch_attention_operator_pair.sh',
                              'configs/attention_operator_training_s42_v1.json'}
    binding={'protocol_sha256':sha(args.config),'probe_directory':str(probe_root),
             'probe_sha256':sha(probe_root/'probe.json'),'probe_status_sha256':sha(probe_root/'status.json'),
             'split_json_sha256':sha(frozen.directory/'split.json'),
             'manifest_sha256':frozen.metadata['sample_manifest_sha256'],
             'sources':{n:sha(ROOT/n) for n in sorted(sources)}}
    return c,prep,p,frozen,binding


def progress(root,phase,**values):
    value={'status':'running','pid':os.getpid(),'phase':phase,**values}
    write(root/'progress.json',value);print(json.dumps(value),flush=True)


def sync(device):
    if device.type=='cuda':
        torch.cuda.synchronize(device)


def evaluate(model,loader,rows,device,root=None,phase='val',export=False):
    model.eval();cm=[[0]*8 for _ in range(8)];out_rows=[];offset=0;started=time.perf_counter()
    with torch.no_grad():
        for step,batch in enumerate(loader,1):
            current=rows[offset:offset+batch.num_graphs]
            require(list(batch.sample_id)==[r['sample_id'] for r in current]
                    and batch.y.view(-1).tolist()==[r['label'] for r in current],'Validation sample order differs')
            batch=batch.to(device);scores=model(batch);alpha=model.last_flow_weights
            require(scores.shape==(batch.num_graphs,8) and alpha.shape==(batch.num_graphs,32)
                    and bool(torch.isfinite(scores).all()) and bool(torch.isfinite(alpha).all()),'Invalid validation output')
            for sid,y,p,a in zip(batch.sample_id,batch.y.view(-1).tolist(),scores.argmax(-1).tolist(),alpha.tolist()):
                cm[y][p]+=1
                if export:
                    out_rows.append({'sample_id':sid,'label':y,'prediction':p,'flow_weights':a})
            offset+=batch.num_graphs
            if root is not None and (step==1 or step%200==0):
                progress(root,phase,samples=offset,total=len(rows),elapsed_seconds=time.perf_counter()-started)
    require(offset==len(rows),'Incomplete raw validation')
    return matrix_report(cm),out_rows


def save_checkpoint(path,model,opt,epoch,metrics,cfg,sampler):
    payload={'model_state':{n:v.detach().cpu().clone() for n,v in model.state_dict().items()},
        'optimizer_state':opt.state_dict(),'epoch':epoch,'validation':metrics,'config':cfg,
        'rng':rng_state(),'sampler_rng':sampler.generator.get_state()}
    temporary=path.with_suffix('.tmp');torch.save(payload,temporary);temporary.replace(path)


def strict_reload(model,path,device,cfg,expected_epoch,metrics):
    saved=torch.load(path,map_location=device,weights_only=False)
    require(saved['config']==cfg and saved['epoch']==expected_epoch and saved['validation']==metrics,
            'Checkpoint metadata/selection differs')
    model.load_state_dict(saved['model_state'],strict=True)
    return saved


def fit(name,model,frozen,train_rows,val_rows,device,c,binding,prep,root,expected_orders=None):
    folder=root/name;folder.mkdir()
    sampler=TrainBatches(len(train_rows),c['batch_size'],c['seed'])
    train_loader=DataLoader(GraphRows(frozen,train_rows,'train'),batch_sampler=sampler,num_workers=c['workers'],
        generator=torch.Generator().manual_seed(c['seed']+100))
    val_loader=DataLoader(GraphRows(frozen,val_rows,'val'),batch_size=c['eval_batch_size'],shuffle=False,num_workers=c['workers'],
        generator=torch.Generator().manual_seed(c['seed']+200))
    opt=torch.optim.Adam([
        {'name':'encoders','params':list(model.flow_encoder.parameters())+list(model.packet_encoder.parameters()),'lr':c['encoder_lr']},
        {'name':'D32_head','params':list(model.fusion.parameters()),'lr':c['head_lr']}],weight_decay=c['weight_decay'])
    require(not opt.state and all(p.requires_grad for p in model.parameters()),'Dirty optimizer or frozen parameter')
    cfg={'variant':name,'protocol':c,'preparation_protocol':prep,'binding':binding,'labels':LABELS,
         'initial_state_sha256':state_digest(model.state_dict())}
    initial={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
    start_rng=rng_hash(rng_state());started=time.perf_counter()
    zero,zero_rows=evaluate(model,val_loader,val_rows,device,root,name+'_epoch0',True)
    save_predictions(folder/'epoch_zero_predictions.jsonl.gz',zero_rows)
    save_checkpoint(folder/'epoch_zero.pt',model,opt,0,zero,cfg,sampler)
    require(rng_hash(rng_state())==start_rng,'Initial evaluation changed training RNG')
    selected,selected_epoch=zero,0;trained=None;trained_epoch=None;history=[];orders=[]
    peak=0
    for epoch in range(1,c['epochs']+1):
        model.train();require(all(m.training for m in (model.flow_encoder,model.packet_encoder,model.fusion)),'Train mode lost')
        if device.type=='cuda':
            torch.cuda.reset_peak_memory_stats(device)
        total_loss=0.;count=0;steps=0;gradient_paths={}
        sync(device);epoch_start=time.perf_counter()
        for step,batch in enumerate(train_loader,1):
            indices=sampler.order[count:count+batch.num_graphs]
            require(list(batch.sample_id)==[train_rows[i]['sample_id'] for i in indices]
                    and batch.y.view(-1).tolist()==[train_rows[i]['label'] for i in indices],'Training sample order differs')
            if step==1 and expected_orders is not None:
                require(sampler.order_sha256==expected_orders[epoch-1],'Paired epoch orders differ')
            batch=batch.to(device);opt.zero_grad(set_to_none=True)
            scores=model(batch);loss=F.nll_loss(scores,batch.y.view(-1))
            require(bool(torch.isfinite(loss)),'Nonfinite training NLL');loss.backward()
            grads=[p.grad for p in model.parameters() if p.grad is not None]
            require(bool(grads) and bool(torch.isfinite(torch.stack([g.norm() for g in grads]).norm())),'Nonfinite/missing gradient')
            if epoch==1 and step<=3:
                gradient_paths[str(step)]={prefix:sum(float(p.grad.detach().abs().sum()) for n,p in model.named_parameters()
                    if n.startswith(prefix) and p.grad is not None) for prefix in ('flow_encoder.','packet_encoder.','fusion.')}
                require(all(v>0 for v in gradient_paths[str(step)].values()),'Training gradient path disconnected')
            opt.step();total_loss+=float(loss.detach())*batch.num_graphs;count+=batch.num_graphs;steps+=1
            if step==1 or step%50==0:
                progress(root,name+'_train',epoch=epoch,samples=count,total=len(train_rows),train_nll=total_loss/count)
        require(count==len(train_rows) and steps==len(sampler),'Incomplete training epoch')
        sync(device);training_seconds=time.perf_counter()-epoch_start
        if device.type=='cuda':
            peak=max(peak,torch.cuda.max_memory_allocated(device))
        validation_start=time.perf_counter()
        measured,rows=evaluate(model,val_loader,val_rows,device,root,name+'_val_epoch'+str(epoch),True)
        validation_seconds=time.perf_counter()-validation_start
        if trained is None or measured['macro_f1']>trained['macro_f1']:
            trained,trained_epoch=measured,epoch
            save_checkpoint(folder/'trained_best.pt',model,opt,epoch,measured,cfg,sampler)
            save_predictions(folder/'trained_best_validation_predictions.jsonl.gz',rows)
        if measured['macro_f1']>selected['macro_f1']:
            selected,selected_epoch=measured,epoch
        if epoch==c['epochs']:
            save_checkpoint(folder/'last.pt',model,opt,epoch,measured,cfg,sampler)
            save_predictions(folder/'last_validation_predictions.jsonl.gz',rows)
        orders.append(sampler.order_sha256)
        row={'epoch':epoch,'train_nll':total_loss/count,'validation':measured,'best_epoch':selected_epoch,
             'best_trained_epoch':trained_epoch,'optimizer_updates':steps,'samples':count,'train_order_sha256':orders[-1],
             'training_seconds_including_graph_io':training_seconds,'validation_seconds_including_graph_io':validation_seconds}
        history.append(row)
        with (folder/'epochs.jsonl').open('a',encoding='utf-8',newline='\n') as f:
            f.write(json.dumps(row)+'\n')
        print(json.dumps({'model':name,**{k:v for k,v in row.items() if k!='validation'},'val_macro_f1':measured['macro_f1']}),flush=True)
        if epoch==1:
            write(folder/'first_updates_gradients.json',gradient_paths)
    changed={prefix:sum(not torch.equal(v.detach().cpu(),initial[n]) for n,v in model.state_dict().items() if n.startswith(prefix))
             for prefix in ('flow_encoder.','packet_encoder.','fusion.')}
    bn_changed=[n for n,v in model.state_dict().items() if 'running_' in n and not torch.equal(v.detach().cpu(),initial[n])]
    require(all(changed.values()) and bool(bn_changed),'Training did not update encoders/BN/head')
    target='epoch_zero.pt' if selected_epoch==0 else 'trained_best.pt'
    shutil.copyfile(folder/target,folder/'best.pt')
    source_predictions='epoch_zero_predictions.jsonl.gz' if selected_epoch==0 else 'trained_best_validation_predictions.jsonl.gz'
    shutil.copyfile(folder/source_predictions,folder/'validation_predictions.jsonl.gz')
    replays={};exports={};load_stages=[]
    for key,filename,epoch,metrics,predfile in [
        ('best','best.pt',selected_epoch,selected,'validation_predictions.jsonl.gz'),
        ('trained_best','trained_best.pt',trained_epoch,trained,'trained_best_validation_predictions.jsonl.gz')]:
        if key=='trained_best' and selected_epoch==trained_epoch:
            replays[key]={**replays['best'],'shared_with_selected_best':True};exports[key]=exports['best'];continue
        strict_reload(model,folder/filename,device,cfg,epoch,metrics)
        replay,export=evaluate(model,val_loader,val_rows,device,root,name+'_'+key+'_raw_replay',True)
        require(replay==metrics,'Raw replay confusion matrix/metrics differ')
        with gzip.open(folder/predfile,'rt',encoding='utf-8') as f:
            reference=[json.loads(x) for x in f]
        require([(r['sample_id'],r['label'],r['prediction']) for r in reference]
                ==[(r['sample_id'],r['label'],r['prediction']) for r in export],'Raw replay predictions differ')
        require(torch.allclose(torch.tensor([r['flow_weights'] for r in reference]),torch.tensor([r['flow_weights'] for r in export]),
                               atol=1e-5,rtol=1e-5),'Raw replay fusion weights differ')
        replays[key]={'strict_state_load':True,'predictions_identical':True,'metrics_identical':True,
                      'samples':len(val_rows),'shared_with_selected_best':False}
        exports[key]=export;load_stages.append({'phase':key+'_replay','samples':len(val_rows)})
    result={'epoch_zero':zero,'best_validation':selected,'best_epoch':selected_epoch,'trained_best_validation':trained,
        'best_trained_epoch':trained_epoch,'last_validation':history[-1]['validation'],'epochs':c['epochs'],
        'optimizer_updates':sum(r['optimizer_updates'] for r in history),'training_samples_per_epoch':len(train_rows),
        'orders':orders,'training_start_rng_sha256':start_rng,'initial_state_sha256':cfg['initial_state_sha256'],
        'early_stop_enabled':False,'stop_reason':'fixed_'+str(c['epochs'])+'_epochs','selected_epoch_zero':selected_epoch==0,
        'training_evidence':{'changed_tensor_counts':changed,'BN_changed_tensors':bn_changed,'encoders_and_bn_updated':True},
        'parameter_count':sum(p.numel() for p in model.parameters()),'replays':replays,
        'cost':{'wall_seconds_including_graph_io_and_replays':time.perf_counter()-started,
                'train_epoch_seconds_sum':sum(r['training_seconds_including_graph_io'] for r in history),
                'validation_epoch_seconds_sum':sum(r['validation_seconds_including_graph_io'] for r in history),
                'peak_train_cuda_allocated_bytes':peak if device.type=='cuda' else None},
        'graph_load_accounting':{'parent_verified_train_samples':len(train_rows)*c['epochs'],
            'parent_verified_val_samples':len(val_rows)*(1+c['epochs']+len(load_stages)),
            'test_samples':0,'replay_stages':load_stages,'meaning':'sample loads, not physical disk IO'},
        'checkpoint_sha256':{n:sha(folder/n) for n in ('epoch_zero.pt','trained_best.pt','best.pt','last.pt')},
        'predictions_sha256':sha(folder/'validation_predictions.jsonl.gz')}
    check_history(history,result,c['epochs'],len(sampler),len(val_rows));write(folder/'metrics.json',result)
    return result,exports['best']


def compare(a,b):
    require(len(a)==len(b) and len({r['sample_id'] for r in a})==len(a),'Bad comparison length/identity')
    counts={k:0 for k in ('repaired','regressed','both_correct','both_wrong')};changed=[]
    for x,y in zip(a,b):
        require((x['sample_id'],x['label'])==(y['sample_id'],y['label']),'Comparison sample order differs')
        old,new=x['prediction']==x['label'],y['prediction']==y['label']
        kind='both_correct' if old and new else 'regressed' if old else 'repaired' if new else 'both_wrong'
        counts[kind]+=1
        if x['prediction']!=y['prediction']:
            changed.append({'sample_id':x['sample_id'],'label':x['label'],'true_class':LABELS[x['label']],
                'old_prediction':x['prediction'],'new_prediction':y['prediction'],
                'change':kind if kind!='both_wrong' else 'both_wrong_changed'})
    return {'counts':counts,'net_correct':counts['repaired']-counts['regressed']},changed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path.cwd())
    p.add_argument('--config',type=Path,default=ROOT/'configs/attention_operator_training_s42_v1.json')
    p.add_argument('--output',type=Path)
    p.add_argument('--metadata-only',action='store_true')
    args=p.parse_args();args.repo=args.repo.resolve()
    c,prep,probe,frozen,binding=metadata(args)
    if args.metadata_only:
        print(json.dumps({'status':'metadata_passed','probe_validated':True,'binding':binding,'training_started':False}),flush=True)
        return
    root=(args.output or args.repo/'outputs/attention_operator_pair_s42_v1').resolve()
    require(root.parent==args.repo/'outputs' and not root.exists(),'Use a new experiment directory under outputs')
    root.mkdir();write(root/'status.json',{'status':'running'})
    try:
        require(torch.cuda.is_available(),'Formal training requires CUDA')
        import torch_geometric
        require(str(torch.__version__).split('+')[0]=='2.3.1' and torch_geometric.__version__=='2.5.3','Wrong torch/PyG version')
        device=torch.device('cuda:0');torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        train_rows,val_rows=choose_rows(frozen,'train'),choose_rows(frozen,'val')
        require((len(train_rows),len(val_rows))==(107621,13446),'Wrong full split counts')
        models={n:OperatorClassifier(n,seed=c['seed']) for n in VARIANTS};pairing=assert_paired(models)
        for name,m in models.items():
            require(state_digest(m.state_dict())==probe['models'][name]['initial_state_sha256'],'Fresh model differs from probed start')
        write(root/'config.json',c);write(root/'binding.json',binding);write(root/'probe_receipt.json',probe)
        write(root/'initialization.json',{'pairing':pairing,'initial_state_sha256':{n:state_digest(m.state_dict()) for n,m in models.items()}})
        write(root/'gpu.json',{'logical_device':str(device),'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'name':torch.cuda.get_device_name(device),'torch':torch.__version__,'pyg':torch_geometric.__version__})
        results={};exports={}
        for name,model in models.items():
            seed_all(c['seed']);model.to(device)
            expected=results['GAT']['orders'] if name=='GATv2' else None
            results[name],exports[name]=fit(name,model,frozen,train_rows,val_rows,device,c,binding,prep,root,expected)
            model.cpu();model.zero_grad(set_to_none=True)
        require(results['GAT']['orders']==results['GATv2']['orders']
                and results['GAT']['training_start_rng_sha256']==results['GATv2']['training_start_rng_sha256'],'Pairing lost during training')
        comparison,changed=compare(exports['GAT'],exports['GATv2'])
        save_predictions(root/'GATv2_vs_GAT_changed_samples.jsonl.gz',changed)
        require(all(sha(ROOT/n)==h for n,h in binding['sources'].items()),'Bound code changed during training')
        probe_root=Path(binding['probe_directory'])
        require(sha(probe_root/'probe.json')==binding['probe_sha256'] and sha(probe_root/'status.json')==binding['probe_status_sha256']
                and sha(frozen.directory/'split.json')==binding['split_json_sha256']
                and sha(frozen.directory/frozen.metadata['sample_manifest'])==binding['manifest_sha256'],'Source data/probe binding changed')
        s={'status':'completed','exit_code':0,'protocol':c,'preparation_protocol':prep,'binding':binding,
           'pairing':pairing,'probe_validated':True,'source_artifacts_unchanged':True,'models':results,'comparison':comparison,
           'changed_samples_sha256':sha(root/'GATv2_vs_GAT_changed_samples.jsonl.gz'),
           'test_evaluated':False,'automatic_followup':False}
        report(root,s);write(root/'summary.json',s);write(root/'status.json',{'status':'completed','exit_code':0})
        write(root/'progress.json',{'status':'completed','training_jobs':2});print('Both arms completed150 epochs; report.md is ready.',flush=True)
    except BaseException as error:
        write(root/'status.json',{'status':'failed','exit_code':1,'error':str(error)})
        raise


if __name__=='__main__':
    main()
