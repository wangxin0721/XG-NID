"""Rel-Mean/Rel-Gate only: frozen GAT messages, trainable relation selection."""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from types import FunctionType

import run_packet_pool_pair as packet
from e3_gate_study import (activate_gpu, checked_gpu, guard_graph_loads, identity, inside,
                           read, require, runtime, sha, write)

HERE = Path(__file__).resolve().parent
MODULE = HERE.parent / 'src/xgnid/relation_selection.py'
VARIANTS = {'Rel-Mean': 'mean', 'Rel-Gate': 'relation_gate'}
TENSORS = ('flow64', 'first64', 'flow_message64', 'packet_message64', 'legacy_packet_mean64', 'incoming', 'ptr', 'labels')
NODE_KEYS = ('first64', 'flow_message64', 'packet_message64')
dim, pair = packet.dim, packet.pair


def check_config(c):
    expected = read(HERE.parent / 'configs/packet_pool_pair_s42_v1.json')
    expected.update(schema='xgnid.relation-pair.v1', variants=VARIANTS,
        inputs='frozen_preaggregation_relation_outputs_train_val_only', score_network=[192,16,2],
        score_output_bias=True, relation_lr=1e-4, J32_reference_macro_f1=.97368275,
        J32_reference_provenance='user_reported_rounded_percent_97.368275',
        changed_location='packet_encoder_second_layer_packet_destination_only',
        empty_relation_policy='original_convolution_outputs_without_new_masks')
    expected.update(min_epochs=50,epochs=150,early_stop_enabled=False,
                    early_stop_reference='trained_epochs_only',final_selection='epoch_zero_or_best_trained')
    del expected['pool_lr']
    require(c == expected, 'Unregistered relation-pair protocol')


def architecture(mode):
    require(mode in VARIANTS.values(), 'Unknown relation mode')
    return {**dim.architecture(32), 'relation_mode': mode, 'score': [192,16,2] if mode=='relation_gate' else None,
        'relation_parameters': 3122 if mode=='relation_gate' else 0,
        'trainable_parameters': 10266 if mode=='relation_gate' else 7144,
        'packet_pool': 'ordered_mean', 'empty_relations': 'unchanged_unmasked_convolution_output'}


def selection_step(initial,best,selected,stale,score,epoch,c):
    require(math.isfinite(score) and 0<=score<=1 and epoch>=1,'Invalid validation score/epoch')
    improved=score>best
    if improved:best,selected,stale=score,epoch,0
    else:stale+=1
    final_epoch=selected if best>initial else 0
    return {'trained_best':best,'trained_epoch':selected,'stale':stale,'improved':improved,
            'final_epoch':final_epoch,'stop':c['early_stop_enabled'] and epoch>=c['min_epochs'] and stale>=c['patience']}


def check_history(history,m,c):
    require(bool(history) and len(history)==m['epochs']==c['epochs']==150, 'Both arms must complete all150 epochs')
    best,selected,stale=-math.inf,None,0
    for epoch,row in enumerate(history,1):
        state=selection_step(m['epoch_zero']['macro_f1'],best,selected,stale,row['val_macro_f1'],epoch,c)
        best,selected,stale=state['trained_best'],state['trained_epoch'],state['stale']
        require(row['epoch']==epoch and row['optimizer_updates']==211
            and row['train_order_sha256']==m['orders'][epoch-1] and row['best_epoch']==state['final_epoch']
            and row['best_trained_epoch']==selected and row['stale_epochs']==stale
            and (epoch==len(history) or not state['stop']), 'Selection history mismatch')
    require((state['stop'] or len(history)==c['epochs']) and m['best_epoch']==state['final_epoch']
            and m['best_trained_epoch']==selected and m['selected_epoch_zero']==(m['best_epoch']==0)
            and m['epoch_zero_eligible'] is True and m['optimizer_updates']==len(history)*211
            and len(m['orders'])==len(history)
            and abs(m['trained_best_validation']['macro_f1']-best)<1e-12
            and abs(m['best_validation']['macro_f1']-max(best,m['epoch_zero']['macro_f1']))<1e-12
            and abs(m['last_validation']['macro_f1']-history[-1]['val_macro_f1'])<1e-12, 'Final selection mismatch')


def train(core,args,model,caches,device,c,binding,encoder_hash,name,expected_orders=None):
    root=args.output/name;root.mkdir()
    cfg={'variant':name,'architecture':architecture(model.pooling),'protocol':c,'binding':binding,
         'labels':list(core.LABELS),'frozen_encoder_sha256':encoder_hash}
    write(root/'config.json',cfg)
    initial,_=evaluate(core,model,caches['val'],device);write(root/'epoch_zero.json',initial)
    core.seed_all(42);start_rng=pair.rng_hash(core,pair.rng_state(core));opt=optimizer(core,model,c)
    sampler=core.TrainBatches(len(caches['train']['ids']),c['batch_size'],c['seed'])
    payload={'model_state':model.state_dict(),'optimizer_state':opt.state_dict(),'epoch':0,'validation':initial,
             'config':cfg,'rng':pair.rng_state(core),'sampler_rng':sampler.generator.get_state()}
    core.checkpoint(root/'epoch_zero.pt',payload);core.checkpoint(root/'best.pt',payload)
    best,trained_epoch,stale=-math.inf,None,0;orders=[];history=[];updates=0;started=time.perf_counter()
    with (root/'epochs.jsonl').open('x',encoding='utf-8') as log:
        for epoch in range(1,c['epochs']+1):
            model.train();count,steps,total=0,0,0.
            for numbers in sampler:
                if steps==0 and expected_orders is not None and epoch<=len(expected_orders):
                    require(sampler.last_order_sha256==expected_orders[epoch-1],'Paired sample order differs')
                f,y=minibatch(core,caches['train'],numbers,device);opt.zero_grad(set_to_none=True)
                scores,_=model.forward_features(*f);loss=core.F.nll_loss(scores,y)
                require(bool(core.torch.isfinite(loss)),'Nonfinite loss');loss.backward()
                require(all(p.grad is None or bool(core.torch.isfinite(p.grad).all()) for p in model.parameters()),'Nonfinite gradients')
                opt.step();total+=float(loss.detach())*len(numbers);count+=len(numbers);steps+=1
            require(count==107621 and steps==211,'Incomplete epoch');dim.assert_frozen(core,model,encoder_hash)
            metrics,_=evaluate(core,model,caches['val'],device)
            state=selection_step(initial['macro_f1'],best,trained_epoch,stale,metrics['macro_f1'],epoch,c)
            best,trained_epoch,stale=state['trained_best'],state['trained_epoch'],state['stale']
            orders.append(sampler.last_order_sha256);updates+=steps
            payload={'model_state':model.state_dict(),'optimizer_state':opt.state_dict(),'epoch':epoch,'validation':metrics,
                     'config':cfg,'rng':pair.rng_state(core),'sampler_rng':sampler.generator.get_state()}
            core.checkpoint(root/'last.pt',payload)
            if state['improved']:
                core.checkpoint(root/'trained_best.pt',payload)
                if state['final_epoch']!=0:core.checkpoint(root/'best.pt',payload)
            row={'epoch':epoch,'train_nll':total/count,'val_macro_f1':metrics['macro_f1'],'optimizer_updates':steps,
                 'train_order_sha256':orders[-1],'best_epoch':state['final_epoch'],'best_trained_epoch':trained_epoch,
                 'best_trained_macro_f1':best,'stale_epochs':stale}
            history.append(row);log.write(json.dumps(row)+'\n');log.flush();dim.progress(args,'training',variant=name,**row)
            if state['stop']:break
    core.sync(device);elapsed=time.perf_counter()-started
    trained=core.torch.load(root/'trained_best.pt',map_location=device,weights_only=False)
    model.load_state_dict(trained['model_state'],strict=True);trained_metrics,trained_exports=evaluate(core,model,caches['val'],device,True)
    require(trained['epoch']==trained_epoch and trained_metrics['confusion_matrix']==trained['validation']['confusion_matrix'],'Trained-best replay differs')
    pair.save_predictions(root/'trained_best_validation_predictions.jsonl.gz',trained_exports)
    chosen=core.torch.load(root/'best.pt',map_location=device,weights_only=False)
    model.load_state_dict(chosen['model_state'],strict=True);measured,exports=evaluate(core,model,caches['val'],device,True)
    require(chosen['epoch']==state['final_epoch'] and measured['confusion_matrix']==chosen['validation']['confusion_matrix'],'Final selected replay differs')
    pair.save_predictions(root/'validation_predictions.jsonl.gz',exports)
    result={'architecture':cfg['architecture'],'epoch_zero':initial,'trained_best_validation':trained_metrics,
        'best_validation':measured,'last_validation':metrics,'best_epoch':chosen['epoch'],'best_trained_epoch':trained_epoch,
        'epochs':epoch,'optimizer_updates':updates,'orders':orders,'epoch_zero_eligible':True,'selected_epoch_zero':chosen['epoch']==0,
        'training_start_rng_sha256':start_rng,'training_wall_seconds':elapsed,'training_evidence':dim.trained_evidence(initial['macro_f1'],history),
        'early_stop_reference':'trained_epochs_only','stop_reason':'patience' if state['stop'] else 'max_epochs',
        'parameter_groups':[{'name':g['name'],'lr':g['lr'],'parameters':sum(p.numel() for p in g['params'])} for g in opt.param_groups],
        'checkpoint_sha256':{n:sha(root/n) for n in ('epoch_zero.pt','trained_best.pt','best.pt','last.pt')},
        'predictions_sha256':sha(root/'validation_predictions.jsonl.gz')}
    check_history(history,result,c);write(root/'metrics.json',result);return result,exports


def metadata(args):
    # Reuse the unchanged mature-D32 provenance checks, replacing only this
    # entry's config validator and source binding; no historical module mutation.
    env = dict(packet.metadata.__globals__, check_config=check_config, MODULE=MODULE, __file__=__file__)
    fn = FunctionType(packet.metadata.__code__, env, 'relation_metadata', packet.metadata.__defaults__)
    meta = list(fn(args))
    meta[-1]['sources'] = {name: sha(path) for name,path in {
        'scripts/run_relation_pair.py': Path(__file__), 'src/xgnid/relation_selection.py': MODULE,
        'scripts/summarize_relation_pair.py': HERE/'summarize_relation_pair.py',
        'scripts/run_packet_pool_pair.py': Path(packet.__file__),
        'src/xgnid/packet_pooling.py': HERE.parent/'src/xgnid/packet_pooling.py'}.items()}
    return tuple(meta)


def load_classes():
    Head, _ = dim.load_classes()
    packet.load_classes()  # Bound ordered-mean implementation, not an installed package copy.
    spec = importlib.util.spec_from_file_location('xgnid.relation_selection', MODULE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return Head, module.RelationClassifier


def build(core, classes, data, device, n0, state, mode):
    Head, Graph = classes
    backbone = dim.build_backbone(core, data, device, n0)
    head = dim.make_head(core, Head, 32, device, dim.initialization_streams(42))
    rng = pair.rng_state(core)
    model = Graph(backbone.flow_encoder, backbone.packet_encoder, head, mode).to(device)
    require(pair.rng_hash(core,rng)==pair.rng_hash(core,pair.rng_state(core)), 'Relation construction changed RNG')
    initial = model.state_dict()
    require({k for k in initial if not k.startswith('relation_gate.')}==set(state), 'Mature state layout changed')
    initial.update(state)
    model.load_state_dict(initial, strict=True)
    require(core.state_hash((k,v) for k,v in model.state_dict().items() if k in state)
            ==core.state_hash(state.items()), 'Mature common weights differ')
    return model.eval()


def optimizer(core, model, c):
    groups=[{'name':'existing_head','params':list(model.fusion.parameters()),'lr':c['head_lr']}]
    if model.relation_gate is not None:
        groups.append({'name':'relation_score','params':list(model.relation_gate.parameters()),'lr':c['relation_lr']})
    flat=[p for g in groups for p in g['params']]
    require(len({id(p) for p in flat})==len(flat)
            and {id(p) for p in flat}=={id(p) for p in model.parameters() if p.requires_grad}
            and sum(p.numel() for p in flat)==architecture(model.pooling)['trainable_parameters'],
            'Optimizer omitted/duplicated/leaked parameters')
    opt=core.torch.optim.Adam(groups,weight_decay=c['weight_decay'])
    require(not opt.state,'Adam must start clean')
    return opt


def tensor_hash(core,cache):
    return core.state_hash((k,cache[k]) for k in TENSORS)


def minibatch(core, cache, numbers, device):
    pieces={k:[] for k in NODE_KEYS}
    sizes=[]
    for i in numbers:
        start,stop=int(cache['ptr'][i]),int(cache['ptr'][i+1])
        sizes.append(stop-start)
        for k in NODE_KEYS:
            pieces[k].append(cache[k][start:stop])
    index=core.torch.repeat_interleave(core.torch.arange(len(numbers),device=device),core.torch.tensor(sizes,device=device))
    features=(cache['flow64'][numbers].to(device),
              *(core.torch.cat(pieces[k]).to(device) for k in NODE_KEYS),index,len(numbers))
    return features,cache['labels'][numbers].to(device)


def validate_cache(core,cache,rows,part,encoder_hash,binding):
    n=len(rows); ptr=cache['ptr']
    require(cache['metadata']=={'schema':'xgnid.pre-relation-cache.v1','partition':part,
        'encoder_sha256':encoder_hash,'split_binding':binding,'samples':n,
        'sample_order_sha256':identity([r['sample_id'] for r in rows])}
        and cache['ids']==[r['sample_id'] for r in rows]
        and len(set(cache['ids']))==n and cache['labels'].tolist()==[r['label'] for r in rows], 'Relation cache identity differs')
    require(ptr.dtype==core.torch.int64 and ptr.shape==(n+1,) and int(ptr[0])==0
        and bool((ptr[1:]>ptr[:-1]).all()) and cache['labels'].dtype==core.torch.int64
        and all(cache[k].shape==(int(ptr[-1]),64) for k in NODE_KEYS)
        and cache['flow64'].shape==cache['legacy_packet_mean64'].shape==(n,64)
        and cache['incoming'].shape==(int(ptr[-1]),2) and cache['incoming'].dtype==core.torch.bool
        and all(bool(core.torch.isfinite(cache[k]).all()) for k in ('flow64','legacy_packet_mean64',*NODE_KEYS)),
        'Relation cache tensors invalid')


def extract(core,args,model,data,device,part,c,encoder_hash,binding,retained):
    rows=core.select_rows(data,part); ids=[]; counts=[]; graph_rows=[]
    keys=('flow64',*NODE_KEYS,'legacy_packet_mean64','incoming','labels')
    pieces={k:[] for k in keys}; max_error=0.; started=time.perf_counter()
    for step,batch in enumerate(packet.loader(core,data,part,c),1):
        batch=batch.to(device)
        features,legacy=model.encode(batch)
        flow,x1,mf,mp,index,size=features
        expected=rows[len(ids):len(ids)+size]
        require(list(batch.sample_id)==[r['sample_id'] for r in expected]
                and batch.y.view(-1).tolist()==[r['label'] for r in expected], 'Extraction ID/label order differs')
        with core.torch.no_grad():
            nodes,_=model.packet_nodes(x1,mf,mp,force_mean=True)
            require(core.torch.allclose(nodes,legacy,atol=1e-6,rtol=1e-6), 'Hooked relation mean does not reproduce original BN/activation')
            max_error=max(max_error,float((nodes-legacy).abs().max()))
            legacy_mean=core.torch_geometric.nn.global_mean_pool(legacy,index,size=size)
            inc=[]
            for relation in (('flow','contain','packet'),('packet','link','packet')):
                ei=batch[relation].edge_index
                inc.append(core.torch.bincount(ei[1],minlength=len(x1)))
            incoming=core.torch.stack([x>0 for x in inc],dim=-1)
            nf,np=mf.norm(dim=-1),mp.norm(dim=-1)
            diff=(mf-mp).norm(dim=-1)
            cosine=core.F.cosine_similarity(mf,mp,dim=-1)
            for j,r in enumerate(expected):
                mask=index==j; count=int(mask.sum()); require(count>0,'Empty Packet graph')
                graph_rows.append({'sample_id':r['sample_id'],'label':r['label'],'packets':count,
                    'flow_edges':int(inc[0][mask].sum()),'packet_edges':int(inc[1][mask].sum()),
                    'no_flow_incoming_nodes':int((~incoming[mask,0]).sum()),
                    'no_packet_incoming_nodes':int((~incoming[mask,1]).sum()),
                    'flow_norm_sum':float(nf[mask].sum()),'packet_norm_sum':float(np[mask].sum()),
                    'difference_sum':float(diff[mask].sum()),'max_difference':float(diff[mask].max()),
                    'cosine_sum':float(cosine[mask].sum()),
                    'relative_difference_sum':float((diff[mask]/(nf[mask]+np[mask]+1e-8)).sum()),
                    'both_zero_norm_nodes':int(((nf[mask]==0)&(np[mask]==0)).sum())})
                counts.append(count)
        values=(flow,x1,mf,mp,legacy_mean,incoming,batch.y.view(-1))
        for k,v in zip(keys,values): pieces[k].append(v.cpu())
        ids.extend(batch.sample_id)
        if step==1 or step%200==0: dim.progress(args,'extract_pre_relation',partition=part,samples=len(ids),total=len(rows))
    require(len(ids)==len(rows),'Incomplete relation extraction')
    chunk_bytes=sum(x.numel()*x.element_size() for ps in pieces.values() for x in ps)
    ptr_bytes=(len(ids)+1)*8; additional=chunk_bytes+2*ptr_bytes
    memory={'graphs':len(ids),'packet_nodes':sum(counts),'final_tensor_bytes':chunk_bytes+ptr_bytes,
        'additional_concat_bytes':additional,'estimated_tensor_peak_bytes':retained+chunk_bytes+additional,
        'retained_cache_bytes':retained,'host_MemAvailable_bytes':packet.available_host_memory(),
        'scope':'cache tensors only; excludes Python objects, graphs, models and allocator/serialization buffers'}
    path=args.output/'cache_memory_estimates.json'; mem=read(path) if path.exists() else {}; mem[part]=memory; write(path,mem)
    dim.progress(args,'relation_cache_before_concat',partition=part,**memory)
    require(memory['host_MemAvailable_bytes'] is None or additional<=memory['host_MemAvailable_bytes'], 'Insufficient memory for relation cache concat')
    cache={k:core.torch.cat(ps) for k,ps in pieces.items()}; del pieces
    cache.update(ids=ids,ptr=core.torch.tensor([0]+counts,dtype=core.torch.long).cumsum(0),graph_diagnostics=graph_rows,
        metadata={'schema':'xgnid.pre-relation-cache.v1','partition':part,'encoder_sha256':encoder_hash,
                  'split_binding':binding,'samples':len(ids),'sample_order_sha256':identity(ids)})
    path=args.output/(part+'_pre_relation.pt'); core.checkpoint(path,cache)
    return cache,{'sha256':sha(path),'tensor_sha256':tensor_hash(core,cache),'memory_estimate':memory,
        'original_node_reproduction_max_abs_error':max_error,'wall_seconds':time.perf_counter()-started,
        'graph_load_stage':packet.graph_load_stage('extract_'+part,part,len(ids),c['workers'])}


def diagnostics(core,cache):
    result={}
    for y,label in enumerate(core.LABELS):
        rows=[r for r in cache['graph_diagnostics'] if r['label']==y]; nodes=sum(r['packets'] for r in rows)
        result[label]={'graphs':len(rows),'packet_nodes':nodes,
            'empty_flow_relation_graphs':sum(r['flow_edges']==0 for r in rows),
            'empty_packet_relation_graphs':sum(r['packet_edges']==0 for r in rows),
            'varying_relation_graphs':sum(r['max_difference']>1e-7 for r in rows),
            **{k:sum(r[k] for r in rows) for k in ('flow_edges','packet_edges','no_flow_incoming_nodes','no_packet_incoming_nodes','both_zero_norm_nodes')},
            **{k.removesuffix('_sum')+'_node_mean':sum(r[k] for r in rows)/nodes
               for k in ('flow_norm_sum','packet_norm_sum','difference_sum','cosine_sum','relative_difference_sum')}}
    return result


def evaluate(core,model,cache,device,export=False):
    model.eval(); cm=[[0]*8 for _ in range(8)]; exports=[]
    with core.torch.no_grad():
        for start in range(0,len(cache['ids']),16):
            numbers=list(range(start,min(start+16,len(cache['ids']))))
            features,ys=minibatch(core,cache,numbers,device); scores,alpha=model.forward_features(*features)
            gamma=model.last_relation_weights
            require(bool(core.torch.isfinite(scores).all()) and bool(core.torch.isfinite(gamma).all())
                    and core.torch.allclose(gamma.sum(-1),core.torch.ones(len(gamma),device=device),atol=1e-7,rtol=1e-7), 'Invalid relation prediction')
            for j,(i,y,p) in enumerate(zip(numbers,ys.tolist(),scores.argmax(-1).tolist())):
                cm[y][p]+=1
                if export: exports.append({'sample_id':cache['ids'][i],'label':y,'prediction':p,
                    'flow_weights':alpha[j].tolist(),'relation_weights':gamma[features[-2]==j].tolist()})
    report=dim.frozen.matrix_report(cm,core.LABELS); dim.verify_metrics(report)
    return report,exports


def check_initial(core,models,cache,device,gold,expected,args):
    maximum=0.; reference_error=0.
    with core.torch.no_grad():
        for start in range(0,len(cache['ids']),16):
            numbers=list(range(start,min(start+16,len(cache['ids'])))); f,_=minibatch(core,cache,numbers,device)
            a,aw=models['Rel-Mean'].forward_features(*f); b,bw=models['Rel-Gate'].forward_features(*f)
            require(core.torch.allclose(a,b,atol=1e-6,rtol=1e-6) and core.torch.allclose(aw,bw,atol=1e-6,rtol=1e-6)
                    and bool((models['Rel-Gate'].last_relation_weights==.5).all()), 'Initial relation pair differs')
            maximum=max(maximum,float((a-b).abs().max()))
            old,_=models['Rel-Mean'].fusion(f[0],cache['legacy_packet_mean64'][numbers].to(device))
            reference_error=max(reference_error,float((a-old).abs().max()))
            require(bool((a.argmax(-1)==old.argmax(-1)).all()), 'Legacy mean predictions differ')
    for model in models.values():
        m,ex=evaluate(core,model,cache,device,True)
        require(m['confusion_matrix']==expected['confusion_matrix'] and
                [(e['sample_id'],e['label'],e['prediction']) for e in ex]==[(e['sample_id'],e['label'],e['prediction']) for e in gold],
                'Initial mature source predictions/CM not reproduced')
    result={'samples':len(gold),'paired_max_abs_error':maximum,'legacy_independent_mean_max_abs_error':reference_error,
        'pair_allclose_1e6':True,'uniform_relations':True,'source_predictions_and_CM_identical':True}
    write(args.output/'initialization.json',result); return result


def probe(core,model,data,cache,device,c,encoder_hash):
    clean={k:v.detach().clone() for k,v in model.state_dict().items()}; digest=core.state_hash(clean.items()); rng=pair.rng_state(core)
    records=[]
    try:
        numbers=[i for i,r in enumerate(cache['graph_diagnostics']) if r['max_difference']>1e-7][:512]
        require(bool(numbers),'No relation differences in training data; inspect relation_diagnostics.json')
        rows=core.select_rows(data,'train'); batch=core.Batch.from_data_list([data.load(rows[i]) for i in numbers]).to(device)
        require(list(batch.sample_id)==[cache['ids'][i] for i in numbers], 'Probe ID order differs')
        require(batch.y.view(-1).tolist()==cache['labels'][numbers].tolist(),'Probe labels differ')
        opt=optimizer(core,model,c)
        for step in range(1,4):
            model.train(); opt.zero_grad(set_to_none=True)
            before={k:v.detach().clone() for k,v in model.named_parameters() if v.requires_grad}
            raw=model(batch); f,y=minibatch(core,cache,numbers,device); cached,_=model.forward_features(*f)
            require(core.torch.allclose(raw,cached,atol=1e-5,rtol=1e-5),'Raw/cache gradient probe differs')
            loss=core.F.nll_loss(raw,y); require(bool(core.torch.isfinite(loss)) and raw.requires_grad,'Disconnected/nonfinite loss')
            loss.backward()
            norms={k:float(v.grad.abs().sum()) if v.grad is not None else 0. for k,v in model.named_parameters() if v.requires_grad}
            require(all(v.grad is None or bool(core.torch.isfinite(v.grad).all()) for v in model.parameters()),'Nonfinite gradient')
            opt.step()
            changed={k:not core.torch.equal(before[k],v) for k,v in model.named_parameters() if v.requires_grad}
            with core.torch.no_grad(): model.forward_features(*f)
            records.append({'step':step,'loss':float(loss.detach()),'gradient_l1':norms,'changed':changed,
                            'max_abs_relation_from_half':float((model.last_relation_weights-.5).abs().max())})
            dim.assert_frozen(core,model,encoder_hash)
        prefixes=['fusion.flow_projection.','fusion.packet_projection.','fusion.gate.','fusion.shared_head.']
        if model.relation_gate is not None:
            prefixes+=['relation_gate.0.','relation_gate.2.']
            require(any(r['max_abs_relation_from_half']>0 for r in records),'Relation weights never learned')
        for prefix in prefixes:
            require(any(any(k.startswith(prefix) and v>0 for k,v in r['gradient_l1'].items()) for r in records)
                    and any(any(k.startswith(prefix) and v for k,v in r['changed'].items()) for r in records),'No gradient/update: '+prefix)
    finally:
        model.load_state_dict(clean,strict=True); model.zero_grad(set_to_none=True); model.eval(); pair.restore_rng(core,rng)
        require(core.state_hash(model.state_dict().items())==digest and pair.rng_hash(core,rng)==pair.rng_hash(core,pair.rng_state(core)), 'Probe restore failed')
        dim.assert_frozen(core,model,encoder_hash)
    return {'updates':records,'clean_model_and_rng_restored':True,'optimizer_discarded':True,'discarded':True,'train_graphs':len(numbers)}


def device_preflight(core,models,device,args):
    rng=pair.rng_state(core); hashes={n:core.state_hash(m.state_dict().items()) for n,m in models.items()}
    r={'status':'running','optimizer_updates':0,'scope':'synthetic relation readout/BN/backward only'}
    try:
        gen=core.torch.Generator().manual_seed(42)
        values=[core.torch.randn(12,64,generator=gen).to(device) for _ in range(3)]
        flow=core.torch.randn(4,64,generator=gen).to(device); index=core.torch.arange(4,device=device).repeat_interleave(3)
        f=(flow,*values,index,4)
        with core.torch.no_grad():
            a,aw=models['Rel-Mean'].forward_features(*f)
            for _ in range(16):
                b,bw=models['Rel-Gate'].forward_features(*f)
                require(core.torch.allclose(a,b,atol=1e-6,rtol=1e-6) and core.torch.allclose(aw,bw,atol=1e-6,rtol=1e-6)
                        and bool((models['Rel-Gate'].last_relation_weights==.5).all()),'Device initial identity differs')
        with core.torch.enable_grad():
            nodes,_=models['Rel-Gate'].packet_nodes(*values)
            grads=core.torch.autograd.grad(nodes.square().sum(),list(models['Rel-Gate'].relation_gate.parameters()))
            require(all(bool(core.torch.isfinite(g).all()) for g in grads) and float(grads[-2].abs().sum())>0,'Gradient through frozen BN disconnected')
        r.update(status='passed',output_weight_gradient_l1=float(grads[-2].abs().sum()))
    except BaseException as error:
        r.update(status='failed',error=str(error)); raise
    finally:
        pair.restore_rng(core,rng)
        r['model_and_rng_unchanged']=all(core.state_hash(m.state_dict().items())==hashes[n] for n,m in models.items()) and pair.rng_hash(core,rng)==pair.rng_hash(core,pair.rng_state(core))
        write(args.output/'cuda_relation_preflight.json',r)
        require(r['model_and_rng_unchanged'],'Device preflight changed state')
    return r


def replay(core,args,model,data,cache,device,c,expected,encoder_hash,phase,original=None):
    model.eval(); cm=[[0]*8 for _ in range(8)]; ex=[]; offset=0; err=0.; legacy_err=0.
    with core.torch.no_grad():
        for step,batch in enumerate(packet.loader(core,data,'val',c),1):
            batch=batch.to(device); f,legacy=model.encode(batch); raw,_=model.forward_features(*f)
            numbers=list(range(offset,offset+batch.num_graphs)); cached_f,y=minibatch(core,cache,numbers,device)
            require(list(batch.sample_id)==[cache['ids'][i] for i in numbers] and batch.y.view(-1).tolist()==y.tolist(),'Replay sample order differs')
            cached,_=model.forward_features(*cached_f)
            require(core.torch.allclose(raw,cached,atol=1e-5,rtol=1e-5) and bool((raw.argmax(-1)==cached.argmax(-1)).all()),'Replay cache output differs')
            err=max(err,float((raw-cached).abs().max()))
            if original is not None:
                nodes,_=model.packet_nodes(f[1],f[2],f[3],force_mean=True)
                matched,_=original.fusion(f[0],model_mean(nodes,f[-2],f[-1]))
                require(core.torch.allclose(raw,matched,atol=1e-6,rtol=1e-6),'Matched original relation readout differs')
                old=original(batch); require(bool((raw.argmax(-1)==old.argmax(-1)).all()),'Original D32 raw predictions differ')
                legacy_err=max(legacy_err,float((raw-old).abs().max()))
            for sid,yy,p in zip(batch.sample_id,y.tolist(),raw.argmax(-1).tolist()):
                cm[yy][p]+=1; ex.append({'sample_id':sid,'label':yy,'prediction':p})
            offset+=batch.num_graphs
            if step==1 or step%200==0:dim.progress(args,phase,samples=offset)
    require(offset==13446 and cm==expected['confusion_matrix'],'Raw graph replay CM differs')
    dim.assert_frozen(core,model,encoder_hash)
    return {'strict_state_load':True,'predictions_identical':True,'confusion_matrix_reproduced':True,'samples':offset,
        'max_abs_raw_cached_error':err,'legacy_source_max_abs_error':legacy_err if original is not None else None,
        'graph_load_stage':packet.graph_load_stage(phase,'val',offset,c['workers'])},ex


def model_mean(nodes,index,size):
    return sys.modules['xgnid.packet_pooling'].ordered_mean_pool(nodes,index,size)


def run(args,meta):
    c,previous,root,source,binding=meta
    _,spec,ref,nroot,_,_=previous[1]
    gpu=checked_gpu(args.physical_gpu,args.allow_shared_gpu); activate_gpu(gpu)
    scripts=str(args.repo/'scripts');sys.path.insert(0,scripts)
    try: import train_grouped as core
    finally: sys.path.remove(scripts)
    device=runtime(core);write(args.output/'gpu.json',gpu)
    data=core.FrozenGraphs(inside(args.repo,spec['split_directory']),ref['binding']['split_id'])
    direct_counts=guard_graph_loads(data,{'train','val'})
    n0=core.torch.load(nroot/'N0/best.pt',map_location=device,weights_only=False)
    saved=core.torch.load(root/'D32-Gate/best.pt',map_location=device,weights_only=False)
    require(saved['epoch']==62 and saved['config']['binding']==source['binding']
            and saved['config']['labels']==list(core.LABELS),'Mature source payload differs')
    classes=load_classes();models={n:build(core,classes,data,device,n0,saved['model_state'],mode) for n,mode in VARIANTS.items()}
    encoder_hash=core.state_hash(dim.encoder_values(models['Rel-Mean']))
    require(encoder_hash==previous[3]['binding']['encoder_source']['encoder_sha256'],'Encoder source changed')
    for m in models.values():dim.assert_frozen(core,m,encoder_hash);optimizer(core,m,c)
    gpu_check=device_preflight(core,models,device,args)
    caches,receipts={},{}
    for part in ('train','val'):
        retained=sum(v[k].numel()*v[k].element_size() for v in caches.values() for k in TENSORS)
        caches[part],receipts[part]=extract(core,args,models['Rel-Mean'],data,device,part,c,encoder_hash,ref['binding'],retained)
        validate_cache(core,caches[part],core.select_rows(data,part),part,encoder_hash,ref['binding'])
    write(args.output/'cache_receipts.json',receipts)
    write(args.output/'relation_diagnostics.json',{p:diagnostics(core,v) for p,v in caches.items()})
    gold=dim.predictions(root/'D32-Gate/validation_predictions.jsonl.gz');expected=source['models']['D32-Gate']['best_validation']
    initial=check_initial(core,models,caches['val'],device,gold,expected,args)
    _,Original=dim.load_classes()
    original=Original(models['Rel-Mean'].flow_encoder,models['Rel-Mean'].packet_encoder,models['Rel-Mean'].fusion).eval()
    init_replays={}
    for name,m in models.items():init_replays[name],_=replay(core,args,m,data,caches['val'],device,c,expected,encoder_hash,'initial_'+name,original)
    del original
    initial['raw_graph_replays']=init_replays;write(args.output/'initialization.json',initial)
    probes={}
    for name,m in models.items():
        probes[name]=probe(core,m,data,caches['train'],device,c,encoder_hash)
        path=args.output/(name+'_preflight.pt');core.checkpoint(path,{'model_state':m.state_dict(),'architecture':architecture(m.pooling)})
        fresh=build(core,classes,data,device,n0,saved['model_state'],m.pooling)
        fresh.load_state_dict(core.torch.load(path,map_location=device,weights_only=False)['model_state'],strict=True)
        f,_=minibatch(core,caches['val'],list(range(16)),device)
        with core.torch.no_grad():require(core.torch.allclose(m.forward_features(*f)[0],fresh.forward_features(*f)[0],atol=1e-6,rtol=1e-6),'Preflight strict reload differs')
        del fresh
    write(args.output/'probes.json',probes);packet.source_pair.assert_source_unchanged(binding)
    results,exports,replays,trained_replays={},{},{},{}
    for name,m in models.items():
        orders=results['Rel-Mean']['orders'] if name=='Rel-Gate' else None
        results[name],exports[name]=train(core,args,m,caches,device,c,binding,encoder_hash,name,orders)
        checkpoint=core.torch.load(args.output/name/'best.pt',map_location=device,weights_only=False)
        require(checkpoint['config']['binding']==binding and checkpoint['config']['variant']==name
                and sha(args.output/name/'best.pt')==results[name]['checkpoint_sha256']['best.pt'],'Best checkpoint identity differs')
        fresh=build(core,classes,data,device,n0,saved['model_state'],m.pooling);fresh.load_state_dict(checkpoint['model_state'],strict=True)
        replays[name],raw=replay(core,args,fresh,data,caches['val'],device,c,checkpoint['validation'],encoder_hash,'best_'+name)
        require([r['prediction'] for r in raw]==[r['prediction'] for r in exports[name]],'Best exported raw predictions differ')
        write(args.output/name/'graph_replay.json',replays[name]);del fresh
        trained=core.torch.load(args.output/name/'trained_best.pt',map_location=device,weights_only=False)
        require(trained['epoch']==results[name]['best_trained_epoch']
                and sha(args.output/name/'trained_best.pt')==results[name]['checkpoint_sha256']['trained_best.pt'],'Trained-best identity differs')
        if trained['epoch']==checkpoint['epoch']:
            require(core.state_hash(trained['model_state'].items())==core.state_hash(checkpoint['model_state'].items()),'Same-epoch selected/trained weights differ')
            trained_replays[name]={**replays[name],'shared_with_selected_best':True}
        else:
            fresh=build(core,classes,data,device,n0,saved['model_state'],m.pooling);fresh.load_state_dict(trained['model_state'],strict=True)
            trained_replays[name],_=replay(core,args,fresh,data,caches['val'],device,c,trained['validation'],encoder_hash,'trained_best_'+name)
            trained_replays[name]['shared_with_selected_best']=False;del fresh
        write(args.output/name/'trained_best_graph_replay.json',trained_replays[name])
    common=min(m['epochs'] for m in results.values())
    require(results['Rel-Mean']['orders'][:common]==results['Rel-Gate']['orders'][:common]
            and results['Rel-Mean']['training_start_rng_sha256']==results['Rel-Gate']['training_start_rng_sha256'],'Pair RNG/order differs')
    for part,v in caches.items():require(tensor_hash(core,v)==receipts[part]['tensor_sha256'] and sha(args.output/(part+'_pre_relation.pt'))==receipts[part]['sha256'],'Relation cache changed')
    for m in models.values():dim.assert_frozen(core,m,encoder_hash)
    packet.source_pair.assert_source_unchanged(binding)
    from summarize_joint_fusion32 import changed_samples
    from summarize_fusion_gate_pair import prediction_transitions
    from summarize_relation_pair import summarize
    comparisons={'Gate_vs_Mean':(exports['Rel-Mean'],exports['Rel-Gate']), 'Mean_vs_D32':(gold,exports['Rel-Mean']),'Gate_vs_D32':(gold,exports['Rel-Gate'])}
    for name in VARIANTS:
        comparisons['Trained_'+name+'_vs_D32']=(gold,dim.predictions(args.output/name/'trained_best_validation_predictions.jsonl.gz'))
    changes,transitions={},{}
    for n,(a,b) in comparisons.items():
        changes[n]=pair.compare_predictions(a,b);transitions[n]=prediction_transitions(a,b,core.LABELS)
        pair.save_predictions(args.output/(n+'_changed_samples.jsonl.gz'),changed_samples(a,b,core.LABELS))
    stages=[r['graph_load_stage'] for group in (receipts,init_replays,replays) for r in group.values()]
    stages +=[r['graph_load_stage'] for r in trained_replays.values() if not r['shared_with_selected_best']]
    accounting=packet.graph_load_accounting(direct_counts,stages)
    write(args.output/'graph_load_accounting.json',accounting)
    summary={'status':'completed','exit_code':0,'protocol':c,'binding':binding,'models':results,'source_model':source['models']['D32-Gate'],
        'initialization':initial,'cuda_relation_preflight':gpu_check,'probes':probes,'graph_replays':replays,'trained_best_graph_replays':trained_replays,
        'relation_diagnostics':read(args.output/'relation_diagnostics.json'),'comparisons':changes,'prediction_transitions':transitions,
        'cache_receipts':receipts,'graph_load_accounting':accounting,'common_epochs_order_checked':common,
        'encoder_parameters_and_BN_unchanged':True,'source_artifacts_unchanged':True,'relation_caches_unchanged':True,
        'tensor_preflight_passed':True,'discarded_probe_updates':6,'training_jobs':2,'test_evaluated':False,'automatic_followup':False}
    write(args.output/'summary.json',summary);print(summarize(args.output),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path.cwd());p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=('check','run'),default='check')
    p.add_argument('--physical-gpu',type=int,default=1);p.add_argument('--allow-shared-gpu',action='store_true');args=p.parse_args()
    args.repo,args.config=args.repo.resolve(),args.config.resolve();args.output=inside(args.repo/'outputs',inside(args.repo,args.output))
    meta=metadata(args);require(not args.output.exists() and args.output!=meta[2] and meta[2] not in args.output.parents,'Refuse output overwrite/source reuse')
    if args.mode=='check':print(json.dumps({'status':'metadata_checks_passed','binding':meta[-1],'training_started':False,'tensor_checks_executed':False}),flush=True);return
    require(os.name=='posix','Formal training runs on the Linux wx server');args.output.mkdir(parents=True)
    write(args.output/'config.json',meta[0]);write(args.output/'binding.json',meta[-1]);write(args.output/'status.json',{'status':'running','phase':'preflight'})
    try:
        run(args,meta);write(args.output/'status.json',{'status':'completed','exit_code':0});write(args.output/'progress.json',{'status':'completed','training_jobs':2})
    except BaseException:
        write(args.output/'status.json',{'status':'failed','exit_code':1});write(args.output/'progress.json',{'status':'failed','exit_code':1})
        (args.output/'error.log').write_text(traceback.format_exc(),encoding='utf-8');raise


if __name__=='__main__':main()
