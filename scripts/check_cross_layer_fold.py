"""No-training best-checkpoint and full-validation CL-Last folding audit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from e3_gate_study import read,require,sha,write
import run_cross_layer_pair as run
from summarize_cross_layer_pair import analyze


def state_hash(values):
    import torch
    h=hashlib.sha256()
    for name,tensor in sorted(values):
        tensor=tensor.detach().cpu().contiguous()
        h.update(json.dumps([name,str(tensor.dtype),list(tensor.shape)]).encode())
        h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def audit(source,device_name='cpu'):
    import torch
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    from xgnid.fold_readout import fold_last
    s=read(source/'summary.json');analyze(s)
    require(read(source/'status.json')=={'status':'completed','exit_code':0},'Source incomplete')
    require(read(source/'config.json')==s['protocol'] and read(source/'binding.json')==s['binding'],'Source binding changed')
    watched=[source/n for n in ('summary.json','status.json','config.json','binding.json','cache_receipts.json','val_cross_layer.pt')]
    watched +=[source/name/file for name in run.VARIANTS for file in ('best.pt','metrics.json','epochs.jsonl','validation_predictions.jsonl.gz')]
    before={str(p):sha(p) for p in watched}
    receipt=read(source/'cache_receipts.json')['val']
    require(before[str(source/'val_cross_layer.pt')]==receipt['sha256'],'Validation cache file hash differs')
    cache=torch.load(source/'val_cross_layer.pt',map_location='cpu',weights_only=False)
    core=SimpleNamespace(torch=torch,state_hash=state_hash)
    require(run.tensor_hash(core,cache)==receipt['tensor_sha256'],'Validation cache tensor hash differs')
    require(len(cache['ids'])==13446 and len(set(cache['ids']))==13446 and cache['labels'].shape==(13446,), 'Invalid validation identity')
    device=torch.device(device_name);torch.set_num_threads(2)
    if device.type=='cuda':
        require(torch.cuda.is_available(),'CUDA unavailable');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    Head,_=run.dim.load_classes();labels=list(s['source_model']['best_validation']['per_class'])
    rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    results={};fold_result=None
    try:
        with torch.no_grad():
            for name,mode in run.VARIANTS.items():
                m=s['models'][name];path=source/name/'best.pt'
                require(sha(path)==m['checkpoint_sha256']['best.pt'],'Best checkpoint bytes differ')
                require(m==read(source/name/'metrics.json'),'Metrics file differs')
                run.check_history([json.loads(x) for x in (source/name/'epochs.jsonl').read_text().splitlines()],m,s['protocol'])
                saved=torch.load(path,map_location='cpu',weights_only=False);state=saved['model_state'];cfg=saved['config']
                require(saved['epoch']==m['best_epoch']==(113 if mode=='last' else 126)
                    and cfg['variant']==name and cfg['binding']==s['binding'] and cfg['protocol']==s['protocol']
                    and cfg['architecture']==run.architecture(mode) and cfg['labels']==labels,'Checkpoint identity differs')
                require(saved['validation']==m['best_validation'],'Saved validation differs')
                enc=state_hash((k,v) for k,v in state.items() if k.startswith(('flow_encoder.','packet_encoder.')))
                require(enc==cfg['frozen_encoder_sha256']==cache['metadata']['encoder_sha256'],'Encoder/cache binding differs')
                head=Head(32,run.dim.initialization_streams(42)).to(device).eval()
                head.load_state_dict({k[len('fusion.'):]:v for k,v in state.items() if k.startswith('fusion.')},strict=True)
                projection=torch.nn.Linear(128,64).to(device).eval()
                projection.load_state_dict({k[len('cross_layer.'):]:v for k,v in state.items() if k.startswith('cross_layer.')},strict=True)
                folded=fold_last(projection,head) if mode=='last' else None
                head_hash=state_hash(head.state_dict().items());projection_hash=state_hash(projection.state_dict().items())
                rows=run.dim.predictions(source/name/'validation_predictions.jsonl.gz')
                require(sha(source/name/'validation_predictions.jsonl.gz')==m['predictions_sha256'],'Prediction file hash differs')
                require([(r['sample_id'],r['label']) for r in rows]==list(zip(cache['ids'],cache['labels'].tolist())),'Prediction/cache order differs')
                cm=[[0]*8 for _ in range(8)];fcm=[[0]*8 for _ in range(8)];max_error=0.;all_bits=True
                for start in range(0,13446,16):
                    stop=min(start+16,13446);flow=cache['flow64'][start:stop].to(device)
                    first=cache['first64'][start:stop].to(device);last=cache['second64'][start:stop].to(device)
                    x=torch.cat((first if mode=='multi' else last,last),dim=-1)
                    out,alpha=head(flow,projection(x));pred=out.argmax(-1).tolist()
                    require(pred==[r['prediction'] for r in rows[start:stop]],'CPU/device cached predictions differ from selected source')
                    for yy,pp in zip(cache['labels'][start:stop].tolist(),pred):cm[yy][pp]+=1
                    if folded is not None:
                        new,weights=folded(flow,last)
                        require(torch.allclose(out,new,atol=1e-5,rtol=1e-5) and torch.allclose(alpha,weights,atol=1e-5,rtol=1e-5),'Fold output/weights exceed registered tolerance')
                        require(new.argmax(-1).tolist()==pred,'Fold changed predictions')
                        max_error=max(max_error,float((out-new).abs().max()));all_bits=all_bits and torch.equal(out,new)
                        for yy,pp in zip(cache['labels'][start:stop].tolist(),new.argmax(-1).tolist()):fcm[yy][pp]+=1
                measured=run.dim.frozen.matrix_report(cm,labels);require(measured==m['best_validation'],'Selected full validation metrics differ')
                results[name]={'epoch':saved['epoch'],'checkpoint_sha256':sha(path),'encoder_sha256':enc,'cached_prediction_replay_identical':True,'metrics':measured,'strict_head_and_projection_load':True}
                require(state_hash(head.state_dict().items())==head_hash and state_hash(projection.state_dict().items())==projection_hash,'Folding mutated original parameters')
                if folded is not None:
                    require(fcm==cm,'Fold confusion matrix differs')
                    fold_result={'status':'passed','samples':13446,'atol':1e-5,'rtol':1e-5,'max_abs_log_probability_error':max_error,
                        'outputs_bitwise_identical':all_bits,'predictions_identical':True,'metrics_identical':True,
                        'construction':'float64 affine composition then source-dtype parameters','newly_replayed_raw_graphs':False}
    finally:
        torch.set_rng_state(rng)
        if cuda_rng:torch.cuda.set_rng_state_all(cuda_rng)
    require({str(p):sha(p) for p in watched}==before,'Source files changed during audit')
    return {'status':'passed','device':str(device),'torch_version':torch.__version__,'source_directory':str(source),
        'checkpoint_verification':results,'folding':fold_result,'optimizer_updates':0,'source_files_unchanged':True,
        'rng_restored':True,'audit_script_sha256':sha(Path(__file__))}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
    require(not a.output.exists(),'Refuse audit overwrite')
    result=audit(a.source.resolve(),a.device);write(a.output,result);print(json.dumps(result),flush=True)
