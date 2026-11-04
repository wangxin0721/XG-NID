"""Implementation/raw-graph probes only; never starts a full training run."""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import torch
from torch.nn import functional as F
from torch_geometric.data import Batch
from xgnid.attention_operator_pair import OperatorClassifier, assert_paired, warm_graph


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_digest(state):
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        h.update(json.dumps([k,str(v.dtype),list(v.shape)]).encode())
        h.update(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def synthetic_graphs():
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator().manual_seed(71042).get_state())
        graphs = []
        for i in range(8):
            g = warm_graph(3,5)
            for node in ('flow','packet'):
                g[node].x.normal_()
            for rel in g.edge_types:
                g[rel].edge_attr.normal_()
            g.y = torch.tensor([i]); g.sample_id = 'synthetic:'+str(i)
            graphs.append(g)
    return graphs


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def probe_one(model, train, val, device, config):
    import numpy as np
    initial = {k:v.detach().clone() for k,v in model.state_dict().items()}
    clean = state_digest(initial)
    rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    py_rng, np_rng = random.getstate(), np.random.get_state()
    opt = torch.optim.Adam([
        {'params':list(model.flow_encoder.parameters())+list(model.packet_encoder.parameters()),'lr':config['encoder_lr']},
        {'params':list(model.fusion.parameters()),'lr':config['head_lr']}], weight_decay=config['weight_decay'])
    if opt.state:
        raise ValueError('Expected a fresh Adam')
    updates = []
    try:
        torch.manual_seed(config['seed'])
        model.eval()
        with torch.no_grad():
            out = model(val)
            if out.shape != (val.num_graphs,8) or not bool(torch.isfinite(out).all()):
                raise ValueError('Wrong/nonfinite prediction shape')
            initial_predictions = out.argmax(-1).tolist()
        baseline = torch.cuda.memory_allocated(device) if device.type == 'cuda' else None
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        for step in range(config['probe_updates']):
            model.train();opt.zero_grad(set_to_none=True)
            sync(device);start=time.perf_counter()
            out=model(train);loss=F.nll_loss(out,train.y.view(-1))
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite NLL')
            loss.backward()
            gradients={n:float(p.grad.detach().abs().sum()) for n,p in model.named_parameters() if p.grad is not None}
            if not gradients or any(not bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None):
                raise ValueError('Missing/nonfinite gradient')
            for prefix in ('flow_encoder.','packet_encoder.','fusion.'):
                if not any(n.startswith(prefix) and value>0 for n,value in gradients.items()):
                    raise ValueError('No nonzero gradient: '+prefix)
            opt.step();sync(device)
            updates.append({'step':step+1,'nll':float(loss.detach()),'seconds':time.perf_counter()-start,
                            'gradient_l1':gradients})
        model.eval();sync(device);start=time.perf_counter()
        with torch.no_grad():
            out=model(val);predictions=out.argmax(-1).tolist()
        sync(device);seconds=time.perf_counter()-start
        if not bool(torch.isfinite(out).all()):
            raise ValueError('Nonfinite post-update validation')
        cm=[[0]*8 for _ in range(8)]
        for y,p in zip(val.y.view(-1).tolist(),predictions):
            cm[y][p]+=1
        from e3_gate_study import matrix_metrics
        changed=[n for n,v in model.state_dict().items() if not torch.equal(v,initial[n])]
        if not any('.convs' in n for n in changed) or not any('.bns' in n and 'running_' in n for n in changed):
            raise ValueError('Convolution or BN did not update')
        result={'updates':updates,'initial_predictions':initial_predictions,'post_probe_predictions':predictions,
                'small_validation_metrics':matrix_metrics(cm),'validation_seconds':seconds,
                'changed_tensors':changed,'total_parameters':sum(p.numel() for p in model.parameters()),
                'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
                'cuda_baseline_allocated_bytes':baseline,
                'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                'peak_extra_allocated_bytes':torch.cuda.max_memory_allocated(device)-baseline if device.type=='cuda' else None,
                'initial_state_sha256':clean}
    finally:
        model.load_state_dict(initial,strict=True);model.zero_grad(set_to_none=True);model.eval()
        opt.state.clear()
        torch.set_rng_state(rng)
        if cuda_rng:
            torch.cuda.set_rng_state_all(cuda_rng)
        random.setstate(py_rng);np.random.set_state(np_rng)
    if (state_digest(model.state_dict())!=clean or not torch.equal(torch.get_rng_state(),rng)
            or any(not torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all() if cuda_rng else []))):
        raise ValueError('Probe failed to restore clean state')
    result.update(clean_state_restored=True,optimizer_discarded=True,rng_restored=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path.cwd())
    p.add_argument('--config',type=Path,default=ROOT/'configs/attention_operator_pair_s42_v1.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--synthetic',action='store_true')
    args=p.parse_args()
    if args.output.exists():
        raise ValueError('Use a new output directory; previous probe evidence is preserved')
    args.output.mkdir(parents=True)
    def write(name,value):
        with (args.output/name).open('w',encoding='utf-8',newline='\n') as f:
            json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
    write('status.json',{'status':'running'})
    try:
        import torch_geometric
        if torch_geometric.__version__!='2.5.3' or str(torch.__version__).split('+')[0]!='2.3.1':
            raise ValueError('Registered implementation requires torch2.3.1 and PyG2.5.3')
        config=json.loads(args.config.read_text(encoding='utf-8'))
        if config!=json.loads((ROOT/'configs/attention_operator_pair_s42_v1.json').read_text(encoding='utf-8')):
            raise ValueError('Use the registered probe protocol')
        write('config.json',config)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        device=torch.device(args.device)
        if device.type=='cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable; no CPU fallback for a GPU probe')
        provenance={'mode':'synthetic' if args.synthetic else 'raw_graphs','test_graph_loads':0}
        if args.synthetic:
            graphs=synthetic_graphs();train_graphs=val_graphs=graphs
        else:
            from grouped_graphs import FrozenGraphs
            frozen=FrozenGraphs(args.repo/config['split_directory'],config['split_id'])
            if frozen.metadata['partition_sizes']!=config['partition_sizes']:
                raise ValueError('Partition counts differ')
            def select(part,per_class):
                counts=[0]*8;rows=[]
                for row in frozen.rows:
                    if row['partition']==part and counts[row['label']]<per_class:
                        counts[row['label']]+=1;rows.append(row)
                if counts!=[per_class]*8:
                    raise ValueError('Missing probe class samples')
                # Only these explicit train/val rows are ever loaded.
                graphs=[]
                for i,row in enumerate(rows,1):
                    graphs.append(frozen.load(row))
                    if i==1 or i%64==0 or i==len(rows):
                        print(json.dumps({'phase':'load_probe_graphs','partition':part,'samples':i,'total':len(rows)}),flush=True)
                return graphs,rows
            train_graphs,tr=select('train',config['probe_train_per_class'])
            val_graphs,vr=select('val',config['probe_val_per_class'])
            provenance.update(split_id=frozen.metadata['split_id'],sample_manifest_sha256=frozen.metadata['sample_manifest_sha256'],
                split_json_sha256=digest(frozen.directory/'split.json'),train_ids=[r['sample_id'] for r in tr],
                val_ids=[r['sample_id'] for r in vr],train_graph_loads=len(tr),val_graph_loads=len(vr))
        train=Batch.from_data_list(train_graphs).to(device);val=Batch.from_data_list(val_graphs).to(device)
        dims=[train_graphs[0][k].x.shape[1] for k in ('flow','packet')]
        models={name:OperatorClassifier(name,*dims,seed=config['seed']) for name in config['variants']}
        pairing=assert_paired(models);results={}
        for name,model in models.items():
            print('Probing '+name,flush=True)
            model.to(device)
            results[name]=probe_one(model,train,val,device,config)
            model.cpu()
        files=['src/xgnid/attention_operator_pair.py','src/xgnid/model.py','src/xgnid/fusion_dimension.py',
               'scripts/probe_attention_operator_pair.py','scripts/grouped_graphs.py','scripts/audit_graph_split.py',
               'scripts/build_grouped_split.py','scripts/e3_gate_study.py','configs/attention_operator_pair_s42_v1.json']
        receipt={'status':'passed','protocol':config,'device':str(device),'torch':torch.__version__,
                 'pyg':torch_geometric.__version__,'provenance':provenance,'pairing':pairing,'models':results,
                 'sources':{n:digest(ROOT/n) for n in files},'formal_training_started':False,
                 'full_validation_evaluated':False,'cost_scope':'3 updates including fresh Adam allocation; subset validation; not a full-run estimate',
                 'train_batch_graphs':train.num_graphs,'val_batch_graphs':val.num_graphs}
        write('probe.json',receipt);write('status.json',{'status':'completed','exit_code':0})
        print('Paired probe passed; full training was not started.',flush=True)
    except BaseException as error:
        write('status.json',{'status':'failed','exit_code':1,'error':str(error)})
        raise


if __name__=='__main__':
    main()
