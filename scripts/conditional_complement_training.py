"""Reuse the existing sampler/atomic recovery; commit only complete paired batches."""
import json
import math
import os
import time
import torch

import joint_transition_training as transaction
from xgnid.conditional_complement import Projected

raw, require = transaction.raw, transaction.require


class Cache:
    def __init__(self, features, labels):
        self.features = Projected(*(v.detach().cpu().contiguous() for v in features))
        self.labels = torch.as_tensor(labels, dtype=torch.long).cpu()
        require(all(len(v) == len(self.labels) for v in self.features), 'Projected cache lengths differ')

    def batch(self, indices, device):
        ix = torch.as_tensor(list(indices), dtype=torch.long)
        return Projected(*(v[ix].to(device) for v in self.features)), self.labels[ix].to(device)


def optimizer(model, settings, preparation=False):
    if preparation:
        groups = [{'params': list(model.parameters()), 'lr': settings['preparation_learning_rate']}]
    else:
        base, new = [], []
        for name, p in model.named_parameters():
            if p.requires_grad:
                (base if name.startswith(('gate.', 'shared_head.')) else new).append(p)
        groups = [{'params': base, 'lr': settings['fusion_learning_rate'], 'name': 'gate_head'}]
        if new:
            groups.append({'params': new, 'lr': settings['new_learning_rate'], 'name': 'correction'})
    return torch.optim.Adam(groups, betas=tuple(settings['adam_betas']), eps=settings['adam_eps'],
                            weight_decay=settings['weight_decay'], amsgrad=False)


@torch.no_grad()
def measure(models, cache, settings, device):
    predictions = {name: [] for name in models}
    for model in models.values():
        model.eval()
    for start in range(0, len(cache.labels), settings['eval_batch_size']):
        features, _ = cache.batch(range(start, min(start+settings['eval_batch_size'], len(cache.labels))), device)
        for name, model in models.items():
            scores = model(features)[0]
            require(bool(torch.isfinite(scores).all()), 'Nonfinite validation output')
            predictions[name].extend(scores.argmax(-1).tolist())
    result = {}
    for name, values in predictions.items():
        records = [{'sample_id': str(i), 'label': int(y), 'prediction': p}
                   for i, (y, p) in enumerate(zip(cache.labels, values))]
        result[name] = transaction.frozen.joint.study.validate_predictions(records, probabilities=False)
    return predictions, result


def save_views(output, state):
    for name in state['models']:
        folder = output/name
        folder.mkdir(exist_ok=True)
        temporary = folder/'epochs.jsonl.tmp'
        rows = [{**{k:v for k,v in r.items() if k not in ('validation','losses','gradient_norms')},
                 'validation': r['validation'].get(name), 'losses': r['losses'].get(name),
                 'gradient_norms': r['gradient_norms'].get(name)} for r in state['history']]
        temporary.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf8')
        os.replace(temporary, folder/'epochs.jsonl')
        transaction.atomic_save(folder/'last.pt', {'state':state['models'][name], 'epoch':state['completed_epoch'],
            'updates':state['updates'], 'binding':state['binding']})
        if name in state['best']:
            transaction.atomic_save(folder/'best.pt', {**state['best'][name], 'binding':state['binding']})
    raw.write(output/'training_progress.json', {'completed_epoch':state['completed_epoch'],
        'updates_per_arm':state['updates'], 'sampler_cursor':state['sampler']['cursor'],
        'best_epochs':{n:v['epoch'] for n,v in state['best'].items()}})


def fit(models, cache, settings, binding, device, output, loss_fn, evaluate=None,
        preparation=False, resume=True, stop_after_updates=None, fault=None):
    """A failed partial batch never overwrites the last all-arm transaction.

    Restart creates fresh models/optimizers then restores every Adam moment, RNG,
    loss accumulator and sampler cursor. No partial in-memory state is published.
    """
    output.mkdir(parents=True, exist_ok=True)
    path = output/'recovery.pt'
    opts = {n:optimizer(m,settings,preparation) for n,m in models.items()}
    sampler = transaction.PairedSampler(len(cache.labels), settings['batch_size'],
        settings['preparation_sampling_seed'] if preparation else settings['sampling_seed'])
    epochs = settings['preparation_epochs'] if preparation else settings['epochs']
    if path.exists():
        require(resume, 'Recovery exists; use --resume auto')
        state = torch.load(path, map_location='cpu', weights_only=True)
        require(state['schema']=='xgnid.conditional-transaction.v1' and state['binding']==binding
            and state['settings']==settings and list(state['models'])==list(models), 'Recovery source/config/code differs')
        for name, model in models.items():
            model.load_state_dict(state['models'][name], strict=True)
            opts[name].load_state_dict(state['optimizers'][name])
            require(all(int(v['step']) == state['updates'] for v in opts[name].state.values()), 'Adam cursor differs')
        sampler.load_state_dict(state['sampler'])
        require(state['updates']==state['completed_epoch']*len(sampler)+state['epoch_updates']
                and state['epoch_samples']==sampler.cursor, 'Recovery cursor differs')
        transaction.restore_rng(state['rng'])
        raw.write(output/'resume_receipt.json', {'passed':True,'updates_per_arm':state['updates'],
            'adam_restored':True,'rng_restored':True,'cursor':sampler.cursor})
    else:
        require(not any(output.glob('*/epochs.jsonl')), 'Ledger exists without recovery')
        predictions, metrics = evaluate(models) if evaluate else ({},{})
        state={'schema':'xgnid.conditional-transaction.v1','binding':binding,'settings':settings,
            'completed_epoch':0,'updates':0,'epoch_updates':0,'epoch_samples':0,
            'loss_sums':{n:{} for n in models},'gradient_sums':{n:{} for n in models},
            'training_seconds':{n:0. for n in models},'best':{},'history':[]}
        if evaluate:
            for n,m in models.items():
                state['best'][n]={'epoch':0,'validation':metrics[n],'predictions':predictions[n],
                                  'state':transaction.cpu_state(m)}
        state['history'].append({'epoch':0,'updates_per_arm':0,'cumulative_updates_per_arm':0,
            'samples_per_arm':0,'order_sha256':None,'validation':metrics,'losses':{},'gradient_norms':{}})

    def persist():
        state.update(models={n:transaction.cpu_state(m) for n,m in models.items()},
            optimizers={n:o.state_dict() for n,o in opts.items()}, sampler=sampler.state_dict(),rng=transaction.rng_state())
        transaction.atomic_save(path,state)

    if not path.exists():
        persist()
        for name in models:
            transaction.atomic_save(output/name/'epoch0.pt',{'binding':binding,'epoch':0,
                'state':state['models'][name],'validation':state['best'].get(name,{}).get('validation')})
    save_views(output,state)
    while state['completed_epoch']<epochs:
        epoch=state['completed_epoch']+1
        if sampler.epoch!=epoch:
            sampler.start(epoch)
            state.update(epoch_updates=0,epoch_samples=0,loss_sums={n:{} for n in models},
                         gradient_sums={n:{} for n in models})
        for m in models.values():
            m.train()
        while sampler.cursor<sampler.size:
            indices=sampler.peek(); features,labels=cache.batch(indices,device)
            for name,model in models.items():
                if device.type=='cuda': torch.cuda.synchronize(device)
                started=time.perf_counter()
                opt=opts[name];opt.zero_grad(set_to_none=True)
                loss,terms=loss_fn(name,model,features,labels)
                require(bool(torch.isfinite(loss)), 'Nonfinite training loss: '+name)
                # Record each objective's gradient norm on the first batch, without altering .grad.
                if state['epoch_updates']==0:
                    params=[p for p in model.parameters() if p.requires_grad]
                    for key,value in terms.items():
                        gradients=torch.autograd.grad(value,params,retain_graph=True,allow_unused=True) if value.requires_grad else []
                        state['gradient_sums'][name]['term_'+key]=math.sqrt(sum(float(g.detach().square().sum()) for g in gradients if g is not None))
                loss.backward()
                require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
                        'Nonfinite gradient: '+name)
                norms={}
                for key,p in model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        prefix=key.split('.')[0]
                        norms[prefix]=norms.get(prefix,0.)+float(p.grad.square().sum())
                for key,value in norms.items():
                    g=state['gradient_sums'][name]; g[key]=g.get(key,0.)+math.sqrt(value)
                opt.step()
                if device.type=='cuda': torch.cuda.synchronize(device)
                state['training_seconds'][name]+=time.perf_counter()-started
                for key,value in {**terms,'total':loss}.items():
                    sums=state['loss_sums'][name];sums[key]=sums.get(key,0.)+float(value.detach())*len(indices)
                if fault: fault(name,state['updates']+1)
            sampler.cursor+=len(indices)
            state['epoch_samples']+=len(indices); state['epoch_updates']+=1;state['updates']+=1
            persist()
            if stop_after_updates==state['updates']: return state
        require(state['epoch_samples']==sampler.size and state['epoch_updates']==len(sampler),'Incomplete paired epoch')
        predictions,metrics=evaluate(models) if evaluate else ({},{})
        for n,m in models.items():
            if evaluate and metrics[n]['macro_f1']>state['best'][n]['validation']['macro_f1']:
                state['best'][n]={'epoch':epoch,'validation':metrics[n],'predictions':predictions[n],
                                  'state':transaction.cpu_state(m)}
        state['history'].append({'epoch':epoch,'updates_per_arm':state['epoch_updates'],
            'cumulative_updates_per_arm':state['updates'],'samples_per_arm':state['epoch_samples'],
            'order_sha256':sampler.order_sha256,'validation':metrics,
            'losses':{n:{k:v/state['epoch_samples'] for k,v in d.items()} for n,d in state['loss_sums'].items()},
            'gradient_norms':{n:{k:v if k.startswith('term_') else v/state['epoch_updates'] for k,v in d.items()}
                              for n,d in state['gradient_sums'].items()}})
        state['completed_epoch']=epoch;state['epoch_updates']=0
        persist();save_views(output,state)
        raw.progress(output,'conditional_preparation' if preparation else 'conditional_paired_training',
            epoch=epoch,total_epochs=epochs,updates_per_arm=state['updates'],
            macro_f1={n:m['macro_f1'] for n,m in metrics.items()})
    require(state['updates']==epochs*len(sampler),'Final update budget differs')
    return state
