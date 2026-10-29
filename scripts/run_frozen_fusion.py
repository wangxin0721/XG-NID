"""Freeze N0, reproduce F0, then train only F1/F2 on train representation caches.

Imports are stdlib-only. Actual tensor/graph work requires the original Linux wx
runtime. Test graphs are forbidden; no resume, overwrite or automatic followup.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import os
from pathlib import Path
import sys
import traceback

from e3_gate_study import (activate_gpu, checked_gpu, guard_graph_loads, identity,
                           inside, read, require, runtime, sha, write)
from run_e2_concat_pair import (compare_predictions, rng_state, restore_rng,
                                rng_hash, save_predictions, selection_state)

VARIANTS = ('F1', 'F2')
HERE = Path(__file__).resolve().parent


def check_spec(spec):
    expected = {'schema': 'xgnid.frozen-fusion.v1', 'variants': list(VARIANTS),
        'seed': 42, 'batch_size': 512, 'cache_batch_size': 16, 'evaluation_batch_size': 16,
        'workers': 2, 'optimizer': 'Adam', 'lr': .001, 'weight_decay': 0., 'epochs': 30,
        'patience': 12, 'probe_batches': 3, 'include_epoch_zero': True,
        'loss': 'plain NLL after frozen H; only fusion weights train',
        'dynamic_input': 'original 21-dimensional representation statistics',
        'dynamic_initialization': 'copy unused N0 gate first layer; zero final layer',
        'proxy_weight_mixture': False, 'test_evaluated': False, 'automatic_followup': False}
    require(all(spec.get(k) == v for k, v in expected.items()), 'Frozen fusion protocol changed')


def metadata(args):
    spec = read(args.config)
    check_spec(spec)
    reference_path = inside(args.config.parent, spec['reference'])
    ref = read(reference_path)
    require(ref['schema'] == 'xgnid.n0-reference.v1' and ref['model']['variant'] == 'N0'
            and ref['model']['best_epoch'] == 5 and ref['model']['test_evaluated'] is False,
            'Wrong registered N0')
    root = inside(args.repo, ref['directory'])
    for folder in (root, root/'N0'):
        status = read(folder/'status.json')
        require((status['status'], status['exit_code']) == ('completed', 0), 'N0 not completed')
    metrics, config = read(root/'N0/metrics.json'), read(root/'N0/config.json')
    require(metrics == ref['model'] and config['binding'] == ref['binding'], 'N0 registration differs')
    summary = read(root/'summary.json')
    require(summary['models']['N0'] == metrics and summary['binding'] == ref['binding'], 'N0 summary differs')
    for name, digest in ref['binding']['sources'].items():
        require(sha(inside(args.repo, name)) == digest, 'Frozen source changed: ' + name)
    require(sha(root/'N0/best.pt') == metrics['checkpoint_sha256']['best.pt'], 'N0 checkpoint changed')
    require(sha(root/'N0/validation_predictions.jsonl.gz') == metrics['predictions_sha256'], 'N0 predictions changed')
    split_dir = inside(args.repo, spec['split_directory'])
    split = read(split_dir/'split.json')
    require(split['split_id'] == ref['binding']['split_id'] and split['partition_sizes'] ==
            {'train': 107621, 'val': 13446, 'test': 13448}, 'Wrong frozen split')
    require(sha(inside(split_dir, split['sample_manifest'])) == split['sample_manifest_sha256'] ==
            ref['binding']['manifest_sha256'], 'Manifest changed')
    binding = {'config_sha256': sha(args.config), 'reference_sha256': sha(reference_path),
        'n0_best_sha256': metrics['checkpoint_sha256']['best.pt'], 'n0_binding': ref['binding'],
        'sources': {n: sha(HERE/n) for n in ('run_frozen_fusion.py', 'run_e2_concat_pair.py', 'e3_gate_study.py')}}
    return spec, ref, root, binding


def matrix_report(cm, names):
    require(len(cm) == 8 and all(len(row) == 8 for row in cm), 'Expected eight classes')
    count = sum(map(sum, cm))
    require(count > 0, 'Empty matrix')
    classes = {}
    for i, name in enumerate(names):
        support, predicted, tp = sum(cm[i]), sum(row[i] for row in cm), cm[i][i]
        classes[name] = {'precision': tp/predicted if predicted else 0.,
            'recall': tp/support if support else 0.,
            'f1-score': 2*tp/(support+predicted) if support+predicted else 0., 'support': support}
    return {'accuracy': sum(cm[i][i] for i in range(8))/count,
        'macro_f1': sum(c['f1-score'] for c in classes.values())/8,
        'weighted_f1': sum(c['f1-score']*c['support'] for c in classes.values())/count,
        'per_class': classes, 'confusion_matrix': cm, 'samples': count}


def verify_order(exports, rows):
    require(len(exports) == len(rows), 'Prediction count differs')
    require(len({r['sample_id'] for r in exports}) == len(rows), 'Duplicate IDs')
    require(all(e['sample_id'] == r['sample_id'] and e['label'] == r['label']
                for e, r in zip(exports, rows)), 'Sample order/label differs from manifest')


def progress(args, phase, **details):
    value = {'status': 'completed' if phase == 'completed' else 'running',
             'phase': phase, 'pid': os.getpid(), **details}
    write(args.output/'progress.json', value)


def extract_cache(core, args, model, frozen, device, part, n0_predictions=None):
    rows = core.select_rows(frozen, part)
    loader = core.DataLoader(core.GraphRows(frozen, rows), batch_size=16, shuffle=False,
        num_workers=2, pin_memory=True, persistent_workers=True, multiprocessing_context='fork')
    pieces = {k: [] for k in ('flow', 'packet', 'features', 'y')}
    ids, full_predictions, max_error = [], [], 0.
    model.eval()
    with core.torch.no_grad():
        for step, batch in enumerate(loader, 1):
            batch = batch.to(device)
            if part == 'val':
                original = model(batch)
                zf, zp = model._last_cache['flow_logits'], model._last_cache['packet_logits']
                cached = core.F.log_softmax(model.shared_head(.5*zf+.5*zp), dim=-1)
                require(core.torch.allclose(original, cached, atol=1e-5, rtol=1e-5), 'F0 forward differs')
                max_error = max(max_error, float((original-cached).abs().max()))
                full_predictions.extend(original.argmax(-1).cpu().tolist())
            else:
                _, _, zf, zp = model._branch_outputs(batch)
            # EXACT input order of the historical 21->16->2 gate. These are
            # representation statistics; H(z) is not used as a prediction here.
            features = core.torch.cat((zf, zp,
                model._confidence_from_logits(zf).unsqueeze(-1),
                model._confidence_from_logits(zp).unsqueeze(-1),
                model._margin_from_logits(zf).unsqueeze(-1),
                model._margin_from_logits(zp).unsqueeze(-1),
                core.F.cosine_similarity(zf, zp, dim=-1).unsqueeze(-1)), dim=-1)
            require(features.shape[1] == 21 and bool(core.torch.isfinite(features).all()), 'Bad cached features')
            for key, tensor in (('flow', zf), ('packet', zp), ('features', features), ('y', batch.y.view(-1))):
                pieces[key].append(tensor.detach().cpu())
            ids.extend(batch.sample_id)
            if step == 1 or step % 100 == 0:
                progress(args, 'cache_'+part, batches=step, samples=len(ids), total=len(rows))
    cache = {k: core.torch.cat(v) for k, v in pieces.items()}
    cache['ids'] = ids
    require(ids == [r['sample_id'] for r in rows] and cache['y'].tolist() == [r['label'] for r in rows],
            'Cache order/labels differ from frozen manifest')
    if part == 'val':
        verify_order(n0_predictions, rows)
        require(full_predictions == [r['prediction'] for r in n0_predictions], 'N0 predictions not reproduced')
    core.checkpoint(args.output/(part+'_cache.pt'), cache)
    return cache, {'samples': len(rows), 'sample_order_sha256': identity(ids),
                   'file_sha256': sha(args.output/(part+'_cache.pt')), 'max_f0_output_error': max_error}


def device_cache(core, cache, device):
    return {k: v.to(device) if core.torch.is_tensor(v) else v for k, v in cache.items()}


def make_fusion(core, model, variant, device):
    if variant == 'F1':
        fusion = core.torch.nn.Module()
        fusion.register_parameter('logit', core.torch.nn.Parameter(core.torch.zeros((), device=device)))
    else:
        require(variant == 'F2', 'Only F1/F2 train')
        fusion = copy.deepcopy(model.gate.net)
        require((fusion[0].in_features, fusion[0].out_features, fusion[-1].out_features) == (21,16,2),
                'Original gate architecture differs')
        with core.torch.no_grad():
            fusion[-1].weight.zero_(); fusion[-1].bias.zero_()
    fusion.requires_grad_(True)
    require(sum(p.numel() for p in fusion.parameters()) == (1 if variant == 'F1' else 386),
            'Unexpected fusion parameter count')
    return fusion


def predict(core, head, fusion, variant, cache, index):
    flow, packet = cache['flow'][index], cache['packet'][index]
    if variant == 'F0':
        weights = flow.new_full((flow.size(0),2), .5)
    elif variant == 'F1':
        alpha = fusion.logit.sigmoid()
        weights = core.torch.stack((alpha, 1-alpha)).expand(flow.size(0),2)
    else:
        weights = core.torch.softmax(fusion(cache['features'][index]), dim=-1)
    mixed = weights[:,:1]*flow+weights[:,1:]*packet
    # H's parameters are frozen; DO NOT put this call under no_grad in training.
    logits = head(mixed)
    return core.F.log_softmax(logits, dim=-1), weights, mixed


def evaluate(core, model, fusion, variant, cache):
    matrix = [[0]*8 for _ in range(8)]
    exports, weights_all = [], []
    with core.torch.no_grad():
        for offset in range(0, len(cache['ids']), 16):
            index = slice(offset, offset+16)
            scores, weights, _ = predict(core, model.shared_head, fusion, variant, cache, index)
            require(bool(core.torch.isfinite(scores).all()), 'Nonfinite evaluation')
            labels, preds = cache['y'][index].tolist(), scores.argmax(-1).tolist()
            ws = weights.cpu().tolist()
            for j, (label, pred) in enumerate(zip(labels, preds)):
                matrix[label][pred] += 1
                exports.append({'sample_id': cache['ids'][offset+j], 'label': label, 'prediction': pred,
                                'flow_weight': ws[j][0], 'packet_weight': ws[j][1]})
            weights_all.append(weights.detach())
    weights = core.torch.cat(weights_all)
    metrics = matrix_report(matrix, core.LABELS)
    metrics['weights'] = {'mean': weights.mean(0).tolist(), 'std': weights.std(0,unbiased=False).tolist(),
                          'min': weights.min(0).values.tolist(), 'max': weights.max(0).values.tolist()}
    return metrics, exports


def update(core, model, fusion, variant, cache, indices, optimizer, probe=False):
    optimizer.zero_grad(set_to_none=True)
    prediction, _, mixed = predict(core, model.shared_head, fusion, variant, cache, indices)
    loss = core.F.nll_loss(prediction, cache['y'][indices])
    require(bool(core.torch.isfinite(loss)), 'Nonfinite loss')
    input_gradient = None
    if probe:
        gradient = core.torch.autograd.grad(loss, mixed, retain_graph=True)[0]
        require(bool(core.torch.isfinite(gradient).all()) and float(gradient.abs().sum()) > 0,
                'Frozen H blocked the fusion input gradient')
        input_gradient = float(gradient.abs().sum())
    loss.backward()
    require(all(p.grad is None for p in model.parameters()) and all(not m.training for m in model.modules()),
            'Frozen model gradient/eval invariant broken')
    norms = {n: float(p.grad.abs().sum()) if p.grad is not None else 0. for n,p in fusion.named_parameters()}
    require(all(p.grad is not None and bool(core.torch.isfinite(p.grad).all()) for p in fusion.parameters()),
            'Fusion gradient missing or nonfinite')
    before = core.state_hash(fusion.state_dict().items()) if probe else None
    optimizer.step()
    changed = before != core.state_hash(fusion.state_dict().items()) if probe else None
    if probe:
        require(changed and sum(norms.values()) > 0, 'Fusion failed to learn')
    return float(loss.detach()), {'head_input_gradient_l1': input_gradient,
                                  'fusion_gradients_l1': norms, 'parameters_changed': changed}


def probe_fusion(core, spec, model, variant, cache, device, start_rng):
    restore_rng(core, start_rng)
    fusion = make_fusion(core, model, variant, device)
    optimizer = core.torch.optim.Adam(fusion.parameters(),lr=spec['lr'],weight_decay=0.)
    initial = core.state_hash(fusion.state_dict().items())
    sampler = core.TrainBatches(len(cache['ids']),512,42)
    observations = []
    for step, indices in enumerate(sampler):
        if step == 3: break
        index = core.torch.tensor(indices,device=device)
        if step == 0:
            with core.torch.no_grad():
                _, weights, _ = predict(core,model.shared_head,fusion,variant,cache,index)
                require(bool((weights == .5).all()), 'Initial weights differ from equal fusion')
        _, receipt = update(core,model,fusion,variant,cache,index,optimizer,probe=True)
        observations.append(receipt)
    require(len(observations) == 3, 'Incomplete probe')
    if variant == 'F2':
        require(any(r['fusion_gradients_l1']['0.weight'] > 0 for r in observations[1:]),
                'Gate first layer did not receive gradients after zero initialization')
    # Discard the entire probe fusion+optimizer. Training constructs them afresh.
    return {'initial_fusion_sha256': initial, 'updates': observations, 'probe_discarded': True}


def train(core, args, spec, model, variant, caches, device, start_rng, f0, baseline, binding, probe):
    restore_rng(core, start_rng)
    fusion = make_fusion(core, model, variant, device)
    require(core.state_hash(fusion.state_dict().items()) == probe['initial_fusion_sha256'], 'Probe state leaked')
    rng_digest = rng_hash(core,rng_state(core))
    optimizer = core.torch.optim.Adam(fusion.parameters(),lr=spec['lr'],weight_decay=0.)
    output = args.output/variant
    output.mkdir()
    cfg = {'variant':variant,'protocol':spec,'binding':binding,'training_start_rng_sha256':rng_digest,
           'trainable_parameters':sum(p.numel() for p in fusion.parameters()), 'frozen_model_parameters_trainable':0}
    write(output/'config.json',cfg)
    initial, exports = evaluate(core,model,fusion,variant,caches['val'])
    require(initial['confusion_matrix'] == f0['confusion_matrix'] and
            [r['prediction'] for r in exports] == [r['prediction'] for r in baseline], 'Epoch0 differs from F0')
    best, best_epoch, stale, updates = f0['macro_f1'], 0, 0, 0
    selected_metrics = initial
    core.checkpoint(output/'best.pt',{'fusion_state':fusion.state_dict(),'epoch':0,'config':cfg,'validation':initial})
    sampler = core.TrainBatches(len(caches['train']['ids']),512,42)
    orders = []
    with (output/'epochs.jsonl').open('x',encoding='utf-8') as log:
        for epoch in range(1,spec['epochs']+1):
            fusion.train()
            samples, steps, loss_sum = 0, 0, 0.
            for indices in sampler:
                index = core.torch.tensor(indices,device=device)
                loss,_ = update(core,model,fusion,variant,caches['train'],index,optimizer)
                samples += len(indices); steps += 1; loss_sum += loss*len(indices)
            require(samples == 107621 and steps == 211, 'Incomplete cached training epoch')
            updates += steps; orders.append(sampler.last_order_sha256)
            fusion.eval()
            validation,_ = evaluate(core,model,fusion,variant,caches['val'])
            best,stale,improved = selection_state(validation['macro_f1'],best,stale)
            if improved:
                best_epoch,selected_metrics = epoch,validation
                core.checkpoint(output/'best.pt',{'fusion_state':fusion.state_dict(),'epoch':epoch,'config':cfg,'validation':validation})
            core.checkpoint(output/'last.pt',{'fusion_state':fusion.state_dict(),'optimizer_state':optimizer.state_dict(),
                'epoch':epoch,'best_epoch':best_epoch,'stale_epochs':stale,'optimizer_updates':updates,
                'sampler_rng_state':sampler.generator.get_state(),'rng':rng_state(core),'config':cfg,'validation':validation})
            entry = {'epoch':epoch,'train_loss':loss_sum/samples,'train_samples':samples,'optimizer_updates':steps,
                'cumulative_optimizer_updates':updates,'train_order_sha256':orders[-1],
                'val_macro_f1':validation['macro_f1'],'val_accuracy':validation['accuracy'],
                'weight_statistics':validation['weights'],'best_epoch':best_epoch,'stale_epochs':stale}
            log.write(json.dumps(entry)+'\n'); log.flush()
            print(json.dumps({'variant':variant,**entry}),flush=True)
            progress(args,'train',variant=variant,epoch=epoch,best_epoch=best_epoch)
            if stale >= spec['patience']: break
    chosen = core.torch.load(output/'best.pt',map_location=device,weights_only=False)
    fusion.load_state_dict(chosen['fusion_state'],strict=True)
    metrics,exports = evaluate(core,model,fusion,variant,caches['val'])
    require(metrics['confusion_matrix'] == selected_metrics['confusion_matrix'], 'Best validation not reproduced')
    save_predictions(output/'validation_predictions.jsonl.gz',exports)
    result = {'variant':variant,'best_epoch':best_epoch,'epochs':epoch,'optimizer_updates':updates,
        'training_samples_seen':107621*epoch,'best_validation':metrics,'last_validation':validation,
        'macro_f1_minus_F0_pp':100*(metrics['macro_f1']-f0['macro_f1']),
        'versus_F0':compare_predictions(baseline,exports),'orders':orders,'training_start_rng_sha256':rng_digest,
        'trainable_parameters':cfg['trainable_parameters'],'test_evaluated':False,
        'checkpoint_sha256':{n:sha(output/n) for n in ('best.pt','last.pt')},
        'predictions_sha256':sha(output/'validation_predictions.jsonl.gz')}
    write(output/'metrics.json',result)
    write(output/'status.json',{'status':'completed','exit_code':0})
    return result,exports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path.cwd())
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--mode',choices=('check','probe','queue'),default='check')
    parser.add_argument('--physical-gpu',type=int,default=1)
    parser.add_argument('--allow-shared-gpu',action='store_true')
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    args.repo=args.repo.resolve(); args.config=(args.repo/args.config).resolve()
    spec,ref,n0,binding=metadata(args)
    if args.mode == 'check':
        print(json.dumps({'status':'metadata_checks_passed','binding':binding,'training_started':False}),flush=True)
        return
    require(os.name == 'posix' and args.output is not None,'Use Linux wx and new output')
    args.output=inside(args.repo/'outputs',inside(args.repo,args.output))
    require(not args.output.exists(),'Refuse overwrite/resume')
    args.output.mkdir(parents=True)
    write(args.output/'status.json',{'status':'running','pid':os.getpid()})
    try:
        gpu=checked_gpu(args.physical_gpu,args.allow_shared_gpu)
        activate_gpu(gpu)
        write(args.output/'hardware.json',gpu)
        sys.path.insert(0,str(args.repo/'scripts'))
        import train_grouped as core
        device=runtime(core)
        core.seed_all(42)
        frozen=core.FrozenGraphs(inside(args.repo,spec['split_directory']),ref['binding']['split_id'])
        guard_graph_loads(frozen,{'train','val'})
        warm=core.Batch.from_data_list([frozen.load(r) for r in frozen.training_candidates()[:16]]).to(device)
        model=core.build_experiment('E2',warm).to(device)
        core.initialize(model,'E2',warm)
        checkpoint=core.torch.load(n0/'N0/best.pt',map_location=device,weights_only=False)
        require(checkpoint['epoch'] == 5 and checkpoint['validation']['confusion_matrix'] ==
                ref['model']['best_validation']['confusion_matrix'] and
                checkpoint['config'] == read(n0/'N0/config.json'),'Wrong N0 selected checkpoint/config')
        model.load_state_dict(checkpoint['model_state'],strict=True)
        model.requires_grad_(False); model.eval(); model.zero_grad(set_to_none=True)
        frozen_hash=core.state_hash(model.state_dict().items())
        with gzip.open(n0/'N0/validation_predictions.jsonl.gz','rt',encoding='utf-8') as stream:
            n0_predictions=[json.loads(line) for line in stream]
        val,vcert=extract_cache(core,args,model,frozen,device,'val',n0_predictions)
        val=device_cache(core,val,device)
        f0,baseline=evaluate(core,model,None,'F0',val)
        require(f0['confusion_matrix'] == ref['model']['best_validation']['confusion_matrix'] and
                abs(f0['macro_f1']-ref['model']['best_validation']['macro_f1']) < 1e-10 and
                [r['prediction'] for r in baseline] == [r['prediction'] for r in n0_predictions], 'F0 failed')
        write(args.output/'F0.json',{'status':'completed','best_validation':f0,'cache':vcert,
                                    'ordered_predictions_reproduced':True,'test_evaluated':False})
        save_predictions(args.output/'F0_validation_predictions.jsonl.gz',baseline)
        print(json.dumps({'phase':'F0_reproduced','macro_f1':f0['macro_f1']}),flush=True)
        training,tcert=extract_cache(core,args,model,frozen,device,'train')
        caches={'train':device_cache(core,training,device),'val':val}
        require(core.state_hash(model.state_dict().items()) == frozen_hash,'Caching changed frozen state')
        binding={**binding,'caches':{'train':tcert,'val':vcert},'frozen_state_sha256':frozen_hash}
        write(args.output/'binding.json',binding)
        start_rng=rng_state(core)
        probes={v:probe_fusion(core,spec,model,v,caches['train'],device,start_rng) for v in VARIANTS}
        require(core.state_hash(model.state_dict().items()) == frozen_hash,'Probe changed frozen model')
        write(args.output/'probe.json',{'status':'completed','variants':probes,'frozen_state_unchanged':True})
        print(json.dumps({'phase':'probe_passed','frozen_head_input_gradient_verified':True}),flush=True)
        if args.mode == 'probe':
            write(args.output/'status.json',{'status':'completed','exit_code':0,'mode':'probe'})
            return
        results,predictions={},{}
        for v in VARIANTS:
            results[v],predictions[v]=train(core,args,spec,model,v,caches,device,start_rng,f0,baseline,binding,probes[v])
            require(core.state_hash(model.state_dict().items()) == frozen_hash,'Training changed frozen parameters/buffers')
        common=min(len(results[v]['orders']) for v in VARIANTS)
        require(results['F1']['orders'][:common] == results['F2']['orders'][:common] and
            results['F1']['training_start_rng_sha256'] == results['F2']['training_start_rng_sha256'],'Unpaired training')
        write(args.output/'summary.json',{'status':'completed','exit_code':0,'binding':binding,'hardware':gpu,
            'F0':f0,'models':results,'F2_minus_F1_macro_f1_pp':100*(results['F2']['best_validation']['macro_f1']-
            results['F1']['best_validation']['macro_f1']),'F2_versus_F1':compare_predictions(predictions['F1'],predictions['F2']),
            'frozen_model_unchanged':True,'paired_sample_order_verified':True,'paired_common_epochs':common,
            'test_evaluated':False,'automatic_followup':False})
        write(args.output/'status.json',{'status':'completed','exit_code':0})
        progress(args,'completed')
    except BaseException:
        write(args.output/'status.json',{'status':'failed','exit_code':1})
        (args.output/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
