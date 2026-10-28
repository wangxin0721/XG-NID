"""Recover interrupted B-H1024 into a new directory; Linux wx only for training.

Metadata/ZIP checks are stdlib-only. Tensor checkpoints are loaded on the server.
Original experiment files and sources are never overwritten.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

from e3_gate_study import (activate_gpu, checked_gpu, guard_graph_loads, inside,
                          read, require, runtime, sha, write, identity)
from fusion_acb import rng_state, restore_rng, rng_hash, update
from fusion_bh import VARIANT, build_model, validate_probe
from fusion_bh_b1024_metadata import metadata
from run_fusion_bh_b1024_server import optimizer, best_diagnostic
from run_e3_gate_ablation_server import snapshot
from bh1024_recovery import (FILES, utc, progress, active_runs, history,
                             source_receipt, validate_checkpoint)

HERE=Path(__file__).resolve().parent

def optimizer_hash(core,state):
    tensors=[]
    scalars={}
    for parameter,values in state['state'].items():
        for name,value in values.items():
            key='%s/%s' % (parameter,name)
            if core.torch.is_tensor(value):
                tensors.append((key,value))
            else:
                scalars[key]=value
    return identity({'tensors':core.state_hash(tensors),'scalars':scalars,'param_groups':state['param_groups']})

def continue_training(core, args, frozen, device, binding, expected_b, gpu, config, cfg, receipt):
    original = args.original / VARIANT
    rows_history, raw_history = history(original / 'epochs.jsonl')
    for filename in FILES:
        shutil.copyfile(original/filename, args.output/filename)
        require(sha(args.output/filename)==receipt['files_sha256'][filename], 'Source changed during recovery copy')
    saved = core.torch.load(args.output/'last.pt',map_location='cpu',weights_only=False)
    restored = validate_checkpoint(saved,cfg,rows_history)
    selected_old = core.torch.load(args.output/'best.pt',map_location='cpu',weights_only=False)
    require(selected_old['config']==cfg and selected_old['epoch']==saved['best_epoch']
            and selected_old['best_val_macro_f1']==saved['best_val_macro_f1'], 'Best and last checkpoints disagree')
    del selected_old
    model, cert = build_model(core, frozen, device, expected_b)
    require(cert==cfg['initialization'], 'Original B-H model initialization/forward certificate differs')
    model.load_state_dict(saved['model_state'],strict=True)
    require(core.state_hash(model.state_dict().items())==core.state_hash(saved['model_state'].items()), 'Model state restore differs')
    opt = optimizer(core,model)
    opt.load_state_dict(saved['optimizer_state'])
    require(optimizer_hash(core,opt.state_dict())==optimizer_hash(core,saved['optimizer_state']), 'Optimizer state restore differs')
    rows = {p:core.select_rows(frozen,p) for p in ('train','val')}
    # Replaying index permutations checks the saved sampler state; zero graph
    # loads and zero optimizer updates occur in this replay.
    replay = core.TrainBatches(len(rows['train']),1024,42)
    for index in range(saved['epoch']):
        for indices in replay:
            pass
        require(replay.last_order_sha256==rows_history[index]['train_order_sha256'], 'Historical sample permutation differs')
    require(core.torch.equal(replay.generator.get_state(),saved['sampler_rng_state']), 'Saved sampler RNG does not follow history')
    sampler = core.TrainBatches(len(rows['train']),1024,42)
    sampler.generator.set_state(saved['sampler_rng_state'])
    # Dedicated worker seeding avoids consuming the restored model RNG when
    # replacing workers. Worker process states were not saved; no bitwise claim.
    kwargs = {'num_workers':2,'pin_memory':True,'persistent_workers':True,'multiprocessing_context':'fork'}
    train_loader = core.DataLoader(core.GraphRows(frozen,rows['train']),batch_sampler=sampler,
                    generator=core.torch.Generator().manual_seed(42),**kwargs)
    val_loader = core.DataLoader(core.GraphRows(frozen,rows['val']),batch_size=16,shuffle=False,
                    generator=core.torch.Generator().manual_seed(43),**kwargs)
    best, best_epoch, stale, updates, stopped = (saved['best_val_macro_f1'],saved['best_epoch'],
                    saved['stale_epochs'],saved['optimizer_updates'],saved['epoch'])
    epoch_start = saved['epoch']+1
    b_orders = [json.loads(line)['train_order_sha256'] for line in
                (inside(args.repo, config['acb_root'])/'B_dynamic/epochs.jsonl').read_text(encoding='utf-8').splitlines()]
    (args.output/'epochs.jsonl').write_bytes(b''.join(raw_history[:saved['epoch']]))
    receipt.update(restored)
    receipt.update({'created_utc':utc(),'resumed_on_physical_gpu':gpu['physical_index'],
                    'model_state_restored':True,'optimizer_state_restored':True,
                    'sampler_rng_and_history_verified':True,'worker_rng_not_saved':True,
                    'dedicated_loader_seed_generators':True,'bitwise_uninterrupted_equivalence_claimed':False,
                    'prior_logged_training_seconds':sum(r['train_seconds'] for r in rows_history[:saved['epoch']]),
                    'recovery_entry_sha256':sha(Path(__file__)),
                    'recovery_helper_sha256':sha(Path(__file__).with_name('bh1024_recovery.py'))})
    restore_rng(core,saved['rng_state'])
    receipt['restored_global_rng_sha256']=rng_hash(core,rng_state(core))
    require(receipt['restored_global_rng_sha256']==rng_hash(core,saved['rng_state']), 'Global RNG restore differs')
    write(args.output.parent/'recovery.json',receipt)
    del saved
    print(json.dumps({'recovery':receipt}),flush=True)
    started = time.perf_counter()
    with (args.output / 'epochs.jsonl').open('a', encoding='utf-8') as log:
        for epoch in range(epoch_start, 101) if stale < 12 else ():
            progress(args.output.parent/'progress.json',phase='training',epoch=epoch,batches_done=0,expected_batches=106)
            core.sync(device)
            tick = time.perf_counter()
            core.torch.cuda.reset_peak_memory_stats(device)
            sums = {k: 0. for k in ('loss_total', 'loss_fusion', 'loss_flow', 'loss_packet')}
            count, epoch_updates = 0, 0
            for batch in train_loader:
                batch = batch.to(device)
                item = update(core, model, opt, batch, 'B_dynamic')
                for k in sums:
                    sums[k] += item[k] * batch.num_graphs
                count += batch.num_graphs
                epoch_updates += 1
                if epoch_updates==1 or epoch_updates%10==0:
                    progress(args.output.parent/'progress.json',phase='training',epoch=epoch,batches_done=epoch_updates,expected_batches=106)
            core.sync(device)
            train_seconds = time.perf_counter() - tick
            require(count == 107621 and epoch_updates == len(sampler) == 106, 'Incomplete training epoch')
            if epoch <= len(b_orders):
                require(sampler.last_order_sha256 == b_orders[epoch-1], 'Epoch data order differs from paired B')
            updates += epoch_updates
            progress(args.output.parent/'progress.json',phase='validation',epoch=epoch,batches_done=epoch_updates)
            validation, _ = core.evaluate(model, val_loader, device)
            improved = validation['macro_f1'] > best
            if improved:
                best, best_epoch, stale = validation['macro_f1'], epoch, 0
            else:
                stale += 1
            entry = {'epoch': epoch, **{k: v/count for k, v in sums.items()}, 'train_samples': count,
                     'train_seconds': train_seconds, 'train_order_sha256': sampler.last_order_sha256,
                     'optimizer_updates': epoch_updates, 'cumulative_optimizer_updates': updates,
                     'val_accuracy': validation['accuracy'], 'val_macro_f1': validation['macro_f1'],
                     'val_weighted_f1': validation['weighted_f1'], 'gate_weights': validation['gate_weights'],
                     'best_epoch': best_epoch, 'stale_epochs': stale,
                     'peak_cuda_allocated_bytes': core.torch.cuda.max_memory_allocated(device),
                     'other_compute_processes_snapshot': snapshot(gpu)}
            payload = {'model_state': model.state_dict(), 'optimizer_state': opt.state_dict(),
                       'config': cfg, 'epoch': epoch, 'best_epoch': best_epoch, 'best_val_macro_f1': best,
                       'validation': validation, 'sampler_rng_state': sampler.generator.get_state(),
                       'recovery_sha256':sha(args.output.parent/'recovery.json'),
                       'rng_state': rng_state(core), 'optimizer_updates': updates, 'stale_epochs': stale}
            core.checkpoint(args.output / 'last.pt', payload)
            if improved:
                core.checkpoint(args.output / 'best.pt', payload)
            log.write(json.dumps(entry) + '\n')
            log.flush()
            print(json.dumps({'variant': args.variant, **entry}), flush=True)
            stopped = epoch
            if stale >= 12:
                break
    selected = core.torch.load(args.output / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(selected['model_state'], strict=True)
    detail = best_diagnostic(core, model, val_loader, rows['val'], device, args.output, selected['validation'])
    expected_ids = read(inside(args.repo, read(args.config)['original_diagnostic']))['models']['E3']['ordered_validation_identity_sha256']
    require(detail['ordered_validation_identity_sha256'] == expected_ids, 'Validation ordering changed')
    write(args.output / 'diagnostics.json', detail)
    metrics = {'variant': args.variant, 'best_epoch': best_epoch, 'stopped_epoch': stopped,
               'best_validation': selected['validation'], 'parameters': cfg['parameters'], 'initialization': cert,
               'optimizer_updates': updates, 'training_samples_seen': 107621 * stopped,
               'wall_seconds': time.perf_counter() - started, 'wall_seconds_scope':'resume_segment_only',
               'recovery_sha256':sha(args.output.parent/'recovery.json'), 'config_sha256': sha(args.output / 'config.json'),
               'checkpoint_sha256': {f: sha(args.output / f) for f in ('best.pt', 'last.pt')},
               'diagnostics_sha256': sha(args.output / 'diagnostics.json'), 'test': None, 'test_evaluated': False}
    write(args.output / 'metrics.json', metrics)
    return {'status': 'completed', 'exit_code': 0, 'mode': 'screen', 'variant': args.variant,
            'binding': binding, 'hardware': gpu, 'model': metrics, 'test_evaluated': False}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path.cwd())
    parser.add_argument('--config',type=Path,default=HERE.parent/'configs/fusion_bh_b1024_study_20261027.json')
    parser.add_argument('--reference-config',type=Path,default=HERE.parent/'configs/e3_gate_study.json')
    parser.add_argument('--original',type=Path,default=Path('outputs/fusion_bh_b1024_s42_v1'))
    parser.add_argument('--output',type=Path,default=Path('outputs/fusion_bh_b1024_s42_resume_v1'))
    parser.add_argument('--physical-gpu',type=int,choices=(0,1),default=1)
    parser.add_argument('--allow-shared-gpu',action='store_true')
    parser.add_argument('--mode',choices=('check','resume'),default='check')
    args=parser.parse_args()
    args.repo=args.repo.resolve()
    args.config,args.reference_config=args.config.resolve(),args.reference_config.resolve()
    args.original=inside(args.repo/'outputs',(args.repo/args.original).resolve())
    args.output=inside(args.repo/'outputs',(args.repo/args.output).resolve())
    require(args.output!=args.repo/'outputs' and args.original!=args.output
            and args.original not in args.output.parents and args.output not in args.original.parents,
            'Recovery requires a separate new output directory')
    require(not args.output.exists(), 'Recovery output exists; inspect it before choosing a new directory')
    targets=(args.original,args.original/VARIANT,args.output,args.output/VARIANT)
    require(not active_runs(targets),'Original or recovery training is still alive; no duplicate launch')
    config,spec,binding,expected_b=metadata(args)
    cfg,receipt=source_receipt(args.original,binding)
    probe=args.original/'probe/summary.json'
    require(sha(probe)==cfg['probe_sha256'],'Original probe report changed')
    validate_probe(read(probe),read(probe.parent/'status.json'),binding)
    if args.mode=='check':
        print(json.dumps({'status':'recovery_metadata_and_zip_checks_passed','source':receipt,
                          'gpu_touched':False,'checkpoint_tensors_loaded':False,'training_started':False}),flush=True)
        return 0
    require(os.name=='posix','Checkpoint loading and recovery run only on Linux wx')
    args.output.mkdir(exist_ok=False)
    child=args.output/VARIANT
    child.mkdir()
    for path,mode in ((args.output,'queue'),(child,'screen')):
        write(path/'status.json',{'status':'running','mode':mode,'pid':os.getpid(),'variant':VARIANT,'recovery':True})
    progress(args.output/'progress.json',phase='restoring_checkpoint')
    try:
        gpu=checked_gpu(args.physical_gpu,args.allow_shared_gpu)
        activate_gpu(gpu)
        sys.path.insert(0,str(args.repo/'scripts'))
        import train_grouped as core
        device=runtime(core)
        frozen=core.FrozenGraphs(inside(args.repo,spec['split_directory']),spec['split_id'])
        guard_graph_loads(frozen,{'train','val'})
        child_args=SimpleNamespace(**vars(args))
        child_args.output,child_args.variant=child,VARIANT
        result=continue_training(core,child_args,frozen,device,binding,expected_b,gpu,config,cfg,receipt)
        write(child/'summary.json',result)
        write(child/'status.json',{'status':'completed','exit_code':0,'mode':'screen','variant':VARIANT})
        progress(args.output/'progress.json',phase='offline_validation_analysis')
        from analyze_fusion_bh_b1024_results import analyze
        analysis=analyze(args.repo,args.output,config,binding)
        write(args.output/'analysis.json',analysis)
        write(args.output/'summary.json',{'status':'completed','exit_code':0,'mode':'queue','variant':VARIANT,
              'binding':binding,'model':result['model'],'recovery':receipt,
              'analysis_sha256':sha(args.output/'analysis.json'),'decision':analysis['decision'],'test_evaluated':False})
        write(args.output/'status.json',{'status':'completed','exit_code':0,'mode':'queue','variant':VARIANT})
        progress(args.output/'progress.json',phase='completed')
        print(json.dumps({'status':'completed','output':str(args.output),'test_evaluated':False}),flush=True)
    except BaseException as exc:
        code=130 if isinstance(exc,KeyboardInterrupt) else 1
        for path,mode in ((args.output,'queue'),(child,'screen')):
            old=read(path/'status.json')
            if old['status']!='completed':
                write(path/'status.json',{'status':'failed','exit_code':code,'mode':mode,'variant':VARIANT})
        progress(args.output/'progress.json',phase='failed',error=repr(exc))
        (args.output/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
        raise
    return 0

if __name__=='__main__':
    sys.exit(main())
