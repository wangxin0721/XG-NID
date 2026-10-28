"""Standard-library validation for interrupted B-H1024 checkpoint recovery."""
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import zipfile
from e3_gate_study import read, require, sha

FILES = ('config.json', 'epochs.jsonl', 'best.pt', 'last.pt')

def utc():
    return datetime.now(timezone.utc).isoformat()

def progress(path, **values):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps({'updated_utc':utc(), **values},indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)

def active_runs(targets, proc=Path('/proc'), own_pid=None):
    """Only inspect processes; never signal or stop another workload."""
    own_pid = os.getpid() if own_pid is None else own_pid
    targets = {str(Path(p).resolve()) for p in targets}
    found = []
    for path in proc.glob('[0-9]*/cmdline'):
        if int(path.parent.name)==own_pid:
            continue
        try:
            argv = path.read_bytes().decode('utf-8',errors='replace').strip('\0').split('\0')
        except (OSError, ValueError):
            continue
        for i,value in enumerate(argv[:-1]):
            if value=='--output' and str(Path(argv[i+1]).resolve()) in targets:
                found.append({'pid':int(path.parent.name),'argv':argv})
                break
    return found

def history(path):
    rows, raw = [], []
    lines = path.read_bytes().splitlines(keepends=True)
    for i,line in enumerate(lines):
        try:
            item = json.loads(line)
        except (ValueError, UnicodeError):
            require(i==len(lines)-1, 'Malformed non-final epoch log entry')
            break  # A crash may leave one partial final log entry.
        require(item['epoch']==len(rows)+1, 'Epoch log is not contiguous')
        require(item['optimizer_updates']==106 and item['cumulative_optimizer_updates']==106*item['epoch']
                and item['train_samples']==107621, 'Original batch/update accounting differs')
        rows.append(item)
        raw.append(line if line.endswith(b'\n') else line+b'\n')
    require(bool(rows), 'No completed epoch in source log')
    return rows, raw

def source_receipt(original, binding):
    folder = original/'B_Hstats'
    cfg = read(folder/'config.json')
    require(cfg['binding']==binding and cfg['variant']=='B_Hstats', 'Interrupted run binding differs')
    expected = {'batch_size':1024,'evaluation_batch_size':16,'initialization_batch_size':16,
                'seed':42,'lambda':.05,'epochs':100,'patience':12,'workers':2,'lr':.001}
    require(all(cfg.get(k)==v for k,v in expected.items()) and cfg['test_evaluated'] is False,
            'Interrupted training controls differ')
    for name in ('last.pt','best.pt'):
        with zipfile.ZipFile(folder/name) as archive:
            require(archive.testzip() is None, 'Corrupt checkpoint ZIP: '+name)
    rows, _ = history(folder/'epochs.jsonl')
    return cfg, {'original_run':str(original), 'files_sha256':{f:sha(folder/f) for f in FILES},
                 'last_logged_epoch':rows[-1]['epoch'], 'cause_of_exit':'unknown',
                 'checkpoint_tensors_loaded':False}

def validate_checkpoint(saved, cfg, rows):
    required = {'model_state','optimizer_state','config','epoch','best_epoch','best_val_macro_f1',
                'validation','sampler_rng_state','rng_state','optimizer_updates','stale_epochs'}
    require(required <= set(saved), 'Checkpoint lacks optimizer/sampler/RNG or stopping state')
    require(saved['config']==cfg, 'Embedded checkpoint config differs')
    epoch, best_epoch, stale = saved['epoch'], saved['best_epoch'], saved['stale_epochs']
    require(type(epoch) is int and 1<=best_epoch<=epoch<=100 and stale==epoch-best_epoch,
            'Invalid saved epoch or early-stopping counter')
    require(saved['optimizer_updates']==106*epoch, 'Checkpoint update budget differs')
    require(len(rows)>=epoch, 'Checkpoint is ahead of complete epoch log; manual audit required')
    row = rows[epoch-1]
    require(row['best_epoch']==best_epoch and row['stale_epochs']==stale
            and abs(row['val_macro_f1']-saved['validation']['macro_f1'])<1e-12,
            'Checkpoint and epoch log disagree')
    require(math.isfinite(saved['best_val_macro_f1']) and
            abs(max(r['val_macro_f1'] for r in rows[:epoch])-saved['best_val_macro_f1'])<1e-12,
            'Saved best score disagrees with history')
    require(set(saved['rng_state'])=={'python','torch','cuda'}, 'Incomplete global RNG state')
    groups = saved['optimizer_state']['param_groups']
    require(bool(groups) and all(g['lr']==.001 and g['weight_decay']==0 for g in groups),
            'Saved optimizer controls differ')
    return {'saved_epoch':epoch,'next_epoch':epoch+1,'best_epoch':best_epoch,'stale_epochs':stale,
            'optimizer_updates':saved['optimizer_updates'],'epoch_limit':100,'patience':12,
            'partial_epoch_discarded':True,'new_optimizer_updates_for_restoration':0}
