"""Read-only cleanup audit. This program never deletes or modifies outputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path

OLD_POOL=('packet_pool_pair_s42_v1','packet_pool_pair_s42_recovery_v1','packet_pool_pair_s42_recovery_v2')
CANONICAL='packet_pool_confirmation_s42_v1'
KEEP=('e2_concat_pair_b512_s42_lr1e4_v1','joint_gate_lrscale01_s42_v1','fusion_dimension_pair_s42_recovery_v1',
      'fusion_gate_pair_s42_v1','cross_layer_pair_s42_v1','cross_layer_fold_check_v1.json',
      'packet_pool_confirmation_s42_v1','relation_pair_s42_v1','residual_pair_s42_v1','residual_pair_s42_retry_v1',
      'r0_graph_rebuild_author_offline_v2','r0_validation_split_s42_v1','server_graph_split_audit_v1')


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def files(root):
    for base,dirs,names in os.walk(root,followlinks=False):
        dirs[:]=[d for d in dirs if not (Path(base)/d).is_symlink()]
        for name in names:
            p=Path(base)/name
            if p.is_file() and not p.is_symlink():yield p


def live_jobs():
    proc=Path('/proc')
    if os.name!='posix' or not proc.is_dir():return {'checked':False,'reason':'Server process state cannot be inferred from copied Windows status/PID files','jobs':[]}
    jobs=[]
    for folder in proc.iterdir():
        if not folder.name.isdigit() or int(folder.name)==os.getpid():continue
        try:args=(folder/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        script=next((x for x in args if x.endswith('.py') and
            (Path(x).name.startswith(('run_','recover_','train_')) or Path(x).name=='main.py')),None)
        if script:
            outputs=[args[i+1] for i,x in enumerate(args[:-1]) if x in ('--output','--output-dir')]
            jobs.append({'pid':int(folder.name),'script':Path(script).name,'outputs':outputs})
    return {'checked':True,'jobs':jobs}


def audit(repo):
    root=repo/'outputs';copies=[];cache=[]
    nested=root/'outputs'
    if nested.is_dir():
        matched=[];unmatched=[];size=0
        for p in files(nested):
            rel=p.relative_to(nested);q=root/rel
            if q.is_file() and not q.is_symlink() and p.stat().st_size==q.stat().st_size and sha(p)==sha(q):
                matched.append(rel.as_posix());size+=p.stat().st_size
            else:unmatched.append(rel.as_posix())
        copies.append({'path':'outputs/','matching_files':len(matched),'matching_bytes':size,'unmatched':unmatched,
            'eligible_as_complete_duplicate':bool(matched) and not unmatched})
    for folder in OLD_POOL:
        for name in ('train_packet_nodes.pt','val_packet_nodes.pt'):
            p=root/folder/name;q=root/CANONICAL/name
            if p.is_file() and q.is_file():
                digest=sha(p);cache.append({'path':folder+'/'+name,'bytes':p.stat().st_size,'sha256':digest,
                    'retained_copy':CANONICAL+'/'+name,'identical':digest==sha(q),
                    'condition':'Old recovery/confirmation hash checks require the original paths; restore them or use hard-link deduplication if those checks must remain runnable'})
    optional=[]
    for rel in ('relation_pair_s42_v1/train_pre_relation.pt','cross_layer_pair_s42_v1/train_cross_layer.pt'):
        p=root/rel
        if p.is_file():optional.append({'path':rel,'bytes':p.stat().st_size,
            'condition':'Completed experiment and verified backup; needed for replaying training, not current residual training. Keep validation cache and all metadata/checkpoints'})
    states={}
    for p in sorted(root.iterdir()):
        if not p.is_dir() or p.is_symlink():continue
        status=p/'status.json'
        if status.is_file():
            try:states[p.name]=json.loads(status.read_text(encoding='utf-8-sig'))
            except (ValueError,OSError):states[p.name]={'status':'unreadable'}
    return {'mode':'read_only_no_deletions','repo':str(repo),'outputs':str(root),
        'duplicate_directory_candidates':copies,'duplicate_cache_candidates':cache,
        'archive_only_training_cache_candidates':optional,
        'keep_roots':[{'path':p,'exists_here':(root/p).exists()} for p in KEEP],
        'on_disk_states_not_proof_of_liveness':states,'server_process_audit':live_jobs(),
        'rule':'Never delete active run files, source-lineage artifacts, best models, metadata or arbitrary *.pt. Candidate is a review suggestion, not authorization to delete.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,default=Path.cwd());a=p.parse_args()
    print(json.dumps(audit(a.repo.resolve()),ensure_ascii=False,indent=2))
