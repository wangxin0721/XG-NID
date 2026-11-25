"""Automatic gate-controlled queue, signals, uploads and immutable GitHub audits."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys

from filelock import FileLock
import launch_joint_transition_fusion as publisher
from sync_server_outputs import lfs_environment

OUTPUT='conditional_complement_v1'


def next_phase(root,completed,lock_only=False,probe_only=False):
    if not completed:return 'lock'
    previous=completed[-1]
    if lock_only:return None
    if previous=='lock':return 'prepare42'
    if previous=='prepare42':return 'seed42'
    if previous=='seed42':
        if probe_only:return None
        d=json.loads((root/'seed42/summary.json').read_text(encoding='utf8'))['decision']
        return 'prepare43' if d['passed'] else None
    chain={'prepare43':'seed43','seed43':'prepare44','prepare44':'seed44','seed44':'aggregate',
           'lock_test':'test42','test42':'test43','test43':'test44','test44':'test_aggregate'}
    if previous=='aggregate':
        result=json.loads((root/'aggregate/summary.json').read_text(encoding='utf8'))
        return 'lock_test' if result['passed'] else None
    return chain.get(previous)


def uploaded_commit(path):
    found=[]
    for line in path.read_text(encoding='utf8',errors='replace').splitlines():
        try:row=json.loads(line)
        except json.JSONDecodeError:continue
        if row.get('status')=='uploaded':found.append(row['commit'])
    if not found:raise RuntimeError('No successful immutable commit receipt in upload.log')
    return found[-1]


def launch(args):
    code=Path(__file__).resolve().parents[1];repo=args.repo.resolve();root=repo/'outputs'/OUTPUT
    root.mkdir(parents=True,exist_ok=True);(repo/'tmp').mkdir(exist_ok=True)
    env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=os.pathsep.join([str(code/'src'),str(code/'scripts'),env.get('PYTHONPATH','')]))
    child=None;caught=None
    def stop(signum,frame):
        nonlocal caught
        caught=signum
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signum) if os.name=='posix' else child.terminate()
    status=1;queue_exit=0;current='initialization';completed=[];history=root/'publication_history';history.mkdir(exist_ok=True)
    attempts_path=root/'phase_exit_codes.json'
    attempts=json.loads(attempts_path.read_text(encoding='utf8'))['attempts'] if attempts_path.exists() else []
    def retry():
        return [sys.executable,'-B',str(code/'scripts/sync_server_outputs.py'),'publish','--repo',str(repo),'--only',OUTPUT]+(
            ['--remote',args.publish_remote] if args.publish_remote else [])
    def publish(audit=False):
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        for name in ('upload.log','exit_codes.json','upload_failure.json'):
            if (root/name).exists():shutil.copy2(root/name,history/(stamp+'_'+name))
        try:
            exits=publisher.publish_results(code,repo,root,status,env,output_name=OUTPUT,remote=args.publish_remote)
            if exits[-1]!=0:raise RuntimeError('Upload attempt exit codes '+str(exits))
            commit=uploaded_commit(root/'upload.log')
            if audit:
                # The evaluator has exited; audit writes only its own receipt after reading GitHub.
                command=[sys.executable,'-B',str(code/'scripts/audit_conditional_complement_archive.py'),
                    '--repo',str(repo),'--commit',commit,'--output',str(history/(stamp+'_github_audit.json'))]
                if args.publish_remote:command+=['--remote',args.publish_remote]
                with (repo/'tmp'/(OUTPUT+'.audit.log')).open('wb') as log:
                    subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                shutil.copy2(repo/'tmp'/(OUTPUT+'.audit.log'),history/(stamp+'_github_audit.log'))
            return commit
        except BaseException as exc:
            publisher.write(root/'upload_failure.json',{'phase':current,'training_exit_code':status,
                'exception':repr(exc),'retry_command':shlex.join(retry())})
            print('UPLOAD/AUDIT FAILED: '+repr(exc)+'\nRetry: '+shlex.join(retry()),flush=True)
            raise
    with FileLock(str(repo/'tmp'/(OUTPUT+'.launcher.lock')),timeout=0):
        handlers={s:signal.signal(s,stop) for s in (signal.SIGINT,signal.SIGTERM)}
        try:
            env['PATH']=lfs_environment(repo)['PATH']
            publisher.write(root/'lfs_preflight.json',{'passed':True,'selective_publish_directory':OUTPUT,
                'publisher_registration':'dynamic immediate-child selection; no static whitelist'})
            while not caught:
                current=next_phase(root,completed,args.lock_only,args.probe_only)
                if current is None:break
                command=[sys.executable,'-u','-B',str(code/'scripts/run_conditional_complement.py'),
                    '--repo',str(repo),'--phase',current,'--resume',args.resume,'--download-inputs']
                if args.data_root:command+=['--data-root',str(args.data_root.resolve())]
                if args.probe_only:command+=['--probe-only']
                started=datetime.now(timezone.utc).isoformat()
                with (root/'training.log').open('ab',buffering=0) as log:
                    log.write(('\n'+started+' '+shlex.join(command)+'\n').encode())
                    child=subprocess.Popen(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=os.name=='posix')
                    status=child.wait()
                if caught or status<0:status=128+(caught or -status)
                attempts.append({'phase':current,'exit_code':status,'started_utc':started,'finished_utc':datetime.now(timezone.utc).isoformat()})
                publisher.write(attempts_path,{'attempts':attempts})
                if status!=0:break
                # Persist and publish each stage before executing its dependent stage.
                publish(audit=True)
                completed.append(current)
            if status==0 and not caught:
                last=completed[-1] if completed else None
                if last=='seed42' and not args.probe_only:
                    summary=json.loads((root/'seed42/summary.json').read_text(encoding='utf8'))
                    publisher.write(root/'summary.json',summary)
                    for name in ('report.md','paper_tables.md','paper_tables.csv'):
                        shutil.copy2(root/'seed42'/name,root/name)
                publisher.write(root/'queue_completion.json',{'completed_phases':completed,'last_phase':last,
                    'stopped_by_registered_gate':last in ('seed42','aggregate') and not args.probe_only,
                    'lock_only':args.lock_only,'probe_only':args.probe_only,
                    'test_evaluated':last=='test_aggregate'})
        except BaseException as exc:
            queue_exit=1
            publisher.write(root/'launcher_failure.json',{'phase':current,'exception':repr(exc),
                'training_exit_code':status,'queue_exit_code':queue_exit})
        finally:
            if caught:status=128+caught
            publisher.write(root/'training_exit.json',{'exit_code':status,'queue_exit_code':status or queue_exit,'signal':caught,'last_phase':current,
                'finished_utc':datetime.now(timezone.utc).isoformat()})
            for sig in handlers:signal.signal(sig,signal.SIG_IGN)
            try:
                publish(audit=False)
            except BaseException:queue_exit=1 # Separate upload_failure evidence; retain the training exit status.
            finally:
                for sig,handler in handlers.items():signal.signal(sig,handler)
    return status or queue_exit


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,default=Path.cwd())
    p.add_argument('--stage',choices=('screen',),default='screen');p.add_argument('--backbone-seed',type=int,choices=(42,),default=42)
    p.add_argument('--resume',choices=('auto','never'),default='auto');p.add_argument('--publish-on-exit',action='store_true',default=True)
    p.add_argument('--data-root',type=Path);p.add_argument('--publish-remote')
    g=p.add_mutually_exclusive_group();g.add_argument('--lock-only',action='store_true');g.add_argument('--probe-only',action='store_true')
    sys.exit(launch(p.parse_args()))


if __name__=='__main__':main()
