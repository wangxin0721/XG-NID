"""Run import -> C43 three arms -> C44 three arms -> aggregate; publish on exits."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys

from filelock import FileLock
import launch_joint_transition_fusion as publisher

OUTPUT_NAME='fixed_transition_stability_v1'


def phases(import_only=False,probe_only=False):
    return ('import',) if import_only else ('import','seed43','seed44') if probe_only else ('import','seed43','seed44','aggregate')


def launch(args):
    code,repo=Path(__file__).resolve().parents[1],args.repo.resolve()
    output=repo/'outputs'/OUTPUT_NAME;output.mkdir(parents=True,exist_ok=True)
    (repo/'tmp').mkdir(exist_ok=True)
    env=os.environ.copy();env['PYTHONUNBUFFERED']='1';env['PYTHONDONTWRITEBYTECODE']='1'
    env['PYTHONPATH']=os.pathsep.join([str(code/'src'),str(code/'scripts'),env.get('PYTHONPATH','')])
    child,caught=None,None
    def stop(signum,frame):
        nonlocal caught
        caught=signum
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signum) if os.name=='posix' else child.terminate()
    with FileLock(str(repo/'tmp'/(OUTPUT_NAME+'.launcher.lock')),timeout=0):
        handlers={s:signal.signal(s,stop) for s in (signal.SIGINT,signal.SIGTERM)}
        status=1;current='preparing'
        ledger_path=output/'phase_exit_codes.json'
        import json
        attempts=json.loads(ledger_path.read_text(encoding='utf-8'))['attempts'] if ledger_path.exists() else []
        uploads_path=output/'backbone_uploads.json'
        uploads=json.loads(uploads_path.read_text(encoding='utf-8'))['uploads'] if uploads_path.exists() else []
        history=output/'publication_history';history.mkdir(exist_ok=True)
        prior_stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        for name in ('upload.log','exit_codes.json','upload_failure.json'):
            if (output/name).is_file():shutil.copy2(output/name,history/('prior_'+prior_stamp+'_'+name))
        def publish():
            try:
                result=publisher.publish_results(code,repo,output,status,env,output_name=OUTPUT_NAME,
                                                 remote=args.publish_remote)
                return result
            except Exception as exc:
                command=[sys.executable,'-B',str(code/'scripts/sync_server_outputs.py'),'publish','--repo',str(repo),'--only',OUTPUT_NAME]
                if args.publish_remote: command+=['--remote',args.publish_remote]
                publisher.write(output/'upload_failure.json',{'phase':current,'training_exit_code':status,
                    'exception':repr(exc),'retry_command':shlex.join(command)})
                print('UPLOAD FAILED: '+repr(exc)+'\nRetry: '+shlex.join(command),file=sys.stderr,flush=True)
                return [1]
        try:
            for current in phases(args.import_only,args.probe_only):
                command=[sys.executable,'-u','-B',str(code/'scripts/run_fixed_transition_stability.py'),
                    '--repo',str(repo),'--phase',current,'--resume','--download-inputs']
                if args.data_root is not None: command+=['--data-root',str(args.data_root.resolve())]
                if args.probe_only: command.append('--probe-only')
                started=datetime.now(timezone.utc).isoformat()
                with (output/'training.log').open('ab',buffering=0) as log:
                    log.write(('\n'+started+' '+shlex.join(command)+'\n').encode())
                    child=subprocess.Popen(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,
                                           start_new_session=os.name=='posix')
                    status=child.wait()
                status=128+(caught or -status) if status<0 or caught else status
                attempts.append({'phase':current,'exit_code':status,'started_utc':started,
                                 'finished_utc':datetime.now(timezone.utc).isoformat()})
                publisher.write(ledger_path,{'attempts':attempts})
                if status!=0: break
                if current in ('seed43','seed44') and not args.probe_only:
                    for s in handlers: signal.signal(s,signal.SIG_IGN)
                    uploaded=publish()
                    for s in handlers: signal.signal(s,stop)
                    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                    for name in ('upload.log','exit_codes.json','upload_failure.json'):
                        if (output/name).is_file():shutil.copy2(output/name,history/(current+'_'+stamp+'_'+name))
                    uploads.append({'phase':current,'upload_attempt_exit_codes':uploaded,'finished_utc':stamp})
                    publisher.write(uploads_path,{'uploads':uploads})
                if caught:
                    status=128+caught;break
        except BaseException as exc:
            status=128+caught if caught else 1
            publisher.write(output/'launcher_failure.json',{'phase':current,'exception':repr(exc),'training_exit_code':status})
        finally:
            publisher.write(output/'training_exit.json',{'exit_code':status,'last_phase':current,'signal':caught,
                'import_only':args.import_only,'probe_only':args.probe_only,'finished_utc':datetime.now(timezone.utc).isoformat()})
            for s in handlers: signal.signal(s,signal.SIG_IGN)
            try: publish()
            finally:
                for s,handler in handlers.items(): signal.signal(s,handler)
    return status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path.cwd());parser.add_argument('--data-root',type=Path)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--import-only',action='store_true');group.add_argument('--probe-only',action='store_true')
    parser.add_argument('--publish-remote')
    sys.exit(launch(parser.parse_args()))


if __name__=='__main__': main()
