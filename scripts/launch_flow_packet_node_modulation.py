"""Publish recovery/logs on completion, catchable error and SIGINT/SIGTERM."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys

from filelock import FileLock
import launch_joint_transition_fusion as publisher
from launch_conditional_complement import uploaded_commit
from sync_server_outputs import lfs_environment

OUTPUT = 'flow_packet_node_modulation_v1'


def launch(args):
    code = Path(__file__).resolve().parents[1]
    repo = args.repo.resolve()
    root = repo / 'outputs' / OUTPUT
    root.mkdir(parents=True, exist_ok=True)
    (repo / 'tmp').mkdir(exist_ok=True)
    # Receipts are outside the sealed computation tree; still inside the published output.
    receipts = root / 'publication'
    receipts.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=os.pathsep.join([str(code / 'src'), str(code / 'scripts'), env.get('PYTHONPATH', '')]))
    child, caught, status, publication_error = None, None, 1, 0

    def stop(signum, frame):
        nonlocal caught
        caught = signum
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signum) if os.name == 'posix' else child.terminate()

    with FileLock(str(repo / 'tmp' / (OUTPUT + '.launcher.lock')), timeout=0):
        handlers = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            env['PATH'] = lfs_environment(repo)['PATH']
            command = [sys.executable, '-u', '-B', str(code / 'scripts/run_flow_packet_node_modulation.py'),
                       '--repo', str(repo), '--resume', args.resume, '--download-inputs']
            if args.data_root:
                command += ['--data-root', str(args.data_root.resolve())]
            if args.lock_only:
                command += ['--lock-only']
            if args.probe_only:
                command += ['--probe-only']
            with (receipts / 'training.log').open('ab', buffering=0) as log:
                log.write(('\n' + datetime.now(timezone.utc).isoformat() + ' ' + shlex.join(command) + '\n').encode())
                child = subprocess.Popen(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=os.name == 'posix')
                status = child.wait()
            if caught or status < 0:
                status = 128 + (caught or -status)
        except BaseException as exc:
            publisher.write(receipts / 'launcher_failure.json', {'exception': repr(exc)})
        finally:
            if caught:
                status = 128 + caught
            publisher.write(receipts / 'training_exit.json', {'exit_code': status, 'signal': caught,
                            'finished_utc': datetime.now(timezone.utc).isoformat()})
            for s in handlers:
                signal.signal(s, signal.SIG_IGN)
            try:
                # Capture publication logs externally to avoid inventory races with the publisher.
                exits = publisher.publish_results(code, repo, receipts, status, env, output_name=OUTPUT,
                                                   remote=args.publish_remote)
                if exits[-1] != 0:
                    raise RuntimeError('Publication failed: ' + str(exits))
                commit = uploaded_commit(receipts / 'upload.log')
                if status == 0:
                    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                    audit = [sys.executable, '-B', str(code / 'scripts/audit_flow_packet_node_archive.py'),
                             '--repo', str(repo), '--commit', commit,
                             '--output', str(receipts / (stamp + '_github_audit.json'))]
                    if args.publish_remote:
                        audit += ['--remote', args.publish_remote]
                    with (receipts / 'audit.log').open('wb') as log:
                        subprocess.run(audit, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                    # Upload the closed independent-audit receipt too.
                    exits = publisher.publish_results(code, repo, receipts, status, env, output_name=OUTPUT,
                                                       remote=args.publish_remote)
                    if exits[-1] != 0:
                        raise RuntimeError('Audit receipt publication failed')
            except BaseException as exc:
                publication_error = 1
                publisher.write(receipts / 'publication_failure.json', {'exception': repr(exc),
                                'training_exit_code': status})
                print('UPLOAD/AUDIT FAILED: ' + repr(exc), flush=True)
            finally:
                for s, handler in handlers.items():
                    signal.signal(s, handler)
    return status or publication_error


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path.cwd())
    p.add_argument('--data-root', type=Path)
    p.add_argument('--resume', choices=('auto', 'never'), default='auto')
    p.add_argument('--publish-remote')
    g = p.add_mutually_exclusive_group()
    g.add_argument('--lock-only', action='store_true')
    g.add_argument('--probe-only', action='store_true')
    sys.exit(launch(p.parse_args()))


if __name__ == '__main__':
    main()
