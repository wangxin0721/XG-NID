"""Publish fixed-test locks, each completed backbone, and final results, including failures."""
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

OUTPUT_NAME = 'fixed_transition_final_test_v1'


def phases(lock_only=False, probe_only=False):
    return ('lock',) if lock_only else ('lock', 'seed42', 'seed43', 'seed44') if probe_only else (
        'lock', 'seed42', 'seed43', 'seed44', 'aggregate')


def launch(args):
    code, repo = Path(__file__).resolve().parents[1], args.repo.resolve()
    output = repo / 'outputs' / OUTPUT_NAME
    output.mkdir(parents=True, exist_ok=True)
    (repo / 'tmp').mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=os.pathsep.join([str(code / 'src'), str(code / 'scripts'), env.get('PYTHONPATH', '')]))
    child, caught = None, None
    def stop(signum, frame):
        nonlocal caught
        caught = signum
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signum) if os.name == 'posix' else child.terminate()
    with FileLock(str(repo / 'tmp' / (OUTPUT_NAME + '.launcher.lock')), timeout=0):
        handlers = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
        status, current = 1, 'preparing'
        ledger_path = output / 'phase_exit_codes.json'
        attempts = json.loads(ledger_path.read_text(encoding='utf-8'))['attempts'] if ledger_path.exists() else []
        history = output / 'publication_history'
        history.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        for name in ('upload.log', 'exit_codes.json', 'upload_failure.json'):
            if (output / name).is_file():
                shutil.copy2(output / name, history / ('prior_' + stamp + '_' + name))
        def publish():
            try:
                result = publisher.publish_results(code, repo, output, status, env, output_name=OUTPUT_NAME,
                    remote=args.publish_remote, exit_key='evaluation_exit_code')
                if result[-1] != 0:
                    publisher.write(output / 'upload_failure.json', {'phase': current,
                        'evaluation_exit_code': status, 'upload_attempt_exit_codes': result,
                        'retry_command': shlex.join(retry_command())})
                return result
            except BaseException as exc:
                command = retry_command()
                publisher.write(output / 'upload_failure.json', {'phase': current,
                    'evaluation_exit_code': status, 'exception': repr(exc), 'retry_command': shlex.join(command)})
                print('UPLOAD FAILED: ' + repr(exc) + '\nRetry: ' + shlex.join(command), file=sys.stderr, flush=True)
                return [1]
        def retry_command():
            command = [sys.executable, '-B', str(code / 'scripts/sync_server_outputs.py'),
                       'publish', '--repo', str(repo), '--only', OUTPUT_NAME]
            return command + ['--remote', args.publish_remote] if args.publish_remote else command
        try:
            # Initialize repository-local Git LFS before the first immutable archive read.
            env['PATH'] = lfs_environment(repo)['PATH']
            publisher.write(output / 'lfs_preflight.json', {'passed': True, 'before_phase': 'lock'})
            for current in phases(args.lock_only, args.probe_only):
                command = [sys.executable, '-u', '-B', str(code / 'scripts/run_fixed_transition_final_test.py'),
                           '--repo', str(repo), '--phase', current, '--resume', '--download-inputs']
                if args.data_root is not None:
                    command += ['--data-root', str(args.data_root.resolve())]
                if args.probe_only:
                    command.append('--probe-only')
                started = datetime.now(timezone.utc).isoformat()
                with (output / 'evaluation.log').open('ab', buffering=0) as log:
                    log.write(('\n' + started + ' ' + shlex.join(command) + '\n').encode())
                    child = subprocess.Popen(command, cwd=repo, env=env, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=os.name == 'posix')
                    status = child.wait()
                status = 128 + (caught or -status) if status < 0 or caught else status
                attempts.append({'phase': current, 'exit_code': status, 'started_utc': started,
                                 'finished_utc': datetime.now(timezone.utc).isoformat()})
                publisher.write(ledger_path, {'attempts': attempts})
                if status != 0:
                    state_path = output / 'status.json'
                    previous = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {}
                    publisher.write(state_path, {**previous, 'status': 'failed', 'phase': current,
                        'queue_completed': False, 'evaluation_exit_code': status, 'signal': caught,
                        'new_optimizer_updates': 0})
                    break
                # The checkpoint lock is uploaded before test phases, independent of test scores.
                if current in ('lock', 'seed42', 'seed43', 'seed44'):
                    for s in handlers:
                        signal.signal(s, signal.SIG_IGN)
                    uploaded = publish()
                    for s in handlers:
                        signal.signal(s, stop)
                    phase_stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                    publisher.write(history / (current + '_' + phase_stamp + '.json'),
                        {'phase': current, 'upload_attempt_exit_codes': uploaded})
                if caught:
                    status = 128 + caught
                    break
        except BaseException as exc:
            status = 128 + caught if caught else 1
            publisher.write(output / 'launcher_failure.json', {'phase': current,
                'exception': repr(exc), 'evaluation_exit_code': status})
            publisher.write(output / 'status.json', {'status': 'failed', 'phase': current,
                'queue_completed': False, 'evaluation_exit_code': status, 'exception': repr(exc),
                'new_optimizer_updates': 0})
        finally:
            publisher.write(output / 'evaluation_exit.json', {'exit_code': status, 'last_phase': current,
                'signal': caught, 'lock_only': args.lock_only, 'probe_only': args.probe_only,
                'new_optimizer_updates': 0, 'finished_utc': datetime.now(timezone.utc).isoformat()})
            for s in handlers:
                signal.signal(s, signal.SIG_IGN)
            try:
                publish()
            finally:
                for s, handler in handlers.items():
                    signal.signal(s, handler)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--data-root', type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--lock-only', action='store_true')
    group.add_argument('--probe-only', action='store_true')
    parser.add_argument('--publish-remote')
    sys.exit(launch(parser.parse_args()))


if __name__ == '__main__':
    main()
