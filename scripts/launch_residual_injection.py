"""Run/resume all three injection arms; publish all artifacts on success or caught failure."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys

from filelock import FileLock
import launch_joint_transition_fusion as publisher

OUTPUT_NAME = 'residual_injection_s42_v1'


def launch(args):
    code, repo = Path(__file__).resolve().parents[1], args.repo.resolve()
    output = repo / 'outputs' / OUTPUT_NAME
    output.mkdir(parents=True, exist_ok=True)
    (repo / 'tmp').mkdir(exist_ok=True)
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'; env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PYTHONPATH'] = os.pathsep.join([str(code / 'src'), str(code / 'scripts'), env.get('PYTHONPATH', '')])
    child, caught = None, None
    def stop(signum, frame):
        nonlocal caught
        caught = signum
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signum) if os.name == 'posix' else child.terminate()
    with FileLock(str(repo / 'tmp' / (OUTPUT_NAME + '.launcher.lock')), timeout=0):
        handlers = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
        status = 1
        try:
            command = [sys.executable, '-u', '-B', str(code / 'scripts/run_residual_injection.py'),
                       '--repo', str(repo), '--resume']
            if args.data_root is not None:
                command += ['--data-root', str(args.data_root.resolve())]
            if args.probe_only:
                command.append('--probe-only')
            with (output / 'training.log').open('ab', buffering=0) as log:
                log.write(('\n' + datetime.now(timezone.utc).isoformat() + ' ' + shlex.join(command) + '\n').encode())
                child = subprocess.Popen(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=os.name == 'posix')
                status = child.wait()
            status = 128 + (caught or -status) if status < 0 or caught else status
        except BaseException as exc:
            with (output / 'training.log').open('a', encoding='utf-8') as log:
                log.write(repr(exc) + '\n')
            status = 128 + caught if caught else 1
        finally:
            publisher.write(output / 'training_exit.json', {'exit_code': status, 'signal': caught,
                'probe_only': args.probe_only, 'finished_utc': datetime.now(timezone.utc).isoformat()})
            for s in handlers:
                signal.signal(s, signal.SIG_IGN)
            try:
                publisher.publish_results(code, repo, output, status, env, output_name=OUTPUT_NAME,
                                          remote=args.publish_remote)
            except Exception as exc:
                retry = [sys.executable, '-B', str(code / 'scripts/sync_server_outputs.py'),
                         'publish', '--repo', str(repo), '--only', OUTPUT_NAME]
                if args.publish_remote is not None:
                    retry += ['--remote', args.publish_remote]
                publisher.write(output / 'upload_failure.json', {'training_exit_code': status,
                    'exception': repr(exc), 'retry_command': shlex.join(retry)})
                print('UPLOAD FAILED: ' + repr(exc) + '\nRetry: ' + shlex.join(retry), file=sys.stderr, flush=True)
            finally:
                for s, handler in handlers.items():
                    signal.signal(s, handler)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--probe-only', action='store_true')
    parser.add_argument('--publish-remote')
    sys.exit(launch(parser.parse_args()))


if __name__ == '__main__':
    main()
