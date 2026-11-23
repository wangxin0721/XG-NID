"""Execute zero-training cross replay and automatically publish all results on exit."""
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

OUTPUT_NAME = 'transition_pair_cross_replay_s42_v1'


def launch(args):
    code, repo = Path(__file__).resolve().parents[1], args.repo.resolve()
    output = repo / 'outputs' / OUTPUT_NAME
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'; env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PYTHONPATH'] = os.pathsep.join([str(code / 'src'), str(code / 'scripts'), env.get('PYTHONPATH', '')])
    child, caught = None, None
    def stop(signum, frame):
        nonlocal caught
        caught = signum
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signum) if os.name == 'posix' else child.terminate()
    # Windows locks deny reads of the locked byte. Keep this live coordination
    # file outside the directory that the publisher must inventory and hash.
    lock_root = repo / 'tmp'
    lock_root.mkdir(exist_ok=True)
    with FileLock(str(lock_root / (OUTPUT_NAME + '.launcher.lock')), timeout=0):
        handlers = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
        status = 1
        try:
            command = [sys.executable, '-u', '-B', str(code / 'scripts/run_transition_pair_cross_replay.py'),
                       '--repo', str(repo), '--device', args.device, '--resume']
            if args.snapshot: command.append('--snapshot')
            if args.archive_root is not None: command += ['--archive-root', str(args.archive_root.resolve())]
            with (output / 'replay.log').open('ab', buffering=0) as stream:
                stream.write(('\n' + datetime.now(timezone.utc).isoformat() + ' ' + shlex.join(command) + '\n').encode())
                child = subprocess.Popen(command, cwd=repo, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                         start_new_session=os.name == 'posix')
                status = child.wait()
            status = 128 + (caught or -status) if status < 0 or caught else status
        except BaseException as exc:
            with (output / 'replay.log').open('a', encoding='utf-8') as stream: stream.write(repr(exc) + '\n')
            status = 128 + caught if caught else 1
        finally:
            publisher.write(output / 'replay_exit.json', {'exit_code': status, 'signal': caught,
                            'training_epochs': 0, 'optimizer_updates': 0})
            for s in handlers: signal.signal(s, signal.SIG_IGN)
            try:
                publisher.publish_results(code, repo, output, status, env, output_name=OUTPUT_NAME,
                                          remote=args.publish_remote, exit_key='replay_exit_code')
            finally:
                for s, handler in handlers.items(): signal.signal(s, handler)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--archive-root', type=Path)
    parser.add_argument('--snapshot', action='store_true')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--publish-remote')
    sys.exit(launch(parser.parse_args()))


if __name__ == '__main__':
    main()
