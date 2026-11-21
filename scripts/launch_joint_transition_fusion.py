"""Run/resume the registered experiment and publish on success, failure, or SIGTERM."""
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
import tempfile

from filelock import FileLock

OUTPUT = 'joint_transition_fusion_s42_v1'


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def publish_results(code, repo, output, training_exit, env, output_name=OUTPUT):
    command = [sys.executable, '-B', str(code / 'scripts/sync_server_outputs.py'),
               'publish', '--repo', str(repo), '--only', output_name]
    # Publish cannot inventory a log that its own subprocess is actively appending to.
    # Capture externally, then publish the closed primary log and exit-code receipt.
    temporary_root = repo / 'tmp'
    temporary_root.mkdir(exist_ok=True)
    attempts = []
    with tempfile.TemporaryDirectory(prefix='joint-fusion-upload-', dir=temporary_root) as temp:
        for attempt in (1, 2):
            log = Path(temp) / 'upload.log'
            try:
                with log.open('wb') as stream:
                    result = subprocess.run(command, cwd=repo, env=env, stdout=stream, stderr=subprocess.STDOUT)
                status = result.returncode
            except Exception as exc:
                log.write_text(repr(exc) + '\n', encoding='utf-8')
                status = 1
            attempts.append(status)
            if attempt == 1:
                shutil.copy2(log, output / 'upload.log')
                write(output / 'exit_codes.json', {'training_exit_code': training_exit,
                      'primary_upload_exit_code': status, 'receipt_publication_follows': True,
                      'retry_command': shlex.join(command)})
            else:
                # Do not mutate the just-published snapshot. Report this receipt upload externally.
                print(log.read_text(encoding='utf-8', errors='replace')[-6000:], flush=True)
    if attempts[-1] != 0:
        print('UPLOAD FAILED: results remain local; retry after training has stopped:\n' + shlex.join(command),
              file=sys.stderr, flush=True)
    print(json.dumps({'training_exit_code': training_exit, 'upload_attempt_exit_codes': attempts,
                      'output': str(output)}), flush=True)
    return attempts


def launch(args, output_name=OUTPUT, runner='run_joint_transition_fusion.py'):
    code = Path(__file__).resolve().parents[1]
    repo = args.repo.resolve()
    output = repo / 'outputs' / output_name
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PYTHONPATH'] = os.pathsep.join([str(code / 'src'), str(code / 'scripts'), env.get('PYTHONPATH', '')])
    child, received_signal = None, None

    def stop(signum, frame):
        nonlocal received_signal
        received_signal = signum
        if child is not None and child.poll() is None:
            if os.name == 'posix':
                os.killpg(child.pid, signum)
            else:
                child.terminate()

    with FileLock(str(output / '.launcher.lock'), timeout=0):
        old_handlers = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
        status = 1
        try:
            command = [sys.executable, '-u', '-B', str(code / 'scripts' / runner),
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
            status = 128 + (received_signal or -status) if status < 0 or received_signal else status
        except BaseException as exc:
            with (output / 'training.log').open('a', encoding='utf-8') as stream:
                stream.write('\nLauncher error: ' + repr(exc) + '\n')
            status = 128 + received_signal if received_signal else 1
        finally:
            write(output / 'training_exit.json', {'exit_code': status, 'signal': received_signal,
                  'finished_utc': datetime.now(timezone.utc).isoformat(), 'probe_only': args.probe_only})
            # A second TERM while uploading must not discard the recoverable evidence.
            for sig in old_handlers:
                signal.signal(sig, signal.SIG_IGN)
            try:
                publish_results(code, repo, output, status, env, output_name=output_name)
            finally:
                for sig, handler in old_handlers.items():
                    signal.signal(sig, handler)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--probe-only', action='store_true')
    sys.exit(launch(parser.parse_args()))


if __name__ == '__main__':
    main()
