"""Preserve the known 29cc938 pre-training CUDA failure before a corrected run."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from filelock import FileLock

OUTPUT = 'flow_packet_node_modulation_v1'
FAILED_COMMIT = '29cc938886727d64eb0578ed8902b89a8140dd46'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def archive_failed_preflight(repo, revision):
    repo = repo.resolve()
    output = repo / 'outputs' / OUTPUT
    receipt = output / 'preflight_restart.json'
    # The same fixed command remains a real resume after the one-time archive.
    if receipt.exists() and read(receipt)['new_code_commit'] == revision:
        return read(receipt)
    if output.is_symlink() or output.resolve().parent != (repo / 'outputs').resolve():
        raise ValueError('Unexpected output directory; preserve it for inspection')
    protocol = read(output / 'protocol_lock.json')
    if protocol['code_commit'] != FAILED_COMMIT or revision == FAILED_COMMIT:
        raise ValueError('Only the known 29cc938 preflight failure can be archived')
    failure = read(output / 'failure.json')['exception']
    if "can't convert cuda:" not in failure or 'numpy' not in failure:
        raise ValueError('Failure is not the known CUDA-to-NumPy preflight error')
    exit_code = read(output / 'publication' / 'training_exit.json')['exit_code']
    if exit_code == 0:
        raise ValueError('Previous launcher did not record failed training')
    if (list(output.rglob('recovery.pt')) or any((output / n).exists() for n in
            ('S1', 'S2', 'S3', 'history.json', 'training_progress.json',
             'summary.json', 'artifact_manifest.json', 'server_probe.json'))):
        raise ValueError('Training/probe artifacts exist; refusing to discard or reset progress')
    pid_file = output / 'launcher.pid'
    if pid_file.exists():
        pid = int(pid_file.read_text().strip())
        if pid <= 0:
            raise ValueError('Invalid recorded launcher PID')
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise ValueError('Previous launcher is still active; wait for its upload to finish')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    saved_name = OUTPUT + '_failed_preflight_' + stamp
    staging = output.parent / saved_name
    # Verify both resolved absolute move targets before moving a directory tree.
    if staging.exists() or staging.resolve().parent != output.parent.resolve():
        raise ValueError('Unexpected archive destination')
    output.rename(staging)
    output.mkdir()
    previous = output / 'previous_attempts'
    previous.mkdir()
    destination = previous / saved_name
    if destination.resolve().parent != previous.resolve():
        raise ValueError('Unexpected preserved-attempt destination: ' + str(staging))
    staging.rename(destination)
    result = {'previous_code_commit': FAILED_COMMIT, 'new_code_commit': revision,
              'reason': 'CPU chronology must precede the in-place CUDA transfer',
              'previous_training_updates': 0,
              'preserved_attempt': destination.relative_to(output).as_posix()}
    receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--code-commit', required=True)
    a = p.parse_args()
    if len(a.code_commit) != 40 or any(c not in '0123456789abcdef' for c in a.code_commit):
        p.error('--code-commit must be a full SHA')
    temporary = a.repo.resolve() / 'tmp'
    temporary.mkdir(exist_ok=True)
    with FileLock(str(temporary / (OUTPUT + '.launcher.lock')), timeout=0), \
         FileLock(str(temporary / (OUTPUT + '.runner.lock')), timeout=0):
        print(json.dumps(archive_failed_preflight(a.repo, a.code_commit), ensure_ascii=False))


if __name__ == '__main__':
    main()
