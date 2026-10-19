"""Pure identity/path checks; never signal a process or delete real run files."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import stop_r0a_v2_server as cleanup


class CleanupChecks(unittest.TestCase):
    def test_missing_pidfd_uses_checked_term_and_never_signals_reused_pid(self):
        info = {'pid': 16724, 'start': '80756', 'args': ['python', 'trainer.py'], 'role': 'trainer'}
        with patch.object(cleanup.os, 'pidfd_open', None, create=True), \
                patch.object(cleanup.os, 'kill') as kill, \
                patch.object(cleanup, 'role_of', return_value='trainer'), \
                patch.object(cleanup, 'process', side_effect=[info, info, None]):
            cleanup.terminate(info, Path.cwd())
            kill.assert_called_once_with(16724, cleanup.signal.SIGTERM)
        with patch.object(cleanup.os, 'pidfd_open', None, create=True), \
                patch.object(cleanup.os, 'kill') as kill, \
                patch.object(cleanup, 'role_of', return_value='trainer'), \
                patch.object(cleanup, 'process', side_effect=[info, dict(info, start='changed')]):
            with self.assertRaisesRegex(RuntimeError, 'PID identity changed'):
                cleanup.terminate(info, Path.cwd())
            kill.assert_not_called()

    def test_scan_does_not_read_unrelated_process_cwd(self):
        with patch.object(cleanup.os, 'getuid', return_value=1000, create=True), \
                patch.object(cleanup.Path, 'stat', return_value=SimpleNamespace(st_uid=1000)), \
                patch.object(cleanup.Path, 'read_bytes', return_value=b'python\0other_service.py\0'), \
                patch.object(cleanup.Path, 'resolve', side_effect=PermissionError('cwd denied')) as resolve:
            self.assertIsNone(cleanup.process(2537, scanning=True))
            resolve.assert_not_called()

    def test_unreadable_candidate_or_recorded_pid_blocks_deletion(self):
        stat = '123 (python) ' + ' '.join(['S'] + ['0'] * 18 + ['12345'])
        with patch.object(cleanup.os, 'getuid', return_value=1000, create=True), \
                patch.object(cleanup.Path, 'stat', return_value=SimpleNamespace(st_uid=1000)), \
                patch.object(cleanup.Path, 'read_bytes', return_value=b'python\0supervisor.py\0'), \
                patch.object(cleanup.Path, 'read_text', return_value=stat), \
                patch.object(cleanup.Path, 'resolve', side_effect=PermissionError('cwd denied')):
            for scanning in (True, False):
                with self.assertRaisesRegex(RuntimeError, 'Cannot verify target/candidate PID'):
                    cleanup.process(2537, scanning=scanning)

    def test_unreadable_cmdline_is_skipped_only_during_scan(self):
        with patch.object(cleanup.os, 'getuid', return_value=1000, create=True), \
                patch.object(cleanup.Path, 'stat', return_value=SimpleNamespace(st_uid=1000)), \
                patch.object(cleanup.Path, 'read_bytes', side_effect=PermissionError('cmdline denied')):
            self.assertIsNone(cleanup.process(2537, scanning=True))
            with self.assertRaisesRegex(RuntimeError, 'Cannot verify target/candidate PID'):
                cleanup.process(2537)

    def test_only_named_v2_paths_are_targets(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'tmp') as folder:
            repo = Path(folder).resolve()
            (repo / 'outputs').mkdir()
            targets = cleanup.deletion_targets(repo)
            self.assertEqual([p.name for p in targets], list(cleanup.TARGET_NAMES))
            self.assertTrue(all(p.parent == repo / 'outputs' for p in targets))
            targets[0].write_text('unexpected file', encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, 'Expected directory'):
                cleanup.deletion_targets(repo)

    def test_process_match_requires_exact_run_repo_mode_and_output(self):
        repo = Path('D:/XG-NID').resolve()
        control = repo / 'outputs/r0_aligned_restart_control_v2'
        parts = ['python', '-B', '-u', str(control / 'run_r0_aligned_server.py'),
                 '--repo', str(repo), '--mode', 'full', '--output', str(repo / 'outputs/r0_aligned_full_s42_v2')]
        info = {'uid': 1000, 'args': parts, 'cwd': str(repo)}
        with patch.object(cleanup.os, 'getuid', return_value=1000, create=True):
            self.assertEqual(cleanup.role_of(info, repo), 'trainer')
            changed = list(parts)
            changed[-1] = str(repo / 'outputs/r0_validation_full_s42_v1')
            self.assertIsNone(cleanup.role_of(dict(info, args=changed), repo))
            changed = list(parts)
            changed[3] = '/tmp/some_other_training.py'
            self.assertIsNone(cleanup.role_of(dict(info, args=changed), repo))
            self.assertIsNone(cleanup.role_of(dict(info, uid=1001), repo))
            supervisor = ['python', str(control / 'supervisor.py'), '--repo', str(repo), '--supervise']
            self.assertEqual(cleanup.role_of(dict(info, args=supervisor), repo), 'supervisor')
            observer = ['python', '/tmp/xgnid_watch_r0a_f1.py', '--repo', str(repo)]
            self.assertEqual(cleanup.role_of(dict(info, args=observer), repo), 'observer')


if __name__ == '__main__':
    unittest.main()
