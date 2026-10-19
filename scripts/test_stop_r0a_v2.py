"""Pure identity/path checks; never signal a process or delete real run files."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import stop_r0a_v2_server as cleanup


class CleanupChecks(unittest.TestCase):
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
