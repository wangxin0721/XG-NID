"""Regress missing system LFS, repository-local PATH and archive-read ordering."""
from contextlib import redirect_stdout
import hashlib
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import fixed_transition_archive as archive
import launch_fixed_transition_stability as launcher
import sync_server_outputs as sync


class LfsBootstrapChecks(unittest.TestCase):
    def test_missing_system_lfs_reuses_verified_repository_local_tool_without_global_path_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo=Path(tmp);folder=repo/'.output-sync-tools/git-lfs-3.8.1-amd64';folder.mkdir(parents=True)
            (folder/'git-lfs').write_bytes(b'cached tool fixture')
            original=os.environ.get('PATH','');seen=[]
            def git(command,**kwargs):
                seen.append(kwargs.get('env'))
                local=kwargs.get('env',{}).get('PATH','').startswith(str(folder)+os.pathsep)
                return SimpleNamespace(returncode=0 if local else 1,stdout=None)
            with patch.object(sync.platform,'system',return_value='Linux'), \
                 patch.object(sync.platform,'machine',return_value='x86_64'), \
                 patch.object(sync.subprocess,'run',side_effect=git), \
                 patch.object(sync.urllib.request,'urlopen') as download:
                env=sync.lfs_environment(repo)
            self.assertTrue(env['PATH'].startswith(str(folder)+os.pathsep))
            self.assertEqual(os.environ.get('PATH',''),original)
            self.assertEqual(len(seen),2);self.assertIsNone(seen[0]);self.assertEqual(seen[1],env)
            download.assert_not_called()

    def test_archive_bootstraps_before_lfs_env_and_forwards_private_path_to_fetch(self):
        for existing in (True,False):
            with self.subTest(existing=existing),tempfile.TemporaryDirectory() as tmp:
                repo=Path(tmp);media=repo/'media';payload=b'locked archive payload'
                digest=hashlib.sha256(payload).hexdigest();target=media/digest[:2]/digest[2:4]/digest
                target.parent.mkdir(parents=True)
                if existing:target.write_bytes(payload)
                env={'PATH':str(repo/'private-lfs')+os.pathsep+os.environ.get('PATH','')}
                spec={'source_result_commit':'fixed-ref','source_directory':'archive',
                      'inputs':{'weights.pt':{'bytes':len(payload),'sha256':digest}}}
                events=[]
                def bootstrap(where):self.assertEqual(where,repo);events.append('bootstrap');return env
                def git(command,**kwargs):
                    self.assertEqual(kwargs.get('env'),env)
                    events.append('fetch' if command[:3]==['git','lfs','fetch'] else 'cat-file')
                    if command[:3]==['git','lfs','fetch']:target.write_bytes(payload)
                    return SimpleNamespace(returncode=0)
                def capture(command,**kwargs):
                    self.assertEqual(kwargs.get('env'),env)
                    if command[:2]==['git','show']:
                        events.append('show')
                        return f'version https://git-lfs.github.com/spec/v1\noid sha256:{digest}\nsize {len(payload)}\n'.encode()
                    self.assertEqual(command,['git','lfs','env']);events.append('lfs-env')
                    return 'LocalMediaDir='+str(media)+'\n'
                with patch.object(archive,'lfs_environment',side_effect=bootstrap), \
                     patch.object(archive.subprocess,'run',side_effect=git), \
                     patch.object(archive.subprocess,'check_output',side_effect=capture):
                    restored=archive.open_archive(repo,spec,download=True)
                self.assertEqual(events[0],'bootstrap');self.assertIn('lfs-env',events)
                self.assertEqual('fetch' in events,not existing)
                self.assertEqual(restored.data('weights.pt'),payload)

    def test_launcher_bootstrap_precedes_child_and_upload_and_bootstrap_failure_still_publishes(self):
        for succeeds in (True,False):
            with self.subTest(succeeds=succeeds),tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
                repo=Path(tmp);args=SimpleNamespace(repo=repo,data_root=None,import_only=True,probe_only=False,publish_remote=None)
                env={'PATH':str(repo/'local-lfs')+os.pathsep+os.environ.get('PATH','')};events=[]
                def bootstrap(where):
                    events.append('bootstrap')
                    if not succeeds:raise RuntimeError('standalone LFS unavailable')
                    return env
                def child(command,**kwargs):
                    events.append('child');self.assertEqual(kwargs['env']['PATH'],env['PATH'])
                    self.assertIn('--phase',command)
                    return SimpleNamespace(wait=lambda:0)
                def publish(code,where,output,status,child_env,**kwargs):
                    events.append('publish');self.assertEqual(status,0 if succeeds else 1)
                    if succeeds:
                        self.assertEqual(child_env['PATH'],env['PATH'])
                        self.assertTrue((output/'lfs_preflight.json').is_file())
                    else:self.assertTrue((output/'launcher_failure.json').is_file())
                    return [0,0]
                with patch.object(launcher,'lfs_environment',side_effect=bootstrap), \
                     patch.object(launcher.subprocess,'Popen',side_effect=child), \
                     patch.object(launcher.publisher,'publish_results',side_effect=publish):
                    self.assertEqual(launcher.launch(args),0 if succeeds else 1)
                self.assertEqual(events,['bootstrap','child','publish'] if succeeds else ['bootstrap','publish'])


if __name__=='__main__':unittest.main()
