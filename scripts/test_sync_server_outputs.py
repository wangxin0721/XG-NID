"""Snapshot integrity and local bare-remote end-to-end tests."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import sync_server_outputs as sync


class Snapshots(unittest.TestCase):
    def test_inventory_and_verification_detect_missing_changed_and_extra_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);out=root/'outputs';out.mkdir();(out/'report.md').write_bytes(b'report')
            files=sync.inventory(out)
            (root/'outputs-manifest.json').write_text(json.dumps({'files':files,'file_count':1,'total_bytes':6}))
            sync.verify(root)
            for change in ('changed','missing','extra'):
                (out/'report.md').write_bytes(b'report')
                if change=='changed':(out/'report.md').write_bytes(b'changed')
                elif change=='missing':(out/'report.md').unlink()
                else:(out/'extra').write_bytes(b'extra')
                with self.assertRaises(RuntimeError):sync.verify(root)

    def test_large_jsonl_and_binary_are_lfs_but_small_reports_are_readable(self):
        attrs=sync.attributes({'small.json':{'bytes':12},'a.pt':{'bytes':1},'a [b].jsonl':{'bytes':11*2**20}})
        self.assertNotIn('small.json',attrs);self.assertIn('a.pt',attrs);self.assertIn('jsonl',attrs)

    def test_publish_twice_then_download_without_changing_source_repository(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);remote=root/'remote.git';repo=root/'source';repo.mkdir()
            sync.command(['git','init','--bare',str(remote)],root)
            sync.command(['git','init'],repo)
            sync.command(['git','config','user.name','Snapshot test'],repo)
            sync.command(['git','config','user.email','test@example.invalid'],repo)
            (repo/'code.txt').write_bytes(b'unchanged code');sync.command(['git','add','code.txt'],repo)
            sync.command(['git','commit','-m','source'],repo)
            head=sync.command(['git','rev-parse','HEAD'],repo,capture=True)
            index=sync.command(['git','ls-files','--stage'],repo,capture=True)
            outputs=repo/'outputs';outputs.mkdir();(outputs/'report.json').write_bytes(b'{"ok":true}')
            (outputs/'model [v1].pt').write_bytes(b'fake model bytes\x00\xff')
            args=argparse.Namespace(repo=repo,remote=str(remote),destination=root/'download')
            # Local bare remotes support LFS. No network or external repository.
            env=os.environ.copy();env['GIT_LFS_SKIP_SMUDGE']='1'
            with patch.object(sync,'lfs_environment',return_value=env):
                sync.publish(args)
                (outputs/'report.json').write_bytes(b'{"ok":false}')
                (outputs/'new.txt').write_bytes(b'new result')
                sync.publish(args)
            sync.download(args)
            tracked=sync.command(['git','lfs','ls-files','--name-only'],args.destination,capture=True)
            self.assertIn('outputs/model [v1].pt',tracked)
            self.assertEqual((args.destination/'outputs/model [v1].pt').read_bytes(),(outputs/'model [v1].pt').read_bytes())
            self.assertEqual((args.destination/'outputs/report.json').read_bytes(),b'{"ok":false}')
            self.assertEqual(sync.command(['git','rev-parse','HEAD'],repo,capture=True),head)
            self.assertEqual(sync.command(['git','ls-files','--stage'],repo,capture=True),index)
            with self.assertRaises(RuntimeError):sync.download(args)


if __name__=='__main__':unittest.main()
