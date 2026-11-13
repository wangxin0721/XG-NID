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
    @contextlib.contextmanager
    def source_links(self,outputs):
        # Emulate OS link metadata for portable tests without Windows symlink
        # privileges. Directory children must never enter the inventory.
        targets={'_organized/alias':'../report.json','broken.pt':'missing-target',
                 'cycle':'.','external':'/outside/server-only'}
        mapped={outputs/name:target for name,target in targets.items()}
        for path in mapped:
            path.parent.mkdir(parents=True,exist_ok=True)
            if path.name=='broken.pt':path.write_bytes(b'missing-target')
            else:
                path.mkdir();(path/'must-not-copy.txt').write_bytes(b'not snapshot content')
        original=Path.is_symlink
        def is_link(path):return path in mapped or original(path)
        def metadata(path):
            target=mapped[path];raw=target.encode('utf-8')
            return {'kind':'symlink','target':target,'bytes':len(raw),'sha256':sync.hashlib.sha256(raw).hexdigest()}
        with patch.object(Path,'is_symlink',is_link),patch.object(sync,'link_metadata',side_effect=metadata):yield targets

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
        self.assertNotIn('broken.pt',sync.attributes({'broken.pt':{'kind':'symlink','bytes':42}}))

    def test_inventory_preserves_directory_dangling_cycle_and_external_links_without_following(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'outputs';out.mkdir();(out/'report.json').write_bytes(b'{}')
            with self.source_links(out) as targets:
                files=sync.inventory(out)
                self.assertEqual(set(files),{'report.json',*targets})
                for name,target in targets.items():
                    self.assertEqual(files[name]['kind'],'symlink');self.assertEqual(files[name]['target'],target)

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
            selected=outputs/'selected';selected.mkdir()
            (selected/'obsolete.json').write_bytes(b'{}')
            args=argparse.Namespace(repo=repo,remote=str(remote),destination=root/'download')
            # Local bare remotes support LFS. No network or external repository.
            env=os.environ.copy();env['GIT_LFS_SKIP_SMUDGE']='1'
            with self.source_links(outputs) as targets,patch.object(sync,'lfs_environment',return_value=env):
                sync.publish(args)
                (outputs/'report.json').write_bytes(b'{"ok":false}')
                (outputs/'new.txt').write_bytes(b'new result')
                sync.publish(args)
                (selected/'obsolete.json').unlink()
                (selected/'best.pt').write_bytes(b'new selected model\x00\xff')
                (outputs/'report.json').write_bytes(b'changed outside selected directory')
                args.only=['selected']
                sync.publish(args)
            sync.download(args)
            tree=sync.command(['git','ls-tree','-r','HEAD'],args.destination,capture=True)
            for name,target in targets.items():
                self.assertTrue(any(line.startswith('120000 ') and line.endswith('\toutputs/'+name) for line in tree.splitlines()))
                self.assertEqual((args.destination/'outputs'/name).read_bytes(),target.encode('utf-8'))
            manifest=json.loads((args.destination/'outputs-manifest.json').read_text())
            self.assertEqual(manifest['symlink_count'],4)
            link_file=args.destination/'outputs/broken.pt'
            link_file.write_bytes(b'tampered-target')
            with self.assertRaises(RuntimeError):sync.verify(args.destination)
            link_file.write_bytes(targets['broken.pt'].encode('utf-8'))
            sync.verify(args.destination)
            tracked=sync.command(['git','lfs','ls-files','--name-only'],args.destination,capture=True)
            self.assertIn('outputs/model [v1].pt',tracked)
            self.assertEqual((args.destination/'outputs/model [v1].pt').read_bytes(),(outputs/'model [v1].pt').read_bytes())
            self.assertEqual((args.destination/'outputs/report.json').read_bytes(),b'{"ok":false}')
            self.assertFalse((args.destination/'outputs/selected/obsolete.json').exists())
            self.assertEqual((args.destination/'outputs/selected/best.pt').read_bytes(),(selected/'best.pt').read_bytes())
            self.assertEqual(manifest['updated_directories'],['selected'])
            for stage in repo.glob('.outputs-publish-*'):
                storage=sync.command(['git','config','--local','--get','lfs.storage'],stage,capture=True).strip()
                self.assertEqual(Path(storage),stage/'.git'/'lfs')
            self.assertEqual(sync.command(['git','rev-parse','HEAD'],repo,capture=True),head)
            self.assertEqual(sync.command(['git','ls-files','--stage'],repo,capture=True),index)
            with self.assertRaises(RuntimeError):sync.download(args)


if __name__=='__main__':unittest.main()
