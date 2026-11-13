"""Publish all or selected outputs, or download and verify the branch snapshot."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

BRANCH='codex/server-outputs'
REMOTE='git@github.com:wangxin0721/XG-NID.git'


def require(ok,message):
    if not ok:raise RuntimeError(message)


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def command(args,cwd,env=None,capture=False):
    return subprocess.run(args,cwd=cwd,env=env,check=True,text=True,
        stdout=subprocess.PIPE if capture else None).stdout


def link_metadata(path):
    target=os.readlink(path)
    data=target.encode('utf-8')
    return {'kind':'symlink','target':target,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}


def inventory(root):
    require(root.is_dir(),'Missing outputs directory: '+str(root))
    files={}
    def failed(error):raise error
    for directory,dirs,names in os.walk(root,followlinks=False,onerror=failed):
        for name in sorted(dirs+names):
            p=Path(directory)/name
            if p.is_symlink():
                before=p.lstat();meta=link_metadata(p);after=p.lstat()
                require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
                        and meta==link_metadata(p),'Symlink changed during inventory: '+str(p))
                files[p.relative_to(root).as_posix()]=meta
                if name in dirs:dirs.remove(name)
            elif name in names:
                require(p.is_file(),'Unsupported special file: '+str(p))
                before=p.stat();digest=sha(p);after=p.stat()
                require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),
                    'Source changed during hashing; retry after training completes: '+str(p))
                files[p.relative_to(root).as_posix()]={'bytes':after.st_size,'sha256':digest}
    return dict(sorted(files.items()))


def uses_lfs(name, meta):
    return meta.get('kind') != 'symlink' and (
        meta['bytes'] >= 10*2**20 or Path(name).suffix.lower() in ('.pt','.pth','.ckpt','.sqlite','.db'))


def attributes(files):
    lines=['# Reports remain ordinary Git text; binary/large artifacts use LFS.']
    for name,meta in files.items():
        if uses_lfs(name, meta):
            # .gitattributes uses C-quoted strings followed by wildmatch parsing.
            pattern='outputs/'+''.join('\\'+c if c in '\\*?[' else c for c in name)
            lines.append(json.dumps(pattern,ensure_ascii=False)+' filter=lfs diff=lfs merge=lfs -text')
    return '\n'.join(lines)+'\n'


def lfs_environment(repo):
    env=os.environ.copy()
    if subprocess.run(['git','lfs','version'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:return env
    require(platform.system()=='Linux','Install Git LFS before local download')
    arch={'x86_64':'amd64','aarch64':'arm64'}.get(platform.machine())
    require(arch is not None,'Unsupported architecture; install Git LFS manually')
    version='3.8.1';folder=repo/'.output-sync-tools'/('git-lfs-'+version+'-'+arch)
    binary=folder/'git-lfs'
    if not binary.exists():
        folder.mkdir(parents=True,exist_ok=True)
        name='git-lfs-linux-'+arch+'-v'+version+'.tar.gz'
        # Pinned from the official release sha256sums.asc, avoiding API limits.
        digest={'amd64':'e72cf723a4a660438862282e8aecf3885fc8123e88bfcad534cf6b941a354b75',
                'arm64':'67895dd77b63329796bc5efdd47b57f218e1f7b98717ebaf936a8238dbcaec0f'}[arch]
        url='https://github.com/git-lfs/git-lfs/releases/download/v'+version+'/'+name
        archive=folder/'download.tar.gz'
        print('Downloading standalone Git LFS from official release',flush=True)
        with urllib.request.urlopen(url,timeout=60) as response,archive.open('wb') as out:
            shutil.copyfileobj(response,out)
        require(sha(archive)==digest,'Git LFS download checksum mismatch')
        temporary=folder/'git-lfs.tmp'
        with tarfile.open(archive) as tar:
            members=[m for m in tar.getmembers() if m.isfile() and Path(m.name).name=='git-lfs']
            require(len(members)==1,'Unexpected Git LFS archive layout')
            with tar.extractfile(members[0]) as source,temporary.open('wb') as out:shutil.copyfileobj(source,out)
        temporary.chmod(0o755)
        temporary.replace(binary)
    env['PATH']=str(folder)+os.pathsep+env.get('PATH','')
    command(['git','lfs','version'],repo,env)
    return env


def publish(args):
    repo=args.repo.resolve();source=repo/'outputs'
    selected = sorted(set(getattr(args, 'only', None) or []))
    def selected_inventory():
        if not selected:
            return inventory(source)
        files = {}
        for name in selected:
            require(name not in ('', '.', '..') and '/' not in name and '\\' not in name,
                    '--only takes a directory name directly under outputs')
            folder = source/name
            require(not folder.is_symlink(), 'Selected directory is a symlink: '+name)
            entries = inventory(folder)
            require(bool(entries), 'Selected directory is empty: '+name)
            files.update({name+'/'+path: meta for path, meta in entries.items()})
        return dict(sorted(files.items()))

    files = selected_inventory()
    require(bool(files),'Refuse empty snapshot')
    env=lfs_environment(repo)
    total=sum(x['bytes'] for x in files.values())
    require(all(x['bytes']<2*2**30 for x in files.values()),'A file exceeds the conservative2GiB LFS limit')
    require(shutil.disk_usage(repo).free>int(total*2.2)+256*2**20,'Need disk space for snapshot and local LFS objects')
    stage=Path(tempfile.mkdtemp(prefix='.outputs-publish-',dir=repo))
    print(json.dumps({'files':len(files),'GiB':total/2**30,'staging_directory':str(stage)},ensure_ascii=False),flush=True)
    # This is an independent Git repository. The user's checkout/index never changes.
    command(['git','init'],stage,env)
    command(['git','symbolic-ref','HEAD','refs/heads/'+BRANCH],stage,env)
    command(['git','remote','add','origin',args.remote],stage,env)
    exists=subprocess.run(['git','ls-remote','--exit-code','--heads',args.remote,'refs/heads/'+BRANCH],cwd=stage,env=env,stdout=subprocess.PIPE,text=True)
    require(exists.returncode in (0,2),'Cannot inspect remote output branch')
    old = {}
    if exists.returncode==0:
        command(['git','fetch','--depth=1','origin',BRANCH],stage,env)
        old=json.loads(command(['git','show','FETCH_HEAD:outputs-manifest.json'],stage,env,True))
        require(old.get('schema') in ('xgnid.outputs-snapshot.v1','xgnid.outputs-snapshot.v2'),'Remote branch is not an outputs snapshot')
        command(['git','read-tree','FETCH_HEAD'],stage,env)
        command(['git','update-ref','HEAD','FETCH_HEAD'],stage,env)
    for key,default in (('user.name','XG-NID artifacts'),('user.email','artifacts@users.noreply.github.com')):
        value=subprocess.run(['git','config',key],cwd=repo,stdout=subprocess.PIPE,text=True).stdout.strip() or default
        command(['git','config',key,value],stage,env)
    command(['git','config','core.autocrlf','false'],stage,env)
    command(['git','config','core.symlinks','false'],stage,env)
    # Keep staging, verification and the pre-push hook on the same absolute
    # object store, even when the user has configured shared LFS storage.
    command(['git','config','--local','lfs.storage',str(stage/'.git'/'lfs')],stage,env)
    command(['git','lfs','install','--local'],stage,env)
    for name,meta in files.items():
        dst=stage/'outputs'/name;dst.parent.mkdir(parents=True,exist_ok=True)
        if meta.get('kind')=='symlink':dst.write_bytes(meta['target'].encode('utf-8'))
        else:shutil.copy2(source/name,dst)
        require(sha(dst)==meta['sha256'],'Source changed during copy: '+name)
    require(selected_inventory()==files,'Source inventory changed; finish training before retrying')
    merged = dict(files)
    if selected:
        merged.update({name: meta for name, meta in old.get('files', {}).items()
                       if name.split('/', 1)[0] not in selected})
    merged = dict(sorted(merged.items()))
    with (stage/'.gitattributes').open('w',encoding='utf-8',newline='\n') as out:out.write(attributes(merged))
    links=sum(x.get('kind')=='symlink' for x in merged.values())
    manifest={'schema':'xgnid.outputs-snapshot.v2','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_directory':str(source),'files':merged,'file_count':len(merged),'symlink_count':links,
        'regular_file_count':len(merged)-links,'total_bytes':sum(meta['bytes'] for meta in merged.values())}
    if selected:
        manifest['updated_directories'] = selected
    with (stage/'outputs-manifest.json').open('w',encoding='utf-8',newline='\n') as out:out.write(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    paths = ['outputs/'+name for name in selected] if selected else ['outputs']
    command(['git','--literal-pathspecs','add','-A','--','.gitattributes','outputs-manifest.json',*paths],stage,env)
    # Store links as Git mode120000 even where the OS cannot create symlinks.
    for name,meta in files.items():
        if meta.get('kind')=='symlink':
            path='outputs/'+name
            blob=command(['git','hash-object','-w','--',path],stage,env,True).strip()
            command(['git','update-index','--add','--cacheinfo','120000',blob,path],stage,env)
    message = 'Update selected server outputs' if selected else 'Snapshot all server outputs with verified SHA256 inventory'
    command(['git','commit','-m',message],stage,env)
    if selected:
        # Older LFS objects live on the remote. Check each selected pointer and
        # its local payload without downloading unrelated model files.
        lfs_env = command(['git','lfs','env'],stage,env,True).splitlines()
        media = next((Path(line.split('=',1)[1]) for line in lfs_env if line.startswith('LocalMediaDir=')), None)
        require(media is not None, 'Git LFS did not report its object directory')
        upload_oids = set()
        for name, meta in files.items():
            if uses_lfs(name, meta):
                digest = meta['sha256']
                pointer = command(['git','show','HEAD:outputs/'+name],stage,env,True).splitlines()
                require(pointer == ['version https://git-lfs.github.com/spec/v1', 'oid sha256:'+digest,
                                    'size '+str(meta['bytes'])], 'Selected LFS pointer differs: '+name)
                if meta['bytes']:
                    location = stage/media/digest[:2]/digest[2:4]/digest
                    require(location.is_file() and sha(location)==digest,'Selected LFS object differs: '+name)
                    upload_oids.add(digest)
        if upload_oids:
            # Upload only the checked payloads before updating the branch.
            # Keep the standard pre-push hook enabled for the final ref push.
            subprocess.run(['git','lfs','push','--object-id','--stdin','origin'],
                           cwd=stage,env=env,check=True,text=True,
                           input='\n'.join(sorted(upload_oids))+'\n')
    else:
        command(['git','lfs','fsck'],stage,env)
    command(['git','push','origin','HEAD:refs/heads/'+BRANCH],stage,env)
    head=command(['git','rev-parse','HEAD'],stage,env,True).strip()
    print(json.dumps({'status':'uploaded','branch':BRANCH,'commit':head,'files':len(files),
        'symlinks':sum(x.get('kind')=='symlink' for x in files.values()),'bytes':total,
        'staging_directory_preserved':str(stage)},ensure_ascii=False),flush=True)


def download(args):
    target=args.destination.resolve()
    require(not target.exists(),'Destination exists; use a new directory to preserve local changes')
    env=os.environ.copy();env['GIT_LFS_SKIP_SMUDGE']='1'
    command(['git','clone','--config','core.symlinks=false','--single-branch','--branch',BRANCH,args.remote,str(target)],args.repo.resolve(),env)
    command(['git','lfs','install','--local'],target,env)
    command(['git','lfs','pull','origin',BRANCH],target,env)
    verify(target)


def verify(root):
    manifest=json.loads((root/'outputs-manifest.json').read_text(encoding='utf-8'))
    actual=inventory(root/'outputs')
    for name,expected in manifest['files'].items():
        if expected.get('kind')=='symlink' and name in actual and actual[name].get('kind')!='symlink':
            # core.symlinks=false represents a Git link as its target text.
            raw=(root/'outputs'/name).read_bytes()
            require(raw==expected['target'].encode('utf-8'),'Git symlink target text differs: '+name)
            actual[name]={'kind':'symlink','target':raw.decode('utf-8'),
                          'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    require(actual==manifest['files'],'Snapshot files differ from manifest (or LFS objects are missing)')
    require(len(actual)==manifest['file_count'] and sum(x['bytes'] for x in actual.values())==manifest['total_bytes'],'Manifest totals differ')
    print(json.dumps({'status':'verified','files':len(actual),'bytes':manifest['total_bytes'],'directory':str(root)},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('publish','download','verify'))
    parser.add_argument('--repo',type=Path,default=Path.cwd())
    parser.add_argument('--remote',default=REMOTE)
    parser.add_argument('--only',action='append',metavar='DIRECTORY',help='Publish this directory under outputs (repeatable)')
    parser.add_argument('--destination',type=Path,default=Path('server_outputs_sync'))
    args=parser.parse_args()
    if args.action=='publish':publish(args)
    elif args.action=='download':download(args)
    else:verify(args.destination.resolve())
