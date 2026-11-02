# Full server outputs snapshot and separate local download

The reported inventory is1480 files,4.125GiB, with7 files larger than100MiB
and none larger than2GiB. Publish every regular file, including archived
runs, failed runs, logs, models, caches, manifests and prediction exports.
Symbolic links, including `_organized` directory aliases, are preserved as
Git mode120000 entries containing the original target text. Directory links,
cycles, dangling links and links outside outputs are not traversed, so targets
are not duplicated or imported from outside the requested directory. Empty
directories have no Git representation. No model result or report is filtered out.

`scripts/sync_server_outputs.py publish` creates an independent temporary
repository under the server checkout and publishes it to the dedicated
`codex/server-outputs` branch. It does not switch branches, stage code,
modify the source outputs, or change the original repository's index.
Later snapshots append normally to the same branch, without force-push.
An existing unrelated branch without our manifest is refused.

Reports remain readable ordinary Git files. Every regular file at least10MiB and
every .pt/.pth/.ckpt/.sqlite/.db file uses LFS, including large JSONL files.
Path-specific attributes support spaces and literal glob characters.
GitHub credentials are reused; billing settings are never modified. Account
LFS storage/download allowances still need to cover existing usage plus the
new snapshot. Official references:
https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github
https://docs.github.com/en/billing/concepts/product-billing/git-lfs

The manifest is now schema v2, with original link targets and their hashes.
It counts regular files and symlinks separately; the total entry count may
therefore exceed the earlier1480 regular-file inventory. Existing v1 snapshot
branches can be advanced normally, without force-push.

If Linux lacks LFS, install a standalone3.8.1 binary under the ignored
`.output-sync-tools/` directory, outside conda. Official release downloads
are checked against the pinned SHA256 values from
https://github.com/git-lfs/git-lfs/releases/download/v3.8.1/sha256sums.asc
No root privileges or GitHub API quota are required.

## Server execution

Fetch the code commit and archive only scripts to a temporary bundle, then
from `/home/nzxn/下载/WX/load/XG-NID` run:

```bash
python_bin=/home/nzxn/miniconda3/envs/wx/bin/python
"$python_bin" -B "$bundle/scripts/sync_server_outputs.py" publish --repo "$PWD"
```

Prefer running after the current experiment finishes. The script hashes all
source files, verifies copied bytes and checks the full source inventory
again before committing. Concurrent changes cause a stop instead of a mixed
snapshot. It checks for roughly2.2times the source size plus256MiB of free
disk space (about9.4GiB for this inventory), to hold the independent snapshot
and local LFS objects. It checks LFS integrity before pushing.

The final JSON includes status=uploaded, the commit, file count, byte count,
and preserved staging directory. If a network error interrupts pushing,
resume the same prepared snapshot without recopying:

```bash
git -C /path/printed/as/staging_directory push origin HEAD:refs/heads/codex/server-outputs
```

No staging directory or source data is automatically deleted.

## Local execution

After upload succeeds, download into a new directory:

```powershell
python -B D:\XG-NID\scripts\sync_server_outputs.py download --repo D:\XG-NID --destination D:\XG-NID\server_outputs_sync
```

Files land at `D:\XG-NID\server_outputs_sync\outputs`. Existing local
`D:\XG-NID\outputs` changes remain untouched. This independent clone tracks
the artifact branch; it does not merge4GiB of results into the code branch.
Git LFS is already installed on the current local machine. Download skips
automatic smudge first, explicitly pulls LFS objects, then hashes all files
and checks exact path/size/SHA256 equality against outputs-manifest.json.
Clone uses core.symlinks=false so Windows requires no symlink privileges:
links appear as small files containing their original target paths. This
preserves the Git link and target text, but `_organized` aliases are not local
clickable directories. Read the actual results under their original output
directories. Verify accepts both native symlinks and these Git target-text
files, and rejects changed link targets.
A pre-existing destination is refused. An interrupted LFS download can be
continued with `git -C D:\XG-NID\server_outputs_sync lfs pull`, then verified:

```powershell
python -B D:\XG-NID\scripts\sync_server_outputs.py verify --destination D:\XG-NID\server_outputs_sync
```

Local tests exercise two real publishes to a temporary bare remote, an LFS
download, files with spaces/brackets, manifest verification, missing/changed/
extra files, and original source HEAD/index preservation. These are local
tests, not evidence that the actual server upload has already happened.

The symlink repair additionally tests a directory alias, a dangling .pt
link (excluded from LFS), a cycle and an external target, then checks their
actual Git120000 modes after two publishes and a download. Source OS-link
metadata is emulated in the Windows tests without requiring symlink creation
privileges; Git storage, LFS transfer and checkout verification are real.
