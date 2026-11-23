# Git LFS 启动失败的修复与恢复

服务器在原 `7c9249e` 入口的 `import` 阶段退出：`git: 'lfs' 不是一个 git 命令`。`Archive.locate()` 调用 `git lfs env` 时没有可用的 LFS PATH；旧发布器会在后续上传时调用 `sync_server_outputs.lfs_environment()`，其仓库内二进制环境只供发布子进程使用，没有传给之前的归档读取进程。GitHub 已保留本次退出1及上传退出0的收据；六次正式训练均未开始。

修复后的入口在首个子进程之前调用现有 LFS helper，并把 PATH 传给所有阶段及上传器。直接调用归档读取时同样先准备 LFS，把环境显式传给 `git show`、`git lfs env`、`git fetch` 和 `git lfs fetch`。Linux 缺少全局 LFS 时，复用或下载仓库 `.output-sync-tools` 下已有哈希校验的独立二进制，无需 sudo 或修改系统安装。

三个新增回归场景覆盖：系统无 LFS 但私有缓存可用；归档读取及缺失对象下载使用私有 PATH；启动顺序与 bootstrap 失败后的退出上传。另回归原57项 CPU 检查。未改变模型、种子、损失、学习率、预算或选模规则。

## 本次建议：在原锁定工作树补齐 LFS 后续跑

沿用服务器现有工作树及其 `7c9249e` 提交，可保留原协议锁、失败日志和任何已生成的预训练前产物。以下命令只补环境并重新调用原入口。启动脚本会从已提交状态续训；本次失败在任何正式更新之前。

```bash
conda activate wx
cd '/home/nzxn/下载/WX/load/XG-NID' || exit 1
xgnid_repo="$PWD"
xgnid_code="$(dirname "$PWD")/XG-NID-fixed-transition-stability"
xgnid_output="$xgnid_repo/outputs/fixed_transition_stability_v1"

if [ -f "$xgnid_output/launcher.pid" ] && kill -0 "$(cat "$xgnid_output/launcher.pid")" 2>/dev/null; then
  echo '旧入口仍在运行或上传；等待退出后再执行本段。'
  exit 1
fi

export PATH="$xgnid_repo/.output-sync-tools/git-lfs-3.8.1-amd64:$xgnid_repo/.output-sync-tools/git-lfs-3.8.1-arm64:$PATH"
PYTHONPATH="$xgnid_code/scripts${PYTHONPATH:+:$PYTHONPATH}" python -B -c \
  'from pathlib import Path; from sync_server_outputs import lfs_environment; lfs_environment(Path.cwd())' || exit 1
git lfs version || exit 1

mkdir -p "$xgnid_repo/tmp" "$xgnid_output"
touch "$xgnid_output/training.log"
nohup python -u -B "$xgnid_code/scripts/launch_fixed_transition_stability.py" \
  --repo "$xgnid_repo" \
  >> "$xgnid_repo/tmp/fixed_transition_stability_v1.launcher.log" 2>&1 < /dev/null &
echo $! > "$xgnid_output/launcher.pid"
tail -F "$xgnid_output/training.log"
```

此处 PATH 同时覆盖 Linux amd64/arm64 的固定工具目录；helper 自动选择当前架构并校验下载。已有二进制会直接复用。若独立工具下载失败，入口会停止并保留失败收据，不能把 `git lfs version` 失败后的步骤当作启动成功。

实时查看安装/发布日志：

```bash
tail -F '/home/nzxn/下载/WX/load/XG-NID/tmp/fixed_transition_stability_v1.launcher.log'
```

进程退出后的独立上传重试仍为：

```bash
python -B "$xgnid_code/scripts/sync_server_outputs.py" publish --repo "$xgnid_repo" --only fixed_transition_stability_v1
```

## 新修复代码与旧协议锁

新版本适用于新建的代码工作树和空实验目录，会自动完成上述 bootstrap，并保存 `lfs_preflight.json`。已经写入协议锁或恢复状态的旧任务继续使用其原代码工作树和补齐后的 PATH。直接替换代码提交会触发现有代码来源锁检查；不要手工篡改协议锁或清除恢复状态。

本地测试不是服务器探测；当前 SSH 仍超时，没有远程启动任何新训练。服务器重启后以 `lfs_preflight.json`（新入口）、第0轮验证收据、采样账本和阶段退出码确认实际进度。
