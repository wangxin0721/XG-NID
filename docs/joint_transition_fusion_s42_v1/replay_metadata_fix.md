# 原 C 验证回放字段比较修复

首次服务器运行（1e47dd4）完成 13,446 条验证缓存后，在 `reference == gold` 处停止，未进入 J0/J1/J2/J3 正式训练。归档原 C 预测包含 `sample_id/label/prediction/flow_weights`，新缓存的预测记录仅包含前三项。直接比较完整字典，即使预测相同也会失败。上方 torch-spline-conv 导入警告不是这次 ValueError 的触发点。

修复仅调整回放比较：严格验证全部样本 ID、标签、顺序、预测类别、由混淆矩阵重算的完整分类指标及原 C 选择时的预测摘要；门控向量作为诊断元数据单独保留。成功和失败均写 `baseline_validation_replay.json`，真实错分会记录变化样本，身份或结构错误也保留错误原因。实际服务器预测是否完全一致，必须由修复后的新回放收据确认。

本地 18 项 CPU 检查全部通过，包含真实归档形状的门控元数据、预测/身份/标签/指标/摘要不匹配拒绝，以及携带归档门控字段的完整合成流程。另使用归档提交 `944ba61faca474d02c409c2038e02f33e5da3d50` 的全部 13,446 条原 C 预测做字段回归：完整字典比较失败、严格预测回放比较通过，见 `archive_schema_check`。这项归档检查没有运行服务器模型。

配置、学习率、损失、原 C 来源和训练预算保持原协议。代码摘要已变化，因此不直接加载旧来源绑定的缓存或手工编辑其来源记录。本次重新准备缓存，失败目录完整移入新输出目录的 `failed_runs`，随新结果一起自动上传。

在原 wx 环境运行下列命令。它等待旧启动器结束自动上传，建立新的修复工作树，归档失败现场后后台启动。若发现 `recovery.pt`，则拒绝移动，避免将已经产生的正式训练状态误当成准备阶段失败。

```bash
bash <<'SH'
set -eu
cd '/home/nzxn/下载/WX/load/XG-NID'
xgnid_repo="$PWD"
xgnid_code="$(dirname "$PWD")/XG-NID-joint-transition-fusion-replayfix"
xgnid_out="$xgnid_repo/outputs/joint_transition_fusion_s42_v1"

# 旧训练已失败，但启动器可能仍在上传；在它退出之前不移动输出目录。
if [ -f "$xgnid_out/launcher.pid" ]; then
  xgnid_pid="$(cat "$xgnid_out/launcher.pid")"
  while ps -p "$xgnid_pid" -o args= | grep -q '[l]aunch_joint_transition_fusion.py'; do
    echo '等待旧启动器完成失败现场上传……'
    sleep 5
  done
fi
test ! -e "$xgnid_out/recovery.pt" || {
  echo '发现正式训练恢复状态，保留现场并停止本次准备阶段重启。'
  exit 1
}

git fetch --no-tags origin refs/heads/codex/joint-transition-fusion
xgnid_revision="$(git rev-parse FETCH_HEAD)"
if [ -e "$xgnid_code" ]; then
  test "$(git -C "$xgnid_code" rev-parse HEAD)" = "$xgnid_revision"
else
  git worktree add --detach "$xgnid_code" "$xgnid_revision"
fi

mkdir -p "$xgnid_repo/tmp"
if [ -d "$xgnid_out" ]; then
  xgnid_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  xgnid_backup="$xgnid_repo/tmp/joint-transition-before-replay-fix-$xgnid_stamp"
  test ! -e "$xgnid_backup"
  mv -- "$xgnid_out" "$xgnid_backup"
  mkdir -p "$xgnid_out/failed_runs"
  mv -- "$xgnid_backup" "$xgnid_out/failed_runs/$xgnid_stamp"
else
  mkdir -p "$xgnid_out"
fi

nohup python -u -B "$xgnid_code/scripts/launch_joint_transition_fusion.py" \
  --repo "$xgnid_repo" \
  > "$xgnid_repo/tmp/joint_transition_fusion_s42_v1.replayfix.launcher.log" 2>&1 < /dev/null &
echo $! > "$xgnid_out/launcher.pid"
SH

tail -F '/home/nzxn/下载/WX/load/XG-NID/outputs/joint_transition_fusion_s42_v1/training.log'
```

如再次停在原 C 回放处，直接查看 `baseline_validation_replay.json`，不要绕过校验继续训练。新入口仍在结束/失败后自动发布整个 `joint_transition_fusion_s42_v1`。训练停止后可单独重试上传：

```bash
xgnid_repo='/home/nzxn/下载/WX/load/XG-NID'
xgnid_code="$(dirname "$xgnid_repo")/XG-NID-joint-transition-fusion-replayfix"
python -B "$xgnid_code/scripts/sync_server_outputs.py" publish --repo "$xgnid_repo" --only joint_transition_fusion_s42_v1
```
