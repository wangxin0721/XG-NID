# Seed42 六组固定检查点回放：零轮训练

本轮只完成固定模型敏感性诊断，独立分支 `codex/transition-pair-cross-replay-s42`，输出目录 `outputs/transition_pair_cross_replay_s42_v1`。代码基于 `f3baba6ce8b80ed22283e6a501f2dfb992d0edf8`，输入绑定结果提交 `7c40c4b6772679d51b20a6a268fc8b4ae2f3d518`。

## 固定条件

| 组别 | 检查点 | 读出配对 | 融合层输入 |
| --- | --- | --- | --- |
| T1 | True 第 191 轮 | 真实配对 | packet + residual |
| T2 | True 第 191 轮 | 归档固定打乱 | packet + residual |
| T0 | True 第 191 轮 | 不注入；诊断记录其原生候选残差 | packet |
| S1 | Shuffle 第 183 轮 | 真实配对 | packet + residual |
| S2 | Shuffle 第 183 轮 | 归档固定打乱 | packet + residual |
| S0 | Shuffle 第 183 轮 | 不注入；诊断记录其原生候选残差 | packet |

0 轮训练、0 次优化器更新、六组各覆盖完整 13,446 条验证流。全部参数冻结且统一 eval。T0/S0 保留各自训练后的融合投影、逐维门控与分类头，只关闭残差注入。重建模块后严格加载全部权重，临时初始化值全部被检查点替换。

本轮直接复用已归档的冻结 Flow/Packet/转移特征，不重新运行图编码器，不访问训练集特征或测试集图。输入包含两个最佳检查点、验证缓存、固定验证映射、原生预测与来源记录，13 个文件的 SHA256 和大小均预先绑定在配置中。验证映射从 sample_id 重新生成并与归档逐项一致。

CPU 可执行该回放。真实数据预检查中，两个原生组在 CPU 上均与原归档的 13,446 条预测完全一致；完整六组结果以新输出目录的收据和报告为准。也支持 CUDA，但必须重新满足原生预测一致性检查。

## 解释前的必要检查

- T1、S2 的样本 ID、标签、预测、完整分类指标及预测摘要必须与相应原生归档一致；否则保存失败收据并停止解释。
- 对冻结 70 维转移特征实际未改变的流，分别检查 T1/T2、S1/S2 的全部 log probability、残差和逐维融合权重逐位一致。
- 回放前后检查两个模型全部参数、缓冲区、grad 与 eval 状态保持不变；结束前再次校验全部输入哈希。
- 比较 T1−T2、S1−S2、T1−T0、S2−S0，并保留 T1−S2 和 T0−S0，支持追踪训练后融合层与残差的分量差异。

本地 33 项 CPU 检查覆盖六路输出、保留训练后融合参数、冻结状态、门控/残差诊断、未变组一致性、原生复现失败阻断报告、输入哈希、完整误报样本清单及失败后自动上传，同时回归原 26 项检查。收据见 `docs/transition_pair_cross_replay_s42_v1/cpu_validation.json` 与 `cpu_validation.log`；该收据是合成检查，不能代替真实 13,446 条回放收据。

## 输出

- `T1`、`T2`、`T0`、`S1`、`S2`、`S0` 各自的完整验证预测。
- `metrics.json`、`per_class.csv`、六组混淆矩阵，包含八类 P/R/F1。
- `native_archive_replay.json`、`unchanged_feature_consistency.json`、`final_verification.json`。
- `sample_diagnostics.jsonl.gz`：每个样本六路 log probability、预测、注入/候选残差 L2、相对 Packet 强度、投影后残差强度、32 维融合权重及相对关闭残差的权重变化。
- `comparisons/*_changed_samples.jsonl.gz`：修复、退步及错误间切换，保留全部六路预测和诊断。
- `webbased_false_positives.json`、`webbased/`：每组 WebBased TP/FP/FN，按真实来源类拆分 FP，新增/消除/共同 FP 的逐样本清单。
- 输入哈希、来源提交、环境、映射摘要、两个原始最佳检查点副本及原验证映射。
- `summary.json`、中文 `report.md`、`replay.log`、`replay_exit.json`、上传日志和退出码。

`raw_residual_l2` 在 T0/S0 表示同一检查点原生配对计算出的未注入候选；`injected_residual_l2` 在 T0/S0 恒为 0。门控权重是原逐维融合中的 Flow 占比，不是尚未训练的 beta 注入门控。

此前 WebBased “净增 9 个误报”需要分解为新增集合与消除集合，并追踪所有对称差样本，不能直接称为只有 9 个新增样本。按真实来源类保存恒等式：

`T1_FP − S2_FP = (T0_FP − S0_FP) + (T1_FP − T0_FP) − (S2_FP − S0_FP)`。

其中 T0−S0 是两套训练后融合参数在无残差输入条件的差异。恒等分解不是训练因果归因；同模型内替换配对或关闭残差也不能替代匹配训练消融。

## 服务器复现

使用已有 wx 环境及原归档目录即可，不需要原始图或训练缓存。入口在正常结束或可捕获失败后将整个新目录发布到 `codex/server-outputs`，大文件沿用 Git LFS。

```bash
conda activate wx
cd '/home/nzxn/下载/WX/load/XG-NID'
xgnid_repo="$PWD"
xgnid_code="$(dirname "$PWD")/XG-NID-transition-pair-cross-replay"
git fetch --no-tags origin refs/heads/codex/transition-pair-cross-replay-s42
xgnid_revision="$(git rev-parse FETCH_HEAD)"
if [ -e "$xgnid_code" ]; then
  test "$(git -C "$xgnid_code" rev-parse HEAD)" = "$xgnid_revision" || exit 1
else
  git worktree add --detach "$xgnid_code" "$xgnid_revision"
fi
mkdir -p "$xgnid_repo/tmp"
nohup python -u -B "$xgnid_code/scripts/launch_transition_pair_cross_replay.py" \
  --repo "$xgnid_repo" --device cpu \
  > "$xgnid_repo/tmp/transition_pair_cross_replay_s42_v1.launcher.log" 2>&1 < /dev/null &
tail -F "$xgnid_repo/outputs/transition_pair_cross_replay_s42_v1/replay.log"
```

`--device cpu` 使用冻结缓存执行推断；支持 `cuda` 或 `auto`。更改设备或来源不能覆盖已有不同绑定的结果目录。入口的 `--resume` 只允许同一绑定重做零训练回放，不涉及训练续训。

如果本地没有物化的输入目录，可从固定 Git 快照及预先下载的 LFS 对象读取：

```bash
git fetch --depth=1 origin codex/server-outputs
git lfs fetch --include='outputs/transition_pair_shuffle_s42_v1/pair_true/best.pt,outputs/transition_pair_shuffle_s42_v1/pair_shuffle/best.pt,outputs/transition_pair_shuffle_s42_v1/frozen_val_features.pt' --exclude='' origin 7c40c4b6772679d51b20a6a268fc8b4ae2f3d518
python -u -B "$xgnid_code/scripts/launch_transition_pair_cross_replay.py" --repo "$xgnid_repo" --snapshot --device cpu
```

上传失败时，待回放日志关闭后单独重试：

```bash
python -B "$xgnid_code/scripts/sync_server_outputs.py" publish --repo "$xgnid_repo" --only transition_pair_cross_replay_s42_v1
```

Windows 上入口的协调锁保存在仓库 `tmp` 下，避免发布器读取输出目录时碰到仍被锁住的字节。首次本地全量回放完成后发现此发布问题；修复只涉及入口协调锁，不重算已完成的六组预测。结果的 `provenance.json` 保留实际推断代码提交，发布修复提交单独可追踪。

当前 True 候选 **97.914033%** 保留；六组诊断不改变上一轮 True−Shuffle **+0.044451 pp <0.05 pp** 的结论。本轮不实现或启动 R0/R1/R2；先按本次残差修复、破坏、误报来源及关闭残差表现决定下一阶段。
