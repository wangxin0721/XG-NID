# 残差注入强度：seed42 三组匹配训练

代码分支：`codex/residual-injection-s42`。独立输出：`outputs/residual_injection_s42_v1`。

本轮只比较 R0/R1/R2，不增加 seed43/44，不访问测试集。保留两项已有结论：残差总体有益；True−Shuffle 的 +0.04445090434977983 个百分点未通过此前 0.05 门槛。True 残差对 WebBased 误报净减少 2 个，因此本候选允许有选择地抑制或增强，不能预设所有流都应该减弱残差。

## 固定设计

| 报告简称 | 内部标识 | beta | 参数总量 |
| --- | --- | --- | ---: |
| R0 | inj_fixed | 恒为 1 | 10,872 |
| R1 | inj_global | 97 维零输入，同一个 MLP 输出全局共享强度 | 12,457 |
| R2 | inj_dynamic | 97 维样本表示，同一个 MLP 输出逐流强度 | 12,457 |

三组均在原最终融合之前计算 `packet + beta * residual`，`beta = 2 * sigmoid(g)`；原有 32 维融合门控继续使用。读出保持既有 Ordered 的 Packet 均值查询、真实连续片段与双尺度等权汇总。R0 与现有非零输出的 Ordered 前向严格相同。

R2 的门控输入按固定顺序组成：

```text
zf = flow_projection(flow)       # 注入前，可训练
zp = packet_projection(packet)   # 注入前，可训练
u = [LN(zf), LN(zp), abs(LN(zf)-LN(zp)), log1p(norm(residual)/(norm(packet)+1e-8))]
```

两个投影各 32 维，总计 97 维。LayerNorm 无仿射参数，eps=1e-5。MLP 为 `Linear(97,16) -> GELU -> Linear(16,1)`，共 1,585 个参数。R1 输入全零，R2 输入真实表示统计；表示差异不称为校准置信度。

R1/R2 初始化完全相同，独立 CPU 随机流种子固定为 **20261123**，不消耗训练 RNG 或采样器 RNG。最后一层权重和偏置均为 0，因此初始 beta=1。原读出零输出初始化、公共读出参数及全部融合参数仍完全相同。三组参数、Adam 状态与原 C 参考模型独立。

R1 因输入恒零，其第一层**输入权重**梯度一直为零是设计结果；隐藏偏置、输出层仍可学习全局强度。不能要求 R1 所有参数都出现非零梯度。R2 的门控输入到两路投影及读出的梯度保持连通，没有 detach。

## 训练协议与来源

- 三组全部调用现有 `run_ordered_transition_readout.load_source(seed=42)`，绑定此前 True/Shuffle 的同一个原 C，而非从 True191 或其他最佳检查点继续微调。
- 原 C 为 `outputs/joint_cross_layer_query_s42_v1/C/best.pt`，第 48 轮，文件 SHA256 `3070acd40203ee3a3944dc5805452eb3a2d73218aa24b31379bdfd5cbb523568`。
- 完整 C 状态摘要必须是 `c29a6433880f6e5ad8a45d387abef07cf54f895277f9f89b313edecfa7feedc6`，由归档来源绑定核验。
- 固定 train/val/test 数量 107621/13446/13448，仅加载 train/val 图。主干及读出 seed42。
- 编码器和跨层模块全部冻结且始终 eval；缓存仅存冻结的 Flow、Packet 均值、跨层 Packet、转移特征及真实连续性标记。每次训练重新计算全部可训练投影、原融合门控、分类头和新增 beta。
- 三组使用同一个显式采样器，同一批样本分别更新；batch512，eval16，200 轮，每轮 211 次、每组 42,200 次更新，不早停。
- Adam，betas=(0.9,0.999)，eps=1e-8，weight_decay=0；融合层 lr=0.0001，读出及新增门控 lr=0.001；最终八分类 NLL，无新增损失。
- 每轮验证；最佳验证 Macro-F1 包含第 0 轮，同分取最早；另保存 30/100/150/200 轮预算窗口。

历史 True 的结果提交固定为 `7c40c4b6772679d51b20a6a268fc8b4ae2f3d518`，Macro-F1 原值 `0.9791403291717622`（97.914033%），只作外部参考。配置绑定其状态、指标、来源、配置、验证预测和最终验证文件哈希。主要判断依据同期匹配训练。

投入条件使用未四舍五入数值：R2−R0 >=0.10 pp、R2−R1 >=0.05 pp，并且 R2 **严格超过**历史 True。三项必须同时满足，门槛不代表统计显著性。若仅全局强度组提高，按全局调节解释，不称为逐流适应收益。

## 必要检查和真实恢复

正式入口会先对真实图执行五次被丢弃的梯度探测，再完整验证第 0 轮的 13,446 条预测。第 1 次更新前读出内部和 beta 梯度为零；读出输出更新后 beta 输出层建立梯度，再检查 beta 隐藏层。探测使用独立模型副本，恢复 RNG 后重新初始化正式模型，不计入正式预算。

每次三组共同更新提交后原子保存 `recovery.pt`，包括所有模型与门控参数、Adam、轮次、累计更新、当前批游标、完整采样顺序与生成器状态、Python/NumPy/Torch/CUDA RNG、损失累计和最佳指标/检查点。进程在任意组更新期间失败，恢复只使用上次三组已共同提交的状态，部分更新不计入提交预算。派生账本和最佳/末轮文件可从恢复状态重新生成。

恢复要求配置、代码提交、源码哈希、来源绑定、环境、图根目录和冻结缓存哈希全部一致；否则明确报错。成功完成后的再启动只回放验证，不重复任何训练更新。最佳模型须通过完整原图验证回放，并且预测摘要与缓存选模一致。

本地已完成：

- **40 项 CPU 检查**，含本候选及原联合训练、配对打乱、交叉回放回归；见 `cpu_validation.json` 与日志。
- 同一模型非零读出下，R0 与已有 Ordered 前向一致；R1/R2 的 beta=1 也完全一致。
- 5 步梯度、活跃投影梯度、冻结参数/缓冲区、采样预算、批中断/轮边界/完成后恢复、三组事务失败回滚、Adam/RNG/采样器精确恢复、合成图最佳检查点回放、失败后自动发布与退出码保留。
- 对已绑定原 C 完整权重及归档验证冻结缓存进行真实 CPU 第 0 轮检查：**三组各 13,446 条输出与原 C 完全一致**，最大 log probability 差为 0，beta 均为 1。该项未执行图编码器，不能称为服务器原图探测或正式训练。见 `real_cached_cpu_preflight.json` 和 `real_cached_cpu_gradients.json`。

**当前执行状态（2026-11-23）：尚未启动正式训练。** 本机 Torch 为 CPU 构建；已配置服务器的跳板及直连 SSH 均超时。服务器 CUDA/真实图探测及 200 轮正式训练仍需在可连接的服务器执行下方入口。没有将合成训练或缓存检查记作正式结果。

## 服务器直接启动三组

使用既有服务器的 wx 环境、原始图、原 C 和来源归档。入口自动进行必要探测，通过后直接训练，无需先单独运行一轮诊断。工作树代码与数据仓库分离，保留主工作区修改和历史结果。

```bash
conda activate wx
cd '/home/nzxn/下载/WX/load/XG-NID' || exit 1
xgnid_repo="$PWD"
xgnid_code="$(dirname "$PWD")/XG-NID-residual-injection"
git fetch --no-tags origin refs/heads/codex/residual-injection-s42 || exit 1
xgnid_revision="$(git rev-parse FETCH_HEAD)"
if [ -e "$xgnid_code" ]; then
  test "$(git -C "$xgnid_code" rev-parse HEAD)" = "$xgnid_revision" || exit 1
else
  GIT_LFS_SKIP_SMUDGE=1 git worktree add --detach "$xgnid_code" "$xgnid_revision" || exit 1
fi
git -C "$xgnid_code" diff --quiet || exit 1
git -C "$xgnid_code" diff --cached --quiet || exit 1
mkdir -p "$xgnid_repo/tmp" "$xgnid_repo/outputs/residual_injection_s42_v1"
touch "$xgnid_repo/outputs/residual_injection_s42_v1/training.log"
nohup python -u -B "$xgnid_code/scripts/launch_residual_injection.py" \
  --repo "$xgnid_repo" \
  > "$xgnid_repo/tmp/residual_injection_s42_v1.launcher.log" 2>&1 < /dev/null &
echo $! > "$xgnid_repo/outputs/residual_injection_s42_v1/launcher.pid"
tail -F "$xgnid_repo/outputs/residual_injection_s42_v1/training.log"
```

如原始图移动过，在启动参数添加 `--data-root '/实际图根目录'`；首次启动后恢复必须使用相同目录。不要分别为三个组启动进程，本入口按共享批次配对更新三组。重复执行相同入口会自动续训；锁阻止并发重复启动。

查看上传与退出状态：

```bash
tail -F "$xgnid_repo/tmp/residual_injection_s42_v1.launcher.log"
cat "$xgnid_repo/outputs/residual_injection_s42_v1/training_exit.json"
cat "$xgnid_repo/outputs/residual_injection_s42_v1/exit_codes.json"
```

## 自动上传与独立重试

正常结束或可捕获失败（含 SIGINT/SIGTERM）后，入口保留训练退出状态并将本轮全部文件发布至 `codex/server-outputs`。`sync_server_outputs.py` 仍负责选择性发布与 Git LFS，包含冻结缓存、最佳权重、末轮状态、恢复状态、预测、账本、beta 诊断、中文报告、日志和失败证据。

发布协调锁位于仓库 `tmp`，不放在待上传目录；训练日志关闭后才上传。先发布结果，再发布关闭的上传日志和退出收据。上传失败会明确打印重试命令，训练退出码不会被上传失败覆盖。

待进程结束后单独重传，不重复训练：

```bash
xgnid_repo='/home/nzxn/下载/WX/load/XG-NID'
xgnid_code="$(dirname "$xgnid_repo")/XG-NID-residual-injection"
python -B "$xgnid_code/scripts/sync_server_outputs.py" publish --repo "$xgnid_repo" --only residual_injection_s42_v1
```

## 结果文件

`config.json`、`provenance.json`、`preparation_environment.json` 保存协议、来源和环境。`raw_gradient_probe.json`、`epoch_zero_replay.json`、`server_probe.json` 保存正式前探测收据。每组的 `epochs.jsonl`、`best.pt`、`last.pt`，以及公共 `recovery.pt` 和 `training_progress.json` 保存预算和恢复状态。

`summary.json`、中文 `report.md`、`budget_comparison.json` 汇总全局选模及全部投入条件。`validation` 保存八类 P/R/F1、混淆矩阵、类别差值、修复/退步清单、WebBased TP/FP/FN 和误报来源。每组完整最佳预测单独导出。

`beta_distributions.json` 包含最佳检查点的总体、逐类、正误、连续片段覆盖和修复/退步分组；`last_beta_distributions.json` 保存末轮分布。逐样本记录原始/注入残差强度、beta、原 32 维融合权重及相对同检查点关闭残差的权重变化、两尺度注意力和完整 log probability。

`selected_validation_replay.json` 与 `final_verification.json` 仅在最佳模型原图回放和完整预算通过后确认成功。最终状态必须结合 `status.json`、训练退出码、上传日志判断。
