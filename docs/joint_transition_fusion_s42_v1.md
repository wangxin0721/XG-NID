# Seed42：报文转移读出与双分支融合层联合优化

本轮准备四组同起点、同样本顺序、同预算的 train/val 配对实验。代码基于 `d03ced606154f1d7b4afd6f29ed0f6f4d2f7011d`，分支为 `codex/joint-transition-fusion`。本地仅完成归档诊断及合成 CPU 验证；服务器 CUDA 探测和正式训练均待执行，测试集未评估。

## 已有证据与诊断

诊断使用结果提交 `944ba61faca474d02c409c2038e02f33e5da3d50` 的全部 13,446 条验证预测，预测文件 SHA256 为 `db1e5a9656df731bc32d70af4dc8ee22023f5a17afaf616ca5954651ff8c4662`。

| 真实类别 | Ordered 相对 Single 修复 | 退步 | 错误类别之间变化 |
| --- | ---: | ---: | ---: |
| Benign | 8 | 9 | 1 |
| WebBased | 4 | 4 | 0 |
| Spoofing | 4 | 1 | 2 |
| Recon | 4 | 8 | 2 |
| Mirai | 0 | 0 | 0 |
| Dos | 2 | 0 | 0 |
| DDos | 0 | 0 | 0 |
| BruteForce | 0 | 0 | 1 |
| 合计 | 22 | 22 | 6 |

WebBased→Recon 修复 3 例，Recon→WebBased 修复 3 例；原正确 Recon 新错为 WebBased 4 例。Spoofing 修复的 4 例原均错为 Benign。BruteForce 的 1 例是 Spoofing→WebBased，前后均错。

修复中 7 例、退步中 5 例只有一条有效转移，确定不存在连续两步片段。其余样本的逐流连续片段数在旧预测中未保存，标记为 unknown。归档覆盖统计为 WebBased 404/544（74.2647%）、Recon 332/2278（14.5742%）、Spoofing 798/1739（45.8884%）、BruteForce 227/233（97.4249%）。覆盖差异是观察；“冻结门控未充分利用新表示”是待验证假设。

完整错分方向、有效一步数量分组和变化样本见 [归档诊断](joint_transition_fusion_s42_v1/archive_diagnosis/report.md)、`diagnosis.json` 和 `changed_samples.jsonl.gz`。服务器训练前，会用原图冻结特征及旧最佳读出重放全部验证样本，补齐真实连续性、两尺度注意力、残差及逐维门控；要求所有旧预测完全一致。连续片段严格使用 `attach_continuity` 的共享中间报文标记，禁止以转移数 ≥2 代替。

## 实验协议

原 C 继续由 `run_ordered_transition_readout.load_source(..., seed=42)` 加载，保留既有来源检查。实际来源为 `outputs/joint_cross_layer_query_s42_v1/C/best.pt`，SHA256 为 `3070acd40203ee3a3944dc5805452eb3a2d73218aa24b31379bdfd5cbb523568`，模型全状态摘要为 `c29a6433880f6e5ad8a45d387abef07cf54f895277f9f89b313edecfa7feedc6`。启动时必须与旧 Ordered 的来源绑定完全相同。

| 内部标识 | 简称 | 新增读出参数 | 可训练融合参数 |
| --- | --- | ---: | ---: |
| joint_base | J0 | 0 | 7,144 |
| joint_single | J1 | 3,200 | 7,144 |
| joint_pointwise | J2 | 3,728 | 7,144 |
| joint_ordered | J3 | 3,728 | 7,144 |

四组融合层从原 C 深复制，参数存储和 Adam 状态独立。J1/J2/J3 公共读出参数相同初始化，J2/J3 context 参数相同初始化，输出投影全零。四组第 0 轮全量验证预测和 log probability 必须与原 C 一致。旧第 59/94/129 轮读出仅用于诊断与冻结结果对比，不作为新训练起点。

两路图编码器、原跨层表示及原查询池均冻结，保持 eval；原 C 参考模型单独保存且不参与任何优化器。缓存只包含 Flow 表示、Packet 均值、冻结跨层 Packet 表示、有效转移特征和真实连续性标记。训练每批重新计算可训练的 Flow/Packet 投影、逐维门控和最终分类头。Single/Pointwise/Ordered 的原定义、Packet 均值查询及可用尺度平均规则均保持原样。

- 固定已有 split、样本清单和八类映射：train/val/test = 107621/13446/13448。本入口只加载 train/val 图。
- 主干与读出 seed42；每个批次依次更新四组，因此每轮样本顺序完全相同。保留旧 `TrainBatches` 的 randperm 与单样本尾批合并规则。
- 完整 200 轮，无早停；train batch512，eval batch16；211 更新/轮，每组 42,200 更新。
- Adam：betas=(0.9,0.999)，eps=1e-8，weight_decay=0，amsgrad=False；读出 lr=0.001，融合层 lr=0.0001。
- 最终八分类 NLL；每轮验证；包含第 0 轮，以 Macro-F1 选最佳，同分选最早。
- 同轨迹汇总 30/100/150/200 轮预算窗口。本轮不追加超参数搜索、seed43/44 或测试评估。

## 恢复与验证

`recovery.pt` 在第 0 轮、每个四组配对更新事务后、每轮验证提交后原子保存，包含四组当前权重、独立优化器、轮次、累计更新、轮内损失、全局随机状态、采样器独立 generator/当前排列/游标、逐轮账本、最佳指标和最佳权重。文件 flush/fsync 后再替换。`--resume` 校验配置、代码提交与文件摘要、来源、环境、输入和冻结缓存摘要；不匹配则明确失败。

若某批次仅更新了部分组就失败，该批次尚未提交；恢复时从上一份完整事务继续。已提交更新不重复或跳过。`best.pt`、`last.pt`、逐轮 JSONL 可从恢复状态重建，避免检查点与账本跨文件写入中断造成歧义。结束后，最佳检查点重新从原图验证回放，并核对缓存选择时的预测摘要和全部分类指标。

CPU 验证使用 Python 3.10.1、Torch 2.3.1+cpu、PyG 2.5.3；记录见 `docs/joint_transition_fusion_s42_v1/cpu_validation.json` 和 `.log`。覆盖真实合成图编码、共同初始化、梯度分阶段建立、冻结参数及缓冲区、逐批/轮末中断恢复、部分组失败回滚、Adam 步数、采样预算、错配拒绝、最佳原图回放、完整入口探测/缓存/训练/完成后恢复，以及失败后结果发布。它不代表服务器 CUDA 验证通过。

旧 Ordered 的一个测试改为让基线和新实现使用同一 packed 切片步幅进行逐位比较，避免 CPU 不同 GEMM 布局的约 1.49e-8 浮点差；读出计算代码未改变。

可在安装好依赖的环境复现 CPU 检查：

```sh
export PYTHONPATH="$xgnid_code/src:$xgnid_code/scripts${PYTHONPATH:+:$PYTHONPATH}"
python -B -m unittest -v test_joint_transition_fusion test_ordered_transition_readout test_packet_transition_readout
```

## 服务器启动

沿用旧 200 轮实验记录的服务器仓库路径。先激活原 `wx` 训练环境；该环境需 CUDA、Torch 2.3.1、PyG 2.5.3，并保有原始图、固定 split、实际原 C 及 `outputs/ordered_transition_readout_200/seed42`。所有旧输入均原位只读并校验哈希。

```sh
conda activate wx
cd '/home/nzxn/下载/WX/load/XG-NID'
xgnid_repo="$PWD"
xgnid_code="$(dirname "$xgnid_repo")/XG-NID-joint-transition-fusion"

git fetch --no-tags origin refs/heads/codex/joint-transition-fusion
xgnid_revision="$(git rev-parse FETCH_HEAD)"
if [ -e "$xgnid_code" ]; then
  test "$(git -C "$xgnid_code" rev-parse HEAD)" = "$xgnid_revision" || {
    echo '已有工作树版本不同，请保留现场并核对版本。'
    exit 1
  }
else
  git worktree add --detach "$xgnid_code" "$xgnid_revision"
fi

mkdir -p "$xgnid_repo/tmp" "$xgnid_repo/outputs/joint_transition_fusion_s42_v1"
nohup python -u -B "$xgnid_code/scripts/launch_joint_transition_fusion.py" \
  --repo "$xgnid_repo" \
  > "$xgnid_repo/tmp/joint_transition_fusion_s42_v1.launcher.log" 2>&1 < /dev/null &
echo $! > "$xgnid_repo/outputs/joint_transition_fusion_s42_v1/launcher.pid"
```

启动器自动执行原图 CUDA 梯度探测、原 C 与四组第 0 轮全量验证、旧 200 轮结果诊断回放，再训练四组。任何检查失败均停止训练并进入上传。图根目录若与 split 中登记的路径不同，在启动器参数后加 `--data-root '/实际图根目录'`；不改 split 或样本清单。

若只做服务器探测，可先前台执行下列入口，随后用上面的后台命令继续同一目录：

```sh
python -u -B "$xgnid_code/scripts/launch_joint_transition_fusion.py" --repo "$xgnid_repo" --probe-only
```

查看实时进度：

```sh
tail -F "$xgnid_repo/outputs/joint_transition_fusion_s42_v1/training.log"
# 上传进度及最终训练/上传退出状态
tail -F "$xgnid_repo/tmp/joint_transition_fusion_s42_v1.launcher.log"
```

中断后，确认旧启动器已退出，再运行同一后台启动命令；它自带 `--resume`。如需可捕获地停止训练并上传现场：

```sh
kill -TERM "$(cat "$xgnid_repo/outputs/joint_transition_fusion_s42_v1/launcher.pid")"
```

## 输出与自动上传

本轮唯一输出目录是 `outputs/joint_transition_fusion_s42_v1`，包含配置/来源/环境/哈希、原图探测、零轮回放、旧结果定向诊断、冻结特征缓存、每组账本/best/last/最佳预测、全量恢复状态、八类指标和混淆矩阵、配对变化样本、门控/残差/两尺度注意力、预算比较、中文报告、训练与上传日志及退出码。

统一启动器在正常退出、异常或可转发的 SIGINT/SIGTERM 后均调用既有发布机制：

```sh
python -B "$xgnid_code/scripts/sync_server_outputs.py" publish --repo "$xgnid_repo" --only joint_transition_fusion_s42_v1
```

目标分支为 `codex/server-outputs`。权重、恢复状态、冻结缓存及其他大文件沿用既有 LFS 规则，其他目录保留。发布时训练日志已关闭；上传日志先写在目录外，关闭后复制到输出目录，并做一次日志/退出码收据发布，避免上传过程改变正在校验的源目录。最终收据上传的实时输出写入目录外 launcher 日志。发布失败明确打印独立重试命令，恢复状态和本地结果保留；启动器始终返回训练退出码，上传退出码另行记录，不替换训练失败状态。

上传独立重试使用上面的 publish 命令；重试时训练与日志写入须已结束。

## 结果判读

以本轮同时实施的四组配对结果为主。报告比较 J3−J1、J3−J2、J3−J0、各组相对原 C，以及 J1/J2/J3 相对对应冻结实验的变化，并保留八类得失、修复/退步方向和连续性分组。旧 Single 前 30 轮重跑存在已记录的小幅差异，不能用旧轨迹替代本轮配对对照。

投入参考按未四舍五入数值判断：J3−J1 ≥0.10 pp，J3−J2 ≥0.05 pp，同时检查 J3 是否超过冻结 Ordered 的实测值 97.845839%。门槛用于资源分配，不代表统计显著性。四组共同提升但 J3 没有拉开差距时，报告为主要来自融合层优化的收益；仅在配对差距支持后，再安排同预算真实连续配对 vs 流内打乱配对，随后考虑 seed43/44，最后锁定模型评估测试集。
