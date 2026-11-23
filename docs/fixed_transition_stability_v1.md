# 固定注入报文转移模型：独立主干稳定性验证

分支 `codex/fixed-transition-stability`，从 `ee456f02f36192716733836bfd579aa2e3903882` 建立独立工作树。输出 `outputs/fixed_transition_stability_v1`，分为 `seed42_import`、`seed43`、`seed44`、`aggregate`。

本轮研究保留模型的整体收益。保留双分支异构图、原跨层模块、Flow 引导读出、32 维融合门控、真实连续配对的 Ordered 读出及固定 `packet + residual` 注入；不实例化 beta 门控，不扩展 True/Shuffle 或其他结构。此前两项消融未通过门槛的结论保持原样。

## 组别及种子含义

| 模型 | 本阶段优化范围 | 可训练参数 |
| --- | --- | ---: |
| base_C | 无，固定原 C 参考 | 0 |
| frozen_single | Single 读出，原 C 全冻结 | 3,200 |
| joint_single | Single + Flow/Packet 投影、原逐维门控及分类头 | 10,344 |
| joint_ordered | Ordered + 同上融合层 | 10,872 |

每个主干的三个训练组从同一个原 C 出发，参数副本和 Adam 状态独立。Single/Ordered 的公共参数相同初始化，读出输出为零，第 0 轮全验证集预测须等于对应原 C。Frozen Single 的融合副本在 `train()` 后仍为 eval 且无梯度；两路编码器、原 Flow 引导读出和跨层模块均不在新优化器中，冻结源模型始终 eval。

准确称为 **三个独立主干上的配对验证**：backbone_seed=42/43/44；本阶段 readout_seed=42、sampling_seed=42，不能写成所有模块以三个不同种子完整重训。seed42 是开发阶段锁定归档，主要新增证据来自 seed43、44。

## 锁定的 seed42 来源

| 模型 | 结果提交 | 路径/组 | 最佳轮 | 验证 Macro-F1 |
| --- | --- | --- | ---: | ---: |
| base_C | 下列归档绑定的同一原 C | joint_cross_layer_query_s42_v1/C | 48 | 97.553803% |
| frozen_single | 944ba61faca474d02c409c2038e02f33e5da3d50 | ordered_transition_readout_200/seed42/single | 94 | 97.825365% |
| joint_single | ac00f1f319aec1b1079e15ca5609c2ec08c85ba9 | joint_transition_fusion_s42_v1/joint_single | 59 | 97.630209% |
| joint_ordered | 7c40c4b6772679d51b20a6a268fc8b4ae2f3d518 | transition_pair_shuffle_s42_v1/pair_true | 191 | 97.914033% |

程序从不可变 Git 快照和哈希匹配的 LFS 对象读取未舍入值、预测、权重和账本。原 C 文件 SHA256 为 `3070acd40203ee3a3944dc5805452eb3a2d73218aa24b31379bdfd5cbb523568`，完整状态摘要为 `c29a6433880f6e5ad8a45d387abef07cf54f895277f9f89b313edecfa7feedc6`。四组源绑定必须一致。

seed42 只导入，不训练；同时保留最佳及末轮轻量权重、原始 C 权重和归档账本。导入时以同一绑定的冻结验证特征在 CPU 回放四个非零最佳模型，逐样本核对全部 13,446 条预测，继承并检查历史原图回放收据。该项不称为新增原图回放或新增训练。

历史代码分别为 `d03ced6`、`4f2dd77`、`f3baba6`。导入记录文件级规范化源码哈希差异、模型功能检查和预测摘要。旧 Frozen Single 的损失配置标签为 `plain_final_NLL`，其他组为 `plain_final_eight_class_NLL`；均沿用最终八分类 NLL。后续增加的显式 `context_right` 接口在真实连续配对路径不启用；旧 Frozen Single 直接调用冻结原融合层，当前封装复制并冻结同一层。

其他 seed42 同结构重跑只写入 `seed42_import/rerun_context.json`：原联合 Ordered 第84轮 97.874050%，固定注入重跑第66轮 97.879172%。它们仅作运行差异背景，**不参与本轮重新择高或三主干统计**。

## 独立原 C43、C44

复用 `run_ordered_transition_readout.load_source`。来源锁来自结果快照 `f0f4130e6ef141b8760dceaaac4158f10ccf5522`，进一步核对独立 N0、C 权重、split、样本清单、原 C 验证预测及其摘要。

| 主干 | 独立 N0 最佳轮 | 原 C 最佳轮 | 原 C 权重 SHA256 | 完整 C 状态摘要 |
| --- | ---: | ---: | --- | --- |
| 43 | 27 | 188 | 6f4d228917e2352b3684b4b1d79dcd0ec8fde02ebe95b7ca6ac6fe732b88bbba | 746ee8c5b2e4f3e461bbd61edc59204d55ab565c58a24803f65fbafa9c0ab45d |
| 44 | 19 | 167 | 418f26d5ef3db580cf347c3237049e2911731d8e9ca2b6431240bcd4a348d7ca | 813afa5e7d19ff35a050dd81a56a114b03e90421b71fb364cea60691ac9576d2 |

N0 权重分别锁定为 `e560fb61...e0443`、`76535b1b...3a1c91`，完整哈希和全部文件清单在配置中。复制 seed42 后改名无法通过状态及来源链检查。两个主干分别生成冻结特征缓存，缓存绑定来源、代码、协议、环境、图根目录与输入哈希，不能互换。

## 固定新增预算

依次运行 seed43 的三个组，再运行 seed44 的三个组，不根据 seed43 得分停止或更改 seed44。每组完整200轮，train107621、val13446，batch512、eval16，每轮211次，每组42200次；共 **6 次训练、253200 次新增更新**。base_C 与 seed42 导入均不计新增训练。

Adam betas=(0.9,0.999)、eps=1e-8、weight_decay=0、amsgrad=False；读出 lr=0.001，联合组融合层 lr=0.0001。最终八分类 NLL，无新损失或超参数搜索。三组在同一批次分别更新，共享显式采样器；每轮全验证集评估，最佳 Macro-F1 包含第0轮，同分取最早。保留30/100/150/200轮预算窗口。

探测使用独立参数副本，检查零输出后梯度逐步建立、联合投影/门控/分类头梯度、冻结模块及缓冲区不变。探测更新被丢弃，不计正式预算。原 C 的残留梯度通过现有 `freeze_c` 的 `zero_grad(set_to_none=True)` 清除。

## 原子恢复与完成核验

每次三个组共同完成一个批次后原子提交 `recovery.pt`：模型、Adam、Python/NumPy/Torch/CUDA RNG、采样器生成器及顺序/游标、轮次/累计更新、损失累计、最优选择，以及每组已提交优化计算耗时。中途某组失败后，只从上次三组共同提交的批次恢复，不重置 Adam，不计入部分更新。

`protocol_lock.json` 在训练前固定全局协议、所有历史/独立来源、代码提交和规范化源码哈希。恢复时还要求各主干来源、环境、图目录和缓存哈希匹配。完成的主干通过文件清单校验后直接跳过训练；不是再训练一轮。技术失败会停止队列并上传证据，修复后同命令续训；科学结果为负不会阻止 seed44。

每个主干完成后核验所有采样顺序及更新预算，完整验证集原图回放最佳权重，再从导出整数预测独立计算混淆矩阵、Precision/Recall/F1 与配对计数，检查编码器、跨层模块和 Frozen Single 融合层不变。测试图加载被禁止。

耗时分别报告本组已提交更新的同步前向/反向/Adam 计算秒及本次主干运行墙钟时间；不把失败回滚、缓存、验证或发布耗时分摊到各模型。旧归档没有每组计时时保留空值，不伪造历史耗时。

## 汇总口径

每个主干保存四模型的 Accuracy/Macro-F1/Weighted-F1、八类 P/R/F1 与混淆矩阵、最佳/末轮指标、参数量、更新预算与耗时，以及四项配对：Ordered−C、Ordered−Joint Single、Ordered−Frozen Single、Joint Single−Frozen Single。

总体同时给出三主干与新增43/44两主干的独立汇总；标准差采用样本标准差（n−1）。配对差先在每个主干内相减，再求均值和样本标准差。仅三个值均严格为正时写“方向一致”；负值、零差均明确列出。只优于 Joint Single 而未优于强 Frozen Single 时，明确报告强对照优势不足。不以最高 seed 代表方法，不作统计显著性声明。

## 当前验证与执行状态

本地 **57 项 CPU 检查通过**，覆盖本轮训练/来源/恢复/汇总/发布与现有实验回归。真实 seed42 归档导入中，四个锁定模型各 13,446 条预测完全一致，新增优化器更新为0。收据放在 `docs/fixed_transition_stability_v1/`。它们不代表服务器 CUDA 探测或六次正式训练完成。当前本机无 CUDA，已配置 GPU 服务器 SSH 跳板连接超时；正式训练状态以输出的 `status.json`、每主干收据及训练退出码为准。

## 服务器一次启动完整队列

使用已完成独立 N0/C 的服务器及既有 wx 环境。主工作区保留，代码放在独立工作树。先取结果快照供固定归档读取，再取代码。缺少的锁定 LFS 对象会由导入阶段按精确来源按需获取。

```bash
conda activate wx
cd '/home/nzxn/下载/WX/load/XG-NID' || exit 1
xgnid_repo="$PWD"
xgnid_code="$(dirname "$PWD")/XG-NID-fixed-transition-stability"
git fetch --no-tags origin refs/heads/codex/server-outputs || exit 1
git fetch --no-tags origin refs/heads/codex/fixed-transition-stability || exit 1
xgnid_revision="$(git rev-parse FETCH_HEAD)"
if [ -e "$xgnid_code" ]; then
  test "$(git -C "$xgnid_code" rev-parse HEAD)" = "$xgnid_revision" || exit 1
else
  GIT_LFS_SKIP_SMUDGE=1 git worktree add --detach "$xgnid_code" "$xgnid_revision" || exit 1
fi
git -C "$xgnid_code" diff --quiet || exit 1
git -C "$xgnid_code" diff --cached --quiet || exit 1
mkdir -p "$xgnid_repo/tmp" "$xgnid_repo/outputs/fixed_transition_stability_v1"
touch "$xgnid_repo/outputs/fixed_transition_stability_v1/training.log"
nohup python -u -B "$xgnid_code/scripts/launch_fixed_transition_stability.py" \
  --repo "$xgnid_repo" \
  > "$xgnid_repo/tmp/fixed_transition_stability_v1.launcher.log" 2>&1 < /dev/null &
echo $! > "$xgnid_repo/outputs/fixed_transition_stability_v1/launcher.pid"
tail -F "$xgnid_repo/outputs/fixed_transition_stability_v1/training.log"
```

如果实际图目录不同，首次启动添加 `--data-root '/实际图根目录'`，之后恢复保持一致。重新运行相同入口即可续训。入口锁在仓库 `tmp`，避免发布读取被锁住的输出文件。

查看队列、退出与发布状态：

```bash
tail -F "$xgnid_repo/tmp/fixed_transition_stability_v1.launcher.log"
cat "$xgnid_repo/outputs/fixed_transition_stability_v1/status.json"
cat "$xgnid_repo/outputs/fixed_transition_stability_v1/phase_exit_codes.json"
cat "$xgnid_repo/outputs/fixed_transition_stability_v1/training_exit.json"
cat "$xgnid_repo/outputs/fixed_transition_stability_v1/exit_codes.json"
```

根状态为 `import_completed` 或 `backbone_completed` 时，完整队列尚未结束；只有汇总完成后才有 `queue_completed: true`。

## 自动发布与独立重试

seed43 完成后发布一次，seed44 完成后发布一次，最终汇总后再次发布；正常结束或可捕获失败均有退出上传处理。复用 `sync_server_outputs.py` 和 LFS，包含所有权重、冻结缓存、恢复状态、预测、账本、报告、运行日志和失败收据。发布日志关闭后再发布日志/退出收据，训练退出码独立保留。上传失败明确提示重试，结果不被删除。`publication_history` 和 `backbone_uploads.json` 跨重启保留各主干的上传失败记录，上传失败本身不会跳过后续主干。

进程结束后独立重传，不增加训练：

```bash
xgnid_repo='/home/nzxn/下载/WX/load/XG-NID'
xgnid_code="$(dirname "$xgnid_repo")/XG-NID-fixed-transition-stability"
python -B "$xgnid_code/scripts/sync_server_outputs.py" publish --repo "$xgnid_repo" --only fixed_transition_stability_v1
```

只需验证归档导入时可给启动入口加 `--import-only`；它不会进入43/44训练。默认不加此参数，直接完成完整队列。本轮结束于验证集稳定性汇总，不自动开始新结构或测试集评估。
