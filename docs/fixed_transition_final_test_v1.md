# 固定注入报文转移模型：检查点锁定与固定测试

本轮只评价固定模型：新增训练0次，反向传播0次，优化器更新0次。候选权重来自不可变结果提交 `5f3975aeecd41c8c1150b97f88813b7ea703e54d`，对应训练代码 `7c9249eafcea21ff40fe1811afcc01414e00acfb`。

## 事前锁定

| 模型 | seed42开发锁定最佳轮 | seed43最佳轮 | seed44最佳轮 |
| --- | ---: | ---: | ---: |
| base_C | 48 | 188 | 167 |
| frozen_single | 94 | 50 | 150 |
| joint_single | 59 | 25 | 131 |
| joint_ordered | 191 | 97 | 123 |

`configs/fixed_transition_final_test_v1.json` 预先固定12个权重文件的路径、字节数、SHA256、验证Macro-F1、配置来源和标签顺序。原C不能因历史同名标签而替换。seed42使用指定开发归档；读出和采样seed均为42，准确称为三个独立主干上的配对评价。

`lock` 阶段不访问原图，核对全部已完成来源、输入哈希、12组验证预测、9份完整账本及最早同分规则，验证三个独立C状态，并验证历史C测试归档的权重与状态一致性。将训练协议、标签映射和权重来源写入 `checkpoint_lock.json`，将当前代码与计算文件哈希写入 `protocol_lock.json`，复制12个精确权重至本轮 `locked_checkpoints/`。正常队列在首次测试前发布锁文件。

验证最佳模型及固定测试期间，所有参数冻结、所有模块保持eval。保留固定 `packet + residual`，不实例化β门控，不更改读出与32维融合门控。

## 固定回放与测试

按 `lock → seed42 → seed43 → seed44 → aggregate` 顺序执行。每个主干先在13,446条验证流上重现四个锁定模型的逐样本预测，然后在相同13,448条测试流上评估四组。

同一主干的验证/测试特征分别仅由该主干的冻结模块提取，四模型共用该份特征。测试特征使用明确的 `TestGraphs` 数据集，不放宽历史训练加载器的train/val隔离。逐批核对样本ID、标签和顺序；数据清单及原图指纹继续使用现有校验。测试先逐样本验证原C与同权重历史预测完全一致，再导出其他模型结果。

每个缓存保存来源、环境、图路径、清单、样本顺序及独立SHA256收据。匹配才复用；缓存与收据之间发生中断时重新提取。已完成主干经产物哈希验证后跳过，部分预测失败可复用缓存续跑，不训练。配置、代码、来源或环境不匹配明确报错，保留原结果。

`--probe-only` 只运行三个主干的验证回放，不加载测试图。默认入口直接完成全部固定测试，不需要先独立执行probe。

## 输出与论文表

输出目录为 `outputs/fixed_transition_final_test_v1/`：

- 根目录：配置与代码锁、12模型锁定权重、状态、日志、退出码、上传收据、最终汇总。
- `seed42/seed43/seed44`：来源、四组验证回放、历史原C测试逐样本对照、冻结特征及校验收据、四组测试预测、指标、八类P/R/F1、混淆矩阵、四项配对修复/退步及全部变化样本。
- `aggregate`：三主干与新增43/44的Accuracy、Macro-F1、Weighted-F1、八类结果的均值和样本标准差；先在同一主干内相减的四项配对差；中文报告及论文实验表Markdown/CSV。

每个主干结束后从导出整数预测独立复算指标与配对计数。最终汇总前再次检查各主干封存哈希、相同测试样本、预测摘要和独立复算。只在12组完整测试完成后生成 `paper_tables.md`；不向有未提交修改的论文草稿自动写入数字。

报告原样保留负值和类别退步，重点列出WebBased、Recon、Spoofing、BruteForce。只有三个主干配对差均正才写方向一致，不作统计显著性声明。98.013342%是seed43验证结果，不是测试分数。

论文表与报告明确披露：**该测试划分已有历史评估使用。本轮在锁定全部验证最佳检查点后进行固定评价，不据测试结果挑检查点、调参或改结构。** 仍保留整体方法收益、Frozen Single强对照、类别取舍，以及True/Shuffle和β门控的既有失败消融。失败消融属于此前seed42验证证据，本轮不追加其测试。

## 服务器启动

在原GPU服务器的交互终端运行。现有主仓库与历史实验工作树均保留，新代码使用独立服务器工作树。先激活原wx环境（Python3.9.23、Torch2.3.1+cu118、PyG2.5.3、CUDA）；脚本按既有协议禁止CPU替代正式原图评价。

```bash
conda activate wx
xgnid_repo='/home/nzxn/下载/WX/load/XG-NID'
xgnid_code="$(dirname "$xgnid_repo")/XG-NID-fixed-transition-final-test"
cd "$xgnid_repo" || exit 1
git fetch --no-tags origin codex/fixed-transition-final-test || exit 1
xgnid_revision="$(git rev-parse origin/codex/fixed-transition-final-test)"

if [ -d "$xgnid_code" ]; then
  [ "$(git -C "$xgnid_code" rev-parse HEAD)" = "$xgnid_revision" ] || {
    echo 'Existing fixed-test worktree has another revision; preserve it and check its output lock.'
    exit 1
  }
else
  git worktree add --detach "$xgnid_code" "$xgnid_revision" || exit 1
fi

mkdir -p "$xgnid_repo/tmp" "$xgnid_repo/outputs/fixed_transition_final_test_v1"
touch "$xgnid_repo/outputs/fixed_transition_final_test_v1/evaluation.log"
nohup python -u -B "$xgnid_code/scripts/launch_fixed_transition_final_test.py" \
  --repo "$xgnid_repo" \
  > "$xgnid_repo/tmp/fixed_transition_final_test_v1.launcher.log" 2>&1 < /dev/null &
echo $! > "$xgnid_repo/outputs/fixed_transition_final_test_v1/launcher.pid"
tail -F "$xgnid_repo/outputs/fixed_transition_final_test_v1/evaluation.log"
```

如原图位置已变，给启动命令加 `--data-root '/实际原图目录'`；仍按固定清单验证，不修改split。已有split与samples.jsonl需保留在原主仓库的数据目录；GitHub结果分支不包含原始图或该数据目录。启动器在首次归档读取前复用 `sync_server_outputs.lfs_environment` 初始化仓库私有LFS，自动取得注册的Git/LFS结果输入。

查看自动上传日志：

```bash
tail -F "$xgnid_repo/tmp/fixed_transition_final_test_v1.launcher.log"
```

锁定、各主干完成、正常结束及可捕获失败均执行选择性发布，包含权重、缓存、预测、报告、日志、退出码和失败证据。保留评价退出码，上传失败明确提示。独立重试：

```bash
python -B "$xgnid_code/scripts/sync_server_outputs.py" publish \
  --repo "$xgnid_repo" --only fixed_transition_final_test_v1
```

最终应检查 `status.json` 的 `queue_completed=true`、`final_verification.json` 的12模型通过记录、`evaluation_exit.json` 的零退出码，以及 `exit_codes.json` 的评价/主上传退出状态。`checkpoint_lock_completed` 或 `server_probe_passed` 均不代表固定测试完成。
