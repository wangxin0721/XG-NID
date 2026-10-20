# E2 训练 batch 256 / 512 独立对照交付

用户于 2026-10-20 要求增加 batch 256 / 512，并明确选择“先做 E2 的 256、512；当前 E3 保持 batch=16”。本次实际增加独立配置与服务器入口，不修改 E0～E3/R0 的冻结代码，不重启、暂停或删除已有任务/产物，不重新制定 v3。

## 比较定义

| 项目 | 已完成的 E2 | 新 E2-B256 | 新 E2-B512 |
|---|---:|---:|---:|
| 训练 batch | 16 | 256 | 512 |
| 每个完整 epoch 参数更新次数 | 6727 | 421 | 211 |
| 初始化 batch / 评价 batch | 16 / 16 | 16 / 16 | 16 / 16 |
| Seed / Adam lr | 42 / 0.001 | 42 / 0.001 | 42 / 0.001 |
| Epoch 上限 / 早停 patience | 100 / 12 | 100 / 12 | 100 / 12 |

其余共用现有 E2 双编码器、等权 0.5/0.5 融合、846800 参数、最终普通 NLL、相同分组 split 与 train/val/test=107621/13446/13448。无自动学习率缩放、scheduler、类别加权、辅助损失、梯度累积、AMP 或有限梯度裁剪。按验证 Macro-F1 严格改善更新 best，重载 best 后测试一次；不依据测试结果选择 batch。每个候选保留完整结果，不只挑分数较高的一项报告。

这是固定 epoch 上限与早停规则的 batch 敏感性比较，不是相同优化器更新次数的比较；早停后实际 epoch、更新次数、样本曝光量和运行时间都单列。更大的 batch 不保证更快收敛或更高分，不据此承诺 16/32 倍加速。PyG 将多张图的节点与边合并处理，容量取决于 batch 内节点/边总数，见 [PyG 2.5.3 批处理说明](https://pytorch-geometric.readthedocs.io/en/2.5.3/advanced/batching.html)。

## 代码与证据约束

- 配置：`configs/e2_batch_comparison.json`。
- 独立入口：`scripts/run_e2_batch_comparison.py`。复用冻结入口中的 loader、sampler、初始化、评价、检查点保存和指标公式；训练循环单独实现，未改写冻结文件。
- 原入口会拒绝与 E0 预算控制不一致的 batch。本入口先以原有 batch16 的真实 E2 记录核验原 smoke/pilot/E0 依赖，再显式登记新 batch 对照协议，不修改或伪造原报告让它通过。
- 所有候选均使用原 16 图初始化批次；初始 state SHA 必须与既有 E2、GPU smoke 一致，否则直接失败。大 batch 容量探测得到的训练后状态不用于正式训练。
- 读取现有图，不生成新图、不改变划分；逐图哈希核验仍由原 FrozenGraphs 完成。探测只加载训练图，不读取验证/测试图张量。
- 先用 NVIDIA 官方查询接口确认指定物理 GPU UUID、已用/可用显存及其计算进程，默认忙碌或查询失败时拒绝运行；用户显式指定 `--allow-shared-gpu` 时允许共享并记录现有进程，不终止任何进程。通过 UUID 设置 CUDA_VISIBLE_DEVICES，程序内部使用 cuda:0。查询是时点观察，不是 GPU 资源预留，见 [NVIDIA SMI 文档](https://docs.nvidia.com/deploy/nvidia-smi/index.html)。
- 每个容量探测使用首个 seed42 训练顺序的三个完整 batch：第一批先做一次预热，再对三批分别做前向、NLL、反向及 Adam 更新；验证参数确有更新、梯度有限、等权仍为 0.5。记录节点/边数、已分配/保留显存峰值、驻留 GPU 的单步耗时。该耗时排除图读取和传输，不冒充完整 epoch 吞吐量。
- 探测不是全图最坏尺寸扫描，不能保证后续永不 OOM。OOM/非有限数值保留失败证据，无自动降 batch、重试或恢复；失败探测不能放行正式训练。
- 正式训练必须提供同一 batch、配置、入口/冻结源码、数据清单、基线报告、运行环境和 GPU UUID 的成功探测。源码或 GPU 改变后重新探测，不绕过绑定。

## 服务器交付与容量探测

本地仅做静态验证，GPU 执行全部在服务器。以下从本分支取新增文件到 `/tmp`，不 pull 或切换现有训练工作树。首次使用取回后续汇报的实际交付提交；长期保留时应把下面分支 ref 改成该提交 SHA，避免将未来版本混在同一对照中。

```bash
. /home/nzxn/miniconda3/etc/profile.d/conda.sh && conda activate wx
cd /home/nzxn/下载/WX/load/XG-NID
git fetch origin codex/r0-edge-preparation
git show origin/codex/r0-edge-preparation:scripts/run_e2_batch_comparison.py > /tmp/xgnid_e2_batch.py
git show origin/codex/r0-edge-preparation:configs/e2_batch_comparison.json > /tmp/xgnid_e2_batch.json
nvidia-smi
```

**默认仅对空闲 GPU 执行；有意共享时使用文末的显式选项。**下面以物理 GPU1 为例，不宣称它当前空闲；若需要 GPU0，显式改为 `--physical-gpu 0`。不要为了新对照中断现有 E3 或另一张卡的任务。探测前台执行，先做 256；通过后再做 512，均不自动串联正式训练。

```bash
python -B -u /tmp/xgnid_e2_batch.py --repo . --config /tmp/xgnid_e2_batch.json --mode probe --batch-size 256 --physical-gpu 1 --output outputs/e2_batch_probe_b256_s42_v1
```

先查看输出和 `outputs/e2_batch_probe_b256_s42_v1/summary.json`，通过后执行：

```bash
python -B -u /tmp/xgnid_e2_batch.py --repo . --config /tmp/xgnid_e2_batch.json --mode probe --batch-size 512 --physical-gpu 1 --output outputs/e2_batch_probe_b512_s42_v1
```

探测成功的 summary 必须为 `status=completed`、`exit_code=0`、`mode=probe`、`optimizer_update_verified=true`、`test=null`、warmup_updates=1、measured_updates=3。报告同时包含初始状态、代码/配置哈希和 GPU UUID。先回传容量探测结果再判定正式执行条件，不能将待执行探测标记为通过。

## 正式训练命令（仅在相应探测实际通过后）

以下示例仍为 GPU1，须与相应探测 UUID 相同。每项只启动一次，目录和日志不存在才执行；原有产物不覆盖。**不要同时提交 B256 和 B512 到同一张卡**；先完成 256，再运行 512。默认入口再次拒绝已有计算进程；显式共享另见下节，没有后台自动队列。

```bash
if [ -e outputs/e2_batch_full_b256_s42_v1 ] || [ -e outputs/e2_batch_full_b256_s42_v1.log ]; then
    echo "B256产物已存在，核查状态，不重复提交。"
else
    nohup python -B -u /tmp/xgnid_e2_batch.py --repo . --config /tmp/xgnid_e2_batch.json --mode full --batch-size 256 --physical-gpu 1 --probe-report outputs/e2_batch_probe_b256_s42_v1/summary.json --output outputs/e2_batch_full_b256_s42_v1 >> outputs/e2_batch_full_b256_s42_v1.log 2>&1 < /dev/null &
    echo "$!" > outputs/e2_batch_full_b256_s42_v1.pid
    echo "B256请求已提交，检查日志确认实际训练。"
fi
```

```bash
if [ -e outputs/e2_batch_full_b512_s42_v1 ] || [ -e outputs/e2_batch_full_b512_s42_v1.log ]; then
    echo "B512产物已存在，核查状态，不重复提交。"
else
    nohup python -B -u /tmp/xgnid_e2_batch.py --repo . --config /tmp/xgnid_e2_batch.json --mode full --batch-size 512 --physical-gpu 1 --probe-report outputs/e2_batch_probe_b512_s42_v1/summary.json --output outputs/e2_batch_full_b512_s42_v1 >> outputs/e2_batch_full_b512_s42_v1.log 2>&1 < /dev/null &
    echo "$!" > outputs/e2_batch_full_b512_s42_v1.pid
    echo "B512请求已提交，检查日志确认实际训练。"
fi
```

以 B256 为例，查看日志用 `tail -n 30 -F outputs/e2_batch_full_b256_s42_v1.log`；退出查看按 Ctrl+C。完成检查读取根 `status.json`、`summary.json` 和 `E2/metrics.json`，不要由 PID 或日志存在推断训练成功。B512 对应替换目录名中的 256。

## 实际验证状态

本地 Python 3.9 AST 语法、复用的冻结模块接口名称、JSON 配置与原冻结控制的一致性、batch 更新次数算术、探测/正式模式约束静态检查均通过；冻结文件 Git 差异为空。没有执行新入口，也没有运行模型或合成/真实数据实验。

首次交付时，服务器容量探测、训练/评价时的显存和吞吐量、同初始化哈希的实际比对、正式训练及性能结论全部待真实执行；后续 B256 共享探测的真实回传见文末。新入口与冻结循环分开实现，尚未用服务器完整小规模 epoch 轨迹验证，不能把静态通过或几步探测当作正式训练正确性已完全证明。原 E2 batch16 测试为 Accuracy 96.9438%、Macro-F1 95.0161%、Weighted-F1 96.8974%，用作已登记对照，不用新测试结果反复调学习率或批大小。

## 用户追加：E2-B256 与 E3 共用 GPU0

2026-10-20 用户询问是否可以让 E2-B256 和 E3 同时在 GPU0 运行。技术上可以让多个 CUDA 进程使用同一张卡；此前默认拒绝是入口的资源/计时保护策略，不是硬件只能运行一个进程。见 [NVIDIA 多进程架构说明](https://docs.nvidia.com/deploy/mps/latest/architecture.html)。本次只放开显式选项，不启用 MPS、不重启驱动、不修改现有 E3。

新选项 `--allow-shared-gpu` 同时适用于 probe / full；两个阶段的绑定都记录该选项、同一 GPU UUID、代码和配置哈希，不能混用旧版本或不同策略的探测。硬件记录包含启动时的其他计算进程、已用/可用显存；每个探测 batch 和正式 epoch 末另记录其他进程快照，快照查询失败会明确记录 unknown，不影响训练本身。

共享时两个模型/优化器仍属于各自独立进程，分别保存结果；但显存与计算资源共同使用。探测通过仅说明抽样 batch 可以运行，不保证未来峰值不冲突。共享执行的墙钟与单步耗时标记为共享环境，不用来独立证明 batch 提速。若要比较纯性能开销，另选无其他计算进程的时段测量；不据现有单次 GPU-Util 或显存截图保证容量。

获取新提交中的入口和配置后，在 GPU0 做 B256 的短探测，E3 保持原样运行：

```bash
python -B -u /tmp/xgnid_e2_batch.py --repo . --config /tmp/xgnid_e2_batch.json --mode probe --batch-size 256 --physical-gpu 0 --allow-shared-gpu --output outputs/e2_batch_probe_b256_s42_shared_v1
```

这份探测真实通过后，再启动相应正式任务：

```bash
if [ -e outputs/e2_batch_full_b256_s42_shared_v1 ] || [ -e outputs/e2_batch_full_b256_s42_shared_v1.log ]; then
    echo "共享B256产物已存在，核查状态，不重复提交。"
else
    nohup python -B -u /tmp/xgnid_e2_batch.py --repo . --config /tmp/xgnid_e2_batch.json --mode full --batch-size 256 --physical-gpu 0 --allow-shared-gpu --probe-report outputs/e2_batch_probe_b256_s42_shared_v1/summary.json --output outputs/e2_batch_full_b256_s42_shared_v1 >> outputs/e2_batch_full_b256_s42_shared_v1.log 2>&1 < /dev/null &
    echo "$!" > outputs/e2_batch_full_b256_s42_shared_v1.pid
    echo "共享B256请求已提交，检查日志确认实际训练。"
fi
```

共享选项初次交付时没有真实执行共享探测或启动共享训练；后续回传见下一节。B512 继续保留既有候选，没有新增同时并发三项任务的安排。

## B256 共享探测真实通过回传（2026-10-20）

来源为用户截图 `codex-clipboard-e7717fd5-dd5b-4ca9-99a8-190847504423.png`。服务器 fetch 到 `db232e4`，入口输出物理 GPU0 / Tesla T4 / sharing_allowed=true。起始查询已用显存 204 MiB、可用 14708 MiB；检测到 PID 268759（此前 ps 已确认为 E3）。这些是当时快照，不据此保证后续容量或推断训练速度。

三个实测完整训练 batch 的回传：

| batch | 图数 | Packet 节点数 | Link 边数 | 驻留 GPU 单步秒数 | PyTorch 已分配峰值（字节） | 保留峰值（字节） |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 256 | 2659 | 4806 | 约 0.0334 | 79360000 | 96468992 |
| 2 | 256 | 2961 | 5410 | 约 0.0401 | 84285952 | 125829120 |
| 3 | 256 | 2781 | 5050 | 约 0.0410 | 81096704 | 125829120 |

峰值已分配约 80.4 MiB，保留峰值 120 MiB；它们是本进程 PyTorch 分配器统计，不是整卡或整个进程的全部显存。单步时间排除读图和传输，且在共享环境测得，不外推 epoch 耗时或相对 batch16 的加速倍数。探测损失约 2.0378、2.0314、2.0242，是一次预热与三次临时更新的结果，不是正式性能评价，不用其选择参数。

三个 batch 的其他进程快照都仍包含 PID 268759。只能确认该进程在查询时仍存在，不能据进程存在替代 E3 的轮次/完成状态验证。本入口未执行终止或暂停其他任务的操作。

入口最终打印 `status=completed`、`mode=probe`、`batch_size=256`、`physical_gpu=0`、`test_evaluated=false`，输出路径为服务器仓库下 `outputs/e2_batch_probe_b256_s42_shared_v1`。按 `db232e4` 执行路径，只有相同初始状态核验、三次完整 batch 前后向、有限梯度、0.5 等权及参数确有更新均通过后，才保存成功报告并打印此状态。本次截图未回传完整 summary.json 的逐项哈希；正式入口将再次读取该实际报告并核对完整绑定。

B256 探测回传时已满足本次抽样容量条件，但当时尚无正式训练证据。后续正式任务已回传 PID、epoch、完成状态及完整指标，见下一节。B512 的服务器探测与训练仍待执行，不自动增加第三个同卡任务。

## B256 共享正式训练最终回传（2026-10-20）

来源：用户完整粘贴 `outputs/e2_batch_full_b256_s42_shared_v1/E2/metrics.json`，附件 `d84a627f-e24e-4e57-8b89-116cd875c6b4/已粘贴的文本.txt`。前一张截图 `codex-clipboard-a6677135-b28f-4f2f-98bf-894250a68050.png` 已显示入口最终 `status=completed`、`mode=full`、`batch_size=256`、`physical_gpu=0`、`test_evaluated=true`。本轮只解析回传数据、核对算术和归档，不重新执行模型或测试推断。

最佳轮 37、停止轮 49，差值 12 符合早停规则。最终测试来自最佳验证检查点，不是直接用第 49 轮 last 权重。模型参数 846800，训练 batch256、评价 batch16，仍为等权融合。验证/测试两路权重 mean/min/max 均为 [0.5,0.5]，std=[0,0]。

| 指标 | B256 最佳验证（13446 图） | B256 最终测试（13448 图） |
|---|---:|---:|
| Accuracy | 97.2036% | 97.2933% |
| Macro-F1 | 95.7038% | 95.6656% |
| Weighted-F1 | 97.1960% | 97.2745% |

测试原始值：accuracy `0.9729327781082688`，macro_f1 `0.9566563961403298`，weighted_f1 `0.9727454187871081`。

### 已登记对照

| 实验 | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| E0（已有台账） | 96.8248% | 95.2321% | 96.8235% |
| E1（已有完整回传） | 96.6761% | 94.8672% | 96.6316% |
| E2-B16（已有完整回传） | 96.9438% | 95.0161% | 96.8974% |
| E2-B256（本次） | 97.2933% | 95.6656% | 97.2745% |

B256 相对 B16 的三项变化分别为 +0.3495、+0.6495、+0.3771 个百分点；相对 E0 的三项变化约 +0.4685、+0.4335、+0.4510 个百分点，E0 差值使用台账四位小数百分数计算。两份 E2 metrics 的 initial_state_sha256 完全相同，均为 `0270349d613f11b453b92e4e928a3c2e7b67d062b5e3ba73c8ba4fb58dbd521a`。

在本 seed42 单次对照中，B256 三项测试指标均高于 B16，且高于既有 E0 台账。不能据此断言 B256 对所有种子最优、统计显著或已经证明双分支独立结构贡献；B512 仍未完成。B256 与 E0 同时有架构及 batch 差异；也不能将 E2-B256 与 E3-B16 的差直接归因于门控。原 E2-B16/E3-B16 配对结果独立保留，若后续需要新 batch 下的门控结论，应另有匹配 batch 的受控实验，当前不自动增加 E3 新任务。

### 测试类别结果与混淆矩阵

| 类别 | F1 | Support |
|---|---:|---:|
| Benign | 96.5702% | 2389 |
| WebBased | 81.5516% | 544 |
| Spoofing | 95.1411% | 1739 |
| Recon | 96.3044% | 2278 |
| Mirai | 100.0000% | 2400 |
| Dos | 99.7628% | 1475 |
| DDos | 99.8743% | 2390 |
| BruteForce | 96.1207% | 233 |

WebBased F1 从 B16 的 78.6275% 升至 81.5516%，增加约 2.9241 个百分点；Recall 从 73.7132% 升至 79.2279%，正确数 401→431，误判为 Recon 的数量 108→76。但 Recon→WebBased 从 34 增至 47，WebBased precision 从 84.2437% 小降为 84.0156%；不能写成所有错误类型均改善，也不根据这些测试反馈搜索参数。

混淆矩阵（行真实、列预测；顺序 Benign / WebBased / Spoofing / Recon / Mirai / Dos / DDos / BruteForce）：

```text
2337    7   39    6    0    0    0    0
  17  431   15   76    0    1    0    4
  78   18 1635    5    0    0    0    3
  18   47    7 2202    0    3    0    1
   0    0    0    0 2400    0    0    0
   0    0    0    3    0 1472    0    0
   0    3    0    3    0    0 2384    0
   1    7    2    0    0    0    0  223
```

### 成本、哈希与证据范围

总更新次数 20629=49×421，训练样本曝光量 5273429=49×107621。B16 实际运行 60 轮、每轮 6727 次更新，因此这不是相同更新次数或相同最终 epoch 的比较。

B256 总墙钟 `10650.379184391` 秒，约 2 小时 57 分 30 秒；训练/验证/检查点阶段 `10562.985185094003` 秒；最终测试含图 IO `87.2675000619929` 秒、模型调用 CUDA event 累计 `26.714854301452636` 秒。B16 的总墙钟约 6 小时 40 分 23 秒。由于实际轮数和共享负载不同，不能将总时间比当作 batch 的纯加速比；共享期间的 E3 耗时也可能受影响。

本次回传配置 SHA 为 `1479f5c4057fd5b11d5f77d01073389d19f5571b0a773ade9ebe4c1d03673e2e`；best.pt SHA 为 `58b5950cd7a07832ac87d34bee327b862c9e5d4ee4e1bf29a625a84d0f1b59e8`，last.pt SHA 为 `96901b85c8449ed64a1ef447bf028ab96523ab70101fcbaa5861c3b710b6c7fb`。这些哈希由服务器 metrics 返回，本次未独立下载权重/配置再计算；完整 summary/binding、预测 CSV 和逐轮日志尚未在本次完整取回。

已在本地用标准库核对验证/测试混淆矩阵的总数、Accuracy、每类 F1、Macro/Weighted-F1，均与回传值一致（误差 <1e-12）；并核对两份 E2 初始哈希相同、B256 更新次数和样本曝光量。这是回传数据算术核对，没有运行训练或新的测试集推断。

下一步按预先登记配置推进 B512，先在所选 GPU 上做对应容量探测，再用该报告启动独立训练。维持 lr=0.001、seed42、评价 batch16 和既定选模规则，不因本次测试分数新增 batch 或学习率候选。B256 结果完整保留，不能覆盖冻结 E2-B16，也不能据此推断 E3 当前运行/完成状态。
