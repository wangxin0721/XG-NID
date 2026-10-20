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
- 先用 NVIDIA 官方查询接口确认指定物理 GPU UUID 及其计算进程，忙碌或查询失败时拒绝运行，不终止任何进程；通过 UUID 设置 CUDA_VISIBLE_DEVICES，程序内部使用 cuda:0。查询是时点观察，不是 GPU 资源预留，见 [NVIDIA SMI 文档](https://docs.nvidia.com/deploy/nvidia-smi/index.html)。
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

**仅对空闲 GPU 执行。**下面以物理 GPU1 为例，不宣称它当前空闲；若需要 GPU0，显式改为 `--physical-gpu 0`。不要为了新对照中断现有 E3 或另一张卡的任务。探测前台执行，先做 256；通过后再做 512，均不自动串联正式训练。

```bash
python -B -u /tmp/xgnid_e2_batch.py --repo . --config /tmp/xgnid_e2_batch.json --mode probe --batch-size 256 --physical-gpu 1 --output outputs/e2_batch_probe_b256_s42_v1
```

先查看输出和 `outputs/e2_batch_probe_b256_s42_v1/summary.json`，通过后执行：

```bash
python -B -u /tmp/xgnid_e2_batch.py --repo . --config /tmp/xgnid_e2_batch.json --mode probe --batch-size 512 --physical-gpu 1 --output outputs/e2_batch_probe_b512_s42_v1
```

探测成功的 summary 必须为 `status=completed`、`exit_code=0`、`mode=probe`、`optimizer_update_verified=true`、`test=null`、warmup_updates=1、measured_updates=3。报告同时包含初始状态、代码/配置哈希和 GPU UUID。先回传容量探测结果再判定正式执行条件，不能将待执行探测标记为通过。

## 正式训练命令（仅在相应探测实际通过后）

以下示例仍为 GPU1，须与相应探测 UUID 相同。每项只启动一次，目录和日志不存在才执行；原有产物不覆盖。**不要同时提交两项到同一张卡**；先完成 256，再运行 512。入口会再次拒绝已有计算进程；没有后台自动队列。

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

服务器容量探测、训练/评价时的显存和吞吐量、同初始化哈希的实际比对、正式训练及性能结论全部待真实执行。新入口与冻结循环分开实现，尚未用服务器小规模轨迹验证；先以真实探测检查运行路径，不能把静态通过当作训练正确性已完全证明。原 E2 batch16 测试为 Accuracy 96.9438%、Macro-F1 95.0161%、Weighted-F1 96.8974%，用作已登记对照，不用新测试结果反复调学习率或批大小。
