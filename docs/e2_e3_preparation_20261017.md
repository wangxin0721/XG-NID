# E2/E3 配置与依赖交付（2026-10-17）

本次新增独立配置和服务器只读核对工具。保持冻结的 `train_grouped.py`、模型、分组划分、E0/E1 产物与正在运行的 R0 不变；未在本地运行模型、训练、冒烟或数据实验。服务器核对尚待真实执行，本文不将其标记为通过。

## 冻结设置

| 项目 | E2 / E3 共用设置 |
|---|---|
| 数据 | 现有 Flow82 / Packet1500 图；Contain/Reverse/Link 属性 4/4/1；原拓扑 |
| 划分 | `ciciot2023_grouped_input_v1`；train 107621 / val 13446 / test 13448 |
| Split ID | `a8ae9c934eec9fee970cc725e60051d0c709e0f16c7dda14749350ff2006975a` |
| 优化 | seed 42；Adam；lr 0.001；默认 weight_decay 0；batch 16；hidden 64 |
| 预算 | 上限 100 epoch；验证 Macro-F1 严格改善才更新 best；patience 12；无 scheduler |
| 目标 | 最终分类普通 NLL；无辅助损失、类别权重、校准或 hard negatives |
| 评价 | 重载最佳验证权重后，固定测试集评价一次 |
| 执行 | cuda:0；workers 2；不自动重试或恢复；每项独立结果目录 |
| 环境 | Python 3.9.23 / torch 2.3.1+cu118 / PyG 2.5.3；沿用服务器 wx |

E2/E3 各有两套独立的完整异构图编码器，分别池化 Flow / Packet，再投影到 8 维中间表示，融合后经 shared_head 分类。E2 权重恒为 0.5；E3 使用原有可微门控和熵补量混合。每个模型总参数 846800，包含休眠或不参与主损失的参数，不代表全部参数都有效更新。8 维中间表示无独立分支分类监督，不称为已经校准的分支分类概率。详细语义见 `e3_mechanism_audit_20261017.md`。

新增 `configs/e2_e3_frozen.json` 供准备工具读取并生成 CLI 参数预览。冻结训练入口仍用其原有 CLI，不新增 `--config`，也不修改训练实现。配置中的 `model_semantics` 是语义说明，训练实际代码由历史报告的源码 SHA 核验。

## 启动依赖核对

`scripts/prepare_e2_e3_server.py` 仅使用 Python 标准库，不导入 torch/PyG/sklearn，不反序列化 `.pt`，不创建 CUDA 上下文，不查询或占用 GPU，不启动训练，不安装依赖。它只新建一个 `outputs/e2_e3_preflight_<UTC>/report.json`，不改写既有文件。

核对范围：

- 重算 sample manifest SHA 和 Split ID；逐行检查身份、分区、标签、顺序、分组不交叉及类别数量；仅检查 32 个训练候选图路径是否存在，不读取图张量或再次扫描所有图文件。
- 检查四模型已通过的 GPU 冒烟报告、配对初始权重哈希、优化器更新、输入不变断言、E2/E3 门控梯度与参数量；核对报告记录的实际源码字节 SHA。
- 检查 E0/E1 预跑及完整训练的 summary、config、metrics、status 一致性、训练源码哈希、预算、初始状态、样本清单、epoch 完整性及验证选模记录。
- 检查已有 best/last 权重、epoch 日志和预测文件存在并记录当前 SHA；E0/E1 预测 CSV 逐行核对 sample_id、label、顺序和数量，不重新推断或计算新测试指标。
- 从安装包元数据核对 Python/torch/PyG 版本，记录 NumPy/sklearn 版本。旧训练配置未冻结后两者版本，因此只能记录当前值，不能声称历史版本一致。
- 检查预定 E2/E3 输出目录和对应日志尚未使用；只读取 R0 启动记录与状态快照，不暂停或终止它。

源码核验以服务器历史报告中的原始文件 SHA 为准。若不符，报告失败和具体文件；不自动覆盖代码，不忽略哈希，也不替换依赖。新增准备工具/配置/文档不在原模型冻结源码清单内。

本工具不验证权重内部内容；历史没有独立权重 SHA 基准时，记录当前 SHA 只是建立本次完整性清单，不能证明此前从未改动。图内容/维度仍由既有冒烟证据与冻结训练 loader 的逐图校验保证，不能把本次元数据通过称为新的模型测试通过。包元数据一致也不等于二进制导入或实时 GPU 可用性已验证。

## R0 运行期间：只执行准备检查

在服务器仓库根目录、新终端执行以下命令。只 fetch 和提取两个新增文件到 `/tmp`，不切换现有分支、不 pull、不修改 R0 所在工作树。`nice` 降低本次 CPU 检查优先级，全部结果仍需服务器回传确认。

```bash
. /home/nzxn/miniconda3/etc/profile.d/conda.sh && conda activate wx
cd /home/nzxn/下载/WX/load/XG-NID
git fetch origin codex/r0-edge-preparation
git show origin/codex/r0-edge-preparation:scripts/prepare_e2_e3_server.py > /tmp/xgnid_prepare_e2_e3.py
git show origin/codex/r0-edge-preparation:configs/e2_e3_frozen.json > /tmp/xgnid_e2_e3_frozen.json
nice -n 10 /home/nzxn/miniconda3/envs/wx/bin/python /tmp/xgnid_prepare_e2_e3.py --repo . --config /tmp/xgnid_e2_e3_frozen.json
```

工具只打印简短状态和实际报告路径。成功状态为 `metadata_checks_passed`，失败为 `blocked`，对应退出码 0 / 1。任一失败均不生成训练命令预览。报告中的 `training_started`、`gpu_touched`、`test_inference_performed` 必须为 false。

读取终端刚打印的具体 `report.json` 路径并回传报告。不要凭目录名挑“最新最好的”结果，不需要发送权重或大型数据。若需要重跑准备检查，会新建另一份独立报告，不覆盖之前证据。

## R0 完成后的启动条件

本次不启动 E2/E3，也不设置自动队列。R0 保存正式结果并正常结束后，确认真实训练进程已退出、GPU 可用，再重新核对依赖并使用报告里的 E2 参数启动。该预览是前台命令，**不是后台 supervisor 或防重复启动工具**；正式启动时需按服务器既有后台管理方式记录日志、PID 和退出码。不要把前台预览直接当作已经启动或能够抗断线的任务。

E2 输出固定为 `outputs/grouped_full_E2_s42_v1`；E3 输出为 `outputs/grouped_full_E3_s42_v1`。若目录已有内容，先核查真实状态，不删除后重跑，不覆盖历史结果。

E2 必须读取原 `outputs/grouped_full_E0_s42_v1/summary.json` 作为基线。E2 完成并核对后，单独启动 E3；其 `--fusion-control-report` 必须指向真实完成的 `outputs/grouped_full_E2_s42_v1/summary.json`。冻结入口会再次核验 E2 的完成状态、split、源码和训练预算；E2 尚未完成时拒绝 E3。报告中的 E3 命令仅是未来参数预览，不表示 E3 依赖已经满足。

R0 是另一个协议的 Edge 重实现，E2/E3 使用分组基准；两类结果分别报告，不用 R0 的测试集或得分选择 E2/E3 参数。

## 本次实际验证与待验证

- 本地：Python 3.9 语法解析、JSON 配置与冻结控制常量一致性、预览 CLI 参数名和训练入口对照、仅标准库导入检查均通过。只读取源码做静态验证，没有执行新增准备工具或运行模型；冻结文件无本次修改。
- 服务器：本新增工具尚未运行；各个依赖的当前实际一致性、权重和预测文件完整性尚待回传。历史 GPU 冒烟与 E0/E1 完成记录不等于本工具已通过。
- E2/E3：未新启动训练，未生成正式模型或性能结果；E3 完整启动还等待 E2 结果。
