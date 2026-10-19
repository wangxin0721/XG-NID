# R0-V：验证集选模、最多100轮、早停12轮

用户于2026-10-19明确确认：从原R0训练部分按输入分组留出约10%作验证，连续12轮验证Macro-F1不提升则早停，最多100轮，加载验证最佳模型后测试一次。这是新的R0-V协议，不覆盖R0/R0-A或分组E0～E3。本地仅实现入口并执行标准库测试；尚无本协议服务器CUDA冒烟、训练或测试结果。

## 固定规则

| 项目 | R0-V |
|---|---|
| 原始训练来源 | 原R0冻结清单中的160000行 |
| 验证来源 | 仅从上述训练行中按类、按node_hash分组留出约10% |
| 原测试集 | 原25557行，成员不变；不参与分组选择、训练、早停或选模 |
| 分组与副本 | 同No Edge输入hash的所有副本及边属性变体同集；跨类冲突直接报错 |
| 留出算法 | 每类group按hash排序，seed=42+label打乱；选择样本数最接近该类10%目标的非空前缀，至少给训练留一个组；同距离选较短前缀 |
| 样本比例 | 预计144000/16000/25557；实际训练/验证数量由整组划分决定并写入split.json，不强制拆组凑数 |
| 重采样 | 不新增、不去重；保留原公开训练CSV已经包含的全部副本，整组划分。验证指标也按保留的行计数 |
| 最大预算 | 100轮，不保证一定执行100轮 |
| 早停指标 | 验证集Macro-F1，固定8类、zero_division=0 |
| 早停规则 | patience=12，min_delta=0；严格大于才提升，相同分数也算未提升，保留较早最佳轮 |
| 最佳模型 | 每次验证提升原子保存best.pt；每轮另存last.pt |
| 最终测试 | 训练结束后严格加载best.pt，一次遍历原测试集，导出预测与指标；冒烟不构造或读取测试图 |
| 保持的参数 | 官方Edge模型，hidden64、BN eps1、Adam lr0.01、wd0、训练batch64、测试batch1、无scheduler、seed42 |
| 验证batch | 64，eval模式、no_grad；每轮直接输出验证Macro-F1 |
| 特殊训练尾批 | 若最后只有1图，合入前一批形成65，保留所有样本，避免训练Flow BatchNorm单例 |
| 初始化 | 首次训练前向进行lazy初始化，无预热前向；每轮数字基础顺序上的随机打乱 |

原论文80:20是划分阶段描述；训练重采样及测试限额后的表4最终数量为160000/25557，因此最终约86.23%/13.77%不等于划分时80%/20%。R0-V在最终训练部分上新增验证集，是方法比较的协议变体，不能据此称为论文原始训练预算的精确复现。

输入分组只保证R0-V训练/验证之间无相同node_hash；不能证明PCAP/捕获/流级独立。既有原始训练/测试输入重叠如实记入split.json，不改变测试集以提高分数。原测试结果此前已经被查看过，本入口的“一次”指本次新运行只有一次正式测试遍历，不宣称该测试集从未被查看。

## 服务器执行

图和旧R0代码仍从原r0_full_launch.json定位，逐一核验冻结代码、原检查点、清单与官方模型身份；无需重生图、复制图或更改旧入口。只在Linux服务器执行。旧R0-A如仍运行，新smoke/full会拒绝启动且不停止任何进程；prepare仅处理JSON元数据，可以先运行。

下载本次新入口和配置，不checkout或覆盖服务器工作树：

```bash
. /home/nzxn/miniconda3/etc/profile.d/conda.sh && conda activate wx
cd /home/nzxn/下载/WX/load/XG-NID
git fetch origin codex/r0-edge-preparation
git show origin/codex/r0-edge-preparation:scripts/run_r0_validation_server.py > /tmp/xgnid_r0_validation.py
git show origin/codex/r0-edge-preparation:configs/r0_validation.json > /tmp/xgnid_r0_validation.json
python -B -u /tmp/xgnid_r0_validation.py --repo . --config /tmp/xgnid_r0_validation.json --mode prepare
```

prepare写入新`outputs/r0_validation_split_s42_v1`，打印真实逐类训练/验证数量。清单已存在会拒绝覆盖，不删除后重抽、不根据结果改seed。

待旧训练结束，执行本协议新的CUDA冒烟；旧R0-A冒烟不能替代新协议冒烟：

```bash
python -B -u /tmp/xgnid_r0_validation.py --repo . --config /tmp/xgnid_r0_validation.json --mode smoke --device cuda:0 --output outputs/r0_validation_smoke_s42_v1
```

冒烟取已冻结训练32/类、验证16/类，训练2轮；校验实际优化更新、6个GAT边投影梯度、BN计数、验证计算、best/last保存及加载最佳权重。smoke摘要必须completed/exit_code=0、epochs_completed=2、optimizer_step_passed=true、test=null、test_passes=0。它证明小样本执行路径，不是性能结果；早停边界另由纯逻辑测试覆盖。

冒烟通过后后台启动正式新运行：

```bash
nohup python -B -u /tmp/xgnid_r0_validation.py --repo . --config /tmp/xgnid_r0_validation.json --mode full --device cuda:0 --smoke-report outputs/r0_validation_smoke_s42_v1/summary.json --output outputs/r0_validation_full_s42_v1 >> outputs/r0_validation_full_s42_v1.log 2>&1 < /dev/null &
```

正式入口核对smoke的配置、划分、源码、运行环境及best/last检查点哈希。输出目录存在就拒绝，不恢复旧训练、不自动重试。保持/tmp中的入口和配置不变，直至本次正式运行结束。

查看进度：

```bash
tail -n 20 -f outputs/r0_validation_full_s42_v1.log
```

每轮JSON包含`val.macro_f1`、`best_epoch`、`best_val_macro_f1`、`stale_epochs`。也可只读：

```bash
cat outputs/r0_validation_full_s42_v1/status.json
tail -n 1 outputs/r0_validation_full_s42_v1/epochs.jsonl
```

正式完成的summary应为completed/exit_code=0、epochs_completed<=100、selection=best_validation_macro_f1、test_passes=1，并记录best_epoch、最佳验证分数、两个检查点哈希和最终测试结果。测试开始前先记录test_started.json；若测试中断，保留失败证据，不自动重复测试。

## 本地验证边界

`python -B -m unittest discover -s scripts -p test_r0_validation_protocol.py -v`：5项标准库测试通过，覆盖分组副本/边变体不跨集、确定性与成员保留、拒绝测试成员/冲突标签/重复ID、12轮早停与并列最佳、训练单例尾批合并，以及冒烟绑定与检查点篡改检测。还核对入口Python3.9语法和JSON配置；没有本地torch导入、真实图加载或模型运行。

入口：[run_r0_validation_server.py](../scripts/run_r0_validation_server.py)；配置：[r0_validation.json](../configs/r0_validation.json)。R0-V当前为“实现与标准库验证完成，服务器执行待回传”，不是已启动或已通过CUDA。
