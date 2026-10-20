# E4：E3 关闭边属性的独立对照（2026-10-20）

用户已要求开始E4，并确认当前两张GPU都忙，先交付命令。本次只做本地开发、语法/元数据验证与GitHub交付，不登录或抢占服务器GPU，不自动运行服务器冒烟或正式训练，不改动冻结E0～E3源文件、配置、划分和结果。不因为E3测试结果重新调参。

## 固定实验定义

E4保持E3两套独立的完整异构图编码器、两层GAT、Flow/Packet分别池化、64→8投影、自适应门控与熵补量混合、8→32→8共享分类头以及最终普通NLL；只将全部12个GAT的edge_attr输入关闭。节点特征82/1500、图边与拓扑、4/4/1属性原文件均不修改；不是删除图边、置零属性或更换SAGE。只在模型消息传递中不用边属性。

使用原有`build_experiment('E4')`，不新增模型结构。该构造器已令`use_gate=True`、两个编码器`use_edge_attr=False`，其余辅助损失权重均为0。休眠边投影保留，总参数预计846800，但有效更新参数不因此与E3相同。

| 项目 | 固定设置 |
|---|---|
| 比较对象 | 已完成的`outputs/grouped_full_E3_s42_resume_v1`，其原始batch16配置 |
| 划分 | 原分组split；107621 train / 13446 val / 13448 test |
| 初始化 | seed42；从头初始化，与E3初始状态配对；不加载E3训练权重 |
| 优化 | Adam lr0.001，默认weight_decay0；batch16；hidden64；workers2 |
| 预算/选模 | 最多100轮；验证Macro-F1严格改善；patience12；无scheduler |
| 测试 | 重载最佳验证权重后，固定测试集评价一次 |
| 新输出 | 冒烟`outputs/grouped_gpu_smoke_E4_s42_v1`；正式`outputs/grouped_full_E4_s42_v1` |

E3/E4是自适应双分支中的边属性消融，不与其他批大小的探索实验混为同一受控比较。E0/E1不能替代该对照；E3/E4单次结果也不能证明跨架构通用收益或统计显著性。

## 独立入口如何复用冻结实现

新增`configs/e4_no_edge.json`与`scripts/run_e4_server.py`。原`train_grouped.py`命令行仅接受E0～E3，不能直接添加`--experiments E4`启动。新入口直接构造E4参数并调用原`run`函数，训练/验证/梯度/采样/选模/检查点/最终测试实现不变。

原训练配置中`use_edge_attr`、`fusion`只有E0～E3的元数据表达式，不适用于E4。新入口只在本进程为E4写配置时将其正确记录为`use_edge_attr=false`、`fusion=adaptive_logits`，并补充真实E4冒烟与父E3证据；不借此更改forward。还在首轮前核对E4和E3的训练/验证sample ID顺序哈希、分区数量及数据清单一致。

原readiness仍核对既有E0～E3 GPU冒烟、E0/E1预跑、E0全量基线及冻结源码。新入口额外验证真实完成的父E3摘要/配置/SHA，以及新E4配对GPU冒烟。仅在返回的内存smoke查找表中添加实际新E4初始状态，使原训练初始化校验可用；历史smoke文件完全不改写。历史四模型检查与本次新E3/E4单步检查的来源各自保留，不伪称全部模型在本次重跑。

模型懒初始化沿用原`initialize`：train-only初始化前向临时启用属性以分配相同边投影参数，eval模式不训练；随即为E4全部关闭。实际E4冒烟探针与训练forward必须不传属性，初始权重哈希必须与原E3一致，不能因边投影未初始化而改变参数规模或种子配对。

## 阶段一：GPU忙时也可做的只读检查

在服务器wx环境、仓库根目录执行。只fetch并提取新文件到/tmp，不pull、不切换现有分支、不安装或替换CUDA依赖。下列分支引用以本次交付提交为准。

```bash
. /home/nzxn/miniconda3/etc/profile.d/conda.sh && conda activate wx
cd /home/nzxn/下载/WX/load/XG-NID
git fetch origin codex/r0-edge-preparation
git show origin/codex/r0-edge-preparation:scripts/run_e4_server.py > /tmp/xgnid_e4.py
git show origin/codex/r0-edge-preparation:configs/e4_no_edge.json > /tmp/xgnid_e4.json
nice -n 10 python -B /tmp/xgnid_e4.py --repo . --config /tmp/xgnid_e4.json --mode check
```

check模式只用标准库读取父E3完整状态、摘要、配置、源文件SHA、样本清单及安装包版本元数据，不导入torch/PyG、不反序列化权重、不创建GPU上下文或输出训练目录。返回`metadata_checks_passed`不代表新E4模型已经冒烟通过。

## 阶段二：GPU空闲后，真实配对冒烟

先用`nvidia-smi`确认目标卡空闲。以下以物理GPU0为例；若明确使用GPU1，只把`CUDA_VISIBLE_DEVICES=0`改为1，进程内部仍为cuda:0。不要在两张卡都忙时执行，也不要停掉别的任务给本次让路。

```bash
CUDA_VISIBLE_DEVICES=0 python -B -u /tmp/xgnid_e4.py --repo . --config /tmp/xgnid_e4.json --mode smoke
```

复用原冒烟工具，只读取冻结train的32候选图，选原16图初始化batch，分别临时检查E3/E4各一个优化步；不读val/test图、不新训完整E3、不改写任何父实验权重。核对：

- E3/E4的初始state SHA都与已完成E3一致，实际样本与原GPU冒烟相同；参数各846800。
- E4全部12个GAT输入edge_attr为None，边投影梯度None或0；逐关系改变边属性，输出变化不超过重复前向数值容差。
- E4自适应门控有非零有限梯度，整体损失/梯度有限、优化器参数确有更新，输入节点/拓扑/属性/标签不变。
- E3同批次保留属性依赖，作为正对照；不把E4属性不敏感误当成整个探针失效。
- 新配置、工具、原源码、父E3、运行环境及smoke artifacts SHA绑定在`e4_binding.json`。

必须看到最终`e4_cuda_smoke_passed`，且`e4_binding.json`为completed，才视为本阶段通过。仅原工具打印`Smoke completed`或raw smoke的status completed还不够，额外配对检查可能随后失败。失败保留报告，不覆盖旧目录或自动重试。

## 阶段三：真实冒烟通过后，正式训练

下列是后续启动命令，当前不代表已经执行。再次确认GPU可用并读取真实冒烟结果；full模式会重新校验绑定证据与源文件，缺少或不一致即拒绝，不回退CPU、不自动补跑冒烟。

```bash
if [ -e outputs/grouped_full_E4_s42_v1 ] || [ -e outputs/grouped_full_E4_s42_v1.log ]; then
    echo "E4目录或日志已存在，先检查，不重复启动。"
else
    CUDA_VISIBLE_DEVICES=0 nohup python -B -u /tmp/xgnid_e4.py --repo . --config /tmp/xgnid_e4.json --mode full >> outputs/grouped_full_E4_s42_v1.log 2>&1 < /dev/null &
    echo "$!" > outputs/grouped_full_E4_s42_v1.pid
    echo "E4请求已提交，请查日志确认实际开始。"
fi
tail -n 30 -F outputs/grouped_full_E4_s42_v1.log
```

出现`Starting full E4 on cuda:0`只表示进入入口；首轮指标才证明训练推进。新目录拒绝覆盖。当前入口仍不提供E4重启恢复功能，不沿用E3专用resume脚本；如计划重启，应先妥善处理该任务。

```bash
cat outputs/grouped_full_E4_s42_v1/status.json
cat outputs/grouped_full_E4_s42_v1/E4/metrics.json
```

只有根status completed、exit_code0且最终metrics完整后才记为正式完成。逐样本predictions.csv仍含sample_id、标签、预测及两路权重；E4训练/测试数据与E3对齐，测试集不用于选择新参数。

## 交付与验证边界

本次新增配置、独立入口和本文档；未修改冻结模型/训练/数据工具。Python3.9语法、E3控制参数一致性、构造器E4开关静态检查通过。纯元数据正例通过，五种反例（属性进入GAT、初始化不配对、门控无梯度、输出依赖边属性、边投影有梯度）均被拒绝；E4配置注释适配及历史smoke不被内存扩展改写的检查通过。没有执行torch导入、真实模型构建、前向、反向、GPU冒烟或训练。记录冒烟/正式的CUDA_VISIBLE_DEVICES映射，不自动挑卡。当前服务器check、E3/E4配对冒烟和E4正式训练均待真实回传，不将命令准备标记为已执行。
