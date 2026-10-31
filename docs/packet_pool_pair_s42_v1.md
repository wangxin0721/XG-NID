# D32-Gate报文池化P0／P1

保留D32-Gate作为基线，本轮只比较报文平均池化P0与注意力池化P1。关系聚合、跨层融合、残差和GATv2分别留待后续独立实验，不把冻结全部编码器的本轮协议直接复用到其他结构改动。

两组严格复用 `outputs/fusion_gate_pair_s42_v1/D32-Gate/best.pt`：第62轮、验证Macro-F1=97.32070542814468%，SHA256=`7815875a111d3e94ceefd0e9ef6cb64dedaf6ff80723645f1ec035b376af31ff`。不使用干净第0轮头或任何联合训练后的权重。旧源码、配置、缓存、检查点和结果均不修改。已有工作树的其他改动不属于本轮。

| 配置 | P0 | P1 |
|---|---|---|
| Packet读出 | 同图平均 | 同图softmax注意力加权和 |
| 新评分网络 | 无 | Linear(64,16)→LeakyReLU→Linear(16,1,bias=False) |
| 新参数 | 0 | 1,056 |
| 可训练总参数 | 7,144 | 8,200 |
| 原64→32投影／逐维门控／分类头 | 从成熟权重继续训练，lr=1e-5 | 相同 |
| 新池化学习率 | 无 | 1e-4 |
| 编码器／BN | 参数和所有缓冲区冻结，eval | 相同 |

P1保留PyG `AttentionalAggregation` 的gate_nn评分参数布局，权重按图softmax，不添加特征变换网络；输出仍为64维。实际读出使用等价的 `mean(h) + sum((beta-1/n)*(h-mean(h)))`，使评分末层零初始化时修正项为零，数值上复现平均池化，同时保留评分梯度。构造新模块只使用独立CPU随机流，不改变CPU／CUDA训练随机状态。[PyG 2.5.3源码](https://pytorch-geometric.readthedocs.io/en/2.5.3/_modules/torch_geometric/nn/aggr/attention.html)提供评分及分组权重的原始实现；本次数值修正和已失败目录的缓存恢复见 `docs/packet_pool_recovery_20261031.md`。

协议固定为seed42、原分组划分，新建Adam、weight_decay=0、原最终输出NLL，train/eval batch=512/16，最多100轮、patience20。以验证Macro-F1严格改善选模；第0轮参与，若后续均未改善则保留第0轮。两组独立重置训练随机状态和优化器，校验所有共同轮次的样本顺序。学习率仅是本轮筛选设置，不声称最优。

## 节点缓存与诊断

旧图级64维缓存无法恢复报文。本轮从原训练／验证图提取冻结的Flow图向量、Packet节点向量和每图指针，存为 `train_packet_nodes.pt`、`val_packet_nodes.pt`。缓存只保存冻结表示；正式训练时按同一图顺序构建变长小批次，在梯度范围内运行池化、投影、门控和分类头。缓存常驻CPU，按批转入GPU。

`packet_diagnostics.json`按划分及八类报告每图Packet数量直方图、多报文比例、表示有差异的多报文图数，以及图内节点到均值的RMS欧氏距离。单报文距离为0；表示差异统计单独看多报文图，避免大量单报文掩盖差异。报告突出WebBased／Recon所在类别，不能因为大多数图只有一个报文便跳过实验。诊断只用训练／验证图，不打开测试图。

整体拼接前先输出总Packet节点数和 `cache_memory_estimates.json`：最终缓存张量大小、分批张量与拼接结果同时存在时的峰值、指针输入／cumsum输出的短暂重叠，以及已保留的另一份缓存。另记录此时Linux的MemAvailable，用新增分配量与可用内存比较，避免把已驻留缓存再扣一次。该估算仅覆盖拼接阶段的缓存张量，不是进程RSS；Python对象、原图、worker预取、模型／batch及分配器／序列化缓冲均未计入。拼接完成立即释放批次张量列表，不再让两份表示持续占用内存。

当前仍用原格式缓存，不预先改为分片。若新增张量分配估算已超过实际MemAvailable，先保留估算并在拼接前退出，再依据服务器数据考虑分片；内存未知时明确记为null，不编造可用容量。

## 图加载计数修正

cb830e8中普通字典无法汇总两个worker里的更新。分区拦截保留，最终 `graph_load_counts` 改为主进程字典中记录的直接加载，加上主进程已核验样本ID／标签的worker阶段样本数。六个阶段分别是训练／验证缓存提取、P0／P1初始回放、P0／P1最佳检查点回放。完整执行时worker阶段应贡献train=107,621、val=67,230；另加模型构造预热、梯度探测等主进程直接加载。若worker数为0，阶段样本已经在主进程字典中，不重复相加。

分阶段依据保存在 `graph_load_accounting.json`，同时写入summary、analysis和report；汇总报告会重新计算总数并核对六个阶段的完整样本量。计量单位是包含重复读取的样本加载次数，不是物理磁盘IO次数，也不是每轮缓存训练消费的样本量。失败或未完成阶段不作为完整验收证据。

## 阻断式验收

1. 启动前核对成熟源结果、协议、划分、编码器来源和权重SHA；两组公共状态严格加载且摘要相同。
2. 完整验证P0／P1初始log概率及门控权重allclose（atol/rtol=1e-6），Packet注意力逐图均匀；逐样本预测和混淆矩阵复现成熟D32-Gate。两组另用全部原验证图回放，并与未改动的历史模型前向比较。
3. 从训练集中有表示差异的多报文图确定性选取最多512图，两组各做3次废弃更新。原图前向与节点缓存前向容差1e-5；检查下游模块及P1评分网络的梯度、参数变化和逐图注意力偏离均匀。末层零初始化允许隐藏层第一步梯度为零，要求后续有更新。若整个训练集没有这种报文表示，记录完整诊断并在训练前退出，不能把这种情况报告为已验证注意力能学习。
4. 每组探测结束恢复所有模型状态、清除梯度、恢复Python／CPU／CUDA RNG；丢弃探测Adam。两组正式训练新建Adam。全部编码器子模块保持eval、无梯度，参数及BN缓冲区摘要不得变化。
5. 保存并严格重载预检检查点；两组最佳检查点分别在新建完整结构中严格加载，使用全部原验证图回放，逐样本log概率allclose、预测和混淆矩阵与节点缓存一致。
6. 训练后复核源产物、节点缓存摘要、共同轮次顺序及编码器／BN；通过后才写completed。任何失败写failed和error.log，不自动重试或扩展实验。

## 服务器执行与产物

从服务器项目根目录运行，本机没有torch／PyG，不能执行正式训练。`bundle`必须指向包含本轮新增源码、scripts及原有configs依赖的完整交付目录；同一checkout内执行可用：

```bash
cd /home/nzxn/下载/WX/load/XG-NID
bash scripts/launch_packet_pool_pair.sh 1
```

启动器先运行小型CPU张量测试及源元数据检查，再后台执行GPU1训练，允许共享，不停止其他任务。Python默认 `/home/nzxn/miniconda3/envs/wx/bin/python`；可通过 `XGNID_PYTHON` 指定。输出固定为新目录 `outputs/packet_pool_pair_s42_v1`，已有目录／日志／PID则拒绝覆盖。与过去图级缓存实验相比，这轮需要重新读取训练／验证原图提取节点，还包含原图探测与回放，不承诺数分钟完成。

小型CPU张量测试暂时屏蔽CUDA设备，只验证数学、梯度、分组、严格加载与状态恢复；正式进程另按指定GPU执行真实原图和CUDA验收。小型测试通过不能代替正式验收。

```bash
tail -n 50 -F outputs/packet_pool_pair_s42_v1.log
cat outputs/packet_pool_pair_s42_v1/progress.json
cat outputs/packet_pool_pair_s42_v1/status.json
```

完成后优先读取 `status.json`、`report.md`、`analysis.json`、`packet_diagnostics.json`。另保留 `summary.json`、`graph_load_accounting.json`、`cache_memory_estimates.json`，以及两组的第0轮、best／last完整权重与优化器状态、逐轮日志、逐类P/R/F1、混淆矩阵、每例32维融合权重和每报文注意力、初始化验收、探测／回放收据、缓存／来源摘要及耗时。

预算判断为 `Δ=F1(P1)-max(F1(P0),F1(D32-Gate))`，按百分点报告。Δ≥0.10个百分点才考虑扩大实验；P0不超过成熟D32时，目标为97.42070542814468%。小于门槛的收益同样记录，门槛不代表统计显著。

报告同时给出P1−P0、P1−D32、P0−D32，WebBased／Recon召回变化、Spoofing／BruteForce取舍。三种比较均导出修复／退步／双方都错但预测变化的样本ID，以及按真类和预测去向聚合的变化。保留八类完整指标，不能只报告总体提升。第0轮被选中不代表新结构有效。

本轮不自动启动关系聚合、组合消融、多seed或测试集评价。池化与关系聚合若各自有效，再做组合与消融；论文贡献需由双分支具体设计和这些匹配实验支持。
