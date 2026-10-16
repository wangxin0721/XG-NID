# Edge / No Edge / 门控对照阶段报告

日期：2026-10-16。P0 已完成，P1 真实数据链路和优化步骤验证通过；E0—E3 已实际完成小规模短程训练。正式全量基线、正式 No Edge 消融及多种子结论尚未完成。没有覆盖历史图、划分、结果或现有未提交修改。

## 三项验收

1. **Edge 实际使用**：前向 hook 记录每一层 GAT 收到的真实属性，Contain/Reverse 为 `[40,4]`，Link 为 `[38,1]`（两个真实图组成的诊断 batch）；两层、两个分支均覆盖。逐关系隔离扰动后，E0 最大预测 log-probability 变化为 Reverse Contain 20.2514；E2 为3.9586，E3 为2.8353。E1/E4 所有关系变化严格为0。
2. **严格 No Edge**：E0/E1 直接使用同一个 vendored `HeteroGNN_Edge`，只切换属性为 Tensor/None，绝不删除图连接。共同初始化 lazy 参数后逐项验证完整 state_dict 相同；输入节点、边索引、边属性、标签均未被 forward 修改。自动配置对比仅有 experiment/use_edge_attr 差异，图文件和 split 的 SHA256 相同。
3. **公平门控对照**：E2/E3 都使用 Edge、同一双异构编码器、分支分类器和 shared head。E2 固定0.5/0.5 logits 权重；E3 使用现有自适应 logits gate（保留现有置信度混合规则）。其他损失全部关闭，使用普通 NLL，无类别加权、hard negative、阈值校准、logit bias。E2 不调用 gate，gate 参数无梯度；E3 gate 有有效梯度。两者初始完整 state_dict 相同，配置对比仅 experiment/fusion 不同。

Contain 和 Link 在当前有向拓扑中每个目标只接收一个源消息，GAT softmax 权重恒为1；两类属性虽传入但当前没有可测的独立贡献，实测变化均为0。反向汇聚多个 Packet 才有注意力竞争。这是原设计的结构限制，不能把全部边属性都称为有效增益。隔离诊断临时只改变一类属性，直接调用 reference API；正常数据管线仍严格检查正反 Contain 镜像一致。

完整证据：`outputs/edge_chain_pilot_20261016_v2/smoke.json`。包含属性输入、前向差异、反向传播/优化更新、配对初始状态一致、图未被修改、错维度拒绝。E4 仅通过冒烟，未进行短程训练。

## 数据恢复与阻塞

见 `docs/edge_P0_20261016.md` 和 `outputs/edge_audit_20261016/source_audit.json`。全部185557行均检查字段对齐、数值有限性及payload解码；没有将缺失字段伪造成边属性。Flow82维、Packet1513维，三关系属性分别4/4/1维。

训练CSV160000行，每类20000；测试25557行。测试有3346条记录与训练完全重复，训练内部有47657条重复；没有原始 flow/capture ID，无法排除同流不同窗口跨集。当前没有历史 .pt 可验证旧样本顺序。正式复现受数据泄漏及身份追踪问题阻塞，不能宣称复现了历史实验。

已恢复 `data/processed/edge_pilot_20261016/graphs.pt` 和 `split.json`：每个源CSV按行序取每类16个内容不重复的真实图，共256图，不截断图节点和边；train/val/test为96/32/128，每类12/4/16。保持源train/test边界，从源train新建验证划分；样本ID为源文件SHA256:行号。记录内容哈希、类别分布、来源、构图代码哈希、字段映射和未使用额外归一化的事实。它是有选择偏差的流程试验集，不能替代正式评估。

## 原始实现与适配

- 参考范围为当前仓库的 `third_party/GNN4ID/Utility/Model.py`，未断言它与论文发布版本逐字一致。工作区原有大量未提交修改，参考文件本身也在其中；本轮没有改它，实验保存实际文件哈希。
- 该参考 Edge 类为两层 GAT、HeteroConv默认sum聚合、BN eps=1、64维、128→64→16→8线性分类头；原 No Edge 类为SAGE，不能作为严格单变量消融。
- E0/E1直接动态加载该参考Edge类，仅包装 `HeteroData` batch API和属性开关。E1是“参考GAT的无边属性消融”，不是原始SAGE No Edge版本。
- 当前项目自定义encoder的mean关系聚合、head激活、可变hidden维度未混称原始基线。E2/E3沿用当前自定义分支编码器，两个分支都接收完整异构图，再分别池化不同节点类型。
- 训练共同采用Adam、lr=0.001、batch16、hidden64、seed42、CPU、早停patience12、验证Macro-F1选优。短程预算3 epoch，不加参考训练脚本的ReduceLROnPlateau调度；这不是完整论文训练方案复现。

## 实际执行结果

最新交付目录：`outputs/edge_chain_pilot_20261016_v2/`。之前的首次短程运行也保留在无v2目录中，最终报告采用v2。两次预测指标相同；v2使E2完全跳过无效的gate计算，增加门控梯度检查。

| 实验 | Accuracy | Macro-F1 | Weighted-F1 | 最佳验证Macro-F1 | 最佳epoch |
|---|---:|---:|---:|---:|---:|
| E0 参考XG-NID Edge | 0.5234 | 0.4250 | 0.4250 | 0.4321 | 3 |
| E1 同GAT No Edge | 0.5234 | 0.4239 | 0.4239 | 0.4722 | 3 |
| E2 双分支固定等权logits Edge | 0.3594 | 0.2583 | 0.2583 | 0.2974 | 3 |
| E3 双分支logits门控 Edge | 0.3359 | 0.2355 | 0.2355 | 0.2894 | 2 |

每组测试各类均16图。每类Precision/Recall/F1/support、8×8混淆矩阵、标签/预测、门控权重和计算耗时位于各组 `metrics.json`；最佳权重/验证指标在 `best.pt`；`config.json` 记录数据和split哈希、种子、配置、代码HEAD和实际文件哈希；`epochs.jsonl` 为逐epoch训练日志，`status.json` 为完成及退出码0。

仅就此次试运行，Edge对No Edge的Macro-F1差为+0.001108；WebBased F1 +0.0541、Mirai +0.0269，Benign -0.0398、DDos -0.0323，其他类别无变化。Recon/BruteForce等困难类尚不能据此判断有改善。E3相对E2的Macro-F1差为-0.022855，不能声称门控获益。测试平均门控权重约0.5012/0.4988；三轮不足以判断塌缩或稳定性。

v2 CPU训练（含验证和保存，不含模型初始化）秒数E0/E1/E2/E3约0.512/0.423/0.710/0.775；测试前向约0.092/0.073/0.122/0.121。均为极短单次计时、无置信区间，不能作为正式开销结论。保留参数数目E0/E1均435032，E2/E3均853456（E2保留但不使用gate参数，以控制参数初始化）。正式开销需同硬件预热、重复测量和报告方差。

环境实际可用：`C:/Users/wx/anaconda3/envs/nids/python.exe`，torch2.8.0+cpu、PyG2.6.1、pandas2.3.3、sklearn1.5.1，无CUDA。默认系统Python缺torch；本轮未安装新依赖。PyG实际本地源码和signature已核查，参考位置参数调用和当前关键字调用均有效。

## 修改清单

| 文件 | 原因 |
|---|---|
| src/xgnid/model.py | Edge开关、逐关系属性检查、直接读取反向store、logits等权融合开关；保留历史模型名 |
| src/xgnid/graph_building.py | 修正空Packet1513维、空Contain(0,4)维度 |
| src/xgnid/edge_experiments.py | 原始Edge适配器、E0—E4受控构造、图完整性检查 |
| scripts/audit_edge_data.py | 无torch依赖的全量CSV来源/重复/字段审查 |
| scripts/run_edge_chain.py | 独立prepare/smoke/train/eval，保存身份、日志、状态和指标，拒绝复用输出目录 |
| scripts/summarize_edge_chain.py | 自动检查成对配置差异，生成真实指标和每类差值 |
| docs/edge_P0_20261016.md、本报告 | 审查证据、限制、实际结果和后续要求 |

旧CLI/训练入口没有被当作本轮标准流程：其自动校准、默认损失和固定Table4重划分仍需调用者知情。新检查点使用新入口eval，不使用旧CLI推断模型。

## 重运行与后续配置

PowerShell示例（每次更换为不存在的output目录）：

```powershell
& 'C:\Users\wx\anaconda3\envs\nids\python.exe' scripts/run_edge_chain.py smoke --data data/processed/edge_pilot_20261016 --output outputs/edge_smoke_new
& 'C:\Users\wx\anaconda3\envs\nids\python.exe' scripts/run_edge_chain.py train --data data/processed/edge_pilot_20261016 --output outputs/edge_chain_new --epochs 3 --experiments E0 E1 E2 E3
& 'C:\Users\wx\anaconda3\envs\nids\python.exe' scripts/run_edge_chain.py eval --data data/processed/edge_pilot_20261016 --checkpoint outputs/edge_chain_pilot_20261016_v2/E1/best.pt --output outputs/edge_eval_new
python scripts/summarize_edge_chain.py outputs/edge_chain_new
```

独立进程重新构造E1、加载检查点并复测已通过，128条标签和预测与首次保存逐项相同，结果在 `outputs/edge_checkpoint_eval_20261016/`（对应首次短程运行；v2同组预测一致）。初次CSV审查遇到csv字段长度默认上限，显式提高后全量重跑成功；审查未发现缺失数值。其余交付运行退出状态全部为0，日志在 `outputs/edge_audit_20261016/`。

正式链条计划仍为E0 Edge → E1同GAT No Edge → E2无门控Edge → E3门控Edge，可选E4门控No Edge。目前仅pilot配置和数据可执行；正式全量配置尚不能在当前身份信息下通过数据门槛。

下一阶段先获取可追溯原始流身份/时间/捕获来源，按流或capture分组划分，先划分再对训练集采样，检查重叠与类别分布并冻结manifest。不要把当前已平衡的160000行再次随机重切成论文基线。随后统一100 epoch上限、patience12、Adam lr0.001、batch16、hidden64、无辅助策略；先seed42，稳定后seed43/44，全部使用同一冻结split；E0/E1与E2/E3分别逐项核查只改变目标变量。正式训练前重做本报告的真实数据冒烟，训练后报告多种子均值/标准差、每类指标、困难类、门控分布/梯度以及重复计时开销。不要将当前短程数值用于论文优越性声明。
