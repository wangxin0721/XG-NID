# 原论文参数的证据分级核查（2026-10-19）

结论：batch=64、30轮、lr=0.01可以确认为作者公开示例的设置，且在2024年的历史版本中已经存在；不能据此确认它们就是产生论文97%结果的完整运行配置。当前R0-A是公开Edge协议重实现，尚不能称为作者论文检查点的精确重现。本次仅补充文档与来源证据，不改冻结配置、源码锁、训练入口或运行中的训练。

## 来源与历史

- 用户提供的期刊PDF：`C:/Users/wx/Desktop/XG-NID(科研通-ablesci.com).pdf`，16页，SHA256 `33a1bdaf53d356ec442d424f91e9f7d4731a65eb636e57325b65eb7e1cb7a83e`。按PDF实际页序定位，检查正文、公式和图表。第1页记录2024-09-19投稿、2025-05-10在线发表。
- [2024-09-07模型notebook](https://github.com/Yasir-ali-farrukh/GNN4ID/blob/194ffd86ff0a29b446fa0099c68be67cf73ea568/GNN4ID_Model.ipynb)：与冻结来源的上游原始SHA256完全相同。2024-08-07的更早notebook也已有相同主要超参数。
- [2024-09-07 Model.py](https://github.com/Yasir-ali-farrukh/GNN4ID/blob/194ffd86ff0a29b446fa0099c68be67cf73ea568/Utility/Model.py)和[2024-07-08 Training.py](https://github.com/Yasir-ali-farrukh/GNN4ID/blob/18c3bebd89d005336fe08b9a3ebfcd382b417b0e/Utility/Training.py)：与冻结来源的上游原始SHA256分别完全相同。依据原始文件、路径提交历史与commit patch日期交叉核验。
- 固定仓库提交`551d1f1…`虽在2026年，patch实际是引用信息修改；不能再据根提交日期暗示模型/训练代码直到2026年才出现。历史代码的存在也不能证明某次论文运行采用了它。
- 默认值依据PyTorch v2.3.1和PyG 2.5.3源码，区别于作者显式赋值，更区别于作者历史运行环境。

## A：论文明确报告的内容

| 项目 | 可确认内容 | 原文位置与边界 |
|---|---|---|
| 目标指标 | CICIoT2023 Proposed约97% Macro-F1 | 第11页图5纵轴明确Macro F1-Score；图表为舍入值，未提供完整精度原始结果 |
| 划分与样本量 | 80:20；训练每类重采样至20000，共160000；测试25557 | 第9～10页§3.5.1、表4；测试六类各4000、WebBased1090、BruteForce467 |
| 报文上限 | 每流最多20个报文 | 第4页§3.1.1 |
| 流提取超时 | idle_timeout=120秒 | 第4页§3.1.1；不能与滚动特征窗口混为一谈 |
| Payload | 1500字节、短报文补零、字节值0～255 | 第4～5页；不是声明缩放至0～1 |
| 提取阶段特征 | 76个流级、14个报文级特征 | 第4～5页；不是对最终Flow输入维度82的直接否定，提取与编码阶段不同 |
| 滚动窗口实验 | 图2中350～400范围达到约97% | 第5页图2；没有唯一确定最终运行取350还是400，也不能直接将代码350解释为350秒 |
| 图与边属性 | Flow/Packet节点、Contain/Link关系；Contain四个属性，Link时间差 | 第5～6页§3.1.2～3.1.3 |
| 模型高层结构 | 两层GAT、BN、激活、按节点类型mean pool、拼接、三层FC、分类输出 | 第6～7页§3.1.4；具体宽度等需查代码 |

论文完整16页文本检索未出现batch_size/batch size、epoch、learning rate、optimizer、Adam、dropout、seed、hidden等训练参数表述，结合相关章节与图表阅读，本轮未找到这些数值的正文披露。关键词未检出本身不是证明历史运行不存在某设置。

## B：官方代码明确或在指定依赖下可推导的设置

以下是公开Edge路径的配置，不能将此表标题改成“论文97%运行参数已证实”。notebook单元编号从0开始。

| 项目 | 实际设置 | 证据与容易误读的地方 |
|---|---|---|
| 训练batch | 64，shuffle=True | notebook单元12 |
| 测试batch | 1，shuffle=False | notebook单元16 |
| 轮数 | 30 | notebook单元8，训练循环使用epochs |
| 优化器、学习率 | Adam、0.01 | notebook单元8及Training.train_with_edge_Att |
| 权重衰减 | 实际0 | args字典虽然写1e-5，但Edge优化器未传weight_decay；采用Adam默认0 |
| Adam其他参数 | betas=(0.9,0.999)，eps=1e-8，amsgrad=False | 指定PyTorch 2.3.1的默认值，不是论文显式报告 |
| 调度器 | Edge路径无有效scheduler | Edge函数构造与step均注释；SAGE路径的有效调度器不能套到Edge |
| 隐层与注意力头 | 每层64维；heads=1 | 64在Model中显式写出；heads采用PyG 2.5.3默认值。args中attn_size=32未被Edge使用 |
| BatchNorm | eps=1.0；momentum=0.1 | eps来自args且确实传入；momentum来自PyTorch默认值。不要擅自改成常见eps=1e-5 |
| 跨关系聚合 | sum | HeteroConv未收到aggr，使用PyG默认sum；构造器aggr="mean"只被保存 |
| 图池化 | 各节点类型global_mean_pool后拼接 | 与跨关系聚合是两个操作 |
| 激活 | GAT层后BN、nn.LeakyReLU，默认斜率0.01 | GAT内部注意力的negative_slope默认0.2是另一处激活 |
| Dropout/自环 | GAT dropout默认0；add_self_loops=False显式设置 | 没有额外Dropout模块 |
| 分类头与损失 | 128→64→16→8连续Linear；log_softmax；NLLLoss | 三个Linear之间没有激活，见下一节差异 |
| 输入 | Flow82；Packet1500；含反向边 | Functions与作者发布CSV及已完成全量重建；不是仅凭论文76维描述推导 |
| 每轮评价 | 训练集Accuracy | train_with_edge_Att内test_edge(train_loader,...)；没有每轮验证集/测试集Macro-F1 |
| 模型选择 | 公开示例训练结束后使用当前模型 | 没有有效best_model更新、验证集选择或早停；示例对应最后一轮，不证明论文历史只测试过一次 |
| 训练随机种子 | 未在示例中设置 | 采样处某些random_state=42不等于训练种子42 |

默认值来源：[Adam v2.3.1](https://github.com/pytorch/pytorch/blob/v2.3.1/torch/optim/adam.py)、[BatchNorm v2.3.1](https://github.com/pytorch/pytorch/blob/v2.3.1/torch/nn/modules/batchnorm.py)、[GATConv 2.5.3](https://github.com/pyg-team/pytorch_geometric/blob/2.5.3/torch_geometric/nn/conv/gat_conv.py)、[HeteroConv 2.5.3](https://github.com/pyg-team/pytorch_geometric/blob/2.5.3/torch_geometric/nn/conv/hetero_conv.py)。

## C：确定存在的描述差异与公开示例的边界

1. **分类头激活不同**：论文第7页式(9)在FC之间写有ReLU，官方Edge前向没有。当前R0-A遵从代码。差异真实存在，但没有证据认定它导致了现有分差，更不能依据测试集分数决定采用哪一版。
2. **边方向描述不统一**：论文第6页文字写连续报文构成有向无环链，图3却画双向箭头；代码ToUndirected生成反向关系。不能声称论文对全图方向给出了唯一一致规范。
3. **notebook活动路径是SAGE**：HeteroGNN/train/test_cm有效，Edge相应调用被注释。因此整本notebook不是一份可直接绑定论文GAT97%的执行记录。R0-A明确选择官方HeteroGNN_Edge和对应Edge训练函数，不为追分切SAGE。
4. **76与82所处阶段不同**：论文描述提取阶段，最终公开CSV经附加特征与编码后为82维Flow。现有证据支持82维公开输入；缺的是作者完整历史变换记录，不能将维数文字差异直接诊断成漏特征。
5. **数据公开版本与论文运行绑定尚缺失**：我们已取得作者当前发布CSV的哈希，并完成185557张图全量重建，所有模型输入与标签等于旧图。这降低了旧图生成错误的可能性，但不证明公开CSV逐字对应论文97%那次私有运行。

## D：仍不确定、不能写成事实的内容

- 产生97%结果那次运行的完整配置、训练seed、划分与采样随机状态、文件成员顺序、初始化权重与RNG状态。
- 97%来自单次、最好一次还是多种子均值，是否还做过其他超参数尝试；未找到对应原始日志与检查点。
- 论文实际采用最后一轮还是另有模型选择；公开示例的最后一轮行为不能替代论文实验记录。
- 论文历史运行的确切依赖、CUDA/GPU与确定性设置；requirements声明不等于运行环境快照。
- 最终滚动窗口究竟350还是400，以及逐PCAP上下文、排序、上游计算与完整数值变换的历史轨迹。
- 论文公式和公开实现不一致时，实际97%用的是哪一版。

## 当前R0-A的定位与后续边界

配置`configs/r0_edge.json`中batch64/test1、30轮、Adam0.01、wd0、无scheduler、hidden64、BN eps1与上述公开Edge路径一致。seed42是我们为了可重复而选择的值，不是已核实的作者训练seed。R0-A额外对齐数字文件索引顺序及首次训练前向初始化，不加入用户双分支门控。

本轮静态核查不提供新的服务器运行状态。上次回传的R0-A v2正在训练；本次未停止、重启或更改它。原R0的Macro-F1为92.8211%，相对论文舍入约97%低约4.18个百分点，差距原因尚未定位。不能将“公开协议执行完成”写成“97%性能已经复现”。

当前应完成固定的R0-A并保存配置、逐轮日志、最后检查点与预测。若仍有明显分差，优先寻求作者原始实验配置/日志/检查点绑定；论文公式版与代码版可作为预先定义、独立命名的协议差异实验，不能混入当前运行或用测试分数挑选版本。

机器可核验来源哈希和历史日期见[r0_parameter_evidence_20261019.json](r0_parameter_evidence_20261019.json)。冻结`r0_source_lock.json`保持原样；本文件是新的补充证据，不追溯改写历史审计时点的未知状态。
