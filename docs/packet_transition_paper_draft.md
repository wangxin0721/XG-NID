# 基于双编码器与报文转移读出的物联网入侵检测

## 1 方法

### 1.1 异构流量图与双编码器

给定一条流的异构图 $G$，本文预测其所属的八类流量之一。每张图包含一个 Flow 节点和一组按采集顺序排列的 Packet 节点，节点输入分别为 82 维和 1500 维。图包含 Flow→Packet、Packet→Flow 和 Packet→Packet 三类关系，边属性维度分别为 4、4 和 1。实验复用从 CICIoT2023 构建的 GNN4ID 流量图。 [CICIoT2023](https://doi.org/10.3390/s23135941) · [GNN4ID](https://github.com/Yasir-ali-farrukh/GNN4ID)

Flow 与 Packet 分支各有一套独立参数的两层异构图注意力编码器。两套编码器均读取完整异构图。每层按边类型进行消息传递，对同一目标节点来自不同关系的输出取均值，再按节点类型执行批归一化和 LeakyReLU。Flow 分支读取第二层的 Flow 节点表示 $\mathbf f\in\mathbb R^{64}$。Packet 分支保留两层 Packet 节点表示 $\mathbf h_i^{(1)},\mathbf h_i^{(2)}\in\mathbb R^{64}$。 [图注意力网络](https://arxiv.org/abs/1710.10903) · [编码器实现](https://github.com/wangxin0721/XG-NID/blob/802cc0d03f10b41868cc88b4e345207f4877aad8/src/xgnid/model.py)

### 1.2 报文节点读出

第一层 Packet 表示取均值得到 $\bar{\mathbf h}^{(1)}$。第二层保留原 C 的 Flow 查询。将 $\mathbf f$ 和第二层节点表示分别映射为 8 维查询与键，在流内对缩放点积做 softmax，得到报文权重 $\beta_i$。记第二层节点均值为 $\bar{\mathbf h}^{(2)}$，实现采用均值加中心化修正的形式

$$
\mathbf p_2=\bar{\mathbf h}^{(2)}
   +\sum_{i=1}^{n}\left(\beta_i-\frac{1}{n}\right)
      \left(\mathbf h_i^{(2)}-\bar{\mathbf h}^{(2)}\right),
\qquad
\mathbf p_0=P\!\left([\bar{\mathbf h}^{(1)};\mathbf p_2]\right).
$$

其中 $P$ 是 $128\to64$ 的线性投影。权重均匀时，$\mathbf p_2$ 回到第二层节点均值。原 C 使用 $\mathbf p_0$ 进行融合，最终模型 B 在此表示上加入报文转移读出的残差。 [原 C 节点池化实现](https://github.com/wangxin0721/XG-NID/blob/802cc0d03f10b41868cc88b4e345207f4877aad8/src/xgnid/flow_query_pooling.py) · [原 C 训练归档](https://github.com/wangxin0721/XG-NID/blob/be62f807c57e7907f99f33dadfb33ffe400d9ee1/outputs/joint_cross_layer_query_s42_v1/report.md)

### 1.3 Packet 均值引导的报文转移读出

对每条流，从已有 Packet→Packet 边中选取按时间顺序的相邻有向边 $i\to j$，其中 $j=i+1$。转移读出只统计同一流内唯一且字段有效的正向边。报文方向与载荷长度按 Packet 节点索引从其所属的 Flow→Packet 边读取。反向边仍参与原图编码。对有效边构造 70 维输入

$$
\mathbf e_{ij}=
\left[
\mathbf h_j^{(2)}-\mathbf h_i^{(2)};
\operatorname{onehot}_4(2d_i+d_j);
\log(1+\Delta t_{ij});
\log(1+\ell_j)-\log(1+\ell_i)
\right].
$$

其中 $d_i,d_j\in\{0,1\}$ 是报文方向，$\Delta t_{ij}$ 以毫秒计，$\ell_i,\ell_j$ 是以字节计的载荷长度。零时间间隔和零载荷长度保留。方向组合由四维独热向量表示，最后一维保留载荷变化的符号。 [转移字段审计](https://github.com/wangxin0721/XG-NID/blob/f7dfb24090e63092657d18a6e2d7e1866b416870/outputs/packet_transition_diagnosis_s42_v1/field_audit.json)

将 $\mathbf e_{ij}$ 经 $70\to16$ 线性层和 GELU 映射为转移向量 $\mathbf t_{ij}$。在最终模型 B 中，查询取自第二层 Packet 节点均值，经 $64\to16$ 线性层得到 $\mathbf q=Q(\bar{\mathbf h}^{(2)})$。权重仅在当前流的有效转移集合 $\mathcal E_G$ 内归一化。

$$
\alpha_{ij}
=\operatorname{softmax}_{(i,j)\in\mathcal E_G}
  \left(\frac{\mathbf q^\top \mathbf t_{ij}}{\sqrt{16}}\right),
\qquad
\mathbf r_G=\sum_{(i,j)\in\mathcal E_G}\alpha_{ij}\mathbf t_{ij},
\qquad
\mathbf p=\mathbf p_0+W\mathbf r_G.
$$

$W$ 是无偏置的 $16\to64$ 线性映射。只有一条有效转移时，其权重为 1。没有有效转移时，$\mathbf r_G=\mathbf 0$，模型保留原 C 的 Packet 表示。训练开始时将 $W$ 置零，使模型在第 0 轮与原 C 等价。B 的转移读出新增 3,200 个可训练参数。 [转移读出实现](https://github.com/wangxin0721/XG-NID/blob/802cc0d03f10b41868cc88b4e345207f4877aad8/src/xgnid/packet_transition_readout.py) · [报文对构造与残差注入](https://github.com/wangxin0721/XG-NID/blob/802cc0d03f10b41868cc88b4e345207f4877aad8/scripts/run_packet_transition_readout.py)

### 1.4 D32 逐维动态融合

将 $\mathbf f$ 与 $\mathbf p$ 分别线性投影到 32 维，得到 $\mathbf z_f,\mathbf z_p$。门控输入包含两路投影、每一路的归一化熵集中度 $1-H(\operatorname{softmax}(\mathbf z))/\log 32$、每一路 softmax 最大与次大分量之差，以及两路投影的余弦相似度。门控网络为 $69\to16\to32$，使用 LeakyReLU 和逐维 Sigmoid，输出 $\mathbf g\in(0,1)^{32}$。融合表示为

$$
\mathbf z=\mathbf g\odot\mathbf z_f+(1-\mathbf g)\odot\mathbf z_p.
$$

最终由共享分类头 $32\to32\to8$ 输出八类 logits。加入报文转移后，门控继续根据更新后的两路表示计算权重。转移权重 $\alpha_{ij}$ 决定同一流内各报文对的汇总权重，门控权重 $\mathbf g$ 决定融合时各特征维度的两路占比。 [D32 实现](https://github.com/wangxin0721/XG-NID/blob/802cc0d03f10b41868cc88b4e345207f4877aad8/src/xgnid/fusion_dimension.py)

## 2 实验设置

原始图集合包含 185,557 张图。按包含节点、拓扑和边属性的精确输入去重后保留 134,515 张图。划分以节点特征和拓扑相同的图为组，将仅边属性不同的变体分配到同一分区。训练、验证、测试分别包含 107,621、13,446 和 13,448 张图。该规则约束的是图输入的跨分区重复。由于归档图缺少原始流与采集标识，流或采集层面的独立性未经验证。 [固定划分实现](https://github.com/wangxin0721/XG-NID/blob/802cc0d03f10b41868cc88b4e345207f4877aad8/scripts/build_grouped_split.py) · [评估协议](https://github.com/wangxin0721/XG-NID/blob/802cc0d03f10b41868cc88b4e345207f4877aad8/configs/joint_cross_layer_query_final_test_s42_v1.json)

E0-FT 为单编码器微调基线，N0 为双编码器等权融合基线。两组沿用各自种子的历史起点，使用相同的微调设置。Adam 学习率为 $10^{-4}$，训练 batch 为 512，最多 30 轮，早停耐心为 12 轮，并允许选中第 0 轮。主结果中的原 C 和 B 按主干种子 42、43、44 配对。seed42 的原 C 沿用第 48 轮的历史检查点，seed43、44 的原 C 从各自的 N0 出发，依次训练 D32 门控、跨层报文读出和 Flow 节点查询。每组 B 冻结对应原 C 的参数和批归一化统计，使用相同的 seed42 读出初始化与采样顺序，只训练新增转移读出。固定主干的转移查询和聚合消融共用 seed42 原 C，其中 B 与 C1 在每个读出种子内共享初始化和采样顺序。每次转移读出训练运行 30 轮，采用 Adam、$10^{-3}$ 的学习率、batch 512 和八类负对数似然。按完整验证集 Macro-F1 选模，包含第 0 轮，同分取最早轮次。固定测试以 batch 16 按清单顺序推理。 [基线微调协议](https://github.com/wangxin0721/XG-NID/blob/802cc0d03f10b41868cc88b4e345207f4877aad8/docs/fair_vector_20261029.md) · [独立主干训练](https://github.com/wangxin0721/XG-NID/blob/3adaefe728098a67075bac14832b325573c1ed84/scripts/run_independent_packet_transition.py) · [固定主干查询配对](https://github.com/wangxin0721/XG-NID/blob/c0e54cd167600cd011d70daa88dd48696aee6e2a/outputs/packet_transition_paired_seeds43_44_v1/report.md)

主要指标为八类 F1 的算术平均值 Macro-F1，同时报告 Accuracy、Weighted-F1 和逐类指标。每份测试检查点均由验证结果锁定，并先通过逐样本验证预测回放。配对比较按样本 ID 对齐。下文的多次结果以均值 ± 样本标准差表示，差值以百分点计。 [基线回放](https://github.com/wangxin0721/XG-NID/blob/3693c112c7dba9624e8343ba483104e466465b29/outputs/early_baselines_final_test_v1/validation_replay.json) · [转移模型回放](https://github.com/wangxin0721/XG-NID/blob/acc9d8f82eea3af6ff4af06c4629717366f96b83/outputs/packet_transition_final_test_v1/validation_replay.json)

## 3 主结果

**表 1　固定划分上的主结果（%）**

| 模型 | 运行数 | 验证 Macro-F1 | 测试 Accuracy | 测试 Macro-F1 | 测试 Weighted-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| E0-FT | 3 | 96.587731 ± 0.428260 | 97.692346 ± 0.282407 | 96.450865 ± 0.418785 | 97.681782 ± 0.294259 |
| N0 | 3 | 97.193702 ± 0.173022 | 98.133551 ± 0.104898 | 97.209246 ± 0.164371 | 98.139227 ± 0.110745 |
| 原 C | 3 | 97.665764 ± 0.236949 | 98.267400 ± 0.007436 | 97.424269 ± 0.062198 | 98.271956 ± 0.004868 |
| B | 3 | 97.710426 ± 0.223177 | 98.284751 ± 0.018714 | **97.454641 ± 0.074335** | 98.288468 ± 0.022264 |

在固定测试集上，N0 相对同 seed 的 E0-FT 分别提升 0.238382、0.703916 和 1.332847 个百分点，平均提升 $0.758382\pm0.549262$ 个百分点。该比较衡量两个编码器方案在相同微调设置下的整体差异，包含模型容量与表示方式的变化。N0 在三次测试中均减少了 WebBased→Recon 错分。反向错分一次减少、两次增加。 [基线固定测试与逐类结果](https://github.com/wangxin0721/XG-NID/blob/3693c112c7dba9624e8343ba483104e466465b29/outputs/early_baselines_final_test_v1/report.md)

B 在三个独立主干上的固定测试 Macro-F1 均值为 97.454641%。相对各自冻结的原 C，seed42、43、44 的配对变化依次为 +0.083160、+0.007955 和 0.000000 个百分点，平均提高 $0.030372\pm0.045889$ 个百分点。三个 B 的验证最佳检查点分别在第 25、4 和 0 轮。seed44 选中第 0 轮，转移残差为零，测试预测与对应的原 C 一致。 [独立主干固定测试](https://github.com/wangxin0721/XG-NID/blob/dacaa08f2119e06f29f224abf579002210ce5c3f/outputs/packet_transition_independent_final_test_v1/report.md) · [seed43 验证选模](https://github.com/wangxin0721/XG-NID/blob/dacaa08f2119e06f29f224abf579002210ce5c3f/outputs/packet_transition_independent_s43_v1/selection.json) · [seed44 验证选模](https://github.com/wangxin0721/XG-NID/blob/dacaa08f2119e06f29f224abf579002210ce5c3f/outputs/packet_transition_independent_s44_v1/selection.json)

B 相对原 C 的测试 WebBased F1 在三个主干上分别变化 +0.794586、+0.040847 和 0.000000 个百分点，平均增加 0.278478 个百分点。Recon→WebBased 错分分别由 44 降至 36 条、由 22 降至 20 条和保持 29 条。WebBased→Recon 错分分别由 11 降至 8 条、由 11 增至 12 条和保持 15 条。 [逐类指标与双向错分](https://github.com/wangxin0721/XG-NID/blob/dacaa08f2119e06f29f224abf579002210ce5c3f/outputs/packet_transition_independent_final_test_v1/summary.json)

## 4 消融实验

**表 2　融合与报文转移读出消融（Macro-F1 为百分数，配对差为百分点）**

| 对照 | 组别 | Seed | 验证 | 固定测试 |
| --- | --- | --- | ---: | ---: |
| 同投影、线性分类头 | 固定等权 | 42 | 96.415130 | 仅验证 |
| 同投影、线性分类头 | D32 动态门控 | 42 | 96.815205 | 仅验证 |
| 同投影、非线性分类头 | 固定等权 | 42 | 96.893939 | 仅验证 |
| 同投影、非线性分类头 | D32 动态门控 | 42 | 97.281326 | 仅验证 |
| 转移聚合 | U · 均匀聚合 | 42 | 97.645342 | 97.364073 |
| 转移聚合 | B · Packet 均值查询 | 42 | 97.677497 | 97.495599 |
| 转移聚合 | C1 · Flow 查询 | 42 | 97.718468 | 97.464150 |
| 转移查询来源（固定 seed42 主干） | B · Packet 均值查询 | 读出 42–44 | 97.685712 ± 0.024564 | 97.482370 ± 0.026623 |
| 转移查询来源（固定 seed42 主干） | C1 · Flow 查询 | 读出 42–44 | 97.676199 ± 0.058058 | 97.454541 ± 0.025273 |
| 逐维门控 | Static · 全部流共用 | 主干 42–44 | 97.498995 ± 0.139999 | 97.368513 ± 0.126285 |
| 逐维门控 | Dynamic · 随流调整 | 主干 42–44 | 97.619523 ± 0.180623 | 97.428506 ± 0.154405 |
| 逐维门控配对差 | Dynamic − Static | 主干 42–44 | +0.120527 ± 0.179513 | +0.059994 ± 0.129412（两升一降） |

**逐维动态门控。** 在同一对冻结投影上，将 D32 融合表示和固定等权表示分别交给同结构、同预算的分类头。以固定第 30 轮比较，D32 在验证集上比等权高 0.400075 个百分点（线性头）和 0.387387 个百分点（非线性头）。两种设置下 WebBased→Recon 错分分别减少 15 条和 22 条。以上结果衡量固定表示和匹配分类头条件下的融合方式。 [匹配融合验证消融](https://github.com/wangxin0721/XG-NID/blob/5a32ddde192c15a7486057c53dd6812ab90cbcf2/outputs/fusion_decision_diagnosis_s42_v1/report.md)

在三个独立训练的 B 主干上，进一步比较门控是否随流调整。每对共用冻结的编码器、报文节点与转移读出和两路投影，复制同一门控网络与分类头，将门控末层置零，使两组均从每维权重 0.5 开始。Static 使用训练集 69 维门控输入的均值，为所有流产生同一套可学习的逐维权重。Dynamic 使用当前流自己的 69 维输入。两组各训练门控与分类头的 2,984 个参数，使用同一采样顺序与 6,330 次更新，按验证 Macro-F1 选模。Dynamic 相对 Static 的测试 Macro-F1 配对变化依次为 −0.086256、+0.159686 和 +0.106551 个百分点，平均提高 0.0600 个百分点。三对的 Recon↔WebBased 双向错分总数分别从 53、42、49 条降至 45、30、47 条。seed42 上 BruteForce F1 下降 1.038 个百分点，是该种子 Macro-F1 下滑的主要来源。 [逐维门控固定测试](https://github.com/wangxin0721/XG-NID/blob/dacaa08f2119e06f29f224abf579002210ce5c3f/outputs/final_d32_gate_ablation_v1/test/report.md) · [逐类指标](https://github.com/wangxin0721/XG-NID/blob/dacaa08f2119e06f29f224abf579002210ce5c3f/outputs/final_d32_gate_ablation_v1/test/summary.json)

**报文转移聚合。** U 单独训练转移映射，并在每条流内对有效转移等权汇总。seed42 下，U 相对原 C 的验证分数提高 0.091539 个百分点，测试分数下降 0.048366 个百分点。B 在同一 seed 的固定测试上比 U 高 0.131526 个百分点。U 仅评估 seed42。对已训练 C1 的同权重推理消融中，临时把 Flow 注意力替换为等权使验证 Macro-F1 下降 1.070370 个百分点，112 条改判均发生在至少有两个有效报文对的流中。这说明该检查点的预测依赖训练形成的权重分布。 [同权重回放及 U 训练](https://github.com/wangxin0721/XG-NID/blob/d94b4cef447d3dc90580b97c07d998503acb3a22/outputs/packet_transition_attention_diagnosis_s42_v1/report.md) · [聚合方式固定测试](https://github.com/wangxin0721/XG-NID/blob/acc9d8f82eea3af6ff4af06c4629717366f96b83/outputs/packet_transition_final_test_v1/report.md)

**查询来源。** C1 与 B 使用相同转移输入和参数量，C1 以 Flow 表示生成转移查询，B 以 Packet 均值生成。C1 在 seed42 的最佳验证轮次比 B 高 0.040971 个百分点，seed43、44 分别低 0.056307 和 0.013204 个百分点。固定第 30 轮的验证比较中，B 在三个 seed 均得分更高。采用各自验证最佳检查点进行固定测试时，B 在三个配对中均领先，C1−B 平均为 $-0.027828\pm0.003140$ 个百分点。该结果支持在本实验设置下采用 Packet 均值作为转移查询。 [三 seed 验证配对](https://github.com/wangxin0721/XG-NID/blob/c0e54cd167600cd011d70daa88dd48696aee6e2a/outputs/packet_transition_paired_seeds43_44_v1/report.md) · [三 seed 固定测试配对](https://github.com/wangxin0721/XG-NID/blob/acc9d8f82eea3af6ff4af06c4629717366f96b83/outputs/packet_transition_final_test_v1/report.md)

## 5 结论范围与补充实验

双编码器基线 N0 相对单编码器基线 E0-FT 在三个种子上均提高测试 Macro-F1，平均提高 0.758382 个百分点。在对应的冻结主干上，报文转移读出平均增加 0.030372 个百分点，两组改善、一组持平。逐流动态门控相对可学习的静态门控平均增加 0.0600 个百分点，两组改善、一组下降。后两项是各自对照下的小幅增量，且随主干而变。三个主干独立训练，各比较沿用同一固定划分。结果反映了训练种子变化下的表现，适用范围限于当前数据和划分。
