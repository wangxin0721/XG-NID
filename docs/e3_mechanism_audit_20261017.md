# 冻结 E2/E3 机制核查（2026-10-17）

仅核查现有 `edge_experiments.py`、`model.py`、`train_grouped.py`，未改模型、损失、历史类名或配置。合成CPU前向公式一致性已通过；未加载服务器真实E3检查点，不给出门控有效性的实证结论。

## 路径、维度与公式

构造器为`build_experiment('E2'/'E3')`，返回`DualBranchLogitGatedHeteroGNN_Edge(hidden_dim=64)`，各辅助权重全为0。两个编码器参数独立，都接收**完整**Flow/Packet异构图和4/4/1边属性；两层GAT、跨关系mean聚合、BN eps1和LeakyReLU。

Flow编码器只池化其Flow输出得到h_f∈R64；Packet编码器只池化其Packet输出得到h_p∈R64。它们不是隔离的单模态网络。特别是每个分支末层通向未池化节点类型的部分卷积不参与最终损失，不能仅以总参数解释有效容量。

两个Linear64→8分别得到z_f/z_p（源码变量flow_logits/packet_logits）。它们是类别数宽度的**中间表示**，实际训练没有独立分支分类监督；不能仅凭变量名认定每个坐标具有确定类别概率含义。

对z的代理量：p=softmax(z)；c=1−[−Σp log(max(p,1e−8))]/log(8)；m=top1(p)−top2(p)；a=cosine_similarity(z_f,z_p)。c是熵补量，m是Softmax间隔，均未校准成经验正确率，也不是已证实可靠性估计。

门控输入[z_f,z_p,c_f,c_p,m_f,m_p,a]共21维，经Linear21→16、LeakyReLU、Linear16→2、Softmax得到g。E3实际权重不是直接g，而是u=0.9g+0.1[c_f,c_p]，w=u/max(sum(u),1e−8)。这是可微计算，**没有detach**，也没有单独的门控平滑正则损失。均匀z时c≈0仍由0.9g保证正常归一化。

融合z=w_f z_f+w_p z_p，`shared_head`为Linear8→32、LeakyReLU、Linear32→8，得到最终分类logits，再log_softmax作为NLL输入。辅助计算的fused_emb=w_f h_f+w_p h_p为64维；它只供辅助头/cache，不是最终shared_head的输入。

E2的use_gate=False直接令w=(0.5,0.5)，跳过g和0.9/0.1混合；c/m/a仍计算。其编码器、投影、shared_head、门控及辅助头定义与E3相同，门控参数保留但E2不训练它。E2/E3初始化及预算配对由冻结入口核验。E0与E2不构成仅有门控一个变量的比较，因为双编码器/池化/头部/容量均不同。

## 实际训练目标

`train_grouped.py`直接执行`F.nll_loss(model(batch), batch.y.view(-1))`，并未调用`model.loss(...)`。因此历史类的辅助分类损失、agreement项、困难类权重等都不进入冻结训练。最终NLL梯度经shared_head、融合、分支投影/编码器传播；E3另经g及c/m/a路径传播。辅助二分类头虽然在forward中算出/cache保存，未连接主损失，梯度为None；不能把计算过输出等同于参与训练。

E2/E3区别是门控路径及其影响的优化；训练目标形式相同。c/m/a没有独立监督。后续检测梯度必须分别报告None/全零/非零有限，不能要求所有模块都更新。E0参考HeteroConv是默认sum，自定义双分支是mean，这也是结构差异之一。

## 独立评估工具必须导出的字段

| 类别 | 必需字段与约束 |
|---|---|
| 运行身份 | experiment、seed、Git SHA、源码/配置SHA、checkpoint SHA/epoch、split ID、samples SHA、partition、设备/torch/PyG版本、类别顺序、干预版本 |
| 样本身份 | sample_id、manifest_index、group_id、真实label、来源路径/offset（可单独保留在服务器）、Edge/NoEdge哈希；所有干预逐样本对齐 |
| 原始表示 | h_f/h_p[64]、z_f/z_p[8]；逐坐标原值，不能只导出argmax |
| 代理量 | softmax(z_f/z_p)、熵、归一化c_f/c_p、m_f/m_p、余弦a；标注“表示代理”，不称分支校准置信度 |
| 门控与融合 | raw gate logits、原始g[2]、混合后u[2]、最终w[2]、融合z[8]、fused_emb[64]；权重有限/非负/和≈1 |
| 分类与干预 | H(z_f)、H(z_p)、H(0.5z_f+0.5z_p)、H(αz_f+(1−α)z_p)、H(w_fz_f+w_pz_p)的logits、Softmax、logprob、预测/对错；原始z的argmax仅另列诊断 |
| α选择 | 仅验证集选择的α及候选网格/并列规则/验证指标/冻结记录；测试不得重新选α；单路H(z)属于同模型推断干预 |
| 汇总诊断 | per-class/support/混淆矩阵/Macro与Weighted-F1、两路同对同错/互补比例、按类别及对错组合的权重分布、可靠性分箱/校准误差；区分原始z代理与H(z)预测 |
| 梯度/容量/成本 | 主损失、各模块None/零/非零有限梯度及实际更新参数、总/有效路径参数；计时含义/预热/重复次数/峰值显存；诊断反向单独运行，不能修改冻结检查点 |

一致性门槛：重算上述公式与冻结forward的logprob一致；E2权重恒0.5；加载检查点前后原预测一致；样本数/顺序/label与清单一致；所有未修改图的原始哈希一致。验证和测试输出分别存放，测试访问记账。原始z无独立监督不证明它必然没有类别语义，需后续验证集诊断；本次不提前实现评估工具或M1。
