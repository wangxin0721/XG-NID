# XG-NID 当前不足与有序双尺度转移读出实验

## 已有证据

主模型 B 在三个独立主干上的固定测试 Macro-F1 均值为 97.454641%，相对原 C 平均提高 0.030372 个百分点。seed42 提高 0.083160，seed43 提高 0.007955，seed44 验证集选中残差为零的第 0 轮。新增读出收益随主干变化。[独立主干报告](https://github.com/wangxin0721/XG-NID/blob/dacaa08f2119e06f29f224abf579002210ce5c3f/outputs/packet_transition_independent_final_test_v1/report.md)

Dynamic 对 Static 的固定测试配对平均差为 +0.059994 个百分点，样本标准差为 0.129412，三个种子两升一降。seed42 的 BruteForce F1 下降约 1.038 个百分点，表明修复两类互扰时可能增加其他类别的误报。两路门控及整体分类头的取舍需要一起观察。[门控消融报告](https://github.com/wangxin0721/XG-NID/blob/dacaa08f2119e06f29f224abf579002210ce5c3f/outputs/final_d32_gate_ablation_v1/test/report.md)

## 当前不足

| 不足 | 证据范围 | 对后续设计的影响 |
| --- | --- | --- |
| 新增转移模块的增益小 | 独立主干两升一平，平均 +0.030372 个百分点 | 需要增加有判别力的行为表示，并配同预算对照 |
| 现有转移读出缺少连续变化的组合 | 固定已有节点表示后，读出分别编码每个相邻边，再做加权和，边集合重排保持输出 | 增加共享中间报文的连续两次转移，以有序组合表示短行为片段 |
| 报文编码器已有顺序上下文，但新增读出只用一步边 | 编码阶段的图消息传递含相邻报文关系 | 假设是读出端还可显式补充顺序，真实收益需要实验验证 |
| 读出训练冻结编码器、投影和门控 | 冻结新增模块实验协议 | 若结构有效，再用匹配微调检验新表示能否与融合层协同 |
| 容易只改善 Recon/WebBased | seed42 门控消融出现 BruteForce 误报增加 | 全八类 Macro-F1 为主，逐类修复、退步和误报数必须保留 |
| 广义双模态自适应融合已有相近方法 | XG-NID 已融合流和报文，XMF-GNN 使用跨模态注意力 | 论文应定位具体的保序片段读出和双尺度残差设计，比较已有方法 |
| 固定测试被多轮查看 | 当前实验历史 | 新增实验先按验证集选择方案，后续固定测试结果作为本划分上的补充结果 |

XMF-GNN 的两层异构图编码及跨模态注意力与本项目主题接近。TB-Graph 通过方向和 burst 关系表达报文交互，并引入位置编码。二者提供相关设计依据。当前方案使用原图的报文表示，在读出端编码连续两次转移，具体差异仍需结合消融和相关工作全文讨论。[XMF-GNN](https://doi.org/10.1016/j.neucom.2025.131285) · [TB-Graph](https://www.techscience.com/cmc/v82n2/59498/html) · [XG-NID](https://arxiv.org/html/2408.16021v2)

## 本轮具体改进

沿用原有 70 维一步转移输入和第二层 Packet 均值查询。已有的节点级 Flow 查询继续参与冻结原 C 的编码。

记相邻报文的转移向量为

$$
\mathbf v_i=\operatorname{GELU}(E\mathbf e_{i,i+1}),\qquad \mathbf v_i\in\mathbb R^{16}.
$$

增加连续两次转移的有序表示

$$
\mathbf c_i=\operatorname{GELU}\left(C[\mathbf v_i;\mathbf v_{i+1}]\right).
$$

它显式保留先后两次变化的角色。只有两条有效正向边共享同一个中间报文时才创建片段。无效边或缺失边之间无法拼出连续片段。每个流分别进行一步和两步的注意力汇总。对存在两步片段的流，两个尺度的汇总向量取均值，再经零初始化的 16→64 输出投影形成残差，加入原 C 的 Packet 表示。没有两步片段时，使用已有一步读出。所有尺度仍使用 Packet 均值查询。

图编码器中已有局部上下文，因此“读出输出对边集合排列不敏感”描述的是固定节点表示条件下的读出性质。它提示了需要检验的结构缺口。该方案尚无真实数据训练结果。

## 配对组别

| 组别 | 一步信息 | 第二尺度 | 新增读出参数 |
| --- | --- | --- | ---: |
| Single | 原转移读出 | 无 | 3,200 |
| Pointwise | 同 Single | 在与 Ordered 完全相同的片段位置，用 [v_i;v_i] 编码 | 3,728 |
| Ordered | 同 Single | 在有效连续片段上，用 [v_i;v_{i+1}] 编码 | 3,728 |

Pointwise 与 Ordered 的片段数量、网络层、参数量、初始化、采样顺序和预算相同。两者的差别是第二个位置是否提供真正的下一次变化。Single 提供与已有 B 相同结构的匹配重训参考。

## 执行安排

1. 先跑 seed42 上三组训练与验证。冻结同一原 C，三组使用 seed42 的读出初始化和每轮采样顺序。每组 Adam lr=0.001、batch512、30 轮、6,330 次更新。允许验证集选中第 0 轮，同分取最早。
2. 输出原 C 验证逐样本回放、每类有效两步片段覆盖率、每轮训练账本、最佳及末轮权重、八类 Precision/Recall/F1、双向错分、修复与退步。
3. 本轮建议的投入参考是 Ordered 在验证集上比 Single 高 0.10 个百分点、比同参数量 Pointwise 高 0.05 个百分点。结合 BruteForce 的完整误报、漏报和其他类变化决定是否继续。该参考用于安排计算预算。
4. 若结果支持继续，按同一配置补 seed43、44 的配对训练。三套独立原 C 来自各自 N0 训练链。
5. 三个种子的全部验证选模锁定后，运行固定测试入口。报告所有种子和配对均值、标准差，披露当前固定测试已被历史实验使用。
6. 如 Ordered 显示收益，再做有序片段与流内打乱片段的同预算训练对照，进一步检验真实相邻关系的作用。随后才考虑两组共同解冻融合投影和门控的低学习率微调。

监督对比学习可用于后续验证类别间间隔，但本轮先用已有八类 NLL 检验读出结构。SupCon 已提出同类聚合、异类分离的通用训练目标，作为后续参考。[SupCon](https://proceedings.neurips.cc/paper_files/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html)

## 服务器启动

在原训练服务器激活 wx 环境，进入既有 XG-NID 仓库。代码使用单独工作树，读取既有仓库的权重、split 和图。下面的训练与验证入口运行结束后自动上传本轮目录内的全部权重、预测、账本和日志到 codex/server-outputs。

```sh
cd '/home/nzxn/下载/WX/load/XG-NID'
xgnid_repo="$PWD"
xgnid_code="$(dirname "$PWD")/XG-NID-ordered-transition"
git fetch --no-tags origin refs/heads/codex/ordered-transition-readout
git worktree add --detach "$xgnid_code" FETCH_HEAD
nohup /bin/sh "$xgnid_code/scripts/launch_ordered_transition_readout.sh" "$xgnid_repo" 42 > "$xgnid_repo/outputs/ordered_transition_launcher.log" 2>&1 < /dev/null &
```

在结果支持继续时，依次运行下面的后续训练和固定测试。每次入口结束后都会上传该实验目录的全部产物。上传使用仓库已有发布逻辑，保留历史结果。

```sh
/bin/sh "$xgnid_code/scripts/launch_ordered_transition_readout.sh" "$xgnid_repo" 43
/bin/sh "$xgnid_code/scripts/launch_ordered_transition_readout.sh" "$xgnid_repo" 44
/bin/sh "$xgnid_code/scripts/launch_ordered_transition_readout.sh" "$xgnid_repo" test
```

主报告在 outputs/ordered_transition_readout_v1/seed42/report.md。固定测试报告在 outputs/ordered_transition_readout_v1/test/report.md。训练失败时，上传脚本会保留并上传失败状态与日志。网络上传失败可单独重试。

```sh
python -B "$xgnid_code/scripts/sync_server_outputs.py" publish --repo "$xgnid_repo" --only ordered_transition_readout_v1
```

本地已完成模块实现和 CPU 结构验证。服务器正式训练和分数待运行。
