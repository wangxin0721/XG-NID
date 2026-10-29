# 固定N0表示的F0／F1／F2筛选

代码与协议于2026-10-29实现。本机无torch/PyG，完成6项标准库检查，另1项真实张量/梯度检查跳过。未在服务器加载N0权重或启动训练。助手负责核验、提交和推送；用户只执行服务器拉取与启动命令。优先物理GPU1，可显式允许共享。

## 起点和边界

固定已完成的seed42 N0最佳检查点：outputs/e2_concat_pair_b512_s42_lr1e4_v1/N0/best.pt，SHA256为ae65d352ff927026bfcada8759b62f1b22c5d8f0166103d2a2d556618e7a3563。它是N0新增第5轮，验证Macro-F1=96.99623354279772%，Accuracy=97.82091328276067%。注册文件保存用户提供的完整N0指标、权重/预测哈希及旧研究binding，服务器启动前逐项比对实际文件。

本轮只有F1/F2两项训练。F0是等权复现与缓存一致性检查，无优化器和更新。两套编码器、投影层、共享H、历史未使用辅助头与旧gate均requires_grad=False并保持eval。每次缓存/探测/训练后核对全部冻结参数及缓冲区哈希；每次更新确认冻结参数无梯度且所有冻结模块仍为eval。

| 项目 | F0 | F1 | F2 |
|---|---|---|---|
| 权重 | 固定.5/.5 | sigmoid(s)，全局标量 | 原21→16→2门控softmax |
| 初态 | 等权 | s=0 | 复制N0未使用gate的第一层，末层W/b置零 |
| 可训练参数 | 0 | 1 | 386 |
| 最终分类 | H(.5zf+.5zp) | H(alpha*zf+(1-alpha)*zp) | 同F1，alpha逐样本变化 |

F2输入顺序严格沿用旧gate：zf八维、zp八维、两路熵统计、两路margin、两路余弦相似度，共21维。统计直接取8维表示，不改成H(z)预测统计，也不混入原.9/.1代理权重。不加入辅助损失、类别加权、校准、新编码器或拼接头。

已有seed42测试结果曾用于前期研究，本轮是验证集探索，不宣称独立确认或未见测试结果；本次不加载测试图、不评估测试集。

## 缓存、梯度和预算

先按冻结manifest顺序处理13446个验证样本。在模型eval及no_grad下，原始N0前向与显式H(.5zf+.5zp)的log-probability要求allclose(atol=rtol=1e-5)；验证ID/标签顺序、逐样本预测、混淆矩阵及Macro-F1必须复现N0。缓存重放F0也要求预测完全一致。F0通过后才缓存107621个训练样本的zf、zp及21维特征；图缓存batch16，不改变BN统计，两项读取同一份无梯度缓存。

固定设置：seed42、Adam lr=.001、weight_decay=0，缓存训练batch512、评价batch16、最多30轮、patience12、严格验证Macro-F1选模。epoch0纳入候选，同分不替换。各项每完整轮211次更新，最多6330次更新；独立早停，报告实际预算与公共轮次顺序。该预算只优化融合，不与此前全模型训练预算等同。

训练时缓存张量已detach，但不对融合→冻结H→NLL这条路径使用no_grad。探测显式计算loss对融合表示的梯度，检查有限且非零；检查融合参数实际更新，F2首层在零末层初始化后的后续batch获得梯度。探测使用同样3个训练batch，随后销毁探测融合器和Adam，恢复公共RNG并重建干净融合器开始正式训练。全程不利用验证标签拟合任何参数或统计，验证只用于复现、选模及最终报告。

每项保存epoch0/best.pt、last.pt、epochs.jsonl、metrics.json及逐样本验证预测。best.pt仅包含融合参数，重建时必须同时加载登记的固定N0。last.pt保存Adam、采样器及RNG，但当前没有resume入口；禁止覆盖既有输出或自动重试。

## 服务器执行

在原服务器仓库普通提示符下执行，使用助手交付的固定提交号替换下面的rev。原模型、已有实验入口与历史结果无需切换或覆盖。启动器固定使用/home/nzxn/miniconda3/envs/wx/bin/python；如原环境位置不同，用XGNID_PYTHON指定已有解释器，不安装替换CUDA环境。

```bash
cd /home/nzxn/下载/WX/load/XG-NID
git fetch origin codex/r0-edge-preparation
rev=$(git rev-parse FETCH_HEAD)
bundle=$(mktemp -d /tmp/xgnid_frozen_fusion.XXXXXX)
git -c core.autocrlf=false archive "$rev" scripts configs | tar -x -C "$bundle"
bash "$bundle/scripts/launch_frozen_fusion.sh" 1 --allow-shared-gpu
```

队列顺序：元数据核验 → F0与验证缓存 → 训练缓存 → F1/F2短探测 → 干净F1训练 → 干净F2训练 → 配对汇总。任何错误停止，不增加其他seed或结构。共享GPU的耗时不能作为独占速度基准。

```bash
tail -n 30 -F outputs/frozen_fusion_n0_b512_s42_v1.log
cat outputs/frozen_fusion_n0_b512_s42_v1/status.json
watch -n 5 cat outputs/frozen_fusion_n0_b512_s42_v1/progress.json
```

Ctrl+C只退出tail/watch。状态与结果文件包括hardware.json、F0.json、binding.json、probe.json、train_cache.pt、val_cache.pt、F1/、F2/和根summary.json。样本及缓存仅存outputs，不提交。失败时根status.json为failed并保留error.log。

## 结果判定

报告F0/F1/F2整体Accuracy、Macro/weighted-F1、完整逐类Precision/Recall/F1及矩阵；F1/F2分别报告相对F0的百分点差、按ID匹配的修复/退步及逐类计数，另报告F2−F1和对应配对计数。记录验证权重的均值/标准差/范围、可训练参数、实际更新数、冻结模型不变证据与公共轮次采样顺序。

F1超过F0而F2没有额外收益时，优先全局权重；F2同时超过F0/F1且幅度有竞争力后再补跨seed验证。仅类似N1的微小领先不认定有效。两者未超过F0则保留N0、结束当前标量门控候选。方案确认前不自动运行测试或追加配置。

跨seed时使用各seed自身E2检查点、先按相同微调流程获得各自N0，再重复冻结实验；最终再补训练机会相当的E0与必要结构消融。本轮不实施这些后续步骤。
