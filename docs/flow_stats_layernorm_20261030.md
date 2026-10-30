# 隐藏层LayerNorm：一次受限的seed42残差候选

本轮只给统计MLP隐藏层增加LayerNorm(64)，不改变编码器、投影、门控、分类头、BN行为、数据预处理、普通融合NLL、各参数组学习率和选模预算。旧实验代码、配置、检查点和结果保持原样。两路仍读取完整异构图，统计路径仍只使用现有Flow特征。

## 模型和起点

独立模型src/xgnid/flow_stats_layernorm.py继承原残差类。MLP从Linear(D,64)→LeakyReLU→Linear(64,64)变为Linear(D,64)→LayerNorm(64,eps=1e-5,elementwise_affine=True)→LeakyReLU→Linear(64,64)。不增加输入归一化、Dropout、其他归一化层或可学习缩放。LayerNorm的两组64维仿射参数增加128个参数，D=82时新路径共9600个。MLP末层索引由2变成3；新探测逐个检查0、1、3层参数，允许第一步首层和LayerNorm梯度为零，要求后续步骤获得梯度并更新。

LayerNorm在每个Flow节点的64维隐藏特征上计算，训练/评价采用相同的归一化计算：[原始论文](https://arxiv.org/abs/1607.06450)。这项性质不保证训练后的残差幅度始终小，也不保证分类指标提高。当前数据每图一个82维Flow节点，原均值池化保留。

重新使用原N0-s42公共参数和选定冻结V2第13轮门控(SHA256=ebc151b5bd60f3d9fc6808d6f1b551ef5559da439ea109487ecded31929d06a7)。不从失败残差last.pt或小学习率J2最佳权重继续。先严格加载N0和V2，再加新模块，末层权重/偏置为0，恢复模块构建前的Python/CPU/CUDA RNG。整份验证集初始输出(容差1e-5)、逐样本预测和混淆矩阵须复现97.093667251942%。沿用真实图回放原小学习率J2对照和新结构检查点严格保存/重建检查。

新配置configs/flow_stats_layernorm_s42_v1.json绑定旧残差配置SHA256=b7f9c2a8b3b1a0a4a5404010046001e6a992f7a63da643df8df3673e446ce74b；除本轮命名、隐藏层LN和新增探测规则外，逐字段要求与旧配置一致。原恢复队列V2来源、缓存和权重完整性校验继续复用，不重跑预热。训练入口通过独立FunctionType命名空间复用原真实图流程和joint.train主体，不改历史模块全局变量。

## 预先固定的训练前扰动检查

固定取seed42采样器的前三个训练batch，各512个样本，执行3次探测更新。正式训练前的稳定性界限如下，适用于每次更新后当前batch，以及始终相同的第一batch；所有累计功能变化均与干净起点在同样样本上的eval输出比较。

| 量 | 上限 |
|---|---:|
| 残差/图表示L2范数比中位数 | 0.10 |
| 范数比95%分位数 | 0.25 |
| 范数比最大值 | 1.00 |
| 同批eval平均NLL相对干净起点的增加 | 0.10 |
| 同批预测相对干净起点的改变比例 | 0.05 |

这些是预先登记的启动筛选界限，尚未经验证为最优值；它们不是显著性检验，也不是控制完整训练轨迹的理论保证。门控变化只记录，不单独用它证明有效或无效。初始BN与公共模块也可能影响功能变化，因此触发界限不能单独归因于LN或统计路径。

每步更新前后均切换eval，记录同批NLL、预测改变、输出最大差、8维门控平均/最大变化及范数比中位数/95%分位数/最大值。另保留固定第一batch的跨步骤轨迹，避免把不同batch损失当成同批曲线。真正反向时仍为原联合train模式，编码器BN计数须按3次更新增加。所有公共模块及新增MLP均检查有限梯度和实际参数改变；MLP每个张量都记录，按层验收后续梯度和更新，避免把FP32微小偏置更新的舍入误判为无梯度。旧停用模块不得改变。

任一步超界即结束探测；finally路径无论通过、拒绝还是异常都恢复参数、BN、梯度、eval模式和RNG，丢弃探测Adam。保存probe.json和probe_snapshots.pt，后者包含固定样本ID、标签、逐样本输出、8维门控及范数，便于重查。拒绝时status=preflight_rejected、exit_code=2、formal_training_updates=0，不创建正式训练目录、不自动降低阈值或换配置重试。

通过后还必须通过新结构的严格保存/重建回放检查，才进入完整训练。

## 一次正式训练与观测

只新增J2_flow_stats_ln一次seed42训练。Adam重置，weight_decay=0；编码器lr=1e-6、投影/共享头lr=1e-5、门控lr=1e-5、新MLP(含LN仿射参数)lr=1e-4。参数正确分组去重。train512/eval16，最多30轮，patience12，严格验证Macro-F1改善，第0轮参与。公共轮次采样顺序在更新前及训练完成后与原小学习率J2比对。测试图禁止加载。

为观察“LN不能保证最终残差受控”，在原验证前向上附加只读hook，逐轮记录门控和范数分布，不增加验证前向次数，不改变优化器、早停规则或候选选择；epoch_scale_diagnostics.jsonl保存第0轮、每轮和最佳权重回放的观测，last_validation也保存末轮尺度统计。观测会增加少量同步/统计耗时，因此不据本轮耗时做独立速度结论。完整训练期间这些分布只作观测，不新增事后停止门槛。

沿用原预算筛选：超过冻结V2与小学习率J2中较强者至少0.10个百分点，当前约97.220712%以上，再人工审查八类取舍。报告不能把整体收益全部归给门控或LN；LN也增加128个参数。若稳定但没有收益，结束该实现，再讨论32维融合；本轮不自动启动新结构或其他seed。

## 本地验证边界与服务器交付

本地无torch/PyG，只执行无torch配置、来源哈希、分组、拒绝门槛、隔离命名空间、观测转发、失败退出和报告测试，以及Python3.9/Bash语法检查。真实等价、LN计算、反向、参数更新和完整权重回放未在本地执行，由服务器先检查；未执行的项目不记为通过。

输出目录独立为outputs/flow_stats_layernorm_joint_s42_v1，拒绝覆盖目录/日志/PID。原模型文件和全部历史配置不改；归档必须同时包含原残差和新LN类两个src文件。助手提交推送后，使用交付的固定提交号替换REV：

```bash
cd /home/nzxn/下载/WX/load/XG-NID
git fetch origin codex/r0-edge-preparation
bundle=$(mktemp -d /tmp/xgnid_stats_ln.XXXXXX)
git -c core.autocrlf=false archive REV scripts configs src/xgnid/flow_stats_residual.py src/xgnid/flow_stats_layernorm.py | tar -x -C "$bundle"
bash "$bundle/scripts/launch_flow_stats_layernorm.sh" 1
```

默认物理GPU1，允许共享，不停止现有任务。

```bash
tail -n 50 -F outputs/flow_stats_layernorm_joint_s42_v1.log
cat outputs/flow_stats_layernorm_joint_s42_v1/status.json
cat outputs/flow_stats_layernorm_joint_s42_v1/progress.json
```

若拒绝，提供status.json和probe.json即可定位触发量。若完成，提供status.json、report.md和analysis.json；包含N0/V2/旧J0/旧J2与独立LN候选的指标、完整逐类结果、修复/退步、最佳/停止轮次、更新量、训练耗时及预热/探测单列记账。历史缺失耗时保持null，额外预热不与对照宣称总训练量相等。独立汇总入口scripts/summarize_flow_stats_layernorm.py ROOT可重新生成新报告。
