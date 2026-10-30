# Flow统计直达路径：一次seed42结构筛选

采用用户最后收敛的方案：只新增一次联合训练，复用已完成的小学习率J2作为结构对照。前一段提出的“冻结编码器C0/C1两次训练”不在本轮执行。两种训练方法不能混为同一对照。本轮编码器正常训练、BatchNorm沿用J2的训练行为；不把统计冻结加入结构变化。

## 结构、输入与初始化

独立模块src/xgnid/flow_stats_residual.py保留原E2/N0公共状态键，先严格加载N0，再严格加载恢复队列V2的21→16→8逐维门控，最后附加flow_stats_mlp。它对已有预处理Flow输入执行Linear(D,64)→LeakyReLU(默认负斜率0.01)→Linear(64,64)，按Flow.batch均值池化，加入原Flow图表示后再通过原flow_classifier(64,8)。Packet路径、门控21维统计定义、8维融合和共享8→32→8分类头不变。没有LayerNorm、BatchNorm、Dropout、新损失或额外可训练缩放系数。

最新简化版没有LayerNorm，所以参数增量为64D+4224；D=82时为9,472，并非带LayerNorm版的9,600。D在服务器从真实图读取，使用原编码器的float和_align_feature_dim规则；原编码器未指定固定维度时要求观测宽度一致。现有FrozenGraphs.load实际要求每图Flow形状为(1,82)，Packet宽1500：这批数据中Flow均值池化等价于取唯一节点的MLP输出。两套图编码器仍读取完整异构图，不能描述成只分别读取两类节点。

MLP末层权重和偏置归零，第一层正常初始化，残差系数固定1。添加新模块后恢复Python/CPU/CUDA随机状态，以免改变公共模块的随机序列。所有初始残差必须为0，整份验证集的输出(容差atol/rtol=1e-5)、预测顺序、逐样本结果、混淆矩阵和Macro-F1均须复现选定V2。

## 起点、对照与预算

配置configs/flow_stats_residual_s42_v1.json注册唯一候选J2_flow_stats。起点是原N0-s42公共参数加outputs/fair_vector_five_runs_recovery_v1/V2/V2/best.pt门控；不是小学习率J2最佳权重。V2为第13轮，SHA256=ebc151b5bd60f3d9fc6808d6f1b551ef5559da439ea109487ecded31929d06a7，验证97.093667251942%。恢复队列、配置绑定、缓存哈希和原V2筛选报告沿用现有完整性校验，仅复用，不重跑预热。

结构对照outputs/joint_gate_lrscale01_s42_v1/J2已完成：第9轮97.12071203419106%，最佳权重SHA256=097cef933f3b0b049d8f0c31c3a8e99dedae7a25ccc8d00c4c9d23f86bd936eb。其公共配置SHA256=b1d2cdbeff2489f2fdba33b6d806524edb56f5db64caee5fb6e11b93469a7368，训练脚本SHA256=b0fb58aebc6483ccda52f52df2d537ee191feebe85ed85f7ee3912b3267da871。运行前核对这些真实文件、完成状态、N0/V2身份、历史逐轮选模和样本顺序；真实图回放旧J2最佳验证结果。旧权重只用于对照回放，不进入新训练起点。

新候选完整复用a6e2088中的joint.train函数主体，仅在独立FunctionType命名空间中替换模型构建及参数分组，不修改历史函数全局变量。训练参数为Adam重置、weight_decay=0、普通融合NLL、seed42、train512/eval16、最多30轮、patience12、严格验证Macro-F1改善、第0轮参与。编码器lr=1e-6，分支投影/共享头lr=1e-5，门控lr=1e-5，新增MLP独立组lr=1e-4。同组共享参数去重，跨不同学习率组的共享参数拒绝执行；所有可训练参数必须入组，旧辅助头和旧标量门控保持停用。

每个公共轮次在首个梯度更新之前比对旧J2采样顺序哈希，再在完成后复核。早停不要求同轮发生。若新候选超过旧J2的21轮，后续仍按同一seed42采样器继续到预先固定预算。旧J2不重跑；预算相同不意味着总更新次数、运行耗时或新增参数量相同。

## 本地与服务器验证边界

本地无torch/PyG，只做Python3.9语法、真实配置/脚本哈希、参数分组逻辑、顺序防护、损坏来源拒绝、选模和报告导出测试，以及Bash语法检查；不声称真实前向/反向或实际检查点回放已通过。本地不启动训练，也不修改服务器任务。

服务器正式训练前自动执行：

1. 从真实图整份验证集复现冻结V2；检查新增路径为0、公共状态相同、公共优化器组相同及初始数值等价；复现旧小学习率J2检查点。
2. 三个真实train batch的前向、反向及实际Adam更新；两路编码器、两个分支层、共享头、门控都有有限有效梯度并更新；新增MLP每个参数在后续更新中获得梯度且改变，残差不再恒为0。首步新MLP第一层梯度为0允许且会记录。
3. 检查联合阶段所有模块train模式，编码器BN计数按预期增加，旧停用参数不变；完整恢复参数、BN缓冲区、梯度、RNG和eval模式，丢弃探测优化器。
4. 将干净新结构保存为preflight_start.pt，重新构建、严格加载全状态并比较32个真实验证样本的输出。上述任一失败均使进程退出，正式训练不会开始。

训练后再重建最佳完整结构，严格加载并复现完整验证混淆矩阵和逐样本预测。训练和探测仅允许train/val图，测试集不参与；读取冻结划分的元数据不属于测试评价。

## 导出与判断

输出独立放在outputs/flow_stats_residual_joint_s42_v1，拒绝覆盖输出目录、日志或PID。根目录summary.json、analysis.json、report.md、per_class.csv汇总N0、冻结V2、已完成小学习率J0/J2和新候选的验证Macro-F1、八类P/R/F1、最佳/停止轮次、总更新次数、训练后最佳/末轮证据和耗时。原始缺失的历史预热耗时保持null，不从文件时间推算；新增前置检查和训练后诊断耗时单列。复用预热5275次更新单独报告。

J2_flow_stats/validation_diagnostics.jsonl.gz每个验证样本保存8维flow_weights及互补packet_weights、64维图表示L2范数、64维残差L2范数和二者比值(分母clamp_min=1e-12)。diagnostics.json保存各维门控及范数比的均值、标准差、范围和分位数；不使用旧标量门控导出格式。

在同一最佳权重上关闭残差路径(只改变推理开关，仍为eval)并重新完整评价，保存validation_stats_off.jsonl.gz和stats_off_metrics.json。它检验最终模型对新增路径的依赖，不能替代重新训练的结构/容量消融，也不用于重新选检查点。

修复/退步与按类计数基于真实ID对齐的逐样本预测，分别相对冻结V2、旧小学习率J2以及同权重关闭路径结果计算。总错误数和Macro-F1一并观察，尤其检查WebBased、Spoofing、Recon和BruteForce的取舍。

沿用用户提出的0.10个百分点投入门槛，并按最后确定的对照口径登记为：候选超过冻结V2和小学习率J2中较强者至少0.10个百分点，即当前约需97.220712%以上，再人工审查类别取舍。只超过起点、只缓解退化或只选中第0轮均不能视为通过。该规则只控制后续投入，不声称统计显著，不改旧冻结V2未通过原筛选门槛的结论，不自动追加seed或调参。

本轮新旧J2的主要差异是Flow统计路径及其新增参数/独立学习率组。N0到新候选的整体变化还包含预热和训练方法，不能全归因于门控；即便结构对照提高，也尚未分离容量和信息路径的各自贡献。

## 服务器启动和结果读取

由助手提交并推送后，使用交付的固定提交号代替REV；启动命令从服务器仓库当前目录运行。候选src文件从归档按精确文件路径加载，复用服务器上已绑定的xgnid.model；不复制覆盖原模型。默认物理GPU1并允许共享，不终止其他进程。

```bash
cd /home/nzxn/下载/WX/load/XG-NID
git fetch origin codex/r0-edge-preparation
bundle=$(mktemp -d /tmp/xgnid_flow_stats.XXXXXX)
git -c core.autocrlf=false archive REV scripts configs src/xgnid/flow_stats_residual.py | tar -x -C "$bundle"
bash "$bundle/scripts/launch_flow_stats_residual.sh" 1
```

```bash
tail -n 30 -F outputs/flow_stats_residual_joint_s42_v1.log
```

完成后提供下面文件即可先判断，不用打包权重：

```bash
cat outputs/flow_stats_residual_joint_s42_v1/status.json
cat outputs/flow_stats_residual_joint_s42_v1/report.md
cat outputs/flow_stats_residual_joint_s42_v1/analysis.json
```

独立汇总入口：wx环境执行python /归档目录/scripts/summarize_flow_stats_residual.py outputs/flow_stats_residual_joint_s42_v1，仅重新整理新结果的报告。配置确认后才考虑等权融合/容量对照、32维融合或保留边属性的GATv2，本轮均不启动。
