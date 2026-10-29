# E0公平微调与逐特征门控：五项固定训练

本轮按E0-FT42→E0-FT43→E0-FT44→V1→V2顺序运行，共五项正式训练。已有E2/N0/F0/F1/F2不重跑，不进行新seed、学习率、隐藏层或轮数搜索，不运行测试集。正式方法尚未定稿；N0是当前对照，动态适应继续作为研究方向。

## 公平微调

各自E0最佳检查点来自原seed42 E0-B512目录，以及baseline_seed_repeats_b512_s43_s44_v1/E0_s43、E0_s44。启动核对完成状态、seed/模型、指标矩阵、源码、配置和检查点哈希。先按验证manifest顺序复现第0轮，再做3-batch可丢弃梯度/实际更新探测，恢复干净状态正式微调。

E0-FT严格读取实际N0协议：新建Adam、不继承动量、lr=.0001、weight_decay=0、训练batch512、评价batch16、workers2、最多30轮、patience12、严格验证Macro-F1选模、第0轮纳入候选。复用原N0训练函数，通过受限AST适配只删除引用双分支名称的首步梯度诊断；以E0编码器/分类头的真实梯度与更新探测替代。同时更正E0不存在历史gate的元数据标签；损失、优化器、数据流、预算和选模均不改，旧代码不被覆盖或猴子补丁修改。

对照每seed自己的已完成N0。新注册文件保存用户提供的三份N0完整指标和文件哈希，服务器逐项核对。最终报告N0−E0-FT的每seed差、均值、样本标准差，两种模型各自微调增益、实际更新数、完整八类指标。相同微调机会并不意味着独立早停后的更新次数相同，也不能替代参数量匹配与池化消融。

## 逐特征融合

固定N0-s42最佳检查点，直接复用已完成冻结融合的train_cache.pt、val_cache.pt。按seed42原记录核验缓存文件哈希、对应冻结模型状态哈希、manifest ID及标签顺序。编码器、投影和H均冻结并保持eval；训练融合→H→NLL时保留输入梯度。

V1为8个全局logit，经Sigmoid得到所有样本共享的8维权重。V2复制原21→16门控首层及LeakyReLU，仅将末层替换为16→8并经Sigmoid；首层来自N0未使用的gate，末层权重和偏置清零。V1/V2可训练参数分别为8、488，所有维度初始为.5。这些分量对应表示维度，不对应攻击类别。

不更换21维输入，不增加辅助监督、类别加权、B-H输入或原代理混合。沿用冻结协议：新建Adam、lr=.001、weight_decay=0、batch512、评价batch16、最多30轮、patience12、第0轮参与选模。同一缓存、同一seed42采样顺序及起始RNG；新层构造的随机数消耗被保存恢复，避免人为改变两项训练起始RNG。训练函数使用独立命名空间适配预测及评价，没有全局修改旧函数。

正式训练前在全部13446个验证样本检查V1/V2初始log-probability与F0 allclose(atol=rtol=1e-5)、预测零差异、矩阵与指标一致，再做各3-batch可丢弃探测。显式检查经过冻结H的融合输入梯度、融合参数实际更新、V2首层后续梯度。探测融合器和Adam被丢弃后重新构造干净起点。

逐样本预测保存全部8维flow_weights及互补packet_weights，权重统计按表示维度报告，不将前两维误标为两分支标量。最后核对冻结模型未变化、V1/V2公共轮次顺序与训练起始RNG相同。

## 预登记门槛与结果

仅当V2−N0≥0.10个百分点且V2−V1≥0.05个百分点时，标记预算数值门槛通过；仍需人工检查八类P/R/F1与修复/退步，确认是否有明显类别损失。该门槛不表示显著性，也不会自动启动seed43/44重复。完整结果及训练后最佳候选（含第0轮是否入选、末轮变化）都会报告，不把第0轮兜底误当成实际收益。

## 服务器交付

助手完成本机检查、提交和推送；用户在服务器运行下列命令，使用助手提供的固定提交号替代rev。优先GPU1，可显式共享。不安装或替换原环境，使用/home/nzxn/miniconda3/envs/wx/bin/python（XGNID_PYTHON可指定已有解释器）。

```bash
cd /home/nzxn/下载/WX/load/XG-NID
git fetch origin codex/r0-edge-preparation
rev=$(git rev-parse FETCH_HEAD)
bundle=$(mktemp -d /tmp/xgnid_fair_vector.XXXXXX)
git -c core.autocrlf=false archive "$rev" scripts configs | tar -x -C "$bundle"
bash "$bundle/scripts/launch_fair_vector.sh" 1 --allow-shared-gpu
```

```bash
tail -n 30 -F outputs/fair_vector_five_runs_v1.log
cat outputs/fair_vector_five_runs_v1/progress.json
cat outputs/fair_vector_five_runs_v1/status.json
```

根summary.json在全部五项训练与核验完成后产生，包含fair_comparison、paired_N0_minus_E0_FT、vector_screening、V2_versus_V1和全部逐类指标。已有输出/日志/PID时拒绝覆盖，不实现resume或自动重试；一项失败停止后续队列。Ctrl+C只退出tail。

本机无torch，不能声称已完成真实检查点初始化等价或梯度检查。标准库/AST检查及NumPy数值检查可在本机执行；真实torch测试本机跳过，服务器队列实际探测必须通过后才正式训练。
