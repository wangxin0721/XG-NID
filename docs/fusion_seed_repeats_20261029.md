# F2固定配置：seed43／44六项训练

本轮只执行N0-s43→F1-s43→F2-s43→N0-s44→F1-s44→F2-s44，另各做一次F0复现。E0/E2已有训练不重跑，不增加新结构或超参数搜索，不评价测试集。助手负责提交推送；用户在服务器执行一个启动命令组。优先GPU1，可显式共享。

2026-10-29本机状态：新队列8项检查通过，包含完整元数据预检的合成文件夹测试及错误权重拒绝；此前N0和冻结融合检查继续通过，其中真实torch检查因本机缺少torch而跳过。尚未在服务器反序列化seed43/44权重或开始新增训练；真实起点复现和梯度探测在服务器执行。

## 起点登记

读取已完成outputs/baseline_seed_repeats_b512_s43_s44_v1/E2_s43/E2和E2_s44/E2的best.pt。登记的验证Macro-F1四位小数分别为97.0766%、96.9707%。启动前核对两个运行的完成状态、模型/seed、原学习率和预算、config/metrics/summary一致性、源码、划分及manifest、best.pt哈希和验证矩阵；使用实际未舍入分数复现，而非用四位显示值训练或选模。

seed42为此前筛选结果，不重跑。configs/frozen_fusion_seed42_result_20261029.json保存用户提供的原始F0/F1/F2指标与绑定记录，启动前与服务器原结果及权重/预测文件核对。旧N0/冻结融合协议的文件哈希、旧训练函数源码哈希必须与seed42结果一致。

## 固定协议与复用边界

原run_e2_concat_pair.py和run_frozen_fusion.py保持字节不变。新入口通过SeedCore薄包装，只将旧函数TrainBatches(...,512,42)的采样器seed映射为当前43或44；其他模型、损失、优化器、加载及训练实现均委托原函数。只有43/44可用，旧采样器参数契约改变时立即拒绝运行，无全局猴子补丁。

N0完整微调：从本seed的E2最佳检查点出发，所有原可训练参数保持原设置，两编码器、两投影和共享头更新。重置Adam，不继承旧动量；lr=.0001、weight_decay=0、batch512、评价batch16、workers2、最多30轮、patience12、严格验证Macro-F1选模，epoch0纳入候选。先复现E2的验证顺序、逐样本标签和矩阵，再做可丢弃3-batch探测，恢复干净起点正式训练。

冻结阶段：重新以当前seed播种并构建原E2结构，严格加载新生成的本seed N0最佳权重，冻结全部参数，保持eval。按原逻辑缓存本seed训练/验证表示。F0必须复现本seed N0的验证顺序、预测、矩阵和Macro-F1，不要求等于seed42的分数。

F1为一个全局标量，F2为原21→16→2门控，从该seed N0未使用的gate复制首层、将末层W/b清零，均从.5/.5开始；不增加B-H输入、代理混合、辅助监督或类别权重。Adam lr=.001、weight_decay=0、batch512、最多30轮、patience12、含epoch0选模，与seed42相同。分类头参数固定，但融合输入的梯度通过H保留。两项共用同一缓存和相同采样顺序，探测状态丢弃后用干净融合器及Adam训练。

每项每完整轮211次更新，最多6330次正式更新；六项最多37980次，不将全模型微调与融合器训练视为等价预算。原协议中的可丢弃短探测保留，不计入正式训练轮数。各项按自己的早停路径结束，不因seed42 F2首轮最佳而缩短预算。

## 选模与报告

epoch0参与选模，因此最终不低于F0并不构成新增收益。每项保留逐轮日志和末轮指标，汇总额外重放日志，核对严格选模及早停，记录是否选中epoch0、训练后最佳候选的轮次、分数、相对起点变化和末轮变化。即使所有训练候选下降、最终选中epoch0，也必须呈现实际下降。

根summary.json包含seed42/43/44的F0/F1/F2原始结果、F2−F0和F2−F1百分点差、三seed模型均值/样本标准差、配对差值均值/样本标准差；新增43/44的配对统计单独报告。所有计算使用未舍入值，使用样本标准差，不作显著性断言。report.md提供简洁表格。

每项保存完整八类P/R/F1、混淆矩阵、逐样本预测、相对F0的修复/退步计数及逐类计数、权重分布、best/last检查点、最佳/停止轮次和实际更新数。每个seed另报告F2对F1的逐样本配对计数，确认冻结模型未变化、两融合器公共轮次顺序及起始RNG相同。权重随样本变化本身不证明效果。

数值筛选只登记新增两seed的F2是否均严格超过各自F0/F1，类别取舍与增益幅度仍需检查。两次均提高且取舍可接受时保留进入最终方案验证；一次有收益时保留探索候选、优先N0；两次无额外优势时结束当前门控候选。三次均提高也只能表述为同一划分下三次训练的平均收益。seed42用于筛选的事实单独标记。没有自动追加训练或最终测试。

## 服务器启动与查看

助手推送后，将固定提交号填入archive。通过/tmp独立交付，不覆盖已有服务器源码、权重或结果；启动器固定使用原wx解释器，可用XGNID_PYTHON指定已有环境位置。

```bash
cd /home/nzxn/下载/WX/load/XG-NID
git fetch origin codex/r0-edge-preparation
rev=$(git rev-parse FETCH_HEAD)
bundle=$(mktemp -d /tmp/xgnid_fusion_repeats.XXXXXX)
git -c core.autocrlf=false archive "$rev" scripts configs | tar -x -C "$bundle"
bash "$bundle/scripts/launch_fusion_seed_repeats.sh" 1 --allow-shared-gpu
```

```bash
tail -n 30 -F outputs/fusion_seed_repeats_s43_s44_v1.log
watch -n 5 cat outputs/fusion_seed_repeats_s43_s44_v1/progress.json
cat outputs/fusion_seed_repeats_s43_s44_v1/status.json
```

根输出下seed43/、seed44/分别包含N0/和fusion/，fusion/含F0.json、缓存、探测记录、F1/、F2/。根summary.json在六项正式训练及核验全部成功后生成；report.md给出三seed表。Ctrl+C只退出tail/watch。

已有输出、日志或PID文件时拒绝重复启动，不实现resume或自动重试。一项失败停止后续排程，保留status.json和error.log；队列不会重新训练已有E0/E2或新增seed42。
