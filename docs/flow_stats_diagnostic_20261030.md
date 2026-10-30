# 残差候选失败后的只读回放

用户返回的analysis.json复核结果：J2_flow_stats最佳第0轮97.093667251942%，等于冻结V2；训练后最佳和末轮均为第12轮94.30319351896747%，相对自身起点低2.79047373297453个百分点。它未通过结构筛选，不保留为当前最佳方案；当前seed42最高的已验证候选仍为小学习率J2的97.12071203419106%。

相对N0的+0.09743370914428个百分点全部已存在于冻结V2。当前残差模型报告中的类别改善、15次修复/15次退步，均是冻结V2与旧小学习率J2的差异，不能归给新增MLP。最佳第0轮的MLP末层仍为零，关闭路径得分不变是预期结果，不能据此判断训练后路径没有梯度、没有使用或没有产生影响。

仅凭所交付analysis.json无法判定退化来自残差幅度、BN运行统计、公共参数适应或它们的交互。新MLP学习率1e-4比公共投影/门控高，且没有独立归一化，只能作为排查线索，不能直接当作已经证实的原因。第12轮是训练后最佳，也不能据此认定过拟合；停止来自12轮未超过第0轮的既定规则。

本次仅新增diagnose_flow_stats_residual.py、启动脚本及轻量测试，不改原模型、训练入口、配置、门槛和完成结果，不新增训练。服务器读取原N0、选定V2和已完成残差last.pt，严格重建原633b7c1结构，执行四个固定的验证回放：

| 回放 | 保留和改变 |
|---|---|
| last_full | 原样末轮，必须复现94.3031935%及其完整验证混淆矩阵 |
| last_stats_off | 所有末轮参数和BN状态相同，仅关闭统计路径 |
| last_with_start_BN | 保留末轮参数及统计路径，只把两路编码器BN运行均值、方差、计数换回V2起点 |
| start_with_last_stats | 保留V2起点的所有公共参数及BN，接入末轮MLP |

四种回放都调用eval和no_grad；代码不创建优化器，不调用反向或参数更新，不选出新检查点，不读测试图。模型形状初始化会读取原来的16个训练图，只用于加载惰性图层并立即恢复检查点，无训练或标签拟合。所有输出放到独立outputs/flow_stats_residual_diagnostic_s42_v1，禁止位于已完成训练目录内。原实验配置、来源绑定、选模日志和全部已登记产物哈希运行前后均复核，拒绝覆盖旧结果或诊断目录。

每种回放输出逐样本8维权重、图/残差范数及比值、完整八类P/R/F1、混淆矩阵和相对冻结V2的修复/退步。summary.json附带原前置梯度检查记录，并注明它属于训练前探测，不能当作整段训练轨迹。report.md汇总四项分数、错误数、范数比均值及95%分位数。

若关闭末轮路径明显恢复分数，说明当前末轮模型中这条路径的实际输入效应有损；结合范数比观察幅度，仍不能只凭范数大小证明因果。若换回起点BN状态恢复较多，说明统计状态对该末轮预测有影响，仍不等于证明固定BN训练一定更好。把末轮MLP放回起点公共参数只是受控推理干预，不能替代配对训练或容量消融。无论结果如何，本入口都不自动延长训练、搜索学习率或启动新结构。

本地缺少torch/PyG，只运行无torch逻辑、来源拒绝、Bash及Python3.9语法检查；实际回放由服务器执行。使用助手交付的固定提交号替换REV：

```bash
cd /home/nzxn/下载/WX/load/XG-NID
git fetch origin codex/r0-edge-preparation
bundle=$(mktemp -d /tmp/xgnid_stats_diag.XXXXXX)
git -c core.autocrlf=false archive REV scripts configs src/xgnid/flow_stats_residual.py | tar -x -C "$bundle"
bash "$bundle/scripts/launch_flow_stats_diagnostic.sh" 1
```

物理GPU1，允许共享，不停止其他任务。查看outputs/flow_stats_residual_diagnostic_s42_v1.log。结束后提供新目录的status.json、report.md和summary.json；训练原目录无需改动或重新打包。
