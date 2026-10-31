# 8/32维配对：修复第0轮等权检查并复用已完成准备

5243cc3服务器日志在D8的train()入口报ValueError: Fresh epoch0 fusion is not equal-weight。该断言在新建D8目录和初始评价之后、创建正式优化器及训练循环之前执行，因此这次失败未产生正式训练更新；报告只在两组训练和原图回放完成后生成，原目录没有report.md、analysis.json符合这个退出位置。

原断言要求每一维统计mean/min/max都用浮点严格等号等于0.5。GPU的mean归约可能产生舍入误差，不能用聚合均值的精确相等作为等权逐元素检查。原始失败没有保存该分布的具体数值，因此修复后额外输出observed_mean/min/max和legacy_aggregate_exact_check_would_pass，便于核实误报；不把未读取的旧均值编造成已观测数据。

修复保留两层约束：门控末层weight/bias必须确实全零；全验证集逐样本权重相对0.5的最大绝对误差必须不超过1e-7。原始均值照常保存，不钳位、不强行改写统计结果。非等权的实际逐元素权重仍会拒绝，即使其均值恰为0.5。模型定义、初始化、学习率、损失和训练顺序均未改变。

本次100轮上限、patience20、第0轮只记录不选模保持不变；配置SHA256仍为07635d84a1c00038c5dbbf550e5b0a5fa15761f95d63b0f159e8820e8eb37b2a。历史配置和src/xgnid/fusion_dimension.py保持不变。

## 恢复策略

scripts/recover_fusion_dimension_pair.py只接受这一次已知的训练前失败，固定核对5243cc3中原入口／汇总／模型的来源哈希、原配置、缓存身份、编码器和初始状态摘要。旧D8必须为空、D32必须未创建；出现best.pt、last.pt、轮次日志或其他正式训练迹象时拒绝自动恢复。还要求原梯度探测和完整结构回放收据均完整、状态已恢复。

源目录outputs/fusion_dimension_pair_s42_v1保持原样，包括failed状态和error.log。恢复写入独立outputs/fusion_dimension_pair_s42_recovery_v1：

1. 复制两份已完成的64维缓存，先后核对文件SHA256、张量内容摘要、划分、样本ID、标签、维度和源编码器身份；不再读取全部图来重新提取表示。
2. 从D8_preflight.pt、D32_preflight.pt严格读取干净初始头部，核对其参数摘要与原initialization.json及当前固定初始化一致；核对其中编码器和BN状态确实是原N0。门控末层须为零。
3. 复用原已完成且恢复干净状态的三步／组梯度探测，不重做这6次已丢弃更新。新恢复过程仍执行真实图的结构重建回放，并在全部验证缓存上做改正后的逐元素等权检查，检查失败继续阻止训练。
4. 随后按原协议运行D8→D32，分别重置Adam和训练RNG，保留共同轮次顺序校验，完整训练后按原流程回放全部真实验证图并生成报告。

运行前检查源任务和恢复任务的相关/proc命令行，无进程被结束。原目录文件在恢复前后逐一核对SHA256；不覆盖、移动或删除任何历史产物。恢复目录／日志／PID已存在时拒绝重复启动。

缓存耗时区分原收据记录的提取时间、恢复复制时间、此次缓存装载／校验时间。已失败任务未保存完整阶段总耗时，不从文件时间推算；原提取时间按已记录两份收据求和，明确不包含未记录的其他准备耗时。报告注明梯度探测为复用历史收据、新探测更新数为0，不把恢复当成额外训练机会。

## 验证和服务器命令

本地无torch/PyG，仅检查浮点均值误拒回归、实际非等权仍拒绝、来源／缓存／权重损坏拒绝、训练进度保护、原目录保持、命名空间隔离、配置不变和Python3.9/Bash语法。真实GPU归约及前向回放由服务器执行，未执行的项目不标记为通过。

助手提交推送后用固定提交号替换REV，运行恢复入口：

```bash
cd /home/nzxn/下载/WX/load/XG-NID
git fetch origin codex/r0-edge-preparation
bundle=$(mktemp -d /tmp/xgnid_dim_recovery.XXXXXX)
git -c core.autocrlf=false archive REV scripts configs src/xgnid/fusion_dimension.py | tar -x -C "$bundle"
bash "$bundle/scripts/launch_fusion_dimension_pair_recovery.sh" 1
```

```bash
tail -n 50 -F outputs/fusion_dimension_pair_s42_recovery_v1.log
cat outputs/fusion_dimension_pair_s42_recovery_v1/progress.json
cat outputs/fusion_dimension_pair_s42_recovery_v1/status.json
```

GPU1，允许共享，正式训练仅在服务器执行。完成后读取新恢复目录的report.md、analysis.json和status.json。原失败目录继续保留failed记录，不会在那里补写结果。
