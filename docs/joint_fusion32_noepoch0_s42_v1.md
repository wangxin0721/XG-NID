# J32后续独立运行：100轮、patience20、第0轮不选模

当前服务器任务继续运行，本交付不停止任务，不覆盖产物，不自动排队。等当前任务结束后，手动启动本版本。

本轮仍只运行seed42的J32-Equal、J32-Gate，分别从原 `outputs/fusion_gate_pair_s42_v1/D32-Equal/best.pt`（第72轮、97.143955%）与 `D32-Gate/best.pt`（第62轮、97.320705%）开始。不会接着当前联合训练权重继续训练，也不重置融合头或重复预热。

独立配置 `configs/joint_fusion32_s42_e100_p20_noepoch0_v1.json` 与100/20基础配置仅有 `include_epoch_zero=false` 的差别。新启动入口为 `scripts/launch_joint_fusion32_noepoch0.sh`，独立结果目录 `outputs/joint_fusion32_s42_e100_p20_noepoch0_v1`。

- 最多100轮、patience20；第0轮完整复现、记录、保存，但不写成最佳检查点。
- 仅从第1轮及后续轮次按验证Macro-F1严格改善选模；以训练后最佳分数计早停。即使训练后所有轮次均低于起点，也保存训练后的最佳权重，如实报告负增益，不回退到第0轮。
- 新建Adam，编码器1e-6、投影／分类头1e-5、动态门控1e-5，weight decay=0；batch512/16、普通最终NLL、真实图输入。
- 编码器参数可训练，编码器保持eval、BN缓冲区固定；同seed两组公共轮次样本顺序一致。
- 保留原模型、标签、划分、源检查点完整性验证、各自第0轮复现、三批梯度探测与状态恢复、最佳检查点原图回放。测试集不参与。

100轮是上限，并不保证训练满100轮。连续20轮未严格改善即停止。预算门槛仍为J32-Gate超过原D32-Gate和本轮J32-Equal中较强者至少0.10个百分点，并审查类别取舍。历史任务若使用30/12预算，不能把两次结果差异仅归因于第0轮选模规则。

使用交付提交号下载归档后，在服务器项目目录运行：

```bash
bash "$bundle/scripts/launch_joint_fusion32_noepoch0.sh" 1
tail -n 50 -F outputs/joint_fusion32_s42_e100_p20_noepoch0_v1.log
```

只应在准备启动时执行上面的启动命令；入口立即后台提交，不等待现有任务。完成后提供新目录的 `status.json`、`report.md`、`analysis.json`。结果同基础版，包含各自起点增益、最终两组差距、逐类最佳／末轮指标、训练后最佳与末轮、修复／退步样本及原图回放凭据。

详见 [联合微调实现说明](joint_fusion32_s42_v1.md)。基础版第0轮选模配置保留为true，新入口显式使用独立false配置。该轮仍需服务器阻断式张量验收，本地轻量检查不能替代真实前后向。
