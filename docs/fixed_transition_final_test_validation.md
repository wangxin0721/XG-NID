# 固定测试入口交付验证（2026-11-24）

## 本地已通过

- CPU环境：Torch2.3.1+cpu、PyG2.5.3。57项相关检查全部通过，覆盖新固定测试入口、既有稳定性训练依赖、LFS初始化与融合层实现。
- 新入口专项15项检查包括非零模型恢复、初始化随机状态保留、冻结参数/缓冲区/模式、验证/测试分区隔离、缓存来源与SHA256、缺失收据重提取、中断续跑、已完成主干复用、错误主干/协议拒绝、历史原C预测不一致时中止、负值及零差报告、锁定先于测试，以及正常/失败/缺少LFS时的上传与退出码。
- 对不可变 `5f3975aeecd41c8c1150b97f88813b7ea703e54d` 的59个注册归档输入验证SHA256/字节数；核对12个权重及验证最佳轮、12组验证预测、9份账本和历史原C测试来源。
- 12个真实模型全部在CPU恢复，三个原C状态哈希分别与锁定来源一致；Frozen Single的融合权重与该原C一致，全部参数冻结且全部模块eval。
- 使用哈希绑定的冻结验证特征对12个真实模型分别回放完整13,446条验证流；逐样本预测全部一致。该检查没有执行原图编码器，没有评价本轮测试候选。
- 合成数据的端到端固定测试与中断续跑中，Adam创建及backward调用均被禁止，流程仍完整通过；新增更新为0。

复现CPU检查：

```powershell
$env:PYTHONPATH = "$PWD\src;$PWD\scripts"
& D:\XG-NID\.residual-audit-env\Scripts\python.exe -u -B -c "import tempfile,pathlib,unittest;tempfile.tempdir=str(pathlib.Path('tmp').resolve());modules=['test_fixed_transition_final_test','test_fixed_transition_stability','test_fixed_transition_lfs','test_joint_transition_fusion'];suite=unittest.defaultTestLoader.loadTestsFromNames(modules);result=unittest.TextTestRunner(verbosity=1).run(suite);raise SystemExit(not result.wasSuccessful())"
```

## 服务器与正式评价状态

本地CUDA不可用。已配置GPU服务器的SSH连接在跳板banner交换阶段超时，因此未完成服务器运行时探测，也未通过本次连接启动正式原图回放或固定测试。服务器启动、日志查看与上传重试命令见 `fixed_transition_final_test_v1.md`。

锁定阶段与本地缓存验证均不等于固定测试完成。当前不填写测试分数，不把seed43的98.013342%验证结果放入测试表。正式完成后入口才生成12组固定测试汇总及 `paper_tables.md`。

用户主工作区的未提交论文草稿、数据及历史结果均保留；新实现位于独立代码分支。正式测试完成后保留全部负值、类别取舍及既有失败消融，不重新挑检查点或调参。
