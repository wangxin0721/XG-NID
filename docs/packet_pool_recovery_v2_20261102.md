# 两次第0轮失败后的有序归约恢复

11月2日截图中，2bf57a5恢复任务已复用缓存，但13,446例验证有1批未通过1e-6检查；旧直接加权写法有11批未通过。两种写法的预测变化均为0。新写法最大Packet读出误差约1.19e-6，最大log概率误差约2.29e-5。此时仍在初始化比较，尚未进入梯度探测或正式训练。

上次中心化修正只消除了 `sum(x/n)` 与 `sum(x)/n` 的差别，没有固定重复CUDA归约的次序，也没有统一softmax均匀权重和计数倒数的计算路径。PyG 2.5.3的sum／mean实际调用 `scatter_add_`，因此不能把另一次独立GPU归约当成精确数值参考。[PyG源码](https://pytorch-geometric.readthedocs.io/en/2.5.3/_modules/torch_geometric/utils/_scatter.html)

本次P0／P1均使用固定PyTorch 2.3.1内的 `torch.segment_reduce` 进行Packet读出。节点按图连续排列；二维CUDA内核对每个图／通道按节点偏移顺序累加，避免读出中的scatter累加。评分网络仍为64→16→1，分图softmax及均匀参考权重都走同一分段归约路径；P1仍计算 `mean + sum((beta-uniform)*(h-mean))`。参考权重也使用零分数softmax，不再另算计数倒数。[固定版本CUDA内核](https://github.com/pytorch/pytorch/blob/v2.3.1/aten/src/ATen/native/cuda/SegmentReduce.cu)

参数量、权重起点、编码器、64→32投影、门控、分类头、学习率、优化器、seed、batch和100／20预算均不改。两组同时更换数值归约后端，历史基线的逐样本预测、混淆矩阵和97.32070542814468%起点仍必须复现。配置文件字节及SHA不变；后端名称记录在结构和源码绑定中。未设置全局确定性标志，也未改编码器CUDA算子。

## 数值验收的边界

1. 新启动在读取节点缓存前，先用实际GPU执行16次合成读出重复检查：P0均值和P1均匀初始输出必须精确相等，beta必须精确等于均匀参考。两组分类输出／门控仍使用atol=rtol=1e-6。另执行分段归约反向检查，评分末层梯度须有限且非零。此阶段没有优化器更新；模型、RNG必须不变。证据写入 `cuda_readout_preflight.json`。
2. 完整验证缓存继续检查P0／P1的log概率及门控1e-6一致、相同节点输入的均值重复一致、均匀权重精确一致，以及成熟D32逐样本预测和CM完全一致。`initialization_comparison.json`另记录旧scatter均值重复误差和旧加权汇总差异。
3. 初始原图回放中，新读出与源分类头在**同一次原图编码所得的相同节点表示**上进行1e-6比较，检验结构和参数等价性。未改动的旧D32仍单独完整前向，预测／CM必须相同；它的独立CUDA归约log误差如实记录为 `legacy_independent_source_outputs_allclose_1e6` 及最大误差，允许该独立误差标志为false。此标志不冒充结构1e-6验收通过。
4. 原图／节点缓存输出仍须通过1e-5比较，逐样本预测及CM精确一致。每组各3次真实图废弃更新、冻结编码器／BN检查、干净状态恢复、相同采样顺序与最佳检查点回放均保留。

以上区分了同一冻结表示下的读出等价和旧CUDA独立前向噪声；不能把“预测一致”用来跳过新结构的数值检查，也不宣称全部编码器／完整CUDA训练位级确定。

## 恢复与来源保护

源节点缓存仍来自 `outputs/packet_pool_pair_s42_v1`。同时审计 `outputs/packet_pool_pair_s42_recovery_v1` 的2bf57a5源码绑定、旧恢复收据、数值诊断和复制后缓存摘要。两者必须都是已知的训练前初始化失败，没有P0／P1目录、初始化通过收据、梯度探测或后续权重。

两份旧目录及日志保持原样，分别保存文件清单／摘要，并在预检后及完成前再次核对。任一来源改变、错误类型不同、已出现训练进度或相关进程仍活动，都会退出。新结果写入独立 `outputs/packet_pool_pair_s42_recovery_v2`，已有目录／日志／PID时拒绝覆盖。

仍只跑P0／P1。从D32-Gate第62轮重新构建干净模型，复制并核验原始训练／验证节点缓存；不重复全量编码器提取。旧提取次数不计入本次原图加载，原内存估算标记为历史提取数据。CPU张量测试增加分段重复、分组错误拒绝和设备预检状态保护，共11项。

在最终交付消息的固定提交bundle内启动：

```bash
bash "$bundle/scripts/launch_packet_pool_pair_recovery_v2.sh" 1
```

```bash
tail -n 50 -F outputs/packet_pool_pair_s42_recovery_v2.log
```

失败时提供新目录的 `error.log`、`cuda_readout_preflight.json`、`initialization_comparison.json` 中已生成的文件。完成后优先提供 `status.json`、`report.md`、`analysis.json`、`packet_diagnostics.json`。预算门槛仍为P1超过max(P0,D32)至少0.10个百分点；不自动追加其他实验。

本地仍无torch／PyG，11项张量测试和实际CUDA预检尚未在本地执行；本地仅执行标准库恢复／协议回归、语法、来源Git对象及文件摘要检查。
