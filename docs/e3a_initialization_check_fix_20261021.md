# E3a初始化核对修复（2026-10-21）

用户截图显示E3a probe完成，随后screen在`certificate == report["initialization"]`处退出。该检查发生在正式screen创建Adam、加载训练批次或写epoch之前，因此本次筛选尚未训练；不需要恢复筛选权重。已有probe、失败screen和成功诊断均保留，不覆盖或删除。

## 原因与修复范围

原证书包含两类信息：原始/变体初态SHA、前向AST SHA、变体/变更张量等身份字段，以及GPU前向所得的`actual_initial_weight_mean`浮点观测。原代码用整字典精确相等把浮点观测也当作逐位身份，过于严格。

在截图报错之前，`initial_model`已核对原始初始化SHA，`apply_variant`已核对E3a没有改变任何初态张量，`new_model`也核对初始eval没有改变模型状态。当前没有收到完整旧probe证书，不能给出两次均值的具体数值差；不据此声称实际模型权重不同。

修复仅在独立`run_e3_gate_ablation_server.py`中：

- 所有6项身份字段仍逐项精确比较，任一变化直接报出字段及双方值；不增加旧probe兼容绕过。
- 两项GPU均值必须是有限的[0,1]数值，和在1e-6内等于1；screen/probe最大绝对差不超过1e-5。超出仍拒绝，并显示实际差值。
- 在日志、config和metrics中记录两次均值、最大绝对差、容差及身份精确匹配状态。该容差只属于浮点诊断，不是模型超参数，也不应用于初始化/源码/配置/检查点哈希。
- 原共享模块、配置、诊断入口、冻结模型和数据代码不变。原验证诊断与缓存分析可继续使用。

由于训练入口文件的真实SHA变化，旧probe的入口绑定不再匹配修复版。必须用修复版做一次短的新probe（1次预热+3次实测），再进入新screen；不重跑验证诊断、不重做缓存分析，不修改旧报告来放行。新的输出使用`e3_gate_probe_E3a_b512_s42_v2`和`e3_gate_screen_E3a_b512_s42_v2`。

## 本地验收

4项纯标准库回归检查通过：接受微小均值舍入差并记录；6项身份各自变化均拒绝；明显数值差、NaN、越界、非归一、错误长度/类型/字段均拒绝；与交付`fda02b1`比较，new_model/update/probe/诊断绑定/parent_receipt/main及完整epoch训练循环AST保持一致。没有本地torch导入、反序列化权重、图加载或模型实验。

代码：[run_e3_gate_ablation_server.py](../scripts/run_e3_gate_ablation_server.py)；回归检查：[test_e3_gate_initialization_comparison.py](../scripts/test_e3_gate_initialization_comparison.py)；机器记录：[e3a_initialization_check_fix_20261021.json](e3a_initialization_check_fix_20261021.json)。修复版服务器probe和screen尚待真实执行，不标记为已通过。

## 服务器执行

fetch后固定实际FETCH_HEAD SHA，提取共用模块、修复入口和原配置到新的不可变/tmp目录；不覆盖旧`/tmp/xgnid_e3_gate_fda02b1`。完整可复制命令见本次交付回复。

1. 使用新入口、原成功诊断、物理GPU0及同一共享策略，输出新probe_v2；该probe不读取验证/测试图。
2. probe_v2成功后，用相同新入口/配置/GPU UUID独立提交screen_v2。保持batch512、lr0.001、seed42、最多100轮、patience12、仅训练和验证。
3. 首先核对日志中的`initialization_comparison`，身份应精确匹配，均值最大差应不大于1e-5，随后才应看到epoch。
4. 输出目录存在时查看其status，不删除重跑、不复用v1失败目录、不自动放宽其他校验。
