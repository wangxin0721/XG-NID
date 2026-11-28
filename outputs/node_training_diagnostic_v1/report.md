# 20轮训练策略诊断

| 组别 | 最佳轮 | 最佳验证 Macro-F1 % | 验证 NLL | 固定训练子集 NLL |
|---|---:|---:|---:|---:|
| Ordered | 0 | 97.914032917 | 0.134565741 | 0.017601352 |
| R1 | 0 | 97.914032917 | 0.134565741 | 0.017601352 |
| R2 | 0 | 97.914032917 | 0.134565741 | 0.017601352 |
| R3 | 0 | 97.914032917 | 0.134565741 | 0.017601352 |

R3延长投入门槛：{'R3_minus_Ordered_pp': 0.0, 'R3_minus_R2_pp': 0.0, 'extension_gate_passed': False, 'threshold_pp_each': 0.05, 'no_arm_improves_source': True, 'R2_and_R3_both_improve_source': False, 'automatic_extension': False, 'test_authorized': False, 'scope': '20_epoch_seed42_resource_decision_not_statistical_significance'}

R1与旧S1相同轮次的差值（pp）：{'1': 0.36037866583770173, '2': 0.22468357376541626, '5': 0.004195582501931305, '10': -0.07853023219450694, '20': 0.170782181817275}
结构比较以R3−R2为主；R1仅诊断训练策略。20轮未通过不等于方法永久无效。无自动扩展、无测试集访问。
全部逐类指标、修复/退步计数、NLL和参数变化见history.json与observations/。
