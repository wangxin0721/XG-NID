# 报文转移读出与双分支融合层联合优化：seed42 配对验证

四组从同一个原 C 开始，主干与跨层模块冻结，仅优化融合层及新增读出。仅使用 train/val。

| 组别 | 最佳轮 | Macro-F1 | 相对原 C（百分点） |
| --- | ---: | ---: | ---: |
| J0 (joint_base) | 39 | 97.568753% | +0.014949 |
| J1 (joint_single) | 59 | 97.630209% | +0.076406 |
| J2 (joint_pointwise) | 84 | 97.747010% | +0.193206 |
| J3 (joint_ordered) | 84 | 97.874050% | +0.320246 |

| 比较 | Macro-F1 差（百分点） | 修复 | 退步 | 错误间变化 |
| --- | ---: | ---: | ---: | ---: |
| joint_ordered-joint_base | +0.305297 | 45 | 22 | 7 |
| joint_ordered-joint_single | +0.243841 | 38 | 18 | 5 |
| joint_ordered-joint_pointwise | +0.127040 | 23 | 16 | 7 |
| joint_base-original_C | +0.014949 | 17 | 11 | 2 |
| joint_single-original_C | +0.076406 | 33 | 24 | 6 |
| joint_pointwise-original_C | +0.193206 | 50 | 28 | 5 |
| joint_ordered-original_C | +0.320246 | 51 | 22 | 5 |
| joint_single-frozen_single | -0.195156 | 18 | 34 | 5 |
| joint_pointwise-frozen_pointwise | -0.043073 | 30 | 30 | 6 |
| joint_ordered-frozen_ordered | +0.028211 | 23 | 19 | 4 |
| frozen_ordered-frozen_single | +0.020474 | 22 | 22 | 6 |

| 同预算轮数 | J0（最佳轮） | J1（最佳轮） | J2（最佳轮） | J3（最佳轮） | J3−J1 pp | J3−J2 pp |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30 | 97.557804% (1) | 97.553803% (0) | 97.642428% (29) | 97.586345% (29) | +0.032541 | -0.056083 |
| 100 | 97.568753% (39) | 97.630209% (59) | 97.747010% (84) | 97.874050% (84) | +0.243841 | +0.127040 |
| 150 | 97.568753% (39) | 97.630209% (59) | 97.747010% (84) | 97.874050% (84) | +0.243841 | +0.127040 |
| 200 | 97.568753% (39) | 97.630209% (59) | 97.747010% (84) | 97.874050% (84) | +0.243841 | +0.127040 |

投入参考按未四舍五入值判断：{'joint_ordered-joint_single': {'delta_pp': 0.24384060844290234, 'reference_pp': 0.1, 'passed': True}, 'joint_ordered-joint_pointwise': {'delta_pp': 0.12703972187545487, 'reference_pp': 0.05, 'passed': True}}
J3 是否超过冻结 Ordered（实测 97.845839%）：True。
这些门槛仅用于资源决策，不代表统计显著性。相对原 C 的全部提升不能归因于有序结构。
两项投入参考均达到，可进一步安排同预算流内打乱配对对照；本轮不自动追加实验或测试。

八类 Precision/Recall/F1 与混淆矩阵见 validation/per_class.csv 及各组 CSV。
所有修复/退步均按真实类别、错分方向、有效一步数量和真实连续两步数量保留。
门控逐维权重、两尺度注意力、残差强度见 validation/validation_predictions.jsonl.gz；
按修复/退步、真实类别和连续片段分组的均值见 validation/diagnostic_groups.json。
门控与残差的相关变化是观察性诊断，不能单独证明冻结门控导致旧错误。
旧 Single 前 30 轮重跑与历史有小幅差异；本轮以同时实施的配对结果为主要依据。
