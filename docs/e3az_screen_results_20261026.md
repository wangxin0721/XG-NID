# E3a-Z训练/验证筛选回传（2026-10-26）

完整metrics回传，variant E3a_equal_init，最佳56轮、停止68轮，更新14348、曝光7318228，与211次更新/轮、107621样本/轮、patience12一致。参数846800，训练batch512、评价batch16，耗时14953.762秒约4小时9分14秒。test=null，未评价测试集。

标准库复算13446样本矩阵及8类Precision/Recall/F1、总Accuracy/Macro/Weighted通过，误差小于1e-12。原配对初态SHA与E3a一致，仅gate.net.2.weight/bias两个张量变化，实际初始权重及probe/screen均值都是.5/.5，精确身份检查通过。这是报告字段一致性，不是本机加载权重独立核验。配置、checkpoint和根summary未以文件回传，不能独立重算其SHA。机器记录见[e3az_screen_results_20261026.json](e3az_screen_results_20261026.json)。

| 候选 | 验证Accuracy | 验证Macro-F1 | 验证Weighted-F1 |
|---|---:|---:|---:|
| E0-B512 | 97.6870% | 96.6729% | 97.6872% |
| E2-B512 | 97.5755% | 96.5994% | 97.5880% |
| E3a | 97.7094% | 96.5613% | 97.7094% |
| E3a-Z | 97.5606% | 96.4594% | 97.5819% |
| 原E3-B512 | 97.3896% | 96.3669% | 97.3839% |

Z相对E3a的Macro低.1019个百分点，相对E2低.1400，相对E0低.2135。当前seed42不支持等权初始化改善总体选模指标，但不能推广为所有seed无效；不再扩展初始化变体。E3a跑92轮而Z跑68轮，实际优化量不同；相同训练策略不等于相同更新总量。

## WebBased与Recon的权衡

Z的WebBased TP494、FN50、FP102，Precision82.8859%、Recall90.8088%、F1 86.6667%。相对E3a TP468/FN76/FP71，增加26个真阳性，同时增加31个假阳性；F1只增加.2401个百分点。WebBased→Recon从60降至26，Recon→WebBased从37增至75，不能只据前一个方向说混淆解决。Z的Recon正确2168，E3a为2216，少48例。

最佳验证最终权重均值Flow.399781/Packet.600219，极端某路>.95合计约36.2859%。初始等权后已经学到样本差异，不是永久等权模型；均值偏Packet不能直接解释具体样本错误。需候选诊断核验，不能套用E3a或原E3统计。

## 下一步

使用已发布8fe7065的diagnose_gate_candidate.py，variant=E3a_equal_init，screen-report=outputs/e3_gate_screen_E3aZ_b512_s42_v1/summary.json，新输出outputs/e3_gate_candidate_E3aZ_b512_s42_v1；同一原始诊断报告及冻结配置。诊断完成后用921e1b6离线工具对齐原E3和Z。已完成的E3a诊断/缓存分析不重跑。

A/C/B和lambda=.05保持，尚未实现/执行其训练入口。先补齐Z的同口径机制证据，随后按登记设置实现并探测三项；不增加类别损失、新融合位置或初始化候选。旧E0/E2保留为强参考，测试集不参与这轮筛选。
