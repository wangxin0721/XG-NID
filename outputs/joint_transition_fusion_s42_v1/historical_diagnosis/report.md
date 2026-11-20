# Ordered 相对 Single 的定向误差诊断

验证样本 13446，预测变化 50。

| 真实类别 | 修复 | 退步 | 错误间变化 |
| --- | ---: | ---: | ---: |
| Benign | 8 | 9 | 1 |
| WebBased | 4 | 4 | 0 |
| Spoofing | 4 | 1 | 2 |
| Recon | 4 | 8 | 2 |
| Mirai | 0 | 0 | 0 |
| Dos | 2 | 0 | 0 |
| DDos | 0 | 0 | 0 |
| BruteForce | 0 | 0 | 1 |

| 类别 | 有效连续两步覆盖 |
| --- | ---: |
| Benign | 1300/2389 (54.4161%) |
| WebBased | 404/544 (74.2647%) |
| Spoofing | 798/1739 (45.8884%) |
| Recon | 332/2278 (14.5742%) |
| Mirai | 2399/2400 (99.9583%) |
| Dos | 493/1475 (33.4237%) |
| DDos | 2384/2388 (99.8325%) |
| BruteForce | 227/233 (97.4249%) |

## repaired

| 真实类别: 原预测 → 新预测 | 数量 |
| --- | ---: |
| Benign: WebBased -> Benign | 2 |
| Benign: Spoofing -> Benign | 4 |
| Benign: Recon -> Benign | 2 |
| WebBased: Recon -> WebBased | 3 |
| WebBased: Benign -> WebBased | 1 |
| Spoofing: Benign -> Spoofing | 4 |
| Recon: Benign -> Recon | 1 |
| Recon: WebBased -> Recon | 3 |
| Dos: Recon -> Dos | 2 |

有效一步转移数量分组：{"8": 2, "10": 1, "2": 4, "5": 1, "3": 3, "19": 2, "9": 2, "1": 7}

有效连续两步片段数量分组：{"7": 2, "9": 1, "1": 4, "4": 1, "2": 3, "18": 2, "8": 2, "0": 7}

## regressed

| 真实类别: 原预测 → 新预测 | 数量 |
| --- | ---: |
| Benign: Benign -> Spoofing | 6 |
| Benign: Benign -> Recon | 2 |
| Benign: Benign -> WebBased | 1 |
| WebBased: WebBased -> Benign | 2 |
| WebBased: WebBased -> Dos | 2 |
| Spoofing: Spoofing -> Recon | 1 |
| Recon: Recon -> WebBased | 4 |
| Recon: Recon -> Dos | 1 |
| Recon: Recon -> Benign | 2 |
| Recon: Recon -> Spoofing | 1 |

有效一步转移数量分组：{"19": 3, "3": 4, "1": 5, "5": 1, "8": 2, "2": 3, "4": 1, "9": 2, "7": 1}

有效连续两步片段数量分组：{"18": 3, "2": 4, "0": 5, "4": 1, "7": 2, "1": 3, "3": 1, "8": 2, "6": 1}

## wrong_changed

| 真实类别: 原预测 → 新预测 | 数量 |
| --- | ---: |
| Benign: WebBased -> Spoofing | 1 |
| Spoofing: Recon -> Benign | 1 |
| Spoofing: DDos -> Recon | 1 |
| Recon: BruteForce -> Spoofing | 1 |
| Recon: Benign -> Spoofing | 1 |
| BruteForce: Spoofing -> WebBased | 1 |

有效一步转移数量分组：{"19": 1, "15": 1, "6": 2, "3": 1, "1": 1}

有效连续两步片段数量分组：{"18": 1, "14": 1, "5": 2, "2": 1, "0": 1}

观察与推测：归档中有 7 例修复、5 例退步只有一条有效转移，因此这些样本没有连续两步片段。
WebBased 与 Recon 的连续片段覆盖明显不均衡；BruteForce 的一例变化为错误类别之间的切换。
转移数大于等于 2 不能推定存在连续片段。未知项需原图连续性标记回放，不能填为 0。
冻结门控限制新增表示利用率属于待检验假设；本诊断不作因果归因。

证据：{"archive_commit": "944ba61faca474d02c409c2038e02f33e5da3d50", "raw_graph_features_replayed": true, "all_historical_predictions_identical": true, "per_sample_continuity_available": true, "attention_residual_and_gate_available": true}
