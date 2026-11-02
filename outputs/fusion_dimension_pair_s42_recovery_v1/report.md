| model | validation Macro-F1 | epoch0 Macro-F1 | best epoch | stop epoch | updates | training seconds |
|---|---:|---:|---:|---:|---:|---:|
| D8 | 97.176708% | 8.382916% | 72 | 92 | 19412 | 239.017 |
| D32 | 97.320705% | 6.738299% | 62 | 82 | 17302 | 204.936 |
| historical lower-LR J2 | 97.120712% | — | 9 | 21 | 4431 | 3934.254 |

D32-D8: +0.143998 pp.
D32-historical J2: +0.199993 pp.
Both new heads are freshly initialized. Epoch0 is recorded and excluded from selection.
Only this pair uses max 100 epochs / patience 20.
Historical J2 is a score reference with a different training protocol, not a matched control.

| class | D8 P/R/F1 (%) | D32 P/R/F1 (%) | D32-D8 F1 pp |
|---|---:|---:|---:|
| Benign | 96.7930 / 97.2792 / 97.0355 | 97.2315 / 97.0280 / 97.1297 | +0.094197 |
| WebBased | 87.2193 / 92.8309 / 89.9377 | 88.4752 / 91.7279 / 90.0722 | +0.134535 |
| Spoofing | 96.2601 / 96.2047 / 96.2324 | 96.2025 / 96.1472 / 96.1749 | -0.057521 |
| Recon | 98.1713 / 96.6198 / 97.3894 | 97.5771 / 97.2344 / 97.4055 | +0.016072 |
| Mirai | 99.9584 / 100.0000 / 99.9792 | 99.9584 / 100.0000 / 99.9792 | +0.000000 |
| Dos | 99.7958 / 99.3898 / 99.5924 | 99.8639 / 99.4576 / 99.6603 | +0.067935 |
| DDos | 99.9580 / 99.6650 / 99.8113 | 99.9161 / 99.7906 / 99.8533 | +0.042061 |
| BruteForce | 97.0213 / 97.8541 / 97.4359 | 97.8723 / 98.7124 / 98.2906 | +0.854701 |

D32 versus D8: repaired=51, regressed=44, net_correct=7.
Shared 64D train/val cache reuse/load/validate seconds: 1.142 (charged once).
Encoder/BN state unchanged: True; raw graph checkpoint replays passed.
Source: /home/nzxn/下载/WX/load/XG-NID/outputs/fusion_dimension_pair_s42_v1
Original cache extraction seconds (recorded receipts): 332.68700449098833
Original gradient probes reused: 3 discarded updates per arm. New recovery probe updates: 0.
D8: trainable parameters=2080; gate=[21, 16, 8]; reused original probe updates=3; new probe updates=0; evidence={'best_trained_epoch': 72, 'best_trained_macro_f1': 0.9717670790956123, 'best_trained_minus_start_pp': 88.79379148980686, 'last_trained_macro_f1': 0.9691496134532347, 'last_trained_minus_start_pp': 88.5320449255691}.
D32: trainable parameters=7144; gate=[69, 16, 32]; reused original probe updates=3; new probe updates=0; evidence={'best_trained_epoch': 62, 'best_trained_macro_f1': 0.9732070542814468, 'best_trained_minus_start_pp': 90.58240654039564, 'last_trained_macro_f1': 0.9722486439617891, 'last_trained_minus_start_pp': 90.48656550842988}.
This pair compares fresh dimension-dependent projection/gate/head pipelines on fixed N0 encoders. It changes gate input/output widths and parameter capacity together; it does not isolate a scalar dimension effect or establish the contribution of dynamic versus equal fusion. Historical J2 uses a different training protocol.
No statistical significance claim. Review class tradeoffs before retaining D32.
Dynamic-versus-equal 32D fusion and other seeds are not automatically launched. No test evaluation.
