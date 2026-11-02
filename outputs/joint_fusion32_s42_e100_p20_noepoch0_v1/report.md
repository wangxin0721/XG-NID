| model | start Macro-F1 | best Macro-F1 | gain pp | last Macro-F1 | best epoch | stop epoch | updates | joint seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| J32-Equal | 97.143955% | 97.135367% | -0.008588 | 97.078130% | 33 | 53 | 11183 | 10310.506 |
| J32-Gate | 97.320705% | 97.368275% | +0.047569 | 97.284535% | 45 | 65 | 13715 | 12235.100 |

Inherited Gate-Equal gap: +0.176751 pp.
Final Gate-Equal gap: +0.232908 pp.
Gap change from joint training: +0.056157 pp.
Gate above stronger original-Gate/new-Equal reference: +0.047569 pp.
Budget screen >= +0.10 pp: False; required score=97.420705%.
Selected epoch0: Equal=False, Gate=False.
Epoch0 is recorded but excluded. Select trained epochs only, even when all are below the starting score.
No head reinitialization or warmup. Fresh Adam: encoders1e-6, projections/head1e-5, gate1e-5; WD0.
Raw graphs, batch512/16, max100/patience20. Encoder modules remain eval but allow gradients; BN buffers fixed.

| class | J32-Equal P/R/F1 (%) | J32-Gate P/R/F1 (%) | Gate F1 gain from own start pp |
|---|---:|---:|---:|
| Benign | 96.9102 / 97.1536 / 97.0318 | 96.9154 / 97.3211 / 97.1178 | -0.011893 |
| WebBased | 89.7959 / 88.9706 / 89.3813 | 89.6243 / 92.0956 / 90.8432 | +0.770953 |
| Spoofing | 96.3668 / 96.0897 / 96.2280 | 96.3605 / 95.9172 / 96.1383 | -0.036535 |
| Recon | 96.8573 / 97.4100 / 97.1329 | 97.7523 / 97.3661 / 97.5588 | +0.153377 |
| Mirai | 100.0000 / 100.0000 / 100.0000 | 99.9584 / 100.0000 / 99.9792 | +0.000000 |
| Dos | 99.7958 / 99.3898 / 99.5924 | 99.7958 / 99.3898 / 99.5924 | -0.067935 |
| DDos | 99.9580 / 99.7487 / 99.8533 | 99.9580 / 99.7487 / 99.8533 | -0.000061 |
| BruteForce | 97.4468 / 98.2833 / 97.8632 | 97.4468 / 98.2833 / 97.8632 | -0.427350 |

Review WebBased FN/FP separately, Spoofing changes and the small BruteForce sample count.
J32-Equal versus own start: repaired=27, regressed=27, net_correct=0.
J32-Equal best-trained evidence: {'best_trained_epoch': 33, 'best_trained_macro_f1': 0.9713536710377247, 'best_trained_minus_start_pp': -0.008587641887036757, 'last_trained_macro_f1': 0.9707812975534628, 'last_trained_minus_start_pp': -0.06582499031323463}.
J32-Equal trainable parameters=850024; groups=[{'name': 'encoder', 'lr': 1e-06, 'parameters': 844544}, {'name': 'head', 'lr': 1e-05, 'parameters': 5480}].
J32-Gate versus own start: repaired=26, regressed=21, net_correct=5.
J32-Gate best-trained evidence: {'best_trained_epoch': 45, 'best_trained_macro_f1': 0.9736827480679938, 'best_trained_minus_start_pp': 0.047569378654699346, 'last_trained_macro_f1': 0.9728453496834999, 'last_trained_minus_start_pp': -0.036170459794693244}.
J32-Gate trainable parameters=851688; groups=[{'name': 'encoder', 'lr': 1e-06, 'parameters': 844544}, {'name': 'head', 'lr': 1e-05, 'parameters': 5480}, {'name': 'gate', 'lr': 1e-05, 'parameters': 1664}].
Final Gate versus Equal: repaired=41, regressed=24, net_correct=17.

J32-Equal best_validation confusion matrix (rows=true):

| true / predicted | Benign | WebBased | Spoofing | Recon | Mirai | Dos | DDos | BruteForce |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Benign | 2321 | 12 | 44 | 10 | 0 | 1 | 0 | 1 |
| WebBased | 9 | 484 | 5 | 44 | 0 | 0 | 0 | 2 |
| Spoofing | 53 | 7 | 1671 | 6 | 0 | 0 | 0 | 2 |
| Recon | 10 | 34 | 11 | 2219 | 0 | 2 | 1 | 1 |
| Mirai | 0 | 0 | 0 | 0 | 2400 | 0 | 0 | 0 |
| Dos | 1 | 1 | 0 | 7 | 0 | 1466 | 0 | 0 |
| DDos | 0 | 0 | 1 | 5 | 0 | 0 | 2382 | 0 |
| BruteForce | 1 | 1 | 2 | 0 | 0 | 0 | 0 | 229 |

J32-Equal last_validation confusion matrix (rows=true):

| true / predicted | Benign | WebBased | Spoofing | Recon | Mirai | Dos | DDos | BruteForce |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Benign | 2321 | 12 | 44 | 10 | 0 | 1 | 0 | 1 |
| WebBased | 10 | 482 | 6 | 44 | 0 | 0 | 0 | 2 |
| Spoofing | 55 | 7 | 1669 | 6 | 0 | 0 | 0 | 2 |
| Recon | 10 | 35 | 11 | 2218 | 0 | 2 | 1 | 1 |
| Mirai | 0 | 0 | 0 | 0 | 2400 | 0 | 0 | 0 |
| Dos | 1 | 1 | 0 | 7 | 0 | 1466 | 0 | 0 |
| DDos | 0 | 0 | 1 | 5 | 0 | 0 | 2382 | 0 |
| BruteForce | 1 | 1 | 2 | 0 | 0 | 0 | 0 | 229 |

J32-Gate best_validation confusion matrix (rows=true):

| true / predicted | Benign | WebBased | Spoofing | Recon | Mirai | Dos | DDos | BruteForce |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Benign | 2325 | 11 | 43 | 7 | 1 | 1 | 0 | 1 |
| WebBased | 7 | 501 | 7 | 26 | 0 | 1 | 0 | 2 |
| Spoofing | 55 | 7 | 1668 | 7 | 0 | 0 | 0 | 2 |
| Recon | 10 | 37 | 10 | 2218 | 0 | 1 | 1 | 1 |
| Mirai | 0 | 0 | 0 | 0 | 2400 | 0 | 0 | 0 |
| Dos | 1 | 2 | 0 | 6 | 0 | 1466 | 0 | 0 |
| DDos | 0 | 0 | 1 | 5 | 0 | 0 | 2382 | 0 |
| BruteForce | 1 | 1 | 2 | 0 | 0 | 0 | 0 | 229 |

J32-Gate last_validation confusion matrix (rows=true):

| true / predicted | Benign | WebBased | Spoofing | Recon | Mirai | Dos | DDos | BruteForce |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Benign | 2324 | 11 | 44 | 7 | 1 | 1 | 0 | 1 |
| WebBased | 7 | 495 | 7 | 32 | 0 | 1 | 0 | 2 |
| Spoofing | 55 | 7 | 1668 | 7 | 0 | 0 | 0 | 2 |
| Recon | 10 | 36 | 10 | 2219 | 0 | 1 | 1 | 1 |
| Mirai | 0 | 0 | 0 | 0 | 2400 | 0 | 0 | 0 |
| Dos | 1 | 2 | 0 | 6 | 0 | 1466 | 0 | 0 |
| DDos | 0 | 0 | 1 | 5 | 0 | 0 | 2382 | 0 |
| BruteForce | 1 | 1 | 2 | 0 | 0 | 0 | 0 | 229 |

Per-class best/last/source metrics: per_class.csv. Prediction destinations: prediction_transitions.csv.
Changed samples: each arm changes_from_own_start.jsonl.gz; root gate_vs_equal_changes.jsonl.gz.
Prediction files retain all 32 gate weights for every validation sample.
Common epoch order/RNG verified: 53; encoder BN buffers unchanged.
Best checkpoints independently reconstructed, strictly loaded and replayed on all raw validation graphs.
Preflight seconds=84.326; discarded probe updates=6.
Joint seconds include training, epoch validation and checkpoint writes; final replay is separately recorded.
The two arms start from their own different mature checkpoints. Own-start deltas measure additional gains from this joint fine-tuning protocol; the inherited gate advantage is not a new joint-training gain. Encoder eval mode fixes BN buffers while encoder affine/other parameters remain trainable. This pair does not isolate gate capacity from adaptation.
No test evaluation, significance claim, extra seed or automatic next training.
