| model | start Macro-F1 | best Macro-F1 | gain pp | last Macro-F1 | best epoch | stop epoch | updates | joint seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| J32-Equal | 97.143955% | 97.143955% | +0.000000 | 97.100336% | 0 | 12 | 2532 | 2806.276 |
| J32-Gate | 97.320705% | 97.320705% | +0.000000 | 97.185821% | 0 | 12 | 2532 | 2457.739 |

Inherited Gate-Equal gap: +0.176751 pp.
Final Gate-Equal gap: +0.176751 pp.
Gap change from joint training: +0.000000 pp.
Gate above stronger original-Gate/new-Equal reference: +0.000000 pp.
Budget screen >= +0.10 pp: False; required score=97.420705%.
Selected epoch0: Equal=True, Gate=True.
Epoch0 is eligible. Matching the starting score alone is not evidence of benefit.
No head reinitialization or warmup. Fresh Adam: encoders1e-6, projections/head1e-5, gate1e-5; WD0.
Raw graphs, batch512/16, max30/patience12. Encoder modules remain eval but allow gradients; BN buffers fixed.

| class | J32-Equal P/R/F1 (%) | J32-Gate P/R/F1 (%) | Gate F1 gain from own start pp |
|---|---:|---:|---:|
| Benign | 96.9507 / 97.1536 / 97.0521 | 97.2315 / 97.0280 / 97.1297 | +0.000000 |
| WebBased | 87.1080 / 91.9118 / 89.4454 | 88.4752 / 91.7279 / 90.0722 | +0.000000 |
| Spoofing | 96.1406 / 95.9747 / 96.0576 | 96.2025 / 96.1472 / 96.1749 | +0.000000 |
| Recon | 97.8695 / 96.7954 / 97.3295 | 97.5771 / 97.2344 / 97.4055 | +0.000000 |
| Mirai | 100.0000 / 100.0000 / 100.0000 | 99.9584 / 100.0000 / 99.9792 | +0.000000 |
| Dos | 99.7958 / 99.3898 / 99.5924 | 99.8639 / 99.4576 / 99.6603 | +0.000000 |
| DDos | 99.8742 / 99.7487 / 99.8114 | 99.9161 / 99.7906 / 99.8533 | +0.000000 |
| BruteForce | 97.4468 / 98.2833 / 97.8632 | 97.8723 / 98.7124 / 98.2906 | +0.000000 |

Review WebBased FN/FP separately, Spoofing changes and the small BruteForce sample count.
J32-Equal versus own start: repaired=0, regressed=0, net_correct=0.
J32-Equal best-trained evidence: {'best_trained_epoch': 12, 'best_trained_macro_f1': 0.9710033649995367, 'best_trained_minus_start_pp': -0.04361824570584538, 'last_trained_macro_f1': 0.9710033649995367, 'last_trained_minus_start_pp': -0.04361824570584538}.
J32-Equal trainable parameters=850024; groups=[{'name': 'encoder', 'lr': 1e-06, 'parameters': 844544}, {'name': 'head', 'lr': 1e-05, 'parameters': 5480}].
J32-Gate versus own start: repaired=0, regressed=0, net_correct=0.
J32-Gate best-trained evidence: {'best_trained_epoch': 8, 'best_trained_macro_f1': 0.9723805081299784, 'best_trained_minus_start_pp': -0.0826546151468377, 'last_trained_macro_f1': 0.9718582061125053, 'last_trained_minus_start_pp': -0.1348848168941541}.
J32-Gate trainable parameters=851688; groups=[{'name': 'encoder', 'lr': 1e-06, 'parameters': 844544}, {'name': 'head', 'lr': 1e-05, 'parameters': 5480}, {'name': 'gate', 'lr': 1e-05, 'parameters': 1664}].
Final Gate versus Equal: repaired=46, regressed=34, net_correct=12.

J32-Equal best_validation confusion matrix (rows=true):

| true / predicted | Benign | WebBased | Spoofing | Recon | Mirai | Dos | DDos | BruteForce |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Benign | 2321 | 13 | 48 | 5 | 0 | 1 | 0 | 1 |
| WebBased | 10 | 500 | 7 | 25 | 0 | 0 | 0 | 2 |
| Spoofing | 51 | 9 | 1669 | 7 | 0 | 0 | 1 | 2 |
| Recon | 10 | 48 | 10 | 2205 | 0 | 2 | 2 | 1 |
| Mirai | 0 | 0 | 0 | 0 | 2400 | 0 | 0 | 0 |
| Dos | 1 | 2 | 0 | 6 | 0 | 1466 | 0 | 0 |
| DDos | 0 | 1 | 0 | 5 | 0 | 0 | 2382 | 0 |
| BruteForce | 1 | 1 | 2 | 0 | 0 | 0 | 0 | 229 |

J32-Equal last_validation confusion matrix (rows=true):

| true / predicted | Benign | WebBased | Spoofing | Recon | Mirai | Dos | DDos | BruteForce |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Benign | 2323 | 12 | 43 | 9 | 0 | 1 | 0 | 1 |
| WebBased | 9 | 483 | 5 | 45 | 0 | 0 | 0 | 2 |
| Spoofing | 54 | 7 | 1669 | 6 | 0 | 0 | 1 | 2 |
| Recon | 11 | 35 | 11 | 2217 | 0 | 2 | 1 | 1 |
| Mirai | 0 | 0 | 0 | 0 | 2400 | 0 | 0 | 0 |
| Dos | 1 | 1 | 0 | 7 | 0 | 1466 | 0 | 0 |
| DDos | 0 | 0 | 1 | 5 | 0 | 0 | 2382 | 0 |
| BruteForce | 1 | 1 | 2 | 0 | 0 | 0 | 0 | 229 |

J32-Gate best_validation confusion matrix (rows=true):

| true / predicted | Benign | WebBased | Spoofing | Recon | Mirai | Dos | DDos | BruteForce |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Benign | 2318 | 11 | 49 | 8 | 1 | 1 | 0 | 1 |
| WebBased | 7 | 499 | 6 | 29 | 0 | 1 | 0 | 2 |
| Spoofing | 48 | 10 | 1672 | 8 | 0 | 0 | 0 | 1 |
| Recon | 10 | 41 | 9 | 2215 | 0 | 0 | 2 | 1 |
| Mirai | 0 | 0 | 0 | 0 | 2400 | 0 | 0 | 0 |
| Dos | 0 | 3 | 0 | 5 | 0 | 1467 | 0 | 0 |
| DDos | 0 | 0 | 0 | 5 | 0 | 0 | 2383 | 0 |
| BruteForce | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 230 |

J32-Gate last_validation confusion matrix (rows=true):

| true / predicted | Benign | WebBased | Spoofing | Recon | Mirai | Dos | DDos | BruteForce |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Benign | 2324 | 11 | 43 | 8 | 1 | 1 | 0 | 1 |
| WebBased | 8 | 491 | 7 | 35 | 0 | 1 | 0 | 2 |
| Spoofing | 57 | 7 | 1666 | 7 | 0 | 0 | 0 | 2 |
| Recon | 10 | 38 | 10 | 2217 | 0 | 1 | 1 | 1 |
| Mirai | 0 | 0 | 0 | 0 | 2400 | 0 | 0 | 0 |
| Dos | 1 | 2 | 0 | 6 | 0 | 1466 | 0 | 0 |
| DDos | 0 | 0 | 1 | 5 | 0 | 0 | 2382 | 0 |
| BruteForce | 1 | 1 | 2 | 0 | 0 | 0 | 0 | 229 |

Per-class best/last/source metrics: per_class.csv. Prediction destinations: prediction_transitions.csv.
Changed samples: each arm changes_from_own_start.jsonl.gz; root gate_vs_equal_changes.jsonl.gz.
Prediction files retain all 32 gate weights for every validation sample.
Common epoch order/RNG verified: 12; encoder BN buffers unchanged.
Best checkpoints independently reconstructed, strictly loaded and replayed on all raw validation graphs.
Preflight seconds=80.900; discarded probe updates=6.
Joint seconds include training, epoch validation and checkpoint writes; final replay is separately recorded.
The two arms start from their own different mature checkpoints. Own-start deltas measure additional gains from this joint fine-tuning protocol; the inherited gate advantage is not a new joint-training gain. Encoder eval mode fixes BN buffers while encoder affine/other parameters remain trainable. This pair does not isolate gate capacity from adaptation.
No test evaluation, significance claim, extra seed or automatic next training.
