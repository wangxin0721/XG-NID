| model | best validation Macro-F1 | last Macro-F1 | epoch0 Macro-F1 | best epoch | stop epoch | updates | train seconds | active parameters |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D32-Equal | 97.143955% | 96.750737% | 6.738299% | 72 | 92 | 19412 | 130.414 | 5480 |
| D32-Gate | 97.320705% | 97.224864% | 6.738299% | 62 | 82 | 17302 | 227.067 | 7144 |
| historical D32 | 97.320705% | 97.224864% | 6.738299% | 62 | 82 | 17302 | 204.936 | 7144 |

Gate-Equal: +0.176751 pp.
Gate-historical D32: +0.000000 pp.
Budget screen (>= +0.10 pp): True; class review still required.
Both arms reuse the same clean D32 snapshot, NOT the epoch62 checkpoint. Epoch0 is recorded but excluded.
Adam lr=1e-3, weight_decay=0, train/eval batch512/16, max100/patience20; encoders/BN fixed in eval.
Equal stores 1,664 unused frozen gate parameters for strict checkpoint compatibility; its active count is 5,480.

| class | Equal best P/R/F1 (%) | Gate best P/R/F1 (%) | F1 delta pp |
|---|---:|---:|---:|
| Benign | 96.9507 / 97.1536 / 97.0521 | 97.2315 / 97.0280 / 97.1297 | +0.077628 |
| WebBased | 87.1080 / 91.9118 / 89.4454 | 88.4752 / 91.7279 / 90.0722 | +0.626764 |
| Spoofing | 96.1406 / 95.9747 / 96.0576 | 96.2025 / 96.1472 / 96.1749 | +0.117309 |
| Recon | 97.8695 / 96.7954 / 97.3295 | 97.5771 / 97.2344 / 97.4055 | +0.075945 |
| Mirai | 100.0000 / 100.0000 / 100.0000 | 99.9584 / 100.0000 / 99.9792 | -0.020829 |
| Dos | 99.7958 / 99.3898 / 99.5924 | 99.8639 / 99.4576 / 99.6603 | +0.067935 |
| DDos | 99.8742 / 99.7487 / 99.8114 | 99.9161 / 99.7906 / 99.8533 | +0.041902 |
| BruteForce | 97.4468 / 98.2833 / 97.8632 | 97.8723 / 98.7124 / 98.2906 | +0.427350 |

Gate versus Equal: repaired=46, regressed=34, net_correct=12.
Inspect WebBased missed attacks, Spoofing losses, and BruteForce gains with the complete eight-class results.
Prediction destinations (all changed samples):

| change | true class | Equal prediction | Gate prediction | samples |
|---|---|---|---|---:|
| both_wrong_changed | Benign | WebBased | Spoofing | 1 |
| both_wrong_changed | Benign | Spoofing | Recon | 1 |
| both_wrong_changed | Benign | Recon | Mirai | 1 |
| both_wrong_changed | WebBased | Benign | Recon | 1 |
| both_wrong_changed | Dos | Benign | WebBased | 1 |
| both_wrong_changed | DDos | WebBased | Recon | 1 |
| regressed | Benign | Benign | Spoofing | 9 |
| regressed | Benign | Benign | Recon | 3 |
| regressed | WebBased | WebBased | Benign | 1 |
| regressed | WebBased | WebBased | Recon | 9 |
| regressed | WebBased | WebBased | Dos | 1 |
| regressed | Spoofing | Spoofing | Benign | 4 |
| regressed | Spoofing | Spoofing | WebBased | 1 |
| regressed | Spoofing | Spoofing | Recon | 3 |
| regressed | Recon | Recon | Benign | 1 |
| regressed | Recon | Recon | WebBased | 2 |
| repaired | Benign | WebBased | Benign | 1 |
| repaired | Benign | Spoofing | Benign | 8 |
| repaired | WebBased | Benign | WebBased | 3 |
| repaired | WebBased | Spoofing | WebBased | 1 |
| repaired | WebBased | Recon | WebBased | 6 |
| repaired | Spoofing | Benign | Spoofing | 7 |
| repaired | Spoofing | Recon | Spoofing | 2 |
| repaired | Spoofing | DDos | Spoofing | 1 |
| repaired | Spoofing | BruteForce | Spoofing | 1 |
| repaired | Recon | Benign | Recon | 1 |
| repaired | Recon | WebBased | Recon | 9 |
| repaired | Recon | Spoofing | Recon | 1 |
| repaired | Recon | Dos | Recon | 2 |
| repaired | Dos | Recon | Dos | 1 |
| repaired | DDos | Recon | DDos | 1 |
| repaired | BruteForce | WebBased | BruteForce | 1 |

D32-Equal best_validation confusion matrix (rows=true, columns=prediction):

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

D32-Equal last_validation confusion matrix (rows=true, columns=prediction):

| true / predicted | Benign | WebBased | Spoofing | Recon | Mirai | Dos | DDos | BruteForce |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Benign | 2327 | 10 | 38 | 12 | 0 | 1 | 0 | 1 |
| WebBased | 11 | 450 | 11 | 70 | 0 | 0 | 0 | 2 |
| Spoofing | 57 | 6 | 1665 | 9 | 0 | 0 | 1 | 1 |
| Recon | 9 | 20 | 9 | 2236 | 0 | 1 | 2 | 1 |
| Mirai | 0 | 0 | 0 | 0 | 2400 | 0 | 0 | 0 |
| Dos | 1 | 1 | 0 | 7 | 0 | 1466 | 0 | 0 |
| DDos | 0 | 0 | 0 | 5 | 0 | 0 | 2383 | 0 |
| BruteForce | 1 | 2 | 2 | 2 | 0 | 0 | 0 | 226 |

D32-Gate best_validation confusion matrix (rows=true, columns=prediction):

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

D32-Gate last_validation confusion matrix (rows=true, columns=prediction):

| true / predicted | Benign | WebBased | Spoofing | Recon | Mirai | Dos | DDos | BruteForce |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Benign | 2317 | 13 | 48 | 8 | 0 | 1 | 0 | 2 |
| WebBased | 8 | 509 | 7 | 16 | 0 | 2 | 0 | 2 |
| Spoofing | 52 | 9 | 1670 | 5 | 0 | 0 | 1 | 2 |
| Recon | 13 | 48 | 10 | 2203 | 0 | 2 | 1 | 1 |
| Mirai | 0 | 0 | 0 | 0 | 2400 | 0 | 0 | 0 |
| Dos | 0 | 3 | 0 | 5 | 0 | 1467 | 0 | 0 |
| DDos | 0 | 1 | 1 | 4 | 0 | 0 | 2382 | 0 |
| BruteForce | 1 | 1 | 2 | 0 | 0 | 0 | 0 | 229 |

Common paired epochs verified: 82.
Public initialization identical: True; full initial validation equality passed.
Encoder parameters/BN unchanged: True; shared caches unchanged: True.
Best checkpoints strictly reloaded and replayed through raw validation graphs; all predictions/CM reproduced.
Shared cache load/validation seconds (charged once): 0.946.
Preflight including cache seconds: 40.169; discarded probe updates: 6.
Historical rerun materially different (>= 0.05 pp): False.
Execution comparison: {'common_protocol_differences': {}, 'source_and_rerun_initial_head_identical': True, 'source_and_rerun_training_rng_identical': True, 'same_model_definition_verified_by_source_sha256': True, 'same_64D_cache_verified_by_sha256': True, 'common_epochs_with_historical_D32': 82, 'max_common_epoch_macro_f1_abs_difference': 0.0, 'max_common_epoch_train_nll_abs_difference': 0.0, 'source_gpu': {'physical_index': 1, 'uuid': 'GPU-a9142835-62e0-974e-aebe-80ab25f7ef02', 'name': 'Tesla T4', 'total_mib': 15360, 'used_mib_at_query': 8696, 'free_mib_at_query': 6216, 'allow_shared_gpu': True, 'other_processes_at_query': [{'pid': 693237, 'name': 'python'}], 'timing_scope': 'sharing permitted; no isolated speed claim'}, 'rerun_gpu': {'physical_index': 1, 'uuid': 'GPU-a9142835-62e0-974e-aebe-80ab25f7ef02', 'name': 'Tesla T4', 'total_mib': 15360, 'used_mib_at_query': 8696, 'free_mib_at_query': 6216, 'allow_shared_gpu': True, 'other_processes_at_query': [{'pid': 693237, 'name': 'python'}], 'timing_scope': 'sharing permitted; no isolated speed claim'}, 'runtime': {'torch': '2.3.1+cu118', 'pyg': '2.5.3', 'python': '3.9.23', 'cuda': '11.8'}, 'unexplained_difference_policy': 'Review hardware/numerical execution if scores materially differ despite matching verified inputs and settings; do not infer a cause from the score alone.'}
If the historical rerun differs materially, explain the execution differences before interpreting gate contribution.
The matched pair measures the effect of adding this trainable dynamic gate, including its 1,664 parameters and optimization. It does not isolate adaptation from gate capacity. The historical D32 is a rerun reference, never a training start. No fraction of the absolute 97.32% score can be assigned to the gate.
No significance claim. No test evaluation, extra seeds or automatic joint training.
