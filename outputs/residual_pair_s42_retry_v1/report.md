| model | start F1 | selected best F1 | best epoch | trained best F1 | trained best epoch | last F1 | stop epoch | updates | seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Res-Base | 97.320705% | 97.467535% | 113 | 97.467535% | 113 | 97.336075% | 150 | 31650 | 3213.085 |
| Res-Skip | 97.320705% | 97.475129% | 37 | 97.475129% | 37 | 97.411989% | 150 | 31650 | 3213.131 |

Skip-Base=+0.007594 pp; Skip-D32=+0.154423 pp.
Skip versus CL-Last97.467535%=+0.007594 pp.
Delta above max(Base,D32)=+0.007594 pp; >=+0.10 pp budget passed=False; required F1=97.567535%.
Skip versus verified CL-Multi 97.509986%=-0.034857 pp. CL-Multi weights are not used to initialize this experiment.
Both begin at mature D32-Gate epoch62. Epoch0 is saved separately; final selection compares it with the best trained epoch, preferring epoch0 on ties.
Only Packet second-layer pre-BN residual differs. Both share the same CL-Last-style128-to64 projection and training budget; old GAT/BN remain frozen/eval.
Residual R=Linear64/64,biasFalse initialized zero; add after the original relation mean and before BN/activation. Empty-relation outputs unchanged.
Adam WD0, existing head1e-5, common projection/residual1e-4, batch512/16, seed42, registered min50/max150; both arms complete150 epochs.
Early stopping is disabled. Stale counters describe trained-epoch progress only; patience20 is recorded but never truncates training.
epoch_zero.pt, trained_best.pt, best.pt and last.pt are independent artifacts. Trained-best predictions are exported separately.
Active parameter counts: Base15400; Skip19496; common projection8256 in both; residual4096 only in Skip.

| class | Skip-Base recall pp | Skip-D32 recall pp | Skip-Base F1 pp | support |
|---|---:|---:|---:|---:|
| Benign | -0.669736 | -0.041859 | +0.143407 | 2389 |
| WebBased | -0.919118 | +1.838235 | +0.004628 | 544 |
| Spoofing | +1.035078 | +0.402530 | +0.149613 | 1739 |
| Recon | +0.307287 | +0.043898 | -0.035628 | 2278 |
| Mirai | +0.000000 | +0.000000 | +0.000000 | 2400 |
| Dos | +0.000000 | -0.067797 | +0.000000 | 1475 |
| DDos | +0.000000 | -0.041876 | +0.000000 | 2388 |
| BruteForce | +0.429185 | -0.429185 | -0.201268 | 233 |

Review WebBased, Recon, Spoofing and BruteForce with all eight classes.
Skip_vs_Base: repaired=33, regressed=28, net=5.
Base_vs_D32: repaired=41, regressed=32, net=9.
Skip_vs_D32: repaired=33, regressed=19, net=14.
Trained_Res-Base_vs_D32: repaired=41, regressed=32, net=9.
Trained_Res-Skip_vs_D32: repaired=33, regressed=19, net=14.

| partition | class | nodes | empty F relation graphs | empty P relation graphs | mean F norm | mean P norm | mean message L2 difference | mean cosine |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| train | Benign | 156716 | 0 | 4569 | 9.80558 | 17.7937 | 20.9364 | -0.0406602 |
| train | WebBased | 37327 | 0 | 61 | 7.35478 | 17.3966 | 19.0262 | 0.0148958 |
| train | Spoofing | 109243 | 0 | 1438 | 9.02856 | 17.8028 | 20.319 | -0.00940036 |
| train | Recon | 56231 | 0 | 2003 | 7.7138 | 17.9075 | 20.3398 | -0.0501959 |
| train | Mirai | 383585 | 0 | 5 | 15.2732 | 9.28894 | 14.9215 | 0.345102 |
| train | Dos | 21072 | 0 | 7102 | 9.52226 | 8.25097 | 13.3642 | 0.114797 |
| train | DDos | 382299 | 0 | 0 | 9.52205 | 13.549 | 18.7192 | -0.2951 |
| train | BruteForce | 24247 | 0 | 13 | 9.08495 | 20.4487 | 19.7825 | 0.302894 |
| val | Benign | 19865 | 0 | 585 | 9.84441 | 17.9362 | 20.9593 | -0.0259179 |
| val | WebBased | 4943 | 0 | 5 | 7.64853 | 17.3738 | 19.0721 | 0.0185873 |
| val | Spoofing | 14367 | 0 | 170 | 9.03139 | 17.8498 | 20.361 | -0.00862951 |
| val | Recon | 6877 | 0 | 264 | 7.65132 | 17.6958 | 20.1277 | -0.0490425 |
| val | Mirai | 47952 | 0 | 0 | 15.2382 | 9.2925 | 14.8873 | 0.346012 |
| val | Dos | 2674 | 0 | 872 | 9.49241 | 8.42975 | 13.4607 | 0.11073 |
| val | DDos | 47668 | 0 | 0 | 9.52196 | 13.5513 | 18.7212 | -0.295101 |
| val | BruteForce | 3139 | 0 | 2 | 9.17276 | 20.9489 | 20.322 | 0.29435 |

Res-Base trained evidence: {'best_trained_epoch': 113, 'best_trained_macro_f1': 0.9746753476680805, 'best_trained_minus_start_pp': 0.1468293386633679, 'last_trained_macro_f1': 0.9733607548753435, 'last_trained_minus_start_pp': 0.015370059389663293}; groups: [{'name': 'existing_head', 'lr': 1e-05, 'parameters': 7144}, {'name': 'common_last_projection', 'lr': 0.0001, 'parameters': 8256}].

Res-Base best_validation CM (rows=true):

| true / predicted | Benign | WebBased | Spoofing | Recon | Mirai | Dos | DDos | BruteForce |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Benign | 2333 | 12 | 36 | 6 | 1 | 1 | 0 | 0 |
| WebBased | 8 | 514 | 4 | 14 | 0 | 2 | 0 | 2 |
| Spoofing | 62 | 7 | 1661 | 7 | 0 | 0 | 0 | 2 |
| Recon | 12 | 45 | 10 | 2209 | 0 | 1 | 1 | 0 |
| Mirai | 0 | 0 | 0 | 0 | 2400 | 0 | 0 | 0 |
| Dos | 1 | 2 | 0 | 6 | 0 | 1466 | 0 | 0 |
| DDos | 0 | 0 | 1 | 5 | 0 | 0 | 2382 | 0 |
| BruteForce | 1 | 1 | 2 | 1 | 0 | 0 | 0 | 228 |

Res-Base trained_best_validation CM (rows=true):

| true / predicted | Benign | WebBased | Spoofing | Recon | Mirai | Dos | DDos | BruteForce |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Benign | 2333 | 12 | 36 | 6 | 1 | 1 | 0 | 0 |
| WebBased | 8 | 514 | 4 | 14 | 0 | 2 | 0 | 2 |
| Spoofing | 62 | 7 | 1661 | 7 | 0 | 0 | 0 | 2 |
| Recon | 12 | 45 | 10 | 2209 | 0 | 1 | 1 | 0 |
| Mirai | 0 | 0 | 0 | 0 | 2400 | 0 | 0 | 0 |
| Dos | 1 | 2 | 0 | 6 | 0 | 1466 | 0 | 0 |
| DDos | 0 | 0 | 1 | 5 | 0 | 0 | 2382 | 0 |
| BruteForce | 1 | 1 | 2 | 1 | 0 | 0 | 0 | 228 |

Res-Base last_validation CM (rows=true):

| true / predicted | Benign | WebBased | Spoofing | Recon | Mirai | Dos | DDos | BruteForce |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Benign | 2325 | 12 | 42 | 8 | 1 | 1 | 0 | 0 |
| WebBased | 7 | 505 | 4 | 23 | 0 | 2 | 0 | 3 |
| Spoofing | 61 | 6 | 1662 | 8 | 0 | 0 | 0 | 2 |
| Recon | 10 | 39 | 10 | 2216 | 0 | 1 | 1 | 1 |
| Mirai | 0 | 0 | 0 | 0 | 2400 | 0 | 0 | 0 |
| Dos | 1 | 2 | 0 | 6 | 0 | 1466 | 0 | 0 |
| DDos | 0 | 0 | 1 | 5 | 0 | 0 | 2382 | 0 |
| BruteForce | 1 | 1 | 2 | 0 | 1 | 0 | 0 | 228 |

Res-Skip trained evidence: {'best_trained_epoch': 37, 'best_trained_macro_f1': 0.9747512882409469, 'best_trained_minus_start_pp': 0.15442339595000965, 'last_trained_macro_f1': 0.9741198865026095, 'last_trained_minus_start_pp': 0.09128322211626738}; groups: [{'name': 'existing_head', 'lr': 1e-05, 'parameters': 7144}, {'name': 'common_last_projection', 'lr': 0.0001, 'parameters': 8256}, {'name': 'new_residual', 'lr': 0.0001, 'parameters': 4096}].

Res-Skip best_validation CM (rows=true):

| true / predicted | Benign | WebBased | Spoofing | Recon | Mirai | Dos | DDos | BruteForce |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Benign | 2317 | 11 | 49 | 9 | 1 | 1 | 0 | 1 |
| WebBased | 5 | 509 | 6 | 20 | 0 | 2 | 0 | 2 |
| Spoofing | 44 | 6 | 1679 | 8 | 0 | 0 | 0 | 2 |
| Recon | 10 | 40 | 9 | 2216 | 0 | 1 | 1 | 1 |
| Mirai | 0 | 0 | 0 | 0 | 2400 | 0 | 0 | 0 |
| Dos | 0 | 3 | 0 | 6 | 0 | 1466 | 0 | 0 |
| DDos | 0 | 0 | 1 | 5 | 0 | 0 | 2382 | 0 |
| BruteForce | 1 | 1 | 2 | 0 | 0 | 0 | 0 | 229 |

Res-Skip trained_best_validation CM (rows=true):

| true / predicted | Benign | WebBased | Spoofing | Recon | Mirai | Dos | DDos | BruteForce |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Benign | 2317 | 11 | 49 | 9 | 1 | 1 | 0 | 1 |
| WebBased | 5 | 509 | 6 | 20 | 0 | 2 | 0 | 2 |
| Spoofing | 44 | 6 | 1679 | 8 | 0 | 0 | 0 | 2 |
| Recon | 10 | 40 | 9 | 2216 | 0 | 1 | 1 | 1 |
| Mirai | 0 | 0 | 0 | 0 | 2400 | 0 | 0 | 0 |
| Dos | 0 | 3 | 0 | 6 | 0 | 1466 | 0 | 0 |
| DDos | 0 | 0 | 1 | 5 | 0 | 0 | 2382 | 0 |
| BruteForce | 1 | 1 | 2 | 0 | 0 | 0 | 0 | 229 |

Res-Skip last_validation CM (rows=true):

| true / predicted | Benign | WebBased | Spoofing | Recon | Mirai | Dos | DDos | BruteForce |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Benign | 2323 | 11 | 41 | 11 | 1 | 1 | 0 | 1 |
| WebBased | 7 | 507 | 6 | 19 | 0 | 2 | 0 | 3 |
| Spoofing | 54 | 5 | 1672 | 6 | 0 | 0 | 0 | 2 |
| Recon | 13 | 38 | 9 | 2215 | 0 | 1 | 1 | 1 |
| Mirai | 0 | 0 | 0 | 0 | 2400 | 0 | 0 | 0 |
| Dos | 0 | 3 | 0 | 5 | 0 | 1467 | 0 | 0 |
| DDos | 0 | 0 | 1 | 5 | 0 | 0 | 2382 | 0 |
| BruteForce | 1 | 1 | 2 | 0 | 1 | 0 | 0 | 228 |

residual_diagnostics.json includes empty relations, per-node missing incoming messages, zero norms and relative differences.
Fresh cache contains pre-BN first-layer/Flow-message/Packet-message tensors. CL graph-level caches cannot recover the required nodes.
Per-sample selected predictions retain all32 fusion weights and residual norm summaries for every graph.
Changed sample IDs: *_changed_samples.jsonl.gz. Best/last/source class P/R/F1: per_class.csv.
Prior checkpoint/fold audit, GPU backward/initial identity, raw gradient probes, clean-state restoration and full best-checkpoint raw validation replays passed.
Matched zero-residual/BN/readout on identical frozen messages uses 1e-6; independent legacy CUDA probabilities recorded separately; all source predictions/CM must match.
Budget threshold is not statistical significance. No adaptive relations, attention pooling, cross-layer shallow readout, new seed, test evaluation or automatic follow-up.
