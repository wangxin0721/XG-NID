| model | start F1 | selected best F1 | best epoch | trained best F1 | trained best epoch | last F1 | stop epoch | updates | seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Rel-Mean | 97.320705% | 97.360584% | 129 | 97.360584% | 129 | 97.318199% | 150 | 31650 | 1359.705 |
| Rel-Gate | 97.320705% | 97.403928% | 134 | 97.403928% | 134 | 97.287865% | 150 | 31650 | 1142.189 |

Gate-Mean=+0.043344 pp; Gate-D32=+0.083223 pp.
Delta above max(Mean,D32)=+0.043344 pp; >=+0.10 pp budget passed=False; required F1=97.460584%.
Gate versus user-reported rounded J32-Gate 97.368275%=+0.035653 pp. J32 artifacts not used or independently verified.
Both begin at mature D32-Gate epoch62. Epoch0 is saved separately; final selection compares it with the best trained epoch, preferring epoch0 on ties.
Only second-layer Packet-destination relation selection changes. All GAT/BN parameters and buffers remain fixed/eval.
Packet readout remains mean. Zero-edge convolution outputs/bias remain present, without new masks.
Adam WD0, existing head1e-5, relation score1e-4, batch512/16, seed42, registered min50/max150; both arms complete150 epochs.
Early stopping is disabled. Stale counters describe trained-epoch progress only; patience20 is recorded but never truncates training.
epoch_zero.pt, trained_best.pt, best.pt and last.pt are independent artifacts. Trained-best predictions are exported separately.
Active parameter counts: Mean7144; Gate10266, including the new3122 relation parameters.

| class | Gate-Mean recall pp | Gate-D32 recall pp | Gate-Mean F1 pp | support |
|---|---:|---:|---:|---:|
| Benign | +0.209293 | +0.418585 | +0.229065 | 2389 |
| WebBased | +0.000000 | +0.551471 | -0.082450 | 544 |
| Spoofing | +0.172513 | -0.287522 | +0.256441 | 1739 |
| Recon | -0.043898 | +0.131694 | -0.065398 | 2278 |
| Mirai | +0.000000 | +0.000000 | +0.000000 | 2400 |
| Dos | +0.000000 | -0.067797 | +0.000000 | 1475 |
| DDos | +0.000000 | -0.041876 | +0.000000 | 2388 |
| BruteForce | +0.429185 | +0.000000 | +0.009093 | 233 |

Review WebBased, Recon, Spoofing and BruteForce with all eight classes.
Gate_vs_Mean: repaired=21, regressed=13, net=8.
Mean_vs_D32: repaired=23, regressed=22, net=1.
Gate_vs_D32: repaired=35, regressed=26, net=9.
Trained_Rel-Mean_vs_D32: repaired=23, regressed=22, net=1.
Trained_Rel-Gate_vs_D32: repaired=35, regressed=26, net=9.

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

Rel-Mean trained evidence: {'best_trained_epoch': 129, 'best_trained_macro_f1': 0.9736058430246577, 'best_trained_minus_start_pp': 0.0398788743210865, 'last_trained_macro_f1': 0.9731819926661666, 'last_trained_minus_start_pp': -0.002506161528026407}; groups: [{'name': 'existing_head', 'lr': 1e-05, 'parameters': 7144}].

Rel-Mean best_validation CM (rows=true):

| true / predicted | Benign | WebBased | Spoofing | Recon | Mirai | Dos | DDos | BruteForce |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Benign | 2323 | 11 | 44 | 8 | 1 | 1 | 0 | 1 |
| WebBased | 7 | 502 | 6 | 26 | 0 | 1 | 0 | 2 |
| Spoofing | 59 | 7 | 1664 | 7 | 0 | 0 | 0 | 2 |
| Recon | 10 | 36 | 10 | 2219 | 0 | 1 | 1 | 1 |
| Mirai | 0 | 0 | 0 | 0 | 2400 | 0 | 0 | 0 |
| Dos | 1 | 2 | 0 | 6 | 0 | 1466 | 0 | 0 |
| DDos | 0 | 0 | 1 | 5 | 0 | 0 | 2382 | 0 |
| BruteForce | 1 | 1 | 2 | 0 | 0 | 0 | 0 | 229 |

Rel-Mean trained_best_validation CM (rows=true):

| true / predicted | Benign | WebBased | Spoofing | Recon | Mirai | Dos | DDos | BruteForce |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Benign | 2323 | 11 | 44 | 8 | 1 | 1 | 0 | 1 |
| WebBased | 7 | 502 | 6 | 26 | 0 | 1 | 0 | 2 |
| Spoofing | 59 | 7 | 1664 | 7 | 0 | 0 | 0 | 2 |
| Recon | 10 | 36 | 10 | 2219 | 0 | 1 | 1 | 1 |
| Mirai | 0 | 0 | 0 | 0 | 2400 | 0 | 0 | 0 |
| Dos | 1 | 2 | 0 | 6 | 0 | 1466 | 0 | 0 |
| DDos | 0 | 0 | 1 | 5 | 0 | 0 | 2382 | 0 |
| BruteForce | 1 | 1 | 2 | 0 | 0 | 0 | 0 | 229 |

Rel-Mean last_validation CM (rows=true):

| true / predicted | Benign | WebBased | Spoofing | Recon | Mirai | Dos | DDos | BruteForce |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Benign | 2323 | 11 | 44 | 8 | 1 | 1 | 0 | 1 |
| WebBased | 7 | 498 | 6 | 30 | 0 | 1 | 0 | 2 |
| Spoofing | 57 | 7 | 1666 | 7 | 0 | 0 | 0 | 2 |
| Recon | 9 | 36 | 10 | 2220 | 0 | 1 | 1 | 1 |
| Mirai | 0 | 0 | 0 | 0 | 2400 | 0 | 0 | 0 |
| Dos | 1 | 2 | 0 | 6 | 0 | 1466 | 0 | 0 |
| DDos | 0 | 0 | 1 | 5 | 0 | 0 | 2382 | 0 |
| BruteForce | 1 | 1 | 2 | 0 | 0 | 0 | 0 | 229 |

Rel-Gate trained evidence: {'best_trained_epoch': 134, 'best_trained_macro_f1': 0.9740392823207363, 'best_trained_minus_start_pp': 0.08322280392895065, 'last_trained_macro_f1': 0.9728786528857153, 'last_trained_minus_start_pp': -0.03284013957315368}; groups: [{'name': 'existing_head', 'lr': 1e-05, 'parameters': 7144}, {'name': 'relation_score', 'lr': 0.0001, 'parameters': 3122}].

Rel-Gate best_validation CM (rows=true):

| true / predicted | Benign | WebBased | Spoofing | Recon | Mirai | Dos | DDos | BruteForce |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Benign | 2328 | 11 | 39 | 8 | 1 | 1 | 0 | 1 |
| WebBased | 6 | 502 | 5 | 27 | 0 | 1 | 0 | 3 |
| Spoofing | 55 | 7 | 1667 | 8 | 0 | 0 | 0 | 2 |
| Recon | 9 | 38 | 10 | 2218 | 0 | 1 | 1 | 1 |
| Mirai | 0 | 0 | 0 | 0 | 2400 | 0 | 0 | 0 |
| Dos | 1 | 2 | 0 | 6 | 0 | 1466 | 0 | 0 |
| DDos | 0 | 0 | 1 | 5 | 0 | 0 | 2382 | 0 |
| BruteForce | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 230 |

Rel-Gate trained_best_validation CM (rows=true):

| true / predicted | Benign | WebBased | Spoofing | Recon | Mirai | Dos | DDos | BruteForce |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Benign | 2328 | 11 | 39 | 8 | 1 | 1 | 0 | 1 |
| WebBased | 6 | 502 | 5 | 27 | 0 | 1 | 0 | 3 |
| Spoofing | 55 | 7 | 1667 | 8 | 0 | 0 | 0 | 2 |
| Recon | 9 | 38 | 10 | 2218 | 0 | 1 | 1 | 1 |
| Mirai | 0 | 0 | 0 | 0 | 2400 | 0 | 0 | 0 |
| Dos | 1 | 2 | 0 | 6 | 0 | 1466 | 0 | 0 |
| DDos | 0 | 0 | 1 | 5 | 0 | 0 | 2382 | 0 |
| BruteForce | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 230 |

Rel-Gate last_validation CM (rows=true):

| true / predicted | Benign | WebBased | Spoofing | Recon | Mirai | Dos | DDos | BruteForce |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Benign | 2327 | 11 | 40 | 8 | 1 | 1 | 0 | 1 |
| WebBased | 5 | 499 | 5 | 30 | 0 | 1 | 0 | 4 |
| Spoofing | 55 | 7 | 1667 | 8 | 0 | 0 | 0 | 2 |
| Recon | 9 | 36 | 10 | 2220 | 0 | 1 | 1 | 1 |
| Mirai | 0 | 0 | 0 | 0 | 2400 | 0 | 0 | 0 |
| Dos | 1 | 2 | 0 | 6 | 0 | 1466 | 0 | 0 |
| DDos | 0 | 0 | 1 | 5 | 0 | 0 | 2382 | 0 |
| BruteForce | 1 | 1 | 2 | 1 | 0 | 0 | 0 | 228 |

relation_diagnostics.json includes empty relations, per-node missing incoming messages, zero norms and relative differences.
Cache contains preaggregation first-layer/Flow-message/Packet-message tensors; old postaggregation caches are not used.
Per-sample selected predictions retain all32 fusion weights and two relation weights for every Packet node.
Changed sample IDs: *_changed_samples.jsonl.gz. Best/last/source class P/R/F1: per_class.csv.
GPU backward/initial identity, raw gradient probes, clean-state restoration and full best-checkpoint raw validation replays passed.
Matched relation mean/BN/readout on identical frozen messages uses 1e-6; independent legacy CUDA probabilities recorded separately; all source predictions/CM must match.
Budget threshold is not statistical significance. No automatic extra model, global weights, new seed or test evaluation.
