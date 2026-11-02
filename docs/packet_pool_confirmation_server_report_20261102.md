| model | epoch0 F1 | trained best F1 | trained epoch | selected F1 | selected epoch | last F1 | epochs | updates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 | 97.320705% | 97.360584% | 129 | 97.360584% | 129 | 97.318199% | 150 | 31650 |
| P1 | 97.320705% | 97.427191% | 146 | 97.427191% | 146 | 97.380197% | 150 | 31650 |

One additional paired confirmation only; this is a training-budget check, not a new model improvement.
Both restart from mature D32-Gate epoch62 with fresh Adam/RNG; no epoch20 model or optimizer is resumed.
Both complete all150 epochs; early stopping disabled. Registered min50 does not trigger stopping.
Epoch0 participates in final selection. trained_best.pt selects epochs1..150 independently; ties retain epoch0.
Frozen encoders and BN; same ordered packet mean/attention; same seed42, split, LR1e-5/1e-4, WD0, NLL, batch512/16.
Trainable parameters unchanged: P0=7144, P1=8200; added attention network=1056.

P1-P0=+0.066607 pp; P1-D32=+0.106486 pp.
P1-max(P0,D32)=+0.066607 pp; +0.10 pp budget passed=False.
Both first20 trajectories reproduced=True; budget verdict changed after20=False.

| model | first20 order same | first20 scores same | exact train NLL same | NLL max abs error | first20 selected F1 | full150 selected F1 | gain after20 pp |
|---|---|---|---|---:|---:|---:|---:|
| P0 | True | True | True | 0 | 97.320705% | 97.360584% | +0.039879 |
| P1 | True | True | True | 0 | 97.320705% | 97.427191% | +0.106486 |

Same P0/P1 models and hyperparameters; only the fixed training budget changes. If the first20 trajectory differs, do not attribute differences solely to removing early stopping.

| class | P1-P0 recall pp | P1-D32 recall pp | P1-P0 F1 pp | support |
|---|---:|---:|---:|---:|
| Benign | +0.125576 | +0.334868 | +0.165829 | 2389 |
| WebBased | +0.367647 | +0.919118 | +0.114762 | 544 |
| Spoofing | +0.230017 | -0.230017 | +0.230814 | 1739 |
| Recon | +0.000000 | +0.175593 | +0.021451 | 2278 |
| Mirai | +0.000000 | +0.000000 | +0.000000 | 2400 |
| Dos | +0.000000 | -0.067797 | +0.000000 | 1475 |
| DDos | +0.000000 | -0.041876 | +0.000000 | 2388 |
| BruteForce | +0.000000 | -0.429185 | +0.000000 | 233 |

Review WebBased/Recon recall and Spoofing/BruteForce tradeoffs.
P1_vs_P0: repaired=11, regressed=2, net=9.
P0_vs_D32: repaired=23, regressed=22, net=1.
P1_vs_D32: repaired=29, regressed=19, net=10.

P0: trained evidence={'best_trained_epoch': 129, 'best_trained_macro_f1': 0.9736058430246577, 'best_trained_minus_start_pp': 0.0398788743210865, 'last_trained_macro_f1': 0.9731819926661666, 'last_trained_minus_start_pp': -0.002506161528026407}; parameter groups=[{'name': 'existing_head', 'lr': 1e-05, 'parameters': 7144}].

P0 best_validation confusion matrix (rows=true):

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

P0 trained_best_validation confusion matrix (rows=true):

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

P0 last_validation confusion matrix (rows=true):

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

P1: trained evidence={'best_trained_epoch': 146, 'best_trained_macro_f1': 0.9742719120553645, 'best_trained_minus_start_pp': 0.10648577739176268, 'last_trained_macro_f1': 0.9738019699520168, 'last_trained_minus_start_pp': 0.05949156705700176}; parameter groups=[{'name': 'existing_head', 'lr': 1e-05, 'parameters': 7144}, {'name': 'new_packet_pool', 'lr': 0.0001, 'parameters': 1056}].

P1 best_validation confusion matrix (rows=true):

| true / predicted | Benign | WebBased | Spoofing | Recon | Mirai | Dos | DDos | BruteForce |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Benign | 2326 | 11 | 41 | 8 | 1 | 1 | 0 | 1 |
| WebBased | 7 | 504 | 5 | 25 | 0 | 1 | 0 | 2 |
| Spoofing | 55 | 7 | 1668 | 7 | 0 | 0 | 0 | 2 |
| Recon | 9 | 37 | 10 | 2219 | 0 | 1 | 1 | 1 |
| Mirai | 0 | 0 | 0 | 0 | 2400 | 0 | 0 | 0 |
| Dos | 1 | 2 | 0 | 6 | 0 | 1466 | 0 | 0 |
| DDos | 0 | 0 | 1 | 5 | 0 | 0 | 2382 | 0 |
| BruteForce | 1 | 1 | 2 | 0 | 0 | 0 | 0 | 229 |

P1 trained_best_validation confusion matrix (rows=true):

| true / predicted | Benign | WebBased | Spoofing | Recon | Mirai | Dos | DDos | BruteForce |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Benign | 2326 | 11 | 41 | 8 | 1 | 1 | 0 | 1 |
| WebBased | 7 | 504 | 5 | 25 | 0 | 1 | 0 | 2 |
| Spoofing | 55 | 7 | 1668 | 7 | 0 | 0 | 0 | 2 |
| Recon | 9 | 37 | 10 | 2219 | 0 | 1 | 1 | 1 |
| Mirai | 0 | 0 | 0 | 0 | 2400 | 0 | 0 | 0 |
| Dos | 1 | 2 | 0 | 6 | 0 | 1466 | 0 | 0 |
| DDos | 0 | 0 | 1 | 5 | 0 | 0 | 2382 | 0 |
| BruteForce | 1 | 1 | 2 | 0 | 0 | 0 | 0 | 229 |

P1 last_validation confusion matrix (rows=true):

| true / predicted | Benign | WebBased | Spoofing | Recon | Mirai | Dos | DDos | BruteForce |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Benign | 2326 | 11 | 41 | 8 | 1 | 1 | 0 | 1 |
| WebBased | 7 | 500 | 5 | 29 | 0 | 1 | 0 | 2 |
| Spoofing | 55 | 7 | 1668 | 7 | 0 | 0 | 0 | 2 |
| Recon | 9 | 36 | 10 | 2220 | 0 | 1 | 1 | 1 |
| Mirai | 0 | 0 | 0 | 0 | 2400 | 0 | 0 | 0 |
| Dos | 1 | 2 | 0 | 6 | 0 | 1466 | 0 | 0 |
| DDos | 0 | 0 | 1 | 5 | 0 | 0 | 2382 | 0 |
| BruteForce | 1 | 1 | 2 | 0 | 0 | 0 | 0 | 229 |

The completed20-epoch screen and all its artifacts remain unchanged; only its frozen node caches are copied and verified.
Cache identities, split, sample order, file/tensor hashes and encoder hashes are checked again.
Cache memory figures describe original extraction, not current copying or whole-process RSS.
Current graph sample loads={'train': 1120, 'val': 53784, 'test': 0}; original cache-extraction reads are excluded.
Initial, selected-best and trained-best checkpoints pass original-graph validation replay (shared when identical).
Matched readout identity retains atol/rtol1e-6; raw/cache1e-5. Independent legacy predictions/CM must match exactly.
Full P/R/F1/support: per_class.csv; changed samples: *_changed_samples.jsonl.gz; destinations: prediction_transitions.csv.
Packet diagnostics: packet_diagnostics.json; load receipts: graph_load_accounting.json.
No test evaluation, extra seed, new hyperparameter search, additional confirmation round or automatic experiment. Budget threshold is not significance.
