| model | start F1 | selected best F1 | best epoch | trained best F1 | trained best epoch | last F1 | stop epoch | updates | seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CL-Last | 97.320705% | 97.467535% | 113 | 97.467535% | 113 | 97.321104% | 150 | 31650 | 423.356 |
| CL-Multi | 97.320705% | 97.509986% | 126 | 97.509986% | 126 | 97.337434% | 150 | 31650 | 433.430 |

Multi-Last=+0.042451 pp; Multi-D32=+0.189281 pp.
Delta above max(Last,D32)=+0.042451 pp; >=+0.10 pp budget passed=False; required F1=97.567535%.
Multi versus reported P1 97.427191%=+0.082795 pp. P1 score is a reference only; its checkpoint is not loaded.
Both begin at mature D32-Gate epoch62. Epoch0 is saved separately; final selection compares it with the best trained epoch, preferring epoch0 on ties.
Only Packet cross-layer readout changes. All encoder and BN parameters/buffers remain frozen/eval.
Both use a128-to64 linear readout. Last inputs[h2,h2]; Multi inputs[h1,h2]. Initialization[0,I],bias0 reproduces h2.
Adam WD0, existing head1e-5, cross-layer projection1e-4, batch512/16, seed42, registered min50/max150; both arms complete150 epochs.
Early stopping is disabled. Stale counters describe trained-epoch progress only; patience20 is recorded but never truncates training.
epoch_zero.pt, trained_best.pt, best.pt and last.pt are independent artifacts. Trained-best predictions are exported separately.
Active parameter counts: Last15400; Multi15400, each including8256 new projection parameters.

| class | Multi-Last recall pp | Multi-D32 recall pp | Multi-Last F1 pp | support |
|---|---:|---:|---:|---:|
| Benign | -0.293010 | +0.334868 | -0.089477 | 2389 |
| WebBased | -0.735294 | +2.022059 | +0.102043 | 544 |
| Spoofing | +0.287522 | -0.345026 | -0.127766 | 1739 |
| Recon | +0.131694 | -0.131694 | +0.024703 | 2278 |
| Mirai | +0.000000 | +0.000000 | +0.000000 | 2400 |
| Dos | +0.000000 | -0.067797 | +0.000000 | 1475 |
| DDos | +0.000000 | -0.041876 | +0.000000 | 2388 |
| BruteForce | +0.429185 | -0.429185 | +0.430108 | 233 |

Review WebBased, Recon, Spoofing and BruteForce with all eight classes.
Multi_vs_Last: repaired=15, regressed=17, net=-2.
Last_vs_D32: repaired=41, regressed=32, net=9.
Multi_vs_D32: repaired=38, regressed=31, net=7.
Trained_CL-Last_vs_D32: repaired=41, regressed=32, net=9.
Trained_CL-Multi_vs_D32: repaired=38, regressed=31, net=7.

| partition | class | graphs | varying depths | layer1 norm | layer2 norm | L2 difference | cosine |
|---|---|---:|---:|---:|---:|---:|---:|
| train | Benign | 19112 | 19112 | 6.58616 | 7.91188 | 7.99722 | 0.443862 |
| train | WebBased | 4359 | 4359 | 6.30772 | 6.47498 | 6.81238 | 0.45033 |
| train | Spoofing | 13914 | 13914 | 6.80696 | 8.08923 | 8.25794 | 0.424122 |
| train | Recon | 18224 | 18224 | 6.18981 | 5.91613 | 6.38261 | 0.453337 |
| train | Mirai | 19200 | 19200 | 4.054 | 4.18002 | 4.74694 | 0.338441 |
| train | Dos | 11802 | 11802 | 5.73185 | 6.33132 | 6.23751 | 0.463284 |
| train | DDos | 19140 | 19140 | 5.22179 | 3.25765 | 4.51623 | 0.51426 |
| train | BruteForce | 1870 | 1870 | 8.16487 | 7.78987 | 8.97852 | 0.403025 |
| val | Benign | 2389 | 2389 | 6.67172 | 8.00906 | 8.1416 | 0.440633 |
| val | WebBased | 544 | 544 | 6.38201 | 6.44397 | 6.84872 | 0.448503 |
| val | Spoofing | 1739 | 1739 | 6.87391 | 8.04438 | 8.29076 | 0.421101 |
| val | Recon | 2278 | 2278 | 6.17443 | 5.9384 | 6.38408 | 0.453427 |
| val | Mirai | 2400 | 2400 | 4.05735 | 4.18269 | 4.75509 | 0.337172 |
| val | Dos | 1475 | 1475 | 5.74806 | 6.31997 | 6.25752 | 0.460161 |
| val | DDos | 2388 | 2388 | 5.2234 | 3.2601 | 4.51901 | 0.514085 |
| val | BruteForce | 233 | 233 | 8.44151 | 7.8562 | 9.23128 | 0.398723 |

CL-Last trained evidence: {'best_trained_epoch': 113, 'best_trained_macro_f1': 0.9746753476680805, 'best_trained_minus_start_pp': 0.1468293386633679, 'last_trained_macro_f1': 0.9732110430950174, 'last_trained_minus_start_pp': 0.00039888135705545835}; groups: [{'name': 'existing_head', 'lr': 1e-05, 'parameters': 7144}, {'name': 'cross_layer_projection', 'lr': 0.0001, 'parameters': 8256}].

CL-Last best_validation CM (rows=true):

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

CL-Last trained_best_validation CM (rows=true):

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

CL-Last last_validation CM (rows=true):

| true / predicted | Benign | WebBased | Spoofing | Recon | Mirai | Dos | DDos | BruteForce |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Benign | 2325 | 12 | 42 | 8 | 1 | 1 | 0 | 0 |
| WebBased | 7 | 504 | 4 | 24 | 0 | 2 | 0 | 3 |
| Spoofing | 61 | 6 | 1662 | 8 | 0 | 0 | 0 | 2 |
| Recon | 10 | 39 | 10 | 2216 | 0 | 1 | 1 | 1 |
| Mirai | 0 | 0 | 0 | 0 | 2400 | 0 | 0 | 0 |
| Dos | 1 | 2 | 0 | 6 | 0 | 1466 | 0 | 0 |
| DDos | 0 | 0 | 1 | 5 | 0 | 0 | 2382 | 0 |
| BruteForce | 1 | 1 | 2 | 0 | 1 | 0 | 0 | 228 |

CL-Multi trained evidence: {'best_trained_epoch': 126, 'best_trained_macro_f1': 0.9750998613318087, 'best_trained_minus_start_pp': 0.18928070503618777, 'last_trained_macro_f1': 0.973374337800741, 'last_trained_minus_start_pp': 0.016728351929418128}; groups: [{'name': 'existing_head', 'lr': 1e-05, 'parameters': 7144}, {'name': 'cross_layer_projection', 'lr': 0.0001, 'parameters': 8256}].

CL-Multi best_validation CM (rows=true):

| true / predicted | Benign | WebBased | Spoofing | Recon | Mirai | Dos | DDos | BruteForce |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Benign | 2326 | 12 | 42 | 7 | 1 | 1 | 0 | 0 |
| WebBased | 7 | 510 | 8 | 16 | 0 | 2 | 0 | 1 |
| Spoofing | 59 | 6 | 1666 | 7 | 0 | 0 | 0 | 1 |
| Recon | 13 | 40 | 10 | 2212 | 0 | 1 | 1 | 1 |
| Mirai | 0 | 0 | 0 | 0 | 2400 | 0 | 0 | 0 |
| Dos | 1 | 2 | 0 | 6 | 0 | 1466 | 0 | 0 |
| DDos | 0 | 0 | 1 | 5 | 0 | 0 | 2382 | 0 |
| BruteForce | 1 | 1 | 2 | 0 | 0 | 0 | 0 | 229 |

CL-Multi trained_best_validation CM (rows=true):

| true / predicted | Benign | WebBased | Spoofing | Recon | Mirai | Dos | DDos | BruteForce |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Benign | 2326 | 12 | 42 | 7 | 1 | 1 | 0 | 0 |
| WebBased | 7 | 510 | 8 | 16 | 0 | 2 | 0 | 1 |
| Spoofing | 59 | 6 | 1666 | 7 | 0 | 0 | 0 | 1 |
| Recon | 13 | 40 | 10 | 2212 | 0 | 1 | 1 | 1 |
| Mirai | 0 | 0 | 0 | 0 | 2400 | 0 | 0 | 0 |
| Dos | 1 | 2 | 0 | 6 | 0 | 1466 | 0 | 0 |
| DDos | 0 | 0 | 1 | 5 | 0 | 0 | 2382 | 0 |
| BruteForce | 1 | 1 | 2 | 0 | 0 | 0 | 0 | 229 |

CL-Multi last_validation CM (rows=true):

| true / predicted | Benign | WebBased | Spoofing | Recon | Mirai | Dos | DDos | BruteForce |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Benign | 2324 | 14 | 40 | 9 | 1 | 1 | 0 | 0 |
| WebBased | 7 | 502 | 5 | 26 | 0 | 2 | 0 | 2 |
| Spoofing | 60 | 6 | 1663 | 8 | 0 | 0 | 0 | 2 |
| Recon | 10 | 39 | 9 | 2217 | 0 | 1 | 1 | 1 |
| Mirai | 0 | 0 | 0 | 0 | 2400 | 0 | 0 | 0 |
| Dos | 1 | 2 | 0 | 6 | 0 | 1466 | 0 | 0 |
| DDos | 0 | 0 | 1 | 5 | 0 | 0 | 2382 | 0 |
| BruteForce | 1 | 1 | 2 | 0 | 0 | 0 | 0 | 229 |

cross_layer_diagnostics.json compares post-BN/activation graph mean representations at both depths.
Cache stores flow64, Packet first64 and second64 graph means; old last-layer-only caches are not reused.
Per-sample selected predictions retain all32 original fusion weights.
Changed sample IDs: *_changed_samples.jsonl.gz. Best/last/source class P/R/F1: per_class.csv.
GPU backward/initial identity, raw gradient probes, clean-state restoration and full best-checkpoint raw validation replays passed.
Matched cross-layer readout on identical frozen graph means uses 1e-6; independent legacy CUDA probabilities recorded separately; all source predictions/CM must match.
Budget threshold is not statistical significance. No residual connection, pooled attention, adaptive relations, new seed, test evaluation or automatic follow-up.
