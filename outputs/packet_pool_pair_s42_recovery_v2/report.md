| model | start Macro-F1 | best Macro-F1 | last Macro-F1 | best epoch | stop epoch | updates | seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 | 97.320705% | 97.320705% | 97.246592% | 0 | 20 | 4220 | 94.313 |
| P1 | 97.320705% | 97.320705% | 97.223279% | 0 | 20 | 4220 | 122.972 |

P1-P0=+0.000000 pp; P1-D32=+0.000000 pp; P0-D32=+0.000000 pp.
Delta=P1-max(P0,D32)=+0.000000 pp; budget >= 0.10 pp: False; required P1=97.420705%.
Both start from D32-Gate epoch62. Epoch0 is eligible; retaining it is not evidence of a gain.
Frozen encoders/BN; existing projection/gate/head lr1e-5; new pool lr1e-4; fresh Adam WD0.
Final NLL, seed42, batch512/16, max100/patience20; active parameters P0=7144, P1=8200.

| class | D32 recall | P0 recall | P1 recall | P1-P0 recall pp | P1-D32 recall pp | P1-P0 F1 pp |
|---|---:|---:|---:|---:|---:|---:|
| Benign | 97.028045% | 97.028045% | 97.028045% | +0.000000 | +0.000000 | +0.000000 |
| WebBased | 91.727941% | 91.727941% | 91.727941% | +0.000000 | +0.000000 | +0.000000 |
| Spoofing | 96.147211% | 96.147211% | 96.147211% | +0.000000 | +0.000000 | +0.000000 |
| Recon | 97.234416% | 97.234416% | 97.234416% | +0.000000 | +0.000000 | +0.000000 |
| Mirai | 100.000000% | 100.000000% | 100.000000% | +0.000000 | +0.000000 | +0.000000 |
| Dos | 99.457627% | 99.457627% | 99.457627% | +0.000000 | +0.000000 | +0.000000 |
| DDos | 99.790620% | 99.790620% | 99.790620% | +0.000000 | +0.000000 | +0.000000 |
| BruteForce | 98.712446% | 98.712446% | 98.712446% | +0.000000 | +0.000000 | +0.000000 |

Review WebBased/Recon recall and Spoofing/BruteForce tradeoffs with support counts in per_class.csv.
P1_vs_P0: repaired=0, regressed=0, net_correct=0.
P0_vs_D32: repaired=0, regressed=0, net_correct=0.
P1_vs_D32: repaired=0, regressed=0, net_correct=0.

| partition | class | graphs | multi-packet graphs | varying multi-packet graphs | mean packets | mean multi-packet RMS distance |
|---|---|---:|---:|---:|---:|---:|
| train | Benign | 19112 | 14543 | 12813 | 8.1999 | 2.0558123096873 |
| train | WebBased | 4359 | 4298 | 4061 | 8.5632 | 2.3877429682792304 |
| train | Spoofing | 13914 | 12476 | 11119 | 7.8513 | 1.3806825091337531 |
| train | Recon | 18224 | 16221 | 4854 | 3.0855 | 0.9766663940659497 |
| train | Mirai | 19200 | 19195 | 19191 | 19.9784 | 0.815344297315532 |
| train | Dos | 11802 | 4700 | 3586 | 1.7855 | 0.0017627813463821367 |
| train | DDos | 19140 | 19140 | 14717 | 19.9738 | 0.005811141473812911 |
| train | BruteForce | 1870 | 1857 | 1813 | 12.9663 | 1.231667028699153 |
| val | Benign | 2389 | 1804 | 1588 | 8.3152 | 2.0344241635689646 |
| val | WebBased | 544 | 539 | 512 | 9.0864 | 2.461851774662442 |
| val | Spoofing | 1739 | 1569 | 1399 | 8.2616 | 1.3915693326092011 |
| val | Recon | 2278 | 2014 | 612 | 3.0189 | 0.9677996227153606 |
| val | Mirai | 2400 | 2400 | 2400 | 19.9800 | 0.8156828046993663 |
| val | Dos | 1475 | 603 | 443 | 1.8129 | 0.012706439780444482 |
| val | DDos | 2388 | 2388 | 1790 | 19.9615 | 0.006336692036093847 |
| val | BruteForce | 233 | 231 | 228 | 13.4721 | 1.3041508135818018 |

P0 trained-epoch evidence: {'best_trained_epoch': 15, 'best_trained_macro_f1': 0.9726189736584797, 'best_trained_minus_start_pp': -0.058808062296711316, 'last_trained_macro_f1': 0.9724659219697434, 'last_trained_minus_start_pp': -0.0741132311703474}; parameter groups: [{'name': 'existing_head', 'lr': 1e-05, 'parameters': 7144}].

P0 best_validation confusion matrix (rows=true):

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

P0 last_validation confusion matrix (rows=true):

| true / predicted | Benign | WebBased | Spoofing | Recon | Mirai | Dos | DDos | BruteForce |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Benign | 2326 | 11 | 41 | 8 | 1 | 1 | 0 | 1 |
| WebBased | 7 | 495 | 6 | 33 | 0 | 1 | 0 | 2 |
| Spoofing | 57 | 7 | 1666 | 7 | 0 | 0 | 0 | 2 |
| Recon | 10 | 39 | 10 | 2216 | 0 | 1 | 1 | 1 |
| Mirai | 0 | 0 | 0 | 0 | 2400 | 0 | 0 | 0 |
| Dos | 1 | 2 | 0 | 6 | 0 | 1466 | 0 | 0 |
| DDos | 0 | 0 | 1 | 5 | 0 | 0 | 2382 | 0 |
| BruteForce | 1 | 1 | 2 | 0 | 0 | 0 | 0 | 229 |

P1 trained-epoch evidence: {'best_trained_epoch': 8, 'best_trained_macro_f1': 0.9728488330893352, 'best_trained_minus_start_pp': -0.03582211921115963, 'last_trained_macro_f1': 0.9722327922964583, 'last_trained_minus_start_pp': -0.09742619849885115}; parameter groups: [{'name': 'existing_head', 'lr': 1e-05, 'parameters': 7144}, {'name': 'new_packet_pool', 'lr': 0.0001, 'parameters': 1056}].

P1 best_validation confusion matrix (rows=true):

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

P1 last_validation confusion matrix (rows=true):

| true / predicted | Benign | WebBased | Spoofing | Recon | Mirai | Dos | DDos | BruteForce |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Benign | 2326 | 11 | 41 | 8 | 1 | 1 | 0 | 1 |
| WebBased | 7 | 493 | 5 | 36 | 0 | 1 | 0 | 2 |
| Spoofing | 56 | 7 | 1667 | 7 | 0 | 0 | 0 | 2 |
| Recon | 10 | 39 | 10 | 2216 | 0 | 1 | 1 | 1 |
| Mirai | 0 | 0 | 0 | 0 | 2400 | 0 | 0 | 0 |
| Dos | 1 | 2 | 0 | 6 | 0 | 1466 | 0 | 0 |
| DDos | 0 | 0 | 1 | 5 | 0 | 0 | 2382 | 0 |
| BruteForce | 1 | 1 | 2 | 0 | 0 | 0 | 0 | 229 |

Full class P/R/F1/support: per_class.csv; prediction destinations: prediction_transitions.csv.
Changed sample IDs: *_changed_samples.jsonl.gz, for both arms versus D32 and P1 versus P0.
Each validation prediction retains all 32 fusion weights and per-packet attention in original node order.
Packet counts and within-graph representation distances: packet_diagnostics.json.
Common epochs with identical order/RNG: 20; discarded probe updates=6.
Both best checkpoints strictly reconstructed and replayed on every original validation graph.
P1-P0 measures the packet readout change including its 1,056 added parameters and training. P0 measures the same mature checkpoint with continued downstream training. The budget threshold is not a statistical significance criterion.
No test evaluation or automatic follow-up experiment.

Prepared node caches reused from the preserved failed run: /home/nzxn/下载/WX/load/XG-NID/outputs/packet_pool_pair_s42_v1
Original cache-extraction reads are provenance only and are excluded from current-run graph-load totals.
The cache memory estimates below describe the original extraction, not current copying/loading.
No formal or probe updates existed before the known initial readout failure.

Graph sample loads (including repeated reads): {'train': 1120, 'val': 53784, 'test': 0}.
Main-process guard counts: {'train': 1120, 'val': 0, 'test': 0}; worker loads use parent-verified stage samples.
These counts describe sample loads, not physical disk IO operations.

| loader phase | partition | parent-verified samples | workers |
|---|---|---:|---:|
| initial_P0 | val | 13446 | 2 |
| initial_P1 | val | 13446 | 2 |
| best_P0 | val | 13446 | 2 |
| best_P1 | val | 13446 | 2 |

| cache | graphs | packet nodes | final tensor GiB | concatenation tensor peak GiB including retained cache |
|---|---:|---:|---:|---:|
| train | 107621 | 1170720 | 0.306384 | 0.612768 |
| val | 13446 | 147485 | 0.038569 | 0.383523 |

Memory estimates cover cache tensor storage at concatenation, not whole-process RSS.
Python objects, source graphs, worker prefetch, models, batches and allocator/serialization buffers are excluded.
Full byte estimates and host MemAvailable before concatenation: cache_memory_estimates.json.
Full load-count receipts: graph_load_accounting.json.

Packet readout uses PyTorch 2.3.1 ordered segment reductions in both arms, with the same zero-score softmax for uniform weights.
Device repeat/uniform/backward preflight passed without model/RNG changes or optimizer updates.
Paired readouts and the source head on identical raw features retain atol/rtol=1e-6.
Independent legacy CUDA graph outputs must reproduce every prediction and the confusion matrix; their log-probability errors are recorded separately.
See initialization.json raw_graph_replays, initialization_comparison.json and cuda_readout_preflight.json.
