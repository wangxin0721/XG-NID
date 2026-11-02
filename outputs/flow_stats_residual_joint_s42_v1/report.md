| model | validation Macro-F1 | best epoch | stop epoch | updates | joint seconds |
|---|---:|---:|---:|---:|---:|
| N0 | 96.996234% | 5 | 17 | 3587 | None |
| frozen_V2 | 97.093667% | 13 | 25 | 5275 | None |
| lower_lr_J0 | 97.022899% | 9 | 21 | 4431 | 4165.293931885972 |
| lower_lr_J2 | 97.120712% | 9 | 21 | 4431 | 3934.2544536161004 |
| J2_flow_stats | 97.093667% | 0 | 12 | 2532 | 2313.2977284520166 |

candidate_minus_N0_pp: +0.097434

candidate_minus_frozen_V2_pp: +0.000000

candidate_minus_lower_lr_J2_pp: -0.027045

candidate_minus_stronger_reference_pp: -0.027045

Budget screen (>= 0.10 pp above the stronger frozen-V2/lower-LR-J2 reference): False. Class review is still required.

Same-checkpoint path-on minus path-off: +0.000000 pp.
Path-off is a post-selection dependency diagnostic, not a separately trained capacity ablation.

Against frozen_V2: repaired=0, regressed=0, net_correct=0.

Against lower_lr_J2: repaired=15, regressed=15, net_correct=0.

Against stats_off_same_checkpoint: repaired=0, regressed=0, net_correct=0.

All eight classes (P / R / F1 in percent):

| class | lower-LR J2 P/R/F1 | residual P/R/F1 | F1 difference pp |
|---|---:|---:|---:|
| Benign | 96.6306 / 97.2373 / 96.9330 | 96.5919 / 97.2792 / 96.9343 | +0.001279 |
| WebBased | 89.8496 / 87.8676 / 88.8476 | 88.8278 / 89.1544 / 88.9908 | +0.143242 |
| Spoofing | 95.9242 / 96.0897 / 96.0069 | 96.2536 / 96.0322 / 96.1428 | +0.135880 |
| Recon | 96.7249 / 97.2344 / 96.9790 | 96.8846 / 96.9271 / 96.9059 | -0.073125 |
| Mirai | 100.0000 / 100.0000 / 100.0000 | 100.0000 / 100.0000 / 100.0000 | +0.000000 |
| Dos | 99.9318 / 99.3898 / 99.6601 | 99.9318 / 99.3898 / 99.6601 | +0.000000 |
| DDos | 100.0000 / 99.6650 / 99.8322 | 100.0000 / 99.6650 / 99.8322 | +0.000000 |
| BruteForce | 99.1342 / 98.2833 / 98.7069 | 98.2833 / 98.2833 / 98.2833 | -0.423635 |

Candidate selected epoch0: True
Candidate training evidence: {'best_trained_epoch': 12, 'best_trained_macro_f1': 0.9430319351896747, 'best_trained_minus_start_pp': -2.79047373297453, 'last_trained_macro_f1': 0.9430319351896747, 'last_trained_minus_start_pp': -2.79047373297453}
Stage A is reused, its historical missing duration remains unavailable: {'directory': '/home/nzxn/下载/WX/load/XG-NID/outputs/fair_vector_five_runs_recovery_v1/V2', 'updates': 5275, 'elapsed_seconds': None, 'timing_available': False, 'timing_note': 'Historical V2 did not record wall time; never infer duration from file timestamps'}
New preflight seconds: 77.44446601800155
New post-training diagnostic seconds: 62.52406340604648
Candidate changes the statistics path and adds its parameters/learning-rate group; overall N0 gains are not an isolated gate effect. Path-off is a post-selection dependency diagnostic at the same checkpoint, not a separately trained ablation.
No test evaluation, no automatic next run, no change to old V2 screening.
