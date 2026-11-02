| model | validation Macro-F1 | best epoch | stop epoch | updates | joint seconds |
|---|---:|---:|---:|---:|---:|
| N0 | 96.996234% | 5 | 17 | 3587 | None |
| frozen_V2 | 97.093667% | 13 | 25 | 5275 | None |
| lower_lr_J0 | 97.022899% | 9 | 21 | 4431 | 4165.293931885972 |
| lower_lr_J2 | 97.120712% | 9 | 21 | 4431 | 3934.2544536161004 |
| J2_flow_stats_ln | 97.093667% | 0 | 12 | 2532 | 2240.4967613349436 |

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
Candidate training evidence: {'best_trained_epoch': 9, 'best_trained_macro_f1': 0.9709042354331445, 'best_trained_minus_start_pp': -0.003243708627542752, 'last_trained_macro_f1': 0.9701200585589266, 'last_trained_minus_start_pp': -0.08166139604933731}
Stage A is reused, its historical missing duration remains unavailable: {'directory': '/home/nzxn/下载/WX/load/XG-NID/outputs/fair_vector_five_runs_recovery_v1/V2', 'updates': 5275, 'elapsed_seconds': None, 'timing_available': False, 'timing_note': 'Historical V2 did not record wall time; never infer duration from file timestamps'}
New preflight seconds: 82.08153966907412
New post-training diagnostic seconds: 61.82207906502299
Candidate changes the statistics path and adds its parameters/learning-rate group; overall N0 gains are not an isolated gate effect. Path-off is a post-selection dependency diagnostic at the same checkpoint, not a separately trained ablation.
No test evaluation, no automatic next run, no change to old V2 screening.
LayerNorm is the only model change from the unnormalized candidate; +128 affine parameters.
Stability probe: passed; discarded updates: 3.
Pre-registered probe bounds: {'updates': 3, 'ratio_median_ceiling': 0.1, 'ratio_p95_ceiling': 0.25, 'ratio_max_ceiling': 1.0, 'eval_nll_increase_ceiling': 0.1, 'prediction_change_fraction_ceiling': 0.05}.
Per-epoch validation norm/gate distributions: epoch_scale_diagnostics.jsonl.
Last trained ratio distribution: {'mean': 0.028436046093702316, 'std': 0.009161137975752354, 'min': 0.0049149817787110806, 'max': 0.08846840262413025, 'quantiles': {'0.0': 0.0049149817787110806, '0.25': 0.021059352904558182, '0.5': 0.029642514884471893, '0.75': 0.03630286082625389, '0.95': 0.037782229483127594, '0.99': 0.043225910514593124, '1.0': 0.08846840262413025}}
Probe bounds are stability screening rules, not validated optima or evidence of generalization.
