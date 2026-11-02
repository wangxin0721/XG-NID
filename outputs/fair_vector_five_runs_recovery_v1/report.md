# Fair fine-tuning and vector fusion

| seed | E0-FT Macro-F1 | N0 Macro-F1 | N0-E0-FT pp |
|---|---:|---:|---:|
| 42 | 96.835190% | 96.996234% | +0.161044 |
| 43 | 96.834785% | 97.318728% | +0.483944 |
| 44 | 96.093219% | 97.266145% | +1.172926 |

| N0-s42 | V1 | V2 | V2-N0 pp | V2-V1 pp |
|---:|---:|---:|---:|---:|
| 96.996234% | 97.039453% | 97.093667% | +0.097434 | +0.054215 |

Budget thresholds: V2-N0 >= 0.10 pp and V2-V1 >= 0.05 pp. Class tradeoffs still require review.
No statistical significance claim, no automatic repetitions, no test evaluation.
