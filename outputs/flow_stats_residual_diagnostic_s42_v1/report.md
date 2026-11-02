| replay | validation Macro-F1 | errors | mean residual/graph norm | p95 ratio |
|---|---:|---:|---:|---:|
| last_full | 94.303194% | 462 | 5.54128 | 11.3605 |
| last_stats_off | 96.599888% | 327 | 0 | 0 |
| last_with_start_BN | 94.319804% | 463 | 5.6614 | 11.5663 |
| start_with_last_stats | 93.938964% | 531 | 5.66583 | 11.6318 |

Source selected epoch 0; diagnosing trained epoch 12.
Frozen V2: 97.093667%; lower-LR J2: 97.120712%.
last_stats_off: keep last parameters/BN buffers, disable residual only.
last_with_start_BN: keep last parameters/residual; restore only encoder BN buffers from the V2 start.
start_with_last_stats: keep V2 public parameters/BN; attach the last trained statistics MLP.
These are fixed inference interventions, not separately trained ablations or candidate selection.
A norm ratio alone cannot prove harmful magnitude. Inspect output/class changes together.
No training, no backward, no test evaluation, no automatic followup. Source files unchanged.
