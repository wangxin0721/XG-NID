# E0/E2 paired baseline repeats

| Seed | E0 Macro-F1 | E2 Macro-F1 | E2−E0 (pp) |
|---|---:|---:|---:|
| 42 | 96.672874% | 96.599400% | -0.073474 |
| 43 | 96.265272% | 97.076612% | +0.811341 |
| 44 | 95.456506% | 96.970718% | +1.514212 |

Paired mean difference: +0.750693 pp; sample standard deviation: 0.795579 pp.
E0 mean: 96.131551%, sample std: 0.619112 pp.
E2 mean: 96.882243%, sample std: 0.250607 pp.
Three seeds provide initial evidence of direction and variability; a small mean lead is not stable/significant superiority.
Full class P/R/F1, confusions, epochs, optimizer updates, exposure and model means/std: analysis.json.
Seed42 was already known during previous exploration. New runs use validation only. Shared-GPU timings are descriptive.
No frozen-fusion experiments are automatically scheduled.

| Seed | Model | Best/stop epoch | Updates | Training samples seen |
|---|---|---:|---:|---:|
| 42 | E0 | 56/68 | 14348 | 7318228 |
| 42 | E2 | 53/65 | 13715 | 6995365 |
| 43 | E0 | 40/52 | 10972 | 5596292 |
| 43 | E2 | 71/83 | 17513 | 8932543 |
| 44 | E0 | 22/34 | 7174 | 3659114 |
| 44 | E2 | 76/88 | 18568 | 9470648 |

| Seed | Model | Class | Precision | Recall | F1 | Support |
|---|---|---|---:|---:|---:|---:|
| 42 | E0 | WebBased | 86.9725% | 87.1324% | 87.0523% | 544 |
| 42 | E0 | Recon | 97.0432% | 96.5320% | 96.7870% | 2278 |
| 42 | E0 | BruteForce | 97.4359% | 97.8541% | 97.6445% | 233 |
| 42 | E2 | WebBased | 84.5754% | 89.7059% | 87.0651% | 544 |
| 42 | E2 | Recon | 97.0588% | 95.6102% | 96.3291% | 2278 |
| 42 | E2 | BruteForce | 97.4359% | 97.8541% | 97.6445% | 233 |
| 43 | E0 | WebBased | 85.5805% | 84.0074% | 84.7866% | 544 |
| 43 | E0 | Recon | 95.6654% | 96.8832% | 96.2704% | 2278 |
| 43 | E0 | BruteForce | 97.8355% | 96.9957% | 97.4138% | 233 |
| 43 | E2 | WebBased | 86.5854% | 91.3603% | 88.9088% | 544 |
| 43 | E2 | Recon | 97.7434% | 96.9710% | 97.3557% | 2278 |
| 43 | E2 | BruteForce | 98.2833% | 98.2833% | 98.2833% | 233 |
| 44 | E0 | WebBased | 85.3119% | 77.9412% | 81.4601% | 544 |
| 44 | E0 | Recon | 95.1655% | 95.9175% | 95.5400% | 2278 |
| 44 | E0 | BruteForce | 97.8070% | 95.7082% | 96.7462% | 233 |
| 44 | E2 | WebBased | 86.9792% | 92.0956% | 89.4643% | 544 |
| 44 | E2 | Recon | 97.6476% | 96.5759% | 97.1088% | 2278 |
| 44 | E2 | BruteForce | 97.0085% | 97.4249% | 97.2163% | 233 |
