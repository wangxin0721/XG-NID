| Seed | Model | Epoch0 F1 | Best0-150 F1 | Best0-200 F1 | Selected epoch | Trained best F1 | Trained epoch | Last F1 | Updates |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | Joint-Mean | 97.320705% | 97.393098% | 97.395206% | 191 | 97.395206% | 191 | 97.383156% | 42200 |
| 42 | Joint-Flow | 97.320705% | 97.493837% | 97.493837% | 95 | 97.493837% | 95 | 97.443073% | 42200 |
| 43 | Joint-Mean | 97.320705% | 97.434015% | 97.465873% | 179 | 97.465873% | 179 | 97.343977% | 42200 |
| 43 | Joint-Flow | 97.320705% | 97.489287% | 97.500104% | 179 | 97.500104% | 179 | 97.451835% | 42200 |
| 44 | Joint-Mean | 97.320705% | 97.438441% | 97.446245% | 185 | 97.446245% | 185 | 97.380757% | 42200 |
| 44 | Joint-Flow | 97.320705% | 97.478505% | 97.478505% | 143 | 97.478505% | 143 | 97.413297% | 42200 |

| Seed | Joint-Flow minus Joint-Mean pp | Joint-Flow minus historical JCL-Multi pp |
|---|---:|---:|
| 42 | +0.098631 | -0.030772 |
| 43 | +0.034231 | -0.024505 |
| 44 | +0.032260 | -0.046104 |

Three-seed paired mean +0.055041 pp; sample SD 0.037764 pp (n=3, ddof=1).
New43/44 paired mean +0.033245 pp; sample SD 0.001394 pp (n=2, ddof=1).
All three positive: True.

Exploratory exception registered before new training; seed42 remains below0.10pp and is not retroactively passed.
All seeds share the same mature D32 checkpoint. Variability covers query/key initialization and joint-stage order/RNG; this is not a full-pretraining multiseed experiment.
Each arm selects independently over epochs0..200; ties keep the earliest epoch. Epoch95 is not forced. Constant learning rates, no scheduler or early stopping.
The experiment tests the query source under joint training. Frozen43/44 are absent; a multiseed enhancement from unfreezing cannot yet be claimed.
Probability vectors are normalized classifier outputs, not additional calibration. Error transitions give behavioral evidence; attention alone does not establish complementarity.

| Model | Three-seed mean Macro-F1 percent | Sample SD pp (n=3, ddof=1) |
|---|---:|---:|
| Joint-Mean | 97.435775 | 0.036479 |
| Joint-Flow | 97.490815 | 0.011112 |

| Seed | True class | Mean P/R/F1 | Flow P/R/F1 | Repaired | Regressed |
|---|---|---|---|---:|---:|
| 42 | WebBased | 89.821429% / 92.463235% / 91.123188% | 89.612676% / 93.566176% / 91.546763% | 7 | 1 |
| 42 | Spoofing | 96.414112% / 95.859689% / 96.136101% | 96.637681% / 95.859689% / 96.247113% | 1 | 1 |
| 42 | Recon | 97.668280% / 97.453907% / 97.560976% | 97.968198% / 97.366111% / 97.666226% | 3 | 5 |
| 43 | WebBased | 89.734513% / 93.198529% / 91.433724% | 89.788732% / 93.750000% / 91.726619% | 4 | 1 |
| 43 | Spoofing | 96.473988% / 95.974698% / 96.223696% | 96.691817% / 95.802185% / 96.244945% | 0 | 3 |
| 43 | Recon | 97.839506% / 97.410009% / 97.624285% | 97.880795% / 97.322212% / 97.600704% | 2 | 4 |
| 44 | WebBased | 90.000000% / 92.647059% / 91.304348% | 89.893617% / 93.198529% / 91.516245% | 5 | 2 |
| 44 | Spoofing | 96.579710% / 95.802185% / 96.189376% | 96.689895% / 95.744681% / 96.214967% | 0 | 1 |
| 44 | Recon | 97.756269% / 97.541703% / 97.648868% | 97.840458% / 97.453907% / 97.646800% | 2 | 4 |

| Seed | All repairs | All regressions | WebBased/Recon repairs | WebBased/Recon regressions |
|---|---:|---:|---:|---:|
| 42 | 17 | 7 | 8 | 4 |
| 43 | 10 | 9 | 5 | 4 |
| 44 | 9 | 7 | 6 | 5 |

Error-type recurrence (true class -> wrong prediction):
| Error type | Seed42 repaired/regressed | Seed43 repaired/regressed | Seed44 repaired/regressed |
|---|---:|---:|---:|
| Benign->Recon | 2/0 | 0/1 | 0/0 |
| Benign->Spoofing | 4/0 | 4/0 | 2/0 |
| Recon->Benign | 0/1 | 0/1 | 0/1 |
| Recon->WebBased | 3/4 | 2/3 | 2/3 |
| Spoofing->Benign | 1/0 | 0/3 | 0/1 |
| Spoofing->WebBased | 0/1 | 0/0 | 0/0 |
| WebBased->Benign | 2/1 | 1/0 | 1/0 |
| WebBased->Recon | 5/0 | 3/1 | 4/2 |
