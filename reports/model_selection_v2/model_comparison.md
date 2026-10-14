# Model comparison

> **WARNING - read before using these numbers.** Trivial-feature baseline (brightness, black margins, tissue extent, file size; no anatomy) reaches test AUC 0.926; the best CNN reaches 1.000 (difference +0.074). The baseline is already very high, so these CNN scores do NOT show that the models learned ovarian pathology: the classes are separable by image characteristics alone (different sources/framing). Do not interpret differences between models as clinical performance.

Seeds per model: densenet121=3, efficientnet_b0=3, resnet50=3  
Threshold: val_f1 (chosen on validation, applied unchanged to test).  
Per-seed columns are mean ± std over seeds (threshold-dependent metrics use each seed's own validation-chosen threshold). 'Seed-avg' = AUC of the probabilities averaged over seeds, with a 95% group-bootstrap CI.

| Model | Params (M) | Val AUC | Test AUC per seed | Test AUC seed-avg [95% CI] | Test acc | Test precision | Test recall | Test specificity | Test F1 | Best epoch | Train min |
|---|---|---|---|---|---|---|---|---|---|---|---|
| densenet121 | 7.0 | 1.000 ± 0.000 | 0.944 ± 0.085 | 0.987 [0.978, 0.997] | 0.841 ± 0.084 | 0.833 ± 0.078 | 0.984 ± 0.028 | 0.467 ± 0.281 | 0.901 ± 0.050 | 10 | 2.1 |
| efficientnet_b0 **(selected)** | 4.0 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 [1.000, 1.000] | 0.998 ± 0.004 | 0.997 ± 0.005 | 1.000 ± 0.000 | 0.992 ± 0.014 | 0.998 ± 0.003 | 17 | 1.9 |
| resnet50 | 23.5 | 1.000 ± 0.000 | 0.988 ± 0.014 | 0.995 [0.980, 1.000] | 0.825 ± 0.069 | 0.810 ± 0.063 | 0.997 ± 0.005 | 0.375 ± 0.250 | 0.893 ± 0.038 | 10 | 1.7 |

## Selection

3 models are within 0.005 validation AUC of the best ('densenet121', 1.0000); 'efficientnet_b0' has the fewest parameters among them.

Paired group-bootstrap difference in test AUC (selected minus other):

- vs densenet121: +0.0122  [+0.0030, +0.0220]  - CI excludes 0
- vs resnet50: +0.0061  [+0.0000, +0.0179]  - CI includes 0 (no demonstrated difference)
