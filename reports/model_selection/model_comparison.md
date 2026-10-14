# Model comparison

> **WARNING - read before using these numbers.** Trivial-feature baseline (brightness, black margins, tissue extent, file size; no anatomy) reaches test AUC 0.755; the best CNN reaches 0.784 (difference +0.029). The CNNs barely beat a baseline that uses no anatomy; interpret with caution.

Seeds per model: densenet121=3, efficientnet_b0=3, resnet50=3  
Threshold: val_f1 (chosen on validation, applied unchanged to test).  
Per-seed columns are mean ± std over seeds (threshold-dependent metrics use each seed's own validation-chosen threshold). 'Seed-avg' = AUC of the probabilities averaged over seeds, with a 95% group-bootstrap CI.

| Model | Params (M) | Val AUC | Test AUC per seed | Test AUC seed-avg [95% CI] | Test acc | Test precision | Test recall | Test specificity | Test F1 | Best epoch | Train min |
|---|---|---|---|---|---|---|---|---|---|---|---|
| densenet121 | 7.0 | 0.999 ± 0.001 | 0.729 ± 0.062 | 0.713 [0.474, 1.000] | 0.658 ± 0.036 | 0.843 ± 0.057 | 0.661 ± 0.019 | 0.650 ± 0.156 | 0.740 ± 0.020 | 2 | 1.4 |
| efficientnet_b0 **(selected)** | 4.0 | 1.000 ± 0.001 | 0.789 ± 0.069 | 0.784 [0.608, 1.000] | 0.715 ± 0.037 | 0.942 ± 0.079 | 0.658 ± 0.018 | 0.875 ± 0.175 | 0.773 ± 0.020 | 1 | 1.5 |
| resnet50 | 23.5 | 0.993 ± 0.008 | 0.758 ± 0.044 | 0.745 [0.541, 1.000] | 0.711 ± 0.044 | 0.924 ± 0.069 | 0.664 ± 0.028 | 0.842 ± 0.142 | 0.772 ± 0.031 | 1 | 1.2 |

## Selection

2 models are within 0.005 validation AUC of the best ('efficientnet_b0', 0.9995); 'efficientnet_b0' has the fewest parameters among them.

Paired group-bootstrap difference in test AUC (selected minus other):

- vs densenet121: +0.0565  [+0.0000, +0.1282]  - CI includes 0 (no demonstrated difference)
- vs resnet50: +0.0316  [+0.0000, +0.0712]  - CI includes 0 (no demonstrated difference)
