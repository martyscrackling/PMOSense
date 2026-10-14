# Model comparison - group K-fold cross-validation

5 folds over 519 images in 409 scan families (Healthy 59 images / **25 families**, PCOS 460 images / 384 families). Every image is tested once; AUC is pooled over all out-of-fold predictions; CI = 95% bootstrap over families.

> **Shortcut baseline.** A gradient-boosting model on 23 trivial image statistics (brightness, contrast, black margins, tissue extent, file size; no anatomy), evaluated on the SAME folds, reaches pooled AUC **0.945** [0.902, 0.982]; the best CNN reaches 0.989 (difference +0.044). The baseline is high, so CNN scores do not show that pathology was learned.

| Model | Params (M) | Pooled OOF AUC [95% CI] | Per-fold AUC mean (min) | Sens. | Spec. | Bal. acc. | Healthy families correct | PCOS families correct |
|---|---|---|---|---|---|---|---|---|
| efficientnet_b0 **(selected)** | 4.0 | 0.989 [0.972, 1.000] | 0.993 (0.966) | 0.961 | 0.847 | 0.904 | 25/25 | 367/384 |
| densenet121 | 7.0 | 0.929 [0.818, 0.996] | 0.980 (0.930) | 0.961 | 0.780 | 0.870 | 24/25 | 381/384 |
| resnet50 | 23.5 | 0.878 [0.728, 0.994] | 0.936 (0.835) | 0.957 | 0.797 | 0.877 | 23/25 | 370/384 |
| *trivial-feature baseline* | - | 0.945 [0.902, 0.982] | - | - | - | - | - | - |

Sens./Spec./Bal. acc. use, for each fold, the threshold that maximised F1 on that fold's validation split. 'Families correct' averages P(PCOS) within a scan family and thresholds at 0.5.

## Selection

Rule: highest pooled out-of-fold AUC; models within 0.005 of the best are tied and the smallest wins.
Tied with the best: efficientnet_b0.  **Selected: efficientnet_b0**

- paired group-bootstrap AUC difference, efficientnet_b0 minus resnet50: +0.1077 [-0.0039, +0.2611] - CI includes 0 (no demonstrated difference)
- paired group-bootstrap AUC difference, efficientnet_b0 minus densenet121: +0.0566 [-0.0084, +0.1681] - CI includes 0 (no demonstrated difference)
- paired group-bootstrap AUC difference, efficientnet_b0 minus trivial-feature baseline: +0.0441 [+0.0124, +0.0854] - CI excludes 0: the CNN beats the trivial-feature baseline

Out-of-fold selection among three candidates is slightly optimistic for the winner; no separate outer test set remains. A different data source is still needed to show the models generalise.