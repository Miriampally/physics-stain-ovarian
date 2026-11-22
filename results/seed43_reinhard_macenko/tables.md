**Slide-level results, clean test folds (mean ± std over folds)**

| Variant | Balanced acc | Macro-F1 | Accuracy | AUC (OvR) | ECE |
|---|---|---|---|---|---|
| N_reinhard | 0.742 ± 0.054 | 0.726 ± 0.057 | 0.738 ± 0.045 | 0.934 ± 0.016 | 0.135 ± 0.036 |
| N_macenko | 0.751 ± 0.052 | 0.741 ± 0.042 | 0.756 ± 0.029 | 0.922 ± 0.010 | 0.165 ± 0.052 |

**Slide-level balanced accuracy under perturbation (reference = clean_sub)**

| Variant | clean_sub | bright0.8 | bright0.65 | bright0.5 | wb | wb0.2 | hscale1.3 | hscale2.0 | escale0.7 | escale0.4 | jpeg30 | noise |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N_reinhard | 0.740 ± 0.050 | 0.711 ± 0.038 | 0.699 ± 0.047 | 0.627 ± 0.048 | 0.717 ± 0.029 | 0.705 ± 0.028 | 0.730 ± 0.052 | 0.621 ± 0.021 | 0.738 ± 0.023 | 0.636 ± 0.042 | 0.436 ± 0.089 | 0.668 ± 0.068 |
| N_macenko | 0.711 ± 0.024 | 0.524 ± 0.050 | 0.423 ± 0.042 | 0.355 ± 0.046 | 0.685 ± 0.049 | 0.591 ± 0.082 | 0.712 ± 0.050 | 0.688 ± 0.061 | 0.725 ± 0.043 | 0.709 ± 0.058 | 0.351 ± 0.046 | 0.519 ± 0.066 |

**Per-class slide-level recall (confusion summed over folds)**

| Variant | HGSC | EC | CC | LGSC | MC |
|---|---|---|---|---|---|
| N_reinhard | 0.74 | 0.66 | 0.83 | 0.67 | 0.80 |
| N_macenko | 0.78 | 0.66 | 0.82 | 0.69 | 0.80 |