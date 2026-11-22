**Slide-level results, clean test folds (mean ± std over folds)**

| Variant | Balanced acc | Macro-F1 | Accuracy | AUC (OvR) | ECE |
|---|---|---|---|---|---|
| B_stainOD | 0.755 ± 0.056 | 0.741 ± 0.053 | 0.762 ± 0.038 | 0.921 ± 0.019 | 0.179 ± 0.059 |
| C_stainOD+consist | 0.771 ± 0.030 | 0.755 ± 0.029 | 0.772 ± 0.034 | 0.926 ± 0.013 | 0.183 ± 0.059 |
| D_attn_only | 0.764 ± 0.044 | 0.768 ± 0.044 | 0.772 ± 0.046 | 0.925 ± 0.022 | 0.132 ± 0.045 |
| F_consist_only | 0.752 ± 0.059 | 0.726 ± 0.064 | 0.740 ± 0.072 | 0.921 ± 0.023 | 0.156 ± 0.049 |

**Slide-level balanced accuracy under perturbation (reference = clean_sub)**

| Variant | clean_sub | bright0.8 | wb | hscale1.3 | escale0.7 | blur | jpeg30 | noise |
|---|---|---|---|---|---|---|---|---|
| B_stainOD | 0.728 ± 0.055 | 0.710 ± 0.038 | 0.709 ± 0.054 | 0.713 ± 0.042 | 0.722 ± 0.050 | 0.243 ± 0.045 | 0.451 ± 0.082 | 0.572 ± 0.114 |
| C_stainOD+consist | 0.755 ± 0.042 | 0.735 ± 0.049 | 0.740 ± 0.037 | 0.741 ± 0.042 | 0.750 ± 0.036 | 0.277 ± 0.067 | 0.435 ± 0.057 | 0.592 ± 0.081 |
| D_attn_only | 0.723 ± 0.048 | 0.423 ± 0.058 | 0.486 ± 0.082 | 0.686 ± 0.060 | 0.705 ± 0.052 | 0.206 ± 0.021 | 0.466 ± 0.069 | 0.654 ± 0.046 |
| F_consist_only | 0.726 ± 0.060 | 0.676 ± 0.064 | 0.662 ± 0.049 | 0.692 ± 0.055 | 0.701 ± 0.057 | 0.244 ± 0.047 | 0.551 ± 0.045 | 0.663 ± 0.044 |

**Per-class slide-level recall (confusion summed over folds)**

| Variant | HGSC | EC | CC | LGSC | MC |
|---|---|---|---|---|---|
| B_stainOD | 0.79 | 0.68 | 0.83 | 0.67 | 0.80 |
| C_stainOD+consist | 0.80 | 0.66 | 0.85 | 0.71 | 0.83 |
| D_attn_only | 0.80 | 0.67 | 0.85 | 0.67 | 0.83 |
| F_consist_only | 0.75 | 0.60 | 0.89 | 0.67 | 0.85 |