**Slide-level Balanced accuracy, out-of-fold, 95% bootstrap CI over slides (reference: A_baseline)**

| Variant | Balanced accuracy [95% CI] | Δ vs A_baseline [95% CI] | bootstrap p |
|---|---|---|---|
| A_baseline | 0.753 [0.707, 0.795] | – | – |
| B_stainOD | 0.754 [0.710, 0.799] | +0.001 [-0.041, +0.040] | 0.948 |
| C_stainOD+consist | 0.771 [0.728, 0.813] | +0.018 [-0.024, +0.059] | 0.389 |
| D_attn_only | 0.764 [0.719, 0.807] | +0.011 [-0.019, +0.039] | 0.463 |
| F_consist_only | 0.751 [0.707, 0.795] | -0.002 [-0.037, +0.034] | 0.952 |
| G_colorjitter | 0.773 [0.731, 0.812] | +0.020 [-0.012, +0.055] | 0.229 |
| H_stainaug | 0.781 [0.740, 0.820] | +0.028 [-0.006, +0.062] | 0.109 |
| E_full | 0.737 [0.691, 0.782] | -0.016 [-0.056, +0.022] | 0.454 |

p-values are not corrected for multiple comparisons.

**Slide-level Balanced accuracy, out-of-fold, 95% bootstrap CI over slides (reference: H_stainaug)**

| Variant | Balanced accuracy [95% CI] | Δ vs H_stainaug [95% CI] | bootstrap p |
|---|---|---|---|
| A_baseline | 0.753 [0.707, 0.795] | -0.028 [-0.062, +0.006] | 0.109 |
| B_stainOD | 0.754 [0.710, 0.799] | -0.027 [-0.068, +0.016] | 0.214 |
| C_stainOD+consist | 0.771 [0.728, 0.813] | -0.010 [-0.051, +0.032] | 0.645 |
| D_attn_only | 0.764 [0.719, 0.807] | -0.017 [-0.054, +0.020] | 0.387 |
| F_consist_only | 0.751 [0.707, 0.795] | -0.030 [-0.061, +0.001] | 0.059 |
| G_colorjitter | 0.773 [0.731, 0.812] | -0.008 [-0.037, +0.020] | 0.602 |
| H_stainaug | 0.781 [0.740, 0.820] | – | – |
| E_full | 0.737 [0.691, 0.782] | -0.044 [-0.083, -0.005] | 0.029 |

p-values are not corrected for multiple comparisons.

**Slide-level Balanced accuracy, out-of-fold, 95% bootstrap CI over slides (reference: C_stainOD+consist)**

| Variant | Balanced accuracy [95% CI] | Δ vs C_stainOD+consist [95% CI] | bootstrap p |
|---|---|---|---|
| A_baseline | 0.753 [0.707, 0.795] | -0.018 [-0.059, +0.024] | 0.389 |
| B_stainOD | 0.754 [0.710, 0.799] | -0.017 [-0.052, +0.016] | 0.318 |
| C_stainOD+consist | 0.771 [0.728, 0.813] | – | – |
| D_attn_only | 0.764 [0.719, 0.807] | -0.007 [-0.047, +0.032] | 0.781 |
| F_consist_only | 0.751 [0.707, 0.795] | -0.020 [-0.060, +0.021] | 0.346 |
| G_colorjitter | 0.773 [0.731, 0.812] | +0.002 [-0.036, +0.039] | 0.907 |
| H_stainaug | 0.781 [0.740, 0.820] | +0.010 [-0.032, +0.051] | 0.645 |
| E_full | 0.737 [0.691, 0.782] | -0.034 [-0.074, +0.003] | 0.077 |

p-values are not corrected for multiple comparisons.