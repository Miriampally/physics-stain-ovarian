**Slide-level Balanced accuracy, out-of-fold, 95% bootstrap CI over slides (reference: A_baseline)**

| Variant | Balanced accuracy [95% CI] | Δ vs A_baseline [95% CI] | bootstrap p |
|---|---|---|---|
| A_baseline | 0.745 [0.702, 0.787] | – | – |
| G_colorjitter | 0.755 [0.711, 0.798] | +0.010 [-0.025, +0.048] | 0.583 |
| H_stainaug | 0.760 [0.718, 0.803] | +0.015 [-0.014, +0.045] | 0.287 |
| C_stainOD+consist | 0.757 [0.713, 0.803] | +0.013 [-0.023, +0.049] | 0.478 |
| N_reinhard | 0.741 [0.698, 0.788] | -0.003 [-0.044, +0.037] | 0.915 |
| N_macenko | 0.752 [0.708, 0.797] | +0.007 [-0.029, +0.044] | 0.683 |

p-values are not corrected for multiple comparisons.

**Slide-level Balanced accuracy, out-of-fold, 95% bootstrap CI over slides (reference: C_stainOD+consist)**

| Variant | Balanced accuracy [95% CI] | Δ vs C_stainOD+consist [95% CI] | bootstrap p |
|---|---|---|---|
| A_baseline | 0.745 [0.702, 0.787] | -0.013 [-0.049, +0.023] | 0.478 |
| G_colorjitter | 0.755 [0.711, 0.798] | -0.002 [-0.035, +0.031] | 0.874 |
| H_stainaug | 0.760 [0.718, 0.803] | +0.002 [-0.029, +0.037] | 0.905 |
| C_stainOD+consist | 0.757 [0.713, 0.803] | – | – |
| N_reinhard | 0.741 [0.698, 0.788] | -0.016 [-0.050, +0.021] | 0.379 |
| N_macenko | 0.752 [0.708, 0.797] | -0.005 [-0.036, +0.024] | 0.698 |

p-values are not corrected for multiple comparisons.