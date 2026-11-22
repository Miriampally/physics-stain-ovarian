# Physics-guided stain-aware classification of ovarian carcinoma subtypes (H&E patches)

Code and result files for the paper *"Physics-guided stain-aware classification of ovarian carcinoma subtypes in H&E whole-slide patches"* (Raghavendra M. V., Kumar N. A.).

The study compares a Beer–Lambert stain and illumination layer in front of an EfficientNet-B0 with colour jitter, stain augmentation and classical Reinhard / Macenko normalisation, on 512 UBC-OCEAN ovarian slides (5 subtypes), with slide-level 5-fold cross-validation, paired bootstrap intervals and test-time shifts (including shifts outside the augmentation range). The main findings and their limits are stated in the paper; in short: no clean-accuracy gain, invariance to brightness / colour-cast / stain-scaling shifts when the radiomic branch is removed (replicated with two seeds), and worse robustness than Reinhard normalisation under additive noise.

## Layout

| Folder | Content |
|---|---|
| `code/` | Training and evaluation code. All patches used in the experiments (Reinhard/Macenko variants, wb colour-cast test, `NO_RADIOMICS` switch) are already applied. |
| `notebooks/` | The Kaggle / Colab notebooks used to run the experiments. Their in-notebook patch cells are redundant with the patched files in `code/`. |
| `results/` | Result files of every run (tables, per-fold results, per-slide out-of-fold predictions). |
| `analysis/` | Bootstrap tables and the script that produces the pooled two-seed table and the seed-42 bootstrap. |

### Result folders

| Folder | Seed | Variants | Radiomic branch |
|---|---|---|---|
| `seed42_main_A_G_H_E`, `seed42_main_B_C_D_F` | 42 | A_baseline, G_colorjitter, H_stainaug, E_full; B_stainOD, C_stainOD+consist, D_attn_only, F_consist_only | on |
| `seed43_A_G_H_C` | 43 | A_baseline, G_colorjitter, H_stainaug, C_stainOD+consist | on |
| `seed43_reinhard_macenko` | 43 | N_reinhard, N_macenko | on |
| `seed43_no_radiomics` | 43 | A_baseline, H_stainaug, C_stainOD+consist, N_reinhard | off |
| `seed42_no_radiomics` | 42 | A_baseline, H_stainaug, C_stainOD+consist, N_reinhard | off |

Each folder has `tables.md` (the tables as in the paper), `results.json` (per fold and condition), `summary_*_level.csv`, `slide_predictions.csv` (out-of-fold probabilities per slide) and `run_meta.json`.
The radiomics-free experiment was designed after seeing the earlier results and then repeated with a second seed; it is not a pre-registered test.

## Data (not included)

The images are from the UBC-OCEAN ovarian cancer subtype classification competition on Kaggle: https://www.kaggle.com/competitions/UBC-OCEAN . Please download them yourself and respect the competition's terms of use. The code expects patches in the layout

```
<ubc_dir>/patches/patch_manifest.csv        # columns include patch_path, label, source_image_id
<ubc_dir>/patches/...                       # patch images
<ubc_dir>/patches_filtered/...              # optional tissue filter
```

How the patches were cut (patch size and magnification) is described in the paper.

## Running

Install the requirements (`pip install -r requirements.txt`), then from `code/`:

```bash
python test_histo_utils.py        # unit tests, should end with ALL TESTS PASSED

# all variants with the radiomic branch (example: seed 42)
python histo_model.py --ubc-dir <ubc_dir> --max-patches-per-slide 16 --folds 5 --epochs 10 --seed 42 \
    --variants A_baseline G_colorjitter H_stainaug C_stainOD+consist N_reinhard N_macenko \
    --conditions clean clean_sub bright0.8 bright0.65 bright0.5 wb wb0.2 hscale1.3 hscale2.0 escale0.7 escale0.4 jpeg30 noise \
    --workers 2 --dl-workers 2 --no-xai --out results_seed42

# radiomic branch disabled (the experiment of Sections 4.8-4.9)
NO_RADIOMICS=1 python histo_model.py --ubc-dir <ubc_dir> --max-patches-per-slide 16 --folds 5 --epochs 10 --seed 42 \
    --variants A_baseline H_stainaug C_stainOD+consist N_reinhard \
    --conditions clean clean_sub bright0.8 bright0.65 bright0.5 wb wb0.2 hscale1.3 hscale2.0 escale0.7 escale0.4 jpeg30 noise \
    --workers 2 --dl-workers 2 --no-xai --resume --out results_seed42_norad
```

`--resume` continues an interrupted run. A GPU is needed for training (the runs reported in the paper used Kaggle T4 GPUs; about 2 GPU hours per radiomics-free run).

To reproduce the pooled two-seed table and the seed-42 bootstrap from the stored result files:

```bash
cd analysis && python pooled_folds_and_seed42_bootstrap.py
```

## Limitations

One dataset, 512 slides, about 100 test slides per fold, two seeds, synthetic test-time shifts, no external-site validation. See Section 6 of the paper.

## Citation and licence

Archived copy with DOI: https://doi.org/10.5281/zenodo.24435575
Citation: to be added after publication.
Licence: MIT (see LICENSE).
