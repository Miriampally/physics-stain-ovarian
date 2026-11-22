"""
novelty_utils.py - framework-agnostic experiment logic (numpy / sklearn only).

Everything that does not need PyTorch lives here so it can be tested on its own:
  * dataset loading with optional patient IDs (group-aware CV, no patient leakage)
  * k-fold cross-validation driver for an ablation study
  * illumination / noise perturbations for the robustness test
  * calibration (ECE) and summary tables (mean +- std over folds)
  * faithfulness metrics for explanations (deletion / insertion curves)
  * permutation importance for the radiomic branch

A "trainer" is any object with
    fit(cfg, X_tr, R_tr, y_tr, X_va, R_va, y_va, seed) -> model
    predict(model, X_uint8, R) -> p(malignant)
"""
import json
import warnings
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import (GroupShuffleSplit, StratifiedGroupKFold,
                                     StratifiedKFold, train_test_split)
from sklearn.preprocessing import StandardScaler

import ovarian_pipeline as op

CLASS_NAMES = op.CLASS_NAMES
ROBUST_CONDITIONS = ["clean", "gain0.5", "gain0.7", "gain1.3", "gradient",
                     "gamma0.7", "gamma1.5", "noise"]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_dataset(data_dir, size, groups_csv=None):
    """Folder layout data/benign, data/malignant. groups_csv: columns filename,patient_id."""
    imgs, ys, names = [], [], []
    for ci, cls in enumerate(CLASS_NAMES):
        files = sorted(p for p in (Path(data_dir) / cls).glob("*") if p.suffix.lower() in op.IMG_EXT)
        if not files:
            raise FileNotFoundError(f"No images in {Path(data_dir) / cls}")
        for p in files:
            im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if im is None:
                warnings.warn(f"cannot read {p}")
                continue
            imgs.append(cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA))
            ys.append(ci)
            names.append(p.name)
    groups = None
    if groups_csv:
        import pandas as pd
        df = pd.read_csv(groups_csv)
        m = dict(zip(df["filename"].astype(str), df["patient_id"].astype(str)))
        groups = np.array([m.get(n) for n in names], dtype=object)
        missing = int(sum(g is None for g in groups))
        if missing:
            raise ValueError(f"{missing} images have no patient_id in {groups_csv}")
    return np.stack(imgs), np.array(ys), names, groups


def radiomics_all(imgs):
    """Radiomics of the (fixed) physics-preprocessed image, one row per image."""
    out = [op.radiomic_features(op.physics_preprocess(im)) for im in imgs]
    return np.nan_to_num(np.stack(out)).astype(np.float32)


def make_folds(y, groups, k, seed):
    X = np.zeros(len(y))
    if groups is None:
        return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y))
    return list(StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(X, y, groups))


def inner_split(tr, y, groups, frac, seed):
    """Split the training fold into fit / validation parts (validation drives early stopping)."""
    if groups is None:
        return train_test_split(tr, test_size=frac, stratify=y[tr], random_state=seed)
    for s in range(seed, seed + 50):
        a, b = next(GroupShuffleSplit(1, test_size=frac, random_state=s).split(tr, groups=groups[tr]))
        if len(np.unique(y[tr][b])) == 2 and len(np.unique(y[tr][a])) == 2:
            return tr[a], tr[b]
    raise RuntimeError("could not build a two-class validation split with patient groups")


# --------------------------------------------------------------------------- #
# Perturbations (robustness test)
# --------------------------------------------------------------------------- #
def np_perturb(img, kind, rng):
    """
    gain<g>   : multiplicative illumination change (I -> g*I)   [physical: I0 scaling]
    gradient  : smooth multiplicative left-right illumination field (0.7 .. 1.3)
    gamma<g>  : sensor non-linearity [NOT covered by Beer-Lambert: expected to hurt]
    noise     : additive Gaussian noise, sigma=10 grey levels
    """
    x = img.astype(np.float32)
    if kind.startswith("gain"):
        x = x * float(kind[4:])
    elif kind == "gradient":
        field = 1.0 + 0.3 * np.linspace(-1, 1, x.shape[1], dtype=np.float32)[None, :]
        x = x * field
    elif kind.startswith("gamma"):
        x = 255.0 * (x / 255.0) ** float(kind[5:])
    elif kind == "noise":
        x = x + rng.normal(0, 10, x.shape)
    elif kind != "clean":
        raise ValueError(kind)
    return np.clip(x, 0, 255).astype(np.uint8)


def perturb_set(imgs, kind, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([np_perturb(im, kind, rng) for im in imgs])


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def ece_score(y, p, bins=10):
    """Expected calibration error of the predicted-class confidence."""
    y, p = np.asarray(y), np.asarray(p)
    conf = np.maximum(p, 1 - p)
    correct = ((p >= 0.5).astype(int) == y).astype(float)
    idx = np.minimum(((conf - 0.5) / 0.5 * bins).astype(int), bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def metrics(y, p):
    pred = (np.asarray(p) >= 0.5).astype(int)
    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    return {"accuracy": float(accuracy_score(y, pred)), "f1": float(f1_score(y, pred, zero_division=0)),
            "auc": float(auc), "ece": ece_score(y, p)}


# --------------------------------------------------------------------------- #
# Cross-validated ablation driver
# --------------------------------------------------------------------------- #
def run_cv(imgs, y, groups, rad_all, variants, trainer, k=5, seed=42, val_frac=0.15,
           conditions=ROBUST_CONDITIONS, out_dir=None, fold_hook=None, verbose=True):
    """
    For each fold: fit the radiomics scaler on the training fold only, train every variant,
    evaluate on the clean test fold and on perturbed copies of it (radiomics recomputed on the
    perturbed images, so every variant is tested end to end).
    Returns results[variant][condition] = [metrics per fold].
    """
    folds = make_folds(y, groups, k, seed)
    results = {v: {c: [] for c in conditions} for v in variants}
    for f, (tr, te) in enumerate(folds):
        tr_in, va = inner_split(tr, y, groups, val_frac, seed + f)
        sc = StandardScaler().fit(rad_all[tr])

        def R(a, sc=sc):
            return np.nan_to_num(sc.transform(a)).astype(np.float32)

        test_sets = {}
        for c in conditions:
            if c == "clean":
                ti, rr = imgs[te], rad_all[te]
            else:
                ti = perturb_set(imgs[te], c, seed)
                rr = radiomics_all(ti)
            test_sets[c] = (ti, R(rr))

        models = {}
        for v, cfg in variants.items():
            if verbose:
                print(f"[fold {f + 1}/{len(folds)}] training {v} ...", flush=True)
            model = trainer.fit(cfg, imgs[tr_in], R(rad_all[tr_in]), y[tr_in],
                                imgs[va], R(rad_all[va]), y[va], seed + f)
            models[v] = model
            for c, (ti, rr) in test_sets.items():
                results[v][c].append(metrics(y[te], trainer.predict(model, ti, rr)))
            if verbose:
                m = results[v]["clean"][-1]
                print(f"    clean: acc={m['accuracy']:.3f} f1={m['f1']:.3f} auc={m['auc']:.3f} ece={m['ece']:.3f}", flush=True)
        if fold_hook is not None:
            fold_hook(f, models, {"imgs_te": imgs[te], "rad_te": test_sets["clean"][1], "y_te": y[te]})
        if out_dir is not None:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            json.dump(results, open(Path(out_dir) / "results_partial.json", "w"), indent=1)
    return results


def summarize(results):
    import pandas as pd
    rows = []
    for v, d in results.items():
        for c, ms in d.items():
            row = {"variant": v, "condition": c}
            for key in ["accuracy", "f1", "auc", "ece"]:
                vals = np.array([m[key] for m in ms], float)
                row[key + "_mean"], row[key + "_std"] = float(np.nanmean(vals)), float(np.nanstd(vals))
            rows.append(row)
    return pd.DataFrame(rows)


def to_markdown(df):
    clean = df[df.condition == "clean"]
    L = ["**Clean test folds (mean ± std over folds)**", "",
         "| Variant | Accuracy | F1 | AUC | ECE |", "|---|---|---|---|---|"]
    for _, r in clean.iterrows():
        L.append(f"| {r.variant} | {r.accuracy_mean:.3f} ± {r.accuracy_std:.3f} | {r.f1_mean:.3f} ± {r.f1_std:.3f} "
                 f"| {r.auc_mean:.3f} ± {r.auc_std:.3f} | {r.ece_mean:.3f} ± {r.ece_std:.3f} |")
    conds = list(dict.fromkeys(df.condition))
    L += ["", "**AUC under perturbation (mean ± std over folds)**", "",
          "| Variant | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
    for v in dict.fromkeys(df.variant):
        d = df[df.variant == v].set_index("condition")
        L.append(f"| {v} | " + " | ".join(f"{d.loc[c].auc_mean:.3f} ± {d.loc[c].auc_std:.3f}" for c in conds) + " |")
    return "\n".join(L)


def plot_robustness(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conds = list(dict.fromkeys(df.condition))
    plt.figure(figsize=(9, 4.8))
    for v in dict.fromkeys(df.variant):
        d = df[df.variant == v].set_index("condition").loc[conds]
        plt.errorbar(range(len(conds)), d.auc_mean, yerr=d.auc_std, marker="o", capsize=3, label=v)
    plt.xticks(range(len(conds)), conds, rotation=30)
    plt.ylabel("AUC"); plt.title("Robustness to illumination / noise shifts")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()


# --------------------------------------------------------------------------- #
# Explanation faithfulness: deletion / insertion curves
# --------------------------------------------------------------------------- #
def faithfulness_masks(sal, steps=10):
    """bool (steps+1,H,W); masks[k] marks the top k/steps fraction of pixels by saliency."""
    flat = sal.ravel()
    order = np.argsort(-flat, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(flat.size)
    frac = ranks / flat.size
    return np.stack([(frac < k / steps).reshape(sal.shape) for k in range(steps + 1)])


def curve_auc(v):
    v = np.asarray(v, float)
    return float(np.sum((v[1:] + v[:-1]) / 2) / (len(v) - 1))


def faithfulness_curves(prob_fn, img01, sal, rad_row, steps=10):
    """
    prob_fn(batch (N,H,W) float in [0,1], rad (N,D)) -> p(malignant) (N,)
    deletion : remove the most salient pixels first (replace by a blurred baseline) -> prob should fall
    insertion: reveal the most salient pixels first on the baseline              -> prob should rise
    """
    base = cv2.GaussianBlur(img01.astype(np.float32), (0, 0), 15)
    masks = faithfulness_masks(sal, steps)
    dele = np.where(masks, base[None], img01[None]).astype(np.float32)
    ins = np.where(masks, img01[None], base[None]).astype(np.float32)
    rad = np.repeat(rad_row[None], steps + 1, 0)
    return prob_fn(dele, rad), prob_fn(ins, rad)


# --------------------------------------------------------------------------- #
# Radiomic permutation importance
# --------------------------------------------------------------------------- #
def permutation_importance(predict_rad_fn, rad, y, names, repeats=3, seed=0):
    """predict_rad_fn(rad) -> p(malignant) for fixed images; returns (base_auc, [(name, auc_drop)...])."""
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, predict_rad_fn(rad))
    drops = np.zeros(rad.shape[1])
    for j in range(rad.shape[1]):
        for _ in range(repeats):
            r = rad.copy()
            r[:, j] = rng.permutation(r[:, j])
            drops[j] += base - roc_auc_score(y, predict_rad_fn(r))
    drops /= repeats
    order = np.argsort(-drops)
    return float(base), [(names[j], float(drops[j])) for j in order]
