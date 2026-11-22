"""
histo_utils.py - numpy / sklearn side of the histopathology (H&E) experiments.
No PyTorch needed, so all of this is unit-tested separately (test_histo_utils.py).

  * UBC-OCEAN patch loader (patch_manifest.csv, optional patches_filtered/ filter)
  * SLIDE-level cross-validation (patches of one slide never straddle train/test)
  * Beer-Lambert stain perturbations for the robustness test
  * radiomics for RGB patches (+ stain-concentration statistics)
  * multi-class metrics at patch and slide level, ECE, per-class recall
  * cross-validated ablation driver, tables, plots, faithfulness helpers
"""
import csv
import hashlib
import os
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

import novelty_utils as nu
import ovarian_pipeline as op

CLASSES = ["HGSC", "EC", "CC", "LGSC", "MC"]

# Ruifrok & Johnston stain vectors (rows: haematoxylin, eosin, residual; columns: R,G,B optical density)
STAIN_M = np.array([[0.650, 0.704, 0.286],
                    [0.072, 0.990, 0.105],
                    [0.268, 0.570, 0.776]], np.float64)
STAIN_M = STAIN_M / np.linalg.norm(STAIN_M, axis=1, keepdims=True)
STAIN_MINV = np.linalg.inv(STAIN_M)
STAIN_FEATURE_NAMES = ["H_mean", "H_std", "H_p90", "E_mean", "E_std", "E_p90"]

# clean = all test patches; clean_sub = the sub-sample the perturbed conditions are computed on
ROBUST_CONDITIONS = ["clean", "clean_sub", "bright0.8", "wb", "hscale1.3", "escale0.7",
                     "blur", "jpeg30", "noise"]
METRIC_KEYS = ["balanced_acc", "macro_f1", "accuracy", "auc", "ece"]


# --------------------------------------------------------------------------- #
# Loading UBC-OCEAN patches
# --------------------------------------------------------------------------- #
def _read_csv(p):
    with open(p, newline="", encoding="utf-8-sig", errors="replace") as f:
        return list(csv.DictReader(f))


def _basename(p):
    return os.path.basename(p.replace("\\", "/"))


_INDEX = {}


def _index(root):
    """basename -> full path for every file under patches_filtered/ and patches/ (built lazily, once)."""
    if root not in _INDEX:
        idx = {}
        for sub in ("patches_filtered", "patches"):
            for dp, _, fs in os.walk(os.path.join(root, sub)):
                for f in fs:
                    idx.setdefault(f, os.path.join(dp, f))
        _INDEX[root] = idx
    return _INDEX[root]


def _resolve(root, row):
    p, lab = row["patch_path"], row["label"]
    b = _basename(p)
    cands = [p, p.replace("\\", os.sep).replace("/", os.sep),
             os.path.join(root, "patches", lab, b), os.path.join(root, "patches", b),
             os.path.join(root, "patches_filtered", lab, b), os.path.join(root, "patches_filtered", b)]
    hit = next((c for c in cands if os.path.isfile(c)), None)
    return hit if hit else _index(root).get(b)


def load_ubc_patches(root, max_per_slide=16, seed=42, use_filtered=True, size=224, task="subtype"):
    """Returns dict(X uint8 (N,H,W,3) RGB, y, slides, files, classes). Patches are capped per slide."""
    man = _read_csv(Path(root) / "patches" / "patch_manifest.csv")
    keep = None
    fdir = Path(root) / "patches_filtered"
    if use_filtered and fdir.is_dir():
        keep = {f for _, _, fs in os.walk(fdir) for f in fs} or None
    if task == "subtype":
        classes = list(CLASSES)
        mapping = {c: i for i, c in enumerate(classes)}
    elif task == "hgsc_vs_rest":
        classes = ["Other", "HGSC"]
        mapping = {c: int(c == "HGSC") for c in CLASSES}
    else:
        raise ValueError(task)
    rows = [r for r in man if r["label"] in mapping and (keep is None or _basename(r["patch_path"]) in keep)]
    by = defaultdict(list)
    for r in rows:
        by[r["source_image_id"]].append(r)
    rng = np.random.default_rng(seed)
    sel = []
    for sid in sorted(by):
        rs = by[sid]
        if len(rs) > max_per_slide:
            rs = [rs[i] for i in sorted(rng.choice(len(rs), max_per_slide, replace=False))]
        sel += rs
    X, y, slides, files, missing = [], [], [], [], 0
    for r in sel:
        p = _resolve(root, r)
        im = cv2.imread(p, cv2.IMREAD_COLOR) if p else None
        if im is None:
            missing += 1
            continue
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if im.shape[0] != size or im.shape[1] != size:
            im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
        X.append(im)
        y.append(mapping[r["label"]])
        slides.append(str(r["source_image_id"]))
        files.append(p)
    if missing > 0.05 * len(sel):
        raise FileNotFoundError(f"{missing}/{len(sel)} patches could not be read - check paths in the manifest")
    if missing:
        warnings.warn(f"{missing} patches could not be read and were skipped")
    return {"X": np.stack(X), "y": np.array(y), "slides": np.array(slides), "files": files, "classes": classes}


def make_synthetic_histo(n_slides=75, patches=6, size=128, seed=0):
    """Fake H&E-like patches, 5 'subtypes' differing in nuclear density/size, per-slide stain variation.
    Smoke-test data only."""
    rng = np.random.default_rng(seed)
    dens = [0.0020, 0.0035, 0.0012, 0.0008, 0.0028]
    rad = [2.5, 2.5, 3.5, 1.8, 2.5]
    X, y, slides = [], [], []
    for s in range(n_slides):
        k = s % 5
        sH, sE = rng.uniform(0.8, 1.25, 2)
        wb = rng.uniform(0.9, 1.0, 3)
        for _ in range(patches):
            pts = (rng.random((size, size)) < dens[k]).astype(np.float32)
            nuc = cv2.GaussianBlur(pts, (0, 0), rad[k])
            nuc = nuc / (nuc.max() + 1e-6)
            sm = cv2.GaussianBlur(rng.random((size, size)).astype(np.float32), (0, 0), 6)
            sm = (sm - sm.min()) / (sm.max() - sm.min() + 1e-6)
            C = np.stack([1.2 * sH * nuc ** 0.7, (0.35 + 0.4 * sm) * sE, 0.05 * np.ones_like(nuc)], -1)
            img = 255.0 * np.exp(-(C @ STAIN_M)) * wb + rng.normal(0, 3, (size, size, 3))
            X.append(np.clip(img, 0, 255).astype(np.uint8))
            y.append(k)
            slides.append(f"S{s:04d}")
    return {"X": np.stack(X), "y": np.array(y), "slides": np.array(slides), "files": None, "classes": list(CLASSES)}


# --------------------------------------------------------------------------- #
# Slide-level folds
# --------------------------------------------------------------------------- #
def slide_folds(y, slides, k, seed):
    us, first = np.unique(slides, return_index=True)
    skf = StratifiedKFold(k, shuffle=True, random_state=seed)
    out = []
    for a, b in skf.split(us, y[first]):
        out.append((np.where(np.isin(slides, us[a]))[0], np.where(np.isin(slides, us[b]))[0]))
    return out


def slide_inner_split(tr, y, slides, frac, seed):
    us, first = np.unique(slides[tr], return_index=True)
    a, b = train_test_split(us, test_size=frac, stratify=y[tr][first], random_state=seed)
    return tr[np.isin(slides[tr], a)], tr[np.isin(slides[tr], b)]


def subsample_per_slide(slides, idx, cap, rng):
    out = []
    for s in np.unique(slides[idx]):
        m = idx[slides[idx] == s]
        out.append(m if len(m) <= cap else rng.choice(m, cap, replace=False))
    return np.sort(np.concatenate(out))


def slide_aggregate(P, slides, y):
    """Mean patch probability per slide -> (slide probs, slide labels, slide ids)."""
    us, inv = np.unique(slides, return_inverse=True)
    Ps = np.zeros((len(us), P.shape[1]))
    np.add.at(Ps, inv, P)
    Ps /= np.bincount(inv)[:, None]
    ys = np.zeros(len(us), int)
    ys[inv] = y
    return Ps, ys, us


# --------------------------------------------------------------------------- #
# Beer-Lambert stain perturbations (numpy, uint8 RGB)
# --------------------------------------------------------------------------- #
def stain_scale_rgb(img, scales):
    """Scale stain concentrations: OD = -ln(I/I0) = C @ M  ->  C' = C*scales  ->  I' = I0 exp(-C' @ M)."""
    od = -np.log((img.astype(np.float64) + 1.0) / 256.0)
    C = od @ STAIN_MINV
    od2 = (C * np.asarray(scales, np.float64)) @ STAIN_M
    return np.clip(256.0 * np.exp(-od2) - 1.0, 0, 255).astype(np.uint8)


def np_perturb_rgb(img, kind, rng):
    """bright<g>: global illumination gain | wb: per-channel white-balance cast | hscale/escale<s>:
    haematoxylin/eosin concentration | blur: defocus | jpeg<q>: compression | noise: sensor noise."""
    if kind in ("clean", "clean_sub"):
        return img
    if kind.startswith("bright"):
        out = img.astype(np.float32) * float(kind[6:])
    elif kind.startswith("wb"):                        # "wb" = gains [1, .9, .8]; "wb<s>" = [1, 1-s, 1-2s]
        s = float(kind[2:]) if len(kind) > 2 else 0.1
        out = img.astype(np.float32) * np.array([1.0, 1 - s, 1 - 2 * s], np.float32)
    elif kind.startswith("hscale"):
        return stain_scale_rgb(img, [float(kind[6:]), 1, 1])
    elif kind.startswith("escale"):
        return stain_scale_rgb(img, [1, float(kind[6:]), 1])
    elif kind.startswith("blur"):                      # "blur" = sigma 1.5 (original), "blur0.7" = sigma 0.7
        out = cv2.GaussianBlur(img, (0, 0), float(kind[4:]) if len(kind) > 4 else 1.5)
    elif kind.startswith("jpeg"):
        ok, buf = cv2.imencode(".jpg", img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(kind[4:])])
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1]
    elif kind == "noise":
        out = img.astype(np.float32) + rng.normal(0, 8, img.shape)
    else:
        raise ValueError(kind)
    return np.clip(out, 0, 255).astype(np.uint8)


def perturb_set_rgb(X, kind, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([np_perturb_rgb(im, kind, rng) for im in X])


# --------------------------------------------------------------------------- #
# Radiomics for RGB patches (+ stain concentration statistics)
# --------------------------------------------------------------------------- #
def histo_radiomics_one(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    f = op.radiomic_features(op.physics_preprocess(gray))
    od = -np.log((img.astype(np.float32) + 1.0) / 256.0).reshape(-1, 3)
    C = od @ STAIN_MINV.astype(np.float32)
    st = []
    for k in (0, 1):
        c = C[:, k]
        st += [c.mean(), c.std(), np.percentile(c, 90)]
    return np.concatenate([f, np.array(st, np.float32)])


def _rad_chunk(chunk):
    return np.stack([histo_radiomics_one(im) for im in chunk])


def radiomics_rgb(X, workers=1):
    if workers <= 1 or len(X) < 64:
        out = _rad_chunk(X)
    else:
        parts = [X[c] for c in np.array_split(np.arange(len(X)), workers * 4) if len(c)]
        with ProcessPoolExecutor(workers) as ex:
            out = np.concatenate(list(ex.map(_rad_chunk, parts)))
    return np.nan_to_num(out).astype(np.float32)


def radiomic_names():
    histo_radiomics_one(np.random.default_rng(0).integers(0, 255, (32, 32, 3)).astype(np.uint8))
    return list(op.RADIOMIC_NAMES) + STAIN_FEATURE_NAMES


def cached_radiomics(X, cache_path, workers=1):
    key = hashlib.md5(np.ascontiguousarray(X[::max(1, len(X) // 50)]).tobytes()).hexdigest() + f"_{len(X)}"
    p = Path(cache_path)
    if p.exists():
        z = np.load(p, allow_pickle=False)
        if str(z["key"]) == key:
            return z["rad"]
    rad = radiomics_rgb(X, workers)
    np.savez(p, rad=rad, key=np.array(key))
    return rad


# --------------------------------------------------------------------------- #
# Metrics (multi-class)
# --------------------------------------------------------------------------- #
def ece_mc(y, P, bins=10):
    conf = P.max(1)
    correct = (P.argmax(1) == y).astype(float)
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def metrics_mc(y, P):
    pred = P.argmax(1)
    K = P.shape[1]
    try:
        if K == 2:
            auc = roc_auc_score(y, P[:, 1])
        else:
            auc = roc_auc_score(y, P, multi_class="ovr", average="macro", labels=list(range(K)))
    except ValueError:
        auc = float("nan")
    return {"accuracy": float(accuracy_score(y, pred)), "balanced_acc": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "auc": float(auc), "ece": ece_mc(y, P)}


def evaluate_both(y, P, slides):
    Ps, ys, _ = slide_aggregate(P, slides, y)
    return {"patch": metrics_mc(y, P), "slide": metrics_mc(ys, Ps)}


# --------------------------------------------------------------------------- #
# Cross-validated ablation driver (slide-level folds)
# --------------------------------------------------------------------------- #
def run_cv_histo(data, rad_all, variants, trainer, k=5, seed=42, val_frac=0.15,
                 conditions=ROBUST_CONDITIONS, robust_per_slide=6, workers=1,
                 out_dir=None, fold_hook=None, verbose=True, resume=False):
    """
    trainer.fit(cfg, tr, va, seed) -> model ; trainer.predict(model, X_uint8, R) -> probs (N,K)
    tr = {X,R,y}, va = {X,R,y,slides}. Returns (results[variant][condition] -> [{'patch','slide'} per fold],
    slide-level confusion matrices summed over folds).
    """
    import json
    X, y, slides, K = data["X"], data["y"], data["slides"], len(data["classes"])
    folds = slide_folds(y, slides, k, seed)
    results = {v: {c: [] for c in conditions} for v in variants}
    cms = {v: np.zeros((K, K), int) for v in variants}
    pred_rows = []                                     # out-of-fold slide-level probabilities (clean test sets)
    meta = {"seed": seed, "k": k, "val_frac": val_frac, "robust_per_slide": robust_per_slide,
            "n_patches": int(len(y)), "n_slides": int(len(np.unique(slides)))}
    n_done = 0
    marker = Path(out_dir) / "confusion_partial.txt" if out_dir is not None else None
    if resume and out_dir is not None and (Path(out_dir) / "results_partial.json").exists():
        prev = json.load(open(Path(out_dir) / "results_partial.json"))
        meta_p = Path(out_dir) / "run_meta.json"
        prev_meta = json.load(open(meta_p)) if meta_p.exists() else None
        if set(prev) != set(variants) or any(set(prev[v]) != set(conditions) for v in variants):
            print("resume: variants/conditions differ from the saved partial results -> starting from fold 1")
        elif prev_meta is not None and prev_meta != meta:
            print("resume: run settings differ from the saved partial results -> starting from fold 1")
        else:
            if prev_meta is None:
                print("resume: no run_meta.json (partial results written by older code); settings cannot be verified")
            n_done = min(min(len(prev[v]["clean"]) for v in variants), len(folds))
            for v in variants:
                for c in conditions:
                    results[v][c] = prev[v][c][:n_done]
            sp_path = Path(out_dir) / "slide_predictions.csv"
            if n_done and sp_path.exists():
                with open(sp_path, newline="") as fh:
                    rd = csv.reader(fh)
                    next(rd)
                    for row in rd:
                        if int(row[1]) < n_done:
                            pred_rows.append([row[0], int(row[1]), row[2], int(row[3])] + [float(q) for q in row[4:]])
                for r_ in pred_rows:                              # rebuild the confusion matrices of finished folds
                    cms[r_[0]][r_[3], int(np.argmax(r_[4:]))] += 1
            elif n_done:
                print("resume: slide_predictions.csv missing -> confusion matrices / saved predictions cover only the "
                      "newly run folds (per-class table will be omitted)")
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                marker.write_text("confusion matrices cover only the folds run after resuming\n")
            print(f"resume: {n_done} of {len(folds)} folds already done, continuing", flush=True)
    if marker is not None and n_done == 0 and marker.exists():
        marker.unlink()
    for f, (tr, te) in enumerate(folds):
        if f < n_done:
            continue
        tr_in, va = slide_inner_split(tr, y, slides, val_frac, seed + f)
        sc = StandardScaler().fit(rad_all[tr])

        def R(a, sc=sc):
            return np.nan_to_num(sc.transform(a)).astype(np.float32)

        sub = subsample_per_slide(slides, te, robust_per_slide, np.random.default_rng(seed + f))
        test_sets = {}
        for c in conditions:
            if c == "clean":
                idx, Xi, rr = te, X[te], rad_all[te]
            elif c == "clean_sub":
                idx, Xi, rr = sub, X[sub], rad_all[sub]
            else:
                idx, Xi = sub, perturb_set_rgb(X[sub], c, seed)
                rr = (np.zeros((len(Xi), rad_all.shape[1]), np.float32) if os.environ.get("NO_RADIOMICS") == "1"  # NO_RADIOMICS
                      else radiomics_rgb(Xi, workers))
            test_sets[c] = (Xi, R(rr), y[idx], slides[idx])
        models = {}
        for v, cfg in variants.items():
            if verbose:
                print(f"[fold {f + 1}/{len(folds)}] training {v} ...", flush=True)
            models[v] = trainer.fit(cfg, {"X": X[tr_in], "R": R(rad_all[tr_in]), "y": y[tr_in]},
                                    {"X": X[va], "R": R(rad_all[va]), "y": y[va], "slides": slides[va]}, seed + f)
            for c, (Xi, Ri, yi, si) in test_sets.items():
                P = trainer.predict(models[v], Xi, Ri)
                results[v][c].append(evaluate_both(yi, P, si))
                if c == "clean":
                    Ps, ys, us = slide_aggregate(P, si, yi)
                    cms[v] += confusion_matrix(ys, Ps.argmax(1), labels=list(range(K)))
                    pred_rows += [[v, f, s_, int(y_)] + [float(q) for q in p_] for s_, y_, p_ in zip(us, ys, Ps)]
            if verbose:
                m = results[v]["clean"][-1]["slide"]
                print(f"    slide-level: bal_acc={m['balanced_acc']:.3f} macroF1={m['macro_f1']:.3f} "
                      f"auc={m['auc']:.3f} ece={m['ece']:.3f}", flush=True)
        if fold_hook is not None:
            fold_hook(f, models, {"X_te": X[te], "R_te": R(rad_all[te]), "y_te": y[te], "slides_te": slides[te]})
        if out_dir is not None:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            json.dump(results, open(Path(out_dir) / "results_partial.json", "w"), indent=1)
            json.dump(meta, open(Path(out_dir) / "run_meta.json", "w"))
            with open(Path(out_dir) / "slide_predictions.csv", "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["variant", "fold", "slide", "y"] + [f"p{k}" for k in range(K)])
                w.writerows(pred_rows)
    return results, cms


def summarize_histo(results, level="slide"):
    import pandas as pd
    rows = []
    for v, d in results.items():
        for c, ms in d.items():
            row = {"variant": v, "condition": c}
            for key in METRIC_KEYS:
                vals = np.array([m[level][key] for m in ms], float)
                row[key + "_mean"], row[key + "_std"] = float(np.nanmean(vals)), float(np.nanstd(vals))
            rows.append(row)
    return pd.DataFrame(rows)


def to_markdown_histo(df, cms=None, classes=None):
    clean = df[df.condition == "clean"]
    L = ["**Slide-level results, clean test folds (mean ± std over folds)**", "",
         "| Variant | Balanced acc | Macro-F1 | Accuracy | AUC (OvR) | ECE |", "|---|---|---|---|---|---|"]
    for _, r in clean.iterrows():
        L.append(f"| {r.variant} | {r.balanced_acc_mean:.3f} ± {r.balanced_acc_std:.3f} | "
                 f"{r.macro_f1_mean:.3f} ± {r.macro_f1_std:.3f} | {r.accuracy_mean:.3f} ± {r.accuracy_std:.3f} | "
                 f"{r.auc_mean:.3f} ± {r.auc_std:.3f} | {r.ece_mean:.3f} ± {r.ece_std:.3f} |")
    conds = [c for c in dict.fromkeys(df.condition) if c != "clean"]
    if conds:
        L += ["", "**Slide-level balanced accuracy under perturbation (reference = clean_sub)**", "",
              "| Variant | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
        for v in dict.fromkeys(df.variant):
            d = df[df.variant == v].set_index("condition")
            L.append(f"| {v} | " + " | ".join(
                f"{d.loc[c].balanced_acc_mean:.3f} ± {d.loc[c].balanced_acc_std:.3f}" if c in d.index else "–"
                for c in conds) + " |")
    if cms is not None and classes is not None:
        L += ["", "**Per-class slide-level recall (confusion summed over folds)**", "",
              "| Variant | " + " | ".join(classes) + " |", "|---|" + "---|" * len(classes)]
        for v, cm in cms.items():
            rec = cm.diagonal() / np.maximum(cm.sum(1), 1)
            L.append(f"| {v} | " + " | ".join(f"{r:.2f}" for r in rec) + " |")
    return "\n".join(L)


def plot_robustness_histo(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conds = [c for c in dict.fromkeys(df.condition) if c != "clean"]
    plt.figure(figsize=(9.5, 4.8))
    for v in dict.fromkeys(df.variant):
        d = df[df.variant == v].set_index("condition")
        cs = [c for c in conds if c in d.index]
        d = d.loc[cs]
        plt.errorbar([conds.index(c) for c in cs], d.balanced_acc_mean, yerr=d.balanced_acc_std,
                     marker="o", capsize=3, label=v)
    plt.xticks(range(len(conds)), conds, rotation=30)
    plt.ylabel("slide-level balanced accuracy")
    plt.title("Robustness to illumination / stain / quality shifts")
    plt.legend(fontsize=7)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# --------------------------------------------------------------------------- #
# Explanation helpers
# --------------------------------------------------------------------------- #
def faithfulness_curves_rgb(prob_fn, img01, sal, rad_row, steps=10):
    """prob_fn(batch (N,H,W,3) float01, rad (N,D)) -> p(target class) (N,). Deletion / insertion curves."""
    base = cv2.GaussianBlur(img01.astype(np.float32), (0, 0), 15)
    masks = nu.faithfulness_masks(sal, steps)[..., None]
    dele = np.where(masks, base[None], img01[None]).astype(np.float32)
    ins = np.where(masks, img01[None], base[None]).astype(np.float32)
    rad = np.repeat(rad_row[None], steps + 1, 0)
    return prob_fn(dele, rad), prob_fn(ins, rad)


def permutation_importance_mc(predict_rad_fn, rad, y, names, repeats=3, seed=0):
    """Drop in balanced accuracy when one radiomic feature is shuffled; predict_rad_fn(rad) -> probs (N,K)."""
    rng = np.random.default_rng(seed)
    base = balanced_accuracy_score(y, predict_rad_fn(rad).argmax(1))
    drops = np.zeros(rad.shape[1])
    for j in range(rad.shape[1]):
        for _ in range(repeats):
            r = rad.copy()
            r[:, j] = rng.permutation(r[:, j])
            drops[j] += base - balanced_accuracy_score(y, predict_rad_fn(r).argmax(1))
    drops /= repeats
    order = np.argsort(-drops)
    return float(base), [(names[j], float(drops[j])) for j in order]


# --------------------------------------------------------------------------- #
# Paired slide-level bootstrap (uses slide_predictions.csv written by run_cv_histo)
# --------------------------------------------------------------------------- #
def load_slide_predictions(path):
    """Out-of-fold slide-level probabilities: one row per (variant, slide)."""
    import pandas as pd
    df = pd.read_csv(path, dtype={"slide": str})
    return df.drop_duplicates(["variant", "slide"], keep="last")


def _slide_arrays(df, variant, slides=None):
    d = df[df.variant == variant].set_index("slide").sort_index()
    if slides is not None:
        d = d.loc[slides]
    pc = [c for c in d.columns if c.startswith("p") and c[1:].isdigit()]
    return d["y"].to_numpy(), d[pc].to_numpy()


def _boot_metric(y, pred, idx, metric):
    if metric == "balanced_acc":
        return balanced_accuracy_score(y[idx], pred[idx])
    if metric == "macro_f1":
        return f1_score(y[idx], pred[idx], average="macro", zero_division=0)
    raise ValueError(metric)


def bootstrap_variant(df, variant, metric="balanced_acc", n_boot=2000, seed=0):
    """Estimate and 95% CI of one variant's slide-level metric (resampling slides)."""
    y, P = _slide_arrays(df, variant)
    pred = P.argmax(1)
    n = len(y)
    rng = np.random.default_rng(seed)
    vals = np.array([_boot_metric(y, pred, rng.integers(0, n, n), metric) for _ in range(n_boot)])
    return {"estimate": float(_boot_metric(y, pred, np.arange(n), metric)),
            "lo": float(np.percentile(vals, 2.5)), "hi": float(np.percentile(vals, 97.5))}


def bootstrap_compare(df, v1, v2, metric="balanced_acc", n_boot=2000, seed=0):
    """Paired bootstrap over slides of metric(v1) - metric(v2) on the slides both variants predicted."""
    common = sorted(set(df[df.variant == v1].slide) & set(df[df.variant == v2].slide))
    if len(common) < 20:
        raise ValueError(f"only {len(common)} slides in common between {v1} and {v2}")
    y, P1 = _slide_arrays(df, v1, common)
    y2, P2 = _slide_arrays(df, v2, common)
    if not (y == y2).all():
        raise ValueError("labels disagree between the two variants for the same slides")
    p1, p2 = P1.argmax(1), P2.argmax(1)
    n = len(y)
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        d[i] = _boot_metric(y, p1, idx, metric) - _boot_metric(y, p2, idx, metric)
    est = _boot_metric(y, p1, np.arange(n), metric) - _boot_metric(y, p2, np.arange(n), metric)
    p = min(1.0, 2 * min(float((d <= 0).mean()), float((d >= 0).mean())))
    return {"delta": float(est), "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)),
            "p_boot": p, "n_slides": n}


def bootstrap_table_md(df, variants, ref, metric="balanced_acc", n_boot=2000, seed=0):
    label = {"balanced_acc": "Balanced accuracy", "macro_f1": "Macro-F1"}[metric]
    L = [f"**Slide-level {label}, out-of-fold, 95% bootstrap CI over slides (reference: {ref})**", "",
         f"| Variant | {label} [95% CI] | Δ vs {ref} [95% CI] | bootstrap p |", "|---|---|---|---|"]
    for v in variants:
        b = bootstrap_variant(df, v, metric, n_boot, seed)
        cell = f"{b['estimate']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}]"
        if v == ref:
            L.append(f"| {v} | {cell} | – | – |")
        else:
            c = bootstrap_compare(df, v, ref, metric, n_boot, seed)
            L.append(f"| {v} | {cell} | {c['delta']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] | {c['p_boot']:.3f} |")
    L += ["", "p-values are not corrected for multiple comparisons."]
    return "\n".join(L)
