#!/usr/bin/env python3
"""
Physics-guided dual-branch hybrid pipeline for ovarian cancer detection
(implementation of the paper's framework).

Pipeline
  1. Physics-guided preprocessing  (median filter -> Beer-Lambert optical density -> CLAHE)
  2. Dual-branch features          (EfficientNet-B0 deep features  +  radiomics)
  3. Fusion + PCA                  (concatenate, then PCA keeping >95% variance)
  4. SMOTE                         (training set only)
  5. LightGBM classifier           (lr=0.01, 100 estimators, max_depth tuned by CV)
  6. Grad-CAM                      (on EfficientNet-B0, see note in gradcam_figure)

Expected data layout (binary):
    data/benign/*.png|jpg|...
    data/malignant/*.png|jpg|...

Usage
    python ovarian_pipeline.py --data-dir data --out outputs
    python ovarian_pipeline.py --synthetic --no-deep --out outputs_smoke   # quick test, no torch needed

Notes
  * All fitting (scaler, PCA, SMOTE, CV) is done on the training split only,
    so the test metrics are not leaked.
  * Augmentation (rotation / flip / scaling) is applied to training images only.
"""
import argparse
import json
import warnings
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import kurtosis, skew
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, regionprops
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CLASS_NAMES = ["benign", "malignant"]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_folder_dataset(data_dir, size):
    images, labels = [], []
    for cls_idx, cls in enumerate(CLASS_NAMES):
        files = sorted(p for p in (Path(data_dir) / cls).glob("*") if p.suffix.lower() in IMG_EXT)
        if not files:
            raise FileNotFoundError(f"No images found in {Path(data_dir) / cls}")
        for p in files:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                warnings.warn(f"Could not read {p}, skipping")
                continue
            images.append(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA))
            labels.append(cls_idx)
    return np.stack(images), np.array(labels)


def make_synthetic_dataset(n, size, seed, benign_frac=500 / 1200):
    """Texture images; 'malignant' ones get an irregular bright region + coarser texture.
    Only for smoke-testing the code, not a substitute for real data."""
    rng = np.random.default_rng(seed)
    n_b = int(n * benign_frac)
    labels = np.array([0] * n_b + [1] * (n - n_b))
    images = []
    yy, xx = np.mgrid[:size, :size]
    for lab in labels:
        base = cv2.GaussianBlur(rng.normal(110, 35, (size, size)).astype(np.float32), (0, 0), 2 + 2 * lab)
        if lab == 1:
            cy, cx = rng.integers(size // 4, 3 * size // 4, 2)
            ry, rx = rng.integers(size // 8, size // 4, 2)
            blob = np.exp(-(((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2))
            base += 70 * blob * (0.7 + 0.3 * rng.random((size, size)))
        base += rng.normal(0, 8, (size, size))
        images.append(np.clip(base, 0, 255).astype(np.uint8))
    return np.stack(images), labels


def augment(img, rng):
    """Rotation, flipping, scaling (paper, Sec. IV-A)."""
    h, w = img.shape
    angle = rng.uniform(-15, 15)
    scale = rng.uniform(0.9, 1.1)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    out = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
    if rng.random() < 0.5:
        out = cv2.flip(out, 0)
    return out


# --------------------------------------------------------------------------- #
# 1. Physics-guided preprocessing
# --------------------------------------------------------------------------- #
def physics_preprocess(img, clip_limit=2.0, tile=8):
    """
    Beer-Lambert: I = I0 * exp(-mu * x)   =>   mu * x = -ln(I / I0)   (optical density)
    Steps: noise filtering -> convert intensity to optical density (which varies
    linearly with attenuation) -> CLAHE contrast enhancement.
    """
    g = cv2.medianBlur(img, 3)                              # noise filtering
    I = g.astype(np.float32) + 1.0
    I0 = np.percentile(I, 99)                               # incident-intensity estimate
    od = -np.log(np.clip(I / I0, 1e-3, 1.0))                # = mu * x
    od = cv2.normalize(od, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return clahe.apply(od)                                  # contrast-limited adaptive HE


# --------------------------------------------------------------------------- #
# 2b. Radiomics branch (F_r)
# --------------------------------------------------------------------------- #
RADIOMIC_NAMES = None


def radiomic_features(img):
    f = {}
    x = img.astype(np.float64)
    p = np.bincount(img.ravel(), minlength=256) / img.size
    nz = p[p > 0]
    # first-order
    f["mean"], f["variance"] = x.mean(), x.var()
    f["skewness"], f["kurtosis"] = skew(x.ravel()), kurtosis(x.ravel())
    f["entropy"] = -(nz * np.log2(nz)).sum()
    f["energy"] = (p ** 2).sum()
    f["min"], f["max"] = x.min(), x.max()
    f["p10"], f["p90"] = np.percentile(x, 10), np.percentile(x, 90)
    f["iqr"] = np.percentile(x, 75) - np.percentile(x, 25)
    # GLCM texture (64 grey levels, 2 distances x 4 angles, averaged)
    q = (img // 4).astype(np.uint8)
    glcm = graycomatrix(q, [1, 2], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=64, symmetric=True, normed=True)
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
        f[f"glcm_{prop}"] = graycoprops(glcm, prop).mean()
    pg = glcm[glcm > 0]
    f["glcm_entropy"] = -(pg * np.log2(pg)).sum() / (glcm.shape[2] * glcm.shape[3])
    # shape of the largest segmented region (Otsu + morphology)
    _, mask = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    props = regionprops(label(mask))
    shape_keys = ["area_frac", "perimeter", "eccentricity", "solidity", "extent",
                  "major_axis", "minor_axis", "circularity"]
    if props:
        r = max(props, key=lambda r: r.area)
        per = max(r.perimeter, 1e-6)
        vals = [r.area / img.size, per, r.eccentricity, r.solidity, r.extent,
                r.axis_major_length, r.axis_minor_length, 4 * np.pi * r.area / per ** 2]
    else:
        vals = [0.0] * len(shape_keys)
    f.update(dict(zip(["shape_" + k for k in shape_keys], vals)))

    global RADIOMIC_NAMES
    RADIOMIC_NAMES = list(f.keys())
    return np.array(list(f.values()), dtype=np.float32)


# --------------------------------------------------------------------------- #
# 2a. Deep branch (F_d): EfficientNet-B0, global-average-pooled 1280-d vector
# --------------------------------------------------------------------------- #
def build_backbone(pretrained):
    import torch
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device), device


def to_tensor(batch_u8, device):
    import torch
    x = torch.from_numpy(batch_u8.astype(np.float32) / 255.0)[:, None].repeat(1, 3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return ((x - mean) / std).to(device)


def deep_features(model, device, images, batch_size=32):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            x = to_tensor(images[i:i + batch_size], device)
            f = model.avgpool(model.features(x)).flatten(1)
            out.append(f.cpu().numpy())
    return np.concatenate(out)


# --------------------------------------------------------------------------- #
# 4. SMOTE (imbalanced-learn if available, otherwise a plain numpy version)
# --------------------------------------------------------------------------- #
def simple_smote(X, y, seed, k=5):
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    target = counts.max()
    Xs, ys = [X], [y]
    for c, n in zip(classes, counts):
        if n == target:
            continue
        Xc = X[y == c]
        kk = min(k, len(Xc) - 1)
        nn = NearestNeighbors(n_neighbors=kk + 1).fit(Xc)
        neigh = nn.kneighbors(Xc, return_distance=False)[:, 1:]
        base = rng.integers(0, len(Xc), target - n)
        nb = neigh[base, rng.integers(0, kk, target - n)]
        lam = rng.random((target - n, 1))
        Xs.append(Xc[base] + lam * (Xc[nb] - Xc[base]))          # x_new = x_i + lam (x_j - x_i)
        ys.append(np.full(target - n, c))
    return np.vstack(Xs), np.concatenate(ys)


def apply_smote(X, y, method, seed):
    try:
        from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
        cls = {"smote": SMOTE, "borderline": BorderlineSMOTE, "adasyn": ADASYN}[method]
        return cls(random_state=seed).fit_resample(X, y)
    except ImportError:
        if method != "smote":
            warnings.warn("imbalanced-learn missing: falling back to plain SMOTE")
        return simple_smote(X, y, seed)


# --------------------------------------------------------------------------- #
# 5. LightGBM (falls back to sklearn HistGradientBoosting if not installed)
# --------------------------------------------------------------------------- #
def make_gbm(max_depth, args):
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(learning_rate=args.lr, n_estimators=args.n_estimators,
                              max_depth=max_depth, random_state=args.seed, verbose=-1)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        warnings.warn("lightgbm not installed: using sklearn HistGradientBoostingClassifier instead")
        return HistGradientBoostingClassifier(learning_rate=args.lr, max_iter=args.n_estimators,
                                              max_depth=None if max_depth < 0 else max_depth,
                                              random_state=args.seed)


def tune_and_fit_gbm(Z_train, y_train, args):
    """Tune max_depth by stratified CV; SMOTE is applied inside each fold (train part only)."""
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)
    best_depth, best_auc = None, -1
    for depth in [3, 5, 7, -1]:
        aucs = []
        for tr, va in skf.split(Z_train, y_train):
            Xb, yb = apply_smote(Z_train[tr], y_train[tr], args.smote, args.seed)
            m = make_gbm(depth, args).fit(Xb, yb)
            aucs.append(roc_auc_score(y_train[va], m.predict_proba(Z_train[va])[:, 1]))
        if np.mean(aucs) > best_auc:
            best_depth, best_auc = depth, float(np.mean(aucs))
    print(f"  LightGBM max_depth={best_depth} (CV AUC {best_auc:.3f})")
    Xb, yb = apply_smote(Z_train, y_train, args.smote, args.seed)
    return make_gbm(best_depth, args).fit(Xb, yb), best_depth


# --------------------------------------------------------------------------- #
# Evaluation / plots
# --------------------------------------------------------------------------- #
def evaluate(y_true, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(y_true, pred, zero_division=0),
        "auc": roc_auc_score(y_true, prob),
        "confusion": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    }


def plot_roc(y_true, probs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 5))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_true, p):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC curves"); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()


def plot_confusion(cm, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mat = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]])
    plt.figure(figsize=(4.5, 4))
    plt.imshow(mat, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, mat[i, j], ha="center", va="center",
                     color="white" if mat[i, j] > mat.max() / 2 else "black", fontsize=14)
    plt.xticks([0, 1], CLASS_NAMES); plt.yticks([0, 1], CLASS_NAMES)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion matrix (proposed)")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_pca(Z_tr, y_tr, Z_te, y_te, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 5))
    for c, col, m in [(0, "tab:blue", "o"), (1, "tab:red", "^")]:
        plt.scatter(Z_tr[y_tr == c, 0], Z_tr[y_tr == c, 1], c=col, marker=m, s=14, alpha=0.5, label=f"{CLASS_NAMES[c]} (train)")
    plt.scatter(Z_te[:, 0], Z_te[:, 1], c="green", marker="*", s=40, label="test")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("PCA of fused features")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


# --------------------------------------------------------------------------- #
# 6. Grad-CAM
# --------------------------------------------------------------------------- #
def gradcam_figure(model, device, head, images_u8, labels, preds, path, n=6):
    """
    LightGBM is not differentiable, so Grad-CAM is computed on a differentiable
    proxy: EfficientNet-B0 feature maps -> global-avg-pool -> a logistic-regression
    head fitted on the deep features. The heatmap therefore shows which regions drive
    the *deep branch's* malignant score, not LightGBM's own decision.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    W = torch.tensor(head.coef_[0], dtype=torch.float32, device=device)
    b = torch.tensor(head.intercept_[0], dtype=torch.float32, device=device)
    n = min(n, len(images_u8))
    fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 5))
    for i in range(n):
        x = to_tensor(images_u8[i:i + 1], device)
        with torch.enable_grad():
            fmap = model.features(x).detach().requires_grad_(True)
            score = fmap.mean((2, 3)).flatten() @ W + b            # malignant logit
            grads, = torch.autograd.grad(score, fmap)
        weights = grads.mean((2, 3), keepdim=True)
        cam = torch.relu((weights * fmap).sum(1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=images_u8.shape[1:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
        base = cv2.cvtColor(images_u8[i], cv2.COLOR_GRAY2RGB)
        overlay = (0.55 * base + 0.45 * heat).astype(np.uint8)
        axes[0, i].imshow(images_u8[i], cmap="gray"); axes[0, i].axis("off")
        axes[0, i].set_title(f"true: {CLASS_NAMES[labels[i]][:3]}", fontsize=8)
        axes[1, i].imshow(overlay); axes[1, i].axis("off")
        axes[1, i].set_title(f"pred: {CLASS_NAMES[preds[i]][:3]}", fontsize=8)
    plt.suptitle("Grad-CAM (deep-branch proxy head)", fontsize=10)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--synthetic", action="store_true", help="use generated toy data (smoke test only)")
    ap.add_argument("--n-synthetic", type=int, default=400)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--n-aug", type=int, default=2, help="augmented copies per training image")
    ap.add_argument("--no-deep", action="store_true", help="skip EfficientNet branch (radiomics only)")
    ap.add_argument("--no-pretrained", action="store_true", help="random-init EfficientNet (no weight download)")
    ap.add_argument("--no-gradcam", action="store_true")
    ap.add_argument("--smote", choices=["smote", "borderline", "adasyn"], default="smote")
    ap.add_argument("--variance", type=float, default=0.95, help="PCA explained variance")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--n-estimators", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ---- data + split (before any fitting) ---------------------------------
    if args.synthetic:
        images, y = make_synthetic_dataset(args.n_synthetic, args.img_size, args.seed)
    else:
        images, y = load_folder_dataset(args.data_dir, args.img_size)
    print(f"Loaded {len(y)} images  (benign={np.sum(y == 0)}, malignant={np.sum(y == 1)})")
    idx = np.arange(len(y))
    tr_idx, te_idx = train_test_split(idx, test_size=args.test_size, stratify=y, random_state=args.seed)
    X_tr_raw, y_tr = images[tr_idx], y[tr_idx]
    X_te_raw, y_te = images[te_idx], y[te_idx]

    # augmentation: training only
    if args.n_aug > 0:
        aug = [augment(im, rng) for im in X_tr_raw for _ in range(args.n_aug)]
        X_tr_raw = np.concatenate([X_tr_raw, np.stack(aug)])
        y_tr = np.concatenate([y_tr, np.repeat(y[tr_idx], args.n_aug)])
    print(f"Train: {len(y_tr)} (after augmentation)   Test: {len(y_te)}")

    # ---- 1. physics-guided preprocessing -----------------------------------
    print("Physics-guided preprocessing...")
    X_tr = np.stack([physics_preprocess(im) for im in X_tr_raw])
    X_te = np.stack([physics_preprocess(im) for im in X_te_raw])

    # ---- 2. dual-branch feature extraction ---------------------------------
    print("Radiomics branch...")
    Fr_tr = np.stack([radiomic_features(im) for im in X_tr])
    Fr_te = np.stack([radiomic_features(im) for im in X_te])
    Fr_tr, Fr_te = np.nan_to_num(Fr_tr), np.nan_to_num(Fr_te)

    use_deep = not args.no_deep
    model = device = None
    if use_deep:
        print("Deep branch (EfficientNet-B0)...")
        model, device = build_backbone(pretrained=not args.no_pretrained)
        Fd_tr = deep_features(model, device, X_tr, args.batch_size)
        Fd_te = deep_features(model, device, X_te, args.batch_size)

    # ---- 3. standardise per branch, fuse, PCA (fit on train only) ----------
    sc_r = StandardScaler().fit(Fr_tr)
    parts_tr, parts_te = [sc_r.transform(Fr_tr)], [sc_r.transform(Fr_te)]
    if use_deep:
        sc_d = StandardScaler().fit(Fd_tr)
        parts_tr.insert(0, sc_d.transform(Fd_tr))
        parts_te.insert(0, sc_d.transform(Fd_te))
    F_tr, F_te = np.hstack(parts_tr), np.hstack(parts_te)          # F = [F_d (+) F_r]
    pca = PCA(n_components=args.variance, svd_solver="full", random_state=args.seed).fit(F_tr)
    Z_tr, Z_te = pca.transform(F_tr), pca.transform(F_te)          # Z = W^T F
    print(f"Fused dim {F_tr.shape[1]} -> PCA dim {Z_tr.shape[1]} (>= {args.variance:.0%} variance)")
    plot_pca(Z_tr, y_tr, Z_te, y_te, out / "pca.png")

    # ---- 4+5. SMOTE + LightGBM --------------------------------------------
    print("Training proposed model (SMOTE + LightGBM)...")
    gbm, depth = tune_and_fit_gbm(Z_tr, y_tr, args)
    probs = {"Proposed": gbm.predict_proba(Z_te)[:, 1]}

    # ---- baselines / ablations on the same split ---------------------------
    print("Baselines...")
    Zs_tr, ys_tr = apply_smote(Z_tr, y_tr, args.smote, args.seed)
    probs["SVM (fused PCA)"] = SVC(probability=True, random_state=args.seed).fit(Zs_tr, ys_tr).predict_proba(Z_te)[:, 1]
    probs["RandomForest (fused PCA)"] = RandomForestClassifier(
        n_estimators=300, random_state=args.seed, n_jobs=-1).fit(Zs_tr, ys_tr).predict_proba(Z_te)[:, 1]
    probs["LightGBM (radiomics only)"] = make_gbm(depth, args).fit(
        *apply_smote(sc_r.transform(Fr_tr), y_tr, args.smote, args.seed)).predict_proba(sc_r.transform(Fr_te))[:, 1]
    head = None
    if use_deep:
        head = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Fd_tr, y_tr)
        probs["EfficientNet features + linear head"] = head.predict_proba(Fd_te)[:, 1]
        probs["LightGBM (deep only)"] = make_gbm(depth, args).fit(
            *apply_smote(sc_d.transform(Fd_tr), y_tr, args.smote, args.seed)).predict_proba(sc_d.transform(Fd_te))[:, 1]

    results = {k: evaluate(y_te, p) for k, p in probs.items()}
    json.dump(results, open(out / "metrics.json", "w"), indent=2)
    print(f"\n{'Model':38s} {'Acc':>6s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'AUC':>6s}")
    for k, r in results.items():
        print(f"{k:38s} {r['accuracy']:6.3f} {r['precision']:6.3f} {r['recall']:6.3f} {r['f1']:6.3f} {r['auc']:6.3f}")

    plot_roc(y_te, probs, out / "roc.png")
    plot_confusion(results["Proposed"]["confusion"], out / "confusion_matrix.png")

    # ---- 6. Grad-CAM -------------------------------------------------------
    if use_deep and not args.no_gradcam:
        preds = (probs["Proposed"] >= 0.5).astype(int)
        order = np.concatenate([np.where(y_te == 1)[0][:3], np.where(y_te == 0)[0][:3]])
        gradcam_figure(model, device, head, X_te[order], y_te[order], preds[order], out / "gradcam.png")
    print(f"\nOutputs written to {out.resolve()}")


if __name__ == "__main__":
    main()
