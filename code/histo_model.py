#!/usr/bin/env python3
"""
histo_model.py - physics-guided hybrid model for H&E histopathology (UBC-OCEAN ovarian subtypes).

Brightfield microscopy is transmitted-light imaging, so Beer-Lambert holds:
    I_c = I0_c * exp(-sum_k C_k * M_kc)        (C_k: stain concentrations, M: stain vectors)
Components
  1. StainODLayer     : per-image white point I0 (99th percentile per channel), optical density
                        OD = -ln(I/I0), LEARNABLE unmixing into stain concentrations (initialised with the
                        Ruifrok H&E vectors), per-image concentration normalisation, and re-synthesis with a
                        LEARNABLE reference stain matrix -> a stain-normalised RGB image for EfficientNet.
                        Invariant by construction to per-channel illumination / white-balance gain and to
                        stain-concentration scaling.
  2. Consistency loss : KL(p(clean) || p(perturbed)) under Beer-Lambert-consistent perturbations
                        (stain-concentration scaling/shift + white-point gain), applied to the whole network.
  3. Cross-attention fusion : radiomic vector (texture + stain statistics) queries EfficientNet tokens.
  4. Explainability   : Grad-CAM, attention map, deletion/insertion faithfulness, Grad-CAM-vs-attention
                        agreement, radiomic permutation importance, drift of the learned stain vectors.
Variants (ablation): A_baseline, B_stainOD, C_stainOD+consist, D_attn_only, E_full, F_consist_only,
                     G_colorjitter (standard colour augmentation: the baseline the physics must beat),
                     H_stainaug (the SAME stain / illumination perturbations that the consistency loss uses,
                     applied as plain data augmentation: the strong, like-for-like baseline).

Evaluation is SLIDE-level: 5-fold CV over slides, patch probabilities averaged per slide.

Usage
    python histo_model.py --ubc-dir "D:\\...\\UBC Cancer subtype" --out results_histo
    python histo_model.py --synthetic --epochs 2 --folds 3 --variants A_baseline E_full --no-pretrained --img-size 128 --out smoke_histo
"""
import argparse
import copy
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

import histo_utils as hu
import stain_norm
from novel_model import ConcatFusion, CrossAttnFusion

VARIANTS = {
    "A_baseline":        dict(physics=False, fusion="concat", consist=False, cjit=False),
    "B_stainOD":         dict(physics=True,  fusion="concat", consist=False, cjit=False),
    "C_stainOD+consist": dict(physics=True,  fusion="concat", consist=True,  cjit=False),
    "D_attn_only":       dict(physics=False, fusion="attn",   consist=False, cjit=False),
    "E_full":            dict(physics=True,  fusion="attn",   consist=True,  cjit=False),
    "F_consist_only":    dict(physics=False, fusion="concat", consist=True,  cjit=False),
    "G_colorjitter":     dict(physics=False, fusion="concat", consist=False, cjit=True),
    "H_stainaug":        dict(physics=False, fusion="concat", consist=False, cjit=False, saug=True),
    "N_reinhard":        dict(physics=False, fusion="concat", consist=False, cjit=False, norm="reinhard"),
    "N_macenko":         dict(physics=False, fusion="concat", consist=False, cjit=False, norm="macenko"),
}
IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# 1. Stain-aware physics layer
# --------------------------------------------------------------------------- #
class StainODLayer(nn.Module):
    """x: (B,3,H,W) RGB in [0,1] -> stain-normalised, ImageNet-normalised image (B,3,H,W)."""

    def __init__(self, white_q=0.99, conc_q=0.99, eps=1e-3, ref=(1.97, 1.03, 0.5), sub=4):
        super().__init__()
        M = torch.tensor(hu.STAIN_M, dtype=torch.float32)
        self.unmix = nn.Parameter(torch.linalg.inv(M).t().contiguous())   # (stain k, channel c)
        self.stain = nn.Parameter(M.clone())                              # reference stain vectors (k, c)
        self.log_ref = nn.Parameter(torch.log(torch.tensor(ref, dtype=torch.float32)))
        self.white_q, self.conc_q, self.eps, self.sub = white_q, conc_q, eps, sub
        self.register_buffer("mean", torch.tensor(IMNET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMNET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        b = x.shape[0]
        w = torch.quantile(x.flatten(2)[:, :, ::self.sub], self.white_q, dim=2).clamp_min(0.3)   # (B,3) I0
        od = -torch.log(x.clamp_min(self.eps) / w.view(b, 3, 1, 1))                              # Beer-Lambert OD
        c = F.relu(torch.einsum("kc,bchw->bkhw", self.unmix, od))                                # concentrations
        q = torch.quantile(c.flatten(2)[:, :, ::self.sub], self.conc_q, dim=2).clamp_min(1e-3)   # (B,3)
        c = c / q.view(b, 3, 1, 1) * self.log_ref.exp().view(1, 3, 1, 1)
        out = torch.exp(-torch.einsum("kc,bkhw->bchw", self.stain, c))                           # re-synthesis
        return (out - self.mean) / self.std


class ImageNetNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor(IMNET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMNET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class HistoNet(nn.Module):
    def __init__(self, n_rad, n_cls, physics, fusion, pretrained=True, d=128):
        super().__init__()
        self.pre = StainODLayer() if physics else ImageNetNorm()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.features = efficientnet_b0(weights=weights).features
        self.fusion = (CrossAttnFusion if fusion == "attn" else ConcatFusion)(1280, n_rad, d, n_cls)

    def encode(self, x):
        return self.features(self.pre(x))

    def fuse(self, fmap, rad):
        tokens = fmap.flatten(2).transpose(1, 2)
        return self.fusion(tokens, fmap.mean((2, 3)), rad)

    def forward(self, x, rad):
        fmap = self.encode(x)
        logits, attn = self.fuse(fmap, rad)
        return logits, attn, fmap


# --------------------------------------------------------------------------- #
# 2. Perturbations used in training (torch, Beer-Lambert consistent)
# --------------------------------------------------------------------------- #
def stain_perturb(x, scale=(0.75, 1.3), shift=0.05, gain=(0.85, 1.0)):
    """x in [0,1] (white = 1): scale / shift stain concentrations, then per-channel white-point gain."""
    b, dev = x.shape[0], x.device
    M = torch.tensor(hu.STAIN_M, dtype=torch.float32, device=dev)
    Minv = torch.linalg.inv(M)
    od = -torch.log(x.clamp_min(1e-3))
    c = torch.einsum("bchw,ck->bkhw", od, Minv)
    s = torch.empty(b, 3, 1, 1, device=dev).uniform_(*scale)
    s[:, 2] = 1.0
    sh = torch.empty(b, 3, 1, 1, device=dev).uniform_(-shift, shift)
    od2 = torch.einsum("bkhw,kc->bchw", c * s + sh, M)
    g = torch.empty(b, 3, 1, 1, device=dev).uniform_(*gain)
    return (torch.exp(-od2) * g).clamp(0, 1)


def color_jitter(x):
    """Standard colour augmentation (baseline G): brightness, contrast, saturation."""
    b, dev = x.shape[0], x.device
    x = x * torch.empty(b, 1, 1, 1, device=dev).uniform_(0.75, 1.25)
    m = x.mean((1, 2, 3), keepdim=True)
    x = (x - m) * torch.empty(b, 1, 1, 1, device=dev).uniform_(0.75, 1.25) + m
    g = x.mean(1, keepdim=True)
    x = (x - g) * torch.empty(b, 1, 1, 1, device=dev).uniform_(0.7, 1.3) + g
    return x.clamp(0, 1)


class PatchDS(Dataset):
    def __init__(self, X, R, y, train, seed):
        self.X, self.R, self.y, self.train = X, R, y, train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        im = self.X[i]
        if self.train:
            im = np.rot90(im, int(self.rng.integers(4)))
            if self.rng.random() < 0.5:
                im = im[:, ::-1]
        x = torch.from_numpy(np.ascontiguousarray(im).astype(np.float32) / 255.0).permute(2, 0, 1)
        return x, torch.from_numpy(self.R[i]), int(self.y[i])


def to_tensor_batch(X_uint8, dev):
    return torch.from_numpy(np.ascontiguousarray(X_uint8).astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(dev)


class HistoTrainer:
    def __init__(self, args, n_cls):
        self.args, self.n_cls = args, n_cls
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @torch.no_grad()
    def predict_float(self, model, X01, R, bs=64):
        """X01: (N,H,W,3) float in [0,1] -> probabilities (N,K)."""
        model.eval()
        out = []
        for i in range(0, len(X01), bs):
            x = torch.from_numpy(np.ascontiguousarray(X01[i:i + bs], dtype=np.float32)).permute(0, 3, 1, 2).to(self.device)
            r = torch.from_numpy(np.asarray(R[i:i + bs], np.float32)).to(self.device)
            out.append(F.softmax(model(x, r)[0], 1).cpu().numpy())
        return np.concatenate(out)

    def predict(self, model, X, R):
        norm = getattr(model, "normaliser", None)
        if norm is not None:                      # classical stain normalisation, applied to every image
            X = norm(X)
        return self.predict_float(model, X.astype(np.float32) / 255.0, R)

    def fit(self, cfg, tr, va, seed):
        a, dev, K = self.args, self.device, self.n_cls
        torch.manual_seed(seed)
        model = HistoNet(tr["R"].shape[1], K, cfg["physics"], cfg["fusion"], not a.no_pretrained).to(dev)
        norm = stain_norm.make_normaliser(cfg["norm"], tr["X"]) if cfg.get("norm") else None
        model.normaliser = norm
        if norm is not None:
            tr = dict(tr, X=norm(tr["X"]))
            print(f"      {cfg['norm']}: train images normalised, fallback {norm.n_fallback}/{norm.n_total}", flush=True)
        backbone = list(model.features.parameters())
        rest = [p for n, p in model.named_parameters() if not n.startswith("features.")]
        opt = torch.optim.AdamW([{"params": backbone, "lr": a.lr_backbone},
                                 {"params": rest, "lr": a.lr_head}], weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
        cnt = np.maximum(np.bincount(tr["y"], minlength=K), 1)
        cw = torch.tensor(len(tr["y"]) / (K * cnt), dtype=torch.float32, device=dev)
        loader = DataLoader(PatchDS(tr["X"], tr["R"], tr["y"], True, seed), batch_size=a.batch_size,
                            shuffle=True, drop_last=len(tr["y"]) > a.batch_size, num_workers=a.dl_workers)
        best, best_state, bad = -1.0, None, 0
        for ep in range(a.epochs):
            model.train()
            for x, r, y in loader:
                x, r, y = x.to(dev), r.to(dev), y.to(dev)
                if cfg["cjit"]:
                    x = color_jitter(x)
                if cfg.get("saug"):                      # stain / illumination augmentation on 80% of the batch
                    m = (torch.rand(x.shape[0], 1, 1, 1, device=dev) < 0.8).float()
                    x = m * stain_perturb(x) + (1 - m) * x
                logits, _, _ = model(x, r)
                loss = F.cross_entropy(logits, y, weight=cw)
                if cfg["consist"]:
                    lp, _, _ = model(stain_perturb(x), r)
                    loss = loss + a.consist_weight * F.kl_div(
                        F.log_softmax(lp, 1), F.softmax(logits.detach(), 1), reduction="batchmean")
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
            Pv = self.predict(model, va["X"], va["R"])
            Ps, ys, _ = hu.slide_aggregate(Pv, va["slides"], va["y"])
            score = balanced_accuracy_score(ys, Ps.argmax(1))
            if score > best:
                best, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= a.patience:
                    break
        model.load_state_dict(best_state)
        return model.eval()


# --------------------------------------------------------------------------- #
# 4. Explainability
# --------------------------------------------------------------------------- #
def explain(trainer, model, data, out_dir, args, rad_names, classes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dev = trainer.device
    X, R, y = data["X_te"], data["R_te"], data["y_te"]
    rng = np.random.default_rng(0)
    pool = rng.choice(len(X), min(args.xai_pool, len(X)), replace=False)
    P = trainer.predict(model, X[pool], R[pool])
    ok = pool[P.argmax(1) == y[pool]]
    if len(ok) == 0:
        print("XAI: no correctly classified patch in the pool, skipping")
        return
    per = int(np.ceil(args.xai_n / len(classes)))
    sel = np.concatenate([ok[y[ok] == k][:per] for k in range(len(classes))])[:args.xai_n]
    model.eval()
    gc_maps, att_maps = [], []
    for i in sel:
        x = to_tensor_batch(X[i:i + 1], dev)
        r = torch.from_numpy(R[i:i + 1]).to(dev)
        with torch.enable_grad():
            fmap = model.encode(x)
            logits, attn = model.fuse(fmap, r)
            grads, = torch.autograd.grad(logits[0, int(y[i])], fmap)
        h, w = fmap.shape[2:]
        cam = F.relu((grads.mean((2, 3), keepdim=True) * fmap).sum(1))[0].detach().cpu().numpy()
        gc_maps.append(cam)
        att_maps.append(None if attn is None else attn[0].detach().cpu().numpy().reshape(h, w))

    H, W = X.shape[1:3]

    def up(m):
        return cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)

    sources = {"gradcam": [up(m) for m in gc_maps], "random": [rng.random((H, W)) for _ in sel]}
    if att_maps[0] is not None:
        sources["attention"] = [up(m) for m in att_maps]

    report = {"n_patches": int(len(sel)), "faithfulness": {}}
    for name, sals in sources.items():
        dels, inss = [], []
        for i, sal in zip(sel, sals):
            t = int(y[i])

            def prob_fn(batch, rd, t=t):
                return trainer.predict_float(model, batch, rd)[:, t]

            d, s = hu.faithfulness_curves_rgb(prob_fn, X[i].astype(np.float32) / 255.0, sal, R[i])
            dels.append(hu.nu.curve_auc(d))
            inss.append(hu.nu.curve_auc(s))
        report["faithfulness"][name] = {
            "deletion_auc_mean(lower=better)": float(np.mean(dels)), "deletion_auc_std": float(np.std(dels)),
            "insertion_auc_mean(higher=better)": float(np.mean(inss)), "insertion_auc_std": float(np.std(inss))}
    if att_maps[0] is not None:
        rhos = [spearmanr(g.ravel(), a.ravel())[0] for g, a in zip(gc_maps, att_maps)]
        report["gradcam_attention_spearman_mean"] = float(np.nanmean(rhos))

    sub = pool[:min(len(pool), 400)]
    with torch.no_grad():
        fmaps = torch.cat([model.encode(to_tensor_batch(X[sub[j:j + 32]], dev)).cpu() for j in range(0, len(sub), 32)])

    def predict_rad(rd):
        out = []
        with torch.no_grad():
            for j in range(0, len(fmaps), 64):
                lg, _ = model.fuse(fmaps[j:j + 64].to(dev), torch.from_numpy(rd[j:j + 64]).to(dev))
                out.append(F.softmax(lg, 1).cpu().numpy())
        return np.concatenate(out)

    base, imp = hu.permutation_importance_mc(predict_rad, R[sub], y[sub], rad_names, repeats=args.perm_repeats)
    report["radiomic_permutation_importance"] = {"base_balanced_acc": base, "top10_balanced_acc_drop": imp[:10]}
    if hasattr(model.pre, "stain"):
        S = model.pre.stain.detach().cpu().numpy()
        S = S / np.linalg.norm(S, axis=1, keepdims=True)
        report["learned_stain_vectors_cosine_to_Ruifrok_init(H,E,residual)"] = [float(v) for v in (S * hu.STAIN_M).sum(1)]
        report["learned_reference_concentration_scale"] = [float(v) for v in model.pre.log_ref.exp().detach().cpu()]
    json.dump(report, open(Path(out_dir) / "xai.json", "w"), indent=2)

    n = min(6, len(sel))
    cols = 3 if att_maps[0] is not None else 2
    fig, ax = plt.subplots(n, cols, figsize=(3 * cols, 3 * n), squeeze=False)
    for row in range(n):
        ax[row, 0].imshow(X[sel[row]])
        ax[row, 0].set_title(classes[int(y[sel[row]])], fontsize=8)
        for col, key in enumerate(["gradcam", "attention"][:cols - 1], start=1):
            s = sources[key][row]
            s = (s - s.min()) / (s.max() - s.min() + 1e-8)
            heat = cv2.applyColorMap((s * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
            ax[row, col].imshow((0.55 * X[sel[row]] + 0.45 * heat).astype(np.uint8))
            ax[row, col].set_title(key, fontsize=8)
        for c in range(cols):
            ax[row, c].axis("off")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "xai_examples.png", dpi=150)
    plt.close()
    print(f"XAI report written to {out_dir}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ubc-dir", default=None, help="folder containing patches/patch_manifest.csv")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--n-synthetic-slides", type=int, default=75)
    ap.add_argument("--task", choices=["subtype", "hgsc_vs_rest"], default="subtype")
    ap.add_argument("--out", default="results_histo")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--max-patches-per-slide", type=int, default=16)
    ap.add_argument("--no-filtered", action="store_true", help="do not restrict to patches_filtered/")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--consist-weight", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--robust-per-slide", type=int, default=6)
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="test conditions (default: clean clean_sub bright0.8 wb hscale1.3 escale0.7 blur jpeg30 noise). "
                         "'blur0.7' = blur with sigma 0.7. 'clean' is always included.")
    ap.add_argument("--workers", type=int, default=1, help="processes for radiomics")
    ap.add_argument("--dl-workers", type=int, default=0, help="DataLoader workers (keep 0 on Windows)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the folds already finished in <out>/results_partial.json")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--no-xai", action="store_true")
    ap.add_argument("--xai-n", type=int, default=30)
    ap.add_argument("--xai-pool", type=int, default=600)
    ap.add_argument("--perm-repeats", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.synthetic:
        data = hu.make_synthetic_histo(args.n_synthetic_slides, 6, args.img_size, args.seed)
        if args.task == "hgsc_vs_rest":
            data["y"] = (data["y"] == 0).astype(int)
            data["classes"] = ["Other", "HGSC"]
    else:
        if not args.ubc_dir:
            raise SystemExit("give --ubc-dir (or --synthetic)")
        data = hu.load_ubc_patches(args.ubc_dir, args.max_patches_per_slide, args.seed,
                                   not args.no_filtered, args.img_size, args.task)
    classes = data["classes"]
    us, first = np.unique(data["slides"], return_index=True)
    print(f"{len(data['y'])} patches from {len(us)} slides")
    print("patches per class:", {c: int((data['y'] == i).sum()) for i, c in enumerate(classes)})
    print("slides per class :", {c: int((data['y'][first] == i).sum()) for i, c in enumerate(classes)})
    print("Computing radiomics (cached after the first run)...")
    rad_all = hu.cached_radiomics(data["X"], out / "radiomics_cache.npz", args.workers)
    if os.environ.get("NO_RADIOMICS") == "1":                      # NO_RADIOMICS
        rad_all = np.zeros_like(rad_all)
        print("radiomic branch disabled: all-zero radiomic input", flush=True)
    rad_names = hu.radiomic_names()

    variants = {v: VARIANTS[v] for v in args.variants}
    trainer = HistoTrainer(args, len(classes))
    xai_variant = "E_full" if "E_full" in variants else list(variants)[-1]

    def hook(fold, models, d):
        for _v, _m in models.items():
            _n = getattr(_m, "normaliser", None)
            if _n is not None:
                print(f"    [{_v}] normaliser fallback so far: {_n.n_fallback}/{_n.n_total} images", flush=True)
        if fold == 0 and not args.no_xai:
            explain(trainer, models[xai_variant], d, out, args, rad_names, classes)

    conds = args.conditions or list(hu.ROBUST_CONDITIONS)
    if "clean" not in conds:
        conds = ["clean"] + conds
    results, cms = hu.run_cv_histo(data, rad_all, variants, trainer, k=args.folds, seed=args.seed,
                                   val_frac=args.val_frac, robust_per_slide=args.robust_per_slide,
                                   workers=args.workers, out_dir=out, fold_hook=hook, conditions=conds,
                                   resume=args.resume)
    json.dump(results, open(out / "results.json", "w"), indent=1)
    partial_cm = (out / "confusion_partial.txt").exists()       # resumed from older code: matrices incomplete
    if not partial_cm:
        json.dump({v: cm.tolist() for v, cm in cms.items()}, open(out / "confusion.json", "w"))
    df = hu.summarize_histo(results)
    df.to_csv(out / "summary_slide_level.csv", index=False)
    hu.summarize_histo(results, "patch").to_csv(out / "summary_patch_level.csv", index=False)
    md = hu.to_markdown_histo(df, None if partial_cm else cms, classes)
    (out / "tables.md").write_text(md, encoding="utf-8")
    hu.plot_robustness_histo(df, out / "robustness.png")
    print("\n" + md)
    print(f"\nAll outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
