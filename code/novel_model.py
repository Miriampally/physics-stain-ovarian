#!/usr/bin/env python3
"""
novel_model.py - the novel contributions, trained end to end in PyTorch.

Novel components (each can be switched on/off -> ablation study)
  1. PhysicsLayer      : learnable flat-field Beer-Lambert layer. The incident-intensity field I0 is
                         estimated as a Gaussian-smoothed copy of the image whose scales (sigma) are
                         LEARNED; optical density OD = -ln(I / I0) is computed at several scales and
                         mixed by a 1x1 conv. OD is exactly invariant to a global gain I -> g*I and
                         approximately invariant to smooth multiplicative illumination fields.
  2. Consistency loss  : KL(p(clean) || p(illumination-perturbed)) so the whole network (not only the
                         first layer) is invariant to physically plausible illumination changes.
  3. Cross-attention fusion : the radiomic vector is the query, the EfficientNet spatial tokens are
                         keys/values, so texture statistics decide WHERE to look; a learned gate mixes
                         attended and pooled evidence. Replaces concatenation + PCA.
  4. Explainability    : Grad-CAM on the end-to-end network, the fusion attention map, a quantitative
                         faithfulness test (deletion / insertion), Grad-CAM-vs-attention agreement,
                         and permutation importance of the radiomic features.

Variants (ablation)
  A_baseline         raw gray input, concat fusion, no consistency loss
  B_physics          + PhysicsLayer
  C_physics+consist  + illumination consistency loss
  D_attn_only        raw gray input, attention fusion
  E_full             PhysicsLayer + consistency + attention fusion
  F_consist_only     raw gray input, concat fusion, consistency loss

Usage
    python novel_model.py --data-dir data --groups-csv patients.csv --out results_novel
    (patients.csv: columns filename,patient_id -> folds never split a patient)
Needs: torch, torchvision, plus the packages in requirements.txt. A GPU is strongly recommended.
"""
import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

import novelty_utils as nu
import ovarian_pipeline as op

VARIANTS = {
    "A_baseline":        dict(physics=False, fusion="concat", consist=False),
    "B_physics":         dict(physics=True,  fusion="concat", consist=False),
    "C_physics+consist": dict(physics=True,  fusion="concat", consist=True),
    "D_attn_only":       dict(physics=False, fusion="attn",   consist=False),
    "E_full":            dict(physics=True,  fusion="attn",   consist=True),
    "F_consist_only":    dict(physics=False, fusion="concat", consist=True),
}


# --------------------------------------------------------------------------- #
# 1. Learnable physics layer
# --------------------------------------------------------------------------- #
class PhysicsLayer(nn.Module):
    """x: (B,1,H,W) intensity in [0,1] -> (B,3,H,W) multi-scale optical density, learned mixing."""

    def __init__(self, sigmas=(4.0, 10.0, 20.0), ksize=65, eps=1e-3):
        super().__init__()
        c = len(sigmas)
        # sigma = softplus(raw) + 0.5  (keeps sigma positive, learnable)
        self.raw_sigma = nn.Parameter(torch.log(torch.expm1(torch.tensor(sigmas) - 0.5)))
        self.mix = nn.Conv2d(c, 3, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(3)
        self.ksize, self.eps = ksize, eps
        self.register_buffer("r", torch.arange(ksize, dtype=torch.float32) - ksize // 2)

    def sigmas(self):
        return F.softplus(self.raw_sigma) + 0.5

    def forward(self, x):
        c = self.raw_sigma.numel()
        s = self.sigmas().view(c, 1)
        k = torch.exp(-(self.r.view(1, -1) ** 2) / (2 * s ** 2))
        k = k / k.sum(1, keepdim=True)                                   # (C,K) normalised kernels
        pad = self.ksize // 2
        xc = x.repeat(1, c, 1, 1)
        bg = F.conv2d(F.pad(xc, (pad, pad, 0, 0), mode="reflect"), k.view(c, 1, 1, -1), groups=c)
        bg = F.conv2d(F.pad(bg, (0, 0, pad, pad), mode="reflect"), k.view(c, 1, -1, 1), groups=c)  # I0 field
        od = -torch.log((xc + self.eps) / (bg + self.eps))               # Beer-Lambert: mu*x = -ln(I/I0)
        return self.bn(self.mix(od))


class GrayRepeat(nn.Module):
    def forward(self, x):
        return ((x - 0.45) / 0.225).repeat(1, 3, 1, 1)


# --------------------------------------------------------------------------- #
# 3. Fusion modules
# --------------------------------------------------------------------------- #
class ConcatFusion(nn.Module):
    def __init__(self, c_feat, n_rad, d, n_cls=2, p=0.3):
        super().__init__()
        self.pd = nn.Sequential(nn.Linear(c_feat, d), nn.GELU())
        self.pr = nn.Sequential(nn.Linear(n_rad, d), nn.GELU())
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(2 * d, n_cls))

    def forward(self, tokens, pooled, rad):
        return self.head(torch.cat([self.pd(pooled), self.pr(rad)], 1)), None


class CrossAttnFusion(nn.Module):
    def __init__(self, c_feat, n_rad, d, n_cls=2, heads=4, p=0.3):
        super().__init__()
        self.pt = nn.Linear(c_feat, d)
        self.ln_t = nn.LayerNorm(d)
        self.pr = nn.Sequential(nn.Linear(n_rad, d), nn.GELU(), nn.LayerNorm(d))
        self.attn = nn.MultiheadAttention(d, heads, dropout=0.1, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(2 * d, d), nn.Sigmoid())
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(2 * d, n_cls))

    def forward(self, tokens, pooled, rad):
        t = self.ln_t(self.pt(tokens))                                   # (B,N,d) spatial tokens
        r = self.pr(rad)                                                 # (B,d)   radiomic query
        ctx, w = self.attn(r.unsqueeze(1), t, t, need_weights=True, average_attn_weights=True)
        ctx, w = ctx.squeeze(1), w.squeeze(1)                            # (B,d), (B,N)
        tm = t.mean(1)
        g = self.gate(torch.cat([tm, r], 1))
        fused = g * ctx + (1 - g) * tm
        return self.head(torch.cat([fused, r], 1)), w


class HybridNet(nn.Module):
    def __init__(self, n_rad, physics, fusion, pretrained=True, d=128):
        super().__init__()
        self.pre = PhysicsLayer() if physics else GrayRepeat()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.features = efficientnet_b0(weights=weights).features
        self.fusion = (CrossAttnFusion if fusion == "attn" else ConcatFusion)(1280, n_rad, d)

    def encode(self, x):
        return self.features(self.pre(x))                                # (B,1280,h,w)

    def fuse(self, fmap, rad):
        tokens = fmap.flatten(2).transpose(1, 2)
        return self.fusion(tokens, fmap.mean((2, 3)), rad)               # logits, attention or None

    def forward(self, x, rad):
        fmap = self.encode(x)
        logits, attn = self.fuse(fmap, rad)
        return logits, attn, fmap


# --------------------------------------------------------------------------- #
# 2. Illumination perturbation + training
# --------------------------------------------------------------------------- #
def illumination_perturb(x, g_range=(0.5, 1.0), grad_strength=0.3):
    """Physically plausible nuisance: global gain x smooth planar multiplicative field. x in [0,1]."""
    b, _, h, w = x.shape
    dev = x.device
    g = torch.empty(b, 1, 1, 1, device=dev).uniform_(*g_range)
    yy = torch.linspace(-1, 1, h, device=dev).view(1, 1, h, 1)
    xx = torch.linspace(-1, 1, w, device=dev).view(1, 1, 1, w)
    a = torch.empty(b, 1, 1, 1, device=dev).uniform_(-grad_strength, grad_strength)
    c = torch.empty(b, 1, 1, 1, device=dev).uniform_(-grad_strength, grad_strength)
    return (x * g * (1 + a * yy + c * xx)).clamp(0, 1)


class ArrayDS(Dataset):
    def __init__(self, imgs, rad, y, train, seed):
        self.imgs, self.rad, self.y, self.train = imgs, rad, y, train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        im = op.augment(self.imgs[i], self.rng) if self.train else self.imgs[i]
        x = torch.from_numpy(im.astype(np.float32) / 255.0)[None]
        return x, torch.from_numpy(self.rad[i]), int(self.y[i])


class TorchTrainer:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @torch.no_grad()
    def predict_float(self, model, X01, R, bs=64):
        """X01: (N,H,W) float in [0,1]."""
        model.eval()
        out = []
        for i in range(0, len(X01), bs):
            x = torch.from_numpy(np.asarray(X01[i:i + bs], np.float32))[:, None].to(self.device)
            r = torch.from_numpy(np.asarray(R[i:i + bs], np.float32)).to(self.device)
            out.append(F.softmax(model(x, r)[0], 1)[:, 1].cpu().numpy())
        return np.concatenate(out)

    def predict(self, model, X, R):
        return self.predict_float(model, X.astype(np.float32) / 255.0, R)

    def fit(self, cfg, Xtr, Rtr, ytr, Xva, Rva, yva, seed):
        a, dev = self.args, self.device
        torch.manual_seed(seed)
        model = HybridNet(Rtr.shape[1], cfg["physics"], cfg["fusion"], not a.no_pretrained).to(dev)
        backbone = list(model.features.parameters())
        rest = [p for n, p in model.named_parameters() if not n.startswith("features.")]
        opt = torch.optim.AdamW([{"params": backbone, "lr": a.lr_backbone},
                                 {"params": rest, "lr": a.lr_head}], weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
        cnt = np.bincount(ytr, minlength=2)
        cw = torch.tensor(len(ytr) / (2.0 * cnt), dtype=torch.float32, device=dev)   # class-balanced CE
        loader = DataLoader(ArrayDS(Xtr, Rtr, ytr, True, seed), batch_size=a.batch_size,
                            shuffle=True, drop_last=len(ytr) > a.batch_size)
        best_auc, best_state, bad = -1.0, None, 0
        for ep in range(a.epochs):
            model.train()
            for x, r, y in loader:
                x, r, y = x.to(dev), r.to(dev), y.to(dev)
                logits, _, _ = model(x, r)
                loss = F.cross_entropy(logits, y, weight=cw)
                if cfg["consist"]:
                    lp, _, _ = model(illumination_perturb(x), r)
                    loss = loss + a.consist_weight * F.kl_div(
                        F.log_softmax(lp, 1), F.softmax(logits.detach(), 1), reduction="batchmean")
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
            auc = roc_auc_score(yva, self.predict(model, Xva, Rva))
            if auc > best_auc:
                best_auc, best_state, bad = auc, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= a.patience:
                    break
        model.load_state_dict(best_state)
        return model.eval()


# --------------------------------------------------------------------------- #
# 4. Explainability
# --------------------------------------------------------------------------- #
def explain(trainer, model, data, out_dir, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dev = trainer.device
    imgs, rad, y = data["imgs_te"], data["rad_te"], data["y_te"]
    p_all = trainer.predict(model, imgs, rad)
    sel = np.where(p_all >= 0.5)[0][:args.xai_n]          # explain the 'malignant' evidence
    if len(sel) == 0:
        print("XAI: no image predicted malignant on this fold, skipping")
        return
    model.eval()
    gc_maps, att_maps = [], []
    for i in sel:
        x = torch.from_numpy(imgs[i].astype(np.float32) / 255.0)[None, None].to(dev)
        r = torch.from_numpy(rad[i:i + 1]).to(dev)
        with torch.enable_grad():
            fmap = model.encode(x)
            logits, attn = model.fuse(fmap, r)
            grads, = torch.autograd.grad(logits[0, 1], fmap)
        h, w = fmap.shape[2:]
        cam = F.relu((grads.mean((2, 3), keepdim=True) * fmap).sum(1))[0].detach().cpu().numpy()
        gc_maps.append(cam)
        att_maps.append(None if attn is None else attn[0].detach().cpu().numpy().reshape(h, w))

    H, W = imgs.shape[1:]
    up = lambda m: cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    rng = np.random.default_rng(0)
    sources = {"gradcam": [up(m) for m in gc_maps], "random": [rng.random((H, W)) for _ in sel]}
    if att_maps[0] is not None:
        sources["attention"] = [up(m) for m in att_maps]

    report = {"n_images": int(len(sel)), "faithfulness": {}}
    for name, sals in sources.items():
        dels, inss = [], []
        for i, sal in zip(sel, sals):
            def prob_fn(batch, rd):
                return trainer.predict_float(model, batch, rd)
            d, s = nu.faithfulness_curves(prob_fn, imgs[i].astype(np.float32) / 255.0, sal, rad[i])
            dels.append(nu.curve_auc(d))
            inss.append(nu.curve_auc(s))
        report["faithfulness"][name] = {
            "deletion_auc_mean(lower=better)": float(np.mean(dels)), "deletion_auc_std": float(np.std(dels)),
            "insertion_auc_mean(higher=better)": float(np.mean(inss)), "insertion_auc_std": float(np.std(inss))}
    if att_maps[0] is not None:
        rhos = [spearmanr(g.ravel(), a.ravel())[0] for g, a in zip(gc_maps, att_maps)]
        report["gradcam_attention_spearman_mean"] = float(np.nanmean(rhos))

    # radiomic permutation importance (backbone features cached; only the fusion head is re-run)
    with torch.no_grad():
        fmaps = torch.cat([model.encode(torch.from_numpy(imgs[i:i + 32].astype(np.float32) / 255.0)[:, None].to(dev)).cpu()
                           for i in range(0, len(imgs), 32)])

    def predict_rad(rd):
        out = []
        with torch.no_grad():
            for i in range(0, len(fmaps), 64):
                lg, _ = model.fuse(fmaps[i:i + 64].to(dev), torch.from_numpy(rd[i:i + 64]).to(dev))
                out.append(F.softmax(lg, 1)[:, 1].cpu().numpy())
        return np.concatenate(out)

    base, imp = nu.permutation_importance(predict_rad, rad, y, op.RADIOMIC_NAMES, repeats=args.perm_repeats)
    report["radiomic_permutation_importance"] = {"base_auc": base, "top10_auc_drop": imp[:10]}
    if hasattr(model.pre, "sigmas"):
        report["learned_physics_sigmas"] = [float(s) for s in model.pre.sigmas().detach().cpu()]
    json.dump(report, open(Path(out_dir) / "xai.json", "w"), indent=2)

    # figure
    n = min(6, len(sel))
    cols = 3 if att_maps[0] is not None else 2
    fig, ax = plt.subplots(n, cols, figsize=(3 * cols, 3 * n), squeeze=False)
    for row in range(n):
        base_rgb = cv2.cvtColor(imgs[sel[row]], cv2.COLOR_GRAY2RGB)
        ax[row, 0].imshow(imgs[sel[row]], cmap="gray")
        for col, key in enumerate(["gradcam", "attention"][:cols - 1], start=1):
            s = sources[key][row]
            s = (s - s.min()) / (s.max() - s.min() + 1e-8)
            heat = cv2.applyColorMap((s * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
            ax[row, col].imshow((0.55 * base_rgb + 0.45 * heat).astype(np.uint8))
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
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--groups-csv", default=None, help="CSV with columns filename,patient_id (recommended)")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--n-synthetic", type=int, default=300)
    ap.add_argument("--out", default="results_novel")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--consist-weight", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--no-xai", action="store_true")
    ap.add_argument("--xai-n", type=int, default=40)
    ap.add_argument("--perm-repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.synthetic:
        imgs, y = op.make_synthetic_dataset(args.n_synthetic, args.img_size, args.seed)
        groups = None
    else:
        imgs, y, _, groups = nu.load_dataset(args.data_dir, args.img_size, args.groups_csv)
    print(f"{len(y)} images (benign={np.sum(y == 0)}, malignant={np.sum(y == 1)}); "
          f"group-aware CV: {groups is not None}")
    print("Computing radiomics...")
    rad_all = nu.radiomics_all(imgs)

    variants = {v: VARIANTS[v] for v in args.variants}
    trainer = TorchTrainer(args)
    xai_variant = "E_full" if "E_full" in variants else list(variants)[-1]

    def hook(fold, models, data):
        if fold == 0 and not args.no_xai:
            explain(trainer, models[xai_variant], data, out, args)

    results = nu.run_cv(imgs, y, groups, rad_all, variants, trainer, k=args.folds, seed=args.seed,
                        val_frac=args.val_frac, out_dir=out, fold_hook=hook)
    json.dump(results, open(out / "results.json", "w"), indent=1)
    df = nu.summarize(results)
    df.to_csv(out / "summary.csv", index=False)
    md = nu.to_markdown(df)
    (out / "tables.md").write_text(md)
    nu.plot_robustness(df, out / "robustness.png")
    print("\n" + md)
    print(f"\nAll outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
