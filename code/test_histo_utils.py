"""Tests for histo_utils.py (no torch needed). Run: python test_histo_utils.py"""
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression

import histo_utils as hu


class StubTrainer:
    """Logistic regression on downsampled pixels + radiomics; stands in for the CNN."""
    def feats(self, X, R):
        small = np.stack([cv2.resize(x, (12, 12)) for x in X]).reshape(len(X), -1) / 255.0
        return np.hstack([small, R])

    def fit(self, cfg, tr, va, seed):
        return LogisticRegression(max_iter=3000).fit(self.feats(tr["X"], tr["R"]), tr["y"])

    def predict(self, model, X, R):
        P = np.zeros((len(X), 5))
        P[:, model.classes_] = model.predict_proba(self.feats(X, R))
        return P


def stain_layer_np(img_uint8, ref=(1.97, 1.03, 0.5), q=0.99):
    """numpy mirror of StainODLayer's maths (white point -> OD -> unmix -> normalise -> re-synthesise)."""
    x = img_uint8.astype(np.float64) / 255.0
    w = np.maximum(np.quantile(x.reshape(-1, 3), q, axis=0), 0.3)
    od = -np.log(np.maximum(x, 1e-3) / w)
    c = np.maximum(od @ hu.STAIN_MINV, 0)
    cq = np.maximum(np.quantile(c.reshape(-1, 3), q, axis=0), 1e-3)
    c = c / cq * np.asarray(ref)
    return np.exp(-(c @ hu.STAIN_M))


def test_stain_roundtrip():
    d = hu.make_synthetic_histo(5, 1, 64, 0)
    img = d["X"][0]
    assert np.abs(hu.stain_scale_rgb(img, [1, 1, 1]).astype(int) - img).max() <= 2
    assert not np.array_equal(hu.stain_scale_rgb(img, [1.5, 1, 1]), img)
    print("stain roundtrip ok")


def test_layer_invariance_maths():
    """The normalisation used by StainODLayer removes white-balance gain and stain-intensity scaling."""
    d = hu.make_synthetic_histo(5, 1, 96, 1)
    rng = np.random.default_rng(0)
    base = d["X"][2]
    ref_out = stain_layer_np(base)
    for kind in ["wb", "hscale1.3", "escale0.7"]:
        pert = hu.np_perturb_rgb(base, kind, rng)
        diff_layer = np.abs(stain_layer_np(pert) - ref_out).mean()
        diff_raw = np.abs(pert.astype(float) - base.astype(float)).mean() / 255.0
        print(f"   {kind}: mean |diff| raw={diff_raw:.4f}  after layer={diff_layer:.4f}")
        assert diff_layer < 0.7 * diff_raw, kind          # reduced (uint8 rounding / clipping leave a floor)
    print("layer invariance maths ok")


def test_slide_folds_no_leakage():
    d = hu.make_synthetic_histo(50, 4, 32, 0)
    folds = hu.slide_folds(d["y"], d["slides"], 5, 0)
    seen = set()
    for tr, te in folds:
        assert not set(d["slides"][tr]) & set(d["slides"][te])
        seen |= set(te)
        tr_in, va = hu.slide_inner_split(tr, d["y"], d["slides"], 0.2, 0)
        assert not set(d["slides"][tr_in]) & set(d["slides"][va])
        assert len(np.unique(d["y"][va])) == 5
    assert len(seen) == len(d["y"])
    print("slide folds ok")


def test_metrics():
    y = np.array([0, 1, 2, 3, 4, 0])
    P = np.eye(5)[y] * 0.875 + 0.025                      # rows sum to exactly 1
    m = hu.metrics_mc(y, P)
    assert m["accuracy"] == 1 and m["macro_f1"] == 1 and m["auc"] == 1.0
    wrong = np.eye(5)[(y + 1) % 5] * 0.95 + 0.01          # confidently wrong, rows sum to 1
    assert hu.metrics_mc(y, wrong)["ece"] > 0.9
    slides = np.array(["a", "a", "b", "b", "c", "c"])
    P2 = np.array([[1, 0], [0.5, 0.5], [0, 1], [0.2, 0.8], [0.3, 0.7], [0.1, 0.9]], float)
    Ps, ys, us = hu.slide_aggregate(P2, slides, np.array([0, 0, 1, 1, 1, 1]))
    assert np.allclose(Ps[0], [0.75, 0.25]) and list(ys) == [0, 1, 1] and list(us) == ["a", "b", "c"]
    assert hu.metrics_mc(np.array([0, 1, 1]), Ps)["auc"] == 1.0          # binary path
    print("metrics ok")


def test_faithfulness_and_perm():
    rng = np.random.default_rng(0)
    H = W = 32
    tpl = np.zeros((H, W), np.float32)
    tpl[8:16, 8:16] = 1
    img = np.stack([0.3 + 0.5 * tpl] * 3, -1)
    prob_fn = lambda b, r: 1 / (1 + np.exp(-(10 * ((b[..., 0] * tpl).sum((1, 2)) / tpl.sum() - 0.5))))
    d_g, i_g = hu.faithfulness_curves_rgb(prob_fn, img, tpl + 0.01 * rng.random((H, W)), np.zeros(3))
    d_r, i_r = hu.faithfulness_curves_rgb(prob_fn, img, rng.random((H, W)), np.zeros(3))
    assert hu.nu.curve_auc(d_g) < hu.nu.curve_auc(d_r) and hu.nu.curve_auc(i_g) > hu.nu.curve_auc(i_r)
    rad = rng.normal(size=(300, 4)).astype(np.float32)
    y = (rad[:, 1] > 0).astype(int)
    fn = lambda r: np.stack([1 / (1 + np.exp(5 * r[:, 1])), 1 / (1 + np.exp(-5 * r[:, 1]))], 1)
    base, imp = hu.permutation_importance_mc(fn, rad, y, list("abcd"), repeats=3)
    assert imp[0][0] == "b" and base > 0.95
    print("faithfulness / permutation importance ok")


def test_loader_paths_and_filter():
    d = hu.make_synthetic_histo(10, 3, 64, 0)
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        rows, k = [], 0
        for s in sorted(set(d["slides"])):
            idx = np.where(d["slides"] == s)[0]
            lab = hu.CLASSES[int(d["y"][idx[0]])]
            (root / "patches" / lab).mkdir(parents=True, exist_ok=True)
            for j, i in enumerate(idx):
                p = root / "patches" / lab / f"{s}_tile_{j:04d}.png"
                cv2.imwrite(str(p), d["X"][i][:, :, ::-1])
                # absolute path for some rows, a stale foreign path for others (must fall back)
                rows.append((str(p) if k % 2 == 0 else "Z:\\gone\\patches\\" + lab + "\\" + p.name, s, lab))
                k += 1
        with open(root / "patches" / "patch_manifest.csv", "w", newline="") as f:
            f.write("patch_path,source_image_id,label\n" + "\n".join(",".join(r) for r in rows) + "\n")
        (root / "patches_filtered").mkdir()
        for r in rows[::2]:                                           # keep half of the files in 'filtered'
            os.makedirs(root / "patches_filtered", exist_ok=True)
            b = r[0].replace("\\", "/").split("/")[-1]
            (root / "patches_filtered" / b).write_bytes(b"x")
        a = hu.load_ubc_patches(str(root), 2, 0, use_filtered=False, size=64)
        assert len(a["y"]) <= 20 and a["X"].shape[1:] == (64, 64, 3)
        assert max(np.unique(a["slides"], return_counts=True)[1]) <= 2
        b = hu.load_ubc_patches(str(root), 99, 0, use_filtered=True, size=64)
        assert 0 < len(b["y"]) < 30
        h = hu.load_ubc_patches(str(root), 99, 0, use_filtered=False, size=64, task="hgsc_vs_rest")
        assert set(h["y"]) <= {0, 1} and h["classes"] == ["Other", "HGSC"]
    print("loader ok")


def test_loader_only_filtered_odd_nesting():
    """Colab layout: no images under patches/, only patches_filtered/ with arbitrary nesting,
    manifest paths come from another machine."""
    d = hu.make_synthetic_histo(10, 3, 64, 0)
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        rows = []
        sub = root / "patches_filtered" / "part_1" / "deeper"
        sub.mkdir(parents=True)
        for s in sorted(set(d["slides"])):
            idx = np.where(d["slides"] == s)[0]
            lab = hu.CLASSES[int(d["y"][idx[0]])]
            for j, i in enumerate(idx):
                name = f"{s}_tile_{j:04d}.png"
                cv2.imwrite(str(sub / name), d["X"][i][:, :, ::-1])
                rows.append(("D:\\x\\patches\\" + lab + "\\" + name, s, lab))
        (root / "patches").mkdir()
        with open(root / "patches" / "patch_manifest.csv", "w", newline="") as f:
            f.write("patch_path,source_image_id,label\n" + "\n".join(",".join(r) for r in rows) + "\n")
        a = hu.load_ubc_patches(str(root), 99, 0, use_filtered=True, size=64)
        assert len(a["y"]) == 30 and a["X"].shape == (30, 64, 64, 3)
    print("loader (filtered-only, odd nesting) ok")


def test_bootstrap():
    import pandas as pd
    rng = np.random.default_rng(0)
    n, K = 200, 5
    y = rng.integers(0, K, n)

    def probs(acc):
        pred = np.where(rng.random(n) < acc, y, rng.integers(0, K, n))
        P = np.full((n, K), 0.02)
        P[np.arange(n), pred] = 0.92
        return P

    rows = []
    for v, acc in [("good", 0.9), ("bad", 0.5)]:
        P = probs(acc)
        rows += [[v, 0, f"s{i:04d}", int(y[i])] + list(P[i]) for i in range(n)]
    df = pd.DataFrame(rows, columns=["variant", "fold", "slide", "y"] + [f"p{k}" for k in range(K)])
    c = hu.bootstrap_compare(df, "good", "bad", n_boot=500)
    assert c["delta"] > 0.2 and c["lo"] > 0 and c["p_boot"] < 0.05 and c["n_slides"] == n
    same = hu.bootstrap_compare(df, "good", "good", n_boot=300)
    assert same["delta"] == 0 and same["lo"] <= 0 <= same["hi"] and same["p_boot"] == 1.0
    b = hu.bootstrap_variant(df, "good", n_boot=300)
    assert b["lo"] <= b["estimate"] <= b["hi"] and b["estimate"] > 0.8
    md = hu.bootstrap_table_md(df, ["good", "bad"], "bad", n_boot=200)
    assert "good" in md and "bootstrap p" in md
    print("bootstrap ok")


def test_tolerant_tables_and_blur_param():
    import pandas as pd
    rows = []
    for v, conds in [("A", ["clean", "clean_sub", "blur"]), ("B", ["clean", "clean_sub", "blur", "blur0.7"])]:
        for c in conds:
            r = {"variant": v, "condition": c}
            for k in hu.METRIC_KEYS:
                r[k + "_mean"], r[k + "_std"] = 0.5, 0.01
            rows.append(r)
    df = pd.DataFrame(rows)
    assert "–" in hu.to_markdown_histo(df)                     # variant A has no blur0.7 column value
    with tempfile.TemporaryDirectory() as t:
        hu.plot_robustness_histo(df, Path(t) / "p.png")
        assert (Path(t) / "p.png").exists()
    img = hu.make_synthetic_histo(5, 1, 64, 0)["X"][0]
    rng = np.random.default_rng(0)
    soft, hard, dflt = (hu.np_perturb_rgb(img, k, rng) for k in ["blur0.5", "blur2.0", "blur"])
    assert np.abs(soft.astype(int) - img).mean() < np.abs(hard.astype(int) - img).mean()
    assert np.array_equal(dflt, cv2.GaussianBlur(img, (0, 0), 1.5))     # original behaviour unchanged
    print("tolerant tables / blur parameter ok")


def test_resume_after_interruption():
    import json
    d = hu.make_synthetic_histo(50, 4, 64, 0)
    rad = hu.radiomics_rgb(d["X"], 1)
    variants = {"A_stub": {}, "B_stub": {}}
    conds = ["clean", "clean_sub", "blur0.7"]
    kw = dict(k=4, seed=0, val_frac=0.25, conditions=conds, robust_per_slide=2, verbose=False)

    class Boom(StubTrainer):
        def __init__(self):
            self.n = 0

        def fit(self, cfg, tr, va, seed):
            self.n += 1
            if self.n > 2 * 2:                       # 2 variants x 2 folds finished -> die in fold 3
                raise RuntimeError("simulated crash")
            return super().fit(cfg, tr, va, seed)

    with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
        ref, ref_cms = hu.run_cv_histo(d, rad, variants, StubTrainer(), out_dir=t1, **kw)
        try:
            hu.run_cv_histo(d, rad, variants, Boom(), out_dir=t2, **kw)
            assert False, "should have crashed"
        except RuntimeError:
            pass
        assert len(json.load(open(Path(t2) / "results_partial.json"))["A_stub"]["clean"]) == 2
        res, cms = hu.run_cv_histo(d, rad, variants, StubTrainer(), out_dir=t2, resume=True, **kw)
        for v in variants:
            for c in conds:
                assert json.dumps(res[v][c]) == json.dumps(ref[v][c]), (v, c)     # identical to an uninterrupted run
            assert (cms[v] == ref_cms[v]).all()
        assert len(hu.load_slide_predictions(Path(t2) / "slide_predictions.csv")) == 100
        assert not (Path(t2) / "confusion_partial.txt").exists()
        # different settings must NOT be resumed silently
        res2, _ = hu.run_cv_histo(d, rad, variants, StubTrainer(), out_dir=t2, resume=True, **{**kw, "seed": 1})
        assert len(res2["A_stub"]["clean"]) == 4

        # partial results written by OLD code (no run_meta.json, no slide_predictions.csv): resume, flag the matrices
        old = Path(t2) / "old"
        old.mkdir()
        part = json.load(open(Path(t1) / "results_partial.json"))
        for v in part:
            for c in part[v]:
                part[v][c] = part[v][c][:2]
        json.dump(part, open(old / "results_partial.json", "w"))
        res3, _ = hu.run_cv_histo(d, rad, variants, StubTrainer(), out_dir=old, resume=True, **kw)
        assert len(res3["A_stub"]["clean"]) == 4 and (old / "confusion_partial.txt").exists()
        assert json.dumps(res3["B_stub"]["clean"]) == json.dumps(ref["B_stub"]["clean"])
    print("resume after interruption ok")


def test_radiomics_parallel_and_cache():
    d = hu.make_synthetic_histo(10, 8, 64, 0)
    X = d["X"][:80]
    r1 = hu.radiomics_rgb(X, 1)
    r2 = hu.radiomics_rgb(X, 2)
    assert r1.shape[1] == len(hu.radiomic_names()) and np.allclose(r1, r2, atol=1e-4)
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "c.npz"
        a = hu.cached_radiomics(X, p, 1)
        assert p.exists() and np.allclose(a, hu.cached_radiomics(X, p, 1))
    print("radiomics parallel/cache ok")


def test_run_cv_end_to_end():
    d = hu.make_synthetic_histo(50, 4, 64, 0)
    rad = hu.radiomics_rgb(d["X"], 1)
    with tempfile.TemporaryDirectory() as t:
        res, cms = hu.run_cv_histo(d, rad, {"A_stub": {}, "B_stub": {}}, StubTrainer(), k=3, seed=0, val_frac=0.25,
                                   conditions=["clean", "clean_sub", "hscale1.3", "blur", "jpeg30"],
                                   robust_per_slide=2, out_dir=t, verbose=False)
        df = hu.summarize_histo(res)
        assert len(df) == 2 * 5 and df.balanced_acc_mean.between(0, 1).all()
        md = hu.to_markdown_histo(df, cms, d["classes"])
        assert "A_stub" in md and "hscale1.3" in md and "HGSC" in md
        hu.plot_robustness_histo(df, Path(t) / "r.png")
        assert (Path(t) / "r.png").exists() and cms["A_stub"].sum() == 50
        sp = hu.load_slide_predictions(Path(t) / "slide_predictions.csv")     # saved out-of-fold predictions
        assert len(sp) == 100 and set(sp.variant) == {"A_stub", "B_stub"} and sp.slide.nunique() == 50
        assert "bootstrap p" in hu.bootstrap_table_md(sp, ["A_stub", "B_stub"], "A_stub", n_boot=200)
    print("run_cv_histo end-to-end ok")
    print(md)


if __name__ == "__main__":
    test_stain_roundtrip()
    test_layer_invariance_maths()
    test_slide_folds_no_leakage()
    test_metrics()
    test_faithfulness_and_perm()
    test_loader_paths_and_filter()
    test_loader_only_filtered_odd_nesting()
    test_bootstrap()
    test_tolerant_tables_and_blur_param()
    test_resume_after_interruption()
    test_radiomics_parallel_and_cache()
    test_run_cv_end_to_end()
    print("\nALL TESTS PASSED")
