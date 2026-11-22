"""Tests for novelty_utils.py using a sklearn stub trainer (no torch needed).
Run: python test_novelty_utils.py"""
import tempfile
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

import novelty_utils as nu
import ovarian_pipeline as op


class StubTrainer:
    """Logistic regression on downsampled pixels + radiomics, standing in for the CNN."""
    def feats(self, X, R):
        small = np.stack([op.cv2.resize(x, (16, 16)) for x in X]).reshape(len(X), -1) / 255.0
        return np.hstack([small, R])

    def fit(self, cfg, Xtr, Rtr, ytr, Xva, Rva, yva, seed):
        return LogisticRegression(max_iter=2000).fit(self.feats(Xtr, Rtr), ytr)

    def predict(self, model, X, R):
        return model.predict_proba(self.feats(X, R))[:, 1]


def test_ece():
    y = np.array([1, 1, 0, 0])
    assert nu.ece_score(y, np.array([1.0, 1.0, 0.0, 0.0])) == 0.0
    assert nu.ece_score(y, np.array([0.0, 0.0, 1.0, 1.0])) > 0.99     # confidently wrong
    print("ece ok")


def test_perturb():
    im = (np.random.default_rng(0).random((32, 32)) * 200).astype(np.uint8)
    rng = np.random.default_rng(0)
    assert np.allclose(nu.np_perturb(im, "gain0.5", rng), im * 0.5, atol=1)
    assert nu.np_perturb(im, "clean", rng).dtype == np.uint8
    for k in nu.ROBUST_CONDITIONS:
        assert nu.np_perturb(im, k, rng).shape == im.shape
    print("perturb ok")


def test_physics_gain_invariance_numpy():
    """Numpy mirror of the PhysicsLayer idea: -ln(I/blur(I)) is invariant to a global gain."""
    import cv2
    im = np.random.default_rng(1).random((64, 64)).astype(np.float32) * 0.4 + 0.2
    f = lambda x: -np.log((x + 1e-3) / (cv2.GaussianBlur(x, (0, 0), 8) + 1e-3))
    assert np.abs(f(im) - f(0.6 * im)).max() < 5e-3
    print("gain invariance ok")


def test_faithfulness():
    rng = np.random.default_rng(0)
    H = W = 32
    template = np.zeros((H, W), np.float32)
    template[8:16, 8:16] = 1.0
    img = 0.3 + 0.5 * template                      # the bright patch is the "evidence"
    prob_fn = lambda b, r: 1 / (1 + np.exp(-(10 * ((b * template).sum((1, 2)) / template.sum() - 0.5))))
    sal_good = template + 0.01 * rng.random((H, W))
    sal_rand = rng.random((H, W))
    d_g, i_g = nu.faithfulness_curves(prob_fn, img, sal_good, np.zeros(3))
    d_r, i_r = nu.faithfulness_curves(prob_fn, img, sal_rand, np.zeros(3))
    assert nu.curve_auc(d_g) < nu.curve_auc(d_r)    # good map: deletion kills the score faster
    assert nu.curve_auc(i_g) > nu.curve_auc(i_r)    # good map: insertion restores it faster
    m = nu.faithfulness_masks(sal_rand, 10)
    assert m[0].sum() == 0 and m[-1].all() and (np.diff(m.sum((1, 2))) >= 0).all()
    print("faithfulness ok")


def test_permutation_importance():
    rng = np.random.default_rng(0)
    rad = rng.normal(size=(200, 5)).astype(np.float32)
    y = (rad[:, 2] > 0).astype(int)
    fn = lambda r: 1 / (1 + np.exp(-5 * r[:, 2]))
    base, imp = nu.permutation_importance(fn, rad, y, list("abcde"), repeats=3)
    assert imp[0][0] == "c" and base > 0.99
    print("permutation importance ok")


def test_run_cv_end_to_end():
    imgs, y = op.make_synthetic_dataset(120, 64, 0)
    rad = nu.radiomics_all(imgs)
    groups = np.array([str(i // 2) for i in range(len(y))], dtype=object)   # 2 images per "patient"
    folds = nu.make_folds(y, groups, 3, 0)
    for tr, te in folds:                                                    # no patient leakage
        assert not set(groups[tr]) & set(groups[te])
    variants = {"A_stub": dict(), "B_stub": dict()}
    with tempfile.TemporaryDirectory() as d:
        res = nu.run_cv(imgs, y, groups, rad, variants, StubTrainer(), k=3, seed=0,
                        conditions=["clean", "gain0.5", "noise"], out_dir=d, verbose=False)
        df = nu.summarize(res)
        assert len(df) == 2 * 3 and df.auc_mean.between(0, 1).all()
        md = nu.to_markdown(df)
        assert "A_stub" in md and "gain0.5" in md
        nu.plot_robustness(df, Path(d) / "rob.png")
        assert (Path(d) / "rob.png").exists() and (Path(d) / "results_partial.json").exists()
    print("run_cv end-to-end ok")
    print(md)


if __name__ == "__main__":
    test_ece(); test_perturb(); test_physics_gain_invariance_numpy()
    test_faithfulness(); test_permutation_importance(); test_run_cv_end_to_end()
    print("\nALL TESTS PASSED")
