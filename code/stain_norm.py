"""
stain_norm.py - classical H&E stain normalisation baselines (numpy + OpenCV only, no extra packages).

  ReinhardNorm : colour transfer in Lab space (Reinhard et al., 2001). Target = mean of the per-image
                 Lab means / stds of a sample of TRAINING patches (so nothing is learned from test slides).
  MacenkoNorm  : stain-vector estimation by SVD + angle percentiles, concentration rescaling and
                 re-synthesis with a fixed reference H&E matrix (Macenko et al., 2009).

Both are applied per image (no slide-level information), to uint8 RGB arrays of shape (N,H,W,3).
If an image cannot be normalised (too little tissue, degenerate stain estimate) it is returned unchanged
and counted in `.n_fallback`, so the fallback rate can be reported.
"""
import cv2
import numpy as np

# Reference stain matrix and maximum concentrations used by the common Macenko implementations
HE_REF = np.array([[0.5626, 0.2159],
                   [0.7201, 0.8012],
                   [0.4062, 0.5581]], dtype=np.float64)
MAXC_REF = np.array([1.9705, 1.0308], dtype=np.float64)


class ReinhardNorm:
    name = "reinhard"

    def __init__(self, target_mean=None, target_std=None, bg_l=230):
        self.mu_t, self.sd_t, self.bg_l = target_mean, target_std, bg_l
        self.n_fallback = 0
        self.n_total = 0

    def _stats(self, lab):
        L = lab[..., 0]
        m = L < self.bg_l                               # drop (near-)white background
        px = lab[m] if m.sum() > 100 else lab.reshape(-1, 3)
        return px.mean(0), px.std(0) + 1e-6

    def fit(self, X, n=400, seed=0):
        idx = np.random.default_rng(seed).choice(len(X), min(n, len(X)), replace=False)
        st = [self._stats(cv2.cvtColor(X[i], cv2.COLOR_RGB2LAB).astype(np.float32)) for i in idx]
        self.mu_t = np.mean([s[0] for s in st], 0)
        self.sd_t = np.mean([s[1] for s in st], 0)
        return self

    def one(self, img):
        self.n_total += 1
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
        mu, sd = self._stats(lab)
        out = (lab - mu) / sd * self.sd_t + self.mu_t
        out = np.clip(out, 0, 255).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_LAB2RGB)

    def __call__(self, X):
        return np.stack([self.one(im) for im in X])


class MacenkoNorm:
    name = "macenko"

    def __init__(self, io=255, alpha=1.0, beta=0.15, he_ref=HE_REF, maxc_ref=MAXC_REF):
        self.io, self.alpha, self.beta = io, alpha, beta
        self.he_ref, self.maxc_ref = he_ref, maxc_ref
        self.n_fallback = 0
        self.n_total = 0

    def fit(self, X, **kw):                              # fixed reference, nothing to fit
        return self

    def one(self, img):
        self.n_total += 1
        h, w, _ = img.shape
        od = -np.log((img.reshape(-1, 3).astype(np.float64) + 1.0) / self.io)
        keep = ~(od < self.beta).any(1)                  # remove transparent pixels
        if keep.sum() < 200:
            self.n_fallback += 1
            return img
        odh = od[keep]
        try:
            _, v = np.linalg.eigh(np.cov(odh.T))
            plane = v[:, 1:3]                            # two largest eigenvectors
            if plane[0, 0] < 0:
                plane[:, 0] *= -1
            if plane[0, 1] < 0:
                plane[:, 1] *= -1
            phi = np.arctan2(odh @ plane[:, 1], odh @ plane[:, 0])
            lo, hi = np.percentile(phi, self.alpha), np.percentile(phi, 100 - self.alpha)
            v1 = plane @ np.array([np.cos(lo), np.sin(lo)])
            v2 = plane @ np.array([np.cos(hi), np.sin(hi)])
            he = np.stack([v1, v2], 1) if v1[0] > v2[0] else np.stack([v2, v1], 1)   # H first (larger red OD)
            c = np.linalg.lstsq(he, od.T, rcond=None)[0]                              # (2, N)
            maxc = np.percentile(c, 99, axis=1)
            if not np.all(np.isfinite(maxc)) or (maxc < 1e-3).any() or not np.all(np.isfinite(he)):
                raise ValueError("degenerate stain estimate")
        except (np.linalg.LinAlgError, ValueError):
            self.n_fallback += 1
            return img
        c = c / maxc[:, None] * self.maxc_ref[:, None]
        out = self.io * np.exp(-self.he_ref @ c)
        return np.clip(out.T.reshape(h, w, 3), 0, 255).astype(np.uint8)

    def __call__(self, X):
        return np.stack([self.one(im) for im in X])


def make_normaliser(name, X_train):
    """name in {'reinhard','macenko'}; X_train used only to set the Reinhard target statistics."""
    n = {"reinhard": ReinhardNorm, "macenko": MacenkoNorm}[name]()
    return n.fit(X_train)
