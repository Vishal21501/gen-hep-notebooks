"""Scoring harness for the calorimeter fast-sim capstone.

Students self-score all day with this one module; the final number is recomputed
server-side on a held-out reference set. Three layers, mirroring the CAPSTONE
metrics (physics first, then the classifier):

1. **Physics observables** (`observables`) — per-shower scalars a physicist reads
   off a shower: total deposited energy, energy response `E_dep / E_inc`,
   longitudinal depth (energy-weighted centroid down the image), lateral width
   (energy-weighted radial RMS) and sparsity (active-cell fraction).
2. **Distribution distances** (`observable_w1s`) — 1-Wasserstein between real and
   generated for each observable (reusing the W1 idea from notebook 05), via
   `scipy.stats.wasserstein_distance`.
3. **The headline metric** (`classifier_two_sample_auc`) — train a small CNN to
   tell real showers from generated ones; **AUC near 0.5 = indistinguishable = a
   good simulator**. This is the CaloChallenge's own yardstick, the image analogue
   of the GNN two-sample test in notebook 05.

`score_showers(...)` packages all three into one dict so a student calls a single
function. Everything degrades gracefully: no torch -> a scikit-learn classifier;
no scipy -> a histogram-based W1, so the harness runs even on a bare box.

Shower tensors are CaloChallenge **Dataset 2** grids, `(N, 45, 9, 16)` in
**physical GeV** — `(layers, radial, angular)`. The longitudinal (depth) axis is
the 45 layers; the lateral (radial) axis is the 9 rings. `incident` is a 1-D
array of incident energies in GeV, aligned with the showers.
"""

from __future__ import annotations

import numpy as np

_ACTIVE_THRESHOLD = 1e-3   # GeV; a voxel above this counts as "hit"


# --------------------------------------------------------------------------- #
# 1. Physics observables (per-shower scalars)
# --------------------------------------------------------------------------- #

def total_energy(showers):
    """Total deposited energy per shower (GeV). Shape (N,)."""
    s = np.asarray(showers, dtype=np.float64)
    return s.sum(axis=(1, 2, 3))


def energy_response(showers, incident):
    """Deposited / incident energy per shower — the sampling fraction. Shape (N,)."""
    inc = np.asarray(incident, dtype=np.float64).ravel()
    return total_energy(showers) / np.clip(inc, 1e-9, None)


def longitudinal_depth(showers):
    """Energy-weighted centroid along the **layer** axis (depth) per shower — "how
    deep did the shower get", in layer units [0, 44]. Shape (N,)."""
    s = np.asarray(showers, dtype=np.float64)
    e_layer = s.sum(axis=(2, 3))                              # (N, L)
    layers = np.arange(s.shape[1], dtype=np.float64)[None]    # (1, L)
    e = np.clip(e_layer.sum(axis=1), 1e-9, None)
    return (e_layer * layers).sum(axis=1) / e


def lateral_width(showers):
    """Energy-weighted centroid along the **radial** axis (lateral spread) per
    shower, in ring units [0, 8] — wider = more spread out. Shape (N,)."""
    s = np.asarray(showers, dtype=np.float64)
    e_ring = s.sum(axis=(1, 3))                               # (N, R)
    rings = np.arange(s.shape[2], dtype=np.float64)[None]     # (1, R)
    e = np.clip(e_ring.sum(axis=1), 1e-9, None)
    return (e_ring * rings).sum(axis=1) / e


def sparsity(showers, threshold=_ACTIVE_THRESHOLD):
    """Fraction of voxels above `threshold` GeV per shower (the "active" fraction).
    Shape (N,). Calorimeter showers are sparse, so this is small."""
    s = np.asarray(showers, dtype=np.float64)
    return (s > threshold).mean(axis=(1, 2, 3))


def observables(showers, incident):
    """All per-shower observables as a dict of (N,) arrays — the physics summary
    a student histograms real-vs-generated."""
    return {
        "total_energy": total_energy(showers),
        "response": energy_response(showers, incident),
        "depth": longitudinal_depth(showers),
        "width": lateral_width(showers),
        "sparsity": sparsity(showers),
    }


# --------------------------------------------------------------------------- #
# 2. Distribution distances (1-Wasserstein per observable)
# --------------------------------------------------------------------------- #

def _w1(a, b):
    """1-Wasserstein between two 1-D samples. Uses scipy when present, else an
    exact sorted-CDF computation (the same quantity, no dependency)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    try:
        from scipy.stats import wasserstein_distance
        return float(wasserstein_distance(a, b))
    except Exception:  # noqa: BLE001 - exact fallback (equal-weight empirical CDFs)
        a, b = np.sort(a), np.sort(b)
        grid = np.concatenate([a, b])
        grid.sort()
        ca = np.searchsorted(a, grid, side="right") / len(a)
        cb = np.searchsorted(b, grid, side="right") / len(b)
        return float(np.sum(np.abs(ca[:-1] - cb[:-1]) * np.diff(grid)))


def observable_w1s(gen, gen_incident, ref, ref_incident):
    """1-Wasserstein distance between generated and reference distributions for
    each observable. Returns a dict {observable: W1}; **lower is better**."""
    g = observables(gen, gen_incident)
    r = observables(ref, ref_incident)
    return {k: _w1(g[k], r[k]) for k in g}


# --------------------------------------------------------------------------- #
# 3. The headline metric — classifier two-sample test (AUC)
# --------------------------------------------------------------------------- #

def roc_auc(y_true, scores):
    """Rank-based ROC AUC (Mann-Whitney U) — no sklearn dependency. Same helper
    as notebook 05, so the two-sample story is identical across tracks."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    auc = (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def _preprocess(showers):
    """log1p-compress the huge dynamic range so the classifier sees a sane scale
    (the same transform the diffusion model trains in)."""
    return np.log1p(np.asarray(showers, dtype=np.float32))


class _ShowerCNN:
    """A small conv classifier for real-vs-generated showers, built lazily so the
    module imports without torch. Two conv blocks + global pool + linear head —
    deliberately modest: the *best* such classifier's AUC is the metric, and a
    huge net would just overfit the few-thousand-shower sets here."""

    def __init__(self, in_ch, device=None):
        import torch.nn as nn
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(32, 1),
        ).to(self.device)

    def fit(self, X, y, epochs=6, batch=128, lr=1e-3):
        torch = self.torch
        ds = torch.utils.data.TensorDataset(
            torch.as_tensor(X), torch.as_tensor(y, dtype=torch.float32))
        loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        bce = torch.nn.functional.binary_cross_entropy_with_logits
        self.net.train()
        for _ in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                loss = bce(self.net(xb).squeeze(-1), yb)
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    def predict(self, X):
        torch = self.torch
        self.net.eval()
        with torch.no_grad():
            logits = self.net(torch.as_tensor(X).to(self.device)).squeeze(-1)
            return torch.sigmoid(logits).cpu().numpy()


def classifier_two_sample_auc(ref, gen, epochs=6, seed=0):
    """Train the best small classifier we can to separate real showers (`ref`,
    label 0) from generated (`gen`, label 1); return its held-out ROC **AUC**.

    **AUC near 0.5** => the classifier can't tell them apart => the generator is
    indistinguishable from real => a *good* fast simulator. AUC toward 1.0 means
    it found a tell. Uses a torch CNN when available, else a scikit-learn
    classifier on flattened pixels, else (no torch, no sklearn) a logistic
    regression in numpy — so the harness always returns a number.
    """
    ref = _preprocess(ref)
    gen = _preprocess(gen)
    n = min(len(ref), len(gen))
    ref, gen = ref[:n], gen[:n]
    X = np.concatenate([ref, gen]).astype(np.float32)
    y = np.concatenate([np.zeros(n), np.ones(n)]).astype(np.float32)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    n_tr = int(0.8 * len(X))
    Xtr, ytr, Xte, yte = X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]

    try:                                            # 1) torch CNN (the real thing)
        import torch  # noqa: F401
        clf = _ShowerCNN(X.shape[1]).fit(Xtr, ytr, epochs=epochs)
        scores = clf.predict(Xte)
        return roc_auc(yte, scores)
    except Exception:                               # noqa: BLE001
        pass

    Xtr_f = Xtr.reshape(len(Xtr), -1)
    Xte_f = Xte.reshape(len(Xte), -1)
    try:                                            # 2) sklearn on flattened pixels
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=200, random_state=seed)
        clf.fit(Xtr_f, ytr)
        return roc_auc(yte, clf.predict_proba(Xte_f)[:, 1])
    except Exception:                               # noqa: BLE001
        pass

    # 3) numpy logistic regression — last-resort, dependency-free
    w = np.zeros(Xtr_f.shape[1]); b = 0.0
    Xs = (Xtr_f - Xtr_f.mean(0)) / (Xtr_f.std(0) + 1e-6)
    for _ in range(300):
        p = 1 / (1 + np.exp(-(Xs @ w + b)))
        g = Xs.T @ (p - ytr) / len(ytr)
        w -= 0.1 * g; b -= 0.1 * float((p - ytr).mean())
    Xte_s = (Xte_f - Xtr_f.mean(0)) / (Xtr_f.std(0) + 1e-6)
    return roc_auc(yte, 1 / (1 + np.exp(-(Xte_s @ w + b))))


# --------------------------------------------------------------------------- #
# One-call self-scoring
# --------------------------------------------------------------------------- #

def score_showers(gen, gen_incident, ref, ref_incident, epochs=6, seed=0):
    """Self-score generated showers against a reference set. Returns a dict with:

      * ``auc``        — classifier two-sample AUC (headline; ~0.5 is best),
      * ``w1``         — per-observable 1-Wasserstein distances (lower better),
      * ``response_slope`` — slope of E_dep vs E_inc (a well-calibrated model ~ const),
      * ``mean_response``  — mean deposited/incident ratio (the sampling fraction).

    This is the number students watch all day; the final grade recomputes ``auc``
    on the held-out reference set server-side.
    """
    auc = classifier_two_sample_auc(ref, gen, epochs=epochs, seed=seed)
    w1 = observable_w1s(gen, gen_incident, ref, ref_incident)
    inc = np.asarray(gen_incident, dtype=np.float64).ravel()
    dep = total_energy(gen)
    slope = float(np.polyfit(inc, dep, 1)[0]) if len(inc) > 1 else float("nan")
    return {
        "auc": auc,
        "w1": w1,
        "response_slope": slope,
        "mean_response": float(np.mean(energy_response(gen, gen_incident))),
    }
