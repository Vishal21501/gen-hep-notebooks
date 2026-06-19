"""Preprocessing for jet point clouds — the piece that makes jet generation
actually match the real distributions.

Jets are (N, P, 3) = (etarel, phirel, ptrel). Two things make raw features a bad
target for a generator:

  * the three features live on very different scales (angles ~0.1, ptrel ~0.03),
    so a plain MSE/critic is dominated by one of them;
  * ptrel is heavy-tailed — the leading particle carries ~20% while most carry
    ~1% — and a model that sees it linearly defaults to a flat ~uniform pt,
    which puts the jet-mass peak in the wrong place.

`JetStandardizer` fixes both: it log-transforms ptrel (so the heavy tail becomes
roughly Gaussian and the concentration is learnable), then z-scores every
feature using training-set statistics. Models train and generate in this
standardized space; `inverse_transform` brings samples back to physical jets.
"""

from __future__ import annotations

import numpy as np


class JetStandardizer:
    def __init__(self, log_pt: bool = True, eps: float = 1e-6, clip: float = 5.0):
        self.log_pt = log_pt
        self.eps = eps
        self.clip = clip  # clamp standardized values to tame padded-slot outliers
        self.mean_ = None
        self.std_ = None

    def _to_feature_space(self, jets):
        """Raw jets -> feature space (optionally log-pt), float64."""
        f = np.asarray(jets, dtype=np.float64).copy()
        if self.log_pt:
            f[..., 2] = np.log(np.clip(f[..., 2], self.eps, None))
        return f

    def fit(self, jets):
        f = self._to_feature_space(jets)
        # Statistics over ACTIVE particles only (ptrel > 0), so zero-padding
        # doesn't drag the mean/std around.
        mask = np.asarray(jets)[..., 2] > 0
        active = f[mask]
        self.mean_ = active.mean(axis=0)
        self.std_ = active.std(axis=0) + 1e-8
        return self

    def transform(self, jets):
        f = self._to_feature_space(jets)
        z = (f - self.mean_) / self.std_
        if self.clip:
            z = np.clip(z, -self.clip, self.clip)
        return z.astype(np.float32)

    def inverse_transform(self, std_jets, renorm_pt=True):
        """Standardized samples -> physical jets (etarel, phirel, ptrel).

        `renorm_pt` rescales each jet's ptrel to sum to 1 (as real jets do). The
        log-pt modelling supplies the *shape* (the concentration); this restores
        the *total*, so the jet-mass scale is correct even if the model's raw
        pt sum drifts a little."""
        f = np.asarray(std_jets, dtype=np.float64) * self.std_ + self.mean_
        if self.log_pt:
            f[..., 2] = np.exp(f[..., 2])
        f[..., 2] = np.clip(f[..., 2], 0.0, None)   # ptrel is non-negative
        if renorm_pt:
            f[..., 2] = f[..., 2] / (f[..., 2].sum(axis=-1, keepdims=True) + 1e-8)
        return f.astype(np.float32)

    def fit_transform(self, jets):
        return self.fit(jets).transform(jets)
