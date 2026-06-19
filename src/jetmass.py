"""The physics spine: jet mass.

One observable, computed and plotted the same way in notebooks 01/02/05/06, so
students can compare a VAE, a GAN and a diffusion model on identical axes. The
rule of thumb for these generative models is *the bulk is easy, the high-mass
tail is hard* — that's the story the recurring plot tells.

Jets here are point clouds: arrays of shape (n_jets, n_particles, 3) whose last
axis is (eta_rel, phi_rel, pt_rel), the standard JetNet particle features. Zero-
padded particles (pt_rel == 0) drop out of the sum. Constituents are treated as
massless, so the jet mass is the invariant mass of the summed four-vectors.
"""

from __future__ import annotations

import numpy as np


def _to_numpy(x) -> np.ndarray:
    """Accept a torch tensor or numpy array; always hand back a detached
    numpy array so the same helper works in every notebook."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def jet_mass(jets) -> np.ndarray:
    """Relative jet mass for each jet in a (n_jets, n_particles, 3) batch.

    Features are (eta_rel, phi_rel, pt_rel). Massless constituents:
        px = pt cos(phi), py = pt sin(phi), pz = pt sinh(eta), E = pt cosh(eta)
    Sum over particles, then m = sqrt(max(E^2 - p^2, 0)). Returns shape (n_jets,).
    """
    j = _to_numpy(jets)
    if j.ndim == 2:  # a single jet -> add a batch axis
        j = j[None]
    eta, phi, pt = j[..., 0], j[..., 1], j[..., 2]
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    E = pt * np.cosh(eta)
    E_tot = E.sum(axis=1)
    px_t, py_t, pz_t = px.sum(axis=1), py.sum(axis=1), pz.sum(axis=1)
    m2 = E_tot**2 - (px_t**2 + py_t**2 + pz_t**2)
    return np.sqrt(np.clip(m2, 0.0, None))


def jet_mass_w1(real, gen) -> float:
    """1-Wasserstein distance between the real and generated jet-mass
    distributions. A single scalar that shrinks as the generator improves —
    the natural thing to return from a checkpoint. Uses scipy if available,
    else a quotient-free quantile approximation."""
    m_real, m_gen = jet_mass(real), jet_mass(gen)
    try:
        from scipy.stats import wasserstein_distance
        return float(wasserstein_distance(m_real, m_gen))
    except ImportError:
        qs = np.linspace(0, 1, 200)
        return float(np.mean(np.abs(np.quantile(m_real, qs) - np.quantile(m_gen, qs))))


def plot_jet_mass(real, gen=None, *, bins=50, mass_range=None,
                  labels=("real", "generated"), title="Jet mass", ax=None):
    """The recurring overlay. Pass real jets alone, or real + generated to
    compare. `mass_range=None` auto-ranges to the 0.5–99.5 percentile of the
    data (so the histogram fills the axes instead of being clipped to a fixed
    window — important when the mass scale differs from real JetNet). Returns
    the matplotlib Axes, so a notebook cell can keep going and end on a scalar
    (what cadence wants for an auto-checkpoint)."""
    import matplotlib.pyplot as plt
    from contextlib import nullcontext
    # Scope the HEP style to THIS plot only — `plt.style.use` would set it
    # globally and persistently, giving every later plot (jet displays, PCA)
    # giant CMS fonts.
    try:
        import mplhep as hep
        style_ctx = plt.style.context(hep.style.CMS)
    except Exception:
        style_ctx = nullcontext()

    m_real = jet_mass(real)
    m_gen = jet_mass(gen) if gen is not None else None
    if mass_range is None:
        pool = m_real if m_gen is None else np.concatenate([m_real, m_gen])
        lo, hi = np.percentile(pool, [0.5, 99.5])
        mass_range = (float(max(0.0, lo)), float(hi if hi > lo else lo + 1e-6))

    with style_ctx:
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))
        # Real as a filled band, generated as a bold step on top.
        ax.hist(m_real, bins=bins, range=mass_range, density=True, histtype="stepfilled",
                alpha=0.35, label=labels[0])
        if m_gen is not None:
            ax.hist(m_gen, bins=bins, range=mass_range, density=True, histtype="step",
                    linewidth=2.2, label=labels[1])
        ax.set_xlabel("relative jet mass")
        ax.set_ylabel("normalised density")
        ax.set_title(title)
        ax.legend()
    return ax


def plot_jets(jets, n=3, titles=None, pt_scale=600.0):
    """Display jets as event pictures: each particle a dot in the (eta_rel,
    phi_rel) plane with marker area proportional to pt_rel — the classic jet
    display. Shows the first `n` jets side by side; returns the Figure."""
    import matplotlib.pyplot as plt
    j = _to_numpy(jets)
    if j.ndim == 2:
        j = j[None]
    n = min(n, len(j))
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.0), squeeze=False)
    for i, ax in enumerate(axes[0]):
        p = j[i]
        mask = p[:, 2] > 0                      # real (non-padded) particles
        ax.scatter(p[mask, 0], p[mask, 1], s=pt_scale * p[mask, 2],
                   alpha=0.6, edgecolors="k", linewidths=0.3)
        ax.set_xlabel(r"$\eta^{\mathrm{rel}}$", fontsize=9)
        if i == 0:
            ax.set_ylabel(r"$\phi^{\mathrm{rel}}$", fontsize=9)
        ax.set_title(titles[i] if titles else f"jet {i}  ({int(mask.sum())} particles)",
                     fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def plot_jet_overlay(jets_a, jets_b, n=3, labels=("real", "reconstructed"),
                     colors=("C0", "C3"), pt_scale=600.0):
    """Overlay two jet sets on the SAME (eta, phi) axes — e.g. a real jet and its
    VAE reconstruction — one colour each, marker area proportional to pt. Shows
    the first `n` paired jets side by side; returns the Figure."""
    import matplotlib.pyplot as plt
    a, b = _to_numpy(jets_a), _to_numpy(jets_b)
    if a.ndim == 2:
        a = a[None]
    if b.ndim == 2:
        b = b[None]
    n = min(n, len(a), len(b))
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.0), squeeze=False)
    for i, ax in enumerate(axes[0]):
        for jets, c, lab in ((a, colors[0], labels[0]), (b, colors[1], labels[1])):
            p = jets[i]
            m = p[:, 2] > 0
            ax.scatter(p[m, 0], p[m, 1], s=pt_scale * p[m, 2], alpha=0.5,
                       color=c, edgecolors="k", linewidths=0.3, label=lab)
        ax.set_xlabel(r"$\eta^{\mathrm{rel}}$", fontsize=9)
        if i == 0:
            ax.set_ylabel(r"$\phi^{\mathrm{rel}}$", fontsize=9)
            ax.legend(fontsize=8, loc="upper right")
        ax.set_title(f"jet {i}", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_aspect("equal")
    fig.tight_layout()
    return fig
