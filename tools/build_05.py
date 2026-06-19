"""Build notebooks/05_evaluation.ipynb — scoring generators.

How do you know a generative model is any *good*? This notebook teaches the
evaluation toolkit for jet generators:

* **W1** — 1-Wasserstein distance on a physics observable (jet mass).
* **FPD / KPD** — Frechet / Kernel Physics Distance (`jetnet.evaluation`), the
  community-standard distribution-level scores.
* **Classifier two-sample test** — train a real GNN to tell real from generated;
  an AUC near 0.5 means it *can't*, i.e. the generator is indistinguishable.

The actual VAE/GAN/diffusion outputs from notebooks 01-03 aren't persisted, so
we make this self-contained: load real jets and synthesize a "good" generator
(lightly perturbed) and a "bad" one (heavily perturbed + clipped tail), so the
metrics have something to rank.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nbbuild import build, md, code, setup, exercise

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "notebooks" / "05_evaluation.ipynb"

cells = [
    md("""
# 05 · Evaluating Generative Models

**Generative Modelling for HEP** — notebook 5 of 6.

You've built a VAE (01), a GAN (02) and a diffusion model (03). Each *produces*
jets — but **how do you know any of them is good?** "The samples look nice" is
not a metric. In HEP we need numbers that compare a generated distribution to
the real one on the observables physicists care about.

This notebook is the **scoring toolkit**:

1. **W1** — the 1-Wasserstein distance on the jet-mass spectrum. Cheap,
   interpretable, per-observable.
2. **FPD** and **KPD** — Frechet / Kernel Physics Distances from
   `jetnet.evaluation`: single-number, distribution-level scores that have
   become the JetNet community standard.
3. A **classifier two-sample test** — we train a real **GNN** to discriminate
   real from generated jets. If the best classifier we can train still only
   reaches **AUC ≈ 0.5**, the two samples are statistically indistinguishable —
   the strongest evidence a generator is good.

> You complete the cells marked **Exercise**. Everything else runs as-is.
"""),

    md("## Setup"),
    setup("""
import numpy as np
import matplotlib.pyplot as plt

import sys; sys.path.insert(0, "..")
from src.seeds import set_seed
from src.data import load_jetnet
from src.jetmass import jet_mass, plot_jet_mass, jet_mass_w1

SEED = 0
N_JETS = 5000
N_PARTICLES = 30
set_seed(SEED)
rng = np.random.default_rng(SEED)
print(f"seed={SEED}  n_jets={N_JETS}  n_particles={N_PARTICLES}")
"""),

    # ----------------------------------------------------------------- Part A
    md("""
## Part A · A bench of synthetic "generators"

We load real gluon jets, then fabricate two stand-in generators so the metrics
have something to rank — no trained model needed:

* **`good`** — real jets with a *tiny* per-particle jitter. A near-perfect
  generator: distributions should overlap almost exactly.
* **`bad`** — real jets with *heavy* jitter **and** a clipped high-mass tail
  (the classic failure mode: generators struggle with the rare, high-mass
  jets). Distributions should visibly disagree.

A good evaluation metric must **rank `good` above `bad`**.
"""),
    setup("""
real = load_jetnet("g", num_particles=N_PARTICLES, max_jets=N_JETS).astype(np.float32)


def perturb(jets, scale, seed):
    \"\"\"Add Gaussian jitter to real particles (pt_rel>0), keeping padding at 0.\"\"\"
    r = np.random.default_rng(seed)
    out = jets.copy()
    mask = out[..., 2] > 0                       # real (non-padded) particles
    noise = r.normal(0.0, scale, size=out.shape).astype(np.float32)
    out += noise * mask[..., None]
    out[..., 2] = np.clip(out[..., 2], 0.0, None)  # pt_rel stays non-negative
    return out


def clip_high_mass(jets, keep_quantile=0.6):
    \"\"\"Drop the high-mass tail and resample from the bulk — mimics a generator
    that never learns the rare high-mass jets.\"\"\"
    m = jet_mass(jets)
    cut = np.quantile(m, keep_quantile)
    bulk = jets[m <= cut]
    idx = np.random.default_rng(SEED).integers(0, len(bulk), size=len(jets))
    return bulk[idx]


# split real into a reference half and an "as-if-real" eval half
half = len(real) // 2
real_ref, real_eval = real[:half], real[half:]

good = perturb(real_eval, scale=0.003, seed=1)               # near-perfect
bad = clip_high_mass(perturb(real_eval, scale=0.03, seed=2)) # heavy + clipped tail

generators = {"good": good, "bad": bad}
print("real_ref:", real_ref.shape, "| good:", good.shape, "| bad:", bad.shape)
"""),
    setup("""
# Eyeball the spine first: real vs good vs bad jet mass.
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
plot_jet_mass(real_ref, good, labels=("real", "good gen"),
              title="Good generator", ax=axes[0])
plot_jet_mass(real_ref, bad, labels=("real", "bad gen"),
              title="Bad generator", ax=axes[1])
plt.tight_layout(); plt.show()
print("plotted jet-mass overlays for both generators")
"""),

    md("""
### Exercise 1 — W1 on the jet mass

The simplest score: the **1-Wasserstein distance** between the real and
generated jet-mass distributions (`src.jetmass.jet_mass_w1`). It's the average
"earth-mover" cost to morph one histogram into the other — **smaller is
better**, and it's in the same units as the observable.

Compute the W1 between `real_ref` and the **good** generator and return it as
`w1_good` (a number).
"""),
    exercise(
        scaffold_body="""
# jet_mass_w1(real, gen) -> float (1-Wasserstein on the jet-mass spectrum)
w1_good = ...
""",
        solution_body="""
w1_good = round(jet_mass_w1(real_ref, good), 5)
print("W1(real, good) =", w1_good)
# sanity check: the bad generator should score *worse* (larger W1)
print("W1(real, bad)  =", round(jet_mass_w1(real_ref, bad), 5))
""",
        answer_var="w1_good",
    ),

    # ----------------------------------------------------------------- Part B
    md("""
## Part B · FPD and KPD

W1 scores **one observable at a time**. The JetNet community standard is to
collapse the whole jet into a feature vector and compare the *distributions* of
those features:

* **FPD** (Frechet Physics Distance) — like FID for images: fits Gaussians to
  the real and generated feature clouds and measures the Frechet distance.
  Lower = better; an unbiased extrapolation removes finite-sample bias.
* **KPD** (Kernel Physics Distance) — a kernel (MMD-style) two-sample distance,
  more robust to outliers. Lower = better.

`jetnet.evaluation` provides `fpd` and `kpd` directly. We compute a small set of
physics-motivated features per jet (the **EFPs**-style summaries: mass plus a
few momentum moments) and feed those in. If `jetnet` isn't installed we fall
back to a simple Frechet distance on the same features, so the cell still runs.
"""),
    setup("""
def jet_features(jets):
    \"\"\"A compact per-jet feature vector for distribution-level metrics:
    jet mass + summary statistics of the particle kinematics. Shape (N, F).\"\"\"
    m = jet_mass(jets)[:, None]
    pt = jets[..., 2]
    eta, phi = jets[..., 0], jets[..., 1]
    n_active = (pt > 0).sum(axis=1, keepdims=True).astype(np.float32)
    pt_sum = pt.sum(axis=1, keepdims=True)
    pt_max = pt.max(axis=1, keepdims=True)
    eta_w = (np.abs(eta) * (pt > 0)).sum(axis=1, keepdims=True)
    phi_w = (np.abs(phi) * (pt > 0)).sum(axis=1, keepdims=True)
    return np.concatenate([m, n_active, pt_sum, pt_max, eta_w, phi_w], axis=1).astype(np.float64)


def frechet_distance(a, b):
    \"\"\"Plain Frechet distance between two Gaussians fit to feature clouds —
    the no-jetnet fallback for FPD.\"\"\"
    mu_a, mu_b = a.mean(0), b.mean(0)
    ca, cb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    diff = mu_a - mu_b
    # sqrt of product of covariances via eigen-decomposition (symmetric PSD)
    prod = ca @ cb
    eigvals = np.linalg.eigvals(prod)
    covmean = np.sqrt(np.clip(eigvals.real, 0, None)).sum()
    return float(diff @ diff + np.trace(ca) + np.trace(cb) - 2 * covmean)


feats_real = jet_features(real_ref)
feats_good = jet_features(good)
feats_bad = jet_features(bad)
print("feature vectors:", feats_real.shape, "(jet mass + 5 kinematic summaries)")
"""),

    md("""
### Exercise 2 — FPD for the bad generator

Compute the **FPD** between the real features and the **bad** generator's
features. Use `jetnet.evaluation.fpd` if available (it returns `(value, error)`
— take the value); otherwise use the provided `frechet_distance` fallback on the
same feature arrays. Return it as `fpd_bad` (a number).

(We expect `fpd_bad` to be clearly *larger* than the FPD of the good generator —
that ordering is the whole point.)
"""),
    exercise(
        scaffold_body="""
# Prefer jetnet.evaluation.fpd; fall back to frechet_distance(feats_real, feats_bad).
# jetnet's fpd returns (value, error) -> keep the value.
try:
    from jetnet.evaluation import fpd
    val, err = ...
    fpd_bad = float(val)
except Exception:
    fpd_bad = ...
""",
        solution_body="""
try:
    from jetnet.evaluation import fpd
    val, err = fpd(feats_real, feats_bad)
    fpd_bad = float(val)
except Exception:
    fpd_bad = frechet_distance(feats_real, feats_bad)
fpd_bad = round(fpd_bad, 5)
print("FPD(real, bad)  =", fpd_bad)
# the good generator should score clearly lower (closer to real)
print("FPD(real, good) =", round(frechet_distance(feats_real, feats_good), 5))
""",
        answer_var="fpd_bad",
    ),

    md("""
### Exercise 3 — Rank the generators

You now have two scores per generator. A metric is only useful if it **orders
the generators correctly**: the good one should beat the bad one. Compute FPD
(or W1, your choice — they should agree here) for both generators, and return
the **name of the better generator** as the string `best_generator` — either
`"good"` or `"bad"`.
"""),
    exercise(
        scaffold_body="""
# answer: string
# Score each generator (lower FPD / W1 = better) and pick the winner's name.
scores = {
    "good": ...,   # FPD or W1 between real and the good generator
    "bad": ...,    # ... and the bad generator
}
best_generator = ...   # the key with the SMALLEST score: "good" or "bad"
""",
        solution_body="""
scores = {
    "good": frechet_distance(feats_real, feats_good),
    "bad": frechet_distance(feats_real, feats_bad),
}
best_generator = min(scores, key=scores.get)
print("scores:", {k: round(v, 5) for k, v in scores.items()})
print("best generator:", best_generator)
""",
        answer_var="best_generator",
    ),

    # ----------------------------------------------------------------- Part C
    md("""
## Part C · The classifier two-sample test

The gold standard. If a generator is perfect, then **no classifier**, however
powerful, can tell its samples from real ones — the best achievable ROC **AUC is
0.5** (pure chance). The further the AUC climbs toward 1.0, the more a
discriminator has found a tell-tale difference, i.e. the worse the generator.

We train a **real GNN** on the jet point clouds (using
`src.gnn.edge_conv_encoder` — a `torch_geometric` EdgeConv message-passing net —
or `src.gnn.DeepSetsEncoder` as a no-PyG fallback) to classify *real (label 0)*
vs *generated (label 1)*, then read its AUC on a held-out split.

> This Mac has no torch, so we can't run the training here — but the code is the
> real thing you'd run on a GPU. Keep the config small.
"""),
    setup("""
import torch
import torch.nn as nn
from src.train import get_device, train
from src.gnn import DeepSetsEncoder

device = get_device()
GNN_EPOCHS = 5
GNN_BATCH = 128


class JetClassifier(nn.Module):
    \"\"\"Real-vs-generated discriminator. Uses a torch_geometric EdgeConv GNN
    when PyG is present, else a permutation-invariant DeepSets encoder. Either
    way: encode the point cloud -> a logit.\"\"\"

    def __init__(self, latent=32):
        super().__init__()
        try:
            from src.gnn import edge_conv_encoder
            self.encoder = edge_conv_encoder(in_features=3, hidden=64, latent=latent)
            self.is_pyg = True
        except Exception:
            self.encoder = DeepSetsEncoder(in_features=3, hidden=64, latent=latent)
            self.is_pyg = False
        self.head = nn.Linear(latent, 1)

    def forward(self, x):
        return self.head(self.encoder(x)).squeeze(-1)


def make_dataset(real_jets, gen_jets):
    \"\"\"Stack real (0) and generated (1) jets into one labelled tensor set.\"\"\"
    x = np.concatenate([real_jets, gen_jets]).astype(np.float32)
    y = np.concatenate([np.zeros(len(real_jets)), np.ones(len(gen_jets))]).astype(np.float32)
    perm = np.random.default_rng(SEED).permutation(len(x))
    return x[perm], y[perm]


def roc_auc(y_true, scores):
    \"\"\"Rank-based ROC AUC (Mann-Whitney U) — no sklearn dependency.\"\"\"
    y_true = np.asarray(y_true); scores = np.asarray(scores)
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = y_true.sum(); n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    auc = (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


print("JetClassifier ready (PyG EdgeConv if available, else DeepSets) "
      f"| epochs={GNN_EPOCHS} batch={GNN_BATCH} device={device}")
"""),

    md("""
### Exercise 4 — Train the GNN classifier, read the AUC

Build the real-vs-generated dataset for the **good** generator, train the
`JetClassifier` with a binary-cross-entropy objective, and evaluate the **ROC
AUC** on a held-out split. Return it as `auc_good` (a number).

A near-perfect generator should drive the AUC down toward **0.5**. (We provide
`make_dataset`, `roc_auc`, and a `train(...)` loop; you write the BCE step and
the eval.)
"""),
    exercise(
        scaffold_body="""
X, Y = make_dataset(real_ref, good)
n_train = int(0.8 * len(X))
Xtr, Ytr = X[:n_train], Y[:n_train]
Xte, Yte = X[n_train:], Y[n_train:]

clf = JetClassifier().to(device)
tr_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.as_tensor(Xtr), torch.as_tensor(Ytr)),
    batch_size=GNN_BATCH, shuffle=True)


def bce_step(model, batch):
    x, y = batch
    logits = model(x)
    # binary cross-entropy with logits between predicted logits and labels y
    return ...

train(clf, tr_loader, bce_step, epochs=GNN_EPOCHS, lr=1e-3, device=device)

clf.eval()
with torch.no_grad():
    # score the held-out jets, then ROC AUC vs the true labels Yte
    scores = ...
auc_good = round(roc_auc(Yte, scores), 3)
""",
        solution_body="""
X, Y = make_dataset(real_ref, good)
n_train = int(0.8 * len(X))
Xtr, Ytr = X[:n_train], Y[:n_train]
Xte, Yte = X[n_train:], Y[n_train:]

clf = JetClassifier().to(device)
tr_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.as_tensor(Xtr), torch.as_tensor(Ytr)),
    batch_size=GNN_BATCH, shuffle=True)


def bce_step(model, batch):
    x, y = batch
    logits = model(x)
    return nn.functional.binary_cross_entropy_with_logits(logits, y)

train(clf, tr_loader, bce_step, epochs=GNN_EPOCHS, lr=1e-3, device=device)

clf.eval()
with torch.no_grad():
    scores = torch.sigmoid(clf(torch.as_tensor(Xte).to(device))).cpu().numpy()
auc_good = round(roc_auc(Yte, scores), 3)
print("classifier two-sample-test AUC (good generator):", auc_good)
""",
        answer_var="auc_good",
    ),

    md("""
### Exercise 5 — Read the verdict

You've seen all three tools agree on the *ranking*, and you know what each score
*means*. Time to interpret.

A classifier two-sample test on a **near-perfect** generator gives an AUC close
to **0.5**: the discriminator does no better than a coin flip, so the generated
jets are statistically indistinguishable from real ones.

**True or false:** *an AUC near 0.5 indicates a good (indistinguishable)
generator.* Set the boolean `auc_05_means_good` accordingly.
"""),
    exercise(
        scaffold_body="""
# answer: bool
# AUC ~ 0.5  => classifier can't tell real from generated => good generator.
# AUC ~ 1.0  => classifier separates them perfectly      => bad generator.
auc_05_means_good = ...   # True or False
""",
        solution_body="""
auc_05_means_good = True
""",
        answer_var="auc_05_means_good",
    ),

    md("""
## Recap

- **No single number is enough** — score generators on the observables that
  matter, with complementary tools.
- **W1** is cheap and per-observable; great for the **jet-mass spine** we've
  tracked since notebook 01.
- **FPD / KPD** (`jetnet.evaluation`) give distribution-level, single-number
  scores — the JetNet community standard. **Lower is better**; a high FPD means
  the generated feature distribution is far from real.
- The **classifier two-sample test** is the strongest test: train the best
  discriminator you can and read its **AUC**. **AUC ≈ 0.5 = indistinguishable =
  good**; AUC → 1.0 means the generator has a tell.
- All three should **rank generators consistently** — if they disagree, you've
  learned something about *which* features your generator gets wrong.
"""),
]

build(OUT, cells)
