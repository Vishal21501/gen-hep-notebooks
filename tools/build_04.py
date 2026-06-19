"""Build notebooks/04_diffusion_calo.ipynb — energy-conditioned diffusion on
calorimeter showers (the second data modality).

Fun spine carries over from 03 (image-like diffusion), but the modality here is
**calorimeter showers**: small energy images (N, 1, H, W) that a particle leaves
behind. We condition a `diffusers` UNet on the **incident energy** and check the
generator's physics: does the deposited energy track the requested energy
(energy response), and does it follow a roughly linear calibration?
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nbbuild import build, md, code, setup, exercise

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "notebooks" / "04_diffusion_calo.ipynb"

cells = [
    md("""
# 04 · Diffusion for Calorimeter Showers

**Generative Modelling for HEP** — notebook 4 of 6.

So far the physics modality has been **jets** (point clouds). Here we switch to
the other big LHC data modality: **calorimeter showers**. When a particle hits
the calorimeter it deposits energy across a grid of cells — effectively a small
*energy image* `(1, H, W)`. The **CaloChallenge** asks us to *generate* these
showers fast, as a surrogate for slow Geant4 simulation.

A shower is not unconditional: a 100 GeV particle deposits far more energy than a
1 GeV one. So we build a **conditional diffusion model** — a `diffusers` UNet
that takes the **incident energy** as a condition (via class embeddings on
energy bins) and learns to denoise showers *for that energy*.

We'll then probe the generator the way a physicist would:

- **energy response** — does generated total energy track the requested energy?
- **calibration / linearity** — what is the deposited-vs-incident ratio?
- a **1-D summary distribution** (our recurring physics spine) — generated vs
  real total-energy spectra.

> You complete the cells marked **Exercise**. Everything else runs as-is.
"""),

    md("## Setup"),
    setup("""
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import sys; sys.path.insert(0, "..")
from src.seeds import set_seed
from src.train import get_device, to_loader, train
from src.data import load_calochallenge
from src.diffusion import make_unet, make_scheduler

SEED = 0
IMG = 16             # showers are (1, IMG, IMG) energy images
N_EBINS = 10         # incident-energy is bucketed into this many class bins
EPOCHS = 6           # small default; trains well under 10 min on one GPU
BATCH = 128
set_seed(SEED)
device = get_device()
print(f"device={device}  img={IMG}x{IMG}  energy_bins={N_EBINS}  epochs={EPOCHS}")
"""),

    # ----------------------------------------------------------------- Part A
    md("""
## Part A · The calorimeter dataset

`load_calochallenge` returns `(showers, incident_energy)`: showers are
`(N, 1, H, W)` energy images (GeV per cell) and `incident_energy` is a 1-D array
of incident particle energies (GeV). If the real CaloChallenge HDF5 isn't mounted
we fall back to a synthetic stand-in with the same shapes and the same physics:
total deposited energy scales with the incident energy.

Two preprocessing choices matter for diffusion:

1. **Normalise** the energy images to roughly `[-1, 1]` (diffusion assumes data
   near unit scale), keeping the per-shower scale so we can invert it later.
2. **Bin** the continuous incident energy into `N_EBINS` discrete classes — the
   UNet's `class_embed` conditioning takes an integer label per sample.
"""),
    setup("""
showers, incident = load_calochallenge(path=None, max_showers=4000)
showers = showers.astype("float32")
incident = incident.astype("float32")
print("showers:", tuple(showers.shape), "| incident GeV range:",
      round(float(incident.min()), 1), "->", round(float(incident.max()), 1))

# Per-dataset normalisation to ~[-1, 1]: log1p tames the huge dynamic range of
# energy deposits, then we scale by the global max. `to_image`/`from_image`
# round-trip between physical GeV and the network's normalised space.
LOG_MAX = float(np.log1p(showers).max())

def to_image(e_gev):
    return (np.log1p(e_gev) / LOG_MAX) * 2.0 - 1.0      # GeV -> [-1, 1]

def from_image(x):
    x = np.asarray(x)
    return np.expm1(((x + 1.0) / 2.0) * LOG_MAX)        # [-1, 1] -> GeV

X = to_image(showers)                                   # (N, 1, IMG, IMG)

# Bin incident energy into N_EBINS classes (uniform in log-energy).
log_e = np.log(incident)
EDGES = np.linspace(log_e.min(), log_e.max() + 1e-6, N_EBINS + 1)

def energy_to_bin(e_gev):
    return np.clip(np.digitize(np.log(e_gev), EDGES) - 1, 0, N_EBINS - 1)

def bin_to_energy(b):
    # representative (geometric-centre) energy of a bin
    lo, hi = EDGES[b], EDGES[b + 1]
    return float(np.exp(0.5 * (lo + hi)))

ebin = energy_to_bin(incident).astype("int64")
print("energy bins populated:", len(np.unique(ebin)), "/", N_EBINS)
"""),

    md("""
We pair each shower image with its energy-bin label in the loader, build the
conditional UNet (`class_embed=True`, `num_classes=N_EBINS`), and a cosine DDPM
scheduler. We optionally warm-start from `checkpoints/calo_diffusion.pt`.
"""),
    setup("""
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

X_t = torch.as_tensor(X, dtype=torch.float32)
y_t = torch.as_tensor(ebin, dtype=torch.long)
calo_loader = DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH, shuffle=True)

unet = make_unet(sample_size=IMG, in_channels=1, base_channels=32,
                 class_embed=True, num_classes=N_EBINS).to(device)
scheduler = make_scheduler(num_train_timesteps=1000)
T_TRAIN = scheduler.config.num_train_timesteps

CKPT = Path("..") / "checkpoints" / "calo_diffusion.pt"
if CKPT.exists():
    unet.load_state_dict(torch.load(CKPT, map_location=device))
    print("loaded checkpoint:", CKPT)
else:
    print("no checkpoint; will train a small model in the next exercise")

print("UNet params:", sum(p.numel() for p in unet.parameters()))
"""),

    md("""
### Exercise 1 — The energy-conditioned training step

Diffusion training is "predict the noise". For a batch of shower images `x0` and
their energy-bin labels `labels`:

1. sample random timesteps `t` and Gaussian `noise`,
2. form the noisy images with `scheduler.add_noise(x0, noise, t)`,
3. ask the UNet to predict the noise **given the energy condition** — pass the
   labels via `class_labels=labels`,
4. return the **MSE** between predicted and true noise.

Implement `calo_step`, train for `EPOCHS`, and report the **final epoch's mean
training loss** as `final_loss` (a number).
"""),
    exercise(
        scaffold_body="""
def calo_step(model, batch):
    x0, labels = batch                       # images in [-1,1], energy-bin ints
    noise = torch.randn_like(x0)
    t = torch.randint(0, T_TRAIN, (x0.shape[0],), device=x0.device).long()
    # 1) noise the images to step t with the scheduler:
    noisy = ...
    # 2) predict the noise, conditioned on the energy bin (class_labels=...):
    pred = ...
    # 3) noise-prediction MSE:
    return ...
""",
        solution_body="""
def calo_step(model, batch):
    x0, labels = batch
    noise = torch.randn_like(x0)
    t = torch.randint(0, T_TRAIN, (x0.shape[0],), device=x0.device).long()
    noisy = scheduler.add_noise(x0, noise, t)
    pred = model(noisy, t, class_labels=labels).sample
    return F.mse_loss(pred, noise)

if CKPT.exists():
    # already warm-started; a single short polish epoch keeps the loss honest
    history = train(unet, calo_loader, calo_step, epochs=1, lr=1e-4, device=device)
else:
    history = train(unet, calo_loader, calo_step, epochs=EPOCHS, lr=2e-4, device=device)
final_loss = round(float(history[-1]), 4)
print("final noise-MSE loss:", final_loss)
""",
        answer_var="final_loss",
    ),

    # ----------------------------------------------------------------- Part B
    md("""
## Part B · Sampling conditioned on energy

To generate a shower at a chosen energy bin we run the reverse diffusion loop,
passing the same `class_labels` at every denoising step. We provide a
`sample_showers(bins)` helper that returns generated showers **in physical GeV**
(it inverts the `from_image` normalisation and floors tiny negatives to zero).
"""),
    setup("""
@torch.no_grad()
def sample_showers(bins, n_steps=50):
    \"\"\"Generate one shower per entry in `bins` (an int array of energy bins).
    Returns showers in physical GeV, shape (len(bins), 1, IMG, IMG).\"\"\"
    unet.eval()
    labels = torch.as_tensor(np.asarray(bins), dtype=torch.long, device=device)
    x = torch.randn(len(labels), 1, IMG, IMG, device=device)
    scheduler.set_timesteps(n_steps)
    for t in scheduler.timesteps:
        eps = unet(x, t, class_labels=labels).sample
        x = scheduler.step(eps, t, x).prev_sample
    gev = from_image(x.cpu().numpy())
    return np.clip(gev, 0.0, None)

_demo = sample_showers(np.array([0, N_EBINS // 2, N_EBINS - 1]))
print("sampled showers:", _demo.shape,
      "| total GeV per shower:", np.round(_demo.sum((1, 2, 3)), 1))
"""),

    md("""
### Exercise 2 — Energy response (the key conditioning test)

A conditional generator is only useful if the condition *takes*. Request a spread
of incident energies, generate a shower for each, and measure the **energy
response**: the Pearson correlation between the **requested incident energy** and
the **total generated deposited energy**.

Steps:

1. pick `req_energy` — a range of incident energies across the dataset,
2. map them to bins with `energy_to_bin`, generate with `sample_showers`,
3. sum each generated shower to get `gen_total` (GeV),
4. compute Pearson `r` between `req_energy` and `gen_total`.

Return `response_r` (a number). A well-conditioned model gives `r` close to 1.
"""),
    exercise(
        scaffold_body="""
req_energy = np.linspace(incident.min() + 1, incident.max() - 1, 200)
gen = sample_showers(energy_to_bin(req_energy))
# total deposited energy per generated shower (sum over the image):
gen_total = ...
# Pearson correlation between requested energy and generated total energy:
response_r = ...
""",
        solution_body="""
req_energy = np.linspace(incident.min() + 1, incident.max() - 1, 200)
gen = sample_showers(energy_to_bin(req_energy))
gen_total = gen.sum(axis=(1, 2, 3))
response_r = float(np.corrcoef(req_energy, gen_total)[0, 1])

plt.figure(figsize=(5, 4))
plt.scatter(req_energy, gen_total, s=8, alpha=0.5)
plt.xlabel("requested incident energy [GeV]")
plt.ylabel("generated deposited energy [GeV]")
plt.title(f"energy response  (Pearson r = {response_r:.3f})")
plt.show()
response_r = round(response_r, 3)
print("energy-response Pearson r:", response_r)
""",
        answer_var="response_r",
    ),

    md("""
### Exercise 3 — Calibration: deposited / incident

The response tells us the *trend* is right; calibration tells us the *scale*. For
a sampling calorimeter the deposited energy is a roughly **linear** function of
the incident energy, so the ratio `deposited / incident` is approximately
constant — the "sampling fraction".

Using the `req_energy` and `gen_total` from Exercise 2, compute the **mean of the
per-shower ratio** `gen_total / req_energy`. Return `mean_ratio` (a number).
"""),
    exercise(
        scaffold_body="""
# mean of the per-shower deposited/incident energy ratio:
mean_ratio = ...
""",
        solution_body="""
ratios = gen_total / req_energy
mean_ratio = float(np.mean(ratios))

plt.figure(figsize=(5, 3))
plt.hist(ratios, bins=30)
plt.axvline(mean_ratio, color="k", ls="--", label=f"mean = {mean_ratio:.3f}")
plt.xlabel("deposited / incident energy"); plt.legend(); plt.show()
mean_ratio = round(mean_ratio, 3)
print("mean deposited/incident ratio:", mean_ratio)
""",
        answer_var="mean_ratio",
    ),

    md("""
### Exercise 4 — Look at a shower, measure its sparsity

Calorimeter showers are **sparse**: most cells are empty, energy concentrates in
a compact core. Generate a single high-energy shower, display the image (log
scale), and compute its **sparsity**: the fraction of cells whose energy is above
a small threshold (`1e-3` GeV).

Return `active_fraction` (a number in `[0, 1]`).
"""),
    exercise(
        scaffold_body="""
one = sample_showers(np.array([N_EBINS - 1]))[0, 0]   # (IMG, IMG) GeV image
# fraction of cells with energy > 1e-3 GeV (the "active" cells):
active_fraction = ...
""",
        solution_body="""
one = sample_showers(np.array([N_EBINS - 1]))[0, 0]
active_fraction = float((one > 1e-3).mean())

plt.figure(figsize=(4, 4))
plt.imshow(np.log1p(one), cmap="inferno")
plt.colorbar(label="log(1 + E[GeV])")
plt.title(f"generated shower  (active cells = {active_fraction:.2f})")
plt.axis("off"); plt.show()
active_fraction = round(active_fraction, 3)
print("active-cell fraction:", active_fraction)
""",
        answer_var="active_fraction",
    ),

    md("""
## Part C · The physics spine

Our recurring habit: compare a **1-D summary distribution** of generated vs real.
Here the summary is the **total deposited energy** spectrum. We overlay the real
showers' total energy against a matched set of generated showers (sampled at the
real events' energy bins).
"""),
    setup("""
real_total = showers.sum(axis=(1, 2, 3))
_gen_match = sample_showers(ebin[:1000]).sum(axis=(1, 2, 3))
plt.figure(figsize=(5, 4))
bins = np.linspace(0, real_total.max(), 40)
plt.hist(real_total[:1000], bins=bins, alpha=0.5, label="real", density=True)
plt.hist(_gen_match, bins=bins, alpha=0.5, label="generated", density=True)
plt.xlabel("total deposited energy [GeV]"); plt.ylabel("density")
plt.title("physics spine: total-energy spectrum"); plt.legend(); plt.show()
print("overlaid real vs generated total-energy spectra")
"""),

    md("""
### Exercise 5 — Where does diffusion struggle?

Look at the overlay above. Like every generator, diffusion matches the **bulk**
of a distribution more easily than the sparse, high-energy **tails** of rare
showers. In **one word**, which part of the energy spectrum is hardest to model —
`"bulk"` or `"tails"`? Set `hardest_region` to that word.
"""),
    exercise(
        scaffold_body="""
# answer: string
hardest_region = ...   # "bulk" or "tails"
""",
        solution_body="""
hardest_region = "tails"
""",
        answer_var="hardest_region",
    ),

    md("""
## Recap

- A calorimeter shower is an **energy image** — a second LHC data modality beyond
  jets, and the target of the **CaloChallenge** fast-simulation effort.
- We trained a **conditional diffusion** model: a `diffusers` UNet with energy-bin
  **class embeddings**, trained with the usual **predict-the-noise** MSE.
- The physics checks that matter: the **energy response** (generated energy tracks
  the requested energy, high Pearson r), the **calibration ratio**
  (deposited/incident ≈ constant), and shower **sparsity**.
- The recurring **1-D spine** — here the total-energy spectrum — shows diffusion,
  like the VAE (01) and GAN (02), works hardest in the **tails**.
"""),
]

build(OUT, cells)
