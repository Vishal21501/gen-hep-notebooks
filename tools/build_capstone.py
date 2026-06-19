"""Build capstone/capstone_calo.ipynb — the full-day fast-sim capstone starter.

Project 2 of the generative track (see capstone/CAPSTONE.md). Students spend the
day building the best **energy-conditioned calorimeter fast simulator** they can,
on the second data modality (calo showers): **CaloChallenge Dataset 2** —
GEANT4 electron showers, 1 GeV–1 TeV, each a 45x9x16 = 6480-voxel grid
(layers x radial x angular). This notebook is the *baseline starter*: it gets any
student to a first submission before lunch, then gets out of the way.

It reuses the week's machinery so nothing here is new tooling:
  * `src.data.load_calochallenge_ds2` — the REAL Dataset 2 (no synthetic fallback;
    run `python tools/prepare_data.py calochallenge` once first),
  * `src.diffusion.make_unet/_scheduler` — the energy-conditioned UNet (nb 04),
    here with the 45 detector layers as input channels,
  * `src.calo_eval.score_showers` — the scoring harness (the capstone's own),
  * cadence submission cells          — the friendly compare-and-discuss board.

Authoring contract is the same as the lesson notebooks: `##` heading per exercise,
exercise cells end on a primitive answer var, setup cells end on a print, and the
only cadence syntax is `# cadence:starter` / `# cadence:end`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nbbuild import build, md, code, setup, exercise

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "capstone" / "capstone_calo.ipynb"

cells = [
    md("""
# 🏗️ Capstone — Calorimeter Fast Simulation

**Generative Modelling for HEP** — *full-day project (≈6h)*

Full detector simulation (Geant4) is the single biggest CPU cost in HEP: the LHC
experiments spend a huge fraction of their grid budget simulating **calorimeter
showers**. Your job today is to build a **generative fast simulator** — a model
that turns an **incident particle energy** `E_inc` (GeV) into a realistic
**calorimeter shower**, thousands of times faster than Geant4. This is exactly the
public **CaloChallenge** task.

We use **CaloChallenge Dataset 2**: GEANT4 **electron** showers with incident
energy log-uniform from **1 GeV to 1 TeV**. Each shower is a **6480-voxel grid**,
`45 layers × 9 radial × 16 angular` — a genuinely 3D energy deposit.

A good fast simulator:

- **conserves energy** sensibly — deposited energy tracks `E_inc`,
- reproduces **shower shapes** — longitudinal depth (which layers light up),
  lateral width (radial spread), sparsity,
- is **fast** — batch-samples thousands of showers per second,
- and is **indistinguishable** from real showers to a trained classifier.

**Architecture is your choice** — you've seen VAEs (01), GANs (02) and diffusion
(04). This starter ships a small **energy-conditioned diffusion baseline** (the
notebook 04 model, with the 45 layers as input channels) so you reach a first
submission fast; then you improve it.

### The day
| Block | Goal |
|---|---|
| **Baseline** | Get *any* conditional generator training + submitting. **First submission before lunch.** |
| **Improve** | Better architecture / conditioning / training; iterate against the metrics + your plots. |
| **Finalize** | Lock your best model; submit final metric + plots + a short reflection. |

> You complete the cells marked **Exercise**. Everything else runs as-is. Submit
> through the **cadence** cells at the end — we walk through the whole cohort's
> boards together at wrap-up. It's a friendly compare-and-discuss, not a ranked grind.
"""),

    md("## Setup"),
    setup("""
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import sys; sys.path.insert(0, "..")
from src.seeds import set_seed
from src.train import get_device, train
from src.data import load_calochallenge_ds2, CALO_DS2_SHAPE
from src.diffusion import make_unet, make_scheduler
from src import calo_eval

SEED = 0
LAYERS, RAD, ANG = CALO_DS2_SHAPE   # (45, 9, 16) = layers, radial, angular
PAD = 16                            # pad radial 9->16 so each layer is a 16x16 image
N_LOAD = 12000                      # showers to load (shipped cache holds 12k); bump if you mount more
N_EBINS = 12                        # incident energy bucketed into this many class bins
EPOCHS = 8                          # small baseline default; bump it once it trains
BATCH = 128
set_seed(SEED)
device = get_device()
print(f"device={device}  voxels={LAYERS}x{RAD}x{ANG}  energy_bins={N_EBINS}  epochs={EPOCHS}")
"""),

    # ----------------------------------------------------------------- Part A
    md("""
## Part A · The data, and a held-out reference

`load_calochallenge_ds2` returns `(showers, incident_energy)`: showers are
`(N, 45, 9, 16)` energy grids in **GeV** (layers, radial, angular), and
`incident_energy` is a 1-D array of incident energies in GeV. This is **real data
only** — run `python tools/prepare_data.py calochallenge` once to download and
skim Dataset 2 into `data/calochallenge_ds2.hdf5`; there is no synthetic fallback.

We split off a **reference** half — you never train on it; it's what you score
your generated showers *against* all day (the final grade recomputes the headline
metric on a separate held-out set server-side). Two preprocessing choices:

1. **Pad** the radial axis 9 → 16 so every layer is a 16×16 image — the diffusion
   UNet sees the 45 layers as 45 input **channels**. We crop back to 9 after sampling.
2. **log-normalise** the (huge dynamic range) energies to ~`[-1, 1]`, and **bin**
   the continuous incident energy into `N_EBINS` classes for the UNet's conditioning.
"""),
    setup("""
showers, incident = load_calochallenge_ds2(max_showers=N_LOAD)   # (N,45,9,16) GeV, (N,) GeV

# Reference half (for scoring) vs training half — never train on the reference.
half = len(showers) // 2
ref_showers, ref_inc = showers[:half], incident[:half]
trn_showers, trn_inc = showers[half:], incident[half:]
print("train:", trn_showers.shape, "| ref:", ref_showers.shape)

# Pad/crop the radial axis (9 <-> 16) so each layer is a 16x16 image for the UNet.
def pad_rad(x):
    out = np.zeros(x.shape[:-2] + (PAD, ANG), dtype=x.dtype)
    out[..., :RAD, :] = x
    return out

def crop_rad(x):
    return x[..., :RAD, :]

# log1p + global-max scaling -> ~[-1, 1]; round-trip back to physical GeV.
LOG_MAX = float(np.log1p(showers).max())

def to_image(e_gev):
    return (np.log1p(e_gev) / LOG_MAX) * 2.0 - 1.0

def from_image(x):
    x = np.asarray(x)
    return np.expm1(((x + 1.0) / 2.0) * LOG_MAX)

X = to_image(pad_rad(trn_showers))   # (N, 45, 16, 16)

# Bin incident energy uniformly in log-energy.
log_e = np.log(incident)
EDGES = np.linspace(log_e.min(), log_e.max() + 1e-6, N_EBINS + 1)

def energy_to_bin(e_gev):
    return np.clip(np.digitize(np.log(e_gev), EDGES) - 1, 0, N_EBINS - 1)

ebin = energy_to_bin(trn_inc).astype("int64")
print("X:", X.shape, "| energy bins populated:", len(np.unique(ebin)), "/", N_EBINS)
"""),

    md("""
We pair each training shower with its energy-bin label, build the conditional
UNet (`in_channels=45` for the layers, `class_embed=True` for the energy), a
cosine DDPM scheduler, and optionally warm-start from
`checkpoints/calo_diffusion_ds2.pt` so a stalled student still has a working model
to analyse and submit.
"""),
    setup("""
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

X_t = torch.as_tensor(X, dtype=torch.float32)
y_t = torch.as_tensor(ebin, dtype=torch.long)
calo_loader = DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH, shuffle=True)

unet = make_unet(sample_size=PAD, in_channels=LAYERS, base_channels=32,
                 class_embed=True, num_classes=N_EBINS).to(device)
scheduler = make_scheduler(num_train_timesteps=1000)
T_TRAIN = scheduler.config.num_train_timesteps

CKPT = Path("..") / "checkpoints" / "calo_diffusion_ds2.pt"
if CKPT.exists():
    unet.load_state_dict(torch.load(CKPT, map_location=device))
    print("warm-started from checkpoint:", CKPT)
else:
    print("no checkpoint; will train a small baseline in Exercise 1")
print("UNet params:", sum(p.numel() for p in unet.parameters()))
"""),

    md("""
### Exercise 1 — Train the conditional baseline

Diffusion training is "predict the noise", **conditioned on the energy bin**. For
a batch of shower grids `x0` (shape `(B, 45, 16, 16)`) and energy-bin labels:

1. sample random timesteps `t` and Gaussian `noise`,
2. noise the grids with `scheduler.add_noise(x0, noise, t)`,
3. predict the noise with the UNet, passing the condition via `class_labels=labels`,
4. return the **MSE** between predicted and true noise.

Implement `calo_step`, train for `EPOCHS`, and report the **final-epoch mean
training loss** as `final_loss` (a number). This is your first milestone — once it
runs you have a generator to sample from.
"""),
    exercise(
        scaffold_body="""
def calo_step(model, batch):
    x0, labels = batch                       # (B,45,16,16) in [-1,1], energy-bin ints
    noise = torch.randn_like(x0)
    t = torch.randint(0, T_TRAIN, (x0.shape[0],), device=x0.device).long()
    # 1) noise the grids to step t:
    noisy = ...
    # 2) predict noise, conditioned on the energy bin (class_labels=...):
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

epochs = 1 if CKPT.exists() else EPOCHS      # warm-started? just a short polish
history = train(unet, calo_loader, calo_step, epochs=epochs, lr=2e-4, device=device)
final_loss = round(float(history[-1]), 4)
print("final noise-MSE loss:", final_loss)
""",
        answer_var="final_loss",
    ),

    md("""
We provide `sample_showers(bins)` — the reverse diffusion loop, conditioned on the
energy bins, returning showers cropped back to the physical `(N, 45, 9, 16)` grid
**in GeV**. You'll use it to generate a matched set against the reference.
"""),
    setup("""
@torch.no_grad()
def sample_showers(bins, n_steps=50):
    \"\"\"Generate one shower per entry in `bins`. Returns (len(bins),45,9,16) GeV.\"\"\"
    unet.eval()
    labels = torch.as_tensor(np.asarray(bins), dtype=torch.long, device=device)
    x = torch.randn(len(labels), LAYERS, PAD, ANG, device=device)
    scheduler.set_timesteps(n_steps)
    for t in scheduler.timesteps:
        eps = unet(x, t, class_labels=labels).sample
        x = scheduler.step(eps, t, x).prev_sample
    grid = from_image(x.cpu().numpy())          # (n,45,16,16) GeV
    return np.clip(crop_rad(grid), 0.0, None)   # -> (n,45,9,16)

# Generate a set matched to the reference showers' energies — reuse this below.
ref_bins = energy_to_bin(ref_inc)
gen_showers = sample_showers(ref_bins)
print("generated", gen_showers.shape, "showers matched to the reference energies")
"""),

    # ----------------------------------------------------------------- Part B
    md("""
## Part B · Self-score with the harness

`src.calo_eval` is your scoring harness — the same toolkit the final grade uses.
Score your generated showers against the held-out reference all day:

- **`score_showers(gen, gen_E, ref, ref_E)`** → `auc`, per-observable `w1`,
  `response_slope`, `mean_response`,
- **`observables(...)`** → per-shower physics (total energy, response, depth over
  the 45 layers, radial width, sparsity) to histogram real-vs-generated,
- **`classifier_two_sample_auc(ref, gen)`** → the headline number directly.
"""),

    md("""
### Exercise 2 — The headline metric (classifier two-sample AUC)

The CaloChallenge yardstick: train the best small classifier you can to tell real
showers from your generated ones. **AUC near 0.5 means it can't → your showers are
indistinguishable → a good fast simulator.** AUC toward 1.0 means it found a tell.

Call `calo_eval.classifier_two_sample_auc(ref_showers, gen_showers)` and return the
result as `auc` (a number). Drive this **toward 0.5** as you improve your model.
"""),
    exercise(
        scaffold_body="""
# calo_eval.classifier_two_sample_auc(real, generated) -> float (AUC; ~0.5 is best)
auc = ...
""",
        solution_body="""
auc = round(calo_eval.classifier_two_sample_auc(ref_showers, gen_showers), 3)
print("classifier two-sample-test AUC (0.5 = indistinguishable):", auc)
""",
        answer_var="auc",
    ),

    md("""
### Exercise 3 — Physics observables: where are you off?

The AUC says *whether* you're off; the physics says *where*. Compute the
per-observable 1-Wasserstein distances between your generated showers and the
reference with `calo_eval.observable_w1s(...)`, **draw the energy-response overlay**
(real vs generated total deposited energy), and return the **total-energy W1** as
`energy_w1` (a number — lower is better).

The overlay is also your `capstone.response` submission plot below.
"""),
    exercise(
        scaffold_body="""
# calo_eval.observable_w1s(gen, gen_E, ref, ref_E) -> {observable: W1}
w1 = ...
energy_w1 = ...   # the "total_energy" entry, rounded to 4 dp
""",
        solution_body="""
w1 = calo_eval.observable_w1s(gen_showers, ref_inc, ref_showers, ref_inc)
energy_w1 = round(w1["total_energy"], 4)
print("per-observable W1 (lower better):", {k: round(v, 4) for k, v in w1.items()})

# energy-response overlay: deposited-energy spectra, real vs generated
real_E = calo_eval.total_energy(ref_showers)
gen_E = calo_eval.total_energy(gen_showers)
plt.figure(figsize=(5, 4))
bins = np.linspace(0, float(real_E.max()), 40)
plt.hist(real_E, bins=bins, alpha=0.5, density=True, label="real")
plt.hist(gen_E, bins=bins, alpha=0.5, density=True, label="generated")
plt.xlabel("total deposited energy [GeV]"); plt.ylabel("density")
plt.title(f"energy response  (total-E W1 = {energy_w1})"); plt.legend(); plt.show()
""",
        answer_var="energy_w1",
    ),

    # ----------------------------------------------------------------- Part C
    md("""
## Part C · Submit to the board (cadence)

These cells push your results to the cohort board. They're **non-blocking** and
**don't replace your own cell outputs** — submit early (baseline before lunch!),
then re-run them whenever you improve. Each is wrapped so the notebook still runs
fine offline. Submit:

- **`capstone.auc`** — your headline AUC (number),
- **`capstone.response`** — the energy-response overlay (plot),
- **`capstone.shower`** — a generated shower, summed over depth (plot),
- **`capstone.writeup`** — 2–3 sentences on what you tried (string → `mark_done`).
"""),
    setup("""
# capstone.auc — the headline number
try:
    import cadence
    cadence.check("capstone.auc", auc)
    print("submitted capstone.auc =", auc)
except Exception as e:
    print("auc submission skipped (offline?):", type(e).__name__)
"""),
    setup("""
# capstone.response — the energy-response overlay
real_E = calo_eval.total_energy(ref_showers); gen_E = calo_eval.total_energy(gen_showers)
fig = plt.figure(figsize=(5, 4))
bins = np.linspace(0, float(real_E.max()), 40)
plt.hist(real_E, bins=bins, alpha=0.5, density=True, label="real")
plt.hist(gen_E, bins=bins, alpha=0.5, density=True, label="generated")
plt.xlabel("total deposited energy [GeV]"); plt.ylabel("density")
plt.title("energy response"); plt.legend()
try:
    import cadence
    cadence.submit_image("capstone.response", fig)
    print("submitted capstone.response")
except Exception as e:
    print("response submission skipped (offline?):", type(e).__name__)
plt.show()
"""),
    setup("""
# capstone.shower — a high-energy generated shower, summed over the 45 layers
# into a (radial x angular) face (log scale).
one = sample_showers(np.array([N_EBINS - 1]))[0]   # (45,9,16) GeV
face = one.sum(axis=0)                              # (9,16) energy per (radial,angular)
fig = plt.figure(figsize=(4, 3))
plt.imshow(np.log1p(face), cmap="inferno", aspect="auto")
plt.colorbar(label="log(1 + E[GeV])"); plt.xlabel("angular"); plt.ylabel("radial")
plt.title("generated shower (summed over depth)")
try:
    import cadence
    cadence.submit_image("capstone.shower", fig)
    print("submitted capstone.shower")
except Exception as e:
    print("shower submission skipped (offline?):", type(e).__name__)
plt.show()
"""),

    md("""
### Exercise 4 — Your write-up

Two or three sentences for the board: which **architecture** you went with, what
**helped**, and what **didn't**. Set `writeup` to that string. (This is your
`capstone.writeup` submission — the wrap-up discussion is built from these.)
"""),
    exercise(
        scaffold_body="""
# answer: string
writeup = ...   # e.g. "Energy-conditioned diffusion, 45 layers as channels. More
                #       epochs tightened the response; the deep, sparse layers stayed hard."
""",
        solution_body="""
writeup = ("Energy-conditioned diffusion baseline (diffusers UNet, 45 layers as "
           "channels). More epochs tightened the energy response; the sparse, "
           "high-radius and deepest layers stayed the hardest to match.")
try:
    import cadence
    cadence.mark_done("capstone.writeup")
    print("submitted capstone.writeup")
except Exception as e:
    print("writeup submission skipped (offline?):", type(e).__name__)
""",
        answer_var="writeup",
    ),

    # ----------------------------------------------------------------- Stretch
    md("""
## Stretch goals (for fast finishers)

You have a baseline and a board submission — now beat it. Ideas, roughly in order
of effort:

- **Train more / bigger.** Raise `EPOCHS` and `N_LOAD`, widen `base_channels`,
  more sampling steps. The cheapest wins first.
- **Better conditioning.** More energy bins, or condition on the *continuous*
  energy instead of bins.
- **Respect the geometry.** The angular axis is **periodic** (it wraps) — try
  circular padding; or model the longitudinal profile explicitly.
- **A different architecture.** A conditional **VAE** or **GAN** on the same data
  — then compare head-to-head on the same metrics (great wrap-up material).
- **Physics-informed loss.** Add an energy-conservation penalty (deposited energy
  should track `E_inc`), or a sparsity penalty.
- **Quantify the speed-up.** Time your sampler in **showers/sec** vs a notional
  Geant4 baseline — the actual selling point of fast sim.

Every idea is judged by the *same* harness, so the metric tells you honestly
whether it helped.
"""),

    md("""
## Recap

- You built an **energy-conditioned generative fast simulator** for CaloChallenge
  **Dataset 2** electron showers — a real 3D, 6480-voxel detector geometry.
- You scored it like a physicist: **physics observables + per-observable W1**
  (total energy, longitudinal depth, radial width, sparsity), and the headline
  **classifier two-sample AUC** (≈ 0.5 = indistinguishable).
- You submitted to the cohort board through **cadence** and reflected on what
  helped. At wrap-up we compare everyone's response curves, sparsity, and
  architectures — best **physics insight** and most **creative approach** get
  recognised alongside the best AUC.

Nice work — that's a real, deployed use of generative modelling in HEP. 🎉
"""),
]

build(OUT, cells)
