"""Build notebooks/06_project.ipynb — Project 1: a JetNet gluon-jet mini capstone.

This is the in-session project notebook. Students have now seen VAEs (01), GANs
(02) and diffusion (03/05); here they pick ANY architecture and try to generate
realistic JetNet **gluon** jets. The headline, auto-checkable metric is the
jet-mass 1-Wasserstein distance (`src.jetmass.jet_mass_w1`) — lower is better, so
it doubles as a friendly leaderboard.

The notebook ships a small working baseline VAE so nobody is stuck on a blank
page; the exercises walk from "load the real jets" through "train a generator"
to "score your leaderboard W1", and finish with a couple of reflective answers.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nbbuild import build, md, code, setup, exercise

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "notebooks" / "06_project.ipynb"

cells = [
    md("""
# 06 · Project 1 — Generate gluon jets 🌫️

**Generative Modelling for HEP** — notebook 6 of 6 · *in-session mini capstone (~1.5h)*

You've built a VAE (01), a GAN (02) and diffusion models (03, 05). Time to put it
together on a real problem and **compete a little**.

### The goal
Generate **JetNet gluon jets** — point clouds of 30 particles, each
`(eta_rel, phi_rel, pt_rel)` — that look like the real thing. Use **any
architecture you like**: a VAE, a GAN, a diffusion model, or something hybrid.

### The leaderboard rule 🏆
We score generators by the **jet-mass 1-Wasserstein distance**,
`src.jetmass.jet_mass_w1(real, generated)` — **lower is better**. The bulk of the
mass spectrum is easy; the high-mass tail is hard. That's where the medals are.

> In the real school you'd *submit* your `mass_w1` to the leaderboard. Here we
> just **compute and print it** — no submission magic needed.

### How this notebook works
- A small **baseline VAE** is provided and trained for you in a setup cell — it
  already produces (mediocre) gluon jets, so you always have something to score.
- The exercises take you from loading the data, to training a generator, to
  scoring your leaderboard metric, with a couple of reflective answers at the end.
- **Tips & scale-up ideas** are sprinkled through — go past the baseline if you
  have time!

You complete the cells marked **Exercise**. Everything else runs as-is.
"""),

    md("## Setup"),
    setup("""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import sys; sys.path.insert(0, "..")
from src.seeds import set_seed
from src.train import get_device, to_loader, train
from src.data import load_jetnet
from src.gnn import DeepSetsEncoder, DeepSetsDecoder
from src.jetmass import jet_mass, plot_jet_mass, jet_mass_w1

SEED = 0
LATENT_DIM = 16
N_PARTICLES = 30
EPOCHS = 8            # small default; bump it once your generator works
BATCH = 128
set_seed(SEED)
device = get_device()
print(f"device={device}  latent_dim={LATENT_DIM}  epochs={EPOCHS}")
"""),

    # ----------------------------------------------------------------- Part A
    md("""
## Part A · Meet the data

Gluon jets have more, softer constituents than quark jets, which pushes their
**jet-mass** spectrum higher. We load the real JetNet gluon jets (with a
synthetic fallback so this always runs), hold some out as a test set, and look at
the target mass distribution you're trying to match.
"""),
    setup("""
gluon = load_jetnet("g", num_particles=N_PARTICLES, max_jets=6000).astype(np.float32)

# A simple train/test split: train your generator on `train_jets`, always SCORE
# your leaderboard W1 against the untouched `test_jets`.
n_test = 1500
test_jets = gluon[:n_test]
train_jets = gluon[n_test:]
jet_loader = to_loader(train_jets, batch_size=BATCH)

print("gluon jets:", gluon.shape,
      "| train:", train_jets.shape, "| test:", test_jets.shape)
"""),

    md("""
### Exercise 1 — Know your target

Before modelling anything, characterise the real gluon jets. Compute the
**median jet mass** of the held-out `test_jets` (using `jet_mass`) and return it
as `median_real_mass`, rounded to 4 dp.

This is the number your generated jets should cluster around — a quick gut-check
for later.
"""),
    exercise(
        scaffold_body="""
# jet_mass(jets) -> array of per-jet relative masses
masses = ...
median_real_mass = ...
""",
        solution_body="""
masses = jet_mass(test_jets)
median_real_mass = round(float(np.median(masses)), 4)
print("median real gluon jet mass:", median_real_mass)
""",
        answer_var="median_real_mass",
    ),

    # ----------------------------------------------------------------- Part B
    md("""
## Part B · The baseline generator

Here is a small **DeepSets VAE** baseline, reusing the permutation-invariant GNN
blocks from notebook 01. It's intentionally modest so you have room to beat it.
We define it and train it for you below, then you'll improve and score it.

**Architecture (the baseline):**
- Encoder: `DeepSetsEncoder` → `(mu, logvar)` in latent space.
- Decoder: `DeepSetsDecoder` → a 30-particle cloud.
- Loss: per-feature reconstruction MSE + a small KL term.

**Scale-up ideas** (if you have time): widen `hidden`, raise `EPOCHS`, anneal the
KL weight, swap in a GAN critic for sharper tails, or try a diffusion decoder.
"""),
    setup("""
class BaselineVAE(nn.Module):
    def __init__(self, latent=LATENT_DIM, n_particles=N_PARTICLES):
        super().__init__()
        self.encoder = DeepSetsEncoder(in_features=3, hidden=128, latent=latent * 2)
        self.decoder = DeepSetsDecoder(latent=latent, n_particles=n_particles)
        self.latent = latent

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = h[:, :self.latent], h[:, self.latent:]
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decoder(z), mu, logvar

    @torch.no_grad()
    def sample(self, n):
        z = torch.randn(n, self.latent, device=next(self.parameters()).device)
        return self.decoder(z)


def vae_step(model, batch):
    x = batch[0]
    x_hat, mu, logvar = model(x)
    recon = F.mse_loss(x_hat, x)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + 1e-3 * kl


model = BaselineVAE().to(device)
print("baseline params:", sum(p.numel() for p in model.parameters()))
"""),

    md("""
### Exercise 2 — Train your generator

Train the baseline (or your own improved version!) and report its **final-epoch
mean loss** as `final_loss`, rounded to 4 dp.

`train(model, loader, step_fn, epochs=..., lr=..., device=...)` returns the
per-epoch loss history. The provided `vae_step` is the objective; if you've built
a different model, pass its own step function instead.
"""),
    exercise(
        scaffold_body="""
# train(...) returns a list: the mean loss for each epoch.
history = ...
final_loss = ...
""",
        solution_body="""
history = train(model, jet_loader, vae_step, epochs=EPOCHS, lr=1e-3, device=device)
final_loss = round(float(history[-1]), 4)
print("final training loss:", final_loss)
""",
        answer_var="final_loss",
    ),

    md("""
### Exercise 3 — Score the leaderboard 🏆

The headline metric. **Generate** as many jets as the test set, **plot** the
jet-mass overlay (real gluon vs your generated jets — the recurring spine), and
return your leaderboard score `mass_w1 = jet_mass_w1(test_jets, gen)`, rounded to
4 dp. **Lower is better.**

The baseline won't win any medals — that's the point. Look at the overlay: where
is it off? Then come back and improve the model to drive this number down.
"""),
    exercise(
        scaffold_body="""
model.eval()
with torch.no_grad():
    # generate len(test_jets) jets from your trained model
    gen = ...
gen = np.asarray(gen) if not isinstance(gen, np.ndarray) else gen
plot_jet_mass(test_jets, gen, labels=("real gluon", "generated")); plt.show()
mass_w1 = ...
""",
        solution_body="""
model.eval()
with torch.no_grad():
    gen = model.sample(len(test_jets)).cpu().numpy()
plot_jet_mass(test_jets, gen, labels=("real gluon", "generated")); plt.show()
mass_w1 = round(jet_mass_w1(test_jets, gen), 4)
print("leaderboard jet-mass W1 (lower is better):", mass_w1)
""",
        answer_var="mass_w1",
    ),

    # ----------------------------------------------------------------- Part C
    md("""
## Part C · Reflect & report

Two quick answers to wrap up your submission: which family of model you went
with, and a snapshot of your generated mass spectrum.
"""),

    md("""
### Exercise 4 — Declare your architecture

Which generative family did your submitted model belong to? Set
`model_choice` to one of `"vae"`, `"gan"` or `"diffusion"`. (The baseline is a
VAE — change this if you swapped it out.)
"""),
    exercise(
        scaffold_body="""
# answer: string
model_choice = ...   # "vae", "gan", or "diffusion"
""",
        solution_body="""
model_choice = "vae"
""",
        answer_var="model_choice",
    ),

    md("""
### Exercise 5 — Summarise your generated mass

Report the shape of your generated mass distribution: compute the **mean** and
**standard deviation** of `jet_mass(gen)` and return them as a two-element list
`mass_summary = [mean, std]`, each rounded to 4 dp.

Compare the mean against `median_real_mass` from Exercise 1 — close means your
generator centres the spectrum well; a small std relative to the real jets is the
classic "too blurry" tell.
"""),
    exercise(
        scaffold_body="""
# answer: list
gen_masses = jet_mass(gen)
mean_mass = ...
std_mass = ...
mass_summary = [mean_mass, std_mass]
""",
        solution_body="""
gen_masses = jet_mass(gen)
mean_mass = round(float(np.mean(gen_masses)), 4)
std_mass = round(float(np.std(gen_masses)), 4)
mass_summary = [mean_mass, std_mass]
print("generated mass [mean, std]:", mass_summary)
""",
        answer_var="mass_summary",
    ),

    md("""
## Wrap-up

- You generated gluon jets end-to-end and scored them on a single, honest metric:
  the **jet-mass 1-Wasserstein distance**.
- The baseline VAE gets the bulk roughly right but smears the **high-mass tail** —
  the recurring lesson of these notebooks.
- **To climb the leaderboard:** train longer, widen the networks, anneal the KL,
  or bring in a GAN critic / diffusion decoder for sharper tails. The metric will
  tell you honestly whether it helped.

Nice work — that's the full arc from VAE to GAN to diffusion to your own
generator. 🎉
"""),
]

build(OUT, cells)
