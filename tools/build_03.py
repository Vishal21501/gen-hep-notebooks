"""Build notebooks/03_diffusion_intro.ipynb — DDPM (denoising diffusion).

Mechanism first: a 2D toy from scratch in plain torch. Students predict the
noise once with their own eyes (forward noising -> reverse denoising) on a
synthetic 2D point distribution, with a 2-layer MLP noise predictor.

Fun spine: creature sprites / "fakemon" with diffusers (UNet + scheduler) — a
brand-new monster emerges from noise, and we animate the half-formed stages as
a gif (a side effect; the exercise cell still ends on a primitive scalar).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nbbuild import build, md, code, setup, exercise

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "notebooks" / "03_diffusion_intro.ipynb"

cells = [
    md("""
# 03 · Denoising Diffusion (DDPM)

**Generative Modelling for HEP** — notebook 3 of 6.

A diffusion model learns to *reverse noise*. The **forward** process slowly
corrupts data into pure Gaussian noise over many small steps; a network then
learns the **reverse** process — at each step it looks at a noisy sample and
**predicts the noise** that was added, so we can subtract a little of it and
walk back toward clean data.

We do the **mechanism first**, by hand, on a tiny **2D point cloud** (eight
Gaussians) with a 2-layer MLP — you'll watch points dissolve into noise and
then re-condense. Only *then* do we reach for `diffusers` to grow a brand-new
**creature sprite** 🐲 out of noise and animate it emerging frame by frame.

> You complete the cells marked **Exercise**. Everything else runs as-is.
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
from src.data import load_images
from src.diffusion import (make_beta_schedule, q_sample, ddpm_p_losses,
                           ddpm_sample, make_unet, make_scheduler)

SEED = 0
TIMESTEPS = 200       # diffusion steps for the 2D toy
EPOCHS = 40           # 2D toy trains in seconds; sprite model is configured below
BATCH = 256
set_seed(SEED)
device = get_device()
print(f"device={device}  timesteps={TIMESTEPS}  epochs={EPOCHS}")
"""),

    # ----------------------------------------------------------------- Part A
    md("""
## Part A · The mechanism, by hand, in 2D

Before any fancy image model, we build the whole DDPM loop on a 2D toy so the
moving parts are visible. We make an **eight-Gaussians** ring inline with numpy,
set up a **linear variance schedule** (`make_beta_schedule`), and define a tiny
**MLP noise predictor** that takes a noisy point `x_t` and the timestep `t` and
outputs its guess of the noise.
"""),
    setup("""
def eight_gaussians(n=4000, radius=4.0, std=0.25, seed=0):
    \"\"\"A ring of 8 Gaussian blobs — a classic 2D generative-model testbed.\"\"\"
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    centers = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    which = rng.integers(0, 8, size=n)
    pts = centers[which] + rng.normal(0, std, size=(n, 2))
    return pts.astype(np.float32)

points = eight_gaussians(n=4000, seed=SEED)
# normalise to unit-ish scale so the schedule (which ends near unit variance) fits
data_mean = points.mean(0, keepdims=True)
data_std = points.std(0, keepdims=True)
points_norm = (points - data_mean) / data_std
X2d = torch.as_tensor(points_norm, dtype=torch.float32)
toy_loader = to_loader(X2d, batch_size=BATCH)

sched = make_beta_schedule(timesteps=TIMESTEPS)        # dict of 1-D tensors

fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(points[:, 0], points[:, 1], s=4, alpha=0.4)
ax.set_title("8 Gaussians (real 2D data)"); ax.set_aspect("equal"); plt.show()


class ToyMLP(nn.Module):
    \"\"\"2-layer MLP noise predictor: (x_t, t) -> predicted noise, shape (B, 2).
    The timestep enters as a single normalised scalar feature.\"\"\"
    def __init__(self, hidden=128, timesteps=TIMESTEPS):
        super().__init__()
        self.timesteps = timesteps
        self.net = nn.Sequential(
            nn.Linear(2 + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2))

    def forward(self, x, t):
        t_feat = (t.float() / self.timesteps).view(-1, 1)
        return self.net(torch.cat([x, t_feat], dim=1))

toy_model = ToyMLP().to(device)
print("ToyMLP params:", sum(p.numel() for p in toy_model.parameters()))
"""),

    md("""
### Exercise 1 — Forward noising with `q_sample`

The forward process adds noise in one shot:
$$x_t = \\sqrt{\\bar\\alpha_t}\\, x_0 + \\sqrt{1-\\bar\\alpha_t}\\, \\epsilon,
  \\qquad \\epsilon \\sim \\mathcal N(0, I).$$

Use `q_sample(x0, t, sched)` to noise the **whole** 2D dataset to the **last**
timestep (`T-1`). By construction the data was standardised, so at the final
step `x_t` should be almost pure unit-variance noise. Compute the **empirical
variance** of the noised points and return it as `var_last` — it should land
close to **1.0**.
"""),
    exercise(
        scaffold_body="""
T = TIMESTEPS
x0 = X2d.to(device)
# a full batch all sitting at the final timestep T-1:
t_last = ...                         # (N,) long tensor, every entry = T-1
x_t, eps = q_sample(x0, t_last, sched)
# empirical variance of the noised coordinates (a single scalar):
var_last = ...
""",
        solution_body="""
T = TIMESTEPS
x0 = X2d.to(device)
t_last = torch.full((x0.shape[0],), T - 1, device=device, dtype=torch.long)
x_t, eps = q_sample(x0, t_last, sched)
var_last = float(x_t.var())

fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for ax, ti in zip(axes, [0, T // 4, T // 2, T - 1]):
    tt = torch.full((x0.shape[0],), ti, device=device, dtype=torch.long)
    xi = q_sample(x0, tt, sched)[0].cpu().numpy()
    ax.scatter(xi[:, 0], xi[:, 1], s=3, alpha=0.3)
    ax.set_title(f"t={ti}"); ax.set_aspect("equal")
fig.suptitle("forward noising: data -> noise"); plt.show()
var_last = round(var_last, 3)
print("variance at last timestep:", var_last)
""",
        answer_var="var_last",
    ),

    md("""
### Exercise 2 — Train the noise predictor (the DDPM objective)

Training a diffusion model is one line of intuition: **noise a sample to a
random step `t`, ask the model to predict the noise, minimise the MSE.** That
is exactly what `ddpm_p_losses(model, x0, t, sched)` does.

Write the `step_fn` that draws a **random timestep per sample** and returns the
DDPM loss, train the `ToyMLP`, and report the **final epoch's mean loss** as
`final_loss`.
"""),
    exercise(
        scaffold_body="""
def ddpm_step(model, batch):
    x0 = batch[0]
    # one random timestep per sample, in [0, TIMESTEPS):
    t = ...
    return ddpm_p_losses(model, x0, t, sched)
""",
        solution_body="""
def ddpm_step(model, batch):
    x0 = batch[0]
    t = torch.randint(0, TIMESTEPS, (x0.shape[0],), device=x0.device)
    return ddpm_p_losses(model, x0, t, sched)

# move the schedule tensors onto the right device for indexing inside q_sample:
sched = {k: v.to(device) for k, v in sched.items()}
history = train(toy_model, toy_loader, ddpm_step, epochs=EPOCHS, lr=2e-3, device=device)
final_loss = round(float(history[-1]), 4)

plt.figure(figsize=(5, 3))
plt.plot(history); plt.xlabel("epoch"); plt.ylabel("noise-MSE")
plt.title("2D DDPM training loss"); plt.show()
print("final training loss:", final_loss)
""",
        answer_var="final_loss",
    ),

    md("""
### Exercise 3 — Reverse denoising: does it recover the data?

Now run the **reverse** process. `ddpm_sample(model, shape, sched, device)`
starts from pure noise and denoises step-by-step back to (hopefully) the data
manifold. Generate **2000** points, plot them over the real data, and measure
how well they match by comparing the **per-coordinate standard deviation** of
generated vs real (both should be ~1 in normalised space).

Set `recovered` (a **bool**) to `True` if the mean absolute difference in those
two standard deviations is below `0.25` — i.e. the reverse process actually
reconstructed the spread of the distribution.
"""),
    exercise(
        scaffold_body="""
# answer: bool
toy_model.eval()
gen = ddpm_sample(toy_model, (2000, 2), sched, device=device).cpu().numpy()
gen_std = gen.std(0)               # per-coordinate std of generated points
real_std = X2d.numpy().std(0)      # per-coordinate std of real (normalised) points
std_gap = ...                      # mean absolute difference of the two stds
recovered = ...                    # bool: is std_gap < 0.25 ?
""",
        solution_body="""
toy_model.eval()
gen = ddpm_sample(toy_model, (2000, 2), sched, device=device).cpu().numpy()
gen_std = gen.std(0)
real_std = X2d.numpy().std(0)
std_gap = float(np.mean(np.abs(gen_std - real_std)))
recovered = bool(std_gap < 0.25)

# de-normalise back to the original coordinates just for a fair-looking overlay
gen_real = gen * data_std + data_mean
fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(points[:, 0], points[:, 1], s=4, alpha=0.3, label="real")
ax.scatter(gen_real[:, 0], gen_real[:, 1], s=4, alpha=0.3, label="generated")
ax.legend(); ax.set_aspect("equal"); ax.set_title("reverse denoising"); plt.show()
print("std gap:", round(std_gap, 3), "| recovered:", recovered)
""",
        answer_var="recovered",
    ),

    # ----------------------------------------------------------------- Part B
    md("""
## Part B · Grow a creature from noise 🐲

The 2D toy showed the whole mechanism. For images we let `diffusers` do the
heavy lifting: a `UNet2DModel` noise predictor and a `DDPMScheduler`. We load
**creature sprites** — openly-licensed **Twemoji** (CC-BY) animals & monsters,
augmented with flips/rotations (synthetic blobs if the sprite cache isn't built
yet) — treat them as `(N, 3, 32, 32)` in `[0, 1]`, and train the diffuser
briefly. Then we sample a brand-new monster and **watch it emerge from noise**.
"""),
    setup("""
sprites = load_images("sprites", n=2000, image_size=32)   # (N, 3, 32, 32) in [0,1]
S = torch.as_tensor(sprites, dtype=torch.float32)
# diffusers UNets expect inputs roughly in [-1, 1]:
S = S * 2.0 - 1.0
sprite_loader = to_loader(S, batch_size=64)

unet = make_unet(sample_size=32, in_channels=3, base_channels=64).to(device)
scheduler = make_scheduler(num_train_timesteps=1000)
print("sprites:", tuple(S.shape),
      "| UNet params:", sum(p.numel() for p in unet.parameters()))


def sprite_step(model, batch):
    \"\"\"diffusers training step: add scheduler noise at random t, predict it.\"\"\"
    x0 = batch[0]
    noise = torch.randn_like(x0)
    t = torch.randint(0, scheduler.config.num_train_timesteps,
                      (x0.shape[0],), device=x0.device).long()
    noisy = scheduler.add_noise(x0, noise, t)
    pred = model(noisy, t).sample
    return F.mse_loss(pred, noise)


@torch.no_grad()
def sprite_sample(model, scheduler, n=4, record_every=100):
    \"\"\"Reverse the scheduler from pure noise, recording intermediate frames.
    Returns (final_images, frames) with images in [0, 1].\"\"\"
    model.eval()
    x = torch.randn(n, 3, 32, 32, device=device)
    frames = []
    for i, t in enumerate(scheduler.timesteps):
        pred = model(x, t).sample
        x = scheduler.step(pred, t, x).prev_sample
        if record_every and (i % record_every == 0 or i == len(scheduler.timesteps) - 1):
            frames.append(((x[0].clamp(-1, 1) + 1) / 2).cpu().numpy())
    final = ((x.clamp(-1, 1) + 1) / 2).cpu().numpy()
    return final, frames

SPRITE_EPOCHS = 6        # brief: a few minutes on one GPU; bump for nicer monsters
print("sprite training config:", SPRITE_EPOCHS, "epochs")
"""),

    md("""
### Exercise 4 — Train the diffuser, animate a monster emerging

Train the sprite UNet briefly with the provided `sprite_step` (use `train(...)`
with `SPRITE_EPOCHS`). Then call `sprite_sample(...)` to denoise from pure noise
while **recording frames every 100 steps**, and stitch those frames into a gif
(`creature_emerges.gif`) so you can watch the monster condense.

The gif is a **side effect** — the cell must end on a primitive. Return the list
of recorded frames' **per-frame mean brightness** as `frame_brightness` (one
float per recorded stage), so its **length** tells you how many frames the
animation has.
"""),
    exercise(
        scaffold_body="""
# answer: list
unet_hist = train(unet, sprite_loader, sprite_step,
                  epochs=SPRITE_EPOCHS, lr=1e-4, device=device)
final_imgs, frames = sprite_sample(unet, scheduler, n=4, record_every=100)
# one brightness value per recorded stage (the gif's frames):
frame_brightness = ...
""",
        solution_body="""
unet_hist = train(unet, sprite_loader, sprite_step,
                  epochs=SPRITE_EPOCHS, lr=1e-4, device=device)
final_imgs, frames = sprite_sample(unet, scheduler, n=4, record_every=100)
frame_brightness = [round(float(np.mean(f)), 3) for f in frames]

# show the final creature + the emergence strip
fig, axes = plt.subplots(1, len(frames), figsize=(2 * len(frames), 2))
for ax, f in zip(np.atleast_1d(axes), frames):
    ax.imshow(np.transpose(np.clip(f, 0, 1), (1, 2, 0))); ax.axis("off")
fig.suptitle("a creature emerging from noise"); plt.show()

# the gif itself — illustrative, guarded in case imageio isn't installed
try:
    import imageio
    gif_frames = [(np.transpose(np.clip(f, 0, 1), (1, 2, 0)) * 255).astype("uint8")
                  for f in frames]
    imageio.mimsave("creature_emerges.gif", gif_frames, duration=0.3)
    print("wrote creature_emerges.gif")
except Exception as e:
    print(f"(gif skipped: {type(e).__name__})")

print("recorded frames:", len(frame_brightness), "| brightness:", frame_brightness)
""",
        answer_var="frame_brightness",
    ),

    md("""
### Exercise 5 — What does the network actually predict?

The single most important idea in DDPM. At each reverse step, the network does
**not** output the clean image directly — it looks at the noisy sample `x_t` and
outputs something we subtract a little of to step back toward the data.

In **one word**, what does the model predict at each step? Set `predicts` to
that word (lower-case).
"""),
    exercise(
        scaffold_body="""
# answer: string
predicts = ...     # one lower-case word
""",
        solution_body="""
predicts = "noise"
""",
        answer_var="predicts",
    ),

    md("""
## Recap

- **Diffusion = learn to reverse noise.** Forward: add Gaussian noise over many
  small steps until data is pure noise. Reverse: a network **predicts the noise**
  and we subtract a little at each step to walk back to data.
- The **training objective is just an MSE on the predicted noise** — you saw it
  on the 2D toy with a 2-layer MLP, and the *same* idea drives the `diffusers`
  UNet that grew a creature sprite from static.
- Diffusion samples are typically **sharper** than a VAE's (notebook 01) and
  **train more stably** than a GAN (notebook 02) — at the cost of many sampling
  steps. We'll put it to work on **calorimeter showers** next (notebook 04).
"""),
]

build(OUT, cells)
