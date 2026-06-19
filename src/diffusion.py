"""Diffusion helpers, in two layers.

1. From-scratch DDPM math (`make_beta_schedule`, `q_sample`, `ddpm_p_losses`,
   `ddpm_sample`) in plain torch — students see "predict the noise" with their
   own eyes once (the 2D toy in notebook 03).
2. A thin `diffusers` wrapper (`make_unet`, `make_scheduler`) for the heavier
   image/voxel models (creature sprites in 03, calorimeter showers in 04), so
   the real tooling does the heavy lifting after the mechanism is understood.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# From-scratch DDPM (the 2D toy)
# --------------------------------------------------------------------------- #

def make_beta_schedule(timesteps=200, beta_start=1e-4, beta_end=2e-2):
    """Linear variance schedule + the derived alphas. Returns a dict of 1-D
    tensors indexed by timestep."""
    betas = torch.linspace(beta_start, beta_end, timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_acp": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_acp": torch.sqrt(1.0 - alphas_cumprod),
    }


def q_sample(x0, t, sched, noise=None):
    """Forward (noising) process: sample x_t given x_0 in one shot.
    x_t = sqrt(acp) * x0 + sqrt(1 - acp) * eps."""
    if noise is None:
        noise = torch.randn_like(x0)
    a = sched["sqrt_acp"][t].view(-1, *([1] * (x0.ndim - 1)))
    b = sched["sqrt_one_minus_acp"][t].view(-1, *([1] * (x0.ndim - 1)))
    return a * x0 + b * noise, noise


def ddpm_p_losses(model, x0, t, sched):
    """The whole training objective: noise x0 to step t, ask the model to
    predict the noise, return the MSE. This single function is the heart of
    DDPM."""
    x_t, noise = q_sample(x0, t, sched)
    pred = model(x_t, t)
    return F.mse_loss(pred, noise)


@torch.no_grad()
def ddpm_sample(model, shape, sched, device="cpu", record_every=0):
    """Reverse (denoising) process. Returns the final sample, or
    (final, trajectory) when `record_every > 0` — the trajectory is what the
    notebooks animate into a 'emerging from noise' gif."""
    T = len(sched["betas"])
    x = torch.randn(shape, device=device)
    betas, alphas, acp = sched["betas"], sched["alphas"], sched["alphas_cumprod"]
    traj = []
    for i in reversed(range(T)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        eps = model(x, t)
        coef = (1 - alphas[i]) / sched["sqrt_one_minus_acp"][i]
        mean = (x - coef * eps) / torch.sqrt(alphas[i])
        x = mean if i == 0 else mean + torch.sqrt(betas[i]) * torch.randn_like(x)
        if record_every and (i % record_every == 0 or i == 0):
            traj.append(x.clone())
    return (x, traj) if record_every else x


# --------------------------------------------------------------------------- #
# diffusers wrapper (the heavy image / voxel models)
# --------------------------------------------------------------------------- #

def make_unet(sample_size=32, in_channels=3, out_channels=None,
              base_channels=64, class_embed=False, num_classes=None):
    """A modest `diffusers` UNet2DModel sized for live training on one GPU.
    Set `class_embed=True` (with `num_classes`) for conditioning — used by the
    energy-conditioned calorimeter model in notebook 04."""
    from diffusers import UNet2DModel
    kw = {}
    if class_embed:
        kw.update(num_class_embeds=num_classes)
    return UNet2DModel(
        sample_size=sample_size,
        in_channels=in_channels,
        out_channels=out_channels or in_channels,
        layers_per_block=2,
        block_out_channels=(base_channels, base_channels, base_channels * 2),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        **kw,
    )


def make_scheduler(num_train_timesteps=1000, schedule="squaredcos_cap_v2"):
    """A `diffusers` DDPM scheduler. The cosine (`squaredcos_cap_v2`) schedule
    trains more stably on small models than plain linear."""
    from diffusers import DDPMScheduler
    return DDPMScheduler(num_train_timesteps=num_train_timesteps,
                         beta_schedule=schedule)
