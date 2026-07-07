# CaloDiffusion interpretability capstone

A ~2-day capstone where students **probe a pretrained diffusion model** that generates
calorimeter showers — they don't train it, they open it up and figure out what it learned.

Everything for the project lives in this one folder.

## Credit

The model, architecture, and pretrained checkpoints are from:

> O. Amram & K. Pedro, *"Denoising diffusion models with geometry adaptation for high
> fidelity calorimeter simulation"*, **Phys. Rev. D 108, 072014 (2023)**,
> [arXiv:2308.03876](https://arxiv.org/abs/2308.03876).
> Code: https://github.com/OzAmram/CaloDiffusionPaper (+ its `CaloChallenge` helper).

Their code is vendored under [`vendor/`](vendor/), essentially unmodified — the only change is
6 `einops.rearrange` calls in `models.py` rewritten with plain `torch` (marked with comments)
so the project needs **no extra dependencies** beyond the course env. The data are skims of the
public [CaloChallenge 2022](https://calochallenge.github.io/homepage/) datasets (Zenodo
records 6366270 for electrons, 8099322 for photons/pions).

## What's here

```
calodiff_probe.py            our friendly wrapper — the only thing students import
capstone_calodiffusion.ipynb scaffolded starter notebook (warm-up + Track B worked)
JUMPING_OFF_POINTS.md        the 5 project tracks (A score field … E EM-vs-hadronic)
data/                        committed shower caches (electron, photon, pion; 12k each)
vendor/                      the paper's code, unmodified
  scripts/                     CaloDiffu, models, utils, consts (+ plot)
  configs/                     per-dataset config JSONs
  trained_models/              dataset2 / dataset1_photon / dataset1_pion checkpoints
  CaloChallenge/code/          geometry XMLHandler + binning XMLs
```

All three shower caches (electron `calochallenge_ds2.hdf5`, photon/pion
`calochallenge_ds1_*.hdf5`) live in this folder's `data/`, so the project is self-contained.

## Requirements

**None beyond the course env** — just `torch`, `numpy`, `h5py`, `pyyaml`, `scikit-learn`,
`matplotlib` (all already present). `einops` was removed (rewritten in torch) and `torchinfo`
is auto-stubbed if missing.

## Quickstart

```python
import calodiff_probe as cp
model = cp.load_model("electron")            # or "photon" / "pion"
x, E, E_inc = cp.load_showers("electron", n=64)
x_t, eps = cp.noise_at(model, x, t=100)      # forward (noising) process
eps_hat  = cp.predict_noise(model, x_t, E, t=100)   # the model's noise estimate (the "score")
grid = cp.regular_grid(model, x)             # (N, layers, angular, radial) for plotting
```

Runs on GPU automatically when available (`CALODIFF_DEVICE=cuda|cpu` to force). Probing is
instant on CPU; **generating** showers (`cp.sample`) is slow on CPU — use the GPU for the
generation-heavy tracks (C, D).

Start from `capstone_calodiffusion.ipynb`, then pick a track from `JUMPING_OFF_POINTS.md`.
