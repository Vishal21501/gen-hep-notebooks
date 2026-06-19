# Generative Modelling for HEP — teaching notebooks

Six teaching notebooks for the generative-modelling track of a high-energy-physics
ML school. Teaching order is **VAE → GAN → diffusion → physics project**. Each
model notebook pairs a playful/visual exercise with a jet-focused physics
extension; notebook 04 adds calorimeter showers as a second data modality.

Students submit through **cadence** (our Jupyter plugin). These are the *teacher*
notebooks — the source of truth. The cadence converter turns them into the
registered + student versions automatically (see "How cadence reads these"
below).

```
notebooks/    01_vae … 06_project        — the teacher notebooks
src/          shared helpers (jet-mass plot, loaders, GNN blocks, diffusion, train loop)
tools/        notebook builders + the converter self-check
data/         (gitignored) downloads land here; loaders fall back to synthetic
checkpoints/  pre-trained jet & calo diffusion models (optional warm-start)
requirements.in        — human-readable env ask
requirements.lock.txt  — pip freeze from the GPU box (generate there; goes to the school)
```

## The notebooks

| File | Goal | Fun spine | Physics |
|---|---|---|---|
| `01_vae.ipynb` | VAE latent space + reconstruction | AFHQ cat→dog latent arithmetic | encode jets (DeepSets GNN), walk quark→gluon |
| `02_gan.ipynb` | GANs + failure modes | QuickDraw doodles, induce mode collapse, fix w/ WGAN-GP | point-cloud jet GAN drops the high-mass tail |
| `03_diffusion_intro.ipynb` | DDPM | 2D toy from scratch; creature sprites emerge from noise (diffusers) | (mechanism focus) |
| `04_diffusion_calo.ipynb` | second modality | — | calorimeter showers, energy-conditioned diffusion |
| `05_evaluation.ipynb` | scoring generators | — | W1 / FPD / KPD + a GNN classifier two-sample test |
| `06_project.ipynb` | in-session mini capstone | — | generate gluon jets, any architecture, jet-mass-W1 leaderboard |

The **jet-mass plot** (`src/jetmass.py`) is the recurring physics spine across
01/02/05/06 — the same axes let students compare a VAE, a GAN and diffusion.

## Environment

GPUs are assumed (default CUDA torch wheel). `requirements.in` is the ask;
**generate the lock file on the GPU provisioning box** and hand that to the
school:

```bash
pip install -r requirements.in
pip freeze > requirements.lock.txt
python -c "import torch, jetnet, diffusers, torch_geometric, sklearn; print(torch.cuda.is_available())"
```

`torch_geometric` is expected from the GNN section's environment. Every data
loader in `src/data.py` has a **synthetic fallback**, so a notebook runs
end-to-end even before the real downloads (JetNet / CaloChallenge / AFHQ /
QuickDraw / sprites) are cached.

## How cadence reads these (authoring contract)

These notebooks are authored for cadence **auto mode**: a plain teaching
notebook is converted by `%cadence_autoregister` → registered notebook →
`%cadence_scaffold` → student notebook. The discipline, kept deliberately minimal:

- **Headings mark exercises.** Each exercise is a markdown `##`/`###` heading
  immediately followed by its code cell. Auto mode pairs them and reads the
  cell's answer from the live kernel. (No `# cadence:checkpoint` markers — a
  single manual marker would flip the whole notebook out of auto mode.)
- **Exercise cells end on a primitive answer.** The last line is a bare variable
  holding a number / string / list / bool, so auto mode can register a
  checkpoint and infer the comparator. A cell ending on a plot/figure/model is
  treated as *setup* and copied verbatim — so for plot deliverables we draw the
  plot as a side effect and end the cell on a scalar (e.g. a jet-mass W1).
- **`# cadence:starter` … `# cadence:end` is the ONLY cadence syntax used.** The
  region between the markers becomes the student's scaffold; everything else in
  the cell is the teacher's reference (stripped from the student notebook, kept
  as the reveal-after-3 solution).
- **Prerequisites live in setup cells before the exercise heading.** Because
  code outside the starter region is stripped, anything the student needs
  (loaded data, trained models, helper functions) goes in a preceding `setup`
  cell. Setup cells end on a `print(...)`.

## Regenerating / checking the notebooks

The notebooks are generated from small builder scripts (so they're reviewable
and reproducible):

```bash
python tools/build_01.py        # ... through build_06.py — writes notebooks/0N_*.ipynb
python tools/check_convert.py    # runs the REAL cadence converter on all 6, GPU-free
```

`tools/check_convert.py` emulates a post-"Run All" kernel (it injects a
primitive for each exercise's answer variable, since this repo box has no GPU)
and runs the actual `autoregister → scaffold` pipeline, asserting every exercise
auto-detects and stubs. Expected output: `[PASS]` for all six, `exercise_fails=0`,
`stubbed == exercises`.

> When you run the real `%cadence_autoregister` in a live GPU kernel, the
> answer values come from your executed cells (not the emulator's placeholders),
> so the registered comparators/solutions reflect real outputs.
