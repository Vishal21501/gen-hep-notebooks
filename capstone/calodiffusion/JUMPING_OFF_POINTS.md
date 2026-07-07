# The project: when does physics emerge in CaloDiffusion?

Read `START_HERE.md` first — it explains the *idea* (the x̂₀ trajectory, coarse-to-fine, the four
observables). **This file is the plan: what to measure and exactly what to hand in.**

The research question is novel: *as the diffusion model builds a shower from noise, in what order
do physical properties appear — and does that order reveal what the model "understands" about
calorimeter physics?* Nobody has measured this. We are **not** re-checking whether the showers
look real (the paper already did that — see "Not the project" below); we are measuring **when**
each piece of physics is decided.

It is **one project in three tiers.** Nail the core, then go deep on *one* thing — don't spread
thin across a menu.

| Tier | What | Who |
|---|---|---|
| **Core (required)** | Measure the **lock-in step** of the four observables on the x̂₀ trajectory | everyone |
| **Nice-to-have** | Does the lock-in ordering differ **EM vs hadronic** (electron/photon vs pion)? | most people, if core works |
| **Optional** | *One* deeper probe (spectral, intervention, or linear probe), chosen with us | based on your interest |

## The calls you'll use

```python
import calodiff_probe as cp
model = cp.load_model("electron")
E = cp.encode_energy("electron", [50.]*8)                 # 8 showers at 50 GeV
x_final, xs, x0s = cp.sample(model, E, debug=True)        # x0s = per-step x̂₀ (the experiment)
for x0 in x0s:                                            # each x0 is one trajectory frame
    grid = cp.to_physical(model, x0, E_inc=[50.]*8)       # -> physical GeV, (N, layer, ang, rad)
    # ... compute the four observables on `grid` ...
```

`to_physical` inverts the preprocessing (the day-waster from START_HERE §2). The **layer axis is
depth**, the last axis is radius. Generation is slow on CPU — **do trajectory runs on the GPU**.

---

## Milestones (same shape whichever tier you reach)

| # | When | Milestone | Artifact |
|---|---|---|---|
| **M0** | end of hour 1 | Tools work — run the notebook, plot one x̂₀ frame in physical units | 1 sanity figure |
| **M1** | end of day 1 | **The emergence curves** — four observables vs step, for one particle | Fig 1 + Table 1 (draft) |
| **M2** | day 2 midday | **Lock-in quantified** (+ particle comparison if you're going there) | the lock-in table |
| **M3** | end | **Write-up** — polished figure(s), the table, 1–2 paragraphs of interpretation | final notebook |

---

## Core (required) — the emergence curves and lock-in table

Generate a batch of showers at a fixed energy, take the x̂₀ trajectory, and compute the **four
observables** (START_HERE §6) on x̂₀ **in physical units** at every step:

1. **Total energy** — sum of all cell energies
2. **Shower depth** — energy-weighted mean layer
3. **Radial spread** — energy-weighted radial RMS
4. **Occupancy** — fraction of cells above a **fixed** threshold (the config's `ECUT`, same at every step)

Normalise each curve to its final (step-0) value, and define the **lock-in step** = the step by
which the observable first reaches — and stays within ~10% of — its final value.

**Deliverables:**
- **Fig 1 (M1) — emergence curves.** The four normalised observables vs diffusion step, one
  figure, averaged over your batch (show the spread as a band). This is the headline plot.
- **Fig 2 — visual sanity.** x̂₀ as a layer×radius heatmap at ~4 steps (early→late): a vague blob
  becoming a shower. Confirms the trajectory is doing what you think.
- **Table 1 (M2) — lock-in steps.** One row per observable: its lock-in step (mean ± spread over
  the batch). Do total energy / depth really lock in **before** radial spread and occupancy?

Report your batch size and the energy you used. That's a complete, publishable result on its own.

---

## Nice-to-have — is the ordering different for different particles?

This is our **headline result if the core works.** EM showers (electron, photon) are compact and
smooth; hadronic showers (pion) are messy and broad. Prediction: the gap between "coarse locks
in" (energy, depth) and "fine locks in" (spread, occupancy) is **wider for pions** — the messy
structure takes more of the trajectory to resolve.

Repeat the core measurement for all three models at **matched incident energies** (`load_model`
+ `encode_energy` with the same GeV list).

**Deliverables:**
- **Table 2 — lock-in step per observable × particle** (electron / photon / pion). The core of
  the comparison.
- **Fig 3 — the coarse→fine gap per particle.** e.g. (occupancy lock-in − energy lock-in) for
  each particle, or the four curves overlaid across particles. Does the gap widen for pions?
- One paragraph: does the ordering hold, widen, or break for hadronic showers? **Any of the three
  is a real finding.**

---

## Optional directions — pick *one*, with us

Only after the core is solid. Each is roughly a half-day; choose by what you found interesting.

- **Spectral emergence.** Instead of scalar observables, watch **frequency bands** appear. FFT
  along the **azimuthal (φ)** axis — periodic, so its modes are real physics (multipoles) — and
  plot when each band locks in. **Dataset 2 (electron) only** (regular grid; DS1 binning makes the
  FFT untrustworthy). Sharp sub-question: does **occupancy** (broadband — a sparse set of spikes
  lives at all frequencies) lock in *separately* from the high-frequency bands? If it does, you've
  shown sparsity and frequency are different axes — something the image-diffusion story can't say.
- **Intervention / conditioning sensitivity.** A *causal* probe: vary the incident-energy
  conditioning and check the observables respond physically (more energy → deeper, more hits) —
  and whether that changes the lock-in ordering. Cheap, high-value, and the most classically
  "interpretability" thing in scope.
- **Linear probe (peeks inside the network).** Forward-hook a UNet block, dump activations, and
  fit `sklearn.LinearRegression` to predict depth (or energy) from them. Run across layers and
  steps: *where and when* does the model first linearly represent depth? The one internal probe
  with a good effort-to-insight ratio (a hook + a regression).

---

## Not the project (so you don't chase it)

- **"Can you tell real from generated?" (classifier / AUC / FPD / KPD).** This is the paper's *own*
  evaluation. We use it **once, as a faithfulness check** — to confirm the model we're probing is
  actually good — and never tune to it. It is **not a deliverable**; re-benchmarking fidelity is
  not novel work.
- **Opening neurons / SAEs / activation patching / circuits.** Real but separate, longer projects
  (see START_HERE §10). The linear probe is the *only* inside-the-network tool in scope, and only
  as an optional stretch.

We are after **novel understanding of what physics the diffusion model learns and when it emerges**
— not another fidelity score.

## What "done" looks like
The emergence figure, the lock-in table, and a paragraph naming **the ordering you found** and
what it says about the model's physics. Negative results ship: "the ordering breaks for hadronic
showers" is a finding. Quality of reasoning over quantity of plots.

## Practical notes
- **Probing is instant; generating is slow on CPU and scales with the number of showers** — each
  of the 400 steps runs the whole batch. Use the **GPU** for trajectory runs; `cp.DEVICE` tells you
  where you are, `CALODIFF_DEVICE=cuda` forces it.
- Use **one fixed `ECUT`** for occupancy at every step — changing it per step invents a fake trend.
- `cp.sample(..., sample_offset=0)` makes showers blow up (two unstable steps); the default
  `sample_offset=2` fixes it. Worth seeing once.
- **Green before clever:** get one emergence curve on screen before optimising anything.
