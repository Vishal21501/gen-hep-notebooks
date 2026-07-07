# Before you start: a primer

You know the basics of diffusion. You probably haven't done "interpretability." This gets you
from there to being able to start the project on day one. Read it once; skim the reading list;
don't try to finish the papers before we begin.

---

## 0. Where everything is — and how to start

**Open `capstone_calodiffusion.ipynb` and run it top to bottom.** That gets the tools working in
your hands (load a model, look at a shower, watch the forward process) before you read another
word. Everything for this project lives in this one folder:

| File / folder | What it's for |
|---|---|
| **`capstone_calodiffusion.ipynb`** | **start here** — runnable warm-up + a worked example |
| `calodiff_probe.py` | the only thing you import (`import calodiff_probe as cp`); the whole API |
| `JUMPING_OFF_POINTS.md` | the concrete **deliverables, milestones, and figures/tables** to produce |
| `README.md` | credit to the CaloDiffusion authors + setup notes |
| `data/` | the three shower caches (electron, photon, pion) |
| `vendor/` | the paper's code — you won't need to touch it |

The API, mapped to the concepts in this primer:

- **x̂₀ trajectory** (§2, the whole experiment) → `x_final, xs, x0s = cp.sample(model, E, debug=True)`.
  `x0s` is the per-step list of x̂₀ (the model's running guess of the finished shower).
- **Invert the preprocessing** (§2's day-wasting trap, "ReverseNorm") → `cp.to_physical(model, x0, E_inc)`
  turns a frame into physical GeV per voxel. Do this before *any* physics.
- **Plot / compute observables** → `cp.regular_grid(model, x0)` gives `(N, layer, angular, radial)`;
  the **layer axis is depth**. Both helpers accept x̂₀ frames straight from the trajectory.
- **Change the particle** → `cp.load_model("photon"|"pion")` and `cp.load_showers("pion", ...)`; same code.
- **Intervention** (§10) → build a conditioning energy with `cp.encode_energy(particle, [E_GeV, ...])`.

Generation (`cp.sample`) is slow on CPU and fast on GPU — do the trajectory runs on the GPU box.
`cp.DEVICE` tells you where you are.

---

## 1. What we're actually doing (in one paragraph)

A diffusion model builds a calorimeter shower by starting from noise and cleaning it up over
many steps. We are going to **watch the shower being built** and ask a physics question:
*which properties of the shower appear early in that process, and which appear late?* That's it.
No retraining, no cracking open the network's neurons. We take a model that already works, run
its generation loop, and measure physics at every step.

## 2. The one object that matters: x̂₀

In the reverse (generation) process, the model repeatedly looks at a noisy image and predicts
how to make it slightly cleaner. Hidden inside that update is a quantity worth naming: at every
step the model effectively produces a **full guess of the final shower** — "given the noise I'm
looking at now, here's my best current estimate of the finished object." We call that guess **x̂₀**
("x-hat-zero": an estimate of x₀, the clean shower).

- Early on, x̂₀ is a vague blob — the model has little to go on.
- Late on, x̂₀ is almost the final shower.
- **We compute all our physics on x̂₀ at each step.** The trajectory of x̂₀ is the whole experiment.

Do **not** compute physics on the noisy image itself (usually written xₜ). That's mostly noise
and tells you nothing until the very end. x̂₀ is the readable object.

⚠️ **Trap you will hit:** inside the model, x̂₀ lives in a squashed "preprocessed" space (a
logit-normalised space), not physical energy units. Before you compute anything physical, you
must invert that transform (the code calls it `ReverseNorm`). Physics on the un-inverted tensor
looks beautiful and means nothing. This is the #1 way people waste a day.

## 3. What "interpretability" means here (and what it doesn't)

"Interpretability" in ML is a big field with two flavours. We're doing the easy, robust one.

- **Mechanistic interpretability** (NOT us): open the model, look at neurons, attention, circuits,
  figure out *how* it computes. Powerful, but slow, and a different project.
- **Behavioural / trajectory analysis** (US): treat the model as a black box, feed the generation
  process, and study the *outputs* across the trajectory. We never look inside. We look at what
  it produces and when.

So when we say "interpret the model," we mean: read off *when* recognisable physics shows up as
the shower is generated. That's honest, reproducible, and doable in two days.

## 4. The "coarse-to-fine" idea, from scratch

People noticed that image diffusion models build pictures **coarse first, fine last**: the overall
layout appears in early steps, sharp details fill in later. There's a clean reason.

- Any image can be split into **frequencies**: low frequency = big smooth structures, high
  frequency = sharp edges and fine texture. (A Fourier transform does this split.)
- Natural images put most of their "energy" in low frequencies, dropping off as a **power law**
  toward high frequencies.
- Diffusion adds **white** noise — equal strength at all frequencies.
- So as you add noise, the weak high frequencies drown first; the strong low frequencies survive
  longest. Run it backwards and the model necessarily recovers **low frequencies first, high last**.

That's the mechanism behind coarse-to-fine. Some people call it "spectral autoregression": each
step conditions the next-finer detail on the coarse structure already cleaned up — like
autoregression, but along the frequency axis instead of time.

**Translate to a shower:** "coarse" ≈ where the bulk energy sits and how deep the shower goes;
"fine" ≈ the exact pattern of which cells are hit and where the shower edges are.

## 5. Why the *calorimeter* case is a real question

Here's the catch that makes this worth a paper rather than a re-run. The whole coarse-to-fine
argument depends on that **power-law frequency spectrum**. Natural images have it. Calorimeter
showers almost certainly **don't** — they're **sparse** (mostly empty voxels), spiky, and sit on
a cylindrical grid. So we genuinely don't know if the ordering holds. Two outcomes, both good:

- It holds anyway → the idea is more general than its assumptions.
- It breaks or changes → we've mapped the **boundary** of a popular framing.

And one sharp point to keep in mind: **occupancy** (how many cells are lit) is *not* a frequency.
A sparse set of spikes is "broadband" — it lives at all frequencies at once. So "when do high
frequencies appear" and "when does the sparsity structure appear" may be **different questions**.
Showing they come apart is exactly the physics the image-diffusion people can't contribute.

## 6. The four observables you'll measure

Computed on x̂₀ (physical units!) at every step:

1. **Total energy** — sum of all cell energies. Expected to settle early (coarse).
2. **Shower depth** — the energy-weighted average layer (how far in the shower peaks). Coarse-ish.
3. **Radial spread** — how wide the shower is, energy-weighted. Finer.
4. **Occupancy** — fraction of cells above a fixed threshold. Expected last (fine + sparse).

Use **one fixed threshold** for occupancy at every step (the config's `ECUT`). Changing it per
step invents a fake trend.

The key output isn't the picture — it's a number per observable: the **lock-in step**, the step at
which the observable first reaches ~90% of its final value. That scalar is what turns "look at
this plot" into "here's a table," and it's what we compare across particle types.

## 7. The particle-type question (probably our best result)

Photons and electrons make **electromagnetic** showers — compact, smooth. Pions make **hadronic**
showers — messy, broad, more fluctuation. Prediction: the messy showers need **more of the
trajectory** to resolve their fine structure, so the gap between "coarse locks in" and "fine locks
in" should be **wider** for pions. We have pretrained models for all three, so this costs nothing
but re-running the pipeline.

## 8. The spectral arm (optional, don't panic)

Same idea as the observables, in frequency space: FFT the shower, and watch each frequency band
emerge across steps. The clean, physically-honest axis to transform is **azimuthal (φ)** — it's
periodic, and its Fourier modes are real physics (multipoles). Do this on **dataset 2** (regular
16×9 grid); dataset 1's irregular binning makes the FFT untrustworthy. If this arm gets messy,
that's fine — the observable curves stand on their own.

## 9. The mindset that keeps this rigorous

- **Separate what you measure from what you judge on.** We measure observables/frequencies; we
  judge the model's overall faithfulness with the official CaloChallenge evaluation (a classifier
  that tries to tell generated from real, plus FPD/KPD). We never tune to those — so if an ordering
  shows up, it's real and emergent, not something we baked in.
- **Negative results ship.** "The ordering breaks for hadronic showers" is a finding, not a failure.
- **Green before clever.** Get one emergence curve on screen before optimising anything.

## 10. The interpretability tools we'll actually use

"Interpretability" sounds like it means cracking open the network's neurons. It can — but that's
only one half of the field, and it's *not* the half we're in. The simplest way to keep the tools
straight is to ask **where in the dataflow you tap**:

- **Output-space (behavioural) tools** — you read what the model *produces* (x̂₀ and things
  computed from it) and treat the network itself as a black box. This is where our whole project
  lives.
- **Internal-space (mechanistic) tools** — you reach *inside* the network and read its hidden
  activations. Powerful, but a different, longer project.

**What we're using (all output-space):**

- **The trajectory / emergence probe** — per-step x̂₀ → physics observables → lock-in curves. This
  is the main instrument, and yes, it *is* an interpretability method (a behavioural one).
- **Spectral decomposition** — the RAPSD and azimuthal-multipole FFTs from §8. A signal-processing
  probe on the same x̂₀ stream.
- **Intervention / sensitivity analysis** — change an input (nudge the incident-energy
  conditioning), watch how the observables respond, and check the response is physical (more
  energy → deeper, more hits). This is the most classically "interpretability" thing in scope: it's
  a *causal* probe — change a cause, measure the effect. Cheap, and high-value.

**One internal tool worth knowing (optional stretch):**

- **Linear probe** — attach a forward hook to a UNet block, dump its activations, and fit a simple
  linear model (sklearn `LinearRegression`) to predict, say, depth from those activations. If it
  predicts well, that quantity is "linearly decodable" from that layer. Run it across layers and
  steps and you can say *where and when* the model first represents depth. It's the one inside-the-
  network probe with a good effort-to-insight ratio — a torch forward hook plus a regression, an
  afternoon of work.

**Names you'll hear but we are NOT using** (each is a whole separate project — know they exist so
you can name them in related work, don't touch them):

- **Sparse autoencoders (SAEs) / temporal SAEs** — train an autoencoder on activations to pull out
  interpretable "features" that switch on across denoising steps.
- **Activation patching / causal tracing** — copy activations between two runs to localise *where*
  the model stores a concept.
- **Gradient attribution** (integrated gradients, saliency) — which input voxels drive an
  observable. Easy to run, but noisy to read on sparse calo data.

**Library stack**, so nobody reinvents anything:

- **torch forward hooks** — the workhorse for anything internal. (Don't reach for
  `TransformerLens`/`nnsight` — those are transformer-shaped; hooks are simpler for a UNet.)
- **numpy.fft** — the spectral arm.
- **sklearn** — the linear probes.
- **captum** — only if you want off-the-shelf gradient attribution.
- The **CaloChallenge classifier** is itself an interpretability tool: a "can you tell real from
  generated" test is a distributional probe, and *which* features it relies on tells you where the
  model's physics is weakest.

Rule of thumb: if a tool reads x̂₀ or an observable, it's in scope. If it needs you to open up the
network's activations, it's optional at best (the linear probe) or out of scope (everything else).

---

## Reading list

Tiered. Start with the three essentials; the rest is for when a specific question bites.

**Essential (read first)**
- Ho, Jain, Abbeel, *Denoising Diffusion Probabilistic Models* (2020), arXiv:2006.11239 — the
  diffusion foundation; make sure the forward/reverse process and x̂₀ are solid.
- Amram & Pedro, *CaloDiffusion with GLaM…* (2023), arXiv:2308.03876 — the exact model and data
  you're using; read the method + how they evaluate.
- Dieleman, *Diffusion is spectral autoregression* (2024), blog:
  sander.ai/2024/09/02/spectral-autoregression.html — the clearest intuition for coarse-to-fine
  and the source of the RAPSD diagnostic. (He includes code.)

**Background on coarse-to-fine / frequency**
- Rissanen, Heinonen, Solin, *Generative Modelling with Inverse Heat Dissipation* (2022),
  arXiv:2206.13397 — coarse-to-fine made explicit via frequency decay; the power-law/PSD argument
  we lean on lives here.
- *DCTdiff* (2024), arXiv:2412.15032 — a theoretical proof that image diffusion is approximately
  spectral autoregression. Skim for the statement, not the algebra.
- Useful counterpoint: Peyman Milanfar has publicly argued the "diffusion = spectral
  autoregression" line is an over-simplification (real denoisers are nonlinear). Worth knowing so
  we don't overclaim — cite it as the caveat.

**Calorimeter context (how the field frames and measures things)**
- Amram et al., *CaloChallenge 2022: a community challenge for fast calorimeter simulation* (2025),
  arXiv:2410.21611 — the datasets, the evaluation battery, and where the models struggle.
- Mikuni & Nachman, *CaloScore v2* (2024), arXiv:2308.03847 — the other main diffusion approach;
  our "does it replicate on a second architecture" stretch model.
- Buss et al., *CaloHadronic* (2025), arXiv:2506.21720 — a diffusion model built specifically for
  hadronic showers; context for the EM-vs-hadronic question.
- *A Comprehensive Evaluation of Generative Models in Calorimeter Shower Simulation* (2024),
  arXiv:2406.12898 — how fidelity is measured; note they praise CaloDiffusion's "interpretability"
  but never analyse the trajectory (that gap is us).
- Lu, Collado, Whiteson, Baldi, *Sparse autoregressive models for scalable generation of sparse
  images in particle physics* (2021), arXiv:2009.14017 — the closest prior work on sparsity +
  autoregression in HEP; good for the occupancy discussion.

**Optional — the "other" interpretability flavour (mechanistic), for the curious**
- *Emergence and Evolution of Interpretable Concepts in Diffusion Models* (2025), arXiv:2504.15473
- *TIDE: Temporal-Aware Sparse Autoencoders for Interpretable Diffusion Transformers* (2025),
  arXiv:2503.07050 — ties feature changes to the coarse-to-fine transition step. Nice bridge, but
  not something we're implementing.
