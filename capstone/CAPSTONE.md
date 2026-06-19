# Full-day capstone — Calorimeter fast simulation

*Project 2 of the generative-modelling track. Audience: students who've just done
the VAE / GAN / diffusion / evaluation notebooks + the in-session jet mini (06).*

---

## 0. The one-liner

**Spend the day building the best generative *fast simulator* for calorimeter
showers** — a model that turns an incident particle energy into a realistic
energy deposit, far faster than full Geant4 simulation. Submit through cadence;
we compare everyone's solutions in a friendly show-and-tell, not a cut-throat
leaderboard.

This is the "boss level" of the week: open-ended, integrative, and on the
**second modality** (calorimeter images/voxels from notebook 04) rather than
jets — so the day feels fresh.

---

## 1. Why fast sim (the motivation to open with)

Full detector simulation (Geant4) is the single biggest CPU cost in HEP — the
LHC experiments spend a large fraction of their grid budget simulating
calorimeter showers. A generative model that samples a shower in milliseconds
instead of minutes, *conditioned on the incident energy*, is a real, deployed
use of exactly what students learned this week. The community even runs a public
benchmark for it: the **CaloChallenge**. That's our task.

---

## 2. The task

Given an **incident energy** `E_inc` (a scalar, GeV), generate a **calorimeter
shower**: the voxelised energy deposit (an image-like tensor). A good model:

- **conserves energy** sensibly (deposited energy tracks `E_inc`),
- reproduces **shower shapes** (radial/longitudinal profiles, sparsity),
- is **fast** (batch-samples thousands of showers per second),
- and is **indistinguishable** from real showers to a trained classifier.

Architecture is the student's choice — they've seen VAEs (01), GANs (02) and
diffusion (04). Diffusion conditioned on energy (extending notebook 04) is the
strong baseline; a conditional VAE or GAN is a perfectly good alternative and
makes for nice cross-architecture comparison in the wrap-up.

### Data
- **CaloChallenge Dataset 2** (electrons) — [Zenodo 6366271](https://zenodo.org/records/6366271).
  GEANT4 electron showers, incident energy log-uniform 1 GeV–1 TeV; each shower is
  **6480 voxels = 45 layers × 9 radial × 16 angular** (a true 3D grid). HDF5 with
  `incident_energies` (N,1) and `showers` (N,6480), energies in **MeV**.
- We pull **one file** (`dataset_2_1.hdf5`, ~1.4 GB, 100k showers) and **skim it to
  half** (~50k) into a small gzip cache: `python tools/prepare_data.py calochallenge`.
- `src/data.py::load_calochallenge_ds2()` reads that cache, reshapes to
  `(N, 45, 9, 16)` and converts to GeV. **Real data only — no synthetic fallback**
  (it raises with a pointer to the prep step if the file is missing).
- The diffusion baseline feeds the **45 layers as input channels** of a 16×16
  image (radial padded 9→16, cropped back after sampling).
- We provide a **held-out test set** (`dataset_2_2.hdf5`, not shown to students) for scoring.

---

## 3. The day (≈6 hours, milestone-driven)

| Time | Block | Goal |
|---|---|---|
| 0:00–0:45 | **Kickoff** | The fast-sim problem; the data; the metrics; how submission/comparison works. Walk through the baseline notebook. |
| 0:45–2:00 | **Block 1 — baseline** | Get *any* conditional generator training and submitting. **Milestone: first submission on the board before lunch.** |
| 2:00–3:00 | Lunch | — |
| 3:00–5:00 | **Block 2 — improve** | Better architecture / conditioning / training; iterate against the metrics and your own plots. |
| 5:00–6:00 | **Block 3 — finalize** | Lock best model; submit final metric + plots + reflection; 3-min lightning talks. |

Milestones (gentle, not ranked): *baseline submitted by lunch* → *beats the
provided baseline on ≥1 metric* → *final submission + reflection*. The point is
momentum for everyone, not a winner.

---

## 4. Evaluation — physics first, then a classifier

Students score their **generated** showers against a **reference** set on:

**Physics observables** (overlaid histograms, real vs generated):
- total deposited energy, and the **energy response** `E_dep / E_inc` vs `E_inc`
  (linearity + resolution),
- **longitudinal** profile (energy per layer) and **lateral** spread (shower width),
- **sparsity** (fraction / number of active voxels).

**Distribution distances:** 1-Wasserstein on each observable (reuse the W1 idea
from 05).

**The headline metric — classifier two-sample test** (the CaloChallenge's own
yardstick): train a small classifier to tell real from generated showers; an
**AUC near 0.5 means indistinguishable = a good simulator**. This reuses exactly
the GNN/classifier idea from notebook 05, just on calo images.

We give students the **scoring harness** so they can self-evaluate all day; the
final number is recomputed on the held-out set at submission.

---

## 5. Friendly comparison via cadence (not a leaderboard grind)

cadence is for *seeing everyone's progress*, so we use it for a **friendly
compare-and-discuss**, not a ranked competition. Each student submits a handful
of checkpoints (cadence auto-types each):

- **`capstone.auc`** — their classifier-test AUC (number) — the headline.
- **`capstone.response`** — energy-response plot (plot).
- **`capstone.shower`** — a generated shower image, or a denoising gif (plot/gif).
- **`capstone.writeup`** — 2–3 sentences: architecture, what helped, what didn't
  (string / `mark_done`).

The teacher dashboard shows the whole cohort's submissions side by side; we walk
through them together at the end — "whose response curve is flattest? who got the
sparsest showers? which architecture won and why?" Recognition is for **best
physics insight / most creative approach**, alongside the best AUC — so it stays
collaborative.

> cadence constraints to honour (same as the lesson notebooks): submission cells
> must be **non-blocking** and must **not replace the student's own cell output**.
> Mark only starter regions; let auto-mode infer the rest.

---

## 6. Deliverables

1. Best model **checkpoint** + the cadence submissions above.
2. A short **report** (a markdown cell is fine): the metrics, the key plots, and
   what worked / didn't.
3. Optional **3-min lightning talk**.

---

## 7. What we build (infrastructure to-do)

- [x] `capstone/CAPSTONE.md` — this doc. ✅
- [x] `capstone/capstone_calo.ipynb` — the **baseline starter** (built from
      `tools/build_capstone.py`): real Dataset 2 load, a small **energy-conditioned
      diffusion baseline** (reusing `src.diffusion.make_unet(class_embed=True)`
      from 04, 45 layers as channels), the **scoring harness**, and the **cadence
      submission cells** (starter-marked). Passes `tools/check_convert.py`.
- [x] `src/calo_eval.py` — the metrics: observables, W1s, and the classifier
      two-sample test, packaged so students call one function (`score_showers`).
- [x] `src/data.py::load_calochallenge_ds2` + `tools/prepare_data.py calochallenge`
      — download `dataset_2_1.hdf5`, skim to ~50k, cache to `data/`.
- [ ] **Run the prep** on the provisioning box (downloads 1.4 GB) and ship the
      skimmed `data/calochallenge_ds2.hdf5` to students.
- [ ] A **held-out reference/test set** (from `dataset_2_2.hdf5`) + a server-side
      scoring note.
- [ ] One or two **pre-trained checkpoints** (`checkpoints/calo_diffusion_ds2.pt`,
      maybe a CVAE) so a student who stalls can still load a working model.

## 8. Stretch goals (for fast finishers)

- Condition on more than energy (incident angle / position).
- Compare two architectures head-to-head on the same metrics.
- Quantify the **speed-up** vs a notional Geant4 baseline (showers/sec) — the
  actual selling point of fast sim.
- Try classifier-guided or physics-informed losses (energy conservation as a
  penalty).

## 9. Risks / logistics

- **GPU time:** diffusion is the slowest; ship a checkpoint so training isn't on
  the critical path. Keep the baseline small (notebook 04's config).
- **Data size/shipping:** the skimmed cache (`data/calochallenge_ds2.hdf5`, 50k
  showers) is **~360 MB** — over GitHub's 100 MB/file limit, so it's *not* committed.
  Ship it as a **GitHub Release asset** (≤2 GB) or any read-only URL and set
  `CALO_DS2_CACHE_URL`; `load_calochallenge_ds2` fetches it once (the AFHQ pattern).
  There is **no synthetic fallback** — the notebook hard-stops with a clear message
  if the cache is absent and no URL is set. Build it once with
  `tools/prepare_data.py calochallenge` (downloads 1.4 GB, skims to 50k, caches).
- **Voxel count / compute:** Dataset 2 is 6480 voxels (vs Dataset 1's ~368), and the
  45-channel UNet is heavier than notebook 04's — keep `N_LOAD`/`EPOCHS` modest for
  the baseline and lean on the shipped checkpoint.
- **Spread of ability:** the milestones + the loadable checkpoint keep weaker
  students moving; the stretch goals keep strong ones busy.
