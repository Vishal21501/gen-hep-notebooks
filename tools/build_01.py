"""Build notebooks/01_vae.ipynb — VAEs: latent space + reconstruction.

Fun spine: AFHQ animal faces + latent arithmetic (a cat->dog vector).
Physics:   encode JetNet jets with a GNN (DeepSets) VAE; walk quark->gluon in
           latent space and watch the jet-mass spine.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nbbuild import build, md, code, setup, exercise

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "notebooks" / "01_vae.ipynb"

cells = [
    md("""
# 01 · Variational Autoencoders

**Generative Modelling for HEP** — notebook 1 of 6.

A VAE learns a smooth, low-dimensional *latent space*: an encoder maps data to a
distribution over latent codes, a decoder maps codes back to data, and a KL term
pulls the codes toward a standard Gaussian so the space is continuous and we can
*sample* and *interpolate* in it.

We'll build the intuition on **AFHQ animal faces** (latent arithmetic — a
"cat → dog" vector), then carry the same idea to **JetNet jets**: encode jets
with a permutation-invariant GNN and walk from **quark** to **gluon** in the
latent space, tracking our recurring **jet-mass** plot.

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
from src.data import load_afhq_local, load_jetnet
from src.gnn import DeepSetsEncoder, DeepSetsDecoder
from src.jets import JetStandardizer
from src.jetmass import jet_mass, plot_jet_mass, jet_mass_w1, plot_jets, plot_jet_overlay

SEED = 0
LATENT_DIM = 64      # more latent capacity -> sharper reconstructions
EPOCHS = 15          # small default; on a CPU drop to ~5, on GPU push to 30+
BATCH = 128
set_seed(SEED)
device = get_device()
print(f"device={device}  latent_dim={LATENT_DIM}  epochs={EPOCHS}")
"""),

    # ----------------------------------------------------------------- Part A
    md("""
## Part A · A VAE on animal faces

We load AFHQ (cats, dogs, wildlife) at low resolution — or a synthetic stand-in
if the download isn't cached yet — and define a small convolutional VAE.
"""),
    setup("""
# Real AFHQ animal faces, loaded from the local data/ folder (data/afhq_32.npz,
# built once by tools/prepare_data.py). Falls back to synthetic blobs if absent.
images, labels = load_afhq_local(n=4000, image_size=32)   # (N, 3, 32, 32) in [0,1]
X = torch.as_tensor(images, dtype=torch.float32)
img_loader = to_loader(X, batch_size=BATCH)
IMG_SHAPE = X.shape[1:]

cat_idx = np.where(labels == 0)[0]   # 0 = cat
dog_idx = np.where(labels == 1)[0]   # 1 = dog
print("images:", tuple(X.shape), "| cats:", len(cat_idx), "dogs:", len(dog_idx))


class ConvVAE(nn.Module):
    def __init__(self, latent=LATENT_DIM):
        super().__init__()
        c = IMG_SHAPE[0]
        def down(i, o):   # halve spatial size
            return nn.Sequential(nn.Conv2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.GELU())

        def up(i, o):     # double spatial size
            return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.GELU())

        # 32 -> 16 -> 8 -> 4, channels 64/128/256 (deeper than before = sharper)
        self.enc = nn.Sequential(down(c, 64), down(64, 128), down(128, 256), nn.Flatten())
        self.fc_mu = nn.Linear(256 * 4 * 4, latent)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent)
        self.fc_dec = nn.Linear(latent, 256 * 4 * 4)
        self.dec = nn.Sequential(
            nn.Unflatten(1, (256, 4, 4)),
            up(256, 128), up(128, 64),                      # 4 -> 8 -> 16
            nn.ConvTranspose2d(64, c, 4, 2, 1), nn.Sigmoid())   # 16 -> 32

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        return self.dec(self.fc_dec(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        return self.decode(z), mu, logvar

vae = ConvVAE().to(device)
print(vae.__class__.__name__, "params:", sum(p.numel() for p in vae.parameters()))
"""),

    md("""
### Exercise 1 — The ELBO loss

A VAE maximises the Evidence Lower BOund. As a loss to *minimise*:

$$\\mathcal{L} = \\underbrace{\\text{BCE}(\\hat x, x)}_{\\text{reconstruction}}
   + \\underbrace{-\\tfrac12 \\sum (1 + \\log\\sigma^2 - \\mu^2 - \\sigma^2)}_{\\text{KL}(q\\,\\|\\,\\mathcal N(0,1))}$$

Implement `vae_loss`, train for a few epochs, and report the **final epoch's
mean loss** as `final_loss`.
"""),
    exercise(
        scaffold_body="""
def vae_loss(model, batch):
    x = batch[0]
    x_hat, mu, logvar = model(x)
    # reconstruction term: summed binary cross-entropy over pixels, per sample
    recon = ...
    # KL term: -0.5 * sum(1 + logvar - mu^2 - exp(logvar)), per sample
    kl = ...
    return (recon + kl) / x.shape[0]
""",
        solution_body="""
def vae_loss(model, batch):
    x = batch[0]
    x_hat, mu, logvar = model(x)
    recon = F.binary_cross_entropy(x_hat, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon + kl) / x.shape[0]

history = train(vae, img_loader, vae_loss, epochs=EPOCHS, lr=1e-3, device=device)
final_loss = round(float(history[-1]), 3)
print("final ELBO loss:", final_loss)
""",
        answer_var="final_loss",
    ),

    md("""
### Exercise 2 — Reconstruction error

Encode a held-out batch to its latent **mean** (no sampling), decode, and
measure the mean per-pixel squared error. Return it as `recon_mse`.
"""),
    exercise(
        scaffold_body="""
vae.eval()
with torch.no_grad():
    x = X[:256].to(device)
    mu, logvar = vae.encode(x)
    # decode the latent MEAN (deterministic reconstruction), then MSE vs x
    x_hat = ...
    recon_mse = ...
""",
        solution_body="""
vae.eval()
with torch.no_grad():
    x = X[:256].to(device)
    mu, logvar = vae.encode(x)
    x_hat = vae.decode(mu)
    recon_mse = float(F.mse_loss(x_hat, x))
recon_mse = round(recon_mse, 4)
print("reconstruction MSE:", recon_mse)
""",
        answer_var="recon_mse",
    ),

    md("""
### Exercise 3 — Latent arithmetic: a "cat → dog" vector  🐱→🐶

The fun bit. Average the latent codes of cats and of dogs, take the difference
to get a **direction**, and add it to a cat's code. Decode along the way to make
a morph grid (we plot it for you). Then quantify the shift: the cosine
similarity between the cat→dog direction and the actual latent displacement you
applied. Return it as `direction_cos` — it should be ~1.0 if you built the
vector correctly.
"""),
    exercise(
        scaffold_body="""
vae.eval()
with torch.no_grad():
    cat_mu = vae.encode(X[cat_idx][:256].to(device))[0].mean(0)
    dog_mu = vae.encode(X[dog_idx][:256].to(device))[0].mean(0)
    # the cat->dog DIRECTION in latent space:
    direction = ...
    # morph one cat toward dog in steps, decode each (shown as a grid):
    base = vae.encode(X[cat_idx[0]][None].to(device))[0]
    steps = [base + a * direction for a in np.linspace(0, 1, 6)]
    grid = torch.cat([vae.decode(z) for z in steps])
    # cosine similarity between `direction` and the full displacement applied:
    applied = steps[-1] - steps[0]
    direction_cos = ...
""",
        solution_body="""
vae.eval()
with torch.no_grad():
    cat_mu = vae.encode(X[cat_idx][:256].to(device))[0].mean(0)
    dog_mu = vae.encode(X[dog_idx][:256].to(device))[0].mean(0)
    direction = dog_mu - cat_mu
    base = vae.encode(X[cat_idx[0]][None].to(device))[0]
    steps = [base + a * direction for a in np.linspace(0, 1, 6)]
    grid = torch.cat([vae.decode(z) for z in steps])
    applied = (steps[-1] - steps[0]).flatten()
    direction_cos = float(F.cosine_similarity(direction.flatten(), applied, dim=0))

fig, axes = plt.subplots(1, 6, figsize=(12, 2))
for ax, im in zip(axes, grid.cpu()):
    ax.imshow(im.permute(1, 2, 0).clip(0, 1).numpy()); ax.axis("off")
fig.suptitle("cat → dog latent morph"); plt.show()
direction_cos = round(direction_cos, 3)
""",
        answer_var="direction_cos",
    ),

    # --------------------------------------------------- PCA & compression
    md("## A linear baseline — PCA & compression"),
    setup("""
from sklearn.decomposition import PCA

# Flatten each image to a 3072-vector and fit a linear PCA at the SAME latent
# budget as the VAE — the classic baseline to compare the learned latent against.
X_flat = X.reshape(len(X), -1).numpy()                 # (N, 3*32*32)
pca = PCA(n_components=LATENT_DIM).fit(X_flat)
print("PCA fit:", X_flat.shape[1], "pixels ->", LATENT_DIM, "components")
"""),

    md("""
### Exercise 4 — PCA: how linear is the data?

PCA finds the best *linear* subspace. How much of the total pixel variance do the
top `LATENT_DIM` principal components capture? Return the fraction as
`variance_explained` (between 0 and 1).
"""),
    exercise(
        scaffold_body="""
# pca.explained_variance_ratio_ holds the fraction of variance per component.
variance_explained = ...        # sum it over the LATENT_DIM components
""",
        solution_body="""
variance_explained = round(float(pca.explained_variance_ratio_.sum()), 3)
print(f"top-{LATENT_DIM} PCA components explain {variance_explained:.1%} of pixel variance")
""",
        answer_var="variance_explained",
    ),

    md("""
### Exercise 5 — Compression: store less, rebuild it

Both PCA and the VAE turn a 3072-pixel image into just `LATENT_DIM` numbers — a
big saving. Compute the **compression ratio** (original dims ÷ latent dims) as
`compression_ratio`. The cell also reconstructs from each codec at the same
budget and prints the error, so you can see the VAE's *nonlinear* code rebuild
more faithfully than linear PCA.
"""),
    exercise(
        scaffold_body="""
# Each image is stored as LATENT_DIM numbers instead of 3*32*32 pixels:
compression_ratio = ...         # original pixel count / LATENT_DIM
""",
        solution_body="""
compression_ratio = round(X_flat.shape[1] / LATENT_DIM, 1)

# Reconstruct a batch from each codec at the same latent budget and compare.
sample = X_flat[:512]
pca_mse = float(np.mean((pca.inverse_transform(pca.transform(sample)) - sample) ** 2))
vae.eval()
with torch.no_grad():
    xb = X[:512].to(device)
    vae_flat = vae.decode(vae.encode(xb)[0]).cpu().numpy().reshape(512, -1)
vae_mse = float(np.mean((vae_flat - sample) ** 2))
print(f"{compression_ratio:.0f}x compression  |  PCA MSE {pca_mse:.4f}  vs  VAE MSE {vae_mse:.4f}")
""",
        answer_var="compression_ratio",
    ),

    md("""
### Visualising PCA

Three views make PCA concrete:

1. **The data in 3D PCA space** (first three components), coloured by class — PCA
   already separates cats / dogs / wild along linear axes.
2. **The component directions as images** — each principal component *is* a
   picture in pixel space (an "eigen-animal"); the data is built by adding these
   up.
3. **Progressive reconstruction** of one image — from PC1 alone, PC2 alone,
   PC1+PC2, up to all `LATENT_DIM` components. Each component adds a bit more
   structure; the linear baseline the VAE improves on.
"""),
    code("""
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
scores = pca.transform(X_flat)                            # (N, LATENT_DIM)

# (1) data in 3D PCA space
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(111, projection="3d")
for cls, name in [(0, "cat"), (1, "dog"), (2, "wild")]:
    s = labels == cls
    ax.scatter(scores[s, 0], scores[s, 1], scores[s, 2], s=3, alpha=0.3, label=name)
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
ax.set_title("data in 3D PCA space"); ax.legend(fontsize=8)
plt.show()


def as_img(vec):
    v = vec.reshape(*IMG_SHAPE).transpose(1, 2, 0)
    return ((v - v.min()) / (np.ptp(v) + 1e-8))             # normalise for display

# (2) the first 6 component directions, as images
fig, axes = plt.subplots(1, 6, figsize=(11, 2))
for k, ax in enumerate(axes):
    ax.imshow(as_img(pca.components_[k])); ax.axis("off")
    ax.set_title(f"PC{k + 1}", fontsize=9)
fig.suptitle("principal components as images (the 'directions')", fontsize=10)
plt.show()

# (3) reconstruct ONE image from PC1 alone, PC2 alone, PC1+2, then all components
i = 0
def recon_from(ks):
    z = np.zeros_like(scores[i]); z[ks] = scores[i, ks]
    return (pca.mean_ + z @ pca.components_).reshape(*IMG_SHAPE).transpose(1, 2, 0)

panels = [("original", X_flat[i].reshape(*IMG_SHAPE).transpose(1, 2, 0)),
          ("PC1", recon_from([0])), ("PC2", recon_from([1])),
          ("PC1+2", recon_from([0, 1])),
          (f"all {LATENT_DIM}", pca.inverse_transform(scores[i:i+1])[0]
           .reshape(*IMG_SHAPE).transpose(1, 2, 0))]
fig, axes = plt.subplots(1, 5, figsize=(11, 2.4))
for ax, (name, img) in zip(axes, panels):
    ax.imshow(np.clip(img, 0, 1)); ax.axis("off"); ax.set_title(name, fontsize=9)
fig.suptitle("progressive PCA reconstruction", fontsize=10)
plt.show()
"""),

    # --------------------------------------------------- animate it (gif util)
    md("""
## Animate it — latent walks as gifs

Several of these ideas are more fun *moving*. `src/anim.py` has a tiny helper:

```python
from src.anim import frames_to_gif, show_gif
frames = [...]   # list of (C,H,W) tensors, HxWxC arrays, OR matplotlib figures
frames_to_gif(frames, "out.gif", fps=12, scale=6)   # scale upsizes small images
show_gif("out.gif")                                  # display inline
```

It takes model image tensors directly, and even matplotlib figures (via
`fig_to_frame`) so you can animate 2D plots — reuse it for the **denoising
trajectory** in notebook 03 or **GAN samples over epochs** in 02. Here we
animate the cat → dog latent walk:
"""),
    code("""
from src.anim import frames_to_gif, show_gif

vae.eval()
with torch.no_grad():
    cmu = vae.encode(X[cat_idx[:256]].to(device))[0].mean(0)
    dmu = vae.encode(X[dog_idx[:256]].to(device))[0].mean(0)
    walk = [cmu + a * (dmu - cmu) for a in np.linspace(0, 1, 24)]   # interpolate
    frames = [vae.decode(z[None])[0].cpu() for z in walk]           # 24 x (3,32,32)

gif_path = frames_to_gif(frames, "cat_to_dog.gif", fps=12, scale=6)
print("wrote", gif_path)
show_gif(gif_path)
"""),

    md("""
### Why still a little soft?

Even with more capacity, VAE samples stay slightly blurry — and that's
fundamental, not a bug. The Gaussian likelihood + the KL term reward the decoder
for hedging (predicting the *average* of plausible pixels) rather than committing
to sharp detail. Levers that help, in order of impact:

- **More latent capacity / depth / epochs** (what we just did).
- **β-VAE**: down-weight the KL (`recon + β·KL`, β < 1) — sharper images, but a
  less regular latent space, so latent arithmetic gets noisier. Try it.
- **A perceptual or adversarial reconstruction loss** (e.g. VAE-GAN) — which is
  exactly the bridge to notebook **02 (GANs)** and **03 (diffusion)**, where
  we'll get crisp samples and then bring that power to jets.

So treat this softness as the baseline to beat across the next two notebooks.
"""),

    # ----------------------------------------------------------------- Part B
    md("""
## Part B · Jets in latent space

Now the physics. Jets are **point clouds** of particles — unordered sets — so we
encode them with a permutation-invariant **GNN** (DeepSets): a per-particle MLP,
a masked pool, then a head to the latent space. We train a jet VAE, then walk
the latent space from the **quark** region to the **gluon** region and watch the
jet mass.
"""),
    setup("""
quark = load_jetnet("q", num_particles=30, max_jets=5000)   # physical (etarel,phirel,ptrel)
gluon = load_jetnet("g", num_particles=30, max_jets=5000)
jets = np.concatenate([quark, gluon]).astype(np.float32)
is_gluon = np.concatenate([np.zeros(len(quark)), np.ones(len(gluon))])

# Standardize jets (log-pt + z-score) so the model targets the right per-feature
# scales and can learn the pt concentration. We train in standardized space and
# inverse-transform samples back to physical jets for the mass plot / displays.
jet_std = JetStandardizer().fit(jets)
jets_s = jet_std.transform(jets)
jet_loader = to_loader(jets_s, batch_size=BATCH)
print("jets:", jets.shape, "| standardized for training")


class JetVAE(nn.Module):
    def __init__(self, latent=LATENT_DIM, n_particles=30):
        super().__init__()
        self.encoder = DeepSetsEncoder(in_features=3, hidden=128, latent=latent * 2,
                                       mask_padding=False)   # standardized jets
        self.decoder = DeepSetsDecoder(latent=latent, n_particles=n_particles)
        self.latent = latent

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = h[:, :self.latent], h[:, self.latent:]
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decoder(z), mu, logvar

jvae = JetVAE().to(device)


def jet_recon(model, batch):
    x = batch[0]
    x_hat, mu, logvar = model(x)
    recon = F.mse_loss(x_hat, x)                       # simple per-feature MSE
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + 1e-3 * kl


# Train the jet VAE up front so the exercise can focus on the latent walk.
_jhist = train(jvae, jet_loader, jet_recon, epochs=EPOCHS, lr=1e-3, device=device)
print("JetVAE params:", sum(p.numel() for p in jvae.parameters()),
      "| final recon+KL:", round(float(_jhist[-1]), 4))
"""),

    md("""
### A look at the jets

Before generating any, see what a jet *is*: each particle is a dot in the
(η, φ) plane with size ∝ pₜ. Quark jets are narrow and collimated; gluon jets
are wider and busier — the structure the GNN has to learn (and the generators in
01/02 have to reproduce).
"""),
    code("""
plot_jets(quark[:3], titles=[f"quark {i}" for i in range(3)]); plt.show()
plot_jets(gluon[:3], titles=[f"gluon {i}" for i in range(3)]); plt.show()
"""),

    md("""
### The quark → gluon latent walk

A quick look at the latent *structure*: take the mean quark code and the mean
gluon code, and decode a few points along the line between them. The jets morph
from narrow (quark-like) to wide (gluon-like) — the latent space is organised by
the physics.
"""),
    code("""
jvae.eval()
with torch.no_grad():
    q_s = torch.as_tensor(jet_std.transform(quark[:512])).to(device)
    g_s = torch.as_tensor(jet_std.transform(gluon[:512])).to(device)
    z_q = jvae.encoder(q_s)[:, :LATENT_DIM].mean(0)
    z_g = jvae.encoder(g_s)[:, :LATENT_DIM].mean(0)
    steps = torch.stack([(1 - a) * z_q + a * z_g
                         for a in torch.linspace(0, 1, 3, device=device)])
    walk_jets = jet_std.inverse_transform(jvae.decoder(steps).cpu().numpy())  # -> physical
plot_jets(walk_jets, titles=["α=0 (quark)", "α=0.5", "α=1 (gluon)"]); plt.show()
"""),

    md("""
### Exercise 6 — Reconstruct gluon jets, watch the jet mass

The fair test of the VAE: **encode** real gluon jets to their latent mean and
**decode** them back. Overlay a few real jets against their reconstructions (same
axes, different colours), plot the jet-mass spine (real vs reconstructed), and
return the **1-Wasserstein distance** between the two mass distributions as
`mass_w1`. Good reconstructions sit almost on top of the real jets; the VAE's
softness shows up as a smeared high-mass tail.
"""),
    exercise(
        scaffold_body="""
jvae.eval()
real_np = gluon[:512]                                   # physical real jets
real_s = torch.as_tensor(jet_std.transform(real_np)).to(device)   # standardized for the model
with torch.no_grad():
    # encode to the latent MEAN (first LATENT_DIM dims), then decode:
    mu_g = ...
    recon_s = ...
recon_g = jet_std.inverse_transform(recon_s.cpu().numpy())        # -> physical jets
# real vs reconstructed: event displays (overlaid) and the jet-mass spine
plot_jet_overlay(real_np, recon_g, n=3, labels=("real gluon", "VAE recon")); plt.show()
plot_jet_mass(real_np, recon_g, labels=("real gluon", "VAE recon")); plt.show()
mass_w1 = ...
""",
        solution_body="""
jvae.eval()
real_np = gluon[:512]
real_s = torch.as_tensor(jet_std.transform(real_np)).to(device)
with torch.no_grad():
    mu_g = jvae.encoder(real_s)[:, :LATENT_DIM]
    recon_s = jvae.decoder(mu_g)
recon_g = jet_std.inverse_transform(recon_s.cpu().numpy())
plot_jet_overlay(real_np, recon_g, n=3, labels=("real gluon", "VAE recon")); plt.show()
plot_jet_mass(real_np, recon_g, labels=("real gluon", "VAE recon")); plt.show()
mass_w1 = round(jet_mass_w1(real_np, recon_g), 4)
print("reconstruction jet-mass W1:", mass_w1)
""",
        answer_var="mass_w1",
    ),

    md("""
### Exercise 7 — What did the VAE miss?

Look at your jet-mass overlay. VAEs are famously a bit *blurry* — they tend to
nail the bulk of a distribution but smear out sharp features. In **one word**,
where does your VAE most struggle to match the real gluon jets — the `"bulk"`
or the `"tail"`? Set `where_it_struggles` to that word.
"""),
    exercise(
        scaffold_body="""
# answer: string
where_it_struggles = ...   # "bulk" or "tail"
""",
        solution_body="""
where_it_struggles = "tail"
""",
        answer_var="where_it_struggles",
    ),

    md("""
## Recap

- A VAE gives a **smooth latent space** you can sample and interpolate in.
- **Latent arithmetic** works because the space is organised by meaningful
  directions (cat → dog; quark → gluon).
- On jets, a **permutation-invariant GNN encoder** respects the point-cloud
  structure — and the **jet-mass spine** shows the classic VAE weakness in the
  tail, which we'll compare against a GAN (02) and diffusion (03, 05).
"""),
]

build(OUT, cells)
