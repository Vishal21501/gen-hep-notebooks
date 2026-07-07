"""calodiff_probe — a friendly handle on the pretrained CaloDiffusion models.

Students shouldn't have to fight the 2023 paper code. This wraps it so a capstone
notebook can load a real model + real showers in a couple of lines, for any of the
three particle types the paper ships:

    import calodiff_probe as cp
    model = cp.load_model("electron")          # or "photon" / "pion"
    x, E, E_inc = cp.load_showers("electron", n=64)
    x_t, eps  = cp.noise_at(model, x, t=100)   # forward (noising) process
    eps_hat   = cp.predict_noise(model, x_t, E, t=100)   # the model's noise estimate
    grid      = cp.regular_grid(model, x)      # (N, layers, angular, radial) for plotting

Particle geometries (the model's *regular* grid, layer axis = shower depth):
    electron : 45 x 16 x 9   (DS2, regular native geometry)
    photon   :  5 x 10 x 30  (DS1, GLaM-mapped from 368 irregular voxels)
    pion     :  7 x 10 x 23  (DS1, GLaM-mapped from 533 irregular voxels)

Conventions (match the math in the notebooks):
  * x      : preprocessed shower tensor the model eats (logit-normed)
  * E      : conditioning energy, shape (N,)   [log-scaled incident E — what the net sees]
  * E_inc  : incident energy in GeV, shape (N,) [the physical number]
  * t      : diffusion step, 0 (clean) .. 399 (pure noise)
"""
import os, sys, types

_ROOT   = os.path.dirname(os.path.abspath(__file__))           # capstone/calodiffusion
_VENDOR = os.path.join(_ROOT, "vendor")                        # the paper's code (unmodified)
_pylibs = os.path.join(_ROOT, "pylibs")                        # optional local deps (dev only)
if os.path.isdir(_pylibs):
    sys.path.insert(0, _pylibs)
sys.path.insert(0, os.path.join(_VENDOR, "scripts"))
os.environ.setdefault("MPLBACKEND", "Agg")

# torchinfo is only used for a one-time model summary printout in the paper code.
# Stub it if absent so the sole hard extra dependency is einops.
try:
    import torchinfo                                           # noqa: F401
except ImportError:
    torchinfo = types.ModuleType("torchinfo"); sys.modules["torchinfo"] = torchinfo
torchinfo.summary = lambda *a, **k: None

_here = os.getcwd(); os.chdir(os.path.join(_VENDOR, "scripts"))  # utils' "../CaloChallenge" is relative
import yaml, numpy as np, torch
from CaloDiffu import CaloDiffu
from utils import XMLHandler, NNConverter
import utils
os.chdir(_here)

def _pick_device():
    """cuda if present, else cpu. Override with CALODIFF_DEVICE=cuda|mps|cpu.
    (mps is opt-in only: 3D convs are unreliable on Apple GPUs in some torch builds.)"""
    env = os.environ.get("CALODIFF_DEVICE")
    if env:
        return torch.device(env)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = _pick_device()          # GPU when available; probing is fine on CPU, generation wants GPU
NSTEPS = 400

# ---- per-particle wiring -------------------------------------------------- #
# All three shower caches live in this folder's data/ (self-contained). Override
# any path with the matching env var.
_HERE_DATA = os.path.join(_ROOT, "data")
PARTICLES = {
    "electron": dict(config="config_dataset2.json", ckpt="dataset2.pth",
                     data=os.environ.get("CALO_DS2", os.path.join(_HERE_DATA, "calochallenge_ds2.hdf5"))),
    "photon":   dict(config="config_dataset1_photon.json", ckpt="dataset1_photon.pth",
                     data=os.environ.get("CALO_DS1_PHOTON", os.path.join(_HERE_DATA, "calochallenge_ds1_photon.hdf5")),
                     xml="photon"),
    "pion":     dict(config="config_dataset1_pion.json", ckpt="dataset1_pion.pth",
                     data=os.environ.get("CALO_DS1_PION", os.path.join(_HERE_DATA, "calochallenge_ds1_pion.hdf5")),
                     xml="pion"),
}


def _cfg(particle):
    return yaml.safe_load(open(os.path.join(_VENDOR, "configs", PARTICLES[particle]["config"])))


def load_model(particle="electron"):
    """Return the pretrained CaloDiffusion model for `particle`, weights loaded, eval mode.
    The returned model carries `.config`, `._particle`, and `._orig_shape` for the helpers."""
    p = PARTICLES[particle]; cfg = _cfg(particle)
    orig_shape = "orig" in cfg.get("SHOWER_EMBED", "")
    NN_embed = None
    if "NN" in cfg.get("SHOWER_EMBED", ""):
        binxml = os.path.join(_VENDOR, "CaloChallenge", "code",
                              f"binning_dataset_1_{'photons' if particle=='photon' else 'pions'}.xml")
        NN_embed = NNConverter(bins=XMLHandler(p["xml"], binxml)).to(DEVICE)
    shape = cfg["SHAPE_ORIG"][1:] if orig_shape else cfg["SHAPE_PAD"][1:]
    m = CaloDiffu(shape, config=cfg, training_obj=cfg["TRAINING_OBJ"],
                  nsteps=NSTEPS, NN_embed=NN_embed).to(DEVICE).eval()
    saved = torch.load(os.path.join(_VENDOR, "trained_models", p["ckpt"]),
                       map_location=DEVICE, weights_only=False)
    m.load_state_dict(saved.get("model_state_dict", saved) if isinstance(saved, dict) else saved)
    m._particle, m._orig_shape, m.config = particle, orig_shape, cfg
    return m


def load_showers(particle="electron", n=64, start=0, data_path=None):
    """Load `n` real showers for `particle`, preprocessed exactly as the model was trained.
    Returns (x, E, E_inc):  x -> model-ready tensor,  E -> (n,) conditioning,  E_inc -> (n,) GeV."""
    p = PARTICLES[particle]; cfg = _cfg(particle)
    orig_shape = "orig" in cfg.get("SHOWER_EMBED", "")
    showers, energies = utils.DataLoader(
        data_path or p["data"], shape=cfg["SHAPE_PAD"], emax=cfg["EMAX"], emin=cfg["EMIN"],
        nevts=n, evt_start=start, max_deposit=cfg["MAXDEP"], ecut=cfg["ECUT"],
        logE=cfg["logE"], showerMap=cfg["SHOWERMAP"], dataset_num=cfg["DATASET_NUM"],
        orig_shape=orig_shape)
    x = torch.tensor(showers, dtype=torch.float32, device=DEVICE)
    if not orig_shape:
        x = x.reshape([-1, *cfg["SHAPE_PAD"][1:]])
    E = torch.tensor(energies.ravel(), dtype=torch.float32, device=DEVICE)
    E_inc = torch.tensor(_e_to_gev(energies.ravel(), cfg), dtype=torch.float32, device=DEVICE)
    return x, E, E_inc


def encode_energy(particle, E_inc_gev):
    """Physical incident energy (GeV) -> the log-scaled number the model conditions on."""
    cfg = _cfg(particle); e = np.asarray(E_inc_gev, dtype=np.float64)
    return torch.tensor(np.log10(e / cfg["EMIN"]) / np.log10(cfg["EMAX"] / cfg["EMIN"]),
                        dtype=torch.float32, device=DEVICE)


@torch.no_grad()
def noise_at(model, x, t, noise=None):
    """Forward process: return (x_t, noise) — x noised to step `t` (int or (N,) tensor)."""
    tt = _as_t(t, x.shape[0])
    if noise is None:
        noise = torch.randn_like(x)
    return model.noise_image(x, tt, noise=noise), noise


@torch.no_grad()
def predict_noise(model, x_t, E, t):
    """The model's estimate of the noise in `x_t` — the core learned 'score'."""
    t_emb = model.do_time_embed(_as_t(t, x_t.shape[0]), model.time_embed)
    return model.pred(x_t, E, t_emb)


@torch.no_grad()
def sample(model, E, num_steps=None, sample_offset=2, debug=False):
    """Generate showers from noise, conditioned on `E` (shape (N,)). Returns model-space x.
    ~50s/batch on CPU, ~batch-size-independent, so batch it. `sample_offset=2` skips the two
    unstable steps (leave at 0 and generation blows up — a good thing for students to discover).
    debug=True also returns (x_final, xs, x0s): the noisy path and the running clean-shower
    prediction (the 'shower emerging from noise' sequence to animate)."""
    out = model.Sample(E, num_steps=num_steps or NSTEPS, sample_offset=sample_offset, debug=debug)
    if debug:
        final, xs, x0s = out
        return torch.tensor(final, device=DEVICE), xs, x0s
    return torch.tensor(out, device=DEVICE)


@torch.no_grad()
def regular_grid(model, x):
    """Model-space x -> (N, layers, angular, radial) numpy in the *regular* geometry.
    This is the space where the layer axis is shower depth, for ALL particles:
    electron is already regular; photon/pion are GLaM-mapped from their irregular voxels.
    Great for plotting and for the depth/energy-conditioning exercises."""
    t = x if torch.is_tensor(x) else torch.as_tensor(np.asarray(x), dtype=torch.float32, device=DEVICE)
    if model.NN_embed is not None:              # DS1: encode irregular -> regular grid
        t = model.NN_embed.enc(t).to(DEVICE)
    return np.asarray(t.detach().cpu()).squeeze(1)   # drop channel -> (N, L, A, R)


def to_physical(model, x, E_inc_gev):
    """Undo preprocessing: model-space x -> physical shower in GeV, native voxel layout
    (electron (N,45,16,9); photon (N,368); pion (N,533)). This is deposited energy per voxel."""
    cfg = model.config
    arr = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    flat = arr.reshape(arr.shape[0], -1)
    e = np.asarray(E_inc_gev).reshape(-1, 1)
    e_scaled = np.log10(e / cfg["EMIN"]) / np.log10(cfg["EMAX"] / cfg["EMIN"])
    voxels, _ = utils.ReverseNorm(flat, e_scaled, shape=cfg["SHAPE_PAD"],
                                  emax=cfg["EMAX"], emin=cfg["EMIN"], max_deposit=cfg["MAXDEP"],
                                  logE=cfg["logE"], showerMap=cfg["SHOWERMAP"],
                                  dataset_num=cfg["DATASET_NUM"], ecut=cfg["ECUT"],
                                  orig_shape=model._orig_shape)
    voxels = np.asarray(voxels)
    if not model._orig_shape:
        voxels = voxels.reshape(-1, *cfg["SHAPE_PAD"][1:]).squeeze(1)
    return voxels


def _e_to_gev(e_scaled, cfg):
    return (cfg["EMIN"] * (cfg["EMAX"] / cfg["EMIN"]) ** np.asarray(e_scaled, dtype=np.float64)).astype(np.float32)


def _as_t(t, n):
    return t.to(DEVICE).long() if torch.is_tensor(t) else torch.full((n,), int(t), device=DEVICE, dtype=torch.long)
