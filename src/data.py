"""Dataset loaders with synthetic fallbacks.

Every loader returns real data when the underlying package/download is present,
and a synthetic stand-in otherwise — so a notebook always runs end-to-end in a
class, even before the school's data cache is warm. Each loader prints which
path it took so there's no silent surprise.

Real sources: JetNet (`jetnet`), CaloChallenge (HDF5), AFHQ / QuickDraw / pixel
sprites (image tensors). Synthetic stand-ins match the real tensors' shapes and
rough statistics so the models and the jet-mass plot behave the same.
"""

from __future__ import annotations

import os as _os

import numpy as np

# Default data folder, resolved relative to THIS file (repo_root/data) rather
# than the kernel's CWD — so loaders find data/ whether a notebook runs from
# notebooks/, the repo root, or anywhere else.
_DEFAULT_DATA_DIR = _os.path.normpath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "data")
)


# --------------------------------------------------------------------------- #
# Jets (the physics spine)
# --------------------------------------------------------------------------- #

def synthetic_jets(n_jets=5000, n_particles=30, jet_type="g", seed=0):
    """Point clouds of (eta_rel, phi_rel, pt_rel) with a realistic-ish mass
    spectrum. Gluon ('g') jets get more, softer constituents than quark ('q')
    jets, so the two separate in the jet-mass plot — enough structure to teach
    with when the real JetNet download isn't available yet."""
    rng = np.random.default_rng(seed)
    spread = 0.12 if jet_type == "g" else 0.08
    n_active = rng.integers(n_particles // 2, n_particles + 1, size=n_jets)
    jets = np.zeros((n_jets, n_particles, 3), dtype=np.float32)
    for i, k in enumerate(n_active):
        eta = rng.normal(0, spread, k)
        phi = rng.normal(0, spread, k)
        pt = rng.dirichlet(np.ones(k) * (0.7 if jet_type == "g" else 1.3))
        jets[i, :k] = np.stack([eta, phi, pt.astype(np.float32)], axis=1)
    return jets


def load_jetnet(jet_type="g", num_particles=30, max_jets=None, data_dir=_DEFAULT_DATA_DIR):
    """JetNet particle clouds, shape (n_jets, num_particles, 3) = (etarel,
    phirel, ptrel). Falls back to `synthetic_jets` if `jetnet` isn't installed
    or the download fails."""
    try:
        from jetnet.datasets import JetNet
        data = JetNet(jet_type=jet_type, data_dir=data_dir,
                      particle_features=["etarel", "phirel", "ptrel"],
                      num_particles=num_particles, download=True)
        jets = np.asarray(data.particle_data, dtype=np.float32)[..., :3]
        if max_jets:
            jets = jets[:max_jets]
        print(f"[data] loaded {len(jets)} real JetNet '{jet_type}' jets")
        return jets
    except Exception as e:  # noqa: BLE001 - any failure -> synthetic
        n = max_jets or 5000
        print(f"[data] JetNet unavailable ({type(e).__name__}); using synthetic jets")
        return synthetic_jets(n_jets=n, n_particles=num_particles, jet_type=jet_type)


# --------------------------------------------------------------------------- #
# Calorimeter showers (the second modality, notebook 04)
# --------------------------------------------------------------------------- #

def synthetic_showers(n=4000, shape=(1, 16, 16), seed=0):
    """Energy-conditioned voxel showers: a radially-decaying blob whose total
    energy tracks the (returned) incident energy, with Poisson-ish noise. Stand-
    in for CaloChallenge until the real HDF5 is mounted."""
    rng = np.random.default_rng(seed)
    c, h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    r2 = (yy - h / 2) ** 2 + (xx - w / 2) ** 2
    incident = rng.uniform(1.0, 100.0, size=n).astype(np.float32)  # GeV
    showers = np.empty((n, c, h, w), dtype=np.float32)
    for i, e in enumerate(incident):
        core = np.exp(-r2 / (2 * (1.5 + 0.02 * e) ** 2))
        showers[i, 0] = (e * core / core.sum()) * rng.gamma(8.0, 1 / 8.0, size=(h, w))
    return showers, incident


def load_calochallenge(path=None, max_showers=4000):
    """Load CaloChallenge showers + incident energies from HDF5 if `path` is
    given and readable, else synthetic. Returns (showers, incident_energy).

    This is the *lesson* loader (notebook 04) — small `(N, 1, 16, 16)` images
    with a synthetic fallback so the mechanism notebook runs offline. The
    **capstone** uses the real-only `load_calochallenge_ds2` below."""
    if path:
        try:
            import h5py
            with h5py.File(path, "r") as f:
                showers = np.asarray(f["showers"][:max_showers], dtype=np.float32)
                inc = np.asarray(f["incident_energies"][:max_showers], dtype=np.float32).ravel()
            print(f"[data] loaded {len(showers)} real CaloChallenge showers")
            return showers, inc
        except Exception as e:  # noqa: BLE001
            print(f"[data] CaloChallenge load failed ({type(e).__name__}); using synthetic")
    else:
        print("[data] no CaloChallenge path given; using synthetic showers")
    return synthetic_showers(n=max_showers)


# --- CaloChallenge Dataset 2 (electrons) — the capstone's REAL data ---------- #
#
# Zenodo record 6366271. `dataset_2_1.hdf5` is 100k GEANT4 electron showers,
# incident energy log-uniform over 1 GeV–1 TeV. Each shower is 6480 voxels =
# 45 longitudinal layers x 9 radial x 16 angular. The HDF5 stores energies in
# **MeV**: `showers` (N, 6480) and `incident_energies` (N, 1). We reshape to the
# (layers, radial, angular) grid and convert to GeV. No synthetic fallback — the
# capstone runs on real data only (run tools/prepare_data.py calochallenge first).
CALO_DS2_SHAPE = (45, 9, 16)                 # (layers, radial, angular)
CALO_DS2_VOXELS = 45 * 9 * 16                 # = 6480
_CALO_DS2_CACHE = "calochallenge_ds2.hdf5"    # skimmed copy lives in data/

# The skimmed cache is ~360 MB — too big to commit (GitHub's 100 MB/file limit),
# so ship it as a GitHub *Release asset* (or any read-only URL) and point this at
# it. When the local cache is missing, load_calochallenge_ds2 fetches it once.
# Override per-call via `url=`, or with the CALO_DS2_CACHE_URL env var.
CALO_DS2_CACHE_URL = None


def _calo_ds2_cache_path(data_dir):
    import os
    return os.path.join(data_dir, _CALO_DS2_CACHE)


def load_calochallenge_ds2(path=None, max_showers=None, data_dir=_DEFAULT_DATA_DIR, url=None):
    """Real CaloChallenge **Dataset 2** electron showers — the capstone data.

    Returns `(showers, incident_energy)`:
      * showers: `(N, 45, 9, 16)` float32, **GeV** per voxel (layers, radial, angular),
      * incident_energy: `(N,)` float32, **GeV**.

    Load order: the local cache `data/calochallenge_ds2.hdf5` (built once by
    `tools/prepare_data.py calochallenge`); else a one-time fetch of that cache
    from `url` / `CALO_DS2_CACHE_URL` / the env var (e.g. a GitHub Release asset),
    saved locally so it's instant thereafter. There is **no synthetic fallback**:
    if the file can't be found *or* fetched this raises, so a class never silently
    trains on fake showers.
    """
    import os
    import h5py
    path = path or _calo_ds2_cache_path(data_dir)
    if not os.path.exists(path):
        src_url = url or CALO_DS2_CACHE_URL or os.environ.get("CALO_DS2_CACHE_URL")
        if src_url:
            try:
                print(f"[data] fetching CaloChallenge DS2 cache from {src_url} (~360 MB, once) ...")
                _download_to(path, src_url, timeout=600, progress=True)
                print(f"[data] saved cache to {path}")
            except Exception as e:  # noqa: BLE001 - fall through to the clear error below
                print(f"[data] cache fetch failed ({type(e).__name__}: {e})")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CaloChallenge Dataset 2 not found at {path}. Either build it locally:\n"
            f"    python tools/prepare_data.py calochallenge\n"
            f"(downloads dataset_2_1.hdf5 from Zenodo 6366271 and skims it), or set "
            f"CALO_DS2_CACHE_URL / pass url= to fetch the prepared cache.")
    with h5py.File(path, "r") as f:
        sl = slice(0, max_showers) if max_showers else slice(None)
        flat = np.asarray(f["showers"][sl], dtype=np.float32)        # (N, 6480) MeV
        inc = np.asarray(f["incident_energies"][sl], dtype=np.float32).ravel()  # MeV
    if flat.shape[1] != CALO_DS2_VOXELS:
        raise ValueError(f"expected {CALO_DS2_VOXELS} voxels/shower, got {flat.shape[1]} "
                         f"— is this really CaloChallenge Dataset 2?")
    showers = flat.reshape(-1, *CALO_DS2_SHAPE) / 1000.0             # -> GeV grid
    inc = inc / 1000.0                                               # -> GeV
    print(f"[data] loaded {len(showers)} real CaloChallenge DS2 showers "
          f"{tuple(showers.shape)} | incident GeV {inc.min():.1f}->{inc.max():.0f}")
    return showers, inc


def save_calochallenge_cache(showers_flat, incident, data_dir=_DEFAULT_DATA_DIR):
    """Write a skimmed, gzip-compressed copy of the Dataset 2 HDF5 (same keys /
    units / `(N, 6480)` layout as Zenodo, just fewer rows) that
    `load_calochallenge_ds2` reads. Showers are sparse, so gzip shrinks it a lot."""
    import os
    import h5py
    os.makedirs(data_dir, exist_ok=True)
    path = _calo_ds2_cache_path(data_dir)
    showers_flat = np.asarray(showers_flat, dtype=np.float32)
    n = len(showers_flat)
    with h5py.File(path, "w") as f:
        # Row-wise chunks (whole showers per chunk) so reading the first N rows is
        # a handful of contiguous gzip blocks — fast partial reads, unlike the
        # Zenodo original's (391, 51) chunking which needs hundreds of requests.
        f.create_dataset("showers", data=showers_flat,
                         chunks=(min(256, n), showers_flat.shape[1]),
                         compression="gzip", compression_opts=4)
        f.create_dataset("incident_energies",
                         data=np.asarray(incident, dtype=np.float32).reshape(-1, 1),
                         compression="gzip", compression_opts=4)
    return path


# --------------------------------------------------------------------------- #
# Image datasets (the fun spine: AFHQ / QuickDraw / sprites)
# --------------------------------------------------------------------------- #

def synthetic_images(n=2000, shape=(3, 32, 32), n_modes=3, seed=0):
    """Coloured blobs in a handful of modes — enough to demonstrate
    reconstruction, mode collapse and denoising without a real download."""
    rng = np.random.default_rng(seed)
    c, h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    imgs = np.zeros((n, c, h, w), dtype=np.float32)
    for i in range(n):
        m = i % n_modes
        cy, cx = rng.uniform(h * 0.3, h * 0.7), rng.uniform(w * 0.3, w * 0.7)
        blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (h / 5) ** 2))
        colour = np.eye(3)[m % 3] if c == 3 else np.array([1.0])
        imgs[i] = blob[None] * colour[:, None, None]
    return np.clip(imgs, 0, 1)


# HF dataset ids to try for AFHQ, in order. The first that resolves wins, so a
# rename upstream or a mirror going down doesn't break the loader.
_AFHQ_HF_IDS = ("huggan/AFHQ", "Dmini/AFHQ-512", "nielsr/afhq")
_AFHQ_CANON = {"cat": 0, "dog": 1, "wild": 2}   # our label convention


# Set this (or the AFHQ_CACHE_URL env var) to a URL on your teaching interface
# that serves the small afhq_<size>.npz built by tools/prepare_data.py. When the
# local cache is missing, load_afhq fetches it once from here — no per-student
# HuggingFace download, no `datasets` install needed at class time.
AFHQ_CACHE_URL = None


def _afhq_cache_path(image_size, data_dir):
    import os
    return os.path.join(data_dir, f"afhq_{image_size}.npz")


def _download_to(path, url, timeout=120, chunk=1 << 20, progress=False):
    """Download `url` to `path` atomically (via a .tmp then rename), streaming in
    `chunk`-byte pieces so a big file (e.g. the ~360 MB calo cache) never loads
    fully into RAM. With `progress`, print a one-line MB counter — useful when a
    student's Binder session fetches the cache on the first cell."""
    import os, urllib.request
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while True:
            buf = r.read(chunk)
            if not buf:
                break
            f.write(buf); done += len(buf)
            if progress and total:
                print(f"\r[data]   {done/1e6:6.0f} / {total/1e6:.0f} MB "
                      f"({100*done/total:4.1f}%)", end="", flush=True)
    if progress and total:
        print()
    os.replace(tmp, path)


def afhq_from_hf(n=4500, image_size=32):
    """Stream AFHQ from HuggingFace and resize into `(images, labels)`.

    images: (~n, 3, image_size, image_size) float32 in [0, 1]; labels 0/1/2,
    sampled BALANCED across cat/dog/wild (n//3 each). Uses `streaming=True`, so
    the multi-GB dataset is NEVER cached to disk — examples stream through, get
    resized to tiny arrays, and the raw data is discarded. Final disk cost is
    just the small `.npz`. Call once via `tools/prepare_data.py`."""
    from datasets import load_dataset
    from PIL import Image

    ds = None
    last_err = None
    for hid in _AFHQ_HF_IDS:
        try:
            ds = load_dataset(hid, split="train", streaming=True)
            break
        except Exception as e:  # noqa: BLE001 - try the next mirror
            last_err = e
    if ds is None:
        raise RuntimeError(f"no AFHQ dataset id resolved on HF ({last_err})")

    feats = getattr(ds, "features", None) or {}
    img_col = next((k for k, v in feats.items() if v.__class__.__name__ == "Image"), None)
    lab_col = next((k for k, v in feats.items() if v.__class__.__name__ == "ClassLabel"), None)
    names = getattr(feats.get(lab_col), "names", None) if lab_col else None
    remap = ({i: _AFHQ_CANON.get(nm.lower(), 2) for i, nm in enumerate(names)}
             if names else {0: 0, 1: 1, 2: 2})

    per_class = max(1, n // 3)
    counts = {0: 0, 1: 0, 2: 0}
    imgs, labels = [], []
    for row in ds:
        # Discover columns from the first row if the stream didn't expose features.
        if img_col is None:
            img_col = next(k for k, v in row.items() if hasattr(v, "convert"))
        if lab_col is None:
            lab_col = next((k for k, v in row.items() if isinstance(v, int)), None)
        lab = remap.get(row[lab_col], 2) if lab_col is not None else (len(imgs) % 3)
        if counts[lab] >= per_class:
            if all(c >= per_class for c in counts.values()):
                break
            continue
        im = row[img_col].convert("RGB").resize((image_size, image_size), Image.BILINEAR)
        imgs.append(np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        labels.append(lab)
        counts[lab] += 1
    return np.stack(imgs), np.array(labels, dtype=np.int64)


def save_afhq_cache(images, labels, image_size=32, data_dir=_DEFAULT_DATA_DIR):
    """Write a small uint8 cache (~18 MB for 6000 @ 32px) that `load_afhq`
    reads instantly. This is what you ship in the repo / on the interface."""
    import os
    os.makedirs(data_dir, exist_ok=True)
    path = _afhq_cache_path(image_size, data_dir)
    u8 = np.clip(images * 255.0, 0, 255).astype(np.uint8).transpose(0, 2, 3, 1)  # NHWC
    np.savez_compressed(path, images=u8, labels=labels.astype(np.int16))
    return path


def load_afhq_local(n=4000, image_size=32, data_dir=_DEFAULT_DATA_DIR):
    """Custom loader — read the prepared AFHQ file straight from the local data
    folder and nothing else.

    Expects `data/afhq_<size>.npz` (uint8 images + labels), built once by
    `tools/prepare_data.py afhq` and shipped with the course / dropped in by the
    provisioning step. Returns `(images, labels)` with images (n,3,H,W) float in
    [0,1] and labels 0=cat/1=dog/2=wild. If the file is missing it prints a clear
    pointer and falls back to coloured synthetic blobs so class never hard-stops.
    """
    import os
    path = _afhq_cache_path(image_size, data_dir)
    if os.path.exists(path):
        d = np.load(path)
        imgs = d["images"][:n].astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        labels = d["labels"][:n].astype(np.int64)
        n_cat = int((labels == 0).sum()); n_dog = int((labels == 1).sum())
        print(f"[data] loaded {len(imgs)} AFHQ images from {path} "
              f"(cats={n_cat}, dogs={n_dog}, wild={len(imgs) - n_cat - n_dog})")
        return imgs, labels
    print(f"[data] {path} not found — run `python tools/prepare_data.py afhq` to "
          f"build it (or drop the provisioned file there). Using synthetic blobs.")
    imgs = synthetic_images(n=n, shape=(3, image_size, image_size))
    return imgs, np.array([i % 3 for i in range(len(imgs))], dtype=np.int64)


def load_afhq(n=4000, image_size=32, data_dir=_DEFAULT_DATA_DIR, url=None):
    """Real AFHQ animal faces as `(images, labels)`.

    images: (n, 3, image_size, image_size) float32 in [0, 1].
    labels: int array, 0=cat, 1=dog, 2=wild (so the cat->dog vector in 01 is real).

    Load order, fastest first:
      1. the small `data/afhq_<size>.npz` cache (instant — built once by
         `tools/prepare_data.py`, shipped in the repo / on the interface);
      2. a one-time fetch of that cache from `url` (or `AFHQ_CACHE_URL` / the
         env var) — e.g. your teaching interface; saved locally so it's instant
         thereafter, and needs no `datasets` install;
      3. a live HuggingFace download (slow — also writes the cache);
      4. coloured synthetic blobs, so the notebook still runs fully offline.
    """
    import os
    cache = _afhq_cache_path(image_size, data_dir)
    # (2) try fetching the prepared cache from the interface before anything heavy.
    if not os.path.exists(cache):
        src_url = url or AFHQ_CACHE_URL or os.environ.get("AFHQ_CACHE_URL")
        if src_url:
            try:
                _download_to(cache, src_url)
                print(f"[data] fetched AFHQ cache from {src_url}")
            except Exception as e:  # noqa: BLE001 - fall through to HF / synthetic
                print(f"[data] cache fetch failed ({type(e).__name__}: {e}); trying HuggingFace")
    if os.path.exists(cache):
        d = np.load(cache)
        imgs = d["images"][:n].astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        labels = d["labels"][:n].astype(np.int64)
        print(f"[data] loaded {len(imgs)} AFHQ images from cache ({cache})")
        return imgs, labels
    try:
        imgs, labels = afhq_from_hf(n=max(n, 6000), image_size=image_size)
        try:
            save_afhq_cache(imgs, labels, image_size, data_dir)  # speed up next run
        except Exception:  # noqa: BLE001 - caching is best-effort
            pass
        imgs, labels = imgs[:n], labels[:n]
        n_cat = int((labels == 0).sum()); n_dog = int((labels == 1).sum())
        print(f"[data] downloaded {len(imgs)} real AFHQ images "
              f"(cats={n_cat}, dogs={n_dog}, wild={len(imgs) - n_cat - n_dog})")
        return imgs, labels
    except Exception as e:  # noqa: BLE001
        print(f"[data] AFHQ unavailable ({type(e).__name__}: {e}); using synthetic blobs")
        imgs = synthetic_images(n=n, shape=(3, image_size, image_size))
        return imgs, np.array([i % 3 for i in range(len(imgs))], dtype=np.int64)


# A few visually-distinct QuickDraw categories -> the GAN's "modes" in notebook
# 02. Order defines the label index (0=cat, 1=fish, ...).
QUICKDRAW_CATEGORIES = ("cat", "fish", "apple", "bicycle")
_QUICKDRAW_URL = ("https://storage.googleapis.com/quickdraw_dataset/full/"
                  "numpy_bitmap/{cat}.npy")


def _fetch_npy_rows(url, n):
    """Fetch only the first `n` rows of a remote .npy via HTTP range requests —
    QuickDraw category files are huge (100k+ rows), but we read just n*784 bytes
    instead of the whole thing, so disk/bandwidth stay tiny."""
    import ast, urllib.request

    def _range(a, b):
        req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
        return urllib.request.urlopen(req, timeout=60).read()

    head = _range(0, 255)
    if head[:6] != b"\x93NUMPY":
        raise ValueError("not a .npy stream")
    major = head[6]
    if major == 1:
        hlen = int.from_bytes(head[8:10], "little"); start = 10 + hlen
        hdr = head[10:10 + hlen]
    else:
        hlen = int.from_bytes(head[8:12], "little"); start = 12 + hlen
        hdr = head[12:12 + hlen]
    meta = ast.literal_eval(hdr.decode("latin1").strip())
    rows, cols = meta["shape"]
    n = min(n, rows)
    raw = _range(start, start + n * cols - 1)
    return np.frombuffer(raw, dtype=np.uint8).reshape(n, cols)


def quickdraw_from_web(n=4000, categories=QUICKDRAW_CATEGORIES):
    """Stream a balanced QuickDraw set (n//len(categories) per category) as
    `(images, labels)`. images: (~n, 1, 28, 28) float [0,1]; labels = category
    index. Space-light (range requests). Call once via tools/prepare_data.py."""
    per = max(1, n // len(categories))
    imgs, labels = [], []
    for idx, cat in enumerate(categories):
        arr = _fetch_npy_rows(_QUICKDRAW_URL.format(cat=cat), per)
        imgs.append(arr.reshape(-1, 28, 28))
        labels.append(np.full(len(arr), idx, dtype=np.int64))
    images = (np.concatenate(imgs)[:, None].astype(np.float32) / 255.0)
    return images, np.concatenate(labels)


def save_quickdraw_cache(images, labels, data_dir=_DEFAULT_DATA_DIR):
    """Write the small uint8 cache (~few MB) that `load_quickdraw_local` reads."""
    import os
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "quickdraw.npz")
    u8 = np.clip(images * 255.0, 0, 255).astype(np.uint8)
    np.savez_compressed(path, images=u8, labels=labels.astype(np.int16))
    return path


def load_quickdraw_local(n=4000, data_dir=_DEFAULT_DATA_DIR):
    """Real QuickDraw doodles as `(images, labels)` from the local data folder
    (`data/quickdraw.npz`, built by `tools/prepare_data.py quickdraw`). images:
    (n,1,28,28) float [0,1]; labels = category index. Falls back to synthetic
    blobs (with i%3 labels) if the file is missing."""
    import os
    path = os.path.join(data_dir, "quickdraw.npz")
    if os.path.exists(path):
        d = np.load(path)
        imgs = d["images"][:n].astype(np.float32) / 255.0
        labels = d["labels"][:n].astype(np.int64)
        cats = ", ".join(f"{QUICKDRAW_CATEGORIES[i]}={int((labels == i).sum())}"
                         for i in np.unique(labels))
        print(f"[data] loaded {len(imgs)} QuickDraw doodles from {path} ({cats})")
        return imgs, labels
    print(f"[data] {path} not found — run `python tools/prepare_data.py quickdraw`. "
          f"Using synthetic blobs.")
    imgs = synthetic_images(n=n, shape=(1, 28, 28))
    return imgs, np.array([i % 3 for i in range(len(imgs))], dtype=np.int64)


# --------------------------------------------------------------------------- #
# Creature sprites (notebook 03) — openly-licensed Twemoji (CC-BY 4.0)
# --------------------------------------------------------------------------- #
# Twemoji is Twitter's emoji set, CC-BY 4.0 — the brief's sanctioned, copyright-
# clean stand-in for "creature sprites" (no Pokemon assets). We grab ~60 animal /
# monster / face emoji and augment (flips + rotations) into a small training set.
_TWEMOJI_CDNS = (
    "https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/{cp}.png",
    "https://cdn.jsdelivr.net/gh/jdecked/twemoji@15.1.0/assets/72x72/{cp}.png",
)
TWEMOJI_CREATURES = [
    "1f436", "1f431", "1f42d", "1f439", "1f430", "1f98a", "1f43b", "1f43c",
    "1f428", "1f42f", "1f981", "1f42e", "1f437", "1f438", "1f435", "1f414",
    "1f427", "1f426", "1f424", "1f986", "1f985", "1f989", "1f987", "1f43a",
    "1f417", "1f434", "1f984", "1f41d", "1f41b", "1f98b", "1f40c", "1f41e",
    "1f41c", "1f577", "1f982", "1f422", "1f40d", "1f98e", "1f996", "1f995",
    "1f419", "1f991", "1f990", "1f980", "1f421", "1f420", "1f41f", "1f42c",
    "1f433", "1f988", "1f40b", "1f47e", "1f47d", "1f479", "1f47a", "1f47b",
    "1f480", "1f608", "1f47f", "1f916", "1f383", "1f409", "1f432",
]


def _fetch_url_bytes(url, timeout=60):
    import urllib.request
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def _fetch_twemoji(cp, size=32):
    """Fetch one Twemoji PNG, composite its transparency onto white, resize ->
    (size, size, 3) float in [0, 1]."""
    import io
    from PIL import Image
    last = None
    for tmpl in _TWEMOJI_CDNS:
        try:
            data = _fetch_url_bytes(tmpl.format(cp=cp))
            im = Image.open(io.BytesIO(data)).convert("RGBA").resize((size, size), Image.BILINEAR)
            bg = Image.new("RGBA", (size, size), (255, 255, 255, 255))
            return np.asarray(Image.alpha_composite(bg, im).convert("RGB"),
                              dtype=np.float32) / 255.0
        except Exception as e:  # noqa: BLE001 - try the next CDN
            last = e
    raise RuntimeError(f"could not fetch twemoji {cp}: {last}")


def sprites_from_web(n=2000, image_size=32, seed=0):
    """Fetch the Twemoji creature set and augment (random h-flip + rotation) into
    `n` sprites. Returns (images, labels): images (n, 3, H, W) float [0,1];
    labels = base-emoji index. Call once via tools/prepare_data.py."""
    from PIL import Image
    rng = np.random.default_rng(seed)
    bases, base_labels = [], []
    for idx, cp in enumerate(TWEMOJI_CREATURES):
        try:
            bases.append(_fetch_twemoji(cp, image_size))
            base_labels.append(idx)
        except Exception:  # noqa: BLE001 - skip any that fail to fetch
            continue
    if not bases:
        raise RuntimeError("no Twemoji sprites could be fetched")
    bases = (np.stack(bases) * 255).astype(np.uint8)            # (B, H, W, 3)
    imgs = np.empty((n, image_size, image_size, 3), dtype=np.float32)
    labels = np.empty(n, dtype=np.int64)
    for i in range(n):
        b = int(rng.integers(len(bases)))
        pil = Image.fromarray(bases[b])
        if rng.random() < 0.5:
            pil = pil.transpose(Image.FLIP_LEFT_RIGHT)
        pil = pil.rotate(float(rng.uniform(-20, 20)), resample=Image.BILINEAR,
                         fillcolor=(255, 255, 255))
        imgs[i] = np.asarray(pil, dtype=np.float32) / 255.0
        labels[i] = base_labels[b]
    return imgs.transpose(0, 3, 1, 2), labels


def save_sprites_cache(images, labels, data_dir=_DEFAULT_DATA_DIR):
    import os
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "sprites_32.npz")
    u8 = np.clip(images * 255.0, 0, 255).astype(np.uint8).transpose(0, 2, 3, 1)
    np.savez_compressed(path, images=u8, labels=labels.astype(np.int16))
    return path


def load_sprites_local(n=2000, data_dir=_DEFAULT_DATA_DIR):
    """Twemoji creature sprites as (images, labels) from data/sprites_32.npz
    (built by `tools/prepare_data.py sprites`). Falls back to synthetic blobs."""
    import os
    path = os.path.join(data_dir, "sprites_32.npz")
    if os.path.exists(path):
        d = np.load(path)
        imgs = d["images"][:n].astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        labels = d["labels"][:n].astype(np.int64)
        print(f"[data] loaded {len(imgs)} Twemoji sprites from {path} "
              f"({len(np.unique(labels))} base emoji)")
        return imgs, labels
    print(f"[data] {path} not found — run `python tools/prepare_data.py sprites`. "
          f"Using synthetic blobs.")
    imgs = synthetic_images(n=n, shape=(3, 32, 32))
    return imgs, np.array([i % 3 for i in range(len(imgs))], dtype=np.int64)


def load_images(dataset="afhq", n=2000, image_size=32, data_dir=_DEFAULT_DATA_DIR):
    """Load an image dataset as a float tensor in [0, 1], shape (n, C, H, W).

    `dataset` in {"afhq", "quickdraw", "sprites"}. Returns images only (use
    `load_afhq` / `load_quickdraw_local` directly if you need the labels). Tries
    the real source for each and falls back to `synthetic_images`. Kept
    deliberately forgiving — the point of these notebooks is the model, not the
    download."""
    c = 1 if dataset == "quickdraw" else 3
    try:
        if dataset == "afhq":
            return load_afhq(n=n, image_size=image_size, data_dir=data_dir)[0]
        if dataset == "quickdraw":
            return load_quickdraw_local(n=n, data_dir=data_dir)[0]
        if dataset == "sprites":
            return load_sprites_local(n=n, data_dir=data_dir)[0]
        raise FileNotFoundError(f"unknown dataset {dataset!r}")
    except Exception as e:  # noqa: BLE001
        print(f"[data] '{dataset}' unavailable ({type(e).__name__}); using synthetic images")
        return synthetic_images(n=n, shape=(c, image_size, image_size))
