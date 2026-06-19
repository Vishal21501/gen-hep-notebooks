"""One-time data preparation — run once on the provisioning box, NOT per student.

Builds the small cached artifacts the notebooks load instantly:

    python tools/prepare_data.py afhq            # -> data/afhq_32.npz  (~18 MB)
    python tools/prepare_data.py afhq --size 64 --n 6000

Ship the resulting `data/*.npz` with the repo (or drop it on the teaching
interface / a shared read-only dir). At class time `src.data.load_afhq` finds
the cache and skips the multi-GB HuggingFace download entirely.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import (afhq_from_hf, save_afhq_cache,
                      quickdraw_from_web, save_quickdraw_cache,
                      sprites_from_web, save_sprites_cache,
                      save_calochallenge_cache, CALO_DS2_VOXELS)

# CaloChallenge Dataset 2 (electrons), Zenodo record 6366271. We pull the
# training file only; the matching reference file is dataset_2_2.hdf5.
_CALO_DS2_URL = ("https://zenodo.org/records/6366271/files/"
                 "dataset_2_1.hdf5?download=1")


def prepare_afhq(n, size, data_dir):
    print(f"[prepare] downloading + resizing {n} AFHQ images to {size}px ...")
    images, labels = afhq_from_hf(n=n, image_size=size)
    path = save_afhq_cache(images, labels, image_size=size, data_dir=data_dir)
    mb = Path(path).stat().st_size / 1e6
    print(f"[prepare] wrote {path}  ({len(images)} images, {mb:.1f} MB) — "
          f"ship this file; students load it instantly.")


def prepare_quickdraw(n, data_dir):
    print(f"[prepare] fetching {n} QuickDraw doodles (range requests, tiny) ...")
    images, labels = quickdraw_from_web(n=n)
    path = save_quickdraw_cache(images, labels, data_dir=data_dir)
    mb = Path(path).stat().st_size / 1e6
    print(f"[prepare] wrote {path}  ({len(images)} doodles, {mb:.1f} MB).")


def prepare_sprites(n, data_dir):
    print(f"[prepare] fetching Twemoji creatures + augmenting to {n} sprites ...")
    images, labels = sprites_from_web(n=n)
    path = save_sprites_cache(images, labels, data_dir=data_dir)
    mb = Path(path).stat().st_size / 1e6
    print(f"[prepare] wrote {path}  ({len(images)} sprites from "
          f"{len(set(labels.tolist()))} emoji, {mb:.1f} MB).")


def _download(url, dest, chunk=1 << 20):
    """Stream `url` to `dest` with a simple progress line; resume-safe via .part."""
    import urllib.request
    dest = Path(dest)
    if dest.exists():
        print(f"[prepare] raw file already present: {dest} "
              f"({dest.stat().st_size / 1e9:.2f} GB) — skipping download")
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[prepare] downloading {url}\n           -> {dest}")
    with urllib.request.urlopen(url) as r:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with open(tmp, "wb") as f:
            while True:
                buf = r.read(chunk)
                if not buf:
                    break
                f.write(buf); done += len(buf)
                if total:
                    print(f"\r[prepare]   {done/1e9:5.2f} / {total/1e9:.2f} GB "
                          f"({100*done/total:4.1f}%)", end="", flush=True)
    print()
    tmp.rename(dest)
    return dest


def prepare_calochallenge(n, data_dir, keep_raw):
    """Download CaloChallenge Dataset 2 (electrons) and skim the first `n`
    showers into the small cache `load_calochallenge_ds2` reads."""
    import h5py
    import numpy as np
    data_dir = Path(data_dir); data_dir.mkdir(parents=True, exist_ok=True)
    raw = _download(_CALO_DS2_URL, data_dir / "dataset_2_1.hdf5")

    with h5py.File(raw, "r") as f:
        keys = list(f.keys())
        print(f"[prepare] raw keys: {keys}")
        sh = f["showers"]; ie = f["incident_energies"]
        n_total = sh.shape[0]
        n = min(n, n_total)
        print(f"[prepare] showers {sh.shape} {sh.dtype} | incident {ie.shape} "
              f"| skimming first {n} / {n_total}")
        if sh.shape[1] != CALO_DS2_VOXELS:
            raise SystemExit(f"unexpected voxel count {sh.shape[1]} (want {CALO_DS2_VOXELS})")
        showers = np.asarray(sh[:n], dtype=np.float32)
        incident = np.asarray(ie[:n], dtype=np.float32)
    print(f"[prepare] incident-energy MeV range: {incident.min():.0f} -> {incident.max():.0f} "
          f"| nonzero voxels/shower (mean): {(showers > 0).sum(1).mean():.0f}")

    path = save_calochallenge_cache(showers, incident, data_dir=str(data_dir))
    mb = Path(path).stat().st_size / 1e6
    print(f"[prepare] wrote {path}  ({n} showers, {mb:.0f} MB compressed)")
    if not keep_raw:
        Path(raw).unlink()
        print(f"[prepare] removed raw {raw} (pass --keep-raw to keep the 1.4 GB original)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=["afhq", "quickdraw", "sprites", "calochallenge"],
                   help="which dataset to prepare")
    p.add_argument("--n", type=int, default=None,
                   help="rows to keep (default: 6000 images; 50000 calo showers = half)")
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--keep-raw", action="store_true",
                   help="keep the raw 1.4 GB calochallenge download after skimming")
    args = p.parse_args()
    if args.dataset == "afhq":
        prepare_afhq(args.n or 6000, args.size, args.data_dir)
    elif args.dataset == "quickdraw":
        prepare_quickdraw(args.n or 6000, args.data_dir)
    elif args.dataset == "sprites":
        prepare_sprites(args.n or 2000, args.data_dir)
    elif args.dataset == "calochallenge":
        prepare_calochallenge(args.n or 50000, args.data_dir, args.keep_raw)
