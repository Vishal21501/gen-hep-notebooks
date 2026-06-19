"""Provisioning smoke test — run ONCE on the GPU box before class.

    python tools/smoke_test.py

Imports every package the notebooks touch (not just a static scan — it also
exercises the two things that fail silently: GPU availability and the
torch_geometric kNN op that needs the compiled `torch-cluster`). Prints a
categorised PASS/FAIL so you know exactly what, if anything, is missing.

Exit code is non-zero if any REQUIRED package is missing.
"""

from __future__ import annotations

import importlib
import sys

# (import name, pip name, note)
REQUIRED = [
    ("numpy", "numpy<2", "arrays (pin <2 for the HEP stack)"),
    ("torch", "torch", "core DL — install the CUDA wheel"),
    ("matplotlib", "matplotlib", "all plots"),
    ("PIL", "pillow", "AFHQ / sprite image decode"),
    ("scipy", "scipy", "Wasserstein / stats"),
    ("sklearn", "scikit-learn", "PCA, baseline classifiers"),
    ("h5py", "h5py", "CaloChallenge HDF5"),
    ("imageio", "imageio", "gifs (denoising, latent walks)"),
    ("diffusers", "diffusers", "UNet + schedulers (03, 04, capstone)"),
    ("jetnet", "jetnet", "JetNet data + FPD/KPD/W1 metrics"),
    ("datasets", "datasets", "real AFHQ download (01)"),
    ("cadence", "cadence-edu", "submission / autoregister / scaffold"),
]

OPTIONAL = [
    ("mplhep", "mplhep", "HEP plot style (jet-mass plot falls back if absent)"),
    ("tqdm", "tqdm", "training progress bars"),
    ("ipywidgets", "ipywidgets", "interactive sliders + cadence upload widget"),
]

# Notebook 05's classifier two-sample test + the capstone use a real PyG GNN,
# with a DeepSets fallback if these are missing.
GNN = [
    ("torch_geometric", "torch-geometric", "EdgeConv classifier (05)"),
    ("torch_cluster", "torch-cluster", "kNN for DynamicEdgeConv — the usual gotcha"),
    ("torch_scatter", "torch-scatter", "PyG scatter ops"),
    ("torch_sparse", "torch-sparse", "PyG sparse ops"),
]

RUNTIME = [
    ("ipykernel", "ipykernel", "the Jupyter kernel students run in"),
    ("IPython", "ipython", "shipped with Jupyter"),
    ("nbformat", "nbformat", "build tooling (tools/*.py) — not needed at student runtime"),
]


def _check(group, label, hard):
    print(f"\n{label}")
    missing = []
    for mod, pip_name, note in group:
        try:
            importlib.import_module(mod)
            print(f"  ok    {mod:18} ({note})")
        except Exception as e:  # noqa: BLE001
            print(f"  MISS  {mod:18} -> pip install {pip_name:16} [{type(e).__name__}]  ({note})")
            missing.append(pip_name)
    return missing


def main():
    print("=" * 72)
    print("Generative-modelling track — environment smoke test")
    print("=" * 72)

    hard_missing = _check(REQUIRED, "REQUIRED (class will break without these):", True)
    _check(OPTIONAL, "OPTIONAL (features degrade quietly if missing):", False)
    gnn_missing = _check(GNN, "GNN extras (05 real classifier; else DeepSets fallback):", False)
    _check(RUNTIME, "RUNTIME / tooling:", False)

    # --- behavioural checks the import list can't catch ---
    print("\nBEHAVIOURAL CHECKS")
    try:
        import torch
        cuda = torch.cuda.is_available()
        print(f"  {'ok  ' if cuda else 'WARN'}  torch.cuda.is_available() = {cuda}"
              f"{'' if cuda else '  <- no GPU visible; live training will be slow'}")
    except Exception as e:  # noqa: BLE001
        print(f"  MISS  torch import failed: {e}")

    # The torch-cluster gotcha: PyG can import fine yet DynamicEdgeConv (kNN) blow
    # up at call time. Exercise it on a tiny tensor.
    try:
        import torch
        from torch_geometric.nn import DynamicEdgeConv
        import torch.nn as nn
        conv = DynamicEdgeConv(nn.Linear(6, 8), k=3)
        x = torch.randn(10, 3)
        batch = torch.zeros(10, dtype=torch.long)
        conv(x, batch)
        print("  ok    DynamicEdgeConv kNN forward works (torch-cluster present)")
    except Exception as e:  # noqa: BLE001
        print(f"  WARN  DynamicEdgeConv kNN unavailable ({type(e).__name__}); "
              f"05 will use the DeepSets fallback")

    try:
        from cadence import load_ipython_extension  # noqa: F401
        print("  ok    cadence extension hook importable (%load_ext cadence will work)")
    except Exception as e:  # noqa: BLE001
        print(f"  MISS  cadence extension hook: {e}")

    print("\n" + "=" * 72)
    if hard_missing:
        print(f"FAIL — install the {len(hard_missing)} REQUIRED package(s): "
              f"{' '.join(hard_missing)}")
        sys.exit(1)
    print("PASS — all required packages present.")
    if gnn_missing:
        print(f"(note: GNN extras missing: {' '.join(gnn_missing)} — 05 falls back to DeepSets)")


if __name__ == "__main__":
    main()
