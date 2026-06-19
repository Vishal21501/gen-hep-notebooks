"""Shared helpers for the generative-modelling-for-HEP notebooks.

Notebooks stay short by importing from here: the recurring jet-mass plot, the
seed-setter, the dataset loaders (with synthetic fallbacks so a notebook runs
before the real downloads land), the point-cloud GNN blocks, the diffusers
UNet/scheduler wrapper, and a generic train loop.

Nothing here is cadence-specific — these are plain torch/numpy utilities.
"""

from .seeds import set_seed
from .jetmass import jet_mass, plot_jet_mass, jet_mass_w1, plot_jets, plot_jet_overlay

__all__ = ["set_seed", "jet_mass", "plot_jet_mass", "jet_mass_w1",
           "plot_jets", "plot_jet_overlay"]
