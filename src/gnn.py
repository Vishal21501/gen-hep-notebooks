"""Point-cloud building blocks for jets.

Jets are unordered sets of particles, so the right inductive bias is
permutation symmetry. Two blocks here:

* `DeepSetsEncoder` — plain-torch, permutation-invariant (per-particle MLP ->
  masked mean/sum pool). Robust, no extra deps; the default for 01/02.
* `edge_conv_encoder` — an `torch_geometric` EdgeConv stack for when you want a
  real message-passing GNN (the capstone classifier in 05). Imported lazily so
  the module still loads without torch_geometric.

All blocks take (batch, n_particles, n_features) with pt_rel == 0 marking
padding, and respect that mask.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def particle_mask(x: torch.Tensor) -> torch.Tensor:
    """True for real particles (pt_rel, the last feature, > 0)."""
    return x[..., 2] > 0


def _mlp(sizes, act=nn.ReLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class DeepSetsEncoder(nn.Module):
    """Permutation-invariant jet encoder: phi(particle) -> masked pool -> rho.
    Maps a jet point cloud to a latent vector (used as the VAE encoder body and
    the GAN/critic feature extractor)."""

    def __init__(self, in_features=3, hidden=128, latent=32, pool="mean",
                 mask_padding=True):
        super().__init__()
        self.phi = _mlp([in_features, hidden, hidden])
        self.rho = _mlp([hidden, hidden, latent])
        self.pool = pool
        # Padding is detected via ptrel > 0, which only holds in RAW feature
        # space. On standardized jets (see src/jets.py) ptrel is z-scored, so
        # set mask_padding=False — every slot is treated as a real particle.
        self.mask_padding = mask_padding

    def forward(self, x):
        if self.mask_padding:
            mask = particle_mask(x).unsqueeze(-1)         # (B, P, 1)
        else:
            mask = torch.ones(*x.shape[:-1], 1, device=x.device, dtype=x.dtype)
        h = self.phi(x) * mask                            # zero out padding
        summed = h.sum(dim=1)
        if self.pool == "mean":
            counts = mask.sum(dim=1).clamp(min=1)
            pooled = summed / counts
        else:
            pooled = summed
        return self.rho(pooled)


class DeepSetsDecoder(nn.Module):
    """Latent vector -> fixed-size particle cloud. Tiny and deliberately simple:
    a per-slot MLP that broadcasts the latent and adds a learned slot
    embedding, so the decoder can place particles differently per slot."""

    def __init__(self, latent=32, n_particles=30, out_features=3, hidden=128):
        super().__init__()
        self.n_particles = n_particles
        self.slot = nn.Parameter(torch.randn(n_particles, 16) * 0.1)
        self.net = _mlp([latent + 16, hidden, hidden, out_features])

    def forward(self, z):
        B = z.shape[0]
        z_rep = z.unsqueeze(1).expand(B, self.n_particles, z.shape[-1])
        slot = self.slot.unsqueeze(0).expand(B, self.n_particles, -1)
        # Raw output in STANDARDIZED feature space — see src/jets.py. The
        # JetStandardizer owns the physical scaling (log-pt + z-score) and its
        # inverse_transform turns these back into (etarel, phirel, ptrel). Doing
        # the scaling here (the old softplus + sum-to-1) flattened pt and put the
        # jet-mass peak in the wrong place.
        return self.net(torch.cat([z_rep, slot], dim=-1))


def edge_conv_encoder(in_features=3, hidden=64, latent=32, k=8):
    """A torch_geometric EdgeConv encoder (real message passing over a k-NN
    graph). Returns an nn.Module operating on PyG `Batch` objects. Imported
    lazily so this file loads without torch_geometric."""
    from torch_geometric.nn import DynamicEdgeConv, global_mean_pool

    class EdgeConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.ec1 = DynamicEdgeConv(_mlp([2 * in_features, hidden, hidden]), k=k)
            self.ec2 = DynamicEdgeConv(_mlp([2 * hidden, hidden, hidden]), k=k)
            self.head = _mlp([hidden, hidden, latent])

        def forward(self, data):
            x, batch = data.x, data.batch
            x = self.ec1(x, batch)
            x = self.ec2(x, batch)
            return self.head(global_mean_pool(x, batch))

    return EdgeConvNet()
