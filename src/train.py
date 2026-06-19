"""A generic training loop so notebooks don't each re-implement one.

`train(model, loader, step_fn, ...)` runs the loop, calls your `step_fn` to get
a scalar loss, steps the optimiser, and returns the per-epoch loss history.
`step_fn(model, batch) -> loss` is where each notebook's actual objective lives
(ELBO, WGAN-GP critic loss, DDPM noise-prediction MSE)."""

from __future__ import annotations

from typing import Callable, List

import torch


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def to_loader(array, batch_size=128, shuffle=True):
    """Wrap a numpy array or tensor in a DataLoader of single tensors."""
    from torch.utils.data import DataLoader, TensorDataset
    t = torch.as_tensor(array, dtype=torch.float32)
    return DataLoader(TensorDataset(t), batch_size=batch_size, shuffle=shuffle)


def train(model, loader, step_fn: Callable, *, epochs=10, lr=1e-3,
          device=None, optimizer=None, progress=True) -> List[float]:
    """Run `epochs` passes over `loader`. Returns the mean loss per epoch."""
    device = device or get_device()
    model.to(device).train()
    opt = optimizer or torch.optim.Adam(model.parameters(), lr=lr)
    history: List[float] = []
    epoch_iter = range(epochs)
    if progress:
        try:
            from tqdm.auto import tqdm
            epoch_iter = tqdm(epoch_iter, desc="train")
        except ImportError:
            pass
    for _ in epoch_iter:
        running, n = 0.0, 0
        for batch in loader:
            batch = [b.to(device) for b in batch] if isinstance(batch, (list, tuple)) else batch.to(device)
            loss = step_fn(model, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss) * (len(batch[0]) if isinstance(batch, (list, tuple)) else len(batch))
            n += (len(batch[0]) if isinstance(batch, (list, tuple)) else len(batch))
        history.append(running / max(n, 1))
    return history
