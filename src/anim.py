"""Turn a sequence of frames into a GIF — for the fun, visual bits.

Reusable across the notebooks:
  * 01 — a latent walk (cat -> dog morph) animated frame by frame;
  * 02 — GAN samples evolving over training epochs;
  * 03 — the diffusion denoising trajectory (a creature emerging from noise).

`frames_to_gif(frames, path)` accepts a list whose items are any of:
  * a HxW or HxWxC numpy array (float in [0,1] or uint8),
  * a CxHxW / 1xHxW torch tensor (channels-first, as models emit),
  * a matplotlib Figure (rendered for you — handy for animating 2D plots).
Returns the path. `show_gif(path)` displays it inline in the notebook.
"""

from __future__ import annotations

import numpy as np


def fig_to_frame(fig):
    """Render a matplotlib Figure to an (H, W, 3) uint8 array. Use the Agg
    backend (`matplotlib.use('Agg')`) or just a normal inline figure."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    return buf[..., :3].copy()


def _to_uint8_image(frame, normalize=False):
    """Coerce one frame to a uint8 image (H, W) or (H, W, 3)."""
    if frame.__class__.__module__.startswith("matplotlib"):
        return fig_to_frame(frame)
    if hasattr(frame, "detach"):                       # torch tensor
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 3 and frame.shape[0] in (1, 3):   # CHW -> HWC
        frame = np.transpose(frame, (1, 2, 0))
    if frame.ndim == 3 and frame.shape[2] == 1:        # drop singleton channel
        frame = frame[:, :, 0]
    if frame.dtype == np.uint8 and not normalize:
        return frame
    f = frame.astype(np.float32)
    if normalize:                                      # min-max to [0,1]
        lo, hi = float(f.min()), float(f.max())
        f = (f - lo) / (hi - lo + 1e-8)
    elif f.max() > 1.5:                                # looks like 0..255 already
        f = f / 255.0
    return (np.clip(f, 0, 1) * 255).astype(np.uint8)


def frames_to_gif(frames, path, fps=10, scale=1, normalize=False, loop=0):
    """Write `frames` to an animated GIF at `path`.

    fps:        frames per second.
    scale:      integer nearest-neighbour upscaling (small 32x32 images make a
                tiny gif — `scale=6` gives a watchable one).
    normalize:  min-max each frame to [0,1] (use for arbitrary-range arrays).
    loop:       0 = loop forever.
    """
    import imageio
    imgs = []
    for fr in frames:
        u8 = _to_uint8_image(fr, normalize=normalize)
        if scale and scale != 1:
            u8 = np.repeat(np.repeat(u8, scale, axis=0), scale, axis=1)
        imgs.append(u8)
    try:
        imageio.mimsave(path, imgs, fps=fps, loop=loop)
    except TypeError:                                  # older/newer imageio API
        imageio.mimsave(path, imgs, duration=1.0 / fps)
    return path


def show_gif(path):
    """Display a saved GIF inline (returns an IPython Image)."""
    from IPython.display import Image
    return Image(filename=path)
