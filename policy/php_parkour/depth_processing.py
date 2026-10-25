"""Training depth preprocessing shared by simulation and hardware (no MuJoCo)."""
import numpy as np
import torch


def preprocess_depth(depth):
    depth = np.asarray(depth, dtype=np.float32)
    if depth.shape != (60, 106) or not np.isfinite(depth).all():
        raise ValueError('PHP camera requires a finite 60x106 metric depth image')
    # Match Holosoma DepthShmPlugin: crop, clamp, antialiased bicubic, far/empty handling.
    tensor = torch.from_numpy(np.ascontiguousarray(np.clip(depth[2:, 4:-4], .3, 3.)))[None, None]
    resized = torch.nn.functional.interpolate(tensor, size=(58, 87), mode='bicubic',
                                               align_corners=False, antialias=True)[0, 0].numpy()
    resized = np.minimum(resized, 3.)
    resized = np.where(resized < .15, 3., resized)
    return ((resized - .3) / 2.7 - .5).astype(np.float32)
