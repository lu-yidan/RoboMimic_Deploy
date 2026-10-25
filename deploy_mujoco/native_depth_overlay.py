"""Depth image upload to the native MuJoCo 3.2.3 viewer backport.

Rendering stays on the native viewer thread; this module never owns a window,
changes the camera, or touches the simulation model/data.
"""
import time
import numpy as np


def depth_rgb(normalized):
    """Display-only color map; the model input is never changed."""
    values = np.asarray(normalized)
    if values.shape != (58, 87) or not np.isfinite(values).all():
        raise ValueError('Depth inset requires a finite 58x87 image')
    near = 1 - np.clip(values + .5, 0, 1)
    return np.stack((255*near, 60+160*near, 200*(1-near)), axis=-1).astype(np.uint8)


def require_native_overlay():
    from mujoco import _simulate
    if not hasattr(_simulate.Simulate, 'set_depth_overlay'):
        raise RuntimeError(
            'Native depth overlay is not installed. With MuJoCo 3.2.3, run: '
            'python tools/install_native_depth_overlay.py. '
            'Alternatively set php_depth_preview=false to use the native viewer without depth.')


class NativeDepthOverlay:
    def __init__(self, viewer):
        require_native_overlay()
        self.viewer = viewer

    def set_depth(self, normalized, captured_at, state_name):
        sim = self.viewer._sim()
        if sim is None:
            return
        if normalized is None:
            rgb = np.zeros((58, 87, 3), dtype=np.uint8)
            label = 'PHP depth | no frame'
        else:
            rgb = depth_rgb(normalized)
            age = max(0, (time.monotonic()-captured_at)*1000)
            label = 'PHP depth | {:.0f} ms'.format(age)
        sim.set_depth_overlay(rgb, label + ' | ' + state_name)
