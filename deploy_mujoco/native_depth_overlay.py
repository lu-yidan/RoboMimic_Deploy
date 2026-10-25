"""Depth inset using only the official MuJoCo passive viewer's public API.

The native viewer owns the window, sidebars, camera and rendering thread.
Images arrive top-to-bottom; Handle.set_images handles the OpenGL flip.
"""
import time
import mujoco
import mujoco.viewer
import numpy as np


def depth_rgb(normalized):
    """Display-only color map; the model input is never changed."""
    values = np.asarray(normalized)
    if values.shape != (58, 87) or not np.isfinite(values).all():
        raise ValueError('Depth inset requires a finite 58x87 image')
    near = 1 - np.clip(values + .5, 0, 1)
    return np.stack((255*near, 60+160*near, 200*(1-near)), axis=-1).astype(np.uint8)


def require_native_overlay():
    required = ('viewport', 'set_images', 'clear_images', 'set_texts', 'clear_texts')
    if not all(hasattr(mujoco.viewer.Handle, name) for name in required):
        raise RuntimeError(
            'Depth preview requires the official MuJoCo viewer image API. '
            'Use the robomimic-mj310 environment (mujoco==3.10.0); '
            'see docs/native_depth_overlay.md. '
            'Set php_depth_preview=false to run an older viewer without the inset.')


class NativeDepthOverlay:
    def __init__(self, viewer):
        require_native_overlay()
        self.viewer = viewer

    def clear(self):
        self.viewer.clear_images()
        self.viewer.clear_texts()

    def set_depth(self, normalized, captured_at, state_name):
        if not self.viewer.is_running():
            return
        # Copy the native scene viewport, excluding either sidebar. The native
        # viewer recalculates it on resize and panel toggles.
        with self.viewer.lock():
            viewport = self.viewer.viewport
            if viewport is None:
                return
            left, bottom, width, height = (
                viewport.left, viewport.bottom, viewport.width, viewport.height)
        scale = min(3, (width - 32)//87, (height - 80)//58)
        if scale < 1:
            self.clear()
            return
        if normalized is None:
            rgb = np.zeros((58, 87, 3), dtype=np.uint8)
            label = 'PHP depth | no frame'
        else:
            rgb = depth_rgb(normalized)
            age = max(0, (time.monotonic()-captured_at)*1000)
            label = 'PHP depth | {:.0f} ms'.format(age)
        rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
        # Padding gives the image a dark border without a custom GL renderer.
        rgb = np.pad(rgb, ((8, 8), (8, 8), (0, 0)), constant_values=15)
        h, w = rgb.shape[:2]
        rect = mujoco.MjrRect(left+width-w-8, bottom+height-h-48, w, h)
        self.viewer.set_images((rect, rgb))
        self.viewer.set_texts((mujoco.mjtFont.mjFONT_NORMAL,
                               mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                               label + ' | ' + state_name, ''))
