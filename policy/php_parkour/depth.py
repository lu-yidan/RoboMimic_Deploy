"""Local MuJoCo D435i capture; no DDS, shared memory or Holosoma runtime needed."""
import time
import numpy as np
import mujoco
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


class DepthCamera:
    def __init__(self, model):
        self.model = model
        self.body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso_link')
        if self.body < 0:
            raise ValueError('PHP camera requires torso_link')
        # Match the external D435i preset's 0.1 m near plane. The default
        # 0.008 m plane renders surfaces inside the torso-mounted camera housing.
        model.vis.map.znear = .1 / float(model.stat.extent)
        self.renderer = mujoco.Renderer(model, height=60, width=106)
        self.renderer.enable_depth_rendering()
        self.camera = mujoco.MjvCamera()
        self.raw_depth = None
        self.local_rotation = np.empty(9)
        quat = np.array([.60290764, .37585401, -.362958, -.60290764])
        mujoco.mju_quat2Mat(self.local_rotation, quat / np.linalg.norm(quat))
        self.local_rotation = self.local_rotation.reshape(3, 3)

    def capture(self, data, state):
        self.renderer.update_scene(data, camera=self.camera)
        rotation = data.xmat[self.body].reshape(3, 3)
        camera_rotation = rotation @ self.local_rotation
        position = data.xpos[self.body] + rotation @ np.array([.01, .01, .44])
        for camera in self.renderer.scene.camera:
            camera.pos[:] = position
            camera.forward[:] = -camera_rotation[:, 2]
            camera.up[:] = camera_rotation[:, 1]
            camera.frustum_center = 0
            camera.frustum_bottom = -camera.frustum_near * np.tan(np.deg2rad(58.5953 / 2))
            camera.frustum_top = -camera.frustum_bottom
        self.raw_depth = self.renderer.render().copy()
        state.php_depth = preprocess_depth(self.raw_depth)
        state.php_depth_time = time.monotonic()

    def close(self):
        self.renderer.close()
