"""Local MuJoCo D435i capture; no DDS, shared memory or Holosoma runtime needed."""
import time
import numpy as np
import mujoco


from policy.php_parkour.depth_processing import preprocess_depth


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
