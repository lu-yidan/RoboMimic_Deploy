"""Yaw-aligned ACP terrain samples from static MuJoCo geometry."""
import time
import numpy as np
from policy.acp.observations import grid_xy, yaw


def flat_grid(pelvis_height, ground_height=0.):
    return np.column_stack((grid_xy(), np.full(187, pelvis_height-ground_height-.5))).astype(np.float32)


class MujocoTerrain:
    def __init__(self, model):
        self.model = model
        self.xy = grid_xy()

    def capture(self, data, state):
        import mujoco
        state.acp_pose_valid = False
        state.pelvis_pos_w = data.qpos[:3].astype(np.float32).copy()
        state.pelvis_quat_w = data.qpos[3:7].astype(np.float32).copy()
        state.acp_ang_vel = data.qvel[3:6].astype(np.float32).copy()
        psi = yaw(state.pelvis_quat_w)
        c, s = np.cos(psi), np.sin(psi)
        xy_world = self.xy @ np.array([[c,s],[-s,c]]) + state.pelvis_pos_w[:2]
        heights = np.full(187, np.nan)
        direction = np.array([0.,0.,-1.])
        geom_id = np.zeros(1, dtype=np.int32)
        for index, xy in enumerate(xy_world):
            origin = np.r_[xy, state.pelvis_pos_w[2]+3.]
            # mj_ray includes dynamic robot geometry. Advance past those hits;
            # only geometry welded to the world supplies terrain truth.
            for _ in range(64):
                distance = mujoco.mj_ray(self.model, data, origin, direction, None, True, -1, geom_id)
                if distance < 0:
                    break
                hit_z = origin[2]-distance
                body = self.model.geom_bodyid[geom_id[0]]
                if self.model.body_weldid[body] == 0:
                    heights[index] = hit_z
                    break
                origin[2] = hit_z-1e-5
        state.acp_terrain = np.column_stack((self.xy, state.pelvis_pos_w[2]-heights-.5)).astype(np.float32)
        stamp = time.monotonic()
        state.acp_pose_time = state.acp_terrain_time = stamp
        state.acp_pose_valid = bool(np.isfinite(heights).all())
