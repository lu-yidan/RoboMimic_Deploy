"""Yaw-aligned ACP terrain samples from static MuJoCo geometry."""
import copy
import time
import numpy as np
from policy.acp.observations import grid_xy, yaw


def flat_grid(pelvis_height, ground_height=0.):
    return np.column_stack((grid_xy(), np.full(187, pelvis_height-ground_height-.5))).astype(np.float32)


class MujocoTerrain:
    def __init__(self, model):
        self.model = model
        self.xy = grid_xy()
        # Filter robot geometry before ray intersection. Advancing a ray past
        # dynamic mesh hits can repeatedly hit the same surface or exceed a
        # fixed retry count. A private model changes only ray visibility; live
        # collision/render groups and all geometry indices remain untouched.
        self.ray_model = copy.copy(model)
        static = model.body_weldid[model.geom_bodyid] == 0
        self.ray_model.geom_group[:] = np.where(static, 0, 1)
        self.ray_groups = np.array([1,0,0,0,0,0], dtype=np.uint8)

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
            distance = mujoco.mj_ray(self.ray_model, data, origin, direction,
                                     self.ray_groups, True, -1, geom_id)
            if distance >= 0:
                heights[index] = origin[2]-distance
        state.acp_terrain = np.column_stack((self.xy, state.pelvis_pos_w[2]-heights-.5)).astype(np.float32)
        stamp = time.monotonic()
        state.acp_pose_time = state.acp_terrain_time = stamp
        state.acp_pose_valid = bool(np.isfinite(heights).all())
