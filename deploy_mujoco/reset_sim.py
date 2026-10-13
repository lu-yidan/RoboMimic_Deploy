"""Explicit user-triggered simulation reset; no persistent external support."""
import mujoco
import numpy as np
from common.utils import FSMCommand


def reset_to_fixed_pose(model, data, fsm, state, output):
    from FSM.FSM import FSMMode
    fsm.cur_policy.exit()
    mujoco.mj_resetData(model, data)
    data.qpos[7:7+state.num_joints] = fsm.fixed_pose_1._default_out
    mujoco.mj_forward(model, data)
    # Place the lowest foot collision geometry just above the floor.
    heights = []
    for name in ('left_ankle_roll_link', 'right_ankle_roll_link'):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for geom in range(model.ngeom):
            if model.geom_bodyid[geom] == body and (model.geom_contype[geom] or model.geom_conaffinity[geom]):
                if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX:
                    extent = np.abs(data.geom_xmat[geom].reshape(3,3)[2]) @ model.geom_size[geom]
                elif model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_SPHERE:
                    extent = model.geom_size[geom,0]
                else:
                    continue
                heights.append(data.geom_xpos[geom,2] - extent)
    if heights:
        data.qpos[2] += .005 - min(heights)
    data.qvel[:] = 0
    data.ctrl[:] = 0
    data.xfrc_applied[:] = 0
    mujoco.mj_forward(model, data)
    state.q = data.qpos[7:7+state.num_joints].copy()
    state.dq = np.zeros(state.num_joints)
    state.pelvis_pos_w = data.qpos[:3].astype(np.float32)
    state.pelvis_quat_w = data.qpos[3:7].astype(np.float32)
    state.root_ang_vel_b = np.zeros(3, np.float32)
    state.skill_cmd = FSMCommand.INVALID
    output.ghost_qpos = None
    output.viz_spheres = None
    fsm.cur_policy = fsm.fixed_pose_1
    fsm.FSMmode = FSMMode.NORMAL
    fsm.sim_counter = 0
    fsm.cur_policy.enter()
    fsm.cur_policy.run()
