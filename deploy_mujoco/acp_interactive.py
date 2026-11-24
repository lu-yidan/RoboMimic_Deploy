"""Live simulation controls and a one-time reset into the actual B/Loco policy.

No transport is created. A reset sets only the initial simulation state; there
is no root support, teleport or interpolation during a Loco -> ACP transition.
"""
from queue import SimpleQueue, Empty

import mujoco
import numpy as np

from common.utils import FSMCommand, get_gravity_orientation
from policy.acp.observations import yaw, rotation


def sync_robot_state(model, data, state):
    n = state.num_joints
    state.q = data.qpos[7:7+n].astype(np.float32)
    state.dq = data.qvel[6:6+n].astype(np.float32)
    state.pelvis_pos_w = data.qpos[:3].astype(np.float32)
    state.pelvis_quat_w = data.qpos[3:7].astype(np.float32)
    state.gravity_ori = get_gravity_orientation(state.pelvis_quat_w)
    state.ang_vel = data.qvel[3:6].astype(np.float32)
    state.acp_ang_vel = state.root_ang_vel_b = state.ang_vel.copy()
    state.root_lin_vel_b = (rotation(state.pelvis_quat_w).T @ data.qvel[:3]).astype(np.float32)
    torso = model.body('torso_link').id
    state.torso_pos_w = data.xpos[torso].astype(np.float32)
    state.torso_quat_w = data.xquat[torso].astype(np.float32)


def velocity_sticks(loco, velocity):
    """Invert LocoMode's asymmetric stick scaling (neutral stick is +0.15 m/s)."""
    limits = np.array([loco.range_velx, loco.range_vely, loco.range_velz])
    return np.clip(2*(np.asarray(velocity)-limits[:,0])/(limits[:,1]-limits[:,0])-1, -1, 1)


def reset_to_loco(model, data, fsm, state, output):
    from FSM.FSM import FSMMode
    loco = fsm.loco_policy
    if model.nu != 29 or model.nq != 36 or model.nv != 35:
        raise ValueError('Loco startup requires the floating-base 29-motor G1 scene')
    fsm.cur_policy.exit()
    mujoco.mj_resetData(model, data)
    data.qpos[:3], data.qpos[3:7] = [0,0,0], [1,0,0,0]
    data.qpos[7+np.asarray(loco.joint2motor_idx)] = loco.default_angles
    mujoco.mj_forward(model, data)
    heights = []
    for side in ('left', 'right'):
        body = model.body(side+'_ankle_roll_link').id
        for geom in np.flatnonzero((model.geom_bodyid == body) & ((model.geom_contype | model.geom_conaffinity) != 0)):
            kind = model.geom_type[geom]
            if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                extent = model.geom_size[geom,0]
            elif kind == mujoco.mjtGeom.mjGEOM_BOX:
                extent = np.abs(data.geom_xmat[geom].reshape(3,3)[2]) @ model.geom_size[geom]
            else:
                raise ValueError('Loco reset requires sphere/box foot collision geometry')
            heights.append(data.geom_xpos[geom,2]-extent)
    if not heights:
        raise ValueError('No foot collision geometry for Loco initialization')
    data.qpos[2] = .002-min(heights)
    mujoco.mj_forward(model, data)
    sync_robot_state(model, data, state)
    state.php_depth, state.php_depth_time = None, 0.
    state.acp_terrain, state.acp_pose_valid = None, False
    state.acp_terrain_time = state.acp_pose_time = 0.
    state.skill_cmd = FSMCommand.INVALID
    state.vel_cmd = velocity_sticks(loco, [0,0,0])
    output.ghost_qpos = output.viz_spheres = None
    loco.action.fill(0)
    fsm.cur_policy, fsm.FSMmode, fsm.sim_counter = loco, FSMMode.NORMAL, 0
    loco.enter()
    loco.run()
    output.executed_fsm_state = loco.name.value


class KeyboardControl:
    def __init__(self, keyboard_velocity=False):
        self.events = SimpleQueue()
        self.velocity = np.zeros(3)
        self.keyboard_velocity = keyboard_velocity
        self.paused = False

    def push(self, key):
        self.events.put(key)

    def poll(self, state, fsm):
        command = None
        while True:
            try:
                key = self.events.get_nowait()
            except Empty:
                break
            text = chr(key).upper() if 32 <= key < 127 else ''
            selected = {'B': FSMCommand.LOCO,
                        'Y': FSMCommand.CMD_ACP if fsm.acp_policy is not None else FSMCommand.CMD_SMP_RECOVERY}.get(text)
            if key == 257: selected = FSMCommand.POS_RESET  # Enter
            if key == 290: selected = FSMCommand.PASSIVE  # F1
            if selected is not None and command != FSMCommand.PASSIVE:
                command = selected
            if text in ('W','S','A','D','Q','E','Z'):
                changes = {'W':(0,.1), 'S':(0,-.1), 'A':(1,.1), 'D':(1,-.1), 'Q':(2,.2), 'E':(2,-.2)}
                if text == 'Z': self.velocity.fill(0)
                else:
                    index, delta = changes[text]
                    self.velocity[index] += delta
                limits = np.array([fsm.loco_policy.range_velx, fsm.loco_policy.range_vely, fsm.loco_policy.range_velz])
                self.velocity[:] = np.clip(self.velocity, limits[:,0], limits[:,1])
                self.keyboard_velocity = True
                print('[Loco keyboard] vx, vy, yaw rate =', self.velocity.round(3).tolist())
            if text in ('1','2','3') and fsm.acp_policy is not None:
                policy = fsm.acp_policy
                heading = yaw(state.pelvis_quat_w)
                direction = np.array([np.cos(heading), np.sin(heading)])
                offset = {'1':direction, '2':np.array([-direction[1],direction[0]]), '3':np.zeros(2)}[text]
                policy.target = (state.pelvis_pos_w[:2]+offset).astype(np.float32)
                policy.heading = heading + (np.pi/2 if text == '2' else 0.)
                print('[ACP goal] xy =', policy.target.round(3).tolist(), 'heading =', round(policy.heading,3))
            if key == 32: self.paused = not self.paused
        if command is not None and state.skill_cmd != FSMCommand.PASSIVE:
            state.skill_cmd = command

    @staticmethod
    def goal_markers(policy):
        pos = np.r_[policy.target, .04]
        end = pos + [.25*np.cos(policy.heading), .25*np.sin(policy.heading), 0.]
        return [dict(pos=pos, radius=.045, rgba=[.9,.1,.8,1.]),
                {'from':pos, 'to':end, 'radius':.012, 'rgba':[.9,.1,.8,1.], 'geom':'arrow'}]
