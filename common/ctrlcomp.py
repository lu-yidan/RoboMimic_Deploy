from common.path_config import PROJECT_ROOT

import numpy as np
from common.utils import FSMCommand


class StateAndCmd:
    def __init__(self, num_joints):
        # robot state
        self.num_joints = num_joints
        self.q = np.zeros(num_joints, dtype=np.float32)
        self.dq = np.zeros(num_joints, dtype=np.float32)
        self.ddq = np.zeros(num_joints, dtype=np.float32)
        self.tau_est = np.zeros(num_joints, dtype=np.float32)
        self.gravity_ori = np.array([0., 0., 1.])
        self.ang_vel = np.zeros(3)
        # body-frame velocities for BeyondMimic (filled in deploy_mujoco)
        self.root_lin_vel_b = np.zeros(3, dtype=np.float32)
        self.root_ang_vel_b = np.zeros(3, dtype=np.float32)
        self.torso_pos_w  = np.zeros(3, dtype=np.float32)
        self.torso_quat_w = np.array([1., 0., 0., 0.], dtype=np.float32)  # [w,x,y,z]
        # pelvis (floating base) state for Score controller (world frame, filled in deploy_mujoco)
        self.pelvis_pos_w  = np.zeros(3, dtype=np.float32)
        self.pelvis_quat_w = np.array([1., 0., 0., 0.], dtype=np.float32)  # [w,x,y,z]
        # ball state for Score controller
        # Simulation: ball_pos_w (world frame, filled by deploy_mujoco from MuJoCo)
        # Real robot: ball_pos_b (pelvis body frame, filled by deploy_real from DDS)
        self.ball_pos_w    = np.zeros(3, dtype=np.float32)
        self.ball_vel_w    = np.zeros(3, dtype=np.float32)
        self.ball_pos_b    = np.zeros(3, dtype=np.float32)  # real robot only
        self.ball_valid    = False                           # real robot only
        # note: target_pos_w lives in Score.__init__ (loaded from score.yaml), not here
        # joy cmd
        self.vel_cmd = np.zeros(3)
        self.skill_cmd = FSMCommand.INVALID
        # skill change cmd
        # self.skill_set = FSMCommand.SKILL_1

class PolicyOutput:
    def __init__(self, num_joints):
        # actions
        self.actions = np.zeros(num_joints, dtype=np.float32)
        self.kps = np.zeros(num_joints, dtype=np.float32)
        self.kds = np.zeros(num_joints, dtype=np.float32)
        # ghost visualization: reference motion pose in world frame, for deploy_mujoco.py
        # Shape: (7 + num_joints,) = [root_pos(3), root_quat_wxyz(4), joint_pos(num_joints)]
        # None when the active policy does not support ghost visualization.
        self.ghost_qpos = None  # np.ndarray (7+n_joints,) or None
        