"""G1 AMP get-up / recovery policy (trained in Isaac Lab, legged_lab g1_rec).

Autonomous fall-recovery: when the robot is on the ground, this policy drives
it back to standing.  No reference trajectory or discriminator is needed at
deploy time — only the actor MLP (obs 480 -> action 29).

This is the NO-KEYBODY variant: key_body_pos_b was removed from the policy
observation during training, so deployment is PROPRIOCEPTION-ONLY (IMU + joint
encoders + last action).  No forward kinematics / key-body plumbing is required.

Observation (term-major, each term keeps 5 history frames, OLDEST-first):
    [ base_ang_vel x5 (15) | root_local_rot_tan_norm x5 (30) |
      joint_pos x5 (145) | joint_vel x5 (145) | last_action x5 (145) ]  = 480

Key correctness points vs the existing AMP locomotion policy:
  * joint_pos is ABSOLUTE (not relative to default).
  * no observation scaling.
  * "root" == pelvis (Isaac Lab articulation root) -> use pelvis_quat_w / root_ang_vel_b.
"""

import os
from collections import deque

import numpy as np
import yaml
import onnxruntime as ort

from FSM.FSMState import FSMState, FSMStateName
from common.ctrlcomp import StateAndCmd, PolicyOutput, FSMCommand

# Reuse BeyondMimic's verified joint mapping + quaternion helpers (single source of truth).
from policy.beyondmimic.BeyondMimic import (
    MUJOCO_TO_ISAAC,
    ISAAC_TO_MUJOCO,
    _yaw_quat,
    _quat_to_matrix,
    _quat_conj,
    _quat_mul,
)


def _root_local_rot_tan_norm(quat_w: np.ndarray) -> np.ndarray:
    """Match legged_lab amp.mdp.observations.root_local_rot_tan_norm.

    Remove yaw from the root quaternion, build the rotation matrix, and return
    column 0 (tangent) and column 2 (normal), concatenated -> (6,).
    """
    q = quat_w.astype(np.float64)
    yaw = _yaw_quat(q)
    q_local = _quat_mul(_quat_conj(yaw), q)
    R = _quat_to_matrix(q_local)
    tan_vec = R[:, 0]   # (3,)
    norm_vec = R[:, 2]  # (3,)
    return np.concatenate([tan_vec, norm_vec]).astype(np.float32)


class AmpRecovery(FSMState):
    def __init__(self, state_cmd: StateAndCmd, policy_output: PolicyOutput):
        super().__init__()
        self.state_cmd = state_cmd
        self.policy_output = policy_output
        self.name = FSMStateName.SKILL_AMP_RECOVERY
        self.name_str = "AMP_RECOVERY"

        cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "config", "amp_recovery.yaml")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        self.kps = np.array(cfg["kps"], dtype=np.float32)
        self.kds = np.array(cfg["kds"], dtype=np.float32)
        self.default_q_mj = np.array(cfg["default_joint_pos"], dtype=np.float32)  # MuJoCo order
        self.action_scale = float(cfg["action_scale"])
        self.clip_actions = float(cfg["clip_actions"])
        self.control_dt = float(cfg["control_dt"])
        self.hist_len = int(cfg["history_len"])
        self.num_obs = int(cfg["num_obs"])

        # History ring buffers, one per obs term. deque keeps OLDEST at the left,
        # which matches Isaac Lab's CircularBuffer.buffer (oldest-first) flatten.
        self._h_ang_vel = deque(maxlen=self.hist_len)   # (3,)
        self._h_rot6d   = deque(maxlen=self.hist_len)   # (6,)
        self._h_jpos    = deque(maxlen=self.hist_len)   # (29,)
        self._h_jvel    = deque(maxlen=self.hist_len)   # (29,)
        self._h_action  = deque(maxlen=self.hist_len)   # (29,)

        self._last_action_il = np.zeros(29, dtype=np.float32)

        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "model", cfg["model_path"])
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(model_path, sess_opts)

        # Warmup
        dummy = np.zeros((1, self.num_obs), dtype=np.float32)
        for _ in range(5):
            self._session.run(["actions"], {"obs": dummy})

        print(f"[AMP_RECOVERY] Initialized. Model: {model_path} (obs {self.num_obs} -> act 29)")

    # ── obs construction ──────────────────────────────────────────────────────

    def _current_frame(self):
        """Compute the per-term observation vectors for the current step."""
        # base_ang_vel: root (pelvis) angular velocity in body frame.
        ang_vel = self.state_cmd.root_ang_vel_b.astype(np.float32)            # (3,)

        # root_local_rot_tan_norm from pelvis quaternion (yaw removed).
        rot6d = _root_local_rot_tan_norm(self.state_cmd.pelvis_quat_w)        # (6,)

        # joint state -> Isaac Lab order (absolute, NOT relative).
        jpos_il = self.state_cmd.q[ISAAC_TO_MUJOCO].astype(np.float32)        # (29,)
        jvel_il = self.state_cmd.dq[ISAAC_TO_MUJOCO].astype(np.float32)       # (29,)

        return ang_vel, rot6d, jpos_il, jvel_il, self._last_action_il.copy()

    def _push_frame(self):
        ang_vel, rot6d, jpos, jvel, action = self._current_frame()
        self._h_ang_vel.append(ang_vel)
        self._h_rot6d.append(rot6d)
        self._h_jpos.append(jpos)
        self._h_jvel.append(jvel)
        self._h_action.append(action)

    def _build_obs(self) -> np.ndarray:
        # Each term: concatenate its 5 frames OLDEST-first, then concatenate terms
        # in the policy-group definition order.
        def flat(dq):
            return np.concatenate(list(dq), axis=0)
        obs = np.concatenate([
            flat(self._h_ang_vel),    # 15
            flat(self._h_rot6d),      # 30
            flat(self._h_jpos),       # 145
            flat(self._h_jvel),       # 145
            flat(self._h_action),     # 145
        ], axis=0).astype(np.float32)
        return obs

    # ── FSMState interface ────────────────────────────────────────────────────

    def enter(self):
        self._last_action_il[:] = 0.0
        for d in (self._h_ang_vel, self._h_rot6d, self._h_jpos,
                  self._h_jvel, self._h_action):
            d.clear()
        # Prime history with the current state repeated hist_len times.
        for _ in range(self.hist_len):
            self._push_frame()

    def run(self):
        self._push_frame()
        obs = self._build_obs()[None, :]

        actions_il = self._session.run(["actions"], {"obs": obs})[0].squeeze(0).astype(np.float32)
        actions_il = np.clip(actions_il, -self.clip_actions, self.clip_actions)
        self._last_action_il[:] = actions_il   # raw action feeds next obs

        # Isaac Lab order -> MuJoCo order, then scale + default offset.
        actions_mj = actions_il[MUJOCO_TO_ISAAC]
        target_q = self.default_q_mj + self.action_scale * actions_mj

        self.policy_output.actions = target_q
        self.policy_output.kps = self.kps
        self.policy_output.kds = self.kds

    def exit(self):
        self._last_action_il[:] = 0.0
        for d in (self._h_ang_vel, self._h_rot6d, self._h_jpos,
                  self._h_jvel, self._h_action):
            d.clear()

    def checkChange(self) -> FSMStateName:
        cmd = self.state_cmd.skill_cmd
        if cmd == FSMCommand.POS_RESET:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.FIXEDPOSE
        if cmd == FSMCommand.PASSIVE:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.PASSIVE
        if cmd == FSMCommand.LOCO:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.LOCOMODE
        if cmd == FSMCommand.CMD_AMP:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.SKILL_AMP
        return FSMStateName.SKILL_AMP_RECOVERY
