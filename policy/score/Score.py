import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))

import os
import numpy as np
import yaml
import onnxruntime
from collections import deque

from FSM.FSMState import FSMState
from common.ctrlcomp import StateAndCmd, PolicyOutput
from common.utils import FSMStateName, FSMCommand, progress_bar

# ---------------------------------------------------------------------------
# Joint index mapping: Isaac Lab BFS order <-> MuJoCo DFS order
# (identical to BeyondMimic)
#
# MuJoCo:    l_hip_p(0) l_hip_r(1) l_hip_y(2) l_knee(3) l_ank_p(4) l_ank_r(5)
#            r_hip_p(6) r_hip_r(7) r_hip_y(8) r_knee(9) r_ank_p(10) r_ank_r(11)
#            waist_y(12) waist_r(13) waist_p(14)
#            l_sho_p(15) l_sho_r(16) l_sho_y(17) l_elbow(18)
#            l_wrist_r(19) l_wrist_p(20) l_wrist_y(21)
#            r_sho_p(22) r_sho_r(23) r_sho_y(24) r_elbow(25)
#            r_wrist_r(26) r_wrist_p(27) r_wrist_y(28)
#
# Isaac Lab: l_hip_p(0) r_hip_p(1) waist_y(2) l_hip_r(3) r_hip_r(4) waist_r(5)
#            l_hip_y(6) r_hip_y(7) waist_p(8) l_knee(9) r_knee(10)
#            l_sho_p(11) r_sho_p(12) l_ank_p(13) r_ank_p(14)
#            l_sho_r(15) r_sho_r(16) l_ank_r(17) r_ank_r(18)
#            l_sho_y(19) r_sho_y(20) l_elbow(21) r_elbow(22)
#            l_wrist_r(23) r_wrist_r(24) l_wrist_p(25) r_wrist_p(26)
#            l_wrist_y(27) r_wrist_y(28)
# ---------------------------------------------------------------------------
MUJOCO_TO_ISAAC = np.array([
     0,  3,  6,  9, 13, 17,   # left leg
     1,  4,  7, 10, 14, 18,   # right leg
     2,  5,  8,               # waist
    11, 15, 19, 21, 23, 25, 27,  # left arm
    12, 16, 20, 22, 24, 26, 28,  # right arm
], dtype=np.int64)

ISAAC_TO_MUJOCO = np.argsort(MUJOCO_TO_ISAAC).astype(np.int64)

# torso_link body index in Isaac Lab BFS NPZ (29 joints, base=0, torso_link=9)
NPZ_ANCHOR_IDX = 9

# Number of history frames in the observation
HISTORY_LEN = 5


# ---------------------------------------------------------------------------
# Math helpers  (quaternion convention: [w, x, y, z])
# ---------------------------------------------------------------------------

def _yaw_quat(q):
    """Return the yaw-only quaternion extracted from q=[w,x,y,z]."""
    w, x, y, z = q
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))
    return np.array([np.cos(yaw * 0.5), 0.0, 0.0, np.sin(yaw * 0.5)],
                    dtype=np.float64)


def _quat_to_matrix(q):
    """[w,x,y,z] -> 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),  2*(x*z + w*y)],
        [2*(x*y + w*z),  1 - 2*(x*x + z*z),  2*(y*z - w*x)],
        [2*(x*z - w*y),      2*(y*z + w*x),  1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def _matrix_to_quat(R):
    """3x3 rotation matrix -> [w,x,y,z]."""
    R = np.asarray(R, dtype=np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def _quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64)


def _rot6d_from_quat(q):
    """First two columns of rotation matrix, flattened row-major -> (6,)."""
    R = _quat_to_matrix(q)
    return R[:, :2].flatten().astype(np.float32)


def _quat_apply_inverse(q, v):
    """Rotate vector v into the body frame defined by quaternion q=[w,x,y,z].
    Equivalent to Isaac Lab's quat_apply_inverse."""
    return _quat_to_matrix(q).T @ v


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class Score(FSMState):
    def __init__(self, state_cmd: StateAndCmd, policy_output: PolicyOutput):
        super().__init__()
        self.state_cmd    = state_cmd
        self.policy_output = policy_output
        self.name     = FSMStateName.SKILL_SCORE
        self.name_str = "skill_score"

        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "score.yaml")
        with open(config_path, "r") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

        onnx_path   = os.path.join(current_dir, "model", cfg["onnx_path"])
        motion_path = os.path.join(current_dir, "model", cfg["motion_path"])
        self.control_dt  = float(cfg["control_dt"])

        # ---- Load motion from NPZ (Isaac Lab BFS order) ----
        motion = np.load(motion_path)
        joint_pos_full = motion["joint_pos"]   # [T, 29]  Isaac Lab order
        joint_vel_full = motion["joint_vel"]   # [T, 29]  Isaac Lab order
        body_quat_full = motion["body_quat_w"] # [T, N_bodies, 4]  [w,x,y,z]
        body_pos_full  = motion["body_pos_w"]  # [T, N_bodies, 3]
        T_full = joint_pos_full.shape[0]

        # Optional time window
        start_s = float(cfg.get("motion_start_s", 0.0))
        end_s   = cfg.get("motion_end_s", None)
        i0 = int(round(start_s / self.control_dt))
        i1 = int(round(float(end_s) / self.control_dt)) if end_s is not None else T_full
        i0 = max(0, min(i0, T_full))
        i1 = max(i0 + 1, min(i1, T_full))
        self.motion_joint_pos = joint_pos_full[i0:i1]
        self.motion_joint_vel = joint_vel_full[i0:i1]
        self.motion_body_quat = body_quat_full[i0:i1]
        self.motion_body_pos  = body_pos_full[i0:i1]
        self.motion_total_steps = self.motion_joint_pos.shape[0]
        print(f"Score motion window: {i0 * self.control_dt:.2f}s ~ "
              f"{i1 * self.control_dt:.2f}s  ({self.motion_total_steps} frames)")

        # ---- Config (Isaac Lab / MuJoCo order) ----
        self.kps             = np.array(cfg["kps"],               dtype=np.float32)
        self.kds             = np.array(cfg["kds"],               dtype=np.float32)
        self.tau_limit       = np.array(cfg["tau_limit"],         dtype=np.float32)
        self.default_q_mj    = np.array(cfg["default_joint_pos"], dtype=np.float32)  # MuJoCo order
        self.action_scale_mj = np.array(cfg["action_scale"],      dtype=np.float32)  # MuJoCo order
        self.clip_actions      = float(cfg.get("clip_actions", 3.0))
        self.WARMUP_STEPS     = int(cfg.get("warmup_steps", 10))
        self.freeze_motion_at_first_frame = bool(cfg.get("freeze_motion_at_first_frame", False))
        self.zero_anchor_pos        = bool(cfg.get("zero_anchor_pos",        False))
        self.ball_as_anchor_pos     = bool(cfg.get("ball_as_anchor_pos",     False))
        self.ball_facing_anchor_ori = bool(cfg.get("ball_facing_anchor_ori", False))
        self.target_pos_w     = np.array(cfg["target_pos"],        dtype=np.float32)  # world frame
        # True on real robot: ball_pos is already in pelvis body frame (from DDS sensor).
        # False in simulation: ball_pos is in world frame and needs coordinate transform.
        self.use_body_frame_ball = bool(cfg.get("use_body_frame_ball", False))
        # soccer_pos_b obs only: when invalid + ~zero sensor ball, substitute (e.g. avoid [0,0,0]).
        _lost_default = cfg.get("ball_obs_default_when_lost", None)
        self._ball_obs_default_when_lost = (
            np.array(_lost_default, dtype=np.float32) if _lost_default is not None else None
        )
        self._ball_obs_lost_norm_max = float(cfg.get("ball_obs_lost_norm_max", 1e-3))

        # Default joint pos in Isaac Lab order (for joint_pos_rel obs)
        self.default_q_il = self.default_q_mj[ISAAC_TO_MUJOCO]

        # ---- ONNX session ----
        self.ort_session = onnxruntime.InferenceSession(onnx_path)

        # ---- Running state ----
        self.time_step      = 0
        self.last_action_il = np.zeros(29, dtype=np.float32)
        self._init_to_world = np.eye(3, dtype=np.float64)
        self._entry_q       = self.default_q_mj.copy()
        self._t0_target_q   = self.default_q_mj.copy()

        # History buffers (oldest → newest, HISTORY_LEN frames each)
        self._ang_vel_buf   = deque([np.zeros(3,  dtype=np.float32)] * HISTORY_LEN, maxlen=HISTORY_LEN)
        self._jpos_buf      = deque([np.zeros(29, dtype=np.float32)] * HISTORY_LEN, maxlen=HISTORY_LEN)
        self._jvel_buf      = deque([np.zeros(29, dtype=np.float32)] * HISTORY_LEN, maxlen=HISTORY_LEN)
        self._action_buf    = deque([np.zeros(29, dtype=np.float32)] * HISTORY_LEN, maxlen=HISTORY_LEN)
        self._ball_pos_buf  = deque([np.zeros(3,  dtype=np.float32)] * HISTORY_LEN, maxlen=HISTORY_LEN)
        self._target_pos_buf = deque([np.zeros(3, dtype=np.float32)] * HISTORY_LEN, maxlen=HISTORY_LEN)

        # Warm-up ONNX
        _dummy_obs = np.zeros((1, 547), dtype=np.float32)
        for _ in range(5):
            self.ort_session.run(["actions"], {"obs": _dummy_obs})

        freeze_note = " [freeze_motion_at_first_frame=ON]" if self.freeze_motion_at_first_frame else ""
        print("Score policy initialized "
              f"(547-dim, {self.motion_total_steps} motion frames){freeze_note}.")

    # ------------------------------------------------------------------

    def enter(self):
        self.time_step      = 0
        self.last_action_il = np.zeros(29, dtype=np.float32)

        # Reset history buffers
        for buf, dim in [(self._ang_vel_buf, 3), (self._jpos_buf, 29),
                         (self._jvel_buf, 29), (self._action_buf, 29),
                         (self._ball_pos_buf, 3), (self._target_pos_buf, 3)]:
            buf.clear()
            buf.extend([np.zeros(dim, dtype=np.float32)] * HISTORY_LEN)

        # ---- Yaw alignment ----
        motion_t0_quat = self.motion_body_quat[0, NPZ_ANCHOR_IDX].astype(np.float64)
        robot_quat     = self.state_cmd.torso_quat_w.astype(np.float64)
        q_rel = _quat_mul(robot_quat, _quat_conj(motion_t0_quat))
        self._init_to_world = _quat_to_matrix(_yaw_quat(q_rel))

        # ---- Anchor reference origin (for relative displacement, avoids needing absolute torso_pos_w) ----
        ref_anchor_0 = self.motion_body_pos[0, NPZ_ANCHOR_IDX].astype(np.float64)
        self._ref_anchor_world_origin = self._init_to_world @ ref_anchor_0
        self._entry_torso_pos_w = self.state_cmd.torso_pos_w.copy().astype(np.float64)

        # ---- Target position in entry pelvis frame (real robot only) ----
        # On real robot we have no absolute world coords, so we fix the target
        # direction at entry time: target_pos_w expressed relative to the
        # pelvis at the moment Score is activated, then kept constant.
        if self.use_body_frame_ball:
            # target_pos is a body-frame offset at entry (+x = forward).
            # Record entry yaw so _build_obs() can track target direction as robot rotates.
            pelvis_quat_entry = self.state_cmd.pelvis_quat_w.astype(np.float64)
            self._entry_yaw_mat = _quat_to_matrix(_yaw_quat(pelvis_quat_entry))  # 3×3
            self.target_pos_b_entry = np.clip(
                self.target_pos_w.astype(np.float64),
                -8.0, 8.0,
            ).astype(np.float32)
        else:
            self.target_pos_b_entry = np.zeros(3, dtype=np.float32)

        # ---- Warm-up interpolation targets ----
        self._entry_q     = self.state_cmd.q.copy()
        # motion_joint_pos is in Isaac Lab order; convert to MuJoCo for warmup
        self._t0_target_q = self.motion_joint_pos[0][MUJOCO_TO_ISAAC].copy()

        max_delta = np.abs(self._t0_target_q - self._entry_q).max()
        print(f"Score enter: warmup {self.WARMUP_STEPS} steps, "
              f"max joint delta = {max_delta:.3f} rad")

    # ------------------------------------------------------------------

    def _build_obs(self) -> np.ndarray:
        """547-dim obs:
        command(58) | anchor_pos_b(3) | anchor_ori_b(6) |
        base_ang_vel(15) | joint_pos(145) | joint_vel(145) |
        actions(145) | soccer_pos_b(15) | target_pos_b(15)
        """
        if self.freeze_motion_at_first_frame:
            t = 0
        else:
            t = min(self.time_step - self.WARMUP_STEPS, self.motion_total_steps - 1)

        # anchor obs uses torso_link as reference body (matches training: anchor_body_name = "torso_link")
        torso_quat_w = self.state_cmd.torso_quat_w.astype(np.float64)
        R_torso_w    = _quat_to_matrix(torso_quat_w)
        torso_pos_w  = self.state_cmd.torso_pos_w.astype(np.float64)

        # ---- command: ref_jpos + ref_jvel (Isaac Lab order) ----
        ref_jpos = self.motion_joint_pos[t]   # (29,) Isaac Lab order
        ref_jvel = self.motion_joint_vel[t]   # (29,) Isaac Lab order

        # Pelvis-frame ball for policy: sensor, or ball_obs_default_when_lost when invalid + ~zero.
        ball_b_effective = None
        if self.use_body_frame_ball:
            ball_b_effective = np.clip(self.state_cmd.ball_pos_b, -8.0, 8.0).astype(np.float32)
            if (
                self._ball_obs_default_when_lost is not None
                and not self.state_cmd.ball_valid
                and float(np.linalg.norm(self.state_cmd.ball_pos_b)) <= self._ball_obs_lost_norm_max
            ):
                ball_b_effective = np.clip(
                    self._ball_obs_default_when_lost, -8.0, 8.0
                ).astype(np.float32)

        # ---- motion_anchor_pos_b (relative to torso, expressed in torso body frame) ----
        # Yaw-align the reference anchor world position, then express in torso body frame.
        init_world_quat      = _matrix_to_quat(self._init_to_world)
        ref_anchor_pos_w     = self.motion_body_pos[t, NPZ_ANCHOR_IDX].astype(np.float64)
        aligned_anchor_pos_w = self._init_to_world @ ref_anchor_pos_w
        if self.zero_anchor_pos:
            anchor_pos_b = np.zeros(3, dtype=np.float32)
        elif self.ball_as_anchor_pos:
            # Use ball position in pelvis body frame directly as anchor_pos_b.
            if self.use_body_frame_ball:
                anchor_pos_b_ref  = (R_torso_w.T @ (aligned_anchor_pos_w - torso_pos_w)).astype(np.float32)
                # Blend with ball when detection is valid OR coast (invalid but non-zero estimate).
                ball_b = ball_b_effective
                use_ball = self.state_cmd.ball_valid or float(np.linalg.norm(ball_b)) > 1e-3
                if use_ball:
                    anchor_pos_b_ball = np.clip(ball_b, -1.0, 1.0).astype(np.float32)
                    anchor_pos_b = 0.1 * anchor_pos_b_ref + 0.9 * anchor_pos_b_ball
                else:
                    anchor_pos_b = anchor_pos_b_ref
                anchor_pos_b[2] = aligned_anchor_pos_w[2] - torso_pos_w[2]
            else:
                _R_pelvis    = _quat_to_matrix(self.state_cmd.pelvis_quat_w.astype(np.float64))
                _ball_rel_w  = self.state_cmd.ball_pos_w.astype(np.float64) - self.state_cmd.pelvis_pos_w.astype(np.float64)
                anchor_pos_b_ref  = (R_torso_w.T @ (aligned_anchor_pos_w - torso_pos_w)).astype(np.float32)
                anchor_cmd_xy = (_R_pelvis.T @ _ball_rel_w)[:2]
                anchor_cmd_xy = anchor_cmd_xy / np.linalg.norm(anchor_cmd_xy)
                anchor_pos_b_ball = 0.05*anchor_cmd_xy
                anchor_pos_b = np.concatenate([anchor_pos_b_ball, [aligned_anchor_pos_w[2] - torso_pos_w[2]]])
        elif self.use_body_frame_ball:
            # Real robot: torso_pos_w is always zero (no odometry).
            # Use relative displacement from entry to avoid feeding raw absolute coords to the policy.
            # Equivalent to training formula when robot and reference start at the same position.
            anchor_disp_w = aligned_anchor_pos_w - self._ref_anchor_world_origin
            robot_disp_w  = torso_pos_w - self._entry_torso_pos_w
            anchor_pos_b  = (R_torso_w.T @ (anchor_disp_w - robot_disp_w)).astype(np.float32)
        else:
            # Simulation: torso_pos_w is accurate. Use absolute coords, matching training exactly.
            anchor_pos_b = (R_torso_w.T @ (aligned_anchor_pos_w - torso_pos_w)).astype(np.float32)

        # Cache for visualization (world-frame anchor position).
        self._debug_anchor_pos_w = (torso_pos_w + R_torso_w @ anchor_pos_b.astype(np.float64)).astype(np.float32)
        self._debug_torso_pos_w  = torso_pos_w.astype(np.float32)

        # ---- motion_anchor_ori_b (relative to torso orientation, in torso body frame) ----
        if self.ball_facing_anchor_ori:
            # World-frame ball position (sim: direct; real robot: transform pelvis-frame → world).
            if self.use_body_frame_ball:
                _R_pelvis = _quat_to_matrix(self.state_cmd.pelvis_quat_w.astype(np.float64))
                ball_pos_w_f64 = (self.state_cmd.pelvis_pos_w.astype(np.float64)
                                  + _R_pelvis @ ball_b_effective.astype(np.float64))
            else:
                ball_pos_w_f64 = self.state_cmd.ball_pos_w.astype(np.float64)

            # Direction from torso to ball; Z uses ref-anchor height (matches training reference).
            to_ball_w = ball_pos_w_f64 - torso_pos_w
            to_ball_w[2] = aligned_anchor_pos_w[2] - torso_pos_w[2]
            norm = np.linalg.norm(to_ball_w)
            if norm < 1e-6:
                to_ball_dir = np.array([1.0, 0.0, 0.0])
            else:
                to_ball_dir = to_ball_w / norm

            # Rodrigues half-angle: quaternion [w,x,y,z] rotating +X onto to_ball_dir.
            # Degenerate case (ball directly behind, d ≈ -1): rotate 180° around Z.
            x_axis = np.array([1.0, 0.0, 0.0])
            d = float(np.dot(x_axis, to_ball_dir))
            if d < -1.0 + 1e-6:
                ball_facing_quat_w = np.array([0.0, 0.0, 0.0, 1.0])  # 180° around Z
            else:
                c = np.cross(x_axis, to_ball_dir)
                q_unnorm = np.array([1.0 + d, c[0], c[1], c[2]])
                ball_facing_quat_w = q_unnorm / np.linalg.norm(q_unnorm)

            rel_quat = _quat_mul(_quat_conj(torso_quat_w), ball_facing_quat_w)
            rel_quat = rel_quat / np.linalg.norm(rel_quat)
            anchor_ori_6d = _rot6d_from_quat(rel_quat)   # (6,)
        else:
            ref_anchor_quat_w = self.motion_body_quat[t, NPZ_ANCHOR_IDX].astype(np.float64)
            aligned_quat      = _quat_mul(init_world_quat, ref_anchor_quat_w)
            rel_quat = _quat_mul(_quat_conj(torso_quat_w), aligned_quat)
            rel_quat = rel_quat / np.linalg.norm(rel_quat)
            anchor_ori_6d = _rot6d_from_quat(rel_quat)   # (6,)

        # ---- Current joint state (Isaac Lab order) ----
        qj_il  = self.state_cmd.q[ISAAC_TO_MUJOCO]
        dqj_il = self.state_cmd.dq[ISAAC_TO_MUJOCO]
        jpos_cur = (qj_il - self.default_q_il).astype(np.float32)   # (29,)
        jvel_cur = dqj_il.astype(np.float32)                        # (29,)

        # ---- Ball and target in pelvis body frame (training uses root/pelvis, not torso) ----
        if self.use_body_frame_ball:
            # Real robot: ball_pos_b for obs (same as ball_b_effective: DDS or lost default).
            # Target direction is corrected each frame for robot yaw rotation since entry:
            #   target_world ≈ R_entry_yaw @ target_pos_b_entry
            #   target_pos_b  = R_current_yaw.T @ target_world
            ball_pos_b = ball_b_effective

            pelvis_quat = self.state_cmd.pelvis_quat_w.astype(np.float64)
            current_yaw_mat = _quat_to_matrix(_yaw_quat(pelvis_quat))
            target_world = self._entry_yaw_mat @ self.target_pos_b_entry.astype(np.float64)
            target_pos_b = np.clip(current_yaw_mat.T @ target_world, -8.0, 8.0).astype(np.float32)
        else:
            # Simulation: transform from world frame using pelvis pos/quat.
            robot_pelvis_pos_w = self.state_cmd.pelvis_pos_w.astype(np.float64)
            R_pelvis    = _quat_to_matrix(self.state_cmd.pelvis_quat_w.astype(np.float64))
            ball_rel_w  = self.state_cmd.ball_pos_w.astype(np.float64) - robot_pelvis_pos_w
            target_rel_w = self.target_pos_w.astype(np.float64) - robot_pelvis_pos_w
            ball_pos_b   = np.clip(R_pelvis.T @ ball_rel_w,   -8.0, 8.0).astype(np.float32)
            target_pos_b = np.clip(R_pelvis.T @ target_rel_w, -8.0, 8.0).astype(np.float32)

        # ---- Update history buffers ----
        self._ang_vel_buf.append(self.state_cmd.root_ang_vel_b.copy())
        self._jpos_buf.append(jpos_cur)
        self._jvel_buf.append(jvel_cur)
        self._action_buf.append(self.last_action_il.copy())
        self._ball_pos_buf.append(ball_pos_b)
        self._target_pos_buf.append(target_pos_b)

        # ---- Assemble obs (oldest→newest within each group) ----
        obs = np.concatenate([
            ref_jpos,                                        # command: ref_jpos  (29)
            ref_jvel,                                        # command: ref_jvel  (29)
            anchor_pos_b,                                    # anchor_pos_b        (3)
            anchor_ori_6d,                                   # anchor_ori_b        (6)
            np.concatenate(list(self._ang_vel_buf)),         # base_ang_vel       (15)
            np.concatenate(list(self._jpos_buf)),            # joint_pos         (145)
            np.concatenate(list(self._jvel_buf)),            # joint_vel         (145)
            np.concatenate(list(self._action_buf)),          # actions           (145)
            np.concatenate(list(self._ball_pos_buf)),        # soccer_pos_b       (15)
            np.concatenate(list(self._target_pos_buf)),      # target_pos_b       (15)
        ], dtype=np.float32)
        return obs   # shape (547,)

    # ------------------------------------------------------------------

    def run(self):
        # ---- Warm-up: interpolate from entry pose to motion t=0 pose ----
        if self.time_step < self.WARMUP_STEPS:
            alpha    = (self.time_step + 1) / self.WARMUP_STEPS
            target_q = (1.0 - alpha) * self._entry_q + alpha * self._t0_target_q
            self.policy_output.actions = target_q
            self.policy_output.kps     = self.kps
            self.policy_output.kds     = self.kds
            self.time_step += 1
            return

        # ---- Policy phase ----
        policy_step = self.time_step - self.WARMUP_STEPS

        obs = self._build_obs()

        out = self.ort_session.run(
            ["actions"],
            {"obs": obs[None, :]},
        )
        actions_il = out[0].squeeze(0)   # (29,) Isaac Lab order
        actions_il = np.clip(actions_il, -self.clip_actions, self.clip_actions)
        self.last_action_il = actions_il.copy()

        # Isaac Lab order -> MuJoCo order, then scale + offset
        actions_mj = actions_il[MUJOCO_TO_ISAAC]
        target_q   = self.default_q_mj + self.action_scale_mj * actions_mj

        # Debug: print for first 3 policy steps
        if policy_step < 30:
            print(f"\n[Score policy_step={policy_step}]")
            print(f"  anchor_pos_b  : {obs[58:61]}")
            print(f"  anchor_ori_6d : {obs[61:67]}")
            print(f"  ball_pos_b    : {obs[529:532]}")   # newest frame of ball_hist   [517:532]
            print(f"  target_pos_b  : {obs[544:547]}")   # newest frame of target_hist [532:547]
            print(f"  actions_il    : min={actions_il.min():.3f}  max={actions_il.max():.3f}")

        self.policy_output.actions = target_q
        self.policy_output.kps     = self.kps
        self.policy_output.kds     = self.kds

        # ---- Visualization: anchor sphere + line from torso to anchor ----
        self.policy_output.viz_spheres = [
            {"pos": self._debug_anchor_pos_w.copy(), "radius": 0.06,
             "rgba": np.array([1.0, 0.5, 0.0, 0.9], dtype=np.float32)},
            {"from": self._debug_torso_pos_w.copy(),
             "to":   self._debug_anchor_pos_w.copy(), "radius": 0.008,
             "rgba": np.array([1.0, 0.5, 0.0, 0.5], dtype=np.float32)},
        ]

        self.time_step += 1
        capped = 0 if self.freeze_motion_at_first_frame else min(policy_step, self.motion_total_steps - 1)
        self.policy_output.ghost_qpos = self._compute_ghost_qpos(capped)
        print(progress_bar(capped * self.control_dt,
                           self.motion_total_steps * self.control_dt),
              end="", flush=True)

    # ------------------------------------------------------------------

    def _compute_ghost_qpos(self, t: int) -> np.ndarray:
        """Compute ghost robot qpos for reference motion visualization.

        Follows mjlab's MotionCommand._update_command() transformation:
          ghost_root_pos  = [robot_anchor.xy, ref_anchor.z]
                          + R_init @ (ref_root_pos - ref_anchor_pos)
          ghost_root_quat = init_world_quat ⊗ ref_root_quat
          ghost_joints    = ref_joint_pos  (MuJoCo order, converted from Isaac Lab)
        """
        ref_anchor_pos = self.motion_body_pos[t, NPZ_ANCHOR_IDX].astype(np.float64)
        ref_root_pos   = self.motion_body_pos[t, 0].astype(np.float64)
        ref_root_quat  = self.motion_body_quat[t, 0].astype(np.float64)

        aligned_anchor_pos = self._init_to_world @ ref_anchor_pos

        torso_pos = self.state_cmd.torso_pos_w.astype(np.float64)
        delta_pos = np.array([torso_pos[0], torso_pos[1], aligned_anchor_pos[2]])

        ghost_root_pos  = delta_pos + self._init_to_world @ (ref_root_pos - ref_anchor_pos)
        ghost_root_quat = _quat_mul(_matrix_to_quat(self._init_to_world), ref_root_quat)

        # NPZ joint_pos is Isaac Lab order; convert to MuJoCo order for qpos.
        ghost_joints_mj = self.motion_joint_pos[t][MUJOCO_TO_ISAAC]

        qpos = np.empty(7 + 29, dtype=np.float32)
        qpos[0:3] = ghost_root_pos
        qpos[3:7] = ghost_root_quat   # [w, x, y, z] — MuJoCo free-joint order
        qpos[7:]  = ghost_joints_mj
        return qpos

    # ------------------------------------------------------------------

    def exit(self):
        self.time_step      = 0
        self.last_action_il = np.zeros(29, dtype=np.float32)
        self.policy_output.ghost_qpos  = None
        self.policy_output.viz_spheres = None
        print()

    def checkChange(self):
        cmd = self.state_cmd.skill_cmd
        if cmd == FSMCommand.LOCO:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.LOCOMODE
        elif cmd == FSMCommand.PASSIVE:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.PASSIVE
        elif cmd == FSMCommand.POS_RESET:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.FIXEDPOSE
        else:
            return FSMStateName.SKILL_SCORE
