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

ISAAC_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

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


def _compile_action_clip_bounds(cfg, fallback_clip):
    """Build per-joint raw action clip bounds in Isaac Lab joint order."""
    fallback_clip = float(fallback_clip)
    lo = np.full(29, -fallback_clip, dtype=np.float32)
    hi = np.full(29, fallback_clip, dtype=np.float32)

    clip_cfg = cfg.get("action_clip", None)
    if clip_cfg is None:
        return lo, hi

    if not isinstance(clip_cfg, dict):
        raise ValueError("action_clip must be a mapping from joint names/patterns to [min, max]")

    matched = set()
    for pattern, bounds in clip_cfg.items():
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"action_clip entry for '{pattern}' must be [min, max]")

        lower, upper = float(bounds[0]), float(bounds[1])
        if lower > upper:
            raise ValueError(f"action_clip entry for '{pattern}' has min > max")

        matched_this_pattern = False
        for i, joint_name in enumerate(ISAAC_JOINT_NAMES):
            if pattern == joint_name or (
                pattern.startswith(".*_") and joint_name.endswith(pattern[3:])
            ):
                if i in matched:
                    raise ValueError(f"action_clip patterns overlap at joint '{joint_name}'")
                lo[i] = lower
                hi[i] = upper
                matched.add(i)
                matched_this_pattern = True

        if not matched_this_pattern:
            raise ValueError(f"action_clip pattern '{pattern}' did not match any Score joint")

    return lo, hi


def _resolve_score_config_path(current_dir, config_file):
    config_path = Path(config_file)
    if config_path.is_absolute():
        return config_path

    candidates = [
        Path(current_dir) / "config" / config_file,
        Path.cwd() / config_file,
        Path(current_dir) / config_file,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class Score(FSMState):
    TARGET_SOURCE_TO_CODE = {
        "none": 0.0,
        "fixed": 1.0,
        "fixed_fallback": 2.0,
        "apriltag": 3.0,
        "imu_hold": 4.0,
        "fixed_sim": 5.0,
    }

    def __init__(
        self,
        state_cmd: StateAndCmd,
        policy_output: PolicyOutput,
        config_file: str = "score.yaml",
    ):
        super().__init__()
        self.state_cmd    = state_cmd
        self.policy_output = policy_output
        self.name     = FSMStateName.SKILL_SCORE
        self.name_str = "skill_" + Path(config_file).stem

        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = _resolve_score_config_path(current_dir, config_file)
        with open(config_path, "r") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        print(f"Score config: {config_path}")

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
        self.default_q_mj    = np.array(cfg["default_joint_pos"], dtype=np.float32)  # MuJoCo order
        self.action_scale_mj = np.array(cfg["action_scale"],      dtype=np.float32)  # MuJoCo order
        self.clip_actions      = float(cfg.get("clip_actions", 3.0))
        self.action_clip_lo_il, self.action_clip_hi_il = _compile_action_clip_bounds(
            cfg,
            self.clip_actions,
        )
        self.WARMUP_STEPS     = int(cfg.get("warmup_steps", 10))
        self.warmup_target_q_mj = np.array(
            cfg.get("warmup_target_joint_pos", self.motion_joint_pos[0][MUJOCO_TO_ISAAC]),
            dtype=np.float32,
        )
        if self.warmup_target_q_mj.shape != (29,):
            raise ValueError("warmup_target_joint_pos must contain 29 joint values in MuJoCo order")
        self.freeze_motion_at_first_frame = bool(cfg.get("freeze_motion_at_first_frame", False))
        self.manual_trigger = bool(cfg.get("manual_trigger", False))
        self.manual_anchor_cmd = bool(cfg.get("manual_anchor_cmd", False))
        self.manual_anchor_pos_scale_xy = np.array(
            cfg.get("manual_anchor_pos_scale_xy", [1.0, 0.6]),
            dtype=np.float32,
        )
        if self.manual_anchor_pos_scale_xy.shape != (2,):
            raise ValueError("manual_anchor_pos_scale_xy must contain [x_scale, y_scale]")
        self.manual_anchor_yaw_scale = float(cfg.get("manual_anchor_yaw_scale", 1.57))
        self.manual_cmd_deadzone = float(cfg.get("manual_cmd_deadzone", 0.05))
        self.zero_anchor_pos        = bool(cfg.get("zero_anchor_pos",        False))
        self.ball_as_anchor_pos     = bool(cfg.get("ball_as_anchor_pos",     False))
        self.ball_anchor_cmd_scale  = float(cfg.get("ball_anchor_cmd_scale", 0.5))
        self.ball_facing_anchor_ori = bool(cfg.get("ball_facing_anchor_ori", False))
        self.target_pos_w     = np.array(cfg["target_pos"],        dtype=np.float32)  # world frame
        self.target_source    = str(cfg.get("target_source", "fixed")).strip().lower()
        if self.target_source not in {"fixed", "apriltag"}:
            raise ValueError(
                f"Unsupported target_source '{self.target_source}'. "
                "Choose one of: ['fixed', 'apriltag']"
            )
        self.target_hold_on_loss_with_imu = bool(cfg.get("target_hold_on_loss_with_imu", True))
        self.target_use_fixed_fallback = bool(cfg.get("target_use_fixed_fallback", True))
        # True on real robot: ball_pos is already in pelvis body frame (from DDS sensor).
        # False in simulation: ball_pos is in world frame and needs coordinate transform.
        self.use_body_frame_ball = bool(cfg.get("use_body_frame_ball", False))
        # soccer_pos_b obs only: when invalid + ~zero sensor ball, substitute (e.g. avoid [0,0,0]).
        _lost_default = cfg.get("ball_obs_default_when_lost", None)
        self._ball_obs_default_when_lost = (
            np.array(_lost_default, dtype=np.float32) if _lost_default is not None else None
        )
        self._ball_obs_lost_norm_max = float(cfg.get("ball_obs_lost_norm_max", 1e-3))
        self._ball_vel_kf_pos_std = float(cfg.get("ball_vel_kf_pos_std", 0.04))
        self._ball_vel_kf_accel_std = float(cfg.get("ball_vel_kf_accel_std", 5.0))
        self._ball_vel_kf_init_vel_std = float(cfg.get("ball_vel_kf_init_vel_std", 2.0))
        self._ball_vel_kf_outlier_dist = float(cfg.get("ball_vel_kf_outlier_dist", 0.8))
        self.runtime_mode = "real" if self.use_body_frame_ball else "sim"
        self.anchor_mode = (
            "zero"
            if self.zero_anchor_pos else
            "ball_cmd"
            if self.ball_as_anchor_pos else
            "motion_ref"
        )
        self.anchor_ori_mode = (
            "ball_facing"
            if self.ball_facing_anchor_ori else
            "motion_ref"
        )

        # ---- Ball-trigger gate: hold at frame 0 until ball enters the circle ----
        self.wait_for_ball    = bool(cfg.get("wait_for_ball",    False))
        self.trigger_radius   = float(cfg.get("trigger_radius",  0.5))
        self.trigger_horizon  = float(cfg.get("trigger_horizon", 0.5))
        self.trigger_base_offset_xy = np.array(
            cfg.get("trigger_base_offset_xy", [0.0, 0.0]),
            dtype=np.float64,
        )
        if self.trigger_base_offset_xy.shape != (2,):
            raise ValueError("trigger_base_offset_xy must be a 2D list, e.g. [-0.1, 0.1]")
        self.trigger_frame    = int(cfg.get("trigger_frame",     252))
        _tpe = cfg.get("trigger_play_end_frame", None)
        self.trigger_play_end_frame = int(_tpe) if _tpe is not None else None
        self.trigger_play_once = bool(cfg.get("trigger_play_once", False))
        self.trigger_gate_enabled = self.wait_for_ball or self.manual_trigger
        self.motion_mode = (
            "freeze"
            if self.freeze_motion_at_first_frame else
            "manual_triggered_once"
            if self.manual_trigger and self.trigger_play_once else
            "manual_triggered"
            if self.manual_trigger else
            "triggered_once"
            if self.wait_for_ball and self.trigger_play_once else
            "triggered"
            if self.wait_for_ball else
            "play"
        )
        self._adapt_play_motion_deprecated = bool(cfg.get("adapt_play_motion", False))

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
        self._motion_triggered    = True   # overwritten in enter()
        self._trigger_policy_step = 0
        # After a finite trigger burst, require ball to leave trigger zone before re-arming.
        self._burst_need_ball_clear = False
        self._prev_ball_pos_b_for_trigger = None
        self._ball_vel_b_est = np.zeros(3, dtype=np.float32)
        self._ball_vel_kf_initialized = False
        self._ball_vel_kf_x = np.zeros(4, dtype=np.float64)  # [x, y, vx, vy] in pelvis frame
        self._ball_vel_kf_P = np.eye(4, dtype=np.float64)
        self._entry_yaw_mat = np.eye(3, dtype=np.float64)
        self._target_world_yaw_vec = None
        self._debug_target_pos_w = np.zeros(3, dtype=np.float32)
        self._debug_target_corrected_pos_w = np.zeros(3, dtype=np.float32)
        self._debug_ball_pos_b = np.zeros(3, dtype=np.float32)
        self._debug_target_pos_b = np.zeros(3, dtype=np.float32)
        self._debug_anchor_x_axis_w = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self._target_debug_source = "fixed"

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

        print("Score policy initialized "
              f"(547-dim, {self.motion_total_steps} motion frames).")
        print(
            f"[Score config] runtime={self.runtime_mode} "
            f"anchor={self.anchor_mode} anchor_ori={self.anchor_ori_mode} "
            f"motion={self.motion_mode} target={self.target_source}"
        )
        if self.manual_anchor_cmd:
            print(
                "[Score config] manual anchor overrides only while joystick "
                "input is outside deadzone."
            )
        if self._adapt_play_motion_deprecated:
            print("[Score config] `adapt_play_motion` is deprecated and ignored.")

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
        self._target_world_yaw_vec = None
        self._debug_target_pos_w = np.zeros(3, dtype=np.float32)
        self._debug_target_corrected_pos_w = np.zeros(3, dtype=np.float32)
        self._debug_ball_pos_b = np.zeros(3, dtype=np.float32)
        self._debug_target_pos_b = np.zeros(3, dtype=np.float32)
        self._debug_anchor_x_axis_w = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self._target_debug_source = "fixed"

        # ---- Warm-up interpolation targets ----
        self._entry_q     = self.state_cmd.q.copy()
        self._t0_target_q = self.warmup_target_q_mj.copy()

        # ---- Trigger gate ----
        # Ball/manual trigger modes hold at frame 0 until the selected event arrives.
        self._motion_triggered    = not self.trigger_gate_enabled
        self._trigger_policy_step = 0
        self._burst_need_ball_clear = False
        self._prev_ball_pos_b_for_trigger = None
        self._reset_ball_vel_kf()
        self.state_cmd.score_manual_trigger = False

        max_delta = np.abs(self._t0_target_q - self._entry_q).max()
        if self.manual_trigger:
            if self.trigger_play_once:
                s0 = int(np.clip(self.trigger_frame, 0, self.motion_total_steps - 1))
                trigger_note = f", waiting for R3 manual trigger -> play once {s0}..end"
            elif self.trigger_play_end_frame is not None:
                s0 = int(np.clip(self.trigger_frame, 0, self.motion_total_steps - 1))
                s1 = int(np.clip(self.trigger_play_end_frame, 0, self.motion_total_steps - 1))
                s1 = max(s0, s1)
                trigger_note = f", waiting for R3 manual trigger -> play frames {s0}..{s1}, then wait"
            else:
                trigger_note = f", waiting for R3 manual trigger -> frame {self.trigger_frame}..end"
        elif self.wait_for_ball:
            if self.trigger_play_once:
                s0 = int(np.clip(self.trigger_frame, 0, self.motion_total_steps - 1))
                trigger_note = (f", waiting for ball (r={self.trigger_radius}m, "
                                f"h={self.trigger_horizon}s → play once {s0}..end)")
            elif self.trigger_play_end_frame is not None:
                s0 = int(np.clip(self.trigger_frame, 0, self.motion_total_steps - 1))
                s1 = int(np.clip(self.trigger_play_end_frame, 0, self.motion_total_steps - 1))
                s1 = max(s0, s1)
                trigger_note = (f", waiting for ball (r={self.trigger_radius}m, "
                                f"h={self.trigger_horizon}s → play frames {s0}..{s1}, then wait)")
            else:
                trigger_note = (f", waiting for ball (r={self.trigger_radius}m, "
                                f"h={self.trigger_horizon}s → frame {self.trigger_frame}..end)")
        else:
            trigger_note = ""
        print(f"Score enter: warmup {self.WARMUP_STEPS} steps, "
              f"max joint delta = {max_delta:.3f} rad{trigger_note}")

    # ------------------------------------------------------------------

    def _trigger_segment_bounds(self):
        """Inclusive segment [s0, s1] in loaded motion indices (clamped)."""
        s0 = int(np.clip(self.trigger_frame, 0, self.motion_total_steps - 1))
        s1 = int(np.clip(
            self.trigger_play_end_frame
            if self.trigger_play_end_frame is not None else self.motion_total_steps - 1,
            0, self.motion_total_steps - 1))
        if s1 < s0:
            s1 = s0
        return s0, s1

    def _motion_frame_index(self, policy_step: int) -> int:
        """Motion frame index for obs / ghost (policy_step excludes warmup)."""
        if self.freeze_motion_at_first_frame or not self._motion_triggered:
            return 0
        steps_since_trigger = policy_step - self._trigger_policy_step
        if self.trigger_gate_enabled and self.trigger_play_once:
            return min(
                int(np.clip(self.trigger_frame, 0, self.motion_total_steps - 1)) + steps_since_trigger,
                self.motion_total_steps - 1,
            )
        if self.trigger_gate_enabled and self.trigger_play_end_frame is not None:
            s0, s1 = self._trigger_segment_bounds()
            t_lin = s0 + steps_since_trigger
            return int(min(t_lin, s1))
        if self.trigger_gate_enabled:
            return min(self.trigger_frame + steps_since_trigger, self.motion_total_steps - 1)
        return min(max(policy_step, 0), self.motion_total_steps - 1)

    def _trigger_play_once_finished(self, policy_step: int) -> bool:
        if not (self.trigger_gate_enabled and self.trigger_play_once and self._motion_triggered):
            return False
        s0 = int(np.clip(self.trigger_frame, 0, self.motion_total_steps - 1))
        s1 = int(np.clip(
            self.trigger_play_end_frame
            if self.trigger_play_end_frame is not None else self.motion_total_steps - 1,
            0,
            self.motion_total_steps - 1,
        ))
        s1 = max(s0, s1)
        return (s0 + (policy_step - self._trigger_policy_step)) >= s1

    def _reset_ball_vel_kf(self):
        self._prev_ball_pos_b_for_trigger = None
        self._ball_vel_b_est[:] = 0.0
        self._ball_vel_kf_initialized = False
        self._ball_vel_kf_x[:] = 0.0
        self._ball_vel_kf_P = np.eye(4, dtype=np.float64)

    def _estimate_ball_vel_b(self) -> np.ndarray:
        """Estimate pelvis-frame ball velocity on real robot from `ball_pos_b`.

        Triggering only needs horizontal velocity, so use a 2D constant-velocity
        Kalman filter on [x, y, vx, vy] instead of raw position differencing.
        """
        if not self.use_body_frame_ball:
            return np.zeros(3, dtype=np.float32)

        if not self.state_cmd.ball_valid:
            self._reset_ball_vel_kf()
            return self._ball_vel_b_est.copy()

        z = self.state_cmd.ball_pos_b[:2].astype(np.float64)
        if not self._ball_vel_kf_initialized:
            self._ball_vel_kf_x[:] = [z[0], z[1], 0.0, 0.0]
            pos_var = max(self._ball_vel_kf_pos_std, 1e-4) ** 2
            vel_var = max(self._ball_vel_kf_init_vel_std, 1e-4) ** 2
            self._ball_vel_kf_P = np.diag([pos_var, pos_var, vel_var, vel_var]).astype(np.float64)
            self._ball_vel_kf_initialized = True
            self._ball_vel_b_est[:] = 0.0
            return self._ball_vel_b_est.copy()

        dt = max(float(self.control_dt), 1e-3)
        F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)
        accel_var = max(self._ball_vel_kf_accel_std, 1e-4) ** 2
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        Q = accel_var * np.array([
            [0.25 * dt4, 0.0,       0.5 * dt3, 0.0      ],
            [0.0,        0.25 * dt4, 0.0,       0.5 * dt3],
            [0.5 * dt3, 0.0,        dt2,       0.0      ],
            [0.0,        0.5 * dt3, 0.0,       dt2      ],
        ], dtype=np.float64)
        H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float64)
        R = (max(self._ball_vel_kf_pos_std, 1e-4) ** 2) * np.eye(2, dtype=np.float64)

        x_pred = F @ self._ball_vel_kf_x
        P_pred = F @ self._ball_vel_kf_P @ F.T + Q
        innovation = z - H @ x_pred
        if (
            self._ball_vel_kf_outlier_dist > 0.0
            and float(np.linalg.norm(innovation)) > self._ball_vel_kf_outlier_dist
        ):
            self._ball_vel_kf_initialized = False
            self._ball_vel_b_est[:] = 0.0
            return self._ball_vel_b_est.copy()

        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        self._ball_vel_kf_x = x_pred + K @ innovation
        I = np.eye(4, dtype=np.float64)
        self._ball_vel_kf_P = (I - K @ H) @ P_pred
        self._ball_vel_kf_P = 0.5 * (self._ball_vel_kf_P + self._ball_vel_kf_P.T)

        self._ball_vel_b_est[:] = [self._ball_vel_kf_x[2], self._ball_vel_kf_x[3], 0.0]
        return self._ball_vel_b_est.copy()

    def _get_effective_ball_pos_b(self):
        """Return pelvis-frame ball position used by the policy.

        In real mode this optionally substitutes a default when the sensor is invalid
        and reports a near-zero vector. In simulation this returns None because the
        ball observation is reconstructed from world-frame state elsewhere.
        """
        if not self.use_body_frame_ball:
            return None

        ball_b_effective = np.clip(self.state_cmd.ball_pos_b, -8.0, 8.0).astype(np.float32)
        if (
            self._ball_obs_default_when_lost is not None
            and not self.state_cmd.ball_valid
            and float(np.linalg.norm(self.state_cmd.ball_pos_b)) <= self._ball_obs_lost_norm_max
        ):
            ball_b_effective = np.clip(
                self._ball_obs_default_when_lost, -8.0, 8.0
            ).astype(np.float32)
        return ball_b_effective

    def _pelvis_to_torso_offset_pelvis(self) -> np.ndarray:
        """Return torso origin relative to pelvis, expressed in pelvis frame."""
        waist_yaw = float(self.state_cmd.q[12])
        R_waist_yaw = _quat_to_matrix(
            np.array(
                [np.cos(waist_yaw * 0.5), 0.0, 0.0, np.sin(waist_yaw * 0.5)],
                dtype=np.float64,
            )
        )
        return R_waist_yaw @ np.array([-0.0039635, 0.0, 0.044], dtype=np.float64)

    def _get_ball_pos_torso_b(
        self,
        ball_b_effective,
        torso_pos_w: np.ndarray,
        R_torso_w: np.ndarray,
    ):
        """Return ball position relative to torso, expressed in torso frame."""
        if self.runtime_mode == "real":
            if ball_b_effective is None:
                return None
            R_pelvis_w = _quat_to_matrix(self.state_cmd.pelvis_quat_w.astype(np.float64))
            ball_rel_torso_pelvis = (
                ball_b_effective.astype(np.float64) - self._pelvis_to_torso_offset_pelvis()
            )
            return (R_torso_w.T @ (R_pelvis_w @ ball_rel_torso_pelvis)).astype(np.float32)

        ball_rel_torso_w = (
            self.state_cmd.ball_pos_w.astype(np.float64) - torso_pos_w
        )
        return (R_torso_w.T @ ball_rel_torso_w).astype(np.float32)

    def _apply_deadzone(self, value: float) -> float:
        value = float(value)
        return 0.0 if abs(value) < self.manual_cmd_deadzone else value

    def _update_manual_anchor_command(self):
        raw_xy = np.asarray(self.state_cmd.score_anchor_pos_raw_b[:2], dtype=np.float32)
        raw_xy = np.array([self._apply_deadzone(raw_xy[0]), self._apply_deadzone(raw_xy[1])], dtype=np.float32)
        anchor_cmd = np.zeros(3, dtype=np.float32)
        anchor_cmd[:2] = raw_xy * self.manual_anchor_pos_scale_xy
        yaw_cmd = self._apply_deadzone(self.state_cmd.score_anchor_yaw_raw) * self.manual_anchor_yaw_scale
        self.state_cmd.score_anchor_pos_cmd_b = anchor_cmd
        self.state_cmd.score_anchor_yaw_cmd = float(yaw_cmd)
        self.state_cmd.score_anchor_pos_cmd_active = bool(np.any(np.abs(raw_xy) > 0.0))
        self.state_cmd.score_anchor_yaw_cmd_active = bool(abs(yaw_cmd) > 0.0)
        self.state_cmd.score_anchor_cmd_active = (
            self.state_cmd.score_anchor_pos_cmd_active
            or self.state_cmd.score_anchor_yaw_cmd_active
        )

    def _manual_anchor_pos_b(
        self,
        torso_pos_w: np.ndarray,
        R_torso_w: np.ndarray,
        aligned_anchor_pos_w: np.ndarray,
    ) -> np.ndarray:
        anchor_pos_b = self.state_cmd.score_anchor_pos_cmd_b.astype(np.float32).copy()
        anchor_pos_b[2] = np.float32(aligned_anchor_pos_w[2] - torso_pos_w[2])
        return anchor_pos_b

    def _manual_anchor_ori_6d(self) -> np.ndarray:
        yaw_rel = float(self.state_cmd.score_anchor_yaw_cmd)
        rel_quat = np.array(
            [np.cos(yaw_rel * 0.5), 0.0, 0.0, np.sin(yaw_rel * 0.5)],
            dtype=np.float64,
        )
        return _rot6d_from_quat(rel_quat)

    def _compute_anchor_pos_b(
        self,
        ball_b_effective,
        torso_pos_w: np.ndarray,
        R_torso_w: np.ndarray,
        aligned_anchor_pos_w: np.ndarray,
    ) -> np.ndarray:
        """Compute anchor position observation in torso body frame."""
        if self.anchor_mode == "zero":
            return np.zeros(3, dtype=np.float32)

        if (self.manual_anchor_cmd
                and self.state_cmd.score_anchor_pos_cmd_active):
            return self._manual_anchor_pos_b(torso_pos_w, R_torso_w, aligned_anchor_pos_w)

        if self.anchor_mode == "ball_cmd":
            ball_pos_torso_b = self._get_ball_pos_torso_b(
                ball_b_effective, torso_pos_w, R_torso_w
            )
            if self.runtime_mode == "real":
                anchor_pos_b_ref = (
                    R_torso_w.T @ (aligned_anchor_pos_w - torso_pos_w)
                ).astype(np.float32)
                if self.state_cmd.ball_valid and ball_pos_torso_b is not None:
                    anchor_cmd_xy = ball_pos_torso_b[:2].astype(np.float32)
                    norm_xy = float(np.linalg.norm(anchor_cmd_xy))
                    if norm_xy > 1e-6:
                        clipped_norm_xy = np.clip(self.ball_anchor_cmd_scale * norm_xy, 0, 1.0)
                        anchor_pos_b_ball = (
                            self.ball_anchor_cmd_scale * clipped_norm_xy * (anchor_cmd_xy / norm_xy)
                        )
                        anchor_pos_b = np.concatenate(
                            [anchor_pos_b_ball, [aligned_anchor_pos_w[2] - torso_pos_w[2]]]
                        )
                    else:
                        anchor_pos_b = anchor_pos_b_ref
                else:
                    anchor_pos_b = anchor_pos_b_ref
                anchor_pos_b[2] = aligned_anchor_pos_w[2] - torso_pos_w[2]
                return anchor_pos_b.astype(np.float32)

            _adisp = aligned_anchor_pos_w - self._ref_anchor_world_origin
            _rdisp = torso_pos_w - self._entry_torso_pos_w
            anchor_pos_b_ref = (R_torso_w.T @ (_adisp - _rdisp)).astype(np.float32)
            anchor_cmd_xy = ball_pos_torso_b[:2]
            norm_xy = float(np.linalg.norm(anchor_cmd_xy))
            if norm_xy > 1e-6:
                clipped_norm_xy = np.clip(self.ball_anchor_cmd_scale * norm_xy, 0, 1.0)
                anchor_pos_b_ball = (
                    self.ball_anchor_cmd_scale * clipped_norm_xy * (anchor_cmd_xy / norm_xy)
                )
                return np.concatenate(
                    [anchor_pos_b_ball, [aligned_anchor_pos_w[2] - torso_pos_w[2]]]
                ).astype(np.float32)
            return anchor_pos_b_ref

        anchor_disp_w = aligned_anchor_pos_w - self._ref_anchor_world_origin
        robot_disp_w  = torso_pos_w - self._entry_torso_pos_w
        return (R_torso_w.T @ (anchor_disp_w - robot_disp_w)).astype(np.float32)

    def _compute_anchor_ori_6d(
        self,
        t: int,
        ball_b_effective,
        torso_quat_w: np.ndarray,
        torso_pos_w: np.ndarray,
        aligned_anchor_pos_w: np.ndarray,
        init_world_quat: np.ndarray,
    ) -> np.ndarray:
        """Compute anchor orientation observation in torso/body frame."""
        if (self.manual_anchor_cmd
                and self.state_cmd.score_anchor_yaw_cmd_active):
            return self._manual_anchor_ori_6d()

        if self.anchor_ori_mode == "ball_facing":
            if self.runtime_mode == "real":
                # Real robot: use the ball direction relative to torso.
                # Only horizontal relative yaw is kept here.
                R_torso_w = _quat_to_matrix(torso_quat_w)
                ball_pos_torso_b = self._get_ball_pos_torso_b(
                    ball_b_effective, torso_pos_w, R_torso_w
                )
                ball_relevant_pos_to_torso = (
                    np.zeros(2, dtype=np.float64)
                    if ball_pos_torso_b is None else
                    ball_pos_torso_b[:2].astype(np.float64)
                )
                norm_xy = float(np.linalg.norm(ball_relevant_pos_to_torso))
                yaw_rel = (
                    0.0 if norm_xy < 1e-6 else
                    float(np.arctan2(
                        ball_relevant_pos_to_torso[1],
                        ball_relevant_pos_to_torso[0],
                    ))
                )
                rel_quat = np.array(
                    [np.cos(yaw_rel * 0.5), 0.0, 0.0, np.sin(yaw_rel * 0.5)],
                    dtype=np.float64,
                )
                return _rot6d_from_quat(rel_quat)

            ball_pos_w_f64 = self.state_cmd.ball_pos_w.astype(np.float64)
            to_ball_w = ball_pos_w_f64 - torso_pos_w
            to_ball_w[2] = aligned_anchor_pos_w[2] - torso_pos_w[2]
            norm = np.linalg.norm(to_ball_w)
            if norm < 1e-6:
                to_ball_dir = np.array([1.0, 0.0, 0.0])
            else:
                to_ball_dir = to_ball_w / norm

            x_axis = np.array([1.0, 0.0, 0.0])
            d = float(np.dot(x_axis, to_ball_dir))
            if d < -1.0 + 1e-6:
                ball_facing_quat_w = np.array([0.0, 0.0, 0.0, 1.0])
            else:
                c = np.cross(x_axis, to_ball_dir)
                q_unnorm = np.array([1.0 + d, c[0], c[1], c[2]])
                ball_facing_quat_w = q_unnorm / np.linalg.norm(q_unnorm)

            rel_quat = _quat_mul(_quat_conj(torso_quat_w), ball_facing_quat_w)
            rel_quat = rel_quat / np.linalg.norm(rel_quat)
            return _rot6d_from_quat(rel_quat)

        ref_anchor_quat_w = self.motion_body_quat[t, NPZ_ANCHOR_IDX].astype(np.float64)
        aligned_quat = _quat_mul(init_world_quat, ref_anchor_quat_w)
        rel_quat = _quat_mul(_quat_conj(torso_quat_w), aligned_quat)
        rel_quat = rel_quat / np.linalg.norm(rel_quat)
        return _rot6d_from_quat(rel_quat)

    def _compute_ball_target_obs_b(self, ball_b_effective):
        """Compute pelvis-frame ball / target observations for the policy."""
        target_bias_vec = np.array([0.0, self.state_cmd.target_y_bias, 0.0], dtype=np.float32)
        ball_bias_vec   = np.array([0.0, self.state_cmd.ball_y_bias,   0.0], dtype=np.float32)

        if self.runtime_mode == "real":
            ball_pos_b = np.clip(ball_b_effective + ball_bias_vec, -8.0, 8.0).astype(np.float32)
            pelvis_quat = self.state_cmd.pelvis_quat_w.astype(np.float64)
            current_yaw_mat = _quat_to_matrix(_yaw_quat(pelvis_quat))
            if self.target_source == "apriltag":
                target_pos_b = self._get_real_target_from_apriltag(current_yaw_mat)
            else:
                target_pos_b = self._get_real_fixed_target_pos_b(current_yaw_mat)
            self._debug_target_pos_w = (
                self.state_cmd.pelvis_pos_w.astype(np.float64)
                + current_yaw_mat @ target_pos_b.astype(np.float64)
            ).astype(np.float32)
            target_pos_b = np.clip(target_pos_b + target_bias_vec, -8.0, 8.0).astype(np.float32)
            self._debug_target_corrected_pos_w = (
                self.state_cmd.pelvis_pos_w.astype(np.float64)
                + current_yaw_mat @ target_pos_b.astype(np.float64)
            ).astype(np.float32)
            return ball_pos_b, target_pos_b

        robot_pelvis_pos_w = self.state_cmd.pelvis_pos_w.astype(np.float64)
        R_pelvis = _quat_to_matrix(self.state_cmd.pelvis_quat_w.astype(np.float64))
        ball_rel_w = self.state_cmd.ball_pos_w.astype(np.float64) - robot_pelvis_pos_w
        target_rel_w = self.target_pos_w.astype(np.float64) - robot_pelvis_pos_w
        ball_pos_b = np.clip(R_pelvis.T @ ball_rel_w, -8.0, 8.0).astype(np.float32)
        target_pos_b = np.clip(
            (R_pelvis.T @ target_rel_w).astype(np.float32) + target_bias_vec, -8.0, 8.0
        ).astype(np.float32)
        self._debug_target_pos_w = self.target_pos_w.astype(np.float32)  # raw (no bias)
        self._debug_target_corrected_pos_w = (
            robot_pelvis_pos_w + R_pelvis @ target_pos_b.astype(np.float64)
        ).astype(np.float32)
        self._target_debug_source = "fixed_sim"
        return ball_pos_b, target_pos_b

    def _get_real_fixed_target_pos_b(self, current_yaw_mat: np.ndarray) -> np.ndarray:
        """Rotate the configured fixed target offset with pelvis yaw."""
        target_world = self._entry_yaw_mat @ self.target_pos_b_entry.astype(np.float64)
        self._target_debug_source = "fixed"
        return np.clip(current_yaw_mat.T @ target_world, -8.0, 8.0).astype(np.float32)

    def _get_real_target_from_apriltag(self, current_yaw_mat: np.ndarray) -> np.ndarray:
        """Use live target_state when valid; otherwise keep aiming with IMU yaw."""
        if self.state_cmd.target_valid:
            target_pos_b = np.clip(self.state_cmd.target_pos_b, -8.0, 8.0).astype(np.float32)
            self._target_world_yaw_vec = current_yaw_mat @ target_pos_b.astype(np.float64)
            self._target_debug_source = "apriltag"
            return target_pos_b

        if self.target_hold_on_loss_with_imu and self._target_world_yaw_vec is not None:
            self._target_debug_source = "imu_hold"
            return np.clip(
                current_yaw_mat.T @ self._target_world_yaw_vec,
                -8.0,
                8.0,
            ).astype(np.float32)

        if self.target_use_fixed_fallback:
            self._target_debug_source = "fixed_fallback"
            return self._get_real_fixed_target_pos_b(current_yaw_mat)

        self._target_debug_source = "none"
        return np.zeros(3, dtype=np.float32)

    def _get_trigger_ball_state_b(self):
        """Return `(ball_pos_b, ball_vel_b, anchor_xy)` in torso frame for trigger gating."""
        R_torso_w = _quat_to_matrix(self.state_cmd.torso_quat_w.astype(np.float64))
        if self.runtime_mode == "real":
            ball_pos_b = self._get_ball_pos_torso_b(
                self.state_cmd.ball_pos_b.astype(np.float32),
                self.state_cmd.torso_pos_w.astype(np.float64),
                R_torso_w,
            ).astype(np.float64)
            R_pelvis_w = _quat_to_matrix(self.state_cmd.pelvis_quat_w.astype(np.float64))
            ball_vel_b = (
                R_torso_w.T @ (R_pelvis_w @ self._estimate_ball_vel_b().astype(np.float64))
            )
        else:
            R_pelvis = _quat_to_matrix(self.state_cmd.pelvis_quat_w.astype(np.float64))
            ball_pos_b = R_torso_w.T @ (
                self.state_cmd.ball_pos_w.astype(np.float64)
                - self.state_cmd.torso_pos_w.astype(np.float64)
            )
            ball_vel_b = (
                R_torso_w.T @ self.state_cmd.ball_vel_w.astype(np.float64)
                - R_torso_w.T @ (R_pelvis @ self.state_cmd.root_lin_vel_b.astype(np.float64))
            )
        return ball_pos_b, ball_vel_b, self.trigger_base_offset_xy.copy()

    def _update_motion_trigger_state(self, policy_step: int):
        """Update finite-burst state machine before trigger evaluation."""
        if (self.trigger_gate_enabled and not self.trigger_play_once and self._motion_triggered
                and self.trigger_play_end_frame is not None):
            s0, s1 = self._trigger_segment_bounds()
            if policy_step - self._trigger_policy_step > (s1 - s0):
                self._motion_triggered = False
                self._burst_need_ball_clear = not self.manual_trigger
                print(
                    f"\n[Score] Played frames {s0}..{s1} -> hold frame 0, "
                    f"wait for next {'R3' if self.manual_trigger else 'ball'}",
                    flush=True,
                )

    def _ball_enters_circle(
        self,
        ball_pos_xy: np.ndarray,  # (2,)
        ball_vel_xy: np.ndarray,  # (2,)
        anchor_xy: np.ndarray,    # (2,)
        radius: float,
        horizon: float,
    ) -> bool:
        """True if the ball's linear trajectory passes within `radius` m of
        `anchor_xy` at any t ∈ [0, horizon] seconds.

        d²(t) = |dp + t·v|²  (dp = ball_pos - anchor)
              = |v|²·t² + 2·dot(dp,v)·t + |dp|²
        Minimum at t* = −dot(dp,v)/|v|²,  clamped to [0, horizon].
        """
        dp  = ball_pos_xy - anchor_xy
        v2  = float(np.dot(ball_vel_xy, ball_vel_xy))
        dpv = float(np.dot(dp, ball_vel_xy))
        dp2 = float(np.dot(dp, dp))

        t_star    = (-dpv / v2) if v2 > 1e-6 else 0.0
        t_clamp   = float(np.clip(t_star, 0.0, horizon))
        min_dist2 = v2 * t_clamp ** 2 + 2.0 * dpv * t_clamp + dp2

        return min_dist2 < radius * radius

    @staticmethod
    def _fmt_vec3(vec: np.ndarray) -> str:
        vec = np.asarray(vec, dtype=np.float32).reshape(3)
        return f"({vec[0]:+.2f},{vec[1]:+.2f},{vec[2]:+.2f})"


    def _build_obs(self) -> np.ndarray:
        """547-dim obs:
        command(58) | anchor_pos_b(3) | anchor_ori_b(6) |
        base_ang_vel(15) | joint_pos(145) | joint_vel(145) |
        actions(145) | soccer_pos_b(15) | target_pos_b(15)
        """
        policy_step = self.time_step - self.WARMUP_STEPS
        t = self._motion_frame_index(policy_step)

        # anchor obs uses torso_link as reference body (matches training: anchor_body_name = "torso_link")
        torso_quat_w = self.state_cmd.torso_quat_w.astype(np.float64)
        R_torso_w    = _quat_to_matrix(torso_quat_w)
        torso_pos_w  = self.state_cmd.torso_pos_w.astype(np.float64)

        # ---- command: ref_jpos + ref_jvel (Isaac Lab order) ----
        ref_jpos = self.motion_joint_pos[t]   # (29,) Isaac Lab order
        ref_jvel = self.motion_joint_vel[t]   # (29,) Isaac Lab order

        # Pelvis-frame ball for policy: sensor, or ball_obs_default_when_lost when invalid + ~zero.
        ball_b_effective = self._get_effective_ball_pos_b()
        if self.manual_anchor_cmd:
            self._update_manual_anchor_command()

        # ---- motion_anchor_pos_b (relative to torso, expressed in torso body frame) ----
        # Yaw-align the reference anchor world position, then express in torso body frame.
        init_world_quat      = _matrix_to_quat(self._init_to_world)
        ref_anchor_pos_w     = self.motion_body_pos[t, NPZ_ANCHOR_IDX].astype(np.float64)
        aligned_anchor_pos_w = self._init_to_world @ ref_anchor_pos_w
        if self._trigger_play_once_finished(policy_step):
            _adisp = aligned_anchor_pos_w - self._ref_anchor_world_origin
            _rdisp = torso_pos_w - self._entry_torso_pos_w
            anchor_pos_b = (R_torso_w.T @ (_adisp - _rdisp)).astype(np.float32)
        else:
            anchor_pos_b = self._compute_anchor_pos_b(
                ball_b_effective, torso_pos_w, R_torso_w, aligned_anchor_pos_w
            )

        # Cache for visualization (world-frame anchor position).
        self._debug_anchor_pos_w = (torso_pos_w + R_torso_w @ anchor_pos_b.astype(np.float64)).astype(np.float32)
        self._debug_torso_pos_w  = torso_pos_w.astype(np.float32)

        # ---- motion_anchor_ori_b (relative to torso orientation, in torso body frame) ----
        anchor_ori_6d = self._compute_anchor_ori_6d(
            t, ball_b_effective, torso_quat_w, torso_pos_w, aligned_anchor_pos_w, init_world_quat
        )
        anchor_x_axis_b = np.array(
            [anchor_ori_6d[0], anchor_ori_6d[2], anchor_ori_6d[4]],
            dtype=np.float64,
        )
        self._debug_anchor_x_axis_w = (
            R_torso_w @ anchor_x_axis_b
        ).astype(np.float32)

        # ---- Current joint state (Isaac Lab order) ----
        qj_il  = self.state_cmd.q[ISAAC_TO_MUJOCO]
        dqj_il = self.state_cmd.dq[ISAAC_TO_MUJOCO]
        jpos_cur = (qj_il - self.default_q_il).astype(np.float32)   # (29,)
        jvel_cur = dqj_il.astype(np.float32)                        # (29,)

        # ---- Ball and target in pelvis body frame (training uses root/pelvis, not torso) ----
        ball_pos_b, target_pos_b = self._compute_ball_target_obs_b(ball_b_effective)
        self._debug_ball_pos_b = ball_pos_b.copy()
        self._debug_target_pos_b = target_pos_b.copy()

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

        # Finite burst finished → hold frame 0 until next ball.
        self._update_motion_trigger_state(policy_step)

        # ---- Trigger gate ----
        if not self._motion_triggered:
            if self.manual_trigger:
                if self.state_cmd.score_manual_trigger:
                    self.state_cmd.score_manual_trigger = False
                    self._motion_triggered = True
                    self._trigger_policy_step = policy_step
                    if self.trigger_play_once:
                        s0 = int(np.clip(self.trigger_frame, 0, self.motion_total_steps - 1))
                        print(f"\n[Score] Manual trigger at policy_step={policy_step} "
                              f"-> play once from frame {s0} to clip end")
                    elif self.trigger_play_end_frame is not None:
                        s0, s1 = self._trigger_segment_bounds()
                        print(f"\n[Score] Manual trigger at policy_step={policy_step} "
                              f"-> play frames {s0}..{s1}")
                    else:
                        print(f"\n[Score] Manual trigger at policy_step={policy_step} "
                              f"-> from frame {self.trigger_frame} to clip end")
            else:
                ball_pos_b, ball_vel_b, anchor_xy = self._get_trigger_ball_state_b()
                in_circle = self._ball_enters_circle(
                    ball_pos_b[:2], ball_vel_b[:2], anchor_xy,
                    self.trigger_radius, self.trigger_horizon)
                if self._burst_need_ball_clear:
                    if not in_circle:
                        self._burst_need_ball_clear = False
                elif in_circle:
                    self._motion_triggered = True
                    self._trigger_policy_step = policy_step
                    if self.trigger_play_once:
                        s0 = int(np.clip(self.trigger_frame, 0, self.motion_total_steps - 1))
                        print(f"\n[Score] Ball trigger at policy_step={policy_step} "
                              f"-> play once from frame {s0} to clip end")
                    elif self.trigger_play_end_frame is not None:
                        s0, s1 = self._trigger_segment_bounds()
                        print(f"\n[Score] Ball trigger at policy_step={policy_step} "
                              f"-> play frames {s0}..{s1}")
                    else:
                        print(f"\n[Score] Ball trigger at policy_step={policy_step} "
                              f"-> from frame {self.trigger_frame} to clip end")
        else:
            self.state_cmd.score_manual_trigger = False

        obs = self._build_obs()

        out = self.ort_session.run(
            ["actions"],
            {"obs": obs[None, :]},
        )
        actions_il = out[0].squeeze(0)   # (29,) Isaac Lab order
        actions_il = np.clip(actions_il, self.action_clip_lo_il, self.action_clip_hi_il)
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
            print(f"  target_src    : {self._target_debug_source}")
            if self.manual_trigger or self.manual_anchor_cmd:
                print(f"  manual        : trigger={self.manual_trigger} "
                      f"anchor_cmd={self.state_cmd.score_anchor_pos_cmd_b} "
                      f"yaw={self.state_cmd.score_anchor_yaw_cmd:+.3f} "
                      f"active=({self.state_cmd.score_anchor_pos_cmd_active},"
                      f"{self.state_cmd.score_anchor_yaw_cmd_active})")
            print(f"  actions_il    : min={actions_il.min():.3f}  max={actions_il.max():.3f}")

        self.policy_output.actions = target_q
        self.policy_output.kps     = self.kps
        self.policy_output.kds     = self.kds
        self.policy_output.debug_target_pos_b = self._debug_target_pos_b.copy()
        self.policy_output.debug_target_source[:] = self.TARGET_SOURCE_TO_CODE.get(
            self._target_debug_source, 0.0
        )

        # ---- Visualization: anchor, anchor orientation, target, and trigger circle ----
        anchor_ori_arrow_to = (
            self._debug_anchor_pos_w.astype(np.float64)
            + 0.45 * self._debug_anchor_x_axis_w.astype(np.float64)
        )
        target_pos_w = self._debug_target_pos_w.astype(np.float64)
        target_marker_pos = np.array(
            [target_pos_w[0], target_pos_w[1], 0.15],
            dtype=np.float64,
        )
        viz = [
            {"pos": self._debug_anchor_pos_w.copy(), "radius": 0.06,
             "rgba": np.array([1.0, 0.5, 0.0, 0.9], dtype=np.float32)},
            {"from": self._debug_torso_pos_w.copy(),
             "to":   self._debug_anchor_pos_w.copy(), "radius": 0.008,
             "rgba": np.array([1.0, 0.5, 0.0, 0.5], dtype=np.float32)},
            {"from": self._debug_anchor_pos_w.copy(),
             "to":   anchor_ori_arrow_to, "radius": 0.025, "geom": "arrow",
             "rgba": np.array([0.6, 0.0, 1.0, 0.9], dtype=np.float32)},
            {"pos": target_pos_w, "radius": 0.07,
             "rgba": np.array([1.0, 0.0, 1.0, 0.95], dtype=np.float32)},
            {"from": self._debug_torso_pos_w.copy(),
             "to":   target_pos_w, "radius": 0.006,
             "rgba": np.array([1.0, 0.0, 1.0, 0.45], dtype=np.float32)},
            {"pos": target_marker_pos, "size": np.array([0.002, 0.15, 0.15]),
             "rgba": np.array([1.0, 0.4, 0.8, 0.85], dtype=np.float32)},
        ]
        # Corrected target (green disc) — only shown when bias is active
        if abs(self.state_cmd.target_y_bias) > 1e-4:
            corrected_marker_pos = np.array(
                [self._debug_target_corrected_pos_w[0], self._debug_target_corrected_pos_w[1], 0.15],
                dtype=np.float64,
            )
            viz.append({"pos": corrected_marker_pos, "size": np.array([0.002, 0.15, 0.15]),
                        "rgba": np.array([0.0, 1.0, 0.4, 0.90], dtype=np.float32)})
        # While waiting for the ball: show a semi-transparent cyan sphere indicating
        # the trigger circle radius around the configured torso-frame offset.
        if self.wait_for_ball and not self.manual_trigger and not self._motion_triggered:
            R_torso_yaw_w = _quat_to_matrix(
                _yaw_quat(self.state_cmd.torso_quat_w.astype(np.float64))
            )
            trigger_offset_w = R_torso_yaw_w @ np.array(
                [self.trigger_base_offset_xy[0], self.trigger_base_offset_xy[1], 0.0],
                dtype=np.float64,
            )
            trigger_center = np.array([
                self.state_cmd.torso_pos_w[0] + trigger_offset_w[0],
                self.state_cmd.torso_pos_w[1] + trigger_offset_w[1],
                0.1,
            ], dtype=np.float64)
            viz.append({"pos": trigger_center, "radius": self.trigger_radius,
                        "rgba": np.array([0.0, 1.0, 1.0, 0.15], dtype=np.float32)})
        self.policy_output.viz_spheres = viz

        self.time_step += 1
        capped = self._motion_frame_index(policy_step)
        self.policy_output.ghost_qpos = self._compute_ghost_qpos(capped)
        if self.trigger_gate_enabled and self.trigger_play_once:
            s0 = int(np.clip(self.trigger_frame, 0, self.motion_total_steps - 1))
            span_ct = self.motion_total_steps - s0
            bar_total = span_ct * self.control_dt
            if self._motion_triggered:
                st = policy_step - self._trigger_policy_step
                bar_prog = min(st + 1, span_ct) * self.control_dt
            else:
                bar_prog = 0.0
        elif self.trigger_gate_enabled and self.trigger_play_end_frame is not None:
            s0, s1 = self._trigger_segment_bounds()
            span_ct = s1 - s0 + 1
            bar_total = span_ct * self.control_dt
            if self._motion_triggered:
                st = policy_step - self._trigger_policy_step
                bar_prog = min(st + 1, span_ct) * self.control_dt
            else:
                bar_prog = 0.0
        else:
            bar_total = self.motion_total_steps * self.control_dt
            bar_prog = capped * self.control_dt
        status_line = progress_bar(bar_prog, bar_total)
        status_line += f" ball_b={self._fmt_vec3(self._debug_ball_pos_b)}"
        status_line += f" target_b={self._fmt_vec3(self._debug_target_pos_b)}"
        if abs(self.state_cmd.target_y_bias) > 1e-4:
            status_line += f" bias_y={self.state_cmd.target_y_bias:+.3f}m"
        if self.manual_trigger:
            status_line += f" trig={'go' if self._motion_triggered else 'R3'}"
        if self.manual_anchor_cmd:
            status_line += f" anchor_cmd={self._fmt_vec3(self.state_cmd.score_anchor_pos_cmd_b)}"
            status_line += f" yaw={self.state_cmd.score_anchor_yaw_cmd:+.2f}"
            status_line += f" active={int(self.state_cmd.score_anchor_cmd_active)}"
        if self.runtime_mode == "real":
            status_line += f" target_src={self._target_debug_source}"
        print(status_line, end="", flush=True)

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
        self.policy_output.debug_target_pos_b[:] = 0.0
        self.policy_output.debug_target_source[:] = 0.0
        print()

    def checkChange(self):
        cmd = self.state_cmd.skill_cmd
        if cmd == FSMCommand.LOCO:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.LOCOMODE
        elif cmd == FSMCommand.CMD_AMP:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.SKILL_AMP
        elif cmd == FSMCommand.CMD_BEYONDMIMIC_MJ:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.SKILL_BEYONDMIMIC_MJ
        elif cmd == FSMCommand.CMD_STANDUP_MJ:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.SKILL_STANDUP_MJ
        elif cmd == FSMCommand.CMD_PINOCCHIO_1_6_MJ:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.SKILL_PINOCCHIO_1_6_MJ
        elif cmd == FSMCommand.PASSIVE:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.PASSIVE
        elif cmd == FSMCommand.POS_RESET:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.FIXEDPOSE
        else:
            return FSMStateName.SKILL_SCORE
