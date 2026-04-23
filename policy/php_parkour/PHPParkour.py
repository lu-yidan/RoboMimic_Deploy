"""Perceptive Humanoid Parkour (PHP) policy — native-MuJoCo port of the
browser-side policyController.js. Runs the same ONNX main policy + depth
backbone, with a 7-step depth-latent delay queue, feeding a PD controller
whose torque is written straight to MjData.ctrl via the direct_torque flag.

Entry:   joystick D-pad LEFT      → CMD_PHP_PARKOUR
Exit:    L1 → PASSIVE, START → POS_RESET, B → LOCO
Speed:   L3 press → toggle high/low mode (mirrors PHP JS 'Y' key)
"""

from common.path_config import PROJECT_ROOT

import json
import os
import time
from collections import deque

import numpy as np
import onnxruntime as ort
import yaml
from scipy.spatial.transform import Rotation

from FSM.FSMState import FSMState
from common.ctrlcomp import StateAndCmd, PolicyOutput
from common.utils import FSMStateName, FSMCommand


# -----------------------------------------------------------------------------
# small helpers
# -----------------------------------------------------------------------------

def _parse_csv_floats(s):
    if s is None or s == "":
        return np.zeros(0, dtype=np.float32)
    return np.array([float(x.strip()) for x in s.split(",") if x.strip()],
                    dtype=np.float32)


def _parse_csv_strs(s):
    if s is None or s == "":
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _ensure_len(arr, n, fill=0.0):
    out = np.full(n, fill, dtype=np.float32)
    k = min(len(arr), n)
    if k > 0:
        out[:k] = arr[:k]
    return out


def _quat_apply_inverse_wxyz(q_wxyz, v):
    """Rotate vector v by the inverse of quat q (MuJoCo wxyz convention)."""
    x, y, z, w = q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]
    return Rotation.from_quat([x, y, z, w]).inv().apply(np.asarray(v))


def _bilinear_resize(img, out_h, out_w):
    """Match policyController.js _resizeBilinear exactly (half-pixel centers)."""
    in_h, in_w = img.shape
    if in_h == out_h and in_w == out_w:
        return img.astype(np.float32, copy=True)
    scale_y = in_h / out_h
    scale_x = in_w / out_w
    ys = (np.arange(out_h, dtype=np.float32) + 0.5) * scale_y - 0.5
    xs = (np.arange(out_w, dtype=np.float32) + 0.5) * scale_x - 0.5
    y0 = np.clip(np.floor(ys).astype(np.int32), 0, in_h - 1)
    y1 = np.clip(y0 + 1, 0, in_h - 1)
    x0 = np.clip(np.floor(xs).astype(np.int32), 0, in_w - 1)
    x1 = np.clip(x0 + 1, 0, in_w - 1)
    wy = (ys - y0).reshape(-1, 1)
    wx = (xs - x0).reshape(1, -1)
    v00 = img[np.ix_(y0, x0)]
    v10 = img[np.ix_(y0, x1)]
    v01 = img[np.ix_(y1, x0)]
    v11 = img[np.ix_(y1, x1)]
    v0 = v00 * (1 - wx) + v10 * wx
    v1 = v01 * (1 - wx) + v11 * wx
    return (v0 * (1 - wy) + v1 * wy).astype(np.float32)


# -----------------------------------------------------------------------------
# main class
# -----------------------------------------------------------------------------

class PHPParkour(FSMState):
    def __init__(self, state_cmd: StateAndCmd, policy_output: PolicyOutput):
        super().__init__()
        self.state_cmd = state_cmd
        self.policy_output = policy_output
        self.name = FSMStateName.SKILL_PHP_PARKOUR
        self.name_str = "php_parkour"

        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "config", "PHPParkour.yaml")) as f:
            cfg = yaml.safe_load(f)

        self.control_dt = float(cfg.get("control_dt", 0.02))
        self.depth_cfg = cfg["depth"]
        self.depth_pre_cfg = cfg["depth_pre"]
        self.cmd_cfg = cfg.get("command", {})
        self.stick_deadzone = float(self.cmd_cfg.get("stick_deadzone", 0.30))
        self.side_from_right_stick = bool(
            self.cmd_cfg.get("side_from_right_stick", False))
        self.trace_cfg = cfg.get("trace", {}) or {}
        self.trace_enabled = bool(self.trace_cfg.get("enabled", False))
        self.trace_path = str(self.trace_cfg.get(
            "path", "/tmp/php_parkour_trace.jsonl"))
        # Resolve relative paths against the deploy project root so that hydra's
        # per-run cwd change doesn't scatter trace files into output/ dirs.
        if self.trace_path and not os.path.isabs(self.trace_path):
            self.trace_path = os.path.join(str(PROJECT_ROOT), self.trace_path)
        self._trace_fp = None
        self._trace_tick = 0

        # --- ONNX sessions ---------------------------------------------------
        model_dir = os.path.join(here, "model")
        policy_path = os.path.join(model_dir, cfg["policy_path"])
        depth_path = os.path.join(model_dir, cfg["depth_backbone_path"])
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(policy_path, sess_options=so,
                                            providers=providers)
        self.depth_session = ort.InferenceSession(depth_path, sess_options=so,
                                                  providers=providers)

        self.input_name = self.session.get_inputs()[0].name
        self.input_names = [i.name for i in self.session.get_inputs()]
        out_names = [o.name for o in self.session.get_outputs()]
        self.output_name = next((n for n in out_names if "action" in n.lower()),
                                out_names[0])
        self.depth_input_name = self.depth_session.get_inputs()[0].name
        self.depth_output_name = self.depth_session.get_outputs()[0].name

        # --- metadata --------------------------------------------------------
        meta = self.session.get_modelmeta().custom_metadata_map
        joint_names = _parse_csv_strs(meta.get("joint_names", ""))
        obs_names = _parse_csv_strs(meta.get("observation_names", ""))
        if not joint_names:
            raise RuntimeError("policy.onnx metadata missing joint_names")
        if not obs_names:
            raise RuntimeError("policy.onnx metadata missing observation_names")

        n = len(joint_names)
        self.joint_names = joint_names
        self.obs_names = obs_names
        action_scale_raw = _parse_csv_floats(meta.get("action_scale", ""))
        if len(action_scale_raw) == 1 and n > 1:
            self.action_scale = np.full(n, action_scale_raw[0], dtype=np.float32)
        else:
            self.action_scale = _ensure_len(action_scale_raw, n, 1.0)
        self.default_joint_pos = _ensure_len(
            _parse_csv_floats(meta.get("default_joint_pos", "")), n, 0.0)
        self.kp = _ensure_len(
            _parse_csv_floats(meta.get("joint_stiffness", "")), n, 0.0)
        self.kd = _ensure_len(
            _parse_csv_floats(meta.get("joint_damping", "")), n, 0.0)

        # --- runtime state ---------------------------------------------------
        self.prev_action = np.zeros(n, dtype=np.float32)
        self.latest_target = self.default_joint_pos.copy()
        self.depth_latency = int(self.depth_pre_cfg.get("latency_steps", 7))
        self.depth_queue = deque(maxlen=self.depth_latency + 1)

        # Model binding (populated by bind_model() via deploy loop).
        self.joint_info = None  # list of dicts per policy joint
        self.root_qpos_adr = 0
        self.root_dof_adr = 0
        self.torso_body_id = -1
        self.gravity_dir = np.array([0, 0, -1], dtype=np.float32)
        self.actuator_ctrlrange = None
        self.n_actuators = 0

        self._is_bound = False
        self._failed = False
        self._warmed = False
        self._warn_no_depth = False
        # Debug: print on cmd/speed change only so the log stays readable.
        self._last_cmd_idx = -1
        self._last_high_speed = None
        self._cmd_label = {0: "IDLE", 1: "W", 2: "W+A", 3: "A", 4: "W+D", 5: "D",
                           6: "W(HI)", 7: "W+A(HI)", 8: "A(HI)", 9: "W+D(HI)",
                           10: "D(HI)"}

    # ------------------------------------------------------------------
    # Model binding — called by deploy_mujoco after MjModel is loaded.
    # ------------------------------------------------------------------
    def bind_model(self, model):
        import mujoco
        # Free joint (root)
        free_jnt = np.where(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE.value)[0]
        root_jid = int(free_jnt[0]) if len(free_jnt) else 0
        self.root_qpos_adr = int(model.jnt_qposadr[root_jid])
        self.root_dof_adr = int(model.jnt_dofadr[root_jid])

        # torso body
        tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.torso_body_id = int(tid) if tid >= 0 else -1

        # gravity
        g = np.asarray(model.opt.gravity, dtype=np.float32)
        gn = float(np.linalg.norm(g))
        self.gravity_dir = (g / gn if gn > 0 else
                            np.array([0, 0, -1], dtype=np.float32))

        # actuator → joint lookup: actuator_trnid[i, 0] = joint id
        self.n_actuators = int(model.nu)
        act_joint_to_ctrl = {}
        for i in range(self.n_actuators):
            jid = int(model.actuator_trnid[i, 0])
            if jid >= 0 and jid not in act_joint_to_ctrl:
                act_joint_to_ctrl[jid] = i

        self.actuator_ctrlrange = np.asarray(model.actuator_ctrlrange,
                                             dtype=np.float32).copy()

        # build per-policy-joint lookup
        info = []
        missing_joints, missing_actuators = [], []
        for name in self.joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                info.append({"name": name, "joint_id": -1,
                             "qposadr": -1, "qveladr": -1, "ctrl_idx": -1})
                missing_joints.append(name)
                continue
            qadr = int(model.jnt_qposadr[jid])
            vadr = int(model.jnt_dofadr[jid])
            cidx = act_joint_to_ctrl.get(jid, -1)
            if cidx < 0:
                missing_actuators.append(name)
            info.append({"name": name, "joint_id": jid,
                         "qposadr": qadr, "qveladr": vadr, "ctrl_idx": cidx})
        self.joint_info = info
        if missing_joints:
            print(f"[PHP] policy joints missing from model: {missing_joints}")
        if missing_actuators:
            print(f"[PHP] policy joints without actuators: {missing_actuators}")
        # One-time mapping dump so the user can eyeball that policy idx i
        # resolves to the right XML actuator/joint with the right kp/kd.
        print("[PHP] joint map (policy_idx -> name -> ctrl_idx  kp  kd)")
        for i, inf in enumerate(self.joint_info):
            print(f"  [{i:2d}] {inf['name']:<28s} ctrl={inf['ctrl_idx']:2d}  "
                  f"kp={self.kp[i]:7.3f}  kd={self.kd[i]:6.3f}  "
                  f"default={self.default_joint_pos[i]:+.3f}")

        # observation buffer size (known once joint count is fixed)
        self.obs_size = self._compute_obs_size()
        self._is_bound = True

    # ------------------------------------------------------------------
    def _compute_obs_size(self):
        size = 0
        n = len(self.joint_names)
        for name in self.obs_names:
            if name in ("base_lin_vel", "base_ang_vel",
                         "projected_gravity", "robot_anchor_projected_gravity",
                         "command"):
                size += 3
            elif name == "placeholder":
                size += 15
            elif name in ("joint_pos", "joint_vel", "actions"):
                size += n
            else:
                raise RuntimeError(f"Unknown observation name: {name}")
        return size

    # ------------------------------------------------------------------
    # FSMState hooks
    # ------------------------------------------------------------------
    def enter(self):
        # Reset latents + action history so we start from a clean slate.
        self.depth_queue.clear()
        n = len(self.joint_names)
        self.prev_action = np.zeros(n, dtype=np.float32)
        self.latest_target = self.default_joint_pos.copy()
        self._failed = False
        # Precompute reordered (actuator-index) kp/kd that the outer 500 Hz
        # PD loop will use. Target is emitted fresh each policy tick.
        num = self.state_cmd.num_joints
        self._kps_reorder = np.zeros(num, dtype=np.float32)
        self._kds_reorder = np.zeros(num, dtype=np.float32)
        for i, info in enumerate(self.joint_info or []):
            ci = info["ctrl_idx"]
            if ci >= 0:
                self._kps_reorder[ci] = self.kp[i]
                self._kds_reorder[ci] = self.kd[i]
        # Emit the initial target (= default pose) so there's no 1-tick gap.
        self._emit_target()
        self.policy_output.direct_torque = False
        # Reset edge-trigger so the first cmd is logged immediately.
        self._last_cmd_idx = -1
        self._last_high_speed = None
        print(f"[PHP] enter  speed={'HIGH' if self.state_cmd.php_high_speed else 'LOW'}")
        # Trace file: overwrite each enter() so a fresh session = fresh file.
        if self._trace_fp is not None:
            try: self._trace_fp.close()
            except Exception: pass
            self._trace_fp = None
        if self.trace_enabled:
            try:
                parent = os.path.dirname(self.trace_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                self._trace_fp = open(self.trace_path, "w")
                self._trace_tick = 0
                self._trace_t0 = time.time()
                print(f"[PHP] trace -> {self.trace_path}")
            except Exception as e:
                print(f"[PHP] trace disabled: cannot open {self.trace_path}: {e}")
                self._trace_fp = None

    def run(self):
        if self._failed:
            # Already requested PASSIVE — just hold zero torque until FSM switches.
            self._emit_zero_torque()
            return
        if not self._is_bound:
            print("[PHP] run() called before bind_model; emitting zero torque")
            self._emit_zero_torque()
            return
        try:
            self._run_once()
        except Exception as e:  # fall back to PASSIVE on any inference failure
            print(f"[PHP] inference failed ({type(e).__name__}: {e}); "
                  f"falling back to PASSIVE")
            self._failed = True
            self.state_cmd.skill_cmd = FSMCommand.PASSIVE
            self._emit_zero_torque()

    def exit(self):
        self.policy_output.direct_torque = False
        self._failed = False
        if self._trace_fp is not None:
            try: self._trace_fp.close()
            except Exception: pass
            self._trace_fp = None
            print(f"[PHP] trace closed ({self._trace_tick} ticks)")

    def checkChange(self):
        cmd = self.state_cmd.skill_cmd
        if cmd == FSMCommand.PASSIVE:
            return FSMStateName.PASSIVE
        if cmd == FSMCommand.POS_RESET:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.FIXEDPOSE
        if cmd == FSMCommand.LOCO:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.LOCOMODE
        return FSMStateName.SKILL_PHP_PARKOUR

    # ------------------------------------------------------------------
    # inference core
    # ------------------------------------------------------------------
    def _run_once(self):
        cmd15 = self._cmd15_from_vel_cmd()

        # Depth backbone → 32-dim latent, with 7-step delay queue.
        depth_img = self.state_cmd.depth_image
        latent = None
        if depth_img is not None:
            latent = self._run_depth_backbone(depth_img)
        elif not self._warn_no_depth:
            print("[PHP] no depth image available; feeding zeros to policy "
                  "(deploy_mujoco did not set state_cmd.depth_image)")
            self._warn_no_depth = True

        if latent is None or latent.shape[-1] != 32:
            latent = np.zeros(32, dtype=np.float32)
        self.depth_queue.append(latent.astype(np.float32, copy=True))
        if len(self.depth_queue) >= self.depth_latency + 1:
            delayed = self.depth_queue[0]
        else:
            delayed = self.depth_queue[0]  # partial queue: use oldest available

        obs = self._build_observation(cmd15)
        full = np.concatenate([obs, delayed]).astype(np.float32).reshape(1, -1)
        feeds = {self.input_name: full}
        if "time_step" in self.input_names:
            feeds["time_step"] = np.zeros((1, 1), dtype=np.float32)
        outputs = self.session.run([self.output_name], feeds)
        action = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        n = len(self.joint_names)
        if action.shape[0] != n:
            raise RuntimeError(f"Policy output shape {action.shape} != joint "
                               f"count {n}")

        # --- trace (before updating prev_action so "prev_action" below is
        #     what was actually fed to the policy this tick) -----------------
        if self._trace_fp is not None:
            self._trace_write(obs, delayed, depth_img, action, cmd15)

        self.prev_action = action.copy()
        self.latest_target = (self.default_joint_pos
                              + self.action_scale * action)

        self._emit_target()

    # ------------------------------------------------------------------
    def _trace_write(self, obs, depth_latent, depth_img, action, cmd15):
        """Dump one JSONL line so we can diff PHP behaviour against the
        browser frame-by-frame. Fields are chosen to cover every input to
        the policy plus the output, so any divergence can be localized."""
        try:
            n = len(self.joint_names)
            # Split obs into its named slots (same order as obs_names).
            slots = {}
            off = 0
            for name in self.obs_names:
                if name in ("base_lin_vel", "base_ang_vel",
                            "projected_gravity",
                            "robot_anchor_projected_gravity", "command"):
                    k = 3
                elif name == "placeholder":
                    k = 15
                else:
                    k = n  # joint_pos / joint_vel / actions
                slots[name] = obs[off:off + k].tolist()
                off += k
            dimg = (np.asarray(depth_img, dtype=np.float32)
                    if depth_img is not None else None)
            rec = {
                "tick": int(self._trace_tick),
                "t_wall": float(time.time() - self._trace_t0),
                "root_pos":  [float(x) for x in self.state_cmd.pelvis_pos_w],
                "root_quat": [float(x) for x in self.state_cmd.pelvis_quat_w],
                "torso_quat":[float(x) for x in self.state_cmd.torso_quat_w],
                "obs_slots": slots,
                "cmd15":     cmd15.tolist(),
                "depth_latent": (depth_latent.tolist()
                                 if depth_latent is not None else None),
                "depth_stats": ({"shape": list(dimg.shape),
                                 "min": float(dimg.min()),
                                 "max": float(dimg.max()),
                                 "mean": float(dimg.mean())}
                                if dimg is not None else None),
                "action":    action.tolist(),
            }
            self._trace_fp.write(json.dumps(rec, separators=(",", ":")) + "\n")
            self._trace_tick += 1
        except Exception as e:
            # Never let tracing crash the run loop.
            print(f"[PHP] trace write failed: {e}")

    def _emit_zero_torque(self):
        # Hold position (target=current qpos) with zero gains so outer PD
        # produces no torque. Used when inference fails or pre-bind.
        num = self.state_cmd.num_joints
        self.policy_output.actions = self.state_cmd.q.astype(np.float32).copy()
        self.policy_output.kps = np.zeros(num, dtype=np.float32)
        self.policy_output.kds = np.zeros(num, dtype=np.float32)
        self.policy_output.direct_torque = False

    def _emit_target(self):
        """Write target (actuator-indexed) + reordered kp/kd into policy_output.
        The outer 500 Hz loop in deploy_mujoco.py then runs:
            tau = kp * (target - qpos) + kd * (0 - qvel)
        every physics step — matching the browser's per-step PD.
        """
        num = self.state_cmd.num_joints
        target_out = self.state_cmd.q.astype(np.float32).copy()
        for i, info in enumerate(self.joint_info):
            ci = info["ctrl_idx"]
            if ci < 0:
                continue
            target_out[ci] = float(self.latest_target[i])
        self.policy_output.actions = target_out
        self.policy_output.kps = self._kps_reorder
        self.policy_output.kds = self._kds_reorder
        self.policy_output.direct_torque = False

    # ------------------------------------------------------------------
    def _cmd15_from_vel_cmd(self):
        """Mirror PHP's keyboard → 15-dim one-hot.
        Forward + L/R from stick; high/low speed from L3-toggled flag.
        """
        arr = np.zeros(15, dtype=np.float32)
        fwd = float(self.state_cmd.vel_cmd[0])
        # vel_cmd[1] = left-stick X (+ = left). vel_cmd[2] = right-stick X
        # (deploy_mujoco maps axis 3 with the same +=left sign). When the
        # user prefers right-stick strafing, take the side value from there.
        if self.side_from_right_stick:
            side = float(self.state_cmd.vel_cmd[2])
        else:
            side = float(self.state_cmd.vel_cmd[1])
        dz = self.stick_deadzone
        is_w = fwd > dz
        is_a = side > dz
        is_d = side < -dz
        if is_w and is_a:
            base = 2
        elif is_w and is_d:
            base = 4
        elif is_w:
            base = 1
        elif is_a:
            base = 3
        elif is_d:
            base = 5
        else:
            base = 0
        if base != 0 and bool(self.state_cmd.php_high_speed):
            idx = {1: 6, 2: 7, 3: 8, 4: 9, 5: 10}[base]
        else:
            idx = base
        arr[idx] = 1.0
        # Edge-triggered log: print whenever the one-hot index or the
        # HIGH/LOW flag changes. Keeps the log readable while still letting
        # the user verify exactly what the policy is being commanded to do.
        hi = bool(self.state_cmd.php_high_speed)
        if idx != self._last_cmd_idx or hi != self._last_high_speed:
            label = self._cmd_label.get(idx, f"idx={idx}")
            print(f"[PHP] cmd={label}  one-hot={idx}  "
                  f"speed={'HIGH' if hi else 'LOW'}  "
                  f"stick=(fwd={fwd:+.2f}, side={side:+.2f})")
            self._last_cmd_idx = idx
            self._last_high_speed = hi
        return arr

    # ------------------------------------------------------------------
    def _build_observation(self, cmd15):
        data = self.state_cmd  # shorthand; see _fill_from_mjdata below
        # NOTE: deploy_mujoco has already written q / dq / ang_vel / gravity_ori
        # into state_cmd each control tick. We still need root lin vel and
        # torso_link gravity projection, which come from state_cmd fields set
        # by the outer loop (torso_quat_w, root_lin_vel_b already computed).
        obs = np.zeros(self.obs_size, dtype=np.float32)
        off = 0
        n = len(self.joint_names)

        for name in self.obs_names:
            if name == "base_lin_vel":
                obs[off:off + 3] = self.state_cmd.root_lin_vel_b
                off += 3
            elif name == "base_ang_vel":
                # PHP reads qvel[root+3:6] directly (body-frame for free joint).
                obs[off:off + 3] = self.state_cmd.root_ang_vel_b
                off += 3
            elif name == "projected_gravity":
                # gravity in root (pelvis) frame
                rq = self.state_cmd.pelvis_quat_w  # [w,x,y,z]
                obs[off:off + 3] = _quat_apply_inverse_wxyz(rq, self.gravity_dir)
                off += 3
            elif name == "robot_anchor_projected_gravity":
                tq = self.state_cmd.torso_quat_w
                obs[off:off + 3] = _quat_apply_inverse_wxyz(tq, self.gravity_dir)
                off += 3
            elif name == "command":
                # PHP hard-codes [0,0,0].
                obs[off:off + 3] = 0.0
                off += 3
            elif name == "placeholder":
                obs[off:off + 15] = cmd15
                off += 15
            elif name == "joint_pos":
                for i, info in enumerate(self.joint_info):
                    if info["qposadr"] < 0:
                        continue
                    # state_cmd.q is indexed by 0..num_joints-1, matching
                    # qpos[7:7+num_joints] order (joint declaration order).
                    # But our policy joint i may not share that index, so
                    # index by qposadr-7 for safety.
                    k = info["qposadr"] - self.root_qpos_adr - 7
                    # root contributes (3 pos + 4 quat) = 7 dofs before hinges
                    # which is why we subtract 7.
                    if 0 <= k < len(self.state_cmd.q):
                        obs[off + i] = (float(self.state_cmd.q[k])
                                        - float(self.default_joint_pos[i]))
                off += n
            elif name == "joint_vel":
                for i, info in enumerate(self.joint_info):
                    if info["qveladr"] < 0:
                        continue
                    k = info["qveladr"] - self.root_dof_adr - 6
                    if 0 <= k < len(self.state_cmd.dq):
                        obs[off + i] = float(self.state_cmd.dq[k])
                off += n
            elif name == "actions":
                obs[off:off + n] = self.prev_action
                off += n
            else:
                raise RuntimeError(f"Unknown observation name: {name}")
        return obs

    # ------------------------------------------------------------------
    def _run_depth_backbone(self, depth_img):
        """PHP: crop → resize → clip+normalize → vflip → 1×H×W → backbone."""
        img = np.asarray(depth_img, dtype=np.float32)
        if img.ndim != 2:
            raise RuntimeError(f"depth_image must be 2D, got shape {img.shape}")
        crop = self.depth_pre_cfg["crop"]
        top, left = int(crop["top"]), int(crop["left"])
        right, bottom = int(crop["right"]), int(crop["bottom"])
        H, W = img.shape
        cropped = img[top:H - bottom, left:W - right]

        out_w = int(self.depth_pre_cfg["resize"]["width"])
        out_h = int(self.depth_pre_cfg["resize"]["height"])
        resized = _bilinear_resize(cropped, out_h, out_w)

        lo = float(self.depth_pre_cfg["clip_min"])
        hi = float(self.depth_pre_cfg["clip_max"])
        normalized = (resized - lo) / (hi - lo) - 0.5
        # Unlike the browser, mujoco.Renderer.render() already applies
        # np.flipud internally (row 0 = top of view). Skipping the vflip
        # the browser does on its OpenGL-ordered buffer.
        tensor = normalized[np.newaxis, :, :].astype(np.float32, copy=True)

        out = self.depth_session.run([self.depth_output_name],
                                     {self.depth_input_name: tensor})[0]
        latent = np.asarray(out, dtype=np.float32).reshape(-1)
        if latent.shape[0] != 32:
            raise RuntimeError(f"Depth backbone output size {latent.shape[0]}"
                               f" != 32")
        return latent

