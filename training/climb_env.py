"""Initial support-pose task with the deployment position-action contract.

Reward/termination use simulation truth. Actor target is an ideal simulated
body-relative observation; perception/noise and force feedback are not modeled.
No old research QP, external base force or kinematic state overwrite is used.
"""
from pathlib import Path
import hashlib
import json
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from policy.climb.contract import ClimbContract

DEFAULT_BUNDLE = Path(__file__).resolve().parents[1] / "assets/climb_mounted"


class ClimbSupportEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, bundle=DEFAULT_BUNDLE, horizon_s=4., render_mode=None):
        if render_mode not in (None, "rgb_array"):
            raise ValueError("Unsupported render mode")
        self.render_mode = render_mode
        self.bundle = Path(bundle).resolve()
        manifest = json.loads((self.bundle / "manifest.json").read_text())
        for name, expected in manifest["output_sha256"].items():
            path = (self.bundle / name).resolve()
            if not path.is_relative_to(self.bundle) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("Scene bundle integrity mismatch: " + name)
        self.contract = ClimbContract.from_json(self.bundle / "contract.json")
        if self.contract.fingerprint != manifest["contract_sha256"]:
            raise ValueError("Bundle contract mismatch")
        self.model = mujoco.MjModel.from_xml_path(str(self.bundle / "model.xml"))
        self.data = mujoco.MjData(self.model)
        self.initial = np.load(self.bundle / "initial_qpos.npy", allow_pickle=False)
        self.joints = np.array([self.model.joint(name).id for name in self.contract.joint_names])
        self.qadr = self.model.jnt_qposadr[self.joints]
        self.vadr = self.model.jnt_dofadr[self.joints]
        self.motors = []
        for jid in self.joints:
            found = np.flatnonzero(self.model.actuator_trnid[:, 0] == jid)
            if len(found) != 1:
                raise ValueError("Expected one motor per declared joint")
            self.motors.append(int(found[0]))
        self.motors = np.asarray(self.motors)
        if self.model.nu != 29 or self.model.neq or np.any(self.model.actuator_biastype) or np.any(self.model.actuator_gaintype):
            raise ValueError("Only unconstrained direct-motor G1 supported")
        np.testing.assert_array_equal(self.model.actuator_gear[self.motors, 0], np.ones(29))
        if not self.model.actuator_ctrllimited[self.motors].all():
            raise ValueError("All motor torque limits must be explicit")
        self.limits = self.model.actuator_ctrlrange[self.motors]
        ratio = self.contract.control_dt / self.model.opt.timestep
        self.substeps = round(ratio)
        if self.substeps < 1 or abs(self.substeps-ratio) > 1e-9:
            raise ValueError("Policy period must be an integer physics-step multiple")
        count = horizon_s / self.contract.control_dt
        if not np.isfinite(count) or count < 1 or abs(count-round(count)) > 1e-9:
            raise ValueError("Horizon must contain an integer number of policy steps")
        self.max_steps = round(count)
        self.pelvis = self.model.body("pelvis").id
        geometry = manifest.get("task_geometry", {})
        self.target = self.model.geom(geometry.get("target_geom", "right_bar")).id
        self.target_axis_offset = float(geometry.get("target_axis_offset_m", 0.))
        self.target_axis_sign = float(geometry.get("target_axis_sign", 1.))
        if self.target_axis_sign not in (-1.,1.):
            raise ValueError("Target axis sign must be -1 or 1")
        if not np.isfinite(self.target_axis_offset):
            raise ValueError("Target offset must be finite")
        self.hand_sites = [self.model.site(name).id for name in ("contact", "right_contact")]
        self.action_space = spaces.Box(-1., 1., (29,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (100,), dtype=np.float32)
        self.renderer = None
        self.previous_action = np.zeros(29, np.float32)
        self.finished = True

    def _observation(self):
        rotation = self.data.xmat[self.pelvis].reshape(3, 3)
        axis_world = self.target_axis_sign * self.data.geom_xmat[self.target].reshape(3, 3)[:, 2]
        target_world = self.data.geom_xpos[self.target] + self.target_axis_offset * axis_world
        position = rotation.T @ (target_world - self.data.xpos[self.pelvis])
        axis = rotation.T @ axis_world
        return self.contract.observation(self.data.qpos[self.qadr], self.data.qvel[self.vadr],
            rotation.T @ np.array([0., 0., -1.]), self.data.qvel[3:6],
            self.previous_action, position, axis, True)

    def _metrics(self):
        root_error = float(np.linalg.norm(self.data.qpos[:3] - self.initial[:3]))
        up = self.data.xmat[self.pelvis].reshape(3, 3)[2, 2]
        tilt = float(np.arccos(np.clip(up, -1., 1.)))
        hand_error = float(np.max(np.linalg.norm(self.data.site_xpos[self.hand_sites] - self.initial_hands, axis=1)))
        return root_error, tilt, hand_error

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError("This first deterministic baseline does not randomize resets")
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.initial
        mujoco.mj_forward(self.model, self.data)
        self.initial_hands = self.data.site_xpos[self.hand_sites].copy()
        self.previous_action.fill(0.)
        self.steps = 0
        self.finished = False
        self.peak_root_error = self.peak_tilt = self.peak_hand_error = 0.
        return self._observation(), self._info("reset", 0., 0., 0., 0.)

    def _info(self, reason, root, tilt, hands, saturation):
        return dict(task="initial_support_pose", reason=reason, simulation_time_s=float(self.data.time),
            root_error_m=root, tilt_rad=tilt, hand_site_error_m=hands,
            motor_saturation_fraction=saturation, contract_sha256=self.contract.fingerprint,
            pose_hold_success=False, bar_load_success_verified=False,
            actor_target_source="ideal_simulation_body_relative_geometry")

    def step(self, action):
        if self.finished:
            raise RuntimeError("Call reset before stepping or after episode end")
        target, clipped = self.contract.decode_action(action)
        difference = clipped - self.previous_action
        self.previous_action = clipped.copy()
        self.steps += 1
        saturation = 0
        executed = 0
        terminated = False
        reason = "running"
        warnings_before = self.data.warning.number.copy()
        for _ in range(self.substeps):
            torque = self.contract.kps*(target-self.data.qpos[self.qadr])-self.contract.kds*self.data.qvel[self.vadr]
            saturation += int(np.sum((torque < self.limits[:, 0]) | (torque > self.limits[:, 1])))
            self.data.ctrl[self.motors] = np.clip(torque, self.limits[:, 0], self.limits[:, 1])
            mujoco.mj_step(self.model, self.data)
            executed += 1
            # Bring kinematic observations to the integrated state. No mj_forward
            # contact/sensor resampling or hidden feedback controller is applied.
            mujoco.mj_kinematics(self.model, self.data)
            root, tilt, hands = self._metrics()
            self.peak_root_error = max(self.peak_root_error, root)
            self.peak_tilt = max(self.peak_tilt, tilt)
            self.peak_hand_error = max(self.peak_hand_error, hands)
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                reason, terminated = "nonfinite_physics", True
            elif np.any(self.data.warning.number > warnings_before):
                reason, terminated = "mujoco_warning", True
            elif self.data.qpos[2] < self.initial[2]-.15 or tilt > np.deg2rad(45):
                reason, terminated = "fall", True
            elif root > .2:
                reason, terminated = "support_region_exit", True
            if terminated:
                break
        truncated = bool(self.steps >= self.max_steps and not terminated)
        self.finished = terminated or truncated
        root, tilt, hands = self._metrics()
        reward = float(np.exp(-(root/.03)**2-(tilt/.15)**2-(hands/.03)**2)
                       -.01*np.mean(clipped**2)-.02*np.mean(difference**2)-float(terminated))
        info = self._info("time_limit" if truncated else reason, root, tilt, hands,
                          saturation/(29*executed))
        info["physics_steps_this_action"] = executed
        info.update(peak_root_error_m=self.peak_root_error, peak_tilt_rad=self.peak_tilt,
                    peak_hand_site_error_m=self.peak_hand_error)
        info["pose_hold_success"] = bool(truncated and self.peak_root_error < .03
            and self.peak_tilt < np.deg2rad(10) and self.peak_hand_error < .03)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError("Nonfinite physics state; no fabricated observation returned")
        return self._observation(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [.29, 0, .88]
        camera.distance, camera.azimuth, camera.elevation = 2.45, -36., -12.
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
