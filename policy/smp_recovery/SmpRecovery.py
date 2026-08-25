"""SMP V3.3 plate-escape recovery policy for Unitree G1."""

import os

import numpy as np
import onnxruntime as ort
import yaml

from FSM.FSMState import FSMState, FSMStateName
from common.ctrlcomp import FSMCommand, PolicyOutput, StateAndCmd


class SmpRecovery(FSMState):
    """Run the normalized 96-dim SMP actor at 50 Hz."""

    def __init__(self, state_cmd: StateAndCmd, policy_output: PolicyOutput):
        super().__init__()
        self.state_cmd = state_cmd
        self.policy_output = policy_output
        self.name = FSMStateName.SKILL_SMP_RECOVERY
        self.name_str = "SMP_RECOVERY"

        current_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(current_dir, "config", "smp_recovery.yaml")) as f:
            cfg = yaml.safe_load(f)

        self.default_q = np.asarray(cfg["default_joint_pos"], dtype=np.float32)
        self.action_scale = np.asarray(cfg["action_scale"], dtype=np.float32)
        self.kps = np.asarray(cfg["kps"], dtype=np.float32)
        self.kds = np.asarray(cfg["kds"], dtype=np.float32)
        self.tau_limit = np.asarray(cfg["tau_limit"], dtype=np.float32)
        self.clip_actions = float(cfg["clip_actions"])
        self.warmup_steps = int(cfg["warmup_steps"])
        self.control_dt = float(cfg["control_dt"])

        for name, value in (
            ("default_joint_pos", self.default_q),
            ("action_scale", self.action_scale),
            ("kps", self.kps),
            ("kds", self.kds),
            ("tau_limit", self.tau_limit),
        ):
            if value.shape != (29,):
                raise ValueError(f"{name} must contain 29 values, got {value.shape}")

        self._last_action = np.zeros(29, dtype=np.float32)
        self._entry_q = self.default_q.copy()
        self._warmup_i = 0

        model_path = os.path.join(current_dir, "model", cfg["model_path"])
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(model_path, options)
        model_input = self._session.get_inputs()[0]
        model_output = self._session.get_outputs()[0]
        if model_input.shape[-1] != 96 or model_output.shape[-1] != 29:
            raise ValueError(
                f"Unexpected SMP model interface: {model_input.shape} -> {model_output.shape}"
            )
        self._input_name = model_input.name
        self._output_name = model_output.name

        dummy = np.zeros((1, 96), dtype=np.float32)
        for _ in range(5):
            self._session.run([self._output_name], {self._input_name: dummy})

        print(f"[SMP_RECOVERY] Initialized: {model_path} (96 -> 29)")

    def _build_obs(self) -> np.ndarray:
        """Match smp.rl.env_cfg actor-term order exactly."""
        obs = np.concatenate(
            (
                np.zeros(3, dtype=np.float32),  # base_lin_vel_b unavailable: fixed zero
                self.state_cmd.root_ang_vel_b,
                self.state_cmd.gravity_ori,
                self.state_cmd.q - self.default_q,
                self.state_cmd.dq,
                self._last_action,
            )
        ).astype(np.float32)
        if obs.shape != (96,):
            raise ValueError(f"SMP recovery observation must be (96,), got {obs.shape}")
        return obs

    def enter(self):
        self._last_action.fill(0.0)
        self._entry_q = self.state_cmd.q.astype(np.float32).copy()
        self._warmup_i = 0

    def run(self):
        obs = self._build_obs()[None, :]
        raw_action = self._session.run(
            [self._output_name], {self._input_name: obs}
        )[0].squeeze(0).astype(np.float32)
        raw_action = np.clip(raw_action, -self.clip_actions, self.clip_actions)
        self._last_action[:] = raw_action

        target_q = self.default_q + self.action_scale * raw_action

        # Ease in from the measured pose to avoid a target discontinuity.
        if self._warmup_i < self.warmup_steps:
            alpha = float(self._warmup_i + 1) / float(self.warmup_steps)
            target_q = (1.0 - alpha) * self._entry_q + alpha * target_q
            self._warmup_i += 1

        # Match training actuator saturation while sending position targets.
        damping = self.kds * self.state_cmd.dq
        target_lo = self.state_cmd.q + (damping - self.tau_limit) / self.kps
        target_hi = self.state_cmd.q + (damping + self.tau_limit) / self.kps
        target_q = np.clip(target_q, target_lo, target_hi)

        self.policy_output.actions = target_q
        self.policy_output.kps = self.kps
        self.policy_output.kds = self.kds

    def exit(self):
        self._last_action.fill(0.0)

    def checkChange(self) -> FSMStateName:
        cmd = self.state_cmd.skill_cmd
        transitions = {
            FSMCommand.POS_RESET: FSMStateName.FIXEDPOSE,
            FSMCommand.PASSIVE: FSMStateName.PASSIVE,
            FSMCommand.LOCO: FSMStateName.LOCOMODE,
            FSMCommand.SKILL_6: FSMStateName.SKILL_BEYONDMIMIC,
            FSMCommand.CMD_BEYONDMIMIC_MJ: FSMStateName.SKILL_BEYONDMIMIC_MJ,
            FSMCommand.CMD_FREEKICK: FSMStateName.SKILL_FREEKICK,
            FSMCommand.CMD_STANDUP_MJ: FSMStateName.SKILL_STANDUP_MJ,
            FSMCommand.CMD_AMP: FSMStateName.SKILL_AMP,
            FSMCommand.CMD_PINOCCHIO_1_6_MJ: FSMStateName.SKILL_PINOCCHIO_1_6_MJ,
        }
        next_state = transitions.get(cmd)
        if next_state is not None:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return next_state
        if cmd == FSMCommand.CMD_SMP_RECOVERY:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
        return FSMStateName.SKILL_SMP_RECOVERY
