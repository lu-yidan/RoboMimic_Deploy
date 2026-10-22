"""Framework-compatible position-policy adapter shared with future training.

No simulation truth, QP controller, or fabricated force sensing is used here.
The default instance cannot activate without an explicitly supplied actor.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from FSM.FSMState import FSMState
from common.utils import FSMStateName, FSMCommand
from policy.climb.contract import ClimbContract


class OnnxActor:
    def __init__(self, model_path, metadata_path, contract):
        model_path = Path(model_path)
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        if metadata.get("contract_sha256") != contract.fingerprint:
            raise ValueError("Model and deployment observation/action contracts differ")
        if metadata.get("model_sha256") != hashlib.sha256(model_path.read_bytes()).hexdigest():
            raise ValueError("Model bytes differ from export metadata")
        import onnxruntime
        self.session = onnxruntime.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != "obs" or inputs[0].type != "tensor(float)" or inputs[0].shape != [1, 100]:
            raise ValueError("Expected one float32 ONNX input obs[1,100]")
        if len(outputs) != 1 or outputs[0].name != "actions" or outputs[0].type != "tensor(float)" or outputs[0].shape != [1, 29]:
            raise ValueError("Expected one float32 ONNX output actions[1,29]")

    def __call__(self, observation):
        return self.session.run(["actions"], {"obs": observation[None, :]})[0][0]


class Climb(FSMState):
    def __init__(self, state_cmd, policy_output, *, config_path=None, actor=None,
                 model_path=None, metadata_path=None):
        super().__init__()
        if state_cmd.num_joints != 29:
            raise ValueError("Climb requires 29 hardware-ordered joints")
        self.state_cmd, self.policy_output = state_cmd, policy_output
        path = config_path or Path(__file__).parent / "config/climb.json"
        self.contract = ClimbContract.from_json(path)
        self.control_dt = self.contract.control_dt
        self.name, self.name_str = FSMStateName.SKILL_CLIMB, "climb"
        if actor is not None and model_path is not None:
            raise ValueError("Choose an injected actor or an ONNX model")
        if model_path is not None:
            if metadata_path is None:
                raise ValueError("An exported model requires contract/hash metadata")
            actor = OnnxActor(model_path, metadata_path, self.contract)
        self.actor = actor
        self.previous_action = np.zeros(29, dtype=np.float32)
        self.last_observation = None
        self.fault = None

    def _damping(self):
        self.policy_output.actions = np.zeros(29, dtype=np.float32)
        self.policy_output.kps = np.zeros(29, dtype=np.float32)
        self.policy_output.kds = np.full(29, 8.0, dtype=np.float32)

    def enter(self):
        self.previous_action.fill(0)
        self.fault = None
        self.last_observation = None
        if self.actor is None:
            self.fault = "No trained climbing actor loaded"
        self._damping()

    def run(self):
        if self.fault is not None:
            self._damping()
            return
        try:
            state = self.state_cmd
            if not state.climb_target_valid:
                raise ValueError("Climbing target is unavailable")
            observation = self.contract.observation(
                state.q, state.dq, state.gravity_ori, state.ang_vel,
                self.previous_action, state.climb_target_pos_b,
                state.climb_target_axis_b, state.climb_target_valid,
            )
            target, clipped = self.contract.decode_action(self.actor(observation.copy()))
            self.policy_output.actions = target
            self.policy_output.kps = self.contract.kps
            self.policy_output.kds = self.contract.kds
            self.previous_action = clipped.copy()
            self.last_observation = observation.copy()
        except Exception as exc:
            # This is a software fallback, not a proved physical recovery.
            self.fault = f"{type(exc).__name__}: {exc}"
            self._damping()

    def exit(self):
        self.previous_action.fill(0)
        self.policy_output.ghost_qpos = None
        self.policy_output.viz_spheres = None
        self._damping()

    def checkChange(self):
        command = self.state_cmd.skill_cmd
        self.state_cmd.skill_cmd = FSMCommand.INVALID
        if self.fault is not None or command == FSMCommand.PASSIVE:
            return FSMStateName.PASSIVE
        if command == FSMCommand.POS_RESET:
            return FSMStateName.FIXEDPOSE
        return FSMStateName.SKILL_CLIMB
