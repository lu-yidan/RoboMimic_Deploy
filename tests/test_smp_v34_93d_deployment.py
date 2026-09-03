from __future__ import annotations

import hashlib
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
import onnxruntime as ort
import yaml

from common.ctrlcomp import PolicyOutput, StateAndCmd
from common.utils import get_gravity_orientation
from policy.smp_recovery.SmpRecovery import SmpRecovery


ROOT = Path(__file__).parents[1]
MODEL_DIR = ROOT / "policy/smp_recovery/model"
CONFIG = ROOT / "policy/smp_recovery/config/smp_recovery.yaml"
MANIFEST = MODEL_DIR / "smp_v34_escape_gate6000_93d.manifest.json"
PROFILE = "v34_93d_gate6000"


class SmpV34DeploymentTest(unittest.TestCase):
    def test_v34_is_the_yaml_default_profile(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SMP_RECOVERY_PROFILE", None)
            policy = SmpRecovery(StateAndCmd(29), PolicyOutput(29))
        self.assertEqual(policy.profile, PROFILE)

        with patch.dict(os.environ, {"SMP_RECOVERY_PROFILE": "a11"}):
            policy = SmpRecovery(StateAndCmd(29), PolicyOutput(29))
        self.assertEqual(policy.profile, "a11")

    def test_model_manifest_and_runtime_interface(self):
        cfg = yaml.safe_load(CONFIG.read_text())
        profile = cfg["model_profiles"][PROFILE]
        manifest = json.loads(MANIFEST.read_text())
        model = MODEL_DIR / profile["model_path"]
        digest = hashlib.sha256(model.read_bytes()).hexdigest()
        self.assertEqual(digest, profile["model_sha256"])
        self.assertEqual(digest, manifest["onnx_sha256"])
        self.assertEqual(manifest["status"], "SIMULATION_COMPARISON_CANARY_NOT_HARDWARE_APPROVED")
        self.assertEqual(manifest["input"]["shape"], [1, 93])
        self.assertTrue(manifest["input"]["normalizer_embedded"])
        self.assertFalse(manifest["runtime_safety"]["action_rate_envelope"])

        session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        self.assertEqual(session.get_inputs()[0].shape, [1, 93])
        self.assertEqual(session.get_outputs()[0].shape, [1, 29])
        action = session.run(None, {"obs": np.zeros((1, 93), dtype=np.float32)})[0]
        self.assertEqual(action.shape, (1, 29))
        self.assertTrue(np.isfinite(action).all())

    def test_headless_mujoco_control_smoke(self):
        model = mujoco.MjModel.from_xml_path(str(ROOT / "g1_description/g1_liao.xml"))
        data = mujoco.MjData(model)
        model.opt.timestep = 0.002
        state = StateAndCmd(29)
        output = PolicyOutput(29)
        with patch.dict(os.environ, {"SMP_RECOVERY_PROFILE": PROFILE}):
            policy = SmpRecovery(state, output)

        data.qpos[:3] = [0.0, 0.0, 0.76]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        data.qpos[7:36] = policy.default_q
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        state.q = data.qpos[7:36].astype(np.float32).copy()
        policy.enter()
        target_q = state.q.copy()
        kps = policy.kps.copy()
        kds = policy.kds.copy()
        max_abs_tau = np.zeros(29, dtype=np.float64)
        for step in range(500):
            q = data.qpos[7:36]
            dq = data.qvel[6:35]
            tau = (target_q - q) * kps - dq * kds
            tau = np.clip(tau, -policy.tau_limit, policy.tau_limit)
            max_abs_tau = np.maximum(max_abs_tau, np.abs(tau))
            data.ctrl[:] = tau
            mujoco.mj_step(model, data)
            if step % 10 == 0:
                state.q = data.qpos[7:36].astype(np.float32).copy()
                state.dq = data.qvel[6:35].astype(np.float32).copy()
                state.root_ang_vel_b = data.qvel[3:6].astype(np.float32).copy()
                state.gravity_ori = get_gravity_orientation(data.qpos[3:7]).astype(
                    np.float32
                )
                policy.run()
                target_q = output.actions.copy()
                kps = output.kps.copy()
                kds = output.kds.copy()

        self.assertTrue(np.isfinite(data.qpos).all())
        self.assertTrue(np.isfinite(data.qvel).all())
        self.assertTrue(np.isfinite(target_q).all())
        self.assertTrue(np.less_equal(max_abs_tau, policy.tau_limit + 1.0e-6).all())


if __name__ == "__main__":
    unittest.main()
