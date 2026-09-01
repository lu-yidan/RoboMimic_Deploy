from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import yaml

from common.ctrlcomp import PolicyOutput, StateAndCmd
from policy.smp_recovery.SmpRecovery import SmpRecovery


ROOT = Path(__file__).parents[1]
MODEL_DIR = ROOT / "policy/smp_recovery/model"
CONFIG = ROOT / "policy/smp_recovery/config/smp_recovery.yaml"
MANIFEST = MODEL_DIR / "smp_a11_grounded_safety_gate1000_93d.manifest.json"


class _FakeSession:
    def __init__(self, *_args, **_kwargs):
        self.action = np.zeros((1, 29), dtype=np.float32)

    def get_inputs(self):
        return [SimpleNamespace(name="obs", shape=[1, 93])]

    def get_outputs(self):
        return [SimpleNamespace(name="actions", shape=[1, 29])]

    def run(self, _outputs, inputs):
        obs = inputs["obs"]
        if obs.shape != (1, 93):
            raise ValueError(f"unexpected observation shape {obs.shape}")
        return [self.action.copy()]


class SmpA11DeploymentTest(unittest.TestCase):
    def _policy(self):
        state = StateAndCmd(29)
        output = PolicyOutput(29)
        with patch("policy.smp_recovery.SmpRecovery.ort.InferenceSession", _FakeSession):
            policy = SmpRecovery(state, output)
        return policy, state, output

    def test_model_and_manifest_are_sha_bound(self):
        cfg = yaml.safe_load(CONFIG.read_text())
        manifest = json.loads(MANIFEST.read_text())
        model = MODEL_DIR / cfg["model_path"]
        digest = hashlib.sha256(model.read_bytes()).hexdigest()
        self.assertEqual(cfg["observation_dim"], 93)
        self.assertEqual(digest, cfg["model_sha256"])
        self.assertEqual(digest, manifest["onnx_sha256"])
        self.assertEqual(manifest["input"]["shape"], [1, 93])
        self.assertTrue(manifest["input"]["normalizer_embedded"])

    def test_observation_order_is_exact_93d_contract(self):
        policy, state, _ = self._policy()
        state.root_lin_vel_b[:] = [91, 92, 93]
        state.root_ang_vel_b[:] = [1, 2, 3]
        state.gravity_ori[:] = [4, 5, 6]
        state.q[:] = policy.default_q + np.arange(29, dtype=np.float32)
        state.dq[:] = np.arange(29, dtype=np.float32) + 30
        policy._last_action[:] = np.arange(29, dtype=np.float32) + 60
        obs = policy._build_obs()
        self.assertEqual(obs.shape, (93,))
        np.testing.assert_allclose(obs[:6], [1, 2, 3, 4, 5, 6])
        np.testing.assert_allclose(obs[6:35], np.arange(29))
        np.testing.assert_allclose(obs[35:64], np.arange(29) + 30)
        np.testing.assert_allclose(obs[64:93], np.arange(29) + 60)
        self.assertFalse(np.isin([91, 92, 93], obs).any())

    def test_nonfinite_observation_and_action_fail_closed(self):
        policy, state, _ = self._policy()
        state.dq[0] = np.nan
        with self.assertRaisesRegex(FloatingPointError, "observation"):
            policy._build_obs()
        state.dq[0] = 0
        policy._session.action[0, 0] = np.inf
        with self.assertRaisesRegex(FloatingPointError, "action"):
            policy.run()


if __name__ == "__main__":
    unittest.main()
