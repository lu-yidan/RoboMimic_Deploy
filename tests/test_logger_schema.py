from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from common.logger import Logger


class LoggerSchemaTest(unittest.TestCase):
    def test_schema_two_records_hardware_evidence_signals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = SimpleNamespace(
                q=np.full(29, 0.1, dtype=np.float32),
                dq=np.full(29, 0.2, dtype=np.float32),
                tau_est=np.arange(29, dtype=np.float32),
                pelvis_pos_w=np.zeros(3, dtype=np.float32),
                pelvis_quat_w=np.array([1, 0, 0, 0], dtype=np.float32),
                ball_pos_b=np.zeros(3, dtype=np.float32),
                ball_pos_w=np.zeros(3, dtype=np.float32),
                ball_valid=False,
                target_pos_b=np.zeros(3, dtype=np.float32),
                target_valid=False,
                vel_cmd=np.zeros(3, dtype=np.float32),
                gravity_ori=np.array([0, 0, -1], dtype=np.float32),
                ang_vel=np.array([0.3, 0.2, 0.1], dtype=np.float32),
                skill_cmd=7,
            )
            policy = SimpleNamespace(
                actions=np.full(29, 0.4, dtype=np.float32),
                kps=np.full(29, 10.0, dtype=np.float32),
                kds=np.full(29, 2.0, dtype=np.float32),
                debug_target_pos_b=np.zeros(3, dtype=np.float32),
                debug_target_source=np.zeros(1, dtype=np.float32),
            )
            logger = Logger(temporary, "schema2")
            logger.log(1, 0.02, state, policy)
            logger.close()
            binary = next(Path(temporary).glob("*_schema2.bin"))
            loaded = Logger.load(str(binary))
            self.assertEqual(loaded["_meta"]["logger_schema_version"], 2)
            np.testing.assert_allclose(loaded["tau_est"][0], state.tau_est)
            np.testing.assert_allclose(loaded["tau_cmd_est"][0], 2.6)
            np.testing.assert_allclose(loaded["gravity_ori"][0], [0, 0, -1])
            self.assertEqual(float(loaded["skill_cmd"][0]), 7.0)

    def test_legacy_metadata_controls_legacy_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary) / "legacy.bin"
            metadata = Path(temporary) / "legacy.json"
            record_dim = sum(size for _, size in Logger.LEGACY_FIELDS)
            np.arange(record_dim, dtype=np.float32).tofile(binary)
            metadata.write_text(
                json.dumps(
                    {
                        "record_dim": record_dim,
                        "fields": dict(Logger.LEGACY_FIELDS),
                    }
                )
            )
            loaded = Logger.load(str(binary))
            self.assertEqual(loaded["q"].shape, (1, 29))
            self.assertNotIn("tau_est", loaded)

    def test_inconsistent_metadata_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary) / "invalid.bin"
            metadata = Path(temporary) / "invalid.json"
            np.zeros(2, dtype=np.float32).tofile(binary)
            metadata.write_text(json.dumps({"record_dim": 2, "fields": {"q": 1}}))
            with self.assertRaisesRegex(ValueError, "field layout"):
                Logger.load(str(binary))


if __name__ == "__main__":
    unittest.main()
