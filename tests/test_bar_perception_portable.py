"""Regression challenges for visible-only proposal contract, using saved RGB-D."""
from pathlib import Path
import json
import unittest
import numpy as np
from perception.orange_bar import CameraCalibration, project
from perception.visible_targets import select_visible_grasp_targets


class VisibleTargetTests(unittest.TestCase):
    source = Path(__file__).resolve().parent / "fixtures/bar_perception"

    def estimate(self, name):
        with np.load(self.source / name / "measurement.npz", allow_pickle=False) as data:
            measurement = {k: data[k] for k in data.files}
        calibration = CameraCalibration(**json.loads((self.source / name / "calibration.json").read_text()))
        result = select_visible_grasp_targets(measurement["rgb"], measurement["depth"],
                                              calibration, measurement["camera_position"],
                                              measurement["camera_rotation"])
        return result, measurement, calibration

    def test_frozen_predictions(self):
        for folder in sorted(self.source.iterdir()):
            with self.subTest(case=folder.name):
                result, _, _ = self.estimate(folder.name)
                expected = json.loads((folder / "expected_prediction.json").read_text())
                self.assertEqual(result, expected)

    def test_proposal_never_grants_identity_or_load(self):
        result, _, _ = self.estimate("clean_initial")
        self.assertGreaterEqual(len(result["targets"]), 2)
        self.assertFalse(result["load_permission"])
        for target in result["targets"]:
            self.assertFalse(target["load_permission"])
            self.assertFalse(target["target_identity_confirmed"])

    def test_targets_are_inside_measured_interval_with_margin(self):
        result, _, _ = self.estimate("high_bar_plus80")
        for target in result["targets"]:
            point = np.asarray(target["target_world_m"])
            ends = np.asarray(target["visible_endpoints_world_m"])
            np.testing.assert_allclose(point, ends.mean(axis=0), atol=1e-12)
            self.assertTrue(np.all(np.linalg.norm(ends - point, axis=1) >= .025))
            self.assertIn("visible", target["endpoint_status"])

    def test_plane_distractor_rejected_without_dropping_real_rods(self):
        result, _, _ = self.estimate("orange_planar_distractor")
        diagnostics = result["candidate_diagnostics"]
        plane_candidates = [d for d in diagnostics if d.get("plane_rmse_m", 1) < 1e-8]
        self.assertEqual(len(plane_candidates), 1)
        plane = plane_candidates[0]
        self.assertFalse(plane["cylinder_model_preferred"])
        self.assertFalse(any(t["candidate_index"] == plane["candidate_index"] for t in result["targets"]))
        self.assertGreaterEqual(len(result["targets"]), 2)

    def test_occluded_pixels_do_not_receive_proposed_points(self):
        result, measurement, calibration = self.estimate("partial_bar_occlusion")
        self.assertGreater(len(result["targets"]), 0)
        for target in result["targets"]:
            pixels, _ = project([target["target_world_m"]], calibration,
                                 measurement["camera_position"], measurement["camera_rotation"])
            x, y = pixels[0]
            self.assertFalse(140 <= x < 200 and 150 <= y < 220)

    def test_heavy_occlusion_abstains(self):
        result, _, _ = self.estimate("heavy_occlusion")
        self.assertEqual(result["targets"], [])


if __name__ == "__main__":
    unittest.main()
