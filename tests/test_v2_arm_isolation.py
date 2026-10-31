"""Experimental runs must not enter the adopted baseline's aggregate or viewer."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from tools import summarize_v2_multicycle as summary
from deploy_mujoco.replay_verified_climb import load_recording

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'outputs/v2_swing_eight_suite_20261029_v2/training_12'
SUITE=ROOT/'outputs/v2_swing_all24_suite_20261029_v1'


class ArmIsolationTests(unittest.TestCase):
    def test_valid_baseline_member_accepts_but_experimental_roles_reject(self):
        protocol=summary.read(SUITE/'protocol.json')
        parent=summary.read(Path(protocol['candidate'])/'manifest.json')
        manifest=summary.read(RUN/'manifest.json')
        summary.verify_suite_member(RUN,'training_12',8,protocol,parent)
        for key in ('contact_admission','timed_loading','scene_compiler'):
            changed=deepcopy(manifest);changed[key]={'experimental':True}
            with patch.object(summary,'read',return_value=changed):
                with self.assertRaises(AssertionError):summary.verify_suite_member(RUN,'training_12',8,protocol,parent)

    def test_baseline_viewer_rejects_before_requiring_experimental_assets(self):
        temporary_root=ROOT/'outputs/test_scratch';temporary_root.mkdir(exist_ok=True)
        import json
        for key in ('contact_admission','timed_loading','scene_compiler'):
            with TemporaryDirectory(dir=temporary_root) as temporary:
                path=Path(temporary).resolve();assert path.is_relative_to(temporary_root.resolve())
                (path/'manifest.json').write_text(json.dumps({key:{'experimental':True}}),encoding='utf8')
                with self.assertRaisesRegex(ValueError,'baseline replay entry'):load_recording(path)

    def test_baseline_telemetry_rejects_experimental_runs_before_launch(self):
        from tools import replay_v2_physics_telemetry as replay
        import json
        temporary_root=ROOT/'outputs/test_scratch';temporary_root.mkdir(exist_ok=True)
        for key in ('contact_admission','timed_loading','scene_compiler'):
            with TemporaryDirectory(dir=temporary_root) as temporary:
                path=Path(temporary).resolve();assert path.is_relative_to(temporary_root.resolve())
                (path/'manifest.json').write_text(json.dumps({key:True}),encoding='utf8')
                with patch.object(replay.sys,'argv',['replay','--run',str(path),'--output',str(path/'new')]), \
                     patch.object(replay.evaluator,'main') as launch:
                    with self.assertRaisesRegex(ValueError,'baseline telemetry entry'):replay.main()
                    launch.assert_not_called()
                self.assertFalse((path/'new').exists())


if __name__=='__main__':unittest.main()
