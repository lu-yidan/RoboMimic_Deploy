import unittest
from pathlib import Path
import json
import mujoco
import numpy as np
from training.wall_support_env import WallSupportEnv
from training.climb_robustness import apply_rung_offsets
from training.tracking_relaxation import TrackingRelaxationController,scaled_reference

ROOT=Path(__file__).resolve().parents[1]


class TrackingRelaxationTests(unittest.TestCase):
    def test_unit_scale_is_bitwise_identity_and_does_not_mutate_state(self):
        e=WallSupportEnv(ROOT/'outputs/wall_support_bundle_pd15_v2');e.reset()
        try:
            q=e.data.qpos.copy();v=e.data.qvel.copy();target=q+np.arange(e.model.nq)*.001
            tq,tv=scaled_reference(e,target,v,1.)
            np.testing.assert_array_equal(tq,target);np.testing.assert_array_equal(tv,v)
            tq[:]=0
            np.testing.assert_array_equal(e.data.qpos,q)
            with self.assertRaises(ValueError):scaled_reference(e,target,v,1.1)
        finally:e.close()

    def test_failure_snapshots_relax_demand_without_relaxing_physical_bounds(self):
        for case in ['row01_z_+5mm','row01_x_-10mm']:
            with self.subTest(case=case):
                run=ROOT/'outputs/clearance_back20_leg14_grid_v1'/case
                e=WallSupportEnv(ROOT/'outputs/wall_support_bundle_pd15_v2');e.reset()
                try:
                    apply_rung_offsets(e,json.loads((run/'scene_variant.json').read_text())['offsets'])
                    f=np.load(run/'failure_state.npz');e.data.qpos[:]=f['qpos'];e.data.qvel[:]=f['qvel']
                    mujoco.mj_forward(e.model,e.data);limits=e.limits.copy()
                    torque,info=TrackingRelaxationController(e,torque_weight=10000.).command(f['qref'],f['vref'],f['active'])
                    self.assertLess(info['tracking_scale'],1.)
                    self.assertLessEqual(info['base_equilibrium_residual'],1e-3)
                    self.assertTrue(np.all(torque>=limits[:,0]-1e-5) and np.all(torque<=limits[:,1]+1e-5))
                    forces=np.asarray(info['force_plan']).reshape(-1,3)
                    self.assertTrue(np.all(forces[:,2]>=-1e-5) and np.all(forces[:,2]<=350.+1e-5))
                    for angle in np.arange(8)*np.pi/4+np.pi/8:
                        self.assertTrue(np.all(forces@np.array([np.cos(angle),np.sin(angle),-.7*np.cos(np.pi/8)])<=1e-3))
                    np.testing.assert_array_equal(e.limits,limits)
                    np.testing.assert_array_equal(e.data.qpos,f['qpos']);np.testing.assert_array_equal(e.data.qvel,f['qvel'])
                    self.assertFalse(np.any(e.data.qfrc_applied));self.assertFalse(np.any(e.data.xfrc_applied))
                finally:e.close()


if __name__=='__main__':unittest.main()
