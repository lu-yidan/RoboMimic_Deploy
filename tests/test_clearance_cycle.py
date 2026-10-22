import hashlib
import json
from pathlib import Path
import unittest
import mujoco
import numpy as np
from training.climb_cycle import LimbIK as OriginalIK, rung
from training.clearance_cycle import LimbIK
from training.wall_support_env import WallSupportEnv
from training.climb_robustness import apply_rung_offsets
from tools.audit_rung_initial_states import inspect_initial

ROOT=Path(__file__).resolve().parents[1]


class ClearanceCycleTests(unittest.TestCase):
    def test_initial_overlap_is_not_misclassified_as_a_control_failure(self):
        e=WallSupportEnv(ROOT/'outputs/wall_support_bundle_pd15_v2');e.reset()
        apply_rung_offsets(e,{'rung_01':[-.01,0,0]})
        live_q=e.data.qpos.copy();live_v=e.data.qvel.copy()
        with np.load(ROOT/'outputs/compact_cycle_plan_v13/plan.npz') as p:
            old=inspect_initial(e,p['qpos'][0])
        self.assertFalse(old['passed'])
        self.assertEqual(len(old['unexpected_overlaps']),2)
        self.assertTrue(all('knee' in c['bodies'][1] for c in old['unexpected_overlaps']))
        with np.load(ROOT/'outputs/clearance_back20_leg14_plan_v1/plan.npz') as p:
            new=inspect_initial(e,p['qpos'][0])
        self.assertTrue(new['passed'])
        np.testing.assert_array_equal(e.data.qpos,live_q)
        np.testing.assert_array_equal(e.data.qvel,live_v)
        e.close()

    def test_default_contact_solver_matches_original_and_does_not_edit_live_state(self):
        e=WallSupportEnv(ROOT/'outputs/wall_support_bundle_pd15_v2');e.reset()
        q=e.data.qpos.copy();v=e.data.qvel.copy()
        old,eo=OriginalIK(e).contact_leg(q,'rf',rung(.3)+[0,-.2,0],0.)
        new,en=LimbIK(e).contact_leg(q,'rf',rung(.3)+[0,-.2,0],0.)
        np.testing.assert_allclose(old,new,atol=1e-12,rtol=0.)
        self.assertAlmostEqual(eo,en,places=12)
        np.testing.assert_array_equal(e.data.qpos,q);np.testing.assert_array_equal(e.data.qvel,v)
        with self.assertRaises(ValueError):LimbIK(e,swing_clearance=np.nan)
        e.close()

    @unittest.skipUnless((ROOT/'outputs/rung_residual_ppo_smoke_v2/manifest.json').exists(),'Historical checkpoint unavailable')
    def test_existing_residual_checkpoint_environment_remains_compatible(self):
        manifest=json.loads((ROOT/'outputs/rung_residual_ppo_smoke_v2/manifest.json').read_text())
        for rel,expected in manifest['sources'].items():
            if rel.startswith('training/'):
                self.assertEqual(hashlib.sha256((ROOT/rel).read_bytes()).hexdigest(),expected,rel)

    @unittest.skipUnless((ROOT/'outputs/clearance_swing8_plan_v3/plan.npz').exists(),'Candidate reference unavailable')
    def test_declared_swing_clearance_and_cycle_closure(self):
        e=WallSupportEnv(ROOT/'outputs/wall_support_bundle_pd15_v2');e.reset();ik=LimbIK(e)
        with np.load(ROOT/'outputs/clearance_swing8_plan_v3/plan.npz') as p:
            np.testing.assert_allclose(p['qpos'][-1,e.qadr],p['qpos'][0,e.qadr],atol=1e-4)
            np.testing.assert_allclose(p['qpos'][-1,:3]-p['qpos'][0,:3],[np.tan(np.deg2rad(5))*.25,0,.25],atol=1e-8)
            minimum=.05
            for key in ['lf','rf']:
                geoms=[g for g in range(e.model.ngeom) if e.model.geom_contype[g] and e.model.geom_bodyid[g]==ik.bodies[key]]
                for index in np.flatnonzero(p['phase']==key+'_swing'):
                    e.data.qpos[:]=p['qpos'][index];mujoco.mj_kinematics(e.model,e.data)
                    for g in geoms:
                        for row in range(5):
                            minimum=min(minimum,mujoco.mj_geomDistance(e.model,e.data,g,e.model.geom(f'rung_{row:02d}').id,.05,None))
            self.assertGreaterEqual(minimum,.0079)
        e.close()


if __name__=='__main__':unittest.main()
