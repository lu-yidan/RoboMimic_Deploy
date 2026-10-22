import unittest
from collections import deque
from pathlib import Path
import hashlib
import mujoco
import numpy as np
from training.climb_robustness import apply_rung_offsets,ReleaseInterlock
from training.wall_support_env import WallSupportEnv
from training.rung_residual_env import RungResidualEnv
from training.climb_cycle import CycleTorqueController
from training.release_feasibility import allocate_before_release

ROOT=Path(__file__).resolve().parents[1]


class RungRobustnessTests(unittest.TestCase):
    @unittest.skipUnless((ROOT/'outputs/rung_guard_v2/row05_x_+10mm/failure_state.npz').exists(),'Recorded release failure unavailable')
    def test_infeasible_release_can_retain_original_supports_without_state_edits(self):
        e=WallSupportEnv(ROOT/'outputs/wall_support_bundle_pd15_v2');e.reset()
        apply_rung_offsets(e,{'rung_05':[.01,0,0]})
        with np.load(ROOT/'outputs/rung_guard_v2/row05_x_+10mm/failure_state.npz') as z:
            e.data.qpos[:]=z['qpos'];e.data.qvel[:]=z['qvel'];q=e.data.qpos.copy();v=e.data.qvel.copy()
            controller=CycleTorqueController(e,torque_weight=10000.)
            torque,info,rejected=allocate_before_release(controller,z['qref'],z['vref'],z['active'],z['qref'],np.ones(4,bool))
        self.assertIsNotNone(rejected);self.assertLess(info['torque_ratio'],1.)
        np.testing.assert_array_equal(e.data.qpos,q);np.testing.assert_array_equal(e.data.qvel,v)
        self.assertFalse(e.data.qfrc_applied.any());self.assertFalse(e.data.xfrc_applied.any());e.close()

    def test_contact_confirmation_rejects_wrong_rod_low_load_and_slip(self):
        e=WallSupportEnv(ROOT/'outputs/wall_support_bundle_pd15_v2');e.reset()
        g=ReleaseInterlock(e)
        contact=dict(group='right_hook',external_geom='rung_04',force_on_robot_world_n=[0,0,30])
        e.contact_metrics=lambda:dict(contacts=[contact])
        e.data.qvel[:]=0.;g.sample(0.)
        self.assertTrue(g.history[-1][1][3])
        contact['external_geom']='rung_03';g.sample(.005)
        self.assertFalse(g.history[-1][1][3])
        contact['external_geom']='rung_04';contact['force_on_robot_world_n']=[0,0,9];g.sample(.01)
        self.assertFalse(g.history[-1][1][3])
        contact['force_on_robot_world_n']=[0,0,30];e.data.qvel[0]=.1;g.sample(.015)
        self.assertFalse(g.history[-1][1][3]);e.close()

    def test_release_needs_continuous_verified_remaining_supports(self):
        g=ReleaseInterlock(None)
        active=np.array([1,1,0,1],dtype=bool)
        g.history=deque([(0.,[True]*4),(.25,[True]*4)])
        self.assertTrue(g.allow(.25,active))
        g=ReleaseInterlock(None)
        g.history=deque([(0.,[True]*4),(.1,[True,True,True,False]),(.25,[True]*4)])
        self.assertFalse(g.allow(.25,active))
        np.testing.assert_array_equal(g.previous_active,[True]*4)
        g.history=deque([(.24,[True]*4),(.25,[True]*4)])
        self.assertFalse(g.allow(.25,active))

    def test_scene_variant_preserves_start_and_frozen_bundle(self):
        e=WallSupportEnv(ROOT/'outputs/wall_support_bundle_pd15_v2');e.reset()
        path=e.bundle/'model.xml';before_hash=hashlib.sha256(path.read_bytes()).hexdigest()
        old=e.model.geom_pos.copy();q=e.data.qpos.copy()
        variant=apply_rung_offsets(e,{'rung_05':[.005,0.,-.01]})
        self.assertEqual(len(variant['changed_geoms']),5)
        np.testing.assert_array_equal(e.model.geom('rung_04').pos,old[e.model.geom('rung_04').id])
        np.testing.assert_allclose(e.model.geom('rung_05').pos-old[e.model.geom('rung_05').id],[.005,0,-.01])
        np.testing.assert_array_equal(q,e.data.qpos)
        spec=mujoco.MjSpec.from_file(str(path))
        for geom in spec.geoms:
            if geom.name=='rung_05' or geom.name.startswith('rung_05_'):
                geom.fromto=np.asarray(geom.fromto)+[.005,0,-.01,.005,0,-.01]
        rebuilt=spec.compile()
        np.testing.assert_array_equal(e.model.bvh_aabb,rebuilt.bvh_aabb)
        self.assertEqual(before_hash,hashlib.sha256(path.read_bytes()).hexdigest())
        with self.assertRaises(ValueError):apply_rung_offsets(e,{'rung_04':[.005,0,0]})
        e.close()

    def test_actor_does_not_receive_true_offset_and_reset_does_not_accumulate(self):
        e=RungResidualEnv()
        obs,_=e.reset(seed=1,options={'offset':[.005,0,0]})
        q=e.env.data.qpos.copy();geom=e.env.model.geom_pos.copy()
        obs2,_=e.reset(seed=2,options={'offset':[-.005,0,0]})
        np.testing.assert_array_equal(obs,obs2)
        self.assertEqual(obs.shape,(77,))
        e.reset(seed=1,options={'offset':[.005,0,0]})
        np.testing.assert_array_equal(e.env.model.geom_pos,geom)
        np.testing.assert_array_equal(e.env.data.qpos,q)
        for _ in range(3):obs,reward,terminated,truncated,info=e.step(np.zeros(7))
        self.assertTrue(np.isfinite(obs).all());self.assertFalse(terminated or truncated)
        self.assertFalse(e.env.data.qfrc_applied.any());self.assertFalse(e.env.data.xfrc_applied.any())
        with self.assertRaises(ValueError):e.step(np.full(7,np.nan))
        e.close()


if __name__=='__main__':unittest.main()
