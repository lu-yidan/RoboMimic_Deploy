import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
import mujoco
import numpy as np
from training.climb_cycle import CycleTorqueController, LimbIK
from training.wall_support_env import WallSupportEnv

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'outputs/wall_support_bundle_pd15_v2'
PLAN=ROOT/'outputs/cycle_contact_plan_v17/plan.npz'


@unittest.skipUnless(PLAN.exists(),'Build the cycle reference first')
class CycleTests(unittest.TestCase):
    def setUp(self):
        self.e=WallSupportEnv(BUNDLE);self.e.reset()
        self.p=np.load(PLAN)

    def tearDown(self):
        self.e.close();self.p.close()

    def test_planning_does_not_change_live_state(self):
        e=self.e;ik=LimbIK(e);q=e.data.qpos.copy();v=e.data.qvel.copy();ctrl=e.data.ctrl.copy()
        candidate=self.p['qpos'][0].copy()
        ik.d.qpos[:]=candidate;mujoco.mj_forward(e.model,ik.d)
        target=ik.d.xpos[ik.bodies['rh']].copy()+[0,0,.02]
        _,error=ik.pose(candidate,{'rh':target})
        self.assertLess(error['rh'],.001)
        np.testing.assert_array_equal(e.data.qpos,q);np.testing.assert_array_equal(e.data.qvel,v)
        np.testing.assert_array_equal(e.data.ctrl,ctrl);self.assertEqual(e.data.time,0.)

    def test_all_four_effectors_end_one_rung_higher(self):
        e=self.e;ik=LimbIK(e);positions=[]
        for q in self.p['qpos'][[0,-1]]:
            ik.d.qpos[:]=q;mujoco.mj_forward(e.model,ik.d)
            positions.append(np.array([ik.d.xpos[b].copy() for b in ik.bodies.values()]))
        shift=np.array([.25*np.tan(np.deg2rad(5)),0,.25])
        np.testing.assert_allclose(positions[1]-positions[0],np.tile(shift,(4,1)),atol=.0003)
        np.testing.assert_allclose(self.p['qpos'][-1,:3]-self.p['qpos'][0,:3],shift,atol=1e-8)
        np.testing.assert_allclose(self.p['qpos'][-1,7:],self.p['qpos'][0,7:],atol=1e-4)

    def test_torque_controller_has_no_root_actuation_or_state_edits(self):
        e=self.e;e.data.qpos[:]=self.p['qpos'][0];mujoco.mj_forward(e.model,e.data)
        q=e.data.qpos.copy();v=e.data.qvel.copy();controller=CycleTorqueController(e)
        for active in [np.ones(4,bool),np.array([1,1,1,0],bool),np.array([1,0,1,1],bool)]:
            tau,info=controller.command(q,np.zeros(e.model.nv),active)
            self.assertTrue(np.all(tau>=e.limits[:,0]-1e-3))
            self.assertTrue(np.all(tau<=e.limits[:,1]+1e-3))
            self.assertLess(info['base_equilibrium_residual'],1e-3)
            self.assertEqual(len(info['force_plan']),int(active.sum())*6)
        np.testing.assert_array_equal(e.data.qpos,q);np.testing.assert_array_equal(e.data.qvel,v)
        self.assertFalse(e.data.qfrc_applied.any());self.assertFalse(e.data.xfrc_applied.any())
        self.assertEqual(e.model.neq,0);self.assertEqual(e.model.nu,29)

    def test_iteration_retry_preserves_bounded_physical_solution(self):
        e=self.e;e.data.qpos[:]=self.p['qpos'][0];mujoco.mj_forward(e.model,e.data)
        controller=CycleTorqueController(e);active=np.ones(4,bool);v=np.zeros(e.model.nv)
        expected,_=controller.command(e.data.qpos.copy(),v,active)
        solver=controller.cache[tuple(active)];solve=solver.solve;calls=[]
        def delayed_solve(**kwargs):
            calls.append(1)
            if len(calls)==1:return SimpleNamespace(info=SimpleNamespace(status_val=7))
            return solve(**kwargs)
        with patch.object(solver,'solve',side_effect=delayed_solve):
            actual,info=controller.command(e.data.qpos.copy(),v,active)
        self.assertEqual(info['solver_retries'],1)
        np.testing.assert_allclose(actual,expected,atol=1e-3)
        self.assertLess(info['base_equilibrium_residual'],1e-3)
        self.assertTrue(np.all(actual>=e.limits[:,0]-1e-3))
        self.assertTrue(np.all(actual<=e.limits[:,1]+1e-3))
        self.assertEqual(solver.settings.max_iter,5000)


if __name__=='__main__':unittest.main()
