import unittest
from pathlib import Path
import numpy as np
from tools.repeat_climb_cycle import repeat_reference
from deploy_mujoco.run_climb_cycle import assess_cycle_end
from training.climb_cycle import CycleTorqueController,solve_force_qp_fallback
from training.wall_support_env import WallSupportEnv

ROOT=Path(__file__).resolve().parents[1]


class ContinuationTests(unittest.TestCase):
    @unittest.skipUnless((ROOT/'outputs/compact_two_cycle_failure_diagnostic_v1/failed_force_qp.npz').exists(),'Recorded failure fixture unavailable')
    def test_contact_corner_fallback_keeps_all_original_constraints(self):
        with np.load(ROOT/'outputs/compact_two_cycle_failure_diagnostic_v1/failed_force_qp.npz') as p:
            H=p['P']+p['P'].T-np.diag(np.diag(p['P']))
            f=solve_force_qp_fallback(H,p['objective'],p['A'],p['lower'],p['upper'])
            self.assertLess(np.max(p['lower']-p['A']@f),1e-6)
            self.assertLess(np.max(p['A']@f-p['upper']),1e-6)

    def test_fallback_does_not_relax_infeasible_bounds(self):
        with self.assertRaises(RuntimeError):
            solve_force_qp_fallback(np.eye(1),np.zeros(1),np.ones((2,1)),np.array([1.,-np.inf]),np.array([np.inf,0.]))

    @unittest.skipUnless((ROOT/'outputs/compact_cycle_plan_v13/plan.npz').exists(),'Build the compact reference first')
    def test_repeat_requires_closure_and_preserves_continuous_boundary(self):
        with np.load(ROOT/'outputs/compact_cycle_plan_v13/plan.npz') as source:
            repeated=repeat_reference(source,2);end=source['time'][-1]
            i=int(np.searchsorted(repeated['time'],end))
            self.assertTrue(np.all(np.diff(repeated['time'])>0))
            # A new cycle retains the original initial settling segment, with no
            # additional jump beyond the source's measured closure tolerance.
            np.testing.assert_allclose(repeated['qpos'][i+1,3:]-repeated['qpos'][i,3:],
                                       source['qpos'][1,3:]-source['qpos'][0,3:],atol=1e-4)
            np.testing.assert_allclose(repeated['qpos'][i+1,:3],repeated['qpos'][i,:3],atol=1e-8)
            self.assertAlmostEqual(repeated['qpos'][-1,2]-repeated['qpos'][0,2],.5)
            bad=dict(source);bad['qpos']=bad['qpos'].copy();bad['qpos'][-1,7]+=.1
            with self.assertRaises(ValueError):repeat_reference(bad,2)

    def test_cycle_window_rejects_incomplete_support_wrong_rung_and_partial_log(self):
        groups=['left_hook','right_hook','left_foot','right_foot'];rows=[];contacts=[]
        for t in np.linspace(18.02,20,100):
            rows.append(dict(time_s=float(t),base_rise_m=.5,periodic_joint_error_rad=.01,
                             **{g+'_vertical_n':50. for g in groups}))
            contacts.append(dict(time_s=float(t),contacts=[dict(group=g,external_geom='rung_06' if 'hook' in g else 'rung_02',force_on_robot_world_n=[0,0,50]) for g in groups]))
        self.assertTrue(assess_cycle_end(rows,contacts,20.,2)['passed'])
        self.assertFalse(assess_cycle_end(rows[-1:],contacts[-1:],20.,2)['passed'])
        rows[20]['left_foot_vertical_n']=0.
        self.assertFalse(assess_cycle_end(rows,contacts,20.,2)['passed'])
        rows[20]['left_foot_vertical_n']=50.
        contacts[20]['contacts'][0]['external_geom']='rung_05'
        self.assertFalse(assess_cycle_end(rows,contacts,20.,2)['passed'])

    @unittest.skipUnless((ROOT/'outputs/compact_cycle_plan_v13/plan.npz').exists(),'Build the compact reference first')
    def test_weighted_allocator_reduces_normalized_torque_without_state_edits(self):
        e=WallSupportEnv(ROOT/'outputs/wall_support_bundle_pd15_v2');e.reset()
        with np.load(ROOT/'outputs/compact_cycle_plan_v13/plan.npz') as p:e.data.qpos[:]=p['qpos'][0]
        q=e.data.qpos.copy();v=e.data.qvel.copy();active=np.ones(4,bool)
        old,_=CycleTorqueController(e).command(q,v,active)
        c=CycleTorqueController(e,torque_weight=10000.)
        for _ in range(2):
            new,info=c.command(q,v,active)
            self.assertLess(np.linalg.norm(new/e.limits[:,1]),np.linalg.norm(old/e.limits[:,1]))
            self.assertLess(info['base_equilibrium_residual'],1e-3)
            self.assertTrue(np.all(new>=e.limits[:,0]));self.assertTrue(np.all(new<=e.limits[:,1]))
        np.testing.assert_array_equal(e.data.qpos,q);np.testing.assert_array_equal(e.data.qvel,v)
        self.assertFalse(e.data.qfrc_applied.any());self.assertFalse(e.data.xfrc_applied.any());e.close()


if __name__=='__main__':unittest.main()
