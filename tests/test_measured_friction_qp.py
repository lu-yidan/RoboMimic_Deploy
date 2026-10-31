import unittest
import mujoco
import numpy as np
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from training.closure_measured_friction_qp import solve_probe


class MeasuredFrictionQPTests(unittest.TestCase):
    def test_saved_high_rod_state_feasibility_and_no_state_change(self):
        b=FullCycleResidualEnv(plan='outputs/support_retarget_plan_v1/plan.npz',cases={'high':{'rung_01':[0.,0.,.015]}})
        try:
            b.reset(options={'case':'high'});e=b.env
            with np.load('outputs/constrained_closure_screen_v1/gain0_drop1_foot_z_plus15/trajectory.npz') as z:
                i=int(np.argmin(abs(z['time_s']-110.)))
                e.data.qpos[:]=z['qpos'][i];e.data.qvel[:]=z['qvel'][i];e.data.time=z['time_s'][i]
            mujoco.mj_forward(e.model,e.data)
            q0=e.data.qpos.copy();v0=e.data.qvel.copy();t=e.data.time
            q,v,active,_=b.reference(t)
            r=solve_probe(e,e.data,q,v,[b.contact.ik.bodies[k] for k in ('lf','rf','lh','rh')])
            self.assertTrue(r['passed']);self.assertLess(r['max_constraint_residual'],1e-3)
            self.assertTrue(np.all(r['torque']>=e.limits[:,0]-1e-3))
            self.assertTrue(np.all(r['torque']<=e.limits[:,1]+1e-3))
            np.testing.assert_array_equal(q0,e.data.qpos);np.testing.assert_array_equal(v0,e.data.qvel)
            self.assertEqual(t,e.data.time);self.assertFalse(e.data.qfrc_applied.any());self.assertFalse(e.data.xfrc_applied.any())
        finally:b.close()


if __name__=='__main__':unittest.main()
