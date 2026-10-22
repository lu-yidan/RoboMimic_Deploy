import unittest
import numpy as np
from training.robust_contact_frame_controller import RobustContactFrameController
from training.robust_tracking_controller import RobustTrackingController
from training.load_transfer_residual_env import LoadTransferResidualEnv
from training.rung_residual_env import ROOT


class ConservativeDynamicsTests(unittest.TestCase):
    def test_allocated_forces_fit_true_friction_cones_despite_normal_bias(self):
        task=LoadTransferResidualEnv(ROOT/'outputs/clearance_back20_leg14_grid_v1',['nominal'])
        try:
            task.reset(options={'case':'nominal'});e=task.env;m=e.model;d=e.data
            q,v,active=task.reference(d.time);q0=d.qpos.copy();v0=d.qvel.copy();limits=e.limits.copy()
            for bias in (-3.,0.,3.):
                controller=RobustContactFrameController(e,torque_weight=10000.,normal_bias_y_deg=bias)
                torque,info=controller.command(q,v,active);forces=np.asarray(info['force_plan']).reshape(-1,3);i=0
                for enabled,key in zip(active,('lf','rf','lh','rh')):
                    if not enabled:continue
                    body=controller.ik.bodies[key]
                    for c in d.contact:
                        a,b=int(c.geom1),int(c.geom2);ba,bb=int(m.geom_bodyid[a]),int(m.geom_bodyid[b])
                        if ba==body and bb==0:external=b;sign=-1.
                        elif bb==body and ba==0:external=a;sign=1.
                        else:continue
                        name=m.geom(external).name
                        if not(name.startswith('rung_') and len(name)==7 and name[-2:].isdigit()):continue
                        actual=sign*c.frame.reshape(3,3)@forces[i];i+=1
                        self.assertGreaterEqual(actual[0],-1e-3)
                        mu=min(.7,float(c.friction[0]),float(c.friction[1]))
                        for angle in np.arange(8)*np.pi/4+np.pi/8:
                            self.assertLessEqual(np.cos(angle)*actual[1]+np.sin(angle)*actual[2]-mu*np.cos(np.pi/8)*actual[0],1e-3)
                self.assertEqual(i,len(forces))
                self.assertTrue(np.all(torque>=limits[:,0]-1e-6) and np.all(torque<=limits[:,1]+1e-6))
                combined=RobustTrackingController(e,torque_weight=10000.,normal_bias_y_deg=bias)
                same,details=combined.command(q,v,active)
                self.assertEqual(details['tracking_scale'],1.)
                np.testing.assert_allclose(same,torque,rtol=0,atol=1e-8)
            np.testing.assert_array_equal(e.limits,limits);np.testing.assert_array_equal(d.qpos,q0);np.testing.assert_array_equal(d.qvel,v0)
        finally:task.close()


if __name__=='__main__':unittest.main()
