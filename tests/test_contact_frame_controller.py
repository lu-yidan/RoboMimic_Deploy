import unittest
import numpy as np
from training.contact_frame_controller import ContactFrameController
from training.load_transfer_residual_env import LoadTransferResidualEnv
from training.rung_residual_env import ROOT


class ContactFrameTests(unittest.TestCase):
    def test_allocated_forces_respect_actual_contact_frames_and_native_torques(self):
        task=LoadTransferResidualEnv(ROOT/'outputs/clearance_back20_leg14_grid_v1',['nominal'])
        try:
            task.reset(options={'case':'nominal'});e=task.env;m=e.model;d=e.data
            q0=d.qpos.copy();v0=d.qvel.copy();limits=e.limits.copy()
            q,v,active=task.reference(d.time);controller=ContactFrameController(e,torque_weight=10000.)
            torque,info=controller.command(q,v,active);forces=np.asarray(info['force_plan']).reshape(-1,3)
            index=0
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
                    frame=sign*c.frame.reshape(3,3);local=frame@forces[index];index+=1
                    self.assertGreaterEqual(local[0],-1e-3);self.assertLessEqual(local[0],350.+1e-3)
                    mu=min(.7,float(c.friction[0]),float(c.friction[1]))
                    for angle in np.arange(8)*np.pi/4+np.pi/8:
                        self.assertLessEqual(np.cos(angle)*local[1]+np.sin(angle)*local[2]-mu*np.cos(np.pi/8)*local[0],1e-3)
            self.assertEqual(index,len(forces))
            self.assertTrue(np.all(torque>=limits[:,0]-1e-5) and np.all(torque<=limits[:,1]+1e-5))
            np.testing.assert_array_equal(e.limits,limits);np.testing.assert_array_equal(d.qpos,q0);np.testing.assert_array_equal(d.qvel,v0)
        finally:task.close()


if __name__=='__main__':unittest.main()
