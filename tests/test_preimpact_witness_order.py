"""Distance witness direction must agree with a known rigid translation."""
import unittest
import mujoco
import numpy as np
from tools.analyze_preimpact_kinematics import ordered_distance


class WitnessOrderTests(unittest.TestCase):
    def test_box_cylinder_orders_and_approach_direction(self):
        m=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <geom name="rod" type="cylinder" pos=".1 0 0" size=".02 .1"/>
          <body><freejoint/><geom name="moving" type="box" size=".02 .02 .02"/></body>
        </worldbody></mujoco>''')
        d=mujoco.MjData(m);d.qvel[0]=.01;mujoco.mj_forward(m,d)
        a=m.geom('moving').id;b=m.geom('rod').id
        gap,ab=ordered_distance(m,d,a,b,.2);reverse,ba=ordered_distance(m,d,b,a,.2)
        self.assertAlmostEqual(gap,.06,places=5);self.assertAlmostEqual(gap,reverse,places=10)
        np.testing.assert_allclose(ab[:3],ba[3:],atol=1e-10)
        np.testing.assert_allclose(ab[3:],ba[:3],atol=1e-10)
        direction=ab[3:]-ab[:3];direction/=np.linalg.norm(direction)
        J=np.zeros((3,m.nv));mujoco.mj_jac(m,d,J,None,ab[:3],int(m.geom_bodyid[a]))
        speed=float(direction@(J@d.qvel));self.assertAlmostEqual(speed,.01,places=8)
        initial=d.qpos.copy();values=[]
        for dt in (-1e-6,1e-6):
            d.qpos[:]=initial;mujoco.mj_integratePos(m,d.qpos,d.qvel,dt);mujoco.mj_forward(m,d)
            values.append(ordered_distance(m,d,a,b,.2)[0])
        self.assertAlmostEqual(-(values[1]-values[0])/2e-6,speed,places=6)


if __name__=='__main__':unittest.main()
