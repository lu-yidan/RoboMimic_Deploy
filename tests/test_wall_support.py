import unittest
from pathlib import Path

import mujoco
import numpy as np

from policy.climb.Climb import OnnxActor
from training.wall_support_env import WallSupportEnv,DEFAULT_WALL_BUNDLE

ROOT=Path(__file__).resolve().parents[1]


class ContactForceTests(unittest.TestCase):
    def test_world_force_sum_matches_known_supported_weight(self):
        model=mujoco.MjModel.from_xml_string('''<mujoco>
          <option timestep=".002"/>
          <worldbody><geom name="floor" type="plane" size="1 1 .1"/>
            <body name="pelvis" pos="0 0 .1"><freejoint/>
              <geom name="test_foot" type="box" size=".1 .1 .1" mass="1"/>
            </body></worldbody></mujoco>''')
        data=mujoco.MjData(model)
        for _ in range(1000):mujoco.mj_step(model,data)
        env=WallSupportEnv.__new__(WallSupportEnv)
        env.model=model;env.data=data;env.robot_bodies={model.body('pelvis').id}
        env.contact_groups={model.body('pelvis').id:'left_foot'};env.robot_weight_n=9.81
        result=env.contact_metrics()
        self.assertGreater(len(result['contacts']),1)
        self.assertAlmostEqual(result['left_foot_vertical_n'],9.81,places=4)
        self.assertAlmostEqual(result['total_contact_vertical_n'],9.81,places=4)


@unittest.skipUnless((DEFAULT_WALL_BUNDLE/'manifest.json').exists(),'Prepare the rung wall task first')
class WallContractTests(unittest.TestCase):
    def test_target_is_right_grasp_point_on_continuous_rung(self):
        env=WallSupportEnv();obs,_=env.reset()
        rotation=env.data.xmat[env.pelvis].reshape(3,3)
        target=rotation@obs[93:96]+env.data.xpos[env.pelvis]
        np.testing.assert_allclose(target,[.6+np.tan(np.deg2rad(5))*1.55-.35,-.2,1.55],atol=1e-7)
        np.testing.assert_allclose(rotation@obs[96:99],[0,1,0],atol=1e-7)
        self.assertEqual(env.model.neq,0)
        env.close()

    def test_old_policy_is_rejected_for_new_geometry_contract(self):
        path=ROOT/'outputs/climb_ppo_smoke_v1/export'
        if not (path/'metadata.json').exists():self.skipTest('Old export unavailable')
        env=WallSupportEnv()
        with self.assertRaisesRegex(ValueError,'contracts differ'):
            OnnxActor(path/'policy.onnx',path/'metadata.json',env.contract)
        env.close()


if __name__=='__main__':unittest.main()
