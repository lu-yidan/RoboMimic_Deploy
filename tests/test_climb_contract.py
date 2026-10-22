import ast
import copy
import json
from pathlib import Path
import re
import unittest
import xml.etree.ElementTree as ET
import numpy as np
from policy.climb.contract import ClimbContract, JOINT_NAMES

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'policy/climb/config/climb.json'


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG.read_text())
        self.c = ClimbContract.from_json(CONFIG)

    def inputs(self):
        return dict(q=self.c.default_q.copy(), dq=np.zeros(29), gravity_body=[0,0,-1],
                    gyro_body=[1,2,3], previous_action=np.zeros(29), target_pos_body=[.1,.2,.3],
                    target_axis_body=[0,0,1], target_valid=True)

    def test_model_order_limits_and_existing_gain_provenance(self):
        model = ET.parse(ROOT/'g1_description/g1_29dof_rev_1_0.xml')
        joints = [j for j in model.iter('joint') if j.get('name') and j.get('type') != 'free']
        self.assertEqual(tuple(j.get('name') for j in joints), JOINT_NAMES)
        ranges = np.array([[float(x) for x in j.get('range').split()] for j in joints], np.float32)
        np.testing.assert_array_equal(self.c.joint_lower, ranges[:,0])
        np.testing.assert_array_equal(self.c.joint_upper, ranges[:,1])
        source = (ROOT/'policy/beyondmimic_mj/config/beyondmimic_mj.yaml').read_text()
        for target, origin in [('default_q','default_joint_pos'),('action_scale','action_scale'),('kps','kps'),('kds','kds')]:
            values = ast.literal_eval(re.search(r'^'+origin+r':\s*(\[[\s\S]*?\])', source, re.M).group(1))
            np.testing.assert_array_equal(getattr(self.c,target),np.array(values,np.float32))

    def test_observation_layout_and_invalid_target_canonicalization(self):
        args=self.inputs(); args['q'] += .01; args['dq'][:]=2; args['previous_action'][:]=.5
        obs=self.c.observation(**args)
        self.assertEqual((obs.shape,obs.dtype),((100,),np.dtype('float32')))
        np.testing.assert_array_equal(obs[:6],[0,0,-1,1,2,3])
        np.testing.assert_allclose(obs[6:35],.01,atol=1e-7)
        np.testing.assert_array_equal(obs[35:64],2*self.c.dq_scale)
        np.testing.assert_array_equal(obs[64:93],.5)
        np.testing.assert_allclose(obs[93:100],[.1,.2,.3,0,0,1,1])
        args['target_valid']=False; args['target_axis_body']=[0,0,0]
        np.testing.assert_array_equal(self.c.observation(**args)[93:],np.zeros(7))

    def test_action_clip_and_model_joint_limit(self):
        target, clipped=self.c.decode_action(np.full(29,10.))
        np.testing.assert_array_equal(clipped,np.ones(29))
        self.assertTrue(np.all(target<=self.c.joint_upper))
        self.assertEqual(target[5],self.c.joint_upper[5])
        zero,_=self.c.decode_action(np.zeros(29))
        np.testing.assert_array_equal(zero,self.c.default_q)
        self.assertFalse(self.c.kps.flags.writeable)

    def test_bad_measurements_actions_and_flags_rejected(self):
        for key,value in [('q',np.zeros((1,29))),('dq',np.full(29,np.nan)),('gravity_body',[0,0,0]),
                          ('target_axis_body',[0,0,2]),('target_valid',1),('previous_action',np.full(29,1.1))]:
            args=self.inputs(); args[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError): self.c.observation(**args)
        for value in (np.zeros(28),np.full(29,np.inf)):
            with self.assertRaises(ValueError): self.c.decode_action(value)

    def test_fingerprint_and_invalid_config(self):
        reordered=dict(reversed(list(self.config.items())))
        self.assertEqual(self.c.fingerprint,ClimbContract(reordered).fingerprint)
        changed=copy.deepcopy(self.config); changed['dq_scale']=.5
        self.assertNotEqual(self.c.fingerprint,ClimbContract(changed).fingerprint)
        changed=copy.deepcopy(self.config); changed['joint_names'][0]='wrong'
        with self.assertRaises(ValueError): ClimbContract(changed)
        self.assertEqual(len(self.c.fingerprint),64)


if __name__=='__main__': unittest.main()
