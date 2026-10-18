"""Contract tests against independent training-runner trajectory fixtures."""
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import onnxruntime as ort
import yaml
from common.ctrlcomp import StateAndCmd, PolicyOutput
from common.utils import FSMCommand,FSMStateName
from policy.amp.Amp import Amp

ROOT=Path(__file__).resolve().parents[1]


class AmpGuidanceDeploymentTest(unittest.TestCase):
    MODEL = "amp_stratified_guidance_60000"
    @classmethod
    def setUpClass(cls):
        cls.cfg=yaml.safe_load((ROOT/'policy/amp/config/amp.yaml').read_text())
        cls.manifest=json.loads((ROOT/f'policy/amp/model/{cls.MODEL}.manifest.json').read_text())
        cls.vectors=np.load(ROOT/f'policy/amp/model/{cls.MODEL}.vectors.npz')

    def make_policy(self,name):
        cfg=dict(self.cfg,model_path=name)
        state=StateAndCmd(29);output=PolicyOutput(29)
        with patch('policy.amp.Amp.yaml.safe_load',return_value=cfg):
            policy=Amp(state,output)
        return policy,state,output

    def test_model_integrity_and_unchanged_config_contract(self):
        for suffix,key in [('.onnx','onnx_sha256')]:
            self.assertEqual(hashlib.sha256((ROOT/f'policy/amp/model/{self.MODEL}{suffix}').read_bytes()).hexdigest(),self.manifest[key])
        self.assertEqual(hashlib.sha256((ROOT/f'policy/amp/model/{self.MODEL}.vectors.npz').read_bytes()).hexdigest(),self.manifest['verification']['vectors_sha256'])
        for key,value in self.manifest['compatible_config'].items():self.assertEqual(self.cfg[key],value,key)
        names=self.manifest['deployment_motor_names']
        self.assertEqual([names[i] for i in self.cfg['joint2motor_idx']],self.manifest['deployment_policy_joint_names'])

    def test_both_models_load_by_changing_only_path(self):
        for name in ['amp_0309_1.onnx',self.MODEL+'.onnx']:
            policy,state,output=self.make_policy(name)
            state.q[:]=policy.default_dof_pos;state.gravity_ori[:]=[0,0,-1]
            policy.enter();policy.run()
            self.assertTrue(np.isfinite(output.actions).all())
            self.assertEqual(policy._session.get_inputs()[0].name,'obs')
            self.assertEqual(policy._session.get_outputs()[0].name,'actions')

    def test_actual_controller_matches_training_motor_targets(self):
        policy,state,output=self.make_policy(self.MODEL+'.onnx')
        idx=policy.joint2motor_idx
        # Frames were built from physical simulator states, not exported ONNX
        # outputs. Expected motor targets came directly from the training runner.
        for obs,expected in zip(self.vectors['obs'],self.vectors['motor_targets']):
            frames=obs.reshape(4,96);current=frames[-1]
            policy._obs_buf[:]=np.concatenate((frames[:1],frames[:3])).reshape(-1)
            state.ang_vel=current[:3].copy();state.gravity_ori=current[3:6].copy()
            policy._cmd_smooth_val[:]=current[6:9]
            state.q[idx]=current[9:38]+policy.default_dof_pos[idx]
            state.dq[idx]=current[38:67]
            policy._last_action[:]=current[67:96]
            with patch.object(policy,'_update_cmd'):
                policy.run()
            np.testing.assert_allclose(policy._obs_buf,obs,atol=2e-7)
            np.testing.assert_allclose(output.actions,expected,atol=5e-5,rtol=2e-5)

    def test_history_initialization_and_roll(self):
        policy,state,_=self.make_policy(self.MODEL+'.onnx')
        state.q[:]=policy.default_dof_pos;state.gravity_ori[:]=[0,0,-1]
        policy.enter();first=policy._obs_buf.reshape(4,96).copy()
        np.testing.assert_array_equal(first,np.repeat(first[:1],4,axis=0))
        state.ang_vel[:]=[.12,.34,.56];policy._push_obs()
        np.testing.assert_array_equal(policy._obs_buf.reshape(4,96)[:3],first[1:])
        np.testing.assert_allclose(policy._obs_buf[-96:-93],[.12,.34,.56])

    def test_recovery_stays_in_amp_and_manual_exits_work(self):
        policy,state,_=self.make_policy(self.MODEL+'.onnx')
        state.gravity_ori[:]=[0,0,1]  # Upside down must not force an automatic exit.
        self.assertEqual(policy.checkChange(),FSMStateName.SKILL_AMP)
        for cmd,result in [(FSMCommand.PASSIVE,FSMStateName.PASSIVE),
                           (FSMCommand.POS_RESET,FSMStateName.FIXEDPOSE),
                           (FSMCommand.LOCO,FSMStateName.LOCOMODE)]:
            state.skill_cmd=cmd;self.assertEqual(policy.checkChange(),result)

class AmpStratifiedSingleDeploymentTest(AmpGuidanceDeploymentTest):
    MODEL = 'amp_stratified_single_60000'


if __name__=='__main__':unittest.main()
