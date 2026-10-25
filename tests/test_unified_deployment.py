"""Offline integration: shared selections, retained models, scenes and FSM routing."""
import builtins
from contextlib import ExitStack
import copy
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import MagicMock, patch

import mujoco
import numpy as np
import onnxruntime as ort
from hydra import compose, initialize_config_dir
import yaml

from common.deployment_config import load_deployment_config, load_scalebfm_config
from common.ctrlcomp import StateAndCmd, PolicyOutput
from common.utils import FSMCommand, FSMStateName, get_gravity_orientation
from deploy_real.config import Config
from deploy_mujoco.scene_builder import load_model, apply_initial_state
from policy.amp.Amp import Amp
from policy.smp_recovery.SmpRecovery import SmpRecovery
from FSM.FSM import FSM

ROOT = Path(__file__).resolve().parents[1]
SMP = ('path_a6_9999', 'a6_no_alpha_9999', 'v34_balanced4_hard19000')
AMP = ('amp_0309_1', 'stratified_guidance_60000', 'stratified_single_60000')


class UnifiedDeploymentTest(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        original = ort.SessionOptions
        def options():
            opts = original()
            opts.intra_op_num_threads = opts.inter_op_num_threads = 1
            return opts
        self.stack.enter_context(patch.object(ort, 'SessionOptions', options))
        # No ambient legacy override may change this integration test.
        self.stack.enter_context(patch.dict(os.environ, {'SMP_RECOVERY_PROFILE': ''}))

    def select(self, cfg):
        folder = self.stack.enter_context(tempfile.TemporaryDirectory())
        path = Path(folder) / 'deploy.yaml'
        path.write_text(yaml.safe_dump(cfg))
        self.stack.enter_context(patch.dict(os.environ, {'LYD_DEPLOY_CONFIG': str(path)}))

    def test_shared_config_selects_all_amp_and_smp_models(self):
        base = load_deployment_config()
        for amp, smp in zip(AMP, SMP):
            cfg = copy.deepcopy(base)
            cfg['amp']['profile'], cfg['smp']['profile'] = amp, smp
            self.select(cfg)
            state, output = StateAndCmd(29), PolicyOutput(29)
            self.assertEqual(Amp(state, output).profile, amp)
            self.assertEqual(SmpRecovery(state, output).profile, smp)
        models = set(p.stem for p in (ROOT/'policy/smp_recovery/model').glob('*.onnx'))
        self.assertEqual(models, {'smp_'+p+'_93d' for p in SMP})

    def test_same_scalebfm_switch_reaches_real_and_simulation(self):
        for enabled in (False, True):
            cfg = load_deployment_config()
            cfg['scalebfm']['enabled'] = enabled
            self.select(cfg)
            self.assertEqual(load_scalebfm_config().enabled, enabled)
            self.assertEqual(Config().scalebfm.enabled, enabled)

    def test_invalid_selectors_and_non_boolean_flag_fail(self):
        cfg = load_deployment_config()
        cfg['scalebfm']['enabled'] = 'false'
        self.select(cfg)
        with self.assertRaises(ValueError): load_deployment_config()
        state, output = StateAndCmd(29), PolicyOutput(29)
        with self.assertRaises(ValueError): Amp(state, output, profile='missing')
        with self.assertRaises(ValueError): SmpRecovery(state, output, profile='missing')

    def make_fsm(self, enabled):
        cfg = load_scalebfm_config()
        cfg.enabled = enabled
        state, output = StateAndCmd(29), PolicyOutput(29)
        state.gravity_ori[:] = [0, 0, -1]
        return FSM(state, output, cfg, [], enable_freekick=False), state, output

    def test_disabled_scalebfm_never_imports_external_runtime_and_keeps_control(self):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.startswith(('scalebridge', 'policy.scalebfm.ScaleBFM')):
                raise AssertionError('Disabled ScaleBFM imported its runtime')
            return original(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=guarded):
            fsm, state, output = self.make_fsm(False)
            self.assertIsNone(fsm.scalebfm_policy)
            output.kds[:] = 0
            state.skill_cmd = FSMCommand.CMD_SCALEBFM
            fsm.run()
            self.assertEqual(fsm.cur_policy.name, FSMStateName.PASSIVE)
            self.assertTrue(np.any(output.kds > 0))
            for cmd, expected in ((FSMCommand.POS_RESET, FSMStateName.FIXEDPOSE),
                                  (FSMCommand.CMD_AMP, FSMStateName.SKILL_AMP),
                                  (FSMCommand.CMD_SMP_RECOVERY, FSMStateName.SKILL_SMP_RECOVERY),
                                  (FSMCommand.PASSIVE, FSMStateName.PASSIVE)):
                state.skill_cmd = cmd
                fsm.run(); fsm.run()
                self.assertEqual(fsm.cur_policy.name, expected)
                self.assertTrue(np.isfinite(output.actions).all())

    def test_enabled_scalebfm_routing_and_recovery_preemption(self):
        # Only the unavailable external GPU policy is substituted; FSM is real.
        module = ModuleType('policy.scalebfm.ScaleBFM')
        tracker = MagicMock()
        tracker.name = FSMStateName.SKILL_SCALEBFM
        tracker.name_str = 'ScaleBFM'
        tracker.checkChange.return_value = FSMStateName.SKILL_SCALEBFM
        module.ScaleBFM = MagicMock(return_value=tracker)
        with patch.dict(sys.modules, {'policy.scalebfm.ScaleBFM': module}):
            fsm, state, output = self.make_fsm(True)
            module.ScaleBFM.assert_called_once()
            for cmd, expected in ((FSMCommand.POS_RESET, FSMStateName.FIXEDPOSE),
                                  (FSMCommand.CMD_SCALEBFM, FSMStateName.SKILL_SCALEBFM),
                                  (FSMCommand.CMD_SMP_RECOVERY, FSMStateName.SKILL_SMP_RECOVERY),
                                  (FSMCommand.PASSIVE, FSMStateName.PASSIVE)):
                state.skill_cmd = cmd
                fsm.run(); fsm.run()
                self.assertEqual(fsm.cur_policy.name, expected)
            tracker.enter.assert_called_once()
            tracker.run.assert_called()
            tracker.exit.assert_called_once()

    def test_retained_smp_runs_in_table_and_box_scenes(self):
        # 1s smoke checks finite control and per-model effort bounds, not recovery success.
        for scene in ('mujoco', 'mujoco_table', 'mujoco_coffin'):
            with initialize_config_dir(config_dir=str(ROOT/'deploy_mujoco/config'), version_base=None):
                cfg = compose(config_name=scene)
            model = load_model(ROOT/cfg.xml_path, cfg.get('scene', {}))
            model.opt.timestep = .002
            self.assertEqual(model.nu, 29)
            for profile in SMP:
                data = mujoco.MjData(model)
                apply_initial_state(model, data, cfg.get('initial_state', {}))
                mujoco.mj_forward(model, data)
                state, output = StateAndCmd(29), PolicyOutput(29)
                policy = SmpRecovery(state, output, profile=profile)
                for step in range(50):
                    state.q = data.qpos[7:36].astype(np.float32).copy()
                    state.dq = data.qvel[6:35].astype(np.float32).copy()
                    state.root_ang_vel_b = data.qvel[3:6].astype(np.float32).copy()
                    state.gravity_ori = get_gravity_orientation(data.qpos[3:7]).astype(np.float32)
                    if step == 0: policy.enter()
                    policy.run()
                    commanded = (output.actions-state.q)*output.kps-output.kds*state.dq
                    self.assertTrue(np.all(np.abs(commanded) <= policy.tau_limit + .002))
                    for _ in range(10):
                        data.ctrl[:] = np.clip((output.actions-data.qpos[7:36])*output.kps-output.kds*data.qvel[6:35], -policy.tau_limit, policy.tau_limit)
                        mujoco.mj_step(model, data)
                    self.assertTrue(np.isfinite(data.qpos).all())
                    self.assertTrue(np.isfinite(data.qvel).all())


if __name__ == '__main__': unittest.main()
