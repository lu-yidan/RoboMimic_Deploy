import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from common.ctrlcomp import StateAndCmd, PolicyOutput
from common.logger import Logger
from policy.smp_recovery.SmpRecovery import SmpRecovery
from bridge.python.bridge_diagnostics_dds import BridgeDiagnostics, DiagnosticsSubscriber, apply_diagnostics

class RecoveryLoggingTest(unittest.TestCase):
    def test_diagnostics_roundtrip_and_missing_is_not_zero(self):
        state = StateAndCmd(29)
        msg = BridgeDiagnostics(tick=123, tau_est=list(range(29)), ddq=[2.]*29,
                                imu_accel=[0., 0., 9.8], motor_temperature=[35.]*58)
        decoded = BridgeDiagnostics.deserialize(msg.serialize())
        apply_diagnostics(state, decoded)
        self.assertTrue(state.tau_est_valid)
        np.testing.assert_equal(state.tau_est, np.arange(29))
        apply_diagnostics(state, None)
        self.assertFalse(state.tau_est_valid)
        self.assertTrue(np.isnan(state.tau_est).all())

    def test_diagnostics_requires_exact_tick(self):
        sub = DiagnosticsSubscriber.__new__(DiagnosticsSubscriber)
        sub.cache = {}
        sub.reader = SimpleNamespace(take=lambda N: [BridgeDiagnostics(tick=40, tau_est=[1.]*29)])
        self.assertIsNone(sub.matching(41))
        self.assertEqual(sub.matching(40).tick, 40)

    def test_exact_actor_input_action_chain_and_legacy_schema_two(self):
        state, output = StateAndCmd(29), PolicyOutput(29)
        with patch.dict(os.environ, {'SMP_RECOVERY_PROFILE':'e1_prone_9999'}):
            policy = SmpRecovery(state, output)
        state.q[:] = policy.default_q
        state.dq[:] = np.linspace(-8, 8, 29)
        state.gravity_ori[:] = [0,0,-1]
        policy.enter()
        with tempfile.TemporaryDirectory() as tmp:
            logger = Logger(tmp, 'test')
            for i in range(12):
                expected_obs = policy._build_obs().copy()
                policy.run()
                output.executed_fsm_state = 18
                d = output.recovery_debug
                np.testing.assert_equal(d['smp_obs'], expected_obs)
                raw = policy._session.run(None, {policy._input_name:expected_obs[None]})[0][0]
                np.testing.assert_allclose(d['smp_raw_action'], raw)
                # Independently reproduce the pre-existing command pipeline.
                action = np.clip(raw, -policy.clip_actions, policy.clip_actions)
                target = policy.default_q + policy.action_scale*action
                alpha = min((i+1)/policy.warmup_steps, 1)
                target = (1-alpha)*policy._entry_q + alpha*target
                target = np.clip(target, state.q+(policy.kds*state.dq-policy.tau_limit)/policy.kps,
                                  state.q+(policy.kds*state.dq+policy.tau_limit)/policy.kps)
                np.testing.assert_allclose(output.actions, target, atol=1e-6)
                logger.log(i, i*.02, state, output)
            logger.close()
            path = next(Path(tmp).glob('*.bin')); log = Logger.load(str(path))
            self.assertEqual(log['smp_obs'].shape, (12,93))
            self.assertEqual(log['_meta']['smp_recovery']['profile'], 'e1_prone_9999')
            self.assertTrue(np.isnan(log['tau_est']).all())
            self.assertTrue((log['executed_fsm_state']==18).all())
            self.assertEqual(len(next(Path(tmp).glob('*.events.jsonl')).read_text().splitlines()),1)
            # Real schema-2 log keeps its original layout, even after upgrading logger.
            old = Path(tmp)/'v2.bin'; fields=Logger.FIELDS[:len(Logger.FIELDS)-len(Logger.EXTRA_FIELDS)]
            np.zeros(sum(n for _,n in fields),np.float32).tofile(old)
            old.with_suffix('.json').write_text(json.dumps({'record_dim':237,'fields':dict(fields)}))
            self.assertEqual(Logger.load(str(old))['tau_est'].shape,(1,29))

if __name__ == '__main__': unittest.main()
