"""A profile may lower deployment effort limits, never raise shared limits."""
import copy
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import yaml
from tools.validation.validate_smp_sim import build_policy

ROOT = Path(__file__).resolve().parents[1]

class ProfileTorqueLimitsTest(unittest.TestCase):
    def test_model_only_limits_match_training(self):
        cfg = yaml.safe_load((ROOT / 'policy/smp_recovery/config/smp_recovery.yaml').read_text())
        policy, _, _ = build_policy('v34_balanced4_hard19000')
        np.testing.assert_allclose(policy.tau_limit, np.asarray(cfg['tau_limit']) * .9)
        np.testing.assert_allclose(policy.log_metadata['effective_tau_limit'], policy.tau_limit)

    def test_existing_profile_keeps_shared_limits(self):
        cfg = yaml.safe_load((ROOT / 'policy/smp_recovery/config/smp_recovery.yaml').read_text())
        policy, _, _ = build_policy('a6_no_alpha_9999')
        np.testing.assert_allclose(policy.tau_limit, cfg['tau_limit'])

    def test_rejects_increased_or_invalid_limits(self):
        cfg = yaml.safe_load((ROOT / 'policy/smp_recovery/config/smp_recovery.yaml').read_text())
        for limits in [[1000.] * 29, [float('nan')] * 29, [0.] * 29, [1.] * 28]:
            trial = copy.deepcopy(cfg)
            trial['model_profiles']['v34_balanced4_hard19000']['tau_limit'] = limits
            with patch('policy.smp_recovery.SmpRecovery.yaml.safe_load', return_value=trial):
                with self.assertRaises(ValueError):
                    build_policy('v34_balanced4_hard19000')

if __name__ == '__main__':
    unittest.main()
