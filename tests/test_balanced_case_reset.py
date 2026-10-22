import unittest
import gymnasium as gym
from training.balanced_case_reset import BalancedCaseReset


class CaseRecorder(gym.Env):
    cases = {'a': {}, 'b': {}, 'c': {}}

    def reset(self, *, seed=None, options=None):
        return None, {'case': options['case']}


class BalancedCaseTest(unittest.TestCase):
    def test_every_block_covers_all_cases(self):
        env = BalancedCaseReset(CaseRecorder())
        sequence = [env.reset(seed=7 if i == 0 else None)[1]['case'] for i in range(12)]
        for start in range(0, 12, 3):
            self.assertEqual(set(sequence[start:start+3]), {'a', 'b', 'c'})
        repeated = [env.reset(seed=7 if i == 0 else None)[1]['case'] for i in range(12)]
        self.assertEqual(sequence, repeated)

    def test_explicit_case_preserves_schedule(self):
        env = BalancedCaseReset(CaseRecorder())
        env.reset(seed=4)
        pending = env.pending_cases.copy()
        self.assertEqual(env.reset(options={'case': 'c'})[1]['case'], 'c')
        self.assertEqual(env.pending_cases, pending)


if __name__ == '__main__':
    unittest.main()
