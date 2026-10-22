import copy
import unittest
from tools.compare_matched_potential_training import MATCHED_FIELDS, validate_pair


class MatchedComparisonTest(unittest.TestCase):
    def pair(self):
        m = {field: 'same' for field in MATCHED_FIELDS}
        m.update(config={'gamma': .9995}, shaping_gamma=.9995, parent_steps=90112,
                 additional_steps_requested=69632, shaping_weight=0.)
        n = copy.deepcopy(m)
        n['shaping_weight'] = 4.
        s = dict(parent_steps=90112, additional_steps=69632, actual_steps=159744)
        return [m, n], [s, copy.deepcopy(s)]

    def test_complete_matched_pair(self):
        validate_pair(*self.pair())

    def test_rejects_shortened_budget_even_if_both_equal(self):
        m, s = self.pair()
        for row in s:
            row.update(additional_steps=1024, actual_steps=91136)
        with self.assertRaisesRegex(ValueError, 'Incomplete requested budget'):
            validate_pair(m, s)

    def test_rejects_changed_scenes_seed_and_pending_evaluation(self):
        for field in ('seed', 'training_cases', 'evaluation_cases', 'sources', 'parent_checkpoint_sha256'):
            m, s = self.pair()
            m[1][field] = 'changed'
            with self.assertRaisesRegex(ValueError, 'Unmatched field'):
                validate_pair(m, s)
        m, s = self.pair()
        s[1]['evaluation_pending'] = True
        with self.assertRaisesRegex(ValueError, 'Unfinished'):
            validate_pair(m, s)


if __name__ == '__main__':
    unittest.main()
