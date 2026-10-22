import unittest
from tools.select_closure_validated_policy import choose


class SelectionTests(unittest.TestCase):
    def test_more_successes_with_regression_is_rejected(self):
        self.assertEqual(choose({'known:a', 'grid:b'}, {'penalty': {'known:a', 'grid:c', 'grid:d'}}), 'parent')

    def test_equal_success_set_does_not_upgrade(self):
        self.assertEqual(choose({'a'}, {'control': {'a'}, 'penalty': {'a'}}), 'parent')

    def test_control_wins_exact_tie(self):
        self.assertEqual(choose({'a'}, {'penalty': {'a', 'b'}, 'control': {'a', 'c'}}), 'control')

    def test_larger_valid_improvement_wins(self):
        self.assertEqual(choose({'a'}, {'penalty': {'a', 'b', 'c'}, 'control': {'a', 'b'}}), 'penalty')

    def test_ineligible_arms_absent(self):
        self.assertEqual(choose({'a'}, {}), 'parent')


if __name__ == '__main__':
    unittest.main()
