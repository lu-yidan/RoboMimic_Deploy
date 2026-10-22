import unittest
from tools.audit_closure_baseline_search import select_candidate


class SelectionTest(unittest.TestCase):
    def test_smaller_error_without_success_gain_retains_zero(self):
        self.assertEqual(select_candidate([
            dict(residual_rad=.003, accepted=2, worst_endpoint_error_m=.001)], 2), 0.)

    def test_success_then_worst_error_then_predeclared_sign(self):
        rows = [dict(residual_rad=-.003, accepted=3, worst_endpoint_error_m=.009),
                dict(residual_rad=.003, accepted=3, worst_endpoint_error_m=.008)]
        self.assertEqual(select_candidate(rows, 2), .003)
        rows[1]['worst_endpoint_error_m'] = .009
        self.assertEqual(select_candidate(rows, 2), -.003)
        rows[1]['accepted'] = 2
        rows[1]['worst_endpoint_error_m'] = .001
        self.assertEqual(select_candidate(rows, 2), -.003)


if __name__ == '__main__':
    unittest.main()
