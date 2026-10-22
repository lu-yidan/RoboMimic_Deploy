import copy
import unittest
from tools.audit_full_cycle_training import validate_evaluation_sets


class EvaluationCompletenessTest(unittest.TestCase):
    def setUp(self):
        self.manifest = {'evaluation_cases': {'nominal': {}, 'low': {}}}
        self.results = {label: [{'case': 'nominal', 'success': True},
                                {'case': 'low', 'success': False}]
                        for label in ('zero', 'after')}
        self.summary = copy.deepcopy(self.results)

    def test_complete_suite_with_task_failure_is_valid_evidence(self):
        validate_evaluation_sets(self.manifest, self.summary, self.results)

    def test_partial_duplicate_pending_and_changed_summary_are_rejected(self):
        for mode in ('partial', 'duplicate', 'pending', 'changed'):
            with self.subTest(mode=mode):
                rows, summary = copy.deepcopy(self.results), copy.deepcopy(self.summary)
                if mode == 'partial':
                    rows['after'].pop()
                    summary['after'] = rows['after']
                elif mode == 'duplicate':
                    rows['after'].append(rows['after'][0])
                    summary['after'] = rows['after']
                elif mode == 'pending':
                    summary['evaluation_pending'] = True
                else:
                    summary['after'][1]['success'] = True
                with self.assertRaises(ValueError):
                    validate_evaluation_sets(self.manifest, summary, rows)


if __name__ == '__main__':
    unittest.main()
