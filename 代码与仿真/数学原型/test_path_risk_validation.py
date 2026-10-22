import unittest

from path_risk_validation import classify_point, confusion_counts, evaluate_thresholds, summarize_rows


def sample(**updates):
    row = {
        'case': 'case_a', 'method': 'current_dls', 'repeat': '1', 'sample_index': '0',
        'segment_type': 'reachable', 'seed_case': 'nominal', 'noise_std_m': '0',
        'target_x_m': '0.5', 'target_y_m': '0.2', 'target_radius_m': '0.54',
        'geometrically_reachable': 'True', 'initial_q1_rad': '0.2', 'initial_q2_rad': '0.3',
        'joint_1_rad': '0.2', 'joint_2_rad': '0.3', 'elbow_sign': '1', 'converged': 'True',
        'reported_error_m': '1e-7', 'actual_error_m': '1e-7', 'solve_time_ms': '0.1',
        'iterations': '', 'peak_damping': '', 'min_singular_value': '0.04',
        'branch_transitions_in_sequence': '0',
    }
    row.update({k: str(v) for k, v in updates.items()})
    return row


class ValidationTests(unittest.TestCase):
    def test_unreachable_and_nonconverged_are_independent_events(self):
        result = classify_point(sample(geometrically_reachable='False', converged='False'))
        self.assertTrue(result['unreachable'])
        self.assertTrue(result['solver_failure'])
        self.assertTrue(result['critical'])

    def test_threshold_confusion_counts_are_exact(self):
        counts = confusion_counts([True, True, False], [True, False, False])
        self.assertEqual(counts, {'tp': 1, 'fp': 0, 'fn': 1, 'tn': 1})

    def test_threshold_sweep_preserves_all_points_and_reports_threshold(self):
        rows = [sample(min_singular_value='0.01'), sample(sample_index='1', min_singular_value='0.03')]
        result = evaluate_thresholds(rows, [0.02, 0.04])
        self.assertEqual([x['threshold'] for x in result], [0.02, 0.04])
        self.assertEqual([x['flagged_points'] for x in result], [1, 2])

    def test_summary_separates_outcomes_from_heuristic_flags(self):
        rows = [sample(min_singular_value='0.01'),
                sample(sample_index='1', geometrically_reachable='False', converged='False'),
                sample(sample_index='2', actual_error_m='0.00001')]
        summary = summarize_rows(rows)
        self.assertEqual(summary['points'], 3)
        self.assertEqual(summary['unreachable_points'], 1)
        self.assertEqual(summary['solver_failure_points'], 1)
        self.assertEqual(summary['error_over_tolerance_points'], 1)
        self.assertEqual(summary['near_singularity_points'], 1)

    def test_invalid_numeric_values_fail_with_column_context(self):
        with self.assertRaisesRegex(ValueError, 'min_singular_value'):
            classify_point(sample(min_singular_value='not-a-number'))


if __name__ == '__main__':
    unittest.main()
