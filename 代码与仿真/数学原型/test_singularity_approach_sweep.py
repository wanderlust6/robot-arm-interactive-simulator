import math
import unittest

import numpy as np

from planar_arm import PlanarArm
from run_singularity_approach_sweep import (
    CASES, reference_path, run_sequence, wrapped_delta,
)


class SingularityApproachSweepTests(unittest.TestCase):
    def test_paths_include_expected_singular_configuration(self):
        for case in CASES:
            q = reference_path(case, 61)
            self.assertEqual(q.shape, (61, 2))
            self.assertTrue(np.all(np.isfinite(q)))
            expected = 0.0 if case.startswith("outer") else math.pi
            self.assertAlmostEqual(float(q[30, 1]), expected, places=10)

    def test_reference_paths_generate_finite_fk_targets(self):
        arm = PlanarArm((0.6, 0.4))
        for case in CASES:
            for q in reference_path(case, 13):
                target = arm.forward_kinematics(q)[-1]
                self.assertEqual(target.shape, (2,))
                self.assertTrue(np.all(np.isfinite(target)))

    def test_curved_approach_is_not_just_a_rotated_linear_path(self):
        straight = reference_path("outer_q1_increasing", 61)
        curved = reference_path("outer_q1_curved_approach", 61)
        self.assertFalse(np.allclose(straight[:, 0], curved[:, 0]))
        self.assertAlmostEqual(float(curved[0, 0]), math.radians(-65), places=10)
        self.assertAlmostEqual(float(curved[-1, 0]), math.radians(-35), places=10)

    def test_wrapped_delta_handles_angle_periodicity(self):
        delta = wrapped_delta(np.radians(179), np.radians(-179))
        self.assertAlmostEqual(abs(float(delta)), math.radians(2), places=10)

    def test_sequence_summary_contains_quality_and_risk_metrics(self):
        arm = PlanarArm((0.6, 0.4))
        rows, summary = run_sequence(arm, "outer_q1_increasing", 9, 1, "current_dls")
        self.assertEqual(len(rows), 9)
        self.assertEqual(summary["samples"], 9)
        for key in ("success_rate", "max_actual_error_m", "min_sigma",
                    "max_joint_step_rad", "max_second_difference_rad",
                    "branch_mismatch_points", "p95_solve_time_ms"):
            self.assertIn(key, summary)
            self.assertTrue(math.isfinite(summary[key]))
        self.assertIn("joint_step_rad", rows[0])
        self.assertIn("reference_joint_error_rad", rows[0])


if __name__ == "__main__":
    unittest.main()
