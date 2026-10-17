import unittest

import numpy as np

from inverse_kinematics import solve_two_link_ik
from joint_limits import JointLimits, filter_ik_solutions


class TestJointLimits(unittest.TestCase):
    def test_accepts_angles_inside_limits(self):
        limits = JointLimits([-1.0, -0.5], [1.0, 0.5])

        self.assertTrue(limits.contains([0.2, -0.1]))
        self.assertEqual(limits.violations([0.2, -0.1]), ())

    def test_accepts_angles_on_limit_boundaries(self):
        limits = JointLimits([-1.0, -0.5], [1.0, 0.5])

        self.assertTrue(limits.contains([-1.0, 0.5]))
        self.assertEqual(limits.violations([-1.0, 0.5]), ())

    def test_reports_each_joint_outside_limits(self):
        limits = JointLimits([-1.0, -0.5], [1.0, 0.5])

        self.assertEqual(limits.violations([1.1, -0.7]), (0, 1))
        self.assertFalse(limits.contains([1.1, -0.7]))

    def test_tolerance_can_absorb_small_numerical_error(self):
        limits = JointLimits([-1.0], [1.0])

        self.assertTrue(limits.contains([1.0 + 1e-10], tolerance=1e-9))
        self.assertFalse(limits.contains([1.0 + 1e-5], tolerance=1e-9))

    def test_rejects_invalid_limit_definitions(self):
        invalid_pairs = [
            ([], []),
            ([-1.0], [1.0, 2.0]),
            ([1.0], [-1.0]),
            ([np.nan], [1.0]),
            ([-1.0], [np.inf]),
        ]
        for lower, upper in invalid_pairs:
            with self.subTest(lower=lower, upper=upper):
                with self.assertRaises(ValueError):
                    JointLimits(lower, upper)

    def test_rejects_invalid_angle_vector_and_tolerance(self):
        limits = JointLimits([-1.0, -1.0], [1.0, 1.0])

        for angles in ([0.0], [0.0, np.nan], [0.0, np.inf]):
            with self.subTest(angles=angles):
                with self.assertRaises(ValueError):
                    limits.contains(angles)

        with self.assertRaises(ValueError):
            limits.contains([0.0, 0.0], tolerance=-1.0)

    def test_keeps_valid_solution_and_reports_rejected_solution(self):
        ik_result = solve_two_link_ik([1.0, 1.0], [1.0, 1.0])
        limits = JointLimits(
            [np.deg2rad(-10), np.deg2rad(80)],
            [np.deg2rad(10), np.deg2rad(100)],
        )

        filtered = filter_ik_solutions(ik_result.solutions, limits)

        self.assertEqual(len(filtered.valid_solutions), 1)
        self.assertEqual(len(filtered.rejected_solutions), 1)
        np.testing.assert_allclose(
            filtered.valid_solutions[0],
            [0.0, np.pi / 2],
            atol=1e-9,
        )
        self.assertEqual(filtered.rejected_solutions[0].violating_joint_indices, (0, 1))

    def test_reports_when_geometry_is_reachable_but_all_solutions_violate_limits(self):
        ik_result = solve_two_link_ik([1.0, 1.0], [1.0, 1.0])
        limits = JointLimits([2.0, 2.0], [2.5, 2.5])

        filtered = filter_ik_solutions(ik_result.solutions, limits)

        self.assertTrue(ik_result.reachable)
        self.assertEqual(filtered.valid_solutions, ())
        self.assertEqual(len(filtered.rejected_solutions), 2)

    def test_filter_supports_any_number_of_joints(self):
        limits = JointLimits([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])

        filtered = filter_ik_solutions(
            [[0.1, 0.2, 0.3], [0.1, 1.2, 0.3]],
            limits,
        )

        self.assertEqual(len(filtered.valid_solutions), 1)
        self.assertEqual(filtered.rejected_solutions[0].violating_joint_indices, (1,))

    def test_boundary_target_can_be_geometrically_reachable_but_rejected_by_limits(self):
        ik_result = solve_two_link_ik([2.0, 0.0], [1.0, 1.0])
        limits = JointLimits([1.0, 1.0], [2.0, 2.0])

        filtered = filter_ik_solutions(ik_result.solutions, limits)

        self.assertTrue(ik_result.reachable)
        self.assertTrue(ik_result.singular)
        self.assertEqual(filtered.valid_solutions, ())
        self.assertEqual(len(filtered.rejected_solutions), 1)

    def test_filter_validates_tolerance_even_when_there_are_no_solutions(self):
        limits = JointLimits([-1.0, -1.0], [1.0, 1.0])

        with self.assertRaises(ValueError):
            filter_ik_solutions([], limits, tolerance=-1.0)

    def test_unreachable_target_has_no_ik_candidates_to_filter(self):
        ik_result = solve_two_link_ik([3.0, 0.0], [1.0, 1.0])
        limits = JointLimits([-1.0, -1.0], [1.0, 1.0])

        filtered = filter_ik_solutions(ik_result.solutions, limits)

        self.assertFalse(ik_result.reachable)
        self.assertEqual(ik_result.solutions, ())
        self.assertEqual(filtered.valid_solutions, ())
        self.assertEqual(filtered.rejected_solutions, ())


if __name__ == "__main__":
    unittest.main()
