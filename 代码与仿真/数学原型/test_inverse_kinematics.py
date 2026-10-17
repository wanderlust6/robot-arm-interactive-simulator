import unittest

import numpy as np

from inverse_kinematics import solve_two_link_ik
from planar_arm import PlanarArm


class TestTwoLinkInverseKinematics(unittest.TestCase):
    def setUp(self):
        self.arm = PlanarArm([1.0, 1.0])

    def test_target_one_one_has_two_valid_solutions(self):
        result = solve_two_link_ik([1.0, 1.0], [1.0, 1.0])

        self.assertTrue(result.reachable)
        self.assertFalse(result.singular)
        self.assertEqual(len(result.solutions), 2)
        for angles in result.solutions:
            np.testing.assert_allclose(
                self.arm.forward_kinematics(angles)[-1],
                [1.0, 1.0],
                atol=1e-9,
            )

    def test_returns_elbow_up_and_elbow_down_branches(self):
        result = solve_two_link_ik([1.0, 1.0], [1.0, 1.0])

        second_angles = sorted(round(solution[1], 8) for solution in result.solutions)
        self.assertEqual(second_angles, [-round(np.pi / 2, 8), round(np.pi / 2, 8)])

    def test_unreachable_target_outside_outer_radius_returns_no_solutions(self):
        result = solve_two_link_ik([2.1, 0.0], [1.0, 1.0])

        self.assertFalse(result.reachable)
        self.assertEqual(result.solutions, ())
        self.assertIn("不可达", result.message)

    def test_target_on_outer_boundary_has_one_singular_solution(self):
        result = solve_two_link_ik([2.0, 0.0], [1.0, 1.0])

        self.assertTrue(result.reachable)
        self.assertTrue(result.singular)
        self.assertEqual(len(result.solutions), 1)
        np.testing.assert_allclose(
            self.arm.forward_kinematics(result.solutions[0])[-1],
            [2.0, 0.0],
            atol=1e-9,
        )

    def test_target_inside_inner_radius_is_unreachable(self):
        result = solve_two_link_ik([0.1, 0.0], [2.0, 1.0])

        self.assertFalse(result.reachable)
        self.assertEqual(result.solutions, ())

    def test_near_base_target_is_not_misclassified_as_infinite_solutions(self):
        result = solve_two_link_ik([1e-5, 0.0], [1.0, 1.0])

        self.assertTrue(result.reachable)
        self.assertFalse(result.singular)
        self.assertEqual(len(result.solutions), 2)

    def test_equal_links_targeted_at_base_reports_infinite_solutions(self):
        result = solve_two_link_ik([0.0, 0.0], [1.0, 1.0])

        self.assertTrue(result.reachable)
        self.assertTrue(result.singular)
        self.assertEqual(result.solutions, ())
        self.assertIn("无穷多组", result.message)

    def test_target_just_outside_boundary_within_tolerance_is_clamped(self):
        result = solve_two_link_ik([2.0 + 1e-10, 0.0], [1.0, 1.0], tolerance=1e-9)

        self.assertTrue(result.reachable)
        self.assertTrue(result.singular)
        self.assertEqual(len(result.solutions), 1)

    def test_rejects_invalid_link_lengths(self):
        for lengths in ([1.0], [1.0, 0.0], [1.0, -1.0], [1.0, np.inf]):
            with self.subTest(lengths=lengths):
                with self.assertRaises(ValueError):
                    solve_two_link_ik([1.0, 1.0], lengths)

    def test_rejects_invalid_target_and_tolerance(self):
        for target in ([1.0], [1.0, np.nan], [np.inf, 0.0]):
            with self.subTest(target=target):
                with self.assertRaises(ValueError):
                    solve_two_link_ik(target, [1.0, 1.0])

        with self.assertRaises(ValueError):
            solve_two_link_ik([1.0, 1.0], [1.0, 1.0], tolerance=-1.0)


if __name__ == "__main__":
    unittest.main()
