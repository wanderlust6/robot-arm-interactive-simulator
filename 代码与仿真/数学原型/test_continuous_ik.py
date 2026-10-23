import unittest

import numpy as np

from continuous_ik import select_continuous_ik_solution
from inverse_kinematics import solve_two_link_ik
from planar_arm import PlanarArm


class TestContinuousIKSelection(unittest.TestCase):
    def setUp(self):
        self.arm = PlanarArm([0.6, 0.4])

    def test_tracks_nearest_equivalent_solution_without_angle_wrap_jump(self):
        previous = np.radians([179.0, 30.0])
        candidates = [
            np.radians([-179.0, 31.0]),
            np.radians([120.0, -20.0]),
        ]

        selected = select_continuous_ik_solution(candidates, previous)

        np.testing.assert_allclose(selected, np.radians([181.0, 31.0]), atol=1e-10)

    def test_selects_nearest_analytic_branch_for_small_target_motion(self):
        q_path = np.column_stack((np.radians(np.linspace(5.0, 25.0, 41)),
                                  np.radians(np.linspace(18.0, 0.6, 41))))
        targets = [self.arm.forward_kinematics(q)[-1] for q in q_path]
        previous = None
        selected_path = []

        for target in targets:
            result = solve_two_link_ik(target, [0.6, 0.4])
            self.assertTrue(result.reachable)
            selected = select_continuous_ik_solution(result.solutions, previous)
            selected_path.append(selected)
            previous = selected

        selected_path = np.asarray(selected_path)
        steps = np.diff(selected_path, axis=0)
        self.assertLess(float(np.max(np.linalg.norm(steps, axis=1))), 0.03)
        for target, angles in zip(targets, selected_path):
            np.testing.assert_allclose(self.arm.forward_kinematics(angles)[-1], target, atol=1e-9)

    def test_empty_candidate_list_returns_none(self):
        self.assertIsNone(select_continuous_ik_solution([], np.zeros(2)))

    def test_empty_numpy_candidate_array_returns_none(self):
        self.assertIsNone(select_continuous_ik_solution(np.empty((0, 2)), np.zeros(2)))

    def test_rejects_candidates_with_wrong_shape(self):
        with self.assertRaises(ValueError):
            select_continuous_ik_solution([[0.1, 0.2, 0.3]], np.zeros(2))

    def test_rejects_non_finite_candidate_angles(self):
        with self.assertRaises(ValueError):
            select_continuous_ik_solution([[np.nan, 0.2]], np.zeros(2))

    def test_rejects_invalid_previous_angles(self):
        with self.assertRaises(ValueError):
            select_continuous_ik_solution([[0.1, 0.2]], [np.inf, 0.0])

    def test_accepts_numpy_array_as_candidate_collection(self):
        candidates = np.radians([[20.0, 30.0], [-20.0, -30.0]])
        selected = select_continuous_ik_solution(candidates, np.radians([-15.0, -25.0]))
        np.testing.assert_allclose(selected, candidates[1], atol=1e-10)

    def test_continuity_does_not_automatically_switch_elbow_branch(self):
        q_path = np.column_stack((
            np.radians(np.linspace(5.0, 25.0, 41)),
            np.radians(np.linspace(18.0, -18.0, 41)),
        ))
        targets = [self.arm.forward_kinematics(q)[-1] for q in q_path]
        previous = None
        selected_path = []
        for target in targets:
            result = solve_two_link_ik(target, [0.6, 0.4])
            self.assertTrue(result.reachable)
            selected = select_continuous_ik_solution(result.solutions, previous)
            selected_path.append(selected)
            previous = selected
        selected_path = np.asarray(selected_path)
        self.assertGreater(float(selected_path[0, 1]), 0.0)
        self.assertGreater(float(selected_path[-1, 1]), 0.0)
        steps = np.diff(selected_path, axis=0)
        self.assertLess(float(np.max(np.linalg.norm(steps, axis=1))), 0.03)


if __name__ == "__main__":
    unittest.main()
