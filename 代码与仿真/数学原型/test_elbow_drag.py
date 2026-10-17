import unittest

import numpy as np

from elbow_drag import solve_elbow_drag_candidates


class TestElbowDrag(unittest.TestCase):
    def test_candidates_preserve_both_link_lengths_and_fixed_line(self):
        lengths = np.array([1.0, 1.0])
        direction = np.array([1.0, 0.0])
        candidates = solve_elbow_drag_candidates(
            [np.cos(np.deg2rad(20)), np.sin(np.deg2rad(20))],
            lengths,
            direction,
            [np.sqrt(2.0), 0.0],
        )

        self.assertTrue(candidates)
        for candidate in candidates:
            elbow = np.asarray(candidate.elbow)
            target = np.asarray(candidate.target)
            np.testing.assert_allclose(np.linalg.norm(elbow), lengths[0], atol=1e-9)
            np.testing.assert_allclose(np.linalg.norm(target - elbow), lengths[1], atol=1e-9)
            np.testing.assert_allclose(target[1], 0.0, atol=1e-9)

    def test_nearest_end_position_is_returned_first(self):
        candidates = solve_elbow_drag_candidates(
            [np.cos(np.deg2rad(20)), np.sin(np.deg2rad(20))],
            [1.0, 1.0],
            [1.0, 0.0],
            [1.5, 0.0],
        )
        distances = [np.linalg.norm(np.asarray(item.target) - [1.5, 0.0]) for item in candidates]
        self.assertEqual(distances, sorted(distances))

    def test_infeasible_mouse_direction_is_clamped_to_feasible_boundary(self):
        candidates = solve_elbow_drag_candidates(
            [0.0, 1.0],
            [1.0, 0.4],
            [1.0, 0.0],
            [1.2, 0.0],
        )

        self.assertTrue(candidates)
        for candidate in candidates:
            elbow = np.asarray(candidate.elbow)
            target = np.asarray(candidate.target)
            np.testing.assert_allclose(np.linalg.norm(elbow), 1.0, atol=1e-9)
            np.testing.assert_allclose(np.linalg.norm(target - elbow), 0.4, atol=1e-9)
            np.testing.assert_allclose(target[1], 0.0, atol=1e-9)

    def test_invalid_line_direction_is_rejected(self):
        with self.assertRaises(ValueError):
            solve_elbow_drag_candidates([1.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
