import unittest
import numpy as np
from planar_arm import PlanarArm
from joint_limits import JointLimits
from numerical_ik import solve_n_link_ik, solve_constrained_drag

class TestNumericalIK(unittest.TestCase):
    def test_reaches_target_with_four_links(self):
        arm = PlanarArm([1.0, 1.0, 0.7, 0.5])
        expected = np.array([0.3, -0.5, 0.4, 0.2])
        target = arm.forward_kinematics(expected)[-1]
        limits = JointLimits([-np.pi] * 4, [np.pi] * 4)
        angles, error, converged = solve_n_link_ik(arm, target, [0.1, -0.2, 0.1, 0.0], limits)
        self.assertTrue(converged, f"error={error}")
        self.assertLess(error, 1e-6)
        np.testing.assert_allclose(arm.forward_kinematics(angles)[-1], target, atol=1e-6)

    def test_constrained_drag_keeps_fixed_endpoint(self):
        arm = PlanarArm([1.0] * 6)
        initial = np.array([0.4, -0.7, 0.5, 0.3, -0.2, 0.1])
        initial_positions = arm.forward_kinematics(initial)
        target = initial_positions[3] + np.array([0.03, -0.02])
        limits = JointLimits([-2 * np.pi] * 6, [2 * np.pi] * 6)
        angles, error, converged = solve_constrained_drag(
            arm, 3, target, initial, {6: initial_positions[6]}, limits
        )
        self.assertTrue(converged, f"error={error}")
        positions = arm.forward_kinematics(angles)
        np.testing.assert_allclose(positions[3], target, atol=1e-5)
        np.testing.assert_allclose(positions[6], initial_positions[6], atol=1e-5)
        np.testing.assert_allclose(
            np.linalg.norm(np.diff(positions, axis=0), axis=1), arm.link_lengths, atol=1e-8
        )

    def test_constrained_drag_preserves_multiple_fixed_nodes(self):
        arm = PlanarArm([1.0] * 6)
        initial = np.array([0.4, -0.7, 0.5, 0.3, -0.2, 0.1])
        initial_positions = arm.forward_kinematics(initial)
        target = initial_positions[4] + np.array([-0.001, 0.001])
        fixed = {2: initial_positions[2], 6: initial_positions[6]}
        angles, error, converged = solve_constrained_drag(
            arm, 4, target, initial, fixed, JointLimits([-2 * np.pi] * 6, [2 * np.pi] * 6)
        )
        self.assertTrue(converged, f"error={error}")
        positions = arm.forward_kinematics(angles)
        np.testing.assert_allclose(positions[2], initial_positions[2], atol=1e-5)
        np.testing.assert_allclose(positions[6], initial_positions[6], atol=1e-5)
        np.testing.assert_allclose(positions[4], target, atol=1e-5)

    def test_rejects_mismatched_initial_angles(self):
        arm = PlanarArm([1.0, 1.0, 1.0])
        with self.assertRaises(ValueError):
            solve_n_link_ik(arm, [1.0, 1.0], [0.0, 0.0])

if __name__ == "__main__":
    unittest.main()
