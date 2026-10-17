import unittest

import numpy as np

from planar_arm import PlanarArm


class TestPlanarArmForwardKinematics(unittest.TestCase):
    def test_two_links_straight_along_positive_x(self):
        arm = PlanarArm([1.0, 1.0])

        points = arm.forward_kinematics([0.0, 0.0])

        np.testing.assert_allclose(
            points,
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            atol=1e-12,
        )

    def test_second_joint_angle_is_relative_to_first_link(self):
        arm = PlanarArm([1.0, 1.0])

        points = arm.forward_kinematics([np.pi / 2, -np.pi / 2])

        np.testing.assert_allclose(
            points,
            [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            atol=1e-12,
        )

    def test_three_link_arm_returns_every_joint_position(self):
        arm = PlanarArm([1.0, 2.0, 0.5])

        points = arm.forward_kinematics([0.0, 0.0, 0.0])

        self.assertEqual(points.shape, (4, 2))
        np.testing.assert_allclose(points[-1], [3.5, 0.0], atol=1e-12)


class TestPlanarArmJacobian(unittest.TestCase):
    def test_jacobian_at_straight_two_link_pose(self):
        arm = PlanarArm([1.0, 1.0])

        jacobian = arm.jacobian([0.0, 0.0])

        np.testing.assert_allclose(
            jacobian,
            [[0.0, 0.0], [2.0, 1.0]],
            atol=1e-12,
        )

    def test_jacobian_matches_central_finite_difference(self):
        arm = PlanarArm([1.2, 0.8, 0.4])
        angles = np.array([0.3, -0.7, 1.1])
        step = 1e-6

        numerical = np.empty((2, len(angles)))
        for index in range(len(angles)):
            plus = angles.copy()
            minus = angles.copy()
            plus[index] += step
            minus[index] -= step
            end_plus = arm.forward_kinematics(plus)[-1]
            end_minus = arm.forward_kinematics(minus)[-1]
            numerical[:, index] = (end_plus - end_minus) / (2 * step)

        np.testing.assert_allclose(
            arm.jacobian(angles),
            numerical,
            rtol=1e-6,
            atol=1e-7,
        )

    def test_straight_two_link_pose_is_singular(self):
        arm = PlanarArm([1.0, 1.0])

        determinant = np.linalg.det(arm.jacobian([0.0, 0.0]))

        self.assertAlmostEqual(determinant, 0.0, places=12)

    def test_bent_two_link_pose_is_not_singular(self):
        arm = PlanarArm([1.0, 1.0])

        determinant = np.linalg.det(arm.jacobian([0.2, 0.8]))

        self.assertGreater(abs(determinant), 1e-3)


class TestPlanarArmInputValidation(unittest.TestCase):
    def test_rejects_empty_link_lengths(self):
        with self.assertRaises(ValueError):
            PlanarArm([])

    def test_rejects_non_positive_link_lengths(self):
        for lengths in ([1.0, 0.0], [1.0, -0.5]):
            with self.subTest(lengths=lengths):
                with self.assertRaises(ValueError):
                    PlanarArm(lengths)

    def test_rejects_non_finite_link_lengths(self):
        with self.assertRaises(ValueError):
            PlanarArm([1.0, np.inf])

    def test_rejects_wrong_number_of_joint_angles(self):
        arm = PlanarArm([1.0, 1.0])

        with self.assertRaises(ValueError):
            arm.forward_kinematics([0.0])

        with self.assertRaises(ValueError):
            arm.jacobian([0.0, 0.0, 0.0])

    def test_rejects_non_finite_joint_angles(self):
        arm = PlanarArm([1.0, 1.0])

        with self.assertRaises(ValueError):
            arm.forward_kinematics([0.0, np.nan])


if __name__ == "__main__":
    unittest.main()
