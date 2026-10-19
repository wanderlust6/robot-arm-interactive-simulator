import unittest

import numpy as np

from dynamics import PlanarArmDynamics
from trajectory_dynamics import calculate_trajectory_torques


class TestTrajectoryDynamics(unittest.TestCase):
    def test_empty_trajectory_returns_empty_torque_matrix(self):
        model = PlanarArmDynamics([1.0, 0.8])
        result = calculate_trajectory_torques(model, [], np.empty((0, 2)))
        self.assertEqual(result.shape, (0, 2))

    def test_single_sample_uses_static_gravity_torque(self):
        model = PlanarArmDynamics([1.2, 0.7], [1.4, 0.8], payload_mass=0.3)
        angles = np.array([[0.4, -0.7]])
        result = calculate_trajectory_torques(model, [0.0], angles)
        np.testing.assert_allclose(result[0], model.gravity_torques(angles[0]), atol=1e-10)

    def test_constant_pose_has_static_gravity_torque_at_every_sample(self):
        model = PlanarArmDynamics([0.9, 1.1], [1.2, 0.6], payload_mass=0.4)
        times = np.linspace(0.0, 2.0, 9)
        angles = np.tile([0.3, -0.8], (len(times), 1))
        result = calculate_trajectory_torques(model, times, angles)
        expected = model.gravity_torques(angles[0])
        np.testing.assert_allclose(result, np.tile(expected, (len(times), 1)), atol=1e-9)

    def test_single_link_constant_angular_acceleration_matches_hand_calculation(self):
        length, mass, payload, acceleration = 1.4, 2.1, 0.6, 1.7
        model = PlanarArmDynamics(
            [length], [mass], payload_mass=payload, gravity=0.0
        )
        times = np.linspace(0.0, 1.0, 11)
        angles = (0.2 + 0.4 * times + 0.5 * acceleration * times**2)[:, None]
        result = calculate_trajectory_torques(model, times, angles)
        inertia = mass * length**2 / 3.0 + payload * length**2
        np.testing.assert_allclose(result[:, 0], inertia * acceleration, atol=1e-8)

    def test_angle_wrap_is_unwrapped_before_differentiation(self):
        model = PlanarArmDynamics([1.0], [1.0], gravity=0.0)
        times = np.linspace(0.0, 1.0, 11)
        continuous = np.linspace(2.8, 3.5, len(times))
        wrapped = (continuous + np.pi) % (2.0 * np.pi) - np.pi
        result = calculate_trajectory_torques(model, times, wrapped[:, None])
        np.testing.assert_allclose(result[:, 0], 0.0, atol=1e-8)

    def test_non_increasing_times_are_rejected(self):
        model = PlanarArmDynamics([1.0])
        with self.assertRaises(ValueError):
            calculate_trajectory_torques(model, [0.0, 0.2, 0.2], [[0.0], [0.1], [0.2]])

    def test_invalid_shapes_and_non_finite_values_are_rejected(self):
        model = PlanarArmDynamics([1.0, 1.0])
        with self.assertRaises(ValueError):
            calculate_trajectory_torques(model, [0.0, 1.0], [[0.0], [0.1]])
        with self.assertRaises(ValueError):
            calculate_trajectory_torques(model, [0.0, np.nan], [[0.0, 0.0], [0.1, 0.1]])


if __name__ == "__main__":
    unittest.main()
