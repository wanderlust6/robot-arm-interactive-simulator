import unittest

import numpy as np

from dynamics import PlanarArmDynamics


class TestGravityTorques(unittest.TestCase):
    def test_horizontal_single_link_torque_matches_mass_times_gravity_times_lever(self):
        model = PlanarArmDynamics([2.0], [3.0], payload_mass=4.0)
        torque = model.gravity_torques([0.0])
        self.assertAlmostEqual(torque[0], (3.0 * 1.0 + 4.0 * 2.0) * 9.80665)

    def test_vertical_arm_has_zero_gravity_torque(self):
        model = PlanarArmDynamics([1.0, 1.0], [1.0, 1.0], payload_mass=2.0)
        torque = model.gravity_torques([np.pi / 2, 0.0])
        np.testing.assert_allclose(torque, [0.0, 0.0], atol=1e-12)

    def test_payload_increases_torque_in_proportion_to_mass(self):
        single = PlanarArmDynamics([1.0, 1.0], [0.0, 0.0], payload_mass=1.0)
        loaded = PlanarArmDynamics([1.0, 1.0], [0.0, 0.0], payload_mass=2.0)
        np.testing.assert_allclose(
            loaded.gravity_torques([0.0, 0.0]),
            2.0 * single.gravity_torques([0.0, 0.0]),
            atol=1e-12,
        )

    def test_each_link_mass_contributes_only_to_upstream_joints(self):
        model = PlanarArmDynamics([1.0, 1.0], [0.0, 1.0])
        torque = model.gravity_torques([0.0, 0.0])
        self.assertAlmostEqual(torque[0], 9.80665 * 1.5)
        self.assertAlmostEqual(torque[1], 9.80665 * 0.5)

    def test_invalid_physical_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            PlanarArmDynamics([1.0, 1.0], [1.0])
        with self.assertRaises(ValueError):
            PlanarArmDynamics([1.0], [-1.0])
        with self.assertRaises(ValueError):
            PlanarArmDynamics([1.0], payload_mass=-0.1)
        with self.assertRaises(ValueError):
            PlanarArmDynamics([1.0], center_of_mass_fractions=[1.1])


if __name__ == "__main__":
    unittest.main()

