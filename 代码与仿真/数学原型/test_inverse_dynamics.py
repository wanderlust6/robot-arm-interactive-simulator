import unittest

import numpy as np

from dynamics import PlanarArmDynamics


class TestInverseDynamics(unittest.TestCase):
    def test_static_horizontal_single_link_matches_hand_calculation(self):
        model = PlanarArmDynamics(
            [1.5], [2.0], center_of_mass_fractions=[0.4], payload_mass=0.7
        )
        torque = model.inverse_dynamics([0.0], [0.0], [0.0])
        expected = 9.80665 * (2.0 * 0.6 + 0.7 * 1.5)
        self.assertAlmostEqual(torque[0], expected, places=10)

    def test_static_vertical_arm_has_zero_gravity_torque(self):
        model = PlanarArmDynamics([1.0, 0.8], [2.0, 1.0], payload_mass=0.5)
        torque = model.inverse_dynamics([np.pi / 2, 0.0], [0.0, 0.0], [0.0, 0.0])
        np.testing.assert_allclose(torque, [0.0, 0.0], atol=1e-10)

    def test_single_link_angular_acceleration_matches_equivalent_inertia(self):
        length, mass, fraction, payload, acceleration = 1.5, 2.0, 0.4, 0.7, 2.0
        model = PlanarArmDynamics(
            [length], [mass], center_of_mass_fractions=[fraction], payload_mass=payload,
            gravity=0.0,
        )
        torque = model.inverse_dynamics([0.3], [0.0], [acceleration])
        com_distance = length * fraction
        inertia_com = mass * length**2 / 12.0
        inertia_base = inertia_com + mass * com_distance**2 + payload * length**2
        self.assertAlmostEqual(torque[0], inertia_base * acceleration, places=10)

    def test_static_inverse_dynamics_matches_existing_gravity_model(self):
        model = PlanarArmDynamics(
            [0.8, 1.2, 0.5], [1.3, 0.7, 2.1],
            center_of_mass_fractions=[0.3, 0.6, 0.45], payload_mass=1.4,
        )
        angles = [0.4, -0.8, 1.1]
        np.testing.assert_allclose(
            model.inverse_dynamics(angles, [0.0] * 3, [0.0] * 3),
            model.gravity_torques(angles), rtol=1e-10, atol=1e-10,
        )

    def test_zero_mass_and_zero_gravity_produce_zero_torque_without_motion(self):
        model = PlanarArmDynamics([1.0, 2.0], [0.0, 0.0], gravity=0.0)
        np.testing.assert_allclose(
            model.inverse_dynamics([0.2, -0.3], [0.0, 0.0], [0.0, 0.0]),
            [0.0, 0.0], atol=1e-12,
        )

    def test_velocity_coupling_generates_torque_in_bent_two_link_arm(self):
        model = PlanarArmDynamics(
            [1.0, 1.0], [0.0, 2.0], center_of_mass_fractions=[0.5, 0.5],
            gravity=0.0, link_inertias=[0.0, 0.0],
        )
        torque = model.inverse_dynamics([0.0, np.pi / 2], [1.0, 0.0], [0.0, 0.0])
        self.assertAlmostEqual(torque[1], 1.0, places=10)

    def test_invalid_dynamic_state_is_rejected(self):
        model = PlanarArmDynamics([1.0, 1.0])
        invalid_states = (
            ([0.0], [0.0, 0.0], [0.0, 0.0]),
            ([0.0, 0.0], [0.0], [0.0, 0.0]),
            ([0.0, 0.0], [0.0, 0.0], [0.0]),
            ([0.0, np.nan], [0.0, 0.0], [0.0, 0.0]),
            ([0.0, 0.0], [0.0, np.inf], [0.0, 0.0]),
            ([0.0, 0.0], [0.0, 0.0], [0.0, -np.inf]),
        )
        for state in invalid_states:
            with self.subTest(state=state), self.assertRaises(ValueError):
                model.inverse_dynamics(*state)

    def test_link_inertia_must_be_nonnegative_and_match_link_count(self):
        with self.assertRaises(ValueError):
            PlanarArmDynamics([1.0, 1.0], link_inertias=[0.1])
        with self.assertRaises(ValueError):
            PlanarArmDynamics([1.0], link_inertias=[-0.1])
        with self.assertRaises(ValueError):
            PlanarArmDynamics([1.0], link_inertias=[np.nan])


def lagrangian_reference(model, q, qd, qdd, step=1e-4):
    """用能量法与中心有限差分构造独立的数值参考解。"""
    q = np.asarray(q, dtype=float)
    qd = np.asarray(qd, dtype=float)
    qdd = np.asarray(qdd, dtype=float)
    n = len(q)
    lengths = model.arm.link_lengths
    masses = model.link_masses
    fractions = model.center_of_mass_fractions
    inertias = model.link_inertias

    def point_positions(angles):
        joints = model.arm.forward_kinematics(angles)
        absolute = np.cumsum(angles)
        centers = np.array([
            joints[i] + fractions[i] * lengths[i]
            * np.array([np.cos(absolute[i]), np.sin(absolute[i])])
            for i in range(n)
        ])
        return centers, joints[-1]

    def potential(angles):
        centers, end = point_positions(angles)
        return model.gravity * (
            sum(masses[i] * centers[i, 1] for i in range(n))
            + model.payload_mass * end[1]
        )

    centers, end = point_positions(q)
    jacobians = []
    for point_index in range(n + 1):
        jacobian = np.zeros((2, n))
        for j in range(n):
            delta = np.zeros(n)
            delta[j] = step
            plus_centers, plus_end = point_positions(q + delta)
            minus_centers, minus_end = point_positions(q - delta)
            plus = plus_centers[point_index] if point_index < n else plus_end
            minus = minus_centers[point_index] if point_index < n else minus_end
            jacobian[:, j] = (plus - minus) / (2 * step)
        jacobians.append(jacobian)

    mass_matrix = np.zeros((n, n))
    for i in range(n):
        mass_matrix += masses[i] * jacobians[i].T @ jacobians[i]
        angular_jacobian = np.zeros(n)
        angular_jacobian[:i + 1] = 1.0
        mass_matrix += inertias[i] * np.outer(angular_jacobian, angular_jacobian)
    mass_matrix += model.payload_mass * jacobians[n].T @ jacobians[n]

    mass_derivatives = np.zeros((n, n, n))
    for k in range(n):
        delta = np.zeros(n)
        delta[k] = step
        mass_derivatives[:, :, k] = (
            reference_mass_matrix(model, q + delta, step)
            - reference_mass_matrix(model, q - delta, step)
        ) / (2 * step)

    coriolis = np.zeros(n)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                christoffel = 0.5 * (
                    mass_derivatives[i, j, k]
                    + mass_derivatives[i, k, j]
                    - mass_derivatives[j, k, i]
                )
                coriolis[i] += christoffel * qd[j] * qd[k]

    gravity_torque = np.zeros(n)
    for j in range(n):
        delta = np.zeros(n)
        delta[j] = step
        gravity_torque[j] = (potential(q + delta) - potential(q - delta)) / (2 * step)

    return mass_matrix @ qdd + coriolis + gravity_torque


def reference_mass_matrix(model, q, step=1e-4):
    """独立能量法参考质量矩阵，供有限差分计算科氏项。"""
    q = np.asarray(q, dtype=float)
    n = len(q)
    lengths = model.arm.link_lengths
    masses = model.link_masses
    fractions = model.center_of_mass_fractions
    joints = model.arm.forward_kinematics(q)
    absolute = np.cumsum(q)
    centers = np.array([
        joints[i] + fractions[i] * lengths[i]
        * np.array([np.cos(absolute[i]), np.sin(absolute[i])])
        for i in range(n)
    ])
    jacobians = []
    for point_index in range(n + 1):
        jacobian = np.zeros((2, n))
        for j in range(n):
            delta = np.zeros(n)
            delta[j] = step
            plus_joints = model.arm.forward_kinematics(q + delta)
            minus_joints = model.arm.forward_kinematics(q - delta)
            plus_abs = np.cumsum(q + delta)
            minus_abs = np.cumsum(q - delta)
            plus_centers = np.array([
                plus_joints[i] + fractions[i] * lengths[i]
                * np.array([np.cos(plus_abs[i]), np.sin(plus_abs[i])])
                for i in range(n)
            ])
            minus_centers = np.array([
                minus_joints[i] + fractions[i] * lengths[i]
                * np.array([np.cos(minus_abs[i]), np.sin(minus_abs[i])])
                for i in range(n)
            ])
            plus = plus_centers[point_index] if point_index < n else plus_joints[-1]
            minus = minus_centers[point_index] if point_index < n else minus_joints[-1]
            jacobian[:, j] = (plus - minus) / (2 * step)
        jacobians.append(jacobian)

    matrix = np.zeros((n, n))
    for i in range(n):
        matrix += masses[i] * jacobians[i].T @ jacobians[i]
        angular_jacobian = np.zeros(n)
        angular_jacobian[:i + 1] = 1.0
        matrix += model.link_inertias[i] * np.outer(angular_jacobian, angular_jacobian)
    matrix += model.payload_mass * jacobians[n].T @ jacobians[n]
    return matrix


class TestInverseDynamicsIndependentReference(unittest.TestCase):
    def test_randomized_states_match_independent_lagrangian_reference(self):
        rng = np.random.default_rng(20261019)
        for joint_count in (2, 3, 4):
            model = PlanarArmDynamics(
                link_lengths=np.linspace(0.45, 1.1, joint_count),
                link_masses=np.linspace(0.6, 1.8, joint_count),
                center_of_mass_fractions=np.linspace(0.35, 0.6, joint_count),
                payload_mass=0.73,
                link_inertias=np.linspace(0.025, 0.11, joint_count),
                gravity=9.80665,
            )
            for case_index in range(4):
                q = rng.uniform(-1.1, 1.1, joint_count)
                qd = rng.uniform(-1.4, 1.4, joint_count)
                qdd = rng.uniform(-2.0, 2.0, joint_count)
                with self.subTest(joint_count=joint_count, case=case_index):
                    actual = model.inverse_dynamics(q, qd, qdd)
                    expected = lagrangian_reference(model, q, qd, qdd)
                    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
