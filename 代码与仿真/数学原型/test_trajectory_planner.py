"""约束轨迹规划模块测试。"""

import unittest

import numpy as np

from dynamics import PlanarArmDynamics
from trajectory_planner import plan_cartesian_line, plan_joint_trajectory, quintic_time_scaling


class QuinticTimeScalingTests(unittest.TestCase):
    def test_start_end_positions_and_zero_endpoint_derivatives(self):
        times, angles, velocities, accelerations = quintic_time_scaling([0.0, 1.0], [1.0, -1.0], 2.0)
        np.testing.assert_allclose(angles[0], [0.0, 1.0])
        np.testing.assert_allclose(angles[-1], [1.0, -1.0])
        np.testing.assert_allclose(velocities[[0, -1]], 0.0, atol=1e-12)
        np.testing.assert_allclose(accelerations[[0, -1]], 0.0, atol=1e-12)
        self.assertAlmostEqual(times[-1], 2.0)

    def test_invalid_inputs_rejected(self):
        for duration in (0, -1, float("nan")):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                quintic_time_scaling([0.0], [1.0], duration)
        with self.assertRaises(ValueError):
            quintic_time_scaling([0.0], [1.0, 2.0], 1.0)


class JointTrajectoryPlannerTests(unittest.TestCase):
    def setUp(self):
        self.model = PlanarArmDynamics([1.0], link_masses=[1.0], gravity=0.0)

    def test_finds_feasible_trajectory_by_extending_duration(self):
        result = plan_joint_trajectory(
            self.model, [0.0], [1.0], max_velocity=0.8,
            max_acceleration=2.0, max_torque=100.0,
            initial_duration=0.5, max_duration=10.0,
        )
        self.assertTrue(result.success, result.message)
        self.assertGreater(result.duration, 0.5)
        self.assertLessEqual(np.max(np.abs(result.joint_velocities)), 0.8 * (1 + 1e-9))
        self.assertLessEqual(np.max(np.abs(result.joint_accelerations)), 2.0 * (1 + 1e-9))
        self.assertLessEqual(np.max(np.abs(result.joint_torques)), 100.0 * (1 + 1e-9))
        self.assertEqual(result.joint_angles.shape, (201, 1))

    def test_reports_torque_limit_that_time_scaling_cannot_fix(self):
        model = PlanarArmDynamics([1.0], link_masses=[1.0], gravity=9.80665)
        result = plan_joint_trajectory(
            model, [0.0], [0.0], max_velocity=2.0,
            max_acceleration=5.0, max_torque=0.01,
            initial_duration=0.5, max_duration=2.0,
        )
        self.assertFalse(result.success)
        self.assertIn("驱动力矩", result.message)

    def test_joint_limit_rejects_out_of_range_endpoint(self):
        result = plan_joint_trajectory(
            self.model, [0.0], [1.0], 2.0, 5.0, 100.0,
            joint_limits=([-0.5], [0.5]),
        )
        self.assertFalse(result.success)
        self.assertIn("关节角限位", result.message)

    def test_invalid_limits_rejected(self):
        with self.assertRaises(ValueError):
            plan_joint_trajectory(self.model, [0.0], [1.0], 0.0, 5.0, 10.0)
        with self.assertRaises(ValueError):
            plan_joint_trajectory(self.model, [0.0], [1.0], 2.0, [1.0, 2.0], 10.0)


class CartesianLinePlannerTests(unittest.TestCase):
    def test_plans_reachable_line_and_preserves_endpoints(self):
        points, angles = plan_cartesian_line([1.0, 1.0], [1.0, 0.0], [0.0, 1.0], sample_count=31)
        self.assertEqual(points.shape, (31, 2))
        self.assertEqual(angles.shape, (31, 2))
        np.testing.assert_allclose(points[0], [1.0, 0.0])
        np.testing.assert_allclose(points[-1], [0.0, 1.0])
        arm = PlanarArmDynamics([1.0, 1.0]).arm
        np.testing.assert_allclose(arm.forward_kinematics(angles[0])[-1], points[0], atol=1e-8)
        np.testing.assert_allclose(arm.forward_kinematics(angles[-1])[-1], points[-1], atol=1e-8)

    def test_rejects_unreachable_line(self):
        with self.assertRaisesRegex(ValueError, "不可达"):
            plan_cartesian_line([1.0, 1.0], [0.5, 0.0], [3.0, 0.0], sample_count=21)

    def test_rejects_joint_limit_violation(self):
        with self.assertRaisesRegex(ValueError, "关节角限位"):
            plan_cartesian_line([1.0, 1.0], [1.0, 0.0], [0.0, 1.0], joint_limits=([-0.1, -0.1], [0.1, 0.1]))


if __name__ == "__main__":
    unittest.main()


class CartesianPathConstraintTests(unittest.TestCase):
    def test_plans_multilink_cartesian_path_with_joint_limits(self):
        from joint_limits import JointLimits
        from trajectory_planner import plan_cartesian_path

        model = PlanarArmDynamics([1.0, 1.0, 1.0], link_masses=[0.0, 0.0, 0.0], gravity=0.0)
        start_angles = np.radians([90.0, -90.0, 90.0])
        arm = model.arm
        start = arm.forward_kinematics(start_angles)[-1]
        end = start + np.array([0.1, -0.1])
        points = np.linspace(start, end, 31)
        result = plan_cartesian_path(
            model, points, start_angles, max_velocity=3.0,
            max_acceleration=10.0, max_torque=1.0,
            initial_duration=2.0, sample_count=41,
            joint_limits=JointLimits([-np.pi] * 3, [np.pi] * 3),
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.joint_angles.shape[1], 3)
        self.assertGreaterEqual(result.joint_angles.shape[0], 41)
        self.assertEqual(result.joint_angles.shape[0], len(result.times))
        self.assertIsNotNone(result.path_progress)
        self.assertIsNotNone(result.path_joint_angles)
        np.testing.assert_allclose(result.joint_velocities[[0, -1]], 0.0, atol=1e-10)
        np.testing.assert_allclose(result.joint_accelerations[[0, -1]], 0.0, atol=1e-10)
        np.testing.assert_allclose(arm.forward_kinematics(result.joint_angles[0])[-1], start, atol=1e-5)
        np.testing.assert_allclose(arm.forward_kinematics(result.joint_angles[-1])[-1], end, atol=1e-4)
        # 每个笛卡尔路径样本都必须保留在输出时间轨迹中，不能被稀疏采样跳过。
        blend_grid = np.linspace(0.0, 1.0, 20001)
        blend_values = 10.0 * blend_grid**3 - 15.0 * blend_grid**4 + 6.0 * blend_grid**5
        expected_times = np.interp(result.path_progress, blend_values, blend_grid) * result.duration
        for expected_time, expected_angles in zip(expected_times, result.path_joint_angles):
            sample_index = int(np.argmin(np.abs(result.times - expected_time)))
            self.assertLess(abs(result.times[sample_index] - expected_time), 1e-6)
            np.testing.assert_allclose(result.joint_angles[sample_index], expected_angles, atol=1e-5)

    def test_key_pose_constraint_is_preserved_by_planned_joint_path(self):
        from numerical_ik import solve_n_link_ik
        from trajectory_planner import plan_cartesian_path

        model = PlanarArmDynamics([1.0, 1.0, 1.0], link_masses=[0.0] * 3, gravity=0.0)
        arm = model.arm
        start_angles = np.radians([90.0, -90.0, 90.0])
        start = arm.forward_kinematics(start_angles)[-1]
        end = start + np.array([0.08, -0.06])
        points = np.linspace(start, end, 31)
        key_index = 15
        key_angles, error, converged = solve_n_link_ik(arm, points[key_index], start_angles)
        self.assertTrue(converged)
        self.assertLess(error, 1e-5)

        result = plan_cartesian_path(
            model, points, start_angles, 100.0, 500.0, 10000.0,
            initial_duration=2.0, sample_count=41, max_duration=2.0,
            pose_constraints={key_index: key_angles},
        )
        self.assertTrue(result.success, result.message)
        progress = result.path_progress
        key_progress = progress[key_index]
        spline = __import__("scipy.interpolate", fromlist=["CubicSpline"]).CubicSpline(
            progress, result.path_joint_angles, axis=0, bc_type="natural"
        )
        np.testing.assert_allclose(spline(key_progress), key_angles, atol=1e-8)

    def test_key_pose_constraint_rejects_pose_at_wrong_cartesian_point(self):
        from trajectory_planner import plan_cartesian_path

        model = PlanarArmDynamics([1.0, 1.0, 1.0], link_masses=[0.0] * 3, gravity=0.0)
        start_angles = np.radians([90.0, -90.0, 90.0])
        start = model.arm.forward_kinematics(start_angles)[-1]
        points = np.linspace(start, start + np.array([0.1, 0.0]), 21)
        wrong_pose = start_angles + np.radians([10.0, 0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "关键姿态与路径位置冲突"):
            plan_cartesian_path(
                model, points, start_angles, 100.0, 500.0, 10000.0,
                pose_constraints={10: wrong_pose},
            )

class CornerStopTimingTests(unittest.TestCase):
    def test_joint_velocity_is_zero_at_declared_corner(self):
        from trajectory_planner import plan_cartesian_path

        model = PlanarArmDynamics([1.0, 1.0, 1.0], link_masses=[0.0] * 3, gravity=0.0)
        start_angles = np.radians([90.0, -90.0, 90.0])
        start = model.arm.forward_kinematics(start_angles)[-1]
        points = np.linspace(start, start + np.array([0.08, -0.06]), 31)
        corner_index = 15
        result = plan_cartesian_path(
            model, points, start_angles, 100.0, 500.0, 10000.0,
            initial_duration=2.0, sample_count=41, max_duration=2.0,
            corner_indices=[corner_index],
        )
        self.assertTrue(result.success, result.message)
        corner_progress = result.path_progress[corner_index]
        corner_time = result.duration * corner_progress
        index = int(np.argmin(np.abs(result.times - corner_time)))
        self.assertLess(abs(result.times[index] - corner_time), 1e-6)
        np.testing.assert_allclose(result.joint_angles[index], result.path_joint_angles[corner_index], atol=1e-5)
        np.testing.assert_allclose(result.joint_velocities[index], 0.0, atol=1e-8)
        np.testing.assert_allclose(result.joint_accelerations[index], 0.0, atol=1e-7)
