import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from app import MainWindow, _unwrap_angle_near
from elbow_drag import solve_elbow_drag_candidates
from joint_limits import JointLimits


class TestDesktopPrototype(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow()

    def tearDown(self):
        self.window.close()

    def _use_two_link_arm(self):
        self.window.arm = __import__("planar_arm").PlanarArm([1.0, 1.0])
        self.window.joint_angles = np.array([np.pi / 4, -np.pi / 2])
        self.window.joint_limits = JointLimits([-2 * np.pi] * 2, [2 * np.pi] * 2)
        self.window.target = self.window.arm.forward_kinematics(self.window.joint_angles)[-1]
        self.window.workspace.set_state(
            self.window.arm.link_lengths, self.window.joint_angles, self.window.target, "valid"
        )

    def test_link_count_can_be_increased_and_decreased(self):
        self.window.link_count_control.setValue(6)
        self.window._set_link_count()
        self.assertEqual(self.window.arm.joint_count, 6)
        self.assertEqual(len(self.window.link_controls), 6)
        self.assertEqual(len(self.window.lower_controls), 6)
        self.assertEqual(len(self.window.upper_controls), 6)
        self.assertAlmostEqual(self.window.arm.link_lengths[4], 1.0)
        np.testing.assert_allclose(np.degrees(self.window.joint_angles), [90.0, -90.0, 90.0, -90.0, 90.0, -90.0])

        self.window.link_count_control.setValue(2)
        self.window._set_link_count()
        self.assertEqual(self.window.arm.joint_count, 2)
        self.assertEqual(len(self.window.link_controls), 2)
        self.assertEqual(len(self.window.lower_controls), 2)
        self.assertEqual(len(self.window.upper_controls), 2)

    def test_main_window_starts_with_three_link_staircase_pose(self):
        self.assertEqual(self.window.arm.joint_count, 3)
        self.assertEqual(len(self.window.link_controls), 3)
        self.assertEqual(len(self.window.lower_controls), 3)
        self.assertEqual(len(self.window.upper_controls), 3)
        np.testing.assert_allclose(np.degrees(self.window.joint_angles), [90.0, -90.0, 90.0])
        self.assertTrue(all(abs(float(length) - 1.0) < 1e-12 for length in self.window.arm.link_lengths))

    def test_angle_representation_stays_continuous_across_pi_boundary(self):
        previous = np.deg2rad(179.0)
        current_wrapped = np.deg2rad(-179.0)
        continuous = _unwrap_angle_near(current_wrapped, previous)

        self.assertAlmostEqual(np.rad2deg(continuous), 181.0, places=7)
        self.assertAlmostEqual(abs(continuous - previous), np.deg2rad(2.0), places=7)

    def test_elbow_drag_preserves_side_across_negative_x_axis(self):
        line_direction = np.array([1.0, 0.0])
        previous_target = np.array([np.sqrt(2.0), 0.0])
        angles = [np.deg2rad(179.0), np.deg2rad(-179.0)]
        results = [
            solve_elbow_drag_candidates(
                [np.cos(angle), np.sin(angle)],
                [1.0, 1.0],
                line_direction,
                previous_target,
            )
            for angle in angles
        ]

        self.assertTrue(all(results))
        elbows = [np.asarray(candidates[0].elbow) for candidates in results]
        crosses = [line_direction[0] * elbow[1] - line_direction[1] * elbow[0] for elbow in elbows]
        self.assertGreater(crosses[0], 0.0)
        self.assertLess(crosses[1], 0.0)
        np.testing.assert_allclose(elbows[0], elbows[1], atol=0.04)

    def test_endpoint_opens_pin_popup(self):
        self.window.show()
        self.application.processEvents()
        workspace = self.window.workspace
        origin, scale = workspace._view_parameters()
        endpoint = workspace._world_to_screen(workspace._joint_positions()[-1], origin, scale)

        QTest.mouseClick(
            workspace,
            Qt.MouseButton.LeftButton,
            pos=QPoint(round(endpoint.x()), round(endpoint.y())),
        )
        self.application.processEvents()

        self.assertEqual(self.window.joint_popup_index, self.window.arm.joint_count)
        self.assertTrue(self.window.joint_popup.isVisible())
        self.assertTrue(self.window.joint_popup_pin.isEnabled())

    def test_fixed_node_popup_does_not_block_selecting_other_nodes(self):
        self.window.show()
        self.application.processEvents()
        workspace = self.window.workspace
        origin, scale = workspace._view_parameters()
        points = workspace._joint_positions()

        first = workspace._world_to_screen(points[2], origin, scale)
        QTest.mouseClick(
            workspace, Qt.MouseButton.LeftButton,
            pos=QPoint(round(first.x()), round(first.y())),
        )
        self.application.processEvents()
        self.window.joint_popup_pin.click()
        self.application.processEvents()
        self.assertIn(2, self.window.fixed_nodes)

        second = workspace._world_to_screen(points[1], origin, scale)
        QTest.mouseClick(
            workspace, Qt.MouseButton.LeftButton,
            pos=QPoint(round(second.x()), round(second.y())),
        )
        self.application.processEvents()

        self.assertEqual(self.window.joint_popup_index, 1)
        self.assertFalse(self.window.joint_popup.isHidden() is False and self.window.joint_popup.geometry().contains(QPoint(round(second.x()), round(second.y()))))

    def test_clicking_link_selects_it_and_updates_properties(self):
        self.window.show()
        self.application.processEvents()
        workspace = self.window.workspace
        origin, scale = workspace._view_parameters()
        points = workspace._joint_positions()
        start = workspace._world_to_screen(points[0], origin, scale)
        end = workspace._world_to_screen(points[1], origin, scale)
        midpoint = QPoint(round((start.x() + end.x()) / 2), round((start.y() + end.y()) / 2))

        QTest.mouseClick(workspace, Qt.MouseButton.LeftButton, pos=midpoint)
        self.application.processEvents()

        self.assertEqual(workspace._selected_object, ("link", 0))

    def test_workspace_renders_in_offscreen_mode(self):
        self.window.show()
        self.application.processEvents()

        rendered_image = self.window.grab()

        self.assertFalse(rendered_image.isNull())

    def test_dragging_to_reachable_target_updates_arm(self):
        self.window._on_target_dragged(1.0, 1.0)

        end_position = self.window.arm.forward_kinematics(self.window.joint_angles)[-1]
        np.testing.assert_allclose(end_position, [1.0, 1.0], atol=1e-8)
        self.assertIn("可达", self.window.status_label.text())

    def test_dragging_elbow_keeps_end_effector_on_initial_line(self):
        self._use_two_link_arm()
        initial_target = self.window.target.copy()
        self.window._on_elbow_drag_started(float(initial_target[0]), float(initial_target[1]))
        mouse_position = [np.cos(np.deg2rad(20)), np.sin(np.deg2rad(20))]
        self.window._on_elbow_dragged(*mouse_position)

        end_position = self.window.arm.forward_kinematics(self.window.joint_angles)[-1]
        np.testing.assert_allclose(end_position, self.window.target, atol=1e-8)
        determinant = (
            initial_target[0] * self.window.target[1]
            - initial_target[1] * self.window.target[0]
        )
        self.assertAlmostEqual(float(determinant), 0.0, places=8)
        self.assertAlmostEqual(
            np.linalg.norm(self.window.arm.forward_kinematics(self.window.joint_angles)[1]),
            self.window.arm.link_lengths[0],
            places=8,
        )
        self.assertAlmostEqual(
            np.linalg.norm(end_position - self.window.arm.forward_kinematics(self.window.joint_angles)[1]),
            self.window.arm.link_lengths[1],
            places=8,
        )
        self.window._on_elbow_drag_finished()

    def test_mouse_drag_on_elbow_uses_fixed_line_interaction(self):
        self._use_two_link_arm()
        self.window.show()
        self.application.processEvents()
        workspace = self.window.workspace
        origin, scale = workspace._view_parameters()
        elbow = workspace._joint_positions()[1]
        elbow_screen = workspace._world_to_screen(elbow, origin, scale)
        target_world = np.array([np.cos(np.deg2rad(20)), np.sin(np.deg2rad(20))])
        target_screen = workspace._world_to_screen(target_world, origin, scale)

        QTest.mousePress(
            workspace,
            Qt.MouseButton.LeftButton,
            pos=QPoint(round(elbow_screen.x()), round(elbow_screen.y())),
        )
        QTest.mouseMove(
            workspace,
            QPoint(round(target_screen.x()), round(target_screen.y())),
            delay=10,
        )
        QTest.mouseRelease(
            workspace,
            Qt.MouseButton.LeftButton,
            pos=QPoint(round(target_screen.x()), round(target_screen.y())),
        )
        self.application.processEvents()

        end_position = self.window.arm.forward_kinematics(self.window.joint_angles)[-1]
        self.assertAlmostEqual(float(end_position[1]), 0.0, places=7)
        np.testing.assert_allclose(end_position, self.window.target, atol=1e-8)
        self.assertIsNone(workspace._drag_line_direction)

    def test_unreachable_target_does_not_change_joint_angles(self):
        original_angles = self.window.joint_angles.copy()

        self.window._on_target_dragged(4.0, 0.0)

        np.testing.assert_allclose(self.window.joint_angles, original_angles, atol=1e-12)
        self.assertIn("不可达", self.window.status_label.text())

    def test_mouse_drag_on_multilink_intermediate_joint_rotates_downstream(self):
        from planar_arm import PlanarArm

        self.window.arm = PlanarArm([1.0] * 4)
        self.window.joint_angles = np.radians([90.0, -90.0, 90.0, -90.0])
        self.window.joint_limits = JointLimits([-2 * np.pi] * 4, [2 * np.pi] * 4)
        self.window.target = self.window.arm.forward_kinematics(self.window.joint_angles)[-1]
        self.window._refresh_workspace()
        self.window.show()
        self.application.processEvents()
        workspace = self.window.workspace
        origin, scale = workspace._view_parameters()
        initial_positions = workspace._joint_positions()
        initial_angles = self.window.joint_angles.copy()
        joint_screen = workspace._world_to_screen(initial_positions[2], origin, scale)
        destination = workspace._world_to_screen([0.0, 2.0], origin, scale)

        QTest.mousePress(
            workspace,
            Qt.MouseButton.LeftButton,
            pos=QPoint(round(joint_screen.x()), round(joint_screen.y())),
        )
        QTest.mouseMove(
            workspace,
            QPoint(round(destination.x()), round(destination.y())),
            delay=10,
        )
        QTest.mouseRelease(
            workspace,
            Qt.MouseButton.LeftButton,
            pos=QPoint(round(destination.x()), round(destination.y())),
        )
        self.application.processEvents()

        positions = self.window.arm.forward_kinematics(self.window.joint_angles)
        np.testing.assert_allclose(positions[1], initial_positions[1], atol=1e-8)
        self.assertNotAlmostEqual(self.window.joint_angles[1], initial_angles[1], places=5)
        np.testing.assert_allclose(
            self.window.joint_angles[[0, 2, 3]],
            initial_angles[[0, 2, 3]],
            atol=1e-8,
        )
        np.testing.assert_allclose(
            np.linalg.norm(np.diff(positions, axis=0), axis=1),
            self.window.arm.link_lengths,
            atol=1e-8,
        )

    def test_dragging_first_internal_joint_changes_only_first_angle(self):
        initial_angles = self.window.joint_angles.copy()
        self.window._on_intermediate_joint_dragged(1, 2.0, 0.0)

        positions = self.window.arm.forward_kinematics(self.window.joint_angles)
        np.testing.assert_allclose(positions[0], [0.0, 0.0], atol=1e-8)
        np.testing.assert_allclose(np.linalg.norm(positions[1]), self.window.arm.link_lengths[0], atol=1e-8)
        self.assertNotAlmostEqual(self.window.joint_angles[0], initial_angles[0], places=5)
        np.testing.assert_allclose(self.window.joint_angles[1:], initial_angles[1:], atol=1e-12)
        np.testing.assert_allclose(
            np.linalg.norm(np.diff(positions, axis=0), axis=1),
            self.window.arm.link_lengths,
            atol=1e-8,
        )

    def test_dragging_multilink_intermediate_joint_changes_only_one_angle(self):
        from planar_arm import PlanarArm

        self.window.arm = PlanarArm([1.0] * 4)
        self.window.joint_angles = np.radians([90.0, -90.0, 90.0, -90.0])
        self.window.joint_limits = JointLimits([-2 * np.pi] * 4, [2 * np.pi] * 4)
        self.window.target = self.window.arm.forward_kinematics(self.window.joint_angles)[-1]
        initial_angles = self.window.joint_angles.copy()
        initial_positions = self.window.arm.forward_kinematics(initial_angles)
        self.window._on_intermediate_joint_dragged(2, 0.0, 2.0)

        positions = self.window.arm.forward_kinematics(self.window.joint_angles)
        np.testing.assert_allclose(positions[1], initial_positions[1], atol=1e-8)
        self.assertNotAlmostEqual(self.window.joint_angles[1], initial_angles[1], places=5)
        np.testing.assert_allclose(
            self.window.joint_angles[[0, 2, 3]],
            initial_angles[[0, 2, 3]],
            atol=1e-12,
        )
        segment_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        np.testing.assert_allclose(segment_lengths, self.window.arm.link_lengths, atol=1e-8)
        np.testing.assert_allclose(self.window.target, positions[-1], atol=1e-8)

    def test_dragging_last_internal_joint_changes_only_its_upstream_angle(self):
        self.window.arm = __import__("planar_arm").PlanarArm([1.0] * 4)
        self.window.joint_angles = np.radians([90.0, -90.0, 90.0, -90.0])
        self.window.joint_limits = JointLimits([-2 * np.pi] * 4, [2 * np.pi] * 4)
        initial_angles = self.window.joint_angles.copy()
        initial_positions = self.window.arm.forward_kinematics(initial_angles)
        self.window._on_intermediate_joint_dragged(3, 1.5, 0.5)

        positions = self.window.arm.forward_kinematics(self.window.joint_angles)
        np.testing.assert_allclose(positions[2], initial_positions[2], atol=1e-8)
        self.assertNotAlmostEqual(self.window.joint_angles[2], initial_angles[2], places=5)
        np.testing.assert_allclose(
            self.window.joint_angles[[0, 1, 3]],
            initial_angles[[0, 1, 3]],
            atol=1e-12,
        )
        segment_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        np.testing.assert_allclose(segment_lengths, self.window.arm.link_lengths, atol=1e-8)
        np.testing.assert_allclose(self.window.target, positions[-1], atol=1e-8)

    def test_dragging_upstream_of_fixed_node_moves_dragged_node_and_preserves_pin(self):
        from planar_arm import PlanarArm

        self.window.arm = PlanarArm([1.0] * 7)
        self.window.joint_angles = np.array([0.25, -0.35, 0.4, -0.2, 0.3, -0.15, 0.1])
        self.window.joint_limits = JointLimits([-2 * np.pi] * 7, [2 * np.pi] * 7)
        before = self.window.arm.forward_kinematics(self.window.joint_angles)
        self.window.fixed_nodes = {5: before[5].copy()}
        target = before[3] + np.array([0.02, -0.02])

        self.window._on_intermediate_joint_dragged(3, *target)

        after = self.window.arm.forward_kinematics(self.window.joint_angles)
        self.assertGreater(float(np.linalg.norm(after[3] - before[3])), 1e-5)
        np.testing.assert_allclose(after[5], before[5], atol=1e-5)
        np.testing.assert_allclose(
            np.linalg.norm(np.diff(after, axis=0), axis=1),
            self.window.arm.link_lengths,
            atol=1e-8,
        )
        self.assertIn(5, self.window.fixed_nodes)
        self.assertIn("约束求解成功", self.window.status_label.text())

    def test_dragging_node_before_fixed_node_keeps_endpoint_still_and_allows_fixed_joint_angle_change(self):
        from planar_arm import PlanarArm

        self.window.arm = PlanarArm([1.0] * 7)
        initial_angles = np.array([0.25, -0.35, 0.4, -0.2, 0.3, -0.15, 0.1])
        self.window.joint_angles = initial_angles.copy()
        self.window.joint_limits = JointLimits([-2 * np.pi] * 7, [2 * np.pi] * 7)
        before = self.window.arm.forward_kinematics(initial_angles)
        self.window.fixed_nodes = {5: before[5].copy()}
        target = before[4] + np.array([0.02, -0.02])

        self.window._on_intermediate_joint_dragged(4, *target)

        after = self.window.arm.forward_kinematics(self.window.joint_angles)
        self.assertIn("约束求解成功", self.window.status_label.text())
        self.assertGreater(float(np.linalg.norm(after[4] - before[4])), 1e-5)
        np.testing.assert_allclose(after[5], before[5], atol=1e-5)
        np.testing.assert_allclose(after[7], before[7], atol=1e-5)
        self.assertNotAlmostEqual(
            float(np.sum(self.window.joint_angles[:5])),
            float(np.sum(initial_angles[:5])),
            places=5,
        )
        np.testing.assert_allclose(
            np.linalg.norm(np.diff(after, axis=0), axis=1),
            self.window.arm.link_lengths,
            atol=1e-8,
        )

    def test_fixed_nodes_three_and_six_keep_all_outer_nodes_still(self):
        from planar_arm import PlanarArm

        self.window.arm = PlanarArm([1.0] * 7)
        initial_angles = np.array([0.25, -0.35, 0.4, -0.2, 0.3, -0.15, 0.1])
        self.window.joint_angles = initial_angles.copy()
        self.window.joint_limits = JointLimits([-2 * np.pi] * 7, [2 * np.pi] * 7)
        before = self.window.arm.forward_kinematics(initial_angles)
        self.window.fixed_nodes = {3: before[3].copy(), 6: before[6].copy()}

        for dragged_index in (4, 5):
            self.window.joint_angles = initial_angles.copy()
            target = before[dragged_index] + np.array([0.01, -0.01])
            self.window._on_intermediate_joint_dragged(dragged_index, *target)

            after = self.window.arm.forward_kinematics(self.window.joint_angles)
            self.assertIn(
                "约束求解成功",
                self.window.status_label.text(),
                msg=f"dragged={dragged_index}: {self.window.status_label.text()}",
            )
            np.testing.assert_allclose(after[[1, 2, 3, 6, 7]], before[[1, 2, 3, 6, 7]], atol=1e-5)
            self.assertGreater(
                float(np.linalg.norm(after[dragged_index] - before[dragged_index])),
                1e-5,
            )
            np.testing.assert_allclose(
                np.linalg.norm(np.diff(after, axis=0), axis=1),
                self.window.arm.link_lengths,
                atol=1e-8,
            )

    def test_fixed_nodes_two_and_five_keep_endpoint_six_still_when_dragging_three_or_four(self):
        from planar_arm import PlanarArm

        self.window.arm = PlanarArm([1.0] * 7)
        initial_angles = np.array([0.25, -0.35, 0.4, -0.2, 0.3, -0.15, 0.1])
        self.window.joint_angles = initial_angles.copy()
        self.window.joint_limits = JointLimits([-2 * np.pi] * 7, [2 * np.pi] * 7)
        before = self.window.arm.forward_kinematics(initial_angles)
        self.window.fixed_nodes = {2: before[2].copy(), 5: before[5].copy()}

        for dragged_index in (3, 4):
            self.window.joint_angles = initial_angles.copy()
            target = before[dragged_index] + np.array([0.01, -0.01])
            self.window._on_intermediate_joint_dragged(dragged_index, *target)

            after = self.window.arm.forward_kinematics(self.window.joint_angles)
            self.assertIn(
                "约束求解成功",
                self.window.status_label.text(),
                msg=f"dragged={dragged_index}: {self.window.status_label.text()}",
            )
            np.testing.assert_allclose(after[2], before[2], atol=1e-5)
            np.testing.assert_allclose(after[5], before[5], atol=1e-5)
            np.testing.assert_allclose(after[6], before[6], atol=1e-5)
            np.testing.assert_allclose(
                np.linalg.norm(np.diff(after, axis=0), axis=1),
                self.window.arm.link_lengths,
                atol=1e-8,
            )

        # 固定节点5只约束其位置，不应把第5个关节角作为固定值写入约束。
        self.window.joint_angles = initial_angles.copy()
        target = before[4] + np.array([0.01, -0.01])
        self.window._on_intermediate_joint_dragged(4, *target)
        self.assertNotAlmostEqual(
            float(self.window.joint_angles[4]),
            float(initial_angles[4]),
            places=5,
        )

    def test_fixed_node_allows_dragging_each_upstream_node(self):
        from planar_arm import PlanarArm

        for fixed_index, dragged_indices in ((3, (1, 2)), (4, (1, 2, 3))):
            for dragged_index in dragged_indices:
                window = MainWindow()
                try:
                    window.arm = PlanarArm([1.0] * 4)
                    window.joint_angles = np.radians([90.0, -90.0, 90.0, -90.0])
                    window.joint_limits = JointLimits([-2 * np.pi] * 4, [2 * np.pi] * 4)
                    before = window.arm.forward_kinematics(window.joint_angles)
                    window.fixed_nodes = {fixed_index: before[fixed_index].copy()}
                    target = before[dragged_index] + np.array([0.03, 0.04])

                    window._on_intermediate_joint_dragged(dragged_index, *target)

                    after = window.arm.forward_kinematics(window.joint_angles)
                    self.assertIn("约束求解成功", window.status_label.text(),
                                  msg=f"fixed={fixed_index}, dragged={dragged_index}: {window.status_label.text()}")
                    np.testing.assert_allclose(after[fixed_index], before[fixed_index], atol=1e-5)
                    np.testing.assert_allclose(
                        after[fixed_index:],
                        before[fixed_index:],
                        atol=1e-5,
                        err_msg=f"fixed={fixed_index}, dragged={dragged_index}: nodes beyond the fixed boundary moved",
                    )
                    self.assertGreater(
                        float(np.linalg.norm(after[dragged_index] - before[dragged_index])),
                        1e-5,
                        msg=f"fixed={fixed_index}, dragged={dragged_index}",
                    )
                finally:
                    window.close()

    def test_dragging_downstream_of_fixed_node_uses_forward_rotation(self):
        from planar_arm import PlanarArm

        self.window.arm = PlanarArm([1.0] * 7)
        self.window.joint_angles = np.array([0.25, -0.35, 0.4, -0.2, 0.3, -0.15, 0.1])
        initial_angles = self.window.joint_angles.copy()
        self.window.joint_limits = JointLimits([-2 * np.pi] * 7, [2 * np.pi] * 7)
        before = self.window.arm.forward_kinematics(self.window.joint_angles)
        self.window.fixed_nodes = {5: before[5].copy()}
        target = before[6] + np.array([0.0, 0.15])

        self.window._on_intermediate_joint_dragged(6, *target)

        after = self.window.arm.forward_kinematics(self.window.joint_angles)
        np.testing.assert_allclose(after[:6], before[:6], atol=1e-8)
        self.assertNotAlmostEqual(self.window.joint_angles[5], initial_angles[5], places=5)
        np.testing.assert_allclose(after[5], self.window.fixed_nodes[5], atol=1e-8)
        np.testing.assert_allclose(
            np.linalg.norm(np.diff(after, axis=0), axis=1),
            self.window.arm.link_lengths,
            atol=1e-8,
        )

    def test_base_target_is_handled_by_numerical_solver(self):
        from planar_arm import PlanarArm

        self.window.arm = PlanarArm([1.0] * 4)
        self.window.joint_angles = np.radians([35.0, -80.0, 65.0, -40.0])
        self.window.joint_limits = JointLimits([-2 * np.pi] * 4, [2 * np.pi] * 4)
        self.window.target = self.window.arm.forward_kinematics(self.window.joint_angles)[-1]
        self.window.workspace.set_state(
            self.window.arm.link_lengths, self.window.joint_angles, self.window.target, "valid"
        )
        self.window._on_target_dragged(0.0, 0.0)

        end_position = self.window.arm.forward_kinematics(self.window.joint_angles)[-1]
        np.testing.assert_allclose(end_position, [0.0, 0.0], atol=1e-6)
        self.assertIn("可达", self.window.status_label.text())

    def test_mismatched_limit_count_is_reported_without_crashing(self):
        original_angles = self.window.joint_angles.copy()
        self.window.joint_limits = JointLimits(
            [np.deg2rad(45), np.deg2rad(80)],
            [np.deg2rad(60), np.deg2rad(100)],
        )

        self.window._on_target_dragged(1.0, 1.0)

        np.testing.assert_allclose(self.window.joint_angles, original_angles, atol=1e-12)
        self.assertIn("数量与机械臂关节数量不一致", self.window.status_label.text())


if __name__ == "__main__":
    unittest.main()
