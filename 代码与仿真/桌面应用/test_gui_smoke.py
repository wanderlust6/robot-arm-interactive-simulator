import os
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QScrollArea

from app import MainWindow, _unwrap_angle_near
from path_trajectory import sample_path_by_arclength, parameterize_path, find_sharp_corner_indices
from elbow_drag import solve_elbow_drag_candidates
from joint_limits import JointLimits
from simulation_plot import SimulationPlot
from experiment_io import write_experiment_package


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

    def test_experiment_package_load_restores_parameters_and_recorded_data(self):
        config = self.window._experiment_configuration()
        angles = np.array([[0.0, 0.1, -0.2], [0.1, 0.0, -0.1]])
        positions = np.array([
            self.window.arm.forward_kinematics(row) for row in angles
        ])
        with tempfile.TemporaryDirectory() as directory:
            write_experiment_package(
                directory, config, [0.0, 0.1], positions, angles,
                np.zeros_like(angles), np.zeros_like(angles), np.zeros_like(angles),
            )
            with patch("app.QFileDialog.getOpenFileName", return_value=(str(__import__("pathlib").Path(directory) / "experiment.json"), "experiment.json")):
                self.window._load_experiment_package()
        self.assertEqual(len(self.window._simulation_log_times), 2)
        np.testing.assert_allclose(self.window._simulation_log_angles, angles)
        np.testing.assert_allclose(self.window._simulation_log_positions, positions)
        np.testing.assert_allclose(self.window.arm.link_lengths, config["model"]["link_lengths_m"])
        self.assertFalse(self.window._simulation_prepared)
        self.assertIsNone(self.window._planned_path_spline)
        self.assertEqual(self.window.simulation_route_indices, config["trajectory"]["selected_route_indices"])
        self.assertEqual(
            self.window.speed_limit_enabled.isChecked(),
            config["constraints"]["joint_speed"]["enabled"],
        )
        self.assertIn("已加载实验包", self.window.statusBar().currentMessage())

    def test_experiment_configuration_captures_model_constraints_and_path(self):
        config = self.window._experiment_configuration()
        self.assertEqual(config["model"]["link_lengths_m"], [1.0, 1.0, 1.0])
        self.assertEqual(len(config["model"]["joint_angles_rad"]), 3)
        self.assertIn("joint_speed", config["constraints"])
        self.assertIn("anchors", config["trajectory"])
        self.assertEqual(config["simulation"]["recorded_samples"], 0)
        self.window.speed_limit_enabled.setChecked(True)
        updated = self.window._experiment_configuration()
        self.assertTrue(updated["constraints"]["joint_speed"]["enabled"])

    def test_plot_click_selects_nearest_curve_point_and_stores_coordinates(self):
        plot = SimulationPlot()
        plot.resize(640, 360)
        plot.set_configuration("time", ["node_x:0"], "shared")
        times = np.array([0.0, 1.0, 2.0])
        positions = np.array([[[0.0, 0.0]], [[1.0, 0.0]], [[2.0, 0.0]]])
        plot.set_data(times, positions, np.zeros((3, 1)), 0.0, 2.0)
        plot.show()
        self.application.processEvents()
        self.assertEqual(len(plot._drawn_points), 3)
        px, py, expected_x, expected_y, _ = plot._drawn_points[1]
        QTest.mouseClick(plot, Qt.MouseButton.LeftButton, pos=QPoint(px, py))
        self.application.processEvents()
        self.assertIsNotNone(plot._selected_point)
        self.assertAlmostEqual(plot._selected_point[2], expected_x)
        self.assertAlmostEqual(plot._selected_point[3], expected_y)
        second = plot._drawn_points[2]
        QTest.mouseClick(plot, Qt.MouseButton.LeftButton, pos=QPoint(second[0], second[1]))
        self.application.processEvents()
        self.assertAlmostEqual(plot._selected_point[2], second[2])
        self.assertAlmostEqual(plot._selected_point[3], second[3])
        plot.close()

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

    def test_first_link_can_cross_minus_180_during_preparation_with_limits_disabled(self):
        from planar_arm import PlanarArm

        self.window.arm = PlanarArm([1.0, 1.0, 1.0])
        self.window.joint_angles = np.deg2rad([-179.0, -90.0, 90.0])
        self.window.joint_limits = JointLimits(
            np.full(3, -np.deg2rad(1e9)), np.full(3, np.deg2rad(1e9))
        )
        self.assertFalse(self.window._joint_limits_enabled())
        self.assertIsNone(self.window._effective_joint_limits())
        endpoint = self.window.arm.forward_kinematics(self.window.joint_angles)[-1]
        self.window.target = endpoint.copy()
        self.window.fixed_nodes = {3: endpoint.copy()}
        self.window.workspace.set_state(
            self.window.arm.link_lengths, self.window.joint_angles,
            self.window.target, "valid", self.window.fixed_nodes,
        )
        target_angle = np.deg2rad(179.0)
        target = np.array([np.cos(target_angle), np.sin(target_angle)])
        self.window._on_intermediate_joint_dragged(1, float(target[0]), float(target[1]))
        self.assertLess(self.window.joint_angles[0], -np.pi, self.window.status_label.text())
        np.testing.assert_allclose(
            self.window.arm.forward_kinematics(self.window.joint_angles)[-1],
            endpoint, atol=1e-5,
        )

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

    def test_trajectory_pen_click_drag_and_delete_anchor(self):
        self.window.show()
        self.application.processEvents()
        workspace = self.window.workspace
        workspace.set_trajectory_edit_mode(True)

        first = QPoint(120, 120)
        second = QPoint(230, 170)
        QTest.mouseClick(workspace, Qt.MouseButton.LeftButton, pos=first)
        QTest.mousePress(workspace, Qt.MouseButton.LeftButton, pos=second)
        QTest.mouseMove(workspace, QPoint(270, 145), delay=20)
        QTest.mouseRelease(workspace, Qt.MouseButton.LeftButton, pos=QPoint(270, 145))
        self.application.processEvents()

        anchors = workspace.path_anchors()
        self.assertEqual(len(anchors), 2)
        self.assertIsNone(anchors[0][1])
        self.assertIsNotNone(anchors[1][1])

        QTest.mouseClick(workspace, Qt.MouseButton.LeftButton, pos=first)
        QTest.keyClick(workspace, Qt.Key.Key_Delete)
        self.application.processEvents()
        self.assertEqual(len(workspace.path_anchors()), 1)

    def test_dragging_blank_workspace_pans_view_without_moving_arm(self):
        self.window.show()
        self.application.processEvents()
        workspace = self.window.workspace
        origin_before, _ = workspace._view_parameters()
        angles_before = self.window.joint_angles.copy()
        start = QPoint(40, 80)
        end = QPoint(95, 125)

        QTest.mousePress(workspace, Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(workspace, end, delay=20)
        QTest.mouseRelease(workspace, Qt.MouseButton.LeftButton, pos=end)
        self.application.processEvents()

        origin_after, _ = workspace._view_parameters()
        self.assertAlmostEqual(origin_after.x() - origin_before.x(), 55.0, delta=2.0)
        self.assertAlmostEqual(origin_after.y() - origin_before.y(), 45.0, delta=2.0)
        np.testing.assert_allclose(self.window.joint_angles, angles_before, atol=1e-12)
        self.assertFalse(workspace._panning)

    def test_path_sampling_preserves_endpoints_and_uniform_time(self):
        anchors = [
            (np.array([0.0, 0.0]), None, None),
            (np.array([2.0, 0.0]), None, None),
        ]
        points = sample_path_by_arclength(anchors, sample_count=21)
        self.assertEqual(points.shape, (21, 2))
        np.testing.assert_allclose(points[0], [0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(points[-1], [2.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(points[:, 1], 0.0, atol=1e-12)
        times, trajectory = parameterize_path(points, 4.0)
        self.assertEqual(len(times), len(trajectory))
        self.assertAlmostEqual(times[0], 0.0)
        self.assertAlmostEqual(times[-1], 4.0)
        self.assertTrue(np.all(np.diff(times) > 0.0))
        np.testing.assert_allclose(trajectory, points, atol=1e-12)

    def test_simulation_panel_prepares_start_and_fixes_endpoint(self):
        self.window._show_panel("simulation")
        self.window.workspace._path_anchors = [
            (np.array([1.0, 1.0]), None, None),
            (np.array([1.2, 1.0]), None, None),
        ]
        self.window.simulation_route_indices = [0, 1]
        self.window._prepare_simulation()
        self.assertTrue(self.window._simulation_prepared)
        self.assertIsNotNone(self.window._planned_joint_angles)
        self.assertEqual(self.window._planned_joint_angles.shape[1], self.window.arm.joint_count)
        self.assertIn(self.window.arm.joint_count, self.window.fixed_nodes)
        np.testing.assert_allclose(
            self.window.arm.forward_kinematics(self.window.joint_angles)[-1],
            self.window.fixed_nodes[self.window.arm.joint_count],
            atol=1e-6,
        )

    def test_toolbar_switches_shared_panel_pages(self):
        self.window._show_panel("simulation")
        self.assertIs(self.window._panel_stack.currentWidget(), self.window._panel_pages["simulation"])
        self.window._show_panel("properties")
        self.assertIs(self.window._panel_stack.currentWidget(), self.window._panel_pages["properties"])
        self.window._show_panel("robot")
        self.assertIs(self.window._panel_stack.currentWidget(), self.window._panel_pages["robot"])

    def test_workspace_selection_only_accepts_existing_path_anchors(self):
        workspace = self.window.workspace
        workspace._path_anchors = [
            (np.array([0.0, 0.5]), None, None),
            (np.array([1.0, 0.5]), None, None),
        ]
        workspace.set_path_point_selection_mode(True)
        selected = []
        workspace.path_anchor_selected.connect(selected.append)
        self.window.show()
        self.application.processEvents()
        origin, scale = workspace._view_parameters()
        anchor = workspace._world_to_screen(workspace._path_anchors[1][0], origin, scale)
        QTest.mouseClick(workspace, Qt.MouseButton.LeftButton, pos=QPoint(round(anchor.x()), round(anchor.y())))
        self.application.processEvents()
        self.assertEqual(selected, [1])
        non_anchor = workspace._world_to_screen([0.5, 0.9], origin, scale)
        QTest.mouseClick(workspace, Qt.MouseButton.LeftButton, pos=QPoint(round(non_anchor.x()), round(non_anchor.y())))
        self.application.processEvents()
        self.assertEqual(selected, [1])

    def test_route_selection_preserves_repeats_and_expands_reverse_segments(self):
        workspace = self.window.workspace
        workspace._path_anchors = [
            (np.array([1.0, 1.0]), np.array([-0.1, 0.0]), np.array([0.1, 0.0])),
            (np.array([1.2, 1.1]), np.array([-0.1, 0.0]), np.array([0.1, 0.0])),
            (np.array([1.4, 1.0]), np.array([-0.1, 0.0]), np.array([0.1, 0.0])),
            (np.array([1.6, 1.1]), np.array([-0.1, 0.0]), np.array([0.1, 0.0])),
        ]
        self.window.select_path_points_button.setChecked(True)
        for index in (0, 2, 0):
            self.window._on_path_anchor_selected(index)
        self.assertEqual(self.window.simulation_route_indices, [0, 2, 0])
        self.assertIn("1 → 3 → 1", self.window.simulation_path_points_label.text())
        anchors = self.window._simulation_anchor_points()
        self.assertEqual(len(anchors), 5)
        np.testing.assert_allclose(
            np.array([item[0] for item in anchors]),
            [[1.0, 1.0], [1.2, 1.1], [1.4, 1.0], [1.2, 1.1], [1.0, 1.0]],
        )
        np.testing.assert_allclose(anchors[3][1], [0.1, 0.0])
        np.testing.assert_allclose(anchors[3][2], [-0.1, 0.0])

    def test_simulation_requires_two_selected_path_points(self):
        self.window._prepare_simulation()
        self.assertFalse(self.window._simulation_timer.isActive())
        self.assertIn("选择路径点", self.window.status_label.text())

    def test_simulation_runs_through_reachable_path_and_finishes(self):
        workspace = self.window.workspace
        workspace._path_anchors = [
            (np.array([1.0, 1.0]), None, None),
            (np.array([1.2, 1.0]), None, None),
        ]
        self.window.simulation_route_indices = [0, 1]
        self.window.max_duration_control.setValue(0.5)
        self.window.max_duration_limit_enabled.setChecked(True)
        self.window._prepare_simulation()
        self.window._start_simulation()
        self.assertTrue(self.window._simulation_timer.isActive())
        max_steps = int(np.ceil(self.window._simulation_duration * 1000 / self.window._simulation_timer.interval())) + 2
        for _ in range(max_steps):
            if not self.window._simulation_timer.isActive():
                break
            self.window._advance_simulation()
        self.assertFalse(self.window._simulation_timer.isActive())
        self.assertAlmostEqual(self.window._simulation_time, 0.5, places=7)
        self.assertIn("完成", self.window.status_label.text())
        self.assertEqual(len(self.window._simulation_log_torques), len(self.window._simulation_log_times))
        self.assertEqual(
            np.asarray(self.window._simulation_log_torques).shape,
            (len(self.window._simulation_log_times), self.window.arm.joint_count),
        )

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


    def test_simulation_plot_records_actual_state_on_shared_time_axis(self):
        self.window._simulation_duration = 2.0
        self.window._simulation_time = 0.5
        self.window._record_simulation_state(0.5)

        actual_endpoint = self.window.arm.forward_kinematics(self.window.joint_angles)[-1]
        self.assertEqual(self.window._simulation_log_times, [0.5])
        np.testing.assert_allclose(self.window._simulation_log_positions[0][-1], actual_endpoint)
        np.testing.assert_allclose(self.window._simulation_log_angles[0], self.window.joint_angles)
        self.assertEqual(self.window.simulation_plot._current_time, 0.5)
        self.assertEqual(self.window.simulation_plot._duration, 2.0)

    def test_simulation_plot_data_is_reset_for_a_new_run(self):
        self.window._record_simulation_state(0.0)
        self.window._record_simulation_state(0.03)
        self.window._clear_simulation_log()

        self.assertEqual(self.window._simulation_log_times, [])
        self.assertEqual(self.window.simulation_plot._times.size, 0)
        self.assertEqual(self.window.simulation_plot._positions.shape[0], 0)

    def test_plot_logs_all_nodes_and_builds_variable_options(self):
        self.window._record_simulation_state(0.0)
        positions = self.window.simulation_plot._positions
        self.assertEqual(positions.shape, (1, self.window.arm.joint_count + 1, 2))
        options = dict(self.window._plot_variables())
        self.assertIn("node_x:1", options)
        self.assertIn("node_speed:3", options)
        self.assertIn("joint_velocity:1", options)

    def test_plot_window_keeps_fixed_size_and_scrolls_when_charts_are_added(self):
        self.window._open_plot_window()
        self.window._plot_window.resize(800, 500)
        self.application.processEvents()
        initial_size = self.window._plot_window.size()
        self.assertIsInstance(self.window._plot_scroll_area, QScrollArea)
        self.assertTrue(self.window._plot_scroll_area.widgetResizable())

        for _ in range(5):
            self.window._add_plot_chart()
        self.application.processEvents()

        self.assertEqual(self.window._plot_window.size(), initial_size)
        self.assertEqual(len(self.window._plot_widgets), 6)
        self.assertIs(self.window._plot_scroll_area.widget(), self.window._plot_charts_container)
        self.assertEqual(self.window._plot_charts_layout.count(), 6)
        self.assertGreater(
            self.window._plot_scroll_area.verticalScrollBar().maximum(), 0,
            "多个图表超出可视区域时，应出现可用的纵向滚动范围",
        )
        self.window._plot_window.close()

    def test_plot_computes_endpoint_path_length_and_node_speed(self):
        from simulation_plot import SimulationPlot

        plot = SimulationPlot()
        times = np.array([0.0, 1.0, 2.0])
        positions = np.array([
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 0.0]],
            [[0.0, 0.0], [2.0, 0.0]],
        ])
        angles = np.zeros((3, 1))
        plot.set_data(times, positions, angles, 1.0, 2.0)
        np.testing.assert_allclose(plot._values("path:1"), [0.0, 1.0, 2.0])
        np.testing.assert_allclose(plot._values("node_speed:1"), [1.0, 1.0, 1.0])
        np.testing.assert_allclose(plot._values("node_ax:1"), [0.0, 0.0, 0.0], atol=1e-10)


    def test_plot_can_read_joint_torque_series(self):
        from simulation_plot import SimulationPlot

        plot = SimulationPlot()
        times = np.array([0.0, 0.5, 1.0])
        positions = np.zeros((3, 3, 2))
        angles = np.zeros((3, 2))
        torques = np.array([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]])
        plot.set_data(times, positions, angles, 0.5, 1.0, torques)

        np.testing.assert_allclose(plot._values("joint_torque:0"), [1.0, 2.0, 3.0])
        np.testing.assert_allclose(plot._values("joint_torque:1"), [-1.0, -2.0, -3.0])
        plot.set_configuration("time", ["joint_torque:0"], "shared")
        plot.resize(640, 320)
        plot.show()
        self.application.processEvents()
        self.assertFalse(plot.grab().isNull())
        plot.close()

    def test_recorded_static_torques_match_dynamics_model(self):
        from dynamics import PlanarArmDynamics

        self.window._record_simulation_state(0.0)
        model = PlanarArmDynamics(
            self.window.arm.link_lengths,
            link_masses=self.window._link_masses,
            payload_mass=self.window._payload_mass,
        )
        expected = model.gravity_torques(self.window.joint_angles)
        self.assertEqual(len(self.window._simulation_log_torques), 1)
        np.testing.assert_allclose(self.window._simulation_log_torques[0], expected, atol=1e-10)
        self.assertIn("joint_torque:0", dict(self.window._plot_variables()))

    def test_incremental_torque_updates_match_full_trajectory_reference(self):
        from dynamics import PlanarArmDynamics
        from trajectory_dynamics import calculate_trajectory_torques

        times = np.linspace(0.0, 0.7, 8)
        angles = np.column_stack((
            0.2 + 0.3 * times + 0.1 * times**2,
            -0.4 + 0.15 * times - 0.05 * times**2,
            0.1 - 0.2 * times + 0.08 * times**2,
        ))
        model = PlanarArmDynamics(
            self.window.arm.link_lengths,
            link_masses=self.window._link_masses,
            payload_mass=self.window._payload_mass,
        )
        for index, time_value in enumerate(times):
            self.window.joint_angles = angles[index].copy()
            self.window._record_simulation_state(float(time_value))
            expected = calculate_trajectory_torques(
                model, times[:index + 1], angles[:index + 1]
            )
            np.testing.assert_allclose(
                np.asarray(self.window._simulation_log_torques),
                expected,
                atol=1e-9,
                err_msg=f"trajectory prefix length {index + 1}",
            )

    def test_payload_mass_change_recalculates_recorded_torques(self):
        self.window._record_simulation_state(0.0)
        original = np.asarray(self.window._simulation_log_torques[0]).copy()

        self.window._payload_mass_control.setValue(1.0)

        self.assertAlmostEqual(self.window._payload_mass, 1.0)
        self.assertFalse(np.allclose(self.window._simulation_log_torques[0], original))

    def test_link_count_change_preserves_link_masses_and_rebuilds_controls(self):
        self.window._link_mass_controls[0].setValue(2.5)
        self.window.link_count_control.setValue(4)
        self.window._set_link_count()

        self.assertEqual(len(self.window._link_mass_controls), 4)
        np.testing.assert_allclose(self.window._link_masses, [2.5, 1.0, 1.0, 1.0])
        self.assertEqual(self.window._simulation_log_times, [])

    def test_center_of_mass_and_inertia_controls_recalculate_torques(self):
        self.window._record_simulation_state(0.0)
        original = np.asarray(self.window._simulation_log_torques[0]).copy()

        self.window._com_fraction_controls[1].setValue(0.25)
        self.assertAlmostEqual(self.window._com_fractions[1], 0.25)
        self.assertFalse(np.allclose(self.window._simulation_log_torques[0], original))

        self.window._clear_simulation_log()
        self.window.joint_angles = np.radians([30.0, -20.0, 15.0])
        self.window._record_simulation_state(0.0)
        self.window.joint_angles = np.radians([35.0, -10.0, 20.0])
        self.window._record_simulation_state(0.1)
        self.window.joint_angles = np.radians([50.0, 5.0, 25.0])
        self.window._record_simulation_state(0.2)
        dynamic_original = np.asarray(self.window._simulation_log_torques).copy()

        self.window._link_inertia_controls[0].setValue(0.25)
        self.assertAlmostEqual(self.window._link_inertias[0], 0.25)
        self.assertFalse(np.allclose(self.window._simulation_log_torques, dynamic_original))

    def test_link_count_change_preserves_com_and_inertia_parameters(self):
        self.window._com_fraction_controls[0].setValue(0.3)
        self.window._link_inertia_controls[0].setValue(0.2)
        self.window.link_count_control.setValue(4)
        self.window._set_link_count()

        self.assertEqual(len(self.window._com_fraction_controls), 4)
        self.assertEqual(len(self.window._link_inertia_controls), 4)
        np.testing.assert_allclose(self.window._com_fractions, [0.3, 0.5, 0.5, 0.5])
        np.testing.assert_allclose(self.window._link_inertias, [0.2, 1 / 12, 1 / 12, 1 / 12], atol=5e-5)

    def _record_test_trajectory(self):
        times = [0.0, 0.1, 0.2, 0.3]
        angles = ([0.0, 0.0, 0.0], [0.1, -0.05, 0.02], [0.2, -0.1, 0.04], [0.3, -0.15, 0.06])
        for time_value, joint_angles in zip(times, angles):
            self.window.joint_angles = np.asarray(joint_angles, dtype=float)
            self.window._record_simulation_state(time_value)

    def test_feasibility_check_passes_when_limits_are_generous(self):
        self._record_test_trajectory()
        self.window.max_joint_speed_control.setValue(100.0)
        self.window.max_joint_acceleration_control.setValue(500.0)
        self.window.max_joint_torque_control.setValue(10000.0)

        self.window._assess_trajectory_feasibility()

        self.assertIn("均未超过设定限值", self.window.feasibility_result_label.text())
        self.assertIn("各关节峰值与发生时刻", self.window.feasibility_result_label.text())
        self.assertIn("关节 1：速度", self.window.feasibility_result_label.text())

    def test_feasibility_check_reports_speed_limit_violation(self):
        self._record_test_trajectory()
        self.window.speed_limit_enabled.setChecked(True)
        self.window.max_joint_speed_control.setValue(0.1)
        self.window.max_joint_acceleration_control.setValue(500.0)
        self.window.max_joint_torque_control.setValue(10000.0)

        self.window._assess_trajectory_feasibility()

        self.assertIn("超限：角速度", self.window.feasibility_result_label.text())
        self.assertIn("关节 1", self.window.feasibility_result_label.text())

    def test_feasibility_check_requires_at_least_three_samples(self):
        self.window._record_simulation_state(0.0)

        self.window._assess_trajectory_feasibility()

        self.assertIn("至少需要 3 个采样点", self.window.feasibility_result_label.text())

    def test_constraint_group_is_always_visible_without_planning_mode_selector(self):
        self.assertFalse(self.window.constraint_group.isHidden())
        self.assertFalse(hasattr(self.window, "trajectory_planning_mode"))
        self.assertTrue(hasattr(self.window, "max_duration_limit_enabled"))


class SharpCornerDetectionTests(unittest.TestCase):
    def test_detects_right_angle_corner_without_changing_samples(self):
        anchors = [
            (np.array([0.0, 0.0]), None, None),
            (np.array([1.0, 0.0]), None, None),
            (np.array([1.0, 1.0]), None, None),
        ]
        sampled = sample_path_by_arclength(anchors, sample_count=20, preserve_anchors=True)
        corners = find_sharp_corner_indices(anchors, sampled)
        self.assertEqual(len(corners), 1)
        np.testing.assert_allclose(sampled[corners[0]], [1.0, 0.0], atol=1e-10)

    def test_does_not_stop_at_collinear_anchor(self):
        anchors = [
            (np.array([0.0, 0.0]), None, None),
            (np.array([1.0, 0.0]), None, None),
            (np.array([2.0, 0.0]), None, None),
        ]
        sampled = sample_path_by_arclength(anchors, sample_count=20, preserve_anchors=True)
        self.assertEqual(find_sharp_corner_indices(anchors, sampled), ())


if __name__ == "__main__":
    unittest.main()
