import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from app import PATH_SCHEME_ORDER, MainWindow
from joint_limits import JointLimits
from numerical_ik import solve_n_link_ik
from planar_arm import PlanarArm


class PathSchemeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow()
        self.arm = PlanarArm([1.0, 0.8, 0.6])
        self.initial = np.array([0.45, -0.7, 0.35])
        self.target = self.arm.forward_kinematics(self.initial)[-1]
        self.limits = JointLimits(
            np.radians([-150.0, -150.0, -150.0]),
            np.radians([150.0, 150.0, 150.0]),
        )
        start = self.arm.forward_kinematics(self.initial)[-1]
        end = self.arm.forward_kinematics(self.initial + np.radians([8.0, -12.0, 10.0]))[-1]
        self.targets = np.linspace(start, end, 21)

    def tearDown(self):
        self.window._simulation_timer.stop()
        self.window.close()

    def _run_path(self, solver, limits=None):
        q = self.initial.copy()
        path = []
        for target in self.targets:
            q_new, error, ok = solver(self.arm, target, q, limits)
            self.assertTrue(ok, f"求解未收敛，末端误差 {error:.3e}")
            self.assertLessEqual(error, 1e-5)
            q = np.asarray(q_new, dtype=float)
            path.append(q.copy())
        return np.asarray(path)

    def test_all_schemes_are_available_in_ui(self):
        self.assertEqual(tuple(self.window.motion_path_checks), PATH_SCHEME_ORDER)
        self.assertEqual(
            set(self.window.motion_path_checks),
            {
                "continuous",
                "joint_bias",
                "weighted",
                "joint_limit_avoidance",
            },
        )
        # 整段平滑优化属于离线研究工具：240 采样点下约需 200 s，
        # 放进同步规划界面会冻结窗口，因此不提供界面入口。
        self.assertNotIn("pathwise_optimization", self.window.motion_path_checks)
        # 奇异值自适应阻尼在常规路径上与基线逐位相同（实测 4e-8°），
        # 只作为近奇异专用策略保留在离线脚本中。
        self.assertNotIn("adaptive_dls", self.window.motion_path_checks)

    def test_adaptive_dls_returns_a_valid_solution(self):
        target = self.arm.forward_kinematics(np.array([0.5, -0.6, 0.25]))[-1]
        q, error, ok = self.window._adaptive_dls_solver(self.arm, target, self.initial, self.limits)
        self.assertTrue(ok)
        self.assertLessEqual(error, 1e-6)
        self.assertTrue(self.limits.contains(q))

    def test_joint_limit_avoidance_respects_limits_and_task(self):
        target = self.arm.forward_kinematics(np.array([0.35, -0.55, 0.4]))[-1]
        q, error, ok = self.window._joint_limit_avoidance_solver(self.arm, target, self.initial, self.limits)
        self.assertTrue(ok)
        self.assertLessEqual(error, 1e-6)
        self.assertTrue(self.limits.contains(q))

    def test_weighted_joint_solver_converges_and_respects_limits(self):
        target = self.arm.forward_kinematics(np.array([0.5, -0.6, 0.25]))[-1]
        q, error, ok = self.window._weighted_joint_solver(self.arm, target, self.initial, self.limits)
        self.assertTrue(ok)
        self.assertLessEqual(error, 1e-6)
        self.assertTrue(self.limits.contains(q))
        self.assertTrue(np.all(np.isfinite(q)))

    def test_weighted_scheme_visibly_differs_from_baseline_on_same_path(self):
        """同一末端路径下，加权关节分配必须给出明显不同的关节路径。"""
        baseline = self._run_path(solve_n_link_ik, self.limits)
        weighted = self._run_path(self.window._weighted_joint_solver, self.limits)

        residuals = [
            float(np.linalg.norm(self.arm.forward_kinematics(q)[-1] - target))
            for q, target in zip(weighted, self.targets)
        ]
        self.assertLessEqual(max(residuals), 1e-5)

        difference_deg = np.degrees(np.abs(weighted - baseline))
        self.assertGreater(
            float(np.max(difference_deg)), 5.0,
            "加权关节分配与基线的关节路径差异过小，说明它退化成了基线求解器",
        )

    def test_joint_limit_avoidance_differs_from_baseline_when_limits_enforced(self):
        """启用限位后，零空间排斥项必须真正改变关节路径，而不是被自身判据否决。"""
        baseline = self._run_path(solve_n_link_ik, self.limits)
        avoidance = self._run_path(self.window._joint_limit_avoidance_solver, self.limits)
        difference_deg = np.degrees(np.abs(avoidance - baseline))
        self.assertGreater(
            float(np.max(difference_deg)), 0.5,
            "限位规避与基线完全相同，说明零空间步被误差判据否决了",
        )

    def test_joint_limit_avoidance_is_not_applicable_without_limits(self):
        window = self.window
        window.workspace._path_anchors = [
            (np.array([1.0, 1.0]), None, None),
            (np.array([1.2, 1.0]), None, None),
        ]
        window.simulation_route_indices = [0, 1]
        for control in window.motion_path_checks.values():
            control.setChecked(False)
        window.motion_path_checks["continuous"].setChecked(True)
        window.motion_path_checks["joint_limit_avoidance"].setChecked(True)

        window._prepare_simulation()

        self.assertTrue(window._simulation_prepared, window.constraint_result_label.text())
        self.assertEqual(window._primary_scheme_key, "continuous")
        skipped = window._planning_results["joint_limit_avoidance"]
        self.assertEqual(skipped.status, "not_applicable")
        self.assertFalse(skipped.success)
        self.assertIn("未启用关节限位", skipped.message)
        self.assertIn("不适用", window.constraint_result_label.text())
        arms = {arm["key"]: arm for arm in window.workspace._scheme_arms}
        self.assertEqual(arms["joint_limit_avoidance"]["status"], "not_applicable")
        self.assertIsNone(arms["joint_limit_avoidance"]["angles"])

    def test_selected_schemes_are_all_exposed_as_overlay_arms(self):
        window = self.window
        window.workspace._path_anchors = [
            (np.array([1.0, 1.0]), None, None),
            (np.array([1.2, 1.0]), None, None),
        ]
        window.simulation_route_indices = [0, 1]
        for control in window.motion_path_checks.values():
            control.setChecked(False)
        for key in ("continuous", "joint_bias", "weighted"):
            window.motion_path_checks[key].setChecked(True)

        window._prepare_simulation()

        self.assertTrue(window._simulation_prepared, window.constraint_result_label.text())
        self.assertEqual(window._primary_scheme_key, "continuous")
        statuses = {arm["key"]: arm["status"] for arm in window.workspace._scheme_arms}
        self.assertEqual(statuses["continuous"], "primary")
        # 主方案之外的两个方案都必须作为可绘制叠加层出现，而不是只保留一个影子臂。
        drawable = [
            arm for arm in window.workspace._scheme_arms
            if arm["role"] != "primary" and arm["angles"] is not None
        ]
        self.assertEqual(len(drawable), 2)

        # 起点被 pose_constraints 固定在相同姿态，各方案在 t=0 必然重合；
        # 差异必须在路径中段才体现出来。
        start_angles = {
            arm["key"]: np.asarray(arm["angles"], dtype=float) for arm in drawable
        }
        np.testing.assert_allclose(
            start_angles["joint_bias"], start_angles["weighted"], atol=1e-9
        )

        mid_time = float(window._simulation_duration) * 0.5
        window._update_scheme_playback(mid_time)
        window._refresh_scheme_arms()
        mid_angles = {
            arm["key"]: np.asarray(arm["angles"], dtype=float)
            for arm in window.workspace._scheme_arms
            if arm["role"] != "primary" and arm["angles"] is not None
        }
        self.assertFalse(
            np.allclose(mid_angles["joint_bias"], mid_angles["weighted"]),
            "路径中段两个方案仍完全重合，说明叠加层没有使用各自的关节轨迹",
        )
        # 每个成功方案都要记录相对主方案的关节差，供界面显示与判断"是否真的不同"。
        for key in ("joint_bias", "weighted"):
            self.assertIn("max_joint_diff_deg", window._planning_results[key].diagnostics)


    def test_overlay_arms_follow_pose_adjustment_after_prepare(self):
        """准备后调整姿态时，叠加臂应跟随当前末端目标重新求解，而不是停在旧姿态。"""
        window = self.window
        window.workspace._path_anchors = [
            (np.array([1.0, 1.0]), None, None),
            (np.array([1.2, 1.0]), None, None),
        ]
        window.simulation_route_indices = [0, 1]
        for control in window.motion_path_checks.values():
            control.setChecked(False)
        for key in ("continuous", "joint_bias", "weighted"):
            window.motion_path_checks[key].setChecked(True)

        window._prepare_simulation()
        self.assertTrue(window._simulation_prepared, window.constraint_result_label.text())

        def overlay_arms():
            return {
                arm["key"]: arm
                for arm in window.workspace._scheme_arms
                if arm["role"] != "primary"
            }

        before = {
            key: np.asarray(arm["angles"], dtype=float).copy()
            for key, arm in overlay_arms().items()
            if arm["angles"] is not None
        }
        self.assertEqual(set(before), {"joint_bias", "weighted"})

        window._on_target_dragged(1.05, 1.02)

        self.assertFalse(window._simulation_prepared)
        self.assertIn("姿态已调整", window.constraint_result_label.text())

        after = overlay_arms()
        for key, arm in after.items():
            self.assertEqual(arm["status"], "preview", f"{key} 未进入跟随预览状态")
            self.assertIsNotNone(arm["angles"], f"{key} 在跟随预览时没有关节构型")
            endpoint = window.arm.forward_kinematics(arm["angles"])[-1]
            np.testing.assert_allclose(
                endpoint, window.target, atol=1e-5,
                err_msg=f"{key} 的叠加臂没有跟到当前末端目标",
            )
            self.assertFalse(
                np.allclose(arm["angles"], before[key]),
                f"{key} 的叠加臂仍停在调整前的姿态",
            )

        # 主机械臂自身仍由当前关节角决定，叠加臂只是跟随它的末端目标。
        self.assertTrue(np.all(np.isfinite(window.joint_angles)))


    def test_scheme_selector_lists_current_schemes(self):
        """路径方案下拉框必须列出当前方案，而不是写死的两项。"""
        from PySide6.QtWidgets import QComboBox

        window = self.window
        combo = QComboBox()
        window._populate_scheme_combo(combo)
        self.assertEqual(
            [combo.itemData(index) for index in range(combo.count())],
            list(PATH_SCHEME_ORDER),
        )

        window.workspace._path_anchors = [
            (np.array([1.0, 1.0]), None, None),
            (np.array([1.2, 1.0]), None, None),
        ]
        window.simulation_route_indices = [0, 1]
        for control in window.motion_path_checks.values():
            control.setChecked(False)
        for key in ("continuous", "weighted"):
            window.motion_path_checks[key].setChecked(True)
        window._prepare_simulation()
        self.assertTrue(window._simulation_prepared, window.constraint_result_label.text())

        window._populate_scheme_combo(combo)
        self.assertEqual(
            [combo.itemData(index) for index in range(combo.count())],
            ["continuous", "weighted"],
        )
        self.assertEqual(combo.itemText(1), "加权关节分配")
        self.assertNotIn("关节偏置", [combo.itemText(i) for i in range(combo.count())])

    def test_axis_dialog_lists_only_current_schemes(self):
        """设置 X/Y 轴对话框的方案下拉框必须反映当前方案，而不是写死的两项。"""
        from PySide6.QtWidgets import QComboBox, QDialog

        window = self.window
        window.workspace._path_anchors = [
            (np.array([1.0, 1.0]), None, None),
            (np.array([1.2, 1.0]), None, None),
        ]
        window.simulation_route_indices = [0, 1]
        for control in window.motion_path_checks.values():
            control.setChecked(False)
        for key in ("continuous", "weighted"):
            window.motion_path_checks[key].setChecked(True)
        window._prepare_simulation()
        self.assertTrue(window._simulation_prepared, window.constraint_result_label.text())

        captured = []

        def fake_exec(dialog):
            for combo in dialog.findChildren(QComboBox):
                data = [combo.itemData(i) for i in range(combo.count())]
                if data and all(
                    isinstance(item, str) and item in PATH_SCHEME_ORDER for item in data
                ):
                    captured.append([combo.itemText(i) for i in range(combo.count())])
            return QDialog.DialogCode.Rejected

        original = QDialog.exec
        QDialog.exec = fake_exec
        try:
            window._set_plot_axis(0, "y", 0)
            window._set_plot_axis(0, "x")
        finally:
            QDialog.exec = original

        self.assertEqual(len(captured), 2, "X 轴与 Y 轴对话框都应包含方案下拉框")
        for texts in captured:
            self.assertEqual(texts, ["连续性优先", "加权关节分配"])
            self.assertNotIn("关节偏置", texts)

    def test_data_table_header_uses_current_scheme_label(self):
        window = self.window
        window.workspace._path_anchors = [
            (np.array([1.0, 1.0]), None, None),
            (np.array([1.2, 1.0]), None, None),
        ]
        window.simulation_route_indices = [0, 1]
        for control in window.motion_path_checks.values():
            control.setChecked(False)
        for key in ("continuous", "weighted"):
            window.motion_path_checks[key].setChecked(True)
        window._prepare_simulation()
        self.assertTrue(window._simulation_prepared, window.constraint_result_label.text())

        window._open_data_window()
        window._data_columns = [("weighted", "node_x:3")]
        window._refresh_data_table()
        header = window._data_table.horizontalHeaderItem(0).text()
        self.assertIn("加权关节分配", header)

        # 方案诊断列：方案由变量索引决定，即使下拉框选的是别的方案也不能错标。
        window._data_columns = [("weighted", "scheme_endpoint_error:0")]
        window._refresh_data_table()
        header = window._data_table.horizontalHeaderItem(0).text()
        self.assertIn("连续性优先", header)
        self.assertNotIn("加权关节分配", header)
        window._data_window.close()


    def test_node_speed_uses_analytic_kinematics(self):
        """节点速度/加速度必须来自雅可比解析式，而不是对位置做数值微分。"""
        window = self.window
        window.workspace._path_anchors = [
            (np.array([1.0, 1.0]), None, None),
            (np.array([1.2, 1.0]), None, None),
        ]
        window.simulation_route_indices = [0, 1]
        for control in window.motion_path_checks.values():
            control.setChecked(False)
        window.motion_path_checks["continuous"].setChecked(True)
        window._prepare_simulation()
        self.assertTrue(window._simulation_prepared, window.constraint_result_label.text())

        primary = window._planning_results[window._primary_scheme_key]
        for step in range(1, 8):
            sample_time = step * 0.01
            window.joint_angles = np.asarray(primary.spline(sample_time), dtype=float)
            window.target = window.arm.forward_kinematics(window.joint_angles)[-1]
            window._update_scheme_playback(sample_time)
            window._record_simulation_state(sample_time)

        window._refresh_simulation_plot()
        plot = window.simulation_plot
        joint_count = window.arm.joint_count
        for index in range(joint_count + 1):
            speeds = np.asarray(plot._values(f"node_speed:{index}", "continuous"), dtype=float)
            expected = np.asarray([
                float(np.linalg.norm(window.arm.node_velocities(q, qd)[index]))
                for q, qd in zip(window._simulation_log_angles, window._simulation_log_velocities)
            ])
            np.testing.assert_allclose(speeds, expected, atol=1e-9)

            accelerations = np.asarray(plot._values(f"node_accel:{index}", "continuous"), dtype=float)
            expected_acc = np.asarray([
                float(np.linalg.norm(window.arm.node_accelerations(q, qd, qdd)[index]))
                for q, qd, qdd in zip(
                    window._simulation_log_angles,
                    window._simulation_log_velocities,
                    window._simulation_log_accelerations,
                )
            ])
            np.testing.assert_allclose(accelerations, expected_acc, atol=1e-9)

    def test_playback_records_uniform_time_grid(self):
        """播放必须按固定步长推进：墙钟抖动不能污染记录的采样间隔。"""
        window = self.window
        window.workspace._path_anchors = [
            (np.array([1.0, 1.0]), None, None),
            (np.array([1.2, 1.0]), None, None),
        ]
        window.simulation_route_indices = [0, 1]
        for control in window.motion_path_checks.values():
            control.setChecked(False)
        window.motion_path_checks["continuous"].setChecked(True)
        window._prepare_simulation()
        self.assertTrue(window._simulation_prepared, window.constraint_result_label.text())

        window._simulation_time = 0.0
        window._clear_simulation_log()
        window._record_simulation_state(0.0)
        for _ in range(24):
            window._advance_simulation()
        window._simulation_timer.stop()

        times = np.asarray(window._simulation_log_times, dtype=float)
        self.assertGreater(len(times), 10)
        steps = np.diff(times)
        expected = window._simulation_timer.interval() / 1000.0 * window._playback_speed
        np.testing.assert_allclose(steps, expected, atol=1e-12)

    def test_diagnose_data_column_reports_ripple_metrics(self):
        """数据列诊断应给出采样间隔与锯齿指标，用于区分数据抖动与显示问题。"""
        from unittest.mock import patch

        from PySide6.QtWidgets import QMessageBox

        window = self.window
        window.workspace._path_anchors = [
            (np.array([1.0, 1.0]), None, None),
            (np.array([1.2, 1.0]), None, None),
        ]
        window.simulation_route_indices = [0, 1]
        for control in window.motion_path_checks.values():
            control.setChecked(False)
        window.motion_path_checks["continuous"].setChecked(True)
        window._prepare_simulation()
        self.assertTrue(window._simulation_prepared, window.constraint_result_label.text())

        # 准备阶段只记录首帧；补充若干帧，才有采样间隔与锯齿指标可算。
        primary = window._planning_results[window._primary_scheme_key]
        for step in range(1, 8):
            sample_time = step * 0.01
            window.joint_angles = np.asarray(primary.spline(sample_time), dtype=float)
            window.target = window.arm.forward_kinematics(window.joint_angles)[-1]
            window._update_scheme_playback(sample_time)
            window._record_simulation_state(sample_time)

        window._open_data_window()
        window._data_columns = [("continuous", "joint_velocity:0")]
        window._refresh_data_table()
        window._data_table.setCurrentCell(0, 0)

        captured = {}

        def fake_information(parent, title, text):
            captured["title"] = title
            captured["text"] = text
            return QMessageBox.StandardButton.Ok

        with patch.object(QMessageBox, "information", side_effect=fake_information):
            window._diagnose_current_data_column()

        self.assertEqual(captured.get("title"), "数据列诊断")
        text = captured.get("text", "")
        self.assertIn("采样间隔", text)
        self.assertIn("均匀重采样后二阶差分", text)
        self.assertIn("相对幅值", text)
        window._data_window.close()


if __name__ == "__main__":
    unittest.main()
