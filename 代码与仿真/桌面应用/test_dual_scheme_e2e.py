import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from unittest.mock import patch
from PySide6.QtWidgets import QApplication

import app as app_module
from app import MainWindow


class DualSchemeEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow()

    def tearDown(self):
        self.window._simulation_timer.stop()
        self.window.close()

    def test_two_planners_prepare_play_and_record_independent_diagnostics(self):
        window = self.window
        window.workspace._path_anchors = [
            (np.array([1.0, 1.0]), None, None),
            (np.array([1.2, 1.0]), None, None),
        ]
        window.simulation_route_indices = [0, 1]
        window.motion_path_checks["continuous"].setChecked(True)
        window.motion_path_checks["joint_bias"].setChecked(True)

        window._prepare_simulation()

        self.assertTrue(window._simulation_prepared, window.constraint_result_label.text())
        self.assertEqual(set(window._planning_results), {"continuous", "joint_bias"})
        self.assertTrue(all(result.success for result in window._planning_results.values()))

        continuous = window._planning_results["continuous"]
        joint_bias = window._planning_results["joint_bias"]
        self.assertIsNot(continuous.plan, joint_bias.plan)
        self.assertIsNot(continuous.spline, joint_bias.spline)
        self.assertEqual(len(continuous.diagnostics["sample_times"]), len(continuous.plan.times))
        self.assertEqual(len(joint_bias.diagnostics["sample_times"]), len(joint_bias.plan.times))

        window._start_simulation()
        self.assertTrue(window._simulation_timer.isActive())
        # 启动入口会重新执行规划；之后断言必须引用本次播放实际使用的结果对象。
        continuous = window._planning_results["continuous"]
        joint_bias = window._planning_results["joint_bias"]

        steps = 0
        while window._simulation_timer.isActive() and steps < 1000:
            window._advance_simulation()
            steps += 1

        self.assertLess(steps, 1000, "双方案播放未在预期步数内结束")
        self.assertFalse(window._simulation_timer.isActive())
        self.assertAlmostEqual(window._simulation_time, window._simulation_duration, places=9)
        self.assertGreater(window._simulation_time, 0.0)
        self.assertGreater(len(window._simulation_log_times), 2)
        progress_values = []
        for key, result in window._planning_results.items():
            playback = result.diagnostics.get("playback", {})
            log = window._simulation_log_scheme_diagnostics[key]
            self.assertIn("joint_angles", playback)
            self.assertIn("joint_velocities", playback)
            self.assertIn("endpoint_error", playback)
            self.assertEqual(len(log["endpoint_errors"]), len(window._simulation_log_times))
            self.assertEqual(len(log["min_singular_values"]), len(window._simulation_log_times))
            self.assertEqual(len(log["condition_numbers"]), len(window._simulation_log_times))
            progress_values.append(playback["path_progress"])
            self.assertTrue(np.all(np.isfinite(playback["joint_angles"])))
            self.assertTrue(np.isfinite(playback["endpoint_error"]))

        # 播放按各方案自身的时间轴推进：每个方案取自身时刻，而不是统一到主方案进度。
        # 总时长不同的方案会在路径的不同进度上停下（较快的更早到达终点）。
        for key, result in window._planning_results.items():
            playback = result.diagnostics["playback"]
            plan = result.plan
            times = np.asarray(plan.times, dtype=float)
            self.assertAlmostEqual(
                playback["time"],
                float(np.clip(window._simulation_duration, times[0], times[-1])),
                places=9,
            )
            progress_samples = (
                np.asarray(plan.time_progress, dtype=float)
                if plan.time_progress is not None
                else np.linspace(0.0, 1.0, len(times))
            )
            self.assertAlmostEqual(
                playback["path_progress"],
                float(np.interp(playback["time"], times, progress_samples)),
                places=9,
            )
        self.assertTrue(all(0.0 <= value <= 1.0 for value in progress_values))
        np.testing.assert_allclose(
            window.workspace._shadow_joint_angles,
            joint_bias.diagnostics["playback"]["joint_angles"],
            atol=1e-10,
        )
        self.assertFalse(np.allclose(
            continuous.diagnostics["playback"]["joint_angles"],
            joint_bias.diagnostics["playback"]["joint_angles"],
        ))

        plot_series = window._plot_scheme_series()
        self.assertEqual(len(plot_series), 2)
        self.assertEqual(plot_series[0]["label"], continuous.label)
        self.assertEqual(plot_series[1]["label"], joint_bias.label)
        self.assertEqual(len(plot_series[0]["endpoint_errors"]), len(window._simulation_log_times))
        self.assertEqual(len(window.simulation_plot._values("scheme_endpoint_error:0")), len(window._simulation_log_times))
        self.assertEqual(len(window.simulation_plot._values("scheme_condition_number:1")), len(window._simulation_log_times))

    def test_csv_export_contains_both_scheme_diagnostic_series(self):
        import csv
        import tempfile
        from pathlib import Path

        window = self.window
        window.workspace._path_anchors = [
            (np.array([1.0, 1.0]), None, None),
            (np.array([1.2, 1.0]), None, None),
        ]
        window.simulation_route_indices = [0, 1]
        window.motion_path_checks["continuous"].setChecked(True)
        window.motion_path_checks["joint_bias"].setChecked(True)
        window._prepare_simulation()
        window._start_simulation()
        for _ in range(1000):
            if not window._simulation_timer.isActive():
                break
            window._advance_simulation()
        self.assertGreater(len(window._simulation_log_times), 0)
        window._open_data_window()
        window._data_columns = [
            "time", "scheme_endpoint_error:0", "scheme_endpoint_error:1",
            "scheme_condition_number:0", "scheme_condition_number:1",
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "scheme_diagnostics.csv")
            with patch.object(app_module.QFileDialog, "getSaveFileName", return_value=(path, "CSV 文件 (*.csv)")):
                window._export_data_csv()
            with open(path, "r", newline="", encoding="utf-8-sig") as stream:
                rows = list(csv.reader(stream))
        self.assertEqual(len(rows), len(window._simulation_log_times) + 1)
        self.assertEqual(len(rows[0]), len(window._data_columns))
        self.assertIn("连续性优先", rows[0][1])
        self.assertIn("参考构型偏置", rows[0][2])
        self.assertTrue(all(len(row) == len(rows[0]) for row in rows[1:]))

    def test_secondary_scheme_failure_preserves_primary_plan_and_reason(self):
        window = self.window
        window.workspace._path_anchors = [
            (np.array([1.0, 1.0]), None, None),
            (np.array([1.2, 1.0]), None, None),
        ]
        window.simulation_route_indices = [0, 1]
        window.motion_path_checks["continuous"].setChecked(True)
        window.motion_path_checks["joint_bias"].setChecked(True)
        original_planner = app_module.plan_cartesian_path
        calls = {"count": 0}

        def planner_with_secondary_failure(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 2:
                raise RuntimeError("测试用：第二方案规划失败")
            return original_planner(*args, **kwargs)

        with patch.object(app_module, "plan_cartesian_path", side_effect=planner_with_secondary_failure):
            window._prepare_simulation()

        self.assertTrue(window._planning_results["continuous"].success)
        secondary = window._planning_results["joint_bias"]
        self.assertFalse(secondary.success)
        self.assertIn("测试用：第二方案规划失败", secondary.message)
        self.assertEqual(window._shadow_plan_status, "failed")
        self.assertIn("第二方案规划失败", window.algorithm_comparison_label.text())
        self.assertTrue(window._simulation_prepared)


if __name__ == "__main__":
    unittest.main()
