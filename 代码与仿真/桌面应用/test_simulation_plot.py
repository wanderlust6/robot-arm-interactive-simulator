import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

from simulation_plot import SimulationPlot, decimate_by_pixel


class PixelDecimationTests(unittest.TestCase):
    """按像素列抽取：只减少绘制点数，不能改变任何数值或极值包络。"""

    def test_small_series_is_returned_unchanged(self):
        points = [QPointF(float(index), float(index % 3)) for index in range(10)]
        reduced, decimated = decimate_by_pixel(points, column_count=100)
        self.assertFalse(decimated)
        self.assertEqual(reduced, points)

    def test_column_min_and_max_are_preserved(self):
        # 每列 20 个点，人为放入一个尖峰和一个谷值，抽取后必须仍存在。
        points = []
        for column in range(50):
            for offset in range(20):
                points.append(QPointF(column + 0.01 * offset, 0.0))
        points[205] = QPointF(points[205].x(), 5.0)     # 尖峰
        points[506] = QPointF(points[506].x(), -4.0)    # 谷值

        reduced, decimated = decimate_by_pixel(points, column_count=50)

        self.assertTrue(decimated)
        self.assertLess(len(reduced), len(points) // 5)
        self.assertAlmostEqual(max(p.y() for p in reduced), 5.0)
        self.assertAlmostEqual(min(p.y() for p in reduced), -4.0)

    def test_first_and_last_points_are_kept(self):
        points = [QPointF(index * 0.001, np.sin(index)) for index in range(5000)]
        reduced, decimated = decimate_by_pixel(points, column_count=300)
        self.assertTrue(decimated)
        self.assertIs(reduced[0], points[0])
        self.assertIs(reduced[-1], points[-1])

    def test_nan_gaps_stay_gaps(self):
        points = [QPointF(float(index), 0.0) for index in range(40)]
        points[15] = None
        reduced, decimated = decimate_by_pixel(points, column_count=4)
        self.assertTrue(decimated)
        self.assertIn(None, reduced)
        self.assertEqual(sum(1 for point in reduced if point is None), 1)

    def test_every_column_keeps_its_extremes(self):
        rng = np.random.default_rng(3)
        values = rng.normal(size=4000)
        points = [QPointF(index * 0.05, float(value)) for index, value in enumerate(values)]
        reduced, decimated = decimate_by_pixel(points, column_count=100)
        self.assertTrue(decimated)

        original_by_column = {}
        for point in points:
            original_by_column.setdefault(int(point.x() // 1), []).append(point.y())
        reduced_by_column = {}
        for point in reduced:
            reduced_by_column.setdefault(int(point.x() // 1), []).append(point.y())

        for column, original_values in original_by_column.items():
            kept = reduced_by_column.get(column, [])
            self.assertTrue(kept, f"列 {column} 没有保留任何点")
            self.assertAlmostEqual(min(kept), min(original_values))
            self.assertAlmostEqual(max(kept), max(original_values))


class DisplaySmoothingTests(unittest.TestCase):
    """显示平滑只作用于绘图路径，不能改变数据表/导出所用的原始值。"""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.plot = SimulationPlot()
        self.plot.resize(640, 360)
        rng = np.random.default_rng(5)
        self.times = np.linspace(0.0, 2.0, 400)
        self.noisy = np.sin(self.times * 6.0) + rng.normal(scale=0.05, size=self.times.size)
        self.plot.set_data(
            self.times,
            np.zeros((self.times.size, 2, 2)),
            np.column_stack((self.noisy, self.noisy)),
            0.0,
            2.0,
            joint_velocities=np.column_stack((self.noisy, self.noisy)),
            joint_accelerations=np.column_stack((self.noisy, self.noisy)),
        )

    def test_smoothing_reduces_zigzag_without_changing_length(self):
        raw = self.plot._values("joint_velocity:0")
        self.plot.set_smoothing(True, 21)
        smoothed = self.plot._values("joint_velocity:0", smooth=True)
        self.assertEqual(len(smoothed), len(raw))
        raw_zigzag = float(np.mean(np.abs(np.diff(raw, n=2))))
        smooth_zigzag = float(np.mean(np.abs(np.diff(smoothed, n=2))))
        self.assertLess(smooth_zigzag, raw_zigzag * 0.5)

    def test_raw_accessor_is_unaffected_by_smoothing(self):
        raw_before = self.plot._values("joint_velocity:0").copy()
        self.plot.set_smoothing(True, 21)
        raw_after = self.plot._values("joint_velocity:0")
        np.testing.assert_allclose(raw_after, raw_before)
        self.assertFalse(np.allclose(self.plot._values("joint_velocity:0", smooth=True), raw_before))

    def test_smoothing_disabled_means_smooth_request_is_raw(self):
        self.plot.set_smoothing(False, 21)
        raw = self.plot._values("joint_velocity:0")
        requested = self.plot._values("joint_velocity:0", smooth=True)
        np.testing.assert_allclose(requested, raw)

    def test_nan_gaps_are_preserved(self):
        from simulation_plot import smooth_series

        values = np.linspace(0.0, 1.0, 50)
        values[20:24] = np.nan
        smoothed = smooth_series(values, 11)
        self.assertTrue(np.all(np.isnan(smoothed[20:24])))
        self.assertTrue(np.all(np.isfinite(smoothed[:20])))
        self.assertTrue(np.all(np.isfinite(smoothed[24:])))

    def test_short_series_and_invalid_window_are_returned_unchanged(self):
        from simulation_plot import smooth_series

        short = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(smooth_series(short, 11), short)

    def test_paint_marks_smoothed_display(self):
        self.plot.set_smoothing(True, 21)
        self.plot.set_configuration("time", ["joint_velocity:0"], "separate")
        self.assertFalse(self.plot.grab().isNull())
        self.assertTrue(self.plot._display_smoothed)

    def test_data_table_values_stay_raw_while_display_is_smoothed(self):
        """界面上的数据表格走默认平滑关闭的取值路径。"""
        self.plot.set_smoothing(True, 21)
        raw = self.plot._values("joint_accel:0").copy()
        # 数据表/导出使用的就是默认调用（smooth 缺省 False）
        np.testing.assert_allclose(self.plot._values("joint_accel:0"), raw)
        smoothed = self.plot._values("joint_accel:0", smooth=True)
        self.assertFalse(np.allclose(smoothed, raw))


class SimulationPlotSchemeDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.plot = SimulationPlot()
        self.plot.resize(640, 360)
        self.times = np.linspace(0.0, 1.0, 11)
        self.positions = np.column_stack((self.times, self.times ** 2))[:, None, :]
        self.angles = np.column_stack((self.times, -self.times))
        self.series = [{
            "key": "continuous",
            "label": "连续性优先",
            "endpoint_errors": (self.times * 0.01).tolist(),
            "min_singular_values": (1.0 - self.times * 0.5).tolist(),
            "condition_numbers": (1.0 + self.times).tolist(),
        }]
        self.plot.set_data(
            self.times, self.positions, self.angles, 0.4, 1.0,
            scheme_series=self.series,
        )

    def test_scheme_diagnostic_series_match_simulation_samples(self):
        errors = self.plot._values("scheme_endpoint_error:0")
        singular_values = self.plot._values("scheme_min_singular_value:0")
        condition_numbers = self.plot._values("scheme_condition_number:0")

        self.assertEqual(len(errors), len(self.times))
        self.assertEqual(len(singular_values), len(self.times))
        self.assertEqual(len(condition_numbers), len(self.times))
        np.testing.assert_allclose(errors, self.times * 0.01)
        np.testing.assert_allclose(singular_values, 1.0 - self.times * 0.5)
        np.testing.assert_allclose(condition_numbers, 1.0 + self.times)

    def test_scheme_diagnostic_plot_renders(self):
        self.plot.set_configuration(
            "time",
            ["scheme_endpoint_error:0", "scheme_min_singular_value:0"],
            "separate",
        )
        self.assertFalse(self.plot.grab().isNull())

    def test_dense_series_renders_with_pixel_decimation(self):
        """上千个采样点在窄控件里绘制时应启用抽取，且仍然渲染成功。"""
        count = 4000
        times = np.linspace(0.0, 10.0, count)
        angles = np.column_stack((np.sin(times), -np.cos(times), np.sin(2 * times)))
        velocities = np.column_stack((np.cos(times), np.sin(times), 2 * np.cos(2 * times)))
        self.plot.set_data(
            times,
            np.zeros((count, 4, 2)),
            angles,
            0.0,
            10.0,
            joint_velocities=velocities,
            joint_accelerations=np.zeros((count, 3)),
        )
        self.plot.set_configuration("time", ["joint_velocity:0"], "separate")
        self.plot.resize(400, 240)
        image = self.plot.grab()
        self.assertFalse(image.isNull())
        self.assertTrue(self.plot._display_decimated)
        # 抽取只影响绘制：曲线取数仍返回全部采样点。
        self.assertEqual(len(self.plot._values("joint_velocity:0")), count)


if __name__ == "__main__":
    unittest.main()
