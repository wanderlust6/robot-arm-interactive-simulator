import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from simulation_plot import SimulationPlot


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


if __name__ == "__main__":
    unittest.main()
