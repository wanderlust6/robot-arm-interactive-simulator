import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from PySide6.QtWidgets import QApplication

from app import MainWindow
from experiment_panel import AutomaticExperimentDialog


class TestAutomaticExperimentDialog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def test_dialog_starts_with_clear_run_action_and_empty_results(self):
        dialog = AutomaticExperimentDialog()
        try:
            self.assertIsNotNone(dialog.run_button)
            self.assertGreater(dialog.results_table.columnCount(), 0)
            self.assertEqual(dialog.results_table.rowCount(), 0)
            self.assertIn("尚未运行", dialog.status_label.text())
        finally:
            dialog.close()

    def test_run_button_executes_small_batch_and_displays_rows(self):
        dialog = AutomaticExperimentDialog()
        try:
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "batch"
                with patch.object(dialog, "_new_output_directory", return_value=output):
                    dialog._run_experiment_batch()
                self.assertEqual(dialog.results_table.rowCount(), 6)
                self.assertEqual(dialog.visualization.scenario_combo.count(), 3)
                self.assertEqual(len(dialog.visualization.left_view.target_points), 9)
                self.assertEqual(len(dialog.visualization.error_view.series), 2)
                self.assertEqual(dialog.visualization.slider.maximum(), 8)
                dialog.visualization.left_view.resize(360, 280)
                self.assertFalse(dialog.visualization.left_view.grab().isNull())
                dialog.visualization.error_view.resize(700, 220)
                self.assertFalse(dialog.visualization.error_view.grab().isNull())
                self.assertIn("实验完成", dialog.status_label.text())
                self.assertTrue((output / "batch_summary.json").exists())
                self.assertTrue((output / "batch_summary.csv").exists())
        finally:
            dialog.close()

if __name__ == "__main__":
    unittest.main()
