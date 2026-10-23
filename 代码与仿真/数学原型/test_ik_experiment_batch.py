import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ik_experiment_batch import run_experiment_batch
from ik_experiment_design import generate_workspace_boundary_scenarios


BASE_CONFIG = {
    "experiment_id": "batch-test",
    "scenario_id": "base",
    "robot": {"link_lengths_m": [0.6, 0.4]},
    "solver": {
        "max_iterations": 100, "tolerance_m": 1e-5,
        "base_damping": 1e-3, "sigma_threshold": 0.02,
        "gain": 0.05, "max_damping": 1.0,
    },
}


class TestIKExperimentBatch(unittest.TestCase):
    def test_batch_runs_each_scenario_and_writes_isolated_results(self):
        scenarios = generate_workspace_boundary_scenarios([0.6, 0.4], sample_count=5, seed=17)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "batch"
            result = run_experiment_batch(scenarios, BASE_CONFIG, output)
            self.assertEqual(result["scenario_count"], 3)
            self.assertEqual(len(result["scenarios"]), 3)
            for item in result["scenarios"]:
                scenario_dir = output / item["scenario_id"]
                self.assertTrue((scenario_dir / "point_results.csv").is_file())
                self.assertTrue((scenario_dir / "summary.json").is_file())
                self.assertIn("current_dls", item["methods"])
                self.assertIn("sigma_adaptive_dls", item["methods"])
            self.assertTrue((output / "batch_summary.json").is_file())
            self.assertTrue((output / "batch_summary.csv").is_file())
            payload = json.loads((output / "batch_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["scenario_count"], 3)

    def test_batch_rejects_duplicate_ids_before_creating_output(self):
        scenarios = generate_workspace_boundary_scenarios([0.6, 0.4], sample_count=5, seed=17)
        scenarios[1] = type(scenarios[0])(
            scenarios[0].scenario_id, scenarios[1].scenario_type,
            scenarios[1].targets_m, scenarios[1].initial_angles_rad, scenarios[1].metadata,
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "batch"
            with self.assertRaises(ValueError):
                run_experiment_batch(scenarios, BASE_CONFIG, output)
            self.assertFalse(output.exists())

    def test_batch_refuses_nonempty_output_directory(self):
        scenarios = generate_workspace_boundary_scenarios([0.6, 0.4], sample_count=5, seed=17)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "batch"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run_experiment_batch(scenarios, BASE_CONFIG, output)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_batch_does_not_mutate_base_config(self):
        scenarios = generate_workspace_boundary_scenarios([0.6, 0.4], sample_count=3, seed=1)
        original = json.loads(json.dumps(BASE_CONFIG))
        with tempfile.TemporaryDirectory() as tmp:
            run_experiment_batch(scenarios[:1], BASE_CONFIG, Path(tmp) / "batch")
        self.assertEqual(BASE_CONFIG, original)

    def test_batch_rejects_invalid_config_before_creating_output(self):
        scenarios = generate_workspace_boundary_scenarios([0.6, 0.4], sample_count=3, seed=1)
        invalid = json.loads(json.dumps(BASE_CONFIG))
        invalid["solver"]["tolerance_m"] = -1
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "batch"
            with self.assertRaises(ValueError):
                run_experiment_batch(scenarios, invalid, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
