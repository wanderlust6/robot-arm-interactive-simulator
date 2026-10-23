import unittest
import tempfile
from pathlib import Path

import numpy as np

from ik_experiment_runner import validate_config, run_paired_sequence, write_results
from planar_arm import PlanarArm


class TestIKExperimentRunner(unittest.TestCase):
    def setUp(self):
        self.config = {
            "experiment_id": "e1-test",
            "scenario_id": "reachable-test",
            "robot": {"link_lengths_m": [0.6, 0.4]},
            "solver": {
                "max_iterations": 250,
                "tolerance_m": 1e-6,
                "base_damping": 1e-3,
                "sigma_threshold": 0.02,
                "gain": 0.05,
                "max_damping": 1.0,
            },
        }

    def test_validate_config_rejects_nonpositive_tolerance(self):
        config = {**self.config, "solver": {**self.config["solver"], "tolerance_m": 0}}
        with self.assertRaisesRegex(ValueError, "tolerance_m"):
            validate_config(config)

    def test_paired_run_uses_same_targets_and_returns_both_methods(self):
        arm = PlanarArm([0.6, 0.4])
        initial = np.array([0.3, 0.7])
        target = arm.forward_kinematics(initial)[-1]
        result = run_paired_sequence(arm, [target], initial, self.config)
        self.assertEqual(set(result["rows_by_method"]), {"current_dls", "sigma_adaptive_dls"})
        for method_rows in result["rows_by_method"].values():
            self.assertEqual(len(method_rows), 1)
            self.assertEqual(method_rows[0]["sample_index"], 0)
            self.assertLessEqual(method_rows[0]["actual_error_m"], self.config["solver"]["tolerance_m"])
            self.assertIn("min_singular_value_before", method_rows[0])
            self.assertIn("min_singular_value_after", method_rows[0])
        self.assertTrue(result["summaries"]["current_dls"]["sequence_success"])
        self.assertTrue(result["summaries"]["sigma_adaptive_dls"]["sequence_success"])

    def test_summary_marks_sequence_failed_if_any_point_fails(self):
        arm = PlanarArm([0.6, 0.4])
        initial = np.array([0.3, 0.7])
        targets = [arm.forward_kinematics(initial)[-1], np.array([5.0, 5.0])]
        config = {**self.config, "solver": {**self.config["solver"], "max_iterations": 1}}
        result = run_paired_sequence(arm, targets, initial, config)
        self.assertFalse(result["summaries"]["current_dls"]["sequence_success"])
        self.assertEqual(result["summaries"]["current_dls"]["sample_count"], 2)

    def test_config_rejects_empty_identifiers_and_malformed_sections(self):
        for config in (
            {**self.config, "experiment_id": " "},
            {**self.config, "scenario_id": ""},
            {**self.config, "robot": None},
            {**self.config, "solver": []},
            {**self.config, "robot": {"link_lengths_m": [True, 0.4]}},
            {**self.config, "solver": {**self.config["solver"], "max_iterations": True}},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_config(config)

    def test_unreachable_target_is_flagged_separately_from_solver_failure(self):
        arm = PlanarArm([0.6, 0.4])
        initial = np.array([0.3, 0.7])
        result = run_paired_sequence(arm, [np.array([1.2, 0.0])], initial, self.config)
        for method in ("current_dls", "sigma_adaptive_dls"):
            row = result["rows_by_method"][method][0]
            self.assertFalse(row["target_reachable"])
            self.assertIn("unreachable_target", row["failure_code"])
            self.assertFalse(result["summaries"][method]["sequence_success"])
            self.assertEqual(result["summaries"][method]["unreachable_target_count"], 1)

    def test_reachable_target_that_fails_solver_is_not_labeled_unreachable(self):
        arm = PlanarArm([0.6, 0.4])
        initial = np.array([0.3, 0.7])
        targets = [arm.forward_kinematics(initial)[-1], np.array([0.0, 1.0])]
        config = {**self.config, "solver": {**self.config["solver"], "max_iterations": 1, "tolerance_m": 1e-12}}
        result = run_paired_sequence(arm, targets, initial, config)
        rows = result["rows_by_method"]["current_dls"]
        self.assertTrue(all(row["target_reachable"] for row in rows))
        self.assertEqual(result["summaries"]["current_dls"]["unreachable_target_count"], 0)

    def test_rejects_empty_targets_before_running_solver(self):
        with self.assertRaisesRegex(ValueError, "targets"):
            run_paired_sequence(PlanarArm([0.6, 0.4]), [], [0.0, 0.0], self.config)

    def test_output_writer_rejects_nonfinite_json_metrics(self):
        arm = PlanarArm([0.6, 0.4])
        target = arm.forward_kinematics([0.0, 0.0])[-1]
        result = run_paired_sequence(arm, [target], [0.0, 0.0], self.config)
        result["summaries"]["current_dls"]["min_singular_value"] = float("nan")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "JSON compliant"):
                write_results(Path(tmp) / "results", self.config, result)


    def test_invalid_experiment_targets_are_rejected_by_schema_validation(self):
        from ik_experiment_runner import validate_experiment_inputs
        config = {**self.config, "targets_m": [[0.2, float("nan")]], "initial_angles_rad": [0.0, 0.0]}
        with self.assertRaisesRegex(ValueError, "targets_m"):
            validate_experiment_inputs(config)

    def test_experiment_initial_angles_must_match_robot_dof(self):
        from ik_experiment_runner import validate_experiment_inputs
        config = {**self.config, "targets_m": [[0.5, 0.2]], "initial_angles_rad": [0.0]}
        with self.assertRaisesRegex(ValueError, "initial_angles_rad"):
            validate_experiment_inputs(config)

    def test_invalid_json_output_does_not_leave_partial_directory(self):
        arm = PlanarArm([0.6, 0.4])
        target = arm.forward_kinematics([0.0, 0.0])[-1]
        result = run_paired_sequence(arm, [target], [0.0, 0.0], self.config)
        result["summaries"]["current_dls"]["min_singular_value"] = float("nan")
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "results"
            with self.assertRaises(ValueError):
                write_results(output, self.config, result)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
