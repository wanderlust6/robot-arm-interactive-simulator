import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ik_experiment_design import (
    ExperimentScenario,
    generate_initial_sensitivity_scenarios,
    generate_target_perturbation_scenarios,
    generate_workspace_boundary_scenarios,
    write_scenario_manifest,
)


class TestIKExperimentDesign(unittest.TestCase):
    def test_workspace_generator_covers_inner_outer_and_regular_regions(self):
        scenarios = generate_workspace_boundary_scenarios(
            link_lengths_m=[0.6, 0.4], sample_count=9, seed=7,
        )
        self.assertEqual({item.scenario_type for item in scenarios}, {"workspace_boundary"})
        self.assertEqual(len(scenarios), 3)
        inner, outer, regular = scenarios
        self.assertEqual(inner.metadata["region"], "inner_boundary")
        self.assertEqual(outer.metadata["region"], "outer_boundary")
        self.assertEqual(regular.metadata["region"], "regular_reachable")
        for scenario in scenarios:
            self.assertEqual(scenario.targets_m.shape, (9, 2))
            self.assertTrue(np.all(np.isfinite(scenario.targets_m)))
            self.assertEqual(scenario.initial_angles_rad.shape, (2,))
        radii = [np.linalg.norm(s.targets_m, axis=1) for s in scenarios]
        self.assertTrue(np.all(radii[0] < 0.2001))
        self.assertTrue(np.all(radii[1] > 0.9998))
        self.assertTrue(np.all((radii[2] > 0.2) & (radii[2] < 1.0)))

    def test_workspace_generator_is_reproducible_for_fixed_seed(self):
        first = generate_workspace_boundary_scenarios([0.6, 0.4], sample_count=13, seed=123)
        second = generate_workspace_boundary_scenarios([0.6, 0.4], sample_count=13, seed=123)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left.targets_m, right.targets_m)
            np.testing.assert_array_equal(left.initial_angles_rad, right.initial_angles_rad)
        other = generate_workspace_boundary_scenarios([0.6, 0.4], sample_count=13, seed=124)
        self.assertFalse(np.array_equal(first[0].targets_m, other[0].targets_m))

    def test_initial_sensitivity_keeps_targets_identical_and_varies_initial_pose(self):
        targets = np.array([[0.7, 0.1], [0.65, 0.2]])
        initial = np.array([0.2, 0.5])
        scenarios = generate_initial_sensitivity_scenarios(
            targets, initial, perturbation_levels_rad=[0.0, 0.05],
            repeats_per_level=3, seed=9,
        )
        self.assertEqual(len(scenarios), 6)
        for scenario in scenarios:
            np.testing.assert_array_equal(scenario.targets_m, targets)
            self.assertEqual(scenario.initial_angles_rad.shape, initial.shape)
        zero_level = scenarios[:3]
        for scenario in zero_level:
            np.testing.assert_array_equal(scenario.initial_angles_rad, initial)
        self.assertTrue(any(not np.array_equal(s.initial_angles_rad, initial) for s in scenarios[3:]))

    def test_target_perturbation_keeps_base_sequence_and_reproduces_noise(self):
        targets = np.array([[0.5, 0.1], [0.55, 0.12], [0.6, 0.15]])
        initial = np.array([0.1, 0.2])
        args = dict(perturbation_levels_m=[0.0, 0.01], repeats_per_level=2, seed=21)
        first = generate_target_perturbation_scenarios(targets, initial, **args)
        second = generate_target_perturbation_scenarios(targets, initial, **args)
        self.assertEqual(len(first), 4)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left.targets_m, right.targets_m)
            np.testing.assert_array_equal(left.initial_angles_rad, right.initial_angles_rad)
        for scenario in first[:2]:
            np.testing.assert_array_equal(scenario.targets_m, targets)
        self.assertTrue(any(not np.array_equal(s.targets_m, targets) for s in first[2:]))

    def test_invalid_inputs_are_rejected(self):
        invalid_calls = [
            lambda: generate_workspace_boundary_scenarios([0.6, 0.4], sample_count=2),
            lambda: generate_workspace_boundary_scenarios([0.6, -0.4]),
            lambda: generate_initial_sensitivity_scenarios([[0.1, 0.2]], [0.0, 0.0], [-0.1], 1),
            lambda: generate_initial_sensitivity_scenarios([[0.1, 0.2]], [0.0, 0.0], [0.1], 0),
            lambda: generate_target_perturbation_scenarios([[float("nan"), 0.0]], [0.0, 0.0], [0.1], 1),
        ]
        for call in invalid_calls:
            with self.subTest(call=call), self.assertRaises(ValueError):
                call()

    def test_scenario_validates_shapes_and_finite_values(self):
        with self.assertRaises(ValueError):
            ExperimentScenario("bad", "test", [[0.0, 1.0, 2.0]], [0.0, 0.0])
        with self.assertRaises(ValueError):
            ExperimentScenario("bad", "test", [[float("inf"), 0.0]], [0.0, 0.0])

    def test_scenario_builds_runner_compatible_config_without_mutating_base(self):
        base = {
            "experiment_id": "e1-auto",
            "scenario_id": "placeholder",
            "robot": {"link_lengths_m": [0.6, 0.4]},
            "solver": {"max_iterations": 100, "tolerance_m": 1e-6},
        }
        scenario = generate_workspace_boundary_scenarios([0.6, 0.4], sample_count=5, seed=3)[0]
        config = scenario.to_experiment_config(base)
        self.assertEqual(config["scenario_id"], scenario.scenario_id)
        np.testing.assert_array_equal(config["targets_m"], scenario.targets_m)
        np.testing.assert_array_equal(config["initial_angles_rad"], scenario.initial_angles_rad)
        self.assertEqual(config["experiment_id"], "e1-auto")
        self.assertEqual(base["scenario_id"], "placeholder")
        self.assertIn("scenario_metadata", config)
    def test_manifest_exports_scenario_metadata_and_arrays(self):
        scenarios = generate_workspace_boundary_scenarios([0.6, 0.4], sample_count=5, seed=2)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "scenario_manifest.json"
            write_scenario_manifest(output, scenarios, seed=2)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["seed"], 2)
            self.assertEqual(len(payload["scenarios"]), 3)
            self.assertEqual(payload["scenarios"][0]["targets_m"].__len__(), 5)
            self.assertEqual(payload["scenarios"][0]["initial_angles_rad"].__len__(), 2)

    def test_manifest_refuses_to_overwrite_existing_file(self):
        scenarios = generate_workspace_boundary_scenarios([0.6, 0.4], sample_count=5, seed=2)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "scenario_manifest.json"
            output.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                write_scenario_manifest(output, scenarios, seed=2)
            self.assertEqual(output.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()

