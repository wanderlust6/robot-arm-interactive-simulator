import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiment_io import write_experiment_package, read_experiment_package


class TestExperimentPackage(unittest.TestCase):
    def test_write_and_read_experiment_package_round_trip(self):
        configuration = {
            "model": {"link_lengths_m": [1.0, 0.8]},
            "constraints": {},
        }
        times = [0.0, 0.5, 1.0]
        positions = np.array([
            [[0.0, 0.0], [1.0, 0.0], [1.8, 0.0]],
            [[0.0, 0.0], [0.9, 0.2], [1.6, 0.3]],
            [[0.0, 0.0], [0.8, 0.4], [1.4, 0.5]],
        ])
        angles = np.array([[0.0, 0.0], [0.2, -0.1], [0.4, -0.2]])
        velocities = np.ones_like(angles) * 0.2
        accelerations = np.ones_like(angles) * 0.1
        torques = np.ones_like(angles) * 1.5
        with tempfile.TemporaryDirectory() as directory:
            summary = write_experiment_package(
                directory, configuration, times, positions, angles, velocities,
                accelerations, torques,
            )
            self.assertEqual(summary["sample_count"], 3)
            self.assertTrue((Path(directory) / "README.md").exists())
            self.assertTrue((Path(directory) / "summary.json").exists())
            package = read_experiment_package(Path(directory) / "experiment.json")
        np.testing.assert_allclose(package["times"], times)
        np.testing.assert_allclose(package["positions"], positions)
        np.testing.assert_allclose(package["angles"], angles)
        np.testing.assert_allclose(package["velocities"], velocities)
        np.testing.assert_allclose(package["accelerations"], accelerations)
        np.testing.assert_allclose(package["torques"], torques)

    def test_package_can_be_loaded_in_a_separate_python_process(self):
        configuration = {
            "model": {"link_lengths_m": [1.0, 0.8]},
            "constraints": {"joint_speed": {"enabled": True, "value_rad_s": 2.0}},
            "trajectory": {"selected_route_indices": [0, 1, 0]},
        }
        times = np.array([0.0, 0.25, 0.5])
        positions = np.array([
            [[0.0, 0.0], [1.0, 0.0], [1.8, 0.0]],
            [[0.0, 0.0], [0.9, 0.1], [1.7, 0.2]],
            [[0.0, 0.0], [0.8, 0.2], [1.6, 0.3]],
        ])
        angles = np.array([[0.0, 0.1], [0.2, -0.1], [0.3, -0.2]])
        velocities = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        accelerations = velocities * 2.0
        torques = velocities * 3.0
        with tempfile.TemporaryDirectory() as directory:
            write_experiment_package(
                directory, configuration, times, positions, angles,
                velocities, accelerations, torques,
            )
            script = (
                "import json, sys; from experiment_io import read_experiment_package; "
                "p = read_experiment_package(sys.argv[1]); "
                "print(json.dumps({'configuration': p['configuration'], "
                "'times': p['times'], 'positions': p['positions'], "
                "'angles': p['angles'], 'velocities': p['velocities'], "
                "'accelerations': p['accelerations'], 'torques': p['torques']}))"
            )
            result = subprocess.run(
                [sys.executable, "-c", script, str(Path(directory) / "experiment.json")],
                cwd=Path(__file__).resolve().parent,
                capture_output=True, text=True, check=True,
            )
        restored = json.loads(result.stdout)
        self.assertEqual(restored["configuration"], configuration)
        for key, expected in (
            ("times", times), ("positions", positions), ("angles", angles),
            ("velocities", velocities), ("accelerations", accelerations), ("torques", torques),
        ):
            np.testing.assert_allclose(restored[key], expected, rtol=0, atol=1e-12)

    def test_rejects_inconsistent_sample_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                write_experiment_package(
                    directory, {"model": {"link_lengths_m": [1.0]}},
                    [0.0, 1.0], [], [], [], [], [],
                )


if __name__ == "__main__":
    unittest.main()
