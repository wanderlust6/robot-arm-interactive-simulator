import csv
import tempfile
import unittest
from pathlib import Path

from visualize_ik_path_risk import build_html, load_groups, summarize_group


FIELDS = [
    "case", "method", "repeat", "sample_index", "segment_type", "seed_case",
    "noise_std_m", "target_x_m", "target_y_m", "target_radius_m",
    "geometrically_reachable", "initial_q1_rad", "initial_q2_rad",
    "joint_1_rad", "joint_2_rad", "elbow_sign", "converged",
    "reported_error_m", "actual_error_m", "solve_time_ms", "iterations",
    "peak_damping", "min_singular_value", "branch_transitions_in_sequence",
]


def row(index, **updates):
    result = {
        "case": "unit_case", "method": "current_dls", "repeat": "1",
        "sample_index": str(index), "segment_type": "test", "seed_case": "nominal",
        "noise_std_m": "0", "target_x_m": str(0.5 + index * 0.01),
        "target_y_m": "0.2", "target_radius_m": "0.54",
        "geometrically_reachable": "True", "initial_q1_rad": "0.2",
        "initial_q2_rad": "0.3", "joint_1_rad": str(0.2 + index * 0.01),
        "joint_2_rad": "0.3", "elbow_sign": "1", "converged": "True",
        "reported_error_m": "1e-7", "actual_error_m": "1e-7",
        "solve_time_ms": "0.1", "iterations": "", "peak_damping": "",
        "min_singular_value": "0.04", "branch_transitions_in_sequence": "0",
    }
    result.update({key: str(value) for key, value in updates.items()})
    return result


class PathRiskTests(unittest.TestCase):
    def load(self, rows):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "input.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return load_groups(path)

    def test_normal_point_has_no_risk_flags(self):
        groups = self.load([row(0), row(1)])
        self.assertEqual(groups[("unit_case", "current_dls", 1)][0]["risk_level"], "normal")

    def test_low_minimum_singular_value_is_high_risk(self):
        groups = self.load([row(0, min_singular_value="0.01")])
        point = groups[("unit_case", "current_dls", 1)][0]
        self.assertEqual(point["risk_level"], "high")
        self.assertIn("near_singularity", point["risk_flags"])

    def test_unreachable_target_is_critical(self):
        groups = self.load([row(0, geometrically_reachable="False", converged="False")])
        point = groups[("unit_case", "current_dls", 1)][0]
        self.assertEqual(point["risk_level"], "critical")
        self.assertIn("geometrically_unreachable", point["risk_flags"])

    def test_endpoint_error_over_tolerance_is_critical(self):
        groups = self.load([row(0, actual_error_m="0.00001")])
        self.assertEqual(groups[("unit_case", "current_dls", 1)][0]["risk_level"], "critical")

    def test_branch_transition_is_reported(self):
        groups = self.load([
            row(0, joint_2_rad="0.3", elbow_sign="1"),
            row(1, joint_2_rad="-0.3", elbow_sign="-1"),
        ])
        points = groups[("unit_case", "current_dls", 1)]
        self.assertIn("elbow_branch_transition", points[1]["risk_flags"])

    def test_summary_counts_match_point_labels(self):
        groups = self.load([row(0), row(1, min_singular_value="0.01"),
                            row(2, geometrically_reachable="False", converged="False")])
        summary = summarize_group(groups[("unit_case", "current_dls", 1)])
        self.assertEqual(summary["samples"], 3)
        self.assertEqual(summary["critical_points"], 1)
        self.assertEqual(summary["high_points"], 1)
        self.assertEqual(summary["normal_points"], 1)
        self.assertEqual(summary["unreachable"], 1)

    def test_dashboard_has_no_unfilled_data_marker(self):
        groups = self.load([row(0), row(1)])
        page = build_html(groups)
        self.assertIn("逆运动学路径级风险分析", page)
        self.assertIn("风险指标", page)
        self.assertNotIn("__DATA__", page)


if __name__ == "__main__":
    unittest.main()
