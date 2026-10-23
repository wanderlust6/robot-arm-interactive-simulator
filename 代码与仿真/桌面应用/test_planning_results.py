import unittest
from types import SimpleNamespace

import numpy as np

from planning_results import PlanningSchemeResult


class TestPlanningSchemeResult(unittest.TestCase):
    def _plan(self, success=True):
        times = np.array([0.0, 1.0, 2.0])
        angles = np.array([[0.0, 0.0], [0.5, -0.2], [1.0, -0.4]])
        velocities = np.array([[0.0, 0.0], [0.5, -0.2], [0.0, 0.0]])
        return SimpleNamespace(
            success=success,
            message="ok" if success else "constraint exceeded",
            times=times,
            joint_angles=angles,
            joint_velocities=velocities,
        )

    def test_successful_plan_keeps_plan_and_builds_spline(self):
        plan = self._plan()
        diagnostics = {"minimum_singular_value": 0.25, "playback": {}}
        result = PlanningSchemeResult.from_plan("continuous", "连续性优先", plan, diagnostics)
        self.assertTrue(result.success)
        self.assertIs(result.plan, plan)
        self.assertIs(result.diagnostics, diagnostics)
        self.assertEqual(result.key, "continuous")
        np.testing.assert_allclose(result.spline(0.0), plan.joint_angles[0])
        np.testing.assert_allclose(result.spline(2.0), plan.joint_angles[-1])

    def test_failed_plan_preserves_diagnostic_message(self):
        plan = self._plan(success=False)
        result = PlanningSchemeResult.from_plan("joint_bias", "冗余关节构型偏置", plan)
        self.assertFalse(result.success)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.message, "constraint exceeded")
        self.assertIs(result.plan, plan)
        self.assertIsNone(result.spline)

    def test_exception_result_is_recorded_without_plan(self):
        error = ValueError("IK did not converge")
        result = PlanningSchemeResult.failed("joint_bias", "冗余关节构型偏置", error)
        self.assertFalse(result.success)
        self.assertIsNone(result.plan)
        self.assertIsNone(result.spline)
        self.assertIn("IK did not converge", result.message)


if __name__ == "__main__":
    unittest.main()
