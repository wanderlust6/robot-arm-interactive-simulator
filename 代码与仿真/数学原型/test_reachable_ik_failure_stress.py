import unittest

import numpy as np

from planar_arm import PlanarArm
from compare_damping_strategies import adaptive_ik
from run_ik_target_branch_reachability_stress import (
    generate_reachable_seed_profiles,
    generate_seed_stress_initials,
)


class ReachableSeedStressTests(unittest.TestCase):
    def test_profiles_generate_reachable_fk_targets_near_both_workspace_boundaries(self):
        arm = PlanarArm((0.6, 0.4))
        profiles = generate_reachable_seed_profiles(arm, sample_count=121)

        self.assertEqual(set(profiles), {"reachable_near_outer", "reachable_near_inner"})
        for joint_path, targets in profiles.values():
            self.assertEqual(joint_path.shape, (121, 2))
            self.assertEqual(targets.shape, (121, 2))
            expected = np.asarray([arm.forward_kinematics(q)[-1] for q in joint_path])
            np.testing.assert_allclose(targets, expected, rtol=0.0, atol=1e-12)
            radii = np.linalg.norm(targets, axis=1)
            self.assertTrue(np.all(radii >= abs(0.6 - 0.4) - 1e-12))
            self.assertTrue(np.all(radii <= 0.6 + 0.4 + 1e-12))

        outer_radius = np.linalg.norm(profiles["reachable_near_outer"][1], axis=1)
        inner_radius = np.linalg.norm(profiles["reachable_near_inner"][1], axis=1)
        self.assertGreater(np.min(outer_radius), 0.99)
        self.assertLess(np.max(inner_radius), 0.22)

    def test_initial_seeds_are_reproducible_and_shared_by_method_comparison(self):
        first = generate_seed_stress_initials(seed=20261022, repeats=12)
        second = generate_seed_stress_initials(seed=20261022, repeats=12)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (12, 2))
        self.assertTrue(np.all(first >= -np.pi))
        self.assertTrue(np.all(first <= np.pi))
        self.assertGreater(np.unique(first, axis=0).shape[0], 1)

    def test_adaptive_dls_recovers_from_rejected_steps_near_singularity(self):
        arm = PlanarArm((0.6, 0.4))
        initial = np.array([-2.2449399401825927, -3.1278733428510788])
        target = np.array([0.9874648447691414, 0.1422738673861409])

        solution, error, converged, _, _ = adaptive_ik(
            arm,
            target,
            initial,
            base_damping=1e-3,
            sigma_threshold=0.02,
            gain=0.05,
            max_damping=0.2,
        )

        self.assertTrue(converged)
        self.assertLessEqual(error, 1e-6)
        actual_error = np.linalg.norm(arm.forward_kinematics(solution)[-1] - target)
        self.assertLessEqual(actual_error, 1e-6)

    def test_default_damping_cap_handles_a_difficult_outer_boundary_seed(self):
        arm = PlanarArm((0.6, 0.4))
        seeds = generate_seed_stress_initials(seed=20261022, repeats=12)
        target = generate_reachable_seed_profiles(arm, sample_count=121)[
            "reachable_near_outer"
        ][1][0]

        solution, error, converged, _, _ = adaptive_ik(arm, target, seeds[7])

        self.assertTrue(converged)
        self.assertLessEqual(error, 1e-6)
        actual_error = np.linalg.norm(arm.forward_kinematics(solution)[-1] - target)
        self.assertLessEqual(actual_error, 1e-6)


if __name__ == "__main__":
    unittest.main()
