import unittest

import numpy as np

from planar_arm import PlanarArm
from run_ik_target_branch_reachability_stress import generate_branch_transition_path


class BranchTransitionPathTests(unittest.TestCase):
    def test_path_crosses_straight_arm_configuration_and_targets_match_fk(self):
        arm = PlanarArm((0.6, 0.4))
        joint_path, targets = generate_branch_transition_path(arm, sample_count=121)

        self.assertGreater(joint_path[0, 1], 0.0)
        self.assertAlmostEqual(joint_path[60, 1], 0.0, places=12)
        self.assertLess(joint_path[-1, 1], 0.0)
        expected = np.asarray([arm.forward_kinematics(q)[-1] for q in joint_path])
        np.testing.assert_allclose(targets, expected, rtol=0.0, atol=1e-12)

    def test_seed_cases_start_on_opposite_elbow_branches(self):
        arm = PlanarArm((0.6, 0.4))
        joint_path, _ = generate_branch_transition_path(arm, sample_count=121)
        same_elbow = joint_path[0]
        opposite_elbow = np.array([joint_path[0, 0], -joint_path[0, 1]])

        self.assertGreater(same_elbow[1], 0.0)
        self.assertLess(opposite_elbow[1], 0.0)
        self.assertAlmostEqual(same_elbow[0], opposite_elbow[0], places=12)
        self.assertAlmostEqual(same_elbow[1], -opposite_elbow[1], places=12)


if __name__ == "__main__":
    unittest.main()
