import unittest
import numpy as np
from scipy.interpolate import CubicSpline
from planar_arm import PlanarArm
from joint_limits import JointLimits
from pathwise_ik_optimizer import optimize_joint_path

class PathwiseOptimizerTests(unittest.TestCase):
    def test_optimizes_joint_sequence_without_changing_cartesian_targets(self):
        arm=PlanarArm([1.0,0.8,0.6])
        anchors_q=np.radians([[25,-38,20],[42,-58,38],[54,-70,46],[38,-52,28],[25,-38,20]])
        anchors=np.array([arm.forward_kinematics(q)[-1] for q in anchors_q])
        spline=CubicSpline(np.linspace(0,1,len(anchors)),anchors,axis=0,bc_type='natural')
        targets=spline(np.linspace(0,1,17))
        initial=np.array([np.radians([25,-38,20]) for _ in targets])
        # Build a feasible initial path by continuation IK at each fixed Cartesian sample.
        from numerical_ik import solve_n_link_ik
        q=anchors_q[0].copy(); path=[]
        for p in targets:
            q,err,ok=solve_n_link_ik(arm,p,q)
            self.assertTrue(ok, msg=f'initial IK failed, error={err}')
            path.append(q.copy())
        limits=JointLimits(np.radians([-150,-145,-150]),np.radians([150,145,150]))
        before=np.array([np.linalg.norm(arm.forward_kinematics(q)[-1]-p) for q,p in zip(path,targets)])
        result=optimize_joint_path(arm,targets,np.asarray(path),limits=limits,max_iterations=120)
        self.assertTrue(result['success'],result['message'])
        self.assertLessEqual(float(np.max(result['endpoint_errors'])),2e-6)
        self.assertTrue(np.all(result['joint_angles'] >= limits.lower-1e-8))
        self.assertTrue(np.all(result['joint_angles'] <= limits.upper+1e-8))
        np.testing.assert_allclose(targets,spline(np.linspace(0,1,17)),atol=0,rtol=0)
        self.assertLessEqual(float(np.max(before)),2e-6)

if __name__=='__main__': unittest.main()
