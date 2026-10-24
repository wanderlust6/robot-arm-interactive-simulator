import json, time
import numpy as np
from scipy.interpolate import CubicSpline
from planar_arm import PlanarArm
from joint_limits import JointLimits
from numerical_ik import solve_n_link_ik
from pathwise_ik_optimizer import optimize_joint_path
arm=PlanarArm([1.0,0.8,0.6])
anchors_q=np.radians([[25,-38,20],[42,-58,38],[54,-70,46],[38,-52,28],[25,-38,20]])
anchors=np.array([arm.forward_kinematics(q)[-1] for q in anchors_q])
spline=CubicSpline(np.linspace(0,1,len(anchors)),anchors,axis=0,bc_type='natural')
t=np.linspace(0,1,17); targets=spline(t)
q=anchors_q[0].copy(); initial=[]
for p in targets:
 q,e,ok=solve_n_link_ik(arm,p,q)
 if not ok: raise RuntimeError(f'initial IK failure {e}')
 initial.append(q.copy())
initial=np.asarray(initial); limits=JointLimits(np.radians([-150,-145,-150]),np.radians([150,145,150]))
def metrics(q):
 d1=np.diff(q,axis=0); d2=np.diff(q,n=2,axis=0); d3=np.diff(q,n=3,axis=0)
 errs=np.array([np.linalg.norm(arm.forward_kinematics(qi)[-1]-p) for qi,p in zip(q,targets)])
 return {'max_endpoint_error_m':float(errs.max()),'sum_joint_step_norm_rad':float(np.linalg.norm(d1,axis=1).sum()),'sum_squared_second_difference':float(np.sum(d2*d2)),'sum_squared_third_difference':float(np.sum(d3*d3)),'max_joint_step_rad':float(np.linalg.norm(d1,axis=1).max()),'minimum_limit_margin_deg':float(np.degrees(np.minimum(q-limits.lower,limits.upper-q).min()))}
start=time.perf_counter(); result=optimize_joint_path(arm,targets,initial,limits=limits,max_iterations=250); elapsed=time.perf_counter()-start
out={'samples':len(targets),'same_fixed_cartesian_targets':True,'optimizer_success':result['optimizer_success'],'validated_success':result['success'],'message':result['message'],'iterations':result['iterations'],'elapsed_seconds':elapsed,'initial':metrics(initial),'optimized':metrics(result['joint_angles'])}
print(json.dumps(out,indent=2))
