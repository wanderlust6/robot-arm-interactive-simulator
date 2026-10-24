import json, sys, time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
APP=HERE.parent/"桌面应用"
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(APP))
from planar_arm import PlanarArm
from numerical_ik import solve_n_link_ik
from joint_limits import JointLimits
from app import MainWindow

arm=PlanarArm([1.0,0.8,0.6])
deg=np.pi/180
q0=np.array([25,-38,20])*deg
limits=JointLimits(np.array([-150,-145,-150])*deg,np.array([150,145,150])*deg)
# One shared end-effector path: forward-kinematics anchors then dense Cartesian interpolation.
q_anchors=np.array([
    [25,-38,20],
    [42,-58,38],
    [54,-70,46],
    [38,-52,28],
    [25,-38,20],
],dtype=float)*deg
anchors=np.array([arm.forward_kinematics(q)[-1] for q in q_anchors])
# Use a smooth interpolation through the same Cartesian anchors for every solver.
from scipy.interpolate import CubicSpline
u0=np.linspace(0,1,len(anchors))
spline=CubicSpline(u0,anchors,axis=0,bc_type="natural")
u=np.linspace(0,1,241)
targets=spline(u)
reference=q0.copy()
reference[-1]+=18*deg
methods={
 "baseline_dls": lambda p,q: solve_n_link_ik(arm,p,q,limits=limits),
 "adaptive_dls": lambda p,q: MainWindow._adaptive_dls_solver(arm,p,q,limits),
 "joint_bias": lambda p,q: MainWindow._joint_bias_path_solver(arm,p,q,limits,reference),
 "joint_limit_avoidance": lambda p,q: MainWindow._joint_limit_avoidance_solver(arm,p,q,limits),
 "minimum_joint_motion": lambda p,q: MainWindow._minimum_joint_motion_solver(arm,p,q,limits),
}
results={}
for name,solver in methods.items():
 q=q0.copy(); qs=[]; errors=[]; failures=0; start=time.perf_counter()
 for p in targets:
  out=solver(p,q); nq,err,ok=out[:3]
  nq=np.asarray(nq,dtype=float)
  actual=float(np.linalg.norm(arm.forward_kinematics(nq)[-1]-p))
  qs.append(nq.copy()); errors.append(actual)
  if not ok or actual>1e-5: failures+=1
  q=nq
 qs=np.asarray(qs); dq=np.diff(qs,axis=0)
 results[name]={
  "success_samples":int(len(targets)-failures),"total_samples":len(targets),
  "max_endpoint_error_m":float(max(errors)),
  "rms_endpoint_error_m":float(np.sqrt(np.mean(np.square(errors)))),
  "joint_path_length_rad":float(np.sum(np.linalg.norm(dq,axis=1))),
  "per_joint_total_variation_rad":np.sum(np.abs(dq),axis=0).tolist(),
  "max_joint_step_rad":float(np.max(np.linalg.norm(dq,axis=1))),
  "min_limit_margin_deg":float(np.min(np.minimum(qs-limits.lower,limits.upper-qs))*180/np.pi),
  "elapsed_s":float(time.perf_counter()-start),
  "_q":qs.tolist(),
 }
base=np.asarray(results["baseline_dls"]["_q"])
for name,r in results.items():
 qs=np.asarray(r.pop("_q"))
 delta=qs-base
 r["max_joint_deviation_from_baseline_deg"]=float(np.max(np.abs(delta))*180/np.pi)
 r["rms_joint_deviation_from_baseline_deg"]=float(np.sqrt(np.mean(delta**2))*180/np.pi)
 r["max_per_sample_joint_vector_deviation_deg"]=float(np.max(np.linalg.norm(delta,axis=1))*180/np.pi)
 r["final_configuration_difference_deg"]=(delta[-1]*180/np.pi).tolist()
print(json.dumps({"path_samples":len(targets),"anchor_points_m":anchors.tolist(),"results":results},indent=2))


