import json,time,numpy as np
from scipy.interpolate import CubicSpline
from planar_arm import PlanarArm
from joint_limits import JointLimits
from numerical_ik import solve_n_link_ik
from pathwise_ik_optimizer import optimize_joint_path
arm=PlanarArm([1.,.8,.6])
wide=JointLimits(np.radians([-150,-145,-150]),np.radians([150,145,150]))
tight=JointLimits(np.radians([-55,-75,-10]),np.radians([60,0,60]))
cases={'bend':(np.radians([[25,-38,20],[42,-58,38],[54,-70,46],[38,-52,28],[25,-38,20]]),wide),'near_singular':(np.radians([[12,-8,4],[8,-4,2],[5,-2,1],[8,-4,2],[12,-8,4]]),wide),'tight':(np.radians([[-35,-48,12],[-25,-55,8],[-20,-60,5],[-28,-52,10],[-35,-48,12]]),tight)}
def build(a,n):
 p=np.array([arm.forward_kinematics(x)[-1] for x in a]); targets=CubicSpline(np.linspace(0,1,len(p)),p,axis=0,bc_type='natural')(np.linspace(0,1,n)); q=a[0].copy(); path=[]
 for x in targets:
  q,e,ok=solve_n_link_ik(arm,x,q)
  if not ok:return targets,None,'initial IK failed '+str(e)
  path.append(q.copy())
 return targets,np.array(path),None
def met(q,t,l):
 d=np.diff(q,axis=0); dd=np.diff(q,n=2,axis=0); e=[np.linalg.norm(arm.forward_kinematics(x)[-1]-p) for x,p in zip(q,t)]
 return {'max_error_m':float(max(e)),'path_length':float(np.linalg.norm(d,axis=1).sum()),'smoothness':float(np.sum(dd*dd)),'max_step_rad':float(np.linalg.norm(d,axis=1).max()),'limit_margin_deg':float(np.degrees(np.minimum(q-l.lower,l.upper-q).min()))}
rows=[]
for name,(a,l) in cases.items():
 for n in (17,41):
  t,q,err=build(a,n); row={'case':name,'samples':n,'initial_error':err}
  if q is None:row['passed']=False;rows.append(row);continue
  row['initial']=met(q,t,l);row['runs']=[]
  for sw,mw in ((.3,.03),):
   start=time.perf_counter()
   try:
    z=optimize_joint_path(arm,t,q,limits=l,smoothness_weight=sw,motion_weight=mw,max_iterations=600,endpoint_tolerance=2e-6,fixed_start=True)
    row['runs'].append({'weights':[sw,mw],'success':z['success'],'optimizer_success':z['optimizer_success'],'message':z['message'],'iterations':z['iterations'],'seconds':time.perf_counter()-start,'optimized':met(z['joint_angles'],t,l),'start_fixed':bool(np.allclose(z['joint_angles'][0],q[0],atol=1e-7))})
   except Exception as ex:row['runs'].append({'weights':[sw,mw],'success':False,'exception':repr(ex)})
  row['passed']=all(x.get('success',False) for x in row['runs']);rows.append(row)
out={'all_runs_passed':all(x.get('passed',False) for x in rows),'cases':rows}
with open(r'D:\Project\机械臂交互式仿真平台\算法研究与验证报告_整段固定路径验证结果.json','w',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps({'all_runs_passed':out['all_runs_passed'],'cases':rows},ensure_ascii=False,indent=2))
