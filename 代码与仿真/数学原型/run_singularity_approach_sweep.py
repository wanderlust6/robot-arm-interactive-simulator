"""Approach-direction and sampling-density sweep for planar-arm singularities."""
import csv, json, math, statistics, sys, time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/"数据与实验"/"奇异位形附近DLS基线实验"/"奇异位形接近方向与采样密度扫描_20261022"
sys.path.insert(0,str(HERE))
from compare_damping_strategies import adaptive_ik
from numerical_ik import solve_n_link_ik
from planar_arm import PlanarArm
LENGTHS=(0.6,0.4); METHODS=("current_dls","sigma_adaptive_dls")
SAMPLE_COUNTS=(61,121,241); REPEATS=3; TOL=1e-6
CASES={
 "outer_q1_increasing":((5,35),(15,-15)),
 "outer_q1_decreasing":((35,5),(15,-15)),
 "outer_q1_curved_approach":((-65,-35),(15,-15)),
 "inner_folded_crossing":((-20,10),(168,192)),
}
BRANCH_EPS=math.sin(math.radians(.25))

def reference_path(case,n):
    if n<3: raise ValueError("sample count must be >= 3")
    if case not in CASES: raise ValueError(f"unknown case: {case}")
    (a,b),(c,d)=CASES[case]
    t=np.linspace(0.0,1.0,n)
    q1=np.linspace(a,b,n)
    if case=="outer_q1_curved_approach":
        q1=q1+5.0*np.sin(np.pi*t)
    return np.radians(np.column_stack((q1,np.linspace(c,d,n))))

def wrapped_delta(a,b):
    x=np.asarray(a)-np.asarray(b)
    return (x+np.pi)%(2*np.pi)-np.pi

def sign(q2):
    v=math.sin(float(q2))
    return 1 if v>BRANCH_EPS else (-1 if v < -BRANCH_EPS else 0)

def solve_point(arm,target,q,method):
    t0=time.perf_counter()
    if method=="current_dls":
        out,reported,ok=solve_n_link_ik(arm,target,q,max_iterations=250,tolerance=TOL,damping=1e-3)
        iterations=peak=None
    elif method=="sigma_adaptive_dls":
        out,reported,ok,iterations,peak=adaptive_ik(arm,target,q,max_iterations=250,tolerance=TOL,
            base_damping=1e-3,sigma_threshold=.02,gain=.05,max_damping=1.0)
    else: raise ValueError(f"unknown method: {method}")
    elapsed=(time.perf_counter()-t0)*1000
    out=np.asarray(out,dtype=float)
    return out,dict(converged=bool(ok),reported_error_m=float(reported),
        actual_error_m=float(np.linalg.norm(arm.forward_kinematics(out)[-1]-target)),
        solve_time_ms=elapsed,iterations=iterations,peak_damping=peak,
        min_singular_value=float(arm.singularity_metrics(out)["min_singular_value"]))

def run_sequence(arm,case,n,repeat,method):
    qr=reference_path(case,n); targets=np.asarray([arm.forward_kinematics(q)[-1] for q in qr])
    offset=np.radians(((0,0),(.35,-.25),(-.35,.25))[repeat-1])
    q=qr[0]+offset; rows=[]
    for i,(target,ref) in enumerate(zip(targets,qr)):
        q,metrics=solve_point(arm,target,q,method)
        rs,qs=sign(ref[1]),sign(q[1])
        rows.append(dict(case=case,sample_count=n,repeat=repeat,method=method,sample_index=i,
            target_x_m=float(target[0]),target_y_m=float(target[1]),
            reference_q1_rad=float(ref[0]),reference_q2_rad=float(ref[1]),
            joint_1_rad=float(q[0]),joint_2_rad=float(q[1]),
            reference_joint_error_rad=float(np.linalg.norm(wrapped_delta(q,ref))),
            reference_branch_sign=rs,observed_branch_sign=qs,
            branch_sign_mismatch=bool(rs and qs and rs!=qs),**metrics))
    uq=np.unwrap(np.asarray([[r["joint_1_rad"],r["joint_2_rad"]] for r in rows]),axis=0)
    steps=np.linalg.norm(np.diff(uq,axis=0),axis=1)
    d2=np.linalg.norm(np.diff(uq,n=2,axis=0),axis=1)
    for i,r in enumerate(rows):
        r["joint_step_rad"]=0. if i==0 else float(steps[i-1])
        r["joint_second_difference_rad"]=0. if i==0 or i==n-1 else float(d2[i-1])
    summary=dict(case=case,sample_count=n,repeat=repeat,method=method,samples=n,
        success_rate=sum(r["converged"] for r in rows)/n,
        failed_points=sum(not r["converged"] for r in rows),
        max_actual_error_m=max(r["actual_error_m"] for r in rows),
        p95_actual_error_m=float(np.percentile([r["actual_error_m"] for r in rows],95)),
        min_sigma=min(r["min_singular_value"] for r in rows),
        max_joint_step_rad=float(max(steps,default=0)),
        p95_joint_step_rad=float(np.percentile(steps,95)) if len(steps) else 0,
        max_second_difference_rad=float(max(d2,default=0)),
        p95_second_difference_rad=float(np.percentile(d2,95)) if len(d2) else 0,
        max_reference_joint_error_rad=max(r["reference_joint_error_rad"] for r in rows),
        branch_mismatch_points=sum(r["branch_sign_mismatch"] for r in rows),
        p95_solve_time_ms=float(np.percentile([r["solve_time_ms"] for r in rows],95)))
    return rows,summary

def write_csv(path,rows):
    if not rows: raise ValueError("empty result table")
    with Path(path).open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def run_experiment(output_dir=OUT):
    arm=PlanarArm(LENGTHS); points=[]; summaries=[]
    for case in CASES:
        for n in SAMPLE_COUNTS:
            for repeat in range(1,REPEATS+1):
                for method in METHODS:
                    rows,s=run_sequence(arm,case,n,repeat,method); points.extend(rows); summaries.append(s)
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"奇异位形逐点结果.csv",points); write_csv(out/"奇异位形分组汇总.csv",summaries)
    aggregate=[]
    for case in CASES:
        for n in SAMPLE_COUNTS:
            for method in METHODS:
                g=[s for s in summaries if s["case"]==case and s["sample_count"]==n and s["method"]==method]
                aggregate.append(dict(case=case,sample_count=n,method=method,repeats=len(g),
                    mean_success_rate=statistics.mean(s["success_rate"] for s in g),
                    worst_actual_error_m=max(s["max_actual_error_m"] for s in g),
                    min_sigma=min(s["min_sigma"] for s in g),
                    mean_max_joint_step_rad=statistics.mean(s["max_joint_step_rad"] for s in g),
                    mean_p95_joint_step_rad=statistics.mean(s["p95_joint_step_rad"] for s in g),
                    mean_max_second_difference_rad=statistics.mean(s["max_second_difference_rad"] for s in g),
                    mean_p95_second_difference_rad=statistics.mean(s["p95_second_difference_rad"] for s in g),
                    mean_max_reference_joint_error_rad=statistics.mean(s["max_reference_joint_error_rad"] for s in g),
                    branch_mismatch_points=sum(s["branch_mismatch_points"] for s in g),
                    mean_p95_solve_time_ms=statistics.mean(s["p95_solve_time_ms"] for s in g)))
    payload=dict(model=dict(link_lengths_m=LENGTHS,sample_counts=SAMPLE_COUNTS,repeats=REPEATS,cases=list(CASES)),
        candidate_parameters=dict(sigma_threshold=.02,gain=.05,max_damping=1.,tolerance_m=TOL),aggregate=aggregate)
    (out/"奇异位形实验汇总.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    lines=["# 奇异位形接近方向与采样密度扫描","",
        f"- 两连杆长度 {LENGTHS[0]} m、{LENGTHS[1]} m；4 条工况，61/121/241 点，3 次重复，2 种方法。",
        "- 工况包括三种伸直奇异位形接近方向和一种折叠奇异位形穿越路径。",
        "- 自适应 DLS 参数：奇异值阈值 0.02、增益 0.05、阻尼上限 1.0；基线阻尼 1e-3。",
        "- 关节步长和二阶差分使用逐关节 unwrap 后的角度；参考关节误差按 2π 周期计算。",
        "- 分支不匹配只在参考与求解姿态均明显远离分支边界时统计。","",
        "| 工况 | 点数 | 方法 | 收敛率 | 最差误差(m) | 最小σ | 平均最大步长(rad) | 平均P95步长(rad) | 平均最大二阶差分(rad) | 平均最大参考关节偏差(rad) | 分支不匹配点 | P95耗时(ms) |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in aggregate:
        lines.append(f'| {r["case"]} | {r["sample_count"]} | {r["method"]} | {r["mean_success_rate"]:.1%} | {r["worst_actual_error_m"]:.3e} | {r["min_sigma"]:.5g} | {r["mean_max_joint_step_rad"]:.5g} | {r["mean_p95_joint_step_rad"]:.5g} | {r["mean_max_second_difference_rad"]:.5g} | {r["mean_max_reference_joint_error_rad"]:.5g} | {r["branch_mismatch_points"]} | {r["mean_p95_solve_time_ms"]:.4g} |')
    lines += ["","## 限制","",
        "- 离散二阶差分是平滑性代理，不等于物理加速度；不同采样密度不能不加说明地直接比较。",
        "- 数据来自正运动学合成目标和确定性初值扰动，不是实机或传感器噪声实验。",
        "- 最小奇异值是局部雅可比指标，不独立决定给定任务方向下的关节速度或求解失败。",
        "- 在 q2 穿越 0 或 π 的工况中，求解器保持初始肘部分支；参考分支在奇异位形另一侧改变，因此出现参考关节偏差。该现象不是末端跟踪失败，但说明仅给定末端位置轨迹不足以保证复现指定关节参考路径。",
        "- 分支不匹配是相对于生成轨迹的关节构型偏差，不应直接解释为瞬时关节跳变；应同时检查关节步长、参考关节偏差和末端误差。",
        "- 若候选方法没有在跟踪误差、关节路径质量和计算成本之间表现出稳定优势，不应替换基线。","",
        f"逐点记录 {len(points)} 条；重复序列 {len(summaries)} 条。",""]
    (out/"奇异位形实验报告.md").write_text("\n".join(lines),encoding="utf-8")
    return points,summaries,aggregate

if __name__=="__main__":
    p,s,a=run_experiment()
    print(f"POINT_ROWS={len(p)} SEQUENCES={len(s)} AGGREGATE_GROUPS={len(a)}")
    print(f"OUTPUT_DIR={OUT}")
    for r in a:
        print(r["case"],r["sample_count"],r["method"],f'success={r["mean_success_rate"]:.3f}',
            f'error={r["worst_actual_error_m"]:.3e}',f'maxstep={r["mean_max_joint_step_rad"]:.5g}',
            f'mismatch={r["branch_mismatch_points"]}')
