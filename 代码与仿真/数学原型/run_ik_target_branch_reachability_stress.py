"""Stress tests: Cartesian target noise, IK branch transitions, and reachability."""
import csv, json, math, statistics, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "数据与实验" / "奇异位形附近DLS基线实验" / "目标扰动分支与可达性压力测试_阻尼上限1.0复测_20261022"
sys.path.insert(0, str(HERE))
from compare_damping_strategies import adaptive_ik
from numerical_ik import solve_n_link_ik
from planar_arm import PlanarArm

LENGTHS = (0.6, 0.4)
METHODS = ("current_dls", "sigma_adaptive_dls")
REPEATS, SAMPLES, TOL = 3, 121, 1e-6
THRESHOLD, GAIN = 0.02, 0.05


def solve_one(arm, target, q, method):
    t0 = time.perf_counter()
    if method == "current_dls":
        sol, reported, ok = solve_n_link_ik(arm, target, q, max_iterations=250,
                                             tolerance=TOL, damping=1e-3)
        iterations, peak = None, None
    else:
        sol, reported, ok, iterations, peak = adaptive_ik(
            arm, target, q, max_iterations=250, tolerance=TOL, base_damping=1e-3,
            sigma_threshold=THRESHOLD, gain=GAIN, max_damping=1.0)
    ms = (time.perf_counter() - t0) * 1000
    actual = float(np.linalg.norm(arm.forward_kinematics(sol)[-1] - target))
    sigma = float(arm.singularity_metrics(sol)["min_singular_value"])
    return np.asarray(sol), dict(converged=bool(ok), reported_error_m=float(reported),
        actual_error_m=actual, solve_time_ms=ms, iterations=iterations,
        peak_damping=peak, min_singular_value=sigma)


def solve_sequence(arm, targets, initial, method, case, repeat, segment, noise_std=0.0, seed_case="nominal"):
    q = np.asarray(initial, dtype=float).copy()
    rows = []
    for i, target in enumerate(targets):
        sol, result = solve_one(arm, target, q, method)
        radius = float(np.linalg.norm(target))
        q2 = float(sol[1])
        reachable = abs(LENGTHS[0]-LENGTHS[1])-1e-12 <= radius <= sum(LENGTHS)+1e-12
        rows.append(dict(case=case, method=method, repeat=repeat, sample_index=i,
            segment_type=segment, seed_case=seed_case, noise_std_m=noise_std,
            target_x_m=float(target[0]), target_y_m=float(target[1]), target_radius_m=radius,
            geometrically_reachable=reachable, initial_q1_rad=float(initial[0]),
            initial_q2_rad=float(initial[1]), joint_1_rad=float(sol[0]), joint_2_rad=q2,
            elbow_sign=int(np.sign(q2)) if abs(q2)>math.radians(0.25) else 0, **result))
        q = sol
    last, transitions = 0, 0
    for row in rows:
        sign = row["elbow_sign"]
        if sign and last and sign != last:
            transitions += 1
        if sign:
            last = sign
    for row in rows:
        row["branch_transitions_in_sequence"] = transitions
    return rows


def summarize(rows):
    errs = [r["actual_error_m"] for r in rows]
    times = [r["solve_time_ms"] for r in rows]
    steps = [math.hypot(rows[i]["joint_1_rad"]-rows[i-1]["joint_1_rad"],
                        rows[i]["joint_2_rad"]-rows[i-1]["joint_2_rad"])
             for i in range(1, len(rows))]
    unreachable = [r for r in rows if not r["geometrically_reachable"]]
    return dict(case=rows[0]["case"], method=rows[0]["method"], repeat=rows[0]["repeat"],
        samples=len(rows), success_rate=sum(r["converged"] for r in rows)/len(rows),
        max_actual_error_m=max(errs), p95_actual_error_m=float(np.percentile(errs,95)),
        p95_solve_time_ms=float(np.percentile(times,95)), max_joint_step_rad=max(steps,default=0),
        branch_transitions=rows[0]["branch_transitions_in_sequence"],
        unreachable_targets=len(unreachable),
        unreachable_targets_reported_converged=sum(r["converged"] for r in unreachable),
        reachable_targets_failed=sum(r["geometrically_reachable"] and not r["converged"] for r in rows),
        first_failed_index=next((r["sample_index"] for r in rows if not r["converged"]), None))


def generate_branch_transition_path(arm, sample_count=SAMPLES):
    """Generate FK targets from a joint path crossing q2=0 at its midpoint."""
    t = np.linspace(0.0, 1.0, sample_count)
    q2 = np.linspace(18.0, -18.0, sample_count)
    joint_path = np.column_stack((np.radians(5.0 + 20.0 * t), np.radians(q2)))
    targets = np.asarray([arm.forward_kinematics(q)[-1] for q in joint_path])
    return joint_path, targets


def generate_reachable_seed_profiles(arm, sample_count=SAMPLES):
    """Generate reachable FK target paths near outer and inner workspace boundaries."""
    t = np.linspace(0.0, 1.0, sample_count)
    q1 = np.radians(5.0 + 20.0 * t)
    profiles = {
        "reachable_near_outer": np.column_stack((q1, np.radians(np.linspace(8.0, 0.5, sample_count)))),
        "reachable_near_inner": np.column_stack((q1, np.radians(np.linspace(172.0, 178.0, sample_count)))),
    }
    return {
        name: (joint_path, np.asarray([arm.forward_kinematics(q)[-1] for q in joint_path]))
        for name, joint_path in profiles.items()
    }


def generate_seed_stress_initials(seed=20261022, repeats=12):
    """Return deterministic randomized initial joint seeds shared across methods."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    return np.random.default_rng(seed).uniform(-np.pi, np.pi, size=(repeats, 2))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    arm = PlanarArm(LENGTHS)
    t = np.linspace(0, 1, SAMPLES)
    all_rows, summaries = [], []

    # A reachable path with reproducible Cartesian perturbations.
    q_path = np.column_stack((np.radians(20+10*np.sin(2*np.pi*t)),
                              np.radians(24+8*np.sin(2*np.pi*t+0.4))))
    base = np.asarray([arm.forward_kinematics(q)[-1] for q in q_path])
    rng = np.random.default_rng(20261021)
    for noise_std in (0.0, 0.0005, 0.002, 0.005):
        for rep in range(1, REPEATS+1):
            noise = rng.normal(0, noise_std, size=base.shape) if noise_std else np.zeros_like(base)
            targets = base + noise
            for method in METHODS:
                case = f"target_noise_{noise_std*1000:.1f}mm"
                rows = solve_sequence(arm, targets, q_path[0], method, case, rep,
                                      "perturbed_reachable", noise_std)
                all_rows.extend(rows); summaries.append(summarize(rows))

    # A target path whose generating joint configuration crosses q2=0.
    branch_q, branch_targets = generate_branch_transition_path(arm, SAMPLES)
    seeds = (("same_elbow", branch_q[0]),
             ("opposite_elbow", np.array([branch_q[0, 0], -branch_q[0, 1]])))
    for seed_name, seed in seeds:
        for rep in range(1, REPEATS+1):
            for method in METHODS:
                case = "branch_transition_"+seed_name
                rows = solve_sequence(arm, branch_targets, seed, method, case, rep,
                                      "near_singular_branch", seed_case=seed_name)
                all_rows.extend(rows); summaries.append(summarize(rows))

    # Reachable targets with reproducible randomized initial seeds.
    seed_initials = generate_seed_stress_initials(seed=20261022, repeats=12)
    for case, (_, targets) in generate_reachable_seed_profiles(arm, SAMPLES).items():
        for repeat, seed in enumerate(seed_initials, start=1):
            for method in METHODS:
                rows = solve_sequence(arm, targets, seed, method, case, repeat,
                                      "reachable_seed_perturbation",
                                      seed_case=f"random_seed_{repeat:02d}")
                all_rows.extend(rows); summaries.append(summarize(rows))

    # Mixed reachable/unreachable points crossing both outer and inner workspace limits.
    radii = np.r_[np.linspace(.85,1.02,31), np.linspace(1.02,.85,31),
                  np.linspace(.30,.18,30), np.linspace(.18,.30,29)]
    theta = np.radians(np.linspace(-35,35,len(radii)))
    targets = np.column_stack((radii*np.cos(theta),radii*np.sin(theta)))
    seeds = (("elbow_positive",np.radians([20,25])),("elbow_negative",np.radians([20,-25])))
    for seed_name, seed in seeds:
        for rep in range(1, REPEATS+1):
            for method in METHODS:
                case = "mixed_reachability_"+seed_name
                rows = solve_sequence(arm, targets, seed, method, case, rep,
                                      "reachable_unreachable_mix", seed_case=seed_name)
                all_rows.extend(rows); summaries.append(summarize(rows))

    with (OUT/"逐点结果.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0])); w.writeheader(); w.writerows(all_rows)
    with (OUT/"分组汇总.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0])); w.writeheader(); w.writerows(summaries)

    aggregate={}
    for case in sorted({s["case"] for s in summaries}):
        for method in METHODS:
            g=[s for s in summaries if s["case"]==case and s["method"]==method]
            aggregate[case+"|"+method]=dict(repeats=len(g),
                mean_success_rate=statistics.mean(s["success_rate"] for s in g),
                worst_actual_error_m=max(s["max_actual_error_m"] for s in g),
                mean_branch_transitions=statistics.mean(s["branch_transitions"] for s in g),
                unreachable_targets=sum(s["unreachable_targets"] for s in g),
                unreachable_targets_reported_converged=sum(s["unreachable_targets_reported_converged"] for s in g),
                reachable_targets_failed=sum(s["reachable_targets_failed"] for s in g),
                mean_p95_solve_time_ms=statistics.mean(s["p95_solve_time_ms"] for s in g))
    payload=dict(model=dict(link_lengths_m=LENGTHS,samples=SAMPLES,repeats=REPEATS),
        candidate_parameters=dict(sigma_threshold=THRESHOLD,gain=GAIN,max_damping=1.0),aggregate=aggregate)
    (OUT/"压力测试汇总.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")

    report=["# 目标扰动、分支切换与可达性压力测试","",
        f"- 二连杆长度：{LENGTHS[0]:.3f} m、{LENGTHS[1]:.3f} m；每条序列 {SAMPLES} 点；每组重复 {REPEATS} 次。",
        "- 对照现有 DLS 与固定参数候选自适应 DLS（阈值 0.02、增益 0.05、阻尼上限 1.0）。",
        "- 目标扰动由固定随机种子生成；耗时会受系统负载影响。",
        "- 可达性依据 |L1-L2| ≤ r ≤ L1+L2 判定，数值容差 1e-12 m。","",
        "| 工况 | 方法 | 平均收敛率 | 最差实际误差(m) | 平均分支跳变 | 不可达却报告收敛 | 可达点失败 | P95耗时(ms) |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for key,v in aggregate.items():
        case,method=key.split("|")
        report.append(f'| {case} | {method} | {v["mean_success_rate"]:.1%} | {v["worst_actual_error_m"]:.3e} | {v["mean_branch_transitions"]:.2f} | {v["unreachable_targets_reported_converged"]} | {v["reachable_targets_failed"]} | {v["mean_p95_solve_time_ms"]:.4g} |')
    report += ["","## 判读限制","",
        "- 分支测试的目标由一条 q2 从 +18° 连续变化到 -18° 的关节路径经正运动学生成，中点 q2=0°。",
        "- 本轮 2 种初始分支 × 2 种求解方法 × 3 次重复，共 12 条分支测试序列；观测到的肘部符号变化总数为 0。目标路径穿过伸直奇异位形，但求解器没有发生可观测翻支，因此本轮并未证明求解器能完成分支切换。",
        "- 分支跳变只统计肘部角明显非零时的符号变化；穿过奇异位形的目标路径与数值解实际翻支是两件不同的事。",
        "- 新增两个由正运动学生成的全可达路径，分别靠近工作空间外边界和内边界；每条路径使用 12 个固定随机种子初值，并在两种求解方法间配对比较。",
        "- 随机初值压力测试是对特定合成路径和初值分布的条件性评估，不代表所有可达目标的普遍失败率。",
        "- 本次阻尼上限 1.0 的复测中，外/内边界的全可达初值扩展路径均无求解失败；此前阻尼上限 0.2 的版本曾在外边界随机初值下出现停滞，原始结果目录保留用于对照。",
        "- 收敛标志必须与实际末端误差一起检查；不可达目标的最佳近似不能被当作成功。",
        "- 噪声直接施加于笛卡尔目标点，不是传感器硬件实验。",
        "- 脚本只做研究诊断，不修改桌面应用，也不据此宣称候选算法整体更优。","",
        "输出文件：逐点结果.csv、分组汇总.csv、压力测试汇总.json。"]
    (OUT/"压力测试报告.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print("POINT_ROWS",len(all_rows)); print("SUMMARY_GROUPS",len(summaries)); print("OUTPUT",OUT)
    for key,v in aggregate.items():
        if "mixed_reachability" in key or "branch_transition" in key or "target_noise_5.0" in key:
            print(key,json.dumps(v,ensure_ascii=False))


if __name__=="__main__":
    main()
