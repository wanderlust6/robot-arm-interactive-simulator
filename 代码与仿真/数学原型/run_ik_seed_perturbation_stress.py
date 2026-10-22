"""Stress-test DLS strategies with perturbed initial joint seeds."""
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from compare_damping_strategies import adaptive_ik
from numerical_ik import solve_n_link_ik
from planar_arm import PlanarArm

OUT = ROOT / "数据与实验" / "奇异位形附近DLS基线实验" / "初始姿态扰动压力测试"
PROFILES = ("crossing_straight", "near_straight", "curved", "approach_retreat")
SEEDS = {
    "nominal": (0.0, 0.0),
    "mild_offset": (15.0, -15.0),
    "moderate_offset": (45.0, -45.0),
    "opposite_elbow_seed": (0.0, 80.0),
}
METHODS = ("current_dls", "sigma_adaptive_dls")
REPEATS = 3
SAMPLES = 121


def profile(name, t):
    rad = np.pi / 180.0
    if name == "crossing_straight":
        q = np.column_stack((15 + 25*np.sin(np.pi*t), 20 - 40*t))
    elif name == "near_straight":
        q = np.column_stack((10 + 20*t, 8 - 7*t))
    elif name == "curved":
        q = np.column_stack((20 + 15*np.sin(2*np.pi*t),
                             35 + 12*np.sin(2*np.pi*t + 0.7)))
    elif name == "approach_retreat":
        q = np.column_stack((5 + 35*t, 15 - 14*np.sin(np.pi*t)))
    else:
        raise ValueError(name)
    return q * rad


def solve_path(arm, targets, initial, method):
    q = np.asarray(initial, dtype=float).copy()
    rows = []
    for i, target in enumerate(targets):
        started = time.perf_counter()
        if method == "current_dls":
            solution, reported, ok = solve_n_link_ik(
                arm, target, q, max_iterations=250, tolerance=1e-6, damping=1e-3)
            iterations, peak_damping = None, None
        else:
            solution, reported, ok, iterations, peak_damping = adaptive_ik(
                arm, target, q, max_iterations=250, tolerance=1e-6,
                base_damping=1e-3, sigma_threshold=0.02, gain=0.05, max_damping=0.2)
        elapsed = (time.perf_counter() - started) * 1000.0
        actual = float(np.linalg.norm(arm.forward_kinematics(solution)[-1] - target))
        sigma = float(arm.singularity_metrics(solution)["min_singular_value"])
        step = float(np.linalg.norm(solution - q))
        rows.append({
            "sample_index": i, "converged": bool(ok), "actual_error_m": actual,
            "solve_time_ms": elapsed, "iterations": iterations,
            "joint_1_rad": float(solution[0]), "joint_2_rad": float(solution[1]),
            "joint_step_rad": step, "min_singular_value": sigma,
            "peak_damping": peak_damping,
        })
        q = np.asarray(solution, dtype=float).copy()
    return rows


def percentile(values, p):
    return float(np.percentile(np.asarray(values), p)) if values else 0.0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    arm = PlanarArm([0.6, 0.4])
    point_rows, summaries = [], []
    for pname in PROFILES:
        joint_path = profile(pname, np.linspace(0.0, 1.0, SAMPLES))
        targets = np.asarray([arm.forward_kinematics(q)[-1] for q in joint_path])
        nominal = joint_path[0].copy()
        for seed_name, offset_deg in SEEDS.items():
            initial = nominal + np.asarray(offset_deg) * np.pi / 180.0
            for method in METHODS:
                for repeat in range(1, REPEATS + 1):
                    rows = solve_path(arm, targets, initial, method)
                    for row in rows:
                        row.update({
                            "scenario": pname, "seed_case": seed_name,
                            "seed_q1_rad": float(initial[0]), "seed_q2_rad": float(initial[1]),
                            "method": method, "repeat": repeat,
                        })
                    point_rows.extend(rows)
                    steps = [r["joint_step_rad"] for r in rows]
                    errors = [r["actual_error_m"] for r in rows]
                    times = [r["solve_time_ms"] for r in rows]
                    q = [(r["joint_1_rad"], r["joint_2_rad"]) for r in rows]
                    d2 = [
                        math.hypot(q[i+1][0]-2*q[i][0]+q[i-1][0],
                                   q[i+1][1]-2*q[i][1]+q[i-1][1])
                        for i in range(1, len(q)-1)
                    ]
                    summaries.append({
                        "scenario": pname, "seed_case": seed_name, "method": method,
                        "repeat": repeat, "samples": len(rows),
                        "success_rate": sum(r["converged"] for r in rows)/len(rows),
                        "max_endpoint_error_m": max(errors),
                        "p95_endpoint_error_m": percentile(errors, 95),
                        "max_joint_step_rad": max(steps),
                        "p95_joint_step_rad": percentile(steps, 95),
                        "max_second_difference_rad": max(d2, default=0.0),
                        "p95_second_difference_rad": percentile(d2, 95),
                        "min_singular_value": min(r["min_singular_value"] for r in rows),
                        "p95_solve_time_ms": percentile(times, 95),
                    })
                    print(f"{pname} | {seed_name} | {method} | repeat {repeat}")

    with (OUT / "初始姿态扰动逐点结果.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(point_rows[0]))
        writer.writeheader()
        writer.writerows(point_rows)
    with (OUT / "初始姿态扰动分组汇总.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    aggregate = []
    for pname in PROFILES:
        for seed_name in SEEDS:
            for method in METHODS:
                group = [s for s in summaries if s["scenario"] == pname
                         and s["seed_case"] == seed_name and s["method"] == method]
                aggregate.append({
                    "scenario": pname, "seed_case": seed_name, "method": method,
                    "repeats": len(group),
                    "mean_success_rate": statistics.mean(s["success_rate"] for s in group),
                    "worst_endpoint_error_m": max(s["max_endpoint_error_m"] for s in group),
                    "mean_max_joint_step_rad": statistics.mean(s["max_joint_step_rad"] for s in group),
                    "mean_p95_joint_step_rad": statistics.mean(s["p95_joint_step_rad"] for s in group),
                    "mean_max_second_difference_rad": statistics.mean(s["max_second_difference_rad"] for s in group),
                    "mean_p95_second_difference_rad": statistics.mean(s["p95_second_difference_rad"] for s in group),
                    "mean_p95_solve_time_ms": statistics.mean(s["p95_solve_time_ms"] for s in group),
                })
    with (OUT / "初始姿态扰动汇总.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)

    lines = [
        "# 初始姿态扰动压力测试", "",
        f"- 路径类型：{', '.join(PROFILES)}",
        f"- 初始种子：{', '.join(SEEDS)}",
        f"- 每条路径采样点：{SAMPLES}",
        f"- 两种方法每组重复：{REPEATS} 次",
        "- 候选参数固定为奇异值阈值 0.02、增益 0.05；本测试不再次调参。",
        "- 同一工况下两种方法使用相同目标点和相同初始关节种子；首个目标点与扰动种子通常不重合。",
        "- 首个点的关节步长包含从扰动种子纠正到第一个目标点的变化，避免遗漏初始化恢复成本。",
        "- 目标路径由可达的合成关节轨迹通过正运动学生成；这仍不是实体机械臂测试。", "",
        "| 路径 | 初始种子 | 方法 | 收敛率 | 最差误差(m) | 最大关节步长(rad) | 最大二阶差分 | P95耗时(ms) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for s in aggregate:
        lines.append(
            f'| {s["scenario"]} | {s["seed_case"]} | {s["method"]} | '
            f'{s["mean_success_rate"]:.1%} | {s["worst_endpoint_error_m"]:.3e} | '
            f'{s["mean_max_joint_step_rad"]:.6f} | {s["mean_max_second_difference_rad"]:.6f} | '
            f'{s["mean_p95_solve_time_ms"]:.4g} |')
    lines += ["", "## 解读限制", "",
              "- 大幅初始姿态偏差会同时改变首次目标误差和可能收敛到的逆运动学分支，需与路径内部的平滑性分开解释。",
              "- 如果出现收敛失败，应回看具体逐点数据，不能只比较平均指标。",
              "- 该实验只用于检验数值求解对初始种子的敏感性，不代表完整的路径规划鲁棒性。", ""]
    (OUT / "初始姿态扰动压力测试报告.md").write_text("\n".join(lines), encoding="utf-8")
    print("POINT_ROWS", len(point_rows))
    print("SUMMARY_GROUPS", len(summaries))
    print("OUTPUT", OUT)


if __name__ == "__main__":
    main()
