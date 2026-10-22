"""Stress-test DLS near the outer and inner workspace boundaries."""
import csv
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

OUT = ROOT / "数据与实验" / "奇异位形附近DLS基线实验" / "工作空间边界压力测试_阻尼上限1.0复测_20261022"
L1, L2 = 0.6, 0.4
RADII = (
    ("outer_98pct", 0.98),
    ("outer_999", 0.999),
    ("outer_99999", 0.99999),
    ("outer_exact", 1.0),
    ("inner_22", 0.22),
    ("inner_2001", 0.2001),
    ("inner_exact", 0.2),
)
SEEDS = ("nominal", "perturbed")
METHODS = ("current_dls", "sigma_adaptive_dls")
N = 121
REPEATS = 3


def inverse_two_link(radius, phi):
    c = (radius*radius - L1*L1 - L2*L2) / (2*L1*L2)
    c = float(np.clip(c, -1.0, 1.0))
    q2 = math.acos(c)
    q1 = phi - math.atan2(L2*math.sin(q2), L1 + L2*math.cos(q2))
    return np.array([q1, q2])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    arm = PlanarArm([L1, L2])
    points, summaries = [], []
    angles = np.linspace(-math.pi/4, math.pi/4, N)
    for case, radius in RADII:
        targets = np.column_stack((radius*np.cos(angles), radius*np.sin(angles)))
        nominal = inverse_two_link(radius, float(angles[0]))
        for seed_name in SEEDS:
            if seed_name == "nominal":
                initial = nominal.copy()
            else:
                offset = np.array([math.radians(15), math.radians(5 if radius >= 0.6 else -5)])
                initial = nominal + offset
            for method in METHODS:
                for repeat in range(1, REPEATS+1):
                    q = initial.copy()
                    rows = []
                    for i, target in enumerate(targets):
                        started = time.perf_counter()
                        if method == "current_dls":
                            solution, reported, ok = solve_n_link_ik(
                                arm, target, q, max_iterations=250, tolerance=1e-6, damping=1e-3)
                            iterations, peak = None, None
                        else:
                            solution, reported, ok, iterations, peak = adaptive_ik(
                                arm, target, q, max_iterations=250, tolerance=1e-6,
                                base_damping=1e-3, sigma_threshold=0.02, gain=0.05, max_damping=1.0)
                        elapsed = (time.perf_counter()-started)*1000
                        actual = float(np.linalg.norm(arm.forward_kinematics(solution)[-1]-target))
                        sigma = float(arm.singularity_metrics(solution)["min_singular_value"])
                        step = float(np.linalg.norm(solution-q))
                        row = {
                            "sample_index": i, "scenario": case, "target_radius_m": radius,
                            "seed_case": seed_name, "method": method, "repeat": repeat,
                            "converged": bool(ok), "actual_error_m": actual,
                            "solve_time_ms": elapsed, "iterations": iterations,
                            "joint_1_rad": float(solution[0]), "joint_2_rad": float(solution[1]),
                            "joint_step_rad": step, "min_singular_value": sigma,
                            "peak_damping": peak,
                        }
                        rows.append(row)
                        points.append(row)
                        q = np.asarray(solution).copy()
                    errors = [r["actual_error_m"] for r in rows]
                    steps = [r["joint_step_rad"] for r in rows]
                    times = [r["solve_time_ms"] for r in rows]
                    qpath = [(r["joint_1_rad"], r["joint_2_rad"]) for r in rows]
                    d2 = [math.hypot(qpath[i+1][0]-2*qpath[i][0]+qpath[i-1][0],
                                     qpath[i+1][1]-2*qpath[i][1]+qpath[i-1][1])
                          for i in range(1, len(qpath)-1)]
                    summaries.append({
                        "scenario": case, "target_radius_m": radius, "seed_case": seed_name,
                        "method": method, "repeat": repeat, "samples": N,
                        "success_rate": sum(r["converged"] for r in rows)/N,
                        "max_endpoint_error_m": max(errors),
                        "p95_endpoint_error_m": float(np.percentile(errors, 95)),
                        "max_joint_step_rad": max(steps),
                        "p95_joint_step_rad": float(np.percentile(steps, 95)),
                        "max_second_difference_rad": max(d2, default=0.0),
                        "p95_second_difference_rad": float(np.percentile(d2, 95)),
                        "min_singular_value": min(r["min_singular_value"] for r in rows),
                        "p95_solve_time_ms": float(np.percentile(times, 95)),
                    })
                    print(f"{case} | {seed_name} | {method} | repeat {repeat}")

    def write_csv(name, rows):
        with (OUT/name).open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_csv("工作空间边界逐点结果.csv", points)
    write_csv("工作空间边界分组汇总.csv", summaries)

    aggregate = []
    for case, radius in RADII:
        for seed in SEEDS:
            for method in METHODS:
                group = [s for s in summaries if s["scenario"] == case
                         and s["seed_case"] == seed and s["method"] == method]
                aggregate.append({
                    "scenario": case, "target_radius_m": radius, "seed_case": seed,
                    "method": method, "repeats": len(group),
                    "mean_success_rate": statistics.mean(s["success_rate"] for s in group),
                    "worst_endpoint_error_m": max(s["max_endpoint_error_m"] for s in group),
                    "mean_max_joint_step_rad": statistics.mean(s["max_joint_step_rad"] for s in group),
                    "mean_max_second_difference_rad": statistics.mean(s["max_second_difference_rad"] for s in group),
                    "mean_p95_second_difference_rad": statistics.mean(s["p95_second_difference_rad"] for s in group),
                    "mean_p95_solve_time_ms": statistics.mean(s["p95_solve_time_ms"] for s in group),
                    "min_singular_value": min(s["min_singular_value"] for s in group),
                })
    write_csv("工作空间边界汇总.csv", aggregate)
    lines = [
        "# 工作空间边界压力测试", "",
        "- 二维 2R 机械臂，连杆长度 0.6 m 与 0.4 m；最大可达半径 1.0 m，最小可达半径 0.2 m。",
        "- 外边界半径：0.98、0.999、0.99999、1.0 m；内边界半径：0.22、0.2001、0.2 m。",
        "- 每条路径 121 个笛卡尔目标，目标沿圆弧从 -45° 到 +45°；每个工况重复 3 次。",
        "- 种子包含解析标称解和关节角扰动解；候选参数固定为阈值 0.02、增益 0.05、阻尼上限 1.0。",
        "- 在外/内边界处，逆运动学解趋于奇异；此处用来测试数值求解行为，不代表路径级避奇异。",
        "- 逐点记录保存每个目标的实际误差、收敛标记、关节步长、最小奇异值与求解时间。", "",
        "| 工况 | 半径(m) | 初始种子 | 方法 | 收敛率 | 最差误差(m) | 最大关节步长(rad) | 最小奇异值 | P95耗时(ms) |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for s in aggregate:
        lines.append(
            f'| {s["scenario"]} | {s["target_radius_m"]:.5f} | {s["seed_case"]} | {s["method"]} | '
            f'{s["mean_success_rate"]:.1%} | {s["worst_endpoint_error_m"]:.3e} | '
            f'{s["mean_max_joint_step_rad"]:.6f} | {s["min_singular_value"]:.3e} | '
            f'{s["mean_p95_solve_time_ms"]:.4g} |')
    lines += ["", "## 解读边界", "",
              "- 外/内边界半径由二维两连杆的可达几何决定；精确边界的雅可比秩会下降。",
              "- 两种方法的成功率、末端误差、关节步长与计算时间需要共同评估。",
              "- 如果在精确边界失败，不能把它直接等同于实际机械臂故障；还需区分数值容差、目标可达性与路径连续性。",
              "- 本实验为合成模型结果，非实体机器人测量。", ""]
    (OUT/"工作空间边界压力测试报告.md").write_text("\n".join(lines), encoding="utf-8")
    print("POINT_ROWS", len(points), "SUMMARY_GROUPS", len(summaries))
    print("OUTPUT", OUT)


if __name__ == "__main__":
    main()
