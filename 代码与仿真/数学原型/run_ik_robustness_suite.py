"""Robustness suite for current DLS and singular-value-adaptive DLS."""
import argparse
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

DEFAULT_OUT = ROOT / "数据与实验" / "奇异位形附近DLS基线实验" / "鲁棒性测试"
METHODS = ("current_dls", "sigma_adaptive_dls")


def make_joint_profile(name, t):
    rad = np.pi / 180.0
    if name == "crossing_straight":
        q1 = 15.0 + 25.0 * np.sin(np.pi * t)
        q2 = 20.0 - 40.0 * t
    elif name == "near_straight":
        q1 = 10.0 + 20.0 * t
        q2 = 8.0 - 7.0 * t
    elif name == "curved":
        q1 = 20.0 + 15.0 * np.sin(2.0 * np.pi * t)
        q2 = 35.0 + 12.0 * np.sin(2.0 * np.pi * t + 0.7)
    elif name == "approach_retreat":
        q1 = 5.0 + 35.0 * t
        q2 = 15.0 - 14.0 * np.sin(np.pi * t)
    else:
        raise ValueError(f"Unknown path profile: {name}")
    return np.column_stack((q1, q2)) * rad


def evaluate_path(arm, targets, initial, method, threshold, gain):
    q = np.asarray(initial, dtype=float).copy()
    rows = []
    for i, target in enumerate(targets):
        started = time.perf_counter()
        if method == "current_dls":
            solution, reported_error, ok = solve_n_link_ik(
                arm, target, q, max_iterations=250, tolerance=1e-6, damping=1e-3)
            iterations = None
            damping_peak = None
        else:
            solution, reported_error, ok, iterations, damping_peak = adaptive_ik(
                arm, target, q, max_iterations=250, tolerance=1e-6,
                base_damping=1e-3, sigma_threshold=threshold, gain=gain,
                max_damping=0.2)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        actual_error = float(np.linalg.norm(arm.forward_kinematics(solution)[-1] - target))
        sigma = float(arm.singularity_metrics(solution)["min_singular_value"])
        step = float(np.linalg.norm(solution - q))
        rows.append({
            "sample_index": i,
            "converged": bool(ok),
            "actual_error_m": actual_error,
            "solve_time_ms": elapsed_ms,
            "iterations": iterations,
            "joint_1_rad": float(solution[0]),
            "joint_2_rad": float(solution[1]),
            "joint_step_rad": step,
            "min_singular_value": sigma,
            "peak_damping": damping_peak,
        })
        q = np.asarray(solution, dtype=float).copy()
    return rows


def percentile(values, p):
    return float(np.percentile(np.asarray(values, dtype=float), p)) if values else 0.0


def summarize(rows, scenario, samples, method, repeat):
    rows = sorted(rows, key=lambda r: r["sample_index"])
    steps = [float(r["joint_step_rad"]) for r in rows[1:]]
    q = [(float(r["joint_1_rad"]), float(r["joint_2_rad"])) for r in rows]
    second = [
        math.hypot(q[i + 1][0] - 2 * q[i][0] + q[i - 1][0],
                   q[i + 1][1] - 2 * q[i][1] + q[i - 1][1])
        for i in range(1, len(q) - 1)
    ]
    errors = [float(r["actual_error_m"]) for r in rows]
    sigmas = [float(r["min_singular_value"]) for r in rows]
    times = [float(r["solve_time_ms"]) for r in rows]
    median_step = statistics.median(steps) if steps else 0.0
    jump_threshold = max(0.05, 5.0 * median_step)
    return {
        "scenario": scenario,
        "samples": samples,
        "method": method,
        "repeat": repeat,
        "success_rate": sum(bool(r["converged"]) for r in rows) / len(rows),
        "max_endpoint_error_m": max(errors, default=0.0),
        "p95_endpoint_error_m": percentile(errors, 95),
        "max_joint_step_rad": max(steps, default=0.0),
        "p95_joint_step_rad": percentile(steps, 95),
        "median_joint_step_rad": median_step,
        "large_jump_threshold_rad": jump_threshold,
        "large_jump_count": sum(step > jump_threshold for step in steps),
        "max_second_difference_rad": max(second, default=0.0),
        "p95_second_difference_rad": percentile(second, 95),
        "min_singular_value": min(sigmas, default=0.0),
        "p95_solve_time_ms": percentile(times, 95),
        "mean_solve_time_ms": statistics.mean(times) if times else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare DLS strategies across path geometry and sample density.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sample-counts", type=int, nargs="+", default=[61, 121, 241])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--sigma-threshold", type=float, default=0.01)
    parser.add_argument("--gain", type=float, default=0.05)
    args = parser.parse_args()
    if args.repeat < 1 or any(n < 3 for n in args.sample_counts):
        parser.error("repeat must be >= 1 and each sample count must be >= 3")
    if len(set(args.sample_counts)) != len(args.sample_counts):
        parser.error("sample counts must be unique")

    args.output.mkdir(parents=True, exist_ok=True)
    arm = PlanarArm([0.6, 0.4])
    profiles = ("crossing_straight", "near_straight", "curved", "approach_retreat")
    all_rows, summaries = [], []
    total = len(profiles) * len(args.sample_counts) * len(METHODS) * args.repeat
    run_index = 0
    for profile_name in profiles:
        for sample_count in args.sample_counts:
            t = np.linspace(0.0, 1.0, sample_count)
            joint_profile = make_joint_profile(profile_name, t)
            targets = np.asarray([arm.forward_kinematics(q)[-1] for q in joint_profile])
            initial = joint_profile[0].copy()
            for method in METHODS:
                for repeat in range(1, args.repeat + 1):
                    result = evaluate_path(
                        arm, targets, initial, method, args.sigma_threshold, args.gain)
                    for row in result:
                        row.update({
                            "scenario": profile_name,
                            "samples": sample_count,
                            "method": method,
                            "repeat": repeat,
                        })
                    all_rows.extend(result)
                    summaries.append(summarize(
                        result, profile_name, sample_count, method, repeat))
                    run_index += 1
                    print(f"[{run_index}/{total}] {profile_name}, N={sample_count}, {method}, repeat={repeat}")

    raw_csv = args.output / "鲁棒性逐点结果.csv"
    with raw_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    summary_csv = args.output / "鲁棒性分组汇总.csv"
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    aggregate = {}
    for profile_name in profiles:
        for sample_count in args.sample_counts:
            key = f"{profile_name}|N={sample_count}"
            aggregate[key] = {}
            for method in METHODS:
                group = [s for s in summaries if s["scenario"] == profile_name
                         and s["samples"] == sample_count and s["method"] == method]
                aggregate[key][method] = {
                    "repeats": len(group),
                    "mean_success_rate": statistics.mean(s["success_rate"] for s in group),
                    "worst_max_endpoint_error_m": max(s["max_endpoint_error_m"] for s in group),
                    "mean_max_joint_step_rad": statistics.mean(s["max_joint_step_rad"] for s in group),
                    "mean_p95_joint_step_rad": statistics.mean(s["p95_joint_step_rad"] for s in group),
                    "mean_max_second_difference_rad": statistics.mean(s["max_second_difference_rad"] for s in group),
                    "mean_p95_solve_time_ms": statistics.mean(s["p95_solve_time_ms"] for s in group),
                    "min_singular_value": min(s["min_singular_value"] for s in group),
                    "mean_large_jump_count": statistics.mean(s["large_jump_count"] for s in group),
                }
    (args.output / "鲁棒性汇总.json").write_text(
        json.dumps({
            "sample_counts": args.sample_counts,
            "repeat": args.repeat,
            "candidate_parameters": {
                "sigma_threshold": args.sigma_threshold,
                "gain": args.gain,
                "base_damping": 1e-3,
                "max_damping": 0.2,
            },
            "summary": aggregate,
        }, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")

    lines = [
        "# DLS 策略鲁棒性测试报告", "",
        f"- 路径类型：{', '.join(profiles)}",
        f"- 采样点数：{', '.join(map(str, args.sample_counts))}",
        f"- 每个（路径×采样密度×方法）重复：{args.repeat} 次",
        f"- 候选参数：sigma_threshold={args.sigma_threshold}, gain={args.gain}, base_damping=1e-3, max_damping=0.2",
        "- 机械臂：二维 2R，连杆长度 0.6 m 和 0.4 m。",
        "- 目标点由预定义关节轨迹经正运动学生成，因此目标均来自该模型的可达工作空间。",
        "- 两种方法对每个工况使用相同目标点和起始姿态；每条路径按目标点顺序 warm-start。",
        "- 关节步长是相邻采样关节角向量的欧氏距离；离散二阶差分仅是平滑性代理，不等于物理加速度。",
        "- 耗时为本机单次计时，容易受系统调度影响，应作为参考而非单独判据。", "",
        "| 路径 | 点数 | 方法 | 收敛率 | 最差末端误差 (m) | 平均最大步长 (rad) | 平均最大二阶差分 (rad) | 平均 P95 耗时 (ms) | 大步长数均值 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, methods in aggregate.items():
        profile_name, count_text = key.split("|")
        for method in METHODS:
            s = methods[method]
            lines.append(
                f"| {profile_name} | {count_text[2:]} | {method} | {s['mean_success_rate']:.1%} "
                f"| {s['worst_max_endpoint_error_m']:.3e} | {s['mean_max_joint_step_rad']:.6f} "
                f"| {s['mean_max_second_difference_rad']:.6f} | {s['mean_p95_solve_time_ms']:.4f} "
                f"| {s['mean_large_jump_count']:.2f} |")
    lines += [
        "", "## 解读边界", "",
        "- 成功率与末端误差是基本正确性指标；只有在它们相近时，才比较路径连续性指标。",
        "- 采样密度会直接影响关节步长和离散二阶差分，跨采样点数比较时不能只看原始数值。",
        "- 当前比较的是两种数值 IK 求解策略，不包含路径优化、动力学控制或实体机器人实验。",
        "- 如果候选方法只在部分工况改善，应报告工况边界，而不是宣称整体更优。",
        "",
    ]
    (args.output / "鲁棒性测试报告.md").write_text("\n".join(lines), encoding="utf-8")
    print("POINT_ROWS", len(all_rows))
    print("GROUPS", len(summaries))
    print("OUTPUT", args.output)


if __name__ == "__main__":
    main()
