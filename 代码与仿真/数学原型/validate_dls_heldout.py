"""Held-out validation on independently generated reachable end-effector paths."""
import argparse
import csv
import json
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


def make_paths(arm, samples):
    t = np.linspace(0.0, 1.0, samples)
    degree = np.pi / 180.0
    profiles = {
        "heldout_cross_straight": np.column_stack((
            (15.0 + 25.0 * np.sin(np.pi * t)) * degree,
            (25.0 - 50.0 * t) * degree,
        )),
        "heldout_near_straight": np.column_stack((
            (10.0 + 20.0 * t) * degree,
            (8.0 - 7.0 * t) * degree,
        )),
        "heldout_curved": np.column_stack((
            (20.0 + 15.0 * np.sin(2.0 * np.pi * t)) * degree,
            (35.0 + 12.0 * np.sin(2.0 * np.pi * t + 0.7)) * degree,
        )),
    }
    result = {}
    for name, joint_profile in profiles.items():
        targets = np.asarray([arm.forward_kinematics(q)[-1] for q in joint_profile])
        result[name] = (targets, joint_profile[0])
    return result


def run(arm, targets, initial, method, threshold, gain):
    q = np.asarray(initial, dtype=float).copy()
    rows = []
    for index, target in enumerate(targets):
        started = time.perf_counter()
        if method == "current_dls":
            solution, reported_error, ok = solve_n_link_ik(
                arm, target, q, max_iterations=250, tolerance=1e-6, damping=1e-3)
            iterations, peak_damping = None, None
        else:
            solution, reported_error, ok, iterations, peak_damping = adaptive_ik(
                arm, target, q, max_iterations=250, tolerance=1e-6,
                base_damping=1e-3, sigma_threshold=threshold, gain=gain, max_damping=0.2)
        elapsed = (time.perf_counter() - started) * 1000
        actual_error = float(np.linalg.norm(arm.forward_kinematics(solution)[-1] - target))
        rows.append({
            "sample_index": index, "method": method, "converged": bool(ok),
            "actual_error_m": actual_error, "solve_time_ms": elapsed,
            "iterations": iterations, "joint_1_rad": float(solution[0]),
            "joint_2_rad": float(solution[1]),
            "joint_step_rad": float(np.linalg.norm(solution - q)),
            "peak_damping": peak_damping,
            "min_singular_value": float(arm.singularity_metrics(solution)["min_singular_value"]),
        })
        q = solution.copy()
    return rows


def main():
    parser = argparse.ArgumentParser(description="在独立生成的可达路径上验证 DLS 候选策略")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=241)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--sigma-threshold", type=float, default=0.01)
    parser.add_argument("--gain", type=float, default=0.05)
    args = parser.parse_args()
    if args.samples < 3 or args.repeat < 1:
        parser.error("samples 必须至少为 3，repeat 必须至少为 1")
    output = args.output or ROOT / "数据与实验" / "奇异位形附近DLS基线实验" / "独立路径验证"
    output.mkdir(parents=True, exist_ok=True)
    arm = PlanarArm([0.6, 0.4])
    all_rows, summaries = [], []
    for scenario, (targets, initial) in make_paths(arm, args.samples).items():
        for method in ("current_dls", "sigma_adaptive_dls"):
            for repeat in range(1, args.repeat + 1):
                rows = run(arm, targets, initial, method, args.sigma_threshold, args.gain)
                for row in rows:
                    row.update({"scenario": scenario, "repeat": repeat})
                all_rows.extend(rows)
                errors = [r["actual_error_m"] for r in rows]
                times = [r["solve_time_ms"] for r in rows]
                summaries.append({
                    "scenario": scenario, "method": method, "repeat": repeat,
                    "samples": len(rows),
                    "success_rate": sum(r["converged"] for r in rows) / len(rows),
                    "max_endpoint_error_m": max(errors),
                    "p95_endpoint_error_m": float(np.percentile(errors, 95)),
                    "p95_solve_time_ms": float(np.percentile(times, 95)),
                    "max_joint_step_rad": max(r["joint_step_rad"] for r in rows),
                    "min_singular_value_on_solution": min(r["min_singular_value"] for r in rows),
                })
        print(f"{scenario}: 验证完成")
    with (output / "独立路径逐点结果.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    (output / "独立路径汇总.json").write_text(
        json.dumps({"samples": args.samples, "repeat": args.repeat,
                    "candidate_parameters": {"sigma_threshold": args.sigma_threshold,
                                             "gain": args.gain, "base_damping": 1e-3,
                                             "max_damping": 0.2},
                    "runs": summaries}, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    report = [
        "# DLS 候选策略独立路径验证", "",
        f"- 每条路径 {args.samples} 点，每种方法重复 {args.repeat} 次。",
        f"- 候选参数：阈值 {args.sigma_threshold}，增益 {args.gain}，基础阻尼 1e-3，最大阻尼 0.2。",
        "- 路径由已知关节姿态经正运动学生成，不使用前一轮实验包中的目标点序列。",
        "- 轨迹分别穿越伸直位形、靠近伸直位形、以及非奇异弯曲路径。", "",
        "| 场景 | 方法 | 成功率 | 最大末端误差 (m) | P95 耗时 (ms) | 最大关节步长 (rad) |",
        "|---|---|---:|---:|---:|---:|"
    ]
    for s in summaries:
        report.append(f'| {s["scenario"]} | {s["method"]} | {s["success_rate"]:.1%} '
                      f'| {s["max_endpoint_error_m"]:.4g} | {s["p95_solve_time_ms"]:.4g} '
                      f'| {s["max_joint_step_rad"]:.4g} |')
    report += [
        "", "## 限制", "",
        "- 这些是合成的可达目标路径，不是实机数据，也不是覆盖所有奇异类型的测试。",
        "- 目标由关节轨迹正运动学生成，候选 IK 的结果不强制等于生成时的关节分支。",
        "- 成功率、末端误差和关节步长需一起解读；单个指标不能证明轨迹质量。",
        "- 候选方法仍未集成到桌面应用，若结果不优于基线，不应为使用自适应阻尼而强行替换。", ""
    ]
    (output / "独立路径验证报告.md").write_text("\n".join(report), encoding="utf-8")
    print(f"结果目录：{output}")


if __name__ == "__main__":
    main()
