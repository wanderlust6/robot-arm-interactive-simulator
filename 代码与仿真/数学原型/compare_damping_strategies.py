"""Compare the current DLS solver with a singular-value-adaptive damping variant."""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_EXPERIMENTS = ROOT / "数据与实验" / "奇异位形附近DLS基线实验"
sys.path.insert(0, str(HERE))
from numerical_ik import solve_n_link_ik
from planar_arm import PlanarArm


def read_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def adaptive_ik(arm, target, initial, max_iterations=250, tolerance=1e-6,
                base_damping=1e-3, sigma_threshold=0.05, gain=0.5, max_damping=0.2):
    """DLS experiment: raise damping as the smallest Jacobian singular value falls."""
    q = np.asarray(initial, dtype=float).copy()
    best = q.copy()
    best_error = float(np.linalg.norm(arm.forward_kinematics(q)[-1] - target))
    peak_damping = base_damping
    for iteration in range(1, max_iterations + 1):
        error_vector = np.asarray(target) - arm.forward_kinematics(q)[-1]
        error = float(np.linalg.norm(error_vector))
        if error < best_error:
            best, best_error = q.copy(), error
        if error <= tolerance:
            return q, error, True, iteration, peak_damping
        jac = arm.jacobian(q)
        singular_values = np.linalg.svd(jac, compute_uv=False)
        sigma_min = float(singular_values[-1]) if singular_values.size else 0.0
        extra = gain * max(0.0, sigma_threshold - sigma_min)
        damping = min(max_damping, base_damping + extra)
        peak_damping = max(peak_damping, damping)
        step = jac.T @ np.linalg.solve(jac @ jac.T + damping**2 * np.eye(2), error_vector)
        norm = float(np.linalg.norm(step))
        if norm > 0.25:
            step *= 0.25 / norm
        candidate = q + step
        candidate_error = float(np.linalg.norm(arm.forward_kinematics(candidate)[-1] - target))
        if candidate_error <= error:
            q = candidate
        else:
            # Match the reference solver's rejection response without changing its implementation.
            fallback = min(max_damping, min(damping * 4.0, 1.0))
            peak_damping = max(peak_damping, fallback)
            step = jac.T @ np.linalg.solve(jac @ jac.T + fallback**2 * np.eye(2), error_vector)
            norm = float(np.linalg.norm(step))
            if norm > 0.25:
                step *= 0.25 / norm
            candidate = q + step
            if np.linalg.norm(arm.forward_kinematics(candidate)[-1] - target) <= error:
                q = candidate
    return best, best_error, best_error <= tolerance, max_iterations, peak_damping


def run_one(arm, targets, initial, method, sigma_threshold=0.05, gain=0.5):
    q = np.asarray(initial, dtype=float).copy()
    rows = []
    for i, target in enumerate(targets):
        start = time.perf_counter()
        if method == "current_dls":
            solution, reported_error, ok = solve_n_link_ik(
                arm, target, q, max_iterations=250, tolerance=1e-6, damping=1e-3)
            iterations = None
            peak_damping = None
        else:
            solution, reported_error, ok, iterations, peak_damping = adaptive_ik(
                arm, target, q, sigma_threshold=sigma_threshold, gain=gain)
        elapsed_ms = (time.perf_counter() - start) * 1000
        actual_error = float(np.linalg.norm(arm.forward_kinematics(solution)[-1] - target))
        metrics = arm.singularity_metrics(solution)
        rows.append({
            "sample_index": i, "method": method, "converged": bool(ok),
            "reported_error_m": float(reported_error), "actual_error_m": actual_error,
            "solve_time_ms": elapsed_ms, "iterations": iterations,
            "min_singular_value": float(metrics["min_singular_value"]),
            "condition_number": float(metrics["condition_number"]),
            "joint_step_rad": float(np.linalg.norm(solution - q)),
            "peak_damping": peak_damping,
        })
        q = solution.copy()
    return rows


def main():
    parser = argparse.ArgumentParser(description="比较当前 DLS 与奇异值自适应阻尼")
    parser.add_argument("--experiments", type=Path, default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--sigma-threshold", type=float, default=0.05)
    parser.add_argument("--gain", type=float, default=0.5)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat 必须至少为 1")
    output = args.output or args.experiments / "阻尼策略对照"
    output.mkdir(parents=True, exist_ok=True)
    cases = sorted(p for p in args.experiments.iterdir()
                   if p.is_dir() and (p / "experiment.json").exists() and (p / "results.csv").exists())
    all_rows, summaries = [], []
    for case in cases:
        config = json.loads((case / "experiment.json").read_text(encoding="utf-8"))
        links = config.get("robot", {}).get("link_lengths_m", [0.6, 0.4])
        arm = PlanarArm(links)
        source = read_rows(case / "results.csv")
        targets = np.asarray([[r["node_2_x_m"], r["node_2_y_m"]] for r in source])
        reference = np.asarray([[r[f"joint_{j + 1}_angle_rad"] for j in range(arm.joint_count)] for r in source])
        for method in ("current_dls", "sigma_adaptive_dls"):
            for run_id in range(1, args.repeat + 1):
                rows = run_one(arm, targets, reference[0], method,
                               sigma_threshold=args.sigma_threshold, gain=args.gain)
                for row in rows:
                    row.update({"scenario": case.name, "run": run_id})
                all_rows.extend(rows)
                errors = [r["actual_error_m"] for r in rows]
                times = [r["solve_time_ms"] for r in rows]
                summary = {
                    "scenario": case.name, "method": method, "run": run_id,
                    "samples": len(rows), "success_rate": sum(r["converged"] for r in rows) / len(rows),
                    "max_endpoint_error_m": max(errors), "p95_endpoint_error_m": float(np.percentile(errors, 95)),
                    "mean_solve_time_ms": float(np.mean(times)), "p95_solve_time_ms": float(np.percentile(times, 95)),
                    "max_joint_step_rad": max(r["joint_step_rad"] for r in rows),
                    "max_iterations": max((r["iterations"] or 0) for r in rows),
                    "max_peak_damping": max((r["peak_damping"] or 0) for r in rows),
                }
                summaries.append(summary)
        print(f"{case.name}: 对照完成")
    with (output / "阻尼策略逐点结果.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    (output / "阻尼策略汇总.json").write_text(
        json.dumps({"repeat": args.repeat, "candidate_parameters": {
            "base_damping": 1e-3, "sigma_threshold": args.sigma_threshold,
            "gain": args.gain, "max_damping": 0.2
        }, "runs": summaries}, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    report = [
        "# DLS 阻尼策略初步对照", "",
        f"- 重复次数：每个场景、每种方法 {args.repeat} 次。",
        "- 当前方法：现有 solve_n_link_ik，初始阻尼 1e-3。",
        "- 候选方法：按最小奇异值提高阻尼；参数仅为实验初值，尚未调优。",
        "- 两种方法使用相同的目标点序列、起始姿态、迭代上限和误差容差。",
        "- 耗时仅供同机同环境粗略比较；需多次重复并检查分布。", "",
        "| 场景 | 方法 | 成功率 | 最大末端误差 (m) | P95 耗时 (ms) | 最大关节步长 (rad) |",
        "|---|---|---:|---:|---:|---:|"
    ]
    for s in summaries:
        report.append(f'| {s["scenario"]} | {s["method"]} | {s["success_rate"]:.1%} | {s["max_endpoint_error_m"]:.4g} | {s["p95_solve_time_ms"]:.4g} | {s["max_joint_step_rad"]:.4g} |')
    report += [
        "", "## 重要限制", "",
        "- 这是研究原型，不会改写或替换桌面应用的逆运动学算法。",
        "- 候选策略在求解失败时采用简单回退，尚未证明收敛性优于基线。",
        "- 该实验按路径点热启动求解；结果不能直接外推到任意目标或实体机械臂。",
        "- 当前方法不返回迭代次数，表中的迭代次数只对候选方法可用。",
        "- 不应只凭一次测试判定优劣；需要检视失败点、关节跳变及误差分布。"
    ]
    (output / "阻尼策略对照报告.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"结果目录：{output}")


if __name__ == "__main__":
    main()
