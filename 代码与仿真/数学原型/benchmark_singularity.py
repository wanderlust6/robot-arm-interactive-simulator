"""Repeatable DLS baseline for the packaged singularity scenarios."""
from __future__ import annotations
import argparse, csv, json, math, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / "数据与实验" / "奇异位形附近DLS基线实验"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from numerical_ik import solve_n_link_ik
from planar_arm import PlanarArm

def load_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]

def percentile(values, p):
    values = np.asarray(values, dtype=float)
    return float(np.percentile(values, p)) if values.size else None

def run_case(case_dir, repeat):
    config = json.loads((case_dir / "experiment.json").read_text(encoding="utf-8"))
    source_rows = load_csv(case_dir / "results.csv")
    links = config.get("robot", {}).get("link_lengths_m", [0.6, 0.4])
    arm = PlanarArm(links)
    if arm.joint_count != 2:
        raise ValueError(f"{case_dir.name}: 当前脚本预期两连杆实验")
    targets = np.asarray([[r["node_2_x_m"], r["node_2_y_m"]] for r in source_rows])
    reference_angles = np.asarray([[r[f"joint_{j + 1}_angle_rad"] for j in range(arm.joint_count)] for r in source_rows])
    ref_metrics = [arm.singularity_metrics(q) for q in reference_angles]
    run_summaries, all_rows = [], []
    for run_index in range(1, repeat + 1):
        initial = reference_angles[0].copy()
        rows, errors, steps, failed, total_seconds = [], [], [], 0, 0.0
        for sample_index, target in enumerate(targets):
            started = time.perf_counter()
            solution, solver_error, converged = solve_n_link_ik(
                arm, target, initial, max_iterations=250, tolerance=1e-6, damping=1e-3)
            elapsed = time.perf_counter() - started
            total_seconds += elapsed
            actual_error = float(np.linalg.norm(arm.forward_kinematics(solution)[-1] - target))
            step = float(np.linalg.norm(solution - initial))
            metrics = arm.singularity_metrics(solution)
            failed += int(not converged)
            errors.append(actual_error)
            steps.append(step)
            row = {
                "scenario": case_dir.name, "run": run_index, "sample_index": sample_index,
                "time_s": source_rows[sample_index].get("time_s"),
                "target_x_m": float(target[0]), "target_y_m": float(target[1]),
                "converged": bool(converged), "solver_error_m": float(solver_error),
                "actual_endpoint_error_m": actual_error, "solve_time_ms": elapsed * 1000.0,
                "joint_step_norm_rad": step, "q1_rad": float(solution[0]), "q2_rad": float(solution[1]),
                "jacobian_rank": int(metrics["rank"]),
                "min_singular_value": float(metrics["min_singular_value"]),
                "condition_number": float(metrics["condition_number"]),
                "reference_min_singular_value": float(ref_metrics[sample_index]["min_singular_value"]),
                "reference_condition_number": float(ref_metrics[sample_index]["condition_number"]),
            }
            rows.append(row)
            initial = solution.copy()
        run_summaries.append({
            "run": run_index, "samples": len(rows), "failed_samples": failed,
            "success_rate": (len(rows) - failed) / len(rows) if rows else 0.0,
            "total_solve_time_s": total_seconds,
            "mean_solve_time_ms": total_seconds * 1000 / len(rows) if rows else None,
            "p95_solve_time_ms": percentile([r["solve_time_ms"] for r in rows], 95),
            "max_endpoint_error_m": max(errors, default=None),
            "p95_endpoint_error_m": percentile(errors, 95),
            "max_joint_step_rad": max(steps, default=None),
            "min_singular_value_on_solution": min((r["min_singular_value"] for r in rows), default=None),
            "max_finite_condition_number": max((r["condition_number"] for r in rows if math.isfinite(r["condition_number"])), default=None),
        })
        all_rows.extend(rows)
    return {
        "scenario": case_dir.name,
        "target_q2_deg": config.get("target", {}).get("joint_angles_deg", [None, None])[-1],
        "reference_samples": len(source_rows),
        "reference_min_singular_value": min(float(m["min_singular_value"]) for m in ref_metrics),
        "reference_singular_sample_count": sum(bool(m["is_singular"]) for m in ref_metrics),
        "runs": run_summaries,
        "notes": [
            "成功率统计当前 solve_n_link_ik 对离散路径点的逐点求解结果。",
            "耗时受计算机负载影响，建议重复运行后比较分布。",
            "奇异性指标基于末端位置雅可比矩阵；条件数可能为无穷大。",
            "当前求解器不返回迭代次数，因此不统计迭代次数。",
            "这是软件仿真基线，不是实体机械臂测量数据。"
        ]
    }, all_rows

def main():
    parser = argparse.ArgumentParser(description="测量奇异位形附近的 DLS 逆运动学基线")
    parser.add_argument("--experiments", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat 必须至少为 1")
    output = args.output or (args.experiments / "基线分析")
    output.mkdir(parents=True, exist_ok=True)
    cases = sorted(p for p in args.experiments.iterdir() if p.is_dir() and (p / "experiment.json").exists() and (p / "results.csv").exists())
    if not cases:
        raise SystemExit(f"未找到实验包：{args.experiments}")
    summaries, point_rows = [], []
    for case in cases:
        summary, rows = run_case(case, args.repeat)
        summaries.append(summary)
        point_rows.extend(rows)
        latest = summary["runs"][-1]
        print(f'{case.name}: 成功率 {latest["success_rate"]:.1%}, 最大末端误差 {latest["max_endpoint_error_m"]:.3e} m')
    with (output / "逐点基线结果.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(point_rows[0].keys()))
        writer.writeheader()
        writer.writerows(point_rows)
    (output / "基线汇总.json").write_text(json.dumps({
        "experiment_root": str(args.experiments), "repeat": args.repeat,
        "solver": "solve_n_link_ik; fixed initial damping 1e-3", "scenarios": summaries
    }, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    report = [
        "# 奇异位形附近 DLS 基线测试报告", "",
        f"- 实验包目录：{args.experiments}", f"- 每个场景重复次数：{args.repeat}",
        "- 求解器：当前版本 solve_n_link_ik，初始阻尼参数 1e-3。",
        "- 数据来源：软件仿真路径；不是实体机械臂测量。", "",
        "## 汇总", "",
        "| 场景 | 路径点数 | 参考路径最小奇异值 | 成功率 | 最大末端误差 (m) | P95 单点耗时 (ms) |",
        "|---|---:|---:|---:|---:|---:|"
    ]
    for s in summaries:
        r = s["runs"][-1]
        report.append(f'| {s["scenario"]} | {s["reference_samples"]} | {s["reference_min_singular_value"]:.6g} | {r["success_rate"]:.1%} | {r["max_endpoint_error_m"]:.6g} | {r["p95_solve_time_ms"]:.4g} |')
    report += [
        "", "## 解释边界", "",
        "- 成功率按离散路径点统计，不代表连续路径无误差或轨迹满足全部动力学约束。",
        "- 参考路径奇异值来自实验包保存的关节角；求解结果指标另存在逐点 CSV。",
        "- 本测试不比较不同阻尼策略，也不证明算法已避免奇异位形；它只建立当前版本的可重复基线。",
        "- 当前求解器不暴露迭代次数，因此本报告不包含迭代次数。", "",
        "## 输出文件", "",
        "- 逐点基线结果.csv：逐点误差、耗时、关节变化与奇异性指标。",
        "- 基线汇总.json：机器可读的统计结果。",
        "- 基线报告.md：便于人工检查的摘要。"
    ]
    report_path = output / "基线报告.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"输出目录：{output}")
    print(f"报告：{report_path}")

if __name__ == "__main__":
    main()
