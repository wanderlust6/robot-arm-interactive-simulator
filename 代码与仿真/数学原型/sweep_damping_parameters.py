"""Systematic parameter sweep for the experimental singular-value-adaptive DLS solver."""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_EXPERIMENTS = ROOT / "数据与实验" / "奇异位形附近DLS基线实验"
sys.path.insert(0, str(HERE))
from compare_damping_strategies import adaptive_ik, read_rows
from numerical_ik import solve_n_link_ik
from planar_arm import PlanarArm


def evaluate(arm, targets, initial, method, threshold=None, gain=None):
    q = np.asarray(initial, dtype=float).copy()
    rows = []
    for i, target in enumerate(targets):
        start = time.perf_counter()
        if method == "current_dls":
            solution, reported_error, ok = solve_n_link_ik(
                arm, target, q, max_iterations=250, tolerance=1e-6, damping=1e-3)
            iterations, peak_damping = None, None
        else:
            solution, reported_error, ok, iterations, peak_damping = adaptive_ik(
                arm, target, q, max_iterations=250, tolerance=1e-6,
                base_damping=1e-3, sigma_threshold=threshold, gain=gain, max_damping=0.2)
        elapsed = (time.perf_counter() - start) * 1000
        actual_error = float(np.linalg.norm(arm.forward_kinematics(solution)[-1] - target))
        metrics = arm.singularity_metrics(solution)
        rows.append({
            "sample_index": i, "method": method, "threshold": threshold, "gain": gain,
            "converged": bool(ok), "actual_error_m": actual_error,
            "solve_time_ms": elapsed, "iterations": iterations,
            "min_singular_value": float(metrics["min_singular_value"]),
            "condition_number": float(metrics["condition_number"]),
            "joint_step_rad": float(np.linalg.norm(solution - q)),
            "peak_damping": peak_damping,
        })
        q = solution.copy()
    return rows


def summarize(rows, scenario, repeat):
    errors = [r["actual_error_m"] for r in rows]
    times = [r["solve_time_ms"] for r in rows]
    return {
        "scenario": scenario, "run": repeat, "method": rows[0]["method"],
        "sigma_threshold": rows[0]["threshold"], "gain": rows[0]["gain"],
        "samples": len(rows), "success_rate": sum(r["converged"] for r in rows) / len(rows),
        "max_endpoint_error_m": max(errors),
        "p95_endpoint_error_m": float(np.percentile(errors, 95)),
        "mean_solve_time_ms": float(np.mean(times)),
        "p95_solve_time_ms": float(np.percentile(times, 95)),
        "max_joint_step_rad": max(r["joint_step_rad"] for r in rows),
        "max_iterations": max((r["iterations"] or 0) for r in rows),
        "max_peak_damping": max((r["peak_damping"] or 0) for r in rows),
    }


def main():
    parser = argparse.ArgumentParser(description="系统扫描奇异值自适应 DLS 参数")
    parser.add_argument("--experiments", type=Path, default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--thresholds", default="0.005,0.01,0.02,0.05")
    parser.add_argument("--gains", default="0.05,0.1,0.25,0.5")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat 必须至少为 1")
    thresholds = [float(x) for x in args.thresholds.split(",")]
    gains = [float(x) for x in args.gains.split(",")]
    if not thresholds or not gains or any(x <= 0 for x in thresholds + gains):
        parser.error("thresholds 和 gains 必须是正数列表")
    output = args.output or args.experiments / "阻尼参数系统扫描"
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
        for run_id in range(1, args.repeat + 1):
            baseline = evaluate(arm, targets, reference[0], "current_dls")
            for row in baseline:
                row.update({"scenario": case.name, "run": run_id})
            all_rows.extend(baseline)
            summaries.append(summarize(baseline, case.name, run_id))
            for threshold in thresholds:
                for gain in gains:
                    rows = evaluate(arm, targets, reference[0], "sigma_adaptive_dls", threshold, gain)
                    for row in rows:
                        row.update({"scenario": case.name, "run": run_id})
                    all_rows.extend(rows)
                    summaries.append(summarize(rows, case.name, run_id))
        print(f"{case.name}: 扫描完成")

    fields = list(all_rows[0])
    with (output / "参数扫描逐点结果.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    payload = {
        "repeat": args.repeat, "thresholds": thresholds, "gains": gains,
        "fixed_parameters": {"base_damping": 1e-3, "max_damping": 0.2,
                             "tolerance_m": 1e-6, "max_iterations": 250, "max_step_rad": 0.25},
        "runs": summaries,
    }
    (output / "参数扫描汇总.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    # Aggregate across repeats and scenarios. Timing is reported descriptively, not treated as a stable ranking.
    grouped = {}
    for s in summaries:
        key = (s["method"], s["sigma_threshold"], s["gain"])
        grouped.setdefault(key, []).append(s)
    aggregate = []
    for (method, threshold, gain), items in grouped.items():
        aggregate.append({
            "method": method, "sigma_threshold": threshold, "gain": gain,
            "mean_success_rate": float(np.mean([x["success_rate"] for x in items])),
            "worst_success_rate": min(x["success_rate"] for x in items),
            "max_endpoint_error_m": max(x["max_endpoint_error_m"] for x in items),
            "mean_p95_time_ms": float(np.mean([x["p95_solve_time_ms"] for x in items])),
            "max_joint_step_rad": max(x["max_joint_step_rad"] for x in items),
            "runs": len(items),
        })
    aggregate.sort(key=lambda x: (
        -x["worst_success_rate"], x["max_endpoint_error_m"], x["mean_p95_time_ms"]))
    report = [
        "# 奇异值自适应 DLS 参数系统扫描", "",
        f"- 场景数：{len(cases)}；每个场景重复：{args.repeat} 次。",
        f"- 阻尼阈值：{thresholds}；增益：{gains}。",
        "- 每轮均运行当前 DLS 基线；候选参数组合逐一测试。",
        "- 误差容差为 1e-6 m，最大迭代 250，最大单步范数 0.25 rad。",
        "- 排序优先级：最差成功率、最大末端误差、平均 P95 耗时；耗时仅作描述性参考。", "",
        "| 方法 | 阈值 | 增益 | 平均成功率 | 最差成功率 | 最大末端误差 (m) | 平均 P95 耗时 (ms) | 最大关节步长 (rad) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    ]
    for a in aggregate:
        report.append(
            f'| {a["method"]} | {a["sigma_threshold"] if a["sigma_threshold"] is not None else "—"} '
            f'| {a["gain"] if a["gain"] is not None else "—"} | {a["mean_success_rate"]:.1%} '
            f'| {a["worst_success_rate"]:.1%} | {a["max_endpoint_error_m"]:.4g} '
            f'| {a["mean_p95_time_ms"]:.4g} | {a["max_joint_step_rad"]:.4g} |')
    report += [
        "", "## 结果解释", "",
        "- 这组扫描只能筛选当前三个固定样例中的候选参数，不能证明对一般机械臂路径的优势。",
        "- 成功率和末端误差是主要指标；运行时间会受系统调度、缓存和数值路径影响。",
        "- 关节步长是相邻求解点的解变化，不能单独代表轨迹平滑性或无跳变保证。",
        "- 现有基线求解器不暴露迭代次数，因此不能直接比较两种方法的迭代效率。",
        "- 候选算法仍是独立研究原型，未修改桌面应用或生产求解器。",
        "- 下一步应将筛选出的参数用于未参与调参的独立目标路径，并统计失败点和解分支变化。", ""
    ]
    (output / "参数扫描报告.md").write_text("\n".join(report), encoding="utf-8")
    print(f"结果目录：{output}")
    print("候选参数排名（前 5）：")
    for a in [x for x in aggregate if x["method"] != "current_dls"][:5]:
        print(f'  threshold={a["sigma_threshold"]}, gain={a["gain"]}: '
              f'worst_success={a["worst_success_rate"]:.1%}, '
              f'max_error={a["max_endpoint_error_m"]:.3g} m, '
              f'mean_P95={a["mean_p95_time_ms"]:.3g} ms')


if __name__ == "__main__":
    main()
