"""Grid-search sensitivity study for singular-value-adaptive DLS parameters."""
import csv
import itertools
import json
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from run_ik_robustness_suite import evaluate_path, make_joint_profile, summarize
from planar_arm import PlanarArm

OUT = ROOT / "数据与实验" / "奇异位形附近DLS基线实验" / "阻尼参数敏感性"
PROFILES = ("crossing_straight", "near_straight", "curved", "approach_retreat")
SAMPLE_COUNTS = (121, 241)
THRESHOLDS = (0.005, 0.01, 0.02, 0.05)
GAINS = (0.01, 0.05, 0.1, 0.25)
REPEATS = 3
METHODS = ("current_dls", "sigma_adaptive_dls")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    arm = PlanarArm([0.6, 0.4])
    all_rows, run_summaries = [], []
    total = len(PROFILES) * len(SAMPLE_COUNTS) * REPEATS * (1 + len(THRESHOLDS) * len(GAINS))
    run_index = 0

    for profile in PROFILES:
        for count in SAMPLE_COUNTS:
            t = np.linspace(0.0, 1.0, count)
            joint_profile = make_joint_profile(profile, t)
            targets = np.asarray([arm.forward_kinematics(q)[-1] for q in joint_profile])
            initial = joint_profile[0].copy()
            configurations = [("current_dls", None, None)]
            configurations.extend(
                ("sigma_adaptive_dls", threshold, gain)
                for threshold, gain in itertools.product(THRESHOLDS, GAINS)
            )
            for method, threshold, gain in configurations:
                for repeat in range(1, REPEATS + 1):
                    rows = evaluate_path(
                        arm, targets, initial, method,
                        threshold if threshold is not None else 0.01,
                        gain if gain is not None else 0.05,
                    )
                    for row in rows:
                        row.update({
                            "scenario": profile,
                            "samples": count,
                            "method": method,
                            "sigma_threshold": threshold,
                            "gain": gain,
                            "repeat": repeat,
                        })
                    all_rows.extend(rows)
                    s = summarize(rows, profile, count, method, repeat)
                    s.update({
                        "sigma_threshold": threshold,
                        "gain": gain,
                    })
                    run_summaries.append(s)
                    run_index += 1
                    print(f"[{run_index}/{total}] {profile}, N={count}, {method}, threshold={threshold}, gain={gain}, repeat={repeat}")

    point_path = OUT / "参数敏感性逐点结果.csv"
    with point_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    summary_path = OUT / "参数敏感性分组汇总.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(run_summaries[0]))
        writer.writeheader()
        writer.writerows(run_summaries)

    aggregated = []
    for profile in PROFILES:
        for count in SAMPLE_COUNTS:
            for method, threshold, gain in configurations:
                group = [s for s in run_summaries
                         if s["scenario"] == profile and s["samples"] == count
                         and s["method"] == method and s["sigma_threshold"] == threshold
                         and s["gain"] == gain]
                aggregated.append({
                    "scenario": profile,
                    "samples": count,
                    "method": method,
                    "sigma_threshold": threshold,
                    "gain": gain,
                    "repeats": len(group),
                    "mean_success_rate": statistics.mean(s["success_rate"] for s in group),
                    "worst_endpoint_error_m": max(s["max_endpoint_error_m"] for s in group),
                    "mean_max_joint_step_rad": statistics.mean(s["max_joint_step_rad"] for s in group),
                    "mean_p95_joint_step_rad": statistics.mean(s["p95_joint_step_rad"] for s in group),
                    "mean_max_second_difference_rad": statistics.mean(s["max_second_difference_rad"] for s in group),
                    "mean_p95_second_difference_rad": statistics.mean(s["p95_second_difference_rad"] for s in group),
                    "mean_p95_solve_time_ms": statistics.mean(s["p95_solve_time_ms"] for s in group),
                    "mean_large_jump_count": statistics.mean(s["large_jump_count"] for s in group),
                    "min_singular_value": min(s["min_singular_value"] for s in group),
                })

    aggregate_path = OUT / "参数敏感性汇总.csv"
    with aggregate_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregated[0]))
        writer.writeheader()
        writer.writerows(aggregated)

    delta_fields = (
        "delta_max_step_vs_baseline", "delta_p95_step_vs_baseline",
        "delta_max_d2_vs_baseline", "delta_p95_d2_vs_baseline",
        "delta_p95_time_vs_baseline", "delta_success_vs_baseline",
    )
    for s in aggregated:
        for field in delta_fields:
            s[field] = None

    comparison = {}
    for profile in PROFILES:
        for count in SAMPLE_COUNTS:
            base = next(s for s in aggregated if s["scenario"] == profile and s["samples"] == count
                        and s["method"] == "current_dls")
            candidates = [s for s in aggregated if s["scenario"] == profile and s["samples"] == count
                          and s["method"] == "sigma_adaptive_dls"]
            for s in candidates:
                s["delta_max_step_vs_baseline"] = s["mean_max_joint_step_rad"] - base["mean_max_joint_step_rad"]
                s["delta_p95_step_vs_baseline"] = s["mean_p95_joint_step_rad"] - base["mean_p95_joint_step_rad"]
                s["delta_max_d2_vs_baseline"] = s["mean_max_second_difference_rad"] - base["mean_max_second_difference_rad"]
                s["delta_p95_d2_vs_baseline"] = s["mean_p95_second_difference_rad"] - base["mean_p95_second_difference_rad"]
                s["delta_p95_time_vs_baseline"] = s["mean_p95_solve_time_ms"] - base["mean_p95_solve_time_ms"]
                s["delta_success_vs_baseline"] = s["mean_success_rate"] - base["mean_success_rate"]

    # Rewrite the aggregate CSV after adding baseline deltas.
    with aggregate_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregated[0]))
        writer.writeheader()
        writer.writerows(aggregated)

    for profile in PROFILES:
        for count in SAMPLE_COUNTS:
            group = [s for s in aggregated if s["scenario"] == profile and s["samples"] == count
                     and s["method"] == "sigma_adaptive_dls"]
            eligible = [s for s in group if s["mean_success_rate"] >= 1.0
                        and s["worst_endpoint_error_m"] <= 1e-6]
            ranked = sorted(eligible, key=lambda s: (
                s["mean_max_joint_step_rad"], s["mean_max_second_difference_rad"],
                s["mean_p95_solve_time_ms"]))
            comparison[f"{profile}|N={count}"] = {
                "candidate_combinations": len(group),
                "eligible_combinations": len(eligible),
                "best_by_joint_step_then_d2_then_time": [
                    {k: s[k] for k in (
                        "sigma_threshold", "gain", "mean_success_rate", "worst_endpoint_error_m",
                        "mean_max_joint_step_rad", "mean_max_second_difference_rad",
                        "mean_p95_solve_time_ms", "delta_max_step_vs_baseline",
                        "delta_max_d2_vs_baseline", "delta_p95_time_vs_baseline")}
                    for s in ranked[:5]
                ],
            }

    (OUT / "参数敏感性结论.json").write_text(
        json.dumps({
            "sample_counts": SAMPLE_COUNTS,
            "thresholds": THRESHOLDS,
            "gains": GAINS,
            "repeats": REPEATS,
            "summary": comparison,
            "ranking_note": "仅在成功率为 100% 且最差末端误差不超过 1e-6 m 的候选中，按平均最大关节步长、最大离散二阶差分、P95耗时排序；该排序不是跨指标的统计显著性检验。",
        }, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")

    lines = [
        "# DLS 阻尼参数敏感性实验", "",
        f"- 路径：{', '.join(PROFILES)}",
        f"- 采样点数：{', '.join(map(str, SAMPLE_COUNTS))}",
        f"- 奇异值阈值：{', '.join(map(str, THRESHOLDS))}",
        f"- 阻尼增益：{', '.join(map(str, GAINS))}",
        f"- 每种方法/参数组合重复：{REPEATS} 次",
        f"- 候选参数组合：{len(THRESHOLDS) * len(GAINS)} 组；每条路径/采样密度均单独对照基线。",
        "- 仅将成功率 100%、最差末端误差 ≤ 1e-6 m 的候选纳入排序；不把此排序解释为统计显著性。",
        "- 关节步长和离散二阶差分依赖采样密度，必须在相同路径和点数下与基线比较。",
        "- 运行耗时受系统调度影响，作为次级指标。", "",
        "| 路径 | 点数 | 方法 | 阈值 | 增益 | 收敛率 | 最差误差(m) | 最大步长(rad) | 最大二阶差分 | P95耗时(ms) | 相对基线最大步长变化 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in aggregated:
        lines.append(
            f'| {s["scenario"]} | {s["samples"]} | {s["method"]} | '
            f'{s["sigma_threshold"] if s["sigma_threshold"] is not None else "-"} | '
            f'{s["gain"] if s["gain"] is not None else "-"} | {s["mean_success_rate"]:.1%} | '
            f'{s["worst_endpoint_error_m"]:.3e} | {s["mean_max_joint_step_rad"]:.6f} | '
            f'{s["mean_max_second_difference_rad"]:.6f} | {s["mean_p95_solve_time_ms"]:.4g} | '
            f'{(s.get("delta_max_step_vs_baseline") or 0.0):+.6f} |')
    lines += ["", "## 结论边界", "",
              "- 参数敏感性实验用于判断候选策略对参数选择是否敏感，不直接证明算法普遍更优。",
              "- 即使某些工况出现更小的平滑性代理指标，也需检查其他路径、误差、收敛率和计算代价。",
              "- 数据为合成可达路径，不是实体机械臂实测结果。", ""]
    (OUT / "参数敏感性报告.md").write_text("\n".join(lines), encoding="utf-8")
    print("POINT_ROWS", len(all_rows))
    print("RUN_GROUPS", len(run_summaries))
    print("AGGREGATE_ROWS", len(aggregated))
    print("OUTPUT", OUT)


if __name__ == "__main__":
    main()
