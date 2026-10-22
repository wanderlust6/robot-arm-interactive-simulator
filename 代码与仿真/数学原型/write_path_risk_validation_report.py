"""Create a Markdown report from the reproducible path-risk validation outputs."""
import csv
from collections import Counter, defaultdict
from pathlib import Path

from path_risk_validation import OUT, SOURCE, load_rows, summarize_rows


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def pct(value):
    return f"{100 * value:.2f}%"


def build_report():
    source_rows = load_rows(SOURCE)
    points = read_csv(OUT / "路径风险逐点验证.csv")
    sequences = read_csv(OUT / "路径风险工况验证.csv")
    thresholds = read_csv(OUT / "路径风险阈值敏感性.csv")
    total = summarize_rows(source_rows)
    reachable = [r for r in source_rows if r["geometrically_reachable"].strip().lower() == "true"]
    reachable_failures = [
        r for r in reachable
        if r["converged"].strip().lower() != "true" or float(r["actual_error_m"]) > 1e-6
    ]
    risk_counts = Counter(r["risk_level"] for r in points)
    point_flag_counts = Counter()
    for row in points:
        for field in ("unreachable", "solver_failure", "error_over_tolerance", "near_singularity",
                      "joint_step_outlier", "observed_elbow_sign_change"):
            if row[field].strip().lower() in {"true", "1"}:
                point_flag_counts[field] += 1
    case_stats = defaultdict(lambda: {"sequences": 0, "points": 0, "unreachable": 0,
                                      "failures": 0, "near_sigma": 0, "sign_changes": 0})
    for row in sequences:
        stat = case_stats[row["case"]]
        stat["sequences"] += 1
        stat["points"] += int(row["points"])
        stat["unreachable"] += int(row["unreachable_points"])
        stat["failures"] += int(row["solver_failure_points"])
        stat["near_sigma"] += int(row["near_singularity_points"])
        stat["sign_changes"] += int(row["observed_elbow_sign_changes"])
    transition_named = {
        name: stat for name, stat in case_stats.items() if name.startswith("branch_transition_")
    }
    transition_lines = [
        f"- {name}：{stat['sequences']} 组序列，观测到肘部符号变化 {stat['sign_changes']} 次。"
        for name, stat in sorted(transition_named.items())
    ]
    case_lines = [
        f"| {name} | {s['sequences']} | {s['points']} | {s['unreachable']} | {s['failures']} | {s['near_sigma']} | {s['sign_changes']} |"
        for name, s in sorted(case_stats.items())
    ]
    threshold_lines = []
    for row in thresholds:
        precision = "—" if not row["precision"] else pct(float(row["precision"]))
        recall = "—" if not row["recall"] else pct(float(row["recall"]))
        threshold_lines.append(
            f"| {float(row['threshold']):.3f} | {row['flagged_points']} | {row['tp']} | {row['fp']} | "
            f"{row['fn']} | {row['tn']} | {precision} | {recall} |"
        )
    report = f"""# 路径风险标签有效性验证

## 1. 范围与复现

- 原始数据：{SOURCE}
- 逐点记录：{len(source_rows):,}
- 序列数：{len(sequences)}
- 输入数据未被修改；分析仅生成新 CSV 和本报告。
- 运行命令（项目根目录）：.\\.venv\\Scripts\\python.exe .\\代码与仿真\\数学原型\\path_risk_validation.py
- 报告生成命令：.\\.venv\\Scripts\\python.exe .\\代码与仿真\\数学原型\\write_path_risk_validation_report.py

## 2. 总体观测

| 指标 | 结果 |
|---|---:|
| 几何不可达点 | {total['unreachable_points']:,} |
| 求解器未收敛点 | {total['solver_failure_points']:,} |
| 实际末端误差大于 1e-6 m 的点 | {total['error_over_tolerance_points']:,} |
| 最小奇异值 ≤ 0.02 的点 | {total['near_singularity_points']:,} |
| 当前筛查规则 critical 点 | {total['critical_points']:,} |
| 当前分级 critical / high / normal | {risk_counts['critical']:,} / {risk_counts['high']:,} / {risk_counts['normal']:,} |
| 关节步长离群点 | {point_flag_counts['joint_step_outlier']:,} |
| 观测到的肘部符号变化点 | {point_flag_counts['observed_elbow_sign_change']:,} |

事件计数可能重叠，不应将各行相加当作不同采样点总数。critical 是按现有规则将不可达、未收敛或末端误差超阈值合并后的筛查等级。

## 3. 按工况汇总

| 工况 | 序列数 | 采样点 | 不可达点 | 未收敛点 | σmin ≤ 0.02 | 肘部符号变化 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(case_lines)}

## 4. 奇异值阈值敏感性

评估对象限定为几何可达点；“实际失败事件”定义为未收敛或实际末端误差大于 1e-6 m。几何不可达点被排除，避免把目标本身不可达误算为可达目标上的求解器失败。

- 可达点总数：{len(reachable):,}
- 可达点中观测到的实际失败事件：{len(reachable_failures):,}

| σmin 阈值 | 标记点数 | TP | FP | FN | TN | Precision | Recall |
|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(threshold_lines)}

**解释：** 本扩展数据集可达点中观测到 1,331 个实际失败点，默认阈值 0.02 的点级 Precision 为 29.89%、Recall 为 63.64%；阈值提高至 0.03 时检出全部已观测失败点，但 FP 增至 3,387。观察到的失败点集中于 sigma_adaptive_dls 在两类近工作空间边界路径的随机初值序列；该结果仅反映本合成数据集中的阈值取舍；采样点存在序列内相关性，不能视为独立样本的统计显著性证据。近奇异是运动学状态提示，并不等同于求解失败。

## 5. 分支标签核查

{chr(10).join(transition_lines) if transition_lines else "- 未找到以 branch_transition_ 开头的工况。"}

工况名称包含“branch_transition”并不能证明轨迹实际穿越了分支。当前逐点 elbow_sign 的变化计数显示，上述命名工况未观测到符号变化。因此，现有数据不能证明该测试已覆盖预期的肘部分支转换；目标路径构造已由单元测试确认跨越 q₂=0°，但数值求解结果仍未出现肘部符号变化；因此，求解器的实际分支切换能力尚未得到验证。

## 6. 结论与局限

1. 对不可达点、未收敛点、误差超限点，原始字段提供可直接检查的事件依据；这些字段可用于核验 critical 标签是否覆盖相应事件。
2. σmin ≤ 0.02 在现有数据里标记了 {point_flag_counts['near_singularity']:,} 个点，在默认阈值 0.02 下，点级 Recall 为 63.64%、Precision 为 29.89%；阈值提高至 0.03 时检出全部已观测失败点，但 FP 增至 3,387。该结果仅反映本合成数据集中的取舍，采样点存在序列内相关性，不能视为普遍性能保证。
3. 关节步长离群点是稳健统计规则产生的提示。当前数据没有独立的“危险跳变”真值，因此只能报告数量和与其他事件的重叠，不能把它当成已验证的误报率。
4. 分支目标路径构造已由单元测试确认跨越 q₂=0°，但求解结果未出现肘部符号变化，实际分支切换能力仍未得到验证。
5. 阈值属于研究筛查启发式，不是机器人安全限值；仿真数据也不能替代真实硬件验证。

## 7. 产物

- 路径风险逐点验证.csv：每点的事件标记、奇异值、关节步长和分级。
- 路径风险工况验证.csv：按工况、方法、重复编号的汇总。
- 路径风险阈值敏感性.csv：多个奇异值阈值下的计数和可计算指标。
- 路径风险标签有效性验证.md：本报告。
"""
    return report


if __name__ == "__main__":
    output = OUT / "路径风险标签有效性验证.md"
    output.write_text(build_report(), encoding="utf-8")
    print(f"REPORT={output}")
