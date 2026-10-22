"""Point-wise association between Jacobian singular values and IK path quality."""
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"D:\Project\机械臂交互式仿真平台")
BASE = ROOT / "数据与实验" / "奇异位形附近DLS基线实验" / "独立路径验证"
SOURCE = BASE / "独立路径逐点结果.csv"
OUT = BASE / "逐点奇异性关联分析"
OUT.mkdir(parents=True, exist_ok=True)

def rank(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg = ((i + 1) + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    return ranks

def corr(xs, ys):
    if len(xs) < 3:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    denom = math.sqrt(sum(x*x for x in dx) * sum(y*y for y in dy))
    return sum(x*y for x, y in zip(dx, dy)) / denom if denom else None

def pct(values, p):
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] if lo == hi else values[lo] * (hi-pos) + values[hi] * (pos-lo)

with SOURCE.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))
groups = defaultdict(list)
for row in rows:
    key = (row["scenario"], row["method"], int(row["repeat"]))
    groups[key].append(row)

results = []
point_rows = []
for (scenario, method, repeat), rs in sorted(groups.items()):
    rs.sort(key=lambda r: int(r["sample_index"]))
    sigma = [float(r["min_singular_value"]) for r in rs]
    error = [float(r["actual_error_m"]) for r in rs]
    step = [float(r["joint_step_rad"]) for r in rs]
    q = [(float(r["joint_1_rad"]), float(r["joint_2_rad"])) for r in rs]
    second = [None] * len(rs)
    for i in range(1, len(rs)-1):
        second[i] = math.hypot(
            q[i+1][0] - 2*q[i][0] + q[i-1][0],
            q[i+1][1] - 2*q[i][1] + q[i-1][1])
    valid = [i for i, v in enumerate(second) if v is not None]
    local = {
        "scenario": scenario, "method": method, "repeat": repeat, "samples": len(rs),
        "spearman_sigma_vs_joint_step": corr(rank(sigma), rank(step)),
        "spearman_sigma_vs_second_difference": corr(
            rank([sigma[i] for i in valid]), rank([second[i] for i in valid])),
        "spearman_sigma_vs_endpoint_error": corr(rank(sigma), rank(error)),
        "pearson_sigma_vs_joint_step": corr(sigma, step),
        "pearson_sigma_vs_second_difference": corr(
            [sigma[i] for i in valid], [second[i] for i in valid]),
        "pearson_sigma_vs_endpoint_error": corr(sigma, error),
        "min_sigma": min(sigma), "max_joint_step_rad": max(step),
        "max_step_index": step.index(max(step)),
        "max_step_sigma": sigma[step.index(max(step))],
        "max_step_endpoint_error_m": error[step.index(max(step))],
        "max_second_difference_rad": max(v for v in second if v is not None),
        "max_second_difference_index": max(valid, key=lambda i: second[i]),
    }
    idx2 = local["max_second_difference_index"]
    local["max_second_difference_sigma"] = sigma[idx2]
    local["max_second_difference_endpoint_error_m"] = error[idx2]
    for threshold in (0.01, 0.02, 0.05):
        low = [i for i, s in enumerate(sigma) if s < threshold]
        high = [i for i, s in enumerate(sigma) if s >= threshold]
        for label, indices in (("low", low), ("rest", high)):
            local[f"sigma_{threshold:g}_{label}_count"] = len(indices)
            local[f"sigma_{threshold:g}_{label}_median_step_rad"] = (
                statistics.median(step[i] for i in indices) if indices else None)
            local[f"sigma_{threshold:g}_{label}_p95_step_rad"] = (
                pct([step[i] for i in indices], .95) if indices else None)
            local[f"sigma_{threshold:g}_{label}_median_second_difference_rad"] = (
                statistics.median(second[i] for i in indices if second[i] is not None)
                if any(second[i] is not None for i in indices) else None)
            local[f"sigma_{threshold:g}_{label}_median_error_m"] = (
                statistics.median(error[i] for i in indices) if indices else None)
    results.append(local)
    for i, r in enumerate(rs):
        point_rows.append({
            "scenario": scenario, "method": method, "repeat": repeat,
            "sample_index": int(r["sample_index"]), "min_singular_value": sigma[i],
            "joint_step_rad": step[i], "second_difference_rad": second[i],
            "endpoint_error_m": error[i],
        })

with (OUT / "逐点关联汇总.csv").open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader(); w.writerows(results)
with (OUT / "逐点指标数据.csv").open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(point_rows[0].keys()))
    w.writeheader(); w.writerows(point_rows)
(OUT / "逐点关联汇总.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

lines = [
    "# 奇异值与 IK 关节路径质量的逐点关联分析", "",
    f"- 输入记录：{len(rows)} 点记录；分组：{len(results)} 个（场景×方法×重复）。",
    "- Pearson 衡量线性关联；Spearman 衡量单调秩关联。相关系数不表示因果关系。",
    "- 关节步长为相邻采样关节角的欧氏距离；二阶差分为离散平滑性代理，不等于物理加速度。",
    "- 低奇异值阈值 0.01、0.02、0.05 仅用于分层观察，不是通用奇异性判据。", "",
    "| 场景 | 方法 | 重复 | min σ | ρ(σ,步长) | ρ(σ,二阶差分) | ρ(σ,末端误差) | 最大步长处 σ | 最大二阶差分处 σ |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|"
]
for r in results:
    def fmt(v):
        return "NA" if v is None else f"{v:.4f}"
    lines.append(
        f"| {r['scenario']} | {r['method']} | {r['repeat']} | {r['min_sigma']:.5f} | "
        f"{fmt(r['spearman_sigma_vs_joint_step'])} | {fmt(r['spearman_sigma_vs_second_difference'])} | "
        f"{fmt(r['spearman_sigma_vs_endpoint_error'])} | {r['max_step_sigma']:.5f} | "
        f"{r['max_second_difference_sigma']:.5f} |")
lines += [
    "", "## 解读原则", "",
    "- 若 σ 与步长/二阶差分呈负相关，表示在本组数据中较低 σ 倾向伴随较大的对应指标；仍需检查极值点和分层统计。",
    "- 末端误差接近数值容差时，相关系数可能受数值噪声支配，不能单独用作方法优劣判断。",
    "- 三条合成路径不足以支持普遍鲁棒性结论；重复运行主要检查可重复性，不等于增加独立路径样本量。",
    "- 本脚本不修改 IK 算法，也不把离散指标解释为真实机械臂的动态性能。", ""
]
(OUT / "逐点关联分析报告.md").write_text("\n".join(lines), encoding="utf-8")
print("INPUT_ROWS", len(rows), "GROUPS", len(results), "POINT_ROWS", len(point_rows))
print("OUTPUT", OUT)
for r in results:
    print(r["scenario"], r["method"], r["repeat"],
          "rho_step", r["spearman_sigma_vs_joint_step"],
          "rho_d2", r["spearman_sigma_vs_second_difference"],
          "rho_error", r["spearman_sigma_vs_endpoint_error"],
          "maxstep_sigma", r["max_step_sigma"],
          "maxd2_sigma", r["max_second_difference_sigma"])
