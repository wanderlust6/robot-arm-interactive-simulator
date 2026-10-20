"""Analyze joint-path continuity in held-out DLS validation data."""
import csv, json, math, statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"D:\Project\机械臂交互式仿真平台")
BASE = ROOT / "数据与实验" / "奇异位形附近DLS基线实验" / "独立路径验证"
SOURCE = BASE / "独立路径逐点结果.csv"
OUT = BASE / "关节路径质量分析"
OUT.mkdir(parents=True, exist_ok=True)

def pct(values, p):
    if not values: return None
    values = sorted(values)
    k = (len(values)-1)*p
    lo, hi = math.floor(k), math.ceil(k)
    return values[lo] if lo == hi else values[lo]*(hi-k)+values[hi]*(k-lo)

with SOURCE.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))
groups = defaultdict(list)
for r in rows:
    groups[(r["scenario"], r["method"], int(r["repeat"]))].append(r)

summaries = []
for (scenario, method, repeat), rs in sorted(groups.items()):
    rs.sort(key=lambda r: int(r["sample_index"]))
    q = [(float(r["joint_1_rad"]), float(r["joint_2_rad"])) for r in rs]
    steps = [math.hypot(q[i][0]-q[i-1][0], q[i][1]-q[i-1][1]) for i in range(1,len(q))]
    second = [math.hypot(q[i+1][0]-2*q[i][0]+q[i-1][0], q[i+1][1]-2*q[i][1]+q[i-1][1]) for i in range(1,len(q)-1)]
    errors = [float(r["actual_error_m"]) for r in rs]
    sigmas = [float(r["min_singular_value"]) for r in rs]
    med = statistics.median(steps) if steps else 0.0
    threshold = max(0.05, 5*med)
    jumps = [i+1 for i,s in enumerate(steps) if s > threshold]
    max_step_i = max(range(len(steps)), key=steps.__getitem__) + 1 if steps else 0
    max_d2_i = max(range(len(second)), key=second.__getitem__) + 1 if second else 0
    summaries.append({
        "scenario":scenario, "method":method, "repeat":repeat, "samples":len(rs),
        "success_rate":sum(r["converged"].lower()=="true" for r in rs)/len(rs) if rs else 0,
        "max_endpoint_error_m":max(errors, default=0), "p95_endpoint_error_m":pct(errors,.95),
        "max_joint_step_rad":max(steps,default=0), "p95_joint_step_rad":pct(steps,.95),
        "median_joint_step_rad":med, "large_jump_threshold_rad":threshold, "large_jump_count":len(jumps),
        "large_jump_sample_indices":jumps,
        "max_second_difference_rad":max(second,default=0), "p95_second_difference_rad":pct(second,.95),
        "max_step_sample_index":max_step_i,
        "max_step_sample_error_m":errors[min(max_step_i,len(errors)-1)] if errors else None,
        "max_step_sample_min_singular_value":sigmas[min(max_step_i,len(sigmas)-1)] if sigmas else None,
        "max_second_difference_center_index":max_d2_i,
        "max_second_difference_center_error_m":errors[min(max_d2_i,len(errors)-1)] if errors else None,
        "max_second_difference_center_min_singular_value":sigmas[min(max_d2_i,len(sigmas)-1)] if sigmas else None,
        "min_singular_value":min(sigmas,default=None),
        "points_with_sigma_below_0_01":sum(s<0.01 for s in sigmas),
    })

csv_path = OUT / "关节路径质量逐次汇总.csv"
with csv_path.open("w",encoding="utf-8-sig",newline="") as f:
    fields=list(summaries[0].keys())
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
    for r in summaries:
        row=r.copy(); row["large_jump_sample_indices"]=json.dumps(row["large_jump_sample_indices"],ensure_ascii=False)
        w.writerow(row)
(OUT/"关节路径质量汇总.json").write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding="utf-8")
lines = [
"# 独立路径 IK 关节路径质量分析","",
f"- 输入逐点记录：{len(rows)} 行；分组（场景×方法×重复）：{len(summaries)} 组。",
"- 关节步长：相邻采样点二维关节角向量的欧氏距离，单位 rad。",
"- 离散二阶差分：相邻关节角增量之差的欧氏距离，单位 rad；仅作平滑性代理，不等于物理加速度。",
"- 大步长阈值：max(0.05 rad, 5×本组步长中位数)。计数是诊断阈值，不代表必然发生运动学错误。","",
"| 场景 | 方法 | 重复 | 收敛率 | 最大末端误差 (m) | 最大步长 (rad) | P95 步长 (rad) | 大步长数 | 最大二阶差分 (rad) | 最小奇异值 |",
"|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
]
for r in summaries:
    lines.append("| {scenario} | {method} | {repeat} | {success_rate:.1%} | {max_endpoint_error_m:.3e} | {max_joint_step_rad:.5f} | {p95_joint_step_rad:.5f} | {large_jump_count} | {max_second_difference_rad:.5f} | {min_singular_value:.5f} |".format(**r))
lines += ["","## 解读限制","","- 必须结合逐点结果判断突跳是否发生在低奇异值区间；仅看全路径最小奇异值不能证明相关性。",
"- 比较耗时不属于本报告指标，避免将不同运行时的环境波动误判为算法优势。",
"- 如果路径中有角度分支切换，关节角差可能表现为突跳；此报告按记录角度直接计算，不做额外 unwrap，以保留求解器实际输出。",""]
(OUT/"关节路径质量分析报告.md").write_text("\n".join(lines),encoding="utf-8")
print("INPUT_ROWS",len(rows),"GROUPS",len(summaries))
print("OUTPUT",OUT)
for r in summaries:
    print(r["scenario"],r["method"],"rep",r["repeat"],"success",round(r["success_rate"],4),
          "max_step",round(r["max_joint_step_rad"],6),"p95_step",round(r["p95_joint_step_rad"],6),
          "jumps",r["large_jump_count"],"max_d2",round(r["max_second_difference_rad"],6),
          "min_sigma",round(r["min_singular_value"],6))
