"""Generate a self-contained HTML dashboard comparing IK path metrics."""
import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"D:\Project\机械臂交互式仿真平台")
BASE = ROOT / "数据与实验" / "奇异位形附近DLS基线实验" / "独立路径验证"
SOURCE = BASE / "独立路径逐点结果.csv"
OUT = BASE / "算法差异可视化"
OUT.mkdir(parents=True, exist_ok=True)

with SOURCE.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))
groups = defaultdict(list)
for row in rows:
    if int(row["repeat"]) == 1:
        groups[(row["scenario"], row["method"])].append(row)
for values in groups.values():
    values.sort(key=lambda r: int(r["sample_index"]))

scenarios = sorted({key[0] for key in groups})
methods = ["current_dls", "sigma_adaptive_dls"]
method_names = {"current_dls": "Current DLS", "sigma_adaptive_dls": "Adaptive DLS"}
colors = {"current_dls": "#2563eb", "sigma_adaptive_dls": "#ea580c"}
metric_labels = {
    "joint_1_rad": "Joint 1 angle (rad)",
    "joint_2_rad": "Joint 2 angle (rad)",
    "min_singular_value": "Minimum singular value",
    "joint_step_rad": "Joint step (rad)",
}
def nice(v):
    return f"{v:.5g}"

def chart_svg(scenario, series, title, y_label, marker_key=None):
    W, H = 900, 300
    left, right, top, bottom = 76, 24, 38, 48
    pw, ph = W-left-right, H-top-bottom
    vals = [float(r[series]) for m in methods for r in groups[(scenario, m)]]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        hi += 0.5
        lo -= 0.5
    pad = (hi-lo)*0.08
    lo, hi = lo-pad, hi+pad
    n = max(len(groups[(scenario, m)]) for m in methods)
    def x(i):
        return left + (i/max(1,n-1))*pw
    def y(v):
        return top + (hi-v)/(hi-lo)*ph
    bits = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}">']
    bits.append(f'<text x="{left}" y="23" font-size="17" font-weight="600" fill="#111827">{html.escape(title)}</text>')
    for j in range(5):
        val = lo + (hi-lo)*j/4
        yy = y(val)
        bits.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{W-right}" y2="{yy:.2f}" stroke="#e5e7eb"/>')
        bits.append(f'<text x="{left-10}" y="{yy+4:.2f}" text-anchor="end" font-size="11" fill="#6b7280">{nice(val)}</text>')
    for j in range(5):
        idx = round((n-1)*j/4)
        xx = x(idx)
        bits.append(f'<line x1="{xx:.2f}" y1="{top}" x2="{xx:.2f}" y2="{H-bottom}" stroke="#f3f4f6"/>')
        bits.append(f'<text x="{xx:.2f}" y="{H-bottom+19}" text-anchor="middle" font-size="11" fill="#6b7280">{idx}</text>')
    bits.append(f'<text x="{left+pw/2}" y="{H-8}" text-anchor="middle" font-size="12" fill="#4b5563">Path sample index</text>')
    bits.append(f'<text transform="translate(18 {top+ph/2}) rotate(-90)" text-anchor="middle" font-size="12" fill="#4b5563">{html.escape(y_label)}</text>')
    for m in methods:
        rs = groups[(scenario, m)]
        pts = " ".join(f"{x(i):.2f},{y(float(r[series])):.2f}" for i,r in enumerate(rs))
        bits.append(f'<polyline points="{pts}" fill="none" stroke="{colors[m]}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
    if marker_key == "joint_step_rad":
        for m in methods:
            rs = groups[(scenario,m)]
            i = max(range(len(rs)), key=lambda k: float(rs[k][marker_key]))
            r = rs[i]
            xx, yy = x(i), y(float(r[marker_key]))
            bits.append(f'<circle cx="{xx:.2f}" cy="{yy:.2f}" r="4.5" fill="{colors[m]}" stroke="white" stroke-width="1.5"><title>{method_names[m]} max step at sample {i}: {nice(float(r[marker_key]))}</title></circle>')
    bits.append('</svg>')
    return "\n".join(bits)

def summary(scenario):
    out = {}
    for m in methods:
        rs = groups[(scenario,m)]
        sigmas = [float(r["min_singular_value"]) for r in rs]
        steps = [float(r["joint_step_rad"]) for r in rs]
        errors = [float(r["actual_error_m"]) for r in rs]
        max_i = max(range(len(rs)), key=lambda i: steps[i])
        min_i = min(range(len(rs)), key=lambda i: sigmas[i])
        out[m] = {
            "samples": len(rs), "min_sigma": min(sigmas),
            "max_step": max(steps), "max_step_index": max_i,
            "sigma_at_max_step": float(rs[max_i]["min_singular_value"]),
            "max_step_error": float(rs[max_i]["actual_error_m"]),
            "min_sigma_index": min_i, "max_error": max(errors),
        }
    return out

scenario_data = {}
for scenario in scenarios:
    scenario_data[scenario] = {
        "name": scenario,
        "charts": [
            chart_svg(scenario, "joint_1_rad", "Joint 1 angle", metric_labels["joint_1_rad"]),
            chart_svg(scenario, "joint_2_rad", "Joint 2 angle", metric_labels["joint_2_rad"]),
            chart_svg(scenario, "min_singular_value", "Minimum singular value", metric_labels["min_singular_value"]),
            chart_svg(scenario, "joint_step_rad", "Joint step (max points marked)", metric_labels["joint_step_rad"], "joint_step_rad"),
        ],
        "summary": summary(scenario),
    }

cards = []
for idx, scenario in enumerate(scenarios):
    data = scenario_data[scenario]
    s = data["summary"]
    cards.append(f'<section class="scenario" id="scenario-{idx}" data-scenario="{html.escape(scenario)}" style="display:{"block" if idx == 0 else "none"}">')
    cards.append(f'<h2>{html.escape(scenario)}</h2><p class="muted">Each method uses repeat 1 for plotting; repeated runs were identical in the current recorded dataset.</p>')
    cards.append('<div class="summary-grid">')
    for m in methods:
        a = s[m]
        cards.append(
            f'<div class="summary"><h3 style="color:{colors[m]}">{method_names[m]}</h3>'
            f'<div>Minimum σ: <b>{a["min_sigma"]:.6g}</b></div>'
            f'<div>Maximum joint step: <b>{a["max_step"]:.6g} rad</b> (sample {a["max_step_index"]})</div>'
            f'<div>σ at max step: <b>{a["sigma_at_max_step"]:.6g}</b></div>'
            f'<div>Maximum endpoint error: <b>{a["max_error"]:.3g} m</b></div></div>')
    cards.append('</div><div class="charts">')
    cards.extend(f'<div class="chart">{chart}</div>' for chart in data["charts"])
    cards.append('</div></section>')

options = "".join(f'<option value="{i}">{html.escape(s)}</option>' for i,s in enumerate(scenarios))
page = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>IK Algorithm Difference Dashboard</title>
<style>
:root{color-scheme:light}body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f3f4f6;color:#111827}
header{background:#111827;color:white;padding:24px max(20px,calc((100% - 1200px)/2))}
main{max-width:1200px;margin:24px auto;padding:0 20px}h1{margin:0 0 8px;font-size:26px}
p{line-height:1.5}.muted{color:#6b7280;font-size:13px}.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:18px 0}
select{padding:10px;border:1px solid #d1d5db;border-radius:8px;background:white;font-size:14px}
.summary-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;margin:18px 0}
.summary,.chart,.note{background:white;border:1px solid #e5e7eb;border-radius:12px;padding:16px;box-shadow:0 1px 2px #00000008}
.summary h3{margin:0 0 12px}.summary div{margin:7px 0;font-size:14px}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:14px}.chart{padding:10px;overflow:hidden}
svg{width:100%;height:auto;display:block}.note{margin-top:20px;line-height:1.6}.legend{display:flex;gap:18px;flex-wrap:wrap}
.swatch{display:inline-block;width:18px;height:4px;vertical-align:middle;margin-right:6px}
@media(max-width:520px){.charts{grid-template-columns:1fr}.chart{padding:4px}.summary-grid{grid-template-columns:1fr}}
</style></head><body>
<header><h1>IK Algorithm Difference Dashboard</h1>
<p>Current DLS vs. singular-value-adaptive DLS · synthetic held-out paths · no production algorithm changed</p>
<div class="legend"><span><i class="swatch" style="background:#2563eb"></i>Current DLS</span><span><i class="swatch" style="background:#ea580c"></i>Adaptive DLS</span></div></header>
<main><div class="controls"><label for="scenario-select">Test path:</label><select id="scenario-select">""" + options + """</select>
<span class="muted">Use the selector to switch between three test paths.</span></div>
""" + "\n".join(cards) + """
<div class="note"><b>How to read this page</b>
<ul><li>Joint angle plots compare each method at the same path sample index.</li>
<li>The singular-value plot indicates local Jacobian conditioning; a smaller value signals proximity to a rank-deficient configuration, relative to this arm and task.</li>
<li>The joint-step plot shows the Euclidean change between adjacent joint-angle samples. Dots mark each method's maximum step.</li>
<li>These are synthetic numerical IK results, not physical robot measurements. Joint step depends on sample spacing, and it is not the same as velocity or acceleration.</li>
<li>Endpoint errors are close to numerical tolerance in these tests, so tiny differences should not be interpreted as a meaningful accuracy advantage.</li></ul></div>
<p class="muted">Source: 独立路径逐点结果.csv · repeat 1 shown for each method. The output is self-contained and works offline.</p></main>
<script>
const select=document.getElementById('scenario-select');
select.addEventListener('change',()=>{document.querySelectorAll('.scenario').forEach((el,i)=>{el.style.display=i===Number(select.value)?'block':'none';});});
</script></body></html>"""
target = OUT / "IK算法差异可视化.html"
target.write_text(page, encoding="utf-8")
print("INPUT_ROWS", len(rows))
print("SCENARIOS", len(scenarios))
print("OUTPUT", target)
for scenario in scenarios:
    s = scenario_data[scenario]["summary"]
    print(scenario,
          "current_max_step", s["current_dls"]["max_step"],
          "adaptive_max_step", s["sigma_adaptive_dls"]["max_step"],
          "current_min_sigma", s["current_dls"]["min_sigma"],
          "adaptive_min_sigma", s["sigma_adaptive_dls"]["min_sigma"])
