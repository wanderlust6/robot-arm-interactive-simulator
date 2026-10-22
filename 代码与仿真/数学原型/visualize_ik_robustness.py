"""Build an offline interactive dashboard for the DLS robustness suite."""
import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"D:\Project\机械臂交互式仿真平台")
BASE = ROOT / "数据与实验" / "奇异位形附近DLS基线实验" / "鲁棒性测试"
SOURCE = BASE / "鲁棒性逐点结果.csv"
OUT = BASE / "鲁棒性差异可视化"
OUT.mkdir(parents=True, exist_ok=True)

with SOURCE.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))
groups = defaultdict(list)
for r in rows:
    if int(r["repeat"]) == 1:
        groups[(r["scenario"], int(r["samples"]), r["method"])].append(r)
for rs in groups.values():
    rs.sort(key=lambda r: int(r["sample_index"]))

profiles = sorted({k[0] for k in groups})
counts = sorted({k[1] for k in groups})
methods = ["current_dls", "sigma_adaptive_dls"]
method_names = {"current_dls": "Current DLS", "sigma_adaptive_dls": "Adaptive DLS"}
colors = {"current_dls": "#2563eb", "sigma_adaptive_dls": "#ea580c"}
profile_labels = {
    "crossing_straight": "Crossing near straight singularity",
    "near_straight": "Near-straight path",
    "curved": "Curved path",
    "approach_retreat": "Approach and retreat",
}

def fnum(x):
    return f"{x:.5g}"

def metric_rows(rs, field):
    return [float(r[field]) for r in rs]

def chart_svg(scenario, count, field, title, ylabel, second=False):
    W, H = 900, 290
    left, right, top, bottom = 80, 24, 38, 44
    pw, ph = W-left-right, H-top-bottom
    data = {}
    for m in methods:
        rs = groups[(scenario, count, m)]
        if second:
            q = [(float(r["joint_1_rad"]), float(r["joint_2_rad"])) for r in rs]
            vals = [0.0] + [
                math.hypot(q[i+1][0]-2*q[i][0]+q[i-1][0],
                           q[i+1][1]-2*q[i][1]+q[i-1][1])
                for i in range(1, len(q)-1)
            ] + [0.0]
        else:
            vals = metric_rows(rs, field)
        data[m] = vals
    all_vals = [v for vals in data.values() for v in vals if math.isfinite(v)]
    lo, hi = min(all_vals), max(all_vals)
    if hi-lo < 1e-12:
        hi += max(abs(hi)*0.05, 0.5)
        lo -= max(abs(lo)*0.05, 0.5)
    pad = (hi-lo)*0.08
    lo, hi = lo-pad, hi+pad
    n = max(len(v) for v in data.values())
    def x(i): return left + i/max(1,n-1)*pw
    def y(v): return top + (hi-v)/(hi-lo)*ph
    bits = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}">']
    bits.append(f'<text x="{left}" y="23" font-size="17" font-weight="600" fill="#111827">{html.escape(title)}</text>')
    for j in range(5):
        v = lo+(hi-lo)*j/4
        yy=y(v)
        bits.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{W-right}" y2="{yy:.2f}" stroke="#e5e7eb"/>')
        bits.append(f'<text x="{left-10}" y="{yy+4:.2f}" text-anchor="end" font-size="11" fill="#6b7280">{fnum(v)}</text>')
    for j in range(5):
        i=round((n-1)*j/4); xx=x(i)
        bits.append(f'<line x1="{xx:.2f}" y1="{top}" x2="{xx:.2f}" y2="{H-bottom}" stroke="#f3f4f6"/>')
        bits.append(f'<text x="{xx:.2f}" y="{H-bottom+18}" text-anchor="middle" font-size="11" fill="#6b7280">{i}</text>')
    bits.append(f'<text x="{left+pw/2}" y="{H-7}" text-anchor="middle" font-size="12" fill="#4b5563">Path sample index</text>')
    bits.append(f'<text transform="translate(18 {top+ph/2}) rotate(-90)" text-anchor="middle" font-size="12" fill="#4b5563">{html.escape(ylabel)}</text>')
    for m in methods:
        pts=" ".join(f"{x(i):.2f},{y(v):.2f}" for i,v in enumerate(data[m]))
        bits.append(f'<polyline points="{pts}" fill="none" stroke="{colors[m]}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
    bits.append("</svg>")
    return "\n".join(bits)

def summary(scenario, count, method):
    rs = groups[(scenario,count,method)]
    steps = metric_rows(rs, "joint_step_rad")[1:]
    errors = metric_rows(rs, "actual_error_m")
    sigmas = metric_rows(rs, "min_singular_value")
    times = metric_rows(rs, "solve_time_ms")
    ok = sum(r["converged"].lower() == "true" for r in rs)
    return {
        "success": ok/len(rs),
        "max_error": max(errors),
        "max_step": max(steps, default=0.0),
        "p95_step": sorted(steps)[min(len(steps)-1, round(0.95*(len(steps)-1)))] if steps else 0.0,
        "min_sigma": min(sigmas),
        "p95_ms": sorted(times)[min(len(times)-1, round(0.95*(len(times)-1)))],
    }

sections=[]
for pidx, profile in enumerate(profiles):
    for cidx, count in enumerate(counts):
        section_id=f"panel-{pidx}-{cidx}"
        sections.append(f'<section class="panel" id="{section_id}" data-profile="{html.escape(profile)}" data-count="{count}" style="display:{"block" if pidx==0 and cidx==0 else "none"}">')
        sections.append(f'<h2>{html.escape(profile_labels.get(profile,profile))}</h2><p class="muted">{count} samples per path · repeat 1 shown for line plots</p><div class="summary-grid">')
        for m in methods:
            s=summary(profile,count,m)
            sections.append(
                f'<div class="summary"><h3 style="color:{colors[m]}">{method_names[m]}</h3>'
                f'<div>Convergence: <b>{s["success"]:.1%}</b></div>'
                f'<div>Max endpoint error: <b>{s["max_error"]:.3g} m</b></div>'
                f'<div>Max joint step: <b>{s["max_step"]:.6g} rad</b></div>'
                f'<div>P95 joint step: <b>{s["p95_step"]:.6g} rad</b></div>'
                f'<div>Minimum σ: <b>{s["min_sigma"]:.6g}</b></div>'
                f'<div>P95 solve time: <b>{s["p95_ms"]:.4g} ms</b></div></div>')
        sections.append('</div><div class="charts">')
        charts=[
            chart_svg(profile,count,"joint_1_rad","Joint 1 angle","rad"),
            chart_svg(profile,count,"joint_2_rad","Joint 2 angle","rad"),
            chart_svg(profile,count,"min_singular_value","Minimum singular value","sigma min"),
            chart_svg(profile,count,"joint_step_rad","Joint step","rad"),
            chart_svg(profile,count,"","Discrete second difference","rad",second=True),
        ]
        sections.extend(f'<div class="chart">{c}</div>' for c in charts)
        sections.append('</div></section>')

profile_options="".join(f'<option value="{html.escape(p)}">{html.escape(profile_labels.get(p,p))}</option>' for p in profiles)
count_options="".join(f'<option value="{c}">{c} samples</option>' for c in counts)
page="""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DLS Robustness Comparison</title><style>
body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f3f4f6;color:#111827}
header{background:#111827;color:#fff;padding:24px max(20px,calc((100% - 1200px)/2))}header h1{margin:0 0 8px;font-size:26px}
main{max-width:1200px;margin:24px auto;padding:0 20px}p{line-height:1.5}.muted{color:#6b7280;font-size:13px}
.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:18px 0}
select{padding:10px;border:1px solid #d1d5db;border-radius:8px;background:white;font-size:14px}
.summary-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:14px;margin:18px 0}
.summary,.chart,.note{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:16px;box-shadow:0 1px 2px #00000008}
.summary h3{margin:0 0 12px}.summary div{margin:7px 0;font-size:14px}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:14px}.chart{padding:10px;overflow:hidden}svg{width:100%;height:auto;display:block}
.legend{display:flex;gap:18px;flex-wrap:wrap}.swatch{display:inline-block;width:18px;height:4px;vertical-align:middle;margin-right:6px}
.note{margin-top:20px;line-height:1.6}@media(max-width:520px){.charts{grid-template-columns:1fr}.chart{padding:4px}}
</style></head><body><header><h1>DLS Robustness Comparison</h1>
<p>Four path geometries × three sample densities × three repeats · synthetic 2R arm data</p>
<div class="legend"><span><i class="swatch" style="background:#2563eb"></i>Current DLS</span><span><i class="swatch" style="background:#ea580c"></i>Adaptive DLS</span></div></header>
<main><div class="controls"><label for="profile">Path:</label><select id="profile">""" + profile_options + """</select>
<label for="count">Sampling:</label><select id="count">""" + count_options + """</select></div>
""" + "\n".join(sections) + """
<div class="note"><b>Reading notes</b><ul>
<li>Endpoint error and convergence are correctness checks; compare continuity metrics only when these are comparable.</li>
<li>Joint step and discrete second difference depend on sampling density. Compare methods at the same path and point count.</li>
<li>Minimum singular value measures local Jacobian conditioning; it does not by itself prove the solver will fail.</li>
<li>Timing varies with machine load. This dashboard is based on synthetic reachable targets, not physical robot measurements.</li>
<li>The current candidate has not been integrated into the desktop application.</li></ul></div>
<p class="muted">Data: 鲁棒性逐点结果.csv · line plots use repeat 1; summary values use repeat 1 as a representative run. See the Markdown report for all repeats.</p></main>
<script>
const p=document.getElementById('profile'),c=document.getElementById('count');
function update(){document.querySelectorAll('.panel').forEach(el=>{el.style.display=(el.dataset.profile===p.value&&el.dataset.count===c.value)?'block':'none';});}
p.addEventListener('change',update);c.addEventListener('change',update);
</script></body></html>"""
target=OUT/"DLS鲁棒性对比.html"
target.write_text(page,encoding="utf-8")
print("INPUT_ROWS",len(rows),"PANEL_COUNT",len(profiles)*len(counts))
print("OUTPUT",target,"BYTES",target.stat().st_size)
print("PROFILES",profiles,"SAMPLE_COUNTS",counts)
