"""Offline path-level risk dashboard for IK stress-test results."""
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "数据与实验" / "奇异位形附近DLS基线实验"
SOURCE = BASE / "目标扰动分支与可达性压力测试_可达初值扩展" / "逐点结果.csv"
OUT = BASE / "路径级风险可视化"
SIGMA_THRESHOLD = 0.02
ERROR_TOLERANCE_M = 1e-6
JOINT_STEP_FLOOR_RAD = 0.15


def as_bool(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def load_groups(source=SOURCE):
    groups = defaultdict(list)
    with Path(source).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            groups[(row["case"], row["method"], int(row["repeat"]))].append(row)
    for rows in groups.values():
        rows.sort(key=lambda r: int(r["sample_index"]))
        previous = None
        for row in rows:
            q = (float(row["joint_1_rad"]), float(row["joint_2_rad"]))
            row["joint_step_rad"] = 0.0 if previous is None else math.hypot(q[0]-previous[0], q[1]-previous[1])
            previous = q
        steps = [float(r["joint_step_rad"]) for r in rows[1:]]
        median = statistics.median(steps) if steps else 0.0
        mad = statistics.median([abs(x-median) for x in steps]) if steps else 0.0
        jump_threshold = max(JOINT_STEP_FLOOR_RAD, median + 6*1.4826*mad)
        previous_sign = 0
        for row in rows:
            flags = []
            reachable = as_bool(row["geometrically_reachable"])
            converged = as_bool(row["converged"])
            error = float(row["actual_error_m"])
            sigma = float(row["min_singular_value"])
            step = float(row["joint_step_rad"])
            sign = int(row["elbow_sign"])
            if not reachable:
                flags.append("geometrically_unreachable")
            if not converged:
                flags.append("solver_not_converged")
            if error > ERROR_TOLERANCE_M:
                flags.append("endpoint_error_over_tolerance")
            if sigma <= SIGMA_THRESHOLD:
                flags.append("near_singularity")
            if int(row["sample_index"]) > 0 and step >= jump_threshold:
                flags.append("joint_step_outlier")
            if sign and previous_sign and sign != previous_sign:
                flags.append("elbow_branch_transition")
            if sign:
                previous_sign = sign
            row["risk_flags"] = flags
            row["risk_level"] = (
                "critical" if any(x in flags for x in (
                    "geometrically_unreachable", "solver_not_converged", "endpoint_error_over_tolerance"))
                else "high" if flags else "normal")
            row["jump_threshold_rad"] = jump_threshold
    return groups


def summarize_group(rows):
    total = len(rows)
    return {
        "samples": total,
        "success_rate": sum(as_bool(r["converged"]) for r in rows)/total if total else 0.0,
        "min_sigma": min(float(r["min_singular_value"]) for r in rows),
        "max_joint_step_rad": max(float(r["joint_step_rad"]) for r in rows),
        "max_error_m": max(float(r["actual_error_m"]) for r in rows),
        "unreachable": sum(not as_bool(r["geometrically_reachable"]) for r in rows),
        "solver_failures": sum(not as_bool(r["converged"]) for r in rows),
        "branch_transitions": sum("elbow_branch_transition" in r["risk_flags"] for r in rows),
        "critical_points": sum(r["risk_level"] == "critical" for r in rows),
        "high_points": sum(r["risk_level"] == "high" for r in rows),
        "normal_points": sum(r["risk_level"] == "normal" for r in rows),
    }


def build_html(groups):
    data = []
    summaries = []
    for (case, method, repeat), rows in sorted(groups.items()):
        summary = summarize_group(rows)
        data.append({"case": case, "method": method, "repeat": repeat,
                     "summary": summary, "rows": rows})
        summaries.append({"case": case, "method": method, "repeat": repeat, **summary})
    return HTML_TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False))


HTML_TEMPLATE = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>IK 路径级风险可视化</title>
<style>
body{margin:0;padding:22px;max-width:1450px;margin-inline:auto;font-family:Segoe UI,"Microsoft YaHei",sans-serif;color:#182230;background:#f4f6f8}
h1{font-size:27px;margin:0 0 6px}.sub,.muted{color:#687383;font-size:13px}.panel,.controls,.card{background:#fff;border:1px solid #dfe4ea;border-radius:11px;padding:15px}
.controls{display:flex;gap:14px;flex-wrap:wrap;margin:16px 0}.controls label{display:flex;flex-direction:column;gap:6px;font-size:13px}
select{min-width:190px;padding:8px;border:1px solid #c8d0da;border-radius:7px;background:white}
.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-bottom:14px}.card .k{font-size:12px;color:#687383}.card .v{font-size:22px;font-weight:650;margin-top:7px;overflow-wrap:anywhere}
.panel{margin-bottom:14px}.panel h2{font-size:17px;margin:0 0 12px}.charts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.chart{border:1px solid #e5e9ef;border-radius:8px;padding:8px;min-width:0}.chart svg{width:100%;height:auto}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{text-align:left;padding:8px;border-bottom:1px solid #e7ebf0;vertical-align:top}th{color:#596474}
.pill{display:inline-block;border-radius:99px;padding:3px 8px;font-size:12px;font-weight:600}.critical{background:#fee2e2;color:#991b1b}.high{background:#ffedd5;color:#9a3412}.normal{background:#dcfce7;color:#166534}
.note{font-size:13px;line-height:1.6;color:#596474}@media(max-width:900px){.cards{grid-template-columns:repeat(2,minmax(0,1fr))}.charts{grid-template-columns:1fr}}@media(max-width:550px){body{padding:10px}.cards{grid-template-columns:1fr}select{width:100%}}
</style></head><body>
<h1>逆运动学路径级风险分析</h1><p class="sub">由目标扰动、分支稳定性与可达性压力测试生成；完全离线，无外部绘图库。</p>
<div class="controls"><label>测试工况<select id="case"></select></label><label>求解方法<select id="method"></select></label><label>重复编号<select id="repeat"></select></label></div>
<div class="cards" id="cards"></div>
<section class="panel"><h2>末端目标路径（XY）</h2><p class="muted">绿色：未触发筛查；橙色：近奇异或关节步长异常；红色：不可达、未收敛或末端误差超阈值。</p><div id="workspace"></div></section>
<section class="panel"><h2>沿路径的风险指标</h2><div class="charts"><div class="chart" id="sigma"></div><div class="chart" id="step"></div><div class="chart" id="error"></div><div class="chart" id="joints"></div></div></section>
<section class="panel"><h2>需关注的采样点</h2><p class="muted">最多展示前 80 个风险点；所有逐点标签另存为 CSV。</p><div style="overflow:auto"><table><thead><tr><th>点序号</th><th>等级</th><th>半径 (m)</th><th>最小奇异值</th><th>关节步长 (rad)</th><th>末端误差 (m)</th><th>标签</th></tr></thead><tbody id="riskRows"></tbody></table></div></section>
<section class="panel"><h2>判读规则与限制</h2><div class="note">
<p><b>红色：</b>几何不可达、IK 未收敛或实际末端误差大于 1e-6 m。</p>
<p><b>橙色：</b>最小奇异值不高于 0.02，或关节步长超过稳健离群阈值 max(0.15 rad, 中位数 + 6×1.4826×MAD)。肘部符号变化另行标记。</p>
<p>阈值是研究用筛查规则，不是机器人安全限值；“绿色”不等于已证明安全。数据来自合成仿真，不代表真实硬件表现。图表展示所选重复序列，汇总 CSV 保留全部重复。</p></div></section>
<script>
var DATA=__DATA__;
var C={critical:"#dc2626",high:"#ea580c",normal:"#16a34a"};
var LABEL={critical:"高风险",high:"需关注",normal:"常规"};
function el(id){return document.getElementById(id)}
function fmt(v,n){v=Number(v);return Number.isFinite(v)?v.toPrecision(n||4):"—"}
function fill(id,values,selected){var e=el(id);e.innerHTML="";values.forEach(function(v){var o=document.createElement("option");o.value=v;o.textContent=v;e.appendChild(o)});if(values.indexOf(selected)>=0)e.value=selected}
function current(){return DATA.find(function(d){return d.case===el("case").value&&d.method===el("method").value&&String(d.repeat)===el("repeat").value})}
function lineChart(rows,field,title,threshold,log){var W=700,H=250,L=65,R=18,T=30,B=35,PW=W-L-R,PH=H-T-B;
 var vals=rows.map(function(r){var v=Number(r[field]);return log?Math.log10(Math.max(v,1e-12)):v});
 var lo=Math.min.apply(null,vals),hi=Math.max.apply(null,vals);if(hi-lo<1e-12){lo-=.5;hi+=.5}var pad=(hi-lo)*.08;lo-=pad;hi+=pad;
 function x(i){return L+i/Math.max(1,rows.length-1)*PW}function y(v){return T+(hi-v)/(hi-lo)*PH}
 var a=['<svg viewBox="0 0 '+W+' '+H+'"><text x="'+L+'" y="20" font-size="15" font-weight="600" fill="#182230">'+title+'</text>'];
 for(var j=0;j<=4;j++){var v=lo+(hi-lo)*j/4,yy=y(v);a.push('<line x1="'+L+'" y1="'+yy+'" x2="'+(W-R)+'" y2="'+yy+'" stroke="#e6eaf0"/>');a.push('<text x="'+(L-7)+'" y="'+(yy+4)+'" text-anchor="end" font-size="10" fill="#667085">'+fmt(log?Math.pow(10,v):v,3)+'</text>')}
 if(threshold!==null){var t=log?Math.log10(Math.max(threshold,1e-12)):threshold;if(t>=lo&&t<=hi)a.push('<line x1="'+L+'" y1="'+y(t)+'" x2="'+(W-R)+'" y2="'+y(t)+'" stroke="#b91c1c" stroke-dasharray="5 4"/>')}
 a.push('<polyline points="'+vals.map(function(v,i){return x(i)+','+y(v)}).join(' ')+'" fill="none" stroke="#2563eb" stroke-width="1.8"/>');
 rows.forEach(function(r,i){if(r.risk_level!=="normal")a.push('<circle cx="'+x(i)+'" cy="'+y(vals[i])+'" r="3" fill="'+C[r.risk_level]+'"><title>点 '+r.sample_index+': '+r.risk_flags.join(", ")+'</title></circle>')});
 a.push('<text x="'+(L+PW/2)+'" y="'+(H-7)+'" text-anchor="middle" font-size="11" fill="#667085">路径采样点</text></svg>');return a.join("")}
function workspace(rows){var W=700,H=350,L=45,R=20,T=15,B=38,xs=rows.map(function(r){return Number(r.target_x_m)}),ys=rows.map(function(r){return Number(r.target_y_m)});
 var minx=Math.min.apply(null,xs),maxx=Math.max.apply(null,xs),miny=Math.min.apply(null,ys),maxy=Math.max.apply(null,ys),span=Math.max(maxx-minx,maxy-miny,.1),cx=(minx+maxx)/2,cy=(miny+maxy)/2;
 minx=cx-span*.58;maxx=cx+span*.58;miny=cy-span*.58;maxy=cy+span*.58;
 function x(v){return L+(v-minx)/(maxx-minx)*(W-L-R)}function y(v){return T+(maxy-v)/(maxy-miny)*(H-T-B)}
 var a=['<svg viewBox="0 0 '+W+' '+H+'">'];for(var i=0;i<=4;i++){var xx=L+(W-L-R)*i/4,yy=T+(H-T-B)*i/4;a.push('<line x1="'+xx+'" y1="'+T+'" x2="'+xx+'" y2="'+(H-B)+'" stroke="#eef1f5"/>');a.push('<line x1="'+L+'" y1="'+yy+'" x2="'+(W-R)+'" y2="'+yy+'" stroke="#eef1f5"/>')}
 a.push('<polyline points="'+rows.map(function(r){return x(Number(r.target_x_m))+','+y(Number(r.target_y_m))}).join(' ')+'" fill="none" stroke="#94a3b8" stroke-width="1.3"/>');
 rows.forEach(function(r){var rad=r.risk_level==="critical"?4:r.risk_level==="high"?3.5:2.5;a.push('<circle cx="'+x(Number(r.target_x_m))+'" cy="'+y(Number(r.target_y_m))+'" r="'+rad+'" fill="'+C[r.risk_level]+'"><title>点 '+r.sample_index+' · '+LABEL[r.risk_level]+' · '+(r.risk_flags.join(", ")||"无标签")+'</title></circle>')});
 a.push('<text x="'+(W/2)+'" y="'+(H-8)+'" text-anchor="middle" font-size="11" fill="#667085">X (m)</text></svg>');return a.join("")}
function render(){var d=current();if(!d)return;var r=d.rows,s=d.summary;
 var cards=[["收敛率",(100*s.success_rate).toFixed(1)+"%"],["最小奇异值",fmt(s.min_sigma,5)],["最大关节步长",fmt(s.max_joint_step_rad,5)+" rad"],["最大末端误差",Number(s.max_error_m).toExponential(3)+" m"],["不可达点",s.unreachable],["未收敛点",s.solver_failures],["红色筛查点",s.critical_points],["橙色筛查点",s.high_points]];
 el("cards").innerHTML=cards.map(function(c){return '<div class="card"><div class="k">'+c[0]+'</div><div class="v">'+c[1]+'</div></div>'}).join("");
 el("workspace").innerHTML=workspace(r);el("sigma").innerHTML=lineChart(r,"min_singular_value","最小奇异值 σmin",.02,false);
 el("step").innerHTML=lineChart(r,"joint_step_rad","关节步长 (rad)",Number(r[0].jump_threshold_rad),false);
 el("error").innerHTML=lineChart(r,"actual_error_m","末端实际误差 (m)",1e-6,true);
 el("joints").innerHTML=lineChart(r,"joint_1_rad","关节角 q1 (rad)",null,false)+lineChart(r,"joint_2_rad","关节角 q2 (rad)",null,false);
 var risky=r.filter(function(v){return v.risk_level!=="normal"||v.risk_flags.indexOf("elbow_branch_transition")>=0}).slice(0,80);
 el("riskRows").innerHTML=risky.length?risky.map(function(v){return '<tr><td>'+v.sample_index+'</td><td><span class="pill '+v.risk_level+'">'+LABEL[v.risk_level]+'</span></td><td>'+fmt(v.target_radius_m,5)+'</td><td>'+fmt(v.min_singular_value,5)+'</td><td>'+fmt(v.joint_step_rad,5)+'</td><td>'+Number(v.actual_error_m).toExponential(3)+'</td><td>'+v.risk_flags.join(", ")+'</td></tr>'}).join(""):'<tr><td colspan="7">该序列未触发风险筛查标签。</td></tr>'}
function updateRepeats(){var d=DATA.filter(function(x){return x.case===el("case").value&&x.method===el("method").value});var reps=[...new Set(d.map(function(x){return String(x.repeat)}))];fill("repeat",reps,reps[0]);render()}
function updateMethods(){var d=DATA.filter(function(x){return x.case===el("case").value}),ms=[...new Set(d.map(function(x){return x.method}))];fill("method",ms,ms[0]);updateRepeats()}
var cases=[...new Set(DATA.map(function(x){return x.case}))];fill("case",cases,cases.find(function(x){return x.indexOf("target_noise_")===0})||cases[0]);updateMethods();
el("case").addEventListener("change",updateMethods);el("method").addEventListener("change",updateRepeats);el("repeat").addEventListener("change",render);
</script></body></html>"""


def main():
    groups = load_groups()
    OUT.mkdir(parents=True, exist_ok=True)
    dashboard = OUT / "路径级风险分析.html"
    dashboard.write_text(build_html(groups), encoding="utf-8")
    point_path = OUT / "路径级风险逐点标签.csv"
    summary_path = OUT / "路径级风险分组汇总.csv"
    point_fields = ["case", "method", "repeat", "sample_index", "target_x_m", "target_y_m",
                    "target_radius_m", "geometrically_reachable", "converged", "actual_error_m",
                    "min_singular_value", "joint_step_rad", "jump_threshold_rad", "risk_level", "risk_flags"]
    with point_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=point_fields)
        writer.writeheader()
        for (case, method, repeat), rows in sorted(groups.items()):
            for row in rows:
                out = {field: row.get(field, "") for field in point_fields}
                out["risk_flags"] = ";".join(row["risk_flags"])
                writer.writerow(out)
    summary_fields = ["case", "method", "repeat", "samples", "success_rate", "min_sigma",
                      "max_joint_step_rad", "max_error_m", "unreachable", "solver_failures",
                      "branch_transitions", "critical_points", "high_points", "normal_points"]
    with summary_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        for (case, method, repeat), rows in sorted(groups.items()):
            writer.writerow({"case": case, "method": method, "repeat": repeat, **summarize_group(rows)})
    print("SEQUENCES", len(groups))
    print("POINTS", sum(len(rows) for rows in groups.values()))
    print("HTML", dashboard)
    print("POINT_CSV", point_path)
    print("SUMMARY_CSV", summary_path)


if __name__ == "__main__":
    main()
