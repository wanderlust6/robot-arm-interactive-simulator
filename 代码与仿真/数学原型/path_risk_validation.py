"""Independent offline validation of heuristic IK path-risk labels."""
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "数据与实验" / "奇异位形附近DLS基线实验"
SOURCE = BASE / "目标扰动分支与可达性压力测试_可达初值扩展" / "逐点结果.csv"
OUT = BASE / "路径级风险可视化"
ERROR_TOLERANCE_M = 1e-6
DEFAULT_SIGMA_THRESHOLD = 0.02
JOINT_STEP_FLOOR_RAD = 0.15

REQUIRED = ("case", "method", "repeat", "sample_index", "geometrically_reachable",
            "converged", "actual_error_m", "min_singular_value", "joint_1_rad",
            "joint_2_rad", "elbow_sign", "branch_transitions_in_sequence")


def as_bool(value):
    value = str(value).strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def number(row, field):
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric value in {field}: {row.get(field)!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"Non-finite numeric value in {field}: {row.get(field)!r}")
    return value


def classify_point(row, sigma_threshold=DEFAULT_SIGMA_THRESHOLD, error_tolerance_m=ERROR_TOLERANCE_M):
    reachable = as_bool(row["geometrically_reachable"])
    converged = as_bool(row["converged"])
    error = number(row, "actual_error_m")
    sigma = number(row, "min_singular_value")
    flags = {
        "unreachable": not reachable,
        "solver_failure": not converged,
        "error_over_tolerance": error > error_tolerance_m,
        "near_singularity": sigma <= sigma_threshold,
    }
    flags["critical"] = flags["unreachable"] or flags["solver_failure"] or flags["error_over_tolerance"]
    flags["heuristic_risk"] = flags["near_singularity"]
    return flags


def confusion_counts(actual, predicted):
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same length")
    tp = fp = fn = tn = 0
    for a, p in zip(actual, predicted):
        if a and p:
            tp += 1
        elif p:
            fp += 1
        elif a:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def load_rows(source=SOURCE):
    rows = []
    with Path(source).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = [field for field in REQUIRED if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError("Missing required CSV columns: " + ", ".join(missing))
        for row in reader:
            for field in ("repeat", "sample_index", "elbow_sign", "branch_transitions_in_sequence"):
                try:
                    int(row[field])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid integer value in {field}: {row.get(field)!r}") from exc
            for field in ("actual_error_m", "min_singular_value", "joint_1_rad", "joint_2_rad"):
                number(row, field)
            as_bool(row["geometrically_reachable"])
            as_bool(row["converged"])
            rows.append(row)
    if not rows:
        raise ValueError(f"No data rows found in {source}")
    return rows


def summarize_rows(rows, sigma_threshold=DEFAULT_SIGMA_THRESHOLD):
    events = [classify_point(row, sigma_threshold) for row in rows]
    count = len(rows)
    return {
        "points": count,
        "unreachable_points": sum(e["unreachable"] for e in events),
        "solver_failure_points": sum(e["solver_failure"] for e in events),
        "error_over_tolerance_points": sum(e["error_over_tolerance"] for e in events),
        "near_singularity_points": sum(e["near_singularity"] for e in events),
        "critical_points": sum(e["critical"] for e in events),
        "success_rate": sum(as_bool(r["converged"]) for r in rows) / count if count else 0.0,
        "min_sigma": min(number(r, "min_singular_value") for r in rows) if count else math.nan,
        "max_error_m": max(number(r, "actual_error_m") for r in rows) if count else math.nan,
    }


def analyze_rows(rows, sigma_threshold=DEFAULT_SIGMA_THRESHOLD):
    groups = defaultdict(list)
    for row in rows:
        key = (row["case"], row["method"], int(row["repeat"]))
        groups[key].append(dict(row))
    analyzed = []
    sequence_summaries = []
    for key, group in sorted(groups.items()):
        group.sort(key=lambda r: int(r["sample_index"]))
        steps = []
        previous_q = None
        previous_sign = 0
        sign_changes = 0
        for row in group:
            q = (number(row, "joint_1_rad"), number(row, "joint_2_rad"))
            step = 0.0 if previous_q is None else math.hypot(q[0] - previous_q[0], q[1] - previous_q[1])
            row["joint_step_rad"] = step
            if previous_q is not None:
                steps.append(step)
            sign = int(row["elbow_sign"])
            changed = bool(sign and previous_sign and sign != previous_sign)
            row["observed_elbow_sign_change"] = changed
            sign_changes += int(changed)
            if sign:
                previous_sign = sign
            previous_q = q
        median = statistics.median(steps) if steps else 0.0
        mad = statistics.median([abs(value - median) for value in steps]) if steps else 0.0
        jump_threshold = max(JOINT_STEP_FLOOR_RAD, median + 6 * 1.4826 * mad)
        for row in group:
            flags = classify_point(row, sigma_threshold)
            row.update(flags)
            row["joint_step_threshold_rad"] = jump_threshold
            row["joint_step_outlier"] = int(row["sample_index"]) > 0 and row["joint_step_rad"] >= jump_threshold
            row["risk_level"] = "critical" if flags["critical"] else ("high" if flags["near_singularity"] or row["joint_step_outlier"] else "normal")
            analyzed.append(row)
        sequence_summaries.append(summarize_rows(group, sigma_threshold) | {
            "case": key[0], "method": key[1], "repeat": key[2],
            "observed_elbow_sign_changes": sign_changes,
            "reported_branch_transitions": max(int(r["branch_transitions_in_sequence"]) for r in group),
            "joint_step_outliers": sum(bool(r["joint_step_outlier"]) for r in group),
        })
    return analyzed, sequence_summaries


def evaluate_thresholds(rows, thresholds):
    eligible = [r for r in rows if as_bool(r["geometrically_reachable"])]
    actual = [not as_bool(r["converged"]) or number(r, "actual_error_m") > ERROR_TOLERANCE_M for r in eligible]
    results = []
    for threshold in thresholds:
        predicted = [number(r, "min_singular_value") <= threshold for r in eligible]
        counts = confusion_counts(actual, predicted)
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        results.append({
            "threshold": float(threshold), "eligible_reachable_points": len(eligible),
            "flagged_points": sum(predicted), **counts,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "actual_failure_points": sum(actual),
        })
    return results


def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def generate_outputs(source=SOURCE, out_dir=OUT):
    rows = load_rows(source)
    analyzed, sequences = analyze_rows(rows)
    thresholds = evaluate_thresholds(rows, [0.005, 0.01, 0.02, 0.03, 0.05, 0.1])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    point_fields = ["case", "method", "repeat", "sample_index", "geometrically_reachable", "converged",
                    "actual_error_m", "min_singular_value", "joint_step_rad", "joint_step_threshold_rad",
                    "unreachable", "solver_failure", "error_over_tolerance", "near_singularity",
                    "joint_step_outlier", "observed_elbow_sign_change", "risk_level"]
    write_csv(out_dir / "路径风险逐点验证.csv", analyzed, point_fields)
    sequence_fields = ["case", "method", "repeat", "points", "success_rate", "min_sigma", "max_error_m",
                       "unreachable_points", "solver_failure_points", "error_over_tolerance_points",
                       "near_singularity_points", "critical_points", "observed_elbow_sign_changes",
                       "reported_branch_transitions", "joint_step_outliers"]
    write_csv(out_dir / "路径风险工况验证.csv", sequences, sequence_fields)
    threshold_fields = ["threshold", "eligible_reachable_points", "flagged_points", "tp", "fp", "fn", "tn",
                        "precision", "recall", "actual_failure_points"]
    write_csv(out_dir / "路径风险阈值敏感性.csv", thresholds, threshold_fields)
    return rows, analyzed, sequences, thresholds


if __name__ == "__main__":
    rows, analyzed, sequences, thresholds = generate_outputs()
    print(f"SOURCE_POINTS={len(rows)}")
    print(f"SEQUENCES={len(sequences)}")
    print(f"ANALYZED_POINTS={len(analyzed)}")
    print(f"OUTPUT_DIR={OUT}")
    print("SIGMA_SWEEP=" + repr(thresholds))
