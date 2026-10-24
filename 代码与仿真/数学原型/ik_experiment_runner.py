"""Config-driven paired experiments for the current and adaptive DLS IK solvers."""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from compare_damping_strategies import adaptive_ik
from numerical_ik import solve_n_link_ik
from planar_arm import PlanarArm

METHODS = ("current_dls", "sigma_adaptive_dls")


def validate_config(config):
    """Validate the stable, JSON-serializable experiment input schema."""
    if not isinstance(config, dict):
        raise ValueError("实验配置必须是 JSON 对象")
    for key in ("experiment_id", "scenario_id", "robot", "solver"):
        if key not in config:
            raise ValueError(f"缺少必需配置项: {key}")
    for key in ("experiment_id", "scenario_id"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} 必须是非空字符串")
    if not isinstance(config["robot"], dict):
        raise ValueError("robot 必须是 JSON 对象")
    if not isinstance(config["solver"], dict):
        raise ValueError("solver 必须是 JSON 对象")
    links = config["robot"].get("link_lengths_m")
    if not isinstance(links, list) or not links or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in links
    ):
        raise ValueError("robot.link_lengths_m 必须是有限正数构成的非空列表")

    solver = config["solver"]
    required = ("max_iterations", "tolerance_m", "base_damping", "sigma_threshold", "gain", "max_damping")
    for key in required:
        if key not in solver:
            raise ValueError(f"缺少必需配置项: solver.{key}")
    if isinstance(solver["max_iterations"], bool) or not isinstance(solver["max_iterations"], int) or solver["max_iterations"] <= 0:
        raise ValueError("solver.max_iterations 必须是正整数")
    for key in ("tolerance_m", "base_damping", "max_damping"):
        value = solver[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"solver.{key} 必须是有限正数")
    for key in ("sigma_threshold", "gain"):
        value = solver[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"solver.{key} 必须是有限非负数")
    if solver["max_damping"] < solver["base_damping"]:
        raise ValueError("solver.max_damping 不能小于 solver.base_damping")
    return config


def validate_experiment_inputs(config):
    """Validate target path and seed vectors from the CLI experiment schema."""
    if not isinstance(config, dict):
        raise ValueError("experiment config must be a JSON object")
    if "targets_m" not in config:
        raise ValueError("missing required field: targets_m")
    if "initial_angles_rad" not in config:
        raise ValueError("missing required field: initial_angles_rad")
    try:
        targets = np.asarray(config["targets_m"], dtype=float)
        initial = np.asarray(config["initial_angles_rad"], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("targets_m and initial_angles_rad must contain numeric values") from exc
    if targets.ndim != 2 or targets.shape[1] != 2 or targets.shape[0] == 0 or not np.all(np.isfinite(targets)):
        raise ValueError("targets_m must be a non-empty finite N×2 array")
    robot = config.get("robot")
    links = robot.get("link_lengths_m") if isinstance(robot, dict) else None
    if not isinstance(links, list):
        raise ValueError("robot.link_lengths_m must be a list")
    if initial.shape != (len(links),) or not np.all(np.isfinite(initial)):
        raise ValueError("initial_angles_rad must be a finite vector matching robot degrees of freedom")
    return targets, initial


def _sequence_summary(rows, tolerance):
    errors = [row["actual_error_m"] for row in rows]
    times = [row["solve_time_ms"] for row in rows]
    failed = next((row for row in rows if not row["point_success"]), None)
    return {
        "method": rows[0]["method"],
        "sample_count": len(rows),
        "successful_points": sum(row["point_success"] for row in rows),
        "unreachable_target_count": sum(not row["target_reachable"] for row in rows),
        "solver_failure_count": sum(not row["converged"] for row in rows),
        "sequence_success": bool(rows) and all(row["point_success"] for row in rows),
        "success_rate": sum(row["point_success"] for row in rows) / len(rows),
        "first_failure_index": None if failed is None else failed["sample_index"],
        "first_failure_code": None if failed is None else failed["failure_code"],
        "max_endpoint_error_m": max(errors),
        "p95_endpoint_error_m": float(np.percentile(errors, 95)),
        "max_joint_step_rad": max(row["joint_step_rad"] for row in rows),
        "min_singular_value": min(row["min_singular_value_after"] for row in rows),
        "total_solve_time_ms": sum(times),
        "p95_solve_time_ms": float(np.percentile(times, 95)),
        "tolerance_m": tolerance,
    }


def run_paired_sequence(arm, targets, initial_angles, config):
    """Run both methods on identical targets and initial angles without changing solvers."""
    validate_config(config)
    targets = np.asarray(targets, dtype=float)
    initial = np.asarray(initial_angles, dtype=float)
    if targets.ndim != 2 or targets.shape[1] != 2 or len(targets) == 0 or not np.all(np.isfinite(targets)):
        raise ValueError("targets 必须是非空、有限的 N×2 坐标数组")
    if initial.shape != (arm.joint_count,) or not np.all(np.isfinite(initial)):
        raise ValueError("initial_angles 必须与机械臂关节数一致且均为有限值")
    if len(config["robot"]["link_lengths_m"]) != arm.joint_count:
        raise ValueError("配置中的连杆数量与机械臂模型不一致")
    if not np.allclose(config["robot"]["link_lengths_m"], arm.link_lengths):
        raise ValueError("配置中的连杆长度与机械臂模型不一致")

    solver = config["solver"]
    link_lengths = np.asarray(arm.link_lengths, dtype=float)
    largest_link = float(np.max(link_lengths))
    inner_radius = max(0.0, largest_link - (float(np.sum(link_lengths)) - largest_link))
    outer_radius = float(np.sum(link_lengths))
    target_radii = np.linalg.norm(targets, axis=1)
    target_reachable = (target_radii >= inner_radius - solver["tolerance_m"]) & (target_radii <= outer_radius + solver["tolerance_m"])
    rows_by_method = {}
    summaries = {}
    for method in METHODS:
        q = initial.copy()
        rows = []
        for index, target in enumerate(targets):
            q_before = q.copy()
            before_metrics = arm.singularity_metrics(q_before)
            started = time.perf_counter()
            iterations = None
            peak_damping = None
            if method == "current_dls":
                solution, reported_error, converged = solve_n_link_ik(
                    arm, target, q_before, max_iterations=solver["max_iterations"],
                    tolerance=solver["tolerance_m"], damping=solver["base_damping"])
            else:
                solution, reported_error, converged, iterations, peak_damping = adaptive_ik(
                    arm, target, q_before, max_iterations=solver["max_iterations"],
                    tolerance=solver["tolerance_m"], base_damping=solver["base_damping"],
                    sigma_threshold=solver["sigma_threshold"], gain=solver["gain"],
                    max_damping=solver["max_damping"])
            elapsed_ms = (time.perf_counter() - started) * 1000
            solution = np.asarray(solution, dtype=float)
            actual_error = float(np.linalg.norm(arm.forward_kinematics(solution)[-1] - target))
            after_metrics = arm.singularity_metrics(solution)
            reachable = bool(target_reachable[index])
            point_success = reachable and bool(converged) and actual_error <= solver["tolerance_m"]
            failure_reasons = []
            if not reachable:
                failure_reasons.append("unreachable_target")
            if not converged:
                failure_reasons.append("non_convergence")
            elif actual_error > solver["tolerance_m"]:
                failure_reasons.append("endpoint_error_exceeded")
            failure_code = ";".join(failure_reasons)
            rows.append({
                "experiment_id": str(config["experiment_id"]),
                "scenario_id": str(config["scenario_id"]),
                "method": method,
                "sample_index": index,
                "target_x_m": float(target[0]),
                "target_y_m": float(target[1]),
                "q_before_rad": json.dumps(q_before.tolist()),
                "q_after_rad": json.dumps(solution.tolist()),
                "reported_error_m": float(reported_error),
                "actual_error_m": actual_error,
                "converged": bool(converged),
                "point_success": point_success,
                "target_reachable": reachable,
                "target_radius_m": float(target_radii[index]),
                "workspace_inner_radius_m": inner_radius,
                "workspace_outer_radius_m": outer_radius,
                "failure_code": failure_code,
                "iterations": iterations,
                "solve_time_ms": elapsed_ms,
                "min_singular_value_before": float(before_metrics["min_singular_value"]),
                "min_singular_value_after": float(after_metrics["min_singular_value"]),
                "condition_number_after": float(after_metrics["condition_number"]),
                "jacobian_rank_after": int(after_metrics["rank"]),
                "joint_step_rad": float(np.linalg.norm(solution - q_before)),
                "peak_damping": peak_damping,
            })
            q = solution.copy()
        rows_by_method[method] = rows
        summaries[method] = _sequence_summary(rows, solver["tolerance_m"])
    return {"rows_by_method": rows_by_method, "summaries": summaries}


def write_results(output_dir, config, result):
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"拒绝覆盖非空输出目录: {output}")
    if not any(result.get("rows_by_method", {}).get(method) for method in METHODS):
        raise ValueError("没有可写入的逐点结果")
    config_json = json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False)
    summary = {
        "experiment_id": config["experiment_id"],
        "scenario_id": config["scenario_id"],
        "solver_config": config["solver"],
        "summaries": result["summaries"],
    }
    summary_json = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)
    all_rows = [row for method in METHODS for row in result["rows_by_method"][method]]
    output.mkdir(parents=True, exist_ok=True)
    (output / "experiment_config.json").write_text(config_json, encoding="utf-8")
    (output / "summary.json").write_text(summary_json, encoding="utf-8")
    with (output / "point_results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    with (output / "sequence_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(next(iter(result["summaries"].values()))))
        writer.writeheader()
        writer.writerows(result["summaries"][method] for method in METHODS)
    report = [
        "# E1 配对实验结果", "",
        f"- 实验：{config['experiment_id']}", f"- 工况：{config['scenario_id']}",
        f"- 目标点数：{len(result['rows_by_method'][METHODS[0]])}",
        f"- 末端误差容差：{config['solver']['tolerance_m']:.6g} m", "",
        "| 方法 | 序列成功 | 点成功率 | 最大末端误差 (m) | 最大关节步长 (rad) | P95 耗时 (ms) | 首次失败点 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = result["summaries"][method]
        report.append(
            f"| {method} | {'是' if row['sequence_success'] else '否'} | {row['success_rate']:.1%} | "
            f"{row['max_endpoint_error_m']:.6g} | {row['max_joint_step_rad']:.6g} | "
            f"{row['p95_solve_time_ms']:.6g} | {row['first_failure_index']} |")
    report += ["", "## 解释限制", "",
               "- 本报告只描述本次合成模型与配置下的结果，不代表实机性能或算法普遍优劣。",
               "- 当前 DLS 函数不返回迭代次数和内部阻尼历史，因此相应字段对该方法为空。",
               "- 最小奇异值在此记录求解前后状态；若用于提前预警，应以求解前状态与后续失败标签进行独立评估。", ""]
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="运行配置驱动的 DLS 配对 IK 实验")
    parser.add_argument("--config", required=True, type=Path, help="实验 JSON 配置文件")
    parser.add_argument("--output", required=True, type=Path, help="本次实验独立输出目录")
    args = parser.parse_args()
    config = validate_config(json.loads(args.config.read_text(encoding="utf-8")))
    try:
        targets, initial_angles = validate_experiment_inputs(config)
    except ValueError as exc:
        parser.error(str(exc))
    arm = PlanarArm(config["robot"]["link_lengths_m"])
    result = run_paired_sequence(arm, targets, initial_angles, config)
    write_results(args.output, config, result)
    print(f"实验完成：{config['experiment_id']} / {config['scenario_id']}")
    print(f"结果目录：{args.output}")


if __name__ == "__main__":
    main()

