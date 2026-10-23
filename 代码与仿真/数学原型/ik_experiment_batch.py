"""批量组织自动生成的逆运动学实验场景并汇总配对结果。"""

import argparse
import copy
import csv
import json
from pathlib import Path

from ik_experiment_runner import (
    METHODS,
    run_paired_sequence,
    validate_config,
    validate_experiment_inputs,
    write_results,
)
from planar_arm import PlanarArm
from ik_experiment_design import ExperimentScenario


def _validate_batch(scenarios, base_config, output_dir):
    if not isinstance(base_config, dict):
        raise ValueError("base_config必须是字典")
    validate_config(base_config)
    items = list(scenarios)
    if not items or any(not isinstance(item, ExperimentScenario) for item in items):
        raise ValueError("scenarios必须是非空ExperimentScenario序列")
    identifiers = [item.scenario_id for item in items]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("批次中的scenario_id必须唯一")
    for identifier in identifiers:
        if identifier in (".", "..") or any(char in identifier for char in ('/', '\\', ':')):
            raise ValueError(f"scenario_id不能用于安全的独立目录名：{identifier!r}")
    link_lengths = base_config["robot"]["link_lengths_m"]
    arm = PlanarArm(link_lengths)
    for scenario in items:
        if len(scenario.initial_angles_rad) != arm.joint_count:
            raise ValueError(f"场景{scenario.scenario_id}的初值维度与机械臂自由度不一致")
        config = scenario.to_experiment_config(copy.deepcopy(base_config))
        validate_config(config)
        targets, initial = validate_experiment_inputs(config)
        if len(config["robot"]["link_lengths_m"]) != arm.joint_count:
            raise ValueError("场景配置的机械臂自由度不一致")
        if targets.shape[0] == 0 or initial.shape != (arm.joint_count,):
            raise ValueError(f"场景{scenario.scenario_id}输入无效")
    output = Path(output_dir)
    if output.exists():
        if not output.is_dir():
            raise FileExistsError(f"批次输出路径不是目录：{output}")
        if any(output.iterdir()):
            raise FileExistsError(f"拒绝使用非空批次输出目录：{output}")
    return items, arm, output


def run_experiment_batch(scenarios, base_config, output_dir):
    """依次执行场景配对实验，输出逐场景结果和批次汇总。"""
    items, arm, output = _validate_batch(scenarios, base_config, output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scenario_summaries = []
    for scenario in items:
        config = scenario.to_experiment_config(copy.deepcopy(base_config))
        targets, initial = validate_experiment_inputs(config)
        result = run_paired_sequence(arm, targets, initial, config)
        write_results(output / scenario.scenario_id, config, result)
        scenario_summaries.append({
            "scenario_id": scenario.scenario_id,
            "scenario_type": scenario.scenario_type,
            "sample_count": len(targets),
            "scenario_metadata": scenario.metadata,
            "methods": result["summaries"],
        })

    payload = {
        "schema_version": 1,
        "experiment_id": base_config["experiment_id"],
        "scenario_count": len(scenario_summaries),
        "methods": list(METHODS),
        "scenarios": scenario_summaries,
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    rows = []
    for scenario in scenario_summaries:
        for method in METHODS:
            metrics = scenario["methods"][method]
            rows.append({
                "scenario_id": scenario["scenario_id"],
                "scenario_type": scenario["scenario_type"],
                "method": method,
                "sample_count": metrics["sample_count"],
                "sequence_success": metrics["sequence_success"],
                "success_rate": metrics["success_rate"],
                "unreachable_target_count": metrics["unreachable_target_count"],
                "solver_failure_count": metrics["solver_failure_count"],
                "max_endpoint_error_m": metrics["max_endpoint_error_m"],
                "p95_solve_time_ms": metrics["p95_solve_time_ms"],
            })
    # Serialize before writing either batch-level file to avoid partial output on invalid JSON metrics.
    csv_fields = list(rows[0])
    output_json = output / "batch_summary.json"
    output_csv = output / "batch_summary.csv"
    if output_json.exists() or output_csv.exists():
        raise FileExistsError("批次汇总文件已存在，拒绝覆盖")
    output_json.write_text(serialized, encoding="utf-8")
    with output_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(rows)
    return payload
def main():
    parser = argparse.ArgumentParser(description="运行场景清单中的配对逆运动学实验批次")
    parser.add_argument("--config", required=True, type=Path, help="基础实验 JSON 配置")
    parser.add_argument("--manifest", required=True, type=Path, help="自动实验设计导出的场景清单 JSON")
    parser.add_argument("--output", required=True, type=Path, help="新建的批次输出目录")
    args = parser.parse_args()
    base_config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("scenarios"), list):
        parser.error("场景清单必须是包含scenarios列表的JSON对象")
    scenarios = []
    for item in manifest["scenarios"]:
        if not isinstance(item, dict):
            parser.error("场景清单中的每个场景必须是JSON对象")
        try:
            scenarios.append(ExperimentScenario(
                scenario_id=item["scenario_id"],
                scenario_type=item["scenario_type"],
                targets_m=item["targets_m"],
                initial_angles_rad=item["initial_angles_rad"],
                metadata=item.get("metadata", {}),
            ))
        except (KeyError, TypeError, ValueError) as exc:
            parser.error(f"无效场景定义：{exc}")
    result = run_experiment_batch(scenarios, base_config, args.output)
    print(f"批次完成：{result['scenario_count']} 个场景")
    print(f"汇总文件：{args.output / 'batch_summary.json'}")


if __name__ == "__main__":
    main()


