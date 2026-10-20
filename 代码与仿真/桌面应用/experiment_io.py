"""Portable experiment-package serialization for the robot-arm simulator."""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1


def write_experiment_package(directory, configuration, times, positions, angles, velocities, accelerations, torques):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    arrays = {
        "times": np.asarray(times, dtype=float),
        "positions": np.asarray(positions, dtype=float),
        "angles": np.asarray(angles, dtype=float),
        "velocities": np.asarray(velocities, dtype=float),
        "accelerations": np.asarray(accelerations, dtype=float),
        "torques": np.asarray(torques, dtype=float),
    }
    count = len(arrays["times"])
    joints = len(configuration["model"]["link_lengths_m"])
    if count:
        for key in ("positions", "angles", "velocities", "accelerations", "torques"):
            if len(arrays[key]) != count:
                raise ValueError(f"{key} 的采样数与时间数组不一致")
        for key in ("angles", "velocities", "accelerations", "torques"):
            if arrays[key].shape != (count, joints):
                raise ValueError(f"{key} 的数据形状应为 ({count}, {joints})")
        if not np.all(np.isfinite(arrays["times"])) or any(not np.all(np.isfinite(arrays[k])) for k in arrays):
            raise ValueError("实验数据包含非有限数值")
    config_doc = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "application": "机械臂交互式仿真平台",
        "configuration": configuration,
    }
    (root / "experiment.json").write_text(json.dumps(config_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    headers = ["time_s"]
    for node in range(joints + 1):
        headers.extend([f"node_{node}_x_m", f"node_{node}_y_m"])
    for joint in range(joints):
        headers.extend([f"joint_{joint+1}_angle_rad", f"joint_{joint+1}_velocity_rad_s",
                        f"joint_{joint+1}_acceleration_rad_s2", f"joint_{joint+1}_torque_Nm"])
    headers.extend([f"node_{joints}_path_length_m"])
    with (root / "results.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        path_length = 0.0
        previous = None
        for i in range(count):
            row = [float(arrays["times"][i])]
            point_set = arrays["positions"][i]
            if point_set.shape != (joints + 1, 2):
                raise ValueError("末端/关节位置数组的节点数与机械臂连杆数不一致")
            endpoint = point_set[-1]
            if previous is not None:
                path_length += float(np.linalg.norm(endpoint - previous))
            previous = endpoint.copy()
            for point in point_set:
                row.extend([float(point[0]), float(point[1])])
            for j in range(joints):
                row.extend([float(arrays["angles"][i, j]), float(arrays["velocities"][i, j]),
                            float(arrays["accelerations"][i, j]), float(arrays["torques"][i, j])])
            row.append(path_length)
            writer.writerow(row)
    summary = {
        "sample_count": count,
        "duration_s": float(arrays["times"][-1]) if count else 0.0,
        "joint_count": joints,
        "max_abs_joint_angle_rad": np.max(np.abs(arrays["angles"]), axis=0).tolist() if count else [],
        "max_abs_joint_velocity_rad_s": np.max(np.abs(arrays["velocities"]), axis=0).tolist() if count else [],
        "max_abs_joint_acceleration_rad_s2": np.max(np.abs(arrays["accelerations"]), axis=0).tolist() if count else [],
        "max_abs_joint_torque_Nm": np.max(np.abs(arrays["torques"]), axis=0).tolist() if count else [],
        "end_effector_path_length_m": path_length,
        "data_note": "速度和加速度按规划轨迹导数记录；力矩为当前理想刚体模型估计值。",
    }
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    readme = """# 机械臂仿真实验记录

## 实验目的
（请填写本次实验要研究或验证的问题。）

## 模型与方法
模型参数、约束、轨迹锚点和曲线配置见 `experiment.json`。仿真时序数据见 `results.csv`。

## 结果摘要
采样点数、仿真时长、关节峰值和末端累计路径长度见 `summary.json`。

## 结论与局限
（请根据数据填写结论，并注明模型假设、数值误差及尚未验证的事项。）

## 文件说明
- `experiment.json`：实验配置与模型/算法条件
- `results.csv`：逐采样时刻的节点位置、关节状态、力矩和末端累计路径长度
- `summary.json`：基础统计摘要
- `README.md`：实验目的、方法和结论记录
"""
    (root / "README.md").write_text(readme, encoding="utf-8")
    return summary


def read_experiment_package(config_path):
    path = Path(config_path)
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"不支持的实验文件版本：{document.get('schema_version')}")
    config = document["configuration"]
    joints = len(config["model"]["link_lengths_m"])
    rows = []
    with (path.parent / "results.csv").open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    times, positions, angles, velocities, accelerations, torques = [], [], [], [], [], []
    for row in rows:
        times.append(float(row["time_s"]))
        positions.append([[float(row[f"node_{n}_{axis}_m"]) for axis in ("x", "y")] for n in range(joints + 1)])
        angles.append([float(row[f"joint_{j+1}_angle_rad"]) for j in range(joints)])
        velocities.append([float(row[f"joint_{j+1}_velocity_rad_s"]) for j in range(joints)])
        accelerations.append([float(row[f"joint_{j+1}_acceleration_rad_s2"]) for j in range(joints)])
        torques.append([float(row[f"joint_{j+1}_torque_Nm"]) for j in range(joints)])
    return {
        "configuration": config,
        "times": times,
        "positions": positions,
        "angles": angles,
        "velocities": velocities,
        "accelerations": accelerations,
        "torques": torques,
    }
