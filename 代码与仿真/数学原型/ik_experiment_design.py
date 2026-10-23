"""可复现的逆运动学自动实验场景生成器。"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _finite_vector(values: Any, name: str, *, positive: bool = False) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}必须是有限数值向量") from exc
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name}必须是非空有限数值向量")
    if positive and np.any(array <= 0):
        raise ValueError(f"{name}必须全部为正数")
    return array.copy()


def _target_matrix(values: Any, name: str = "targets_m") -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}必须是有限的 N×2 坐标数组") from exc
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] != 2:
        raise ValueError(f"{name}必须是非空 N×2 坐标数组")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name}必须全部为有限数值")
    return array.copy()


def _positive_integer(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name}必须是大于或等于 {minimum} 的整数")
    return int(value)


def _levels(values: Any, name: str) -> list[float]:
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}必须是非空有限非负数序列") from exc
    if result.ndim != 1 or result.size == 0 or not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ValueError(f"{name}必须是非空有限非负数序列")
    return [float(item) for item in result]


@dataclass(frozen=True)
class ExperimentScenario:
    """一个可直接交给实验执行流程的场景。角度单位 rad，坐标单位 m。"""

    scenario_id: str
    scenario_type: str
    targets_m: np.ndarray
    initial_angles_rad: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.scenario_id, str) or not self.scenario_id.strip():
            raise ValueError("scenario_id不能为空")
        if not isinstance(self.scenario_type, str) or not self.scenario_type.strip():
            raise ValueError("scenario_type不能为空")
        targets = _target_matrix(self.targets_m)
        initial = _finite_vector(self.initial_angles_rad, "initial_angles_rad")
        try:
            metadata = json.loads(json.dumps(self.metadata, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata必须可序列化为符合标准的JSON") from exc
        object.__setattr__(self, "targets_m", targets)
        object.__setattr__(self, "initial_angles_rad", initial)
        object.__setattr__(self, "metadata", metadata)

    def to_experiment_config(self, base_config: dict[str, Any]) -> dict[str, Any]:
        """将场景注入执行器配置，保留其他实验参数且不修改输入字典。"""
        if not isinstance(base_config, dict):
            raise ValueError("base_config必须是字典")
        config = dict(base_config)
        config["scenario_id"] = self.scenario_id
        config["targets_m"] = self.targets_m.tolist()
        config["initial_angles_rad"] = self.initial_angles_rad.tolist()
        config["scenario_metadata"] = dict(self.metadata)
        return config
    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "scenario_type": self.scenario_type,
            "targets_m": self.targets_m.tolist(),
            "initial_angles_rad": self.initial_angles_rad.tolist(),
            "metadata": self.metadata,
        }


def generate_workspace_boundary_scenarios(
    link_lengths_m: Iterable[float] = (0.6, 0.4),
    *,
    sample_count: int = 21,
    seed: int = 0,
    boundary_offset_m: float = 1e-4,
) -> list[ExperimentScenario]:
    """生成内边界、外边界和普通可达区三类径向目标序列。"""
    lengths = _finite_vector(link_lengths_m, "link_lengths_m", positive=True)
    count = _positive_integer(sample_count, "sample_count", minimum=3)
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed必须是非负整数")
    try:
        offset = float(boundary_offset_m)
    except (TypeError, ValueError) as exc:
        raise ValueError("boundary_offset_m必须是有限正数") from exc
    longest = float(np.max(lengths))
    inner = max(0.0, longest - (float(np.sum(lengths)) - longest))
    outer = float(np.sum(lengths))
    if not np.isfinite(offset) or offset <= 0 or 2 * offset >= outer - inner:
        raise ValueError("boundary_offset_m必须为正，且小于工作空间径向宽度的一半")

    rng = np.random.default_rng(int(seed))
    angle = rng.uniform(-np.pi, np.pi, count)
    radii = (
        np.full(count, inner + offset),
        np.full(count, outer - offset),
        np.full(count, (inner + outer) / 2.0),
    )
    labels = ("inner_boundary", "outer_boundary", "regular_reachable")
    scenarios = []
    for label, radius in zip(labels, radii):
        targets = np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))
        initial = rng.uniform(-np.pi, np.pi, lengths.size)
        scenarios.append(ExperimentScenario(
            scenario_id=f"workspace-{label}",
            scenario_type="workspace_boundary",
            targets_m=targets,
            initial_angles_rad=initial,
            metadata={
                "region": label,
                "link_lengths_m": lengths.tolist(),
                "inner_radius_m": inner,
                "outer_radius_m": outer,
                "sample_count": count,
                "boundary_offset_m": offset,
                "seed": int(seed),
            },
        ))
    return scenarios


def generate_initial_sensitivity_scenarios(
    targets_m: Any,
    initial_angles_rad: Any,
    perturbation_levels_rad: Iterable[float],
    repeats_per_level: int,
    *,
    seed: int = 0,
) -> list[ExperimentScenario]:
    """保持目标序列不变，对初始关节角施加可复现的均匀随机扰动。"""
    targets = _target_matrix(targets_m)
    initial = _finite_vector(initial_angles_rad, "initial_angles_rad")
    levels = _levels(perturbation_levels_rad, "perturbation_levels_rad")
    repeats = _positive_integer(repeats_per_level, "repeats_per_level")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed必须是非负整数")
    rng = np.random.default_rng(int(seed))
    scenarios = []
    for level_index, level in enumerate(levels):
        for repeat in range(repeats):
            perturbed = initial.copy() if level == 0 else initial + rng.uniform(-level, level, initial.size)
            scenarios.append(ExperimentScenario(
                scenario_id=f"initial-sensitivity-l{level_index}-r{repeat}",
                scenario_type="initial_sensitivity",
                targets_m=targets,
                initial_angles_rad=perturbed,
                metadata={
                    "perturbation_level_rad": level,
                    "repeat_index": repeat,
                    "level_index": level_index,
                    "seed": int(seed),
                    "perturbation_distribution": "uniform_per_joint",
                },
            ))
    return scenarios


def generate_target_perturbation_scenarios(
    targets_m: Any,
    initial_angles_rad: Any,
    perturbation_levels_m: Iterable[float],
    repeats_per_level: int,
    *,
    seed: int = 0,
) -> list[ExperimentScenario]:
    """保持初值不变，对基准目标序列施加独立高斯坐标扰动。"""
    targets = _target_matrix(targets_m)
    initial = _finite_vector(initial_angles_rad, "initial_angles_rad")
    levels = _levels(perturbation_levels_m, "perturbation_levels_m")
    repeats = _positive_integer(repeats_per_level, "repeats_per_level")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed必须是非负整数")
    rng = np.random.default_rng(int(seed))
    scenarios = []
    for level_index, level in enumerate(levels):
        for repeat in range(repeats):
            perturbed = targets.copy() if level == 0 else targets + rng.normal(0.0, level, targets.shape)
            scenarios.append(ExperimentScenario(
                scenario_id=f"target-perturbation-l{level_index}-r{repeat}",
                scenario_type="target_perturbation",
                targets_m=perturbed,
                initial_angles_rad=initial,
                metadata={
                    "perturbation_sigma_m": level,
                    "repeat_index": repeat,
                    "level_index": level_index,
                    "seed": int(seed),
                    "perturbation_distribution": "independent_gaussian_xy",
                },
            ))
    return scenarios


def write_scenario_manifest(
    output_path: str | Path,
    scenarios: Iterable[ExperimentScenario],
    *,
    seed: int,
) -> Path:
    """将场景列表写入JSON清单；拒绝覆盖已有文件。"""
    path = Path(output_path)
    if path.exists():
        raise FileExistsError(f"拒绝覆盖已有场景清单：{path}")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed必须是非负整数")
    items = list(scenarios)
    if not items or any(not isinstance(item, ExperimentScenario) for item in items):
        raise ValueError("scenarios必须是非空ExperimentScenario序列")
    ids = [item.scenario_id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("scenario_id必须唯一")
    payload = {"schema_version": 1, "seed": int(seed), "scenario_count": len(items),
               "scenarios": [item.to_dict() for item in items]}
    try:
        serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    except (TypeError, ValueError) as exc:
        raise ValueError("场景清单无法序列化为标准JSON") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")
    return path



