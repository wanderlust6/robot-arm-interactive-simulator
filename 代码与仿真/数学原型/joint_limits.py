"""二维或多关节机械臂的关节角限位检查工具。"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RejectedIKSolution:
    """记录一组违反关节限位的逆运动学解。"""

    joint_angles: tuple[float, ...]
    violating_joint_indices: tuple[int, ...]


@dataclass(frozen=True)
class IKSolutionFilterResult:
    """保存通过限位检查的解，以及被拒绝的解。"""

    valid_solutions: tuple[tuple[float, ...], ...]
    rejected_solutions: tuple[RejectedIKSolution, ...]


class JointLimits:
    """保存每个关节的最小角度和最大角度，角度单位为弧度。"""

    def __init__(
        self,
        lower: Sequence[float],
        upper: Sequence[float],
    ) -> None:
        try:
            lower_values = np.asarray(lower, dtype=float)
            upper_values = np.asarray(upper, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("关节限位必须是数值序列") from exc

        if (
            lower_values.ndim != 1
            or upper_values.ndim != 1
            or lower_values.size == 0
            or lower_values.shape != upper_values.shape
        ):
            raise ValueError("上下限必须是长度相同且非空的一维序列")
        if not np.all(np.isfinite(lower_values)) or not np.all(np.isfinite(upper_values)):
            raise ValueError("关节角上下限必须是有限数值")
        if np.any(lower_values > upper_values):
            raise ValueError("每个关节的下限都不能大于上限")

        self._lower = lower_values.copy()
        self._upper = upper_values.copy()

    @property
    def joint_count(self) -> int:
        """返回配置了限位的关节数量。"""
        return int(self._lower.size)

    @property
    def lower(self) -> np.ndarray:
        """返回各关节下限的副本，避免外部修改内部配置。"""
        return self._lower.copy()

    @property
    def upper(self) -> np.ndarray:
        """返回各关节上限的副本，避免外部修改内部配置。"""
        return self._upper.copy()

    def violations(
        self,
        joint_angles: Sequence[float],
        tolerance: float = 1e-9,
    ) -> tuple[int, ...]:
        """返回违反限位的关节索引；容差用于吸收边界附近的数值误差。"""
        try:
            angles = np.asarray(joint_angles, dtype=float)
            tol = float(tolerance)
        except (TypeError, ValueError) as exc:
            raise ValueError("关节角和容差必须是数值") from exc

        if angles.ndim != 1 or angles.size != self.joint_count:
            raise ValueError("关节角数量必须与已配置的限位数量一致")
        if not np.all(np.isfinite(angles)):
            raise ValueError("关节角必须是有限数值")
        if not np.isfinite(tol) or tol < 0:
            raise ValueError("容差必须是有限的非负数")

        outside = (angles < self._lower - tol) | (angles > self._upper + tol)
        return tuple(int(index) for index in np.flatnonzero(outside))

    def contains(
        self,
        joint_angles: Sequence[float],
        tolerance: float = 1e-9,
    ) -> bool:
        """判断一组关节角是否满足全部关节限位。"""
        return not self.violations(joint_angles, tolerance)


def filter_ik_solutions(
    solutions: Sequence[Sequence[float]],
    joint_limits: JointLimits,
    tolerance: float = 1e-9,
) -> IKSolutionFilterResult:
    """按关节限位筛选逆运动学解，不改变几何可达性的判断结果。"""
    try:
        tol = float(tolerance)
    except (TypeError, ValueError) as exc:
        raise ValueError("容差必须是数值") from exc
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("容差必须是有限的非负数")

    valid_solutions = []
    rejected_solutions = []

    for solution in solutions:
        try:
            angles = np.asarray(solution, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("每组逆运动学解都必须是数值序列") from exc

        violations = joint_limits.violations(angles, tol)
        normalized_angles = tuple(float(angle) for angle in angles)
        if violations:
            rejected_solutions.append(
                RejectedIKSolution(
                    joint_angles=normalized_angles,
                    violating_joint_indices=violations,
                )
            )
        else:
            valid_solutions.append(normalized_angles)

    return IKSolutionFilterResult(
        valid_solutions=tuple(valid_solutions),
        rejected_solutions=tuple(rejected_solutions),
    )
