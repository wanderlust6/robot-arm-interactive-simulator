"""两连杆平面机械臂的解析逆运动学。"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InverseKinematicsResult:
    """保存逆运动学求解结果。

    solutions 中每组角度依次为第一、第二关节角，单位为弧度。
    若 reachable 为 True 但 solutions 为空，表示目标位于基座且两根
    连杆等长；此时存在无穷多组折叠解，无法用有限列表完整表示。
    """

    solutions: tuple[tuple[float, float], ...]
    reachable: bool
    singular: bool
    message: str


def solve_two_link_ik(
    target: Sequence[float],
    link_lengths: Sequence[float],
    tolerance: float = 1e-9,
) -> InverseKinematicsResult:
    """求解两连杆平面机械臂的解析逆运动学。

    第一关节角相对于 x 轴正方向，第二关节角相对于第一根连杆。
    目标坐标与连杆长度必须使用相同的长度单位。

    当两种肘部构型不同且都存在时，返回两组解。目标位于可达工作空间
    之外时，返回 reachable=False。工作空间边界附近的微小浮点误差由
    tolerance 控制。
    """
    try:
        point = np.asarray(target, dtype=float)
        lengths = np.asarray(link_lengths, dtype=float)
        tol = float(tolerance)
    except (TypeError, ValueError) as exc:
        raise ValueError("目标坐标、连杆长度和容差都必须是数值") from exc

    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError("目标坐标必须恰好包含两个有限数值")
    if lengths.shape != (2,) or not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
        raise ValueError("连杆长度必须恰好包含两个有限的正数")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("容差必须是有限的非负数")

    x, y = point
    l1, l2 = lengths
    radius = float(np.hypot(x, y))
    inner_radius = abs(float(l1 - l2))
    outer_radius = float(l1 + l2)

    if radius > outer_radius + tol or radius < inner_radius - tol:
        return InverseKinematicsResult(
            solutions=(),
            reachable=False,
            singular=False,
            message="目标点不可达：目标距离超出机械臂的可达工作空间。",
        )

    if radius == 0.0 and l1 == l2:
        return InverseKinematicsResult(
            solutions=(),
            reachable=True,
            singular=True,
            message="目标点位于基座，且两根连杆等长；存在无穷多组折叠解。",
        )

    cos_theta2 = (
        radius * radius - l1 * l1 - l2 * l2
    ) / (2.0 * l1 * l2)
    cos_theta2 = float(np.clip(cos_theta2, -1.0, 1.0))
    elbow_angle = float(np.arccos(cos_theta2))

    candidate_angles = [elbow_angle]
    if not (
        np.isclose(elbow_angle, 0.0, atol=1e-12, rtol=0.0)
        or np.isclose(elbow_angle, np.pi, atol=1e-12, rtol=0.0)
    ):
        candidate_angles.append(-elbow_angle)

    target_direction = float(np.arctan2(y, x))
    solutions = []
    for theta2 in candidate_angles:
        theta1 = target_direction - float(
            np.arctan2(l2 * np.sin(theta2), l1 + l2 * np.cos(theta2))
        )
        solutions.append((theta1, theta2))

    singular = len(solutions) == 1
    message = (
        "目标可达，但位于工作空间边界的奇异构型。"
        if singular
        else "目标可达，已返回两组肘部构型解。"
    )
    return InverseKinematicsResult(
        solutions=tuple(solutions),
        reachable=True,
        singular=singular,
        message=message,
    )
