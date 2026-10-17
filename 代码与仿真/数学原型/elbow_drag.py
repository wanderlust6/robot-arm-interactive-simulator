"""两连杆机械臂沿固定末端直线拖动肘关节的几何求解。"""

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np


@dataclass(frozen=True)
class ElbowDragCandidate:
    """保存一次肘关节拖动对应的可行机械臂姿态。"""

    joint_angles: tuple[float, float]
    target: tuple[float, float]
    elbow: tuple[float, float]


def solve_elbow_drag_candidates(
    mouse_position: Sequence[float],
    link_lengths: Sequence[float],
    line_direction: Sequence[float],
    previous_target: Sequence[float],
    tolerance: float = 1e-9,
) -> tuple[ElbowDragCandidate, ...]:
    """求解肘关节拖动时，末端位于固定直线且连杆长度不变的候选姿态。

    鼠标决定肘关节的期望方向；肘关节实际位置投影到以基座为圆心、
    第一根连杆长度为半径的圆上。末端被限制在拖动开始时确定的固定直线上，
    并与肘关节保持第二根连杆长度。若鼠标方向没有几何可行解，则将肘关节
    方向限制到最近的可行边界。

    返回的候选解按末端与 previous_target 的距离从近到远排序。
    """

    try:
        mouse = np.asarray(mouse_position, dtype=float)
        lengths = np.asarray(link_lengths, dtype=float)
        direction = np.asarray(line_direction, dtype=float)
        old_target = np.asarray(previous_target, dtype=float)
        tol = float(tolerance)
    except (TypeError, ValueError) as exc:
        raise ValueError("输入参数必须为数值") from exc

    if mouse.shape != (2,) or not np.all(np.isfinite(mouse)):
        raise ValueError("鼠标位置必须包含两个有限数值")
    if lengths.shape != (2,) or not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
        raise ValueError("两根连杆长度必须为有限正数")
    if direction.shape != (2,) or not np.all(np.isfinite(direction)):
        raise ValueError("固定直线方向必须包含两个有限数值")
    if old_target.shape != (2,) or not np.all(np.isfinite(old_target)):
        raise ValueError("原末端位置必须包含两个有限数值")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("容差必须为有限的非负数")

    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= tol:
        raise ValueError("固定直线方向不能为零向量")
    direction = direction / direction_norm

    l1, l2 = map(float, lengths)
    mouse_norm = float(np.linalg.norm(mouse))
    if mouse_norm <= tol:
        mouse_angle = float(np.arctan2(old_target[1], old_target[0]))
    else:
        mouse_angle = float(np.arctan2(mouse[1], mouse[0]))

    line_angle = float(np.arctan2(direction[1], direction[0]))

    if l2 < l1:
        max_offset = float(np.arcsin(np.clip(l2 / l1, 0.0, 1.0)))
        candidate_directions = (line_angle, line_angle + np.pi)
        offsets = [
            (angle + np.pi) % (2.0 * np.pi) - np.pi
            for angle in (mouse_angle - candidate_directions[0],
                          mouse_angle - candidate_directions[1])
        ]
        nearest_index = int(np.argmin(np.abs(offsets)))
        base_angle = candidate_directions[nearest_index]
        offset = offsets[nearest_index]
        mouse_angle = base_angle + float(np.clip(offset, -max_offset, max_offset))

    elbow = l1 * np.array([np.cos(mouse_angle), np.sin(mouse_angle)])
    along = float(np.dot(elbow, direction))
    perpendicular_sq = max(0.0, float(np.dot(elbow, elbow)) - along * along)
    discriminant = l2 * l2 - perpendicular_sq
    if discriminant < -tol:
        return ()
    root = float(np.sqrt(max(0.0, discriminant)))
    distances = (along - root, along + root)

    candidates = []
    seen_targets = []
    for distance in distances:
        target = distance * direction
        if any(np.linalg.norm(target - existing) <= tol for existing in seen_targets):
            continue
        seen_targets.append(target)

        second_vector = target - elbow
        theta1 = float(np.arctan2(elbow[1], elbow[0]))
        absolute_theta2 = float(np.arctan2(second_vector[1], second_vector[0]))
        theta2 = (absolute_theta2 - theta1 + np.pi) % (2.0 * np.pi) - np.pi

        candidates.append(
            ElbowDragCandidate(
                joint_angles=(theta1, theta2),
                target=(float(target[0]), float(target[1])),
                elbow=(float(elbow[0]), float(elbow[1])),
            )
        )

    candidates.sort(
        key=lambda item: float(np.linalg.norm(np.asarray(item.target) - old_target))
    )
    return tuple(candidates)
