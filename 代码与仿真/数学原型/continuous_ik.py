"""连续轨迹中的解析逆运动学解选择工具。"""

from collections.abc import Sequence

import numpy as np


def select_continuous_ik_solution(
    candidates: Sequence[Sequence[float]],
    previous: Sequence[float] | None,
) -> np.ndarray | None:
    """从候选关节角中选择与上一姿态最接近的一组。

    角度按 2π 周期展开到上一姿态附近，再比较关节空间中的欧氏距离。
    没有候选解时返回 None；没有上一姿态时保留候选列表中的第一组。
    此策略优先保持局部连续性，不保证跟随用户指定的肘部构型切换。
    """
    if len(candidates) == 0:
        return None

    angles = np.asarray(candidates, dtype=float)
    if angles.ndim != 2 or angles.shape[1] != 2 or not np.all(np.isfinite(angles)):
        raise ValueError("候选解必须是由两关节有限角度组成的二维序列")

    if previous is None:
        return angles[0].copy()

    previous_angles = np.asarray(previous, dtype=float)
    if previous_angles.shape != (2,) or not np.all(np.isfinite(previous_angles)):
        raise ValueError("上一姿态必须包含两个有限关节角")

    unwrapped = angles + 2.0 * np.pi * np.round(
        (previous_angles[None, :] - angles) / (2.0 * np.pi)
    )
    distances = np.linalg.norm(unwrapped - previous_angles[None, :], axis=1)
    return unwrapped[int(np.argmin(distances))].copy()
