"""从离散关节轨迹估计速度、加速度并计算逆动力学力矩。"""

from collections.abc import Sequence

import numpy as np

from dynamics import PlanarArmDynamics


def calculate_trajectory_torques(
    model: PlanarArmDynamics,
    times: Sequence[float],
    joint_angles: Sequence[Sequence[float]],
) -> np.ndarray:
    """根据采样时刻和相对关节角，返回各采样点的关节力矩（N·m）。

    关节速度与加速度由展开后的角度序列按时间数值微分估计。
    少于 3 个采样点时无法可靠估计加速度，暂按零加速度处理。
    """
    try:
        time_values = np.asarray(times, dtype=float)
        angles = np.asarray(joint_angles, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("时间和关节角必须是有限数值数组") from exc

    joint_count = model.arm.joint_count
    if time_values.ndim != 1:
        raise ValueError("时间必须是一维数组")
    if angles.ndim != 2 or angles.shape != (time_values.size, joint_count):
        raise ValueError("关节角数组形状必须为 (采样点数量, 关节数量)")
    if not np.all(np.isfinite(time_values)) or not np.all(np.isfinite(angles)):
        raise ValueError("时间和关节角必须全部为有限数值")
    if time_values.size > 1 and np.any(np.diff(time_values) <= 0.0):
        raise ValueError("采样时刻必须严格递增")

    sample_count = time_values.size
    if sample_count == 0:
        return np.empty((0, joint_count), dtype=float)

    unwrapped_angles = np.unwrap(angles, axis=0)
    velocities = np.zeros_like(unwrapped_angles)
    accelerations = np.zeros_like(unwrapped_angles)

    if sample_count >= 2:
        edge_order = 2 if sample_count >= 3 else 1
        velocities = np.gradient(
            unwrapped_angles, time_values, axis=0, edge_order=edge_order
        )
    if sample_count >= 3:
        edge_order = 2 if sample_count >= 3 else 1
        accelerations = np.gradient(
            velocities, time_values, axis=0, edge_order=edge_order
        )

    return np.asarray([
        model.inverse_dynamics(q, qd, qdd)
        for q, qd, qdd in zip(unwrapped_angles, velocities, accelerations)
    ])
