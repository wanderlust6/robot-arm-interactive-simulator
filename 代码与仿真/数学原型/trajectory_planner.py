"""基于五次时间标定的机械臂约束轨迹规划。"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.interpolate import PchipInterpolator

from dynamics import PlanarArmDynamics
from inverse_kinematics import solve_two_link_ik
from planar_arm import PlanarArm
from trajectory_dynamics import calculate_trajectory_torques


@dataclass(frozen=True)
class TrajectoryPlanResult:
    """规划结果及可诊断的失败信息。角度为 rad，力矩为 N·m。"""

    success: bool
    message: str
    times: np.ndarray
    joint_angles: np.ndarray
    joint_velocities: np.ndarray
    joint_accelerations: np.ndarray
    joint_torques: np.ndarray
    duration: float
    iterations: int
    path_progress: Optional[np.ndarray] = None
    path_joint_angles: Optional[np.ndarray] = None


def _validate_limits(values, count, name):
    try:
        limits = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}必须是有限正数") from exc
    if limits.ndim == 0:
        limits = np.full(count, float(limits))
    if limits.shape != (count,) or not np.all(np.isfinite(limits)) or np.any(limits <= 0):
        raise ValueError(f"{name}必须是正数，或长度为关节数的正数序列")
    return limits


def quintic_time_scaling(start, end, duration, sample_count=201):
    """生成五次时间标定轨迹，起终点速度和加速度均为零。"""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    duration = float(duration)
    if start.ndim != 1 or end.shape != start.shape or start.size == 0:
        raise ValueError("起点和终点必须是形状相同的非空关节角向量")
    if not np.all(np.isfinite(start)) or not np.all(np.isfinite(end)):
        raise ValueError("起点和终点角度必须为有限数值")
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("运动时间必须为正数")
    if isinstance(sample_count, bool) or int(sample_count) != sample_count or sample_count < 3:
        raise ValueError("采样点数必须是至少为 3 的整数")
    times = np.linspace(0.0, duration, int(sample_count))
    s = times / duration
    blend = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
    blend_d = (30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4) / duration
    blend_dd = (60.0 * s - 180.0 * s**2 + 120.0 * s**3) / duration**2
    delta = end - start
    angles = start + blend[:, None] * delta
    velocities = blend_d[:, None] * delta
    accelerations = blend_dd[:, None] * delta
    return times, angles, velocities, accelerations


def plan_joint_trajectory(
    model: PlanarArmDynamics,
    start_angles,
    end_angles,
    max_velocity,
    max_acceleration,
    max_torque,
    *,
    initial_duration=1.0,
    sample_count=201,
    max_duration=120.0,
    max_iterations=60,
    joint_limits=None,
    duration_growth=1.25,
) -> TrajectoryPlanResult:
    """固定关节空间直线路径，逐步延长时间直到速度/加速度/力矩满足限制。

    这是可行性搜索，不承诺全局最短时间。默认假定起终点静止；力矩使用
    逆动力学模型计算。若重力力矩本身超过限值，延长时间不会解决该问题。
    """
    arm = model.arm
    start = arm._validated_angles(start_angles)
    end = arm._validated_angles(end_angles)
    n = arm.joint_count
    v_limit = _validate_limits(max_velocity, n, "角速度上限")
    a_limit = _validate_limits(max_acceleration, n, "角加速度上限")
    t_limit = _validate_limits(max_torque, n, "力矩上限")
    duration = float(initial_duration)
    max_duration = float(max_duration)
    growth = float(duration_growth)
    if not np.isfinite(duration) or duration <= 0 or not np.isfinite(max_duration) or max_duration < duration:
        raise ValueError("初始运动时间和最大搜索时间必须有效，且最大时间不小于初始时间")
    if not np.isfinite(growth) or growth <= 1.0:
        raise ValueError("duration_growth 必须大于 1")
    if isinstance(max_iterations, bool) or int(max_iterations) != max_iterations or max_iterations < 1:
        raise ValueError("max_iterations 必须是正整数")
    if joint_limits is not None:
        try:
            lower, upper = (np.asarray(part, dtype=float) for part in joint_limits)
        except (TypeError, ValueError) as exc:
            raise ValueError("joint_limits 必须为 (下限, 上限) 两个向量") from exc
        if (lower.shape != (n,) or upper.shape != (n,) or not np.all(np.isfinite(lower))
                or not np.all(np.isfinite(upper)) or np.any(lower >= upper)):
            raise ValueError("关节角上下限必须是形状正确且下限小于上限的有限向量")
        if np.any(start < lower) or np.any(start > upper) or np.any(end < lower) or np.any(end > upper):
            empty = np.empty((0, n), dtype=float)
            return TrajectoryPlanResult(False, "起点或终点超出关节角限位。", np.empty(0), empty, empty, empty, empty, 0.0, 0)
        if np.any(np.minimum(start, end) < lower) or np.any(np.maximum(start, end) > upper):
            empty = np.empty((0, n), dtype=float)
            return TrajectoryPlanResult(False, "关节空间路径超出关节角限位。", np.empty(0), empty, empty, empty, empty, 0.0, 0)

    last_result = None
    for iteration in range(1, int(max_iterations) + 1):
        if duration > max_duration * (1.0 + 1e-12):
            break
        times, angles, velocities, accelerations = quintic_time_scaling(
            start, end, duration, sample_count
        )
        torques = np.asarray([
            model.inverse_dynamics(q, qd, qdd)
            for q, qd, qdd in zip(angles, velocities, accelerations)
        ])
        last_result = (times, angles, velocities, accelerations, torques)
        speed_ok = np.all(np.abs(velocities) <= v_limit[None, :] * (1.0 + 1e-9))
        accel_ok = np.all(np.abs(accelerations) <= a_limit[None, :] * (1.0 + 1e-9))
        torque_ok = np.all(np.abs(torques) <= t_limit[None, :] * (1.0 + 1e-9))
        if speed_ok and accel_ok and torque_ok:
            return TrajectoryPlanResult(True, "已找到满足所设速度、加速度和力矩上限的轨迹。", *last_result, duration, iteration)
        if duration >= max_duration:
            break
        duration = min(duration * growth, max_duration)

    if last_result is None:
        empty = np.empty((0, n), dtype=float)
        return TrajectoryPlanResult(False, "未生成候选轨迹；请检查时间搜索参数。", np.empty(0), empty, empty, empty, empty, 0.0, 0)
    times, angles, velocities, accelerations, torques = last_result
    reasons = []
    if np.any(np.abs(velocities) > v_limit[None, :] * (1.0 + 1e-9)):
        reasons.append("角速度")
    if np.any(np.abs(accelerations) > a_limit[None, :] * (1.0 + 1e-9)):
        reasons.append("角加速度")
    if np.any(np.abs(torques) > t_limit[None, :] * (1.0 + 1e-9)):
        reasons.append("驱动力矩")
    message = "在搜索时间范围内未找到满足全部约束的轨迹；仍超限：" + "、".join(reasons) + "。可能需要改变路径或放宽限制。"
    return TrajectoryPlanResult(False, message, times, angles, velocities, accelerations, torques, float(times[-1]), min(int(max_iterations), iteration))


def plan_cartesian_line(
    link_lengths,
    start_point,
    end_point,
    *,
    sample_count=201,
    elbow_sign=None,
    joint_limits=None,
):
    """将两连杆机械臂的笛卡尔直线段离散成连续关节角路径。

    elbow_sign 可为 None、1 或 -1。None 时选择与前一采样解最接近的 IK 解；
    显式指定符号可固定肘部构型。若路径任一点不可达或关节限位被违反，则报错。
    返回 (末端路径点, 关节角路径)。
    """
    arm = PlanarArm(link_lengths)
    if arm.joint_count != 2:
        raise ValueError("当前解析笛卡尔直线规划仅支持两连杆平面机械臂")
    start = np.asarray(start_point, dtype=float)
    end = np.asarray(end_point, dtype=float)
    if start.shape != (2,) or end.shape != (2,) or not np.all(np.isfinite(start)) or not np.all(np.isfinite(end)):
        raise ValueError("笛卡尔起点和终点必须是有限的二维坐标")
    if isinstance(sample_count, bool) or int(sample_count) != sample_count or sample_count < 3:
        raise ValueError("采样点数必须是至少为 3 的整数")
    if elbow_sign not in (None, -1, 1):
        raise ValueError("elbow_sign 只能是 None、1 或 -1")
    points = np.linspace(start, end, int(sample_count))
    limits = None
    if joint_limits is not None:
        limits = tuple(np.asarray(part, dtype=float) for part in joint_limits)
        if any(part.shape != (2,) or not np.all(np.isfinite(part)) for part in limits) or np.any(limits[0] >= limits[1]):
            raise ValueError("关节角上下限格式无效")
    joint_path = []
    previous = None
    for index, point in enumerate(points):
        result = solve_two_link_ik(point, link_lengths)
        if not result.reachable or not result.solutions:
            raise ValueError(f"笛卡尔直线路径第 {index + 1} 个采样点不可达：{result.message}")
        candidates = list(result.solutions)
        if elbow_sign is not None:
            candidates = [solution for solution in candidates if np.sign(solution[1]) == elbow_sign or abs(solution[1]) < 1e-10]
            if not candidates:
                raise ValueError(f"笛卡尔路径第 {index + 1} 个采样点无法满足指定肘部构型")
        if previous is None:
            selected = candidates[0]
        else:
            selected = min(candidates, key=lambda candidate: np.linalg.norm(np.asarray(candidate) - previous))
        selected = np.asarray(selected, dtype=float)
        if limits is not None and (np.any(selected < limits[0]) or np.any(selected > limits[1])):
            raise ValueError(f"笛卡尔路径第 {index + 1} 个采样点违反关节角限位")
        if previous is not None:
            selected += 2.0 * np.pi * np.round((previous - selected) / (2.0 * np.pi))
        joint_path.append(selected)
        previous = selected
    return points, np.asarray(joint_path)


def plan_cartesian_path(
    model: PlanarArmDynamics,
    cartesian_points,
    initial_angles,
    max_velocity,
    max_acceleration,
    max_torque,
    *,
    initial_duration=1.0,
    sample_count=201,
    max_duration=120.0,
    max_iterations=60,
    joint_limits=None,
    pose_constraints=None,
):
    """沿给定笛卡尔路径连续求逆运动学，并自动搜索满足约束的运动时间。

    几何路径保持固定；使用热启动数值 IK 连续求解各点，再对关节路径使用
    五次时间标定。路径点必须可达，且所有解必须满足关节限位。
    """
    from joint_limits import JointLimits
    from numerical_ik import solve_n_link_ik

    arm = model.arm
    points = np.asarray(cartesian_points, dtype=float)
    seed = arm._validated_angles(initial_angles)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.all(np.isfinite(points)):
        raise ValueError("笛卡尔路径必须是至少包含两个有限二维点的数组")
    if joint_limits is not None:
        if isinstance(joint_limits, JointLimits):
            limits = joint_limits
        else:
            limits = JointLimits(*joint_limits)
    else:
        limits = None
    pose_constraints = {} if pose_constraints is None else dict(pose_constraints)
    normalized_constraints = {}
    for raw_index, raw_angles in pose_constraints.items():
        if isinstance(raw_index, bool) or int(raw_index) != raw_index or not 0 <= int(raw_index) < len(points):
            raise ValueError("关键姿态约束的路径采样索引无效")
        constrained_angles = arm._validated_angles(raw_angles)
        constrained_point = arm.forward_kinematics(constrained_angles)[-1]
        error = float(np.linalg.norm(constrained_point - points[int(raw_index)]))
        if error > 1e-4:
            raise ValueError(f"第 {int(raw_index) + 1} 个采样点的关键姿态与路径位置冲突（末端误差 {error:.3g} m）")
        if limits is not None and (np.any(constrained_angles < limits.lower - 1e-9) or np.any(constrained_angles > limits.upper + 1e-9)):
            raise ValueError(f"第 {int(raw_index) + 1} 个采样点的关键姿态违反关节角限位")
        normalized_constraints[int(raw_index)] = constrained_angles.copy()
    joint_path = []
    previous = seed.copy()
    for index, point in enumerate(points):
        if index in normalized_constraints:
            angles = normalized_constraints[index].copy()
            error = float(np.linalg.norm(arm.forward_kinematics(angles)[-1] - point))
            converged = error <= 1e-5
        else:
            angles, error, converged = solve_n_link_ik(arm, point, previous, limits=limits)
        if not converged or error > 1e-5:
            raise ValueError(f"路径第 {index + 1} 个点不可达或逆运动学未收敛（误差 {error:.3g}）")
        if limits is not None and (np.any(angles < limits.lower - 1e-9) or np.any(angles > limits.upper + 1e-9)):
            raise ValueError(f"路径第 {index + 1} 个点违反关节角限位")
        joint_path.append(angles.copy())
        previous = angles
    joint_path = np.unwrap(np.asarray(joint_path), axis=0)
    # 以笛卡尔路径弧长作为统一进度参数，不能用关节角弧长替代几何路径进度。
    # 否则冗余机械臂的关节变化会扭曲路径参数，导致末端在关键节点附近切角。
    cartesian_distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if np.any(cartesian_distances <= 1e-12):
        raise ValueError("笛卡尔路径含有连续重复点，无法建立严格递增的路径参数。")
    progress = np.concatenate(([0.0], np.cumsum(cartesian_distances)))
    progress /= progress[-1]

    v_limit = _validate_limits(max_velocity, arm.joint_count, "角速度上限")
    a_limit = _validate_limits(max_acceleration, arm.joint_count, "角加速度上限")
    t_limit = _validate_limits(max_torque, arm.joint_count, "力矩上限")
    duration = float(initial_duration)
    max_duration = float(max_duration)
    if not np.isfinite(duration) or duration <= 0 or not np.isfinite(max_duration) or max_duration < duration:
        raise ValueError("初始运动时间和最大搜索时间必须有效，且最大时间不小于初始时间")
    # 使用保形分段三次插值，避免普通三次样条在路径节点附近过冲，
    # 进而导致末端越过目标后又折返。时间参数仍采用五次标定。
    path_spline = PchipInterpolator(progress, joint_path, axis=0)
    last = None
    # 每个几何路径样本都必须对应一个时间样本；否则时间离散化可能跨过节点 2/3。
    # 通过单调五次时间标定的数值逆映射，求出每个路径进度对应的准确时刻。
    inverse_grid = np.linspace(0.0, 1.0, 20001)
    inverse_blend = 10.0 * inverse_grid**3 - 15.0 * inverse_grid**4 + 6.0 * inverse_grid**5
    path_sample_times = np.interp(progress, inverse_blend, inverse_grid)
    for iteration in range(1, int(max_iterations) + 1):
        regular_times = np.linspace(0.0, duration, int(sample_count))
        times = np.unique(np.concatenate((regular_times, path_sample_times * duration)))
        s = times / duration
        blend = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
        blend_d = (30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4) / duration
        blend_dd = (60.0 * s - 180.0 * s**2 + 120.0 * s**3) / duration**2
        angles = path_spline(blend)
        path_velocity = path_spline(blend, 1)
        path_acceleration = path_spline(blend, 2)
        velocities = path_velocity * blend_d[:, None]
        accelerations = path_acceleration * (blend_d[:, None] ** 2) + path_velocity * blend_dd[:, None]
        torques = np.asarray([model.inverse_dynamics(q, qd, qdd) for q, qd, qdd in zip(angles, velocities, accelerations)])
        last = (times, angles, velocities, accelerations, torques)
        passed = (np.all(np.abs(velocities) <= v_limit[None, :] * (1 + 1e-9))
                  and np.all(np.abs(accelerations) <= a_limit[None, :] * (1 + 1e-9))
                  and np.all(np.abs(torques) <= t_limit[None, :] * (1 + 1e-9)))
        if passed:
            return TrajectoryPlanResult(True, "已找到满足所设速度、加速度和力矩上限的笛卡尔路径轨迹。", *last, duration, iteration, progress.copy(), joint_path.copy())
        if duration >= max_duration:
            break
        duration = min(duration * 1.25, max_duration)
    times, angles, velocities, accelerations, torques = last
    diagnostics = []
    for name, values, limits, unit in (
        ("角速度", np.abs(velocities), v_limit, "rad/s"),
        ("角加速度", np.abs(accelerations), a_limit, "rad/s²"),
        ("驱动力矩", np.abs(torques), t_limit, "N·m"),
    ):
        peaks = np.max(values, axis=0)
        exceeded = np.flatnonzero(peaks > limits * (1 + 1e-9))
        if exceeded.size:
            details = "、".join(
                f"关节 {index + 1} {peaks[index]:.3g}/{limits[index]:.3g} {unit}"
                for index in exceeded
            )
            diagnostics.append(f"{name}超限（实际峰值/上限）：{details}")
    message = (
        "在最长搜索时间内仍无法满足约束。" + "；".join(diagnostics)
        + "。可尝试放宽对应限制，或调整路径/机械臂参数。"
    )
    return TrajectoryPlanResult(False, message, times, angles, velocities, accelerations, torques, float(times[-1]), int(iteration), progress.copy(), joint_path.copy())
