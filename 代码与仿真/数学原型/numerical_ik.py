"""适用于任意段数平面串联机械臂的数值逆运动学。"""

import numpy as np
from scipy.optimize import least_squares, minimize
from planar_arm import PlanarArm
from joint_limits import JointLimits


def solve_n_link_ik(arm: PlanarArm, target, initial_angles, limits: JointLimits | None = None,
                    max_iterations: int = 250, tolerance: float = 1e-6,
                    damping: float = 1e-3):
    """以阻尼最小二乘法求解末端位置；返回 (angles, error, converged)。"""
    target = np.asarray(target, dtype=float)
    angles = np.asarray(initial_angles, dtype=float).copy()
    if target.shape != (2,) or not np.all(np.isfinite(target)):
        raise ValueError("目标位置必须是有限的二维坐标")
    if angles.shape != (arm.joint_count,) or not np.all(np.isfinite(angles)):
        raise ValueError("初始关节角数量必须与连杆数量一致且为有限值")
    if max_iterations <= 0 or tolerance <= 0 or damping <= 0:
        raise ValueError("迭代次数、容差和阻尼系数必须为正数")
    if limits is not None and limits.joint_count != arm.joint_count:
        raise ValueError("关节限位数量必须与连杆数量一致")

    best = angles.copy()
    best_error = float(np.linalg.norm(arm.forward_kinematics(best)[-1] - target))
    for _ in range(max_iterations):
        positions = arm.forward_kinematics(angles)
        delta = target - positions[-1]
        error = float(np.linalg.norm(delta))
        if error < best_error:
            best, best_error = angles.copy(), error
        if error <= tolerance:
            return angles, error, True

        jacobian = arm.jacobian(angles)
        step = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + (damping ** 2) * np.eye(2), delta
        )
        step_norm = float(np.linalg.norm(step))
        if step_norm > 0.25:
            step *= 0.25 / step_norm
        candidate = angles + step
        if limits is not None:
            candidate = np.clip(candidate, limits.lower, limits.upper)
        candidate_error = float(np.linalg.norm(arm.forward_kinematics(candidate)[-1] - target))
        if candidate_error > error:
            damping = min(damping * 4.0, 1.0)
        else:
            angles = candidate
            damping = max(damping * 0.8, 1e-6)

    return best, best_error, best_error <= tolerance


def solve_constrained_drag(arm: PlanarArm, target_index: int, target,
                           initial_angles, fixed_positions, limits: JointLimits | None = None,
                           tolerance: float = 1e-5, max_iterations: int = 80,
                           target_position=None):
    """求解受固定节点约束的拖拽；优先单次热启动，失败时再尝试多初值。"""
    target = np.asarray(target, dtype=float)
    initial = np.asarray(initial_angles, dtype=float)
    if target.shape != (2,) or not np.all(np.isfinite(target)):
        raise ValueError("拖拽目标必须是有限的二维坐标")
    if initial.shape != (arm.joint_count,) or not np.all(np.isfinite(initial)):
        raise ValueError("初始关节角与连杆数量不匹配")
    if not 1 <= target_index <= arm.joint_count:
        raise ValueError("拖拽节点索引超出范围")
    if limits is not None and limits.joint_count != arm.joint_count:
        raise ValueError("关节限位数量必须与连杆数量一致")

    pins = {int(i): np.asarray(p, dtype=float) for i, p in fixed_positions.items()}
    if any(i < 1 or i > arm.joint_count or p.shape != (2,) or not np.all(np.isfinite(p))
           for i, p in pins.items()):
        raise ValueError("固定节点索引或坐标无效")
    pins.pop(target_index, None)

    track_mouse_position = target_position is not None
    if track_mouse_position:
        target_position = np.asarray(target_position, dtype=float)
        if target_position.shape != (2,) or not np.all(np.isfinite(target_position)):
            raise ValueError("拖动目标必须是有限的二维坐标")
    else:
        target_position = target

    lower = np.full(arm.joint_count, -np.inf) if limits is None else limits.lower
    upper = np.full(arm.joint_count, np.inf) if limits is None else limits.upper

    def residual_vector(angles):
        positions = arm.forward_kinematics(angles)
        target_weight = 1e6
        pin_weight = 1e8
        residuals = [(positions[target_index] - target_position) * target_weight]
        for index, point in sorted(pins.items()):
            residuals.append((positions[index] - point) * pin_weight)
        # 轻微偏好连续姿态，减少鼠标移动时关节突然跳变。
        residuals.append((angles - initial) * 1e-3)
        return np.concatenate(residuals)

    def assess(result):
        angles = np.asarray(result.x, dtype=float)
        if not np.all(np.isfinite(angles)):
            return None
        positions = arm.forward_kinematics(angles)
        pin_errors = [
            float(np.max(np.abs(positions[index] - point)))
            for index, point in sorted(pins.items())
        ]
        error = max(pin_errors, default=0.0)
        if not track_mouse_position:
            error = max(error, float(np.max(np.abs(positions[target_index] - target))))
        if error > tolerance or (not track_mouse_position and not result.success):
            return None
        target_error = float(np.linalg.norm(positions[target_index] - target_position))
        angle_change = float(np.dot(angles - initial, angles - initial))
        return target_error + 1e-5 * angle_change, angles, error

    # 鼠标事件相邻帧的初始姿态就是上一帧解；正常情况下只需一次求解。
    starts = [np.clip(initial.copy(), lower, upper)]
    solutions = []

    def solve_from(start, budget):
        result = least_squares(
            residual_vector,
            start,
            bounds=(lower, upper),
            max_nfev=budget * max(1, arm.joint_count),
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
        )
        candidate = assess(result)
        if candidate is not None:
            solutions.append(candidate)

    solve_from(starts[0], max_iterations)

    # 只有热启动无法满足固定节点时，才启用多初值恢复，避免每个鼠标事件都跑 2n+1 次优化。
    if not solutions and track_mouse_position and pins:
        for joint_index in range(arm.joint_count):
            for offset in (-0.12, 0.12):
                candidate = initial.copy()
                candidate[joint_index] = np.clip(
                    candidate[joint_index] + offset,
                    lower[joint_index],
                    upper[joint_index],
                )
                solve_from(candidate, max(20, max_iterations // 2))

    if not solutions:
        return initial.copy(), float("inf"), False

    _, angles, error = min(solutions, key=lambda item: item[0])
    return angles, error, True
