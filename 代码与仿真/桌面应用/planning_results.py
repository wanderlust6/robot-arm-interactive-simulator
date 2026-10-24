"""统一保存各路径方案的规划结果，避免方案数据分散在影子变量中。"""

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import BPoly, CubicHermiteSpline


class QuinticHermiteSpline:
    """按 q、q'、q'' 匹配的分段五次 Hermite 插值，二阶导连续（C²）。

    ``CubicHermiteSpline`` 只匹配 q 与 q'，二阶导在每个节点处不连续，因此由它
    求出的关节角加速度会逐节点跳变。这里额外匹配规划器给出的角加速度。

    构造时直接使用规划器的时间网格：试过把节点数据重采样到均匀网格再插值，
    那会破坏 q̈ 与 q、q̇ 的自洽性，实测加速度抖动反而更差（中位比例 0.05% → 13%）。
    """

    def __init__(self, times, values, first_derivatives, second_derivatives):
        times = np.asarray(times, dtype=float)
        values = np.asarray(values, dtype=float)
        first = np.asarray(first_derivatives, dtype=float)
        second = np.asarray(second_derivatives, dtype=float)
        if values.ndim != 2 or values.shape[0] != times.size:
            raise ValueError("关节角数组形状必须为 (采样点数量, 关节数量)")
        if first.shape != values.shape or second.shape != values.shape:
            raise ValueError("角速度/角加速度形状必须与关节角一致")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("采样时刻必须严格递增")
        self.x = times
        self._splines = [
            BPoly.from_derivatives(
                times,
                np.column_stack((values[:, joint], first[:, joint], second[:, joint])),
            )
            for joint in range(values.shape[1])
        ]

    def __call__(self, x, nu=0):
        array = np.asarray(x, dtype=float)
        scalar = array.ndim == 0
        columns = [np.atleast_1d(spline(array, nu)) for spline in self._splines]
        values = np.column_stack(columns)
        return values[0] if scalar else values


@dataclass(frozen=True)
class PlanningSchemeResult:
    """单个路径方案的完整规划结果与播放插值器。"""

    key: str
    label: str
    status: str
    plan: object = None
    spline: object = None
    message: str = ""
    diagnostics: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == "ready" and self.plan is not None and bool(self.plan.success)

    @classmethod
    def from_plan(cls, key: str, label: str, plan, diagnostics=None):
        if not bool(plan.success):
            return cls(key, label, "failed", plan=plan, message=str(plan.message), diagnostics={} if diagnostics is None else diagnostics)
        times = np.asarray(plan.times, dtype=float)
        angles = np.asarray(plan.joint_angles, dtype=float)
        velocities = np.asarray(plan.joint_velocities, dtype=float)
        accelerations = getattr(plan, "joint_accelerations", None)
        spline = None
        if accelerations is not None and np.shape(accelerations) == angles.shape:
            try:
                spline = QuinticHermiteSpline(times, angles, velocities, accelerations)
            except (ValueError, IndexError):
                spline = None
        if spline is None:
            # 兼容没有角加速度字段的规划结果，退回只匹配位置与速度的 C¹ 插值。
            spline = CubicHermiteSpline(times, angles, velocities, axis=0)
        return cls(key, label, "ready", plan=plan, spline=spline, message=str(plan.message), diagnostics={} if diagnostics is None else diagnostics)

    @classmethod
    def failed(cls, key: str, label: str, error):
        return cls(key, label, "failed", message=str(error), diagnostics={})
