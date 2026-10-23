"""统一保存各路径方案的规划结果，避免方案数据分散在影子变量中。"""

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import CubicHermiteSpline


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
        spline = CubicHermiteSpline(
            np.asarray(plan.times, dtype=float),
            np.asarray(plan.joint_angles, dtype=float),
            np.asarray(plan.joint_velocities, dtype=float),
            axis=0,
        )
        return cls(key, label, "ready", plan=plan, spline=spline, message=str(plan.message), diagnostics={} if diagnostics is None else diagnostics)

    @classmethod
    def failed(cls, key: str, label: str, error):
        return cls(key, label, "failed", message=str(error), diagnostics={})
