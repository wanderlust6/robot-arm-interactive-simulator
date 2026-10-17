"""二维平面串联机械臂的运动学计算。"""

from collections.abc import Sequence

import numpy as np


class PlanarArm:
    """由转动关节和固定长度连杆组成的平面机械臂。

    关节角采用弧度制，并且是相对角。第一根连杆相对于 x 轴正方向
    逆时针转动；后续关节角相对于前一根连杆计算。
    """

    def __init__(self, link_lengths: Sequence[float]) -> None:
        try:
            lengths = np.asarray(link_lengths, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("连杆长度必须是有限的正数序列") from exc

        if lengths.ndim != 1 or lengths.size == 0:
            raise ValueError("至少需要一根连杆")
        if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
            raise ValueError("所有连杆长度都必须是有限的正数")

        self._link_lengths = lengths.copy()

    @property
    def link_lengths(self) -> np.ndarray:
        """返回连杆长度的副本，避免外部修改内部数据。"""
        return self._link_lengths.copy()

    @property
    def joint_count(self) -> int:
        """返回机械臂的关节数量。"""
        return int(self._link_lengths.size)

    def _validated_angles(self, joint_angles: Sequence[float]) -> np.ndarray:
        try:
            angles = np.asarray(joint_angles, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("关节角必须是有限数值序列") from exc

        if angles.ndim != 1 or angles.size != self.joint_count:
            raise ValueError("关节角数量必须与连杆数量一致")
        if not np.all(np.isfinite(angles)):
            raise ValueError("所有关节角都必须是有限数值")
        return angles

    def forward_kinematics(self, joint_angles: Sequence[float]) -> np.ndarray:
        """计算基座、各关节和末端位置，返回形状为 (n + 1, 2) 的数组。"""
        angles = self._validated_angles(joint_angles)
        absolute_angles = np.cumsum(angles)

        link_vectors = np.column_stack(
            (
                self._link_lengths * np.cos(absolute_angles),
                self._link_lengths * np.sin(absolute_angles),
            )
        )
        positions = np.vstack((np.zeros((1, 2)), np.cumsum(link_vectors, axis=0)))
        return positions

    def jacobian(self, joint_angles: Sequence[float]) -> np.ndarray:
        """计算末端二维位置的雅可比矩阵，返回形状为 (2, n) 的数组。"""
        positions = self.forward_kinematics(joint_angles)
        end_position = positions[-1]
        joint_to_end = end_position - positions[:-1]

        return np.vstack((-joint_to_end[:, 1], joint_to_end[:, 0]))
