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

    def node_jacobians(self, joint_angles: Sequence[float]) -> np.ndarray:
        """返回各节点的位置雅可比矩阵，形状为 (n + 1, 2, n)。

        第 i 个节点（0 为基座）对第 m 个关节的偏导为
        ``sum_{k=m}^{i-1} L_k * [-sin(theta_k), cos(theta_k)]``，
        其中 ``theta_k`` 为第 k 根连杆的绝对角。
        """
        angles = self._validated_angles(joint_angles)
        absolute_angles = np.cumsum(angles)
        count = self.joint_count
        jacobians = np.zeros((count + 1, 2, count), dtype=float)
        for node in range(1, count + 1):
            for joint_index in range(node):
                span = slice(joint_index, node)
                jacobians[node, 0, joint_index] = -np.sum(
                    self._link_lengths[span] * np.sin(absolute_angles[span])
                )
                jacobians[node, 1, joint_index] = np.sum(
                    self._link_lengths[span] * np.cos(absolute_angles[span])
                )
        return jacobians

    def node_velocities(self, joint_angles: Sequence[float], joint_velocities: Sequence[float]) -> np.ndarray:
        """返回各节点速度，形状为 (n + 1, 2)，单位 m/s。

        用雅可比与关节角速度解析求得，避免对位置序列做数值微分——数值微分会把
        采样上的微小不一致放大成可见抖动。
        """
        jacobians = self.node_jacobians(joint_angles)
        velocities = self._validated_angles(joint_velocities)
        return np.einsum("nij,j->ni", jacobians, velocities)

    def node_accelerations(
        self,
        joint_angles: Sequence[float],
        joint_velocities: Sequence[float],
        joint_accelerations: Sequence[float],
    ) -> np.ndarray:
        """返回各节点加速度，形状为 (n + 1, 2)，单位 m/s²。

        ``a_i = J_i q̈ + (dJ_i/dt) q̇``，其中
        ``(dJ_i/dt) q̇ = sum_{k<i} L_k * [-cos(theta_k), -sin(theta_k)] * omega_k²``，
        ``omega_k`` 为到第 k 个关节为止的累计角速度（θ̇_k 与 q̇ 的求和各贡献一次）。
        """
        angles = self._validated_angles(joint_angles)
        velocities = self._validated_angles(joint_velocities)
        accelerations = self._validated_angles(joint_accelerations)
        absolute_angles = np.cumsum(angles)
        absolute_velocities = np.cumsum(velocities)
        jacobians = self.node_jacobians(angles)
        result = np.einsum("nij,j->ni", jacobians, accelerations)
        for node in range(1, self.joint_count + 1):
            for k in range(node):
                direction = np.array([-np.cos(absolute_angles[k]), -np.sin(absolute_angles[k])])
                result[node] += (
                    self._link_lengths[k] * absolute_velocities[k] ** 2 * direction
                )
        return result

    def singularity_metrics(self, joint_angles: Sequence[float]) -> dict[str, object]:
        """返回末端位置雅可比矩阵的秩、奇异值和条件数。

        rank 采用 NumPy 的数值秩判定。对单关节机械臂，满秩上限为 1，
        因此不会仅因无法覆盖二维平面的全部方向就被误判为奇异位形。
        """
        jacobian = self.jacobian(joint_angles)
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        rank = int(np.linalg.matrix_rank(jacobian))
        max_rank = min(jacobian.shape)
        min_singular_value = float(singular_values[-1]) if singular_values.size else 0.0
        max_singular_value = float(singular_values[0]) if singular_values.size else 0.0

        if rank < max_rank or min_singular_value == 0.0:
            condition_number = float("inf")
        else:
            condition_number = max_singular_value / min_singular_value

        return {
            "rank": rank,
            "max_rank": max_rank,
            "singular_values": singular_values.copy(),
            "min_singular_value": min_singular_value,
            "condition_number": float(condition_number),
            "is_singular": rank < max_rank,
        }
