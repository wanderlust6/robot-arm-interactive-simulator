"""二维平面串联机械臂的重力与逆动力学模型。"""

from collections.abc import Sequence

import numpy as np

from planar_arm import PlanarArm


class PlanarArmDynamics:
    """基于刚性连杆和末端点负载的平面机械臂动力学模型。

    坐标约定：x 向右、y 向上，重力沿 -y；关节角为相对角，逆时针为正。
    连杆质心位于连杆轴线上，由质心比例指定。未提供转动惯量时，
    假设连杆为均匀细杆，使用 I_com = m * L**2 / 12。

    当前不包含摩擦、柔性、齿轮传动、接触力或电机能力限制。
    """

    def __init__(
        self,
        link_lengths: Sequence[float],
        link_masses: Sequence[float] | None = None,
        center_of_mass_fractions: Sequence[float] | None = None,
        payload_mass: float = 0.0,
        gravity: float = 9.80665,
        link_inertias: Sequence[float] | None = None,
    ) -> None:
        self.arm = PlanarArm(link_lengths)
        count = self.arm.joint_count
        lengths = self.arm.link_lengths
        self._link_masses = self._validate_vector(
            [1.0] * count if link_masses is None else link_masses,
            count, "连杆质量", allow_zero=True,
        )
        self._com_fractions = self._validate_vector(
            [0.5] * count if center_of_mass_fractions is None else center_of_mass_fractions,
            count, "质心比例", allow_zero=True,
        )
        if np.any(self._com_fractions > 1.0):
            raise ValueError("质心比例必须位于 [0, 1]")

        default_inertias = self._link_masses * lengths**2 / 12.0
        self._link_inertias = self._validate_vector(
            default_inertias if link_inertias is None else link_inertias,
            count, "连杆转动惯量", allow_zero=True,
        )
        self.payload_mass = self._validate_nonnegative(payload_mass, "末端负载质量")
        try:
            self.gravity = float(gravity)
        except (TypeError, ValueError) as exc:
            raise ValueError("重力加速度必须是有限的非负数") from exc
        if not np.isfinite(self.gravity) or self.gravity < 0:
            raise ValueError("重力加速度必须是有限的非负数")

    @staticmethod
    def _validate_nonnegative(value, name: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}必须是有限的非负数") from exc
        if not np.isfinite(result) or result < 0:
            raise ValueError(f"{name}必须是有限的非负数")
        return result

    @classmethod
    def _validate_vector(cls, values, count, name, allow_zero=False):
        try:
            result = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}必须是有限数值序列") from exc
        if result.ndim != 1 or result.size != count or not np.all(np.isfinite(result)):
            raise ValueError(f"{name}数量必须与连杆数量一致，且全部为有限数值")
        if np.any(result < 0) or (not allow_zero and np.any(result <= 0)):
            qualifier = "非负数" if allow_zero else "正数"
            raise ValueError(f"{name}必须为{qualifier}")
        return result.copy()

    @property
    def link_masses(self) -> np.ndarray:
        return self._link_masses.copy()

    @property
    def center_of_mass_fractions(self) -> np.ndarray:
        return self._com_fractions.copy()

    @property
    def link_inertias(self) -> np.ndarray:
        """返回各连杆绕自身质心的平面转动惯量副本，单位 kg·m²。"""
        return self._link_inertias.copy()

    def gravity_torques(self, joint_angles: Sequence[float]) -> np.ndarray:
        """返回各关节抵抗重力所需的力矩，单位 N·m。"""
        angles = self.arm._validated_angles(joint_angles)
        absolute_angles = np.cumsum(angles)
        positions = self.arm.forward_kinematics(angles)
        torques = np.zeros(self.arm.joint_count, dtype=float)

        for link_index, (length, mass, fraction, angle) in enumerate(
            zip(self.arm.link_lengths, self._link_masses, self._com_fractions, absolute_angles)
        ):
            if mass == 0 or self.gravity == 0:
                continue
            com = positions[link_index] + fraction * length * np.array(
                [np.cos(angle), np.sin(angle)]
            )
            for joint_index in range(link_index + 1):
                lever = com - positions[joint_index]
                torques[joint_index] += mass * self.gravity * lever[0]

        if self.payload_mass > 0 and self.gravity > 0:
            end = positions[-1]
            for joint_index in range(self.arm.joint_count):
                lever = end - positions[joint_index]
                torques[joint_index] += self.payload_mass * self.gravity * lever[0]

        return torques

    @staticmethod
    def _validated_state(values: Sequence[float], count: int, name: str) -> np.ndarray:
        try:
            result = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}必须是有限数值序列") from exc
        if result.ndim != 1 or result.size != count:
            raise ValueError(f"{name}数量必须与关节数量一致")
        if not np.all(np.isfinite(result)):
            raise ValueError(f"{name}必须全部为有限数值")
        return result

    @staticmethod
    def _cross2(first: np.ndarray, second: np.ndarray) -> float:
        """二维向量叉积的 z 分量。"""
        return float(first[0] * second[1] - first[1] * second[0])

    def inverse_dynamics(
        self,
        joint_angles: Sequence[float],
        joint_velocities: Sequence[float],
        joint_accelerations: Sequence[float],
    ) -> np.ndarray:
        """用平面递归牛顿—欧拉算法计算驱动力矩，单位 N·m。

        输入关节角、角速度和角加速度均采用相对关节坐标，单位分别为
        rad、rad/s、rad/s²。输出正值表示需要施加逆时针驱动力矩。
        """
        count = self.arm.joint_count
        q = self._validated_state(joint_angles, count, "关节角")
        qd = self._validated_state(joint_velocities, count, "关节角速度")
        qdd = self._validated_state(joint_accelerations, count, "关节角加速度")

        lengths = self.arm.link_lengths
        absolute_angles = np.cumsum(q)
        absolute_omegas = np.cumsum(qd)
        absolute_alphas = np.cumsum(qdd)
        positions = self.arm.forward_kinematics(q)
        gravity_vector = np.array([0.0, -self.gravity])

        link_forces = []
        endpoint_accelerations = []
        com_levers = []
        endpoint_levers = []
        origin_acceleration = np.zeros(2, dtype=float)

        for i in range(count):
            angle = absolute_angles[i]
            direction = np.array([np.cos(angle), np.sin(angle)])
            com_lever = self._com_fractions[i] * lengths[i] * direction
            end_lever = lengths[i] * direction
            omega = absolute_omegas[i]
            alpha = absolute_alphas[i]

            com_acceleration = (
                origin_acceleration
                + alpha * np.array([-com_lever[1], com_lever[0]])
                - omega**2 * com_lever
            )
            link_forces.append(self._link_masses[i] * (com_acceleration - gravity_vector))
            endpoint_acceleration = (
                origin_acceleration
                + alpha * np.array([-end_lever[1], end_lever[0]])
                - omega**2 * end_lever
            )
            endpoint_accelerations.append(endpoint_acceleration)
            com_levers.append(com_lever)
            endpoint_levers.append(end_lever)
            origin_acceleration = endpoint_acceleration

        distal_force = self.payload_mass * (endpoint_accelerations[-1] - gravity_vector)
        distal_moment = 0.0
        torques = np.zeros(count, dtype=float)

        for i in range(count - 1, -1, -1):
            body_force = link_forces[i]
            total_force = body_force + distal_force
            moment_at_origin = (
                self._link_inertias[i] * absolute_alphas[i]
                + self._cross2(com_levers[i], body_force)
                + self._cross2(endpoint_levers[i], distal_force)
                + distal_moment
            )
            torques[i] = moment_at_origin
            distal_moment = moment_at_origin
            distal_force = total_force

        return torques

