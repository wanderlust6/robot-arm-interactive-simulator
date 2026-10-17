"""Planar serial robot arm kinematics."""

from collections.abc import Sequence

import numpy as np


class PlanarArm:
    """A planar serial arm with revolute joints and fixed link lengths.

    Joint angles are relative angles in radians. The first link angle is
    measured counterclockwise from the positive x-axis.
    """

    def __init__(self, link_lengths: Sequence[float]) -> None:
        try:
            lengths = np.asarray(link_lengths, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("link_lengths must be a sequence of finite positive numbers") from exc

        if lengths.ndim != 1 or lengths.size == 0:
            raise ValueError("link_lengths must contain at least one link")
        if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
            raise ValueError("all link lengths must be finite and greater than zero")

        self._link_lengths = lengths.copy()

    @property
    def link_lengths(self) -> np.ndarray:
        """Return a copy of the link lengths."""
        return self._link_lengths.copy()

    @property
    def joint_count(self) -> int:
        return int(self._link_lengths.size)

    def _validated_angles(self, joint_angles: Sequence[float]) -> np.ndarray:
        try:
            angles = np.asarray(joint_angles, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("joint_angles must be a sequence of finite numbers") from exc

        if angles.ndim != 1 or angles.size != self.joint_count:
            raise ValueError("the number of joint angles must match the number of links")
        if not np.all(np.isfinite(angles)):
            raise ValueError("all joint angles must be finite")
        return angles

    def forward_kinematics(self, joint_angles: Sequence[float]) -> np.ndarray:
        """Return base and joint positions as an array with shape (n + 1, 2)."""
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
        """Return the end-effector position Jacobian with shape (2, n)."""
        positions = self.forward_kinematics(joint_angles)
        end_position = positions[-1]
        joint_to_end = end_position - positions[:-1]

        return np.vstack((-joint_to_end[:, 1], joint_to_end[:, 0]))
