import numpy as np
from planar_arm import PlanarArm

arm = PlanarArm([1.0, 1.0])

poses = {
    "弯曲姿态": [np.pi / 4, np.pi / 2],
    "完全伸直": [0.0, 0.0],
    "完全折叠": [0.0, np.pi],
}

for name, angles in poses.items():
    J = arm.jacobian(angles)

    print(f"\n{name}")
    print("雅可比矩阵:")
    print(J)
    print("行列式:", np.linalg.det(J))
    print("矩阵秩:", np.linalg.matrix_rank(J))
    print("奇异值:", np.linalg.svd(J, compute_uv=False))