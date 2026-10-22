# 二维机械臂数学原型

本目录实现二维平面串联机械臂的基础运动学，不依赖桌面图形界面。核心计算独立于绘图逻辑，并通过自动化测试验证。

## 当前功能

- 支持配置任意数量的平面转动关节和固定长度连杆。
- 正运动学：计算基座、各关节和末端位置。
- 末端位置雅可比矩阵：描述关节角微小变化与末端位置微小变化之间的局部线性关系。
- 两连杆平面机械臂解析逆运动学：返回不同肘部构型的解。
- 目标点可达性判断、工作空间边界容差处理和奇异边界标记。
- 独立关节角限位配置、检查与逆运动学解筛选。
- 任意段数阻尼最小二乘数值逆运动学及末端/中间节点拖动约束求解。
- 输入校验与自动化测试。

逆运动学的解析求解目前只支持两连杆机械臂；关节限位工具可以检查任意数量关节的角度组合。

## 坐标系与单位约定

- 基座位于二维坐标原点。
- x 轴向右，y 轴向上。
- 角度单位为弧度，正方向为逆时针。
- 每个关节角是相对角；第一关节角相对于 x 轴正方向，后续关节角相对于前一根连杆。
- 连杆长度必须是有限的正数。
- 同一次计算中的目标坐标和连杆长度必须使用相同的长度单位。
- 关节角限位的上下界都使用弧度，且下限不能大于上限。

## 模块说明

### `planar_arm.py`

- `PlanarArm(link_lengths)`：创建平面串联机械臂。
- `forward_kinematics(joint_angles)`：返回形状为 `(n + 1, 2)` 的 NumPy 数组，第一行是基座，最后一行是末端位置。
- `jacobian(joint_angles)`：返回末端二维位置雅可比矩阵，形状为 `(2, n)`。

### `inverse_kinematics.py`

- `solve_two_link_ik(target, link_lengths, tolerance=1e-9)`：求解两连杆机械臂的解析逆运动学。
- 返回 `InverseKinematicsResult`，包含 `solutions`、`reachable`、`singular` 和 `message`。
- 每组解为 `(theta1, theta2)`，单位为弧度；普通可达目标通常返回两组肘部构型解。
- 不可达目标返回 `reachable=False` 且 `solutions=()`。
- 两根连杆等长且目标恰在基座时存在无穷多组折叠解；该情况会通过结果消息明确说明。
- `tolerance` 用于吸收工作空间边界附近的浮点误差，不代表关节角限位或控制精度。

### continuous_ik.py

- select_continuous_ik_solution(candidates, previous)：从两连杆解析逆运动学候选解中，选择按每个关节的 2π 周期展开后与上一姿态欧氏距离最小的解。
- 空候选集返回 None；没有上一姿态时返回第一组候选解的副本。
- 候选集必须能转换为形状为 (N, 2) 的有限浮点数组；上一姿态必须是两个有限关节角。
- 该函数只负责局部连续性偏好，不负责关节限位、全局路径优化或速度/加速度约束。详见 [连续解析逆运动学解选择](../../docs/CONTINUOUS_IK.md)。

### `elbow_drag.py`

- `solve_elbow_drag_candidates(mouse_position, link_lengths, line_direction, previous_target, tolerance=1e-9)`：求解拖动第二关节时的可行姿态。
- `line_direction` 是拖动开始时由第一关节指向末端的固定方向；本次拖动期间末端只能沿该直线移动。
- 候选姿态保持两根连杆长度不变，并按末端位置与上一次有效末端位置的距离排序，减少不必要的跳变。
- 若鼠标方向对应的肘关节位置不可行，会将其限制到最近的可行方向；最终仍需通过独立关节限位检查。
- `ElbowDragCandidate` 包含关节角、末端位置和第二关节位置，可用于界面更新和几何约束测试。

### run_singularity_approach_sweep.py

- 对两连杆机械臂的三种伸直奇异位形接近路径和一种折叠奇异位形穿越路径，比较基线 DLS 与自适应 DLS。
- 每条路径采用 61、121、241 个采样点，并记录末端误差、最小奇异值、关节步长、离散二阶差分、参考关节偏差、分支不匹配和 P95 求解耗时。
- 这是离线研究脚本，不会修改桌面应用或生产求解器。实验方法、结果和解释边界见 [奇异位形扫描报告](../../docs/SINGULARITY_APPROACH_SWEEP.md)。

### `numerical_ik.py`

- `solve_n_link_ik(...)`：使用阻尼最小二乘迭代求解任意段数机械臂的末端位置逆运动学，并检查关节限位。
- `solve_constrained_drag(...)`：在节点位置约束下求解中间关节拖动，供桌面交互层调用。
- 数值求解可能受初始姿态、约束可行性和收敛容差影响；调用方需要检查收敛状态与末端误差。

### `joint_limits.py`

- `JointLimits(lower, upper)`：配置每个关节的角度下限和上限。
- `contains(joint_angles)`：判断所有关节角是否满足限位。
- `violations(joint_angles)`：返回违反限位的关节索引，从 0 开始计数。
- `filter_ik_solutions(solutions, joint_limits)`：把逆运动学解分为符合限位的解和违反限位的解。
- 返回的 `IKSolutionFilterResult` 包含 `valid_solutions` 和 `rejected_solutions`；每个被拒绝的解会记录违反限位的关节索引。
- 几何可达性仍由逆运动学结果中的 `reachable` 判断。即使目标在几何上可达，也可能没有任何解符合指定的关节限位。
- 默认容差为 1e-9，用于吸收角度边界附近的浮点误差。

## 安装与测试

需要 Python 3.10 或更新版本，以及 NumPy。测试使用 Python 自带的 `unittest`，不要求安装 pytest。

在本目录运行：

```powershell
python -m pip install -r requirements.txt
python -m unittest -v
```

当前测试覆盖正运动学、雅可比矩阵的中心有限差分验证、奇异与非奇异姿态、两连杆解析逆运动学、任意段数数值逆运动学、节点约束拖动、不可达目标、工作空间边界、关节限位、部分解筛选、非法输入和浮点容差。

## 最小示例

```python
import numpy as np

from inverse_kinematics import solve_two_link_ik
from joint_limits import JointLimits, filter_ik_solutions
from planar_arm import PlanarArm

arm = PlanarArm([1.0, 1.0])
result = solve_two_link_ik([1.0, 1.0], [1.0, 1.0])

limits = JointLimits(
    [np.deg2rad(-10), np.deg2rad(80)],
    [np.deg2rad(10), np.deg2rad(100)],
)
filtered = filter_ik_solutions(result.solutions, limits)

print("几何上是否可达：", result.reachable)
print("符合关节限位的解：", filtered.valid_solutions)
print("违反限位的解：", filtered.rejected_solutions)

for solution in filtered.valid_solutions:
    print("末端位置：", arm.forward_kinematics(solution)[-1])
```

## 已知限制

- 逆运动学目前只支持两连杆平面机械臂，不支持任意数量连杆的通用解析逆解。
- 关节限位筛选目前直接比较输入角度与上下限，不会自动把角度加减整周后再匹配多圈角度范围。
- 暂未实现电机能力约束、摩擦、接触动力学及动力学约束轨迹规划。
- 运动学模块只描述平面几何；动力学模块另行模拟刚性连杆质量、质心、平面转动惯量、重力和关节驱动力矩。
- 雅可比矩阵只针对末端二维位置，不包括末端姿态。
- 两连杆奇异边界测试不能覆盖空间六自由度机械臂的全部奇异类型。

### dynamics.py

- `PlanarArmDynamics(...)`：建立二维平面刚性串联机械臂动力学模型。
- 连杆质量默认每根 1 kg，质心默认位于连杆中点，末端负载默认 0 kg；未指定 `link_inertias` 时，假设连杆为均匀细杆，使用 `I_com = m * L**2 / 12`。
- `gravity_torques(joint_angles)`：返回静态抵抗重力所需的关节力矩，单位 N·m。
- `inverse_dynamics(joint_angles, joint_velocities, joint_accelerations)`：采用平面递归牛顿—欧拉算法计算指定关节状态下的驱动力矩，角度/角速度/角加速度单位分别为 rad、rad/s、rad/s²。
- 末端负载当前按末端点质量处理；模型不包含摩擦、柔性、齿轮传动、接触力或电机能力限制。
- 输出表示理想刚体模型下的理论驱动力矩，不足以单独判断真实电机能否执行轨迹。

动力学测试覆盖手算静态力矩、竖直姿态、单连杆角加速度等效惯量、静态逆动力学与重力模型的一致性、速度耦合、非法输入，以及通过中心有限差分构造的拉格朗日能量法参考解对 2–4 连杆随机动态状态的交叉验证。参考解用于独立核验递归牛顿—欧拉实现，不应将其视为解析精确解。

### `trajectory_dynamics.py`

- `calculate_trajectory_torques(model, times, joint_angles)`：根据离散时间戳和关节角轨迹估计角速度、角加速度，并逐采样点调用逆动力学返回关节力矩矩阵，单位 N·m。
- 关节角先按时间轴展开，减少跨越 ±π 时的差分跳变；采样时间必须严格递增。
- 少于三个采样点时，无法可靠估计加速度，函数按零加速度处理；实际曲线精度受轨迹采样间隔和数值微分误差影响。

### `trajectory_planner.py`

- `quintic_time_scaling(start, end, duration, sample_count)`：生成起终点速度和加速度均为零的五次关节空间时间曲线。
- `plan_joint_trajectory(...)`：在关节空间起终点之间使用五次时间标定，逐步增加运动时间，直到角速度、角加速度与逆动力学力矩满足上限；可检查关节角限位。
- `plan_cartesian_path(...)`：对用户给定的笛卡尔路径连续求解数值逆运动学，再进行时间标定和动力学约束检查，支持任意连杆数量。
- `plan_cartesian_line(...)`：提供两连杆机械臂笛卡尔直线的解析逆运动学路径生成。
- 自动延时只会改变时间参数，不会修复几何不可达、关节限位冲突或静态重力力矩已超限等问题；搜索结果是当前固定路径上的可行解，不保证全局最短时间。所有速度、加速度与力矩检查均基于离散采样，实际执行仍需留出安全裕度。
