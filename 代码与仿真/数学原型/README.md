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

### `elbow_drag.py`

- `solve_elbow_drag_candidates(mouse_position, link_lengths, line_direction, previous_target, tolerance=1e-9)`：求解拖动第二关节时的可行姿态。
- `line_direction` 是拖动开始时由第一关节指向末端的固定方向；本次拖动期间末端只能沿该直线移动。
- 候选姿态保持两根连杆长度不变，并按末端位置与上一次有效末端位置的距离排序，减少不必要的跳变。
- 若鼠标方向对应的肘关节位置不可行，会将其限制到最近的可行方向；最终仍需通过独立关节限位检查。
- `ElbowDragCandidate` 包含关节角、末端位置和第二关节位置，可用于界面更新和几何约束测试。

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
- 暂未实现速度、加速度、转矩约束、轨迹规划、动力学和图形界面。
- 只描述平面内的几何运动学，不模拟质量、惯量、重力、摩擦或关节转矩。
- 雅可比矩阵只针对末端二维位置，不包括末端姿态。
- 两连杆奇异边界测试不能覆盖空间六自由度机械臂的全部奇异类型。
