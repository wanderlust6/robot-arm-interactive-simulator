"""机械臂交互式仿真平台的首个桌面界面原型。"""

import math
import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QFrame,
    QPushButton,
    QSpinBox,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MATH_DIRECTORY = PROJECT_ROOT / "代码与仿真" / "数学原型"
sys.path.insert(0, str(MATH_DIRECTORY))

from elbow_drag import solve_elbow_drag_candidates
from inverse_kinematics import solve_two_link_ik
from numerical_ik import solve_n_link_ik, solve_constrained_drag
from joint_limits import JointLimits, filter_ik_solutions
from planar_arm import PlanarArm
from workspace import ArmWorkspace


class MainWindow(QMainWindow):
    """管理机械臂参数、逆运动学求解和工作区显示。"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("机械臂交互式仿真平台")
        self.resize(1180, 760)

        self.arm = PlanarArm([1.0, 1.0, 1.0])
        self.joint_angles = np.radians([90.0, -90.0, 90.0])
        self.joint_limits = JointLimits(
            [math.radians(-180)] * self.arm.joint_count,
            [math.radians(180)] * self.arm.joint_count,
        )
        self.target = self.arm.forward_kinematics(self.joint_angles)[-1]
        self.target_status = "valid"
        self._updating_controls = False
        self.fixed_nodes = {}

        self._elbow_drag_line_direction = None
        self.workspace = ArmWorkspace(self)
        self.setCentralWidget(self.workspace)
        self.workspace.target_dragged.connect(self._on_target_dragged)
        self.workspace.elbow_drag_started.connect(self._on_elbow_drag_started)
        self.workspace.elbow_dragged.connect(self._on_elbow_dragged)
        self.workspace.elbow_drag_finished.connect(self._on_elbow_drag_finished)
        self.workspace.intermediate_joint_dragged.connect(self._on_intermediate_joint_dragged)
        self.workspace.object_selected.connect(self._on_object_selected)

        self._build_toolbar()
        self._build_properties_panel()
        self._build_joint_popup()
        self._sync_controls()
        self._refresh_workspace()

        self.statusBar().showMessage(
            "就绪：拖动橙色末端移动目标；拖动中间关节调整上游姿态。"
        )

    def _build_toolbar(self):
        toolbar = self.addToolBar("主工具栏")
        toolbar.setMovable(False)

        reset_action = QAction("重置姿态", self)
        reset_action.triggered.connect(self._reset_pose)
        toolbar.addAction(reset_action)

        toolbar.addSeparator()
        toolbar.addWidget(QLabel("连杆数量"))
        self.link_count_control = QSpinBox()
        self.link_count_control.setRange(1, 12)
        self.link_count_control.setValue(self.arm.joint_count)
        self.link_count_control.setToolTip("设置平面串联机械臂的连杆数量")
        toolbar.addWidget(self.link_count_control)
        apply_count = QPushButton("应用")
        apply_count.clicked.connect(self._set_link_count)
        toolbar.addWidget(apply_count)
        toolbar.addSeparator()
        hint = QLabel("固定节点后拖拽其他节点，系统会自动协调未固定关节")
        toolbar.addWidget(hint)

    def _build_properties_panel(self):
        dock = QDockWidget("属性与状态", self)
        dock.setObjectName("属性与状态面板")
        dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        dock.setMinimumWidth(290)

        panel = QWidget(dock)
        layout = QVBoxLayout(panel)

        link_group = QGroupBox("连杆参数")
        link_form = QFormLayout(link_group)
        self.link_form = link_form
        self.link_controls = []
        for index, length in enumerate(self.arm.link_lengths):
            control = self._make_spinbox(0.1, 10.0, 2, 0.1)
            control.setValue(float(length))
            control.valueChanged.connect(self._on_link_lengths_changed)
            self.link_controls.append(control)
            link_form.addRow(f"第 {index + 1} 段长度", control)
        layout.addWidget(link_group)

        limit_group = QGroupBox("关节角限位（度）")
        limit_form = QFormLayout(limit_group)
        self.limit_form = limit_form
        self.lower_controls = []
        self.upper_controls = []
        for index in range(self.arm.joint_count):
            lower = self._make_spinbox(-360.0, 360.0, 1, 1.0)
            upper = self._make_spinbox(-360.0, 360.0, 1, 1.0)
            lower.valueChanged.connect(self._on_limits_changed)
            upper.valueChanged.connect(self._on_limits_changed)
            self.lower_controls.append(lower)
            self.upper_controls.append(upper)
            limit_form.addRow(f"关节 {index + 1} 下限", lower)
            limit_form.addRow(f"关节 {index + 1} 上限", upper)
        layout.addWidget(limit_group)

        status_group = QGroupBox("求解状态")
        status_layout = QVBoxLayout(status_group)
        self.status_label = QLabel("等待操作")
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumHeight(64)
        status_layout.addWidget(self.status_label)
        layout.addWidget(status_group)
        layout.addStretch(1)

        dock.setWidget(panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)


    @staticmethod
    def _make_spinbox(minimum, maximum, decimals, step):
        control = QDoubleSpinBox()
        control.setRange(minimum, maximum)
        control.setDecimals(decimals)
        control.setSingleStep(step)
        control.setKeyboardTracking(False)
        return control

    def _set_link_count(self):
        """重建任意数量连杆的参数控件，并保留仍然存在的参数。"""
        count = int(self.link_count_control.value())
        old_count = self.arm.joint_count
        old_lengths = self.arm.link_lengths.copy()
        old_angles = self.joint_angles.copy()
        old_lower = self.joint_limits.lower.copy()
        old_upper = self.joint_limits.upper.copy()

        lengths = [
            float(old_lengths[index]) if index < old_count else 1.0
            for index in range(count)
        ]
        angles = np.radians([90.0 if index % 2 == 0 else -90.0 for index in range(count)])
        lower = np.full(count, math.radians(-180))
        upper = np.full(count, math.radians(180))
        preserved = min(count, old_count)
        angles[:preserved] = old_angles[:preserved]
        lower[:preserved] = old_lower[:preserved]
        upper[:preserved] = old_upper[:preserved]

        while self.link_form.rowCount():
            self.link_form.removeRow(0)
        while self.limit_form.rowCount():
            self.limit_form.removeRow(0)

        self.arm = PlanarArm(lengths)
        self.joint_angles = angles
        self.joint_limits = JointLimits(lower, upper)
        self.fixed_nodes.clear()
        self.link_controls = []
        self.lower_controls = []
        self.upper_controls = []

        for index, length in enumerate(lengths):
            control = self._make_spinbox(0.1, 10.0, 2, 0.1)
            control.setValue(length)
            control.valueChanged.connect(self._on_link_lengths_changed)
            self.link_controls.append(control)
            self.link_form.addRow(f"第 {index + 1} 段长度", control)

        for index in range(count):
            lower_control = self._make_spinbox(-360.0, 360.0, 1, 1.0)
            upper_control = self._make_spinbox(-360.0, 360.0, 1, 1.0)
            lower_control.setValue(float(np.degrees(lower[index])))
            upper_control.setValue(float(np.degrees(upper[index])))
            lower_control.valueChanged.connect(self._on_limits_changed)
            upper_control.valueChanged.connect(self._on_limits_changed)
            self.lower_controls.append(lower_control)
            self.upper_controls.append(upper_control)
            self.limit_form.addRow(f"关节 {index + 1} 下限", lower_control)
            self.limit_form.addRow(f"关节 {index + 1} 上限", upper_control)

        self.target = self.arm.forward_kinematics(self.joint_angles)[-1]
        self.target_status = "valid"
        self._sync_controls()
        self._refresh_workspace()
        self._set_status(f"机械臂已调整为 {count} 段，保留了原有可用参数。", "valid")

    def _sync_controls(self):
        self._updating_controls = True
        controls = []
        controls.extend(zip(self.link_controls, self.arm.link_lengths))
        controls.extend(zip(self.lower_controls, np.degrees(self.joint_limits.lower)))
        controls.extend(zip(self.upper_controls, np.degrees(self.joint_limits.upper)))
        blockers = [QSignalBlocker(control) for control, _ in controls]
        for control, value in controls:
            control.setValue(float(value))
        del blockers
        self._updating_controls = False

    def _on_link_lengths_changed(self, _value):
        if self._updating_controls:
            return
        try:
            self.arm = PlanarArm([control.value() for control in self.link_controls])
        except ValueError as exc:
            self._set_status(str(exc), "invalid")
            return
        self.target = self.arm.forward_kinematics(self.joint_angles)[-1]
        self._update_pose_status()
        self._refresh_workspace()

    def _on_angles_changed(self, _value):
        if self._updating_controls:
            return
        self.joint_angles = np.radians([self.angle_1.value(), self.angle_2.value()])
        self.target = self.arm.forward_kinematics(self.joint_angles)[-1]
        self._update_pose_status()
        self._refresh_workspace()

    def _on_limits_changed(self, _value):
        if self._updating_controls:
            return
        lower = np.radians([control.value() for control in self.lower_controls])
        upper = np.radians([control.value() for control in self.upper_controls])
        try:
            self.joint_limits = JointLimits(lower, upper)
        except ValueError as exc:
            self.target_status = "invalid"
            self._set_status(str(exc), self.target_status)
            self._refresh_workspace()
            return

        self._on_target_dragged(float(self.target[0]), float(self.target[1]))

    def _on_target_dragged(self, x, y):
        self.target = np.array([x, y], dtype=float)
        if float(np.linalg.norm(self.target)) > float(np.sum(self.arm.link_lengths)) + 1e-8:
            self.target_status = "invalid"
            self._set_status("目标不可达：超出机械臂最大可达距离。", self.target_status)
            self._refresh_workspace()
            return
        if self.joint_limits.joint_count != self.arm.joint_count:
            self.target_status = "invalid"
            self._set_status("关节限位数量与机械臂关节数量不一致，请先同步限位设置。", self.target_status)
            self._refresh_workspace()
            return
        angles, error, converged = solve_n_link_ik(
            self.arm, self.target, self.joint_angles, self.joint_limits
        )
        if not converged:
            self.target_status = "invalid"
            self._set_status(f"数值逆运动学未收敛，末端误差为 {error:.5f}。", self.target_status)
            self._refresh_workspace()
            return
        self.joint_angles = angles
        self.target_status = "singular" if np.linalg.matrix_rank(self.arm.jacobian(angles)) < 2 else "valid"
        self._sync_controls()
        self._set_status(f"目标可达，已求解 {self.arm.joint_count} 个关节；末端误差 {error:.2e}。", self.target_status)
        self._refresh_workspace()

    def _on_elbow_drag_started(self, target_x, target_y):
        """记录拖动开始时的末端方向，并将其固定为本次拖动的直线。"""
        direction = np.array([target_x, target_y], dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            direction = self.arm.forward_kinematics(self.joint_angles)[-1]
            norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            self._elbow_drag_line_direction = np.array([1.0, 0.0])
        else:
            self._elbow_drag_line_direction = direction / norm

        self.workspace.set_drag_line_direction(self._elbow_drag_line_direction)
        self.statusBar().showMessage("正在拖动第二关节：末端被限制在拖动开始时的固定直线上。")

    def _on_intermediate_joint_dragged(self, joint_index, x, y):
        """拖动节点；存在固定节点时使用带位置约束的逆运动学。"""
        if not 1 <= joint_index <= self.arm.joint_count:
            return
        if joint_index in self.fixed_nodes:
            self._set_status(f"节点 {joint_index} 已固定，请先解除固定再拖动。", "invalid")
            return
        if self.joint_limits.joint_count != self.arm.joint_count:
            self.target_status = "invalid"
            self._set_status("关节限位数量与机械臂关节数量不一致。", self.target_status)
            self._refresh_workspace()
            return

        upstream_fixed_indices = [
            index for index in self.fixed_nodes if index < joint_index
        ]
        downstream_fixed_indices = [
            index for index in self.fixed_nodes if index > joint_index
        ]
        if downstream_fixed_indices:
            current_positions = self.arm.forward_kinematics(self.joint_angles)
            drag_constraints = {
                index: point.copy() for index, point in self.fixed_nodes.items()
            }

            if upstream_fixed_indices:
                # 拖动点位于两个固定节点之间时，区间外所有节点都保持原位置。
                left_boundary = max(upstream_fixed_indices)
                right_boundary = min(downstream_fixed_indices)
                for index in range(1, left_boundary + 1):
                    drag_constraints.setdefault(index, current_positions[index].copy())
                for index in range(right_boundary, self.arm.joint_count + 1):
                    drag_constraints.setdefault(index, current_positions[index].copy())
            else:
                # 没有上游固定节点时，以基座为左边界；最近的下游固定节点右侧全部保持原位。
                right_boundary = min(downstream_fixed_indices)
                for index in range(right_boundary, self.arm.joint_count + 1):
                    drag_constraints.setdefault(index, current_positions[index].copy())
            angles, error, converged = solve_constrained_drag(
                self.arm, joint_index, [x, y], self.joint_angles,
                drag_constraints, self.joint_limits,
                target_position=[x, y],
            )
            if not converged:
                self.target_status = "invalid"
                self._set_status(
                    f"当前拖拽目标与固定节点/关节限位不兼容（约束误差 {error:.2e}），已保持上一有效姿态。",
                    self.target_status,
                )
                self._refresh_workspace()
                return
            self.joint_angles = angles
            self.target = self.arm.forward_kinematics(angles)[-1]
            self.target_status = (
                "singular" if np.linalg.matrix_rank(self.arm.jacobian(angles)) < 2 else "valid"
            )
            self._sync_controls()
            self._set_status(
                f"约束求解成功：节点 {joint_index} 已移动，固定节点保持不动，误差 {error:.2e}。",
                self.target_status,
            )
            self._refresh_workspace()
            return

        positions = self.arm.forward_kinematics(self.joint_angles)
        pivot = positions[joint_index - 1]
        current_vector = positions[joint_index] - pivot
        target_vector = np.array([x, y], dtype=float) - pivot
        if float(np.linalg.norm(target_vector)) <= 1e-12:
            return

        current_absolute_angle = math.atan2(float(current_vector[1]), float(current_vector[0]))
        target_absolute_angle = math.atan2(float(target_vector[1]), float(target_vector[0]))
        delta = (target_absolute_angle - current_absolute_angle + math.pi) % (2.0 * math.pi) - math.pi

        candidate_angles = self.joint_angles.copy()
        angle_index = joint_index - 1
        candidate_angles[angle_index] += delta
        if not self.joint_limits.contains(candidate_angles):
            self.target_status = "invalid"
            self._set_status(
                f"拖动将使关节 {angle_index + 1} 超出角度限位，已保持上一有效姿态。",
                self.target_status,
            )
            self._refresh_workspace()
            return

        self.joint_angles = candidate_angles
        self.target = self.arm.forward_kinematics(self.joint_angles)[-1]
        self.target_status = (
            "singular"
            if np.linalg.matrix_rank(self.arm.jacobian(self.joint_angles)) < 2
            else "valid"
        )
        self._sync_controls()
        self._set_status(
            f"已旋转关节 {angle_index + 1}：其上游节点固定，下游机械臂整体绕该节点旋转。",
            self.target_status,
        )
        self._refresh_workspace()

    def _on_elbow_dragged(self, x, y):
        """根据鼠标位置求解满足固定直线和连杆长度约束的姿态。"""
        if self.arm.joint_count != 2:
            self._set_status("多段机械臂的末端拖动已支持；肘关节固定直线拖动仍仅支持两段模型。", "invalid")
            return
        if self._elbow_drag_line_direction is None:
            return
        candidates = solve_elbow_drag_candidates(
            [x, y],
            self.arm.link_lengths,
            self._elbow_drag_line_direction,
            self.target,
        )
        if not candidates:
            self.target_status = "invalid"
            self._set_status("该拖动位置不存在满足固定直线和连杆长度约束的姿态。", self.target_status)
            self._refresh_workspace()
            return

        continuous_candidates = []
        for candidate in candidates:
            angles = np.array(
                [
                    _unwrap_angle_near(candidate.joint_angles[index], self.joint_angles[index])
                    for index in range(2)
                ],
                dtype=float,
            )
            if not self.joint_limits.contains(angles):
                continue
            delta = angles - self.joint_angles
            angle_change = float(np.dot(delta, delta))
            target_change = float(np.linalg.norm(np.asarray(candidate.target) - self.target))
            continuous_candidates.append((angle_change, target_change, angles, candidate))

        if not continuous_candidates:
            self.target_status = "invalid"
            self._set_status("该拖动位置没有同时满足关节限位与连续性要求的姿态，已保持上一有效姿态。", self.target_status)
            self._refresh_workspace()
            return

        _, _, selected_angles, selected = min(
            continuous_candidates,
            key=lambda item: (item[0], item[1]),
        )

        self.joint_angles = selected_angles
        self.target = np.asarray(selected.target, dtype=float)
        self.target_status = (
            "singular"
            if abs(math.sin(self.joint_angles[1])) < 1e-8
            else "valid"
        )
        self._sync_controls()
        if self.target_status == "singular":
            self._set_status("肘关节已移动；末端沿固定直线运动，当前姿态位于奇异边界。", self.target_status)
        else:
            self._set_status("肘关节已移动；末端沿固定直线运动，连杆长度与关节限位均满足。", self.target_status)
        self._refresh_workspace()

    def _on_elbow_drag_finished(self):
        self._elbow_drag_line_direction = None

    def _build_joint_popup(self):
        self.joint_popup = QFrame(self.workspace)
        self.joint_popup.setObjectName("jointStatePopup")
        self.joint_popup.setFrameShape(QFrame.Shape.StyledPanel)
        self.joint_popup.setStyleSheet("QFrame#jointStatePopup { background: white; border: 1px solid #aebaca; border-radius: 8px; } QLabel { border: none; }")
        popup_layout = QVBoxLayout(self.joint_popup)
        popup_layout.setContentsMargins(10, 8, 10, 8)
        self.joint_popup_title = QLabel("关节状态")
        self.joint_popup_angle = QLabel()
        self.joint_popup_position = QLabel()
        self.joint_popup_pin = QPushButton("固定")
        self.joint_popup_pin.setCheckable(True)
        self.joint_popup_pin.toggled.connect(self._set_joint_popup_fixed)
        popup_layout.addWidget(self.joint_popup_title)
        popup_layout.addWidget(self.joint_popup_angle)
        popup_layout.addWidget(self.joint_popup_position)
        popup_layout.addWidget(self.joint_popup_pin)
        self.joint_popup_index = None
        self.joint_popup_fixed = False
        self.joint_popup.hide()

    def _set_joint_popup_fixed(self, fixed):
        index = self.joint_popup_index
        if index is None or index == 0:
            self.joint_popup_pin.setChecked(False)
            self.joint_popup_pin.setEnabled(False)
            return
        if fixed:
            positions = self.arm.forward_kinematics(self.joint_angles)
            self.fixed_nodes[index] = positions[index].copy()
        else:
            self.fixed_nodes.pop(index, None)
        self.joint_popup_fixed = bool(fixed)
        self.joint_popup_pin.setText("已固定" if fixed else "固定")
        self._refresh_workspace()
        if fixed:
            self._set_status(f"节点 {index} 已固定在当前坐标。", "valid")
        else:
            self._set_status(f"节点 {index} 已解除固定。", "valid")

    def _refresh_joint_popup(self):
        if self.joint_popup_index is None:
            return
        positions = self.arm.forward_kinematics(self.joint_angles)
        index = self.joint_popup_index
        angle = 0.0 if index == 0 else math.degrees(float(np.sum(self.joint_angles[:index])))
        self.joint_popup_title.setText("基座节点（始终固定）" if index == 0 else f"节点 {index}")
        self.joint_popup_angle.setText(f"角度：{angle:.2f}°")
        self.joint_popup_position.setText(f"坐标：({positions[index][0]:.3f}, {positions[index][1]:.3f})")
        self.joint_popup_pin.blockSignals(True)
        self.joint_popup_pin.setEnabled(index != 0)
        self.joint_popup_pin.setChecked(index in self.fixed_nodes)
        self.joint_popup_pin.setText("已固定" if index in self.fixed_nodes else "固定")
        self.joint_popup_pin.blockSignals(False)
        self.joint_popup_fixed = index in self.fixed_nodes

    def _on_object_selected(self, object_type, index):
        if object_type == "joint":
            self.joint_popup_index = index
            self._refresh_joint_popup()
            self.joint_popup.adjustSize()
            popup_x = max(8, self.workspace.width() - self.joint_popup.width() - 12)
            self.joint_popup.move(popup_x, 64)
            self.joint_popup.show()
            self.joint_popup.raise_()
        else:
            self.joint_popup.hide()
            self.joint_popup_index = None
            self.joint_popup_fixed = False

    def _update_pose_status(self):
        violations = self.joint_limits.violations(self.joint_angles)
        if violations:
            names = "、".join(str(index + 1) for index in violations)
            self._set_status(f"当前姿态违反第 {names} 个关节的角度限位。", "invalid")
            return
        self.target_status = "valid"
        self._set_status("当前姿态满足关节限位。", self.target_status)

    def _set_status(self, message, status):
        self.status_label.setText(message)
        self.statusBar().showMessage(message)
        colors = {
            "valid": "#237a57",
            "singular": "#a56a12",
            "invalid": "#b33131",
        }
        self.status_label.setStyleSheet(f"color: {colors.get(status, '#27364b')};")

    def _refresh_workspace(self):
        self._refresh_joint_popup()
        self.workspace.set_state(
            self.arm.link_lengths,
            self.joint_angles,
            self.target,
            self.target_status,
            self.fixed_nodes,
        )

    def _reset_pose(self):
        self.joint_angles = np.radians([
            90.0 if index % 2 == 0 else -90.0
            for index in range(self.arm.joint_count)
        ])
        self.target = self.arm.forward_kinematics(self.joint_angles)[-1]
        self.target_status = "valid"
        self._sync_controls()
        self._update_pose_status()
        self._refresh_workspace()


def _unwrap_angle_near(angle, reference):
    """将等价角转换到最接近参考角的表示，避免跨越 ±π 时数值突变。"""
    return float(reference + (angle - reference + math.pi) % (2.0 * math.pi) - math.pi)


def main():
    """启动桌面应用。"""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
