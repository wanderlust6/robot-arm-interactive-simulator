"""机械臂交互式仿真平台的首个桌面界面原型。"""

import csv
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicHermiteSpline, CubicSpline, PchipInterpolator
from PySide6.QtCore import QSignalBlocker, Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QCheckBox,
    QDockWidget,
    QDoubleSpinBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QInputDialog,
    QMessageBox,
    QScrollArea,
    QVBoxLayout,
    QWidget,
    QFrame,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QLayout,
    QSizePolicy,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MATH_DIRECTORY = PROJECT_ROOT / "代码与仿真" / "数学原型"
sys.path.insert(0, str(MATH_DIRECTORY))

from elbow_drag import solve_elbow_drag_candidates
from inverse_kinematics import solve_two_link_ik
from numerical_ik import solve_n_link_ik, solve_constrained_drag
from joint_limits import JointLimits, filter_ik_solutions
from planar_arm import PlanarArm
from dynamics import PlanarArmDynamics
from trajectory_dynamics import calculate_trajectory_torques
from trajectory_planner import plan_cartesian_path
from workspace import ArmWorkspace
from path_trajectory import sample_path_by_arclength, parameterize_path, find_sharp_corner_indices
from simulation_plot import SimulationPlot, AxisSelectionDialog, VARIABLES, variable_key
from experiment_io import write_experiment_package, read_experiment_package


class MainWindow(QMainWindow):
    """管理机械臂参数、逆运动学求解和工作区显示。"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("机械臂交互式仿真平台")
        self.resize(1180, 760)

        self.arm = PlanarArm([1.0, 1.0, 1.0])
        self._link_masses = np.ones(self.arm.joint_count, dtype=float)
        self._com_fractions = np.full(self.arm.joint_count, 0.5, dtype=float)
        self._link_inertias = self.arm.link_lengths**2 / 12.0
        self._payload_mass = 0.0
        self.joint_angles = np.radians([90.0, -90.0, 90.0])
        self.joint_limits = JointLimits(
            [math.radians(-180.0)] * self.arm.joint_count,
            [math.radians(180.0)] * self.arm.joint_count,
        )
        self.target = self.arm.forward_kinematics(self.joint_angles)[-1]
        self.target_status = "valid"
        self._updating_controls = False
        self.fixed_nodes = {}
        self._simulation_timer = QTimer(self)
        # 以 100 Hz 记录仿真数据，提升短时轨迹的绘图采样密度。
        self._simulation_timer.setInterval(10)
        self._simulation_timer.timeout.connect(self._advance_simulation)
        self._simulation_times = np.array([], dtype=float)
        self._simulation_points = np.empty((0, 2), dtype=float)
        self._simulation_time = 0.0
        self._simulation_duration = 5.0
        self._simulation_log_times = []
        self._simulation_log_positions = []
        self._simulation_log_angles = []
        self._simulation_log_velocities = []
        self._simulation_log_accelerations = []
        self._simulation_log_torques = []
        self._planned_joint_angles = None
        self._planned_path_progress = None
        self._planned_path_spline = None
        self._planned_cartesian_path = None
        self._planned_pose_constraints = {}
        self._anchor_pose_constraints = {}

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

        self.trajectory_edit_action = QAction("轨迹编辑", self)
        self.trajectory_edit_action.setCheckable(True)
        self.trajectory_edit_action.setToolTip("切换钢笔式轨迹编辑模式：单击创建直线点，按住拖动创建曲线点")
        self.trajectory_edit_action.toggled.connect(self._set_trajectory_edit_mode)
        toolbar.addAction(self.trajectory_edit_action)

        self.simulation_panel_action = QAction("模拟运行", self)
        self.simulation_panel_action.triggered.connect(lambda: self._show_panel("simulation"))
        toolbar.addAction(self.simulation_panel_action)

        self.properties_panel_action = QAction("属性与状态", self)
        self.properties_panel_action.triggered.connect(lambda: self._show_panel("properties"))
        toolbar.addAction(self.properties_panel_action)

        self.robot_settings_action = QAction("机械臂设置", self)
        self.robot_settings_action.triggered.connect(lambda: self._show_panel("robot"))
        toolbar.addAction(self.robot_settings_action)

        toolbar.addSeparator()
        self.save_experiment_action = QAction("保存实验包", self)
        self.save_experiment_action.setShortcut("Ctrl+S")
        self.save_experiment_action.triggered.connect(self._save_experiment_package)
        toolbar.addAction(self.save_experiment_action)

        self.load_experiment_action = QAction("加载实验包", self)
        self.load_experiment_action.setShortcut("Ctrl+O")
        self.load_experiment_action.triggered.connect(self._load_experiment_package)
        toolbar.addAction(self.load_experiment_action)

        self.link_count_control = QSpinBox()
        self.link_count_control.setRange(1, 12)
        self.link_count_control.setValue(self.arm.joint_count)
        self.link_count_control.setToolTip("设置平面串联机械臂的连杆数量")
        self.link_count_control.setVisible(False)
        apply_count = QPushButton("应用")
        apply_count.clicked.connect(self._set_link_count)
        apply_count.setVisible(False)
        self._apply_link_count_button = apply_count

    def _prepare_simulation(self):
        self._simulation_timer.stop()
        self._simulation_prepared = False
        self.constraint_result_label.setText("正在搜索满足约束的轨迹……")
        anchors = self._simulation_anchor_points()
        if anchors is None:
            message = "请先点击“选择路径点”，并按顺序选择至少两个紫色锚点。"
            self.constraint_result_label.setText(f"规划失败：{message}")
            self._set_status(message, "invalid")
            return
        try:
            # 每段路径至少分配 240 个几何采样点，减少相邻关键锚点映射到同一采样点的误报。
            sampled = sample_path_by_arclength(anchors, sample_count=max(240, 240 * (len(anchors) - 1)), preserve_anchors=True)
            corner_indices = find_sharp_corner_indices(anchors, sampled)
            self._planned_joint_angles = None
            self._planned_path_progress = None
            self._planned_path_spline = None
            self._planned_cartesian_path = None
            self._planned_pose_constraints = {}
            current_endpoint = self.arm.forward_kinematics(self.joint_angles)[-1]
            if np.linalg.norm(current_endpoint - sampled[0]) <= 1e-4:
                start_angles = self.joint_angles.copy()
                start_error, start_ok = float(np.linalg.norm(current_endpoint - sampled[0])), True
            else:
                start_angles, start_error, start_ok = solve_n_link_ik(
                    self.arm, sampled[0], self.joint_angles,
                    limits=self.joint_limits if self._joint_limits_enabled() else None,
                )
            if not start_ok:
                raise ValueError(f"路径起点逆运动学未收敛（误差 {start_error:.3g}），无法自动规划。")
            model = PlanarArmDynamics(
                self.arm.link_lengths, link_masses=self._link_masses,
                center_of_mass_fractions=self._com_fractions,
                link_inertias=self._link_inertias, payload_mass=self._payload_mass,
            )
            max_duration = self.max_duration_control.value() if self.max_duration_limit_enabled.isChecked() else 120.0
            initial_duration = min(1.0, max_duration)
            pose_constraints = {0: start_angles.copy()}
            plan = plan_cartesian_path(
                model, sampled, start_angles,
                self.max_joint_speed_control.value() if self.speed_limit_enabled.isChecked() else 1e12,
                self.max_joint_acceleration_control.value() if self.acceleration_limit_enabled.isChecked() else 1e12,
                self.max_joint_torque_control.value() if self.torque_limit_enabled.isChecked() else 1e12,
                initial_duration=initial_duration,
                sample_count=240, max_duration=max_duration,
                joint_limits=self.joint_limits if self._joint_limits_enabled() else None,
                pose_constraints=pose_constraints,
                corner_indices=corner_indices,
                max_end_effector_speed=(self.max_end_effector_speed_control.value() if self.end_effector_speed_limit_enabled.isChecked() else float("inf")),
                max_joint_jerk=(self.max_joint_jerk_control.value() if self.jerk_limit_enabled.isChecked() else float("inf")),
            )
            if not plan.success:
                raise ValueError(plan.message)
            times = plan.times
            self._planned_joint_angles = plan.joint_angles
            self._planned_path_progress = plan.path_progress
            # 使用规划器给出的关节角和角速度构造 Hermite 插值，保留拐点处的零速度条件。
            # 仅对时间轨迹插值，不修改用户定义的笛卡尔路径点。
            self._planned_path_spline = CubicHermiteSpline(times, plan.joint_angles, plan.joint_velocities, axis=0)
            self._planned_cartesian_path = sampled.copy()
            self._planned_pose_constraints = {
                int(index): np.asarray(angles, dtype=float).copy()
                for index, angles in pose_constraints.items()
            }
            # 仿真播放的目标位置直接来自笛卡尔路径，不再由关节空间样条反算末端位置。
            points = sampled.copy()
            self._set_status(f"规划成功：总时间 {plan.duration:.2f} s，识别到 {len(corner_indices)} 个尖角停速点，搜索 {plan.iterations} 次。", "valid")
        except ValueError as exc:
            self._planned_joint_angles = None
            self._planned_path_progress = None
            self._planned_path_spline = None
            self._planned_cartesian_path = None
            self._planned_pose_constraints = {}
            self.constraint_result_label.setText(f"规划失败：{exc}")
            self._set_status(str(exc), "invalid")
            return
        self.constraint_result_label.setText(f"路径规划成功：总时间 {float(times[-1]):.2f} s；已识别 {len(corner_indices)} 个尖角并在各拐点停速，播放时沿用规划关节轨迹。")
        self._simulation_times = times
        self._simulation_points = points
        self._simulation_duration = float(times[-1])
        self._simulation_time = 0.0
        if self._planned_joint_angles is not None:
            self.joint_angles = self._planned_joint_angles[0].copy()
            self.target = self.arm.forward_kinematics(self.joint_angles)[-1]
            self.target_status = "valid"
            self.workspace.set_state(self.arm.link_lengths, self.joint_angles, self.target, self.target_status, self.fixed_nodes)
            self._sync_controls()
        else:
            self._on_target_dragged(float(points[0, 0]), float(points[0, 1]))
        if self.target_status == "invalid":
            self._set_status("起点不可达，无法准备仿真；请调整起点或机械臂参数。", "invalid")
            return
        self.fixed_nodes[self.arm.joint_count] = self.target.copy()
        self.workspace.set_state(
            self.arm.link_lengths, self.joint_angles, self.target, self.target_status, self.fixed_nodes
        )
        self._simulation_prepared = True
        self._clear_simulation_log()
        self._record_simulation_state(0.0)
        self.select_path_points_button.setChecked(False)
        self.simulation_run_button.setEnabled(True)
        self.simulation_pause_button.setEnabled(False)
        self.simulation_stop_button.setEnabled(False)
        self.simulation_pause_button.setText("暂停")
        self._set_status("仿真已准备：机械臂已到达起点。调整姿态后点击运行。", "valid")

    def _set_path_point_selection_mode(self, enabled):
        if enabled and self.trajectory_edit_action.isChecked():
            self.trajectory_edit_action.setChecked(False)
        if enabled and self._simulation_prepared:
            self.fixed_nodes.pop(self.arm.joint_count, None)
            self._refresh_workspace()
            self._simulation_prepared = False
        self.workspace.set_path_point_selection_mode(enabled)
        if enabled:
            self.simulation_route_indices.clear()
            self.simulation_path_points_label.setText("尚未选择路径点")
            self._simulation_prepared = False
            self.simulation_run_button.setEnabled(False)
            self._show_panel("simulation")
            self._set_status("请依次点击轨迹上的紫色锚点；重复点击同一点会再次加入路径序列。", "valid")
        else:
            self._set_status("路径点选择结束。", "valid")

    def _on_path_anchor_selected(self, index):
        if not self.select_path_points_button.isChecked():
            return
        anchors = self.workspace.path_anchors()
        if not 0 <= index < len(anchors):
            return
        self.simulation_route_indices.append(index)
        self.simulation_path_points_label.setText("路径顺序：" + " → ".join(str(i + 1) for i in self.simulation_route_indices))
        self._refresh_pose_anchor_options()
        self._simulation_prepared = False
        self.simulation_run_button.setEnabled(False)

    def _refresh_pose_anchor_options(self):
        if not hasattr(self, "pose_anchor_combo"):
            return
        previous = self.pose_anchor_combo.currentData()
        self.pose_anchor_combo.clear()
        expanded_route = []
        for start, end in zip(self.simulation_route_indices, self.simulation_route_indices[1:]):
            step = 1 if end >= start else -1
            segment = list(range(start, end + step, step))
            if expanded_route:
                segment = segment[1:]
            expanded_route.extend(segment)
        available_anchors = list(dict.fromkeys(expanded_route))
        for route_position, anchor_index in enumerate(available_anchors):
            self.pose_anchor_combo.addItem(f"路径位置 {route_position + 1}（锚点 {anchor_index + 1}）", (route_position, anchor_index))
        selected = self.pose_anchor_combo.findData(previous)
        if selected >= 0:
            self.pose_anchor_combo.setCurrentIndex(selected)
        self._refresh_pose_constraint_label()

    def _activate_pose_anchor(self, _index):
        selected = self.pose_anchor_combo.currentData()
        if selected is None:
            return
        _route_position, anchor_index = selected
        anchors = self.workspace.path_anchors()
        if not 0 <= anchor_index < len(anchors):
            return
        anchor_point = np.asarray(anchors[anchor_index][0], dtype=float)
        self._simulation_timer.stop()
        self._simulation_prepared = False
        self.simulation_run_button.setEnabled(False)
        self.fixed_nodes.pop(self.arm.joint_count, None)
        self._on_target_dragged(float(anchor_point[0]), float(anchor_point[1]))
        if self.target_status == "invalid":
            self._set_status(f"锚点 {anchor_index + 1} 当前不可达，未固定末端。", "invalid")
            return
        self.target = anchor_point.copy()
        self.fixed_nodes[self.arm.joint_count] = anchor_point.copy()
        self.workspace.set_state(self.arm.link_lengths, self.joint_angles, self.target, self.target_status, self.fixed_nodes)
        self._sync_controls()
        self._refresh_pose_constraint_label()
        self._set_status(f"末端已移动并固定在锚点 {anchor_index + 1}。现在可调整其他关节，再记录姿态。", "valid")

    def _refresh_pose_constraint_label(self):
        if not hasattr(self, "pose_constraints_label"):
            return
        if not self._anchor_pose_constraints:
            self.pose_constraints_label.setText("尚未设置关键姿态；默认使用连续逆运动学解。")
            return
        lines = [f"锚点 {index + 1}：" + ", ".join(f"{value:.1f}°" for value in np.degrees(angles)) for index, angles in sorted(self._anchor_pose_constraints.items())]
        self.pose_constraints_label.setText("已记录姿态（关节角）：\n" + "\n".join(lines))

    def _record_anchor_pose_constraint(self):
        selected = self.pose_anchor_combo.currentData()
        if selected is None:
            self._set_status("请先选择路径点，再记录关键姿态。", "invalid")
            return
        _route_position, anchor_index = selected
        anchors = self.workspace.path_anchors()
        if not 0 <= anchor_index < len(anchors):
            self._set_status("所选锚点已不存在，请重新选择路径点。", "invalid")
            return
        anchor_point = np.asarray(anchors[anchor_index][0], dtype=float)
        endpoint = self.arm.forward_kinematics(self.joint_angles)[-1]
        error = float(np.linalg.norm(endpoint - anchor_point))
        if error > 1e-4:
            self._set_status(f"当前姿态不能记录：末端与锚点相差 {error:.4g} m。请先将机械臂末端移动到该锚点。", "invalid")
            return
        self._anchor_pose_constraints[anchor_index] = self.joint_angles.copy()
        self._refresh_pose_constraint_label()
        self._simulation_prepared = False
        self.simulation_run_button.setEnabled(False)
        self._set_status(f"已记录锚点 {anchor_index + 1} 的关键姿态。准备时将作为硬约束检查。", "valid")

    def _clear_anchor_pose_constraint(self):
        selected = self.pose_anchor_combo.currentData()
        if selected is None:
            return
        _route_position, anchor_index = selected
        self._anchor_pose_constraints.pop(anchor_index, None)
        self._refresh_pose_constraint_label()
        self._simulation_prepared = False
        self.simulation_run_button.setEnabled(False)
        self._set_status(f"已清除锚点 {anchor_index + 1} 的关键姿态约束。", "valid")

    def _simulation_anchor_points(self):
        anchors = self.workspace.path_anchors()
        if len(self.simulation_route_indices) < 2 or not anchors:
            return None
        route = []
        for start, end in zip(self.simulation_route_indices, self.simulation_route_indices[1:]):
            step = 1 if end >= start else -1
            segment = list(range(start, end + step, step))
            if route:
                segment = segment[1:]
            route.extend(segment)
        directed_anchors = []
        for position, index in enumerate(route):
            point, incoming, outgoing = anchors[index]
            previous_index = route[position - 1] if position > 0 else None
            next_index = route[position + 1] if position + 1 < len(route) else None
            if previous_index is None:
                directed_incoming = None
            elif previous_index < index:
                directed_incoming = incoming
            else:
                directed_incoming = outgoing
            if next_index is None:
                directed_outgoing = None
            elif next_index > index:
                directed_outgoing = outgoing
            else:
                directed_outgoing = incoming
            directed_anchors.append((point, directed_incoming, directed_outgoing))
        return directed_anchors

    def _start_simulation(self):
        # ?????????????????????????????????????
        # ??????????????????????????????
        self._prepare_simulation()
        if not getattr(self, "_simulation_prepared", False):
            return
        self._simulation_timer.stop()
        if self.trajectory_edit_action.isChecked():
            self.trajectory_edit_action.setChecked(False)
        self.fixed_nodes.pop(self.arm.joint_count, None)
        self._refresh_workspace()
        self._simulation_time = 0.0
        self._clear_simulation_log()
        start = self._simulation_points[0]
        if self._planned_joint_angles is not None:
            self.joint_angles = self._planned_joint_angles[0].copy()
            self.target = self.arm.forward_kinematics(self.joint_angles)[-1]
            self.target_status = "valid"
            self.workspace.set_state(self.arm.link_lengths, self.joint_angles, self.target, self.target_status, self.fixed_nodes)
            self._sync_controls()
        else:
            self._on_target_dragged(float(start[0]), float(start[1]))
        if self.target_status == "invalid":
            self._set_status("起点姿态当前不可执行，请重新准备。", "invalid")
            self._simulation_prepared = False
            return
        self._record_simulation_state(0.0)
        self._simulation_timer.start()
        self.simulation_pause_button.setEnabled(True)
        self.simulation_stop_button.setEnabled(True)
        self.simulation_pause_button.setText("暂停")
        self._set_status(f"轨迹仿真运行中：时长 {self._simulation_duration:.1f} s。", "valid")

    def _pause_simulation(self):
        if self._simulation_timer.isActive():
            self._simulation_timer.stop()
            self.simulation_pause_button.setText("继续")
            self._set_status(f"仿真已暂停：t = {self._simulation_time:.2f} s。", "valid")
        elif len(self._simulation_times) and self._simulation_time < self._simulation_duration:
            self._simulation_timer.start()
            self.simulation_pause_button.setText("暂停")
            self._set_status(f"轨迹仿真继续：t = {self._simulation_time:.2f} s。", "valid")

    def _stop_simulation(self):
        self._simulation_timer.stop()
        self._simulation_prepared = False
        self.fixed_nodes.pop(self.arm.joint_count, None)
        self._refresh_workspace()
        self._simulation_time = 0.0
        self.simulation_pause_button.setEnabled(False)
        self.simulation_stop_button.setEnabled(False)
        self.simulation_pause_button.setText("暂停")
        if len(self._simulation_points):
            start = self._simulation_points[0]
            self._on_target_dragged(float(start[0]), float(start[1]))
        self._refresh_simulation_plot()
        self._refresh_data_table()
        self._set_status("轨迹仿真已停止，已保留停止前的数据。", "valid" if self.target_status != "invalid" else "invalid")

    def _clear_simulation_log(self):
        self._simulation_log_times = []
        self._simulation_log_positions = []
        self._simulation_log_angles = []
        self._simulation_log_velocities = []
        self._simulation_log_accelerations = []
        self._simulation_log_torques = []
        self._refresh_simulation_plot()

    def _record_simulation_state(self, time_value):
        """记录当前逆运动学求解后的真实末端位置和关节角。"""
        positions = self.arm.forward_kinematics(self.joint_angles)
        self._simulation_log_times.append(float(time_value))
        self._simulation_log_positions.append(positions.copy())
        self._simulation_log_angles.append(self.joint_angles.copy())
        if self._planned_path_spline is not None:
            sample_time = float(time_value)
            self._simulation_log_velocities.append(np.asarray(self._planned_path_spline(sample_time, 1), dtype=float))
            self._simulation_log_accelerations.append(np.asarray(self._planned_path_spline(sample_time, 2), dtype=float))
        else:
            self._simulation_log_velocities.append(np.zeros(self.arm.joint_count, dtype=float))
            self._simulation_log_accelerations.append(np.zeros(self.arm.joint_count, dtype=float))
        self._recalculate_simulation_torques(incremental=True)
        self._refresh_simulation_plot()
        self._refresh_data_table()

    def _recalculate_simulation_torques(self, incremental=False):
        """根据当前已记录的关节轨迹估计角速度/角加速度并计算力矩。"""
        sample_count = len(self._simulation_log_times)
        if sample_count == 0:
            self._simulation_log_torques = []
            return

        times = np.asarray(self._simulation_log_times, dtype=float)
        angles = np.asarray(self._simulation_log_angles, dtype=float)
        if angles.shape != (sample_count, self.arm.joint_count):
            self._simulation_log_torques = []
            return

        model = PlanarArmDynamics(
            self.arm.link_lengths,
            link_masses=self._link_masses,
            center_of_mass_fractions=self._com_fractions,
            link_inertias=self._link_inertias,
            payload_mass=self._payload_mass,
        )
        if incremental and sample_count > 1 and len(self._simulation_log_torques) == sample_count - 1:
            window_start = max(0, sample_count - 5)
            window_torques = calculate_trajectory_torques(
                model, times[window_start:], angles[window_start:]
            )
            if window_start == 0:
                self._simulation_log_torques = [row.copy() for row in window_torques]
            else:
                self._simulation_log_torques.append(window_torques[-1].copy())
                for index in range(max(0, sample_count - 3), sample_count):
                    self._simulation_log_torques[index] = window_torques[index - window_start].copy()
        else:
            torques = calculate_trajectory_torques(model, times, angles)
            self._simulation_log_torques = [row.copy() for row in torques]

    def _on_dynamics_parameters_changed(self, _value=None):
        self._link_masses = np.asarray(
            [control.value() for control in self._link_mass_controls], dtype=float
        )
        self._com_fractions = np.asarray(
            [control.value() for control in self._com_fraction_controls], dtype=float
        )
        self._link_inertias = np.asarray(
            [control.value() for control in self._link_inertia_controls], dtype=float
        )
        self._payload_mass = float(self._payload_mass_control.value())
        self._recalculate_simulation_torques()
        self._refresh_simulation_plot()
        self._refresh_data_table()
        self.statusBar().showMessage("动力学参数已更新，当前记录轨迹的力矩已重新计算。", 4000)

    def _refresh_simulation_plot(self):
        plot = getattr(self, "simulation_plot", None)
        if plot is None:
            return
        plot.set_data(
            self._simulation_log_times,
            self._simulation_log_positions,
            self._simulation_log_angles,
            self._simulation_time,
            self._simulation_duration,
            self._simulation_log_torques,
            self._simulation_log_velocities,
            self._simulation_log_accelerations,
        )
        if self._plot_window is None:
            plot.set_configuration(self._plot_configs[0]["x"], self._plot_configs[0]["ys"], "shared")
            plot.chart_title = self._chart_title(self._plot_configs[0], 1)
            plot.update()
        else:
            while len(self._plot_widgets) < len(self._plot_configs):
                extra = SimulationPlot(self._plot_charts_container)
                extra.setMinimumHeight(320)
                self._plot_widgets.append(extra)
                self._plot_charts_layout.addWidget(extra, 0)
            for index, config in enumerate(self._plot_configs):
                self._plot_widgets[index].set_data(self._simulation_log_times, self._simulation_log_positions, self._simulation_log_angles, self._simulation_time, self._simulation_duration, self._simulation_log_torques, self._simulation_log_velocities, self._simulation_log_accelerations)
                self._plot_widgets[index].set_configuration(config["x"], config["ys"], "shared")
                self._plot_widgets[index].chart_title = self._chart_title(config, index + 1)
                self._plot_widgets[index].update()
            for index in range(len(self._plot_widgets) - 1, len(self._plot_configs) - 1, -1):
                self._plot_widgets[index].hide()
            # QScrollArea 在 widgetResizable=True 时可能把内容容器压回视口高度，
            # 因此显式按当前可见图表数量设置内容高度，确保纵向滚动条真正生效。
            chart_height = 320
            chart_spacing = self._plot_charts_layout.spacing()
            visible_count = len(self._plot_configs)
            content_height = visible_count * chart_height + max(0, visible_count - 1) * chart_spacing
            self._plot_charts_container.setMinimumHeight(content_height)

    def _chart_title(self, config, chart_index):
        labels = dict(self._plot_variables())
        x_label = labels.get(config["x"], config["x"])
        y_labels = [labels.get(key, key) for key in config["ys"]]
        return "图 " + str(chart_index) + "  ·  X：" + x_label + "  ·  Y：" + "、".join(y_labels)

    def _plot_summary_text(self):
        labels = dict(self._plot_variables())
        lines = []
        for chart_index, config in enumerate(self._plot_configs, 1):
            lines.append("图 " + str(chart_index))
            lines.append("  X 轴：" + labels.get(config["x"], config["x"]))
            for series_index, key in enumerate(config["ys"], 1):
                lines.append("  曲线 " + str(series_index) + "，Y 轴：" + labels.get(key, key))
        return chr(10).join(lines)

    def _open_plot_window(self):
        if self._plot_window is None:
            self._plot_window = QDialog(self)
            self._plot_window.setWindowTitle("仿真数据绘图")
            self._plot_window.resize(1100, 760)
            layout = QVBoxLayout(self._plot_window)
            # 子控件的 sizeHint 不应反向撑大整个对话框；图表溢出交给滚动区域处理。
            layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
            self._plot_window_layout = layout
            self._plot_scroll_area = QScrollArea(self._plot_window)
            self._plot_scroll_area.setMinimumSize(0, 0)
            self._plot_scroll_area.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            self._plot_scroll_area.setWidgetResizable(True)
            self._plot_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            self._plot_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            self._plot_charts_container = QWidget()
            self._plot_charts_layout = QVBoxLayout(self._plot_charts_container)
            self._plot_charts_layout.setContentsMargins(0, 0, 0, 0)
            self._plot_charts_layout.setSpacing(10)
            self._plot_widgets = [self.simulation_plot]
            self.simulation_plot.setMinimumHeight(320)
            self._plot_charts_layout.addWidget(self.simulation_plot, 0)
            self._plot_scroll_area.setWidget(self._plot_charts_container)
            layout.addWidget(self._plot_scroll_area, 1)
            self._data_button = QPushButton("查看数据")
            self._data_button.clicked.connect(self._open_data_window)
            layout.addWidget(self._data_button)
            export_image = QPushButton("导出图像")
            export_image.clicked.connect(self._export_plot_image)
            layout.addWidget(export_image)
            self._refresh_simulation_plot()
        self._plot_window.show()

    def _experiment_configuration(self):
        anchors = []
        for point, incoming, outgoing in self.workspace.path_anchors():
            anchors.append({
                "point": np.asarray(point, dtype=float).tolist(),
                "incoming": None if incoming is None else np.asarray(incoming, dtype=float).tolist(),
                "outgoing": None if outgoing is None else np.asarray(outgoing, dtype=float).tolist(),
            })
        return {
            "model": {
                "link_lengths_m": self.arm.link_lengths.tolist(),
                "joint_angles_rad": self.joint_angles.tolist(),
                "target_m": self.target.tolist(),
                "fixed_nodes": {str(k): np.asarray(v, dtype=float).tolist() for k, v in self.fixed_nodes.items()},
                "joint_limits_rad": {
                    "lower": self.joint_limits.lower.tolist(), "upper": self.joint_limits.upper.tolist(),
                    "lower_enabled": [c.isChecked() for c in self.lower_limit_enabled],
                    "upper_enabled": [c.isChecked() for c in self.upper_limit_enabled],
                },
            },
            "dynamics": {
                "link_masses_kg": self._link_masses.tolist(),
                "center_of_mass_fractions": self._com_fractions.tolist(),
                "link_inertias_kg_m2": self._link_inertias.tolist(),
                "payload_mass_kg": float(self._payload_mass),
            },
            "constraints": {
                "max_duration": {"enabled": self.max_duration_limit_enabled.isChecked(), "value_s": self.max_duration_control.value()},
                "joint_speed": {"enabled": self.speed_limit_enabled.isChecked(), "value_rad_s": self.max_joint_speed_control.value()},
                "end_effector_speed": {"enabled": self.end_effector_speed_limit_enabled.isChecked(), "value_m_s": self.max_end_effector_speed_control.value()},
                "joint_acceleration": {"enabled": self.acceleration_limit_enabled.isChecked(), "value_rad_s2": self.max_joint_acceleration_control.value()},
                "joint_jerk": {"enabled": self.jerk_limit_enabled.isChecked(), "value_rad_s3": self.max_joint_jerk_control.value()},
                "joint_torque": {"enabled": self.torque_limit_enabled.isChecked(), "value_nm": self.max_joint_torque_control.value()},
            },
            "trajectory": {
                "anchors": anchors, "selected_route_indices": list(self.simulation_route_indices),
            },
            "plots": {
                "configurations": self._plot_configs,
                "display_mode": self._plot_display_mode,
                "data_columns": list(getattr(self, "_data_columns", ["time"])),
            },
            "algorithm": {
                "planner": "plan_cartesian_path",
                "path_sampling": "sample_path_by_arclength",
                "minimum_samples_per_anchor_segment": 240,
                "time_scaling": "quintic",
                "joint_path_interpolation": "PCHIP, segmented at detected sharp corners",
                "sharp_corner_angle_tolerance_deg": 1.0,
                "playback_sampling_target_hz": 100,
                "max_planning_duration_s": self.max_duration_control.value() if self.max_duration_limit_enabled.isChecked() else 120.0,
            },
            "simulation": {"duration_s": float(self._simulation_duration), "recorded_samples": len(self._simulation_log_times)},
        }

    def _save_experiment_package(self):
        parent_directory = QFileDialog.getExistingDirectory(
            self, "选择实验包存放位置", str(PROJECT_ROOT / "数据与实验")
        )
        if not parent_directory:
            return
        default_name = "实验_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        name, accepted = QInputDialog.getText(
            self, "新建实验文件夹", "实验文件夹名称：", text=default_name
        )
        if not accepted:
            return
        name = name.strip()
        if not name or name in {".", ".."} or any(ch in name for ch in '<>:"/\\|?*'):
            QMessageBox.warning(self, "文件夹名称无效", "请输入有效的文件夹名称。")
            return
        root = Path(parent_directory) / name
        if root.exists() and any(root.iterdir()):
            answer = QMessageBox.question(
                self, "覆盖实验包", f"文件夹“{name}”已存在且不为空，是否覆盖其中的同名实验文件？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        root.mkdir(parents=True, exist_ok=True)
        if self._simulation_timer.isActive():
            self._simulation_timer.stop()
            self.simulation_pause_button.setText("继续")
        try:
            summary = write_experiment_package(
                root, self._experiment_configuration(), self._simulation_log_times,
                self._simulation_log_positions, self._simulation_log_angles,
                self._simulation_log_velocities, self._simulation_log_accelerations,
                self._simulation_log_torques,
            )
        except (OSError, ValueError, TypeError) as exc:
            self._set_status(f"实验包保存失败：{exc}", "invalid")
            return
        self._set_status(f"实验包已保存到 {root}；记录 {summary['sample_count']} 个采样点。", "valid")

    def _load_experiment_package(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "加载实验包", str(PROJECT_ROOT / "数据与实验"), "实验配置 (experiment.json)"
        )
        if not path:
            return
        try:
            package = read_experiment_package(path)
            config = package["configuration"]
            model = config["model"]
            lengths = model["link_lengths_m"]
            if not 1 <= len(lengths) <= 12:
                raise ValueError("实验文件中的连杆数量必须在 1 到 12 段之间。")
            self._simulation_timer.stop()
            self.link_count_control.setValue(len(lengths))
            self._set_link_count()
            for control, value in zip(self.link_controls, lengths):
                control.setValue(float(value))
            self.arm = PlanarArm(lengths)
            self.joint_angles = np.asarray(model["joint_angles_rad"], dtype=float)
            if self.joint_angles.shape != (self.arm.joint_count,):
                raise ValueError("实验文件中的关节角数量与连杆数量不一致。")
            limits = model["joint_limits_rad"]
            self.joint_limits = JointLimits(limits["lower"], limits["upper"])
            for controls, values in ((self.lower_controls, np.degrees(self.joint_limits.lower)),
                                     (self.upper_controls, np.degrees(self.joint_limits.upper))):
                for control, value in zip(controls, values):
                    control.setValue(float(value))
            for controls, values in ((self.lower_limit_enabled, limits["lower_enabled"]),
                                     (self.upper_limit_enabled, limits["upper_enabled"])):
                for control, value in zip(controls, values):
                    control.setChecked(bool(value))
            dynamics = config["dynamics"]
            self._link_masses = np.asarray(dynamics["link_masses_kg"], dtype=float)
            self._com_fractions = np.asarray(dynamics["center_of_mass_fractions"], dtype=float)
            self._link_inertias = np.asarray(dynamics["link_inertias_kg_m2"], dtype=float)
            self._payload_mass = float(dynamics["payload_mass_kg"])
            for name, values in (("_link_mass_controls", self._link_masses),
                                 ("_com_fraction_controls", self._com_fractions),
                                 ("_link_inertia_controls", self._link_inertias)):
                for control, value in zip(getattr(self, name), values):
                    control.setValue(float(value))
            self._payload_mass_control.setValue(self._payload_mass)
            constraints = config["constraints"]
            control_map = (
                ("max_duration", self.max_duration_limit_enabled, self.max_duration_control, "value_s"),
                ("joint_speed", self.speed_limit_enabled, self.max_joint_speed_control, "value_rad_s"),
                ("end_effector_speed", self.end_effector_speed_limit_enabled, self.max_end_effector_speed_control, "value_m_s"),
                ("joint_acceleration", self.acceleration_limit_enabled, self.max_joint_acceleration_control, "value_rad_s2"),
                ("joint_jerk", self.jerk_limit_enabled, self.max_joint_jerk_control, "value_rad_s3"),
                ("joint_torque", self.torque_limit_enabled, self.max_joint_torque_control, "value_nm"),
            )
            for key, checkbox, spinbox, value_key in control_map:
                checkbox.setChecked(bool(constraints[key]["enabled"]))
                spinbox.setValue(float(constraints[key][value_key]))
            trajectory = config["trajectory"]
            self.workspace._path_anchors = [
                (np.asarray(a["point"], dtype=float),
                 None if a["incoming"] is None else np.asarray(a["incoming"], dtype=float),
                 None if a["outgoing"] is None else np.asarray(a["outgoing"], dtype=float))
                for a in trajectory["anchors"]
            ]
            self.workspace.update()
            self.simulation_route_indices = [int(i) for i in trajectory["selected_route_indices"]]
            if any(i < 0 or i >= len(self.workspace._path_anchors) for i in self.simulation_route_indices):
                raise ValueError("实验文件中的路径序号超出锚点范围。")
            self.simulation_path_points_label.setText(
                "路径顺序：" + " → ".join(str(i + 1) for i in self.simulation_route_indices)
                if self.simulation_route_indices else "尚未选择路径点"
            )
            self._refresh_pose_anchor_options()
            plots = config.get("plots", {})
            self._plot_configs = plots.get("configurations", [{"x": "time", "ys": ["node_x:1"]}])
            self._plot_display_mode = plots.get("display_mode", "shared")
            self._data_columns = plots.get("data_columns", ["time"])
            self._refresh_plot_configuration_labels()
            self._simulation_duration = float(config.get("simulation", {}).get("duration_s", 5.0))
            self.target = np.asarray(model["target_m"], dtype=float)
            self.fixed_nodes = {int(k): np.asarray(v, dtype=float) for k, v in model.get("fixed_nodes", {}).items()}
            self.target_status = "valid"
            self._simulation_time = 0.0
            self._simulation_prepared = False
            self._planned_path_spline = self._planned_joint_angles = self._planned_cartesian_path = None
            self._simulation_log_times = package["times"]
            self._simulation_log_positions = package["positions"]
            self._simulation_log_angles = package["angles"]
            self._simulation_log_velocities = package["velocities"]
            self._simulation_log_accelerations = package["accelerations"]
            self._simulation_log_torques = package["torques"]
            self._sync_controls()
            self.workspace.set_state(self.arm.link_lengths, self.joint_angles, self.target, self.target_status, self.fixed_nodes)
            self._refresh_workspace()
            self._refresh_simulation_plot()
            self._refresh_data_table()
            self._set_status(f"已加载实验包：{Path(path).parent}。重新运行前请点击“准备”。", "valid")
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            self._set_status(f"实验包加载失败：{exc}", "invalid")

    def _export_plot_image(self):
        if self._plot_window is None or not self._plot_widgets:
            self.statusBar().showMessage("请先打开绘图窗口。", 4000)
            return
        path, selected_filter = QFileDialog.getSaveFileName(
            self._plot_window,
            "导出当前图像",
            "仿真曲线.png",
            "PNG 图像 (*.png);;JPEG 图像 (*.jpg *.jpeg);;BMP 图像 (*.bmp)",
        )
        if not path:
            return
        suffix = Path(path).suffix.lower()
        if suffix not in (".png", ".jpg", ".jpeg", ".bmp"):
            suffix = ".jpg" if "JPEG" in selected_filter else ".bmp" if "BMP" in selected_filter else ".png"
            path += suffix
        image = self._plot_widgets[0].grab()
        if image.isNull() or not image.save(path):
            self.statusBar().showMessage("图像导出失败，请更换保存位置或格式。", 5000)
            return
        self.statusBar().showMessage(f"图像已导出：{path}", 6000)

    def _open_data_window(self):
        if not self._simulation_log_times:
            self.statusBar().showMessage("当前没有可查看的仿真数据。", 4000)
            return
        if getattr(self, "_data_window", None) is None:
            self._data_window = QDialog(self)
            self._data_window.setWindowTitle("仿真数据分析")
            self._data_window.resize(1000, 650)
            layout = QVBoxLayout(self._data_window)
            self._data_table = QTableWidget(self._data_window)
            self._data_table.setAlternatingRowColors(True)
            self._data_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            layout.addWidget(self._data_table, 1)
            controls = QHBoxLayout()
            add_column = QPushButton("+ 添加数据列")
            add_column.clicked.connect(self._add_data_column)
            remove_column = QPushButton("删除选中列")
            remove_column.clicked.connect(self._remove_data_column)
            controls.addWidget(add_column)
            controls.addWidget(remove_column)
            export_data = QPushButton("导出 CSV")
            export_data.clicked.connect(self._export_data_csv)
            controls.addWidget(export_data)
            controls.addStretch(1)
            layout.addLayout(controls)
            self._data_columns = ["time"]
        self._refresh_data_table()
        self._data_window.show()

    def _export_data_csv(self):
        if not self._simulation_log_times:
            self.statusBar().showMessage("当前没有可导出的仿真数据。", 4000)
            return
        path, _ = QFileDialog.getSaveFileName(
            self._data_window,
            "导出仿真数据",
            "仿真数据.csv",
            "CSV 文件 (*.csv)",
        )
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        labels = dict(self._plot_variables())
        columns = list(getattr(self, "_data_columns", ["time"]))
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as file:
                writer = csv.writer(file)
                writer.writerow([labels.get(key, key) for key in columns])
                values_by_column = [self.simulation_plot._values(key) for key in columns]
                row_count = min((len(values) for values in values_by_column), default=0)
                for row in range(row_count):
                    writer.writerow([float(values[row]) for values in values_by_column])
        except (OSError, ValueError) as exc:
            self.statusBar().showMessage(f"导出失败：{exc}", 6000)
            return
        self.statusBar().showMessage(f"数据已导出：{path}", 6000)

    def _add_data_column(self):
        dialog = QDialog(self._data_window)
        dialog.setWindowTitle("添加数据列")
        form = QFormLayout(dialog)
        objects = QComboBox()
        quantities = QComboBox()
        objects.addItem("时间", "time")
        for i in range(1, self.arm.joint_count + 1):
            objects.addItem("节点 " + str(i), "node:" + str(i))
        for i in range(self.arm.joint_count):
            objects.addItem("关节/连杆 " + str(i + 1), "joint:" + str(i))

        def fill(_index=None):
            selected = objects.currentData()
            quantities.clear()
            if selected == "time":
                quantities.addItem("时间", "time")
            else:
                prefix, idx_text = selected.split(":")
                idx = int(idx_text)
                for key, label in self._plot_variables():
                    kind, _, variable_index = key.partition(":")
                    if prefix == "node" and (kind.startswith("node_") or kind == "path") and int(variable_index or 0) == idx:
                        quantities.addItem(VARIABLES[kind][0], key)
                    elif prefix == "joint" and (kind.startswith("joint_") or kind == "link_angle") and int(variable_index or 0) == idx:
                        quantities.addItem(VARIABLES[kind][0], key)

        objects.currentIndexChanged.connect(fill)
        fill()
        form.addRow("对象", objects)
        form.addRow("物理量", quantities)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            key = quantities.currentData()
            if key and key not in self._data_columns:
                self._data_columns.append(key)
                self._refresh_data_table()

    def _remove_data_column(self):
        column = self._data_table.currentColumn()
        if column <= 0 or column >= len(self._data_columns):
            return
        self._data_columns.pop(column)
        self._refresh_data_table()

    def _refresh_data_table(self):
        if not hasattr(self, "_data_table"):
            return
        labels = dict(self._plot_variables())
        self._data_table.setRowCount(len(self._simulation_log_times))
        self._data_table.setColumnCount(len(self._data_columns))
        self._data_table.setHorizontalHeaderLabels([labels.get(key, key) for key in self._data_columns])
        for col, key in enumerate(self._data_columns):
            values = self.simulation_plot._values(key)
            for row, value in enumerate(values):
                if row >= self._data_table.rowCount():
                    break
                item = QTableWidgetItem(f"{float(value):.8g}")
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self._data_table.setItem(row, col, item)

    def _plot_variables(self):
        variables = [("time", VARIABLES["time"][0])]
        for index in range(1, self.arm.joint_count + 1):
            node_name = f"节点 {index}"
            for kind in ("path", "node_x", "node_y", "node_vx", "node_vy", "node_speed",
                         "node_ax", "node_ay", "node_accel"):
                variables.append((variable_key(kind, index), f"{node_name} · {VARIABLES[kind][0]}"))
        for index in range(self.arm.joint_count):
            for kind in ("joint_angle", "link_angle", "joint_velocity", "joint_accel", "joint_torque"):
                variables.append((variable_key(kind, index), f"关节/连杆 {index + 1} · {VARIABLES[kind][0]}"))
        return variables

    def _variable_group(self, key):
        kind = key.split(":", 1)[0]
        return VARIABLES.get(kind, ("", "其他", ""))[1]

    def _remove_plot_curve(self, chart_index, series_index):
        config = self._plot_configs[chart_index]
        if len(config["ys"]) <= 1:
            return
        config["ys"].pop(series_index)
        self._refresh_plot_configuration_labels()
        self._refresh_simulation_plot()

    def _add_plot_chart(self):
        self._plot_configs.append({"x": "time", "ys": ["node_x:3"]})
        self._refresh_plot_configuration_labels()
        self._refresh_simulation_plot()

    def _remove_plot_chart(self, chart_index):
        if len(self._plot_configs) <= 1:
            return
        self._plot_configs.pop(chart_index)
        self._refresh_plot_configuration_labels()
        self._refresh_simulation_plot()

    def _add_plot_curve(self, chart_index):
        config = self._plot_configs[chart_index]
        group = self._variable_group(config["ys"][0])
        config["ys"].append(config["ys"][0])
        series_index = len(config["ys"]) - 1
        self._refresh_plot_configuration_labels()
        self._set_plot_axis(chart_index, "y", series_index)
        self._refresh_plot_configuration_labels()
        self._refresh_simulation_plot()

    def _object_kind(self, key):
        kind, index = key.split(":", 1) if ":" in key else (key, "0")
        index = int(index)
        if kind == "time":
            return "time", "time", 0
        if kind == "path":
            return "node:" + str(index), "path", index
        if kind.startswith("node_"):
            return "node:" + str(index), kind, index
        return "joint:" + str(index), kind, index

    def _set_plot_axis(self, chart_index, axis, series_index=0):
        config = self._plot_configs[chart_index]
        current = config["x"] if axis == "x" else config["ys"][series_index]
        required_unit = (
            VARIABLES[config["ys"][0].partition(":")[0]][2]
            if axis == "y" and series_index > 0
            else None
        )
        dialog = QDialog(self)
        dialog.setWindowTitle("设置 X 轴" if axis == "x" else "设置曲线 " + str(series_index + 1) + " 的 Y 轴")
        layout = QFormLayout(dialog)
        object_combo = QComboBox()
        quantity_combo = QComboBox()
        options = []
        if axis == "x":
            options.append(("time", "时间"))
        for i in range(1, self.arm.joint_count + 1):
            node_keys = [variable_key(kind, i) for kind in ("path", "node_x", "node_y", "node_vx", "node_vy", "node_speed", "node_ax", "node_ay", "node_accel")]
            options.append(("node:" + str(i), "节点 " + str(i)))
        for i in range(self.arm.joint_count):
            joint_keys = [variable_key(kind, i) for kind in ("joint_angle", "link_angle", "joint_velocity", "joint_accel")]
            options.append(("joint:" + str(i), "关节/连杆 " + str(i + 1)))
        for value, label in options:
            object_combo.addItem(label, value)
        current_object, current_kind, current_index = self._object_kind(current)
        object_combo.setCurrentIndex(max(object_combo.findData(current_object), 0))

        def fill_quantities(_index=None):
            selected_object = object_combo.currentData()
            quantity_combo.clear()
            if selected_object == "time":
                quantity_combo.setEnabled(False)
                quantity_combo.addItem("无（时间本身）", "time")
                return
            quantity_combo.setEnabled(True)
            if selected_object == "path":
                if required_unit is None or VARIABLES["path"][2] == required_unit:
                    quantity_combo.addItem("累计路径长度", "path")
                else:
                    quantity_combo.addItem("无匹配单位", "")
                return
            prefix, index_text = selected_object.split(":")
            index = int(index_text)
            candidates = []
            for key, label in self._plot_variables():
                kind, _, var_index = key.partition(":")
                same_unit = required_unit is None or VARIABLES[kind][2] == required_unit
                if prefix == "node" and (kind.startswith("node_") or kind == "path") and int(var_index or 0) == index and same_unit:
                    candidates.append((key, kind, label))
                elif prefix == "joint" and (kind.startswith("joint_") or kind == "link_angle") and int(var_index or 0) == index and same_unit:
                    candidates.append((key, kind, label))
            for key, quantity, label in candidates:
                quantity_combo.addItem(VARIABLES[quantity][0], quantity)
            if quantity_combo.count() == 0:
                quantity_combo.addItem("无可用物理量", "")
            idx = quantity_combo.findData(current_kind)
            if idx >= 0:
                quantity_combo.setCurrentIndex(idx)

        object_combo.currentIndexChanged.connect(fill_quantities)
        fill_quantities()
        quantity_combo.setCurrentIndex(max(quantity_combo.findData(current_kind), 0))
        layout.addRow("对象", object_combo)
        layout.addRow("物理量", quantity_combo)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_object = object_combo.currentData()
            if selected_object == "time":
                key = "time"
            elif selected_object == "path":
                key = "path:0"
            else:
                prefix, index_text = selected_object.split(":")
                index = int(index_text)
                kind = quantity_combo.currentData()
                if not kind:
                    return
                key = variable_key(kind, index)
            if axis == "x":
                config["x"] = key
            else:
                config["ys"][series_index] = key
            self._refresh_plot_configuration_labels()
            self._refresh_simulation_plot()

    def _short_variable_label(self, key):
        if key == "time":
            return "时间"
        if key.startswith("path"):
            return "节点" + str(int(key.partition(":")[2] or 0)) + "·累计路径长度"
        kind, _, index_text = key.partition(":")
        index = int(index_text or 0)
        if kind.startswith("node_"):
            obj = "基座" if index == 0 else "节点" + str(index)
        else:
            obj = "关节" + str(index + 1)
        quantity = VARIABLES.get(kind, (kind, "", ""))[0]
        quantity = quantity.replace("节点 X 坐标", "X坐标").replace("节点 Y 坐标", "Y坐标")
        quantity = quantity.replace("节点 X 速度", "X速度").replace("节点 Y 速度", "Y速度")
        quantity = quantity.replace("节点速度大小", "速度大小").replace("节点 X 加速度", "X加速度").replace("节点 Y 加速度", "Y加速度")
        quantity = quantity.replace("节点加速度大小", "加速度大小").replace("关节角度", "角度").replace("连杆绝对角度", "方向角")
        quantity = quantity.replace("关节角速度", "角速度").replace("关节角加速度", "角加速度")
        return obj + "·" + quantity

    def _refresh_plot_configuration_labels(self):
        while self._plot_config_layout.count():
            item = self._plot_config_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()
        labels = dict(self._plot_variables())
        for chart_index, config in enumerate(self._plot_configs):
            group = QGroupBox("图 " + str(chart_index + 1))
            layout = QVBoxLayout(group)
            chart_header = QHBoxLayout()
            chart_header.addWidget(QLabel("图 " + str(chart_index + 1)), 1)
            remove_chart = QPushButton("×")
            remove_chart.setFixedWidth(30)
            remove_chart.setEnabled(len(self._plot_configs) > 1)
            remove_chart.setToolTip("删除这张图")
            remove_chart.clicked.connect(lambda checked=False, i=chart_index: self._remove_plot_chart(i))
            chart_header.addWidget(remove_chart)
            layout.addLayout(chart_header)
            x_button = QPushButton("X：" + self._short_variable_label(config["x"]))
            x_button.clicked.connect(lambda checked=False, i=chart_index: self._set_plot_axis(i, "x"))
            layout.addWidget(x_button)
            for series_index, key in enumerate(config["ys"]):
                row = QHBoxLayout()
                y_button = QPushButton("曲线 " + str(series_index + 1) + "：" + self._short_variable_label(key))
                y_button.clicked.connect(lambda checked=False, i=chart_index, j=series_index: self._set_plot_axis(i, "y", j))
                row.addWidget(y_button, 1)
                remove_button = QPushButton("×")
                remove_button.setFixedWidth(30)
                remove_button.setEnabled(len(config["ys"]) > 1)
                remove_button.setToolTip("删除这条曲线")
                remove_button.clicked.connect(lambda checked=False, i=chart_index, j=series_index: self._remove_plot_curve(i, j))
                row.addWidget(remove_button)
                layout.addLayout(row)
            add_curve = QPushButton("＋ 添加曲线")
            add_curve.clicked.connect(lambda checked=False, i=chart_index: self._add_plot_curve(i))
            layout.addWidget(add_curve)
            self._plot_config_layout.addWidget(group)
        add_chart = QPushButton("＋ 新增图")
        add_chart.clicked.connect(self._add_plot_chart)
        self._plot_config_layout.addWidget(add_chart)

    def _assess_trajectory_feasibility(self):
        """Compare recorded trajectory estimates against user-defined joint limits."""
        sample_count = len(self._simulation_log_times)
        if sample_count < 3:
            message = "数据不足：请先完成一段仿真，至少需要 3 个采样点。"
            self.feasibility_result_label.setText(message)
            self._set_status(message, "invalid")
            return

        times = np.asarray(self._simulation_log_times, dtype=float)
        angles = np.unwrap(np.asarray(self._simulation_log_angles, dtype=float), axis=0)
        torques = np.asarray(self._simulation_log_torques, dtype=float)
        expected_shape = (sample_count, self.arm.joint_count)
        if angles.shape != expected_shape or torques.shape != expected_shape or not np.all(np.isfinite(times)):
            message = "检查失败：轨迹数据不完整或包含无效值，请重新运行仿真。"
            self.feasibility_result_label.setText(message)
            self._set_status(message, "invalid")
            return
        if np.any(np.diff(times) <= 0.0):
            message = "检查失败：采样时间必须严格递增，请重新运行仿真。"
            self.feasibility_result_label.setText(message)
            self._set_status(message, "invalid")
            return

        edge_order = 2 if sample_count >= 3 else 1
        velocities = np.gradient(angles, times, axis=0, edge_order=edge_order)
        accelerations = np.gradient(velocities, times, axis=0, edge_order=edge_order)
        metrics = []
        if self.speed_limit_enabled.isChecked():
            metrics.append(("角速度", np.abs(velocities), float(self.max_joint_speed_control.value()), "rad/s"))
        if self.acceleration_limit_enabled.isChecked():
            metrics.append(("角加速度", np.abs(accelerations), float(self.max_joint_acceleration_control.value()), "rad/s²"))
        if self.torque_limit_enabled.isChecked():
            metrics.append(("驱动力矩", np.abs(torques), float(self.max_joint_torque_control.value()), "N·m"))
        violations = []
        lines = [f"检查采样点：{sample_count}；限值均按绝对值逐关节比较。"]
        for name, values, limit, unit in metrics:
            peaks = np.max(values, axis=0)
            over = np.argwhere(values > limit * (1.0 + 1e-9))
            if len(over):
                joint_ids = sorted(set(int(item[1]) + 1 for item in over))
                violations.append(name)
                detail = "、".join(f"关节 {index}" for index in joint_ids)
                lines.append(f"超限：{name}峰值 {float(np.max(peaks)):.3g} {unit}（上限 {limit:g}），涉及{detail}。")
            else:
                lines.append(f"通过：{name}最大值 {float(np.max(peaks)):.3g} {unit}（上限 {limit:g}）。")
        lines.append("各关节峰值与发生时刻：")
        for joint_index in range(self.arm.joint_count):
            speed_peak_index = int(np.argmax(np.abs(velocities[:, joint_index])))
            acceleration_peak_index = int(np.argmax(np.abs(accelerations[:, joint_index])))
            torque_peak_index = int(np.argmax(np.abs(torques[:, joint_index])))
            lines.append(
                f"关节 {joint_index + 1}：速度 {abs(velocities[speed_peak_index, joint_index]):.3g} rad/s "
                f"@ {times[speed_peak_index]:.3g} s；加速度 "
                f"{abs(accelerations[acceleration_peak_index, joint_index]):.3g} rad/s² "
                f"@ {times[acceleration_peak_index]:.3g} s；力矩 "
                f"{abs(torques[torque_peak_index, joint_index]):.3g} N·m "
                f"@ {times[torque_peak_index]:.3g} s。"
            )

        if violations:
            lines.insert(1, "结论：当前轨迹至少有一项超过设定限值。")
            self._set_status("轨迹可执行性检查：发现超限项。", "invalid")
        else:
            lines.insert(1, "结论：所检查的三项指标均未超过设定限值。")
            self._set_status("轨迹可执行性检查：当前设定下未发现超限。", "valid")
        lines.append("注意：本检查的角速度和角加速度由离散轨迹数值微分估算；绘图中的关节角速度/角加速度优先采用规划样条的解析导数。检查不包含碰撞、摩擦、电机动态或控制误差。可在结果区域内滚动查看完整说明。")
        self.feasibility_result_label.setText("\n".join(lines))

    def _advance_simulation(self):
        if len(self._simulation_times) < 2:
            self._simulation_timer.stop()
            return
        self._simulation_time = min(
            self._simulation_duration,
            self._simulation_time + self._simulation_timer.interval() / 1000.0,
        )
        if self._planned_joint_angles is not None:
            # 规划阶段已生成连续的关节空间轨迹；播放时必须沿用同一轨迹。
            # 不再逐帧重新求 IK，也不在关键点附近硬切关节角，避免构型分支跳变。
            sample_angles = self._planned_joint_angles
            if self._simulation_time >= self._simulation_duration:
                self.joint_angles = sample_angles[-1].copy()
            elif self._planned_path_spline is not None:
                self.joint_angles = np.asarray(
                    self._planned_path_spline(self._simulation_time), dtype=float
                )
            else:
                self.joint_angles = np.asarray([
                    np.interp(self._simulation_time, self._simulation_times, sample_angles[:, joint])
                    for joint in range(self.arm.joint_count)
                ], dtype=float)
            self.target = self.arm.forward_kinematics(self.joint_angles)[-1]
            self.target_status = "valid"
            self.workspace.set_state(
                self.arm.link_lengths, self.joint_angles, self.target, self.target_status, self.fixed_nodes
            )
            self._sync_controls()
        else:
            x = float(np.interp(self._simulation_time, self._simulation_times, self._simulation_points[:, 0]))
            y = float(np.interp(self._simulation_time, self._simulation_times, self._simulation_points[:, 1]))
            self._on_target_dragged(x, y)
        if self.target_status == "invalid":

            self._simulation_timer.stop()
            self.simulation_pause_button.setEnabled(False)
            self.simulation_stop_button.setEnabled(True)
            self._set_status("轨迹仿真中断：当前路径点不可达或逆运动学未收敛。", "invalid")
            return
        self._record_simulation_state(self._simulation_time)
        self.statusBar().showMessage(
            f"轨迹仿真运行中：t = {self._simulation_time:.2f} / {self._simulation_duration:.2f} s。"
        )
        if self._simulation_time >= self._simulation_duration:
            self._simulation_timer.stop()
            self.simulation_pause_button.setEnabled(False)
            self.simulation_stop_button.setEnabled(False)
            self.simulation_pause_button.setText("暂停")
            self._set_status("轨迹仿真完成。", "valid")

    def _set_trajectory_edit_mode(self, enabled):
        if enabled and self._simulation_timer.isActive():
            self._simulation_timer.stop()
        if enabled:
            self.select_path_points_button.setChecked(False)
            self.simulation_route_indices.clear()
            self._anchor_pose_constraints.clear()
            self._refresh_pose_anchor_options()
            self.simulation_path_points_label.setText("轨迹已进入编辑模式，请编辑完成后重新选择路径点。")
            self._simulation_prepared = False
            self.simulation_run_button.setEnabled(False)
        self.workspace.set_trajectory_edit_mode(enabled)
        self.statusBar().showMessage(
            "轨迹编辑模式：单击添加直线锚点，按住拖动创建曲线锚点。"
            if enabled
            else "机械臂操作模式：可拖动末端目标和关节。"
        )

    def _show_panel(self, page):
        self._panel_stack.setCurrentWidget(self._panel_pages[page])
        self._panel_dock.setWindowTitle({
            "properties": "属性与状态",
            "simulation": "模拟运行",
            "robot": "机械臂设置",
        }[page])
        if not self._panel_dock.isVisible():
            self._panel_dock.show()

    def _build_properties_panel(self):
        dock = QDockWidget("属性与状态", self)
        dock.setObjectName("共用侧边面板")
        dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        dock.setMinimumWidth(290)
        self._panel_dock = dock
        self._panel_stack = QStackedWidget(dock)
        self._panel_pages = {}

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
            link_form.addRow(f"第 {index + 1} 段长度（m）", control)
        layout.addWidget(link_group)

        structure_group = QGroupBox("机械臂结构与默认状态")
        structure_layout = QVBoxLayout(structure_group)
        structure_layout.addWidget(QLabel("连杆数量"))
        self.link_count_control.setVisible(True)
        self._apply_link_count_button.setVisible(True)
        structure_layout.addWidget(self.link_count_control)
        structure_layout.addWidget(self._apply_link_count_button)
        self.reset_pose_button = QPushButton("恢复默认状态")
        self.reset_pose_button.clicked.connect(self._reset_pose)
        structure_layout.addWidget(self.reset_pose_button)
        layout.addWidget(structure_group)

        limit_group = QGroupBox("关节角限位（°；内部使用 rad）")
        limit_form = QFormLayout(limit_group)
        self.limit_form = limit_form
        self.lower_controls = []
        self.upper_controls = []
        self.lower_limit_enabled = []
        self.upper_limit_enabled = []
        for index in range(self.arm.joint_count):
            lower = self._make_spinbox(-360.0, 360.0, 1, 1.0)
            upper = self._make_spinbox(-360.0, 360.0, 1, 1.0)
            lower.setValue(-180.0)
            upper.setValue(180.0)
            lower_enabled = QCheckBox("启用")
            upper_enabled = QCheckBox("启用")
            lower_enabled.setChecked(False)
            upper_enabled.setChecked(False)
            lower.valueChanged.connect(self._on_limits_changed)
            upper.valueChanged.connect(self._on_limits_changed)
            lower_enabled.toggled.connect(self._on_limits_changed)
            upper_enabled.toggled.connect(self._on_limits_changed)
            self.lower_controls.append(lower)
            self.upper_controls.append(upper)
            self.lower_limit_enabled.append(lower_enabled)
            self.upper_limit_enabled.append(upper_enabled)
            lower_row = QHBoxLayout()
            lower_row.addWidget(lower_enabled)
            lower_row.addWidget(lower, 1)
            upper_row = QHBoxLayout()
            upper_row.addWidget(upper_enabled)
            upper_row.addWidget(upper, 1)
            limit_form.addRow(f"关节 {index + 1} 下限", lower_row)
            limit_form.addRow(f"关节 {index + 1} 上限", upper_row)
        layout.addWidget(limit_group)

        status_group = QGroupBox("求解状态与奇异性")
        status_layout = QVBoxLayout(status_group)
        self.singularity_label = QLabel("雅可比分析：等待计算")
        self.singularity_label.setWordWrap(True)
        status_layout.addWidget(self.singularity_label)
        self.status_label = QLabel("等待操作")
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumHeight(64)
        status_layout.addWidget(self.status_label)
        layout.addWidget(status_group)
        layout.addStretch(1)

        self._panel_pages["properties"] = panel
        self._panel_stack.addWidget(panel)

        simulation_scroll = QScrollArea(dock)
        simulation_scroll.setWidgetResizable(True)
        simulation_scroll.setMinimumSize(0, 0)
        simulation_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        simulation_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        simulation_panel = QWidget()
        simulation_layout = QVBoxLayout(simulation_panel)
        simulation_scroll.setWidget(simulation_panel)
        simulation_layout.addWidget(QLabel("按顺序选择已编辑轨迹上的紫色锚点。可重复选择同一锚点。"))
        self.simulation_route_indices = []
        self.simulation_path_points_label = QLabel("尚未选择路径点")
        self.simulation_path_points_label.setWordWrap(True)
        simulation_layout.addWidget(self.simulation_path_points_label)
        self.select_path_points_button = QPushButton("选择路径点")
        self.select_path_points_button.setCheckable(True)
        self.select_path_points_button.toggled.connect(self._set_path_point_selection_mode)
        simulation_layout.addWidget(self.select_path_points_button)
        self.workspace.path_anchor_selected.connect(self._on_path_anchor_selected)

        self.max_duration_limit_enabled = QCheckBox("启用")
        self.max_duration_limit_enabled.setChecked(False)
        self.max_duration_control = self._make_spinbox(0.5, 120.0, 1, 0.5)
        self.max_duration_control.setValue(5.0)
        self.max_duration_control.setToolTip("启用后，轨迹必须在该总时间上限内满足其他约束；未启用时最多搜索 120 秒")
        duration_limit_row = QHBoxLayout()
        duration_limit_row.addWidget(self.max_duration_limit_enabled)
        duration_limit_row.addWidget(self.max_duration_control, 1)

        feasibility_group = QGroupBox("约束限制（理想模型）")
        self.constraint_group = feasibility_group
        feasibility_layout = QVBoxLayout(feasibility_group)
        feasibility_form = QFormLayout()
        self.max_joint_speed_control = self._make_spinbox(0.01, 100.0, 2, 0.1)
        self.max_joint_speed_control.setValue(3.0)
        self.max_joint_speed_control.setToolTip("各关节允许的最大绝对角速度，单位 rad/s")
        self.speed_limit_enabled = QCheckBox("启用")
        self.speed_limit_enabled.setChecked(False)
        speed_limit_row = QHBoxLayout()
        speed_limit_row.addWidget(self.speed_limit_enabled)
        speed_limit_row.addWidget(self.max_joint_speed_control, 1)
        feasibility_form.addRow("角速度上限（rad/s）", speed_limit_row)
        self.max_end_effector_speed_control = self._make_spinbox(0.001, 100.0, 3, 0.05)
        self.max_end_effector_speed_control.setValue(0.5)
        self.max_end_effector_speed_control.setToolTip("末端执行器允许的最大线速度，单位 m/s")
        self.end_effector_speed_limit_enabled = QCheckBox("启用")
        end_speed_row = QHBoxLayout()
        end_speed_row.addWidget(self.end_effector_speed_limit_enabled)
        end_speed_row.addWidget(self.max_end_effector_speed_control, 1)
        feasibility_form.addRow("末端线速度上限（m/s）", end_speed_row)
        self.max_joint_acceleration_control = self._make_spinbox(0.01, 500.0, 2, 0.5)
        self.max_joint_acceleration_control.setValue(10.0)
        self.max_joint_acceleration_control.setToolTip("各关节允许的最大绝对角加速度，单位 rad/s²")
        self.acceleration_limit_enabled = QCheckBox("启用")
        self.acceleration_limit_enabled.setChecked(False)
        acceleration_limit_row = QHBoxLayout()
        acceleration_limit_row.addWidget(self.acceleration_limit_enabled)
        acceleration_limit_row.addWidget(self.max_joint_acceleration_control, 1)
        feasibility_form.addRow("角加速度上限（rad/s²）", acceleration_limit_row)
        self.max_joint_jerk_control = self._make_spinbox(0.01, 10000.0, 2, 1.0)
        self.max_joint_jerk_control.setValue(50.0)
        self.max_joint_jerk_control.setToolTip("各关节允许的最大角加加速度，单位 rad/s³；用于限制角加速度变化过快")
        self.jerk_limit_enabled = QCheckBox("启用")
        jerk_limit_row = QHBoxLayout()
        jerk_limit_row.addWidget(self.jerk_limit_enabled)
        jerk_limit_row.addWidget(self.max_joint_jerk_control, 1)
        feasibility_form.addRow("角加加速度上限（rad/s³）", jerk_limit_row)
        self.max_joint_torque_control = self._make_spinbox(0.01, 10000.0, 2, 0.5)
        self.max_joint_torque_control.setValue(20.0)
        self.max_joint_torque_control.setToolTip("各关节允许的最大绝对理论驱动力矩，单位 N·m")
        self.torque_limit_enabled = QCheckBox("启用")
        self.torque_limit_enabled.setChecked(False)
        torque_limit_row = QHBoxLayout()
        torque_limit_row.addWidget(self.torque_limit_enabled)
        torque_limit_row.addWidget(self.max_joint_torque_control, 1)
        feasibility_form.addRow("力矩上限（N·m）", torque_limit_row)
        feasibility_form.addRow("总时间上限（s）", duration_limit_row)
        feasibility_layout.addLayout(feasibility_form)
        self.constraint_result_label = QLabel("点击“准备”后，程序会自动搜索满足已启用约束的轨迹，并说明失败原因。")
        self.feasibility_result_label = self.constraint_result_label
        self.constraint_result_label.setWordWrap(True)
        self.constraint_result_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        feasibility_layout.addWidget(self.constraint_result_label)
        simulation_layout.addWidget(feasibility_group)

        self.simulation_prepare_button = QPushButton("准备")
        self.simulation_prepare_button.clicked.connect(self._prepare_simulation)
        simulation_layout.addWidget(self.simulation_prepare_button)
        self.simulation_run_button = QPushButton("运行")
        self.simulation_run_button.clicked.connect(self._start_simulation)
        self.simulation_run_button.setEnabled(False)
        simulation_layout.addWidget(self.simulation_run_button)
        self.simulation_pause_button = QPushButton("暂停")
        self.simulation_pause_button.clicked.connect(self._pause_simulation)
        self.simulation_pause_button.setEnabled(False)
        simulation_layout.addWidget(self.simulation_pause_button)
        self.simulation_stop_button = QPushButton("停止")
        self.simulation_stop_button.clicked.connect(self._stop_simulation)
        self.simulation_stop_button.setEnabled(False)
        simulation_layout.addWidget(self.simulation_stop_button)
        self.simulation_status_label = QLabel("先设置路径点，再点击准备。")
        self.simulation_status_label.setWordWrap(True)
        simulation_layout.addWidget(self.simulation_status_label)
        self._plot_variables_cache = None
        self._plot_configs = [{"x": "time", "ys": ["node_x:3"]}]
        self._plot_display_mode = "shared"
        self._plot_config_layout = QVBoxLayout()
        simulation_layout.addLayout(self._plot_config_layout)
        self.simulation_plot = SimulationPlot()
        self._plot_window = None
        self.plot_summary_label = QLabel()
        self.plot_summary_label.setWordWrap(True)
        self.plot_summary_label.setVisible(False)
        simulation_layout.addWidget(self.plot_summary_label)
        self.open_plot_window_button = QPushButton("打开绘图窗口")
        self.open_plot_window_button.clicked.connect(self._open_plot_window)
        simulation_layout.addWidget(self.open_plot_window_button)
        self._refresh_plot_configuration_labels()
        simulation_layout.addStretch(1)
        self._panel_pages["simulation"] = simulation_scroll
        self._panel_stack.addWidget(simulation_scroll)

        robot_panel = QWidget(dock)
        robot_layout = QVBoxLayout(robot_panel)
        robot_layout.addWidget(QLabel("固定节点后拖拽其他节点，可通过约束求解协调关节。"))

        dynamics_group = QGroupBox("动力学参数")
        dynamics_layout = QVBoxLayout(dynamics_group)
        self._mass_form = QFormLayout()
        self._link_mass_controls = []
        for index, mass in enumerate(self._link_masses):
            control = self._make_spinbox(0.0, 100.0, 2, 0.1)
            control.setValue(float(mass))
            control.valueChanged.connect(self._on_dynamics_parameters_changed)
            self._link_mass_controls.append(control)
            self._mass_form.addRow(f"第 {index + 1} 段质量（kg）", control)
        dynamics_layout.addLayout(self._mass_form)

        self._com_form = QFormLayout()
        self._com_fraction_controls = []
        for index, fraction in enumerate(self._com_fractions):
            control = self._make_spinbox(0.0, 1.0, 3, 0.05)
            control.setValue(float(fraction))
            control.setToolTip("质心沿连杆轴线的位置比例：0 为近端关节，1 为远端端点")
            control.valueChanged.connect(self._on_dynamics_parameters_changed)
            self._com_fraction_controls.append(control)
            self._com_form.addRow(f"第 {index + 1} 段质心比例", control)
        dynamics_layout.addLayout(self._com_form)

        self._inertia_form = QFormLayout()
        self._link_inertia_controls = []
        for index, inertia in enumerate(self._link_inertias):
            control = self._make_spinbox(0.0, 10000.0, 4, 0.01)
            control.setValue(float(inertia))
            control.setToolTip("连杆绕自身质心的平面转动惯量，单位 kg·m²")
            control.valueChanged.connect(self._on_dynamics_parameters_changed)
            self._link_inertia_controls.append(control)
            self._inertia_form.addRow(f"第 {index + 1} 段转动惯量（kg·m²）", control)
        dynamics_layout.addLayout(self._inertia_form)

        payload_form = QFormLayout()
        self._payload_mass_control = self._make_spinbox(0.0, 100.0, 2, 0.1)
        self._payload_mass_control.setValue(self._payload_mass)
        self._payload_mass_control.valueChanged.connect(self._on_dynamics_parameters_changed)
        payload_form.addRow("末端负载质量（kg）", self._payload_mass_control)
        dynamics_layout.addLayout(payload_form)
        dynamics_layout.addWidget(QLabel("质心默认位于连杆中点，连杆按均匀细杆估算转动惯量。力矩是理想刚体模型的估计值。"))
        robot_layout.addWidget(dynamics_group)
        robot_layout.addStretch(1)
        self._panel_pages["robot"] = robot_panel
        self._panel_stack.addWidget(robot_panel)

        dock.setWidget(self._panel_stack)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        dock.hide()
        self._simulation_prepared = False


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
        old_masses = self._link_masses.copy()
        old_com_fractions = self._com_fractions.copy()
        old_inertias = self._link_inertias.copy()
        old_angles = self.joint_angles.copy()
        old_lower = np.radians([control.value() for control in self.lower_controls])
        old_upper = np.radians([control.value() for control in self.upper_controls])
        old_lower_enabled = [control.isChecked() for control in self.lower_limit_enabled]
        old_upper_enabled = [control.isChecked() for control in self.upper_limit_enabled]

        lengths = [
            float(old_lengths[index]) if index < old_count else 1.0
            for index in range(count)
        ]
        masses = np.asarray([
            float(old_masses[index]) if index < old_count else 1.0
            for index in range(count)
        ])
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
        self._link_masses = masses
        self._com_fractions = np.asarray([
            old_com_fractions[index] if index < old_count else 0.5
            for index in range(count)
        ], dtype=float)
        self._link_inertias = np.asarray([
            old_inertias[index] if index < old_count else masses[index] * lengths[index] ** 2 / 12.0
            for index in range(count)
        ], dtype=float)
        self.joint_angles = angles
        self.joint_limits = JointLimits(lower, upper)
        self.fixed_nodes.clear()
        self.link_controls = []
        self.lower_controls = []
        self.upper_controls = []
        self.lower_limit_enabled = []
        self.upper_limit_enabled = []

        for index, length in enumerate(lengths):
            control = self._make_spinbox(0.1, 10.0, 2, 0.1)
            control.setValue(length)
            control.valueChanged.connect(self._on_link_lengths_changed)
            self.link_controls.append(control)
            self.link_form.addRow(f"第 {index + 1} 段长度（m）", control)

        for index in range(count):
            lower_control = self._make_spinbox(-360.0, 360.0, 1, 1.0)
            upper_control = self._make_spinbox(-360.0, 360.0, 1, 1.0)
            lower_control.setValue(float(np.degrees(old_lower[index])) if index < old_count else -180.0)
            upper_control.setValue(float(np.degrees(old_upper[index])) if index < old_count else 180.0)
            lower_enabled = QCheckBox("启用")
            upper_enabled = QCheckBox("启用")
            lower_enabled.setChecked(old_lower_enabled[index] if index < old_count else False)
            upper_enabled.setChecked(old_upper_enabled[index] if index < old_count else False)
            lower_control.valueChanged.connect(self._on_limits_changed)
            upper_control.valueChanged.connect(self._on_limits_changed)
            lower_enabled.toggled.connect(self._on_limits_changed)
            upper_enabled.toggled.connect(self._on_limits_changed)
            self.lower_controls.append(lower_control)
            self.upper_controls.append(upper_control)
            self.lower_limit_enabled.append(lower_enabled)
            self.upper_limit_enabled.append(upper_enabled)
            lower_row = QHBoxLayout()
            lower_row.addWidget(lower_enabled)
            lower_row.addWidget(lower_control, 1)
            upper_row = QHBoxLayout()
            upper_row.addWidget(upper_enabled)
            upper_row.addWidget(upper_control, 1)
            self.limit_form.addRow(f"关节 {index + 1} 下限", lower_row)
            self.limit_form.addRow(f"关节 {index + 1} 上限", upper_row)

        while self._mass_form.rowCount():
            self._mass_form.removeRow(0)
        self._link_mass_controls = []
        for index, mass in enumerate(self._link_masses):
            control = self._make_spinbox(0.0, 100.0, 2, 0.1)
            control.setValue(float(mass))
            control.valueChanged.connect(self._on_dynamics_parameters_changed)
            self._link_mass_controls.append(control)
            self._mass_form.addRow(f"第 {index + 1} 段质量（kg）", control)

        for form, controls, values, label, minimum, maximum, decimals, step, tooltip in (
            (self._com_form, "_com_fraction_controls", self._com_fractions, "质心比例", 0.0, 1.0, 3, 0.05,
             "质心沿连杆轴线的位置比例：0 为近端关节，1 为远端端点"),
            (self._inertia_form, "_link_inertia_controls", self._link_inertias, "转动惯量（kg·m²）", 0.0, 10000.0, 4, 0.01,
             "连杆绕自身质心的平面转动惯量，单位 kg·m²"),
        ):
            while form.rowCount():
                form.removeRow(0)
            new_controls = []
            for index, value in enumerate(values):
                control = self._make_spinbox(minimum, maximum, decimals, step)
                control.setValue(float(value))
                control.setToolTip(tooltip)
                control.valueChanged.connect(self._on_dynamics_parameters_changed)
                new_controls.append(control)
                form.addRow(f"第 {index + 1} 段{label}", control)
            setattr(self, controls, new_controls)

        self._simulation_prepared = False
        self._simulation_timer.stop()
        self._clear_simulation_log()
        self.target = self.arm.forward_kinematics(self.joint_angles)[-1]
        self.target_status = "valid"
        self._sync_controls()
        self._on_limits_changed(0)
        self._refresh_workspace()
        self._set_status(f"机械臂已调整为 {count} 段，保留了原有可用参数。", "valid")

    def _sync_controls(self):
        self._updating_controls = True
        controls = []
        controls.extend(zip(self.link_controls, self.arm.link_lengths))
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
        self._recalculate_simulation_torques()
        self._refresh_simulation_plot()
        self._refresh_data_table()

    def _on_angles_changed(self, _value):
        if self._updating_controls:
            return
        self.joint_angles = np.radians([self.angle_1.value(), self.angle_2.value()])
        self.target = self.arm.forward_kinematics(self.joint_angles)[-1]
        self._update_pose_status()
        self._refresh_workspace()

    def _joint_limits_enabled(self):
        return any(
            control.isChecked()
            for control in (*self.lower_limit_enabled, *self.upper_limit_enabled)
        )

    def _effective_joint_limits(self):
        """????????????????????????"""
        return self.joint_limits if self._joint_limits_enabled() else None

    def _on_limits_changed(self, _value):
        if self._updating_controls:
            return
        lower_degrees = [
            control.value() if enabled.isChecked() else -1e9
            for control, enabled in zip(self.lower_controls, self.lower_limit_enabled)
        ]
        upper_degrees = [
            control.value() if enabled.isChecked() else 1e9
            for control, enabled in zip(self.upper_controls, self.upper_limit_enabled)
        ]
        lower = np.radians(lower_degrees)
        upper = np.radians(upper_degrees)
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
            self.arm, self.target, self.joint_angles, self._effective_joint_limits()
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
                drag_constraints, self._effective_joint_limits(),
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
        if self._joint_limits_enabled() and not self.joint_limits.contains(candidate_angles):
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
            if self._joint_limits_enabled() and not self.joint_limits.contains(angles):
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
        self.joint_popup_angle.setText(f"累计角度：{angle:.2f}°（{math.radians(angle):.4f} rad）")
        self.joint_popup_position.setText(f"坐标（m）：({positions[index][0]:.3f}, {positions[index][1]:.3f})")
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
        if hasattr(self, "simulation_status_label"):
            self.simulation_status_label.setText(message)
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
        self._refresh_singularity_status()

    def _refresh_singularity_status(self):
        """更新当前姿态的几何奇异性与关节角限位状态。"""
        metrics = self.arm.singularity_metrics(self.joint_angles)
        singular_text = (
            "几何状态：奇异位形"
            if metrics["is_singular"]
            else "几何状态：非奇异"
        )
        min_sv = metrics["min_singular_value"]
        condition = metrics["condition_number"]
        condition_text = "∞" if not math.isfinite(condition) else f"{condition:.4g}"
        limit_violations = []
        for index, angle in enumerate(self.joint_angles):
            lower_enabled = index < len(self.lower_limit_enabled) and self.lower_limit_enabled[index].isChecked()
            upper_enabled = index < len(self.upper_limit_enabled) and self.upper_limit_enabled[index].isChecked()
            lower = math.radians(self.lower_controls[index].value()) if index < len(self.lower_controls) else -math.inf
            upper = math.radians(self.upper_controls[index].value()) if index < len(self.upper_controls) else math.inf
            tolerance = 1e-9
            if lower_enabled and angle < lower - tolerance:
                limit_violations.append(f"关节 {index + 1} 低于下限")
            if upper_enabled and angle > upper + tolerance:
                limit_violations.append(f"关节 {index + 1} 高于上限")
        limit_text = "关节角限位：违规（" + "、".join(limit_violations) + "）" if limit_violations else "关节角限位：未发现违规"
        self.singularity_label.setText(
            f"{singular_text}\n雅可比秩：{metrics['rank']}/{metrics['max_rank']}\n"
            f"最小奇异值：{min_sv:.4g}\n条件数：{condition_text}\n{limit_text}"
        )
        self.singularity_label.setStyleSheet(
            "color: #a56a12;" if metrics["is_singular"] else "color: #237a57;"
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
