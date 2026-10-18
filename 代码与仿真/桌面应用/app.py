"""机械臂交互式仿真平台的首个桌面界面原型。"""

import csv
import math
import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSignalBlocker, Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QDoubleSpinBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QFrame,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
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
from path_trajectory import sample_path_by_arclength, parameterize_path
from simulation_plot import SimulationPlot, AxisSelectionDialog, VARIABLES, variable_key


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
        self._simulation_timer = QTimer(self)
        self._simulation_timer.setInterval(30)
        self._simulation_timer.timeout.connect(self._advance_simulation)
        self._simulation_times = np.array([], dtype=float)
        self._simulation_points = np.empty((0, 2), dtype=float)
        self._simulation_time = 0.0
        self._simulation_duration = 5.0
        self._simulation_log_times = []
        self._simulation_log_positions = []
        self._simulation_log_angles = []

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
        anchors = self._simulation_anchor_points()
        if anchors is None:
            self._set_status("请先点击“选择路径点”，并按顺序选择至少两个紫色锚点。", "invalid")
            return
        try:
            sampled = sample_path_by_arclength(anchors, sample_count=240)
            times, points = parameterize_path(sampled, self.simulation_duration_control.value())
        except ValueError as exc:
            self._set_status(str(exc), "invalid")
            return
        self._simulation_times = times
        self._simulation_points = points
        self._simulation_duration = float(times[-1])
        self._simulation_time = 0.0
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
        self._simulation_prepared = False
        self.simulation_run_button.setEnabled(False)

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
        if not getattr(self, "_simulation_prepared", False):
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
        self._refresh_simulation_plot()

    def _record_simulation_state(self, time_value):
        """记录当前逆运动学求解后的真实末端位置和关节角。"""
        positions = self.arm.forward_kinematics(self.joint_angles)
        self._simulation_log_times.append(float(time_value))
        self._simulation_log_positions.append(positions.copy())
        self._simulation_log_angles.append(self.joint_angles.copy())
        self._refresh_simulation_plot()
        self._refresh_data_table()

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
        )
        if self._plot_window is None:
            plot.set_configuration(self._plot_configs[0]["x"], self._plot_configs[0]["ys"], "shared")
            plot.chart_title = self._chart_title(self._plot_configs[0], 1)
            plot.update()
        else:
            while len(self._plot_widgets) < len(self._plot_configs):
                extra = SimulationPlot(self._plot_window)
                self._plot_widgets.append(extra)
                self._plot_window_layout.addWidget(extra, 1)
            for index, config in enumerate(self._plot_configs):
                self._plot_widgets[index].set_data(self._simulation_log_times, self._simulation_log_positions, self._simulation_log_angles, self._simulation_time, self._simulation_duration)
                self._plot_widgets[index].set_configuration(config["x"], config["ys"], "shared")
                self._plot_widgets[index].chart_title = self._chart_title(config, index + 1)
                self._plot_widgets[index].update()
            for index in range(len(self._plot_widgets) - 1, len(self._plot_configs) - 1, -1):
                self._plot_widgets[index].hide()

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
            self._plot_window_layout = layout
            self._plot_widgets = [self.simulation_plot]
            layout.addWidget(self.simulation_plot, 1)
            self._data_button = QPushButton("查看数据")
            self._data_button.clicked.connect(self._open_data_window)
            layout.addWidget(self._data_button)
            export_image = QPushButton("导出图像")
            export_image.clicked.connect(self._export_plot_image)
            layout.addWidget(export_image)
            self._refresh_simulation_plot()
        self._plot_window.show()

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
            for kind in ("joint_angle", "link_angle", "joint_velocity", "joint_accel"):
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

    def _advance_simulation(self):
        if len(self._simulation_times) < 2:
            self._simulation_timer.stop()
            return
        self._simulation_time = min(
            self._simulation_duration,
            self._simulation_time + self._simulation_timer.interval() / 1000.0,
        )
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

        self._panel_pages["properties"] = panel
        self._panel_stack.addWidget(panel)

        simulation_panel = QWidget(dock)
        simulation_layout = QVBoxLayout(simulation_panel)
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

        duration_form = QFormLayout()
        self.simulation_duration_control = self._make_spinbox(0.5, 120.0, 1, 0.5)
        self.simulation_duration_control.setValue(5.0)
        self.simulation_duration_control.setToolTip("从起点运行到终点所用的仿真时间")
        duration_form.addRow("仿真时长（s）", self.simulation_duration_control)
        simulation_layout.addLayout(duration_form)
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
        self._panel_pages["simulation"] = simulation_panel
        self._panel_stack.addWidget(simulation_panel)

        robot_panel = QWidget(dock)
        robot_layout = QVBoxLayout(robot_panel)
        robot_layout.addWidget(QLabel("固定节点后拖拽其他节点，可通过约束求解协调关节。"))
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
            self.link_form.addRow(f"第 {index + 1} 段长度（m）", control)

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
