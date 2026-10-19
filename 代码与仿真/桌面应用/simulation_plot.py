"""Configurable simulation plots and axis-selection dialogs."""

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout,
    QLabel, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget,
)


VARIABLES = {
    "time": ("时间 t (s)", "时间", "s"),
    "path": ("节点累计路径长度 (m)", "路径长度", "m"),
    "node_x": ("节点 X 坐标 (m)", "位置", "m"),
    "node_y": ("节点 Y 坐标 (m)", "位置", "m"),
    "node_vx": ("节点 X 速度 (m/s)", "线速度", "m/s"),
    "node_vy": ("节点 Y 速度 (m/s)", "线速度", "m/s"),
    "node_speed": ("节点速度大小 (m/s)", "线速度", "m/s"),
    "node_ax": ("节点 X 加速度 (m/s²)", "线加速度", "m/s²"),
    "node_ay": ("节点 Y 加速度 (m/s²)", "线加速度", "m/s²"),
    "node_accel": ("节点加速度大小 (m/s²)", "线加速度", "m/s²"),
    "joint_angle": ("关节相对角 θ (°)", "角度", "°"),
    "link_angle": ("连杆绝对方向 φ (°)", "角度", "°"),
    "joint_velocity": ("关节角速度 (rad/s)", "角速度", "rad/s"),
    "joint_accel": ("关节角加速度 (rad/s²)", "角加速度", "rad/s²"),
    "joint_torque": ("关节驱动力矩 τ (N·m)", "力矩", "N·m"),
}


def variable_key(kind, index=0):
    return f"{kind}:{int(index)}"


def parse_variable(key):
    kind, _, index = key.partition(":")
    return kind, int(index or 0)


class AxisSelectionDialog(QDialog):
    def __init__(self, title, variables, selected, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(340)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("选择一个变量作为横坐标。" if title.startswith("设置 X") else
                                "可添加多条曲线。每条曲线可选择不同节点或关节。"))
        self.combo = QComboBox()
        for key, label in variables:
            self.combo.addItem(label, key)
        index = self.combo.findData(selected)
        self.combo.setCurrentIndex(max(index, 0))
        if title.startswith("设置 X"):
            form = QFormLayout()
            form.addRow("横坐标变量", self.combo)
            layout.addLayout(form)
        else:
            self.list_widget = QListWidget()
            for key in selected:
                label = next((name for value, name in variables if value == key), key)
                self.list_widget.addItem(QListWidgetItem(label))
                self.list_widget.item(self.list_widget.count() - 1).setData(Qt.ItemDataRole.UserRole, key)
            layout.addWidget(self.list_widget)
            row = QHBoxLayout()
            add_button = QPushButton("添加曲线")
            remove_button = QPushButton("删除选中")
            add_button.clicked.connect(self._add_series)
            remove_button.clicked.connect(self._remove_series)
            row.addWidget(add_button)
            row.addWidget(remove_button)
            layout.addLayout(row)
            self.mode_combo = QComboBox()
            self.mode_combo.addItem("相同物理量：共用坐标尺度", "shared")
            self.mode_combo.addItem("不同物理量：分图显示", "separate")
            self.mode_combo.addItem("不同物理量：仍共用坐标尺度", "mixed")
            form = QFormLayout()
            form.addRow("多曲线显示", self.mode_combo)
            layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                   QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _add_series(self):
        key = self.combo.currentData()
        label = self.combo.currentText()
        item = QListWidgetItem(label)
        item.setData(Qt.ItemDataRole.UserRole, key)
        self.list_widget.addItem(item)

    def _remove_series(self):
        row = self.list_widget.currentRow()
        if row >= 0:
            self.list_widget.takeItem(row)

    def selected_x(self):
        return self.combo.currentData()

    def selected_series(self):
        return [self.list_widget.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(self.list_widget.count())]

    def selected_mode(self):
        return self.mode_combo.currentData()


class SimulationPlot(QWidget):
    """Draw configurable simulation variables against a shared x-axis."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(300)
        self.setMinimumWidth(250)
        self._times = np.array([], dtype=float)
        self._positions = np.empty((0, 1, 2), dtype=float)
        self._angles = np.empty((0, 0), dtype=float)
        self._torques = np.empty((0, 0), dtype=float)
        self._current_time = 0.0
        self._duration = 1.0
        self.x_variable = "time"
        self.y_variables = ["node_x:3", "node_y:3", "joint_angle:1"]
        self.display_mode = "separate"
        self.chart_title = ""

    def set_data(self, times, positions, angles, current_time, duration, torques=None):
        self._times = np.asarray(times, dtype=float).copy()
        pos = np.asarray(positions, dtype=float)
        if pos.size == 0:
            self._positions = np.empty((0, 1, 2), dtype=float)
        elif pos.ndim == 2:
            self._positions = pos[:, None, :]
        else:
            self._positions = pos.copy()
        angle_values = np.asarray(angles, dtype=float)
        self._angles = angle_values.copy() if angle_values.size else np.empty((0, 0), dtype=float)
        torque_values = np.asarray(torques, dtype=float) if torques is not None else np.empty((0, 0), dtype=float)
        self._torques = torque_values.copy() if torque_values.size else np.empty((0, 0), dtype=float)
        self._current_time = max(0.0, float(current_time))
        self._duration = max(float(duration), 1e-9)
        self.update()

    def set_configuration(self, x_variable, y_variables, display_mode="separate"):
        self.x_variable = x_variable
        self.y_variables = list(y_variables) or ["node_x:0"]
        self.display_mode = display_mode
        self.update()

    def _values(self, key):
        kind, index = parse_variable(key)
        count = len(self._times)
        if kind == "time":
            return self._times.copy()
        if not count:
            return np.array([], dtype=float)
        if kind == "path":
            node = min(max(index, 0), self._positions.shape[1] - 1)
            points = self._positions[:, node, :]
            return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        if kind.startswith("node_"):
            node = min(max(index, 0), self._positions.shape[1] - 1)
            coords = self._positions[:, node, :]
            dt = np.diff(self._times)
            safe_dt = np.where(dt > 1e-12, dt, np.nan)
            velocity = np.gradient(coords, self._times, axis=0, edge_order=1) if count > 1 else np.zeros_like(coords)
            acceleration = np.gradient(velocity, self._times, axis=0, edge_order=1) if count > 2 else np.zeros_like(coords)
            if kind == "node_x": return coords[:, 0]
            if kind == "node_y": return coords[:, 1]
            if kind == "node_vx": return velocity[:, 0]
            if kind == "node_vy": return velocity[:, 1]
            if kind == "node_speed": return np.linalg.norm(velocity, axis=1)
            if kind == "node_ax": return acceleration[:, 0]
            if kind == "node_ay": return acceleration[:, 1]
            if kind == "node_accel": return np.linalg.norm(acceleration, axis=1)
        if kind == "joint_torque":
            return self._torques[:, index] if index < self._torques.shape[1] else np.zeros(count)
        if kind.startswith("joint_") or kind == "link_angle":
            if kind == "joint_angle":
                return np.degrees(self._angles[:, index]) if index < self._angles.shape[1] else np.zeros(count)
            if kind == "link_angle":
                return np.degrees(np.cumsum(self._angles, axis=1)[:, index]) if index < self._angles.shape[1] else np.zeros(count)
            unwrapped = np.unwrap(self._angles, axis=0)
            velocity = np.gradient(unwrapped, self._times, axis=0, edge_order=1) if count > 1 else np.zeros_like(unwrapped)
            acceleration = np.gradient(velocity, self._times, axis=0, edge_order=1) if count > 2 else np.zeros_like(velocity)
            if index >= self._angles.shape[1]:
                return np.zeros(count)
            if kind == "joint_velocity": return velocity[:, index]
            if kind == "joint_accel": return acceleration[:, index]
        return np.zeros(count)

    def _x_values(self):
        return self._values(self.x_variable)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#ffffff"))
        painter.setPen(QColor("#263449"))
        painter.drawText(10, 20, self.chart_title or "仿真曲线")
        if not len(self._times):
            painter.setPen(QColor("#64748b"))
            painter.drawText(16, 44, "准备或运行仿真后显示曲线")
            painter.end()
            return
        x = self._x_values()
        series = [(key, self._values(key)) for key in self.y_variables]
        valid = [(key, values) for key, values in series if len(values) == len(x)]
        if not valid:
            painter.drawText(16, 44, "请选择有效曲线")
            painter.end()
            return
        units = [VARIABLES.get(parse_variable(key)[0], ("", "", ""))[2] for key, _ in valid]
        different = len(set(units)) > 1
        separate = self.display_mode == "separate" and different
        groups = [[item] for item in valid] if separate else [valid]
        top, bottom, left, right = 48, self.height() - 32, 52, 12
        gap = 22
        panel_h = max(50, (bottom - top - gap * (len(groups) - 1)) // len(groups))
        for group_index, group in enumerate(groups):
            area_top = top + group_index * (panel_h + gap)
            area_bottom = area_top + panel_h
            painter.setPen(QPen(QColor("#cbd5e1"), 1))
            painter.drawRect(left, area_top, max(10, self.width() - left - right), panel_h)
            all_values = np.concatenate([values[np.isfinite(values)] for _, values in group if np.any(np.isfinite(values))]) if any(np.any(np.isfinite(values)) for _, values in group) else np.array([0.0])
            xfinite = x[np.isfinite(x)]
            xmin, xmax = (float(np.min(xfinite)), float(np.max(xfinite))) if len(xfinite) else (0.0, 1.0)
            if abs(xmax - xmin) < 1e-12: xmax = xmin + 1.0
            ymin, ymax = float(np.min(all_values)), float(np.max(all_values))
            if abs(ymax - ymin) < 1e-12:
                ymin -= 0.5
                ymax += 0.5
            else:
                margin = (ymax - ymin) * 0.1
                ymin -= margin
                ymax += margin
            painter.setPen(QColor("#475569"))
            for tick in range(5):
                fraction = tick / 4.0
                tick_y = area_bottom - fraction * panel_h
                tick_value = ymin + fraction * (ymax - ymin)
                painter.drawLine(left - 4, int(tick_y), left, int(tick_y))
                painter.drawText(2, int(tick_y + 4), f"{tick_value:.3g}")
                tick_x = left + fraction * (self.width() - left - right)
                x_value = xmin + fraction * (xmax - xmin)
                painter.drawLine(int(tick_x), area_bottom, int(tick_x), area_bottom + 4)
                painter.drawText(int(tick_x - 15), area_bottom + 14, f"{x_value:.3g}")
            for curve_index, (key, values) in enumerate(group):
                color = QColor.fromHsv((curve_index * 67 + 205) % 360, 190, 190)
                points = []
                for xv, yv in zip(x, values):
                    if not np.isfinite(xv) or not np.isfinite(yv):
                        points.append(None)
                        continue
                    px = left + (xv - xmin) / (xmax - xmin) * (self.width() - left - right)
                    py = area_bottom - (yv - ymin) / (ymax - ymin) * panel_h
                    points.append((int(px), int(py)))
                painter.setPen(QPen(color, 1.8))
                previous = None
                for point in points:
                    if point is None:
                        previous = None
                    else:
                        if previous is not None:
                            painter.drawLine(previous[0], previous[1], point[0], point[1])
                        previous = point
                label = VARIABLES.get(parse_variable(key)[0], (key, "", ""))[0]
                painter.setPen(color)
                painter.drawText(4, area_top + 15 + curve_index * 14, label)
            painter.setPen(QColor("#475569"))
            x_label = VARIABLES.get(parse_variable(self.x_variable)[0], (self.x_variable, "", ""))[0]
            painter.drawText(left, min(self.height() - 4, area_bottom + 29), x_label)
        painter.end()
