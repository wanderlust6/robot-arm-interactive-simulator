"""Configurable simulation plots and axis-selection dialogs."""

import numpy as np
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout,
    QLabel, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget,
)
from scipy.signal import savgol_filter


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
    "min_singular_value": ("\u6700\u5c0f\u5947\u5f02\u503c \u03c3min (m)", "\u5947\u5f02\u6027\u8bca\u65ad", "m"),
    "condition_number": ("雅可比条件数 κ", "奇异性诊断", ""),
    "endpoint_error": ("末端跟踪误差 (m)", "误差", "m"),
    "scheme_endpoint_error": ("方案末端跟踪误差 (m)", "方案诊断", "m"),
    "scheme_min_singular_value": ("方案最小奇异值", "方案诊断", "m"),
    "scheme_condition_number": ("方案雅可比条件数", "方案诊断", ""),
}


def variable_key(kind, index=0):
    return f"{kind}:{int(index)}"


def parse_variable(key):
    kind, _, index = key.partition(":")
    return kind, int(index or 0)


def decimate_by_pixel(points, column_count, minimum_points_per_column=2.0):
    """按像素列对折线做 min-max 抽取，保留每列的极值。

    ``points`` 中允许 ``None`` 表示断点。返回 ``(抽取后的点列, 是否发生抽取)``。
    同一像素列内的多个点只保留该列的**最小值与最大值**（按原始顺序），因此曲线
    的极值包络完全不变，没有任何数值被修改；只是不再把上千个点硬塞进几百个像素。
    点数不超过 ``minimum_points_per_column × column_count`` 时原样返回。
    """
    if column_count <= 0 or not points:
        return list(points), False
    usable = sum(1 for point in points if point is not None)
    if usable <= minimum_points_per_column * column_count:
        return list(points), False

    reduced = []
    bucket = []
    current_column = None

    def flush():
        if not bucket:
            return
        if len(bucket) == 1:
            reduced.append(bucket[0])
            return
        low = high = 0
        for index, point in enumerate(bucket):
            if point.y() < bucket[low].y():
                low = index
            if point.y() > bucket[high].y():
                high = index
        first, second = (low, high) if low <= high else (high, low)
        reduced.append(bucket[first])
        if second != first:
            reduced.append(bucket[second])

    for point in points:
        if point is None:
            flush()
            bucket = []
            current_column = None
            reduced.append(None)
            continue
        column = int(point.x() // 1)
        if current_column is None or column == current_column:
            bucket.append(point)
            current_column = column
        else:
            flush()
            bucket = [point]
            current_column = column
    flush()

    # 保证首尾点精确保留：它们是曲线的起止值，不应因为不在极值上而被丢弃。
    original = [point for point in points if point is not None]
    if original:
        first_index = next(i for i, point in enumerate(reduced) if point is not None)
        if reduced[first_index] is not original[0]:
            reduced.insert(first_index, original[0])
        last_index = next(i for i in range(len(reduced) - 1, -1, -1) if reduced[i] is not None)
        if reduced[last_index] is not original[-1]:
            reduced.insert(last_index + 1, original[-1])
    return reduced, True


def smooth_series(values, window=11, polyorder=3):
    """用于**绘图显示**的零相位平滑（Savitzky-Golay）。

    只应作用在绘制路径上：数据表、CSV 导出与可执行性检查必须使用原始值。
    选择 Savitzky-Golay 而不是滑动平均：它用局部多项式拟合，能更好地保留峰形；
    且是零相位滤波，不会让曲线在时间上左右平移。

    - ``window`` 为窗口点数，会被强制成不小于 ``polyorder + 2`` 的奇数。
    - 含 NaN 的序列先按有限样本线性插值再滤波，滤波后恢复原 NaN 位置，保持断点。
    - 点数不足或参数无效时原样返回，不做任何改动。
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 5:
        return values
    usable = np.isfinite(values)
    if np.count_nonzero(usable) < 5:
        return values
    points = int(window)
    if points % 2 == 0:
        points += 1
    points = min(points, values.size if values.size % 2 == 1 else values.size - 1)
    if points <= polyorder + 1:
        return values
    order = min(int(polyorder), points - 2)

    if np.all(usable):
        return savgol_filter(values, points, order, mode="interp")
    filled = values.copy()
    filled[~usable] = np.interp(
        np.flatnonzero(~usable), np.flatnonzero(usable), values[usable]
    )
    filtered = savgol_filter(filled, points, order, mode="interp")
    return np.where(usable, filtered, np.nan)


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
        self.setMouseTracking(True)
        self._drawn_points = []
        self._selected_point = None
        # 本次绘制是否发生了按像素列的极值抽取（只影响绘制，不影响数值）。
        self._display_decimated = False
        # 显示平滑：默认关闭，只影响绘制路径，不改变任何被记录或被导出的数值。
        self._smoothing_enabled = False
        self._smoothing_window = 11
        self._smoothing_polyorder = 3
        self._display_smoothed = False
        self._min_singular_values = np.empty(0, dtype=float)
        self._condition_numbers = np.empty(0, dtype=float)
        self._endpoint_errors = np.empty(0, dtype=float)
        self._scheme_series = []
        self._scheme_states = {}
        self.scheme_key = "continuous"
        self.x_scheme_key = "continuous"
        self.y_scheme_keys = []

    def set_data(self, times, positions, angles, current_time, duration, torques=None, joint_velocities=None, joint_accelerations=None, min_singular_values=None, condition_numbers=None, endpoint_errors=None, scheme_series=None, scheme_states=None, node_velocities=None, node_accelerations=None):
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
        velocity_values = np.asarray(joint_velocities, dtype=float) if joint_velocities is not None else np.empty((0, 0), dtype=float)
        acceleration_values = np.asarray(joint_accelerations, dtype=float) if joint_accelerations is not None else np.empty((0, 0), dtype=float)
        self._joint_velocities = velocity_values.copy() if velocity_values.size else np.empty((0, 0), dtype=float)
        self._joint_accelerations = acceleration_values.copy() if acceleration_values.size else np.empty((0, 0), dtype=float)
        self._min_singular_values = np.asarray(min_singular_values, dtype=float).copy() if min_singular_values is not None else np.empty(0, dtype=float)
        self._condition_numbers = np.asarray(condition_numbers, dtype=float).copy() if condition_numbers is not None else np.empty(0, dtype=float)
        self._endpoint_errors = np.asarray(endpoint_errors, dtype=float).copy() if endpoint_errors is not None else np.empty(0, dtype=float)
        self._scheme_series = list(scheme_series or [])
        self._scheme_states = dict(scheme_states or {})
        # 节点速度/加速度的解析值，形状 (采样数, 节点数, 2)；缺失时退回数值微分。
        node_velocity_values = (
            np.asarray(node_velocities, dtype=float)
            if node_velocities is not None else np.empty((0, 0, 2), dtype=float)
        )
        node_acceleration_values = (
            np.asarray(node_accelerations, dtype=float)
            if node_accelerations is not None else np.empty((0, 0, 2), dtype=float)
        )
        self._node_velocities = node_velocity_values
        self._node_accelerations = node_acceleration_values
        self._current_time = max(0.0, float(current_time))
        self._duration = max(float(duration), 1e-9)
        self.update()

    def set_configuration(self, x_variable, y_variables, display_mode="separate", x_scheme_key="continuous", y_scheme_keys=None):
        self.x_variable = x_variable
        self.y_variables = list(y_variables) or ["node_x:0"]
        self.display_mode = display_mode
        self.x_scheme_key = x_scheme_key or "continuous"
        self.y_scheme_keys = list(y_scheme_keys or ["continuous"] * len(self.y_variables))
        while len(self.y_scheme_keys) < len(self.y_variables):
            self.y_scheme_keys.append("continuous")
        self.scheme_key = self.x_scheme_key
        self.update()

    def _values(self, key, scheme_key=None, smooth=False):
        """曲线取值。

        ``smooth=True`` 仅供**绘制**使用：启用显示平滑时对序列做零相位
        Savitzky-Golay 滤波。默认 ``False``，因此数据表格、CSV 导出、
        可执行性检查与「诊断当前列」拿到的始终是原始数值。
        """
        values = self._raw_values(key, scheme_key)
        if smooth and self._smoothing_enabled:
            return smooth_series(values, self._smoothing_window, self._smoothing_polyorder)
        return values

    def _raw_values(self, key, scheme_key=None):
        kind, index = parse_variable(key)
        count = len(self._times)
        if kind == "time":
            return self._times.copy()
        if not count: return np.array([], dtype=float)
        selected_scheme = scheme_key or self.scheme_key
        state = self._scheme_states.get(selected_scheme, {})
        sp = np.asarray(state.get("positions", []), dtype=float)
        sa = np.asarray(state.get("angles", []), dtype=float)
        if len(sp) == count and len(sa) == count:
            if kind == "path":
                node=min(max(index,0),sp.shape[1]-1); pts=sp[:,node,:]
                return np.r_[0.0,np.cumsum(np.linalg.norm(np.diff(pts,axis=0),axis=1))]
            if kind.startswith("node_"):
                node=min(max(index,0),sp.shape[1]-1); coords=sp[:,node,:]
                vel, acc = self._node_kinematics(node, coords, count)
                if kind=="node_x": return coords[:,0]
                if kind=="node_y": return coords[:,1]
                if kind=="node_vx": return vel[:,0]
                if kind=="node_vy": return vel[:,1]
                if kind=="node_speed": return np.linalg.norm(vel,axis=1)
                if kind=="node_ax": return acc[:,0]
                if kind=="node_ay": return acc[:,1]
                if kind=="node_accel": return np.linalg.norm(acc,axis=1)
            if kind=="joint_angle": return np.degrees(sa[:,index]) if index<sa.shape[1] else np.zeros(count)
            if kind=="link_angle": return np.degrees(np.cumsum(sa,axis=1)[:,index]) if index<sa.shape[1] else np.zeros(count)
            if kind in ("joint_velocity", "joint_accel"):
                if index >= sa.shape[1]:
                    return np.zeros(count)
                # 优先使用该方案样条的解析导数。只有缺失时才退回数值微分：
                # 对记录角度做两次差分会被墙钟采样的时间抖动放大成加速度噪声。
                stored = state.get("velocities" if kind == "joint_velocity" else "accelerations")
                if stored is not None:
                    stored = np.asarray(stored, dtype=float)
                    if stored.shape == sa.shape and np.any(np.isfinite(stored[:, index])):
                        return stored[:, index]
                unwrapped = np.unwrap(sa, axis=0)
                scheme_velocity = np.gradient(unwrapped, self._times, axis=0, edge_order=1) if count > 1 else np.zeros_like(unwrapped)
                if kind == "joint_velocity":
                    return scheme_velocity[:, index]
                scheme_acceleration = np.gradient(scheme_velocity, self._times, axis=0, edge_order=1) if count > 2 else np.zeros_like(unwrapped)
                return scheme_acceleration[:, index]
        if kind == "path":
            node = min(max(index, 0), self._positions.shape[1] - 1)
            points = self._positions[:, node, :]
            return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        if kind.startswith("node_"):
            node = min(max(index, 0), self._positions.shape[1] - 1)
            coords = self._positions[:, node, :]
            velocity, acceleration = self._node_kinematics(node, coords, count)
            if kind == "node_x": return coords[:, 0]
            if kind == "node_y": return coords[:, 1]
            if kind == "node_vx": return velocity[:, 0]
            if kind == "node_vy": return velocity[:, 1]
            if kind == "node_speed": return np.linalg.norm(velocity, axis=1)
            if kind == "node_ax": return acceleration[:, 0]
            if kind == "node_ay": return acceleration[:, 1]
            if kind == "node_accel": return np.linalg.norm(acceleration, axis=1)
        if kind == "min_singular_value":
            return self._min_singular_values.copy() if len(self._min_singular_values) == count else np.zeros(count)
        if kind == "condition_number":
            return self._condition_numbers.copy() if len(self._condition_numbers) == count else np.zeros(count)
        if kind == "endpoint_error":
            return self._endpoint_errors.copy() if len(self._endpoint_errors) == count else np.zeros(count)
        if kind in ("scheme_endpoint_error", "scheme_min_singular_value", "scheme_condition_number"):
            if index >= len(self._scheme_series):
                return np.full(count, np.nan)
            field = {
                "scheme_endpoint_error": "endpoint_errors",
                "scheme_min_singular_value": "min_singular_values",
                "scheme_condition_number": "condition_numbers",
            }[kind]
            values = np.asarray(self._scheme_series[index].get(field, []), dtype=float)
            return values.copy() if len(values) == count else np.full(count, np.nan)
        if kind == "joint_torque":
            return self._torques[:, index] if index < self._torques.shape[1] else np.zeros(count)
        if kind.startswith("joint_") or kind == "link_angle":
            if kind == "joint_angle":
                return np.degrees(self._angles[:, index]) if index < self._angles.shape[1] else np.zeros(count)
            if kind == "link_angle":
                return np.degrees(np.cumsum(self._angles, axis=1)[:, index]) if index < self._angles.shape[1] else np.zeros(count)
            if index >= self._angles.shape[1]:
                return np.zeros(count)
            if kind == "joint_velocity":
                if self._joint_velocities.shape == self._angles.shape:
                    return self._joint_velocities[:, index]
                unwrapped = np.unwrap(self._angles, axis=0)
                return np.gradient(unwrapped, self._times, axis=0, edge_order=1) if count > 1 else np.zeros(count)
            if kind == "joint_accel":
                if self._joint_accelerations.shape == self._angles.shape:
                    return self._joint_accelerations[:, index]
                unwrapped = np.unwrap(self._angles, axis=0)
                velocity = np.gradient(unwrapped, self._times, axis=0, edge_order=1) if count > 1 else np.zeros_like(unwrapped)
                return np.gradient(velocity, self._times, axis=0, edge_order=1) if count > 2 else np.zeros(count)
        return np.zeros(count)

    def _node_kinematics(self, node, coords, count):
        """返回某节点的速度与加速度。

        优先使用规划样条导数经雅可比求得的**解析值**（由 set_data 传入）。
        只有缺失时才退回对位置序列做数值微分——那会把采样上的微小不一致
        放大成可见抖动，而且对"速度大小/加速度大小"这类模长量尤其明显。
        """
        velocities = np.asarray(self._node_velocities, dtype=float)
        accelerations = np.asarray(self._node_accelerations, dtype=float)
        if (
            velocities.ndim == 3
            and velocities.shape[0] == count
            and node < velocities.shape[1]
            and accelerations.shape == velocities.shape
        ):
            return velocities[:, node, :], accelerations[:, node, :]
        velocity = np.gradient(coords, self._times, axis=0, edge_order=1) if count > 1 else np.zeros_like(coords)
        acceleration = np.gradient(velocity, self._times, axis=0, edge_order=1) if count > 2 else np.zeros_like(coords)
        return velocity, acceleration

    def set_smoothing(self, enabled, window=11, polyorder=3):
        """设置"仅用于绘图"的显示平滑。数值与导出数据不受影响。"""
        self._smoothing_enabled = bool(enabled)
        self._smoothing_window = int(window)
        self._smoothing_polyorder = int(polyorder)
        self.update()

    def smoothing_enabled(self):
        return bool(self._smoothing_enabled)

    def _x_values(self, scheme_key=None):
        return self._values(self.x_variable, scheme_key or self.x_scheme_key)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._drawn_points:
            mx, my = event.position().x(), event.position().y()
            nearest = min(
                self._drawn_points,
                key=lambda point: (point[0] - mx) ** 2 + (point[1] - my) ** 2,
            )
            distance_squared = (nearest[0] - mx) ** 2 + (nearest[1] - my) ** 2
            if distance_squared <= 324:
                self._selected_point = nearest
                self.update()
                event.accept()
                return
        super().mousePressEvent(event)

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
        x = self._x_values(self.x_scheme_key)
        self._drawn_points = []
        self._display_decimated = False
        self._display_smoothed = bool(self._smoothing_enabled)
        series = [
            (key, self._values(key, scheme, smooth=True), scheme)
            for key, scheme in zip(self.y_variables, self.y_scheme_keys)
        ]
        valid = [(key, values, scheme) for key, values, scheme in series if len(values) == len(x)]
        if not valid:
            painter.drawText(16, 44, "请选择有效曲线")
            painter.end()
            return
        units = [VARIABLES.get(parse_variable(key)[0], ("", "", ""))[2] for key, _, _scheme in valid]
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
            all_values = np.concatenate([values[np.isfinite(values)] for _, values, _scheme in group if np.any(np.isfinite(values))]) if any(np.any(np.isfinite(values)) for _, values, _scheme in group) else np.array([0.0])
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
            for curve_index, (key, values, scheme) in enumerate(group):
                color = QColor.fromHsv((curve_index * 67 + 205) % 360, 190, 190)
                points = []
                for xv, yv in zip(x, values):
                    if not np.isfinite(xv) or not np.isfinite(yv):
                        points.append(None)
                        continue
                    px = left + (xv - xmin) / (xmax - xmin) * (self.width() - left - right)
                    py = area_bottom - (yv - ymin) / (ymax - ymin) * panel_h
                    # 用浮点坐标绘制：每个采样点只占 1–2 像素时，整数取整会在相邻点
                    # 之间产生 ±1 px 抖动，在陡峭段表现为锯齿，看起来像"数据在抖"。
                    pixel_point = QPointF(px, py)
                    points.append(pixel_point)
                    self._drawn_points.append((px, py, float(xv), float(yv), key))
                painter.setPen(QPen(color, 1.8))
                # 采样点多于像素列时按列做 min-max 抽取：每列保留该列内的最小与
                # 最大值，包络和峰值完整保留，所有数值一个都不改，只是不再把上千个
                # 点硬挤进几百像素（那会造成视觉上的锯齿）。
                area_width = int(max(1, round(self.width() - left - right)))
                drawn_points, decimated = decimate_by_pixel(points, area_width)
                if decimated:
                    self._display_decimated = True
                previous = None
                for point in drawn_points:
                    if point is None:
                        previous = None
                    else:
                        if previous is not None:
                            painter.drawLine(previous, point)
                        previous = point
                kind, series_index = parse_variable(key)
                label = VARIABLES.get(kind, (key, "", ""))[0]
                if kind.startswith("scheme_") and series_index < len(self._scheme_series):
                    label = f"{self._scheme_series[series_index].get('label', '方案')} · {label}"
                painter.setPen(color)
                scheme_label = next((item.get("label", scheme) for item in self._scheme_series if item.get("key") == scheme), scheme)
                label = f"{scheme_label} · {label}"
                painter.drawText(4, area_top + 15 + curve_index * 14, label)
            if len(self._times) > 0 and len(x) == len(self._times):
                finite_pairs = np.isfinite(self._times) & np.isfinite(x)
                if np.count_nonzero(finite_pairs) >= 2:
                    marker_x_value = float(np.interp(
                        self._current_time,
                        self._times[finite_pairs],
                        x[finite_pairs],
                    ))
                    if xmin <= marker_x_value <= xmax:
                        marker_x = int(
                            left + (marker_x_value - xmin) / (xmax - xmin)
                            * (self.width() - left - right)
                        )
                        painter.setPen(QPen(
                            QColor("#dc2626"), 1.5, Qt.PenStyle.DashLine
                        ))
                        painter.drawLine(marker_x, area_top, marker_x, area_bottom)
            painter.setPen(QColor("#475569"))
            x_label = VARIABLES.get(parse_variable(self.x_variable)[0], (self.x_variable, "", ""))[0]
            painter.drawText(left, min(self.height() - 4, area_bottom + 29), x_label)
        if self._display_smoothed and self.width() > 520:
            painter.setPen(QColor("#b45309"))
            painter.drawText(
                self.width() - 330, 20,
                f"显示已平滑（Savitzky-Golay，{self._smoothing_window} 点；数值未改）",
            )
        if self._display_decimated and self.width() > 520:
            # 明确标注：只是绘制时按像素列抽取极值，数值、峰值与导出数据均未改变。
            painter.setPen(QColor("#94a3b8"))
            painter.drawText(
                self.width() - 260, 20,
                "显示：按像素列抽取极值（数值未改）",
            )
        if self._selected_point is not None:
            sx, sy, selected_x, selected_y, selected_key = self._selected_point
            painter.setPen(QPen(QColor("#dc2626"), 2))
            painter.drawEllipse(sx - 5, sy - 5, 10, 10)
            x_name = VARIABLES.get(parse_variable(self.x_variable)[0], (self.x_variable, "", ""))[0]
            y_name = VARIABLES.get(parse_variable(selected_key)[0], (selected_key, "", ""))[0]
            coordinate_text = f"{x_name}: {selected_x:.6g}    {y_name}: {selected_y:.6g}"
            text_rect = painter.fontMetrics().boundingRect(coordinate_text).adjusted(-8, -5, 8, 5)
            text_rect.moveTopRight(self.rect().topRight() + QPoint(-8, 28))
            painter.fillRect(text_rect, QColor("#fff7ed"))
            painter.setPen(QPen(QColor("#b91c1c"), 1))
            painter.drawRect(text_rect)
            painter.drawText(text_rect.adjusted(8, 5, -8, -5), Qt.AlignmentFlag.AlignVCenter, coordinate_text)
        painter.end()
