"""用于绘制和拖动二维机械臂的工作区控件。"""

import math

from PySide6.QtCore import QPointF, Qt, Signal, QTimer
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

import numpy as np


class ArmWorkspace(QWidget):
    """绘制坐标网格、机械臂和目标点，并支持拖动末端目标。"""

    target_dragged = Signal(float, float)
    elbow_drag_started = Signal(float, float)
    elbow_dragged = Signal(float, float)
    elbow_drag_finished = Signal()
    object_selected = Signal(str, int)
    intermediate_joint_drag_started = Signal(int, float, float)
    intermediate_joint_dragged = Signal(int, float, float)
    intermediate_joint_drag_finished = Signal()
    path_anchor_selected = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(560, 420)
        self.setMouseTracking(True)
        self._link_lengths = np.array([1.0, 1.0], dtype=float)
        self._joint_angles = np.array([math.pi / 4, -math.pi / 2], dtype=float)
        self._target = np.array([math.sqrt(2), 0.0], dtype=float)
        self._target_status = "valid"
        self._fixed_nodes = {}
        self._zoom = 1.0
        self._view_offset = QPointF(0.0, 0.0)
        self._panning = False
        self._pan_last_position = None
        self._drag_mode = "forward"
        self._dragging_target = False
        self._dragging_elbow = False
        self._dragging_intermediate_joint = False
        self._dragged_joint_index = None
        self._drag_line_direction = None
        self._selected_object = ("joint", 0)
        self._trajectory_edit_mode = False
        self._path_point_selection_mode = False
        self._path_anchors = []
        self._path_press_position = None
        self._path_dragged = False
        self._path_preview_anchor = None
        self._selected_path_anchor = None
        self._path_edit_target = None
        self._path_edit_last_world = None
        self._pending_drag_update = None
        self._drag_update_timer = QTimer(self)
        self._drag_update_timer.setSingleShot(True)
        self._drag_update_timer.setInterval(16)
        self._drag_update_timer.timeout.connect(self._flush_pending_drag_update)

    def _queue_drag_update(self, mode, *values):
        """合并高频鼠标事件，限制数值求解频率，同时保留最新鼠标位置。"""
        self._pending_drag_update = (mode, values)
        if not self._drag_update_timer.isActive():
            self._drag_update_timer.start()

    def _flush_pending_drag_update(self):
        pending = self._pending_drag_update
        self._pending_drag_update = None
        if pending is None:
            return
        mode, values = pending
        if mode == "elbow":
            self.elbow_dragged.emit(*values)
        elif mode == "joint":
            self.intermediate_joint_dragged.emit(*values)
        elif mode == "target":
            self.target_dragged.emit(*values)

    def set_drag_mode(self, mode):
        self._drag_mode = mode
        self.update()

    def set_path_point_selection_mode(self, enabled):
        self._path_point_selection_mode = bool(enabled)
        self.update()

    def set_trajectory_edit_mode(self, enabled):
        self._trajectory_edit_mode = bool(enabled)
        self._path_press_position = None
        self._path_preview_anchor = None
        self._path_dragged = False
        self._path_edit_target = None
        self._path_edit_last_world = None
        if not self._trajectory_edit_mode:
            self._selected_path_anchor = None
        self.update()

    def path_anchors(self):
        return [
            (
                point.copy(),
                None if incoming is None else incoming.copy(),
                None if outgoing is None else outgoing.copy(),
            )
            for point, incoming, outgoing in self._path_anchors
        ]

    def clear_path(self):
        self._path_anchors.clear()
        self._path_preview_anchor = None
        self._selected_path_anchor = None
        self.update()

    def _delete_selected_path_anchor(self):
        if self._selected_path_anchor is None:
            if self._path_anchors:
                self._path_anchors.pop()
                self.update()
            return
        if 0 <= self._selected_path_anchor < len(self._path_anchors):
            self._path_anchors.pop(self._selected_path_anchor)
        self._selected_path_anchor = None
        self.update()

    def keyPressEvent(self, event):
        if self._trajectory_edit_mode and event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self._delete_selected_path_anchor()
            event.accept()
            return
        if self._trajectory_edit_mode and event.key() == Qt.Key.Key_Escape:
            self._path_press_position = None
            self._path_preview_anchor = None
            self._path_dragged = False
            self.update()
            event.accept()
            return
        super().keyPressEvent(event)

    def set_state(self, link_lengths, joint_angles, target, target_status="valid", fixed_nodes=None):
        """更新工作区显示状态。"""
        self._link_lengths = np.asarray(link_lengths, dtype=float).copy()
        self._joint_angles = np.asarray(joint_angles, dtype=float).copy()
        self._target = np.asarray(target, dtype=float).copy()
        self._target_status = target_status
        self._fixed_nodes = {
            int(index): np.asarray(point, dtype=float).copy()
            for index, point in (fixed_nodes or {}).items()
        }
        self.update()

    def _base_view_parameters(self):
        total_length = max(float(np.sum(self._link_lengths)), 0.5)
        width = max(self.width(), 1)
        height = max(self.height(), 1)
        scale = min((width - 100) / (2.4 * total_length), (height - 110) / (1.6 * total_length))
        scale = max(scale, 1.0)
        origin = QPointF(width * 0.5, height * 0.76)
        return origin, scale

    def _view_parameters(self):
        base_origin, base_scale = self._base_view_parameters()
        origin = QPointF(
            base_origin.x() + self._view_offset.x(),
            base_origin.y() + self._view_offset.y(),
        )
        return origin, base_scale * self._zoom

    def wheelEvent(self, event):
        """以鼠标所在的世界坐标为锚点缩放工作区。"""
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return

        cursor = event.position()
        old_origin, old_scale = self._view_parameters()
        anchor = self._screen_to_world(cursor, old_origin, old_scale)

        factor = 1.15 ** (delta / 120.0)
        self._zoom = min(5.0, max(0.2, self._zoom * factor))

        base_origin, base_scale = self._base_view_parameters()
        new_scale = base_scale * self._zoom
        desired_origin = QPointF(
            cursor.x() - float(anchor[0]) * new_scale,
            cursor.y() + float(anchor[1]) * new_scale,
        )
        self._view_offset = QPointF(
            desired_origin.x() - base_origin.x(),
            desired_origin.y() - base_origin.y(),
        )
        self.update()
        event.accept()

    def _world_to_screen(self, point, origin, scale):
        return QPointF(origin.x() + float(point[0]) * scale, origin.y() - float(point[1]) * scale)

    def _screen_to_world(self, point, origin, scale):
        return np.array(
            [(point.x() - origin.x()) / scale, (origin.y() - point.y()) / scale],
            dtype=float,
        )

    def _joint_positions(self):
        absolute_angles = np.cumsum(self._joint_angles)
        vectors = np.column_stack(
            (
                self._link_lengths * np.cos(absolute_angles),
                self._link_lengths * np.sin(absolute_angles),
            )
        )
        return np.vstack((np.zeros((1, 2)), np.cumsum(vectors, axis=0)))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#f7f9fc"))

        origin, scale = self._view_parameters()
        self._draw_grid(painter, origin, scale)
        self._draw_drag_line(painter, origin, scale)

        positions = self._joint_positions()
        screen_points = [self._world_to_screen(point, origin, scale) for point in positions]

        for index in range(len(screen_points) - 1):
            selected = self._selected_object == ("link", index)
            pen = QPen(QColor("#e0a13a" if selected else "#4776a8" if index == 0 else "#48a6a0"), 17 if selected else 13)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(screen_points[index], screen_points[index + 1])

        for index, point in enumerate(screen_points[:-1]):
            selected = self._selected_object == ("joint", index)
            painter.setPen(QPen(QColor("#e0a13a" if selected else "#ffffff"), 4 if selected else 3))
            painter.setBrush(QColor("#31577d" if index == 0 else "#2f8884"))
            radius = 13 if selected else 10 if index == 0 else 9
            painter.drawEllipse(point, radius, radius)

        end_point = screen_points[-1]
        painter.setPen(QPen(QColor("#ffffff"), 3))
        painter.setBrush(QColor("#f0a34a"))
        painter.drawEllipse(end_point, 10, 10)

        for index in self._fixed_nodes:
            if 0 < index < len(screen_points):
                point = screen_points[index]
                painter.setPen(QPen(QColor("#d04a4a"), 3))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(point, 16, 16)
                painter.drawLine(QPointF(point.x() - 5, point.y()), QPointF(point.x() + 5, point.y()))
                painter.drawLine(QPointF(point.x(), point.y() - 5), QPointF(point.x(), point.y() + 5))

        target_point = self._world_to_screen(self._target, origin, scale)
        target_color = {
            "valid": QColor("#2f9b75"),
            "singular": QColor("#d58a22"),
            "invalid": QColor("#d64d4d"),
        }.get(self._target_status, QColor("#d64d4d"))
        target_pen = QPen(target_color, 2, Qt.PenStyle.DashLine)
        painter.setPen(target_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(target_point, 15, 15)
        painter.drawLine(
            QPointF(target_point.x() - 7, target_point.y()),
            QPointF(target_point.x() + 7, target_point.y()),
        )
        painter.drawLine(
            QPointF(target_point.x(), target_point.y() - 7),
            QPointF(target_point.x(), target_point.y() + 7),
        )

        self._draw_trajectory_path(painter, origin, scale)

        painter.setPen(QColor("#27364b"))
        painter.drawText(18, 28, "二维机械臂工作区（坐标单位：m）")
        painter.setPen(QColor("#63738a"))
        instruction = (
            "轨迹编辑：单击添加直线锚点，按住拖动创建曲线锚点"
            if self._trajectory_edit_mode
            else (
                "拖动橙色末端移动目标；拖动第二关节可沿固定直线移动末端"
                if len(self._link_lengths) == 2
                else "拖动橙色末端移动目标；拖动中间节点可带动下游整体绕前一节点旋转"
            )
        )
        painter.drawText(18, 50, instruction)

    def _draw_trajectory_path(self, painter, origin, scale):
        anchors = list(self._path_anchors)
        if self._path_preview_anchor is not None:
            anchors.append(self._path_preview_anchor)
        if not anchors:
            return

        screen = []
        for point, incoming, outgoing in anchors:
            screen.append((
                self._world_to_screen(point, origin, scale),
                None if incoming is None else incoming,
                None if outgoing is None else outgoing,
            ))

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#111111"), 2.0))
        for index in range(len(screen) - 1):
            p0, _in0, out0 = screen[index]
            p3, in3, _out3 = screen[index + 1]
            if out0 is None and in3 is None:
                painter.drawLine(p0, p3)
                continue
            dx, dy = p3.x() - p0.x(), p3.y() - p0.y()
            c1 = QPointF(p0.x() + (out0[0] * scale if out0 is not None else dx / 3),
                         p0.y() - (out0[1] * scale if out0 is not None else -dy / 3))
            c2 = QPointF(p3.x() + (in3[0] * scale if in3 is not None else -dx / 3),
                         p3.y() - (in3[1] * scale if in3 is not None else dy / 3))
            previous = p0
            for step in range(1, 31):
                t = step / 30
                u = 1 - t
                point = QPointF(
                    u**3 * p0.x() + 3*u*u*t*c1.x() + 3*u*t*t*c2.x() + t**3*p3.x(),
                    u**3 * p0.y() + 3*u*u*t*c1.y() + 3*u*t*t*c2.y() + t**3*p3.y(),
                )
                painter.drawLine(previous, point)
                previous = point

        for index, (point, incoming, outgoing) in enumerate(screen):
            if self._trajectory_edit_mode and index == self._selected_path_anchor:
                for handle in (incoming, outgoing):
                    if handle is None:
                        continue
                    handle_point = QPointF(
                        point.x() + handle[0] * scale,
                        point.y() - handle[1] * scale,
                    )
                    painter.setPen(QPen(QColor("#c7cbd1"), 1.5, Qt.PenStyle.DashLine))
                    painter.drawLine(point, handle_point)
                    painter.setPen(QPen(QColor("#a6abb3"), 1.5))
                    painter.setBrush(QColor("#ffffff"))
                    painter.drawEllipse(handle_point, 4, 4)
            painter.setPen(QPen(QColor("#2563eb") if index == self._selected_path_anchor else QColor("#7c3aed"), 2))
            painter.setBrush(QColor("#ffffff"))
            painter.drawEllipse(point, 5, 5)

    def _draw_drag_line(self, painter, origin, scale):
        """拖动肘关节时显示末端被限制运动的固定直线。"""
        if self._drag_line_direction is None:
            return
        direction = self._drag_line_direction
        extent = max(self.width(), self.height()) / max(scale, 1.0) * 1.5
        start = self._world_to_screen(-extent * direction, origin, scale)
        end = self._world_to_screen(extent * direction, origin, scale)
        pen = QPen(QColor("#d58a22"), 1.5, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(start, end)

    def set_drag_line_direction(self, direction):
        """设置或清除肘关节拖动时使用的固定直线方向。"""
        if direction is None:
            self._drag_line_direction = None
        else:
            vector = np.asarray(direction, dtype=float)
            norm = float(np.linalg.norm(vector))
            self._drag_line_direction = vector / norm if norm > 1e-12 else None
        self.update()

    def _draw_grid(self, painter, origin, scale):
        grid_pen = QPen(QColor("#e3e8f0"), 1)
        painter.setPen(grid_pen)
        spacing = max(25, scale * 0.5)
        x = origin.x() % spacing
        while x < self.width():
            painter.drawLine(QPointF(x, 0), QPointF(x, self.height()))
            x += spacing
        y = origin.y() % spacing
        while y < self.height():
            painter.drawLine(QPointF(0, y), QPointF(self.width(), y))
            y += spacing

        axis_pen = QPen(QColor("#8998aa"), 1.5)
        painter.setPen(axis_pen)
        painter.drawLine(QPointF(0, origin.y()), QPointF(self.width(), origin.y()))
        painter.drawLine(QPointF(origin.x(), 0), QPointF(origin.x(), self.height()))
        painter.setPen(QColor("#63738a"))
        painter.drawText(int(self.width() - 24), int(origin.y() - 8), "x")
        painter.drawText(int(origin.x() + 8), 18, "y")
        painter.drawText(int(origin.x() + 8), int(origin.y() + 18), "0")

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        if self._path_point_selection_mode and not self._trajectory_edit_mode:
            cursor = event.position()
            origin, scale = self._view_parameters()
            for index, (point, _incoming, _outgoing) in enumerate(self._path_anchors):
                anchor_screen = self._world_to_screen(point, origin, scale)
                if math.hypot(cursor.x() - anchor_screen.x(), cursor.y() - anchor_screen.y()) <= 12:
                    self.path_anchor_selected.emit(index)
                    event.accept()
                    return
            event.accept()
            return
        if self._trajectory_edit_mode:
            self.setFocus(Qt.FocusReason.MouseFocusReason)
            cursor = event.position()
            origin, scale = self._view_parameters()
            world_cursor = self._screen_to_world(cursor, origin, scale)

            for index, (point, incoming, outgoing) in enumerate(self._path_anchors):
                anchor_screen = self._world_to_screen(point, origin, scale)
                for name, handle in (("in", incoming), ("out", outgoing)):
                    if handle is None:
                        continue
                    handle_screen = QPointF(
                        anchor_screen.x() + handle[0] * scale,
                        anchor_screen.y() - handle[1] * scale,
                    )
                    if math.hypot(cursor.x() - handle_screen.x(), cursor.y() - handle_screen.y()) <= 9:
                        self._selected_path_anchor = index
                        self._path_edit_target = ("handle", index, name)
                        self._path_edit_last_world = world_cursor
                        self.update()
                        event.accept()
                        return
                if math.hypot(cursor.x() - anchor_screen.x(), cursor.y() - anchor_screen.y()) <= 10:
                    self._selected_path_anchor = index
                    self._path_edit_target = ("anchor", index)
                    self._path_edit_last_world = world_cursor
                    self.update()
                    event.accept()
                    return

            self._selected_path_anchor = None
            self._path_edit_target = None
            self._path_press_position = event.position()
            self._path_dragged = False
            self._path_preview_anchor = None
            event.accept()
            return

        origin, scale = self._view_parameters()
        positions = self._joint_positions()
        elbow_point = self._world_to_screen(positions[1], origin, scale)
        end_point = self._world_to_screen(positions[-1], origin, scale)
        target_point = self._world_to_screen(self._target, origin, scale)
        cursor = event.position()
        self.object_selected.emit("none", -1)

        if self._drag_mode == "forward" and math.hypot(
            cursor.x() - end_point.x(), cursor.y() - end_point.y()
        ) <= 24:
            self._selected_object = ("joint", len(positions) - 1)
            self.object_selected.emit("joint", len(positions) - 1)
            self.update()
            self._dragging_intermediate_joint = True
            self._dragged_joint_index = len(positions) - 1
            world = self._screen_to_world(cursor, origin, scale)
            self.intermediate_joint_drag_started.emit(
                self._dragged_joint_index, float(world[0]), float(world[1])
            )
            event.accept()
            return

        if math.hypot(cursor.x() - target_point.x(), cursor.y() - target_point.y()) <= 24 or math.hypot(
            cursor.x() - end_point.x(), cursor.y() - end_point.y()
        ) <= 24:
            self._dragging_target = True
            self.target_dragged.emit(float(self._target[0]), float(self._target[1]))
            event.accept()
            return

        if len(self._link_lengths) == 2 and math.hypot(cursor.x() - elbow_point.x(), cursor.y() - elbow_point.y()) <= 20:
            self._selected_object = ("joint", 1)
            self.object_selected.emit("joint", 1)
            self.update()
            self._dragging_elbow = True
            self.elbow_drag_started.emit(float(self._target[0]), float(self._target[1]))
            event.accept()
            return

        if len(self._link_lengths) > 2:
            for index, point in enumerate(positions[1:-1], start=1):
                point_screen = self._world_to_screen(point, origin, scale)
                if math.hypot(cursor.x() - point_screen.x(), cursor.y() - point_screen.y()) <= 20:
                    self._selected_object = ("joint", index)
                    self.object_selected.emit("joint", index)
                    self.update()
                    self._dragging_intermediate_joint = True
                    self._dragged_joint_index = index
                    world = self._screen_to_world(cursor, origin, scale)
                    self.intermediate_joint_drag_started.emit(index, float(world[0]), float(world[1]))
                    event.accept()
                    return

        for index, point in enumerate(positions[:-1]):
            point_screen = self._world_to_screen(point, origin, scale)
            if math.hypot(cursor.x() - point_screen.x(), cursor.y() - point_screen.y()) <= 16:
                self._selected_object = ("joint", index)
                self.object_selected.emit("joint", index)
                self.update()
                event.accept()
                return

        for index in range(len(screen_points := [self._world_to_screen(point, origin, scale) for point in positions]) - 1):
            start, end = screen_points[index], screen_points[index + 1]
            dx, dy = end.x() - start.x(), end.y() - start.y()
            length_sq = dx * dx + dy * dy
            if length_sq <= 1e-12:
                continue
            t = max(0.0, min(1.0, ((cursor.x() - start.x()) * dx + (cursor.y() - start.y()) * dy) / length_sq))
            nearest_x, nearest_y = start.x() + t * dx, start.y() + t * dy
            if math.hypot(cursor.x() - nearest_x, cursor.y() - nearest_y) <= 10:
                self._selected_object = ("link", index)
                self.object_selected.emit("link", index)
                self.update()
                event.accept()
                return

        if self._drag_mode == "forward" and math.hypot(
            cursor.x() - end_point.x(), cursor.y() - end_point.y()
        ) <= 24:
            self._selected_object = ("joint", len(positions) - 1)
            self.object_selected.emit("joint", len(positions) - 1)
            self.update()
            self._dragging_intermediate_joint = True
            self._dragged_joint_index = len(positions) - 1
            world = self._screen_to_world(cursor, origin, scale)
            self.intermediate_joint_drag_started.emit(
                self._dragged_joint_index, float(world[0]), float(world[1])
            )
            event.accept()
            return

        if math.hypot(cursor.x() - target_point.x(), cursor.y() - target_point.y()) <= 24 or math.hypot(
            cursor.x() - end_point.x(), cursor.y() - end_point.y()
        ) <= 24:
            self._dragging_target = True
            self.target_dragged.emit(float(self._target[0]), float(self._target[1]))
            event.accept()
            return

        self._panning = True
        self._pan_last_position = cursor
        event.accept()

    def mouseMoveEvent(self, event):
        if self._panning and self._pan_last_position is not None:
            cursor = event.position()
            delta = cursor - self._pan_last_position
            self._view_offset = QPointF(
                self._view_offset.x() + delta.x(),
                self._view_offset.y() + delta.y(),
            )
            self._pan_last_position = cursor
            self.update()
            event.accept()
            return
        if self._trajectory_edit_mode and self._path_edit_target is not None:
            cursor = self._screen_to_world(event.position(), *self._view_parameters())
            mode, index, *extra = self._path_edit_target
            point, incoming, outgoing = self._path_anchors[index]
            if mode == "anchor":
                delta = cursor - self._path_edit_last_world
                point = point + delta
                self._path_anchors[index] = (point, incoming, outgoing)
            else:
                vector = cursor - point
                if extra[0] == "in":
                    incoming = vector
                else:
                    outgoing = vector
                self._path_anchors[index] = (point, incoming, outgoing)
            self._path_edit_last_world = cursor
            self.update()
            event.accept()
            return
        if self._trajectory_edit_mode and self._path_press_position is not None:
            cursor = event.position()
            distance = math.hypot(cursor.x() - self._path_press_position.x(), cursor.y() - self._path_press_position.y())
            if distance > 5:
                self._path_dragged = True
            origin, scale = self._view_parameters()
            point = self._screen_to_world(self._path_press_position, origin, scale)
            incoming = outgoing = None
            if self._path_dragged:
                current = self._screen_to_world(cursor, origin, scale)
                outgoing = current - point
                incoming = -outgoing
            self._path_preview_anchor = (point, incoming, outgoing)
            self.update()
            event.accept()
            return
        if self._dragging_elbow:
            origin, scale = self._view_parameters()
            position = self._screen_to_world(event.position(), origin, scale)
            self._queue_drag_update("elbow", float(position[0]), float(position[1]))
            event.accept()
            return
        if self._dragging_intermediate_joint and self._dragged_joint_index is not None:
            origin, scale = self._view_parameters()
            position = self._screen_to_world(event.position(), origin, scale)
            self._queue_drag_update(
                "joint", self._dragged_joint_index, float(position[0]), float(position[1])
            )
            event.accept()
            return
        if self._dragging_target:
            origin, scale = self._view_parameters()
            target = self._screen_to_world(event.position(), origin, scale)
            self._queue_drag_update("target", float(target[0]), float(target[1]))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._panning:
            self._panning = False
            self._pan_last_position = None
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._trajectory_edit_mode and self._path_edit_target is not None:
            self._path_edit_target = None
            self._path_edit_last_world = None
            self.update()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._trajectory_edit_mode and self._path_press_position is not None:
            origin, scale = self._view_parameters()
            point = self._screen_to_world(self._path_press_position, origin, scale)
            incoming = outgoing = None
            if self._path_dragged:
                current = self._screen_to_world(event.position(), origin, scale)
                outgoing = current - point
                incoming = -outgoing
            self._path_anchors.append((point, incoming, outgoing))
            self._selected_path_anchor = len(self._path_anchors) - 1
            self._path_press_position = None
            self._path_preview_anchor = None
            self._path_dragged = False
            self.update()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            if self._drag_update_timer.isActive():
                self._drag_update_timer.stop()
            self._flush_pending_drag_update()
            self._dragging_target = False
            if self._dragging_elbow:
                self._dragging_elbow = False
                self.set_drag_line_direction(None)
                self.elbow_drag_finished.emit()
            if self._dragging_intermediate_joint:
                self._dragging_intermediate_joint = False
                self._dragged_joint_index = None
                self.intermediate_joint_drag_finished.emit()
        super().mouseReleaseEvent(event)
