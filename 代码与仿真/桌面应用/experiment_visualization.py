import csv
import json
import math
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen, QFont
from PySide6.QtWidgets import (
    QComboBox, QGroupBox, QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget,
)


class WorkspaceView(QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.title = title
        self.links = (0.6, 0.4)
        self.target_points = []
        self.q = [0.0, 0.0]
        self.current_index = 0
        self.failed = False
        self.setMinimumSize(260, 250)

    def set_sequence(self, points, q, index, failed=False):
        self.target_points = points
        self.q = q
        self.current_index = index
        self.failed = failed
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#fbfcfe"))
        painter.setPen(QColor("#263445"))
        painter.setFont(QFont("Microsoft YaHei UI", 10, QFont.Weight.DemiBold))
        painter.drawText(12, 22, self.title)
        w, h = self.width(), self.height()
        scale = min((w - 36) / 2.2, (h - 62) / 2.2)
        ox, oy = w / 2, h * 0.57
        def screen(pt):
            return (ox + pt[0] * scale, oy - pt[1] * scale)
        outer = self.links[0] + self.links[1]
        inner = abs(self.links[0] - self.links[1])
        painter.setPen(QPen(QColor("#d8e0ea"), 1, Qt.PenStyle.DashLine))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        x, y = screen((0, 0))
        painter.drawEllipse(int(x - outer * scale), int(y - outer * scale), int(2 * outer * scale), int(2 * outer * scale))
        painter.drawEllipse(int(x - inner * scale), int(y - inner * scale), int(2 * inner * scale), int(2 * inner * scale))
        painter.setPen(QPen(QColor("#b7c4d4"), 1))
        painter.drawLine(int(ox - outer * scale), int(oy), int(ox + outer * scale), int(oy))
        painter.drawLine(int(ox), int(oy - outer * scale), int(ox), int(oy + outer * scale))
        if self.target_points:
            painter.setPen(QPen(QColor("#6b7280"), 1.5, Qt.PenStyle.DashLine))
            poly = [screen(p) for p in self.target_points]
            for a, b in zip(poly, poly[1:]):
                painter.drawLine(int(a[0]), int(a[1]), int(b[0]), int(b[1]))
            tx, ty = screen(self.target_points[self.current_index])
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#dc2626"))
            painter.drawEllipse(int(tx - 4), int(ty - 4), 8, 8)
        q1, q2 = self.q
        p0 = screen((0, 0))
        p1 = screen((self.links[0] * math.cos(q1), self.links[0] * math.sin(q1)))
        p2 = screen((self.links[0] * math.cos(q1) + self.links[1] * math.cos(q1 + q2),
                     self.links[0] * math.sin(q1) + self.links[1] * math.sin(q1 + q2)))
        painter.setPen(QPen(QColor("#2563eb"), 5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(int(p0[0]), int(p0[1]), int(p1[0]), int(p1[1]))
        painter.setPen(QPen(QColor("#0f766e"), 5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#1d4ed8"))
        painter.drawEllipse(int(p0[0] - 5), int(p0[1] - 5), 10, 10)
        painter.setBrush(QColor("#0f766e"))
        painter.drawEllipse(int(p1[0] - 5), int(p1[1] - 5), 10, 10)
        painter.setBrush(QColor("#dc2626") if self.failed else QColor("#15803d"))
        painter.drawEllipse(int(p2[0] - 5), int(p2[1] - 5), 10, 10)
        painter.setPen(QColor("#475569"))
        painter.setFont(QFont("Microsoft YaHei UI", 8))
        painter.drawText(10, h - 10, "虚线：目标路径  |  圆环：工作空间边界")


class ErrorCurveView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.series = []
        self.index = 0
        self.setMinimumHeight(180)

    def set_series(self, series, index):
        self.series = series
        self.index = index
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#ffffff"))
        w, h = self.width(), self.height()
        left, right, top, bottom = 62, w - 18, 18, h - 42
        painter.setPen(QColor("#263445"))
        painter.setFont(QFont("Microsoft YaHei UI", 9, QFont.Weight.DemiBold))
        painter.drawText(12, 15, "末端实际误差（对数刻度）")
        if not self.series:
            painter.drawText(20, 50, "运行实验后显示逐点误差曲线")
            return
        vals = [max(float(v), 1e-12) for s in self.series for v in s["errors"] if math.isfinite(float(v))]
        if not vals:
            return
        lo, hi = min(vals), max(vals)
        log_lo, log_hi = math.log10(lo), math.log10(hi)
        if log_hi - log_lo < 0.5:
            log_lo -= 0.5
            log_hi += 0.5
        def xy(i, val, count):
            x = left + (right - left) * (i / max(count - 1, 1))
            y = bottom - (math.log10(max(float(val), 1e-12)) - log_lo) / (log_hi - log_lo) * (bottom - top)
            return int(x), int(y)
        painter.setPen(QPen(QColor("#dbe2ea"), 1))
        for j in range(5):
            y = top + (bottom - top) * j / 4
            painter.drawLine(left, int(y), right, int(y))
        colors = [QColor("#2563eb"), QColor("#d97706")]
        for k, s in enumerate(self.series):
            pts = [xy(i, v, len(s["errors"])) for i, v in enumerate(s["errors"])]
            painter.setPen(QPen(colors[k % len(colors)], 2))
            for a, b in zip(pts, pts[1:]):
                painter.drawLine(a[0], a[1], b[0], b[1])
            if pts:
                x, y = pts[min(self.index, len(pts) - 1)]
                painter.setBrush(colors[k % len(colors)])
                painter.drawEllipse(x - 4, y - 4, 8, 8)
            painter.setPen(colors[k % len(colors)])
            painter.drawText(left + 10 + k * 185, h - 12, s["label"])
        painter.setPen(QColor("#475569"))
        painter.setFont(QFont("Microsoft YaHei UI", 8))
        painter.drawText(8, top + 8, f"{10**log_hi:.1e} m")
        painter.drawText(8, bottom, f"{10**log_lo:.1e} m")
        painter.drawText(left, h - 25, "采样点 0")
        painter.drawText(right - 60, h - 25, f"采样点 {max(len(self.series[0]['errors']) - 1, 0)}")


class ExperimentVisualization(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.output_dir = None
        self.scenarios = {}
        self.rows = {}
        self.targets = []
        self.point_index = 0
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("工况："))
        self.scenario_combo = QComboBox()
        self.scenario_combo.currentTextChanged.connect(self._scenario_changed)
        controls.addWidget(self.scenario_combo, 1)
        self.point_label = QLabel("采样点：—")
        controls.addWidget(self.point_label)
        layout.addLayout(controls)
        views = QHBoxLayout()
        self.left_view = WorkspaceView("当前 DLS")
        self.right_view = WorkspaceView("奇异值自适应 DLS")
        views.addWidget(self.left_view, 1)
        views.addWidget(self.right_view, 1)
        layout.addLayout(views, 3)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self._point_changed)
        layout.addWidget(self.slider)
        self.error_view = ErrorCurveView()
        layout.addWidget(self.error_view, 2)
        self.detail_label = QLabel("运行自动实验后，可逐点查看实际姿态与误差。")
        self.detail_label.setWordWrap(True)
        layout.addWidget(self.detail_label)

    def load_batch(self, output_dir, payload):
        self.output_dir = Path(output_dir)
        self.scenarios = {s["scenario_id"]: s for s in payload["scenarios"]}
        self.rows = {}
        for scenario in payload["scenarios"]:
            scenario_id = scenario["scenario_id"]
            path = self.output_dir / scenario_id / "point_results.csv"
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                records = list(csv.DictReader(stream))
            self.rows[scenario_id] = {}
            for method in ("current_dls", "sigma_adaptive_dls"):
                method_rows = [r for r in records if r["method"] == method]
                method_rows.sort(key=lambda r: int(r["sample_index"]))
                self.rows[scenario_id][method] = method_rows
        self.scenario_combo.blockSignals(True)
        self.scenario_combo.clear()
        self.scenario_combo.addItems(self.scenarios.keys())
        self.scenario_combo.blockSignals(False)
        if self.scenario_combo.count():
            self.scenario_combo.setCurrentIndex(0)
            self._scenario_changed(self.scenario_combo.currentText())

    def _scenario_changed(self, scenario_id):
        if scenario_id not in self.rows:
            return
        self.targets = [(float(r["target_x_m"]), float(r["target_y_m"])) for r in self.rows[scenario_id]["current_dls"]]
        count = len(self.targets)
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(count - 1, 0))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.point_index = 0
        self._refresh_views()

    def _point_changed(self, index):
        self.point_index = index
        self._refresh_views()

    def _refresh_views(self):
        scenario_id = self.scenario_combo.currentText()
        if scenario_id not in self.rows:
            return
        series = []
        detail_parts = []
        for method, view, label in (
            ("current_dls", self.left_view, "当前 DLS"),
            ("sigma_adaptive_dls", self.right_view, "自适应 DLS"),
        ):
            rows = self.rows[scenario_id][method]
            if not rows:
                continue
            row = rows[min(self.point_index, len(rows) - 1)]
            q = json.loads(row["q_after_rad"])
            failed = row["point_success"].lower() != "true"
            view.set_sequence(self.targets, q, self.point_index, failed)
            errors = [float(r["actual_error_m"]) for r in rows]
            series.append({"label": label, "errors": errors})
            detail_parts.append(
                f"{label}：误差 {float(row['actual_error_m']):.3e} m；"
                f"最小奇异值 {float(row['min_singular_value_after']):.3e}；"
                f"关节步长 {float(row['joint_step_rad']):.3e} rad；"
                f"耗时 {float(row['solve_time_ms']):.3f} ms；"
                f"结果 {'成功' if not failed else '失败：' + (row['failure_code'] or '误差超限')}"
            )
        self.error_view.set_series(series, self.point_index)
        self.point_label.setText(f"采样点：{self.point_index + 1}/{len(self.targets)}")
        self.detail_label.setText("  |  ".join(detail_parts))
