"""桌面端自动实验入口：生成工作空间场景、执行配对比较并呈现汇总。"""

from datetime import datetime
from pathlib import Path
import sys

from PySide6.QtCore import Qt
from experiment_visualization import ExperimentVisualization
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MATH_DIRECTORY = PROJECT_ROOT / "代码与仿真" / "数学原型"
if str(MATH_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MATH_DIRECTORY))

from ik_experiment_batch import run_experiment_batch
from ik_experiment_design import generate_workspace_boundary_scenarios


class AutomaticExperimentDialog(QDialog):
    """运行一个小型、可复现的 DLS 配对实验，并显示批次汇总。"""

    COLUMNS = (
        ("工况", "scenario_id"),
        ("算法", "method"),
        ("目标点数", "sample_count"),
        ("整条序列成功", "sequence_success"),
        ("点成功率", "success_rate"),
        ("不可达目标", "unreachable_target_count"),
        ("求解失败点", "solver_failure_count"),
        ("最大末端误差 (m)", "max_endpoint_error_m"),
        ("P95 耗时 (ms)", "p95_solve_time_ms"),
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("自动实验与算法对比")
        self.resize(1240, 940)
        layout = QVBoxLayout(self)
        intro = QLabel(
            "自动生成内边界、外边界和普通可达区三个工况，使用相同目标序列配对比较当前 DLS 与奇异值自适应 DLS。"
            "本演示使用二维两连杆模型（0.6 m、0.4 m）和固定随机种子。"
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.status_label = QLabel("尚未运行。点击“运行自动实验”开始小规模批次。")
        self.status_label.setObjectName("experimentStatusLabel")
        layout.addWidget(self.status_label)
        self.results_table = QTableWidget(0, len(self.COLUMNS), self)
        self.results_table.setHorizontalHeaderLabels([title for title, _ in self.COLUMNS])
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.results_table, 1)
        self.output_label = QLabel("结果将保存到项目的“数据与实验/自动实验批次”目录。")
        self.output_label.setWordWrap(True)
        layout.addWidget(self.output_label)
        self.visualization = ExperimentVisualization(self)
        layout.addWidget(self.visualization, 2)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.run_button = QPushButton("运行自动实验")
        self.run_button.setObjectName("runAutomaticExperimentButton")
        self.run_button.clicked.connect(self._run_experiment_batch)
        buttons.addWidget(self.run_button)
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.close)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)

    def _new_output_directory(self):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        return PROJECT_ROOT / "数据与实验" / "自动实验批次" / f"桌面运行-{stamp}"

    def _run_experiment_batch(self):
        self.run_button.setEnabled(False)
        self.results_table.setRowCount(0)
        self.status_label.setText("正在生成场景并运行配对实验……")
        self.status_label.repaint()
        try:
            scenarios = generate_workspace_boundary_scenarios(
                link_lengths_m=(0.6, 0.4), sample_count=9, seed=20261022,
                boundary_offset_m=1e-4,
            )
            base_config = {
                "experiment_id": "desktop-auto-demo",
                "scenario_id": "pending-scenario",
                "robot": {"link_lengths_m": [0.6, 0.4]},
                "solver": {
                    "max_iterations": 120,
                    "tolerance_m": 1e-5,
                    "base_damping": 0.05,
                    "sigma_threshold": 0.02,
                    "gain": 0.05,
                    "max_damping": 1.0,
                },
            }
            output_dir = self._new_output_directory()
            payload = run_experiment_batch(scenarios, base_config, output_dir)
            self.visualization.load_batch(output_dir, payload)
            for scenario in payload["scenarios"]:
                for method, metrics in scenario["methods"].items():
                    row_values = {
                        "scenario_id": scenario["scenario_id"],
                        "method": method,
                        **metrics,
                    }
                    row = self.results_table.rowCount()
                    self.results_table.insertRow(row)
                    for column, (_, key) in enumerate(self.COLUMNS):
                        value = row_values[key]
                        if key == "sequence_success":
                            display = "是" if value else "否"
                        elif key == "success_rate":
                            display = f"{value:.1%}"
                        elif isinstance(value, float):
                            display = f"{value:.6g}"
                        else:
                            display = str(value)
                        item = QTableWidgetItem(display)
                        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                        self.results_table.setItem(row, column, item)
            self.results_table.resizeColumnsToContents()
            self.status_label.setText(
                f"实验完成：{payload['scenario_count']} 个工况，{self.results_table.rowCount()} 条算法汇总记录。"
                "成功率仅代表本次合成场景，不代表算法普遍优劣。"
            )
            self.output_label.setText(f"结果目录：{output_dir}\n批次汇总：batch_summary.json / batch_summary.csv")
        except Exception as exc:
            self.status_label.setText("实验未完成，请查看错误信息。")
            QMessageBox.critical(self, "自动实验失败", f"运行自动实验时发生错误：\n{exc}")
        finally:
            self.run_button.setEnabled(True)
