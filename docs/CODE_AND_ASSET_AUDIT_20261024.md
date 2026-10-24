# 代码与资产审查报告（2026-10-24）

> 审查对象：`D:\Project\机械臂交互式仿真平台`
>
> 审查性质：**只读**。除新增本报告外，截至生成时未删除、未移动、未修改任何项目文件。
>
> 审查方法：逐文件阅读、AST 调用关系检查、Qt 信号连接复核、SHA-256 对比、`unittest`、`py_compile`、目录与 Git 状态盘点。
>
> 结论口径：**P0 可直接删** / **P1 建议归档** / **P2 保留但瘦身** / **P3 需确认**。

---

## 0. 结论先行

项目主体是健康的：70 个非备份 Python 文件全部可编译；数学原型 143 项、桌面应用 102 项测试通过；核心数学模块、GUI、实验包和离线研究脚本都在正常工作。

真正的问题不是“代码太多”，而是四类可清理的沉积物：

1. **13 个编辑器备份**、**12 个重复自动实验批次**、**1 个与文件夹逐文件相同的 ZIP**、**1 个一次性补丁脚本**和**1 个本地快捷方式**混在仓库里；
2. 一批“第一轮审查 / 旧界面”文档仍在描述已经不存在或已经变化的能力；
3. `app.py` 里有一段永远不会执行的“对比两种算法”旧分支，以及一个只完成了一半、从未接入界面的“姿态锚点”功能；
4. 实验包元数据仍写 `PCHIP`，而当前规划器已经改为 `CubicSpline`。

建议顺序：

- 先做 **P0**：约 **2.1 MB**，零研究价值，删除风险最低；
- 再做 **P1**：约 **3.7 MB**，只归档、不删除，保留证据链；
- **P2** 按主题拆成小提交：先删死代码，再修文档，最后修元数据与可移植性；
- **P3** 逐项确认，尤其是 `阻尼参数系统扫描` 和旧研究脚本，不能仅凭“当前无引用”删除。

一句话：**不要为了省 66 MB 去删实验数据；真正该删的是备份、重复批次和过期样例，真正该修的是死代码和文档口径。**

---

## 1. 审查范围、环境与限制

### 1.1 规模

| 范围 | 文件数 | 大小 | 备注 |
|---|---:|---:|---|
| 项目文件总计 | 423 | 约 68.6 MB | 不含 `.git`、`.venv` 等本地环境 |
| `数据与实验/` | 313 | 66.16 MB | 绝大部分是实验证据 |
| `数据与实验/奇异位形附近DLS基线实验/` | 82 | 65.53 MB | 核心研究数据 |
| `数据与实验/自动实验批次/` | 221 | 0.54 MB | 13 个同配置批次 |
| `数据与实验/算法对比演示包_奇异位形附近/` | 5 | 0.01 MB | 过期样例 |
| `数据与实验/规划策略筛选研究_20261023/` | 3 | 0.08 MB | 2026-10-23 报告的数据源 |
| 代码、文档、根目录文件 | 约 110 | 约 2.4 MB | 含 70 个非备份 `.py` |

### 1.2 环境与验证

- Python 3.13.13、NumPy 2.5.4、SciPy 1.18.1、PySide6 6.12.0。
- 数学原型 **143 项** `unittest` 通过；桌面应用 **102 项** `unittest` 通过（与 `CHANGELOG.md:188` 一致）。
- 70 个非备份 `.py` 文件全部通过 `py_compile`。
- 当前环境未安装 `pytest`、`ruff`、`coverage`；因此未做覆盖率、lint 和 pytest 风格检查。
- 多个文件带 UTF-8 BOM，AST 与读取统一使用 `utf-8-sig`。
- Git：`main` 分支、11 个提交、**57 项未提交/未跟踪变更**。审查期间未执行 `git clean`、`git reset --hard` 或任何删除。

### 1.3 判断原则

- **运行期证据优先于名称**：`*_backup`、`*_old` 不自动等于可删；`*_test` 也不自动等于可删。
- **有输出数据的研究脚本不按“无 import”删除**：数据目录里的报告就是它们的产物。
- **历史实验包不按当前代码回写**：旧 `experiment.json` 是历史快照，修正元数据只影响新生成的包。
- **归档不等于释放空间**：在仓库内移动只减少主线干扰；要真正减容，必须移到仓库外或压缩。

---

## 2. P0：可直接删除

> 这一级的共同特征：**可由当前代码/数据重建，或本来就不属于项目资产**。
>
> 合计约 **2.1 MB**。删除前仍建议先提交或打 tag，见第 7 节。

### P0-1 十三个编辑器备份（1,545,261 B）

**路径与清单：**

```text
代码与仿真/桌面应用/
  app.py.bak_before_time_sync_fix
  app.py.bak_bias_fix
  app.py.bak_diagnostics_question_mark_root_cause
  app.py.bak_export_all_plots
  app.py.bak_fix_diagnostics_punctuation
  app.py.bak_fix_question_marks_v2
  app.py.bak_fix_visible_newlines
  app.py.bak_multi_scheme_panels
  app.py.bak_scheme_dimension
  app.py.pre_multischeme.bak
  simulation_plot.py.bak_scheme_dimension
代码与仿真/数学原型/
  trajectory_planner.py.bak
  trajectory_planner.py.bak_bias_fix
```

**证据：**

- 全部是 2026-10-22 至 2026-10-24 的编辑前快照，命名直接对应一次修复或一次界面重构。
- 当前 `app.py`、`simulation_plot.py`、`trajectory_planner.py` 均可编译，且 143 + 102 项测试通过；备份不参与任何 import。
- `.gitignore` 目前只有 `*.bak`，所以 11 个 `.bak_*` 文件仍以未跟踪状态出现在 `git status`；另外 2 个 `*.bak` 虽被忽略，但仍占磁盘。
- 这些备份没有对应的文档引用，也没有测试依赖。

**动作：**

1. 删除上述 13 个文件；
2. 在 `.gitignore` 增加 `*.bak_*`、`*.pre_*`，防止再次出现；
3. 如果确实需要保留某次重构前的版本，应使用 Git 提交/tag，而不是文件副本。

---

### P0-2 十二个重复的自动实验批次（约 0.49 MB）

**路径：**

```text
数据与实验/自动实验批次/桌面运行-20261022-*
```

共 13 个目录，每个约 42.1 KB、17 个文件。建议保留最新的：

```text
数据与实验/自动实验批次/桌面运行-20261022-173413-697325/
```

删除其余 12 个。

**证据：**

- 13 个目录的 3 份 `experiment_config.json`（inner / outer / regular）分别具有**完全相同的 SHA-256**；也就是说 13 次运行使用的是同一套模型、目标、种子和求解参数。
- 13 个批次的 `batch_summary.json` 指标一致：
  - `workspace-inner_boundary`：`current_dls` 成功率 1.0，`sigma_adaptive_dls` 成功率 0.0；
  - `workspace-outer_boundary`：`current_dls` 成功率 1.0，`sigma_adaptive_dls` 成功率 0.0；
  - `workspace-regular_reachable`：两者成功率均为 1.0。
- 差异主要在 `solve_time_ms`、时间戳和浮点末位；指标一致。`point_results.csv` 因计时列不同而不是逐字节相同，但这不是独立实验。
- 13 次运行集中在 2026-10-22 17:12–17:34，属于同一轮界面演示的重复点击。

**例外：** 如果确实要把这 13 次当作“运行时波动”的重复样本，就应保留全部并在报告中说明用途；当前没有任何文档或分析引用这 13 次重复，因此默认只保留 1 个。

---

### P0-3 与文件夹完全重复的 ZIP（3.8 KB）

**路径：**

```text
数据与实验/算法对比演示包_奇异位形附近.zip
```

**证据：** ZIP 内 5 个条目与同名文件夹的 5 个文件逐一 SHA-256 相同：

| 文件 | SHA-256 前 16 位 | 与文件夹 |
|---|---|---|
| `experiment.json` | `e6e40cf9b6af164f` | 相同 |
| `README.md` | `b26a82c259bad4b6` | 相同 |
| `results.csv` | `0cc1a9fdc9226339` | 相同 |
| `summary.json` | `935afc656f7d58a0` | 相同 |
| `对比诊断记录.md` | `5e3c4157c68c2a05` | 相同 |

**动作：** 删除 ZIP。文件夹本身不建议直接删，见 P1-3。

---

### P0-4 一次性补丁脚本（633 B）

**路径：**

```text
代码与仿真/桌面应用/_patch_export_all_plots.py
```

**证据：**

- 脚本硬编码绝对路径 `D:\Project\机械臂交互式仿真平台\...`（第 3 行）；
- 只做一次字符串替换：`image = self._plot_widgets[0].grab()` → 容器 `grab()`（第 5–6 行）；
- 脚本自己会写出 `app.py.bak_export_all_plots`（第 8–9 行），该备份已存在，说明补丁已经执行过；
- 不属于应用运行路径，也不被任何测试/文档引用。

**动作：** 删除脚本；对应备份已在 P0-1 中删除。

---

### P0-5 本地快捷方式（1,885 B）

**路径：**

```text
启动仿真.exe.lnk
```

**证据：**

- Windows 快捷方式，不是源码、配置或实验数据；
- 不被代码、README、测试引用；
- 放在仓库根目录只会制造“项目自带可执行文件”的误解。

**动作：** 从仓库删除。若需要快捷方式，放在桌面或开始菜单，不要提交到仓库。

---

### P0-6 工具生成的 ACL 报告（59,165 B）

**路径：**

```text
.dsh-acl-reports/
  acl-backup-9f5a1c56a0514acba1f66b4e2ee79ae4.json
  acl-backup-9f5a1c56a0514acba1f66b4e2ee79ae4.json.ps1
  acl-report-cb37b88d70d14d259b05c1c675eaa253.jsonl
```

**证据：**

- `acl-backup-*.json` 记录的是仓库根目录的 Windows DACL 备份（`D:\Project\机械臂交互式仿真平台`），不是项目内容；
- 这是运行工具（而非项目代码）产生的审计/备份产物；
- 不被 `.gitignore` 覆盖，当前会出现在版本控制视野中。

**动作：** 删除，或在确认 ACL 备份不再需要后删除；同时在 `.gitignore` 增加 `.dsh-acl-reports/`。若担心 ACL 恢复，先把整个目录移到仓库外。

---

### P0 汇总

| 项目 | 数量 | 大小 |
|---|---:|---:|
| 编辑器备份 | 13 | 1,545,261 B |
| 重复自动实验批次 | 12 | 约 0.49 MB |
| 重复 ZIP | 1 | 3.8 KB |
| 一次性补丁脚本 | 1 | 633 B |
| 本地快捷方式 | 1 | 1,885 B |
| ACL 工具产物 | 3 | 59,165 B |
| **合计** | **31 个文件/目录项**（13 个备份 + 12 个批次目录 + 3 个单文件 + 3 个 ACL 文件） | **约 2.1 MB** |

---

## 3. P1：建议归档，不直接删除

> 目标目录建议：
>
> - 文档：`docs/archive/`
> - 数据：`数据与实验/归档/`
>
> 这两个目录当前为空。归档时在每个归档目录放一份 `ARCHIVE_README.md`，写清：归档日期、原路径、替代它的新路径、归档原因。
>
> 注意：仓库内归档**不释放磁盘空间**，只降低主线干扰；如果要减容，应移到仓库外或压缩。

### P1-1 旧的工作空间边界压力测试（1.86 MB）

**路径：**

```text
数据与实验/奇异位形附近DLS基线实验/工作空间边界压力测试/
```

**替代者：**

```text
数据与实验/奇异位形附近DLS基线实验/工作空间边界压力测试_阻尼上限1.0复测_20261022/
```

**证据：**

- 新旧目录的 `outer_98pct / nominal / current_dls` 最差误差均为 `9.381034659751649e-08 m`；`工作空间边界汇总.csv` 的指标逐项相同，只有耗时不同。
- 新目录记录的是 `max_damping=1.0` 的复测；当前脚本的默认输出目录已经指向新目录：
  - `代码与仿真/数学原型/run_ik_workspace_boundary_stress.py:18`
  - 求解参数 `max_damping=1.0` 在同文件第 70 行。
- 旧目录不再被脚本写出，也不再是任何当前文档的输入。

**动作：** 归档旧目录，保留新目录。归档说明中写明“旧目录指标与新目录等价，已被 `max_damping=1.0` 复测替代”。

---

### P1-2 旧的目标扰动基础测试（1.86 MB）

**路径：**

```text
数据与实验/奇异位形附近DLS基线实验/目标扰动分支与可达性压力测试/
```

**不要动这两个：**

```text
目标扰动分支与可达性压力测试_可达初值扩展/
目标扰动分支与可达性压力测试_阻尼上限1.0复测_20261022/
```

**证据：**

- 基础目录是较早的版本，不含 `reachable_near_inner` / `reachable_near_outer`；`_可达初值扩展` 在其基础上增加了这两组可达初值，是它的超集。
- `_可达初值扩展` **不是**可随手归档的旧数据：它是 `docs/PATH_LEVEL_RISK_VISUALIZATION.md:9` 明确写出的输入（96 组、11,616 条逐点记录），也是“1,331 个可达点求解失败、阈值 0.02 Precision/Recall 29.89%/63.64%”这一结论的数据源。
- `_阻尼上限1.0复测_20261022` 是最新修正版：同一扩展工况上，自适应 DLS 的成功率从 41.7% 恢复到 100%。它是“当前应保留”的版本，但不能替代 `_可达初值扩展` 的历史证据角色。

**动作：** 只归档基础目录；保留扩展目录和复测目录。归档说明中写明“基础目录已被扩展目录包含，扩展目录仍是路径风险看板的输入”。

> 这是本报告最重要的一条“反直觉”结论：**最新替代版不等于旧证据可以删**。旧扩展数据记录了修复前的失败，删掉就看不出 `max_damping=1.0` 到底修了什么。

---

### P1-3 过期的算法对比演示包（约 10 KB）

**路径：**

```text
数据与实验/算法对比演示包_奇异位形附近/
```

**证据：**

- `README.md` 第 5–9 行仍要求用户把界面切换到“对比两种算法（显示影子机械臂）”；该入口已从当前界面移除，当前是“路径方案（可多选）”叠加显示。
- `results.csv` 只有表头（README 自己也写明“只有表头，导入后重新规划”）。
- `experiment.json` 的 `algorithm.joint_path_interpolation` 仍写 `PCHIP`，而当前规划器使用 `CubicSpline`。
- 唯一仍有价值的是 `对比诊断记录.md`：它记录了“两种算法在当前参数下几乎完全重合（最大差异约 5.13e-9 rad）”这一负结果，是后续移除自适应 DLS 常规选项的证据之一。

**动作：**

- 删除同名 ZIP（已在 P0-3）；
- 将整个文件夹归档到 `数据与实验/归档/`；
- 在归档 README 中加一句“原界面入口已移除；本包仅作为 2026-10-22 的负结果记录保留”。

---

### P1-4 过期的首版界面说明（2.8 KB）

**路径：**

```text
docs/UI_PROTOTYPE.md
```

**证据：**

- 第 5 行：“当前原型不承担完整仿真、轨迹编辑或实验管理功能。”
- 第 43 行：“当前不包含轨迹编辑、仿真时间播放、数据曲线、数据导出、场景保存/加载、负载对象、动力学、碰撞检测和三维视图。”

当前桌面应用已经具备轨迹编辑、播放、曲线、CSV/图像导出、实验包保存/加载、动力学力矩与部分约束检查。该文档描述的是首版原型，不是当前状态。

**动作：** 归档到 `docs/archive/UI_PROTOTYPE_2026-10.md`，并在归档说明中注明“历史首版说明，已被当前桌面应用 README 与总规划取代”。

---

### P1-5 过期的第一轮能力审查（6.1 KB）

**路径：**

```text
docs/CURRENT_CAPABILITY_AUDIT.md
```

**证据：**

- 第 7–8 行仍写“数学原型 79 项、GUI 55 项”；当前是 **143 + 102**（`CHANGELOG.md:188`、`docs/PROJECT_MASTER_PLAN.md:9`）。
- 第 20 行仍写“当前实现没有基于最小奇异值/条件数的自适应阻尼策略”；当前 `compare_damping_strategies.adaptive_ik` 已实现，并已用于多组实验。
- 第 24 行仍写“分段 PCHIP”；当前规划器使用 `CubicSpline`，见 `trajectory_planner.py:7`。
- 第 27–28 行仍写“实验包跨进程一致性未完成、公平算法比较尚未完成”；当前已有实验包 v1、自动实验批次和多组对照数据。

**动作：** 归档为“第一轮审查（历史）”。不要在原文件上改数字，否则会丢失“当时结论如何演变”的记录。

---

### P1-6 已完成的实施计划（2.6 KB）

**路径：**

```text
docs/superpowers/plans/2026-10-22-path-risk-label-validation.md
```

**证据：**

- 计划中的交付物都已存在：
  - `代码与仿真/数学原型/path_risk_validation.py`
  - `代码与仿真/数学原型/test_path_risk_validation.py`
  - `数据与实验/奇异位形附近DLS基线实验/路径级风险可视化/路径风险标签有效性验证.md`
  - `docs/PATH_LEVEL_RISK_VISUALIZATION.md` 已补充复现命令。
- 文件里的复选框仍未勾选，但这是计划文档未回填，不代表任务未完成。

**动作：** 归档到 `docs/archive/plans/`，并在归档说明中写“已完成；保留供追溯，不再作为待办”。

---

### P1-7 与新总规划重叠的旧研究路线（4.9 KB）

**路径：**

```text
docs/PROJECT_GOAL_AND_RESEARCH_PLAN.md
```

**证据：**

- 该文档的“多约束运动规划 / 阶段 A-B-C / 实体机械臂验证”框架，已被 `docs/PROJECT_MASTER_PLAN.md` 的 14 周执行计划取代。
- 但它仍有一段总规划没有展开的内容：**阶段 C 的实体机械臂验证设想**（设备可用时如何做仿真—实测对比）。
- 直接删除会丢失这段设想；直接保留又会让读者以为它是当前路线。

**动作：**

1. 把“阶段 C 实体机械臂验证”的独特内容合并进 `docs/PROJECT_MASTER_PLAN.md` 的后续/实体验证章节；
2. 将原文件归档到 `docs/archive/`；
3. 若不想改总规划，至少在原文件顶部加一行“历史路线，以 `PROJECT_MASTER_PLAN.md` 为准”。

---

### P1 汇总

| 项目 | 大小 | 处理 |
|---|---:|---|
| 旧工作空间边界测试 | 1.86 MB | 归档，保留新复测 |
| 旧目标扰动基础测试 | 1.86 MB | 归档，保留扩展与复测 |
| 过期算法对比演示包 | 约 10 KB | 归档，保留负结果记录 |
| `UI_PROTOTYPE.md` | 2.8 KB | 归档 |
| `CURRENT_CAPABILITY_AUDIT.md` | 6.1 KB | 归档 |
| 已完成实施计划 | 2.6 KB | 归档 |
| `PROJECT_GOAL_AND_RESEARCH_PLAN.md` | 4.9 KB | 合并后归档 |
| **合计** | **约 3.7 MB** | 仓库内归档不释放空间 |

---

## 4. P2：保留但应瘦身/修正

> 这一级不是“文件该不该删”，而是“代码和文档该不该继续背着历史包袱”。
>
> 建议每个主题单独一个提交，改完立即跑测试。

### P2-1 `app.py` 中永远不执行的“对比两种算法”分支

**证据链：**

```text
app.py:648    algorithm = "current_dls"
app.py:674    if algorithm == "compare":
app.py:762    if algorithm != "compare":
```

- `algorithm` 在 648 行被赋值为常量 `"current_dls"`，此后没有任何地方重新赋值；因此 674 行的条件恒为假。
- 674–745 行整段不可达，包括：
  - `comparison_plans` 的构建；
  - `plan_summary` 嵌套函数（714–737）；
  - 影子机械臂轨迹 `_shadow_*` 的设置（699–713）；
  - `CubicHermiteSpline` 的使用（702–705）。
- 762 行的 `if algorithm != "compare":` 恒为真，可以把条件去掉，保留其内部“已选路径方案”的文本设置。
- `CubicHermiteSpline` 只在第 10 行导入、第 702 行使用；删除死分支后该导入也可删。
- `_adaptive_dls_solver` 目前只被这个死分支和 `test_path_schemes.py:68` 使用；该方法本身应保留，因为它是离线研究接口。

**动作：**

1. 删除 674–745 行；
2. 将 762 行的条件改为无条件执行（或直接去掉 `if`、保留代码块）；
3. 删除第 10 行 `CubicHermiteSpline` 导入；
4. 顺手修正 747–761 行当前异常缩进的 `plan = plan_cartesian_path(...)` 调用；
5. 跑 143 + 102 项测试。

预计减少约 **115 行**。`algorithm_comparison_label` 仍被 826 行使用，不要删控件；它现在实际显示“已选路径方案”，后续可考虑改名为 `scheme_summary_label`。

---

### P2-2 从未接入界面的“姿态锚点”半成品

**证据链：**

- `app.py:981–999` `_refresh_pose_anchor_options()` 第一行就检查 `hasattr(self, "pose_anchor_combo")`，但全文件从未创建 `pose_anchor_combo`，所以永远 early-return。
- `app.py:1025–1032` `_refresh_pose_constraint_label()` 同样检查 `pose_constraints_label`，该控件也从未创建。
- `app.py:1001–1023` `_activate_pose_anchor()`、`1034–1054` `_record_anchor_pose_constraint()`、`1056–1065` `_clear_anchor_pose_constraint()` 没有任何调用点。
- `_anchor_pose_constraints` 只在 `174` 行初始化、`2613/3115` 行清空、`1050/1061` 行读写；规划器从未把它转成 `pose_constraints`（实际只有 `app.py:647` 的 `{0: start_angles}`）。
- 调用 `_refresh_pose_anchor_options()` 的地方只有 `977`、`2614`、`3117`，全部会因为控件不存在而空转。

**动作：**

- 删除 981–1065 行的方法；
- 删除 `977`、`2614`、`3117` 的调用；
- 删除 `174` 的初始化与 `2613`、`3115` 的清空；
- 不要误删 `_simulation_anchor_points()`（1067 行起）：它处理的是路径锚点，不是姿态约束。

预计减少约 **90 行**。如果这个功能还有产品价值，正确做法是重新设计并一次性接好 UI、状态和规划器，而不是保留一个永远 early-return 的半成品。

---

### P2-3 其他确认无调用的方法/类

| 文件:行号 | 符号 | 证据 | 建议 |
|---|---|---|---|
| `app.py:1484–1492` | `_plot_summary_text` | 全仓无调用；实际摘要文本由其他方法生成 | 删除 |
| `app.py:2183–2185` | `_set_plot_scheme` | 全仓无调用；方案选择走新的组合框 | 删除 |
| `app.py:3159–3165` | `_on_angles_changed` | 全仓无调用；角度控件已不连接此槽 | 删除 |
| `simulation_plot.py:146–211` | `AxisSelectionDialog` | 全仓无实例化；`app.py:60` 只有导入 | 删除类与导入 |
| `workspace.py:82–84` | `ArmWorkspace.set_drag_mode` | 全仓无调用；`_drag_mode` 只写不读 | 删除方法与 `line 41` 的初始化 |
| `workspace.py:111–115` | `ArmWorkspace.clear_path` | 全仓无调用 | 删除 |

**明确不要误删：**

- `workspace.py:242` `wheelEvent`：Qt 事件重写，由框架调用，不是死代码。
- `simulation_plot.py:434` `smoothing_enabled`：公开 getter，当前无外部调用但属于绘图 API，保留。
- `app.py` 的 `_adaptive_dls_solver`：被 `test_path_schemes.py:68` 使用，保留。

---

### P2-4 未使用的 import

以下为 AST 检查结果（排除字符串、注释和同名局部变量；BOM 文件按 `utf-8-sig` 读取）：

| 文件:行号 | 未使用符号 |
|---|---|
| `app.py:49` | `solve_two_link_ik` |
| `app.py:52` | `filter_ik_solutions` |
| `app.py:59` | `parameterize_path` |
| `app.py:60` | `AxisSelectionDialog` |
| `numerical_ik.py:4` | `minimize` |
| `trajectory_planner.py:12` | `calculate_trajectory_torques` |
| `compare_damping_strategies.py:5` | `math` |
| `visualize_ik_comparison.py:4` | `json` |
| `visualize_ik_comparison.py:5` | `math` |
| `visualize_ik_robustness.py:4` | `json` |
| `run_ik_seed_perturbation_stress.py:3` | `json` |

说明：

- `app.py:49/52/59` 是早期两连杆/路径 API 的遗留导入；删除前确认没有动态 `eval`/`getattr` 使用（当前检查为无）。
- `trajectory_planner.py:12` 的 `calculate_trajectory_torques` 在文件内没有再被调用；力矩计算走 `dynamics`/`trajectory_dynamics` 的其他路径。
- 这些删除不影响运行逻辑，但建议和对应模块的测试一起跑。

---

### P2-5 文档口径修正（不归档、直接改）

| 文件:行号 | 当前问题 | 建议改法 |
|---|---|---|
| `CONTRIBUTING.md:3` | “本项目目前处于设计阶段” | 改为“本项目目前是可运行的研究原型，仍在持续验证” |
| `代码与仿真/桌面应用/README.md:49` | 仍称“场景保存/加载、速度约束……尚未实现”，且称“速度与加速度由离散采样数值微分估算” | 与同文件 `18`、`22–23`、`53–57` 行统一：已有实验包保存/加载、速度/加速度/力矩约束检查、解析导数；只保留真实边界（无摩擦/接触/电机饱和/三维/实机认证）。若“场景保存”特指路径场景而非实验包，请保留这一细分说明 |
| `README.md:41–42` | 仓库结构只列 `docs/DEVELOPMENT_ROADMAP.md` | 补上 `PROJECT_MASTER_PLAN.md`、审计报告、技术文档 |
| `README.md:30` | “贡献与开发约定（如文件存在）” | 文件已存在，去掉“如文件存在” |
| `docs/DEVELOPMENT_ROADMAP.md` 顶部 | 仍是“当前状态”口吻，但 README 已标注为历史路线 | 在标题下加“历史路线，以 `PROJECT_MASTER_PLAN.md` 为准”；不要重写历史阶段 |
| `算法差异验证报告_20261023.md:43–44` | “参考构型偏置当前实现不合格，最大步长 0.246 rad”对应修复前实现 | 在文首加版本注记：该结论对应 2026-10-23 旧实现；`CHANGELOG.md:92` 已记录修复后同一指标降至 0.02–0.04 rad，需重新评估。不要直接改原表数字 |

补充：

- `代码与仿真/数学原型/README.md` 本次检查未发现“动力学尚未实现”的旧声明；该文件已描述 `dynamics.py` 与 `trajectory_planner.py`，不需要按旧清单修改。
- `docs/UI_PROTOTYPE.md`、`docs/CURRENT_CAPABILITY_AUDIT.md` 走归档，不在原文件上修补。

---

### P2-6 实验包元数据漂移与可移植性

#### (1) `PCHIP` 已过时

**证据：**

- 当前规划器：`代码与仿真/数学原型/trajectory_planner.py:7` 导入 `CubicSpline`；桌面 README `:57` 明确写“规划器的关节路径插值使用 C² 三次样条。此前用的是 C¹ 的 PCHIP”。
- 但新实验包的配置写入仍在 `app.py:1581`：
  ```python
  "joint_path_interpolation": "PCHIP, segmented at detected sharp corners",
  ```
- 历史数据同样带旧值，例如 `数据与实验/奇异位形附近DLS基线实验/01_参考姿态/experiment.json:123`。

**建议：**

1. 修改 `app.py:1581` 为新口径，例如：
   ```python
   "joint_path_interpolation": "CubicSpline (C2), segmented at detected sharp corners",
   ```
2. 增加一个显式的 `package_format_version`，避免以后再把“当前实现”写死在历史包里；
3. **不要批量回写历史 `experiment.json`**。它们是历史快照，回写会伪造“当时就是这样”的记录。需要说明时，在数据 README 或归档说明中加注记。

#### (2) 绝对路径

- `数据与实验/奇异位形附近DLS基线实验/实验包索引.json` 的每个 case 都有 `"path": "D:\\Project\\机械臂交互式仿真平台\\..."`。
- `基线分析/基线报告.md:3` 也写死了绝对路径。
- 代码中同样有硬编码根目录：
  - `visualize_ik_comparison.py:9`
  - `visualize_ik_robustness.py:9`
  - `analyze_ik_path_quality.py:6`

**建议：**

- 脚本统一改为：
  ```python
  ROOT = Path(__file__).resolve().parents[2]
  ```
- 新建的实验包索引改为相对路径（相对仓库根或相对索引文件）；
- 历史报告里的绝对路径保留，归档说明中注明“历史生成物，路径未回写”。

#### (3) `.gitignore`

当前只有 `*.bak`。建议追加：

```gitignore
*.bak_*
*.pre_*
.dsh-acl-reports/
```

如果决定保留 `启动仿真.exe.lnk`，再加 `*.lnk`；本报告建议直接删除该快捷方式。

---

## 5. P3：需确认后再处理

### P3-1 `阻尼参数系统扫描` 是否归档

**路径：**

```text
数据与实验/奇异位形附近DLS基线实验/阻尼参数系统扫描/
```

**事实：**

- 约 **11.9 MB**，主要是 `参数扫描逐点结果.csv`（12.08 MB）。
- 设计：3 个基线场景 × 3 次重复 × 4 个阈值 × 4 个增益，`max_damping=0.2`。
- 与 `阻尼参数敏感性` **不是重复数据**：后者是 4 条合成路径 × 2 种采样密度 × 同一参数网格，带完整报告（14.16 MB CSV）。
- 它仍被 `CHANGELOG.md` 引用，且 `sweep_damping_parameters.py` 的默认输出目录就是它。

**建议：**

- 如果磁盘不紧张：**保留**。66 MB 的总量对研究项目不算大，删掉会削弱“系统扫描 vs 敏感性分析”的对照。
- 如果确实要减容：整套（数据 + 脚本说明）移到仓库外归档，并在 CHANGELOG 里写明“已归档，不在仓库内”；不要只删 CSV 留一个空目录。

**需要你确认：** 保留，还是移到仓库外？

---

### P3-2 `_shadow_*` 与 `set_scheme_arms` 的兼容层

**事实：**

- 这不是死代码。当前 `app.py:813–817` 仍会为“第一个次级方案”设置 `_shadow_joint_angles` 等状态；`workspace.py:297–311` 在存在叠加臂时优先画 `set_scheme_arms`，只有在没有叠加臂时才回退到影子臂。
- `test_dual_scheme_e2e.py:102` 明确断言影子角等于 `joint_bias` 的播放角；`test_gui_smoke.py:802/814` 也直接测试 `set_shadow_state`。
- 因此现在删 `_shadow_*` 会破坏测试和旧接口，不是“清理”，而是“重构”。

**建议：**

- 先确认新叠加臂的图例、播放、失败态、跟随态都已覆盖所有使用场景；
- 再把测试从 `_shadow_*` 迁到 `_scheme_arms`；
- 最后删除 `set_shadow_state`、`_shadow_*` 和相关绘制分支；
- 这是一个独立重构项，不要混在本次清理里。

**需要你确认：** 是否把“移除影子臂兼容层”排进下一轮重构？

---

### P3-3 旧研究脚本：不要按“0 引用”删除

以下脚本在文档/其他代码中没有直接 import，但它们**有数据产物**：

| 脚本 | 证据/产物 | 建议 |
|---|---|---|
| `analyze_ik_path_quality.py` | 生成 `独立路径验证/关节路径质量分析/`（4,338 行输入、18 组汇总） | 保留 |
| `validate_dls_heldout.py` | 生成 `独立路径验证/` 的 held-out 数据 | 保留 |
| `sweep_damping_parameters.py` | 生成 `阻尼参数系统扫描/` | 保留或随数据一起归档 |
| `benchmark_singularity.py` | 生成 `基线分析/` 的可重复基线 | 保留 |
| `research_ik_path_differences.py` | `算法差异验证报告_20261023.md:57` 明确引用 | 保留 |
| `research_ik_path_differences_wide_limits.py` | 与上一脚本配套的宽松限位实验 | 保留 |
| `explore_jacobian.py` | 506 B 的教学式脚本，打印三种姿态的雅可比/行列式/秩/奇异值 | 可归档或删除，象征意义大于研究价值 |

**需要你确认：** `explore_jacobian.py` 是否保留？其余脚本建议不动。

---

### P3-4 `论文与资料/` 只有 `.gitkeep`

**事实：**

- 目录中只有 `.gitkeep`，没有论文或资料。
- 但 `docs/PROJECT_GOAL_AND_RESEARCH_PLAN.md:81`、总规划和 README 都把“文献调研”列为研究步骤。

**建议：**

- **不要删目录**。这是项目声明的研究资料区，删掉会让“文献调研”失去落点。
- 要么补一个 `README.md` 说明“待补充的文献清单/检索策略”，要么开始放第一批评论文笔记。
- 这是内容建设问题，不是清理问题。

**需要你确认：** 是否现在补一个资料区 README？

---

### P3-5 历史报告的处理方式

- `算法差异验证报告_20261023.md`：建议“加版本注记”，不建议重写。它是 2026-10-23 的筛选记录，重写会丢失决策轨迹。
- `docs/DEVELOPMENT_ROADMAP.md`：建议“加历史横幅”，不建议删。它记录了阶段划分，仍有追溯价值。
- `设计总档案.md`：保留。它是设计决策总档案，和总规划的分工不同；后续只需在总规划里引用它，不必合并。

**需要你确认：** 是否接受“历史文档只加注记、不改正文”的原则？

---

## 6. 建议保留：不要误删的资产

### 6.1 数据

| 路径 | 大小 | 保留理由 |
|---|---:|---|
| `01_参考姿态` / `02_接近奇异` / `03_完全伸直` | 约 405 KB | 三个可导入的基线实验包，README 明确说明用途 |
| `基线分析` | 1.2 MB | 当前 DLS 基线报告与逐点数据 |
| `初始姿态扰动压力测试` | 2.56 MB | 种子扰动实验，CHANGELOG 有结论引用 |
| `奇异位形接近方向与采样密度扫描_20261022` | 3.38 MB | 当前奇异扫描主数据 |
| `独立路径验证` | 1.49 MB | 含关节路径质量、算法差异可视化、逐点奇异性关联三套子分析 |
| `鲁棒性测试` | — | 4 路径 × 3 采样密度 × 2 方法 × 3 重复 |
| `阻尼策略对照` + `阻尼策略对照_低增益` | 各约 1.4 MB | **参数不同**（默认阈值 0.05/增益 0.5；低增益 0.01/0.1），是独立对照，不是重复 |
| `阻尼参数敏感性` | 14.02 MB | 当前最完整的路径级参数研究，带报告 |
| `路径级风险可视化` | 14.02 MB | 风险标签结论与 10 MB HTML 看板；输入是 `_可达初值扩展` |
| `目标扰动..._可达初值扩展` | 3.75 MB | 路径风险看板的输入，记录修复前失败 |
| `目标扰动..._阻尼上限1.0复测_20261022` | 3.75 MB | 修复后的当前版本 |
| `规划策略筛选研究_20261023` | 0.08 MB | 2026-10-23 报告的复现数据 |

### 6.2 代码与测试

- `代码与仿真/数学原型/` 的核心模块：`planar_arm.py`、`inverse_kinematics.py`、`numerical_ik.py`、`continuous_ik.py`、`joint_limits.py`、`dynamics.py`、`trajectory_planner.py`、`path_trajectory.py`、`planning_results.py`。
- `代码与仿真/桌面应用/` 的 `app.py`、`workspace.py`、`simulation_plot.py`、`experiment_io.py`、`experiment_panel.py`、`path_trajectory.py`。
- 全部 `test_*.py`：143 + 102 项测试是当前项目的回归基线。
- `代码与仿真/数学原型/visualize_ik_path_risk.py`、`path_risk_validation.py`、`write_path_risk_validation_report.py`：风险看板与验证链。
- `.venv/`：本地环境，已被 `.gitignore` 忽略；**不是项目清理对象**，不要删。

### 6.3 文档

- `docs/PROJECT_MASTER_PLAN.md`：当前执行级总规划，245 项测试的口径与之一致。
- `设计总档案.md`：设计决策与架构记录。
- `CHANGELOG.md`：最可靠的事实来源之一，记录了修复前后差异。
- `docs/CONTINUOUS_IK.md`、`docs/SINGULARITY_APPROACH_SWEEP.md`、`docs/PATH_LEVEL_RISK_VISUALIZATION.md`、`docs/逆动力学模块设计.md`：当前技术文档。
- `代码与仿真/数学原型/README.md`、`代码与仿真/桌面应用/README.md`：当前使用说明（桌面 README 的“当前边界”需按 P2-5 修正）。

---

## 7. 建议执行顺序（Runbook）

> 以下命令是建议稿，**尚未执行**。执行前请先确认 P0/P1 范围。

### 第 0 步：安全垫

当前有 57 项未提交/未跟踪变更，先不要删任何东西。

```powershell
cd "D:\Project\机械臂交互式仿真平台"
git status --short
git switch -c cleanup/20261024
git add -A
git commit -m "chore: checkpoint before asset cleanup"
```

如果不愿提交，至少把仓库（排除 `.venv`、`.git`）压缩一份到仓库外。**不要**用 `git clean -fdx` 代替人工清理。

### 第 1 步：P0 删除（先 `-WhatIf` 预览）

```powershell
$root = "D:\Project\机械臂交互式仿真平台"

# 1) 备份文件
Get-ChildItem -LiteralPath "$root\代码与仿真" -Recurse -File |
  Where-Object { $_.Name -match '\.(bak|bak_.*|pre_.*)$' } |
  ForEach-Object { Remove-Item -LiteralPath $_.FullName -WhatIf }

# 2) 重复自动批次：保留最新一个
Get-ChildItem -LiteralPath "$root\数据与实验\自动实验批次" -Directory |
  Where-Object { $_.Name -ne '桌面运行-20261022-173413-697325' } |
  ForEach-Object { Remove-Item -LiteralPath $_.FullName -Recurse -WhatIf }

# 3) 其余 P0
Remove-Item -LiteralPath "$root\数据与实验\算法对比演示包_奇异位形附近.zip" -WhatIf
Remove-Item -LiteralPath "$root\代码与仿真\桌面应用\_patch_export_all_plots.py" -WhatIf
Remove-Item -LiteralPath "$root\启动仿真.exe.lnk" -WhatIf
Remove-Item -LiteralPath "$root\.dsh-acl-reports" -Recurse -WhatIf
```

确认输出无误后，去掉 `-WhatIf` 再执行。删除后立即跑第 4 步的测试。

### 第 2 步：P1 归档

```powershell
$root = "D:\Project\机械臂交互式仿真平台"
New-Item -ItemType Directory -Force "$root\数据与实验\归档"
New-Item -ItemType Directory -Force "$root\docs\archive"

Move-Item -LiteralPath "$root\数据与实验\奇异位形附近DLS基线实验\工作空间边界压力测试" `
  -Destination "$root\数据与实验\归档\工作空间边界压力测试"

Move-Item -LiteralPath "$root\数据与实验\奇异位形附近DLS基线实验\目标扰动分支与可达性压力测试" `
  -Destination "$root\数据与实验\归档\目标扰动分支与可达性压力测试"

Move-Item -LiteralPath "$root\数据与实验\算法对比演示包_奇异位形附近" `
  -Destination "$root\数据与实验\归档\算法对比演示包_奇异位形附近"

Move-Item -LiteralPath "$root\docs\UI_PROTOTYPE.md" -Destination "$root\docs\archive\UI_PROTOTYPE_2026-10.md"
Move-Item -LiteralPath "$root\docs\CURRENT_CAPABILITY_AUDIT.md" -Destination "$root\docs\archive\CURRENT_CAPABILITY_AUDIT_2026-10.md"
Move-Item -LiteralPath "$root\docs\superpowers\plans\2026-10-22-path-risk-label-validation.md" `
  -Destination "$root\docs\archive\2026-10-22-path-risk-label-validation.md"
```

归档后：

- 在 `数据与实验/归档/ARCHIVE_README.md` 写清每个目录的替代关系；
- 在 `docs/archive/ARCHIVE_README.md` 写清每份文档的归档原因；
- 检查 `README.md`、`PROJECT_MASTER_PLAN.md`、`设计总档案.md` 中是否还有指向旧路径的链接。

### 第 3 步：P2 代码与文档

建议拆成 5 个提交：

1. `refactor: remove unreachable algorithm comparison branch`
2. `refactor: remove unused pose-anchor scaffolding`
3. `refactor: remove unused dialogs and methods`
4. `docs: align capability and boundary statements`
5. `fix: correct experiment metadata and hard-coded roots`

每个提交后跑：

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
.\.venv\Scripts\python.exe -m unittest discover -s "代码与仿真\数学原型" -p "test_*.py"
.\.venv\Scripts\python.exe -m unittest discover -s "代码与仿真\桌面应用" -p "test_*.py"
```

### 第 4 步：验收

- 数学原型 143 项、桌面应用 102 项通过；
- 70 个非备份 `.py` 全部 `py_compile`；
- `app.py` 离屏启动无异常；
- `git status` 只剩预期变更；
- `git diff --check` 通过；
- 归档后的旧路径不再被任何“当前文档”引用（历史报告中的引用可保留，但要注明“历史路径”）。

---

## 8. 需要你确认的 5 件事

1. **P0 + P1 是否按本报告执行？** 我建议先执行 P0，再执行 P1；P0 是真正释放空间，P1 主要是降噪。
2. **`阻尼参数系统扫描`（11.9 MB）保留还是移出仓库？** 它不是重复数据，但与 `阻尼参数敏感性` 功能重叠。
3. **`docs/PROJECT_GOAL_AND_RESEARCH_PLAN.md` 是否先合并“阶段 C 实体验证”再归档？**
4. **`算法差异验证报告_20261023.md` 是加版本注记，还是重写结论？** 我建议只加注记，保留决策轨迹。
5. **`explore_jacobian.py` 是否保留？** 它是 506 B 的教学脚本，删了不影响任何数据或测试。

确认后，我可以按 P0 → P1 → P2 的顺序逐批执行，并在每批后跑测试；P3 则按你的选择处理。

---

## 附录 A：P0 执行清单

- [ ] 删除 13 个 `.bak` / `.pre_*` 备份
- [ ] 保留 1 个自动批次，删除其余 12 个
- [ ] 删除重复 ZIP
- [ ] 删除 `_patch_export_all_plots.py`
- [ ] 删除 `启动仿真.exe.lnk`
- [ ] 删除 `.dsh-acl-reports/`
- [ ] `.gitignore` 增加 `*.bak_*`、`*.pre_*`、`.dsh-acl-reports/`
- [ ] 跑 143 + 102 项测试
- [ ] `py_compile` 70 个文件
- [ ] `git status` 复核

## 附录 B：关键证据索引

| 结论 | 证据位置 |
|---|---|
| 13 个备份、1,545,261 B | 本报告 P0-1 清单；目录逐文件核对 |
| 13 个自动批次配置同哈希 | 3 份 `experiment_config.json` 的 SHA-256 分组结果 |
| ZIP 与文件夹逐文件相同 | 5 个条目的 SHA-256 对比 |
| 旧工作空间测试与新复测等价 | `工作空间边界汇总.csv` 的 `outer_98pct` 行；`run_ik_workspace_boundary_stress.py:18,70` |
| 扩展数据是路径风险看板输入 | `docs/PATH_LEVEL_RISK_VISUALIZATION.md:9` |
| 新复测修复 41.7% 失败 | `目标扰动..._阻尼上限1.0复测_20261022/压力测试汇总.json` |
| compare 分支不可达 | `app.py:648,674,762` |
| 姿态锚点半成品 | `app.py:981–1065,174,2613,3115` |
| AxisSelectionDialog 无实例化 | `simulation_plot.py:146–211`；`app.py:60` 仅导入 |
| 未使用 import | AST 检查结果，见 P2-4 |
| PCHIP 元数据过时 | `app.py:1581`；`trajectory_planner.py:7`；`01_参考姿态/experiment.json:123` |
| 测试数字 79/55 过时 | `docs/CURRENT_CAPABILITY_AUDIT.md:7–8`；`CHANGELOG.md:188` |
| UI_PROTOTYPE 过期 | `docs/UI_PROTOTYPE.md:5,43` |
| 2026-10-23 报告结论部分失效 | 报告 `:43–44`；`CHANGELOG.md:92` |

---

**报告结束。**

> 再次说明：本报告除新增自身外只读完成，未执行任何删除或移动。P0/P1 的具体执行需要你的确认。
