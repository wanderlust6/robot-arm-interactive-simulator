# Path Risk Label Validation Implementation Plan

> **For agentic workers:** Execute task-by-task and verify each step before marking complete.

**Goal:** Quantify how existing path-level heuristic labels relate to independently observable IK stress-test outcomes.

**Architecture:** Add a pure offline analysis module that loads the existing point CSV, derives outcome events independently from display labels, evaluates threshold sweeps and group summaries, and writes a Markdown report plus CSV evidence. Existing source data, dashboard, UI, and IK solver remain unchanged.

**Tech Stack:** Python standard library (`csv`, `statistics`, `json`, `unittest`), existing project virtual environment.

**Spec:** `docs/PATH_LEVEL_RISK_VISUALIZATION.md` and existing `visualize_ik_path_risk.py`.

## Global Constraints

- Do not modify the production IK solver or desktop UI.
- Do not overwrite the source CSV or existing dashboard outputs.
- Treat thresholds as research heuristics, not robot safety limits.
- Do not claim ground-truth classification for heuristic-only events.
- Do not commit or push.

---

### Task 1: Define validation semantics with tests

**Files:**
- Create: `代码与仿真/数学原型/test_path_risk_validation.py`
- Create later: `代码与仿真/数学原型/path_risk_validation.py`

- [ ] Test event classification from raw CSV fields (unreachable, solver failure, error tolerance, near singularity).
- [ ] Test exact confusion-matrix counts and threshold-sweep counts.
- [ ] Run tests and verify they fail because the implementation module is missing.

### Task 2: Implement reproducible offline analysis

**Files:**
- Create: `代码与仿真/数学原型/path_risk_validation.py`

- [ ] Load and validate required columns and numeric values.
- [ ] Compute independent outcome events and current heuristic labels.
- [ ] Sweep sigma thresholds and joint-step thresholds; report only comparisons supported by the data.
- [ ] Aggregate by case/method/repeat and write machine-readable CSV outputs.

### Task 3: Generate report and verify

**Files:**
- Create: `数据与实验/奇异位形附近DLS基线实验/路径级风险可视化/路径风险标签有效性验证.md`
- Update: `docs/PATH_LEVEL_RISK_VISUALIZATION.md` only if needed to document how to rerun.

- [ ] Run unit tests and full analysis.
- [ ] Verify row/sequence counts reconcile with the source CSV and output summaries.
- [ ] Run syntax checks and `git diff --check`.
- [ ] Review the report to ensure heuristic associations are not described as causal or as robot safety certification.