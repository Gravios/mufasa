# Mufasa documentation

Index of documents in `docs/`. Two tracks: **workflow / user-facing** (top section) and **Kalman smoother design** (bottom section). Each doc states its audience and scope in its first lines.

---

## Workflow & usage

### `workflow_audit.md`
**Topological view of the system.** AST inventory of all workbench pages, forms, backend classes, the canonical data-flow pipeline, project-layout schemas, and the plotting backend table. Generated post-patch 122bj.

Read this when you want to know "which class does X" or "what's the data shape at stage Y".

### `data_source_guides.md`
**Per-source import guide.** Twelve pose-data importers (DLC, SLEAP, MARS, TRK, FaceMap, YOLO, SuperAnimal-TopView, SimBA blob), eight third-party annotation appenders, plus a decision matrix mapping data shape → recommended import path.

Read this when you have data and need to get it INTO mufasa.

### `workflow_recipes.md`
**Eleven end-to-end recipes** for common experimental setups: single-animal classification, two-animal social, ROI-bounded behaviors, cue-light experiments, spontaneous alternation, pup retrieval, third-party annotation reuse, unsupervised discovery, YOLO pipeline, heuristic-only, feature subsets for non-ML use.

Read this when you have a research question and need a starting checklist.

### `workflows.md`
**Developer-facing technical audit.** One entry per workflow with status tags (🟢 deeply audited / 🟡 shallow listed / 🔴 known broken / ⚪ not yet examined), salvage notes, branch-by-branch traces, Tk-vs-Qt port coverage.

Read this when you're working on the codebase itself, planning a migration patch, or trying to figure out why a workflow misbehaves.

### `tk_surface_audit.md`
**Tk-vs-Qt surface inventory and removal plan.** Status-tagged audit of all 96 files in `mufasa/ui/` plus `mufasa/SimBA.py`: which are load-bearing for Qt, which are reachable only via the legacy Tk launcher, which are unreferenced, what migration steps are required to remove the Tk surface entirely.

Read this when planning UI work or considering a SimBA → Mufasa rename in Tk-only code (short answer: don't — it's slated for removal).

### `tk_to_qt_consolidation_plan.md`
**Comprehensive redesign mapping every Tk popup to its Qt destination.** Lays out the target Qt workbench layout, maps the ~85 Tk popups to existing Qt forms or proposed new sections, identifies the ~7 genuine gaps that need new Qt work, and orders the migration into four tiers (verify → small new sections → unsupervised port → drop + cleanup).

Read this when planning the next porting patch — it tells you what's left to build and what's already covered (existing Qt is further along than the AST audit alone suggests).

### `qt_form_runtime_gaps.md`
**Audit of Qt forms whose UI is wired but whose backend raises `NotImplementedError` at runtime.** Four forms with seven failing operations: `AverageFrameForm` (entirely broken — spelling mismatch + kwarg shape), `VideoFiltersForm` (3 of 5 ops + CLAHE interactive preview unwired), `CropVideosForm` (multi-crop sub-mode unwired), `DropBodypartsForm` (entirely broken — missing backend), `ROIFeaturesForm` (Remove action unwired).

Read this before launching the Qt workbench to know which buttons will surface errors. Includes recommended stop-gap (disable broken options) plus priority order for backend wiring.

### `backend_audit.md`
**Backend-side companion to `qt_form_runtime_gaps.md`** — two-part audit covering the actual availability of "missing" backends (5 of 7 turn out to be findable under different names, only 2 are genuinely missing) and the 25 backend modules that import from `tkinter_functions.py` (blocking Tier-4 cleanup). Each Tk importer is categorised: dies-with-unsupervised / needs-refactor / simple-popup-replacement / build-infrastructure.

Read this when planning a fix-the-broken-form patch or planning the Tier-4 backend-decoupling work. The Quick Wins section in §4a lists ≤1-hour fixes that close 4 of the 7 runtime gaps.

---

## Kalman smoother — design track

These three documents are a separate concern (the Kalman v2 smoother design + audit). Not workflow-related; kept in `docs/` for proximity to the rest of the technical writing.

### `kalman_smoother_design.md`
Triplet-egocentric covariance Kalman smoother design document (1408 lines, May 2026). Mathematical formulation, state space, EM noise fitting, integration plan. No implementation yet at time of writing — design only.

### `v2_next_steps_design.md`
Follow-up design notes addressing `q_root_pos` ceiling-hit observed in the v2 real-data run. Patches 113–118 context.

### `v2_pipeline_audit.txt`
Output of `audit_v2_pipeline.py` (below). Module structure inventory of the v2 pipeline.

### `audit_v2_pipeline.py`
The script that generated `v2_pipeline_audit.txt`. Re-runnable.

---

## How the workflow docs were produced

The three workflow companions (`workflow_audit.md`, `data_source_guides.md`, `workflow_recipes.md`) were generated via five AST passes over `mufasa/`:

1. Qt workbench page registration — extract pages, sections, form classes from `mufasa/ui_qt/pages/*.py` by walking `add_page(...)` / `add_section(...)` call sites.
2. Pose importers — class inventory from `mufasa/pose_importers/`.
3. Feature extractors — class inventory from `mufasa/feature_extractors/`, separating hardcoded-schema specialty extractors from project-purpose extractors.
4. Models / processors / ROI / outliers — class inventory across `mufasa/model/`, `mufasa/data_processors/`, `mufasa/roi_tools/`, `mufasa/outlier_tools/`, `mufasa/labelling/`, `mufasa/unsupervised/`, `mufasa/video_processors/`, `mufasa/cue_light_tools/`, `mufasa/bounding_box_tools/`.
5. Plotting backends + project layout — class inventory of `mufasa/plotting/`; v1 TOML schema sections + path keys via regex on `mufasa/project_layout.py`.

The recipes are inferred from the resulting class graph + docstring text. They reflect what the codebase makes possible, not necessarily what users actually do. See caveats in each doc.

---

## When to add a new doc

If you're tempted to add a new doc to `docs/`:

1. **Is it user-facing or developer-facing?** Match the convention: lowercase_with_underscores.md, first paragraph states audience.
2. **Does it duplicate something here?** Better to extend an existing doc than fork. The Kalman track shows what fragmentation looks like — three docs that conceptually want to be one.
3. **Is it transient or stable?** Migration-arc work belongs in patch commit messages, not docs. Patch 122bf's snapshot-drift fix list isn't in any doc — it's in the commit.
4. **Update `README.md` too** (this file). Keep the index in sync.
