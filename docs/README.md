# Mufasa documentation

Index of documents in `docs/`. Three tracks: **workflow / user-facing**, **Tk → Qt migration & audits** (developer-facing), and **Kalman smoother design** (separate concern). Each doc states its audience and scope in its first lines.

---

## User-facing entry points

If you're new to Mufasa, start here.

### `v1_project_layout.md` (122df)
**v1 project layout reference.** What a v1 project looks like on disk, run-id semantics, the path-abstraction layer (`mufasa.project_layout.project_paths_from_config`) for backend code, and how to create a fresh v1 project.

Read this if you're starting a new project or want to understand how v1 differs from legacy SimBA.

### `migration_guide.md` (122df)
**Migrate legacy → v1.** How to use `python -m mufasa.cli.migrate_project` to move an existing SimBA-layout project to v1. Includes dry-run workflow, troubleshooting, and rollback.

Read this if you have an existing `project_folder/`-style project and want to adopt v1.

### `lint_status.md` (122dg + 122do)
**Codebase lint snapshot + follow-up plan.** Audit of `ruff check` findings across the whole codebase after the 122dg targeted sweep (F401/W292/W293) and the 122do modernization sweep (UP045/UP006/UP007/UP035/I001/F401 cascade). Top-rule breakdown, per-directory disposition, recommended tier-1/2/3 follow-up patches.

Read this when planning the next typing/lint sweep or asking "is this file lint-clean?"

### `testing_workflow.md` (122dk)
**End-to-end test workflow.** Step-by-step verification plan tailored to parquet pose data + cage videos + multi-condition setups. Identifies the spots that need specific manual checks because recent patches changed behaviour vs the legacy SimBA implementation.

Read this when you have a real dataset and want to validate the patch series before relying on it.

### `roi_enhancements_proposal.md` (122dk)
**ROI tool audit + proposed enhancements.** Honest status of the ROI tool's current capabilities (what's wired, what's fixed, what's missing). Two design proposals: "Apply to selected videos" subset-apply (122dk-future) and drag-to-adjust placed ROIs (122dl-future).

Read this when planning the next ROI feature patches or deciding whether the current ROI tool meets a specific user need.

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

---

## Tk → Qt migration & audits

These five docs together cover the v1 migration: the Tk surface that needs to go, the Qt surface that's replacing it, the runtime gaps that block "the Qt workbench is feature-complete", and the backend coupling that blocks the final Tk deletion. They're chained: read them in roughly the order below for the full picture.

### `tk_surface_audit.md`
**Tk-vs-Qt surface inventory and removal plan.** Status-tagged audit of all 96 files in `mufasa/ui/` plus `mufasa/SimBA.py`: which are load-bearing for Qt, which are reachable only via the legacy Tk launcher, which are unreferenced, what migration steps are required to remove the Tk surface entirely.

Updated counts (post-122cm): 8 files dropped via Tier-4 work since the original audit (2 in 122bx + 6 cue-light in 122ck), plus 2 from elsewhere in `mufasa/` (`bounding_box_tools/boundary_menus.py` + `video_processors/batch_process_menus.py` in 122cm). 86 of 96 `mufasa/ui/` files remaining.

Read this when planning UI work or considering a SimBA → Mufasa rename in Tk-only code (short answer: don't — it's slated for removal).

### `tk_to_qt_consolidation_plan.md`
**Comprehensive redesign mapping every Tk popup to its Qt destination.** Lays out the target Qt workbench layout, maps the ~85 Tk popups to existing Qt forms or proposed new sections, identifies the ~7 genuine gaps that need new Qt work, and orders the migration into four tiers (verify → small new sections → unsupervised port → drop + cleanup).

Tier status (post-122cm):
* Tier 1 (verify existing Qt) — ✓ done by 122bt/bu/bv.
* Tier 2 (small new sections) — ✓ done by 122bv close-out.
* Tier 3a (Blob quick-check) — ✓ 122bw.
* Tier 3b (Unsupervised analysis) — pending (multi-day port).
* Tier 4 (drop + cleanup) — in progress; module-level Tk-importer count 25 → 19; 10 file deletions so far.

Read this when planning the next porting patch — it tells you what's left to build and what's already covered.

### `qt_form_runtime_gaps.md`
**Audit of Qt forms whose UI was wired but whose backend raised `NotImplementedError` at runtime.** Originally identified 7 failing operations across 4 forms; all 7 closed in patches 122ca through 122cf. The CLAHE interactive-preview Qt-dialog port (originally tracked as a separate partial-failure case) shipped in 122ci. **No remaining `NotImplementedError` raises in the Qt form surface.**

Read this for the historical record of how each gap was closed, or to understand the per-form fix patterns (form rewrite vs backend addition vs UI redesign).

### `backend_audit.md`
**Backend-side companion to `qt_form_runtime_gaps.md`.** Originally covered (a) the actual availability of "missing" backends (5 of 7 turn out to be findable under different names) and (b) the 25 backend modules that imported from `tkinter_functions.py` (blocking Tier-4 cleanup). Now extended with:

* **§3d "Strategic disposition"** (added 122cm) — four-bucket classification of remaining Tk importers (delete-only / dies-with / deferred / lazy) plus a decision rule for future audits ("Tk use is a single intrusion in a pure-backend file → decouple; file is a Tk surface → wait for parent work item; don't decouple piecemeal"). Supersedes the earlier Group A/B/C/D classification.
* **§4d Backend decoupling progress** — items 11, 12, 13 marked done in 122ch + 122cl; item 14 (annotator_mixin) RECLASSIFIED in 122cm (dies with labelling Qt port, not decouple-able).
* **§4f Tier-4 close-out** — items 17, 18 done in 122cm; item 16 (labelling Qt port) pending.

Read this when planning Tier-4 work or when considering whether a backend file's Tk use is a candidate for decoupling vs delete-with-parent. The decision rule in §3d is the actionable summary.

### `qt_form_registration_audit.md`
**Registration check for OperationForm subclasses.** Confirms every form defined under `mufasa/ui_qt/forms/` is reachable from at least one page's `add_section()` call. All 60 subclasses registered (zero orphans) as of patch 122cf. Includes the reproducible AST script and a regression-guard smoke test that flags any future orphan addition.

Read this when adding a new form to remember to wire it; the smoke test will fail loudly otherwise.

---

## Kalman smoother — design track

These four documents are a separate concern (the Kalman v2 smoother design + audit). Not workflow-related; kept in `docs/` for proximity to the rest of the technical writing.

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
2. **Does it duplicate something here?** Better to extend an existing doc than fork. The Kalman track shows what fragmentation looks like — four docs that conceptually want to be one.
3. **Is it transient or stable?** Migration-arc work belongs in patch commit messages, not docs. Patch 122bf's snapshot-drift fix list isn't in any doc — it's in the commit.
4. **Update `README.md` too** (this file). Keep the index in sync.
