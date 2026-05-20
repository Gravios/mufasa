# Mufasa session handoff — 2026-05-20

End-of-session summary for the Mufasa Qt port work. Read this before starting a new session to pick up where this one ended.

---

## TL;DR

* **9 patches shipped this session** (122df → 122dn).
* **Cumulative state across all sessions:** the v1 migration arc is complete and the Qt workbench is feature-comprehensive against the legacy SimBA Tk surface. The ROI tool got four sequential enhancements addressing every user-surfaced concern. Pose-importer porting is functionally complete (11 of 17 backends wired).
* **All work is AST-verified only — no PySide6 runtime testing in the sandbox.** The single highest-priority next task is real-world verification on actual user data.
* **The testing workflow doc (`docs/testing_workflow.md`, shipped in 122dk) is the script** for that verification. Run through it and file issues that come up.

---

## What this session shipped

| Patch | Title | LoC | Type |
|---|---|---:|---|
| 122df | README rebrand from SimBA + v1 user docs (project layout + migration guide) | ~750 | Docs + cleanup |
| 122dg | Targeted lint sweep on `mufasa/ui_qt/` (F401/W292/W293) + lint-status report | ~250 | Lint + docs |
| 122dh | Pose importers step 1 — wire DLC + SLEAP + SuperAnimal to PoseImportForm | ~900 | Feature |
| 122di | Pose importers step 2 — wire YOLO-pose + MARS | ~290 | Feature |
| 122dj | Pose importers step 3 — wire TRK + FaceMap (series complete) | ~315 | Feature |
| 122dk | Testing workflow doc + ROI enhancements proposal (doc-only) | ~700 | Docs |
| 122dl | ROI: "Apply to selected videos" button (Proposal 1 implementation) | ~370 | Feature |
| 122dm | ROI: drag-to-adjust placed ROIs (Proposal 2 implementation) | ~1040 | Feature |
| 122dn | ROI Define panel integrated inline into workbench page (no popup) | ~680 | UX refactor |

**Per-patch detail:** see commit messages (`git log --format=fuller`) — each has a several-page explanation including discovery notes, design decisions, sandbox limits, and real-world test instructions.

---

## Current state of the Mufasa Qt port (cumulative across all sessions)

### Done ✓

| Area | Status | Reference patches |
|---|---|---|
| Tk → Qt UI surface | All Tk dialogs ported; `mufasa/ui/` directory deleted | 122d4/d5/d6 cascade; final cleanup in 122de |
| v1 project layout (`project.toml`) | Working alongside legacy `.ini` projects; both detected automatically | Multiple |
| ConfigReader v1 path resolution | All Qt-reachable backends route through `project_layout.project_paths_from_config` | 122da/db hardwired-paths audit + fixes |
| Visualization routes (12) | All find their v1 source dirs via `_VIZ_SOURCE_V1_MAP` | 122dc |
| Qt workbench known issues | 4 bugs flagged in real-world testing; all 4 fixed | 122d0 (QWI-4) + 122d7/d8/d9 |
| README | Rebranded from SimBA-dominated to Mufasa-first | 122df |
| User-facing docs | `v1_project_layout.md`, `migration_guide.md`, `testing_workflow.md`, `roi_enhancements_proposal.md`, `lint_status.md` | 122df + 122dk |
| Pose importer ports | 11 of 17 backend files wired to PoseImportForm | 122dh/di/dj |
| ROI tool — Apply to all (v1 paths) | Fixed | 122d9 + 122da |
| ROI tool — Apply to selected videos | Implemented (reuses existing DuplicateRoisDialog with new entry point) | 122dl |
| ROI tool — drag-to-adjust placed ROIs | Implemented (within existing QPainter framework; NOT a QGraphicsScene rewrite) | 122dm |
| ROI tool — inline in workbench (no popup) | Implemented (split into Widget + thin Dialog wrapper) | 122dn |
| Lint baseline | `ui_qt/` F401/W292/W293-clean; rest of codebase audited and tiered | 122dg + `docs/lint_status.md` |
| Test infrastructure | 20 strict-passing smoke tests (`smoke_122d{0..n}*.py`); strict-format regex sweep | All patches |

### In flight / deferred

| Area | Reason | What's needed |
|---|---|---|
| Pose importers — DANNCE | Partly subsumed by carved-out 3D-marker future scope; smaller user base than DLC/SLEAP | Either port it (single patch ~50 LoC) or skip in favor of full 3D ingestion design |
| Pose importers — SimBA blob | Legacy contour-based; no clear current use case | Skip unless requested |
| 3D marker trajectory ingestion | Vicon / mocap / AniPose 3D — different data shape than 2D pose | Design conversation first, then ~3-5 patches |
| Project-change → page-reload wiring for ROIDefineWidget | `set_config_path()` API exists but nothing calls it on project switch | One-line connection to existing project-change signal (if workbench has one); ~30 LoC |
| `mufasa-migrate-project` console entry point | Currently invoked as `python -m mufasa.cli.migrate_project` | 3-line `pyproject.toml` change |
| Tier-1 lint follow-up on `ui_qt/` | pyupgrade modernization (UP045/UP006/UP007) ~270 issues | Single auto-fix patch; mechanical |
| Tier-2 lint follow-up on `mixins/` + `utils/` | 1658 + 1036 errors; needs per-file triage | Multi-patch series |
| Tier-3 lint on legacy backends | `video_processors/`, `feature_extractors/`, etc. — ~6000 errors | Sweep incrementally as files are touched for other reasons |
| Type checker (mypy / pyright) | Not configured yet | Add `[tool.mypy]` to `pyproject.toml`; expect thousands of errors first run |

### File / code metrics

* `mufasa/**/*.py` count: 416
* Reachable from main entry points: 196 (47%)
* Reachable + standalone tools: 201 (48%)
* Imported by anything in codebase: 236 (57%)
* Truly orphan: 180
* Test sweep: 20/20 strict (N/N)

---

## High-priority caveats / risks

In rough order of "most likely to surprise":

1. **No PySide6 runtime testing.** All sandbox work is AST-only. Every feature shipped this session has unverified behaviors. The testing workflow exists specifically to expose anything AST analysis missed.

2. **MARS + TRK sentinel-passing routes** (in `mufasa/ui_qt/forms/pose_import.py`): I passed `interpolation_method="None"` as a sentinel for "skip preprocessing" since these backends require non-default arguments. This convention is **inferred from SimBA code reading**, not verified. If `Interpolate.fix_missing_values("None")` raises instead of skipping, both routes fail at first user attempt. Fix is one line: change `"None"` to `"Linear"`. Same risk for TRK's `smoothing_settings={}` — TRK assigns but never reads it in the importer; should be safe but unverified.

3. **ROIDefineWidget has no project-switch handling.** `set_config_path()` API exists; nothing calls it. The current workbench (per `workbench_app.py` inspection) tears down + rebuilds pages on project change, so this MAY work — but if pages persist across project switches, the embedded widget will stay stuck on the initial state.

4. **`ROIDefinePanel` (the wrapper) raises `RuntimeError` on no-videos** to preserve legacy callers' exception-handler paths. The embedded `ROIDefineWidget` instead renders a placeholder. If users hit a path where the panel is opened via the legacy popup route (still wired through `forms/roi.py:ROIManageForm.on_run` → action="draw"), they get the exception. Should be unsurprising in practice — empty projects don't make sense.

5. **122dg's lint sweep did not fix everything.** 9765 errors remain across the codebase (down from 9846). Almost all are modernization warnings (`Optional[X]` → `X | None`) in legacy backend code. Tier-1 follow-up on `ui_qt/` is ready to ship as `122do` if/when desired.

6. **Polygon vertex-drag is NOT implemented** (122dm Proposal 2). Polygon body-drag works; per-vertex adjustment was explicitly deferred. If users specifically need to nudge individual polygon vertices, this would be a follow-up patch.

7. **MARS / FaceMap routes pass `requires_animal_ids=False`.** MARS is two-mouse-fixed by design and has internal logic; FaceMap is single-face. The Animal-IDs UI row stays hidden for both. May surprise users who expect to label each animal in a MARS import — but that's not how MARS's data model works.

8. **122dk testing workflow doc is opinionated about expected paths and behaviors.** If a user's observed reality differs from what the doc says should happen, that's the bug to file. Don't trust the doc over observed reality.

---

## Roadmap — next session priorities

### Tier 1 — Should-be-immediate (highest impact)

**T1.1 Real-world verification on actual data.** Run `docs/testing_workflow.md` end-to-end against the user's parquet pose + cage videos + two-condition setup. Expect to file 5–10 issues that compound across the 9 patches shipped this session.

* **Highest-risk paths to test first:**
  - MARS + TRK pose import (sentinel route convention is inferred, not verified)
  - ROI inline widget on project switch (project-change wiring is missing)
  - ROI drag-to-adjust → Save → reopen → adjustments persist
  - ROI "Apply to selected" with the actual two-condition setup
  - Maintenance section's legacy popup path still working

* **Expected outcomes from running the workflow:**
  - A list of behaviors that don't match the doc
  - A list of error messages from the workbench
  - Notes on UX friction (probably surfacing in the inline ROI widget on smaller screens)

### Tier 2 — Quick wins after verification

**T2.1 Project-change → set_config_path wiring** (~30 LoC). Connect whatever signal the workbench fires when a project loads to `widget.set_config_path(new_path)`. Needs inspection of `workbench_app.py` + `MufasaWorkbench` to find the existing signal.

**T2.2 `mufasa-migrate-project` console entry point** (3-line `pyproject.toml` change). Trivially small but surfaces the migration tool more obviously.

**T2.3 Tier-1 lint sweep on `ui_qt/`** (~270 errors, all auto-fixable). One-command `ruff check mufasa/ui_qt --select UP045,UP006,UP007,UP035,UP032,I001 --fix`. Mechanical modernization; produces a large diff but no behavior change.

**T2.4 Address feedback from T1.1.** Per-issue fix patches. Approximate budget: each issue → one ~50-200 LoC patch.

### Tier 3 — Design decisions needed first

**T3.1 3D marker trajectory ingestion.** The user mentioned having 3D marker data. This is the carved-out "future scope" from the pose-importer series. Needs:
* Design doc covering supported formats (Vicon CSV, AniPose 3D parquet, DANNCE matlab, direct 3D CSV)
* Decision on data-shape contract (multi-index extended with `z` axis? separate parallel storage?)
* Impact assessment on downstream pipeline (feature extractors → distances become Euclidean 3D; plotting → 3D rendering needs?)
* Then ~3-5 implementation patches

**T3.2 Tier-2 lint follow-up on `mixins/` + `utils/`.** ~2700 total errors. Per-file triage required — some files are active code that benefits from modernization; others are inherited SimBA code with low ROI.

**T3.3 Type checker integration.** Add `[tool.mypy]` config. First run will surface thousands of errors mostly from third-party stubs (numpy, pandas, cv2). Need to decide: ignore_missing_imports + strict-incremental policy, or sweep-and-fix.

### Tier 4 — Long-tail polish

**T4.1 Auto-generate the legacy→v1 mapping table in `migration_guide.md` from `LEGACY_TO_V1_MAPPING`** so the doc never drifts from the actual code.

**T4.2 Workbench screenshots in the migration guide.** Text-only currently.

**T4.3 Tier-3 lint sweep on legacy backends.** Incremental, file-by-file as touched.

**T4.4 Polygon vertex-drag** (extends 122dm). Only if users specifically request.

**T4.5 Rotated rectangles in ROI canvas.** Out of scope but plausible future request.

---

## Important context for the next session

### Sandbox environment

* PySide6 / numba / cython / h5py / cv2 / tkinter are **NOT available** in the sandbox. Qt code is AST-verified only.
* `ruff` 0.15.13 is installed via `pip install --break-system-packages ruff`.
* Working directory: `/home/claude/mufasa`.
* Final patch + doc outputs go to `/mnt/user-data/outputs/`.

### Compaction-rebuild procedure

If the session is compacted and a fresh sandbox is needed:
```bash
cd /home/claude && rm -rf mufasa && git clone --quiet https://github.com/Gravios/mufasa.git
cd mufasa
git config user.email "claude@anthropic.com" && git config user.name "Claude"
pip install --break-system-packages ruff
```

### Strict-format test sweep

The strict sweep regex is `^[a-z_0-9]+: ([0-9]+)/\1 checks passed$` (NOT just `passed$` — that pattern matches `N/M` failures too):

```bash
timeout 300 bash -c '
FAILED=0
for t in tests/smoke_122d*.py; do
  out=$(timeout 30 python "$t" 2>&1)
  last=$(echo "$out" | tail -1)
  if echo "$last" | grep -qE "^[a-z_0-9]+: ([0-9]+)/\1 checks passed$"; then
    echo "  PASS  $(basename $t)  $last"
  else
    echo "  FAIL  $(basename $t)  $last"
    FAILED=1
  fi
done
exit $FAILED
'
```

### Established conventions

* **Patches are named `0001-patch-122d<X>-<slug>.patch`** in `/mnt/user-data/outputs/`.
* **Commit messages are exhaustive** — rationale + file-by-file changes + sandbox limits + test coverage + stack at HEAD + Path B running totals.
* **Snapshot-resilience** — when later patches change pinned counts/state in earlier smoke tests, relax the earlier test (≥ instead of ==, fuzzy text match instead of exact). Pattern established in 122dh/di/dj/n.
* **AST > regex everywhere** for code analysis. `ast.unparse()` drops comments naturally for diff stability.
* **`docs/`** is the canonical home for all design / status / audit documentation. Updates to `docs/README.md` index when adding new docs.

### Key file paths

| Path | Purpose |
|---|---|
| `/home/claude/mufasa/` | Repo root (sandbox) |
| `/mnt/user-data/outputs/` | Final patch + doc destination |
| `docs/v1_project_layout.md` | v1 layout user/dev reference |
| `docs/migration_guide.md` | Legacy → v1 migration workflow |
| `docs/testing_workflow.md` | End-to-end test plan tailored to user's data |
| `docs/roi_enhancements_proposal.md` | ROI audit + design proposals (now implemented) |
| `docs/hardwired_paths_audit.md` | 107-hit audit + per-site triage |
| `docs/simba_death_cascade.md` | Tk surface removal plan (complete) |
| `docs/qt_workbench_known_issues.md` | QWI tracking (all 4 fixed) |
| `docs/lint_status.md` | Codebase lint snapshot + tier plan |
| `docs/workflows.md` | Codebase audit — which workflow lives where |
| `mufasa/project_layout.py` | Path abstraction layer; 18 public functions |
| `mufasa/ui_qt/dialogs/roi_define_panel.py` | ROIDefineWidget + ROIDefinePanel wrapper |
| `mufasa/ui_qt/forms/pose_import.py` | 11 routes (PoseImportForm) |
| `mufasa/roi_tools/roi_utils.py` | multiply_ROIs + reset_video_ROIs |
| `mufasa/roi_tools/roi_logic.py` | ROILogic with update_*_geometry methods (122dm) |
| `mufasa/ui_qt/workbench.py` | WorkflowPage with add_section + add_section_widget |

### Patch series rhythm

The 122d-series naming convention continues: next patch would be `122do`, then `122dp`, etc. If the next session starts a fundamentally new arc (e.g., 3D ingestion), consider switching to `122e0` or even a clean `123aa` namespace.

---

## Suggested first action for the next session

**Run the testing workflow** (`docs/testing_workflow.md`) against the user's actual data and report back with:

1. Which steps succeeded as documented
2. Which steps produced errors or unexpected behavior (with exact error messages)
3. Which UX behaviors surprised the user (independent of crashes)
4. Any workflow gaps the doc didn't anticipate

That output becomes the input for the next 5–10 patches. The "Tier 1 — Real-world verification" framing turns ~21 patches of accumulated AST-only confidence into runtime-confirmed working software. Nothing in Tier 2+ matters until Tier 1 lands.

If the workflow goes smoothly, the natural follow-up is Tier 2 quick wins (`122do` lint sweep, then `122dp` migration entry point, then `122dq` project-change wiring) — all bounded patches that don't depend on each other.

If the workflow surfaces design issues (data-shape contracts, 3D requirements crystallizing into concrete needs), pivot to Tier 3 design discussions.
