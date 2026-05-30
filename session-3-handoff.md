# mufasa session-3 handoff

**Date:** May 30, 2026
**Last commit:** `42e8791` (patch 122fo — per-family 'already computed' badge)
**Strict smoke tests passing:** 76/76
**Cumulative session-3 patches:** 62 (`122df` → `122fo`)

---

## Quick state

| | |
|---|---|
| Working tree | `/home/claude/mufasa` (or your local checkout) |
| Output dir | `/mnt/user-data/outputs/` (one `.patch` file per commit, presented via `present_files`) |
| Test sweep cmd | `for t in tests/smoke_122d*.py tests/smoke_122e*.py tests/smoke_122f*.py; do …`(see below) |
| Total SECTIONS | 16 |
| ui_bound sections | 13 |
| Unique section_ids wired on forms | 7 |
| Sections with `detect_path` | 11 |
| Sections with `content_predicate` (new in 122fk) | 2 |

---

## How this session worked (preserve in next session)

### Patch + smoke + ship loop

For every user request:
1. **Source edit** in the appropriate module(s)
2. **AST-based smoke test** at `tests/smoke_122f<x>_<name>.py` (PySide6/cv2/h5py unavailable in sandbox → no runtime tests of UI code)
3. **Sweep**: rerun every existing strict smoke to catch reciprocal tripwires that the change invalidated
4. **Flip tripwires** that fired (pinned counts/strings that changed)
5. **Commit** with a structured multi-section message (see "commit message structure" below)
6. **`git format-patch -1 -o /mnt/user-data/outputs/`** + **`present_files`**

### Strict smoke format

Every smoke prints a final line matching this regex:
```
^[a-z_0-9]+: (N)/\1 checks passed$
```

Sweep command (the canonical pass-check):
```bash
for t in tests/smoke_122d*.py tests/smoke_122e*.py tests/smoke_122f*.py; do
  out=$(python "$t" 2>&1)
  last=$(echo "$out" | tail -1)
  if echo "$last" | grep -qE "^[a-z_0-9]+: ([0-9]+)/\1 checks passed$"; then
    :
  else
    echo "FAIL: $(basename $t) — $last"
  fi
done
echo "(done) strict tests: $(ls tests/smoke_122d*.py tests/smoke_122e*.py tests/smoke_122f*.py 2>/dev/null | wc -l)"
```

### AST-extraction pattern (sandbox-safe testing)

Because PySide6/cv2/h5py aren't importable in this sandbox, smokes verify behaviour by:
- `ast.parse(open(f).read())` → walk to find class / function nodes
- `ast.unparse(node)` to get source-as-string, then substring-check
- **`ast.unparse` emits SINGLE quotes for string literals** — when checking raw `.read_text()` source, accept BOTH `'.parquet'` and `".parquet"` (quote-style gotcha that bit smoke_122fm)
- Pre-compute "reference implementations" inside the smoke to validate algorithm correctness (the smoke_122fi transition-jump pattern, smoke_122fn bridge-truth-table pattern)

### Reciprocal tripwires

Many smokes pin counts/strings that other patches change. When a patch invalidates a sibling's pin, the sweep surfaces it; flip to the new expected value AND update the comment to record the date of the change ("…flipped post-122fX because…").

Common ones to watch:
- `len(SECTIONS)` count
- `len(wired_ids)` / "X of Y wired forms"
- `len(ui_bound)` count
- `len(with_detect)` count
- Section title strings (e.g., "Frame labelling" vs "Frame Labeling")
- Form class member lists / counts

### Commit message structure (the convention)

Multi-section format:
```
patch 122fX — one-line summary

User request (date):
> verbatim quote of what the user asked for (if any)

CONTEXT / WHY
=============
brief problem statement

WHAT THIS PATCH LANDED
======================
file-by-file, what changed

WHAT THIS PATCH DID NOT CHANGE
==============================
explicit notes on what was deliberately out of scope

DESIGN DECISIONS / TRADE-OFFS
==============================
choices a future reader would otherwise have to reconstruct

RECIPROCAL FLIPS
================
list of which sibling smokes were flipped and why

NEW SMOKE: <name>.py (N checks)
* bullet per check group

N strict-format smoke tests pass (M + 122fX).
Cumulative session-3: K patches (122df → 122fX).

CLASS-LEVEL OBSERVATION  (optional, when a reusable pattern emerged)
========================
```

---

## What landed in session-3 (theme summary)

Patches 122df → 122fo, grouped by theme:

### Project layout + docs (122df, 122dg, 122dk, 122do, 122dp→122dx removal cycle)
- README rebrand from SimBA, v1 user docs, lint sweep, console-entry migration tool (later removed)
- Removed legacy Tk chooser, ConfigReader `_is_v1` branch, deprecated shims

### Pose importer wiring (122dh, 122di, 122dj)
- 7 new pose-format importers wired to `PoseImportForm` (DLC h5/csv, SLEAP, SuperAnimal, YOLO-pose, MARS, TRK, FaceMap)

### ROI tools (122dl, 122dm, 122dn, 122fb, 122fd, 122fe)
- "Apply to selected videos" button, drag-to-adjust placed ROIs, inline-not-popup ROI panel
- Removed standalone ROI Maintenance section (folded into Definitions popup button)
- **122fd: fixed the live ROI move-bug** (dead-code guard at `roi_canvas.py` mouseMoveEvent line ~875 was missing `SHAPE_MOVING` / `HANDLE_DRAGGING` from allowlist). Reported 3 times before being found — class-level lesson: "look harder when the user repeats themselves; trace every state-change including guards."
- 122fe: Manage classifiers redesigned as `QTableWidget` (Name | Key | Info | Delete, with "+" add)

### Section provenance + badge system (the session backbone)
- **122ds, 122dt:** foundation (`section_provenance.py` module, `publish_to_stage`, `record_run`)
- **122du:** section-status badges on `QToolBox` section headers
- **122dr:** content-aware run resolution (empty runs no longer shadow populated ones)
- **122ei:** filesystem-evidence fallback (`detect_path`) — implicit badge for projects pre-dating provenance wiring
- **122el, 122em:** section-binding audit + drift-prevention test
- **122en:** centralized v1 layout knowledge in `v1_project_paths`
- **122eo, 122fl:** distinguish KeyError vs runtime in `_record_provenance`; **122fl surfaces failures to the user via the Done dialog** (no more silent white badges)
- **122ep, 122ew, 122ez, 122es-hotfix, 122fc:** detect_path coverage extensions
- **122fk:** **smarter `detect_path` semantics** — new `content_predicate` field on `SectionSpec` (gates whether the evidence counts; replaces 122fc's sentinel-path workaround)
- **122ex-hotfix, 122fj:** wired `section_id` on `EgocentricAlignmentForm`, `ROIFeaturesForm`, `FeatureSubsetExtractorForm`

### Classifier page reshape (122ey, 122fa)
- Split into 6 standalone tabs (Train, Validate, Inference, Manage…)
- Renamed for clarity; YOLO merge

### Frame labelling redesign (122ff, 122fg, 122fh, 122fi)
- Playback keys + continuous-label/delete state (per-classifier hotkey toggles a "writing 1s to X" mode; arrow keys play forward/back and write to active label)
- New `LabelTimeseriesPlot` widget (custom paint, ±2s window, no matplotlib)
- Up/Down → playback rate ×1.25 / ÷1.25 (symmetric, clamped to [1, 240] fps)
- Page Up/Down → jump to prev/next label transition in active label
- "Frame labelling" → "Frame Labeling" (user-facing strings only; class name and module path kept)

### Stray-file defense (122eu-hotfix consumer side; 122fm producer side)
- **122eu-hotfix:** `get_fn_ext` handles extension-less files (was crashing on `SKIPPED` sentinels in run dirs)
- **122fm:** audit of every data-dir enumeration site. Found ONE gap — pose_import's snapshot symlinked stray files into run dirs. Fixed with `_POSE_DATA_EXTS` extension filter at the source. **Class-level invariant established: never assume every file in a data dir is a data file.**

### Sibling-API drift resolution (122et-hotfix live; 122fn dormant)
- **122et-hotfix:** `ROIPlotMultiprocess` accepts `show_bbox` (Qt form had been passing single-core kwarg to MP backend → TypeError crash on workers>1)
- **122fn:** applied the same shim pattern to two dormant pairs (`plot_clf_results` / `single_run_model_validation_video`). All three sibling pairs now consistent. Smokes pin against future re-drift.

### Features UI (122fj, 122fo)
- **122fj:** section-level badges for ROI:Features + Features:Compute feature subset (white/green via `record_run`)
- **122fo:** per-family "already computed" checkmark badges in the feature-subset selector lists (filesystem-truthful, complementary to the section badge)

---

## Current SECTIONS inventory

| section_id | page | ui_bound | detect_path | predicate | wired form |
|---|---|---|---|---|---|
| `import_pose` | Data Import | ✓ | ✓ | ✓ | yes |
| `import_video` | Data Import | ✓ | ✓ | ✓ | (record_run only via pose-import publish) |
| `pixels_per_mm` | Preprocessing | ✓ | ✓ | | no |
| `interpolate` | Preprocessing | ✓ | ✓ | | yes |
| `kalman_v2` | Preprocessing | ✓ | ✓ | | yes |
| `outlier_correction` | Preprocessing | ✓ | ✓ | | yes |
| `savitzky_golay` | Preprocessing | | | | |
| `egocentric` | Preprocessing | ✓ | ✓ | | yes |
| `drop_body_parts` | Preprocessing | | | | |
| `roi_definitions` | ROI | ✓ | ✓ | | no |
| `features_subject` | Features | | | | (ui_bound=False placeholder) |
| `features_compute_subset` | Features | ✓ | | | yes (122fj) |
| `features_roi` | ROI | ✓ | | | yes (122fj) |
| `annotation` | Annotation | ✓ | ✓ | | no |
| `classifier_train` | Train | ✓ | ✓ | | no |
| `classifier_run` | Inference | ✓ | ✓ | | no |

---

## Architectural patterns established (preserve)

### 1. Three categories of `detect_path` semantics

Established by 122fk. When wiring a new badge:

| Pattern | When to use | Example |
|---|---|---|
| **Single-section evidence** | "Does THIS section's output exist?" | Most sections — `kalman_v2`, `interpolate`, `egocentric` |
| **Cross-section consistency** (`content_predicate`) | "Do TWO sections agree?" | Data Import bases-match (`import_pose`, `import_video`) |
| **None / `record_run` only** | Output destination shared with siblings; only explicit provenance can disambiguate | `features_roi`, `features_compute_subset` (both write into `derived/features/`) |

The right shape for "evidence exists AND meets a constraint":
```python
detect_path=lambda root: <canonical location>,
content_predicate=<bool-returning callable taking project_root>,
```
NOT `detect_path=lambda root: <conditional return>` (the 122fc workaround pattern, refactored away in 122fk).

### 2. Stray-file defense at BOTH layers

Established by 122eu-hotfix (consumer) + 122fm (producer). Future enumeration sites must:
- Filter by expected extension(s)
- Skip dotfiles (`name.startswith(".")` or `p.name.startswith(".")`)
- Use `os.path.isfile` / `Path.is_file()` guards

When introducing a new data dir, surface a `_<DOMAIN>_EXTS` constant for the valid extensions (cf. `_POSE_DATA_EXTS`, `_POSE_EXTS`, `_VIDEO_DATA_EXTS`).

### 3. Provenance failures must be loud

Established by 122fl. `_record_provenance` returns `str | None`; runtime failures emit `logging.warning` (not `print`, invisible in packaged apps) AND set an error string that `_on_success` appends to the "Done." dialog. A silent white badge is now impossible — the user always learns *why* if it didn't update.

### 4. Sibling-API parity for single-core / `_mp` plotting classes

Established by 122et + 122fn. When a single-core class and its `_mp` sibling expose the same option under different names/types, the MP class accepts the single-core name as a bridged alias so Qt forms can pass identical kwargs to both backends without `TypeError`. Apply this pattern any time a new plotting MP class is added.

### 5. Two-layer status indicators

Established by 122fj + 122fo. Section-level badges (white/green via provenance, freshness-aware via deps) coexist with per-item filesystem badges (truthful for *this specific output right now*). The two layers answer different questions and complement each other.

### 6. Look harder when the user repeats themselves

Established by 122fd. The ROI move-bug was reported THREE times before being found because the cause was inside a guard clause (allowlist missing two states) that I'd glossed over twice. When a bug report repeats, trace EVERY line of the entry → state-change path including guards, allowlists, and short-circuits.

---

## TODO list (prioritized)

All remaining items are lower priority than what shipped this session — none are user-facing bugs. Listed by remaining value.

### Medium value

1. **`light_dark_box_analyzer.py` v1 conversion** (filed 122ev as `ev-hotfix` — partial only)
   - **Problem:** CLI-only analyzer uses `pd.read_csv` + manual DLC multi-index parse + `extensions=['.csv']`. Crashes / wrong output on v1 parquet projects.
   - **Fix:** swap to `read_df(check_multiindex=True)`; support both v0 (csv) and v1 (parquet); honor project `file_type`.
   - **Caveat:** can't validate the data-transformation correctness in this sandbox (pandas/h5py unavailable). Best done in an env where you can run a v1 project through it.
   - **Priority:** functional bug but CLI-only — niche user impact.

2. **Form-title vs `add_section` title drift audit** (standing)
   - **Problem:** 34 deliberate mismatches between `FormClass.title` and the `add_section` title that hosts it. Verify all are intentional; surface any accidental mismatches.
   - **Reality check:** badges attach via `add_section` title → `SECTIONS.section_title`, which IS audited (122el). FormClass.title mismatches are cosmetic, not badge-breaking.
   - **Priority:** verification with likely-no-fix outcome. Value mostly documentary.

### Lower value (refactor / docs)

3. **Constants extraction** (`PAGE_*` / `SECTION_TITLE_*`)
   - Page names ("Preprocessing", "Features", etc.) and section titles are string literals scattered through page builders. Centralize as module constants.
   - Pure refactor; no behaviour change.

4. **`feature_io.py` legacy branches** (standing)
   - Several v0/v1 fork branches that can collapse now that v0 support is dropped.
   - Backend cleanup; AST/structural changes only.

5. **In-importer interpolation deprecation** (standing)
   - Pose importers still have an inline interpolation path that pre-dates the dedicated Interpolate section.
   - Deprecate the inline path; force users through the Interpolate page (with a migration warning for projects that relied on it).

6. **Tk-flow descriptions in `workflows.md`** (standing)
   - Doc references to the (now-removed) Tk chooser flow. Update to reflect the Qt-only path.

7. **Egocentric STALE diagnosis** (informational, deferred by user)
   - The egocentric badge sometimes reads STALE incorrectly when `outlier_correction` evidence is newer than `rotated/` content. Math is correct (STALE *is* the right answer given the freshness graph); user wants the math itself revisited.

---

## Key files quick-reference

### Provenance / badge core
- `mufasa/section_provenance.py` — `SECTIONS` dict, `SectionSpec` dataclass (now with `content_predicate`), `_resolve_run_at`, `_path_mtime_if_has_content`, `_data_import_bases_match`, `record_run`, `get_status`, `get_all_statuses`
- `mufasa/ui_qt/workbench.py` — `OperationForm` base class with `_record_provenance` (returns `str | None`) and `_on_success` dialog

### Frame labeling (the redesigned annotation workflow)
- `mufasa/ui_qt/frame_labeller.py` — `FrameLabellerWidget` (playback / continuous-mode / transition-jump / timeseries integration)
- `mufasa/ui_qt/frame_scrubber.py` — `FrameScrubberWidget` (`_fps` native vs `_playback_fps` user-adjustable)
- `mufasa/ui_qt/label_timeseries_plot.py` — `LabelTimeseriesPlot` (custom-paint, no matplotlib)

### Forms with wired section_id (record_run-based badges)
- `mufasa/ui_qt/forms/pose_import.py` — `import_pose` + `_POSE_DATA_EXTS` (122fm)
- `mufasa/ui_qt/forms/preprocessing.py` — `kalman_v2`, `interpolate`, `outlier_correction`, `egocentric`
- `mufasa/ui_qt/forms/roi.py` — `ROIFeaturesForm.section_id = "features_roi"` (122fj)
- `mufasa/ui_qt/forms/features.py` — `FeatureSubsetExtractorForm.section_id = "features_compute_subset"` (122fj), `_computed_family_slugs` + `_make_family_item` for per-family badges (122fo)

### Sibling-API shim pattern
- `mufasa/plotting/roi_plotter_mp.py` — `show_bbox` alias bridging to `bbox` (122et)
- `mufasa/plotting/plot_clf_results_mp.py` — `show_bbox` + `print_timers` aliases (122fn)
- `mufasa/plotting/single_run_model_validation_video_mp.py` — `show_animal_bounding_boxes` alias (122fn)

### Critical bug fixes referenced often
- `mufasa/ui_qt/dialogs/roi_canvas.py` — `ROICanvas.mouseMoveEvent` (122fd dead-code guard, the move-bug fix)
- `mufasa/utils/read_write.py` — `get_fn_ext` (122eu-hotfix extension-less guard)

---

## Starter prompt for the next session

Copy/paste-ready primer for a fresh session:

```
I'm continuing work on Gravios/mufasa (PySide6 behavioural-analysis workbench).
Session-3 ended at commit 42e8791 (patch 122fo). 76 strict smoke tests pass.

Workflow conventions:
- Each user request → one or more patches named 122f<letter>.
- Per patch: source edit + AST-based smoke at tests/smoke_122f<x>_*.py
  printing "<name>: N/N checks passed" as the final line + multi-section
  commit message + git format-patch -1 -o /mnt/user-data/outputs/ +
  present_files.
- PySide6/cv2/h5py unavailable in sandbox → all smokes are AST-extraction.
- Sweep cmd: see handoff doc.
- Reciprocal tripwires: many smokes pin counts/strings that other patches
  change; when sibling smokes fire after a change, flip the pinned values.

See sessions-3-handoff.md (in /mnt/user-data/uploads/ if attached) for:
- Full inventory of what shipped
- Architectural patterns to preserve (3 detect_path categories, content_
  predicate, stray-file defense at both layers, sibling-API parity, etc.)
- Prioritized TODO list

Top of the TODO: light_dark_box_analyzer.py v1 conversion (CLI-only,
needs pd.read_csv → read_df). Also: form-title vs add_section drift audit.

Next request follows below.
```
