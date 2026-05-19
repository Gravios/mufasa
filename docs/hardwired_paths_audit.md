# Hardwired-paths audit (patch 122da)

## Background

Mufasa supports two project layouts:

* **Legacy SimBA layout** — `<project>/{videos, csv/input_csv, csv/outlier_corrected_movement_location, csv/features_extracted, csv/targets_inserted, csv/machine_results, logs, models, frames}` etc. Driven by a `project_config.ini` file. Inherited from SimBA.
* **v1 layout** — `<project>/{sources/{videos, pose, annotations, video_info.csv}, derived/{outlier_corrected, features, labels, classifications, frames}, logs, models, project.toml}`. Run-id-scoped subdirectories under `derived/`.

The detection rule everywhere: `config_path.endswith(".toml")` → v1; else legacy.

## The path abstraction layer

**`mufasa.project_layout.project_paths_from_config(config_path)`** returns a dict of layout-agnostic paths. Same rule the rest of the codebase uses.

Public surface (18 functions in `mufasa.project_layout`):

| Function | Role |
|---|---|
| `project_paths_from_config(config_path)` | **Main entry point.** Returns dict with keys: `project_root`, `video_dir`, `input_pose_dir`, `logs_dir`, `video_info_path`, `models_dir`, `machine_results_dir`, `roi_definitions_path`, `derived_features_dir`, `derived_labels_dir`, `derived_classifications_dir` |
| `project_metadata_from_config(config_path)` | Animal count, body parts, file type, etc. |
| `resolve_v1_project_root(config_path)` | Best-effort root resolution |
| `detect_layout(path)` | Returns `"v1"` or `"legacy"` |
| `read_project_toml` / `write_project_toml` | v1 project.toml I/O |
| `read_run_toml` / `write_run_toml` | Per-run provenance files |
| `global_model_cache_dir` | `~/.config/mufasa/models/` |
| `file_sha256` | Model fingerprinting |
| `import_model_into_project` / `mirror_model_to_global_cache` | Model import / cache mirror |
| `read_classifier_inference_settings` / `write_…` | v1 classifier settings I/O |
| `read_classifier_training_settings` / `write_…` | v1 training settings I/O |
| `generate_run_id` / `is_run_id` | Run-id allocation |

**Secondary:** `ConfigReader(config_path=…)` exposes `self.video_dir`, `self.input_csv_dir`, `self.logs_path`, etc. — same layout-agnostic resolution but with the cost of full project metadata read. Use it when you need the broader config alongside paths.

## Scan results

**107 hits** total across `mufasa/**/*.py` for patterns matching `os.path.join(..., "<legacy_subpath>")`, `Path / "<legacy_subpath>"`, and multi-segment legacy path literals. After categorization:

| Disposition | Count | Description |
|---|---:|---|
| DEFINITION | 33 | `project_layout.py` + `config_reader.py` + `legacy_layout.py` + `cli/migrate_project.py`. Intentional — they define the layouts. |
| LEGACY-BRANCH (inside `if not self._is_v1:`) | — | Many hits within DEFINITION files are inside legacy branches; correct. |
| LEGACY-ONLY CLUSTER | 10 | `bounding_box_tools/` — SimBA-only backend, no Qt entry. |
| DEFENSIVE-TRY | 3 | Hardcodes inside `try/except` fallbacks for malformed configs. |
| **POTENTIAL BUG** | **38** | Unconditional joins to legacy subpaths in files that should support either layout. |

## Triage: the 38 potential bugs

### Qt-reachable bugs (require fixing for v1 workflows)

**Note (post-re-triage):** the initial scan flagged 12 sites; closer inspection during 122db reduced this to just **3 actually-broken** sites. The rest were false positives in three categories:

* **`self.project_path` + same-path-both-layouts.** ConfigReader's `self.project_path` is set to the project root for BOTH layouts (v1 root for v1; legacy root for legacy). Subpaths like `logs/` are identical in both layouts (`<root>/logs/`), so `os.path.join(self.project_path, "logs")` is correct for both. Sites: `data_processors/pup_retrieval_calculator.py:83`.
* **Defensive `except` fallbacks.** Several flagged hits were inside `try/except` blocks where the primary path uses `project_paths_from_config` correctly; the `except` branch fires only on malformed configs and uses a best-effort legacy guess. Acceptable. Sites: `roi_logic.py:145, 149`, `video_info.py:265, 291, 292`.
* **Branch-gated legacy paths.** Sites that look like hardcoded legacy paths but are inside `if not v1` / `if v1_root is None` branches where they're meant to fire. Acceptable. Sites: `pose_cleanup.py:1855, 1960`.

| Site | Status | Notes |
|---|---|---|
| `roi_tools/roi_utils.py:462` (multiply_ROIs videos_dir) | ✓ Fixed 122d9 | The original QWI-1 fix |
| `roi_tools/roi_utils.py:474` (multiply_ROIs roi_coordinates_path) | ✓ Fixed 122da | Sibling miss from 122d9 |
| `roi_tools/roi_utils.py:561` (reset_video_ROIs roi_coordinates_path) | ✓ Fixed 122da | Same sibling pattern |
| `video_processors/video_processing.py:2207` | ✓ Fixed 122db | `extract_frames_from_all_videos_in_directory` used `read_config_entry(config, "General settings", "project_path")` (fails on v1 .toml) + hardcoded legacy `frames/input/` subpath. Now uses `project_paths_from_config()["project_root"]` and branches to `derived/frames/extracted/` for v1 |
| `ui_qt/forms/pose_cleanup.py:1318` | ✓ Fixed 122db | Unconditionally set `<root>/csv/smoothed_v2/` as default-output even for v1 projects. Now gated to legacy-only (same as the L1855 sibling) so v1 projects leave the field blank and target()'s run-dir allocator handles it. |
| `ui_qt/forms/visualizations.py:1233` (per-route src_dir) | Deferred | Hardcoded `<root>/csv/<subdir>`. Requires route-metadata refactor (each route should declare its v1 path explicitly). Real bug, but the scope is larger than a one-liner. Tracked for a separate "visualizations v1 routing" patch. |
| `data_processors/pup_retrieval_calculator.py:83` | NOT A BUG (re-triage) | `self.project_path` + `"logs"` — both correct for both layouts. |
| `ui_qt/forms/video_info.py:265, 291, 292` | NOT A BUG (re-triage) | All inside `except Exception:` defensive fallback. Primary path uses helper. |
| `roi_tools/roi_logic.py:145, 149` | NOT A BUG (re-triage) | Same — defensive fallback inside try/except. |
| `ui_qt/forms/pose_cleanup.py:1855` | NOT A BUG (re-triage) | Explicitly gated `if not str(self.config_path).lower().endswith(".toml")` — legacy-only by design. |
| `ui_qt/forms/pose_cleanup.py:1960` | NOT A BUG (re-triage) | Inside `if v1_root is None:` (legacy) branch of a `resolve_v1_project_root` check at L1934. |
| `utils/project_reconfigure.py:96` | NOT A BUG (re-triage) | Legacy-only by design — v1 stores body_parts in `project.toml`, not a separate `bp_names` CSV. This function reconfigures the legacy `bp_names` CSV; it wouldn't be called for v1. |

### Qt-internal intentional decisions (NOT bugs — flagged by scan but author-justified)

The hardwired-path scan flagged these, but inspection of the surrounding comments shows the original authors made deliberate decisions about where these resources live:

| Site | Path | Justification |
|---|---|---|
| `ui_qt/clip_review.py:313` | `<project_root>/csv/validation_results/<video>.csv` | Comment at L304-307: "validation_results lives under '<root>/csv/validation_results/' for both layouts (it's not currently surfaced as a key in project_paths_from_config — derived from project_root + convention)." Human-validation-ratings have no canonical v1 home yet; both layouts use the same convention. |
| `ui_qt/targeted_clips.py:142` | `<project_root>/frames/input/advanced_clip_annotator/<video>/` | Comment at L130-132: staging dir "under both v1 and legacy roots. Not currently a key in the layout helper since this is the only consumer; could be added if more emerge." Transient frame-staging; sole consumer. |
| `ui_qt/input_source_picker.py:155, 158, 168` | `<project>/csv/{input_csv, outlier_corrected_movement_location, smoothed_v2}` | Function is documented as the **legacy-discovery half** of the picker. Enumerating legacy paths IS its job. The v1-half is separate. |
| `tools/csv_to_parquet.py:106` | `<project>/csv/` | Tool's **purpose** is converting legacy CSV trees to parquet. Hardcoded legacy paths are the input domain. |
| `utils/toml_to_configparser.py:154` | `<project_root>/models/` | v1-to-legacy translator. Output domain is the legacy layout by definition. |

**Process lesson:** when a file flags as having hardcoded paths, **read the surrounding comments first** before classifying as a bug. The original authors of `clip_review.py` and `targeted_clips.py` were aware of `project_paths_from_config` and explicitly chose not to route through it. Wrong reflex would be to "fix" these and break the conventions they established.

### Unreachable from Qt workbench (legacy-only; defer)

These files have no Qt-side importer; they only run when called directly or via legacy code (which is now deleted):

| File | Hits |
|---|---:|
| `data_processors/severity_bout_based_calculator.py` | 1 |
| `data_processors/severity_frame_based_calculator.py` | 1 |
| `data_processors/timebins_clf_calculator.py` | 1 |
| `data_processors/timebins_movement_calculator.py` | 2 |
| `data_processors/timebins_movement_calculator_mp.py` | 2 |
| `model/inference_validation.py` | 1 |
| `plotting/annotation_videos.py` | 1 |
| `pose_importers/read_DANNCE_mat.py` | 1 |
| `pose_processors/reverse_pose.py` | 3 |
| `third_party_label_appenders/transform/litpose_merge_projects.py` | 2 |
| `tools/csv_to_parquet.py` | 1 |
| `utils/data.py` | 2 (legacy MARS smoothers; noted in QWI-1 audit) |
| `utils/toml_to_configparser.py` | 1 |

**Disposition:** these don't break any Qt workflow. They're safe to leave alone unless/until one of these legacy paths gets a v1 lifecycle. The right move per cluster:

* **`data_processors/*` and `model/inference_validation.py`** — these are calculator backends. If they get ported to v1 (or surfaced under a Qt analysis page), they need the helper. Right now they only work on legacy projects.
* **`pose_importers/read_DANNCE_mat.py`, `pose_processors/reverse_pose.py`, `utils/data.py`** — legacy import/cleanup paths. Defer until MARS/DANNCE/etc. become v1 priorities.
* **`tools/csv_to_parquet.py`** — explicit legacy-CSV-to-parquet conversion tool. **Intentional legacy scan.** Not a bug; it's what the tool does.
* **`utils/toml_to_configparser.py`** — v1-to-legacy translator. Hardcoded legacy paths are part of its job. **Intentional.**
* **`third_party_label_appenders/transform/litpose_merge_projects.py`** — uses `self.master_dir / "videos"` for SimBA-style project merging. Legacy-only operation.

### Intentional legacy-discovery (not bugs)

* **`ui_qt/input_source_picker.py:155, 158, 168`** — explicitly enumerates `csv/input_csv`, `csv/outlier_corrected_movement_location`, `csv/smoothed_v2` as candidates when scanning a legacy project for existing pose data. The function is documented as the legacy-half of the picker; the v1-half is separate.
* **`tools/csv_to_parquet.py`** — same idea: scan legacy CSV trees for conversion.

## Fix pattern

For any site that needs to support both layouts:

**Before:**
```python
project_path = config.get(GENERAL_SETTINGS, PROJECT_PATH)
videos_dir = os.path.join(project_path, "videos")
roi_path = os.path.join(project_path, "logs", "measures", "ROI_definitions.h5")
```

**After:**
```python
from mufasa.project_layout import project_paths_from_config
paths = project_paths_from_config(config_path=config_path)
videos_dir = paths["video_dir"]
roi_path = paths["roi_definitions_path"]
```

For sites that need the legacy directory as a *starting point* (e.g. a CSV-to-parquet conversion tool that walks `csv/`), the hardcoded path stays — the tool's job is to find legacy data.

## Fix-coverage status

**Fixed in 122da:**

| File | Sites | What |
|---|---|---|
| `roi_tools/roi_utils.py` | L474 + L561 | ROI coordinates path resolution. Two siblings of the L462 fix from 122d9; both functions (`multiply_ROIs` and `reset_video_ROIs`) had the same hardcoded `os.path.join(project_path, "logs", ...)` pattern. Now route through `project_paths_from_config(...)["roi_definitions_path"]`. |

**Decided NOT to fix (intentional designs, see above):** `ui_qt/clip_review.py:313`, `ui_qt/targeted_clips.py:142`, `ui_qt/input_source_picker.py:155+`, `tools/csv_to_parquet.py:106`, `utils/toml_to_configparser.py:154`.

**Open follow-up (not in 122da scope):** the 7 remaining deferred Qt-reachable sites + the 13 legacy-only files. These don't block v1 workflows today but constrain the project to remain "legacy works first" rather than "v1 is canonical". Recommended split into focused patches:

* **122db** — Qt-reachable backend bugs (data_processors/pup_retrieval, video_processors/video_processing frames extraction). Both need the layout helper applied AND a decision about where v1 puts the output (e.g., v1 frames go to `derived/frames/extracted/`).
* **122dc** — Qt-internal form bugs (pose_cleanup × 3, video_info × 3, visualizations × 1). Mix of unconditional bugs and defensive fallbacks; needs per-site triage.
* **122dd** — `roi_tools/roi_logic.py` × 2. Qt-reachable; needs to use the helper.
* **(deferred)** — legacy-only files (data_processors/severity, timebins, MARS smoothers, etc.). Disposition: fix only if the cluster gets a v1 lifecycle.

## Process lessons

1. **Sibling audits should be exhaustive.** The 122d9 sibling audit only checked for the exact same `os.path.join(project_path, "videos")` pattern. It missed `os.path.join(project_path, "logs", ...)` two lines below — same root cause, different subpath. The cleaner approach: any function that uses `config.get(GENERAL_SETTINGS, PROJECT_PATH)` is suspect; sweep all path-joins inside it.

2. **"Qt-reachable" is the right scoping question** for fix prioritization. Files unreachable from Qt aren't breaking any current workflow; fixing them is "improve coverage", not "fix bugs".

3. **Intentional legacy-discovery is OK.** `input_source_picker`, `csv_to_parquet`, `toml_to_configparser` all hardcode legacy paths *as their job*. Don't reflexively replace those with the abstraction layer.

4. **The abstraction layer is in good shape.** 18 public functions; both layouts handled cleanly; consistent detection rule. The bugs aren't from the helper being missing — they're from earlier code that wasn't touched during the layout migration. Migration-by-attrition rather than migration-by-sweep.

5. **Triage rule: not every join to a legacy-looking subpath is a bug.** Three classes of false positive surfaced during 122db's re-triage:

   a) **`self.project_path` is layout-resolved.** On any `ConfigReader` subclass, `self.project_path` is set to the project root for BOTH layouts (v1 root for v1; legacy root for legacy). Subpaths that are identical in both layouts (`logs/`, `derived/features/`, `derived/labels/`) are safe to join directly.
   
   b) **Defensive try/except fallbacks.** Many files use `project_paths_from_config` in the primary path and a hardcoded legacy guess in the `except` branch. The `except` fires only on malformed configs — acceptable.
   
   c) **Branch-gated legacy paths.** Code paths inside `if not v1:` / `else:` branches of explicit layout checks are SUPPOSED to construct legacy paths.

   The right triage workflow:
   1. Find `os.path.join(X, "Y", …)` hits.
   2. For each: does subpath Y differ between layouts? (If `logs/`, `derived/features/`, `models/` after the v1 layout absorbs them, the answer is "no" → false positive.)
   3. Where does X come from? `self.project_path` on a ConfigReader subclass is layout-resolved. `config.get(GENERAL_SETTINGS, PROJECT_PATH)` directly on a configparser is **broken for v1**.
   4. Is the join inside a `try/except` whose primary path uses the helper? → defensive fallback, not a bug.
   5. Is it inside an explicit `if v1_root is None:` legacy branch? → intentional.
   
   Following this workflow during 122db reduced the initially-flagged 12 "Qt-reachable bugs" to 3 actual ones.
