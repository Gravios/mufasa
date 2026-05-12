# CHANGELOG

Working notes on what landed in each session, what's deferred, and what
to pick up next time. Keep entries dated and grouped by patch series so
"where did this behavior come from" is answerable from git log alone.

---

## Session 2026-05-12 — TOML-aware ConfigReader + configparser removal from forms

Two patches: **122e** makes `ConfigReader` (959 lines, inherited
by 268+ files across the codebase) read v1 `project.toml` while
preserving its legacy attribute surface, so every backend
continues to work transparently against v1 projects without
per-file modification. **122f** removes direct `configparser`
usage from every UI form method, encapsulating the legacy INI
branch in a handful of layout-agnostic helpers.

Directional commitment: as of this session, Mufasa no longer
depends on `project_config.ini` for any production code path.
INI parsing is kept alive in three places only — the migration
tool (`legacy_layout.py` + `migrate_project.py`), the
`read_config_file` fallback for legacy projects opened post-
migration, and the encapsulated legacy branches of the new
read/write helpers.

### Shipped

#### TOML-aware ConfigReader (patch 122e)

New module **`mufasa/utils/toml_to_configparser.py`** translates
a parsed v1 `project.toml` into a synthetic
`configparser.ConfigParser` populated with the legacy section
and key names. Coverage:

- `[General settings]` — `project_path` (= v1 root, not a
  `project_folder/` subdir), `project_name`, `file_type`,
  `workflow_file_type`, `animal_no`, `os_system`
- `[SML settings]` — `model_dir`, `no_targets`, `target_name_N`,
  `model_path_N` for each classifier
- `[threshold_settings]`, `[Minimum_bout_lengths]` — per-
  classifier placeholders (`"NaN"`)
- `[create ensemble settings]` — pose preset code +
  ML-training placeholders (`"NaN"` for unset)
- `[Multi animal IDs]` — comma-joined `id_list`
- `[Outlier settings]` — movement / location criteria
- Plus `[Frame settings]`, `[Line plot settings]`,
  `[Path plot settings]`, `[ROI settings]`, `[Directionality
  settings]`, `[process movement settings]` (empty placeholders
  the legacy creator wrote)

Missing TOML values default to `"NaN"` (the legacy
`Dtypes.NONE` sentinel) so existing
`read_config_entry(..., default_value=...)` calls behave
identically.

**`mufasa/utils/read_write.py`** — `read_config_file` now
branches on the suffix: `.toml` → delegates to the shim;
anything else → the original INI parser. Case-insensitive
(`.TOML` works too). Legacy projects opened after a migration
keep their existing path verbatim.

**`mufasa/mixins/config_reader.py`** — `ConfigReader.__init__`
detects v1 by suffix, pre-loads the parsed TOML on
`self._v1_toml_data`, and routes the body-parts handling
through TOML rather than the `project_bp_names.csv` file. A
new method `_apply_v1_path_overrides` runs early in `__init__`
(before any filesystem read) and rewrites the 30+ path
attributes — `input_csv_dir`, `outlier_corrected_dir`,
`features_dir`, `targets_folder`, `machine_results_dir`,
`video_dir`, `video_info_path`, all the plot dirs, the SHAP /
directionality / ROI / clf-validation dirs, plus `logs_path`
and `roi_coordinates_path` — to v1 equivalents under
`sources/`, `derived/`, `models/`, `logs/`. Multi-run stages
(outlier_corrected, features, classifications) resolve to the
**latest run** subdir (`derived/<stage>/<run_id>/`) when any
exist, falling back to the stage parent. File lists are re-
globbed against the new paths.

#### configparser removal from ui_qt forms (patch 122f)

Two new helpers in **`mufasa/project_layout.py`**:

- `project_paths_from_config(config_path)` → dict mapping
  `project_root`, `video_dir`, `input_pose_dir`, `logs_dir`,
  `video_info_path`, `models_dir` to v1 or legacy equivalents
  depending on the config suffix.
- `project_metadata_from_config(config_path)` → dict with
  `animal_count`, `file_type`, `body_parts`, `animal_ids`,
  `classifier_targets`, `pose_config_code`. v1 reads from
  `project.toml` directly; legacy parses the INI plus the
  `project_bp_names.csv` file.

Forms migrated (every direct `configparser.ConfigParser()` call
in form methods replaced):

- **pose_cleanup.py** — `SmoothingForm.target`,
  `InterpolateForm.target`, `OutlierSettingsForm.build/target`,
  `EgocentricAlignmentForm.build/collect_args`,
  `KalmanV2SmoothingForm.build` (legacy fallback),
  `RunOutlierCorrectionForm.build/target`. The
  read-modify-write of outlier settings is encapsulated in two
  module-level helpers `_read_outlier_settings` and
  `_write_outlier_settings`. v1's TOML schema gains a nested
  `[outlier_settings.references]` table
  (`Animal_X = ["bp1", "bp2"]`) mirroring the legacy
  per-animal reference keys.
- **classifier.py** — `_refresh_remove_options`,
  `_add_classifier`, `_remove_classifier` now use two
  module-level helpers `_read_classifiers` and
  `_write_classifiers`. v1 read-modify-writes `project.toml`'s
  `[classifiers].targets`.
- **analysis.py** — `_load_classifier_names` delegates to
  `project_metadata_from_config`. Top-level `import
  configparser` removed.
- **visualizations.py** — the data-paths auto-population path
  uses `project_paths_from_config` + `project_metadata_from_config`.
  Inline `import configparser` inside `target` removed.
- **addons.py** — `_load_cue_light_names` branches on suffix.
  v1 returns `[]` (cue-light ROI names aren't in the v1 schema
  yet; form falls back to free-text entry).
- **annotation.py**, **roi.py** — dead `import configparser`
  lines removed (no other use).

Net effect: every form *method* is `configparser`-free.
`configparser` references that remain are in three places, all
encapsulated:

1. `pose_cleanup.py` — legacy branches of `_read_outlier_settings`
   and `_write_outlier_settings`
2. `classifier.py` — legacy branch of `_write_classifiers`
3. `addons.py` — legacy branch of `_load_cue_light_names`

### Test counts at session end

| suite                                    | count    |
|------------------------------------------|----------|
| smoke_project_layout                     |   9/9    |
| smoke_migrate_project                    |   5/5    |
| smoke_recent_project                     |   6/6    |
| smoke_pose_cleanup_v2_wiring             |   2/2    |
| smoke_model_dual_save                    |  23/23   |
| smoke_input_source_picker                |  29/29   |
| smoke_outlier_forms_wiring               |  40/40   |
| smoke_empty_classifier                   |   1/1    |
| smoke_config_creator_v1                  |  38/38   |
| smoke_v1_configreader (NEW, 122e)        |  69/69   |
| smoke_v1_form_configparser_removal       |  37/37   |
| (NEW, 122f)                              |          |

### Sandbox testing limitation

ConfigReader itself can't be instantiated in this sandbox —
its import chain pulls cv2, h5py, trafaret, and tkinter, none
of which are available. The 122e test exercises the three
layers it can reach (the TOML shim behaviorally, the
`read_config_file` routing via AST, the ConfigReader v1 logic
via AST). Full behavioral verification — actually constructing
a `ConfigReader(config_path=project.toml)` and inspecting its
attribute surface against the expected v1 paths — is the
user's to confirm on a real install. The wiring is testable
even if the runtime isn't.

### Deliberately deferred

**Backend writes leak provenance.** ConfigReader's
`_apply_v1_path_overrides` resolves multi-run stage attributes
to the **latest run** subdir. That's correct for reads but wrong
for writes — a backend that does `os.makedirs(self.outlier_corrected_dir,
exist_ok=True)` and then writes files there will mutate the
prior run's directory in place, corrupting provenance. Each
writing backend needs `ProjectPaths.stage_run_dir(...)`
allocation at run time instead. Track as the "v1-aware backend
writes" thread.

**Configs directory.** `configs_meta_dir` falls back to
`<root>/configs/` for v1; the v1 layout doesn't model a configs
dir yet.

**Body-parts path in v1.** `body_parts_path` is set to the
project.toml path itself (informational only — v1 reads body
parts from TOML data directly). Any legacy code that tries to
parse this file as a CSV will fail loudly.

**reconfigure_project_user_defined.** Still INI-only.
Currently unreachable from CreateProjectDialog. Kept for the
File → Reconfigure menu action; v1 rewrite is a small patch
when needed.

**Cue-light ROI metadata in v1.** `addons.py._load_cue_light_names`
returns `[]` for v1. If cue-lights matter going forward, the v1
schema needs a `[roi.cue_lights]` table plus form-side writers.

### Pickup checklist for next session

1. Confirm tests green:
   ```bash
   for t in tests/smoke_*.py; do PYTHONPATH=. python "$t"; done
   ```
2. **Full ConfigReader behavioral verification** on a real
   install — instantiate against a v1 project.toml and walk
   the attribute surface. Anything still under `csv/` instead
   of `sources/`/`derived/` is a missed override.
3. **v1-aware backend writes** — the highest-priority deferred
   thread. Each writing backend (outlier correction, feature
   extraction, classifier training, visualization) needs
   `ProjectPaths.stage_run_dir(...)` allocation at run time
   instead of reading a "latest" path from ConfigReader.
4. **Batch pipeline through inference** — still the original
   ask from session 2026-05-11; never started.
5. **Migration prompt on legacy projects** — when
   `detect_layout(project) == 'legacy'`, dialog at workbench
   launch offering migration. Small patch.

---

## Session 2026-05-11 — Dual-save model store + InputSourcePicker + pose-cleanup redesign + v1-only new projects

Three patch series landed: **122b** factors model provenance and
input-source selection into reusable primitives; **122c** uses
them to redesign the pose-cleanup page with explicit Run / Skip
outlier-correction forms and the legacy forms folded into an
"Advanced / legacy" section; **122d** cuts new-project creation
over to v1-only and makes the Kalman v2 form v1-aware so the
fresh-project pose-cleanup workflow runs end-to-end without
touching legacy paths.

### Shipped

#### Dual-save model store (patch 122b)

Trained models now live in two places. The **global cache** at
`~/.config/mufasa/models/<name>.npz` is a cross-project library
— a model trained once is discoverable from any project on the
same machine. The **project store** at `<project>/models/<name>/`
holds `model.npz` plus a `card.toml` carrying provenance
(`source_path`, `sha256`, `copied_at`, `mufasa_version`). Both
get a copy on every save / load crossing.

- New `project_layout.py` helpers: `import_model_into_project`,
  `mirror_model_to_global_cache`, `file_sha256`,
  `global_model_cache_dir`, `resolve_v1_project_root`.
  Content-hash idempotence: re-importing/mirroring the same
  model is a no-op. Different-content collision raises
  `FileExistsError` rather than silently overwriting (training
  output passes `overwrite=True`; load-side imports use the
  default no-overwrite to protect known-good models).
- `KalmanV2SmoothingForm.target`: a `_post_train_dual_save()`
  closure runs after every `smooth_pose_v2(save_model=...)`
  call (both single-pass and two-pass training paths); the
  load branch imports the chosen model into the project's
  `models/` before passing it to the smoother. Failures in
  either are soft — logged, but the original `.npz` is left
  intact.

#### InputSourcePicker (patch 122b)

New module `mufasa/ui_qt/input_source_picker.py`.

- `discover_input_sources(config_path, project_root)` — pure
  function. Walks both layouts: legacy `csv/{input_csv,
  outlier_corrected_movement, outlier_corrected_movement_location,
  smoothed_v2}/` and v1 `sources/pose/` + `derived/smoothed/
  <flavor>/<run>/` + `derived/outlier_corrected/<run>/`. v1
  candidates come first when both are present (post-migration
  transient). Multi-run stages list one entry per run, newest
  first. Empty / nonexistent dirs are skipped silently.
- `_DEFAULT_PREFER_ORDER`: kalman_v2 → outlier_corrected →
  savitzky → raw. Exactly one source is marked `is_default`.
  Overridable per call (the new `RunOutlierCorrectionForm`
  prefers raw, since outlier correction usually precedes
  smoothing).
- `InputSourcePicker` Qt widget: combobox + path field + browse
  + refresh (`↻`). Custom-path sentinel un-greys the path field.
  PySide6 imported lazily so the pure discovery function tests
  headless.
- **First consumer**: `EgocentricAlignmentForm`. The previous
  auto-detect of `csv/outlier_corrected_movement_location/`-vs-
  `csv/input_csv/` is gone; users can now feed egocentric
  alignment any prior stage's output, including a specific
  Kalman v2 run.

#### Pose cleanup redesign (patch 122c)

- New `RunOutlierCorrectionForm`. Chains
  `OutlierCorrecterMovement` → `OutlierCorrecterLocation`. Input
  via `InputSourcePicker` (preference RAW). Per-stage checkboxes
  let users disable movement or location independently. Output
  dir auto-generates `derived/outlier_corrected/<run_id>/` on v1
  projects (with `run.toml` provenance written after) and falls
  back to legacy `csv/outlier_corrected_movement_location/`
  otherwise. When both stages run, movement writes to a
  `_movement_intermediate/` sibling that location reads from.
- New `SkipOutlierCorrectionForm`. Thin wrapper around
  `OutlierCorrectionSkipper` — no fields, just a Run button.
  Writes `run.toml` with `params.skipped = true` and a
  `SKIPPED` marker file when run inside a v1 project so future
  layout-aware tooling can distinguish "skipped" from "missing"
  runs.
- Pose cleanup page sections reordered to match the conceptual
  pipeline:

  ```
  1. Interpolate missing frames
  2. Kalman v2 smoothing
  3. Run outlier correction         ← NEW
  4. Skip outlier correction        ← NEW
  5. Egocentric alignment           (now picker-driven)
  6. Advanced / legacy              (stacks SmoothingForm,
                                     OutlierSettingsForm,
                                     DropBodypartsForm)
  ```

  Legacy `Smoothing` / `Outlier correction settings` /
  `Drop body-parts` top-level sections are gone; their forms
  still exist (settings is still consumed by the new outlier
  forms — it writes the thresholds they read) but they're
  folded into one Advanced section using
  `WorkflowPage._instantiate`'s existing form-stacking.

#### v1-only new projects + Kalman v2 v1-awareness (patch 122d)

`ProjectConfigCreator` was rewritten end-to-end. The legacy
SimBA tree generator (the `__create_directories` /
`__create_configparser_config` pair that built `project_folder/
csv/{input_csv, outlier_corrected_*, features_extracted, ...}`
+ `project_config.ini`) is gone. New projects land as v1:

```
<parent>/<project_name>/
├── project.toml                       (new)
├── sources/{videos,pose,annotations}/  (via ProjectPaths.ensure_skeleton)
├── derived/
├── models/
└── logs/
```

The new `project.toml` schema is small and explicit:

- top-level: `project_layout_version`, `project_name`,
  `created`, `mufasa_version`, `os_platform`
- `[pose]`: `animal_count`, `file_type`, `body_parts` (list,
  in order), `pose_config_code`, `pose_config_idx`, `animal_ids`
- `[classifiers]`: `targets` (may be empty)
- `[outlier_settings]`: `movement_criterion`, `location_criterion`
  (initial `"NaN"` matches the legacy `Dtypes.NONE` sentinel
  so existing settings code can read it verbatim once it's TOML-
  aware)

The Qt create-project dialog (`CreateProjectDialog`) was
simplified accordingly: the previous "create with placeholder
preset, then call `reconfigure_project_user_defined` to patch
in the autodetected body parts" two-step is gone. Both the
preset and autodetect paths now funnel body parts directly into
`ProjectConfigCreator` via the new `body_parts=` parameter, so
the project is fully configured on first write. The dialog's
docstring and the project-setup-page banner were updated to
say `project.toml` instead of `project_config.ini` (with the
banner still acknowledging legacy `project_config.ini` for
projects opened from before this cutover).

`KalmanV2SmoothingForm` was made v1-aware so the fresh-project
pose-cleanup workflow runs end-to-end on v1 without any legacy
path resolution:

- **Input dir** defaults to `<project>/sources/pose/`.
- **Output dir** defaults to
  `<project>/derived/smoothed/kalman_v2/<run_id>/`. A fresh
  `run_id` is allocated at form-build time (cached on
  `self._v1_run_id`) so the path the user sees in the field is
  exactly what gets written on Run. Closing and reopening the
  form gives a fresh `run_id`; the form never overwrites a
  prior run.
- **Save-model path** defaults to `<run_dir>/model.npz`,
  co-located with the smoothed pose data. The 122b dual-save
  flow still mirrors to `~/.config/mufasa/models/model.npz` and
  imports into `<project>/models/model/` on top.
- After smoothing completes, the form writes a
  `<run_dir>/run.toml` provenance file tagged
  `stage = "smoothed.kalman_v2"`, with `params` capturing every
  scalar/list kwarg passed to `smooth_pose_v2` plus the mode
  (`"train"` or `"load"`). Soft-fails (logged) if anything goes
  wrong — pose data already on disk is left intact.

Legacy projects still work via the existing `configparser`
fallback in `build()`. `resolve_v1_project_root` distinguishes
the two cases.

Net effect: the data flow for the v1 pose-cleanup pipeline is
now self-consistent — raw pose lives at `sources/pose/`, the
Kalman v2 smoother writes to
`derived/smoothed/kalman_v2/<run_id>/`, and the egocentric
alignment form's `InputSourcePicker` surfaces both raw and any
prior smoothed run as candidates (with the most-processed run
as the default). Subsequent pose-cleanup operations read from
raw or smoothed exactly as intended.

### Test counts at session end

| suite                          | count    |
|--------------------------------|----------|
| smoke_kalman_pose_smoother_v2  | 152/152  |
| smoke_project_layout           |   9/9    |
| smoke_migrate_project          |   5/5    |
| smoke_recent_project           |   6/6    |
| smoke_pose_cleanup_v2_wiring   |   2/2    |
| smoke_model_dual_save          |  23/23   |
| smoke_input_source_picker      |  29/29   |
| smoke_outlier_forms_wiring     |  40/40   |
| smoke_empty_classifier         |   1/1    |
| smoke_config_creator_v1        |  38/38   |

`smoke_model_dual_save` redirects `HOME` so it doesn't pollute
the real `~/.config/mufasa/models/`. `smoke_input_source_picker`
covers the pure discovery function behaviorally and the Qt
widget at AST level (PySide6-free sandbox). The wiring tests
are AST-only. `smoke_empty_classifier` was rewritten from its
legacy INI shape into a v1-`project.toml` shape; the original
contract (empty `target_list` is accepted) is preserved.

### Deliberately deferred

**Model store**

- Trained-model auto-naming. The form's default save path is
  now `<run_dir>/model.npz` for v1 projects (one model per run,
  collision-free by construction); the global cache still flat-
  names them as `model.npz` which IS collision-prone. Probably
  want to use the run_id as the global-cache name too.
- Card.toml schema is minimal — `model_name`, `source_path`,
  `sha256`, `copied_at`, `mufasa_version`. No training params,
  no eval metrics, no fingerprint of the training data. The
  changelog's earlier note on `card.toml` for classifiers still
  applies; this is the v0.

**InputSourcePicker**

- Picker doesn't show file counts or last-modified time per
  source. For a project with many runs it'd be useful to know
  which has data and how fresh it is. Two lines per item in the
  combobox would do it.
- No source preview / inspect button. Picking a run and seeing
  its `run.toml` parameters before committing would help.

**Pose cleanup**

- "Drop body-parts" still lives in Advanced / legacy. It's
  really a project-setup decision, not a per-run cleanup step.
  Move it to the project-setup page once that's the right
  surface.
- `RunOutlierCorrectionForm` writes `run.toml` for v1 but
  doesn't yet enumerate the actual input file list (params has
  the `data_dir`, not the per-file paths). The `write_run_toml`
  `inputs` field accepts a list; populating it would make
  full reproducibility checkable.
- `SkipOutlierCorrectionForm` writes a v1 `run.toml` next to a
  `SKIPPED` marker file but doesn't actually populate the v1
  `derived/outlier_corrected/<run>/` with the skip-copied
  files. That's because `OutlierCorrectionSkipper` writes to
  the legacy `csv/outlier_corrected_movement_location/` and
  isn't yet v1-aware. Fixing this is part of the broader
  "make each backend v1-aware" thread.
- The legacy `SmoothingForm` (Savitzky-Golay) is still mounted
  in Advanced. Once we're confident Kalman v2 covers every
  pre-Kalman use case the dataset hit, this form can be
  removed entirely.

**v1 cutover (122d) — the iceberg below the waterline**

The 122d patch makes new projects v1, but the downstream forms
that consume those projects are still mostly INI-driven. The
**only** form that fully works in a fresh v1 project right now
is `KalmanV2SmoothingForm` (because its target reads pose
directly from `input_dir` without ever touching
`config_path`). Every other form — `InterpolateForm`,
`RunOutlierCorrectionForm`, `SkipOutlierCorrectionForm`,
`EgocentricAlignmentForm`, the legacy three, plus every form
on every other page — calls into a backend that uses
`ConfigReader` to parse `config_path` as an INI. In a v1
project `config_path` points to `project.toml`, and those
backends will fail to read it.

The migration thread that fixes this:

1. Add a `ConfigReader` subclass (or a `read_project_toml`
   helper) that reads `project.toml` and exposes the same
   attributes the legacy ConfigReader does (`animal_cnt`,
   `body_parts`, `file_type`, `input_csv_dir`, etc.). For v1,
   `input_csv_dir` resolves to `sources/pose/`,
   `outlier_corrected_dir` to a latest-run path under
   `derived/outlier_corrected/`, and so on.
2. Route every `ConfigReader`-using backend through the
   subclass.
3. Make each form's `target()` pass the right v1 paths down
   so its backend never has to guess.

That's a multi-session thread; the priority order matches the
pose-cleanup page's section order (Interpolate, Outlier-Run,
Outlier-Skip, Egocentric backend, Advanced/legacy).

Until that's done, the practical v1 workflow is:

- Drop raw pose into `<project>/sources/pose/`.
- Run **Kalman v2 smoothing** — produces
  `derived/smoothed/kalman_v2/<run>/`.
- (Other pose-cleanup forms will fail with a config error.)
- Egocentric alignment partially works — the form constructs
  fine (no INI needed in the form itself, since the picker
  drives the data path), but its backend
  (`EgocentricalAligner`) reads `config_path` and may fall
  over depending on how deeply it relies on the INI.

In short: **122d unlocks the smoother in v1 mode; the rest of
the pipeline catches up patch by patch.**

### Open questions

- Should `RunOutlierCorrectionForm`'s save_dir picker accept a
  v1 run directory the user already created (rather than always
  auto-generating)? Useful for re-running with tweaked
  thresholds in the same provenance dir. Currently auto-gen
  only.
- The InputSourcePicker shows v1 runs newest-first. For a
  project with dozens of runs this could get unwieldy. A
  "latest only" toggle (showing just the most recent run per
  flavor) would help; the full list could be reachable via an
  "All runs…" entry. Not urgent.
- The legacy `reconfigure_project_user_defined` helper still
  exists but isn't called by `CreateProjectDialog` anymore.
  Keep it for the File → Reconfigure project from DLC file…
  menu action, or retire it entirely? Probably keep until the
  v1-aware reconfigure equivalent exists.

### Pickup checklist for next session

1. Confirm tests still green:
   ```bash
   PYTHONPATH=. python tests/smoke_project_layout.py && \
   PYTHONPATH=. python tests/smoke_migrate_project.py && \
   PYTHONPATH=. python tests/smoke_recent_project.py && \
   PYTHONPATH=. python tests/smoke_pose_cleanup_v2_wiring.py && \
   PYTHONPATH=. python tests/smoke_model_dual_save.py && \
   PYTHONPATH=. python tests/smoke_input_source_picker.py && \
   PYTHONPATH=. python tests/smoke_outlier_forms_wiring.py && \
   PYTHONPATH=. python tests/smoke_empty_classifier.py && \
   PYTHONPATH=. python tests/smoke_config_creator_v1.py
   ```
2. Three threads ready to pick up:
   - **v1-aware ConfigReader.** Highest priority — every form
     beyond Kalman v2 fails in a fresh v1 project until this
     exists. See "iceberg below the waterline" section above.
   - **Batch pipeline through inference** (still the original
     ask from this session — scoped out, never started).
     Schema for `pipeline.toml`, CLI scaffold, ephemeral-project
     synthesizer, four stage adapters. Estimated 1500–2500 LOC
     across one or two sessions.
   - **Migration prompt on legacy projects.** Original
     deferred item from 122a: when `detect_layout(project) ==
     'legacy'`, surface a dialog at workbench launch offering
     migration. Small patch.

---

## Session 2026-05-08 to 2026-05-11 — Kalman v2 smoother feature completion + workbench wiring + project-layout redesign

### Shipped

#### Smoother backend (patch 121d–e)

- **121d — constant-acceleration extension** for selected segments.
  Listed segments gain an acceleration block at the end of the state
  vector (+4 dims for root translation + orientation, +2 dims per
  non-root segment). F gets integrator cross-terms (`v_new = v + a·dt`,
  `x_new = x + v·dt + a·dt²/2`); Q gets diagonal jerk noise on accel
  slots. FK and Jacobian unchanged. New CLI flag
  `--const-accel-segments`. Backward compatible (empty list = no-op).
  Tests 145-148 (4 new on top of 121b/c's 144).

- **121e — M-step learning of q_jerk_***. EM now adapts `q_jerk_root_pos`,
  `q_jerk_root_ori`, `q_jerk_seg_ori` from Q_hat's accel-block
  diagonals instead of holding them at the initial 10×-q-root default.
  Floor/ceiling/hard-cap pattern from 121c
  (`_M_STEP_Q_JERK_*_HARD_CAP = 500000 / 500 / 500`). Per-session
  aggregator still passes through prev values — deferred. Tests 149-152.

#### Workbench wiring (patch 121f–i)

- **121f — Kalman v2 smoother in Pose Cleanup page**. New
  `KalmanV2SmoothingForm` section sitting below the legacy Savitzky-
  Golay smoother. Calls `smooth_pose_v2` via Python API (not
  subprocess) so errors surface in the progress dialog. Fields:
  input/output dirs, fps, likelihood threshold, EM iters, workers,
  per-marker drift, orientation drift segments, const-accel segments,
  save-model checkbox. PySide6-independent AST-level wiring smoke
  test.

- **121g — train/load mode + training subset + EM parameters**. Mode
  selector at top of form (radio buttons: train new / load saved).
  Training-mode group adds a file-list picker for subset selection
  (auto-refreshes on input-dir change, with select-all/clear/refresh
  buttons), full EM parameter surface (tolerance, damping, pooled-vs-
  per-session aggregation, warm-start σ, perspective, validation),
  save-model path with browse. Load-mode group surfaces the model
  file picker. Two-pass workflow runs when training on a subset: fit
  on subset → load → smooth all.

- **121h — standard model directory at `~/.config/mufasa/models/`**.
  Module-level helper `_default_model_dir()`. Five touchpoints
  (placeholder, save-browse, load-browse, collect_args, two-pass
  workflow) route through it. Models trained once are discoverable
  from any project on the same machine.

- **121i — persistent recent-project at `~/.config/mufasa/recent`**.
  New `mufasa/ui_qt/recent_project.py` module (PySide6-free, testable
  headless). Launch priority: `--project` → CWD auto-discover →
  recent file → none. `--no-recent` flag disables fallback.
  `_switch_to_project` saves on every project switch. Stale/empty/
  unwriteable entries fail silently.

#### Project layout redesign (patch 122a)

Major restructure — patch number stepped to 122 to signal the break.

- **122a — full project-layout v1 + migration tool**. New layout:

  ```
  <project_root>/
  ├── project.toml                     # was project_config.ini
  ├── sources/{videos,pose,annotations}/   # read-only inputs
  ├── derived/                         # generated; safe to delete
  │   ├── smoothed/<flavor>/<run_id>/
  │   ├── outlier_corrected/<run_id>/
  │   ├── features/<run_id>/
  │   ├── classifications/<run_id>/
  │   └── frames/{extracted,annotated}/
  ├── models/<model_name>/
  └── logs/<run_id>/
  ```

  Three new modules: `mufasa/project_layout.py` (`ProjectPaths`,
  `Stages`, `SmoothingFlavors`, `generate_run_id`, TOML I/O, version
  validation), `mufasa/legacy_layout.py` (SimBA-to-v1 mapping table,
  INI parser preserving unknown sections under `[legacy.*]`,
  body-part-name file reader), and `mufasa/cli/migrate_project.py`
  (`MigrationPlan` separates planning from execution, dry-run default,
  `--commit` to actually move, idempotent on already-migrated
  projects, `MIGRATION.toml` manifest written for auditability).

  Run-id format: `YYYYMMDD-HHMMSS-<6hex>` — sortable lexically =
  sortable chronologically. Per-run `run.toml` carries stage,
  parameters, input file list, mufasa version, results.

  Tests: `tests/smoke_project_layout.py` (9 checks: constants, run-id
  format, ProjectPaths skeleton, stage_run_dir, list_runs/latest_run,
  TOML roundtrip, version guard, run.toml roundtrip, detect_layout)
  and `tests/smoke_migrate_project.py` (5 checks: synthetic SimBA
  project → migration → verify all files at expected v1 locations +
  project.toml content + MIGRATION.toml manifest + idempotence + dry-
  run + unknown-path exit code + project_folder-pointed call shape).

### Test counts at session end

| suite                              | count    |
|------------------------------------|----------|
| smoke_kalman_pose_smoother_v2      | 152/152  |
| smoke_workbench_auto_discover      |  10/10   |
| smoke_workbench_launcher           |  10/10   |
| smoke_pose_cleanup_v2_wiring       |   2/2    |
| smoke_recent_project               |   6/6    |
| smoke_project_layout               |   9/9    |
| smoke_migrate_project              |   5/5    |

### Real-data observations worth preserving

Confirmed during 121bc real-data runs (67-session dataset, 30 fps):

- `--orient-drift-segments body` made the tail look better and gave
  honest larger variance circles (the over-confidence from before was
  hiding real uncertainty), but head markers became too constrained —
  root angular residual no longer absorbing.
- Adding `head` to drift loosened it. `body,head` worked.
- With drift wired and `--const-accel-segments body,head`, motion was
  slightly over-smoothed during fast turns. Hypothesis: the constant-
  velocity predictor lags on accelerating motion; const-accel should
  help. 121d/e land that.
- Useful knobs found during tuning: `--likelihood-threshold` (lower
  includes more obs, less prior reliance), q-scaling via save/load
  model.
- Recommended config for this dataset:
  `--with-drift --orient-drift-segments body,head --const-accel-segments body,head --fps 30 --em-max-iter 20 --workers 12`.

### Deliberately deferred

Notes on shape, not commitment to implementation — pick whichever
matters next.

**Smoother backend**

- M-step learning of q_jerk in the per-session aggregator. Currently
  only the pooled path (the default) learns; the per-session path
  passes through prev_params. Pattern would mirror what's in
  `finalize_m_step_v2` already — same sufficient stats, same caps,
  per-session aggregation via the existing scalar/median machinery.

**Workbench**

- Live progress reporting from `smooth_pose_v2`. The form currently
  shows an indeterminate progress dialog; the `[smoother-v2] ...`
  lines still go to stdout. A Qt-friendly progress-callback parameter
  on `smooth_pose_v2` plus a signal-based bridge into the dialog
  would surface them. Worth doing but not blocking.
- Cancel button. Dialog has one but `smooth_pose_v2`'s worker pool
  doesn't poll for cancellation between sessions.
- Multi-segment pickers. Orientation-drift and const-accel segments
  are typed as comma-separated strings. Checkbox grid against the
  layout's actual segment list would be a UX upgrade.
- Model inspection. Loading a model just feeds it to the smoother;
  the form doesn't surface the layout extensions or fitted q values
  from the loaded file. An "Inspect" button could read the npz and
  show a summary.

**Project layout / consumer migration**

The 122a patch lands the layout primitives but NOT the form
integration. Recommended order for the next session:

1. **Launcher UX**: when `detect_layout(project_root) == 'legacy'`,
   pop a dialog offering to run `migrate_project`. Small patch
   touching `workbench_app.py`'s startup path.
2. **`KalmanV2SmoothingForm` v1-aware**: when `config_path` resolves
   to a v1 project, default output_dir to
   `ProjectPaths.smoothed_run_dir(SmoothingFlavors.KALMAN_V2)` (auto-
   generated run id) and default save_model to
   `<run_dir>/model.npz`. Falls back to legacy defaults for legacy
   projects. Write `run.toml` after each successful run.
3. **Each other form** (outlier correction, features, classification)
   gets the same one-form-at-a-time treatment. Stop once the user's
   workflow doesn't hit a legacy path anymore.
4. **`ConfigReader` subclass** that reads `project.toml`. Existing
   consumers route through it transparently.
5. **Run-id selector** in the v2 form: optional "Run name" field
   (default: auto), and the "Load saved model" picker enumerates
   `derived/smoothed/kalman_v2/*/model.npz` from the current project
   when one is loaded.
6. Once all forms migrated: deprecate the legacy INI reader. The
   legacy-layout reader stays as long as we want migrate-project to
   work.

**Schemas to design later**

- `models/<model_name>/card.toml` for trained classifiers — what they
  were trained on, when, what params. Migration just moves
  `models/` as-is right now; the format for new training runs is open.
- Anything that wants a stable schema in `project.toml` beyond what
  `parse_legacy_config` produces (animal_count, body_parts, stage
  defaults). Worth letting it grow organically as consumer code
  migrates.

### Open questions

- Is `~/.config/mufasa/models/` the right place for the global model
  cache once projects have their own `<project>/models/`? Two options:
  keep both (global cache for cross-project reuse, project-local for
  reproducibility) or drop the global one and require explicit copy.
  No urgency — current 121h behavior is fine for the immediate need.
- Migration of `logs/` is bucketed under `logs/imported_<date>/`
  alongside per-run dirs. If existing log-parsing code expects flat
  `logs/`, will need a small compat shim — but nothing observed yet
  to suggest the migration would break anything.

### Pickup checklist for next session

1. `cd` into the project, confirm tests still green:
   `python tests/smoke_kalman_pose_smoother_v2.py && \
    PYTHONPATH=. python tests/smoke_project_layout.py && \
    PYTHONPATH=. python tests/smoke_migrate_project.py`
2. Decide between the layout-consumer work (item 1 above) vs. the
   smoother polish work (live progress, cancel) vs. continuing
   real-data tuning of the 121d/e behavior.
3. If continuing real-data tuning: run with the recommended config
   above, compare against the 121bc baseline you have. q_jerk values
   are now in the saved model; check whether EM is adapting them
   reasonably or hitting caps. If hitting caps, the cap values may
   need rethinking.
