# CHANGELOG

Working notes on what landed in each session, what's deferred, and what
to pick up next time. Keep entries dated and grouped by patch series so
"where did this behavior come from" is answerable from git log alone.

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
