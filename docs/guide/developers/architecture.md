# D1. Architecture

This chapter is for people changing Mufasa's code. It covers how the package is laid out, how data
moves through it, and the few APIs most code goes through.

## The package

```text
mufasa/
├── project_layout.py        the project on disk: paths, runs, metadata, model library  (start here)
├── section_provenance.py    which workbench section has run on what (the ✓ / ○ marks)
├── legacy_layout.py         the only module that knows the old SimBA layout
├── cli/                     `mufasa` and its commands (commands.py, run.py, workbench_launcher.py)
├── ui_qt/                   the workbench (PySide6)
│   ├── workbench.py           the main window, pages, sections, OperationForm
│   ├── workbench_app.py       builds the window: every page, in sidebar order
│   ├── pages/                 one module per sidebar page (build_*_page)
│   ├── forms/                 the sections' forms (OperationForm subclasses)
│   ├── dialogs/ · popups/     modal and pop-out windows
│   ├── feature_viewer.py · feature_panels.py · pose3d_viewer.py · pose3d_animation.py
│   │                          the viewers (their own windows)
│   ├── frame_scrubber.py      the video player every viewer uses
│   ├── panel_capture.py       recording a viewer's visible panels into a video
│   └── progress.py            run_with_progress: background work with a progress bar
├── pose_importers/          tracker formats → the project's pose files
├── data_processors/         smoothing (Kalman v2), interpolation, alignment, …
├── outlier_tools/           SimBA outlier correction
├── feature_extractors/      feature families (feature_subsets.py, …_kernels.py, …_orchestration.py)
├── labelling/ · third_party_label_appenders/   labels, and other programs' annotations
├── model/                   classifiers: training, inference, YOLO fitting, sessions
├── roi_tools/               ROI analysis and features
├── plotting/                every visualisation
├── video_processors/        video editing
├── pose3d/                  3-D: camera, mono and stereo inference, Kalman, gait, files, the GPU solver
├── bodymodel/               the rigged body model: rig import, fitting
├── biomech/                 the virtual rodent in MuJoCo: dynamics, muscles, optimisation, smoother
├── tools/                   the other console tools (mufasa-pose3d, mufasa-view3d, …)
├── utils/                   I/O, paths, config, enums, …
├── mixins/                  shared base classes (config readers, plotting, …)
├── _native/                 compiled feature kernels (Cython; a proof of concept so far)
└── assets/                  lookups, icons, built-in models (the virtual rodent, the Rat7M prior)
```

The workbench code lives only in `mufasa/ui_qt`. The other packages need no Qt, so the command line
and scripts can import them, and so can tests run without a display. Keep it that way: a module
outside `ui_qt` must not import PySide6. `mufasa/pose3d/files.py` was split out of the 3-D viewer
for this reason.

## How data moves

```text
sources/pose/  ──import──►  derived/pose/<run>/
                                 │
       interpolate · Kalman v2 · outlier correction
                                 │  (each writes derived/<stage>/<run>/ and publishes it to…)
                                 ▼
                     derived/outlier_corrected/<run>/   ◄── what later stages read
                                 │
        ┌────────────────────────┼─────────────────────────────┐
        ▼                        ▼                             ▼
 derived/features/<family>/   derived/pose3d/<run>/        ROI analysis
        │                     (_3d, _gait, _dynamics)
        ▼
 derived/labels/ ─► models/generated_models/*.sav ─► derived/classifications/ ─► analysis, plots
```

Everything is parquet, keyed by the video's name (the stem). Files are matched across stages by
that name.

## The project layout API

Code never builds project paths by hand. It goes through `mufasa.project_layout`:

```python
from mufasa.project_layout import (
    ProjectPaths, find_project_config, generate_run_id, global_model_cache_dir,
    project_metadata_from_config, project_paths_from_config, publish_to_stage)

cfg = find_project_config(Path.cwd())          # the project.toml above a folder, or None
paths = project_paths_from_config(cfg)          # dict: project_root, video_dir, input_pose_dir,
                                                #   models_dir, derived_features_dir, logs_dir, …
meta = project_metadata_from_config(cfg)        # dict: body_parts, animal_count, animal_ids,
                                                #   classifier_targets, file_type, …

pp = ProjectPaths(Path(paths["project_root"]))
run = pp.stage_run_dir("smoothed", flavor="kalman_v2")   # derived/smoothed/kalman_v2/<new run id>/
latest = pp.latest_run("pose3d")                          # the newest run of a stage, or None
```

* `project_paths_from_config` also answers for old-layout projects, so callers don't branch on
  the layout.
* `generate_run_id()` gives a run id (`YYYYMMDD-HHMMSS-<6 hex>`), and `is_run_id(name)` recognises
  one. Ids sort by time.
* `publish_to_stage(...)` links a run into another stage. This is how a Kalman v2 run becomes the
  `outlier_corrected` input of the later stages.
* `global_model_cache_dir()` is `~/.config/mufasa/models/`, the user's model library.

The layout in full: [`../../v1_project_layout.md`](../../v1_project_layout.md).

## Provenance: the ✓ and ○ marks

`mufasa.section_provenance.SECTIONS` declares each producing section:

* its id;
* its page and section title;
* the sections it depends on;
* where its output is (`detect_path`).

After a section runs, `record_run(config, section_id, run_id)` stores the run under
`[provenance.<section_id>]` in `project.toml`. `get_status` compares a section's last run with
those of the sections it depends on: a dependency that ran later makes it stale (○). The
dependencies form a DAG, checked by `_validate_dag` at import time.

## The workbench

* **`Workbench`** (`workbench.py`) is the main window:
  * the sidebar and a stack of pages;
  * the File, Tools and Help menus;
  * the status bar.

  `add_page(title, icon_name)` adds a page, and `add_tools_action(label, callback)` adds an
  entry to the Tools menu.
* **`WorkflowPage`** is a column of collapsible sections.
  * `add_section(title, [(FormClass, kwargs), …])` adds a section of forms.
  * `add_section_widget(title, factory)` adds any widget, such as the ROI editor or the 3-D
    viewer launcher.

  Sections are built when first opened, so start-up stays quick.
* **`OperationForm`** is the base of almost every section:
  * `build()` lays out the form in `self.body_layout`;
  * `collect_args()` reads and validates the inputs, raising `RuntimeError` with a message on a
    bad input;
  * `target(**kwargs)` does the work.

  The base class wires the Run button, runs `target` in the background through
  `run_with_progress` (progress bar, cancel), and records provenance. A form opts into
  provenance with the class attributes `section_id` and `publish_*_stage`. To record a run, it
  sets `self._last_run_id` in `target`.
* **`workbench_app.py`** builds the window: one `build_*_page(wb, config_path=…)` call per page,
  in sidebar order.

The viewers (feature viewer, 3-D viewer) are not pages. They open in windows of their own
(`feature_panels.ViewerWindow`), share a `FrameScrubberWidget` for playback, and record through
`panel_capture`.

## The command line

* `mufasa` is `mufasa.cli.workbench_launcher:main`. A first argument in its `COMMANDS` goes to
  `mufasa.cli.commands`; otherwise it opens the workbench.
* `mufasa.cli.commands.COMMANDS` maps each command's name to its handler and its one-line help,
  and `LIST_SUBJECTS` does the same for `mufasa list`.
* `mufasa run` is `mufasa.cli.run`. `STEPS` maps each pipeline name to its `mufasa-pose3d`
  subcommand, and `CHAIN` is the order `run all` follows.
* The other tools are console scripts in `pyproject.toml`'s `[project.scripts]`.

The commands import no Qt, and tests check that they don't.

## Native code and the GPU

* The feature kernels are NumPy and numba (`feature_extractors/feature_subset_kernels.py`).
  `mufasa/_native/` holds Cython ports, compiled on install. So far there is one, the inside-ROI
  test, which is checked for byte-equal output and not yet the default.
* The 3-D solver has three back ends (`mufasa/pose3d/gpu.py`):
  * PyTorch on CUDA;
  * PyTorch on the CPU;
  * SciPy.

  They are chosen at run time (`--device auto`).
* Video encoding goes through FFmpeg, with NVENC when present (`ui_qt/linux_env.py` detects it).
