# v1 project layout

User and developer reference for Mufasa's v1 project layout.

**Audience:** users creating new projects + developers writing backend code that needs to read/write project paths.

**Scope:** explains what the v1 layout looks like on disk, why it differs from legacy SimBA, the run-id semantics, and the path-abstraction layer that lets code work on both layouts.

---

## TL;DR

A v1 Mufasa project is a directory containing a `project.toml` file plus a structured set of subdirectories for sources, derived data, models, and logs. Pipeline stages write to per-run subdirectories under `derived/`, so multiple runs of the same stage don't clobber each other.

```
my_project/
├── project.toml                    ← project metadata; layout marker
├── sources/                        ← inputs you provide; never written
│   ├── videos/
│   ├── pose/
│   ├── annotations/                ← optional, for supervised training
│   └── video_info.csv              ← calibration: fps, resolution, pixels/mm
├── derived/                        ← pipeline-stage outputs
│   ├── pose/20260319-104230-a3f1b9/            ← import run (links to sources/pose)
│   ├── interpolated/<run_id>/
│   ├── smoothed/kalman_v2/<run_id>/
│   ├── outlier_corrected/<run_id>/             ← downstream input; runs published here
│   ├── features/<family>/<video>.parquet
│   ├── labels/<video>.parquet
│   ├── classifications/<video>.parquet
│   └── plots/ · frames/{extracted,annotated}/ · directionality/ · …
├── models/                         ← Kalman models and classifiers
├── logs/                           ← run logs and project-wide measures
│   └── measures/
│       └── rois.json               ← ROI definitions for every video
└── backups/                        ← snapshots before in-place edits
```

Compare to the legacy SimBA layout, where everything lives flat under `<project>/csv/`, `<project>/videos/`, etc., with no run separation.

---

## Why v1 exists

The legacy layout has two problems v1 fixes:

* **No run separation.** Re-running outlier correction with different parameters overwrites the previous output. v1's per-run subdirectories (`derived/<stage>/<run_id>/`) preserve every run; the latest is the default unless code explicitly picks an older one.
* **Inconsistent grouping.** Legacy mixes inputs (`videos/`), intermediate pipeline outputs (`csv/features_extracted/`), final outputs (`csv/machine_results/`), and provenance (`logs/`) in a single flat tree. v1 separates them into `sources/` (inputs you provide), `derived/` (computed), `models/` (trained), `logs/` (provenance).

Only the v1 layout is supported; legacy SimBA projects (`project_config.ini`) are not read (support removed in patches 122dw–122dz).

---

## Directory contents

### `sources/`

Inputs you provide. Pipeline reads these; never writes to them.

| Path | Contents |
|---|---|
| `sources/videos/` | Video files. Subdirectories OK; the workbench's video picker recurses. |
| `sources/pose/` | Raw pose estimation outputs (DLC, SLEAP, MARS, etc.) prior to outlier correction. |
| `sources/annotations/` | Per-video annotation CSVs (one row per frame) used as training targets. Optional. |
| `sources/video_info.csv` | Per-video metadata: pixels/mm, fps, resolution. Edited via Preprocessing → Video Calibration ("Save calibration"). |

### `derived/`

Pipeline-stage outputs. Each major stage has its own subdirectory; within most of them, runs are scoped by run-id.

| Stage | Path | Run-scoped? | Notes |
|---|---|---|---|
| Pose import | `derived/pose/<run_id>/` | Yes | Relative links to the imported files in `sources/pose/`. |
| Interpolation | `derived/interpolated/<run_id>/` | Yes | Missing frames filled. |
| Kalman v2 smoothing | `derived/smoothed/kalman_v2/<run_id>/` | Yes | `<video>_smoothed_v2.<hash>.parquet` plus `run.toml`. |
| Outlier correction | `derived/outlier_corrected/<run_id>/` | Yes | The pose that downstream stages read. Import, interpolation and smoothing runs are *published* here as relative links, renamed to plain video names so ROIs and calibration match. |
| Feature extraction | `derived/features/<family>/<video>.parquet` | No (per family) | One subtree per feature family, so families can be added incrementally (122ae). |
| Labels (targets) | `derived/labels/<video>.parquet` | No (flat) | Training targets aligned with features; edited in place. `derived/labels/annotation_status.toml` records which videos the annotator marked complete (122jc). |
| Classifications | `derived/classifications/` | No (flat) | Per-video predictions; one parquet per video. Patch 122ax decision: flat for simpler downstream reads. |
| Directionality | `derived/directionality/` | No (flat) | Animal-to-animal directing data. |
| Frame extraction | `derived/frames/extracted/<video>/` | No | Per-video extracted frame dumps. |
| Annotated frames | `derived/frames/annotated/<video>/` | No | Frames with annotations overlaid. |

When a backend needs the "current data" for a stage, it picks the latest run (lexicographic-sort, since run-ids are timestamp-prefixed). When it needs to write fresh output, it allocates a new run-id and writes there.

### `models/`

Trained models.

```
models/
├── 20260319-104230-a3f1b9/
│   ├── model.npz              ← fitted Kalman v2 smoother model
│   └── card.toml              ← where it came from (source path, hash)
└── attack.sav                 ← trained classifier (or .onnx with the onnx extra)
```

Kalman models are mirrored to a global cache at `~/.config/mufasa/models/` so a model fitted in one project can be loaded in another (`import_model_into_project` / `mirror_model_to_global_cache` in `mufasa.project_layout`). Classifier training and inference settings are not separate files; they live in `project.toml` under `[classifier_training]` and `[classifier_inference.<name>]`.

### `logs/`

Run-level logs and project-wide measures.

| Path | Contents |
|---|---|
| `logs/measures/rois.json` | ROI shapes for every video, keyed by video name (122il; replaces SimBA's `ROI_definitions.h5`). |
| `logs/measures/pose_configs/` | Pose configuration metadata. |
| `logs/<run_id>/run.toml` | Per-run provenance (parameters, input refs, completion status). |

### `project.toml`

Top-level project metadata. Sections:

```toml
[project]
name           = "my_project"
created        = "2026-03-19T10:42:30Z"
layout_version = "v1"

[pose]
file_type      = "parquet"
animal_count   = 2
body_parts     = ["nose", "left_ear", "right_ear", "tail_base", ...]

[settings]
# Per-project overrides for inference / training defaults.

[features]
# Optional (122iz): which markers the HEAD VELOCITY IN BODY FRAME family
# treats as head and body. Without these, the "head" segment and the root
# segment of [[pose.segments]] are used, else marker names (head / nose /
# ear vs back / body / center / spine …).
head_markers = ["head_front", "head_back"]
body_markers = ["back_T4", "back_T8"]
# or pick segments of [[pose.segments]] by name:
# head_segment = "head"
# body_segment = "back"
```

Backends read this via `mufasa.project_layout.read_project_toml(path)`.

---

## Run-id format

Run-ids are timestamp-prefixed strings:

```
YYYYMMDD-HHMMSS-XXXXXX
20260319-104230-a3f1b9
```

* 8-digit date + 6-digit time + 6 hex characters.
* Generated by `mufasa.project_layout.generate_run_id()`.
* String-sortable: lexicographic order matches chronological order.
* The 6 hex characters de-duplicate runs started in the same second.
* Validated by `mufasa.project_layout.is_run_id(name)`.

Backend code that wants "the latest run for stage X" follows this pattern (lifted from `ConfigReader._apply_v1_path_overrides`):

```python
from mufasa.project_layout import is_run_id

def latest_run_or_parent(stage_dir: Path) -> Path:
    """Latest run subdir under stage_dir, or stage_dir itself if
    no run subdirs exist."""
    if stage_dir.is_dir():
        runs = sorted(d for d in stage_dir.iterdir()
                      if d.is_dir() and is_run_id(d.name))
        if runs:
            return runs[-1]
    return stage_dir
```

---

## Path abstraction layer

Backend code should never hardcode legacy vs v1 paths. Use the layout-agnostic helpers in `mufasa.project_layout`.

### `project_paths_from_config(config_path)`

The main entry. Returns a dict with layout-agnostic paths:

```python
from mufasa.project_layout import project_paths_from_config

paths = project_paths_from_config(config_path="my_project/project.toml")
# Returns:
# {
#   "project_root":                  "/abs/path/to/my_project",
#   "video_dir":                     "/abs/path/to/my_project/sources/videos",
#   "input_pose_dir":                "/abs/path/to/my_project/sources/pose",
#   "logs_dir":                      "/abs/path/to/my_project/logs",
#   "video_info_path":               "/abs/path/to/my_project/sources/video_info.csv",
#   "models_dir":                    "/abs/path/to/my_project/models",
#   "roi_definitions_path":          ".../logs/measures/rois.json",
#   "derived_features_dir":          ".../derived/features",
#   "derived_labels_dir":            ".../derived/labels",
#   "derived_classifications_dir":   ".../derived/classifications",
#   "machine_results_dir":           ".../derived/classifications",   # legacy-named alias
# }
```

Passing a legacy `.ini` path returns the corresponding legacy paths instead — same keys, different values. The caller doesn't need to know which layout.

### Other helpers

```python
detect_layout(path)               # "v1" | "legacy" | "unknown"
resolve_v1_project_root(path)     # locates v1 root from any path inside
read_project_toml(path)           # parses project.toml into a dict
write_project_toml(path, data)    # serializes a dict to project.toml
project_metadata_from_config(...) # animal count, body parts, file type
generate_run_id()                 # fresh "YYYYMMDD-HHMMSS-XXXXXX"
is_run_id(name)                   # validates the format
read_run_toml(path)               # per-run provenance file
write_run_toml(path, data)        # ditto
```

### Triage rules (when reviewing existing code)

A scan for `os.path.join(X, "Y")` patterns will surface many false positives. Use these rules:

| Pattern | Bug? | Why |
|---|---|---|
| `os.path.join(self.project_path, "logs", …)` on a `ConfigReader` subclass | NO | `self.project_path` IS the project root for both layouts; `logs/` resolves identically. |
| `os.path.join(self.project_path, "videos", …)` | YES (v1-breaking) | Legacy uses `<root>/videos`; v1 uses `<root>/sources/videos`. |
| `os.path.join(self.project_path, "csv", …)` | YES (v1-breaking) | v1 has no top-level `csv/` tree. |
| `os.path.join(project_path, …)` where `project_path` is from `config.get("General settings", "project_path")` | YES | `configparser.get()` doesn't work on `.toml`; the value is bogus for v1. Use `project_paths_from_config()` instead. |
| Inside `try: … except: <legacy fallback>` | NO | Defensive fallback for malformed configs. Acceptable. |
| Inside `if not is_v1:` / `if v1_root is None:` branches | NO | Branch-gated to legacy; intentional. |
| In `utils/toml_to_configparser.py`, `tools/csv_to_parquet.py` | NO | These tools' job IS the legacy layout. Intentional. (Patch 122dw deleted `cli/migrate_project.py`, which used to be on this list too.) |

See [`hardwired_paths_audit.md`](hardwired_paths_audit.md) for the full audit + per-site disposition.

---

## ConfigReader integration

`ConfigReader` (in `mufasa/mixins/config_reader.py`) detects layout from the config path and sets layout-aware attributes on `self`. The cost of instantiating one is non-trivial (parses TOML or INI, validates body parts, etc.), so for one-off path lookups prefer `project_paths_from_config()`. For backend modules that already extend `ConfigReader`, use its attributes directly — `self.video_dir`, `self.input_csv_dir` (yes, name is legacy-flavored but the value is layout-correct), `self.outlier_corrected_movement_dir`, `self.features_dir`, `self.machine_results_dir`, `self.input_frames_dir`, etc.

For v1 projects, ConfigReader's `_apply_v1_path_overrides` resolves multi-run stages to the latest run subdir (or the stage parent if no runs exist yet) — i.e., the attribute always points at a directory that should contain data, not at the parent of a tree of runs.

---

## Creating a new v1 project

Two paths:

**Workbench:** Use the workbench's project chooser → New Project. The form writes `project.toml`, creates the required directory tree, and seeds an empty `sources/video_info.csv`.

**Programmatically** (for scripts / CI):

```python
from pathlib import Path
from mufasa.project_layout import write_project_toml

root = Path("my_project")
for sub in [
    "sources/videos", "sources/pose", "sources/annotations",
    "derived", "models", "logs/measures",
]:
    (root / sub).mkdir(parents=True, exist_ok=True)

write_project_toml(root / "project.toml", {
    "project": {
        "name": "my_project",
        "created": "2026-03-19T10:42:30Z",
        "layout_version": "v1",
    },
    "pose": {
        "file_type": "parquet",
        "animal_count": 2,
        "body_parts": [
            "nose", "left_ear", "right_ear", "tail_base",
            # …whatever your pose model produces…
        ],
    },
    "settings": {},
})
```

Then drop videos into `sources/videos/`, pose CSVs into `sources/pose/`, and open the project in the workbench.

---

## Migrating an existing legacy project

**No longer supported.** Patch 122dw deleted the legacy → v1 migration tool. v1 is the only supported project layout going forward; if you have an existing `project_folder/`-style project, create a fresh v1 project (see "Creating a new v1 project" above) and copy your source data into the new `sources/` tree by hand. The legacy reader in `mufasa.legacy_layout` is still in the codebase for backward source-compatibility of in-repo code paths, but no longer exposes a migration path.

---

## References

* Layout helpers: [`mufasa/project_layout.py`](../mufasa/project_layout.py) — 18 public functions; the source of truth.
* ConfigReader integration: [`mufasa/mixins/config_reader.py`](../mufasa/mixins/config_reader.py), specifically `_apply_v1_path_overrides`.
* Triage rules: [`hardwired_paths_audit.md`](hardwired_paths_audit.md).
