# Mufasa workflow audit

This document maps the operational paths through Mufasa (and its
SimBA ancestor) — what's there, what each branch does, what's
broken, what the Qt port covers vs. misses. It's a reverse-
engineered guide to a codebase that grew organically over many
years.

## Audit status legend

Each workflow entry is tagged:

- 🟢 **Deeply audited** — code path traced, reads/writes verified,
  branches mapped, status assessed.
- 🟡 **Shallow listed** — entry point and rough purpose noted, but
  internals not yet traced. Treat anything not contradicted by
  Anthropic's testing as unknown.
- 🔴 **Known broken** — confirmed not-functional in current state,
  with reason.
- ⚪ **Not yet examined** — listed for completeness but no trace
  done.

This file is a living document. Each session adds depth.

## How Mufasa is organized

The legacy launcher (`mufasa/SimBA.py` → `mufasa-tk` console script)
is the reference implementation. It opens a top-level window with
a menu bar (file/process-videos/tools/etc.) plus a per-project
window with **10 tabs**:

```
Further imports | Video parameters | Outlier correction | ROI |
Extract features | Label behavior | Train machine model |
Run machine model | Visualizations | Add-ons
```

The Qt port (`mufasa-qt`, `mufasa/ui_qt/`) consolidates these into
a smaller workbench. Coverage as of 2026-04-30:

| Tk tab | Qt page | Coverage |
|--------|---------|----------|
| Further imports | Data Import | Partial |
| Video parameters | Data Import (Video parameters & calibration section) | New, full |
| Outlier correction | — | None |
| ROI | ROI | Full |
| Extract features | Compute → Feature subsets | Partial (subset families only) |
| Label behavior | — | None |
| Train machine model | — | None |
| Run machine model | — | None |
| Visualizations | — | None |
| Add-ons | — | None |
| (Tools menu, video processing) | Video Processing | Substantial |

The tabular gap here is the source of truth for "what's missing
in Qt": classifier training/inference, label-behavior, and
visualizations are entirely Tk-only at present.

---

# Project lifecycle

## Create a new project 🟡
**Tk entry**: File → Create a new project (`ProjectCreatorPopUp`)
**Qt entry**: Project Setup page → "Create new project" form
**Reads**: nothing (writes a new tree)
**Writes**: `project_folder/project_config.ini`,
  `project_folder/{csv,videos,logs,frames,models}/...` skeleton
**Branches**:
- Pose-config dropdown (4bp / 7bp / 8bp / 9bp / 14bp / 16bp /
  user-defined / multi-animal variants) — sets `bp_headers` in
  the config and creates `project_bp_names.csv`
- File type radio: csv / parquet — sets `[General settings]
  file_type`
- Animals-per-classifier integer
- Initial classifier names (comma-separated)

## Load project 🟡
**Tk entry**: File → Load project (`LoadProjectPopUp`)
**Qt entry**: File → Open project, or auto-discover via
`mufasa-qt` reading recent-projects file
**Reads**: `project_config.ini`
**Branches**: none (it's just a path picker that hands off to
`SimbaProjectPopUp`).

## Restart 🟡
**Tk entry**: File → Restart
**Qt entry**: none
Spawns a fresh Python process with the same args. Used for
clearing application state without manually killing the window.

## Recent projects ⚪
Tracks last-N opened project paths in
`~/.mufasa_recent_projects.json` (or similar). Both launchers
read the same file.

---

# Stage 1: Data ingestion (the "Further imports" tab + standalone importers)

The canonical pipeline starts with importing pose-estimation
data into `csv/input_csv/`. There are many possible source
formats.

## Pose import: DLC (DeepLabCut) 🟢
**Tk entry**: ProjectCreator's "Import pose" section, or via
project popup's "Import pose data" submenus
**Qt entry**: Data Import page → "Import pose-estimation data"
form
**Backend**: `mufasa/pose_importers/dlc_h5_importer.py`,
  `mufasa/pose_importers/dlc_csv_importer.py`
**Reads**: source `.h5` / `.csv` files (DLC's multi-row-header
format), `project_bp_names.csv`
**Writes**: `csv/input_csv/<video>.csv` or `.parquet` (depending
on project file_type), preserving DLC's 3-row multi-index header
when written via `write_df(multi_idx_header=True)`. **This is
the format that broke our parquet migration tool — see patch 64
csv_to_parquet_multiindex.**

**Branches**:
- Single-animal vs multi-animal DLC (different importers for
  each — see `dlc_multi_animal_importer.py` and `madlc_importer.py`)
- File detection: `_dlc_autodetect_format` decides h5 vs csv
  based on header probing
- Likelihood masking: optional post-import step that NaN-replaces
  poses below a per-bodypart confidence threshold
- Interpolation: optional gap-filling for masked / missing values
  (linear, quadratic, nearest)

**Status**: Functional in both Tk and Qt. The Qt port covers
DLC h5 + csv directly. Likelihood masking happens at import
time in both UIs.

**Recent fixes**:
- `dlc_h5_importer_fix`, `dlc_csv_importer_fix`,
  `likelihood_threshold_fix`, `likelihood_mask_suffix_fix`,
  `dlc_autodetect_fix` — all addressing edge cases in the
  importer detection / threshold-application chain

## Pose import: SLEAP 🟡
**Tk entry**: ProjectCreator → "Import pose" → SLEAP h5 / SLEAP csv
**Qt entry**: not yet ported
**Backend**: `mufasa/pose_importers/sleap_h5_importer.py`,
  `sleap_csv_importer.py`
**Status**: Functional in Tk. Format is similar enough to DLC
that porting to Qt is straightforward when needed.

## Pose import: MARS, MADLC, FaceMap, NicCage, TRK 🟡
**Tk entry**: ProjectCreator → "Import pose" → various
**Qt entry**: not ported
**Backends**:
- `mars_importer.py` (Mouse Action Recognition System)
- `madlc_importer.py` (multi-animal DLC variant)
- `facemap_importer.py` (mouse facial pose)
- `niccage_importer.py` (Nicholas-Cage-style — older format,
  rarely seen)
- `trk_importer.py` (LightningPose .trk files)
**Status**: Believed functional in Tk; not exercised in any
recent session. **At risk** of bit-rot — none of these have
been touched in our 67-patch stack.

## Import frame directory 🟡
**Tk entry**: Further imports → IMPORT FRAMES DIRECTORY TO SIMBA
PROJECT (`ImportFrameDirectoryPopUp`)
**Qt entry**: not ported
Copies a directory of pre-extracted PNG/JPG frames into
`project_folder/frames/input/<video>/`. Used when working from
already-decoded frames rather than video files.

## Add classifier / Remove classifier 🟢
**Tk entry**: Further imports → ADD/REMOVE CLASSIFIER
**Qt entry**: not ported
**Backend**: edits `[SML settings]` and `[create_ensemble_settings]`
sections of `project_config.ini`, plus body-part mapping CSVs.
**Status**: Functional. Edits config + creates empty model
folders; doesn't touch any data files. Low-risk operation.

## Archive processed files 🟡
**Tk entry**: Further imports → ARCHIVE PROCESSED FILES
**Qt entry**: not ported
**Backend**: `archive_files_pop_up.py`
Moves a snapshot of `csv/`, `models/`, `logs/` into a timestamped
archive directory inside the project. Used to preserve a known-
good state before destructive operations.

## Reverse tracking identities 🔴
**Tk entry**: Further imports → REVERSE TRACKING IDENTITIES IN SIMBA PROJECT
**Status**: **Wired to `cmd=None`** in the launcher source — the
button exists but does nothing. Either the implementation was
never finished or it was deliberately disabled and the button
forgotten. (Source: `mufasa/SimBA.py` line near
`reverse_btn = SimbaButton(... cmd=None)`.)

## Interpolate pose 🟢
**Tk entry**: Further imports → INTERPOLATE POSE IN SIMBA PROJECT
(`InterpolatePopUp` from `mufasa/ui/pop_ups/interpolate_pop_up.py`)
**Qt entry**: not ported
**Backend**: `mufasa/data_processors/interpolate.py` →
  `Interpolate.run()` → `animal_interpolator` /
  `body_part_interpolator` from `mufasa/utils/data.py`
**Reads**: `csv/<subdir>/<video>.csv` (any subdirectory under
  `csv/`; user picks via the popup),
  `project_bp_names.csv` (via ConfigReader's `bp_headers`)
**Writes**: same path as input — **interpolation is in-place**.
  If `copy_originals=True`, the original is first copied to a
  timestamped sibling directory:
  `<input_dir>/Pre_<method>_<type>_interpolation_<datetime>/<video>.csv`
**Branches**:
- **Type dropdown** — two algorithms with completely different
  detection logic (these are not interchangeable):
    - `MISSING BODY-PARTS` → `body_part_interpolator`. Per
      bodypart per animal: a frame is "missing" when both x AND
      y of that bodypart are ≤ 0.0. Each bodypart interpolated
      independently in time.
    - `MISSING ANIMALS` → `animal_interpolator`. Per animal:
      a frame is "missing" when **all bodypart x-values are
      identical** (i.e., the entire animal collapsed to a point).
      Interpolates the whole animal block at once. **Only useful
      for multi-animal projects** where one animal can disappear
      while another stays present.
- **Method dropdown** — passed through to pandas's `.interpolate()`:
    - `NEAREST` (default), `LINEAR`, `QUADRATIC`. Applied via
      `df[col].interpolate(method=method).ffill().bfill()` so
      gaps at the start or end of the timeline get filled with
      the nearest valid value.
- **Save originals** — boolean. Off by default in the Tk popup
  but Mufasa users typically leave it on; without it, a bad
  interpolation run silently overwrites the source data.
- **Single file vs directory** — the popup has both a
  FileSelect and a FolderSelect. FolderSelect processes every
  `.csv` (or `.parquet` per `file_type`) in the folder.
- **Multi-index header detection** (popup, line 82-84): uses
  `Path.resolve().absolute()` comparison against
  `self.input_csv_dir`. **Correct** — this path is what passes
  the auto-detection challenge that the Smoothing popup gets
  wrong (see below).

**Important behavior — what gets interpolated**:
- Only `X_bps + Y_bps` columns are touched. The `_p` (likelihood)
  columns are **NOT** interpolated, which is correct: a likelihood
  is a measurement confidence and shouldn't be synthesized.
- For body-part mode, missing detection is `(x <= 0) AND (y <= 0)`.
  This means a bodypart at the literal corner (0, 0) is treated
  as missing. In practice, DLC and SLEAP rarely produce exactly
  (0, 0) values for valid detections (they output floats with
  noise), but it's a corner case worth knowing about.
- For animal mode, missing detection is "all x columns equal" —
  detects when an entire animal collapsed to one point (e.g., a
  legacy DLC behavior when no detection passes confidence).

**Status**: Functional. Backend has been touched defensively
(see "Audit A1 fix" comment in `interpolate.py:83-89` adding
auto-detection of multi-index headers when `multi_index_df_headers`
is `None`).

**Watch-outs**:
- `df.fillna(0)` then `df < 0 → 0` then `astype(int)` (line 1302
  for animals, line 1345 for body-parts) — input data gets
  **forcibly cast to int32** before interpolation. **This loses
  sub-pixel precision** if the user's pose-estimation produces
  floating-point coordinates (DLC does; SLEAP does). After
  interpolation, the values are written as float32 again, but
  the precision damage is done. Not catastrophic for behavior
  research at typical pixel-scale resolutions but worth knowing
  about for high-resolution work.
- "Animal missing" detection compares against the FIRST column
  (`animal_df.iloc[:, 0]`). If your first body-part is genuinely
  motionless across many frames (e.g., a tail-tracker on a
  mostly-stationary animal), you'd get false positives. Reorder
  body parts so the most-mobile one is first, or use body-part
  mode instead.
- Output is converted to `np.float32` on write (`interpolate.py:120`).
  If your downstream code assumes float64, you'd see precision
  divergence. Mufasa's own pipeline tolerates float32 fine.

## Smooth pose 🟢
**Tk entry**: Further imports → SMOOTH POSE IN SIMBA PROJECT
(`SmoothingPopUp` from `mufasa/ui/pop_ups/smoothing_popup.py`)
**Qt entry**: not ported
**Backend**: `mufasa/data_processors/smoothing.py` →
  `Smoothing.run()` → `savgol_smoother` / `df_smoother` from
  `mufasa/utils/data.py`
**Reads**: same as interpolate (any csv subdir),
  `logs/video_info.csv` OR raw video file (for FPS lookup)
**Writes**: in-place at the input path. Same backup convention
  as interpolate when `copy_originals=True`:
  `<input_dir>/Pre_<method>_<time_window>_smoothing_<datetime>/`.
**Branches**:
- **Method dropdown** — two routes:
    - `Savitzky Golay` → `savgol_smoother`. Polynomial fit
      (order 3) within a sliding window. Handles edges via
      `mode='nearest'`.
    - `Gaussian` → `df_smoother(method='gaussian')`. Pandas
      rolling window with Gaussian weighting (`std=5`,
      hardcoded). Centered window, NaN edges filled with
      original values via `.fillna(data[c])`.
- **Time window** — milliseconds, converted to frames via
  `frms_in_smoothing_window = int(time_window / (1000 / fps))`.
  Default 100ms in the popup. **Critical**: if `time_window`
  is short relative to `fps`, the result will be a no-op:
  `if frms_in_smoothing_window <= 1: return data` (line 1396 for
  savgol, 1452 for gaussian).
- **Polynomial order** for savgol is **hardcoded to 3** in the
  kernel — the popup doesn't expose it. To override, you'd have
  to call `savgol_smoother` programmatically.
- **Save originals** — same as interpolate.

**FPS source** — required because time window converts to frames
via FPS:
1. First tries `find_video_of_file(self.video_dir, video_name)`
   to locate the source video; reads FPS via
   `get_video_meta_data` (`cv2.VideoCapture` under the hood)
2. Falls back to reading `logs/video_info.csv` if no video file
   is found
3. Raises `NoFilesFoundError` if neither source is available

**This means smoothing requires either the actual video file
in `videos/` or a populated `logs/video_info.csv` — the same
calibration table we just built the Qt form for.** Without it,
smoothing flat-out fails.

**Critical bug — `_p` likelihood columns get smoothed too**:
Both kernels iterate over all DataFrame columns (`for c in
data.columns`). They don't distinguish between coordinate
columns (`_x`, `_y`) and likelihood columns (`_p`). This means:
- `_p` values get rolling-averaged across the time window.
- After smoothing, a likelihood of 0.99 followed by a 0.01
  followed by 0.99 (a single-frame dropout) becomes ~0.66 across
  three frames.
- Downstream code (e.g., likelihood masking in feature
  extraction) reads the smoothed likelihood and may treat the
  three frames as borderline-valid when only the middle one was
  actually low-confidence.

This is a **real correctness issue** for the standard pipeline,
not just a code-smell. Compare to `body_part_interpolator` and
`animal_interpolator` which deliberately only operate on
`X_bps + Y_bps`. The smoothers should match that behavior.

**Critical bug — popup's multi-index detection is broken**:
`smoothing_popup.py:83`:
```python
multi_index_df_headers = False
if data_dir == self.input_csv_dir: multi_index_df_headers = True
```
This is raw string equality. It fails when:
- The user picks the directory via FolderSelect with a trailing
  slash (`/path/csv/input_csv/` vs `/path/csv/input_csv`)
- Different case on Windows
- Symlinks in the path
- Any `os.path.normpath` quirk

The interpolate popup uses `Path.resolve().absolute()` (correct)
for the equivalent check. The smoothing backend has an
"Audit A1 fix" comment at `smoothing.py:92-102` that adds the
correct detection — but **only when the popup passes
`multi_index_df_headers=None`**. The popup hardcodes the value
it computed (line 89), so the backend fallback is unreachable
from the Tk UI.

**Result**: smoothing a project's `csv/input_csv/` files via
the GUI may write back without re-inserting the multi-index
header rows on environments where the path comparison fails.
The next pipeline stage (outlier correction, feature extraction)
would then fail to read the file with `read_df(check_multiindex=True)`
because the expected 3-row IMPORTED_POSE header is gone.

**Workaround**: smooth files in `csv/outlier_corrected_movement_location/`
or later instead of `csv/input_csv/` — those don't have multi-
index headers and the comparison doesn't matter.

**Watch-outs**:
- Smoothing in-place is permanent. Always run with
  Save Originals = True the first time on a new project.
- Smoothing the `_p` columns (above) — until that's fixed,
  re-run likelihood-mask after any smoothing pass to overwrite
  the corrupted likelihoods with zeros.
- Both savgol and gaussian smoothers `.clip(lower=0)` the data.
  Pose data shouldn't be negative anyway, but if you have
  centered/transformed pose data (e.g., post-egocentric-alignment
  where coordinates are relative to the animal's center and CAN
  be negative), running smoothing afterward will silently zero
  out half your data. Smooth BEFORE alignment.

**Recommended fix priority**: medium. The `_p` smoothing is the
most concerning because it silently corrupts confidence values
that downstream code trusts. The multi-index detection bug only
fires in `csv/input_csv/` and only on certain paths; most users
smooth elsewhere in the pipeline.

---

## Dead code — advanced interpolator/smoother variants 🔴
There are TWO `AdvancedInterpolator` class definitions in the
codebase:
- `mufasa/data_processors/advanced_interpolator.py` (266 lines)
- `mufasa/data_processors/interpolation_smoothing.py:428` (in a
  819-line module that also contains an `AdvancedSmoothing` class)

Plus:
- `mufasa/data_processors/advanced_smoothing.py` (213 lines)

**Nothing in the codebase imports any of these.** Confirmed via
`grep -rn "from mufasa.data_processors.advanced"`. They're
orphaned drafts. Likely the result of a refactor where one
version was promoted and the others were never deleted.

**Recommendation**: delete them in a future cleanup pass. Not
worth doing now — they don't break anything by sitting there,
and confirming "really nothing imports these" needs more
verification than I've done. Flagging for the workflow audit
so future maintainers don't trip on them.

## Egocentric alignment 🟢
**Tk entry**: Further imports → EGOCENTRICALLY ALIGN POSE AND VIDEO
(`EgocentricAlignPopUp`)
**Qt entry**: not ported
**Backend**: `mufasa/data_processors/egocentric_alignment.py`,
  `egocentric_align_gpu.py`
**Reads**: `csv/input_csv/<video>.csv`, source video file
**Writes**: aligned pose CSV + aligned (rotated/translated) video
**Branches**:
- CPU path (`egocentric_alignment.py`) — pure NumPy, multiprocessing
- GPU path (`egocentric_align_gpu.py`) — CuPy-based, optional;
  requires `cupy` import to succeed
- Per-animal vs whole-project mode (sets which body-part anchors
  the rotation per video)
- Anchor body-part dropdown (typically "Center" or "Tail_base")
**Status**: Functional. We touched this in patches
`egocentric_alignment_fix`, `egocentric_pool_lifecycle_fix`,
`egocentric_gpu_empty_batch_fix`, `egocentric_gpu_detect_fix` —
fixed shape-mismatch bugs and a CuPy import-detection issue.
Specific to behavioral neuroscience workflows where the animal
needs to be in a fixed orientation across frames (e.g. for
side-by-side comparison or for downstream models requiring a
canonical pose).

---

# Stage 2: Video parameters / calibration

## Video info table 🟢
**Tk entry**: Tab "Video parameters" → CONFIGURE VIDEO PARAMETERS
(`create_video_info_table` → opens `VideoInfoTable` from
`mufasa/ui/video_info_ui.py`)
**Qt entry**: Data Import page → "Video parameters & calibration"
section (`VideoInfoForm` from `mufasa/ui_qt/forms/video_info.py`)
**Reads**: `videos/`, `csv/input_csv/` (for fallback discovery),
  `logs/video_info.csv` if it exists
**Writes**: `logs/video_info.csv` with columns: Video, fps,
  Resolution_width, Resolution_height, Distance_in_mm, pixels/mm
**Branches**:
- Auto-fill FPS / resolution from `cv2.VideoCapture` (or
  manual entry in the Tk form)
- Per-row pixel calibration via the OpenCV
  CalculatePixelDistanceTool (Tk) or the new Qt
  PixelCalibrationDialog (Qt — replaced the OpenCV widget after
  the truncated-instructions bug)
- "Apply row 1 to all" mode for projects with one camera setup
**Status**: Both functional after patches 65 (form),
67 (calibration class fix), 68 (Qt dialog), 69 (distance writeback).

**Critical**: Without `logs/video_info.csv` populated, ALL
distance-converting feature kernels emit values in pixels not
millimeters. This is the calibration we audited heavily.

## Calibrate pixels/mm (Tools-menu helper) 🟢
**Tk entry**: not present
**Qt entry**: Tools menu → "Calibrate pixels/mm…"
**Backend**: opens the Qt `PixelCalibrationDialog` for a single
chosen video, displays result, doesn't persist
**Status**: Functional. One-off helper for sanity-checking a
single video's calibration; for project-wide persistence, use
the Data Import form.

---

# Stage 3: Outlier correction

## Run outlier correction 🟢
**Tk entry**: Tab "Outlier correction" → RUN OUTLIER CORRECTION
(`SimbaProjectPopUp.correct_outlier`)
**Qt entry**: not ported
**Backend**: `mufasa/outlier_tools/outlier_corrector_movement.py`
+ `outlier_corrector_location.py` run sequentially
**Pipeline**:
1. `OutlierCorrecterMovement(config_path).run()` — flags frames
   where animal moved more than `criterion × inter-bodypart
   distance` between adjacent frames; replaces them with prior
   frame's values
2. `OutlierCorrecterLocation(config_path).run()` — flags frames
   where bodyparts are further apart than expected; same
   correction strategy
**Reads**: `csv/input_csv/<video>.csv`,
  `[outlier_settings]` section of `project_config.ini`
**Writes**: `csv/outlier_corrected_movement/<video>.csv`,
  `csv/outlier_corrected_movement_location/<video>.csv`
**Branches**:
- "Settings" button (`OutlierSettingsPopUp`) — configure the
  per-bodypart criterion multipliers stored in config
- "Skip outlier correction" — `OutlierCorrectionSkipper.run()`
  copies input_csv → outlier_corrected_movement_location
  unchanged. **Useful when poses are already clean**, e.g.
  from MARS or a hand-curated DLC model.
- Multiprocessing variants (`outlier_corrector_*_mp.py`) used
  internally for large projects
- "Advanced" variants (`outlier_corrector_*_advanced.py`) — more
  configurable criteria; opt-in
**Status**: Functional. The standard non-MP path is what most
users hit.

---

# Stage 4: ROI definitions and analysis

(Covered comprehensively by the Qt ROI page after patches
qt_roi_*. The legacy Tk ROI tab and Qt ROI page diverge significantly.)

## Define ROIs 🟢
**Tk entry**: Tab "ROI" → DEFINE ROIs (`ROIVideoTable`)
**Qt entry**: ROI page → Definitions → Action: "Draw ROIs (interactive)"
**Backend**: `mufasa/roi_tools/`, Qt canvas in
  `mufasa/ui_qt/dialogs/roi_canvas.py`
**Reads**: video first frame (cv2 VideoCapture), existing ROI
  definitions from `logs/measures/<shape>/<video>.h5`
**Writes**: ROI definition `.h5` files in `logs/measures/`
**Branches**:
- Three shape types: Rectangle, Circle, Polygon
- Per-video ROI sets (each video has its own definitions)
- "Apply to all videos" replicate mode
- Standardise (rescale ROIs based on a per-video transformation)
- Import ROIs from another project / format
**Status**: Functional in both Tk (legacy) and Qt (extensively
patched). Qt path now uses native canvas (`qt_native_roi_canvas`).

## Delete all ROI definitions 🟡
**Tk entry**: Tab "ROI" → DELETE ALL ROI DEFINITIONS
**Qt entry**: ROI page (similar action)
Wipes all `.h5` files under `logs/measures/`. Confirmation prompt.

## Analyze ROI data: aggregates 🟡
**Tk entry**: Tab "ROI" → ANALYZE ROI DATA: AGGREGATES
(`ROIAggregateDataAnalyzerPopUp`)
**Qt entry**: ROI page → Analyze
**Backend**: `roi_tools/roi_analyzer.py` (likely)
Computes per-video, per-animal, per-ROI statistics: time spent
inside, entry counts, mean velocity inside, transitions between
ROIs, etc. Outputs to `logs/measures/` as a CSV.
**Reads**: `csv/outlier_corrected_movement_location/`,
`logs/measures/` ROI .h5 files, `logs/video_info.csv` (for
distance/velocity unit conversion)
**Writes**: `logs/measures/ROI_<analysis>.csv`

## Analyze ROI data: time-bins 🟡
Same as above but bucketed into user-defined N-second time bins.

## Visualize ROI tracking 🟡
**Tk entry**: VISUALIZE ROI TRACKING (`VisualizeROITrackingPopUp`)
**Qt entry**: ROI page → Visualize
Renders output videos with ROIs overlaid + bodypart trails.

## Visualize ROI features 🟡
**Tk entry**: VISUALIZE ROI FEATURES (`VisualizeROIFeaturesPopUp`)
**Qt entry**: ROI page → Visualize (same form, different action)
Renders heatmaps of bodypart distance to / time spent in each
ROI. Shows where the animal spends its time relative to the
defined geometry.

---

## Other analyses (the "OTHER ANALYSES / VISUALIZATIONS" section in the ROI tab)

These are project-state analyses that don't strictly belong
to the ROI workflow but live in the same UI tab.

| Workflow | Backend popup | Status |
|----------|---------------|--------|
| Movement / velocity aggregates | `MovementAnalysisPopUp` | 🟡 |
| Movement / velocity time-bins | `MovementAnalysisTimeBinsPopUp` | 🟡 |
| Location heatmaps | `HeatmapLocationPopup` | 🟡 |
| Path plots | `EzPathPlotPopUp` | 🟡 |
| Animal-directing-animal analysis | `AnimalDirectingAnimalPopUp` | 🟡 |
| Animal-directing-animal viz | `DirectingOtherAnimalsVisualizerPopUp` | 🟡 |
| Direction between body parts | `DirectionAnimalToBodyPartSettingsPopUp` | 🟡 |
| Boolean conditional aggregate stats | `BooleanConditionalSlicerPopUp` | 🟡 |
| Spontaneous alternation | `SpontaneousAlternationPopUp` | 🟡 |
| Distance analysis (aggregates) | `DistanceAnalysisPopUp` | 🟡 |
| Distance analysis (time-bins) | `TimBinDistanceAnalysisPopUp` | 🟡 |

All of these read from `csv/outlier_corrected_movement_location/`,
some from `logs/measures/` ROI definitions, and all consume
`logs/video_info.csv` for unit conversion.

**None ported to Qt yet.** Some of these are routinely used
in behavioral neuroscience papers (path plots, heatmaps,
spontaneous alternation specifically for T-maze / Y-maze tests)
and would be valuable Qt port targets.

---

# Stage 5: Feature extraction

## Extract features 🟢
**Tk entry**: Tab "Extract features" → EXTRACT FEATURES (config-
specific; the popup has many checkboxes)
**Qt entry**: Compute → Feature subsets (covers a subset of the
full extraction)
**Backend**: `mufasa/feature_extractors/feature_extractor_*.py`
(one per body-part scheme: 4bp, 7bp, 8bp, etc.)
**Reads**: `csv/outlier_corrected_movement_location/<video>.csv`,
  `logs/video_info.csv`, optionally ROI definitions for ROI-relative
  features
**Writes**: `csv/features_extracted/<video>.csv`
**Branches**:
- Body-part scheme dropdown — picks which `feature_extractor_*.py`
  module runs. Each has different feature families.
- Feature subset families (Qt port has these as explicit choices;
  Tk has them implicitly bundled). The subset code lives in
  `mufasa/feature_extractors/feature_subsets.py` after our
  refactor patches.
- "Apply user-defined feature extraction script" — point to a
  custom `.py` file with a `feature_extractor` function. Power-
  user escape hatch.
- Append vs save-to-new-dir destination (Qt port has explicit
  radios after `destination_radios` patch).
**Status**: Functional in both. The Qt port has been extensively
worked on (feature_kernels_extract, feature_orchestration_extract,
cython_*, parallel_spawn_context, hull_parallel_rewrite,
feature_overwrite_confirm, preflight_columns_only, …).

## Append ROI features 🟡
**Tk entry**: Tab "Extract features" → APPEND ROI FEATURES (two
buttons: append by animal, append by body-part)
**Qt entry**: not ported
**Backends**: `append_roi_features_animals_pop_up.py`,
  `append_roi_features_bodypart_pop_up.py`
Adds per-frame ROI columns to existing feature CSVs: distance to
each ROI center, inside/outside booleans, etc. Skipped silently
when no ROIs are defined (`feature_skip_missing_rois_fix` patch).

---

# Stage 6: Behavior labelling

## Label behavior 🟡
**Tk entry**: Tab "Label behavior" → LABEL BEHAVIOR (entry to
  the canonical annotation tool)
**Qt entry**: not ported
**Backend**: `mufasa/labelling/labelling_interface.py`
Frame-by-frame manual annotation interface: scroll through video,
mark behavior class on each frame, save to
`csv/targets_inserted/<video>.csv`. Hotkeys for class assignment.

## Pseudo-labelling 🟡
**Tk entry**: Tab "Label behavior" → PSEUDO-LABELLING
**Backend**: `pseudo_labelling_pop_up.py`
Bootstraps annotations by running an existing model on unlabelled
videos and using high-confidence predictions as labels for
re-training. Useful for active-learning workflows.

## Advanced label behavior 🟡
**Tk entry**: Tab "Label behavior" → ADVANCED LABEL BEHAVIOR
Wider feature set than basic labelling: multi-animal annotation,
severity scoring, etc.

## Targeted clip annotator 🟡
**Tk entry**: Tab "Label behavior" → TARGETED CLIP ANNOTATOR
**Backend**: `targeted_annotator_pop_up.py`
Annotation flow that pre-clips short candidate sequences (e.g.
from a coarse model) and asks the user to confirm/correct each.
Faster than frame-by-frame for sparse behaviors. Related to
`mufasa/ui_qt/targeted_clips.py` which we touched once.

## Import third-party behavior annotations 🔴 (likely)
**Tk entry**: Tab "Label behavior" → IMPORT THIRD-PARTY
**Backends**: a family of importers under `third_party_label_appenders/`:
- BORIS
- BORIS multi-animal
- ETHOVISION
- DeepEthogram
- Solomon
- Observer
- BENTO
**Status**: Likely partially-broken. These haven't been touched
in years; format spec changes from the upstream tools may have
moved on. **Recommend testing with a known-good file from each
tool before relying on them.** Worth a focused audit if any of
these matter for your workflow.

---

# Stage 7: Train machine model

## Train single classifier 🟡
**Tk entry**: Tab "Train machine model" → TRAIN SINGLE MODEL
(`MachineModelSettingsPopUp`)
**Qt entry**: not ported
**Backend**: `mufasa/model/train_*.py`
**Reads**: `csv/targets_inserted/<video>.csv` (features + labels),
  `[create_ensemble_settings]` config
**Writes**: `models/generated_models/<classifier_name>.sav` (pickled
  sklearn / xgboost model), `models/model_eval/` evaluation CSVs
  and plots
**Branches**:
- Classifier algorithm: random forest, XGBoost
- Hyperparameter grid search vs fixed values
- SMOTE / SMOTEENN class balancing on/off
- Bootstrap iterations for ensemble (creates multiple models with
  different bootstrap samples; `BootstrapEnsembleSamplerPopUp`
  is a separate entry)
- Evaluation outputs: PR curve, learning curve, feature importance,
  classification report, etc. (each is opt-in)
**Status**: Functional in Tk. Heavyweight — actually trains
sklearn/xgboost models, can take minutes to hours.

## Train multiple classifiers (one per behavior) 🟡
**Tk entry**: Tab "Train machine model" → TRAIN MULTIPLE MODELS
(`MachineModelMultipleSettingsPopUp` or similar)
Repeats single-model training for each classifier defined in the
project config.

## Hyperparameter search 🟡
**Tk entry**: Tab "Train machine model" → GRID SEARCH (or RANDOM
SEARCH)
Wraps the train flow with a parameter sweep. Outputs best params
to a CSV; user then trains a final model with those params.

---

# Stage 8: Run machine model

## Validate model on single video 🟡
**Tk entry**: Tab "Run machine model" → VALIDATE MODEL ON SINGLE VIDEO
**Backend**: `validate_model_run_clf.py`
Runs the trained model on a single video and produces an
annotated visualization. Used for sanity-checking model
quality before batch inference.

## Run machine model (batch) 🟡
**Tk entry**: Tab "Run machine model" → RUN MACHINE MODEL
(`InferenceBatchPopUp`)
**Reads**: trained `.sav` files in `models/`, feature CSVs in
  `csv/features_extracted/`
**Writes**: `csv/machine_results/<video>.csv` with per-frame
  predicted classes + probabilities

## Analyze machine results 🟡
A family of post-inference analysis popups:
- `clf_descriptive_statistics_pop_up.py` — bout count, total time,
  mean bout duration
- `clf_by_roi_pop_up.py` — class breakdown per ROI
- `clf_by_timebins_pop_up.py` — class breakdown per N-second window
- `clf_annotation_counts_pop_up.py` — confusion / agreement with
  ground-truth labels
- `kleinberg_pop_up.py` — burst detection on prediction sequences
- `severity_pop_up.py` — intensity-weighted classification

---

# Stage 9: Visualizations

## Data visualizations (the "DATA VISUALIZATIONS" section)

| Workflow | Popup | Status |
|----------|-------|--------|
| Visualize classifications | `VisualizeClfPopUp` | 🟡 |
| Probability plots | `clf_probability_plot_pop_up` | 🟡 |
| Validation videos | `clf_validation_plot_pop_up` | 🟡 |
| Gantt plots | `gantt_pop_up` | 🟡 |
| Distance plots | `distance_plot_pop_up` | 🟡 |
| Path plots | `path_plot_pop_up` | 🟡 |
| Heatmaps (classifications) | `heatmap_clf_pop_up` | 🟡 |
| Annotated bouts videos | `annotated_bouts_videos_pop_up` | 🟡 |
| SHAP values | `shap_pop_up` | 🟡 |

Most of these read `csv/machine_results/` and write annotated
video files into `frames/output/<plot_type>/<video>/`.

## Merge frames 🟡
**Tk entry**: Tab "Visualizations" → MERGE FRAMES
Stitch together the various visualization output frames into a
single combined output video (e.g. raw video + classification
overlay + heatmap + path plot side-by-side).

---

# Stage 10: Add-ons

These are SimBA expansions — separate analytical pipelines that
share the project structure but are conceptually distinct.

| Add-on | Status |
|--------|--------|
| Multi-classifier behavior probability ensembles | 🟡 |
| Severity-of-classification scoring | 🟡 |
| FSTTC (Forward / Spontaneous Two-Time Control) | 🟡 |
| Cue light analysis | 🟡 (specialized — uses a separate cue_lights module) |
| Pup retrieval analysis | 🟡 |

---

# Out-of-band utilities (the menu bar)

These don't belong to any pipeline; they're standalone tools.

## Process Videos menu

| Tool | Popup class | Status | Qt? |
|------|-------------|--------|-----|
| Batch pre-process videos | `BatchPreProcessPopUp` | 🟡 | No |
| Blob tracking (init) | `InitializeBlobTrackerPopUp` | 🟡 | No |
| Blob tracking (visualize) | `BlobVisualizerPopUp` | 🟡 | No |
| Train YOLO model | `YOLOPoseTrainPopUP` | 🔴 (likely — depends on ultralytics being installed; gated by `yolo_state` flag) | No |
| Predict with YOLO | `YOLOPoseInferencePopUP` | 🔴 (same) | No |
| Visualize YOLO results | `YoloPoseVisualizerPopUp` | 🔴 (same) | No |

The YOLO popups are conditionally enabled based on whether the
ultralytics package imports cleanly. On systems without it, the
buttons are greyed out (`state=DISABLED`).

## Tools menu — Video processing

This is the largest group: ~60 popups for general video
manipulation. The Qt port consolidates these into 11 forms across
10 sections of the Video Processing page (substantial coverage).

Categories:
- **Change FPS** — single, multiple, upsample-with-interpolation
- **Clip videos** — by time, by frames, into multiple subclips
- **Crop videos** — rectangles, circles, polygons, multi-crop
- **Convert file formats** — image (PNG/JPEG/BMP/TIFF/WEBP),
  video (MP4/AVI/WEBM/MOV)
- **Color processing** — grayscale, B&W, CLAHE (interactive +
  batch), remove specific colors
- **Concatenate** — stack two or N videos
- **Convert ROI definitions** between formats
- **Convert annotations** — COCO ↔ YOLO ↔ DLC ↔ Labelme ↔ SLEAP
  (a big set of cross-format converters)
- **Frame extraction** — extract every N frames, extract by
  range, extract by classification result
- **Image processing** — average frame, time-lapse, register
  multi-camera, etc.

The Qt Video Processing page covers most of the common operations.
The cross-format annotation converters are still Tk-only.

## Tools menu — Annotation conversions

A submenu specifically for converting between pose-estimation /
annotation file formats. Used as a one-off prep step, doesn't
need a project. ~10 popups, none ported to Qt; they live as
the "Tools page" in the Qt port (separate from project state).

---

# What I'd prioritize for porting next

Based on the audit:

1. **Outlier correction** — first non-trivial pipeline gap in the
   Qt port. Currently users have to switch to mufasa-tk between
   pose import and feature extraction. ~1 day to port (small
   form, well-defined backend).
2. **Train machine model (single)** — the goal of the entire
   pipeline. Currently entirely Tk-only. Mid-sized port (form
   has many options for grid search, SMOTE, etc).
3. **Run machine model (batch inference)** — the natural follow-up.
4. **Path plots and heatmaps** — most-used visualization outputs;
   show up in papers; relatively self-contained.
5. **Label behavior** — specialty form, but if you don't already
   have annotations from another tool, you need it. Big port.

Things I'd specifically NOT prioritize:
- Third-party annotation appenders (use upstream tool's CSV
  export instead)
- YOLO pose training popups (use ultralytics CLI directly)
- Older importers (MARS, NicCage, FaceMap) — port on demand if
  someone actually has data in those formats

---

# Areas where this audit is genuinely shallow

I want to be honest about what I haven't done:

- **Not traced**: ~75% of the popup `pop_up.py` files are listed
  but not opened. Their actual reads/writes are unverified.
- **Not tested**: I haven't run any of the legacy popups in a
  fresh environment; "functional" claims are based on code reading
  + the assumption that what users were able to use historically
  still works.
- **No testing at all on Windows or macOS** — the audit assumes
  Linux. Some Tk paths have OS branches that may behave
  differently.
- **No regression testing of Tk popups against any specific
  Mufasa version** — the launcher imports from many places and
  upstream changes (sklearn, matplotlib, opencv) may have broken
  things that nobody notices because nobody runs them.

This document grows over time as we audit more deeply. **Tag any
specific workflow you're considering using and I'll do a deep
trace before you trust it.**
