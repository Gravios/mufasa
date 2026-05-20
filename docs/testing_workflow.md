# Mufasa testing workflow — parquet pose + cage videos

End-to-end test plan for a Mufasa user with:

* **Pose data:** parquet files with marker locations.
* **Videos:** mouse-in-cage recordings.
* **Two conditions / arena layouts** — different physical setups requiring different ROI shapes.

This workflow exercises the v1 project layout, the pose-import paths, the ROI tool's Apply-all behavior, and the downstream pipeline. It also identifies the spots that **need specific manual checks** because patches 122da → 122dj have changed behaviour vs the legacy SimBA implementation.

---

## Step 0 — identify your parquet format (5 min)

Two different parquet formats land at Mufasa. Knowing which one you have determines the import path.

### Option A — Mufasa / SimBA internal format

Multi-index column headers, three rows above data:

```
scorer       my_model  my_model  my_model  my_model  ...
bodypart     nose      nose      nose      left_ear  ...
coord        x         y         likelihood  x       ...
0            123.4     456.7     0.99      ...
1            123.5     456.6     0.99      ...
```

**Test:** `python3 -c "import pandas as pd; df = pd.read_parquet('your_file.parquet'); print(df.columns[:8]); print(df.head(3))"`

If columns are tuples or you see `(scorer, bodypart, coord)`-style names → Option A. **You can skip the importer entirely.** Drop the parquet files directly into `<project_root>/sources/pose/` (v1) or `<project>/csv/input_csv/` (legacy). The downstream pipeline reads this format natively.

### Option B — DLC 3.0+ parquet export

DLC 3.0 added direct parquet output (same column structure as Option A but written by DLC's exporter). Treat exactly like Option A — drop into the pose directory.

### Option C — Custom / unknown shape

If the columns look different (e.g., flat names like `nose_x`, `nose_y`), you need to convert. Use Mufasa's `tools/csv_to_parquet.py` to inspect, or rename columns to the multi-index shape and rewrite.

---

## Step 1 — create a v1 project (10 min)

Two paths:

### Workbench
```bash
mufasa-workbench
# File → New Project → pick a parent directory + project name
# → Form writes <name>/project.toml + the directory tree
```

### Programmatic (for scripted setups)
```python
from pathlib import Path
from mufasa.project_layout import write_project_toml

root = Path("/path/to/test_project")
for sub in ["sources/videos", "sources/pose", "sources/annotations",
            "derived", "models", "logs/measures"]:
    (root / sub).mkdir(parents=True, exist_ok=True)

write_project_toml(root / "project.toml", {
    "project": {
        "name": "cage_test",
        "created": "2026-05-19T00:00:00Z",
        "layout_version": "v1",
    },
    "pose": {
        "file_type": "parquet",          # match your data
        "animal_count": 1,                # single mouse
        "body_parts": ["nose", "left_ear", "right_ear", "tail_base"],
        # ↑ replace with your actual body part list, matching the
        # parquet column order
    },
    "settings": {},
})
```

**Verify:** project root has `project.toml` + the expected `sources/derived/models/logs/` subdirectories.

---

## Step 2 — drop in pose data + videos (5 min)

For Option A/B (parquet already in the right shape):

```bash
# Skip the importer; place files directly.
cp /path/to/pose/*.parquet  /path/to/test_project/sources/pose/
cp /path/to/videos/*.mp4    /path/to/test_project/sources/videos/
```

**Verify:** `ls test_project/sources/pose/` and `ls test_project/sources/videos/` show matching filenames (the pose `.parquet` for `video1.mp4` should be `video1.parquet`).

For Option C (need conversion), open the workbench → Data Import → Import Pose Data → pick the closest matching route → Browse to source directory → Run.

---

## Step 3 — set video parameters (5 min)

Workbench → Data Import → Video Parameters & Calibration:

* For each video: set pixels-per-mm (click "Calibrate from frame" → opens Qt dialog → click two points spanning a known distance → enter the distance → OK).
* Set resolution + fps (usually auto-detected from the video).

**Specific test from patch 122de:** the calibration dialog is the **Qt-native PixelCalibrationDialog** now (no separate cv2 OS window). Confirm:
- A modal dialog opens INSIDE the workbench (not a standalone OS window).
- Click two points → line appears between them → enter known distance → OK populates the ppm field.

**Verify:** `<root>/sources/video_info.csv` exists and has one row per video.

---

## Step 4 — ROI tool: targeted tests (15–30 min)

This is the section to spend real attention on. **Your two-conditions/two-arenas scenario directly hits the gap between what works today and what would be most useful.**

### Step 4.1 — draw an ROI on a "condition A" video

Workbench → ROI → ROI Definitions:

* Pick one video from condition A (e.g., `cond_A_mouse1.mp4`).
* Select rectangle / circle / polygon → click "Draw" → click + drag on the frame.
* The shape commits when you release (rectangle) / click again (circle) / close the polygon by clicking the first vertex.
* Click Save.

**Verify:** the ROI persists across re-opening the dialog (it's stored in `<root>/logs/measures/ROI_definitions.h5`).

### Step 4.2 — "Apply to all" — verify the 122d9/122da fix

This is the bug you described — pre-122d9 this was broken for v1 projects.

* With your `cond_A` ROI drawn → click **"Apply to all"**.
* Workbench should now show the same ROI on every video in the project.

**Watch for:** any error pointing at `<v1_root>/csv/input_csv/...` or `os.path.join(project_path, "logs", ...)` — that would mean the fix didn't fully land. Both `multiply_ROIs:462` (videos_dir) and `multiply_ROIs:474` (roi_coordinates_path) plus `reset_video_ROIs:561` were fixed across 122d9 + 122da. The error you saw before was the latter two; if it persists, file a bug.

### Step 4.3 — handle your two conditions: workaround for now

Apply-to-all is currently **all-or-nothing**. With two conditions/arena layouts, you have a few workarounds:

**Workaround 1 — apply, then reset per-video:**
1. Draw `cond_A` ROI on a condition-A video.
2. "Apply to all" → propagates to every video including condition-B.
3. Open each condition-B video → Reset → redraw `cond_B` ROI.

Tedious but works.

**Workaround 2 — apply per-condition group, using a temporary project split:**

Create two sub-folders or use the `cli/migrate_project.py` `--v1-root` flag (or a manual copy) to make two parallel projects, one per condition. Apply ROIs in each. Then merge if needed. Heavier but cleaner.

**Workaround 3 — script via the backend:**
```python
from mufasa.roi_tools.roi_utils import multiply_ROIs, reset_video_ROIs

# Apply cond_A ROI to all videos
multiply_ROIs(config_path=...)

# Reset cond_B videos
for vid in cond_B_videos:
    reset_video_ROIs(config_path=..., filename=vid)
# Then manually open the workbench and redraw on one cond_B
# video → apply-to-all again (this clobbers cond_A!)
# So in practice you can only do this once per "majority condition."
```

**Proper fix (not yet implemented):** a patch that adds "Apply to selected videos…" — see `docs/roi_enhancements_proposal.md` for the proposed design.

### Step 4.4 — drag-to-adjust placed ROIs

**Current state: not implemented.** Once a shape is committed, clicking on it in the canvas does nothing. To adjust an existing ROI you must Reset → redraw from scratch. This is a real UX gap that came up directly from your testing.

**Workaround:** reset the affected video and redraw. Each redraw can be quick if you know the rough size — Mufasa's ROI canvas keeps the last-used draw mode active, so a click + drag commits a new rectangle immediately.

**Proper fix (not yet implemented):** see `docs/roi_enhancements_proposal.md` for the proposed select-and-drag patch.

---

## Step 5 — pose cleanup (10 min)

Workbench → Preprocessing → Outlier Correction:

* Pick a body part for the movement criterion (typically `nose` or `tail_base`).
* Set the criterion threshold (default 1.5 — body lengths per frame).
* Run.

**Verify:** `<root>/derived/outlier_corrected/<run_id>/` populated; the workbench prints the run_id (a timestamp-prefixed string like `20260519-152431-a3f1b9`).

Optional: smooth with Savitzky-Golay (under Preprocessing → Smoothing). Run → `<root>/derived/outlier_corrected/<run_id>/` updated or new run created depending on settings.

---

## Step 6 — feature extraction (5–10 min, depends on video count)

Workbench → Features → Extract Features:

* Pick the latest outlier-correction run (auto-selected).
* Run.

**Verify:** `<root>/derived/features/<run_id>/` has one parquet per video. The parquet should have hundreds of feature columns (distances, angles, speeds, bouts of motion, etc.) per frame.

---

## Step 7 — visualizations (5 min) — verify 122dc fix

Workbench → Visualizations → pick a route (e.g., "Path Plot"):

This is where patch 122dc lives. Pre-122dc, every visualization route on a v1 project would fail with "data source directory not found" pointing at `<v1_root>/csv/<stage>/`. Post-122dc it resolves to `<v1_root>/derived/<stage>/<latest_run>/`.

**Run a Path Plot:** the route uses `data_paths_source="outlier_corrected_movement_location"`.

**Verify:** plot renders without error. If it raises pointing at `<v1_root>/csv/...`, the 122dc fix needs another look.

---

## Step 8 — classifier (optional; ~30 min if you have annotations)

If your test data includes hand-labels:

* Workbench → Annotation → Append Targets.
* Workbench → Classifier → Train.

If not, skip — you've already exercised the bulk of the pipeline.

---

## Quick sanity-check checklist

Run through this at the end. Each ✓ exercises a recently-fixed code path.

| Check | Tests |
|---|---|
| `project.toml` exists at root, valid TOML | Step 1 project creation |
| Pose files in `<root>/sources/pose/` (NOT `<root>/csv/input_csv/`) | v1 layout detection + automatic routing |
| `<root>/sources/video_info.csv` has correct rows | Step 3 video parameters |
| ROI HDF at `<root>/logs/measures/ROI_definitions.h5` | ROI persistence (path is same in both layouts) |
| Apply-to-all does NOT error pointing at a `csv/` directory | 122d9 + 122da fixes |
| Outlier correction output at `<root>/derived/outlier_corrected/<run_id>/` | v1 run-id allocation |
| Features at `<root>/derived/features/<run_id>/` | Feature stage v1 layout |
| Path plot renders without "csv/ not found" error | 122dc visualizations fix |
| Pixel calibration is an embedded Qt dialog (not standalone cv2 window) | 122de port |
| `mufasa.ui` directory doesn't exist on disk | 122de cleanup |

If anything in this checklist fails, file a bug with the exact error + the failing step number.

---

## Bugs to specifically watch for

These are the highest-risk paths from the recent patch series:

1. **MARS + TRK route runtime failures** (patches 122di, 122dj). I passed `interpolation_method="None"` as a sentinel; if the backend's `Interpolate.fix_missing_values()` actually rejects "None" instead of skipping, the import fails. The fix is to change the sentinel to `"Linear"` in `mufasa/ui_qt/forms/pose_import.py` — won't matter for the DLC / SLEAP / YOLO / FaceMap routes.

2. **Empty `<v1_root>/csv/` directory appearing** (patch 122dc / earlier). If any visualization or analysis form creates this directory inadvertently, it means a hardwired-paths bug slipped through. Should never appear in a v1 project.

3. **`<root>/sources/pose/` not populating** — if pose import "succeeds" but the directory is empty, the backend may be writing to the legacy `csv/input_csv/` path. Check `<root>/csv/input_csv/` and report which backend was used.

4. **PixelCalibrationDialog crashing on Run** (patch 122de). The `on_run` override is supposed to keep it on the GUI thread. If you see a Qt error about "QDialog called from non-GUI thread", the override didn't work.

---

## Reporting issues

For each issue:

* Step number that failed.
* Exact error message (copy from the workbench's status panel or terminal).
* Workbench log location: `<root>/logs/workbench.log` if it exists, otherwise the terminal stdout/stderr from launching `mufasa-workbench`.
* Mufasa version / commit hash: `cd mufasa && git log --oneline -1`.
