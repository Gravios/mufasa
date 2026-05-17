# Qt form — runtime gaps audit

**Generated:** post-patch 122bx (May 2026).
**Purpose:** Inventory the Qt workbench forms whose UI is wired but whose backend code raises `NotImplementedError` (or similar) at runtime. These are "looks complete, fails on Run" cases — the user can fill in the form and click Run, but the operation never completes.

This audit is the output of patch 122by, Path C of the post-122bw work. No fixes are made in this patch; the audit identifies the gaps so they can be prioritised by the team.

---

## 1. Summary

**Status after patches 122ca / 122cb / 122cc / 122cd:** of 7 originally-counted runtime gaps, 5 have been closed (B&W, blur, brightness/contrast, AverageFrameForm, ROIFeatures Remove). 2 form-redesign cases remain (DropBodyparts constructor mismatch, CropVideos multi-crop semantics) plus the CLAHE interactive-preview Qt-dialog port.

| Form | Failing operations | Failure mode | Status |
|---|---|---|---|
| `VideoFiltersForm` (video_filters.py) | ~~Black & white~~ | ~~Backend functions not present in this fork~~ | ✓ 122ca |
| `VideoFiltersForm` | ~~Box blur~~, ~~Brightness/contrast~~ | ~~Backend functions not present~~ | ✓ 122cb (new FFmpeg backends) |
| `VideoFiltersForm` | CLAHE with "Interactive preview" checked | Dialog not yet wired | pending |
| `CropVideosForm` (video_editing.py) | Multi-crop from single video | Semantics mismatch with `MultiCropper` (folder vs single) | pending (form redesign) |
| `AverageFrameForm` (image_conversion.py) | ~~All — every Run~~ | ~~Calls `create_average_frame`; backend is `create_average_frm`; kwargs mismatch~~ | ✓ 122cc (form rewritten) |
| `DropBodypartsForm` (pose_cleanup.py) | All — every Run | Constructor signature mismatch with `KeypointRemover` | pending (form redesign) |
| `ROIFeaturesForm` (roi.py) | ~~Remove-ROI-features action~~ | ~~Backend needs `data_dir` field not surfaced~~ | ✓ 122cd (field added) |

Plus `VisualizationForm` raises `RuntimeError` (not `NotImplementedError`) when the project context is unavailable — that's defensive guarding, not a gap. Excluded from this audit.

---

## 2. Per-form detail

### 2a. `AverageFrameForm` (image_conversion.py) — ✓ FIXED in patch 122cc

~~**Severity: high.** Every Run fails. The form is currently unusable.~~

~~**Root cause:**~~ ~~The form's `target()` looks for `mufasa.video_processors.video_processing.create_average_frame` (with an `e`). The actual function is named `create_average_frm` (no `e`).~~

**Resolved in 122cc:** Form rewritten to match the actual backend signature. Drops the unsupported `method` (Mean/Median) and `stride` fields — the backend is mean-only over all frames in the requested window. Surfaces the real backend parameters:

* **Window mode selector** — "Whole video" / "Frame range" / "Time range (HH:MM:SS)" via a QStackedWidget. Backend rejects mixing frame and time ranges; the mode selector makes that impossible to violate.
* **Save path** — optional file picker. If empty, defaults to `<source>_avgframe_<timestamp>.png` alongside the source.

The dispatch now calls `create_average_frm(video_path, start_frm, end_frm, start_time, end_time, save_path, verbose=False)` with kwargs assembled from the selected window mode.

### 2b. `VideoFiltersForm` (video_filters.py) — 1 OPERATION FAILS (was 4)

| Op | What fails | Reason |
|---|---|---|
| ~~Black & white (binarise)~~ | ~~Always~~ | ✓ **FIXED in patch 122ca** — rewired to existing `video_to_bw` backend (with threshold range scaling 0–255 → 0.0–1.0; `invert` checkbox dropped since backend doesn't support it) |
| ~~Box / Gaussian blur~~ | ~~Always~~ | ✓ **FIXED in patch 122cb** — new `video_blur` backend added to `video_processors/video_processing.py` (FFmpeg's `gblur` filter; method=gaussian default, `box` available) |
| ~~Brightness / contrast~~ | ~~Always~~ | ✓ **FIXED in patch 122cb** — new `video_brightness_contrast` backend added (FFmpeg's `eq` filter; ranges map directly) |
| CLAHE | Only with "Interactive preview" checked | Dialog not wired; non-interactive CLAHE works |

**Severity (post-122cb): low — only the CLAHE interactive preview remains**. The main CLAHE path works; B&W is fixed; blur and brightness/contrast have working FFmpeg-backed implementations.

The form has 5 operations in its dropdown; 1 of those 5 currently has a partial failure (CLAHE interactive preview only).

**Fix scope (remaining):**
* CLAHE interactive preview: medium — needs a Qt dialog like blob quick-check (live frame display + tunable parameters).

**Stop-gap (no backend work):** disable the two remaining failing options in the dropdown until they're wired. Same posture as before; the broken-options scope just shrank by one.

### 2c. `CropVideosForm` (video_editing.py) — 1 OPERATION FAILS

`Multi-crop from a single video` is the failing sub-mode. Other sub-modes (rect / circle / polygon, single + directory) work.

**Fix scope:** small. The Tk source `MultiCropPopUp` exists with the backend already wired; just needs the corresponding Qt path completed.

**Docstring inconsistency:** the `CropVideosForm` docstring doesn't reference its replaced Tk popups by `:class:` name like the other forms do. Should list `CropVideoPopUp`, `CropVideoCirclesPopUp`, `CropVideoPolygonsPopUp`, `MultiCropPopUp` so future readers know which Tk surface this absorbs.

### 2d. `DropBodypartsForm` (pose_cleanup.py) — ALL OPERATIONS FAIL

**Severity: high.** Backend module `keypoint_dropper` is missing from this fork. Form is currently unusable; every Run raises `NotImplementedError`.

**Fix scope:** medium-to-large. Port `keypoint_dropper.py` from the legacy SimBA branch (per the docstring note in the form). May need adjustments for v1 project layout.

**Stop-gap:** disable the form (don't register it in the page). The current state is worse than no form — users see a UI promising functionality that doesn't exist.

### 2e. `ROIFeaturesForm` (roi.py) — ✓ FIXED in patch 122cd

~~The "Remove ROI features" action raises `NotImplementedError`; "Append by animal" and "Append by body-part" work.~~

**Resolved in 122cd:** Form's Remove action rewired to call `ConfigReader.remove_roi_features(data_dir)`. The 122bz audit found the backend exists as a method on `ConfigReader` (not a free function); this patch surfaces a new `data_dir` field and dispatches correctly.

* New `data_dir_edit` QLineEdit + Browse button. Field is enabled only when the Remove action is selected.
* Auto-populates the default `<project>/csv/features_extracted` when switching to Remove if that directory exists.
* `collect_args()` raises ValueError if Remove is selected but no data_dir picked, or if the path doesn't exist.
* Dispatch instantiates `ConfigReader(config_path=config_path, read_video_info=False, create_logger=False)` to avoid the heavy startup work, then calls `.remove_roi_features(data_dir=data_dir)`.

---

## 3. Recommendations

In priority order:

1. **Fix `AverageFrameForm`** — it's currently entirely broken. Quick win for users who land on the "Metadata & audit" section expecting it to work. Medium-sized form rewrite (a few hours).

2. **Disable broken options in `VideoFiltersForm`** — a 2-line fix: remove three entries from the `OPS` list. Users stop seeing dropdown items that fail. Re-add when backends are wired.

3. **Disable `DropBodypartsForm`** — remove from `pose_cleanup_page.py` registration. Re-add when `keypoint_dropper` lands.

4. **Disable Remove action in `ROIFeaturesForm`** — small UX fix.

5. **Fix `CropVideosForm` docstring** — list its replaced Tk popups for consistency with the other 122b-series forms.

6. **Port the missing backends** — medium-to-large per item. `keypoint_dropper` is probably the biggest, followed by the three VideoFiltersForm ops, then `remove_roi_features`, then `CropVideosForm`'s multi-crop sub-mode.

The "disable broken options" stop-gap is recommended over leaving the UI as-is — broken UI is worse than missing UI because it implies functionality that doesn't exist.

---

## 4. Audit methodology

Reproducible. The audit script scans every Qt form file for `target()` methods raising `NotImplementedError`:

```python
import ast
from pathlib import Path

for f in Path("mufasa/ui_qt/forms").glob("*.py"):
    src = f.read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for stmt in node.body:
                if (isinstance(stmt, ast.FunctionDef)
                        and stmt.name == "target"):
                    body_src = ast.unparse(stmt)
                    if "NotImplementedError" in body_src:
                        print(f"{f.name}:{node.name}: {body_src[:200]}")
```

Limitations:
* Doesn't catch backends that fail at runtime with a different exception type (e.g., `KeyError`, `ValueError`).
* Doesn't detect partial failures (a backend that runs but produces wrong output).
* Doesn't detect forms whose `on_run()` override skips `target()` entirely (those need separate inspection).

A runtime smoke pass against a real workbench is the only way to find the last 10–20% of issues this audit misses.

---

## 5. Caveats

* **No fixes applied in this patch.** This is documentation only. The list of "currently failing" operations is honest about the state of the Qt port; future patches choose which to fix and in what order.
* **The "stop-gap disable" recommendation is a UX judgment, not a hard rule.** A maintainer may prefer to leave the failing options visible as a documented TODO. Either way, fixing the actual root cause is better than disabling.
* **The fork-divergence framing matters.** Several `NotImplementedError` raises mention "this fork" — the Mufasa codebase forked from SimBA and some backends weren't pulled across. Porting them isn't necessarily "writing new code"; it may be a copy-and-adapt from the upstream branch.
* **The audit was performed at the time of patch 122by**. Future patches that wire backends should update or remove the corresponding entry here.
