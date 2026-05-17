# Qt form — runtime gaps audit

**Generated:** post-patch 122bx (May 2026).
**Purpose:** Inventory the Qt workbench forms whose UI is wired but whose backend code raises `NotImplementedError` (or similar) at runtime. These are "looks complete, fails on Run" cases — the user can fill in the form and click Run, but the operation never completes.

This audit is the output of patch 122by, Path C of the post-122bw work. No fixes are made in this patch; the audit identifies the gaps so they can be prioritised by the team.

---

## 1. Summary

Seven Qt form/operation pairs raise `NotImplementedError` or equivalent at runtime when the user clicks Run. All seven are spread across four forms.

| Form | Failing operations | Failure mode |
|---|---|---|
| `VideoFiltersForm` (video_filters.py) | Black & white, Box blur, Brightness/contrast | Backend functions not present in this fork |
| `VideoFiltersForm` | CLAHE with "Interactive preview" checked | Dialog not yet wired |
| `CropVideosForm` (video_editing.py) | Multi-crop from single video | Backend wiring pending |
| `AverageFrameForm` (image_conversion.py) | All — every Run | Calls `create_average_frame`; backend is named `create_average_frm` (missing `e`); also kwargs mismatch |
| `DropBodypartsForm` (pose_cleanup.py) | All — every Run | `keypoint_dropper` backend not in this fork |
| `ROIFeaturesForm` (roi.py) | Remove-ROI-features action | `remove_roi_features` not in this fork |

Plus `VisualizationForm` raises `RuntimeError` (not `NotImplementedError`) when the project context is unavailable — that's defensive guarding, not a gap. Excluded from this audit.

---

## 2. Per-form detail

### 2a. `AverageFrameForm` (image_conversion.py) — ALL OPERATIONS FAIL

**Severity: high.** Every Run fails. The form is currently unusable.

**Root cause:** The form's `target()` looks for `mufasa.video_processors.video_processing.create_average_frame` (with an `e`). The actual function is named `create_average_frm` (no `e`). The form's defensive `getattr` returns `None`, triggers the `AttributeError` branch, and raises `NotImplementedError`.

**Secondary issue:** Even if the spelling were corrected, the kwargs are incompatible. The form passes `video_path=path, method=method, frame_stride=stride`, but the backend signature is:

```python
def create_average_frm(video_path, start_frm=None, end_frm=None,
                       start_time=None, end_time=None,
                       save_path=None, verbose=False):
```

No `method`, no `frame_stride`. The form's UI surfaces parameters the backend doesn't accept.

**Fix scope:** medium. The form needs a redesign:
* Drop `method` and `frame_stride` from the UI.
* Add window controls (start/end frame OR start/end time).
* Add `save_path` (optional file picker).
* Add `verbose` checkbox.
* Update the dispatch to call `create_average_frm`.

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

### 2e. `ROIFeaturesForm` (roi.py) — REMOVE ACTION FAILS

The "Remove ROI features" action raises `NotImplementedError`; "Append by animal" and "Append by body-part" work.

**Severity: low.** The Add operations are the common use cases; Remove is rare.

**Fix scope:** medium. Port `remove_roi_features` backend from legacy branch.

**Stop-gap:** remove the "Remove" option from the action dropdown until wired.

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
