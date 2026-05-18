# Backend audit

**Generated:** post-patch 122by (May 2026).
**Purpose:** Two-part audit of backend code with two distinct cleanup interactions: (a) the "missing backends" flagged in `qt_form_runtime_gaps.md`, and (b) the backend modules that import from `mufasa.ui.tkinter_functions` and embed Tk-popup launches (blocking the Tier-4 Tk-tree deletion).

**Companion docs:** `qt_form_runtime_gaps.md` (Qt form runtime gaps as observed from the form side), `tk_surface_audit.md` (per-Tk-file inventory), `tk_to_qt_consolidation_plan.md` (overall consolidation plan).

---

## 1. Summary

| Audit topic | Items | Action class |
|---|---:|---|
| Unwired-backend gaps from Qt forms | 7 ops | **5 solvable by name fix / re-wiring; 2 genuinely missing** |
| Backend modules embedding Tk UI | 25 files | Migration order: refactor / die-with-unsupervised / replace with Qt confirmation |

The §1 finding is the more impactful: the 122by audit overstated the gap count by understating availability. Most "missing" backends actually exist under different names. **Five of seven failing operations are recoverable with a rewiring patch, not a backend port.**

---

## 2. §1 Unwired-backend gaps — revised availability

For each operation that raised `NotImplementedError` in the 122by audit, look for the actual backend in `mufasa/` (excluding `ui_qt/`) by exact name, near matches, and substring scan.

### 2a. `AverageFrameForm` — spelling mismatch (REVERIFIES 122by)

Form calls `create_average_frame`. Real function:

```
mufasa/video_processors/video_processing.py:4045
    def create_average_frm(video_path, start_frm=None, end_frm=None,
                            start_time=None, end_time=None,
                            save_path=None, verbose=False):
```

The spelling mismatch confirmed; also the kwargs mismatch (form passes `method`, `frame_stride` — backend takes window-bounds + `save_path`). **Fix scope: medium.** Rewrite form to match the actual signature. ~30 minutes plus design (decide how to surface time-window controls).

### 2b. `VideoFiltersForm (Black & white)` — SOLVABLE BY RENAME

Form calls `convert_to_black_and_white`. Real function:

```
mufasa/video_processors/video_processing.py
    def video_to_bw(...)
```

Same module the form already imports. **Fix scope: tiny.** Change one identifier:

```python
# in target() branch for "black_white":
_vp.video_to_bw(video_path=path, threshold=params["threshold"],
                invert=params["invert"])
```

(Backend signature should be verified; the existing form params `threshold` + `invert` match a typical B&W signature.)

### 2c. `VideoFiltersForm (Box / Gaussian blur)` — GENUINELY MISSING

No function under any of: `convert_to_bw_blur`, `video_box_blur`, `blur_video`, `gaussian_blur_video`. **Fix scope: small-to-medium.** Either:

* Port the backend from the SimBA branch (likely a small OpenCV wrapper using `cv2.boxFilter` or `cv2.GaussianBlur`), or
* Write fresh using OpenCV: ~30 lines for a video-stream wrapper around `cv2.boxFilter` / `cv2.GaussianBlur`.

### 2d. `VideoFiltersForm (Brightness/Contrast)` — GENUINELY MISSING

No function under any of: `brightness_contrast_video`, `video_brightness_contrast`, `adjust_brightness_contrast`, `adjust_video`. **Fix scope: small-to-medium.** Same options as 2c: port from SimBA or write fresh (`cv2.convertScaleAbs` is the standard call).

### 2e. `DropBodypartsForm` — SOLVABLE BY REWIRING TO EXISTING CLASS

Form raises NotImplementedError citing missing `keypoint_dropper` module. The actual implementation:

```
mufasa/pose_processors/remove_keypoints.py
    class KeypointRemover:
```

Already imported by the Tk side at `mufasa/ui/pop_ups/pose_bp_drop_pop_up.py`. **Fix scope: small.** Rewire the form's `target()` to instantiate `KeypointRemover` and call `.run()` (or equivalent).

### 2f. `ROIFeaturesForm (Remove)` — SOLVABLE BY METHOD CALL

Form raises NotImplementedError citing missing `remove_roi_features`. The function exists:

```
mufasa/mixins/config_reader.py
    def remove_roi_features(...):  # method of ConfigReader
```

It's a method on `ConfigReader`, not a free function. The form just needs to instantiate `ConfigReader(config_path)` and call `.remove_roi_features(...)`. **Fix scope: small.**

### 2g. `CropVideosForm (multi-crop)` — SOLVABLE BY REWIRING TO EXISTING CLASS

Form raises NotImplementedError citing missing multi-crop backend. The actual implementation:

```
mufasa/video_processors/multi_cropper.py
    class MultiCropper:
```

Already imported by the Tk side at `mufasa/ui/pop_ups/video_processing_pop_up.py:60`. **Fix scope: small.** Rewire the form's multi-crop branch.

### 2h. Updated `qt_form_runtime_gaps.md` priorities

The audit's priority recommendations stand, but the **work content differs from what was assumed**:

| Op | 122by said | Reality (this audit) |
|---|---|---|
| AverageFrameForm | "medium rewrite" | medium rewrite (confirmed — kwarg shape changes too) |
| VideoFiltersForm (B&W) | "medium port from SimBA" | **tiny — rename one call** |
| VideoFiltersForm (blur) | "medium port from SimBA" | small-to-medium (genuinely missing) |
| VideoFiltersForm (brightness/contrast) | "medium port from SimBA" | small-to-medium (genuinely missing) |
| DropBodypartsForm | "medium-to-large port from SimBA" | **small — rewire to KeypointRemover** |
| ROIFeaturesForm (Remove) | "medium port from SimBA" | **small — call existing ConfigReader method** |
| CropVideosForm (multi-crop) | "small" | small (confirmed — rewire to MultiCropper) |

**Five of seven failing operations are fixable in well under an hour each.** The remaining two (blur + brightness) need ~30 minutes each of new OpenCV code. This is much smaller than the 122by audit estimated.

---

## 3. §2 Backend modules with embedded Tk UI

**Post-patch 122ch:** 23 modules under `mufasa/` (excluding `ui/`, `ui_qt/`, and `SimBA.py`) import `mufasa.ui.tkinter_functions` at module-load time (was 25 pre-122ch). Plus 1 lazy importer — `mufasa.utils.confirm` — which imports inside a function body only when the default Tk-backed `confirm_two_option` actually fires; designed to be replaceable by a Qt override at workbench startup. 23 + 1 = 24 total importers.

The module-level importers are the ones that block Tier-4 cleanup (removing `tkinter_functions.py` would break them at load). The lazy importer doesn't have that property — `confirm.py` would survive `tkinter_functions.py` deletion as long as a Qt override is installed first, or the stdin/auto-yes fallback is acceptable.

### 3a. Inventory by category (post-122ch)

```
unsupervised/  (13 files)   — the entire unsupervised module
labelling/     (2 files)    — frame labelling + standard_labeller
mixins/        (2 files)    — annotator_mixin, pop_up_mixin
                              (train_model_mixin decoupled in 122ch)
roi_tools/     (2 files)    — roi_ruler, roi_ui_mixin
bounding_box_tools/ (1)     — boundary_menus
cue_light_tools/   (1)      — cue_light_main_popup
video_processors/  (1)      — batch_process_menus
                              (video_processing decoupled in 122ch)
utils/         (1, lazy)    — confirm (the abstraction; not blocking)
```

### 3b. Migration disposition

#### Group A: dies with Tier 3b (Unsupervised port) — **13 files**

All 13 `unsupervised/` files import Tk only to render the legacy unsupervised UI (`UnsupervisedGUI` + its sub-popups: dim-reduction, cluster fitting, validation, visualization, comparison, XAI, data extraction). When Tier 3b ships the Qt port of the unsupervised pipeline, this whole subtree gets deleted. No per-file migration work; the entire `unsupervised/pop_ups/` directory plus `unsupervised/unsupervised_main.py` go away in one delete.

#### Group B: tightly-coupled UI + backend, needs refactor — **5 files**

These define classes/functions where the Tk UI is mixed with backend logic — refactoring is required before the Tk import can be removed.

| File | Tk symbols imported | Migration |
|---|---|---|
| `mixins/pop_up_mixin.py` | `DropDownMenu`, `Entry_Box`, `FileSelect`, `SimbaButton`, `hxtScrollbar` | Foundation class for ~85 Tk popups. Delete with Tier 4 once popups are gone. |
| `mixins/annotator_mixin.py` | `Entry_Box` | Used by `mufasa/labelling/`. Replace `Entry_Box` with a Qt or backend-pure equivalent. |
| `mixins/train_model_mixin.py` | ~~`TwoOptionQuestionPopUp`~~ | ~~Single yes/no confirmation. Replace with `QMessageBox.question` or a backend-side `bool` parameter.~~ ✓ **DONE 122ch** — routed through `mufasa.utils.confirm.confirm_two_option`. |
| `labelling/labelling_interface.py` | `CreateLabelFrameWithIcon`, `Entry_Box`, `SimBALabel`, `SimbaButton`, `SimbaCheckbox` | Labelling UI itself — the whole interface needs a Qt port (similar shape to `blob_quick_check`: parameter form + interactive viewer dialog). |
| `labelling/standard_labeller.py` | Same as above | Sibling labeller — should be ported alongside. |

#### Group C: backend-launches-popup, simple refactor — **5 files**

These are backend modules that fire a Tk popup as part of an operation (e.g., a confirmation, an interactive parameter setter). The backend logic itself is fine; just the popup invocation needs to go.

| File | Tk symbols imported | Migration |
|---|---|---|
| `video_processors/video_processing.py` | ~~`TwoOptionQuestionPopUp`~~ | ~~Replace with `QMessageBox.question`. Or split: backend takes a `confirm_callback` parameter; Tk version supplies the popup, Qt version supplies a dialog.~~ ✓ **DONE 122ch** — routed through `mufasa.utils.confirm.confirm_two_option` (lazy Tk import inside the helper; Qt override via module attribute reassignment). |
| `cue_light_tools/cue_light_main_popup.py` | `CreateLabelFrameWithIcon`, `MufasaDropDown`, `SimBALabel`, `SimbaButton` | This is the launcher that's already replaced by the Add-ons Cue-light forms in the Qt workbench. Drop it. |
| `roi_tools/roi_ruler.py` | `SimBALabel` | Read the code — the SimBALabel is probably a status display for the ruler. Refactor: emit a callback signal that callers can route to Qt or Tk. |
| `roi_tools/roi_ui_mixin.py` | `CreateLabelFrameWithIcon`, `DropDownMenu`, `Entry_Box`, `MufasaDropDown`, `SimBALabel`, …5 more | Mixin used by `mufasa/ui/pop_ups/roi_*`. Replaced by the Qt ROI surface (`ROIManageForm` + `ROIDefinePanel`). Drop. |
| `bounding_box_tools/boundary_menus.py` | `CreateLabelFrameWithIcon`, `DropDownMenu`, `Entry_Box` | Boundary box menu Tk surface. Status: not surfaced in the Qt workbench yet (no `BoundingBoxForm` exists). Either port to Qt or accept removal as a Tk-only feature loss. |

#### Group D: build infrastructure, low priority — **2 files**

| File | Tk symbols imported | Migration |
|---|---|---|
| `video_processors/batch_process_menus.py` | `CreateLabelFrameWithIcon`, `Entry_Box`, `MufasaDropDown`, `MufasaSeparator`, `SimBALabel`, …2 more | Batch processing menu surface, Tk-only. Should be ported into the Qt workbench as a "Batch processing" page or absorbed into existing Video Processing forms. Substantial work. |

### 3c. Most-imported Tk symbols

| Symbol | Importers | Replacement strategy |
|---|---:|---|
| `DropDownMenu` | 14 | `QComboBox` |
| `Entry_Box` | 11 | `QLineEdit` / `QSpinBox` |
| `FileSelect` | 11 | `QFileDialog.getOpenFileName` (or a small Qt wrapper widget) |
| `FolderSelect` | 9 | `QFileDialog.getExistingDirectory` |
| `CreateLabelFrameWithIcon` | 6 | `QGroupBox` + label |
| `SimbaButton` | 6 | `QPushButton` |
| `SimBALabel` | 6 | `QLabel` |
| `MufasaDropDown` | 3 | `QComboBox` (or the Qt port at `ui_qt/widgets.py`) |
| `SimbaCheckbox` | 3 | `QCheckBox` |
| `TwoOptionQuestionPopUp` | 2 | `QMessageBox.question` |
| `hxtScrollbar` | 2 | `QScrollArea` |

The 1-to-1 Qt equivalents are well-known. The real friction is the backend ↔ UI coupling in Group B + C — once those refactors are done, the symbol substitution is mechanical.

---

## 4. §3 Combined cleanup plan

Recommended order:

### 4a. Quick wins — REVISED post-patch 122ca

The original audit categorised 4 fixes as "small, ≤1 hour each." A pre-implementation review during patch 122ca discovered that **only 1 of the 4 is truly trivial**; the other 3 need form redesign because the Qt form was written against a different mental model than what the backend supports.

Honest revised estimates:

| Lane | Original audit said | 122ca review found | Status |
|---|---|---|---|
| **`VideoFiltersForm` B&W** → `video_to_bw` | 1-line rename | tiny — threshold scaling + drop `invert` field; ≈15 lines | ✓ shipped 122ca |
| **`ROIFeaturesForm` Remove** → `ConfigReader.remove_roi_features` | ~10 lines | medium — backend's `remove_roi_features(self, data_dir)` requires a `data_dir` parameter the form doesn't currently surface | ✓ shipped 122cd |
| **`CropVideosForm` multi-crop** → `MultiCropper` | ~15 lines | medium — the form's mental model is "ONE video → MULTIPLE outputs"; `MultiCropper`'s real model is "MANY videos in folder → N crops each". Different operation; form's UX needs redesign or backend's behavior needs explaining | ✓ shipped 122cf (chose to loop ROISelector + crop_video directly; bypasses MultiCropper) |
| **`DropBodypartsForm`** → `KeypointRemover` | ~10 lines | medium — constructor signature mismatch. Form expects `KeyPointRemover(config_path, body_parts, copy_originals)`. Real class is `KeypointRemover(data_folder, pose_tool, file_format)` + `.run(animal_names, bp_to_remove_list)`. Form needs config-to-data-folder resolution + selection transformation | ✓ shipped 122ce |

**Net 122ca outcome:** 1 of 7 runtime gaps closed (B&W). The other 3 from this list need follow-up patches with form-side changes, not just dispatch rewires.

The audit underestimated complexity because it stopped at "is the backend named in the codebase?" without verifying constructor signatures and semantic models match the form's UX assumptions.

### 4b. Follow-up patches needed (medium, 1-3 hours each)

5. ~~**Redesign `ROIFeaturesForm` Remove action**~~ ✓ **DONE in patch 122cd.** Added `data_dir` QLineEdit + Browse button. Field enabled only for the Remove action; auto-populates `<project>/csv/features_extracted` as default when switching to Remove if that path exists. Dispatch rewired to instantiate `ConfigReader(read_video_info=False, create_logger=False)` and call `.remove_roi_features(data_dir=data_dir)`.
6. ~~**Redesign `CropVideosForm` multi-crop semantics**~~ ✓ **DONE in patch 122cf.** Picked the "match form's UX with a loop of single-crop calls" path: the multi-crop branch now loops `ROISelector` + `crop_video` directly, bypassing `MultiCropper` entirely. Added a `crop_count` QSpinBox (2–20, default 2). Output files are suffixed `_crop1`, `_crop2`, …, `_cropN`; collision-detection appends a timestamp if files already exist. Multi-crop is gated to rectangle-shape + single-file scope via reactive `_refresh_multi_state`.
7. ~~**Redesign `DropBodypartsForm`**~~ ✓ **DONE in patch 122ce.** Took the "add resolution + transform" path (Option A): the form is project-aware so we keep that mental model. Added a `data_folder` QLineEdit (auto-defaults to `<project>/csv/input_csv`); added a status label showing inferred `pose_tool` (DLC for 1 animal, maDLC for >1) and `file_format` from project metadata; dropped the misleading `copy_originals` checkbox (the backend always writes to a new `Reorganized_bp_<datetime>` subdirectory). Dispatch transforms the form's `[(animal, bp), ...]` selection into the backend's split `animal_names` + `bp_to_remove_list` lists.
8. ~~**Rewrite `AverageFrameForm`**~~ ✓ **DONE in patch 122cc.** Form rewritten to match `create_average_frm` signature: dropped `method` (Mean/Median) and `stride` fields (backend doesn't support); added a window-mode selector (Whole video / Frame range / Time range) via QStackedWidget; added an optional save-path field that defaults to a timestamped name alongside source.

### 4c. Medium fixes — genuinely missing backends (port from SimBA or write fresh)

9. ~~**Port box-blur backend**~~ ✓ **DONE in patch 122cb.** New `video_blur(video_path, kernel_size, method, save_dir, gpu)` in `mufasa/video_processors/video_processing.py`. Uses FFmpeg's `gblur` / `boxblur` filter. Wired into `VideoFiltersForm.target()` blur branch.
10. ~~**Port brightness/contrast backend**~~ ✓ **DONE in patch 122cb.** New `video_brightness_contrast(video_path, brightness, contrast, save_dir, gpu)` in same module. Uses FFmpeg's `eq` filter; form's brightness (−1..+1) and contrast (0..3) ranges map directly.

After 4b + 4c: ~~all 7~~ **4 of the 7 form runtime gaps closed** (B&W, blur, brightness/contrast, plus the 3 deferred from §4b once those follow-ups land).

### 4d. Backend-Tk-coupling decoupling (per-file)

Order suggested:

11. ~~**`video_processors/video_processing.py`** — replace `TwoOptionQuestionPopUp` import.~~ ✓ **DONE in patch 122ch.** Both the module-level Tk import and the `TwoOptionQuestionPopUp(...)` call site (in `extract_frames_from_all_videos_in_directory`) replaced with a `from mufasa.utils.confirm import confirm_two_option` import + a `confirm_two_option(...)` call. The new helper lazy-imports Tk only if no Qt override is installed; backend file is now Tk-import-free at module load.
12. ~~**`mixins/train_model_mixin.py`** — same `TwoOptionQuestionPopUp` replacement.~~ ✓ **DONE in patch 122ch.** Same pattern: Tk import dropped; `TrainModelMixin.read_meta_dicts_from_dir` (META CONFIG FILE ERROR confirmation) now routes through `confirm_two_option`.
13. **`roi_tools/roi_ruler.py`** — refactor `SimBALabel` use into a callback.
14. **`mixins/annotator_mixin.py`** — refactor `Entry_Box` use.

**New helper:** `mufasa/utils/confirm.py` — provides `confirm_two_option(question, option_one, option_two, title)`. Default implementation lazy-imports the Tk popup; falls back to stdin prompt if Tk unavailable; falls back to `option_one` if stdin unavailable. Qt code overrides by reassigning the module-level binding at workbench startup (see the module docstring for the pattern).

**Qt-side override installed in patch 122cj.** `mufasa.ui_qt.qt_confirm.install_qt_confirm_override()` is called from `workbench_app.main()` immediately after the `QApplication` is constructed. After install, every backend call to `confirm_two_option(...)` routes through `QMessageBox.question`-style dialog with the caller's option labels preserved (e.g., "SKIP" / "TERMINATE" for the training-meta-config error path, not Qt's standard "Yes" / "No" constants).

### 4e. Bulk-drop (after Tier 3b)

12. **Drop `mufasa/unsupervised/pop_ups/`** + unsupervised_main.py (13 files) — handled by Tier 3b.
13. ~~**Drop `cue_light_tools/cue_light_main_popup.py`**~~ ✓ **DONE in patch 122ck.** Deleted 6 files in one batch: `mufasa/cue_light_tools/cue_light_main_popup.py`, `mufasa/ui/pop_ups/cue_light_main_popup.py`, and 4 sub-popups (`cue_light_clf_analyzer_popup.py`, `cue_light_data_analyzer_popup.py`, `cue_light_movement_analyzer_popup.py`, `cue_light_visualizer_popup.py`). All 4 sub-popups were orphans after the main popup went — only consumer was the main popup itself. SimBA.py's import + button creation + grid call surgically removed; remaining reference is the breadcrumb comment at SimBA.py's import block.
14. ~~**Drop `roi_tools/roi_ui_mixin.py`**~~ — **NOT safe to drop yet** (re-audited 122ck). `roi_tools/roi_ui.py:ROI_ui` subclasses `ROI_mixin` from this file, and `roi_ui.py` is transitively consumed by the Qt ROI surface at `mufasa/ui_qt/dialogs/roi_canvas.py` + `roi_define_panel.py`. Either port `ROI_ui` to be self-contained or accept that this file stays alongside the Qt ROI dialogs. Defer until the Qt ROI dialogs themselves are reviewed.
15. **Drop `mixins/pop_up_mixin.py`** — once no Tk popups remain.

### 4f. Last items (Tier 4 close-out)

16. **`labelling/labelling_interface.py` + `standard_labeller.py`** — Qt port of the labelling UI. Substantial.
17. **`video_processors/batch_process_menus.py`** — Qt port or accept removal.
18. **`bounding_box_tools/boundary_menus.py`** — Qt port or accept removal.

After 4a-4f: `mufasa/ui/tkinter_functions.py` is unreachable; can be deleted, completing the Tier 4 cleanup.

---

## 5. §4 Audit methodology

Two AST-based passes; both reproducible.

**Pass 1 (unwired backend search):** for each form's failing operation, search the codebase (excluding `ui_qt/`) for the literal name plus near-name variants. Top-level `def` and `class` matches are "exact"; substring matches in any line are "near".

**Pass 2 (Tk-importer scan):** every `.py` under `mufasa/` excluding `ui/`, `ui_qt/`, and `SimBA.py` is parsed; any `from mufasa.ui.tkinter_functions import …` is collected with its imported symbols. Aggregation by directory gives the disposition categories.

Both scripts are inlined in patch 122bz's commit message (and reproducible by anyone with the codebase open).

---

## 6. Caveats

* **The "near match" search uses 4–5 candidate names per failing op.** A backend named in a completely unexpected way (e.g., `bw_video_creator` for `video_to_bw`) might be missed. The fact that 5 of 7 *did* turn up under reasonable guesses suggests the search space wasn't too narrow, but the genuinely-missing 2 (blur, brightness) might exist somewhere I didn't think to look.
* **Method-on-class vs free function ambiguity.** `remove_roi_features` is a method of `ConfigReader`. The AST scan picks it up as a `FunctionDef` named `remove_roi_features` regardless. Callers need to instantiate the class first — not a drop-in replacement for a free function.
* **Group C "drop" recommendations** (in §3b) assume the Qt surface is feature-complete for the corresponding Tk code path. Each one needs a runtime check on the Qt workbench before deletion — same caveat as `tk_surface_audit.md`.
* **`video_processors/video_processing.py` imports `TwoOptionQuestionPopUp` from the Tk layer.** That's a backend file with a heavy Tk dependency. Decoupling is high priority because this file is used by 25+ other backend modules; the Tk import propagates virally.
* **The 25 importer count is post-122br.** Future refactors that introduce or remove Tk imports will shift the number; the §2 numbers in `tk_surface_audit.md` may need refreshing periodically.
* **No backend functional verification.** This audit confirms names exist; it doesn't confirm the existing functions are correct, complete, or even runnable. The "tiny fix — just rename the call" claim in §2b assumes `video_to_bw` actually produces a B&W video; a real test would confirm.
