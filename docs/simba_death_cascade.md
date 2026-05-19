# SimBA.py death cascade — scope

**Status:** scoped post-patch 122cw; pending execution (122cy onwards).
**Authority:** evidence-based reachability audit. All findings reproducible via the AST + grep checks in `tests/smoke_122cx_simba_death_cascade_scope.py`.

---

## Summary

`mufasa.SimBA` is the legacy Tk entry point (`mufasa-tk` in `[project.scripts]`). The Qt entry points (`mufasa`, `mufasa-chooser`, `mufasa-workbench`) do not reach into SimBA.py at any point. Removing the `mufasa-tk` entry point + deleting SimBA.py triggers a clean cascade-delete of **101 files** (plus 1 config edit) across the Tk surface, leaving the Qt workbench fully functional with two exceptions documented below.

| Stage | Files | Notes |
|---|---:|---|
| Stage A (entry point) | 0 + 1 config edit | ✓ **EXECUTED 122d4** — `mufasa-tk` removed from `pyproject.toml`. |
| Stage B (cascade) | 114 | ✓ **EXECUTED 122d5** — bulk deletion of SimBA.py + all Tk-only files. |
| Stage C (tail) | 3 | ✓ **EXECUTED 122d6** — `tkinter_functions.py` + `pop_up_mixin.py` (planned) + `unsupervised_mixin.py` (newly exposed by Stage B; folded in). Was scoped as 2 in 122cx; +1 for `unsupervised_mixin.py` which lives in `mufasa/mixins/` and so wasn't swept by Stage B's `git rm -r mufasa/unsupervised/`. |
| **Total** | **118 files** | (was 117 at 122cx scoping; +1 for the unsupervised_mixin discovery in 122d6) |

After the cascade, **two Tk dependencies remain** in the Qt-using path:

1. **`mufasa/ui/px_to_mm_ui.py`** — used by `ui_qt/forms/video_utilities.py:318` to launch a **cv2-based** (NOT Tk-based, as the 122cx audit initially claimed) pixel-calibration UI. Uses `cv2.namedWindow` + `cv2.imshow` + `cv2.setMouseCallback` — pure OpenCV. Survives the cascade because the cascade scope was "Tk surface", and this file isn't Tk. The "Tk dependency in Qt path" framing in earlier patch notes was incorrect; the actual disposition is "cv2-window dependency, opens a separate OS window when the Qt video-calibration form is run." A future Qt-native port (embedded QGraphicsView + Qt mouse events instead of a standalone cv2 window) would be a UX polish item, not a Tk-elimination item.
2. **`mufasa/utils/confirm.py`** — lazy Tk importer (already documented in `backend_audit.md` §3d Bucket 4); fires only if no Qt override is installed. Survives.

These two are real Tier-4 close-out work for a follow-on patch (after Stage C).

---

## Stage A: Remove the `mufasa-tk` entry point — ✓ EXECUTED 122d4

**Scope:** 1-line config edit in `pyproject.toml`.

**Status (post-122d4):** the entry point is commented out with a 122d4 breadcrumb. The legacy launcher (`mufasa-tk`) no longer appears in the installed scripts. SimBA.py stays in tree as a `python -m mufasa.SimBA` backstop until Stage B (122d5) deletes it.

**Before (pre-122d4):**
```toml
[project.scripts]
mufasa-tk        = "mufasa.SimBA:main"          # legacy Tk entry point
mufasa           = "mufasa.cli.workbench_launcher:main"
mufasa-chooser   = "mufasa.ui_qt.app:main"
mufasa-workbench = "mufasa.ui_qt.workbench_app:main"
```

**After (post-122d4):**
```toml
[project.scripts]
# mufasa-tk        = "mufasa.SimBA:main"          # removed 122d4
mufasa           = "mufasa.cli.workbench_launcher:main"
mufasa-chooser   = "mufasa.ui_qt.app:main"
mufasa-workbench = "mufasa.ui_qt.workbench_app:main"
```

**Risk realisation:** users with shell history / shell aliases / docs referencing `mufasa-tk` will get `command not found` on a fresh install / `pip install -e .` re-run. The `python -m mufasa.SimBA` backstop softens the transition; Stage B removes that too.

**Why kept the comment line instead of clean delete:** future archaeology — easier to see "this used to exist" than to need a git blame. Cosmetic; either acceptable.

---

## Stage B: The cascade (99 files)

**Order of deletion within this stage doesn't matter** — every file has only SimBA.py-tree consumers. Listed here grouped by directory for review-clarity.

### B1. SimBA.py itself + ui/ Tk helpers (5 files)

| File | Sole consumer |
|---|---|
| `mufasa/SimBA.py` | Entry point only (entry point removed in Stage A) |
| `mufasa/ui/machine_model_settings_ui.py` | SimBA.py:93 |
| `mufasa/ui/utils.py` | SimBA.py |
| `mufasa/ui/get_tree_view.py` | `ui/pop_ups/print_video_meta_popup.py:4` (also dying this stage) |
| `mufasa/ui/video_timelaps.py` | `ui/pop_ups/video_processing_pop_up.py:28` (also dying this stage) |

### B2. ui/pop_ups (71 files — was 75 pre-122cz)

Every file in `mufasa/ui/pop_ups/` is imported by SimBA.py at module-load time, and ZERO of them have any other consumer. The whole directory cascade-deletes.

(Was 75 files at 122cx scoping. Four popups have been pre-deleted as part of cascade-prep work: 122cz deleted `direction_animal_to_bodypart_settings_pop_up.py` (the workflow-gap unblock); 122d1/d2/d3 deleted the 3 YOLO/conversion popups (`simba_rois_to_yolo_pop_up.py`, `yolo_inference_popup.py`, `yolo_pose_train_popup.py`) as Qt-port ports rather than feature drops.)

Subgroups for review (just for orientation; deletion is bulk):

* **Analytics/visualisation popups (~30 files):** `path_plot`, `heatmap_*`, `distance_*`, `movement_*`, `clf_*`, `gantt`, etc.
* **Tracking/import popups (~20 files):** `dlc_*`, `sleap_*`, `coco_*`, `yolo_*`, etc.
* **Project-mgmt popups (~15 files):** `csv_2_parquet`, `print_video_meta`, `check_videos_seekable`, `merge_*`, etc.
* **Other (~10 files):** ROI-related (kept after 122cr cluster-deletion; the 4 subprocess popups deleted in 122cs/ct/cu/cv leave the rest), `blob_visualizer`, `delete_all_rois`, `about_simba`, etc.

### B3. Tk labelling cluster (5 files)

| File | Status |
|---|---|
| `mufasa/labelling/labelling_interface.py` | SimBA.py-only (122cw evidence) |
| `mufasa/labelling/labelling_advanced_interface.py` | SimBA.py-only |
| `mufasa/labelling/standard_labeller.py` | SimBA.py-only |
| `mufasa/labelling/targeted_annotations_clips.py` | SimBA.py-only |
| `mufasa/mixins/annotator_mixin.py` | consumed only by `labelling/targeted_annotations_clips.py` (also dying) |

The other 5 files in `mufasa/labelling/` (`extract_labelled_frames`, `extract_labelling_meta`, `mitra_style_appender`, `play_annotation_video`, `single_clf_appender_excel`) are **backend utilities consumed by Qt forms** — they stay.

### B4. Unsupervised cluster (30 files)

Entire `mufasa/unsupervised/` directory:

* `unsupervised_main.py` (sole external consumer = SimBA.py:725 deferred import — 122cw evidence)
* 13 × `unsupervised/pop_ups/*.py` (each consumed only by unsupervised_main.py)
* 15 × algorithm-backend modules: `bout_aggregator.py`, `cluster_frequentist_calculator.py`, `cluster_validation_calculator.py`, `cluster_video_visualizer.py`, `cluster_xai_calculator.py`, `clusterer_comparison_calculator.py`, `data_extractor.py`, `dataset_creator.py`, `dbcv_calculator.py`, `embedding_correlation_calculator.py`, `grid_search_visualizers.py`, `hdbscan_clusterer.py`, `outlier_detector.py`, `tsne.py`, `umap_embedder.py` — all consumed only inside the cluster (verified in 122cx: zero outside `mufasa/unsupervised/`)
* `enums.py` (cluster-internal enum constants; zero outside consumers)

Plus `data_map.yaml` (non-py asset; dies with the directory removal).

**The 122cw audit undercounted this cluster as 14 files.** That was the popup + main count — the 15 algorithm-backend modules + `enums.py` were missed. The 122cx re-audit corrects to 30 files. The disposition is unchanged (SimBA.py finale; zero Qt reach); only the file count moves.

Zero Qt-side reach anywhere in the cluster. Cascade-deletes cleanly.

### B5. Stage B file count: 5 + 71 + 5 + 30 = **111 files** (post-122d3)

---

## Stage C: Tail (3 files) — ✓ EXECUTED 122d6

After Stage B cleared, two planned + one newly-discovered orphan went fully unreferenced:

| File | Why it died in Stage C | Detected when |
|---|---|---|
| `mufasa/ui/tkinter_functions.py` | Consumed by SimBA.py + 75 popups + 4 Tk labelling + 14 unsupervised + 2 mixins. All gone in Stage B. The remaining importer (`utils/confirm.py`'s lazy `from mufasa.ui.tkinter_functions import TwoOptionQuestionPopUp`) is wrapped in a `try/except ImportError`; post-Stage-C it always raises and the function falls back to `_stdin_confirm` — **working as designed**, NOT a broken-fallback. | 122cx scoping |
| `mufasa/mixins/pop_up_mixin.py` | Fan-in from every Tk popup. All deleted in Stage B. Zero surviving importers post-122d5. | 122cx scoping |
| `mufasa/mixins/unsupervised_mixin.py` | **Newly discovered in 122d6.** The 122cx audit missed this — it lives in `mufasa/mixins/`, not `mufasa/unsupervised/`, so Stage B's `git rm -r mufasa/unsupervised/` didn't sweep it. Zero surviving importers after Stage B. | 122d6 pre-flight |

**Re-evaluation of `utils/confirm.py` (the audit's "broken-fallback" claim):**

122cx's scoping doc said the lazy import would "fail at call time" after Stage C, calling it a "broken fallback path". On closer inspection of the file, that's not quite right — `_default_confirm` already wraps the Tk import in `try/except ImportError`:

```python
try:
    from mufasa.ui.tkinter_functions import TwoOptionQuestionPopUp
except ImportError:
    return _stdin_confirm(question, option_one, option_two, title)
```

After 122d6 deletes `tkinter_functions.py`, the import always raises `ImportError`, the `except` branch fires, and the function gracefully routes to stdin. This was already the intended fallback for headless / minimal environments (per the module docstring's §"Default behaviour" point #2). It's **not** broken; it's the documented headless path.

No follow-on `confirm.py` body rewrite is needed. The 122d7+ "cleanup" item for `confirm.py` is reduced to a cosmetic doc tweak (remove the now-misleading "Tk fallback" wording in the docstring, since Tk is no longer in the tree) — or just leave the docstring alone since it describes the general design pattern.

**Pre-existing orphan note:** `mufasa/mixins/network_mixin.py` was already orphan **before** Stage B (NetworkMixin had no importers in any of the Stage-B-deleted files). It's pre-existing dead code, NOT exposed by the cascade. Disposition deferred to a separate "pre-existing-orphans" cleanup patch (or formally accept it as a library-API entry point — `NetworkMixin` is feature-grade analytical code that user code might subclass). Not Stage C's concern.

---

## Tier-4 remaining after Stage C

The following two files **do not die in this cascade** and need separate Tier-4 work:

### 1. `mufasa/ui/px_to_mm_ui.py`

**Consumers (post-Stage-B):**
- `ui_qt/forms/video_utilities.py:318` — imports `GetPixelsPerMillimeterInterface`

**Correction (patch 122dd):** This file is NOT a Tk surface, despite the 122cx audit's framing. It uses pure OpenCV (`cv2.namedWindow`, `cv2.imshow`, `cv2.setMouseCallback`) — no Tk imports anywhere. The "Qt falls through to Tk" framing was wrong; it's "Qt falls through to cv2 standalone window."

When the user clicks "calibrate pixels per millimeter" on the Qt video-utilities form, an OpenCV window opens for click-to-place calibration. This works for both v1 and legacy projects (the function takes `video_path` directly, no config_path resolution).

**Disposition:**
- **Option A:** Port `GetPixelsPerMillimeterInterface` to a Qt dialog with `QGraphicsView` + `QGraphicsScene` + mouse-event handlers. ~150–250 lines. Pure UX polish — embedded panel instead of standalone window. Doesn't change behaviour or fix any bug.
- **Option B (current):** Leave as-is. The cv2 window opens, the user calibrates, the form receives the result via the `iface.ppm` attribute. Functional; just visually inconsistent with the rest of the Qt workbench.

Either way, this file survives Stages A–C cleanly. Scheduled for a future UX polish patch if/when desired.

### 2. `mufasa/utils/confirm.py`

**Correction (patch 122dd):** The pre-122dd characterization of this file ("broken fallback path", "needs body rewrite") was wrong. Closer inspection found:

* `_default_confirm` already wraps the Tk import in `try/except ImportError`. Post-Stage-C, the import always raises ImportError, the `except` branch fires, and the function gracefully routes to `_stdin_confirm`. **Working as designed** — the stdin fallback is the documented headless path.
* The Qt override IS installed at workbench startup (patch 122cj — `mufasa/ui_qt/qt_confirm.py` + workbench_app.py:185). Backend `confirm_two_option` calls from a Qt session route through `QMessageBox.question` automatically. The `_default_confirm` stdin path is only reached in headless / CLI contexts.

**Status:** patch 122dd cosmetic-rewrites the module docstring to reflect post-Stage-C reality and adds a comment clarifying that the lazy Tk-import try/except is retained for diff-stability rather than rewritten. No behaviour change. The file is correctly designed for both interactive (Qt) and headless (stdin) contexts.

---

## Risk register

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| Users with shell aliases / docs reference `mufasa-tk` | High | Low | Stage A's commit message + release notes. SimBA.py stays a stage longer so `python -m mufasa.SimBA` works during transition. |
| Some workflow has no Qt equivalent and was Tk-only | Medium | High | Pre-Stage-B audit checklist (see below). Each major feature category in B2 should have a checked Qt counterpart. |
| Hidden runtime consumer of one of the 99 files | Low | Medium | The AST audit caught zero non-SimBA consumers. The 122cr lesson (string-literal subprocess imports invisible to AST) applies — grep the codebase for popup filenames in any string context as a pre-flight check. |
| `tkinter_functions.py` consumed somewhere unexpected | Low | Low | Audit complete (no Qt consumers). If a Stage-B file accidentally remains, Stage C's deletion fails the AST check and we catch it. |
| `utils/confirm.py` lazy import fires after Stage C | Low | Low | Only fires if no Qt override is set. Workbench installs override at startup. Acceptable broken fallback. |

### Pre-Stage-B audit checklist

Before running Stage B, verify each major feature category has either (a) a Qt counterpart in `mufasa/ui_qt/`, or (b) an explicit "feature not in v1" decision logged. Categories from B2:

- [ ] Analytics/visualisation: path-plot, heatmap, distance, movement, clf-validation, gantt, severity, ROI-aggregate, kleinberg, FSTTC, mutual-exclusivity, pup-retrieval, spontaneous-alternation
- [ ] Tracking imports: DLC, SLEAP, COCO-keypoints, YOLO-keypoints, labelme
- [ ] Project management: CSV-to-parquet, print-video-meta, check-videos-seekable, merge-coco-keypoint-files
- [ ] ROI: delete-all-rois (already covered by Qt ROIManageForm)
- [ ] Video processing: video_processing_pop_up (parts likely covered by `ui_qt/forms/video_utilities.py`)
- [ ] About / settings: about_simba_pop_up

**Sweep completed in 122cy; blocker resolved in 122cz; 3 feature-decision items resolved in 122d1/d2/d3 (all ported).** See `docs/stage_b_checklist.md` for the full results. **TL;DR:**

* **69** of 75 popups have verified Qt counterparts (covered).
* **2** are hard drops (about, splash — cosmetic/admin).
* ~~**1** is a **blocking workflow gap**~~ ✓ **RESOLVED 122cz** — directing-bodypart settings ported to `DirectingBodyPartSettingsForm` on addons_page.
* ~~**3** are non-blocking gaps requiring feature-disposition decisions~~ ✓ **RESOLVED 122d1/d2/d3** — all 3 YOLO/conversion popups ported (SimBARoisToYoloForm on tools_page; YOLOPoseInferenceForm + YOLOPoseTrainForm on classifier_page).

Stage B is now fully ready for execution.

---

## Staging plan (actuals)

| Patch | Stage | Files deleted | Config edits | Status |
|---|---|---:|---|---|
| **122cx** | Scoping | 0 | 0 | ✓ Done — scoping doc + smoke test |
| **122cy** | Pre-Stage-B checklist | 0 | 0 | ✓ Done — checklist sweep, 1 blocker + 3 feature-decisions surfaced |
| **122cz** | Stage B prep (blocker port) | 1 | 0 | ✓ Done — directing-bodypart settings ported to Qt |
| **122d0** | Drive-by (QWI-4) | 0 | 1 (`workbench_app.py`) | ✓ Done — Qt-workbench bug tracking + page order fix |
| **122d1** | Stage B prep (YOLO port #1) | 1 | 0 | ✓ Done — simba_rois_to_yolo ported |
| **122d2** | Stage B prep (YOLO port #2) | 1 | 0 | ✓ Done — yolo_inference ported |
| **122d3** | Stage B prep (YOLO port #3) | 1 | 0 | ✓ Done — yolo_pose_train ported |
| **122d4** | Stage A | 0 | 1 (`pyproject.toml`) | ✓ Done — `mufasa-tk` entry point removed |
| **122d5** | Stage B | 114 | 0 | ✓ Done — bulk delete |
| **122d6** | Stage C tail | 3 | 0 | ✓ **Done — tail deletion (this patch); 2 planned + 1 newly discovered orphan** |
| **122d7+** | Cleanup | 0 | 0 | Optional — README sweep, QWI-1/2/3 fixes, network_mixin disposition, px_to_mm_ui Qt port |

Total elapsed: 9 patches (122cx → 122d5). Stage C is mechanical; cleanup is optional polish.

Alternatively, Stage B could be split by directory:

* 122cz.1 — ui/pop_ups (75 files)
* 122cz.2 — labelling Tk UI (5 files)
* 122cz.3 — unsupervised (14 files)
* 122cz.4 — SimBA.py + ui/ Tk helpers (5 files)

Split is safer (smaller blast radius per patch) but each split needs the same pre-cascade verification, multiplying review overhead. **Recommended: monolithic Stage B** assuming the pre-Stage-B checklist passes cleanly.

---

## Boundary regression guards

`tests/smoke_122cx_simba_death_cascade_scope.py` pins the exact numbers in this document. If a future patch:

* Adds a Qt-side import of an unsupervised file
* Adds a non-SimBA consumer of a `ui/pop_ups` file
* Adds a non-Qt consumer of `px_to_mm_ui.py`

...the smoke test fails and the scoping is re-examined before Stage B can land.
