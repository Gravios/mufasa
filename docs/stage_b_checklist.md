# Pre-Stage-B checklist sweep

**Status:** completed in patch 122cy.
**Authority:** automated coverage analysis (keyword + class-name match against `mufasa/ui_qt/forms,pages,dialogs/`) + manual verification of edge cases. All 75 popups in B2 audited.

## Summary

| Category | Count |
|---|---:|
| Covered by Qt workbench | **69** |
| Hard drops (admin/cosmetic) | **2** |
| Workflow gaps requiring Qt work before Stage B | ~~1~~ **0 (resolved 122cz)** |
| Feature decisions required (YOLO/conversion) | ~~3~~ **0 (resolved 122d1/d2/d3 by porting)** |
| **Total** | **75 → 71 (4 ported in 122cz/d1/d2/d3)** |

**Verdict (post-122d3):** Stage B is **fully ready for execution**. All 4 popup gaps from the original 122cy sweep are now resolved — 1 via Qt port (122cz; was blocker), 3 via Qt port (122d1/d2/d3; were feature decisions). No more pre-Stage-B preparation needed.

---

## §1. Covered by Qt workbench (69 popups)

These have verified Qt counterparts — either by direct class-name match or by route/form mapping in the workbench pages. Safe to bulk-delete in Stage B.

### Analysis / Statistics (10 popups)
- `animal_directing_other_animals_pop_up.py` → AnalysisForm "Directing toward other animals" route
- `clf_annotation_counts_pop_up.py` → AnnotationReportsForm (annotation_page)
- `clf_by_roi_pop_up.py` → ROIAnalysisForm
- `clf_by_timebins_pop_up.py` → AnalysisForm classifier time-bins route
- `clf_descriptive_statistics_pop_up.py` → AnalysisForm classifier descriptive route
- `distance_analysis_pop_up.py` → AnalysisForm "Distance between body-parts" route
- `distance_timebins_popup.py` → AnalysisForm "Distance by time bins" route (line 252-263 in analysis.py)
- `fsttc_pop_up.py` → AnalysisForm
- `movement_analysis_pop_up.py` → AnalysisForm "Movement analysis" route
- `movement_analysis_time_bins_pop_up.py` → AnalysisForm "Movement analysis by time bins" route
- `severity_analysis_pop_up.py` → AnalysisForm severity route (with `severity_mode` selector)

### Visualizations (15 popups)
- `annotated_bouts_videos_pop_up.py` → VisualizationForm
- `blob_visualizer_pop_up.py` → VisualizationForm "BlobVisualizer" route (line 676 in visualizations.py)
- `clf_plot_pop_up.py` → VisualizationForm
- `clf_probability_plot_pop_up.py` → VisualizationForm
- `clf_validation_plot_pop_up.py` → ValidateClassifierForm
- `data_plot_pop_up.py` → VisualizationForm
- `directing_animal_to_bodypart_plot_pop_up.py` → VisualizationForm
- `directing_other_animals_plot_pop_up.py` → VisualizationForm
- `distance_plot_pop_up.py` → VisualizationForm
- `ez_path_plot_popup.py` → VisualizationForm
- `gantt_pop_up.py` → VisualizationForm
- `heatmap_clf_pop_up.py` → VisualizationForm
- `heatmap_location_pop_up.py` → VisualizationForm
- `make_path_plot_pop_up.py` → VisualizationForm
- `path_plot_pop_up.py` → VisualizationForm
- `quick_path_plot_pop_up.py` → VisualizationForm
- `validation_plot_pop_up.py` → ValidateClassifierForm
- `visualize_pose_in_dir_pop_up.py` → VisualizationForm
- `yolo_plot_results.py` → VisualizationForm "YOLOPoseVisualizer" route (line 657 in visualizations.py)

### ROI (8 popups)
- `append_roi_features_animals_pop_up.py` → ROIFeaturesForm
- `append_roi_features_bodypart_pop_up.py` → ROIFeaturesForm
- `delete_all_rois_pop_up.py` → ROIManageForm (already wired via Qt; 122cs era)
- `remove_roi_features_pop_up.py` → ROIFeaturesForm (line 422 in roi.py — rewired in 122cd)
- `roi_aggregate_stats_popup.py` → ROIAnalysisForm
- `roi_analysis_time_bins_pop_up.py` → ROIAnalysisForm
- `roi_features_plot_pop_up.py` → ROIVisualizeForm
- `roi_tracking_plot_pop_up.py` → ROIVisualizeForm

### Classifier (3 popups)
- `clf_add_remove_print_pop_up.py` → ClassifierManageForm
- `run_machine_models_popup.py` → RunInferenceForm
- (clf_validation_plot covered in Visualizations above)

### Add-ons (4 popups)
- `kleinberg_pop_up.py` → KleinbergForm (addons_page)
- `mutual_exclusivity_pop_up.py` → MutualExclusivityForm (addons_page)
- `pup_retrieval_pop_up.py` → PupRetrievalForm (addons_page)
- `spontaneous_alternation_pop_up.py` → SpontaneousAlternationForm (addons_page)

### Conversion / Import (15 popups)
- `coco_keypoints_to_yolo_popup.py` → ConverterForm (tools_page)
- `csv_2_parquet_pop_up.py` → ExportToCSVForm (tools_page)
- `dlc_h5_inference_to_yolo_popup.py` → ConverterForm
- `dlc_to_labelme_popup.py` → ConverterForm
- `dlc_to_yolo_keypoints_popup.py` → ConverterForm
- `labelme_bbox_to_yolo_bbox_popup.py` → ConverterForm
- `labelme_to_df_popup.py` → ConverterForm
- `labelme_to_imgs_popup.py` → ConverterForm
- `merge_coco_keypoint_files_pop_up.py` → ConverterForm
- `simba_to_yolo_keypoints_popup.py` → SimBA→YOLO Keypoints (data_import.py:298 SimBA2Yolo backend)
- `sleap_annotations_to_yolo_popup.py` → SLEAPToYoloForm (tools_page)
- `sleap_h5_inference_to_yolo_popup.py` → SLEAPToYoloForm

### Annotation (3 popups)
- `select_video_for_labelling_popup.py` → FrameLabellingLauncher (annotation_page)
- `select_video_for_pseudo_labelling_popup.py` → annotation_page (pseudo labelling route)
- `third_party_annotator_appender_pop_up.py` → ThirdPartyAppenderForm (annotation_page)

### Pose / Tracking (3 popups)
- `egocentric_alignment_pop_up.py` → EgocentricAlignmentForm (pose_cleanup_page)
- `pose_bp_drop_pop_up.py` → DropBodypartsForm (pose_cleanup_page)
- `smoothing_popup.py` → SmoothingForm (pose_cleanup_page)

### Video Processing (4 popups)
- `check_videos_seekable_pop_up.py` → CheckVideoSeekableForm (video_processing_page Utilities)
- `multiple_videos_to_frames_popup.py` → ExtractFramesForm (video_processing_page)
- `print_video_meta_popup.py` → AverageFrameForm (video_processing_page Metadata) or VideoInfoForm
- `single_video_to_frames_popup.py` → ExtractFramesForm
- `video_processing_pop_up.py` → distributed across video_processing_page forms

### Features (2 popups)
- `boolean_conditional_slicer_pup_up.py` → AnalysisForm "Boolean conditional slicing" (analysis.py:283)
- `subset_feature_extractor_pop_up.py` → FeatureSubsetExtractorForm (features_page)

---

## §2. Hard drops — admin/cosmetic (2 popups)

Acceptable to delete with no Qt replacement; not user-facing workflows.

- `about_simba_pop_up.py` — "About SimBA" dialog. Qt workbench shows version info in project_setup_page. Cosmetic info dialog; no workflow lost.
- `splash_popup.py` — `SplashMovie` startup splash. Qt workbench has its own loading sequence; the legacy SimBA splash isn't a user-facing feature anyone runs deliberately.

---

## §3. Workflow gaps (1 popup — ~~BLOCKING~~ ✓ RESOLVED 122cz)

These popups have **no Qt counterpart**, and the workflow they support **cannot be invoked without them**.

### 3.1. `direction_animal_to_bodypart_settings_pop_up.py` — ✓ RESOLVED 122cz

**Class:** `DirectionAnimalToBodyPartSettingsPopUp`
**What it did:** let the user pick, per animal, which body-part is the "direction-from" reference for the directing-to-bodypart analysis. Wrote selections to project config:
- Section: `ConfigKey.DIRECTIONALITY_SETTINGS.value`
- Key: `bodypart_direction`

**Qt dependency:** `AnalysisForm`'s "Directing toward body-part — statistics" route (line 315 in `analysis.py`) calls `DirectingAnimalsToBodyPartAnalyzer(config_path=...)`. The backend takes only `config_path` and reads the body-part selection from the project's directing-settings — which the popup wrote.

**Resolution (122cz):** Ported to a Qt-native form — `DirectingBodyPartSettingsForm` in `mufasa/ui_qt/forms/addons.py`. Wired into `addons_page.py` as the "Directing — body-part settings" section. The Tk popup file is deleted; SimBA.py's import + button + grid wiring are commented out (same surgical pattern as 122ck/122cr).

Notable port differences:
* **Single dropdown** instead of one-per-animal. The legacy Tk popup looped per animal but wrote the same single key (only the last animal's choice persisted), so the per-animal UI was misleading. The Qt form transparently writes the single key with a single dropdown.
* **Body-part choices** are the union of body-part names across all animals (matches the backend's `bp_x_name = bodypart_direction + '_x'` lookup, which requires the name to exist for every animal).
* **Settings-only** — the Tk popup also kicked off the analyzer at the end of save(). The Qt form is settings-only; users run the analysis from AnalysisForm separately (cleaner separation of concerns).

After this patch, the AnalysisForm's "Directing toward body-part — statistics" route has a complete Qt path. Stage B no longer has a blocking workflow gap.

---

## §4. Feature-decision-required ~~(3 popups — NON-BLOCKING)~~ ✓ ALL 3 PORTED 122d1/d2/d3

The 3 non-blocking gap popups identified in 122cy were ported to Qt rather than dropped, after the user opted to keep the YOLO/ROI conversion features in v1.

### 4.1. `simba_rois_to_yolo_pop_up.py` — ✓ Ported 122d1

Replaced by `SimBARoisToYoloForm` on the Tools workbench page, section "SimBA ROIs → YOLO conversion". Mirrors the existing `SLEAPToYoloForm` pattern (sibling on the same page). Supports the same parameters as the Tk popup (config path, optional video_dir/save_dir, OBB toggle, frame count, train size, greyscale, CLAHE, verbose).

### 4.2. `yolo_inference_popup.py` — ✓ Ported 122d2

Replaced by `YOLOPoseInferenceForm` on the Classifier workbench page, section "YOLO pose — inference". Single form with a Mode selector (Single video / Video directory) replacing the Tk popup's two-button layout. 14 parameters from the original popup mapped to Qt widgets. Branches between `YOLOPoseInference` and `YOLOPoseTrackInference` based on optional tracker .yml. CUDA + ultralytics availability checked in `collect_args` (friendly error path).

### 4.3. `yolo_pose_train_popup.py` — ✓ Ported 122d3

Replaced by `YOLOPoseTrainForm` on the Classifier workbench page, section "YOLO pose — train". Detached subprocess pattern preserved — `subprocess.Popen` launches `python -m mufasa.model.yolo_fit` so the workbench doesn't wait for hours-long training. Info dialog confirms launch with save directory.

---

## §5. Recommended Stage B preparation patches

To unblock Stage B execution:

1. **122cy.1** (already this patch) — record this checklist; no code changes.
2. **122cy.2** — **REQUIRED:** Port `DirectionAnimalToBodyPartSettingsPopUp` to a Qt dialog. ~80–120 lines, similar shape to 122cs `ROISizeStandardizerDialog`. After this lands, the AnalysisForm's body-part directing route has a complete Qt path.
3. **122cy.3** (optional) — feature-disposition decisions:
   - Drop simba_rois_to_yolo / yolo_inference / yolo_pose_train? (Recommended.) Or port them?
   - Record decisions in this doc.

After 122cy.2 lands (and optionally 122cy.3), Stage B can execute as planned in `simba_death_cascade.md` — 115 file deletions.

**If you accept the recommendation to drop the 3 YOLO/ROI conversion popups:** Stage B's effective scope unchanged (still 115 files; just that 3 of them go away as dropped features rather than ported ones).

**If you want to keep all 4 features:** Stage B is delayed by 4 small Qt-port patches (similar to 122cs/ct/cu/cv pattern). Adds maybe 1 day of work.

---

## §6. Methodology notes

The coverage analysis used three filters:

1. **Class-name substring match** — does the popup's primary class name (case-insensitive) appear anywhere in `mufasa/ui_qt/forms,pages,dialogs/`? If yes, HIGH confidence.
2. **Keyword overlap** — do all of the popup's name-derived keywords (snake_case parts ≥ 3 chars, excluding generic words) appear in the Qt source? If yes, MEDIUM confidence.
3. **Manual verification** — for MEDIUM cases and questionable HIGH cases, grep the Qt source for code-level (not docstring) references to confirm a real wiring exists.

The class-name filter alone is **too permissive** — docstring "this replaces FooBarPopUp" notes count as matches even when no Qt form wires the backend. The 122cy sweep checks both the docstring + actual code-level wiring (route labels, form definitions, page additions).

False-positive rate of the original class-name filter: ~5% (4 of 73 HIGH+MEDIUM cases needed correction after manual check, with 1 actually corrected to "real gap" and 3 to "no Qt feature").

For future audits of this kind, prefer "match in Qt page wiring" (a form added via `add_section`) as the strongest indicator, rather than any textual match in the Qt source.
