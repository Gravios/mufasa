# Pre-Stage-B checklist sweep

**Status:** completed in patch 122cy.
**Authority:** automated coverage analysis (keyword + class-name match against `mufasa/ui_qt/forms,pages,dialogs/`) + manual verification of edge cases. All 75 popups in B2 audited.

## Summary

| Category | Count |
|---|---:|
| Covered by Qt workbench | **69** |
| Hard drops (admin/cosmetic) | **2** |
| Workflow gaps requiring Qt work before Stage B | **1** |
| Feature decisions required (YOLO/conversion) | **3** |
| **Total** | **75** |

**Verdict:** Stage B is **not yet ready** for monolithic execution. One real workflow gap (`direction_animal_to_bodypart_settings`) blocks the "Directing toward body-part — statistics" analysis route. Three additional gaps need feature-disposition decisions but don't block other workflows.

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

## §3. Workflow gaps (1 popup — BLOCKING)

These popups have **no Qt counterpart**, and the workflow they support **cannot be invoked without them**.

### 3.1. `direction_animal_to_bodypart_settings_pop_up.py` — BLOCKING

**Class:** `DirectionAnimalToBodyPartSettingsPopUp`
**What it does:** lets the user pick, per animal, which body-part is the "direction-from" reference for the directing-to-bodypart analysis. Writes selections to project config:
- Section: `ConfigKey.DIRECTIONALITY_SETTINGS.value`
- Key: `bodypart_direction`

**Qt dependency:** `AnalysisForm`'s "Directing toward body-part — statistics" route (line 315 in `analysis.py`) calls `DirectingAnimalsToBodyPartAnalyzer(config_path=...)`. The backend takes only `config_path` and reads the body-part selections from the project's directing-settings — which the popup writes.

**Comment in `analysis.py:311-313` explicitly says:**
> "the body-part configuration is read from the project's directing-settings (set via the dedicated settings popup or project config)."

**Impact if deleted in Stage B:** users running the Qt route would hit a config-read error (missing keys). They'd have to hand-edit `project_config.ini`.

**Disposition options:**
1. **Port to Qt as a settings dialog.** Small popup (~50 lines): per-animal dropdown for body-part picks. Similar pattern to 122cs (`ROISizeStandardizerDialog`). **Recommended.**
2. **Inline the selectors into AnalysisForm.** When the directing-to-bodypart route is selected, surface a body-part dropdown per animal. Saves a dialog but adds complexity to the universal AnalysisForm. Trade-off.
3. **Drop the directing-to-bodypart route entirely.** Acceptable if the feature is not in v1 scope. Removes both the popup AND the AnalysisForm route. Cleanest but a feature loss.

**Recommendation:** Option 1 (port to Qt). Estimated 1 patch (~80–120 lines of Qt code, similar shape to 122cs/cu).

---

## §4. Feature-decision-required (3 popups — NON-BLOCKING)

These popups have no Qt counterpart, but the workflow has no Qt-side caller — deleting them just drops the GUI for a feature that's not exposed in the workbench. Backend modules survive.

### 4.1. `simba_rois_to_yolo_pop_up.py`

**Class:** `SimBAROIs2YOLOPopUp`
**What it does:** converts SimBA ROI definitions to YOLO bounding-box training data.

**Disposition options:**
1. **Add a converter to tools_page.** ConverterForm already covers similar formats (DLC, SLEAP, Labelme); adding ROI→YOLO is consistent.
2. **Drop the feature.** The workbench's existing converters cover all the main pose-data formats. ROI→YOLO is a niche workflow; users wanting it can use the backend directly.

**Recommendation:** Drop unless explicitly required. If kept, port via tools_page in a follow-on patch.

### 4.2. `yolo_inference_popup.py`

**Class:** `YOLOPoseInferencePopUP`
**What it does:** GUI for `YOLOPoseInference` — runs a trained YOLO pose model on input videos.

**Disposition options:**
1. **Port to a workbench form.** Could go on classifier_page next to RunInferenceForm (which handles SimBA classifiers, not YOLO).
2. **Drop the GUI.** YOLO inference is typically a CLI workflow; users with YOLO models will scripts invoking `YOLOPoseInference` directly.

**Recommendation:** Drop unless YOLO model inference is a v1 workbench feature. The backend stays.

### 4.3. `yolo_pose_train_popup.py`

**Class:** `YOLOPoseTrainPopUP`
**What it does:** GUI for training a YOLO pose model — wraps the YOLO training subprocess with CUDA detection and parameter configuration.

**Disposition options:**
1. **Port to a workbench form.** Similar to TrainClassifierForm but for YOLO pose.
2. **Drop the GUI.** YOLO training is a long-running offline operation typically done outside a GUI; the backend training script can be invoked directly.

**Recommendation:** Drop. The backend stays accessible from CLI.

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
