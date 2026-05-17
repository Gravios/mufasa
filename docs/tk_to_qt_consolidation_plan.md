# Tk → Qt consolidation plan

**Generated:** post-patch 122br (May 2026).
**Purpose:** Comprehensive redesign of the Tk popup surface as Qt workbench forms. Defines the target layout, maps every Tk popup to its Qt destination (existing or proposed-new), identifies gaps, and orders the migration work.

**Status:** Design document. No code changes in this iteration. Implementation is multiple follow-up patches.

This doc supersedes the recipe-by-recipe "rename Tk-only labels" cleanup lane that was cancelled in patch 122br once we recognized the Tk surface is slated for removal.

**Companion docs:** `tk_surface_audit.md` (per-file inventory + removal-dependency graph), `workflow_audit.md` (canonical pipeline + page topology), `workflows.md` (developer-facing status-tagged audit).

---

## 1. State of play

| | Count |
|---|---:|
| Tk popups invoked from `mufasa/SimBA.py` | ~70 distinct classes |
| Tk popups also reachable via backend modules (e.g. `cue_light_main_popup.py`) | ~15 more |
| Total Tk popups in `mufasa/ui/pop_ups/` (and adjacent) | **~85** |
| Current Qt workbench pages | 14 |
| Tk popups already covered by Qt forms (shared backend class) | **≈ 25 direct + ≈ 30 via `VisualizationForm` route table** |
| Tk popups with NO direct Qt equivalent | **≈ 30** |

The Qt workbench is **further along than it looks**. Most video-processing, ROI, preprocessing, and visualization work is already consolidated in Qt. The gaps are concentrated in:

- A handful of dedicated analysis-stats popups (movement/distance/direction analysis)
- A small number of data-format conversion popups
- Some admin/utility popups (model info, video metadata, etc.)
- The unsupervised module (`UnsupervisedGUI` — entirely Tk)

The remainder of this document maps the entire Tk surface to its Qt destination and orders the closure work.

---

## 2. Target Qt workbench layout

Below is the **proposed final layout** of the Qt workbench after all Tk popups are retired. Existing pages are kept; new sections are added where needed; the count of forms per page stays manageable (≤ 8 sections per page where possible).

Sections marked **(existing)** already exist in Qt. Sections marked **(new)** are proposed additions. The number in parentheses after each section title is the count of Tk popups it absorbs.

### Page 1 — Projects (no change)
- **Create or open project** (existing) → from `LoadProjectPopUp`, `SimbaProjectPopUp` ✓
- **Project information** (existing)

### Page 2 — Data Import (no change)
- **Import Pose Data** (existing)
- **Import video** (existing)

### Page 3 — Video Processing (substantial, mostly done)
- **Format conversion** (existing, 6) → `Convert2AVI`, `Convert2MP4`, `Convert2MOV`, `Convert2WEBM`, `Convert2BlackWhite`, `Greyscale` ✓
- **Trim & split** (existing, 5) → `ClipVideo`, `ClipMultipleByTimestamps`, `ClipMultipleByFrameNumbers`, `ClipSingleByFrameNumbers`, `MultiShorten` ✓
- **Crop & mask** (existing, 4) → `CropVideo`, `CropVideoCircles`, `CropVideoPolygons`, `MultiCrop` ✓
- **Resize & rate** (existing, 5) → `Downsample{Single,Multiple}`, `Upsample`, `ChangeFps{Single,Multiple}` ✓
- **Rotate & flip** (existing, 3) → `RotateVideoSetDegrees`, `VideoRotator`, `FlipVideos` ✓
- **Filters & enhancement** (existing, 4) → `BoxBlur`, `BrightnessContrast`, `CLAHE`, `InteractiveClahe` ✓
- **Overlay / burn-in** (existing, 7) → `SuperImposeFrameCount`, `SuperimposeTimer`, `SuperimposeProgressBar`, `SuperimposeText`, `SuperimposeVideoNames`, `SuperimposeWatermark`, `SuperimposeVideo` ✓
- **Frame extraction** (existing, 2) → `SingleVideo2Frames`, `MultipleVideos2Frames` ✓
- **Merge frames → video** (existing)
- **Import frame directory** (existing)
- **Join & transition** (existing, 4) → `Concatenating`, `Concatenator`, `VideoTemporalJoin`, `ManualTemporalJoin` ✓
- **Image format conversion** (existing)
- **Utilities** (existing, 3) → reverse, change speed, pixels-per-mm. Add (1) → `CheckVideoSeekable`
- **Metadata & audit** (existing, 2) → `AverageFrame`, `PrintVideoMetaData` ✓
- **Background removal** (new, 2) → `BackgroundRemoverSingleVideo`, `BackgroundRemoverDirectory` ⚠ GAP

**Net status:** 14 sections cover ~33 Tk popups. **1 gap** (background removal).

### Page 4 — Preprocessing (mostly done)
- **Preprocess Videos** (existing)
- **Video Calibration** (existing)
- **Interpolate missing frames** (existing)
- **Kalman v2 smoothing** (existing)
- **Run outlier correction** (existing)
- **Skip outlier correction** (existing)
- **Egocentric alignment** (existing, 1) → `EgocentricAlign` ✓
- **Advanced / legacy** (existing) — has `SmoothingForm` ← from `Smoothing` ✓, `OutlierSettings`, `DropBodyparts` ← from `DropTrackingData` ✓

**Net status:** All Tk preprocessing popups covered.

### Page 5 — ROI (mostly done)
- **Definitions** (existing) — covers `ROIManage`, `DuplicateROIsBySourceTarget` ✓
- **Analyze** (existing, 2) — covers `ROIAnalysis`, `ROIAnalysisTimeBins`, `ROIAggregateStats` ✓ (route through existing form)
- **Visualize** (existing, 1) → `ROITrackingPlot` ✓
- **Features** (existing, 3) → `AppendROIFeaturesByAnimal`, `AppendROIFeaturesByBodyPart`, `ROIFeaturesPlot` ✓
- **Import ROI CSV** (new, 1) → `ROIDefinitionsCSVImporter` ⚠ exists as backend, needs Qt surfacing
- **Size standardiser** (new, 1) → `ROISizeStandardizer` ⚠ exists as backend, needs Qt surfacing

**Net status:** 4 existing sections, 2 gaps that should be additional sections (small popups; both backend ports already exist).

### Page 6 — Features (small)
- **Compute feature subsets** (existing, 1) → `FeatureSubsetExtractor` ✓

**Net status:** ✓ done.

### Page 7 — Annotation (substantial)
- **Frame labelling** (existing, 1) → `FrameLabelling` ✓
- **Targeted annotation clips** (existing, 1)
- **Third-party annotation import** (existing, 1) → `ThirdPartyAppender` ✓
- **Review classifier predictions** (existing, 1)
- **Reports** (existing, 1) → `ClfAnnotationCount` ✓
- **Annotated bouts → videos** (new, 1) → `PlotAnnotatedBouts` ⚠ GAP

**Net status:** 5 existing, 1 gap.

### Page 8 — Classifier (mostly done)
- **Manage classifiers** (existing) → covers `AddClf`, `RemoveAClassifier`, `PrintModelInfo` ✓
- **Run inference** (existing) → covers `RunMachineModels` ✓
- **Train classifier** (existing)
- **Validate classifier** (existing) → covers `ClassifierValidationClips`, `ClfValidationPlot` ✓
- **Descriptive statistics** (new, 1) → `AggregateClfCalculator` (also via `AnalysisForm`; need to clarify which one is canonical) ⚠ POSSIBLE DUPLICATE
- **Classifier by ROI / by time bins** (new, 2) → `ClfByROI`, `ClfByTimebins` ⚠ GAPS

**Net status:** 4 existing, possibly 3 gaps (1 is a duplicate question).

### Page 9 — Visualizations (consolidated already)
- **Create visualisation** (existing, single form with 29 routes)

This form is the single canonical entry point for plotting backends. It already routes to:

`BlobVisualizer`, `DataPlotter`, `DirectingOtherAnimalsVisualizer`, `DirectingAnimalsToBodyPartVisualizer`, `DistancePlotter`, `EzPathPlot`, `GanttCreator`, `MakePathPlot`, `PlotAnnotatedBouts`, `PlotSklearnResults`, `TresholdPlotCreator`, `VisualizePoseInFolder`, `YoloPoseVisualizer`, and ~16 others — covering essentially all plot-related Tk popups.

**Action items here:**
- Verify every Tk plot popup is routed through `VisualizationForm`. If any aren't (e.g., `BlobVisualizer`, `MakePathPlot`), add the route.
- Retire the Tk plot popups (they're now redundant).

### Page 10 — Analysis (small)
- **Run analysis** (existing) → covers `MovementAnalysis`, `MovementAnalysisTimeBins`, `DistanceAnalysis`, `DistanceTimebins`, `DirectingOtherAnimals`, `DirectionAnimalToBodyPartSettings`, `BooleanConditionalSlicer`, `FSTTC`, `AnimalDirectingOtherAnimals` (via dispatcher table)

**Action item:** Confirm the dispatcher covers all 9 Tk analysis popups. If not, add routes. Same model as `VisualizationForm` — single form with internal routing.

**Net status:** Likely ✓ done; needs verification pass.

### Page 11 — Add-ons (mostly done)
- **Cue-light — data analysis** (existing)
- **Cue-light — classifier statistics** (existing)
- **Cue-light — movement statistics** (existing)
- **Cue-light — visualizer** (existing)
- **Kleinberg burst smoothing** (existing)
- **Mutual exclusivity corrector** (existing)
- **Pup retrieval** (existing)
- **Spontaneous alternation** (existing)
- **Blob tracker — initialise** (existing)
- **Blob quick-check** (new, 1) → `BlobQuickChecker` ⚠ GAP (Tk-only)
- **Unsupervised analysis** (new, 1) → consolidates `UnsupervisedGUI` (Tk-only Tk frame) ⚠ LARGE GAP

**Net status:** 9 existing, 2 gaps. The unsupervised lane is the single largest remaining piece of work.

### Page 12 — Tools (extend)
- **Convert pose / annotation data** (existing) → consolidates `Csv2Parquet`, `Parquet2Csv`, `DLC2Labelme`, `Labelme2Img`, `Labelme2DataFrame`, `LabelmeBbox2YoloBbox`, `DLC2Yolo`, `SLEAPAnnotations2Yolo`, `DLCH5Inference2Yolo`, `SLEAPH5Inference2Yolo`, `COCOKeypoints2Yolo`, `SimBA2YoloKeypoints`, `MergeCOCOKeypointFiles`, `SimBAROIs2YOLO`, `ConvertROIDefinitions` ✓ (mostly via dispatcher; **verify all 15 are routed**)
- **Re-order pose keypoints** (existing)
- **SLEAP → YOLO conversion** (existing) ← could fold into "Convert pose" above
- **Export to CSV** (existing)

**Net status:** 4 existing sections cover ~15 Tk popups via a converter dispatcher. **Verify dispatcher coverage.** Possibly consolidate "SLEAP → YOLO" back into "Convert pose / annotation data" to eliminate redundancy.

### Page 13 — Help/About (proposed)
- **About** (new, 1) → `AboutSimBAPopUp` — currently in Tk launcher's Help menu

**Net status:** New page (or new section under an existing page). Tiny.

---

## 3. Gap summary

The full list of Tk popups with **no current Qt route or section**:

### High-priority gaps (substantial functionality, no Qt equivalent)

| Tk popup | Backend | Proposed home | Notes |
|---|---|---|---|
| `UnsupervisedGUI` | `mufasa/unsupervised/*` | Add-ons → Unsupervised analysis (new section, large form) | The unsupervised pipeline (UMAP, HDBSCAN, cluster visualization) is entirely Tk. Substantial port. |
| `BlobQuickChecker` | `mufasa/ui/blob_quick_check_interface.py` | Add-ons → Blob quick-check (new section) | Standalone Tk interface; backend already in place. |
| `BackgroundRemoverSingleVideoPopUp` + `BackgroundRemoverDirectoryPopUp` | `video_bg_substraction` | Video Processing → Background removal (new section) | Two popups → one form with a single-vs-batch toggle. |
| `ROIDefinitionsCSVImporter` | already in `roi_tools` | ROI → Import ROI CSV (new section) | Small form. |
| `ROISizeStandardizer` | already in `roi_tools` | ROI → Size standardiser (new section) | Small form. |
| `PlotAnnotatedBouts` | already in `plotting` | Annotation → Annotated bouts → videos (new section) | Or route through `VisualizationForm`. |

### Medium-priority gaps (admin/diagnostic popups)

| Tk popup | Proposed home | Notes |
|---|---|---|
| `CheckVideoSeekablePopUp` | Video Processing → Utilities | One-button popup; folds into existing Utilities. |
| `ClfByROI`, `ClfByTimebins` | Classifier → new section "Classifier × ROI / time bins" | Small forms; could be a single section with mode selector. |
| Any `clf_descriptive_statistics_pop_up` not covered | Confirm `AnalysisForm` covers it | Likely already routed; verify only. |

### Likely-already-covered gaps (need verification, not new work)

The following 30+ Tk popups appear "unmapped" in the AST audit but are almost certainly routed via `VisualizationForm`'s 29-route table or `AnalysisForm`'s dispatcher:

```
GanttCreator                          → VisualizationForm
PlotSklearnResults                    → VisualizationForm
TresholdPlotCreator                   → VisualizationForm
DistancePlotter                       → VisualizationForm
DataPlotter                           → VisualizationForm
DirectingOtherAnimalsVisualizer       → VisualizationForm
DirectingAnimalsToBodyPartVisualizer  → VisualizationForm
EzPathPlot                            → VisualizationForm
PlotAnnotatedBouts                    → VisualizationForm (or new Annotation section)
ClassifierValidationClips             → ValidateClassifierForm
MovementCalculator                    → AnalysisForm
DistanceCalculator                    → AnalysisForm
DistanceTimeBinCalculator             → AnalysisForm
TimeBinsClfCalculator                 → AnalysisForm
DirectingOtherAnimalsAnalyzer         → AnalysisForm
DirectingAnimalsToBodyPartAnalyzer    → AnalysisForm
FSTTCCalculator                       → AnalysisForm
BooleanConditionalCalculator          → AnalysisForm
AggregateClfCalculator                → AnalysisForm
PlotMakePath                          → VisualizationForm
BlobVisualizer                        → VisualizationForm
... and others
```

**Action:** verify-pass against `VisualizationForm.ROUTES` and `AnalysisForm`'s dispatcher table. Add any missing routes. Then retire the Tk popups.

---

## 4. Consolidation principles

When porting a Tk popup, choose between three patterns:

1. **Add a new section** to an existing page when the popup is a distinct workflow (e.g., "Background removal" is a distinct task from "Filters").

2. **Add a route** to an existing dispatcher form (`VisualizationForm`, `AnalysisForm`, `ConverterForm`) when the popup is a parameter variant of an existing capability (e.g., adding "PlotAnnotatedBouts" as a new route in `VisualizationForm`).

3. **Add a mode selector to an existing section** when the Tk popups differ only by single vs. multiple-file scope (e.g., `DownsampleSingleVideo` + `DownsampleMultipleVideos` → one section with a "Mode" toggle: Single / Batch).

**Reject:** creating one Qt section per Tk popup. The Qt workbench should have fewer, broader sections — exactly what the user asked for.

**Soft cap:** ~8 sections per page. Already breached by Video Processing (14); that's acceptable because video work is genuinely diverse. Other pages should stay at or below the cap.

---

## 5. Drop candidates

Tk popups that should be **deleted, not ported**:

| Tk popup | Reason |
|---|---|
| `LoadProjectPopUp`, `SimbaProjectPopUp` | Qt Projects page covers it. |
| `PrintModelInfoPopUp` | Qt `ClassifierManageForm` covers it (or should). |
| `AboutSimBAPopUp` | Move content to a Qt About dialog or skip; it's vestigial branding. |
| `RestartSimBAPopUp` (if present) | The Qt app doesn't need a self-restart popup. |
| `BackgroundRemoverDirectoryPopUp` | Folds into a single `BackgroundRemover` form with a "Batch" mode toggle (and `BackgroundRemoverSingleVideo` is the other mode). |
| `InteractiveClahePopUp` | Already merged with `CLAHEPopUp` per the existing Qt design. |
| `MultiCropPopUp` | Already covered by existing `CropVideosForm`. |
| Helper pop-ups like `pop_ups/helpers.py` | UNREFERENCED in the Tk surface audit; just delete. |

---

## 6. Migration order

Recommended order of patches:

### Tier 1 — Verification (no new code, AST audit only)
1. **Confirm `VisualizationForm.ROUTES` covers the ~20 plot popups** in §3's "likely-already-covered" list. For each missing route, add it.
2. **Confirm `AnalysisForm`'s dispatcher covers the ~9 analysis popups.** Add missing routes.
3. **Confirm `ConverterForm` covers the ~15 data-conversion popups.** Add missing routes.

After Tier 1, the "unmapped" pool drops from 30 to roughly 7 genuinely missing forms.

### Tier 2 — Add new Qt sections (small forms, 1-3 hours each)
4. **Video Processing → Background removal** (consolidates 2 Tk popups)
5. **ROI → Import ROI CSV** (1 Tk popup)
6. **ROI → Size standardiser** (1 Tk popup)
7. **Annotation → Annotated bouts → videos** (1 Tk popup — or route through `VisualizationForm`)
8. **Add-ons → Blob quick-check** (1 Tk popup)
9. **Classifier → Classifier × ROI / time bins** (2 Tk popups → 1 section with mode selector)

### Tier 3 — Substantial new Qt work
10. **Add-ons → Unsupervised analysis** (entirely new Qt port of `UnsupervisedGUI` + cluster visualization). Large.

### Tier 4 — Drop + cleanup
11. **Delete Tk popups with confirmed Qt coverage** (after Tiers 1-3 finish). Mechanical bulk delete.
12. **Migrate 25 backend importers of `mufasa.ui.tkinter_functions`** (the §3 cross-coupling problem from `tk_surface_audit.md`). Each backend module needs its embedded Tk-popup invocations replaced with Qt-form launches or refactored to be UI-free.
13. **Drop `mufasa-tk` entry point + delete `mufasa/SimBA.py` + `mufasa/ui/` tree.**

---

## 7. Per-popup mapping appendix

Complete map of every Tk popup → its Qt destination. Generated post-122br; refresh periodically.

### Already covered by a Qt form sharing the same backend (25)

```
append_roi_features_animals_pop_up.py        → roi.py        (Features section)
append_roi_features_bodypart_pop_up.py       → roi.py
clf_annotation_counts_pop_up.py              → annotation.py (Reports section)
cue_light_clf_analyzer_popup.py              → addons.py     (Cue-light — clf stats)
cue_light_data_analyzer_popup.py             → addons.py     (Cue-light — data analysis)
cue_light_movement_analyzer_popup.py         → addons.py
cue_light_visualizer_popup.py                → addons.py
egocentric_alignment_pop_up.py               → pose_cleanup.py
import_roi_csv_popup.py                      → roi.py        (needs surfacing as section)
kleinberg_pop_up.py                          → addons.py
mutual_exclusivity_pop_up.py                 → addons.py
pup_retrieval_pop_up.py                      → addons.py
roi_aggregate_stats_popup.py                 → roi.py        (Analyze section)
roi_analysis_time_bins_pop_up.py             → roi.py        (Analyze section)
roi_features_plot_pop_up.py                  → roi.py        (Features section)
roi_size_standardizer_popup.py               → roi.py        (needs surfacing as section)
roi_tracking_plot_pop_up.py                  → roi.py
run_machine_models_popup.py                  → classifier.py (Run inference)
sleap_annotations_to_yolo_popup.py           → pose_tools.py (Convert pose data)
smoothing_popup.py                           → pose_cleanup.py
spontaneous_alternation_pop_up.py            → addons.py
subset_feature_extractor_pop_up.py           → features.py
third_party_annotator_appender_pop_up.py     → annotation.py
validation_plot_pop_up.py                    → validate_classifier.py
video_processing_pop_up.py                   → annotation.py (Targeted clips)
```

### Likely covered via dispatcher table (verify pass needed, ~30)

All plot popups → VisualizationForm. All analysis popups → AnalysisForm. All data conversion popups → ConverterForm. See §3 "Likely-already-covered gaps" for the list.

### Genuine gaps (new Qt work needed, ~7)

See §3 "High-priority gaps" + "Medium-priority gaps".

### Drop candidates (~8)

See §5.

---

## 8. Why this design

**Consolidation, not preservation.** The Tk launcher had ~85 popups because every backend got its own popup file. The Qt workbench takes the opposite stance: group related capabilities into shared forms with mode selectors, route similar plots through a single dispatcher, and add new functionality as sections within existing pages rather than as new top-level entries.

**Pages map to research stages.** Projects → Data Import → Preprocessing → ROI → Features → Annotation → Classifier → Visualizations → Analysis → Add-ons → Tools. A user can flow top-to-bottom through a typical workflow. The Tk launcher offered the same capabilities scattered across menus organized by file type and tool name; the Qt layout reorganizes by user goal.

**Dispatcher forms scale.** `VisualizationForm` has 29 routes today; it could grow to 50 without breaking the user model — pick a plot type, fill in parameters, run. Adding a 30th plotting backend is 5 lines of route declaration, not a new top-level page. Same for `AnalysisForm` and `ConverterForm`.

**Per-page caps are aspirational, not hard.** Video Processing has 14 sections; that's accepted because video work is genuinely diverse and grouping further would obscure capabilities. Pages with fewer sections (Projects, Features, Analysis) can grow without restructuring.

**Drop, don't port.** Several Tk popups exist only because the Tk launcher needed a popup for everything (`LoadProjectPopUp`, `AboutSimBAPopUp`, the various restart/info utilities). Qt has better idioms for these (the Projects page, an About dialog in the menu bar, log views in the main window). Don't recreate the Tk noise in Qt.

---

## 9. Caveats and unknowns

- **Verification is required.** This doc is an audit-based proposal, not a runtime test. The "likely covered via dispatcher" claim in §3 is based on backend class names; the actual `VisualizationForm.ROUTES` table needs an explicit scan. Tier 1 of the migration order exists exactly to do that scan.
- **`UnsupervisedGUI` is large.** The Tk file is 200+ lines and chains multiple sub-popups (UMAP setup, HDBSCAN setup, cluster validation, video clip generation, etc.). A faithful Qt port is many sections and probably its own subsection within Add-ons.
- **Backend module embedded popups.** Per `tk_surface_audit.md`, 25 backend modules (mixins, labelling, video_processors, etc.) import from `tkinter_functions.py` and embed Tk popup launches. These are independent of the SimBA.py menu surface but block Tk removal. The migration order in §6 covers them as Tier 4 item 12.
- **No visual verification possible.** Qt isn't runnable in the design-doc environment. Final design choices (button placement, label wording, defaults) should be tested by someone running the workbench. This doc covers structure; visual polish is downstream.
- **Per-popup file inventory may drift.** A future patch that adds or removes Tk popups would shift the count. The numbers in §1 are post-122br; refresh by re-running the AST audit (recipe in §7 of `tk_surface_audit.md`).
- **"Drop candidates" need user signoff.** §5 lists popups that are arguably obsolete; an actual researcher may have a use case that argues for keeping (or porting) one of them. Don't auto-delete from this list without a sanity check.
- **The 14-section Video Processing page is at the edge of usability.** If it grows much further, splitting into "Video Processing — basic" and "Video Processing — advanced" may be needed. Not in scope here.
