# Mufasa Workflow Audit


> **See also:** `docs/workflows.md` (per-workflow technical audit, status-tagged),
> `docs/README.md` (doc index). All three companion docs were generated together
> post-patch 122bj; see commit log for the AST audit methodology.
**Generated:** Friday, May 16, 2026 (post-122bj)
**Scope:** AST inventory of all workflow entry points, data-flow stages, and backend classes across `mufasa/` after the v1 migration arc closed.
**Method:** AST walk of every `*.py` in `mufasa/`, cross-referenced with the Qt workbench page registrations.

---

## 1. Workbench topology

The Qt workbench (`mufasa.ui_qt.workbench`) is built from 14 pages, each registering one or more form classes via `page.add_section(title, [(FormClass, kwargs)])`. The page-load order in `mufasa.ui_qt.workbench.build_app()` defines the suggested top-to-bottom user flow.

| # | Page | Sections | Stage |
|---|---|---|---|
| 1 | **Projects** | Create or open project; Project information | Setup |
| 2 | **Data Import** | Import Pose Data; Import video | Ingest |
| 3 | **Video Processing** | 14 sections (format/trim/crop/resize/rotate/filters/overlay/extract/merge/import/join/image/utilities/audit) | Preprocessing (video) |
| 4 | **Preprocessing** | Preprocess Videos; Video Calibration; Interpolate; Kalman v2; Outlier (run / skip); Egocentric alignment; Advanced/legacy | Preprocessing (pose) |
| 5 | **ROI** | Definitions; Analyze; Visualize; Features | Spatial config |
| 6 | **Features** | Compute feature subsets | Featurization (non-ML) |
| 7 | **Annotation** | Frame labelling; Targeted clips; Third-party import; Review predictions; Reports | Labelling |
| 8 | **Classifier** | Manage; Train; Run inference; Validate | Modeling |
| 9 | **Visualizations** | Create visualisation (29 routes) | Output |
| 10 | **Analysis** | Run analysis | Output |
| 11 | **Add-ons** | Cue-light (4); Kleinberg; Mutual exclusivity; Pup retrieval; Spontaneous alternation; Blob tracker init | Special workflows |
| 12 | **Tools** | Convert pose data; Re-order keypoints; SLEAP→YOLO; Export CSV | Utility |

The Visualizations page is special: it has one section (`Create visualisation`) wrapping a dispatcher (`VisualizationForm`) that exposes **29 routes** through a route table — not 29 sections in the UI. The route dispatcher uses 10 extras kinds (`int`, `float`, `bool`, `choice`, `color`, `str`, `list`, `dict`, `pickle`, `file`) to render per-backend parameter inputs.

---

## 2. Canonical pipeline

Below is the canonical end-to-end data flow. Each stage names its input/output artifacts under the v1 project layout (TOML projects, current as of patch 122bj).

```
                  ┌──────────────────────┐
                  │ 0. Project setup     │   creates project.toml
                  │   NewProjectForm     │   creates derived/, sources/, logs/
                  └──────────┬───────────┘
                             │
       ┌─────────────────────┴────────────────────┐
       │                                          │
       v                                          v
┌──────────────┐                       ┌──────────────────────┐
│ 1a. Import   │                       │ 1b. Import video     │
│  Pose Data   │                       │   VideoImportForm    │
│ PoseImport-  │                       │                      │
│   Form       │                       │   sources/videos/    │
│              │                       │   (v1)               │
│ → sources/   │                       │   videos/ (legacy)   │
│   pose/      │                       └──────────────────────┘
└──────┬───────┘
       │
       v
┌──────────────────────────┐
│ 2. Preprocessing (pose)  │
│   InterpolateForm        │
│   KalmanV2SmoothingForm  │
│   RunOutlierCorrection-  │   logs/ contains outlier reports
│     Form (or Skip)       │   sources/pose/*.parquet updated in-place
│   EgocentricAlignment-   │
│     Form                 │
└──────┬───────────────────┘
       │ (optional: 2'. ROI definitions; required for ROI-feature workflows)
       │
       v
┌──────────────────────────┐
│ 3. Feature extraction    │   derived/features/<video>.parquet (wide)
│   FeatureSubsetExtractor │   derived/features/<family>/<video>.parquet
│   <or specialty class>   │   (per-family for non-ML use)
└──────┬───────────────────┘
       │
       v
┌──────────────────────────┐
│ 4. Annotation            │   derived/labels/<video>.parquet
│   FrameLabellingLauncher │   (one parquet per video, columns =
│   (or Targeted clips     │    classifier names)
│    or Third-party        │
│    appender)             │
└──────┬───────────────────┘
       │
       v
┌──────────────────────────┐
│ 5. Classifier training   │   models/<classifier>.sav
│   TrainClassifierForm    │   logs/<classifier>_meta.json
│   → TrainRandomForest-   │
│     Classifier           │
└──────┬───────────────────┘
       │
       v
┌──────────────────────────┐
│ 6. Inference             │   derived/classifications/<video>.parquet
│   RunInferenceForm       │   (columns: Probability_<T>, <T>)
│   → InferenceBatch (or   │
│     InferenceMulticlass) │
└──────┬───────────────────┘
       │
       ├──> ┌─────────────────────────┐
       │    │ 7a. Analysis            │  logs/<various>_<stamp>.csv
       │    │   AnalysisForm          │
       │    └─────────────────────────┘
       │
       ├──> ┌─────────────────────────┐
       │    │ 7b. Visualization       │  derived/visualizations/<stamp>/
       │    │   VisualizationForm     │
       │    │   (29 routes)           │
       │    └─────────────────────────┘
       │
       └──> ┌─────────────────────────┐
            │ 7c. Add-on workflows    │  varies (cue light, spontaneous
            │   (cue light, kleinberg,│  alternation, mutual exclusivity,
            │    pup retrieval, ...)  │  etc.)
            └─────────────────────────┘
```

Notable structural facts:
- **v1 derived/ tree** is the single source of truth for downstream data. The legacy `csv/features_extracted/`, `csv/targets_inserted/`, `csv/machine_results/` paths were retired in the 122ax/122bb close-out.
- **Per-video parquet, not aggregated CSVs.** Every stage writes one parquet per video, indexed by frame number. This makes per-video re-runs cheap (no need to re-read the whole project's data).
- **Two feature trees coexist.** `derived/features/<video>.parquet` is wide (all families merged, for ML training/inference). `derived/features/<family>/<video>.parquet` per-family is for non-ML analytical use via `FeatureSubsetsCalculator`.

---

## 3. Data source inventory (12 pose importers)

| Importer class | Module | Format | Animals | Notes |
|---|---|---|---|---|
| `DLCSingleAnimalCSVImporter` | `dlc_csv_importer` | DLC CSV | 1 | Single animal, named body-parts |
| `DLCSingleAnimalH5Importer` | `dlc_h5_importer` | DLC H5 | 1 | Standard DLC HDF5 export |
| `MADLCImporterH5` | `madlc_importer` | maDLC H5 | 2+ | Multi-animal DLC; needs animal-name mapping |
| `SLEAPImporterCSV` | `sleap_csv_importer` | SLEAP CSV | 1+ | SLEAP analysis export |
| `SLEAPImporterH5` | `sleap_h5_importer` | SLEAP H5 | 1+ | SLEAP H5 native |
| `SLEAPImporterSLP` | `sleap_slp_importer` | SLEAP `.slp` | 1+ | SLEAP project file |
| `MarsImporter` | `import_mars` | MARS JSON | 2 | Mouse social analysis (Caltech) |
| `TRKImporter` | `trk_importer` | TRK | varies | Animal Part Tracker |
| `FaceMapImporter` | `facemap_h5_importer` | FaceMap H5 | 1 | Face-keypoint tracking |
| `SimBAYoloImporter` | `simba_yolo_importer` | YOLO results | 1+ | YOLO pose, optional interpolation |
| `SimBABlobImporter` | `simba_blob_importer` | blob CSV | 1 | Centroid + bbox (no keypoints) |
| `SuperAnimalTopViewImporter` | `superanimal_import` | DLC-format | 1 | SuperAnimal-TopView pretrained mouse model |

`DLCAutodetectError` is raised when an importer can't parse body-part names — typically signals a malformed CSV header.

---

## 4. Feature extractors

### 4a. Specialty extractors (hardcoded body-part / animal config)

These have fixed schemas — they only work on pose data matching their exact body-part naming.

| Class | Animals × Body-parts | Project type |
|---|---|---|
| `ExtractFeaturesFrom4bps` | 1 × 4 | Generic single-animal (open field, etc.) |
| `ExtractFeaturesFrom7bps` | 1 × 7 | Single-animal extended |
| `ExtractFeaturesFrom8bps` | 1 × 8 | Single-animal full skeleton |
| `ExtractFeaturesFrom9bps` | 1 × 9 | Single-animal w/ tail tip |
| `ExtractFeaturesFrom14bps` | 2 × 7 | Two-animal social |
| `ExtractFeaturesFrom16bps` | 2 × 8 | Two-animal extended |
| `ExtractFeaturesFrom8bps2Animals` | 2 × 4 | Two-animal minimal |

### 4b. Project-purpose specialty extractors

These produce features tailored to a specific research project. They are NOT interchangeable — each assumes the body-part schema of the original study.

| Class | Purpose |
|---|---|
| `AgressionFeatureExtractor` | Two-animal aggression scoring |
| `AmberFeatureExtractor` | AMBER (autism mouse) pipeline |
| `BoundaryRearingFeaturizer` | Rearing detection (walls + perimeter) |
| `CaveFishFeaturizer` | Mexican cave fish anxiety |
| `GerbilFeaturizer` | Gerbil social via SLEAP NPY |
| `MitraFeatureExtractor` | Grooming + rearing (Mitra lab) |
| `RatSocialFeaturizer` | Rat social behavior |
| `RiptortusFeaturizer` | Riptortus pedestris feeding (insect) |
| `StraubTailAnalyzer` | Tail behavior (egocentric video) |
| `WingWaveFeatureExtractor` | Insect wing-wave behavior |

### 4c. Generic extractors

- `UserDefinedFeatureExtractor` — works on any body-part schema by reading the project's pose layout. **The default choice for novel body-part configurations.**
- `FeatureSubsetsCalculator` — computes individual feature families (distances, angles, hull, velocity, etc.) **without** producing the wide ML-ready frame. Use for analytical / publication-figure work where you want specific feature columns and skip the rest.

---

## 5. Model / training / inference

### Training

| Class | Use case |
|---|---|
| `TrainRandomForestClassifier` | Single binary classifier from one meta-config |
| `GridSearchRandomForestClassifier` | Grid-search multiple meta-configs |
| `TrainMultiClassRandomForestClassifier` | Multi-class (one-hot targets) |
| `TrainMultiLabelRandomForestClassifier` | Multi-label (independent binary heads) |
| `GridSearchMulticlassRandomForestClassifier` | Grid-search × multi-class |
| `OrdinalClassifier` | Ordinal targets (multiple binary heads sharing structure) |

### Inference

| Class | Use case |
|---|---|
| `InferenceBatch` | Binary classifier(s) on every video in project |
| `InferenceMulticlassBatch` | Multi-class inference |
| `InferenceValidation` | Single classifier × single video (for validation video output) |
| `SamInference` | SAM (Segment Anything Model) inference |

### YOLO sub-pipeline

Distinct from the Random-Forest pipeline. YOLO classes handle detection/pose/segmentation:

- `FitYolo` (training)
- `YoloInference` / `YoloNVDECInference` (detection, plus NVDEC accelerated variant)
- `YOLOPoseInference` / `YOLOPoseTrackInference` (pose + tracking)
- `YOLOSegmentationInference` (segmentation)

---

## 6. Data processors (analysis stage)

Post-inference aggregators. Each consumes `derived/classifications/<video>.parquet` and/or pose data and produces a per-project summary CSV under `logs/`.

| Class | Output |
|---|---|
| `AggregateClfCalculator` | Bout counts, durations, intervals by classifier |
| `BooleanConditionalCalculator` | Time-/frame-counts under per-frame boolean conditions |
| `MovementCalculator` | Distance, velocity, time-immobile per animal |
| `TimeBinsClfCalculator` | Classification stats binned by time |
| `TimeBinsMovementCalculator` | Movement stats binned by time |
| `DistanceCalculator`, `DistanceTimeBinCalculator` | Body-part-pair distances |
| `DirectingOtherAnimalsAnalyzer`, `DirectingAnimalsToBodyPartAnalyzer` | Gaze/heading direction analysis |
| `SeverityCalculator`, `SeverityBoutCalculator`, `SeverityFrameCalculator` | Movement-weighted severity scores |
| `FSTTCCalculator` | Forward Spike-Time Tiling Coefficient (sequence statistic) |
| `KleinbergCalculator` | Burst smoothing of classifier output |
| `MutualExclusivityCorrector` | Post-hoc exclusivity rules between classifiers |
| `LightDarkBoxAnalyzer` | Light/dark-box paradigm-specific |
| `PupRetrieverCalculator` | Pup-retrieval task (Winters 2022) |
| `SpontaneousAlternationCalculator` | Y-maze alternation |
| `CueLightAnalyzer` + 2 follow-ups | Cue-light experiments |
| `FreezingDetector`, `CirclingDetector` | Heuristic detectors (no classifier needed) |
| `GibbSampler` | Motif discovery in categorical sequences |

---

## 7. ROI tooling

ROIs (Regions of Interest) are stored at `logs/measures/ROI_definitions.h5`. They power three feature flows:

1. **ROI features** added to wide feature frame via `ROIFeatureCreator` — used during feature extraction.
2. **ROI analysis** via `ROIAnalyzer` / `ROIClfCalculator` / `ROITimebinCalculator` — produces aggregate stats.
3. **ROI visualization** via `ROIPlotter` / `ROIfeatureVisualizer` — overlay videos.

ROI shape classes:
- `ROISelector` (rectangle), `ROISelectorCircle` (circle), `ROISelectorPolygon` (polygon)
- `InteractiveROIModifier` / `InteractiveROIBufferer` — post-hoc shape editing
- `ROIRuler` — distance measurements
- `ROISizeStandardizer` — normalize sizes against a reference video

`ROIDefinitionsCSVImporter` lets users round-trip ROI definitions through CSV (useful for sharing or batch-editing ROI sets).

---

## 8. Outlier correction

Three forms feed three backends:

| Form | Backend | Strategy |
|---|---|---|
| `RunOutlierCorrectionForm` | `OutlierCorrecterLocation` + `OutlierCorrecterMovement` | Heuristic thresholds on location (deviation from body center) and movement (frame-to-frame Euclidean) |
| `SkipOutlierCorrectionForm` | `OutlierCorrectionSkipper` | Marks data as processed without modification |
| `Advanced / legacy` section | `OutlierCorrecterLocationAdvanced` + `OutlierCorrecterMovementAdvanced` | Per-animal and per-body-part thresholds |

Kalman v2 (`KalmanV2SmoothingForm`) is a separate smoothing path with a richer state model (`StateLayout`, `BodyLayout`, `FittedLengths` types) and EM-fitted noise parameters (`NoiseParamsV2`). Strongly recommended over `Smooth` (the legacy moving-average smoother) for downstream feature quality.

---

## 9. v1 TOML project layout

A v1 project is a TOML file (`project.toml`) plus a directory tree:

```
<project_root>/
├── project.toml                 # canonical config
├── sources/
│   ├── pose/                    # imported pose data (parquet per video)
│   └── videos/                  # imported video files (or symlinks)
├── derived/
│   ├── features/                # wide features per video + per-family subtrees
│   ├── labels/                  # annotations per video (parquet)
│   └── classifications/         # inference output per video (parquet)
├── logs/
│   ├── measures/
│   │   └── ROI_definitions.h5
│   └── <stage>_<timestamp>.csv  # analysis outputs
└── models/                      # trained classifiers (.sav + meta.json)
```

The 11 keys returned by `project_paths_from_config(toml_path)`:
- `project_root` — absolute path
- `input_pose_dir` — `sources/pose/`
- `video_dir` — `sources/videos/`
- `video_info_path` — `logs/video_info.csv`
- `derived_features_dir` — `derived/features/`
- `derived_labels_dir` — `derived/labels/`
- `derived_classifications_dir` — `derived/classifications/`
- `machine_results_dir` — legacy alias (only present on legacy INI projects)
- `models_dir` — `models/`
- `logs_dir` — `logs/`
- `roi_definitions_path` — `logs/measures/ROI_definitions.h5`

### v1 TOML sections recognized:

- `[classifier_inference.<classifier>]` — per-classifier inference params (`model_path`, `threshold`, `min_bout_ms`)
- `[classifier_training]` — training params (25 canonical keys: `model_to_run`, `rf_n_estimators`, sampling, evaluation, SHAP, …)

Legacy `.ini` projects expose the equivalent settings in `[SML settings]`, `[threshold_settings]`, `[Minimum_bout_lengths]`, `[create_ensemble_settings]`. The form layer uses `mufasa.project_layout.write_classifier_inference_settings` and `write_classifier_training_settings` to dispatch on project format.

---

## 10. Plotting backends (29 surfaced in form + 7 not)

29 routes are surfaced through `VisualizationForm`. 7 plotting classes are either internal helpers, validators, or scaffolding (e.g., `CircularPlotting` base class, `FrameMergererFFmpeg` ffmpeg wrapper, `ShapAggregateStatisticsCalculator` aggregator).

### Available kinds in the form

| Kind | Source | Use case |
|---|---|---|
| `int`, `float`, `bool` | base | scalar params |
| `choice` | base | enum dropdown |
| `color` | base | RGB triple |
| `str` | base | text |
| `list` | 122be | comma-separated → `list[str]` |
| `dict` | 122bg | JSON → `dict` |
| `pickle` | 122bh | file picker + `pickle.load` |
| `file` | base | file picker (returns path string) |

Every `mufasa.plotting` backend with user-facing parameters is reachable through the form. There are no remaining "callable but not surfaced" backends after patch 122bh.

---

## 11. Third-party annotation appenders

Mufasa can ingest annotations from other tools and write them as `derived/labels/<video>.parquet`:

| Importer | Source tool | Format |
|---|---|---|
| `BorisAppender` | BORIS | CSV |
| `BentoAppender` | BENTO | CSV |
| `DeepEthogramImporter` | DeepEthogram | CSV |
| `ImportEthovision` | Noldus EthoVision | XLSX |
| `NoldusObserverImporter` | Noldus Observer | XLSX |
| `SolomonImporter` | Solomon Coder | CSV |
| `MitraStyleAnnotationAppender` | Mitra lab format | XLSX |
| `SingleClfAppenderExcel` | Generic Excel | XLSX (single classifier) |
| `BorisSourceCleaner` | (helper) | Pre-process BORIS exports |

The unified entry point is `ThirdPartyLabelAppender` which dispatches by format.

---

## 12. Workflow shape — read this as a flowchart

The shortest accurate description of mufasa as a system:

> Pose-estimation data goes in. Per-video parquet flows through preprocessing → featurization → annotation → training → inference, producing per-video parquet at each stage in the v1 `derived/` tree. Analysis and visualization read from `derived/` and write to `logs/` or `derived/visualizations/`. The form layer dispatches user input to dataclass-driven route tables; backends are independently callable from Python.

Everything else is configuration, format variants, or specialty extractors for specific research contexts.
