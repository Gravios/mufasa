# Mufasa Data Source Guides


> **See also:** `docs/workflows.md` (per-workflow technical audit, status-tagged),
> `docs/README.md` (doc index). All three companion docs were generated together
> post-patch 122bj; see commit log for the AST audit methodology.
**Purpose:** Per-source instructions for getting data INTO mufasa. Each section covers file format expectations, importer choice, body-part schema requirements, and the suggested next stage.

For end-to-end recipes (data source → analysis output), see `WORKFLOW_RECIPES.md`.

---

## DeepLabCut (DLC)

### DLC single-animal — CSV

**Importer:** `DLCSingleAnimalCSVImporter` (in `mufasa.pose_importers.dlc_csv_importer`)
**Form:** `PoseImportForm` → format = "DLC CSV"

**File format:** standard DLC CSV with 3 header rows:
1. `scorer` — model identifier
2. `bodyparts` — body-part names
3. `coords` — `x`, `y`, `likelihood` for each

**Body-part schema:** any. Imported as-is; the project's pose layout in `project.toml` records the names.

**Recommended:**
- If body-part count matches a specialty extractor (4 / 7 / 8 / 9), you can use that downstream.
- If it doesn't, use `UserDefinedFeatureExtractor` at the feature-extraction stage.

**Next stage:** preprocessing → outlier correction → feature extraction.

### DLC single-animal — H5

**Importer:** `DLCSingleAnimalH5Importer` (in `mufasa.pose_importers.dlc_h5_importer`)
**Form:** `PoseImportForm` → format = "DLC H5"

**File format:** DLC `.h5` produced via `deeplabcut.analyze_videos(..., save_as_csv=False)`.

**Advantage over CSV:** smaller, faster to read, no float-precision issues. Preferred when both are available.

**Body-part schema and next stage:** identical to DLC CSV.

### maDLC (multi-animal DLC)

**Importer:** `MADLCImporterH5` (in `mufasa.pose_importers.madlc_importer`)
**Form:** `PoseImportForm` → format = "maDLC H5"

**File format:** multi-animal DLC `.h5`. Each tracklet has an `individual` index alongside body-part and coord.

**Requires animal mapping:** the importer needs to know which maDLC `individual` index maps to which mufasa animal name. Form prompts for this.

**Body-part schema:** number-of-animals × body-parts-per-animal. The specialty extractors are sized for:
- 2 × 4 → `ExtractFeaturesFrom8bps2Animals`
- 2 × 7 → `ExtractFeaturesFrom14bps`
- 2 × 8 → `ExtractFeaturesFrom16bps`

If your maDLC project doesn't match, use `UserDefinedFeatureExtractor`.

**Common pitfall:** maDLC sometimes produces "swapped identity" tracklets where animal labels flip mid-video. Run heuristic post-processing (e.g., velocity-continuity check) before featurization, or use SLEAP's tracker output.

---

## SLEAP

### SLEAP — CSV

**Importer:** `SLEAPImporterCSV` (in `mufasa.pose_importers.sleap_csv_importer`)
**Form:** `PoseImportForm` → format = "SLEAP CSV"

**File format:** SLEAP analysis CSV. SLEAP produces this via `sleap-convert --format analysis.csv`.

**Body-part schema:** flexible. SLEAP supports both single-animal and multi-animal; the CSV format encodes this.

### SLEAP — H5

**Importer:** `SLEAPImporterH5` (in `mufasa.pose_importers.sleap_h5_importer`)
**Form:** `PoseImportForm` → format = "SLEAP H5"

**File format:** SLEAP `.h5` from `sleap-convert --format analysis.h5`. Includes tracking metadata SLEAP CSV drops.

**Preferred over CSV** for multi-animal SLEAP data.

### SLEAP — `.slp` project file

**Importer:** `SLEAPImporterSLP` (in `mufasa.pose_importers.sleap_slp_importer`)

**File format:** SLEAP project file (the working file SLEAP saves). Contains everything: video reference, skeleton, labels, predictions.

**When to use:** if you have the SLEAP project and want to bring across the full provenance. Heavier than the CSV/H5 routes.

---

## YOLO pose estimation

### Pre-trained YOLO results

**Importer:** `SimBAYoloImporter` (in `mufasa.pose_importers.simba_yolo_importer`)
**Form:** `PoseImportForm` → format = "YOLO"

**File format:** YOLO pose output. Mufasa expects coordinate files produced by `YOLOPoseInference` or `YOLOPoseTrackInference` (from `mufasa.model`).

**Workflow loop:** unique to YOLO — you typically *train YOLO inside mufasa* and feed the output back as pose data.

1. Annotate keypoints (Tools page → SLEAP→YOLO conversion if migrating from SLEAP).
2. `FitYolo` on the labeled set.
3. `YOLOPoseInference` (or `YOLOPoseTrackInference`) on new videos.
4. Import the result via `SimBAYoloImporter` → feeds back into the standard mufasa pipeline.

**GPU acceleration:** `YoloNVDECInference` uses NVIDIA NVDEC + TensorRT for ~5-10× speedup on long videos. Requires NVIDIA hardware + appropriate runtime.

### Blob tracking (no keypoints)

**Importer:** `SimBABlobImporter` (in `mufasa.pose_importers.simba_blob_importer`)
**Form:** Add-ons → "Blob tracker — initialise" (`BlobTrackerInitLauncher`)

**File format:** centroid + bounding-box CSV produced by `BlobTrackingExecutor`.

**When to use:** behavior scoring that doesn't need keypoint detail — e.g., simple location-based behaviors in low-resolution videos where DLC/SLEAP would be unreliable.

**Limitations:**
- No per-body-part features. Most specialty extractors won't apply.
- Use `FeatureSubsetsCalculator` for centroid-based features (velocity, ROI dwell, etc.) or `UserDefinedFeatureExtractor` with a minimal single-point "body-part."

---

## Specialty importers (specific research contexts)

### MARS — two-animal social

**Importer:** `MarsImporter` (in `mufasa.pose_importers.import_mars`)

**File format:** MARS JSON output (Caltech mouse social analysis pipeline).

**Animal count:** fixed 2. Body-parts mapped to MARS's standard schema.

**When to use:** if you're continuing a MARS-based pipeline. For new projects, prefer DLC/SLEAP — MARS is an older tool.

### TRK — Animal Part Tracker

**Importer:** `TRKImporter` (in `mufasa.pose_importers.trk_importer`)

**File format:** TRK (Animal Part Tracker `.trk` files).

**When to use:** legacy TRK data only. New projects should use DLC, SLEAP, or YOLO.

### FaceMap — face keypoint tracking

**Importer:** `FaceMapImporter` (in `mufasa.pose_importers.facemap_h5_importer`)

**File format:** FaceMap H5 output.

**When to use:** facial behavior / expression analysis. Body-parts are face landmarks (eye, mouth, etc.), not full-body.

**Compatible extractors:** `UserDefinedFeatureExtractor` only. Specialty extractors assume body-pose schema.

### SuperAnimal-TopView

**Importer:** `SuperAnimalTopViewImporter` (in `mufasa.pose_importers.superanimal_import`)

**File format:** DLC-format CSV/H5 from the SuperAnimal-TopView pretrained model (top-down mouse).

**When to use:** no manual labeling required — SuperAnimal is pretrained. Run it via DLC's modelzoo and import the result.

**Recommended next stage:** outlier correction is essential — pretrained model performance is variable.

---

## Third-party annotation imports (NOT pose)

If you already have **behavioral annotations** (frame-by-frame labels, not pose), use the Annotation page → "Third-party annotation import":

| Source tool | Importer |
|---|---|
| BORIS | `BorisAppender` |
| BENTO | `BentoAppender` |
| DeepEthogram | `DeepEthogramImporter` |
| Noldus EthoVision | `ImportEthovision` |
| Noldus Observer | `NoldusObserverImporter` |
| Solomon Coder | `SolomonImporter` |
| Generic Excel (single classifier) | `SingleClfAppenderExcel` |
| Mitra-lab format | `MitraStyleAnnotationAppender` |

**Workflow note:** these write directly to `derived/labels/<video>.parquet`. Skip the Frame Labelling form — you already have labels. Proceed straight to classifier training.

---

## Decision matrix

| Your data | Recommended import path |
|---|---|
| Single mouse, top-down, DLC-tracked | DLC CSV/H5 → 8bp or 9bp extractor |
| Single mouse, side-view, DLC | DLC CSV/H5 → 4bp or 7bp extractor |
| Two mice, social interaction, DLC | maDLC H5 → 14bp / 16bp extractor |
| Multi-animal, anything > 2 | SLEAP H5 → User-defined extractor |
| Pretrained model, no labels | SuperAnimal-TopView → User-defined |
| Insect / non-mammal | DLC/SLEAP → User-defined extractor |
| Face only | FaceMap H5 → User-defined extractor |
| No keypoints, just location | SimBABlob → FeatureSubsetsCalculator |
| Already have BORIS / EthoVision labels | Skip pose import, use Third-party appender |
| GPU available, long videos | YOLO + `YoloNVDECInference` |

---

## Project file format choice — v1 TOML vs legacy INI

When you create a project (Projects → Create or open project), you choose the project format:

- **v1 TOML** (`project.toml`): recommended for all new projects. Cleaner schema, per-classifier sections under `[classifier_inference.<name>]`, designed for round-tripping. Data tree under `derived/`.
- **Legacy INI** (`project_config.ini`): SimBA-compatible. Use only if you need to share with an external SimBA workflow. Data tree under `csv/features_extracted/`, `csv/targets_inserted/`, `csv/machine_results/`.

The form layer dispatches both formats transparently (via `mufasa.project_layout.write_classifier_inference_settings` / `write_classifier_training_settings`). All importers, extractors, training, and inference work on either. **Visualization and analysis output structure differs** between the two — legacy INI projects don't have a `derived/` tree.

Conversion: not automated. To migrate a legacy project to v1, the cleanest approach is to create a new TOML project pointing at the same videos and re-run import + feature extraction.
