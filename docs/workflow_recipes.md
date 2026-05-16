# Mufasa Workflow Recipes


> **See also:** `docs/workflows.md` (per-workflow technical audit, status-tagged),
> `docs/README.md` (doc index). All three companion docs were generated together
> post-patch 122bj; see commit log for the AST audit methodology.
**Purpose:** End-to-end recipes for common experimental setups. Each recipe maps data + research question → ordered list of mufasa pages/forms/backends.

For per-source import details, see `DATA_SOURCE_GUIDES.md`. For the topological view of pages and stages, see `WORKFLOW_AUDIT.md`.

---

## Recipe selector

Pick the row that best describes your experiment.

| If your experiment is… | Use recipe |
|---|---|
| Single mouse, open field, score a few behaviors | **R1. Single-animal binary classification** |
| Two mice, social interaction, score aggression/mounting/etc | **R2. Two-animal social classification** |
| Light/dark box, sucrose preference, etc. — ROI-bounded behavior | **R3. ROI-bounded behavior** |
| Cue lights drive expected behavior | **R4. Cue-light experiment** |
| Y-maze / T-maze | **R5. Spontaneous alternation** |
| Pup retrieval task | **R6. Pup retrieval** |
| Already have BORIS / EthoVision annotations, want to train classifier | **R7. Third-party annotation → classifier** |
| Unknown behaviors — exploratory clustering | **R8. Unsupervised discovery** |
| Detection (boxes) more than keypoints | **R9. YOLO-based pipeline** |
| Heuristic-only, no classifier | **R10. Heuristic-only (freezing / circling / movement)** |
| Already annotated, want to publish features for analysis | **R11. Feature subsets for non-ML use** |

---

## R1. Single-animal binary classification

**Scenario:** One mouse in an open field. You want to detect one or more discrete behaviors (e.g., grooming, rearing).

**Data:** DLC or SLEAP, single-animal, 4–9 body-parts.

### Steps

1. **Projects → Create or open project.** Choose v1 TOML. Set body-part count.
2. **Data Import → Import Pose Data.** Choose DLC CSV/H5 or SLEAP H5.
3. **Data Import → Import video.** Default symlinks (no copy needed).
4. **Preprocessing → Video Calibration.** Set frame rate + pixels-per-mm per video.
5. **Preprocessing → Interpolate missing frames.** Linear interpolation usually fine.
6. **Preprocessing → Kalman v2 smoothing.** Recommended over the legacy moving-average smoother. Use defaults.
7. **Preprocessing → Run outlier correction.** Defaults if data is clean; tune thresholds if pose is noisy.
8. **Features → Compute feature subsets** (NO — that's for non-ML use). Instead: invoke the matching specialty extractor via *Classifier → Train classifier* (the train flow runs feature extraction implicitly), or via API:
   - 4 body-parts → `ExtractFeaturesFrom4bps`
   - 7 → `ExtractFeaturesFrom7bps`
   - 8 → `ExtractFeaturesFrom8bps`
   - 9 → `ExtractFeaturesFrom9bps`
   - Other → `UserDefinedFeatureExtractor`
9. **Annotation → Frame labelling.** Annotate 10–20% of frames per video.
10. **Classifier → Train classifier.** Random Forest, 2000 estimators, balanced class weights. Save model to `models/`.
11. **Classifier → Validate classifier.** Run on one held-out video; inspect the validation video output.
12. **Classifier → Run inference.** Applies to all videos in the project; writes `derived/classifications/<video>.parquet`.
13. **Analysis → Run analysis.** Choose `AggregateClfCalculator` for bout counts + durations.
14. **Visualizations.** Recommended routes:
    - "Path plot (per video)" → spatial coverage check
    - "Heat map (classifier, single core)" → where behavior happened
    - "Sklearn results (single core)" → overlay video with behavior labels

### Pitfalls

- **Insufficient annotations.** RF needs ~5000+ labeled frames across multiple videos for binary classifiers. Less → biased decision boundary.
- **Class imbalance.** If target behavior is <5% of frames, enable under-sampling in `[classifier_training]` (set `under_sample_setting = "Random undersample"`, `under_sample_ratio = 1.0`).
- **Feature extractor mismatch.** Running 14bp extractor on 8bp data → silent feature corruption. Confirm body-part count before training.

---

## R2. Two-animal social classification

**Scenario:** Two animals, score social behaviors (attack, sniffing, mounting, etc.).

**Data:** maDLC H5 or SLEAP multi-animal H5. Each animal has 4 / 7 / 8 body-parts.

### Steps

1. **Projects** → create v1 TOML, 2 animals.
2. **Data Import → Import Pose Data.** maDLC H5 (or SLEAP H5). **Confirm animal identity mapping** in the importer dialog.
3. **Preprocessing**, same as R1.
4. **Feature extraction** — choose based on body-part count:
   - 2 × 4 → `ExtractFeaturesFrom8bps2Animals`
   - 2 × 7 → `ExtractFeaturesFrom14bps`
   - 2 × 8 → `ExtractFeaturesFrom16bps`
   - Other → `UserDefinedFeatureExtractor`
   - For attack/aggression specifically: `AgressionFeatureExtractor` produces tuned features.
5. **Annotation → Frame labelling.**
6. **Classifier → Train + Validate + Run inference.**
7. **Analysis:**
   - `AggregateClfCalculator` — basic stats
   - `DirectingOtherAnimalsAnalyzer` — gaze direction toward the other animal (rich feature for social analysis)
   - `DistanceCalculator` — inter-animal distance over time

### Identity-swap caveat

Pose trackers occasionally swap animal identities mid-video. Symptoms: classifier confidence drops, distance metrics jitter. Detection:
- `derived/classifications/<video>.parquet` shows mid-video probability cliff.
- Distance plot has unexplained spikes.

Mitigations:
- SLEAP's `--tracking-tracker flow` is more robust than maDLC's default tracker.
- For maDLC, the post-`stitchtracklets` output is more identity-stable than raw `analyze_videos` output.
- `Tools → Re-order pose keypoints` can re-label after the fact for short videos.

---

## R3. ROI-bounded behavior

**Scenario:** Behavior is only meaningful within specific spatial regions (e.g., "drinking from spout A" requires being near spout A's coordinates).

### Steps

1. R1 or R2 through step 7 (preprocessing complete).
2. **ROI → Definitions.** Draw rectangles/circles/polygons per video. Save.
3. **ROI → Analyze.** Computes time in each ROI per animal.
4. **ROI → Features** (`ROIFeaturesForm`). Adds ROI-relative features (in/out indicator, distance to ROI edge) to the wide feature frame.
5. Feature extraction now produces ROI-aware features. Continue with annotation + training as R1/R2.
6. **Analysis: `ROIClfCalculator`** — bouts per ROI × per classifier.

### Choice: per-video ROIs vs reference ROIs

- **Per-video ROIs** — draw on every video. Tedious but accurate when camera position shifts between recordings.
- **Reference ROIs + standardization** — draw once on a reference video, use `ROISizeStandardizer` to scale to other videos. Faster setup but assumes consistent framing.

### When NOT to use ROI features

If your behavior is purely about animal posture (e.g., grooming) and not where it happens, ROI features add noise. Stick with the standard featurizers.

---

## R4. Cue-light experiment

**Scenario:** Visual cue (LED) signals a stimulus; you want behavior around cue onsets.

### Steps

1. R1 through step 7.
2. **Add-ons → Cue-light — data analysis** (`CueLightDataForm`). Provide:
   - Video file or CSV of cue-light timestamps.
   - Cue-light names (now a list-kind field — type comma-separated names).
3. (Optional, if scoring behavior IN cue ON intervals): finish R1's classifier flow first.
4. **Add-ons → Cue-light — classifier statistics** (`CueLightClfForm`). Cross-tabulates classifier output against cue-light states.
5. **Add-ons → Cue-light — movement statistics** (`CueLightMovementForm`). Velocity / immobility around cue events.
6. **Add-ons → Cue-light — visualizer** (`CueLightVisualizerForm`) — overlay video showing cue state + animal trajectory.
7. **Visualizations → Cue light visualizer route** — alternative entry point with the same backend.

### Two cue-light sources

- **Video-detected cue state** — `CueLightAnalyzer` reads pixel intensity from a small ROI in the video frame.
- **External CSV cue state** — supply a CSV with frame-indexed ON/OFF columns.

The form picks based on which input you provide.

---

## R5. Spontaneous alternation (Y/T-maze)

### Steps

1. R1 through step 7 (preprocessing).
2. **ROI → Definitions.** Draw one ROI per maze arm + one "Center" ROI.
3. **Add-ons → Spontaneous alternation** (`SpontaneousAlternationForm`):
   - Arm ROI names (list-kind field after 122bi).
   - Center ROI name.
   - Animal area threshold (pixels).
   - Detection threshold.
   - Buffer frames around transitions.
4. Backend: `SpontaneousAlternationCalculator`. Writes per-video alternation count + classical alternation ratio.
5. **Visualizations → Spontaneous alternation plot** — overlay video showing arm visits and alternations.

No classifier training needed — it's a heuristic.

---

## R6. Pup retrieval

**Scenario:** Maternal pup-retrieval assay. Score retrievals (pup moved from periphery to nest).

### Steps

1. R1 through preprocessing.
2. **Add-ons → Pup retrieval** (`PupRetrievalForm`):
   - Supply pup positions per frame (separate pose track or manual CSV).
   - Specify dam body-part to track.
3. Backend: `PupRetrieverCalculator` (Winters 2022). Outputs retrieval events with frame indices.

Requires custom pose tracking for the pup. Often involves blob tracking the pup separately.

---

## R7. Third-party annotations → classifier

**Scenario:** You've already annotated behavior in BORIS / EthoVision / Solomon / etc. You want to train a classifier without redoing the annotation in mufasa.

### Steps

1. R1 through preprocessing + feature extraction.
2. **Annotation → Third-party annotation import** (`ThirdPartyAppenderForm`). Choose your source tool. Provide the annotation file(s).
3. Inspect `derived/labels/<video>.parquet` to confirm labels imported correctly.
4. **Classifier → Train + Validate + Run inference** (R1 steps 10–12).
5. Analysis + visualization as needed.

### Source-specific quirks

- **BORIS** exports vary by version. If import fails, run `BorisSourceCleaner` on the file first.
- **EthoVision** state-event vs point-event distinction: only state-events ("behavior X happened between frame A and B") map cleanly to mufasa's frame-binary labels.
- **DeepEthogram** outputs probability + binary; mufasa imports the binary.

---

## R8. Unsupervised discovery

**Scenario:** Don't know what behaviors are in your data; want to discover clusters.

### Steps

1. R1 through feature extraction.
2. **Annotation:** SKIP. Unsupervised doesn't need labels.
3. Use the unsupervised module (`mufasa.unsupervised`) — not currently surfaced through the standard form layer; use `UnsupervisedGUI` (Tk popup) or call backends programmatically:
   - `DatasetCreator` — aggregate raw features to bout-level / video-level vectors.
   - `UmapEmbedder` — UMAP grid search.
   - `HDBSCANClusterer` — HDBSCAN grid search.
   - `ClusterValidators` — internal cluster quality metrics.
   - `ClusterXAICalculator` — explainability (train surrogate RF on cluster labels).
   - `ClusterVideoVisualizer` — generates per-cluster example video clips.
4. **Result interpretation:** clusters are unlabeled. Use `ClusterVideoVisualizer` output to label them by inspection.

### Recommended grid

- UMAP: `n_neighbors ∈ {10, 30, 100}`, `min_dist ∈ {0.0, 0.1}`, `n_components = 2`.
- HDBSCAN: `min_cluster_size ∈ {50, 200, 500}`, `min_samples ∈ {10, 30}`.

Cross-validate cluster stability with `ClustererComparisonCalculator` before committing to a clustering.

---

## R9. YOLO-based pipeline

**Scenario:** Detection-heavy task (where is the animal?) more than pose-heavy. GPU available.

### Steps

1. **Tools → SLEAP → YOLO conversion** if you have SLEAP annotations; otherwise manually label bounding boxes.
2. `FitYolo` — train YOLO model from labels.
3. `YOLOPoseInference` (or `YOLOPoseTrackInference` for multi-frame tracking; or `YoloNVDECInference` for fast detection without pose).
4. **Data Import → Import Pose Data** with format = "YOLO". Imports the inference output.
5. From here, standard mufasa pipeline (preprocessing → features → annotation → classifier).

### YOLO variants

| Inference class | Output | Speed |
|---|---|---|
| `YoloInference` | Bounding boxes | Slow (per-frame) |
| `YoloNVDECInference` | Bounding boxes | Fast (NVDEC + TRT) |
| `YOLOPoseInference` | Keypoints | Slow |
| `YOLOPoseTrackInference` | Keypoints + identity | Slowest (tracker overhead) |
| `YOLOSegmentationInference` | Masks | Moderate |

For long videos with multiple animals: `YOLOPoseTrackInference` is the canonical mufasa entry. For short videos or detection-only: skip pose and use blob tracking.

---

## R10. Heuristic-only (no classifier)

**Scenario:** Simple behaviors detectable from pose alone — freezing, circling, immobility. No annotation needed.

### Heuristic backends

- `FreezingDetector` — velocity threshold + duration filter. Configurable per body-part.
- `CirclingDetector` — angular velocity + total rotation.
- `MovementCalculator` — distance traveled, mean velocity, time-immobile.

### Steps

1. R1 through preprocessing.
2. Run the heuristic backend directly (call from Python or via Analysis page).
3. Output goes to `logs/<heuristic>_<timestamp>.csv`.

No model training, no annotation, no inference. Skip the entire Classifier section.

### When heuristic beats ML

- Behavior is well-defined kinematically.
- Annotation cost is prohibitive (rare behavior, lots of videos).
- Reproducibility is critical (heuristic is deterministic; RF is not).

### When ML beats heuristic

- Behavior depends on context (posture + location + history).
- Multiple animals interact.
- You have annotations already.

---

## R11. Feature subsets for non-ML use

**Scenario:** Publication figure or downstream non-ML analysis. You want specific feature families (distances, angles, hulls) without committing to the full ML pipeline.

### Steps

1. R1 through preprocessing.
2. **Features → Compute feature subsets** (`FeatureSubsetExtractorForm`).
3. Choose feature families:
   - `TWO-POINT BODY-PART DISTANCES (MM)`
   - `WITHIN-ANIMAL THREE-POINT BODY-PART ANGLES (DEGREES)`
   - `BODY-PART CONVEX HULL` (area / perimeter)
   - `BODY-PART INTERPOLATED FRAME-TO-FRAME EUCLIDEAN VELOCITY`
   - and others.
4. Backend: `FeatureSubsetsCalculator`. Writes per-family parquet to `derived/features/<family_slug>/<video>.parquet`.
5. Read parquet files in pandas / R / Julia / matlab for your analysis.

### Why use this instead of the wide feature frame?

- **Smaller files.** One family is much smaller than the kitchen-sink wide frame.
- **Cleaner provenance.** Each parquet is labeled by family.
- **No ML commitment.** You're not implicitly training on a feature you'll use to compute the dependent variable.

### Drawback

Per-family parquet is not what `TrainRandomForestClassifier` reads. If you later decide to train a classifier, you'll re-run the wide feature extraction. Both can coexist — they live in different subdirectories.

---

## Cross-recipe notes

### Per-video parallelism

Every feature extractor, analyzer, and most plotters accept a `core_cnt` parameter. Default is 1 (sequential). Recommended: set to `cpu_count() - 1` for batch operations. Forms expose this as a numeric input.

### GPU usage

Mostly limited to YOLO (R9) and one or two visualizers. The rest of mufasa is CPU-bound.

### Video file vs symlink

Video Import defaults to **symlink** (after patch 122v). Faster, no disk waste. Switch to copy if your project will move between filesystems.

### v1 vs legacy project format

All recipes above assume v1 TOML. Legacy INI projects work identically through the form layer for everything except the `derived/` tree — those projects use `csv/features_extracted/`, `csv/targets_inserted/`, `csv/machine_results/` instead. For new projects, **always use v1 TOML**.

### Validation discipline

For every classifier:
1. Hold out at least one video from training.
2. Run `InferenceValidation` → review the validation video.
3. Sample 200 random frames from `derived/classifications/<video>.parquet`, inspect agreement.
4. Iterate before running inference on all videos.

A common failure mode: training on 80% of frames within a video and validating on the other 20%. Same video → near-zero variance → false confidence. Always hold out **whole videos**.

### Reproducibility

Each pipeline stage writes timestamped log files to `logs/`. To make a run fully reproducible:
- Pin the random seed at training (mufasa uses `random_state=1` by default).
- Save `project.toml`, `models/<classifier>.sav`, and `logs/<classifier>_meta.json` together.
- Record the pose-importer version / DLC model checkpoint separately — those upstream artifacts aren't in mufasa's reproducibility scope.
