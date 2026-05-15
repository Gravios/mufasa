# CSV-Subtree Removal Audit

**Goal:** Determine what would break if `features_extracted_dir`,
`targets_inserted_dir` keys are dropped from
`project_paths_from_config`, and the legacy
`csv/features_extracted/` + `csv/targets_inserted/` trees stop
being written to entirely.

**Scope:** `features_extracted/` and `targets_inserted/` only.
`machine_results/` is OUT of scope — classifier inference still
writes there; that's separate work.

**Result:** 3 active blockers, 3 dead-code patches, 1 attribute
deprecation. Total: ~6 small follow-up patches before the keys
can be dropped from `project_paths_from_config`.

---

## Category A — ACTIVE BLOCKERS (must migrate before removal)

These code paths actively WRITE to or READ from the legacy
trees. Removing the layout keys would cause silent data loss
or runtime exceptions.

### A1. `directing_other_animals_calculator.py` — feature write-back

```python
# line 217 (in run loop, when append_bool_tables_to_features=True):
write_df(
    df=df, file_type=self.file_type,
    save_path=os.path.join(
        self.features_dir, f"{video_name}.{self.file_type}",
    ),
)
```

122ae-5b migrated the READ. The WRITE still targets the legacy
location to "append directionality boolean tables to features."
After 122ak (where `load_features_for_video` no longer reads
legacy), this write is **orphaned**: it creates a file that no
consumer reads.

**Fix:** Replace with `write_wide_features_v1(...)` (the helper
from 122ae-4). One-line swap, no semantic change for v1
consumers. Estimated effort: trivial.

### A2. Seven specialty feature extractors — never migrated to v1 sidecar

| File | Save target |
|---|---|
| `aggression_feature_extractor.py` | `save_dir = self.features_dir if save_dir is None` |
| `amber_feature_extractor.py` | writes to `self.features_dir` |
| `boundary_rearing_analyzer.py` | `save_dir = self.features_dir if save_dir is None` |
| `cave_fish_featurizer.py` | writes to `self.features_dir` |
| `gerbil_featurizer.py` | writes to `self.features_dir` |
| `mitra_feature_extractor.py` | writes to `self.features_dir` |
| `rat_social_featurizer.py` | writes to `self.features_dir` |
| `wingwave_extractor.py` | writes to `self.features_dir` |

122ae-4 instrumented the 8 standard extractors (4/7/8/9/14/16bp
+ 8bps_2_animals + user_defined) with the v1 wide-parquet
sidecar but left these 7 specialty ones on legacy paths only.

**Fix:** Same one-line `write_wide_features_v1` injection per
file as 122ae-4. Mechanical work. Estimated effort: 1 small
patch covers all 7.

### A3. `single_clf_appender_excel.py` — XLSX → targets write

```python
data_dir = self.features_dir if data_dir is None else data_dir
self.save_dir = self.targets_folder if save_dir is None else save_dir
```

XLSX-import tool that reads features + Excel labels and writes
combined targets. Same shape as the third-party appenders that
were migrated in 122ak.

**Fix:** Retarget to write via `save_labels_for_video` (mirror
the BENTO / BORIS / deepethogram migrations from 122ak). One
file, ~5 lines changed.

---

## Category B — DEAD CODE GATED BY USER FLAGS

User-visible API surface that does nothing meaningful anymore.
Misleading more than broken.

### B1. `feature_subsets.py` `append_to_features_extracted` /
`append_to_targets_inserted` kwargs

Two boolean kwargs on `FeatureSubsetsCalculator` (and the
matching UI form fields in `features.py` lines 587-588 and
609-610, and `subset_feature_extractor_pop_up.py` lines 84-85)
that perform PRE-FLIGHT checks for legacy directory presence
but never trigger an actual write. The writer was retargeted
to per-family parquet in 122ae-3; these flags became inert.

**Fix:** Remove the kwargs from the calculator, the two UI
forms, and the popup. The pre-flight check at
`feature_subsets.py:252-257` goes too. ~25 lines deleted across
4 files.

### B2. `select_video_for_labelling_popup.py` existence checks

```python
# Around lines that decide "continue" vs "new" mode based on
# whether csv/targets_inserted/<video>.csv exists.
if os.path.isfile(targets_inserted_file_path):
    ... # offer continue mode
```

After 122ak, labels live under `derived/labels/<video>.parquet`.
The legacy check returns False even when v1 labels exist, so
the popup may incorrectly offer "new" when "continue" is valid.

**Fix:** Replace the `os.path.isfile` check with
`try: load_labels_for_video(...); except FileNotFoundError`.
~3 lines per check site.

### B3. ConfigReader's legacy glob populations

```python
# mufasa/mixins/config_reader.py
self.feature_file_paths = glob.glob(
    self.features_dir + f"/*.{self.file_type}",
)
self.target_file_paths = glob.glob(
    self.targets_folder + f"/*.{self.file_type}",
)
```

After 122ae-5b + 122ae-5e, the 5 RF training modules + inference
batch ignore these populated lists when they're empty and
re-derive via `list_video_stems_with_*`. So the ConfigReader
glob is redundant. Not harmful, just unnecessary work on every
ConfigReader instantiation.

**Fix:** Drop the two glob.glob lines from ConfigReader. Audit
to confirm no remaining consumer depends on the pre-populated
list when it's non-empty. (Risk: the consumer's "if not
self.target_file_paths" fallback is what triggers the v1
discovery; removing the glob makes that the only path, which
is fine.)

---

## Category C — HARMLESS LEGACY REFERENCES

41 sites that mention the legacy paths but don't do I/O against
them. Safe to leave OR clean up cosmetically.

### C1. Pseudo-paths constructed for stem extraction

Training modules build `os.path.join(self.targets_folder,
f"{stem}.{self.file_type}")` paths to populate
`self.target_file_paths`. Those paths are then consumed by
`get_fn_ext(file_path)` for the stem only — the file is never
opened. Files affected:
- `mufasa/labelling/labelling_interface.py` (2 sites)
- `mufasa/labelling/labelling_advanced_interface.py` (2 sites)
- `mufasa/labelling/standard_labeller.py` (2 sites)
- `mufasa/model/train_rf.py` etc. (5 RF modules, 1 site each)
- `mufasa/feature_extractors/{mitra,rat_social,wingwave}` (1 each)

These are harmless. Cleanup would be cosmetic — replace the
pseudo-path construction with a list of stems directly. Not
necessary for csv/ removal because the path string never
touches disk.

### C2. Error message string interpolations

16 files reference the legacy directory only inside f-strings
like `f"No data files found in {self.targets_folder}"`. These
are user-facing error messages; the path string is just used
for display. Removing the path attribute would make these
NameError at error-time.

**Fix when keys are dropped:** Replace each f-string with
either the v1 path or generic phrasing. Mechanical work, ~16
edits.

Files in this category:
- `mufasa/data_processors/directing_other_animals_calculator.py`
  (also Category A1; the error msg here is fine)
- `mufasa/feature_extractors/{boundary_rearing,user_defined,
  rat_social,wingwave,feature_subsets}.py`
- `mufasa/labelling/extract_labelling_meta.py`
- `mufasa/mixins/annotator_mixin.py`
- `mufasa/model/{grid_search_rf,train_rf}.py`
- `mufasa/plotting/annotation_videos.py`
- `mufasa/roi_tools/ROI_feature_analyzer.py`
- `mufasa/third_party_label_appenders/{BENTO,BORIS,observer,
  solomon,third_party}_*.py`
- `mufasa/ui/pop_ups/{annotated_bouts_videos,
  boolean_conditional_slicer,clf_annotation_counts,
  run_machine_models,select_video_for_labelling,
  third_party_annotator_appender,video_processing}_*.py`

---

## Recommended order

1. **Patch 122al** — A1 + A2 + A3 together (all active writes
   migrate to v1 helpers). 1 file for A1, 7 files for A2, 1
   file for A3. ~9 files, mostly mechanical write-site swaps.
   Closes the active blocker bucket entirely.

2. **Patch 122am** — B1 + B2 (kill dead API). Drops the
   `append_to_*` kwargs from `FeatureSubsetsCalculator` and the
   UI forms; rewrites the labelling-popup existence check.
   Pure subtractive change.

3. **Patch 122an** — B3 + key removal from
   `project_paths_from_config`. Drops the ConfigReader globs;
   removes `features_extracted_dir` and `targets_inserted_dir`
   from the layout helper. The Category C ERROR_MSG references
   become NameError unless they're cleaned up — sweep all 16
   in this patch with sed-style replacements.

After 122an, the only csv/ subtree remaining in
`project_paths_from_config` is `machine_results_dir`. That can
move to `derived/classifications/` in a future patch when the
inference write paths are themselves migrated.

---

## Risk assessment

**Low risk** (mechanical, well-isolated):
- A1, A3 — single-site write swaps.
- A2 — pattern-replication across 7 files (the 122ae-4
  template).
- B1 — pure deletion of dead UI kwargs.
- B3 — pure deletion of redundant ConfigReader globs.

**Medium risk**:
- B2 — the popup existence-check change has subtle UX impact
  (continue vs new mode dispatch). Worth a manual test in the
  UI before shipping.
- 122an's mass-edit of 16 ERROR_MSG references — easy to typo;
  AST verify with tests.

**No risk**:
- Category C1 (pseudo-paths) — no action required.

---

## Status of `machine_results_dir`

Out of scope here. 35 files reference it. Active writers:
`inference_batch.py` and `inference_multiclass_batch.py` (the
classifier inference outputs). Various downstream analysis
modules read it.

Migrating `machine_results/` to `derived/classifications/<run_id>/`
is conceptually similar to the features migration but touches
more consumer modules. Defer until v1-features and v1-labels
prove themselves in production use.
