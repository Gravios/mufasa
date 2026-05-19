# Migration guide: legacy SimBA → v1

How to migrate a legacy SimBA-layout Mufasa project to the v1 layout using the `mufasa.cli.migrate_project` tool.

**Audience:** users with existing SimBA-style projects who want to adopt the v1 layout.

**Scope:** when to migrate, how the tool works, recommended workflow, troubleshooting, and what to verify after migration.

---

## TL;DR

```bash
# 1. Dry run — prints every operation but doesn't touch files
python -m mufasa.cli.migrate_project /path/to/legacy_project

# 2. Commit
python -m mufasa.cli.migrate_project /path/to/legacy_project --commit

# 3. Open the new project in the workbench
mufasa-workbench
# → File → Open Project → /path/to/legacy_project/project.toml
```

After the migration:
* `project.toml` appears at the project root.
* Existing data lives under `derived/<stage>/imported_<YYYYMMDD>/` runs.
* The original `project_folder/` is left empty (rmdir manually once verified).
* A `MIGRATION.toml` manifest records every move for auditability.

---

## When to migrate

**Migrate if:**
* You want per-run separation (re-running outlier correction with different params no longer clobbers prior output).
* You're starting new analyses on an existing project and want the cleaner directory grouping.
* You're using a workflow that v1 prefers (e.g., the visualization page's `derived/`-routed source-dir resolution from patch 122dc).

**Don't bother if:**
* The project is "done" — final classifications computed, papers submitted. Legacy works fine for archival use; migration adds noise without value.
* You're mid-experiment and don't want to risk path-resolution differences during analysis. Finish the experiment in legacy, then migrate (or don't).

Both layouts remain supported indefinitely. There's no deprecation timeline. The migration tool is for users who want v1's improvements; nothing forces the move.

---

## What the tool does

`mufasa.cli.migrate_project` walks a legacy project and:

1. **Detects layout.** If already v1, exits as a no-op. If unrecognized, exits with an error.
2. **Plans the move.** Walks the `LEGACY_TO_V1_MAPPING` table (in `mufasa/legacy_layout.py`), maps each legacy subdirectory to its v1 destination, and groups all existing data under one `imported_<YYYYMMDD>` run label.
3. **Creates the v1 skeleton.** `sources/`, `derived/`, `models/`, `logs/` plus required subdirectories.
4. **Moves files.** Verbatim; no re-encoding or transformation. CSV stays CSV, parquet stays parquet.
5. **Writes `project.toml`** at the new project root, populated from the legacy `project_config.ini` + body-part-name files.
6. **Writes `MIGRATION.toml`** manifest at the new root listing every operation. This is your audit trail — if anything looks wrong after the migration, this file tells you exactly what moved where.

The tool **does not**:

* Re-encode or transform pose data.
* Touch `models/` contents (trained classifiers stay where they are).
* Delete the original `project_folder/` directory (left empty for manual review and removal).
* Migrate trained model artifacts to the global cache (do that separately via `mufasa.project_layout.mirror_model_to_global_cache`).

**Idempotence:** running the tool on an already-migrated project is a safe no-op. Re-running on a partial migration (e.g., if the prior run crashed) is **not** safe — verify by hand or restore from backup.

---

## Recommended workflow

### 1. Back up the project

`mufasa.cli.migrate_project` moves files in place. There's no built-in undo. Make a copy first:

```bash
cp -r /path/to/legacy_project /path/to/legacy_project.bak
```

For large projects (videos can be many GB), use a snapshot tool (btrfs/zfs/rsync hardlinks) rather than a full copy.

### 2. Dry run

```bash
python -m mufasa.cli.migrate_project /path/to/legacy_project
```

The output lists every operation the tool would perform, in order. Read through it:

* Are all expected legacy directories represented?
* Do the destination paths look right?
* Are there any files in unexpected places (e.g., files dropped directly under `project_folder/` instead of under `csv/something/`)?

If something looks off, **stop**. Investigate before committing. Common causes:
* Body-parts CSV in a non-canonical location → fix by hand before migration.
* Extra subdirectories under `csv/` that the mapping doesn't recognize → they'll be skipped; either move them by hand or extend `LEGACY_TO_V1_MAPPING` first.

### 3. Commit

When the dry run looks right:

```bash
python -m mufasa.cli.migrate_project /path/to/legacy_project --commit
```

Watch for errors. Per-operation output goes to stdout; errors go to stderr. If a single move fails the tool continues with the rest, so check stderr at the end.

### 4. Verify

After the tool exits:

* `<project_root>/project.toml` exists and `head` it to sanity-check the metadata.
* `<project_root>/MIGRATION.toml` exists. Look through it.
* `<project_root>/derived/<stage>/imported_<YYYYMMDD>/` contains the moved data files.
* `<project_root>/project_folder/` (the original) is now empty (or near-empty — body-parts CSV will likely still be inside; that's expected).

### 5. Open in the workbench

```bash
mufasa-workbench
# File → Open Project → <project_root>/project.toml
```

The workbench should detect v1 from the `.toml` extension and route all path lookups through the v1 helper. Verify:

* Project chooser shows the project name from `project.toml`.
* Video Info form lists all videos.
* ROI Definitions dialog finds the existing ROI HDF (at `logs/measures/ROI_definitions.h5`; the path didn't change).
* Visualization routes find their source data (look at `derived/<stage>/imported_<YYYYMMDD>/`).

If the visualization or analysis pages can't find data, the most likely cause is the migration mapped a stage to a different `derived/` subdirectory than the workbench expects. Check `MIGRATION.toml` and the [v1 project layout doc](v1_project_layout.md) for the canonical names.

### 6. Clean up

Once you're confident:

```bash
rmdir /path/to/legacy_project/project_folder  # if empty
rm /path/to/legacy_project/project_config.ini # if present
```

Keep `MIGRATION.toml` — it's your audit trail. The backup from step 1 can be deleted once you've run a real workflow against the migrated project and everything works.

---

## Tool flags

```
python -m mufasa.cli.migrate_project <path> [options]

Positional:
  path                 Legacy project location. Either the parent of
                       project_folder/ or project_folder/ itself.

Options:
  --commit             Actually perform the migration. Without this,
                       the tool prints the plan and exits.
  --v1-root <path>     Write the v1 layout to a different directory
                       than the legacy project's location. Useful for
                       parallel-tree migrations (keep legacy intact,
                       build a v1 copy elsewhere).
  --quiet              Suppress per-operation output. Errors still go
                       to stderr.
```

---

## What gets moved where

Approximate mapping (see `mufasa/legacy_layout.py` for the authoritative table):

| Legacy | v1 |
|---|---|
| `<project>/videos/` | `<v1_root>/sources/videos/` |
| `<project>/csv/input_csv/` | `<v1_root>/sources/pose/` |
| `<project>/csv/outlier_corrected_movement_location/` | `<v1_root>/derived/outlier_corrected/imported_<DATE>/` |
| `<project>/csv/features_extracted/` | `<v1_root>/derived/features/imported_<DATE>/` |
| `<project>/csv/targets_inserted/` | `<v1_root>/derived/labels/imported_<DATE>/` |
| `<project>/csv/machine_results/` | `<v1_root>/derived/classifications/` (flat; one parquet per video) |
| `<project>/logs/` | `<v1_root>/logs/` (preserved verbatim, including `measures/ROI_definitions.h5`) |
| `<project>/models/` | `<v1_root>/models/` (preserved verbatim) |
| `<project>/frames/input/` | `<v1_root>/derived/frames/extracted/` |
| `<project>/frames/output/` | `<v1_root>/derived/frames/annotated/` |
| `<project>/logs/video_info.csv` | `<v1_root>/sources/video_info.csv` |
| `<project>/project_config.ini` | (read; not moved) — used to populate `project.toml`. The original is left in place. |
| Body-parts CSV | (read; not moved) — body parts go into `project.toml`'s `[pose].body_parts` array. |

Anything not in the table — extra `csv/` subdirectories, third-party-tool outputs, scratch files you added — is skipped. The dry run lists them as `skip`; either move them by hand after the migration or extend `LEGACY_TO_V1_MAPPING` first.

---

## Troubleshooting

### "Not a recognized Mufasa project layout"

`detect_layout()` looks for either a `project.toml` (v1) or a `project_folder/project_config.ini` (legacy). If neither is present:

* You may have passed the wrong path. The tool accepts either the parent of `project_folder/` or `project_folder/` itself. If your structure is different, try the other.
* The `project_config.ini` may be missing. Restore it from backup or recreate manually (the legacy SimBA tutorials cover this).

### "Already a v1-layout project. Nothing to do."

The tool detected a `project.toml` at the given path. This is the no-op idempotent case — running again on an already-migrated project. If you actually wanted to migrate a different project, double-check the path.

### Some files weren't moved

Read the dry-run output carefully. Any operation marked `skip` is something the tool noticed but chose not to move (e.g., an unrecognized subdirectory under `csv/`). The `MIGRATION.toml` records skips too.

Move skipped files by hand if needed. If the same skip pattern keeps showing up across multiple projects, consider adding to `LEGACY_TO_V1_MAPPING` in `mufasa/legacy_layout.py` so future migrations handle it automatically.

### "FileNotFoundError" mid-migration

The tool fails per-operation rather than rolling back. If a move failed:

1. Look at stderr for which operation failed and why.
2. Check `MIGRATION.toml` — operations recorded there have completed; everything after the failure point has not.
3. Manually correct the underlying cause (permissions, disk space, etc.).
4. Re-running the tool is NOT safe in this state — it expects an all-or-nothing state. Either restore from backup and re-migrate, or manually finish the remaining moves and create `project.toml` by hand.

### Body parts list is empty in `project.toml`

The migration reads body parts from the legacy `project_bp_names.csv` (typically at `<project>/logs/measures/pose_configs/bp_names/project_bp_names.csv`). If that file is missing, malformed, or empty, the v1 `project.toml` ends up with an empty `[pose].body_parts` array. Edit `project.toml` manually to add the body parts, or use `mufasa.utils.project_reconfigure.reconfigure_bp_names()` on the legacy project before migrating.

### Workbench can't find data after migration

If a workbench page reports "data source directory not found":

* Check the path it's looking for against the `MIGRATION.toml` operations log.
* The visualization page in particular uses `_VIZ_SOURCE_V1_MAP` (in `mufasa/ui_qt/forms/visualizations.py`, added in patch 122dc) to map route source names to v1 directories. If the page is looking at a stage the map doesn't cover, this surfaces as the "not found" error.

---

## Rollback

The tool has no built-in rollback. To revert:

1. Restore the original `project_folder/` from your backup (step 1 above).
2. Delete the migrated v1 artifacts: `project.toml`, `MIGRATION.toml`, `sources/`, `derived/`, `models/` (only if it was created fresh; if it was pre-existing legacy `models/`, the migration moved it in place).
3. The legacy `project_config.ini` and body-parts CSV should still be in place (the migration reads but doesn't move them).

**Always** keep the step-1 backup until you've run a real workflow against the migrated project and confirmed everything works.

---

## References

* Tool source: [`mufasa/cli/migrate_project.py`](../mufasa/cli/migrate_project.py)
* Mapping table: [`mufasa/legacy_layout.py`](../mufasa/legacy_layout.py) — `LEGACY_TO_V1_MAPPING`
* v1 layout: [`v1_project_layout.md`](v1_project_layout.md)
* Path-abstraction layer: [`mufasa/project_layout.py`](../mufasa/project_layout.py)
