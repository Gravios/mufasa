"""
mufasa.tools.csv_to_parquet
============================

Migrate a Mufasa project from CSV to parquet storage. Walks the
project's pose-data directories, converts every CSV to parquet,
updates ``project_config.ini`` to reflect the new file_type, and
optionally removes the original CSVs.

Why bother
----------

Parquet is binary, columnar, and compressed. For typical pose
data (50K-row × ~40-col) the file is 5-10× smaller than the
equivalent CSV. Reading a column header is microseconds (just
metadata), vs. seconds for a CSV the same size. Mufasa's
preflight check (which reads headers from every existing file)
drops from minutes to milliseconds when the project is parquet.

The downside: parquet isn't human-readable. If you frequently
inspect raw values with a text editor, CSV stays more
convenient. But ``pd.read_parquet(path).head()`` is fast and
gives you the same view in Python.

What gets converted
-------------------

Only files in the standard Mufasa pose-data subdirectories
under ``project_folder/csv/``:

  - input_csv/
  - outlier_corrected_movement/
  - outlier_corrected_movement_location/
  - features_extracted/
  - targets_inserted/
  - machine_results/

Files in ``logs/`` and ``logs/measures/`` are NOT converted —
those are ROI definitions, video metadata, and annotation logs,
none of which are read by the file_type-aware machinery.

Files in temporary directories (matching ``temp_*`` or
``Prior_to_*``) are skipped — those are leftover scratch from
in-progress / aborted runs.

Conversion order
----------------

1. Walk every source directory and build a list of CSVs to convert
2. (Optional dry-run) print what would be converted, exit
3. For each CSV, write a sibling .parquet (same basename) and
   verify row count + column set match
4. Once all conversions succeed, update project_config.ini
   file_type = parquet
5. (Optional) remove the original CSVs

The conversion is atomic-ish: parquets are written alongside
CSVs, NOT replacing them, until all are verified. If any
conversion fails partway, the project_config.ini is NOT updated
and you're left with a mixed CSV+parquet state — but no data
loss.

CLI
---

    # Dry run: just show what would be converted
    python -m mufasa.tools.csv_to_parquet \\
        --config /path/to/project_config.ini --dry-run

    # Actually convert (keeps CSVs as backup)
    python -m mufasa.tools.csv_to_parquet \\
        --config /path/to/project_config.ini

    # Convert AND delete original CSVs (irreversible)
    python -m mufasa.tools.csv_to_parquet \\
        --config /path/to/project_config.ini --delete-csv
"""
from __future__ import annotations

import argparse
import configparser
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


# Standard subdirectories under csv/ that contain pose-data files.
# Files in OTHER csv/ subdirectories (or in logs/) are skipped.
POSE_DATA_DIRS = (
    "input_csv",
    "outlier_corrected_movement",
    "outlier_corrected_movement_location",
    "features_extracted",
    "targets_inserted",
    "machine_results",
)


def discover_files(project_folder: str) -> List[str]:
    """Return list of CSV files in pose-data directories that
    should be converted. Skips temp directories and hidden files."""
    project_folder = Path(project_folder)
    csv_root = project_folder / "csv"
    if not csv_root.is_dir():
        raise FileNotFoundError(
            f"Expected csv/ subdirectory under {project_folder} "
            f"(is this really a Mufasa project_folder?)"
        )

    found: List[str] = []
    for sub in POSE_DATA_DIRS:
        sub_path = csv_root / sub
        if not sub_path.is_dir():
            continue
        # Only top-level CSVs in each directory; deeper paths
        # (temp_*, Prior_to_*) are scratch/backup we don't migrate.
        for entry in sorted(sub_path.iterdir()):
            if entry.is_file() and entry.suffix == ".csv" and not entry.name.startswith("."):
                found.append(str(entry))
    return found


def convert_csv_to_parquet(
    csv_path: str, parquet_path: Optional[str] = None,
    has_index: bool = True,
) -> Tuple[int, int]:
    """Convert a single CSV to parquet.

    :param has_index: True if the CSV has a leading unnamed
        index column (pandas to_csv default). Mufasa writes with
        this, so default True.
    :return: (n_rows, n_cols) of the converted file. Caller can
        compare against the original to verify.
    """
    if parquet_path is None:
        parquet_path = str(Path(csv_path).with_suffix(".parquet"))
    df = pd.read_csv(csv_path, index_col=0 if has_index else None)
    df.to_parquet(parquet_path)
    return len(df), len(df.columns)


def verify_parquet(
    csv_path: str, parquet_path: str, has_index: bool = True,
) -> bool:
    """Read both files and check shape + column names match.

    Doesn't compare values element-wise (would double the I/O for
    large projects). The header + row count match catches the
    overwhelming majority of conversion errors.
    """
    csv_df = pd.read_csv(csv_path, index_col=0 if has_index else None, nrows=5)
    pq_df = pd.read_parquet(parquet_path)
    csv_cols = list(csv_df.columns)
    pq_cols = list(pq_df.columns)
    if csv_cols != pq_cols:
        return False
    # Row count: read schema metadata for parquet (cheap) and
    # count CSV rows by streaming
    import pyarrow.parquet as pq
    pq_rows = pq.read_metadata(parquet_path).num_rows
    csv_rows = 0
    with open(csv_path, "rb") as f:
        # Subtract 1 for the header line
        csv_rows = sum(1 for _ in f) - 1
    return pq_rows == csv_rows


def update_config_file_type(config_path: str, new_file_type: str = "parquet") -> None:
    """Update [General settings] file_type in project_config.ini."""
    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    if "General settings" not in cfg:
        raise ValueError(
            f"{config_path} has no [General settings] section — "
            f"is this really a Mufasa project_config.ini?"
        )
    cfg["General settings"]["file_type"] = new_file_type
    with open(config_path, "w") as f:
        cfg.write(f)


def find_project_folder(config_path: str) -> str:
    """Given a project_config.ini path, find the project_folder
    (parent directory that contains csv/, videos/, etc.)."""
    return os.path.dirname(os.path.abspath(config_path))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mufasa-csv-to-parquet",
        description=(
            "Migrate a Mufasa project from CSV to parquet storage. "
            "Converts pose-data CSVs in standard project subdirectories, "
            "verifies each, and updates project_config.ini."
        ),
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="path to project_config.ini",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="list files that would be converted and exit (no changes)",
    )
    parser.add_argument(
        "--delete-csv", action="store_true",
        help=(
            "delete original CSVs after successful conversion + "
            "verification + config update. Irreversible. Default: "
            "keep CSVs alongside parquets as backup."
        ),
    )
    parser.add_argument(
        "--no-index", action="store_true",
        help=(
            "skip the leading index column when reading CSVs "
            "(default: assume an index column exists, which is "
            "Mufasa's standard write format)"
        ),
    )
    args = parser.parse_args(argv)

    if not os.path.isfile(args.config):
        print(f"Config file not found: {args.config}", file=sys.stderr)
        return 1

    project_folder = find_project_folder(args.config)
    print(f"Project folder: {project_folder}")

    # Read current file_type for the report
    cfg = configparser.ConfigParser()
    cfg.read(args.config)
    current_type = cfg.get(
        "General settings", "file_type", fallback="(unknown)",
    )
    print(f"Current file_type in config: {current_type}")

    if current_type == "parquet":
        print("Project is already configured for parquet. Nothing to do.")
        return 0

    files = discover_files(project_folder)
    if not files:
        print(
            "No CSVs found in standard pose-data directories. "
            "Either the project is empty or the directory layout is "
            "non-standard."
        )
        return 0

    print(f"\nFound {len(files)} CSV file(s) to convert:")
    by_dir = {}
    for f in files:
        d = os.path.basename(os.path.dirname(f))
        by_dir.setdefault(d, []).append(os.path.basename(f))
    for d, names in sorted(by_dir.items()):
        print(f"  {d}/: {len(names)} files")

    if args.dry_run:
        print("\n--dry-run: no changes made.")
        return 0

    has_index = not args.no_index

    # Phase 1: convert all
    print("\nConverting CSVs to parquet...")
    failures: List[Tuple[str, str]] = []
    converted: List[Tuple[str, str]] = []  # (csv_path, parquet_path)
    for i, csv_path in enumerate(files, 1):
        parquet_path = str(Path(csv_path).with_suffix(".parquet"))
        try:
            n_rows, n_cols = convert_csv_to_parquet(
                csv_path, parquet_path, has_index=has_index,
            )
            if not verify_parquet(csv_path, parquet_path, has_index=has_index):
                failures.append((csv_path, "verification failed"))
                # Remove the bad parquet so it doesn't pollute
                try:
                    os.unlink(parquet_path)
                except Exception:
                    pass
                continue
            converted.append((csv_path, parquet_path))
            print(
                f"  [{i}/{len(files)}] {os.path.basename(csv_path)}: "
                f"{n_rows} rows × {n_cols} cols"
            )
        except Exception as exc:
            failures.append((csv_path, f"{type(exc).__name__}: {exc}"))

    if failures:
        print(
            f"\n{len(failures)} file(s) failed to convert:",
            file=sys.stderr,
        )
        for path, reason in failures:
            print(f"  {path}: {reason}", file=sys.stderr)
        print(
            "\nProject config NOT updated. Fix the failures and "
            "rerun the migration. The successfully-converted "
            "parquets remain on disk alongside their CSVs.",
            file=sys.stderr,
        )
        return 1

    # Phase 2: update project_config.ini
    print("\nUpdating project_config.ini file_type to 'parquet'...")
    update_config_file_type(args.config, "parquet")

    # Phase 3 (optional): remove original CSVs
    if args.delete_csv:
        print(f"\n--delete-csv: removing {len(converted)} original CSVs...")
        for csv_path, _ in converted:
            try:
                os.unlink(csv_path)
            except Exception as exc:
                print(
                    f"  WARNING: failed to delete {csv_path}: {exc}",
                    file=sys.stderr,
                )
        print("Done.")
    else:
        print(
            f"\nKept {len(converted)} CSVs as backup alongside the "
            f"new parquets. To remove them, re-run with --delete-csv "
            f"or delete manually."
        )

    print("\nMigration complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
