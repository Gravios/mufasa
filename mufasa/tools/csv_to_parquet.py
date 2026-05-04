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


def _detect_header_rows(csv_path: str, max_probe_lines: int = 10) -> int:
    """Detect the number of non-numeric header rows in a CSV.

    Mufasa's input_csv/ files preserve the DLC-style multi-row
    header (typically 3 rows: scorer, bodypart, coord). Other
    pipeline stages (outlier_corrected_*, features_extracted/,
    targets_inserted/, machine_results/) write flat single-row
    headers via pyarrow's csv.write_csv path.

    Returns the row index where actual data starts. 1 means the
    standard "first row is header" case (so 0-indexed line 0 is
    the header). 3 means a 3-row multi-index header (rows 0-2
    are headers, line 3 is first data).

    Algorithm: read up to ``max_probe_lines`` lines, try to
    parse each row's columns (after the first index column) as
    float. The first row that's all-float marks the start of
    data; the count of non-data rows above it is the header count.
    """
    import csv as _csv
    header_count = 0
    with open(csv_path, "r", newline="") as f:
        reader = _csv.reader(f)
        for i, row in enumerate(reader):
            if i >= max_probe_lines:
                break
            if not row:
                continue
            # Skip the first column (Mufasa's index pad) when
            # checking — it's expected to be either empty or a
            # numeric row index.
            data_cells = row[1:] if len(row) > 1 else row
            try:
                # If every cell parses as float, this is a data row.
                for cell in data_cells:
                    if cell == "" or cell.lower() == "nan":
                        continue
                    float(cell)
                # All cells are numeric — this row is data.
                return header_count
            except ValueError:
                # Non-numeric cell present — this is a header row.
                header_count += 1
    # If we didn't find a data row in the probe window, assume the
    # standard flat header (1 row). The caller will likely catch
    # any inconsistency at write/verify time.
    return 1 if header_count == 0 else header_count


def convert_csv_to_parquet(
    csv_path: str, parquet_path: Optional[str] = None,
    has_index: bool = True,
) -> Tuple[int, int]:
    """Convert a single CSV to parquet.

    Auto-detects whether the CSV has a single-row header (the
    standard Mufasa output for processed pipeline stages) or a
    multi-row header (Mufasa's input_csv/ files preserve a 3-row
    DLC-style header: scorer, bodypart, coord). The two paths
    differ:

    - Flat single-row header: use pyarrow.csv.read_csv → table →
      parquet directly. Fastest path. Doesn't suffer from
      pandas's chunked-inference DtypeWarning because pyarrow
      uses per-column inference.

    - Multi-row header: pandas read_csv with header=list(range(N))
      to parse the multi-index, flatten by taking the LAST level
      of the multi-index (which carries the bodypart_x / _y / _p
      column names — the actual identifiers downstream code uses),
      then write parquet via pandas. Slower per file but only
      applies to the input_csv/ stage.

    :param has_index: True if the CSV has a leading unnamed
        index column. Mufasa writes with this, so default True.
    :return: (n_rows, n_cols) of the converted file.
    """
    if parquet_path is None:
        parquet_path = str(Path(csv_path).with_suffix(".parquet"))

    n_header_rows = _detect_header_rows(csv_path)

    if n_header_rows <= 1:
        # Flat header — fast path via pyarrow.
        try:
            from pyarrow import csv as pa_csv
            from pyarrow import parquet as pa_parquet
            table = pa_csv.read_csv(csv_path)
            if has_index:
                col_names = table.column_names
                if col_names:
                    table = table.drop([col_names[0]])
            pa_parquet.write_table(table, parquet_path)
            return table.num_rows, table.num_columns
        except ImportError:
            pass
        # Pandas fallback for single-row case
        df = pd.read_csv(
            csv_path,
            index_col=0 if has_index else None,
            low_memory=False,
        )
        df.to_parquet(parquet_path)
        return len(df), len(df.columns)

    # Multi-row header path. read_csv with header=[0,1,2] for
    # 3-row case. Pandas builds a MultiIndex from those rows;
    # we flatten by taking just the last level (the actual
    # bodypart_x / _y / _p column names).
    df = pd.read_csv(
        csv_path,
        index_col=0 if has_index else None,
        header=list(range(n_header_rows)),
        low_memory=False,
    )
    # Flatten MultiIndex columns to just the last level. This
    # matches what Mufasa's downstream code expects after read_df
    # processes a multi-idx file (see read_df with check_multiindex
    # in utils/read_write.py — it drops the header rows and
    # essentially uses the column names from the bottom level).
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    # Coerce all numeric — multi-index files may have data
    # parsed as object dtype if pandas saw the header rows
    # ambiguously. Force-cast.
    df = df.apply(pd.to_numeric, errors="coerce")
    df.to_parquet(parquet_path)
    return len(df), len(df.columns)


def verify_parquet(
    csv_path: str, parquet_path: str, has_index: bool = True,
) -> bool:
    """Read both files and check shape + column names match.

    For multi-row-header CSVs (input_csv/ files), the column
    names in the parquet are derived from the LAST row of the
    multi-index header (matching convert_csv_to_parquet's
    flattening). So we compare against the last header line of
    the CSV.

    Row count for parquet comes from metadata. For CSV it's
    streamed line count minus the number of header rows.
    """
    n_header_rows = _detect_header_rows(csv_path)

    # Get CSV columns (last header row, the data-level names)
    try:
        with open(csv_path, "r", newline="") as f:
            import csv as _csv
            reader = _csv.reader(f)
            header_lines = []
            for i, row in enumerate(reader):
                if i >= n_header_rows:
                    break
                header_lines.append(row)
        if not header_lines:
            return False
        csv_cols = header_lines[-1]  # last header row = column names
        if has_index:
            csv_cols = csv_cols[1:]  # drop index slot
    except Exception:
        # Fall back to pandas if our streaming reader chokes
        df = pd.read_csv(
            csv_path,
            index_col=0 if has_index else None,
            header=list(range(n_header_rows)) if n_header_rows > 1 else 0,
            nrows=0,
        )
        if isinstance(df.columns, pd.MultiIndex):
            csv_cols = list(df.columns.get_level_values(-1))
        else:
            csv_cols = list(df.columns)

    # Parquet columns from metadata (no row data read)
    import pyarrow.parquet as pq_module
    pq_schema = pq_module.read_schema(parquet_path)
    pq_cols = list(pq_schema.names)

    if csv_cols != pq_cols:
        return False

    pq_rows = pq_module.read_metadata(parquet_path).num_rows
    # Stream-count CSV rows minus header rows
    with open(csv_path, "rb") as f:
        total_lines = sum(1 for _ in f)
    csv_rows = total_lines - n_header_rows
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


def _convert_one(args: Tuple[str, str, bool]) -> Tuple[str, str, Optional[str], int, int]:
    """Worker: convert one CSV to parquet and verify.

    Returns (csv_path, parquet_path, error_or_None, n_rows, n_cols).
    Top-level so ProcessPoolExecutor can pickle it.
    """
    csv_path, parquet_path, has_index = args
    try:
        n_rows, n_cols = convert_csv_to_parquet(
            csv_path, parquet_path, has_index=has_index,
        )
        if not verify_parquet(csv_path, parquet_path, has_index=has_index):
            try:
                os.unlink(parquet_path)
            except Exception:
                pass
            return (csv_path, parquet_path, "verification failed", 0, 0)
        return (csv_path, parquet_path, None, n_rows, n_cols)
    except Exception as exc:
        return (
            csv_path, parquet_path,
            f"{type(exc).__name__}: {exc}", 0, 0,
        )


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
    parser.add_argument(
        "--n-workers", type=int, default=None,
        help=(
            "number of parallel processes for conversion. Default: "
            "min(8, os.cpu_count()). Pyarrow's CSV reader is already "
            "internally parallel for a single file; high process "
            "counts mostly help when many small files saturate per-"
            "process I/O setup overhead. 1 = sequential."
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

    # Phase 1: convert all (in parallel)
    # ProcessPoolExecutor isolates each conversion — a single
    # failed/malformed CSV doesn't bring down the others.
    # Default workers: min(8, cpu_count). Going higher than 8
    # rarely helps because pyarrow's CSV reader is already
    # multi-threaded internally; the wins from parallelism here
    # come from overlapping I/O setup across files.
    n_workers = args.n_workers
    if n_workers is None:
        cpu = os.cpu_count() or 4
        n_workers = min(8, cpu)
    n_workers = max(1, n_workers)

    print(
        f"\nConverting {len(files)} CSV(s) to parquet "
        f"(n_workers={n_workers})..."
    )
    failures: List[Tuple[str, str]] = []
    converted: List[Tuple[str, str]] = []  # (csv_path, parquet_path)

    work_items = [
        (csv_path, str(Path(csv_path).with_suffix(".parquet")), has_index)
        for csv_path in files
    ]

    if n_workers == 1:
        # Sequential — useful for debugging or constrained envs.
        results = (_convert_one(item) for item in work_items)
    else:
        from concurrent.futures import ProcessPoolExecutor
        executor = ProcessPoolExecutor(max_workers=n_workers)
        try:
            results = executor.map(_convert_one, work_items)
        except KeyboardInterrupt:
            executor.shutdown(wait=False, cancel_futures=True)
            raise

    completed = 0
    try:
        for csv_path, parquet_path, error, n_rows, n_cols in results:
            completed += 1
            if error is not None:
                failures.append((csv_path, error))
            else:
                converted.append((csv_path, parquet_path))
            print(
                f"  [{completed}/{len(files)}] "
                f"{os.path.basename(csv_path)}: "
                + (
                    f"{n_rows} rows × {n_cols} cols"
                    if error is None
                    else f"FAILED — {error}"
                )
            )
    except KeyboardInterrupt:
        if n_workers > 1:
            executor.shutdown(wait=False, cancel_futures=True)
        print("\nInterrupted by user.", file=sys.stderr)
        # Treat as failure so we don't update config
        return 130
    finally:
        if n_workers > 1:
            executor.shutdown(wait=True)

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
