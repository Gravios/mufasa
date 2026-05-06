#!/usr/bin/env python3
"""Strip non-marker columns from pose data files.

Reads a CSV or parquet file (or recursively a directory of them),
identifies columns that form a complete marker triple
(`<name>_x` + `<name>_y` + `<name>_p`), and writes a new file
containing ONLY those columns in marker-grouped order.

Use cases:
  - Cleaning files where SimBA-style feature extraction has been
    layered on top of raw pose (Movement_*, Euclidean_distance_*,
    rolling-window stats, _FEATURE_SUBSET features, etc.)
  - Sanity-checking which directories actually contain pose data
    versus computed features
  - Producing minimal pose files for downstream tools
    (kalman_diagnostic, analysis scripts) that only need raw pose

Files where NO marker triples are detected are reported and
skipped — no empty file is ever written.

Examples:
  # Single file → writes Cacna_87.pose.parquet next to the input
  python strip_to_pose.py /path/to/Cacna_87.parquet

  # Directory → recurses, writes one .pose.<ext> per matching file
  python strip_to_pose.py /path/to/csv/

  # Custom output directory
  python strip_to_pose.py /path/to/csv/ --output-dir /tmp/pose/

  # Inspect only — don't write anything
  python strip_to_pose.py /path/to/file.parquet --dry-run

  # Be loud about what's happening per file
  python strip_to_pose.py /path/to/csv/ --verbose

Exits non-zero if zero files contained marker columns.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

SUFFIX_X = "_x"
SUFFIX_Y = "_y"
SUFFIX_P = "_p"
ALL_SUFFIXES = (SUFFIX_X, SUFFIX_Y, SUFFIX_P)

# Suffixes that pose-stage columns can pick up from upstream
# pipeline steps and that we should tolerate on input. Stripping
# these reveals the canonical marker name.
TOLERATED_TRAILING_TAGS = (
    "_FEATURE_SUBSET",      # mufasa.feature_extractors.feature_subset_*
    "_shifted",             # _shifted columns from movement extractors
)


def _strip_trailing_tags(name: str) -> str:
    """Remove any number of tolerated trailing tags from a column
    name. ``foo_x_FEATURE_SUBSET_shifted`` → ``foo_x``."""
    changed = True
    while changed:
        changed = False
        for tag in TOLERATED_TRAILING_TAGS:
            if name.endswith(tag):
                name = name[: -len(tag)]
                changed = True
    return name


def find_marker_columns(
    columns: List[str],
) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    """Identify marker triples in ``columns``.

    A marker is a base name that has all three of ``<name>_x``,
    ``<name>_y``, ``<name>_p`` present (case-insensitive on the
    suffix). Tolerated trailing tags (``_FEATURE_SUBSET``,
    ``_shifted``) are stripped before matching.

    Returns
    -------
    markers : dict
        ``{canonical_marker_name: {"x": orig_col, "y": orig_col,
        "p": orig_col}}`` — keyed by the lowercased base name with
        tolerated trailing tags stripped, mapping each suffix
        letter to the ORIGINAL (untransformed) column name in
        ``columns``.
    other : list of str
        Original column names that were NOT part of any complete
        triple. Includes incomplete triples (e.g. only ``_x`` and
        ``_y`` present, missing ``_p``).
    """
    by_marker: Dict[str, Dict[str, str]] = {}
    for orig in columns:
        # Normalize to detect the suffix in any case
        canonical = _strip_trailing_tags(str(orig))
        canonical_lower = canonical.lower()
        for suffix in ALL_SUFFIXES:
            if canonical_lower.endswith(suffix):
                base = canonical_lower[: -len(suffix)]
                # Empty base name (e.g. column literally named "_x")
                # is not a real marker
                if not base:
                    continue
                by_marker.setdefault(base, {})[suffix[1]] = orig
                break

    complete: Dict[str, Dict[str, str]] = {
        name: triple
        for name, triple in by_marker.items()
        if {"x", "y", "p"} <= set(triple.keys())
    }

    used_originals = set()
    for triple in complete.values():
        used_originals.update(triple.values())

    other = [c for c in columns if c not in used_originals]
    return complete, other


def detect_format(path: Path) -> str:
    """Return 'csv' or 'parquet' based on file extension."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return "parquet"
    if suffix in (".csv", ".tsv"):
        return "csv"
    raise ValueError(
        f"Unsupported file extension {suffix!r} for {path}; "
        f"expected .csv, .tsv, or .parquet"
    )


def read_df(path: Path) -> pd.DataFrame:
    """Read a pose-data file. Handles flat CSV (header row 0),
    parquet, and 3-row IMPORTED_POSE multi-index CSVs."""
    fmt = detect_format(path)
    if fmt == "parquet":
        return pd.read_parquet(path)

    # CSV — peek at the first lines to decide flat vs multi-row
    # Use the same heuristic as csv_to_parquet._detect_header_rows:
    # the first row whose data cells (past the index) are all
    # numeric is the start of data.
    n_header = _detect_header_rows(path)
    if n_header == 1:
        return pd.read_csv(path, index_col=0, low_memory=False)
    df = pd.read_csv(
        path, index_col=0,
        header=list(range(n_header)),
        low_memory=False,
    )
    # Flatten multi-index by using the last header row (the
    # bodypart_x / _y / _p names — what downstream tools expect)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def write_df(df: pd.DataFrame, path: Path) -> None:
    fmt = detect_format(path)
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def _detect_header_rows(csv_path: Path, max_probe_lines: int = 10) -> int:
    """Lightweight copy of csv_to_parquet._detect_header_rows so
    this script remains standalone (no mufasa imports)."""
    import csv as _csv
    header_count = 0
    with open(csv_path, "r", newline="") as f:
        reader = _csv.reader(f)
        for i, row in enumerate(reader):
            if i >= max_probe_lines:
                break
            if not row:
                continue
            data_cells = row[1:] if len(row) > 1 else row
            try:
                for cell in data_cells:
                    if cell == "" or cell.lower() == "nan":
                        continue
                    float(cell)
                return header_count
            except ValueError:
                header_count += 1
    return 1 if header_count == 0 else header_count


def process_file(
    in_path: Path,
    out_path: Optional[Path],
    dry_run: bool,
    verbose: bool,
) -> Tuple[int, int]:
    """Process one file. Returns (n_marker_cols_kept, n_other_cols_stripped).

    Returns (0, n_other) if no marker triples were found — caller
    treats this as "skipped, no markers."
    """
    df = read_df(in_path)
    markers, other = find_marker_columns(list(df.columns))

    if verbose:
        print(f"  {in_path}")
        print(f"    total columns: {len(df.columns)}")
        print(f"    marker triples found: {len(markers)}")
        print(f"    other (non-marker) columns: {len(other)}")
        if other and verbose:
            preview = other[:5]
            print(f"    other preview: {preview}")

    if not markers:
        return 0, len(other)

    # Build output column order: bp1_x, bp1_y, bp1_p, bp2_x, ...
    # in alphabetical bp order for stability.
    out_cols: List[str] = []
    for bp in sorted(markers.keys()):
        triple = markers[bp]
        out_cols.extend([triple["x"], triple["y"], triple["p"]])

    out_df = df[out_cols].copy()
    # Rename to canonical lowercase form so consumers don't have
    # to care about Original_Case or _FEATURE_SUBSET pollution.
    rename_map = {}
    for bp in sorted(markers.keys()):
        triple = markers[bp]
        rename_map[triple["x"]] = f"{bp}_x"
        rename_map[triple["y"]] = f"{bp}_y"
        rename_map[triple["p"]] = f"{bp}_p"
    out_df = out_df.rename(columns=rename_map)

    if dry_run:
        if verbose:
            print(f"    [dry-run] would write {len(out_cols)} cols "
                  f"to {out_path}")
        return len(markers), len(other)

    if out_path is None:
        raise ValueError("out_path required when dry_run=False")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_df(out_df, out_path)
    if verbose:
        print(f"    wrote {len(out_cols)} cols → {out_path}")
    return len(markers), len(other)


def discover_input_files(root: Path) -> List[Path]:
    """Recursively find pose-data files. Skips files that already
    look like our outputs (.pose.csv / .pose.parquet)."""
    if root.is_file():
        return [root]
    files: List[Path] = []
    for ext in (".csv", ".parquet", ".tsv"):
        files.extend(root.rglob(f"*{ext}"))
    # Filter out our own outputs
    files = [f for f in files if not _looks_like_pose_output(f)]
    # Filter hidden files
    files = [f for f in files if not any(p.startswith(".") for p in f.parts)]
    return sorted(files)


def _looks_like_pose_output(path: Path) -> bool:
    return ".pose." in path.name


def derive_output_path(
    in_path: Path,
    output_dir: Optional[Path],
    in_place: bool,
) -> Path:
    """Decide where the stripped output should go."""
    fmt = detect_format(in_path)
    ext = ".parquet" if fmt == "parquet" else ".csv"
    out_name = in_path.stem + ".pose" + ext
    if in_place:
        # Replace original — caller is responsible for backups
        return in_path
    if output_dir is not None:
        return output_dir / out_name
    return in_path.parent / out_name


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Strip non-marker columns from pose-data CSV/parquet "
            "files, leaving only <bp>_x, <bp>_y, <bp>_p triples."
        ),
    )
    parser.add_argument(
        "path", help="Input file or directory (recursive)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help=(
            "Where to write outputs. Default: alongside each "
            "input as <name>.pose.<ext>. Ignored if --in-place."
        ),
    )
    parser.add_argument(
        "--in-place", action="store_true",
        help=(
            "OVERWRITE the input file with the stripped version. "
            "Make a backup first if you care about the originals."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Read and analyze files but write nothing.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-file details.",
    )
    args = parser.parse_args(argv)

    root = Path(args.path).resolve()
    if not root.exists():
        print(f"ERROR: {root} does not exist", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    if output_dir and args.in_place:
        print(
            "ERROR: --output-dir and --in-place are mutually exclusive",
            file=sys.stderr,
        )
        return 1

    files = discover_input_files(root)
    if not files:
        print(f"No CSV/parquet files found under {root}")
        return 1

    print(f"Found {len(files)} candidate file(s)")
    if args.dry_run:
        print("(dry-run — nothing will be written)")

    n_kept = 0
    n_skipped_no_markers = 0
    n_failed = 0
    for f in files:
        try:
            out_path = derive_output_path(f, output_dir, args.in_place)
            markers_found, other_stripped = process_file(
                f, out_path, dry_run=args.dry_run, verbose=args.verbose,
            )
        except Exception as e:
            print(f"  FAILED {f}: {type(e).__name__}: {e}", file=sys.stderr)
            n_failed += 1
            continue
        if markers_found == 0:
            n_skipped_no_markers += 1
            if not args.verbose:
                print(f"  SKIP {f.name}: no marker triples "
                      f"({other_stripped} non-marker cols)")
        else:
            n_kept += 1
            if not args.verbose:
                print(f"  OK   {f.name}: {markers_found} markers kept, "
                      f"{other_stripped} stripped")

    print()
    written_label = "would-write" if args.dry_run else "written"
    print(f"Summary: {n_kept} {written_label}, {n_skipped_no_markers} skipped "
          f"(no markers), {n_failed} failed")
    if n_kept == 0:
        print(
            "WARNING: zero files contained marker triples. The files "
            "you targeted may be feature-only outputs (e.g. "
            "feature_subsets) rather than pose data.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
