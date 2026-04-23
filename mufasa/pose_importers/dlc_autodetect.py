"""
mufasa.pose_importers.dlc_autodetect
====================================

Body-part auto-detection from DeepLabCut output files (.h5 or .csv).

Purpose: Mufasa's built-in body-part presets cap at 9 per single animal,
and many DLC projects either exceed that count or use bodypart names
that don't match any preset. Rather than asking users to transcribe
body-part lists into ``project_bp_names.csv`` by hand (error-prone —
ordering mistakes silently corrupt every downstream coordinate), this
module reads the body-part list straight from any DLC output file in
the project's column order.

Public API:

    extract_bodyparts(path)           # dispatches on extension
    extract_bodyparts_from_h5(path)
    extract_bodyparts_from_csv(path)

All three return ``List[str]`` in DLC's original column order, with
duplicates removed (each body part appears three times in DLC output,
once each for x/y/likelihood).

Raises ``DLCAutodetectError`` with an explanatory message on:
  * multi-animal H5 (4 column levels — use maDLC importer instead)
  * malformed headers
  * unsupported extensions
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import List, Union


class DLCAutodetectError(Exception):
    """Raised when a DLC file can't be parsed for body-part names."""


def _dedupe_preserve_order(names: List[str]) -> List[str]:
    """Remove duplicates while keeping first-appearance order."""
    seen: dict = {}
    for n in names:
        if n and n not in seen:
            seen[n] = True
    return list(seen.keys())


def extract_bodyparts_from_h5(h5_path: Union[str, os.PathLike]) -> List[str]:
    """Read body-part names from a DLC single-animal H5 file.

    DLC single-animal H5 has a 3-level column ``MultiIndex``:
    (scorer → bodyparts → coords). multi-animal H5 has 4 levels with
    an extra ``individuals`` level — rejected here because the
    resulting body-part list wouldn't map cleanly onto a single-animal
    Mufasa project anyway.
    """
    import pandas as pd  # deferred — only needed when this is called
    p = Path(h5_path)
    try:
        df = pd.read_hdf(p)
    except Exception as exc:
        raise DLCAutodetectError(
            f"Could not read H5 file {p.name}: {type(exc).__name__}: {exc}"
        ) from exc

    if not isinstance(df.columns, pd.MultiIndex):
        raise DLCAutodetectError(
            f"{p.name}: expected a MultiIndex column header (DLC writes "
            f"a 3-level scorer/bodyparts/coords index), got flat "
            f"columns. Is this really a DLC output H5?"
        )
    if df.columns.nlevels == 4:
        raise DLCAutodetectError(
            f"{p.name}: has 4 column levels, looks like a multi-animal "
            f"DLC (maDLC) file. This autodetection path is for "
            f"single-animal projects only — use the maDLC importer "
            f"instead."
        )
    if df.columns.nlevels != 3:
        raise DLCAutodetectError(
            f"{p.name}: unexpected column level count "
            f"({df.columns.nlevels}). DLC single-animal H5 should have "
            f"3 levels (scorer, bodyparts, coords)."
        )

    # Level 1 is "bodyparts". Each bp appears 3× (x, y, likelihood);
    # preserve first-appearance order.
    bps = _dedupe_preserve_order(list(df.columns.get_level_values(1)))
    if not bps:
        raise DLCAutodetectError(
            f"{p.name}: parsed column header but no body-part names "
            f"were present. Is the file empty?"
        )
    return bps


def extract_bodyparts_from_csv(csv_path: Union[str, os.PathLike]) -> List[str]:
    """Read body-part names from a DLC CSV file.

    DLC CSVs store a 3-row multi-header:
      row 0: scorer name (repeated)
      row 1: body-part names (each repeated 3×)
      row 2: coord type (x / y / likelihood)

    The first column of the data portion is the frame index label.
    We read just the header rows — no need to slurp the whole file.
    """
    p = Path(csv_path)
    try:
        with open(p, "r", newline="") as f:
            reader = csv.reader(f)
            rows = [next(reader) for _ in range(3)]
    except StopIteration:
        raise DLCAutodetectError(
            f"{p.name}: has fewer than 3 rows. DLC CSV needs a 3-row "
            f"multi-header."
        )
    except Exception as exc:
        raise DLCAutodetectError(
            f"Could not read CSV file {p.name}: {type(exc).__name__}: "
            f"{exc}"
        ) from exc

    if len(rows) < 3:
        raise DLCAutodetectError(
            f"{p.name}: expected 3 header rows, found {len(rows)}."
        )
    # Row 1 holds body-part names. First cell is a label like
    # 'bodyparts' or empty — skip it.
    bodyparts_row = rows[1][1:] if rows[1] else []
    bps = _dedupe_preserve_order(bodyparts_row)
    if not bps:
        raise DLCAutodetectError(
            f"{p.name}: couldn't parse body-parts from row 2 of the "
            f"header. First 3 rows: {rows!r}"
        )
    return bps


def extract_bodyparts(dlc_path: Union[str, os.PathLike]) -> List[str]:
    """Extract body-part names from a DLC output file, dispatching on
    extension. Supported: ``.h5``, ``.csv``.
    """
    p = Path(dlc_path)
    ext = p.suffix.lower()
    if ext == ".h5":
        return extract_bodyparts_from_h5(p)
    elif ext == ".csv":
        return extract_bodyparts_from_csv(p)
    else:
        raise DLCAutodetectError(
            f"{p.name}: unsupported extension {ext!r}. Expected .h5 "
            f"or .csv."
        )


__all__ = [
    "DLCAutodetectError",
    "extract_bodyparts",
    "extract_bodyparts_from_h5",
    "extract_bodyparts_from_csv",
]
