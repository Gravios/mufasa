"""
mufasa.tools.csv_to_dlc
========================

Convert Mufasa's flat per-video CSVs back into DeepLabCut's
multi-index CSV format.

Mufasa's flat format (one row per frame):

    Nose_x, Nose_y, Nose_p, Tail_base_x, Tail_base_y, Tail_base_p, ...
    123.4, 567.8, 0.99, 234.5, 678.9, 0.98, ...

DLC's multi-index format (3 header rows, one row per frame):

    scorer,    DLC_resnet50_..., DLC_resnet50_..., DLC_resnet50_..., ...
    bodyparts, Nose,             Nose,             Nose,             ...
    coords,    x,                y,                likelihood,       ...
    0,         123.4,            567.8,            0.99,             ...

The conversion is information-preserving for the columns it
covers: Mufasa's `_p` suffix maps to DLC's `likelihood` coord.
The scorer string is read from the project config if available
(under `[create_ensemble_settings] -> scorer`), or synthesized
from a CLI flag, or defaulted to "mufasa_export".

CAVEAT: this produces "what the DLC output would look like for
these coordinates", NOT what DLC originally produced. If your
Mufasa CSVs are from `csv/outlier_corrected_movement_location/`,
those values have been processed by Mufasa's outlier correction —
they're not the raw DLC output. The converter is useful for
re-importing into DLC tooling but won't reconstruct the original
DLC output bit-for-bit.

Usage
-----

CLI:

    python -m mufasa.tools.csv_to_dlc \\
        --in-dir  /path/to/project_folder/csv/outlier_corrected_movement_location \\
        --out-dir /path/to/dlc_export \\
        --scorer  DLC_resnet50_my_project

    # Single file:
    python -m mufasa.tools.csv_to_dlc \\
        --in /path/to/Video_42.csv \\
        --out /path/to/Video_42_dlc.csv

Programmatic:

    from mufasa.tools.csv_to_dlc import flat_to_dlc
    dlc_df = flat_to_dlc(flat_df, scorer="DLC_resnet50_my_project")
    dlc_df.to_csv("out.csv")
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd


_DEFAULT_SCORER = "mufasa_export"


def parse_flat_columns(columns: Iterable[str]) -> List[Tuple[str, str]]:
    """Parse Mufasa's flat column names into (bodypart, coord) pairs.

    Mufasa convention: ``{bodypart}_x``, ``{bodypart}_y``, ``{bodypart}_p``.
    Returns the columns in original order, mapped to (bp, dlc_coord)
    where dlc_coord is one of ``"x"``, ``"y"``, ``"likelihood"``.

    Raises ``ValueError`` for columns that don't match the convention.
    """
    out = []
    for col in columns:
        col = str(col)
        # Suffix-based parsing. The bodypart name itself may contain
        # underscores (e.g. "Tail_base"), so we only split on the
        # FINAL underscore.
        if "_" not in col:
            raise ValueError(
                f"Column {col!r} doesn't match Mufasa's "
                f"`{{bodypart}}_{{x|y|p}}` convention"
            )
        idx = col.rfind("_")
        bp = col[:idx]
        suffix = col[idx + 1:]
        if suffix == "x":
            out.append((bp, "x"))
        elif suffix == "y":
            out.append((bp, "y"))
        elif suffix == "p":
            # Mufasa's _p → DLC's likelihood
            out.append((bp, "likelihood"))
        else:
            raise ValueError(
                f"Column {col!r} has unexpected suffix {suffix!r} "
                f"(expected one of x, y, p)"
            )
    return out


def flat_to_dlc(
    flat_df: pd.DataFrame,
    scorer: str = _DEFAULT_SCORER,
) -> pd.DataFrame:
    """Convert a flat Mufasa DataFrame to DLC multi-index format.

    :param flat_df: DataFrame with columns named like
        ``{bodypart}_x``, ``{bodypart}_y``, ``{bodypart}_p``.
        Order is preserved (DLC convention is bp-grouped, but if
        the input is already bp-grouped the output is too).
    :param scorer: scorer string for the top level of the
        multi-index. Defaults to ``"mufasa_export"``.
    :return: DataFrame with a 3-level column MultiIndex
        ``(scorer, bodypart, coord)``.
    """
    pairs = parse_flat_columns(flat_df.columns)
    multi_index = pd.MultiIndex.from_tuples(
        [(scorer, bp, coord) for bp, coord in pairs],
        names=["scorer", "bodyparts", "coords"],
    )
    out = flat_df.copy()
    out.columns = multi_index
    return out


def discover_scorer(config_path: Optional[str]) -> str:
    """Look up the scorer string from a Mufasa project config, if
    one is available. Falls back to the default."""
    if config_path is None or not os.path.isfile(config_path):
        return _DEFAULT_SCORER
    import configparser
    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    # Mufasa config doesn't reliably store a scorer string. Some
    # projects have one under [create_ensemble_settings] from older
    # SimBA versions; try a few likely keys before falling back.
    for section in ("create_ensemble_settings", "General settings"):
        if section in cfg:
            for key in ("scorer", "model_name"):
                if key in cfg[section]:
                    val = cfg[section][key].strip()
                    if val:
                        return val
    return _DEFAULT_SCORER


def convert_file(
    in_path: str,
    out_path: str,
    scorer: str = _DEFAULT_SCORER,
    has_index: bool = True,
) -> None:
    """Convert a single Mufasa flat CSV to a DLC multi-index CSV.

    :param has_index: True if the input has an unnamed first column
        that pandas wrote as the index. Mufasa's read/write helpers
        round-trip with a leading index column by default, so this
        defaults to True. Set False if your CSV has no leading
        index column.
    """
    if has_index:
        flat_df = pd.read_csv(in_path, index_col=0)
    else:
        flat_df = pd.read_csv(in_path)
    dlc_df = flat_to_dlc(flat_df, scorer=scorer)
    # DLC convention: index column named 'coords' written as the
    # first column (the frame number). pandas writes the multi-index
    # to_csv with three header rows, which is what we want.
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    dlc_df.to_csv(out_path)


def convert_directory(
    in_dir: str,
    out_dir: str,
    scorer: str = _DEFAULT_SCORER,
    has_index: bool = True,
    pattern: str = "*.csv",
) -> List[str]:
    """Convert every CSV in a directory. Returns list of output paths.

    Skips files whose names start with ``.`` (hidden). Output filename
    matches input filename. Overwrites without warning if the output
    file already exists — the caller is responsible for collision
    handling.
    """
    in_path = Path(in_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    written = []
    for src in sorted(in_path.glob(pattern)):
        if src.name.startswith("."):
            continue
        dst = out_path / src.name
        try:
            convert_file(
                in_path=str(src), out_path=str(dst),
                scorer=scorer, has_index=has_index,
            )
            written.append(str(dst))
            print(f"  converted: {src.name}")
        except Exception as exc:
            print(
                f"  SKIPPED {src.name}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
    return written


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mufasa-csv-to-dlc",
        description=(
            "Convert Mufasa's flat per-video CSVs back into "
            "DeepLabCut's multi-index CSV format."
        ),
    )
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        "--in", dest="in_file", type=str,
        help="single input CSV file (use --in-dir for a directory)",
    )
    src_group.add_argument(
        "--in-dir", type=str,
        help="directory containing Mufasa flat CSVs",
    )
    parser.add_argument(
        "--out", dest="out_file", type=str,
        help="output path (required when --in is set)",
    )
    parser.add_argument(
        "--out-dir", type=str,
        help="output directory (required when --in-dir is set)",
    )
    parser.add_argument(
        "--scorer", type=str, default=None,
        help=(
            f"DLC scorer string. If not given, tries to read from "
            f"--config; defaults to {_DEFAULT_SCORER!r}."
        ),
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Mufasa project_config.ini to source the scorer string from",
    )
    parser.add_argument(
        "--no-index", action="store_true",
        help="skip the leading index column when reading (default: assume an index column exists)",
    )
    args = parser.parse_args(argv)

    scorer = args.scorer
    if scorer is None:
        scorer = discover_scorer(args.config)
    has_index = not args.no_index

    if args.in_file:
        if not args.out_file:
            parser.error("--out is required when --in is set")
        convert_file(
            in_path=args.in_file,
            out_path=args.out_file,
            scorer=scorer,
            has_index=has_index,
        )
        print(f"Wrote {args.out_file}")
        return 0
    else:
        if not args.out_dir:
            parser.error("--out-dir is required when --in-dir is set")
        written = convert_directory(
            in_dir=args.in_dir,
            out_dir=args.out_dir,
            scorer=scorer,
            has_index=has_index,
        )
        print(f"\nConverted {len(written)} file(s) to {args.out_dir}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
