"""
mufasa.utils.feature_io
========================

Single read API for per-video features. Reads from the v1 per-
family parquet tree first; falls back to the legacy wide CSV
under ``csv/features_extracted/`` so consumers can adopt this
helper before writers retarget to parquet in patches 122ae-3
and 122ae-4.

Shipped in patch 122ae-2. Replaces ~7 call sites that each
read ``csv/features_extracted/<video>.<ext>`` directly and
branch on ``file_type``: classifier training, classifier
inference, frame labeller, clip review, analysis plots,
visualization plots, and the FeatureSubsetsCalculator's
internal read paths.

Public API
----------
:func:`family_slug` — display name (FEATURE_FAMILIES constant)
    to filesystem-safe directory name. Used by both the reader
    (this module) and the writers (122ae-3, 122ae-4) so the
    on-disk layout stays in sync.

:func:`load_features_for_video` — read all (or selected)
    feature families for one video into one wide DataFrame.

Per-family parquet layout
-------------------------
::

    <project_root>/derived/features/
    ├── two_point_body_part_distances_mm/
    │   ├── video_001.parquet
    │   └── video_002.parquet
    ├── within_animal_three_point_body_part_angles_degrees/
    │   ├── video_001.parquet
    │   └── …
    └── …

Each parquet file holds the columns for exactly one family for
one video. Concatenating per-family files horizontally
(``pd.concat(..., axis=1)``) yields the same wide schema the
legacy CSV produced — by construction, since the writers will
slice the wide DataFrame by family and write each slice
separately.

Legacy fallback
---------------
If the per-family tree doesn't exist OR doesn't contain any
matching files for ``video_name``, the helper falls back to
reading the legacy wide CSV at
``<features_extracted_dir>/<video>.<ext>`` (ext from the
project's ``import_file_type``). This lets:

* Existing projects (CSV-only on disk) keep working without
  migration.
* The reader land in patches 122ae-2 (this patch) before any
  writer retargets to parquet in 122ae-3 + 122ae-4.

After 122ae-4 ships, the legacy fallback stays in place
indefinitely so that historical CSV projects remain readable.
The fallback is read-only — this helper never writes the
legacy format.
"""
from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import List, Optional

import pandas as pd

from mufasa.project_layout import (project_metadata_from_config,
                                   project_paths_from_config)


__all__ = ["family_slug", "load_features_for_video"]


_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


def family_slug(family_name: str) -> str:
    """Convert a ``FEATURE_FAMILIES`` display name to a filesystem-
    safe directory name.

    Display names in :mod:`mufasa.feature_extractors.feature_subsets`
    look like ``'TWO-POINT BODY-PART DISTANCES (MM)'`` —
    uppercase, spaced, parenthesised. Slug them by lowercasing,
    replacing every non-alphanumeric run with a single
    underscore, then stripping leading/trailing underscores.

    Examples:
        >>> family_slug('TWO-POINT BODY-PART DISTANCES (MM)')
        'two_point_body_part_distances_mm'
        >>> family_slug('WITHIN-ANIMAL THREE-POINT CONVEX HULL PERIMETERS (MM)')
        'within_animal_three_point_convex_hull_perimeters_mm'
        >>> family_slug('ENTIRE ANIMAL CONVEX HULL AREA (MM2)')
        'entire_animal_convex_hull_area_mm2'

    The same function is used by writers (122ae-3, 122ae-4) so
    reads and writes hit the same directory names.
    """
    slug = _SLUG_PATTERN.sub("_", family_name.lower()).strip("_")
    return slug or "untitled"


def _strip_video_ext(video_name: str) -> str:
    """Allow callers to pass either 'video_001' or 'video_001.mp4'.
    Returns the bare stem either way."""
    return Path(video_name).stem


def _legacy_csv_path(video_name: str, *, config_path: str) -> Optional[str]:
    """Resolve the legacy wide-CSV path for ``video_name`` based on
    the project's ``import_file_type``. Returns None if the
    project's metadata is unreadable (defensive — keeps the
    caller's error story consistent regardless of cause)."""
    try:
        paths = project_paths_from_config(config_path)
        meta = project_metadata_from_config(config_path)
    except Exception:
        return None
    ext = meta.get("import_file_type", meta.get("file_type", "csv"))
    return os.path.join(
        paths["features_extracted_dir"],
        f"{_strip_video_ext(video_name)}.{ext}",
    )


def _read_legacy(path: str) -> pd.DataFrame:
    """Read a legacy wide features file (CSV / parquet / H5).

    Mirrors :func:`mufasa.utils.read_write.read_df` for the formats
    actually used by the legacy ``features_extracted`` tree but
    sidesteps that helper's quirk of stripping the first CSV column
    regardless of ``has_index``. The frame index in legacy CSVs is
    represented as an unnamed first column; reading it as data and
    discarding it via ``df.iloc[:, 1:]`` would lose information for
    feature files where the first column IS a feature, so we read
    the file straight and let the caller treat the index column the
    same way the legacy reader did.
    """
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext in (".h5", ".hdf5"):
        return pd.read_hdf(path)
    raise ValueError(
        f"Unsupported legacy features extension {ext!r} at {path}",
    )


def load_features_for_video(
    video_name: str,
    config_path: str,
    *,
    families: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load features for one video as a wide DataFrame.

    Reads from the v1 per-family parquet tree
    (``derived/features/<family_slug>/<video>.parquet``) when
    that tree exists. Falls back to the legacy
    ``csv/features_extracted/<video>.<ext>`` wide file when the
    per-family tree is empty or absent. Caller doesn't have to
    care which path was used.

    :param video_name: Video stem; ``.mp4`` / ``.avi`` / etc.
        extension is tolerated and stripped.
    :param config_path: Path to ``project.toml`` (v1) or
        ``project_config.ini`` (legacy).
    :param families: Optional list of FEATURE_FAMILIES display
        names (e.g. ``'TWO-POINT BODY-PART DISTANCES (MM)'``).
        If ``None``, every family found on disk under
        ``derived/features/`` is loaded. The list is silently
        filtered by the slugifier — passing an unknown family
        name is a no-op rather than an error.

    :returns: DataFrame whose columns are the concatenation of
        all matched family files' columns, in the order the
        ``families`` argument requested (or directory-listing
        order when ``families`` is None). Rows align by position
        — all family files for one video must be the same
        length, which by construction they are when written by
        the per-family writer (122ae-3 + 122ae-4).

    :raises FileNotFoundError: if neither the per-family tree
        nor the legacy wide file contains any data for the
        requested video.
    """
    stem = _strip_video_ext(video_name)

    # ---- Per-family parquet tree (v1 / post-migration) ----
    try:
        paths = project_paths_from_config(config_path)
    except Exception:
        paths = {}
    derived_root = paths.get("derived_features_dir")
    parquet_frames: list[pd.DataFrame] = []

    if derived_root and os.path.isdir(derived_root):
        # Pick the subdirs to scan. If the caller passed an explicit
        # families list, slug each entry; otherwise enumerate
        # everything on disk so the helper is family-list-agnostic.
        if families is None:
            subdirs = sorted(
                d for d in os.listdir(derived_root)
                if os.path.isdir(os.path.join(derived_root, d))
            )
        else:
            subdirs = [family_slug(f) for f in families]

        for sub in subdirs:
            fpath = os.path.join(
                derived_root, sub, f"{stem}.parquet",
            )
            if not os.path.isfile(fpath):
                continue
            try:
                parquet_frames.append(pd.read_parquet(fpath))
            except Exception as exc:
                warnings.warn(
                    f"Failed to read {fpath}: {exc}. Skipping "
                    f"this family for {stem}.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    if parquet_frames:
        # Concat horizontally, aligned by position. Duplicate
        # column names are possible if a family was renamed in
        # the writer between runs — flag them, keep first occurrence.
        merged = pd.concat(parquet_frames, axis=1)
        dupes = merged.columns[merged.columns.duplicated()].tolist()
        if dupes:
            warnings.warn(
                f"Duplicate columns across feature families for "
                f"{stem}: {dupes!r}. Keeping the first occurrence; "
                f"check that family slug names don't collide.",
                RuntimeWarning,
                stacklevel=2,
            )
            merged = merged.loc[:, ~merged.columns.duplicated()]
        return merged

    # ---- Legacy fallback ----
    legacy_path = _legacy_csv_path(stem, config_path=config_path)
    if legacy_path and os.path.isfile(legacy_path):
        return _read_legacy(legacy_path)

    # ---- Nothing found ----
    raise FileNotFoundError(
        f"No features found for video {stem!r}. Looked under "
        f"{derived_root!r} (per-family parquet tree) and "
        f"{legacy_path!r} (legacy wide file)."
    )
