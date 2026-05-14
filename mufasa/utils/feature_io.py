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


__all__ = [
    "family_slug",
    "load_features_for_video",
    "write_wide_features_v1",
]


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

    Patch 122ae-5: aligned with :func:`mufasa.utils.read_write.read_df`'s
    default behaviour of stripping the leading pad column that
    SimBA's ``write_df(has_index=True)`` writes. The pad column
    is a positional index serialised as the first CSV column;
    SimBA's own readers drop it on load (see read_write.py
    line 146 in the post-122ae-1 tree: ``df = df.iloc[:, 1:]``).
    Without this strip, swapping a ``read_df(...)`` call site to
    ``load_features_for_video(...)`` would leave consumers with
    an extra unnamed first column that breaks their column-name
    assumptions.

    The strip is CSV-only — parquet files written by 122ae-3
    and 122ae-4 don't have a pad column (they're written with
    ``index=False``), and H5 files come through pd.read_hdf
    with their original index handling intact.

    Mirrors read_df's default ``has_index=True`` semantic.
    Callers that legitimately wrote a CSV without a pad column
    would now get their first column silently dropped — but no
    known caller does that for the features_extracted tree
    specifically (it's exclusively written via write_df), so
    this is safe.
    """
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path)
        # Strip the leading pad column. Matches read_df default.
        if df.shape[1] > 0:
            df = df.iloc[:, 1:]
        return df
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
    wide_parquet_df: Optional[pd.DataFrame] = None

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

        # Patch 122ae-4: also check for a wide parquet at the
        # derived root (NOT inside a per-family subdir). This is
        # what the full feature extractor writes — it produces one
        # wide DataFrame per video and the per-family attribution
        # isn't easily recoverable from column names, so it lands
        # as a sidecar wide file rather than as per-family files.
        # The families= filter is intentionally ignored for the
        # wide file (since the wide file has all families merged;
        # filtering would require column-name pattern matching
        # which is deferred).
        wide_path = os.path.join(derived_root, f"{stem}.parquet")
        if os.path.isfile(wide_path):
            try:
                wide_parquet_df = pd.read_parquet(wide_path)
            except Exception as exc:
                warnings.warn(
                    f"Failed to read wide parquet {wide_path}: "
                    f"{exc}. Falling through to per-family / "
                    f"legacy for {stem}.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    if parquet_frames or wide_parquet_df is not None:
        # Patch 122ae-4: merge precedence —
        #   per-family columns OVERRIDE wide-parquet columns when
        #   both exist with the same name. Rationale: per-family
        #   files come from FeatureSubsetsCalculator runs that
        #   were SPECIFICALLY re-computed by the user; the wide
        #   parquet from the full extractor is the canonical
        #   "all features" baseline. When the user re-runs a
        #   family, they want the new values, not the baseline.
        per_family = (
            pd.concat(parquet_frames, axis=1)
            if parquet_frames else pd.DataFrame()
        )
        # Drop duplicate columns WITHIN the per-family stack first
        # (could happen if two slug dirs claim overlapping names).
        if per_family.shape[1]:
            dupes_pf = per_family.columns[
                per_family.columns.duplicated()
            ].tolist()
            if dupes_pf:
                warnings.warn(
                    f"Duplicate columns across feature families "
                    f"for {stem}: {dupes_pf!r}. Keeping the first "
                    f"occurrence; check that family slug names "
                    f"don't collide.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                per_family = per_family.loc[
                    :, ~per_family.columns.duplicated()
                ]
        # Now merge with wide. Per-family wins for duplicates.
        if wide_parquet_df is None:
            return per_family
        if per_family.empty:
            return wide_parquet_df
        # Both present — drop wide columns that per-family also has,
        # then concat the rest. Aligned by row position.
        wide_unique_cols = [
            c for c in wide_parquet_df.columns
            if c not in per_family.columns
        ]
        merged = pd.concat(
            [per_family.reset_index(drop=True),
             wide_parquet_df[wide_unique_cols].reset_index(drop=True)],
            axis=1,
        )
        return merged

    # ---- Legacy fallback ----
    legacy_path = _legacy_csv_path(stem, config_path=config_path)
    if legacy_path and os.path.isfile(legacy_path):
        return _read_legacy(legacy_path)

    # ---- Nothing found ----
    wide_path_str = (
        os.path.join(derived_root, f"{stem}.parquet")
        if derived_root else None
    )
    raise FileNotFoundError(
        f"No features found for video {stem!r}. Looked under "
        f"{derived_root!r} (per-family tree), "
        f"{wide_path_str!r} (wide parquet), and "
        f"{legacy_path!r} (legacy wide file)."
    )


def write_wide_features_v1(
    df: pd.DataFrame,
    video_name: str,
    config_path: str,
) -> Optional[str]:
    """Write a wide-features DataFrame as a v1-native sidecar
    parquet at ``<derived_features_dir>/<video>.parquet``.

    Patch 122ae-4: bulk feature extractors (``feature_extractor_*bp``,
    ``feature_extractor_user_defined``) produce a single wide
    DataFrame per video — there's no easy per-family attribution
    to recover from column names, so they can't write the
    per-family layout that ``FeatureSubsetsCalculator`` writes
    (via 122ae-3). Instead, they sidecar a wide parquet at the
    derived-features root, picked up by
    :func:`load_features_for_video`'s wide-parquet branch.

    No-op for legacy (non-TOML) projects so projects that haven't
    migrated don't gain a stray ``derived/`` subtree alongside
    their ``csv/`` tree. v1 detection is by config file extension
    — matches what
    :class:`mufasa.mixins.config_reader.ConfigReader._is_v1` does.

    :param df: The wide DataFrame to write. Written as-is; no
        column-name suffixing or sorting (the legacy wide CSV
        write already handled any such normalisation upstream).
    :param video_name: Video stem; ``.mp4`` etc. tolerated and
        stripped.
    :param config_path: Path to ``project.toml`` (v1) or
        ``project_config.ini`` (legacy). Used to (a) detect
        v1-ness and (b) resolve ``derived_features_dir`` via the
        layout helper.

    :returns: The path written to, or ``None`` if no write
        happened (legacy project, or the layout helper couldn't
        resolve a path).
    """
    if not str(config_path).lower().endswith(".toml"):
        # Legacy project — skip the v1 write so the project tree
        # stays clean. Callers' legacy wide-CSV write was
        # already performed upstream.
        return None
    try:
        paths = project_paths_from_config(config_path)
    except Exception as exc:
        warnings.warn(
            f"Cannot resolve derived_features_dir from "
            f"{config_path!r}: {exc}. Skipping v1 wide-parquet "
            f"sidecar write for {video_name!r}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    derived_features_dir = paths.get("derived_features_dir")
    if not derived_features_dir:
        return None
    os.makedirs(derived_features_dir, exist_ok=True)
    stem = _strip_video_ext(video_name)
    out_path = os.path.join(derived_features_dir, f"{stem}.parquet")
    df.to_parquet(out_path, index=False)
    return out_path
