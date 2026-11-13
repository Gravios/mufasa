"""
mufasa.utils.feature_io
========================

v1-native read API for per-video features. Reads from
``derived/features/`` only — both the per-family parquet tree
(from FeatureSubsetsCalculator) and the wide-parquet sidecar
(from the bulk feature extractors). The legacy
``csv/features_extracted/`` fallback was removed in patch
122ak; v1 layout is the only supported layout.

Shipped in patch 122ae-2. v1-only in 122ak.

Public API
----------
:func:`family_slug` — display name (FEATURE_FAMILIES constant)
    to filesystem-safe directory name. Used by both the reader
    (this module) and the writers (122ae-3, 122ae-4) so the
    on-disk layout stays in sync.

:func:`load_features_for_video` — read all (or selected)
    feature families for one video into one wide DataFrame.
    Raises FileNotFoundError if no v1 features exist.

:func:`write_wide_features_v1` — write a wide DataFrame as a
    v1-native sidecar parquet.

:func:`list_video_stems_with_features` — enumerate video stems
    with features anywhere under ``derived/features/``.

Per-family parquet layout
-------------------------
::

    <project_root>/derived/features/
    ├── <video_001>.parquet                # wide-parquet sidecar
    │                                        from bulk extractors
    ├── two_point_body_part_distances_mm/
    │   ├── <video_001>.parquet
    │   └── <video_002>.parquet
    ├── within_animal_three_point_body_part_angles_degrees/
    │   ├── <video_001>.parquet
    │   └── …
    └── …

Per-family files (subdirs) take precedence over the wide
parquet for any column-name conflict — per-family files come
from explicit user re-computations and represent the most
recent values; the wide parquet is the baseline.
"""
from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from mufasa.project_layout import project_paths_from_config

__all__ = [
    "RoiFamilyStatus",
    "WIDE_FAMILY",
    "family_slug",
    "feature_catalog",
    "list_video_stems_with_features",
    "load_feature_columns",
    "load_features_for_video",
    "roi_family_status",
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
    # Patch 122jc: strip only a real video / table extension. Path.stem
    # also cut dotted video names ("rec.v2" -> "rec"), so their labels,
    # features and classifications landed under the wrong name.
    from mufasa.utils.annotation_status import strip_media_ext
    return strip_media_ext(video_name)


# Patch 122ak: _legacy_csv_path and _read_legacy removed. v1
# projects are the only supported layout; features live under
# derived/features/. If callers need to import legacy CSVs as
# input, they should run feature extraction (which now writes
# v1-native parquets) from scratch.


def load_features_for_video(
    video_name: str,
    config_path: str,
    *,
    families: list[str] | None = None,
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
        ``project.toml`` (legacy).
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
    wide_parquet_df: pd.DataFrame | None = None

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

    # ---- Nothing found ----
    # Patch 122ak: legacy fallback removed. v1 projects store
    # features under derived/features/<video>.parquet (wide
    # extractor) and/or derived/features/<slug>/<video>.parquet
    # (per-family writer). If neither produced a file for this
    # video, raise — features must be extracted first.
    wide_path_str = (
        os.path.join(derived_root, f"{stem}.parquet")
        if derived_root else None
    )
    raise FileNotFoundError(
        f"No features found for video {stem!r}. Looked under "
        f"{derived_root!r} (per-family tree) and "
        f"{wide_path_str!r} (wide parquet). Run feature "
        f"extraction first."
    )


def write_wide_features_v1(
    df: pd.DataFrame,
    video_name: str,
    config_path: str,
) -> str | None:
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
    their ``csv/`` tree. v1 detection is by config file extension —
    same suffix check that ``ConfigReader.__init__`` does (122dy
    deleted ``ConfigReader._is_v1`` because v1 is now the only
    supported path; this helper's local check is retained because
    it's still called from contexts that may receive a legacy
    path before ConfigReader gets to reject it).

    :param df: The wide DataFrame to write. Written as-is; no
        column-name suffixing or sorting (the legacy wide CSV
        write already handled any such normalisation upstream).
    :param video_name: Video stem; ``.mp4`` etc. tolerated and
        stripped.
    :param config_path: Path to ``project.toml`` (v1) or
        ``project.toml`` (legacy). Used to (a) detect
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


# ---------------------------------------------------------------------- #
# Patch 122je — column catalogue and projected loads (Feature
# Visualization). Read parquet schemas / only the requested columns, so
# browsing hundreds of features across dozens of sessions stays cheap.
# ---------------------------------------------------------------------- #
WIDE_FAMILY = "(wide feature file)"


def _feature_files(video_name: str, config_path: str) -> list[tuple[str, str]]:
    """[(family, parquet path)] for one video: per-family files first
    (family = folder name), then the wide sidecar."""
    stem = _strip_video_ext(video_name)
    try:
        root = project_paths_from_config(config_path).get("derived_features_dir")
    except Exception:
        root = None
    out: list[tuple[str, str]] = []
    if root and os.path.isdir(root):
        for sub in sorted(os.listdir(root)):
            f = os.path.join(root, sub, f"{stem}.parquet")
            if os.path.isdir(os.path.join(root, sub)) and os.path.isfile(f):
                out.append((sub, f))
        wide = os.path.join(root, f"{stem}.parquet")
        if os.path.isfile(wide):
            out.append((WIDE_FAMILY, wide))
    return out


def feature_catalog(video_name: str, config_path: str) -> dict[str, list[str]]:
    """{family: [columns]} for one video, from parquet schemas only (no
    data read). A column in both a family file and the wide file is
    listed under the family (per-family wins, as in
    :func:`load_features_for_video`)."""
    import pyarrow.parquet as pq
    seen: set[str] = set()
    out: dict[str, list[str]] = {}
    for fam, path in _feature_files(video_name, config_path):
        try:
            names = [c for c in pq.read_schema(path).names
                     if c not in seen and not c.startswith("__index")]
        except Exception:
            continue
        if names:
            out[fam] = names
            seen.update(names)
    return out


def load_feature_columns(video_name: str, config_path: str,
                         columns: list[str]) -> pd.DataFrame:
    """Only ``columns`` for one video, reading just the files and columns
    that hold them. Missing columns are absent from the result (not an
    error); order follows ``columns``."""
    import pyarrow.parquet as pq
    want = list(dict.fromkeys(columns))
    got: dict[str, pd.Series] = {}
    for _fam, path in _feature_files(video_name, config_path):
        todo = [c for c in want if c not in got]
        if not todo:
            break
        try:
            have = set(pq.read_schema(path).names)
        except Exception:
            continue
        cols = [c for c in todo if c in have]
        if cols:
            df = pd.read_parquet(path, columns=cols)
            for c in cols:
                got[c] = df[c].reset_index(drop=True)
    return pd.DataFrame({c: got[c] for c in want if c in got})


# ---------------------------------------------------------------------- #
# Patch 122ji — is a ROI family's output still current?
# ---------------------------------------------------------------------- #
@dataclass
class RoiFamilyStatus:
    """How a ROI feature family's outputs compare with the current ROIs.

    * ``changed`` — computed from different ROIs (a ROI added, removed,
      renamed, moved or reshaped since);
    * ``missing`` — videos that now have ROIs but no output;
    * ``removed`` — outputs for videos that no longer have ROIs;
    * ``unknown`` — outputs written before ROI tracking (122ji) whose file
      is older than rois.json, so the ROIs may have changed.
    """
    computed: list[str] = field(default_factory=list)
    current: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)

    @property
    def modified(self) -> bool:
        return bool(self.computed) and bool(
            self.changed or self.missing or self.removed or self.unknown)

    def summary(self) -> str:
        parts = []
        if self.changed:
            parts.append(f"ROIs changed in {len(self.changed)} video(s)")
        if self.missing:
            parts.append(f"{len(self.missing)} video(s) with ROIs not computed")
        if self.removed:
            parts.append(f"ROIs removed from {len(self.removed)} video(s)")
        if self.unknown:
            parts.append(f"{len(self.unknown)} video(s) computed before rois.json "
                         "was last saved")
        return "; ".join(parts)


def roi_family_status(config_path: str, slug: str) -> RoiFamilyStatus:
    """Compare ``derived/features/<slug>/`` with the project's rois.json."""
    from mufasa.feature_extractors.feature_subset_orchestration import (
        read_roi_fingerprint,
    )
    from mufasa.roi_tools.roi_store import RoiFileError, load_rois
    st = RoiFamilyStatus()
    paths = project_paths_from_config(config_path)
    fam_dir = os.path.join(paths["derived_features_dir"], slug)
    roi_path = paths.get("roi_definitions_path", "")
    try:
        store = load_rois(roi_path)
    except (RoiFileError, OSError):
        store = None
    with_rois = set(store.videos_with_rois()) if store else set()
    roi_mtime = os.path.getmtime(roi_path) if roi_path and os.path.isfile(roi_path) else 0.0
    outs = {}
    if os.path.isdir(fam_dir):
        outs = {os.path.splitext(f)[0]: os.path.join(fam_dir, f)
                for f in os.listdir(fam_dir)
                if f.endswith(".parquet") and not f.startswith(".")}
    st.computed = sorted(outs)
    for v, f in sorted(outs.items()):
        if v not in with_rois:
            st.removed.append(v)
            continue
        try:
            fp = read_roi_fingerprint(f)
        except Exception:
            fp = None
        if fp is None:
            (st.unknown if os.path.getmtime(f) < roi_mtime else st.current).append(v)
        elif fp == store.fingerprint(v):
            st.current.append(v)
        else:
            st.changed.append(v)
    st.missing = sorted(with_rois - set(outs)) if outs else []
    return st


def list_video_stems_with_features(
    config_path: str,
) -> list[str]:
    """Return the sorted union of video stems for which features
    exist anywhere in the project (v1 layout AND legacy layout
    combined).

    Patch 122ae-5b: needed by consumers that previously did
    directory-level discovery via
    :func:`mufasa.utils.read_write.find_files_of_filetypes_in_directory`
    on the legacy ``csv/features_extracted/`` tree. For v1
    projects, features live across multiple locations
    (per-family parquet subdirs, wide-parquet sidecar at the
    derived/features/ root, AND the legacy tree during the
    migration window), so a single-directory scan misses
    videos. This helper consolidates discovery.

    Lookup order:

    1. Wide-parquet sidecar at
       ``<derived_features_dir>/<stem>.parquet``.
    2. Per-family files at
       ``<derived_features_dir>/<family_slug>/<stem>.parquet``.
       Multiple family-subdir occurrences of the same stem
       count once.
    3. Legacy wide CSV / parquet at
       ``<features_extracted_dir>/<stem>.<ext>``.

    The output is the union — a stem appears once even if it
    has files in multiple layouts. Consumers can then call
    :func:`load_features_for_video` per stem; that helper
    merges the available sources.

    :param config_path: Path to ``project.toml`` (v1) or
        ``project.toml`` (legacy).
    :returns: Sorted list of video stems. Empty list if the
        project has no features anywhere.
    """
    try:
        paths = project_paths_from_config(config_path)
    except Exception:
        return []
    stems: set[str] = set()

    derived_root = paths.get("derived_features_dir")
    if derived_root and os.path.isdir(derived_root):
        # Wide parquet (122ae-4 sidecar): files directly at the
        # derived_features_dir root with .parquet extension.
        for name in os.listdir(derived_root):
            full = os.path.join(derived_root, name)
            if os.path.isfile(full) and name.endswith(".parquet"):
                stems.add(Path(name).stem)
        # Per-family parquets (122ae-3): one subdir per family,
        # each containing <stem>.parquet files.
        for sub in os.listdir(derived_root):
            sub_path = os.path.join(derived_root, sub)
            if not os.path.isdir(sub_path):
                continue
            for name in os.listdir(sub_path):
                full = os.path.join(sub_path, name)
                if (
                    os.path.isfile(full)
                    and name.endswith(".parquet")
                ):
                    stems.add(Path(name).stem)

    # Patch 122ak: legacy csv/features_extracted/ scan removed.
    # v1 layout (derived/features/) is the only source of truth.

    return sorted(stems)
