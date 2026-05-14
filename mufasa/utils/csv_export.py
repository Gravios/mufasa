"""
mufasa.utils.csv_export
=======================

CSV export helpers for v1 projects — read whatever's currently on
disk (per-family parquet, wide parquet, or legacy CSV — whichever
:func:`mufasa.utils.feature_io.load_features_for_video` finds) and
write it out as a wide CSV at a user-picked destination. Useful
when a v1 project's data needs to go to an external tool that
expects the SimBA-style features-extracted CSV shape.

Shipped in patch 122ae-6. Two public helpers:

:func:`export_features_csv` — write the feature wide DataFrame
    for one video.

:func:`export_labels_csv` — write the classifier-target columns
    for one video.

:func:`export_combined_csv` — features and labels concatenated
    column-wise, matching the legacy ``csv/targets_inserted/``
    shape (features first, label columns last).

Pad-column convention
---------------------
``read_df()`` strips a leading pad column on read (the unnamed
positional index that ``write_df(has_index=True)`` writes). To
make exported CSVs round-trip cleanly through legacy SimBA tools
that use ``read_df`` to consume them, the export defaults to
``include_index=True`` — the CSV's first column is the positional
index. Set ``include_index=False`` for a pad-free shape that
matches what e.g. pandas' ``to_csv(index=False)`` produces
directly (useful for scripts that read with ``pd.read_csv()``).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

from mufasa.utils.feature_io import load_features_for_video
from mufasa.utils.label_io import load_labels_for_video


__all__ = [
    "export_features_csv",
    "export_labels_csv",
    "export_combined_csv",
]


def _strip_video_ext(video_name: str) -> str:
    return Path(video_name).stem


def _write_csv(df: pd.DataFrame, dest_path: str,
               include_index: bool) -> None:
    """Single write site for the three exporters. Centralises the
    pad-column decision so the convention stays consistent."""
    df.to_csv(dest_path, index=include_index)


def export_features_csv(
    video_name: str,
    config_path: str,
    dest_dir: str,
    *,
    include_index: bool = True,
) -> str:
    """Export the feature columns for one video as a CSV.

    Reads via :func:`load_features_for_video` so the source is
    automatically the right place for v1 (per-family parquet
    and/or wide parquet) or legacy (``csv/features_extracted/``)
    projects.

    :param video_name: Video stem; ``.mp4`` etc. tolerated and
        stripped.
    :param config_path: Path to ``project.toml`` (v1) or
        ``project_config.ini`` (legacy).
    :param dest_dir: Directory to write into. Created if needed.
    :param include_index: When True (default), include a leading
        positional pad column so legacy ``read_df`` consumers
        round-trip cleanly. When False, write without index —
        matches a vanilla ``df.to_csv(index=False)``.

    :returns: The path written to.

    :raises FileNotFoundError: if no features can be found for
        ``video_name`` in any layout. Propagated from
        load_features_for_video; message names every probed
        path so users can diagnose without reading source.
    """
    df = load_features_for_video(video_name, config_path)
    stem = _strip_video_ext(video_name)
    os.makedirs(dest_dir, exist_ok=True)
    out_path = os.path.join(dest_dir, f"{stem}.csv")
    _write_csv(df, out_path, include_index)
    return out_path


def export_labels_csv(
    video_name: str,
    config_path: str,
    dest_dir: str,
    *,
    include_index: bool = True,
) -> str:
    """Export classifier labels for one video as a CSV.

    Reads via :func:`load_labels_for_video`. Output has one
    column per classifier target from the project's metadata.
    Targets absent from the source file land as all-NA columns
    (matches load_labels_for_video's missing-target stability).

    :param video_name: Video stem.
    :param config_path: Path to project config.
    :param dest_dir: Destination directory (auto-created).
    :param include_index: Pad column toggle, same semantic as
        export_features_csv.

    :returns: Path written to.

    :raises FileNotFoundError: if no labels found in any layout.
    """
    df = load_labels_for_video(video_name, config_path)
    stem = _strip_video_ext(video_name)
    os.makedirs(dest_dir, exist_ok=True)
    out_path = os.path.join(dest_dir, f"{stem}.csv")
    _write_csv(df, out_path, include_index)
    return out_path


def export_combined_csv(
    video_name: str,
    config_path: str,
    dest_dir: str,
    *,
    include_index: bool = True,
) -> str:
    """Export features + labels combined for one video.

    Mirrors the legacy ``csv/targets_inserted/<video>.<ext>``
    shape: features columns first, classifier-target columns
    appended at the end. Row alignment is positional (both
    DataFrames cover the same frame range; this is true by
    construction when both come from the same video).

    Row-count mismatches between features and labels surface as
    a :class:`ValueError` rather than a silent shape mangle —
    callers can decide whether to skip the video or abort the
    batch.

    :param video_name: Video stem.
    :param config_path: Path to project config.
    :param dest_dir: Destination directory.
    :param include_index: Pad column toggle.

    :returns: Path written to.

    :raises FileNotFoundError: if features OR labels can't be
        found in any layout. (Either missing is fatal; exporting
        only-features or only-labels has dedicated functions
        above for that.)
    :raises ValueError: if feature and label DataFrames have
        different row counts — would otherwise produce a
        misaligned concat.
    """
    features = load_features_for_video(video_name, config_path)
    labels = load_labels_for_video(video_name, config_path)
    if len(features) != len(labels):
        raise ValueError(
            f"Features and labels for {video_name!r} have "
            f"different row counts: {len(features)} vs "
            f"{len(labels)}. Cannot combine for export — fix "
            f"the source data or use the features-only / "
            f"labels-only exports instead."
        )
    combined = pd.concat(
        [features.reset_index(drop=True),
         labels.reset_index(drop=True)],
        axis=1,
    )
    stem = _strip_video_ext(video_name)
    os.makedirs(dest_dir, exist_ok=True)
    out_path = os.path.join(dest_dir, f"{stem}.csv")
    _write_csv(combined, out_path, include_index)
    return out_path
