"""
mufasa.utils.classification_io
==============================

v1-native read/write API for per-video classifier predictions.
Reads from and writes to
``derived/classifications/<video>.parquet`` — the v1 home for
classifier inference output.

Patch 122at (shipped initial helpers + dual-write era).

What's stored
-------------
Per-video parquet at
``derived/classifications/<video_name>.parquet``:

* For each classifier target ``T``:

  - ``Probability_T`` (float, 0.0–1.0) — probability of class ``T``
    being present in this frame.
  - ``T`` (Int64) — the thresholded + min-bout-corrected boolean
    prediction (0 / 1).

* Row index = frame number (0-indexed), positional — not stored
  as a column. Joins with feature data by row position.

Features are NOT included — they already live in
``derived/features/<video>.parquet``. Consumers that need
features+predictions read both and join positionally.

Public API
----------
:func:`load_classifications_for_video` — read predictions for one
  video as a DataFrame.

:func:`save_classifications_for_video` — write a predictions
  DataFrame for one video.

:func:`list_video_stems_with_classifications` — enumerate stems
  with predictions on disk.

Dual-write era
--------------
Patch 122at is the OPEN of the dual-write era for classifier
predictions. :class:`InferenceBatch` writes BOTH to the legacy
``csv/machine_results/`` (combined features + predictions, same
shape consumers have always read) AND to v1
``derived/classifications/`` (predictions-only). Consumers
migrate to the v1 read path in subsequent patches; once all
consumers read v1, the legacy write is dropped.

Mirrors the labels migration arc:
* 122ae-3.5 — open dual-write for labels.
* 122ae-5b → 5e — consumer migrations.
* 122ak — drop legacy write.

Classification migration:
* 122at — open dual-write (this patch).
* 122au+ — consumer migrations (each subsystem).
* 122a?? — drop legacy write.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import List, Optional

import pandas as pd

from mufasa.project_layout import (project_metadata_from_config,
                                   project_paths_from_config)


__all__ = [
    "list_video_stems_with_classifications",
    "load_classifications_for_video",
    "save_classifications_for_video",
]


def _strip_video_ext(video_name: str) -> str:
    """Accept either 'video_001' or 'video_001.mp4'. Returns the
    bare stem either way."""
    return Path(video_name).stem


def _classifications_dir(config_path: str) -> Path:
    """Resolve the v1 classifications directory from the layout
    helper. Creates the directory on first use so save sites
    don't need to."""
    paths = project_paths_from_config(config_path)
    out = Path(paths["derived_classifications_dir"])
    out.mkdir(parents=True, exist_ok=True)
    return out


def _prediction_columns(df: pd.DataFrame,
                        classifier_targets: List[str]) -> List[str]:
    """Return the subset of ``df.columns`` that represent
    predictions for the given classifier targets (probability +
    boolean columns). Used by the dual-write site to extract
    just the v1-relevant subset from the combined CSV that the
    Tk-era pipeline produces."""
    cols: List[str] = []
    for clf in classifier_targets:
        prob = f"Probability_{clf}"
        if prob in df.columns:
            cols.append(prob)
        if clf in df.columns:
            cols.append(clf)
    return cols


def save_classifications_for_video(
    video_name: str,
    config_path: str,
    predictions: pd.DataFrame,
) -> str:
    """Write per-frame classifier predictions for one video to
    ``derived/classifications/<video>.parquet``.

    :param video_name: bare stem or filename — ``Path().stem`` is
        used either way.
    :param config_path: path to project's ``project.toml`` (v1)
        or ``project_config.ini`` (legacy).
    :param predictions: DataFrame with one row per frame. Columns
        should be a subset of:

        * ``Probability_<classifier>`` — float, prob of the
          classifier firing.
        * ``<classifier>`` — Int64 nullable, the thresholded +
          bout-corrected boolean.

        Extra columns are written as-is (no curation) — this lets
        callers pass through diagnostic columns like the per-frame
        feature subset score during the dual-write era without
        the helper losing them. Consumers should be tolerant of
        extra columns.

    :returns: absolute path of the written parquet file.

    The parquet index is not preserved (frame order is positional).
    """
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError(
            f"predictions must be a DataFrame, got "
            f"{type(predictions).__name__}",
        )
    stem = _strip_video_ext(video_name)
    out_dir = _classifications_dir(config_path)
    out_path = out_dir / f"{stem}.parquet"
    predictions.reset_index(drop=True).to_parquet(out_path)
    return str(out_path)


def load_classifications_for_video(
    video_name: str,
    config_path: str,
    *,
    targets: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load classifier predictions for one video.

    :param video_name: bare stem or filename.
    :param config_path: project config path.
    :param targets: if given, return only the columns
        ``Probability_<t>`` and ``<t>`` for each target. Missing
        columns are filled with NaN so the result shape is stable.
        If None, returns all columns present in the parquet.

    :returns: DataFrame, frame-indexed positionally (0..N-1).

    :raises FileNotFoundError: if no v1 predictions exist for
        this video. Callers that want a legacy-fallback should
        wrap this in try/except and read the combined CSV from
        ``machine_results_dir`` themselves until they migrate to
        v1 fully.
    """
    stem = _strip_video_ext(video_name)
    out_dir = _classifications_dir(config_path)
    path = out_dir / f"{stem}.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"No v1 classifications for video {stem!r} at {path}",
        )
    df = pd.read_parquet(path)
    if targets is not None:
        wanted: List[str] = []
        for t in targets:
            wanted.append(f"Probability_{t}")
            wanted.append(t)
        missing = [c for c in wanted if c not in df.columns]
        if missing:
            for c in missing:
                df[c] = pd.NA
        df = df[wanted]
    return df


def list_video_stems_with_classifications(
    config_path: str,
) -> List[str]:
    """List the video stems that have classifier predictions on
    disk under ``derived/classifications/``. Returns a sorted
    list — stable ordering across runs."""
    try:
        out_dir = _classifications_dir(config_path)
    except Exception:
        return []
    if not out_dir.is_dir():
        return []
    stems = [
        p.stem for p in sorted(out_dir.glob("*.parquet"))
        if p.is_file()
    ]
    return stems
