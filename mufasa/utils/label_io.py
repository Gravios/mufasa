"""
mufasa.utils.label_io
=====================

Single read/write API for per-video classifier labels. Reads from
``derived/labels/<video>.parquet`` first; falls back to extracting
the classifier-target columns from the legacy wide
``csv/targets_inserted/<video>.<ext>`` file. Writes only to the
new per-video parquet location.

Shipped in patch 122ae-3.5. The labels split from
``csv/targets_inserted`` (which conflated labels with the
features they were paired with) into a focused per-video parquet
under ``derived/labels/`` is one of the v1 layout decisions
reified in 122ae-1. This module is the API surface; the
frame-labeller retarget that actually wires these helpers up
lands in 122ae-5.

Public API
----------
:func:`load_labels_for_video` — read labels for one video as a
    DataFrame with one column per classifier target (Int64 /
    nullable int — 0 / 1 / NaN for unannotated frames).

:func:`save_labels_for_video` — write a labels DataFrame for one
    video to ``derived/labels/<video>.parquet``. Optionally
    merges with existing labels rather than overwriting.

Schema on disk
--------------
Per-video parquet at ``derived/labels/<video_name>.parquet``:

* One column per classifier target (named exactly as the target
  appears in ``project_metadata['classifier_targets']``).
* Row index = frame number (0-indexed), positional — not stored
  as a column. Consumers join with feature data by row position.
* Values: Int64 nullable. 0 / 1 mean the frame was annotated as
  not-occurring / occurring. ``pd.NA`` means the frame was never
  annotated (annotator skipped over it, or the labeller hasn't
  visited it yet).

Legacy fallback (read-only)
---------------------------
Reading from ``csv/targets_inserted/<video>.<ext>`` extracts only
the classifier-target columns. Other columns (features, pose
data) are discarded — they don't belong in the labels file.

Writes ONLY go to the new location. There's no write-back to the
legacy wide file. After 122ae-5 lands and the labeller switches
to ``save_labels_for_video``, projects accumulate labels under
``derived/labels/`` and the legacy ``csv/targets_inserted/`` tree
stops being updated.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import List, Optional

import pandas as pd

from mufasa.project_layout import (project_metadata_from_config,
                                   project_paths_from_config)


__all__ = ["load_labels_for_video", "save_labels_for_video"]


def _strip_video_ext(video_name: str) -> str:
    """Allow callers to pass either 'video_001' or 'video_001.mp4'.
    Returns the bare stem either way. Mirrors feature_io's
    behaviour for API consistency."""
    return Path(video_name).stem


def _read_legacy(path: str) -> pd.DataFrame:
    """Read a legacy wide targets file. Same extension dispatch as
    feature_io._read_legacy."""
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext in (".h5", ".hdf5"):
        return pd.read_hdf(path)
    raise ValueError(
        f"Unsupported legacy targets extension {ext!r} at {path}",
    )


def load_labels_for_video(
    video_name: str,
    config_path: str,
    *,
    targets: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load classifier labels for one video.

    Reads from the v1 per-video parquet
    (``derived/labels/<video>.parquet``) when that file exists.
    Falls back to extracting the classifier-target columns from
    the legacy wide ``csv/targets_inserted/<video>.<ext>`` file
    when the v1 file is absent. Caller doesn't have to care
    which path was used.

    :param video_name: Video stem; ``.mp4`` / ``.avi`` / etc.
        extension is tolerated and stripped.
    :param config_path: Path to ``project.toml`` (v1) or
        ``project_config.ini`` (legacy).
    :param targets: Optional list of classifier-target names to
        project to. ``None`` returns every classifier target the
        project knows about (per
        ``project_metadata['classifier_targets']``). Targets the
        project knows about but the file doesn't have land as
        all-NA columns in the returned DataFrame.

    :returns: DataFrame with one column per requested classifier
        target. Dtype is ``Int64`` (pandas nullable int) so 0,
        1, and pd.NA all round-trip correctly. Row count
        matches whatever's in the underlying file; consumers
        align with feature data by row position.

    :raises FileNotFoundError: if neither the v1 file nor the
        legacy wide file contains any labels for the requested
        video.
    """
    stem = _strip_video_ext(video_name)

    # Resolve target list. If the caller passed an explicit
    # filter, honour it as-is; otherwise read the project's
    # classifier_targets metadata.
    try:
        meta = project_metadata_from_config(config_path)
        all_targets = list(meta.get("classifier_targets", []))
    except Exception:
        meta = {}
        all_targets = []
    requested = (
        list(targets) if targets is not None else all_targets
    )

    # ---- v1 per-video parquet ----
    try:
        paths = project_paths_from_config(config_path)
    except Exception:
        paths = {}
    derived_labels_dir = paths.get("derived_labels_dir")
    v1_path = (
        os.path.join(derived_labels_dir, f"{stem}.parquet")
        if derived_labels_dir else None
    )
    if v1_path and os.path.isfile(v1_path):
        df = pd.read_parquet(v1_path)
        return _project_to_targets(df, requested)

    # ---- Legacy fallback ----
    legacy_dir = paths.get("targets_inserted_dir")
    import_ft = meta.get(
        "import_file_type", meta.get("file_type", "csv"),
    )
    legacy_path = (
        os.path.join(legacy_dir, f"{stem}.{import_ft}")
        if legacy_dir else None
    )
    if legacy_path and os.path.isfile(legacy_path):
        wide = _read_legacy(legacy_path)
        return _project_to_targets(wide, requested)

    raise FileNotFoundError(
        f"No labels found for video {stem!r}. Looked at "
        f"{v1_path!r} (v1 per-video parquet) and "
        f"{legacy_path!r} (legacy wide targets file)."
    )


def _project_to_targets(df: pd.DataFrame,
                        targets: List[str]) -> pd.DataFrame:
    """Slice the input DataFrame to just the requested target
    columns. Missing targets land as all-NA Int64 columns so the
    output schema is stable regardless of which targets the
    underlying file actually had.

    Empty target list → DataFrame with ``len(df)`` rows and zero
    columns (matches what classifier-training code expects when
    the project has no targets defined yet).
    """
    out = pd.DataFrame(index=range(len(df)))
    for t in targets:
        if t in df.columns:
            # Cast to Int64 (nullable) so 0 / 1 / NA all
            # round-trip. Existing column might be int64 (no NAs
            # tolerated), float64 (with NaN), or object — let
            # pandas figure out the conversion.
            col = df[t]
            try:
                out[t] = col.astype("Int64")
            except (TypeError, ValueError):
                # Last-ditch: coerce via to_numeric. NaNs survive
                # as pd.NA after the Int64 cast.
                out[t] = pd.to_numeric(
                    col, errors="coerce",
                ).astype("Int64")
        else:
            # Target not present in this file — emit all-NA so
            # the column shape is stable.
            out[t] = pd.array(
                [pd.NA] * len(df), dtype="Int64",
            )
    return out


def save_labels_for_video(
    video_name: str,
    config_path: str,
    labels: pd.DataFrame,
    *,
    merge: bool = True,
) -> str:
    """Write labels for one video to
    ``derived/labels/<video>.parquet``.

    :param video_name: Video stem; extension tolerated and
        stripped.
    :param config_path: Path to project.toml / project_config.ini.
    :param labels: DataFrame with one column per classifier
        target. Values should be 0 / 1 / NaN (or pd.NA). Dtype
        flexibility same as load_labels_for_video — cast to
        Int64 nullable on write.
    :param merge: When True (default) and a labels file already
        exists at the target path, columns from the new
        ``labels`` DataFrame are merged onto the existing one
        — new columns added, existing columns OVERWRITTEN by
        the new values (one column at a time; row positions
        align). When False the existing file is overwritten
        wholesale. The merge behaviour matches the frame-
        labeller's mental model: each save adds or updates one
        classifier's labels without disturbing the others.

    :returns: The path written to.

    :raises ValueError: if the project's
        derived_labels_dir can't be resolved (malformed config).
    """
    stem = _strip_video_ext(video_name)
    try:
        paths = project_paths_from_config(config_path)
    except Exception as exc:
        raise ValueError(
            f"Cannot resolve project paths from {config_path!r}: "
            f"{exc}"
        )
    derived_labels_dir = paths.get("derived_labels_dir")
    if not derived_labels_dir:
        raise ValueError(
            f"Project at {config_path!r} doesn't expose "
            f"'derived_labels_dir'. Is the layout helper "
            f"up-to-date?"
        )
    os.makedirs(derived_labels_dir, exist_ok=True)
    out_path = os.path.join(derived_labels_dir, f"{stem}.parquet")

    # Coerce input columns to Int64 nullable for consistent
    # on-disk dtype.
    coerced = pd.DataFrame(index=labels.index)
    for col in labels.columns:
        s = labels[col]
        try:
            coerced[col] = s.astype("Int64")
        except (TypeError, ValueError):
            coerced[col] = pd.to_numeric(
                s, errors="coerce",
            ).astype("Int64")

    if merge and os.path.isfile(out_path):
        # Existing labels file present; merge column-wise. Rows
        # are aligned by position (assume both DataFrames cover
        # the same frame range, which they will for a single
        # video; the labeller never produces partial-frame
        # files).
        existing = pd.read_parquet(out_path)
        if len(existing) != len(coerced):
            warnings.warn(
                f"Existing labels for {stem!r} have "
                f"{len(existing)} rows but the new write has "
                f"{len(coerced)}; row counts don't align. "
                f"Falling back to overwrite of the new "
                f"DataFrame (no merge). Consumer should "
                f"investigate.",
                RuntimeWarning,
                stacklevel=2,
            )
            coerced.to_parquet(out_path, index=False)
            return out_path
        # Drop columns from existing that the new write also
        # has — the new values win — then concatenate.
        keep_cols = [
            c for c in existing.columns
            if c not in coerced.columns
        ]
        merged = pd.concat(
            [existing[keep_cols].reset_index(drop=True),
             coerced.reset_index(drop=True)],
            axis=1,
        )
        merged.to_parquet(out_path, index=False)
        return out_path

    # No existing file, or overwrite requested: write straight.
    coerced.to_parquet(out_path, index=False)
    return out_path
