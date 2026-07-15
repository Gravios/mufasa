"""
mufasa.utils.data_sample
========================

Discovery and cheap head-sampling of a project's data files, for the Data
inspector page.

Two jobs:

* :func:`list_project_data_files` — what data exists, grouped by pipeline
  stage (imported pose, smoothed, outlier-corrected, features,
  classifications, ...), including the run id a derived file came from.
* :func:`load_sample` — read the first N rows of one file *without* pulling
  the whole thing into memory. Pose files run to tens of thousands of frames;
  an inspector that reads them entirely to show 50 rows is unusable.

Sampling is per-format:

* **parquet** — ``ParquetFile.metadata.num_rows`` gives the exact row count
  with no data read at all, and ``iter_batches`` reads only the first batch.
* **csv** — ``read_csv(nrows=)`` stops early. The total row count is left
  unknown rather than paying a full scan to find it.
* **h5** — ``read_hdf(stop=)`` where the file's format allows it, otherwise
  a full read is the only option and the sample is taken afterwards.

Everything is imported lazily so merely opening the page costs nothing.
"""
from __future__ import annotations

import os
from typing import Any

_DATA_SUFFIXES = (".parquet", ".csv", ".h5")

# Derived stage directory -> display label. Order is pipeline order, which
# is also the order the inspector lists them in.
_DERIVED_STAGES: tuple[tuple[str, str], ...] = (
    ("smoothed", "Smoothed"),
    ("outlier_corrected", "Outlier-corrected"),
    ("features", "Features"),
    ("classifications", "Classifications"),
    ("labels", "Labels"),
)


def _is_data_file(name: str) -> bool:
    return (
        not name.startswith(".")
        and name.lower().endswith(_DATA_SUFFIXES)
    )


def list_project_data_files(config_path) -> dict[str, list[str]]:
    """Return ``{stage_label: [file_path, ...]}`` for a project.

    "Pose (imported)" comes from the project's input pose directory; the
    derived stages are discovered by walking ``<root>/derived/<stage>/``,
    which may nest a flavour directory (e.g. ``smoothed/kalman_v2/``) and
    then per-run directories. Stages with no files are omitted, so the
    inspector only offers sources that actually have something in them.
    """
    from mufasa.project_layout import project_paths_from_config

    out: dict[str, list[str]] = {}
    try:
        paths = project_paths_from_config(config_path)
    except (ValueError, OSError):
        return out

    pose_dir = paths.get("input_pose_dir", "")
    if pose_dir and os.path.isdir(pose_dir):
        files = sorted(
            os.path.join(pose_dir, f)
            for f in os.listdir(pose_dir)
            if _is_data_file(f)
        )
        if files:
            out["Pose (imported)"] = files

    root = paths.get("project_root", "")
    derived = os.path.join(root, "derived") if root else ""
    if derived and os.path.isdir(derived):
        for stage_dir, label in _DERIVED_STAGES:
            stage_path = os.path.join(derived, stage_dir)
            if not os.path.isdir(stage_path):
                continue
            found: list[str] = []
            for dirpath, _dirs, filenames in os.walk(stage_path):
                found.extend(
                    os.path.join(dirpath, f)
                    for f in filenames
                    if _is_data_file(f)
                )
            if found:
                out[label] = sorted(found)
    return out


def describe_path(path: str, config_path=None) -> str:
    """A short label for a file: its name, plus the run id when the file
    lives under a run directory (which is how derived files are told
    apart — same video name, different run)."""
    from mufasa.project_layout import is_run_id

    name = os.path.basename(path)
    parts = os.path.normpath(path).split(os.sep)
    runs = [p for p in parts if is_run_id(p)]
    if runs:
        return f"{name}  ·  {runs[-1]}"
    return name


def load_sample(path: str, n_rows: int = 50) -> tuple[Any, dict]:
    """Read the first ``n_rows`` of ``path``.

    :returns: ``(DataFrame, info)``. ``info`` carries ``total_rows`` (exact
        for parquet, ``None`` when determining it would cost a full scan),
        ``n_columns``, ``file_size``, ``sampled_rows`` and ``nan_fraction``
        (over the sample — an all-NaN column is the single most common
        symptom of a marker/name mismatch, so it's worth surfacing).
    :raises ValueError: for unreadable or unsupported files.
    """
    import pandas as pd

    if not os.path.isfile(path):
        raise ValueError(f"{path} does not exist.")
    n_rows = max(1, int(n_rows))
    size = os.path.getsize(path)
    low = path.lower()
    total: int | None = None

    if low.endswith(".parquet"):
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ValueError(f"pyarrow is required to read parquet: {exc}")
        try:
            pf = pq.ParquetFile(path)
            total = pf.metadata.num_rows          # no data read
            if total == 0:
                df = pf.read().to_pandas()
            else:
                batch = next(pf.iter_batches(batch_size=n_rows))
                df = batch.to_pandas()
        except StopIteration:
            df = pd.DataFrame()
        except Exception as exc:
            raise ValueError(f"Could not read parquet: {exc}")
    elif low.endswith(".csv"):
        try:
            df = pd.read_csv(path, nrows=n_rows)
        except Exception as exc:
            raise ValueError(f"Could not read csv: {exc}")
    elif low.endswith(".h5"):
        try:
            df = pd.read_hdf(path, stop=n_rows)
        except (TypeError, ValueError, NotImplementedError):
            # Fixed-format stores don't support start/stop — full read is
            # the only route.
            try:
                df = pd.read_hdf(path).head(n_rows)
            except Exception as exc:
                raise ValueError(f"Could not read h5: {exc}")
        except Exception as exc:
            raise ValueError(f"Could not read h5: {exc}")
    else:
        raise ValueError(f"Unsupported file type: {os.path.basename(path)}")

    try:
        n_cells = int(df.size)
        nan_frac = float(df.isna().sum().sum()) / n_cells if n_cells else 0.0
    except Exception:
        nan_frac = 0.0

    info = {
        "total_rows": total,
        "sampled_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "file_size": size,
        "nan_fraction": nan_frac,
    }
    return df, info


def format_columns(df) -> list[str]:
    """Flatten column labels for display; MultiIndex columns (the
    IMPORTED_POSE 3-level header) collapse to their innermost level, which
    is the part that names the marker."""
    import pandas as pd

    if isinstance(df.columns, pd.MultiIndex):
        return [str(t[-1]) for t in df.columns]
    return [str(c) for c in df.columns]


__all__ = [
    "describe_path",
    "format_columns",
    "list_project_data_files",
    "load_sample",
]
