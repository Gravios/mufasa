"""
mufasa.model.session_ingest
===========================

Add new recordings to an existing project and bring their derived data up to
date, without re-running the whole project.

The contract is "same format and same pose as the originals": a new file is
only accepted if its markers match the project's ``body_parts`` exactly. That
check is the point of this module. A mismatched import is not a loud failure
— it silently produces pose columns the rest of the pipeline can't find,
which surfaces much later as all-NaN arrays (see the layout regression fixed
in 122gv/122gx). Better to refuse at the door and say exactly which markers
differ.

Two steps, separable:

* :func:`check_pose_compatibility` — dry inspection: what files are there,
  do they parse, do their markers match the project?
* :func:`ingest_sessions` — import the accepted files, then optionally
  refresh derived data for **only those sessions**, reusing the latest
  existing model rather than retraining.

Only smoothing is wired to a model here, because it is the only stage whose
"latest model" is unambiguous (see :func:`find_latest_smoothing_model`).
Feature extraction and classifier inference are deliberately left out; see
the module notes in ``docs/`` and the caller's stage flags.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_POSE_SUFFIXES = (".fdlc.parquet", ".parquet", ".csv", ".h5")


def _candidate_files(source: str | os.PathLike) -> list[str]:
    """Expand a file or directory into a sorted list of pose files."""
    s = os.fspath(source)
    if os.path.isfile(s):
        return [s]
    if not os.path.isdir(s):
        return []
    out = [
        os.path.join(s, f)
        for f in sorted(os.listdir(s))
        if not f.startswith(".") and f.lower().endswith(_POSE_SUFFIXES)
    ]
    return out


def _markers_of(path: str) -> list[str]:
    """Read a pose file's marker names without loading the whole thing.

    Handles the two shapes Mufasa sees: FreeDLC's long/tidy table (a
    ``bodypart`` column) and the wide ``<bp>_x/_y/_p`` layout (flat or under
    the IMPORTED_POSE MultiIndex).
    """
    import pandas as pd

    low = path.lower()
    if low.endswith(".parquet"):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        names = [c for c in pf.schema_arrow.names]
        if "bodypart" in names:
            # long format — the distinct bodyparts are the marker set
            col = pq.read_table(path, columns=["bodypart"]).column("bodypart")
            import pyarrow.compute as pc
            return [str(v) for v in pc.unique(col).to_pylist()]
        try:
            head = next(pf.iter_batches(batch_size=1)).to_pandas()
        except StopIteration:
            head = pd.DataFrame(columns=names)
        cols = head.columns
    elif low.endswith(".csv"):
        head = pd.read_csv(path, nrows=1)
        cols = head.columns
    else:
        raise ValueError(f"Unsupported pose file: {os.path.basename(path)}")

    labels = [
        str(c[-1]) if isinstance(c, tuple) else str(c) for c in cols
    ]
    markers: list[str] = []
    for lab in labels:
        for suf in ("_x", "_y", "_p", "_likelihood"):
            if lab.endswith(suf):
                bp = lab[: -len(suf)]
                if bp and bp not in markers:
                    markers.append(bp)
                break
    return markers


def check_pose_compatibility(
    source: str | os.PathLike, config_path: str | os.PathLike
) -> dict[str, Any]:
    """Inspect ``source`` (a file or folder) against the project's pose.

    :returns: ``{"files": [...], "accepted": [...], "rejected": {path: why},
        "project_markers": [...]}``. A file is accepted only when its marker
        set equals the project's ``body_parts`` — same names, no missing, no
        extras. Order is not required to match: the importers align by name.
    """
    from mufasa.project_layout import project_metadata_from_config

    meta = project_metadata_from_config(config_path)
    project_markers = [str(b) for b in meta.get("body_parts", [])]
    expected = set(project_markers)

    files = _candidate_files(source)
    accepted: list[str] = []
    rejected: dict[str, str] = {}
    for fp in files:
        try:
            found = _markers_of(fp)
        except Exception as exc:  # noqa: BLE001 - report, don't abort the scan
            rejected[fp] = f"could not read: {type(exc).__name__}: {exc}"
            continue
        if not found:
            rejected[fp] = "no marker columns found"
            continue
        got = set(found)
        if got == expected:
            accepted.append(fp)
            continue
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        bits = []
        if missing:
            bits.append(f"missing {missing}")
        if extra:
            bits.append(f"unexpected {extra}")
        rejected[fp] = "pose differs from the project: " + "; ".join(bits)

    return {
        "files": files,
        "accepted": accepted,
        "rejected": rejected,
        "project_markers": project_markers,
    }


def find_latest_smoothing_model(config_path: str | os.PathLike) -> str | None:
    """Return the most recently modified smoothing model in the project.

    Models are stored as ``<project>/models/<name>/model.npz`` (see
    :func:`mufasa.project_layout.import_model_into_project`). "Latest" is by
    modification time, which matches how they're produced — train, and the
    newest one is the current one. Returns ``None`` when the project has no
    model, in which case the caller must train rather than reuse.
    """
    from mufasa.project_layout import project_paths_from_config

    try:
        models_dir = Path(project_paths_from_config(config_path)["models_dir"])
    except (ValueError, OSError, KeyError):
        return None
    if not models_dir.is_dir():
        return None
    candidates = sorted(
        models_dir.glob("*/model.npz"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(candidates[0]) if candidates else None


def ingest_sessions(
    config_path: str | os.PathLike,
    source: str | os.PathLike,
    *,
    smooth: bool = True,
    fps: float = 30.0,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Import new sessions and refresh their derived data.

    :param config_path: Project ``project.toml``.
    :param source: A pose file or a folder of them.
    :param smooth: Re-run smoothing for the new sessions using the project's
        latest model (:func:`find_latest_smoothing_model`). Skipped with a
        note when no model exists — this function never trains.
    :param fps: Frame rate passed to the smoother.
    :param dry_run: Report what would happen; write nothing.
    :returns: Summary with the compatibility report, imported files, the
        model used, and the smoothed output directory.
    """
    from mufasa.project_layout import generate_run_id, project_paths_from_config

    report = check_pose_compatibility(source, config_path)
    summary: dict[str, Any] = {
        "checked": len(report["files"]),
        "accepted": list(report["accepted"]),
        "rejected": dict(report["rejected"]),
        "imported": [],
        "smoothing_model": None,
        "smoothed_dir": None,
        "notes": [],
        "dry_run": dry_run,
    }
    if not report["accepted"]:
        summary["notes"].append("Nothing to import: no file matched the project's pose.")
        return summary
    if dry_run:
        return summary

    paths = project_paths_from_config(config_path)
    input_dir = paths["input_pose_dir"]
    os.makedirs(input_dir, exist_ok=True)

    # Import through the project's own importer so alignment, likelihood
    # handling and the skeleton sidecar are treated exactly as they were for
    # the original sessions.
    from mufasa.pose_importers.fdlc_parquet_importer import FDLCParquetImporter

    fdlc = [f for f in report["accepted"] if f.lower().endswith(".fdlc.parquet")]
    if fdlc:
        staging = {os.path.dirname(f) for f in fdlc}
        for folder in sorted(staging):
            FDLCParquetImporter(
                config_path=os.fspath(config_path), data_folder=folder,
            ).run()
        summary["imported"].extend(fdlc)
    else:
        # Already in Mufasa's wide layout — copy into the project untouched.
        import shutil
        for f in report["accepted"]:
            dest = os.path.join(input_dir, os.path.basename(f))
            shutil.copy2(f, dest)
            summary["imported"].append(dest)

    if not smooth:
        return summary

    model = find_latest_smoothing_model(config_path)
    summary["smoothing_model"] = model
    if model is None:
        summary["notes"].append(
            "Smoothing skipped: the project has no saved model to reuse. "
            "Train one first (Preprocessing -> Kalman v2), then re-run."
        )
        return summary

    stems = [
        os.path.splitext(os.path.basename(f))[0].replace(".fdlc", "")
        for f in summary["imported"]
    ]
    ft = str(
        __import__("mufasa.project_layout", fromlist=["x"])
        .project_metadata_from_config(config_path)
        .get("file_type", "parquet")
    )
    new_pose = [
        os.path.join(input_dir, f"{s}.{ft}")
        for s in stems
        if os.path.isfile(os.path.join(input_dir, f"{s}.{ft}"))
    ]
    if not new_pose:
        summary["notes"].append(
            "Smoothing skipped: could not locate the imported pose files."
        )
        return summary

    out_dir = os.path.join(
        paths["project_root"], "derived", "smoothed", "kalman_v2",
        generate_run_id(),
    )
    from mufasa.data_processors.kalman_pose_smoother_v2 import smooth_pose_v2

    smooth_pose_v2(
        pose_input=new_pose,
        output_dir=out_dir,
        load_model=model,
        fps=fps,
        verbose=True,
    )
    summary["smoothed_dir"] = out_dir
    return summary


__all__ = [
    "check_pose_compatibility",
    "find_latest_smoothing_model",
    "ingest_sessions",
]
