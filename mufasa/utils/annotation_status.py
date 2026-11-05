"""
mufasa.utils.annotation_status
==============================

Per-video annotation progress for the Frame Labeling page (patch 122jc).

Two facts per video:

* **labels** — whether ``derived/labels/<video>.parquet`` exists, and how
  many frames carry each behaviour (read from that file);
* **complete** — set by the annotator with the labeller's "Mark video
  complete" checkbox, kept in ``derived/labels/annotation_status.toml``
  next to the labels themselves::

      [videos."K7_cam_20251124_1510_A10-0-051"]
      complete = true
      updated = "2026-11-05T16:40:12+0100"

The status file is small TOML so it survives hand edits and diffs; it is
not a parquet, so label discovery (``list_video_stems_with_labels``) does
not mistake it for a video.
"""
from __future__ import annotations

import os
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from mufasa.project_layout import project_paths_from_config, write_project_toml

STATUS_FILENAME = "annotation_status.toml"
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg")
_STRIP_EXTS = VIDEO_EXTS + (".parquet", ".csv", ".h5", ".pkl", ".pickle", ".feather")


def strip_media_ext(name: str) -> str:
    """'rec.mp4' / 'dir/rec.mp4' → 'rec'; 'rec.v2' → 'rec.v2' (only known
    video / table extensions are removed, so dotted names survive)."""
    base = Path(str(name)).name
    for ext in _STRIP_EXTS:
        if base.lower().endswith(ext):
            return base[: -len(ext)]
    return base


def _labels_dir(config_path) -> Path:
    return Path(project_paths_from_config(str(config_path))["derived_labels_dir"])


def _read(p: Path) -> dict:
    """Parse the status file (plain TOML — not a project.toml, so not
    read_project_toml, which insists on a layout version)."""
    try:
        with open(p, "rb") as fh:
            return tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def status_path(config_path) -> Path:
    return _labels_dir(config_path) / STATUS_FILENAME


def read_status(config_path) -> dict[str, dict]:
    """{video: {"complete": bool, "updated": str}}; empty when no file."""
    p = status_path(config_path)
    if not p.is_file():
        return {}
    data = _read(p)
    vids = data.get("videos", {})
    return {str(k): dict(v) for k, v in vids.items() if isinstance(v, dict)}


def is_complete(config_path, video: str) -> bool:
    return bool(read_status(config_path).get(video, {}).get("complete", False))


def set_complete(config_path, video: str, complete: bool) -> Path:
    """Record (or clear) the complete mark for one video."""
    p = status_path(config_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = _read(p) if p.is_file() else {}
    vids = data.setdefault("videos", {})
    vids[video] = {"complete": bool(complete),
                   "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    write_project_toml(p, data)
    return p


@dataclass
class VideoAnnotation:
    video: str
    video_path: str | None
    has_labels: bool
    n_frames: int = 0
    labelled: dict[str, int] = field(default_factory=dict)   # behaviour → frames
    any_labelled: int = 0                                    # frames with ≥ 1 behaviour
    complete: bool = False
    saved: float | None = None                               # labels mtime

    @property
    def state(self) -> str:
        if self.complete:
            return "complete"
        if self.has_labels and any(self.labelled.values()):
            return "in progress"
        if self.has_labels:
            return "started"
        return "not started"


def project_videos(config_path) -> dict[str, str]:
    """{video name: path} for the videos under ``sources/videos``."""
    vd = project_paths_from_config(str(config_path)).get("video_dir")
    out: dict[str, str] = {}
    if vd and os.path.isdir(vd):
        for p in sorted(Path(vd).rglob("*")):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS and not p.name.startswith("."):
                out.setdefault(p.stem, str(p))
    return out


def summarize(config_path, classifiers: list[str] | None = None) -> list[VideoAnnotation]:
    """One row per project video (plus any labelled video whose file is
    gone), with label counts and the complete mark."""
    videos = project_videos(config_path)
    ldir = _labels_dir(config_path)
    labelled = {p.stem: p for p in ldir.glob("*.parquet")} if ldir.is_dir() else {}
    status = read_status(config_path)
    rows = []
    for name in sorted(set(videos) | set(labelled)):
        row = VideoAnnotation(name, videos.get(name), name in labelled,
                              complete=bool(status.get(name, {}).get("complete")))
        if name in labelled:
            p = labelled[name]
            row.saved = p.stat().st_mtime
            try:
                df = pd.read_parquet(p)
                row.n_frames = len(df)
                cols = classifiers if classifiers else list(df.columns)
                hit = {c: (pd.to_numeric(df[c], errors="coerce") == 1).to_numpy()
                       for c in cols if c in df.columns}
                row.labelled = {c: int(v.sum()) for c, v in hit.items()}
                if hit:
                    row.any_labelled = int(np.logical_or.reduce(list(hit.values())).sum())
            except Exception:
                pass
        rows.append(row)
    return rows


__all__ = [
    "STATUS_FILENAME",
    "strip_media_ext",
    "VideoAnnotation",
    "is_complete",
    "project_videos",
    "read_status",
    "set_complete",
    "status_path",
    "summarize",
]
