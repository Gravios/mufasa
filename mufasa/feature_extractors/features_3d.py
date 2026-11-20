"""
mufasa.feature_extractors.features_3d
=====================================

Patch 122ko: 3-D feature families for *Compute Features*, from a video's
3-D pose file (``derived/pose3d/<run>/<video>_3d.parquet`` from 3-D
inference, or ``sources/pose3d/`` motion capture). Millimetres and
degrees, world frame with z up.

Families (display names are the public API, like the 2-D ones):

* ``3-D BODY-PART HEIGHTS (MM)`` — ``Height (mm) <marker>``: each
  marker's height above the floor;
* ``3-D TWO-POINT BODY-PART DISTANCES (MM)`` — ``3-D distance (mm) a-b``
  for every marker pair;
* ``3-D FRAME-BY-FRAME BODY-PART MOVEMENTS (MM)`` —
  ``3-D movement <marker> (mm)``: distance moved since the previous frame;
* ``3-D BODY POSTURE ANGLES (DEGREES)`` — head pitch and roll, body pitch
  (tail base → neck), head pitch relative to the body, the signed bend
  in the sagittal plane at each joint of the spine (dorsal flexion
  positive), plus the run's own ``pelvis_pitch_deg`` / ``lumbar_angle_deg``
  when present. Marker roles are the rat skeleton's
  (:class:`mufasa.pose3d.mono.MonoConfig`); angles whose markers are
  missing are left out.

Rows follow the video's frames (the 2-D pose file's rows): the 3-D file's
``frame`` column places each row, frames it lacks are missing.
"""
from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HEIGHTS_3D = "3-D BODY-PART HEIGHTS (MM)"
DISTANCES_3D = "3-D TWO-POINT BODY-PART DISTANCES (MM)"
MOVEMENT_3D = "3-D FRAME-BY-FRAME BODY-PART MOVEMENTS (MM)"
POSTURE_3D = "3-D BODY POSTURE ANGLES (DEGREES)"
FAMILIES_3D = (HEIGHTS_3D, DISTANCES_3D, MOVEMENT_3D, POSTURE_3D)
# keep NaN for missing values (heights can be ≤ 0 for mocap; angles are signed)
SIGNED_3D = frozenset({HEIGHTS_3D, POSTURE_3D})

# spine, tail end → head, as in MonoConfig.spine_map / the 3-D chain
SPINE_ORDER = ("tail_V6", "back_V2", "back_L6", "back_L2", "back_T8", "back_T4", "head_back")


# --------------------------------------------------------------------------- #
# finding and loading
# --------------------------------------------------------------------------- #
def pose3d_sources(project_root) -> list[Path]:
    """Folders that hold 3-D pose files, newest 3-D run first, then
    motion capture."""
    root = Path(project_root)
    runs = sorted((d for d in (root / "derived" / "pose3d").glob("*") if d.is_dir()),
                  key=lambda d: d.name, reverse=True)
    mocap = root / "sources" / "pose3d"
    return runs + ([mocap] if mocap.is_dir() else [])


def find_pose3d_file(video: str, folders: Iterable[Path]) -> Path | None:
    """``<video>_3d.parquet`` (else ``<video>*.parquet``) in the first
    folder that has one."""
    for d in folders:
        p = Path(d) / f"{video}_3d.parquet"
        if p.is_file():
            return p
        hits = sorted(Path(d).glob(f"{video}*.parquet"))
        if hits:
            return hits[0]
    return None


def load_pose3d_aligned(path, n_frames: int) -> tuple[list[str], np.ndarray, dict]:
    """(markers, xyz (n_frames, M, 3), extra per-frame columns) with row k
    = video frame k; frames the file lacks are NaN."""
    df = pd.read_parquet(path)
    from mufasa.pose3d.io import marker_names
    names = marker_names(df)                                  # 122kz: the shared rule
    frames = df["frame"].to_numpy(int) if "frame" in df.columns else np.arange(len(df))
    keep = (frames >= 0) & (frames < n_frames)
    xyz = np.full((n_frames, len(names), 3), np.nan)
    xyz[frames[keep]] = np.stack([df[[m + "_x", m + "_y", m + "_z"]].to_numpy(float) for m in names],
                                 1)[keep]
    extra = {}
    for c in ("pelvis_pitch_deg", "lumbar_angle_deg"):
        if c in df.columns:
            v = np.full(n_frames, np.nan)
            v[frames[keep]] = df[c].to_numpy(float)[keep]
            extra[c] = v
    return names, xyz, extra


# --------------------------------------------------------------------------- #
# families
# --------------------------------------------------------------------------- #
def compute_heights(names, xyz) -> dict[str, np.ndarray]:
    return {f"Height (mm) {m}": xyz[:, j, 2] for j, m in enumerate(names)}


def compute_distances(names, xyz) -> dict[str, np.ndarray]:
    order = sorted(range(len(names)), key=lambda j: names[j].lower())
    out = {}
    for a, b in combinations(order, 2):
        out[f"3-D distance (mm) {names[a]}-{names[b]}"] = np.linalg.norm(xyz[:, a] - xyz[:, b], axis=1)
    return out


def compute_movement(names, xyz) -> dict[str, np.ndarray]:
    step = np.full(xyz.shape[:2], np.nan)
    step[1:] = np.linalg.norm(np.diff(xyz, axis=0), axis=2)
    step[0] = 0.0
    return {f"3-D movement {m} (mm)": step[:, j] for j, m in enumerate(names)}


def _elevation(a, b, heading=None) -> np.ndarray:
    """Elevation of a→b (degrees); along ``heading`` (unit, xy) when given,
    so a bend backwards past vertical stays signed."""
    d = b - a
    horiz = np.linalg.norm(d[:, :2], axis=1) if heading is None else np.einsum("ij,ij->i", d[:, :2], heading)
    return np.degrees(np.arctan2(d[:, 2], horiz))


def compute_posture(names, xyz, extra: dict | None = None) -> dict[str, np.ndarray]:
    idx = {m: j for j, m in enumerate(names)}

    def P(m):
        return xyz[:, idx[m]]

    def have(*ms):
        return all(m in idx for m in ms)
    out: dict[str, np.ndarray] = {}
    if have("head_back", "head_nose"):
        out["Head pitch (degrees)"] = _elevation(P("head_back"), P("head_nose"))
    if have("head_left", "head_right"):
        d = P("head_right") - P("head_left")
        out["Head roll (degrees)"] = np.degrees(np.arctan2(d[:, 2], np.linalg.norm(d[:, :2], axis=1)))
    tail = next((m for m in ("tail_V6", "back_V2") if m in idx), None)
    neck = next((m for m in ("head_back", "back_T4") if m in idx), None)
    heading = None
    if tail and neck:
        out["Body pitch (degrees)"] = _elevation(P(tail), P(neck))
        h = P(neck)[:, :2] - P(tail)[:, :2]
        n = np.linalg.norm(h, axis=1, keepdims=True)
        heading = np.where(n > 1e-9, h / np.where(n > 1e-9, n, 1), np.nan)
        if "Head pitch (degrees)" in out:
            out["Head pitch relative to body (degrees)"] = out["Head pitch (degrees)"] - out["Body pitch (degrees)"]
    chain = [m for m in SPINE_ORDER if m in idx]
    if heading is not None and len(chain) >= 3:
        el = [_elevation(P(a), P(b), heading) for a, b in zip(chain[:-1], chain[1:])]
        for k in range(1, len(chain) - 1):
            bend = (el[k] - el[k - 1] + 180.0) % 360.0 - 180.0
            out[f"Spine bend at {chain[k]} (degrees)"] = bend
    for c, label in (("pelvis_pitch_deg", "Pelvis pitch (degrees)"), ("lumbar_angle_deg", "Lumbar angle (degrees)")):
        if extra and c in extra:
            out[label] = extra[c]
    return out


def compute_family(family: str, names, xyz, extra=None) -> dict[str, np.ndarray]:
    if family == HEIGHTS_3D:
        return compute_heights(names, xyz)
    if family == DISTANCES_3D:
        return compute_distances(names, xyz)
    if family == MOVEMENT_3D:
        return compute_movement(names, xyz)
    if family == POSTURE_3D:
        return compute_posture(names, xyz, extra)
    raise ValueError(f"not a 3-D family: {family!r}")


__all__ = ["DISTANCES_3D", "FAMILIES_3D", "HEIGHTS_3D", "MOVEMENT_3D", "POSTURE_3D", "SIGNED_3D",
           "SPINE_ORDER", "compute_distances", "compute_family", "compute_heights", "compute_movement",
           "compute_posture", "find_pose3d_file", "load_pose3d_aligned", "pose3d_sources"]
