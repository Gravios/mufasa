"""
mufasa.pose3d.io
================

Patch 122kf: 3-D pose files; patch 122kz: their contract, in one place.

**The 3-D pose file** (``<video>_3d.parquet``):

* ``<marker>_x``, ``<marker>_y``, ``<marker>_z``: millimetres, in a world frame
  with **z up and the floor at z = 0** (gait measures the legs' reach to it);
* ``frame``: the video frame of each row (rows in order, each frame once);
* per marker, optionally: ``<marker>_likelihood`` (0–1) and ``<marker>_reproj_px``;
* per frame, optionally: derived values (``lumbar_angle_deg``, ``pelvis_pitch_deg``…);
* metadata under ``mufasa_pose3d`` (JSON): ``units`` ("mm"), ``fps``, ``method``,
  ``camera``, ``edges``, ``box_x/_y/_z``, ``report``.

**Markers** are found with :func:`pose_markers`, the one rule every reader uses: a
name with ``_x``, ``_y`` (and ``_z``) columns, except a name ``<m>_var`` when ``<m>``
is a marker too — the per-axis variances (``<m>_var_x``…) the Kalman v2 smoother
writes into 2-D pose files, which once became markers of their own (K7: 15 such,
up to 2,000 km away). :func:`check_pose3d` lists what breaks the contract.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

META_KEY = b"mufasa_pose3d"


def write_pose3d(df: pd.DataFrame, path, meta: dict) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t = pa.Table.from_pandas(df, preserve_index=False)
    t = t.replace_schema_metadata({**(t.schema.metadata or {}), META_KEY: json.dumps(meta).encode()})
    pq.write_table(t, path)
    return path


def read_pose3d(path) -> tuple[pd.DataFrame, dict]:
    import pyarrow.parquet as pq
    t = pq.read_table(path)
    raw = (t.schema.metadata or {}).get(META_KEY)
    return t.to_pandas(), (json.loads(raw) if raw else {})


# columns <m><suffix>_x… derived from marker <m>, never markers themselves
DERIVED_SUFFIXES = ("_var",)


def pose_markers(columns, dims: int = 3) -> list[str]:
    """Marker names in a pose table's ``columns``, in column order: names with ``_x`` and
    ``_y`` columns (and ``_z`` for ``dims=3``), except ``<m><suffix>`` (e.g. ``<m>_var``,
    variances) when ``<m>`` is a marker too."""
    cols = set(map(str, columns))
    need = ("_y", "_z") if dims == 3 else ("_y",)
    names = [c[:-2] for c in map(str, columns)
             if c.endswith("_x") and all(c[:-2] + s in cols for s in need)]
    have = set(names)
    return [n for n in names
            if not any(n.endswith(s) and n[:-len(s)] in have for s in DERIVED_SUFFIXES)]


def marker_names(df: pd.DataFrame) -> list[str]:
    """The 3-D markers of a pose table (:func:`pose_markers`)."""
    return pose_markers(df.columns, 3)


def check_pose3d(df: pd.DataFrame, meta: dict | None = None, names: list[str] | None = None) -> list[str]:
    """Where a 3-D pose table breaks the file contract (see the module), as sentences; empty
    when it keeps it."""
    import numpy as np
    meta = meta or {}
    names = marker_names(df) if names is None else names
    out = []
    if not names:
        return ["no <marker>_x/_y/_z columns"]
    if not meta.get("fps"):
        out.append("no frame rate in the file (metadata 'fps'): times assume a default")
    if str(meta.get("units", "mm")).lower() not in ("mm", "millimetre", "millimeter"):
        out.append(f"units '{meta.get('units')}', not mm")
    if "frame" in df.columns:
        fr = df["frame"].to_numpy()
        if len(fr) > 1 and (np.diff(fr) <= 0).any():
            out.append("frames out of order or repeated")
        elif len(fr) > 1 and (np.diff(fr) > 1).any():
            out.append(f"{int((np.diff(fr) > 1).sum())} gaps in the frames")
    z = df[[n + "_z" for n in names]].to_numpy(float)
    z = z[np.isfinite(z)]
    if len(z) > 100:
        lo = float(np.percentile(z, 1))
        if lo < -10.0:
            out.append(f"markers below the floor (1st percentile of height {lo:.0f} mm): is the floor at z = 0?")
    cols = set(map(str, df.columns))
    skipped = [c[:-2] for c in map(str, df.columns)
               if c.endswith("_x") and c[:-2] + "_y" in cols and c[:-2] + "_z" in cols and c[:-2] not in names]
    if skipped:
        out.append(f"{len(skipped)} derived column sets are not markers ({', '.join(skipped[:3])}"
                   f"{'…' if len(skipped) > 3 else ''}): written from a smoothed 2-D file")
    return out


def skeleton_edges(names: list[str], segments: dict | None = None) -> list[list[str]]:
    """Viewer edges: the rat defaults for the markers present, or a chain
    per project segment (``segments``: name → [markers])."""
    rat = [["head_nose", "head_mid"], ["head_mid", "head_back"], ["head_left", "head_back"],
           ["head_right", "head_back"], ["head_left", "head_mid"], ["head_right", "head_mid"],
           ["head_back", "back_T4"], ["back_T4", "back_T8"], ["back_T8", "back_L2"], ["back_L2", "back_L6"],
           ["back_L6", "hip_left"], ["back_L6", "hip_right"], ["hip_left", "back_V2"], ["hip_right", "back_V2"],
           ["back_L6", "back_V2"], ["back_V2", "tail_V6"], ["tail_V6", "tail_V18"], ["tail_V18", "tail_V32"]]
    have = set(names)
    edges = [e for e in rat if e[0] in have and e[1] in have]
    for ms in (segments or {}).values():
        ms = [m for m in ms if m in have]
        edges += [[a, b] for a, b in zip(ms[:-1], ms[1:]) if [a, b] not in edges]
    return edges


__all__ = ["DERIVED_SUFFIXES", "META_KEY", "check_pose3d", "marker_names", "pose_markers", "read_pose3d",
           "skeleton_edges", "write_pose3d"]
