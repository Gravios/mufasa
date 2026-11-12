"""
mufasa.pose3d.io
================

Patch 122kf: 3-D pose files — parquet with ``<marker>_x/_y/_z`` (mm),
``frame``, per-marker ``_likelihood`` / ``_reproj_px``, and a JSON
``mufasa_pose3d`` schema entry (box, camera, edges, method) that the 3-D
viewer (patch 122ke) reads.
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


def marker_names(df: pd.DataFrame) -> list[str]:
    return [c[:-2] for c in df.columns
            if c.endswith("_x") and c[:-2] + "_y" in df.columns and c[:-2] + "_z" in df.columns]


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


__all__ = ["META_KEY", "marker_names", "read_pose3d", "skeleton_edges", "write_pose3d"]
