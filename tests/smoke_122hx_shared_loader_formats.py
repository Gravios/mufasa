"""Smoke test for patch 122hx — shared pose loader handles all three formats.

pose_video_overlay / pose_viewer._load_pose_file failed on a raw FreeDLC
long-format parquet with "Could not load ... (no marker columns found)" — the
same raw-long shape 122hw fixed in the *smoother's* loader, but in a separate
copy. Rather than patch each loader, 122hx teaches the shared fallback both
delegate to — kalman_diagnostic.load_pose_file — the two non-flat FreeDLC
shapes (imported MultiIndex, raw long), so pose_viewer inherits the fix by
falling through to it.

This is dependency-light (kalman_diagnostic imports pandas/numpy only), so it
runs the real loader against real parquet round-trips for all three formats.
"""
from __future__ import annotations

import pathlib
import sys
import tempfile
from pathlib import Path

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mufasa.data_processors.kalman_diagnostic import (  # noqa: E402
    _pivot_fdlc_long_to_wide,
    load_pose_file,
)

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


def _write_long(path, markers, n=8, individual="a0"):
    rows = []
    for f in range(n):
        for bp in markers:
            rows.append({"frame": f, "individual": individual, "bodypart": bp,
                         "x": float(f), "y": float(f) * 2, "likelihood": 0.9})
    pd.DataFrame(rows).to_parquet(path)


def _write_imported(path, markers):
    flat = []
    for m in markers:
        flat += [f"{m}_x", f"{m}_y", f"{m}_p"]
    df = pd.DataFrame(np.random.rand(6, len(flat)), columns=flat)
    df.columns = pd.MultiIndex.from_tuples(
        [("IMPORTED_POSE", "IMPORTED_POSE", c) for c in flat])
    df.to_parquet(path)


def _write_wide(path, markers):
    cols = {}
    for m in markers:
        cols[f"{m}_x"] = [1.0, 2.0]
        cols[f"{m}_y"] = [3.0, 4.0]
        cols[f"{m}_p"] = [0.9, 0.9]
    pd.DataFrame(cols).to_parquet(path)


MARKERS = ["head_nose", "back_T4", "back_L2", "tail_V6"]

with tempfile.TemporaryDirectory() as d:
    dd = Path(d)

    # 1. raw FreeDLC long — the reported failure
    p_long = dd / "raw.fdlc.parquet"
    _write_long(p_long, MARKERS)
    df1, m1 = load_pose_file(str(p_long))
    check("shared loader reads a raw FreeDLC long parquet",
          set(m1) == set(MARKERS))
    check("long-format markers keep their case (back_T4)", "back_T4" in m1)

    # 2. imported MultiIndex (122ht form)
    p_imp = dd / "imported.parquet"
    _write_imported(p_imp, MARKERS)
    df2, m2 = load_pose_file(str(p_imp))
    check("shared loader reads an imported MultiIndex parquet",
          set(m2) == set(MARKERS))

    # 3. wide flat (smoothed output / plain) — must still work, case preserved
    p_wide = dd / "smoothed.parquet"
    _write_wide(p_wide, MARKERS)
    df3, m3 = load_pose_file(str(p_wide))
    check("shared loader still reads a wide flat parquet",
          set(m3) == set(MARKERS))
    check("wide-flat markers keep their case", "back_T4" in m3)

    # pose_viewer inherits the fix: its direct read finds no markers on the
    # long file (so it falls back), and the shared loader then loads it.
    def _viewer_direct_markers(path):
        df = pd.read_parquet(path)
        cols = [str(c).lower() for c in df.columns]
        return {c[:-2] for c in cols
                if c.endswith("_x") and f"{c[:-2]}_y" in cols}

    check("pose_viewer's direct read finds no markers on a long file",
          _viewer_direct_markers(p_long) == set())
    # ...and falling back to the shared loader (what pose_viewer does) works
    dfv, mv = load_pose_file(str(p_long))
    check("pose_viewer's diagnostic fallback now loads the long file",
          set(mv) == set(MARKERS))

# pivot helper unit checks (the piece the shared loader calls)
def _long_df(markers, n=5, individual="a0", likelihood=0.9):
    rows = []
    for f in range(n):
        for bp in markers:
            rows.append({"frame": f, "individual": individual, "bodypart": bp,
                         "x": float(f), "y": float(f), "likelihood": likelihood})
    return pd.DataFrame(rows)


w = _pivot_fdlc_long_to_wide(_long_df(MARKERS))
check("pivot emits _x/_y/_p per bodypart",
      all(f"{m}_x" in w.columns and f"{m}_y" in w.columns
          and f"{m}_p" in w.columns for m in MARKERS))
check("pivot clips the -1 sentinel to 0",
      _pivot_fdlc_long_to_wide(_long_df(["n"], 1, likelihood=-1.0))
      ["n_p"].iloc[0] == 0.0)
try:
    _pivot_fdlc_long_to_wide(
        pd.concat([_long_df(["n"], 2, "a0"), _long_df(["n"], 2, "a1")]))
    check("pivot rejects multi-animal with an explicit message", False)
except ValueError as exc:
    check("pivot rejects multi-animal with an explicit message",
          "individual" in str(exc).lower())

# wiring: the shared loader's parquet branch handles both non-flat shapes
src = (REPO / "mufasa/data_processors/kalman_diagnostic.py").read_text()
check("shared loader flattens a MultiIndex header",
      "isinstance(df.columns, pd.MultiIndex)" in src
      and "col[-1] for col in df.columns" in src)
check("shared loader pivots the long schema",
      '{"frame", "bodypart", "x", "y"}.issubset(' in src
      and "_pivot_fdlc_long_to_wide(df)" in src)

n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122hx_shared_loader_formats: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
