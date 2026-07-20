"""Smoke test for patch 122hw — smoother reads a raw FreeDLC long-format parquet.

FreeDLC's native .fdlc.parquet is LONG/tidy: columns (frame, individual,
bodypart, x, y, likelihood), one row per bodypart per frame. The smoother's
loader expects wide per-marker columns (<bp>_x/_y/_p). 122ht handled the
*imported* wide MultiIndex form, but a *raw* FreeDLC file (never run through the
importer, e.g. smoothed standalone with --load-model --beside-input) is long,
has no _x columns, and failed with "Could not load ... Tried direct read and DLC
multi-row header parsing".

122hw adds _pivot_fdlc_long_to_wide and detects the long schema in the loader,
pivoting to wide before the marker check.
"""
from __future__ import annotations

import pathlib
import sys
import types

_tk = types.ModuleType("tkinter")
_tk.messagebox = types.ModuleType("tkinter.messagebox")
_tk.messagebox.showerror = lambda *a, **k: None
sys.modules.setdefault("tkinter", _tk)
sys.modules.setdefault("tkinter.messagebox", _tk.messagebox)

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

import mufasa.data_processors.kalman_pose_smoother_v2 as K  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


def _long(markers, n_frames=10, individual="a0", likelihood=0.9):
    rows = []
    for f in range(n_frames):
        for bp in markers:
            rows.append({"frame": f, "individual": individual, "bodypart": bp,
                         "x": float(f), "y": float(f) * 2,
                         "likelihood": likelihood})
    return pd.DataFrame(rows)


# ---- the pivot helper ----
check("_pivot_fdlc_long_to_wide is defined",
      hasattr(K, "_pivot_fdlc_long_to_wide"))

MARKERS = ["head_nose", "back_T4", "back_L2", "tail_V6"]
wide = K._pivot_fdlc_long_to_wide(_long(MARKERS, n_frames=12))
found = {c[:-2] for c in wide.columns if c.endswith("_x")}
check("pivot recovers all bodyparts as <bp>_x", found == set(MARKERS))
check("pivot emits _x/_y/_p per bodypart",
      all(f"{m}_x" in wide.columns and f"{m}_y" in wide.columns
          and f"{m}_p" in wide.columns for m in MARKERS))
check("pivot preserves the frame count (contiguous)", len(wide) == 12)
check("marker names keep their case (back_T4)", "back_T4" in found)

# multi-animal rejected — with the explicit, actionable message (not pandas'
# incidental "duplicate entries" reshape error)
try:
    K._pivot_fdlc_long_to_wide(
        pd.concat([_long(["n"], 2, "a0"), _long(["n"], 2, "a1")]))
    check("multi-animal file is rejected", False)
    check("multi-animal rejection names the individuals", False)
except ValueError as exc:
    check("multi-animal file is rejected", True)
    check("multi-animal rejection names the individuals",
          "individual" in str(exc).lower()
          and "single-animal" in str(exc).lower())

# -1 sentinel likelihood clipped to 0
w = K._pivot_fdlc_long_to_wide(_long(["n"], 1, likelihood=-1.0))
check("the -1 no-detection sentinel is clipped to 0", w["n_p"].iloc[0] == 0.0)

# a gap in frame numbers is filled (reindexed to contiguous)
g = _long(["n"], 5)
g = g[g["frame"] != 2]  # drop frame 2
wg = K._pivot_fdlc_long_to_wide(g)
check("a missing frame is reindexed (contiguous 0..max)", len(wg) == 5)

# ---- loader wiring (AST): long detection precedes the marker check ----
src = (REPO / "mufasa/data_processors/kalman_pose_smoother_v2.py").read_text()
check("loader detects the FreeDLC long schema",
      '{"frame", "bodypart", "x", "y"}.issubset(' in src)
check("loader pivots long input before the marker check",
      "_pivot_fdlc_long_to_wide(df_direct)" in src)
# ordering: long-detect/pivot must precede the "_x" marker detection
pivot_pos = src.find("_pivot_fdlc_long_to_wide(df_direct)")
marker_pos = src.find('if col.endswith("_x"):', pivot_pos)
check("the long pivot precedes the direct-read marker detection",
      0 <= pivot_pos < marker_pos)
# the long-detect guard must not fire on wide MultiIndex data (122ht path)
check("long detection is gated to non-MultiIndex columns",
      "not isinstance(df_direct.columns, pd.MultiIndex)" in src
      and "{\"frame\", \"bodypart\", \"x\", \"y\"}.issubset(" in src)

# wide input is NOT treated as long (guard needs frame/bodypart present)
wide_df = pd.DataFrame({"head_nose_x": [1.0, 2.0], "head_nose_y": [3.0, 4.0],
                        "head_nose_p": [0.9, 0.9]})
check("a wide DataFrame lacks the long schema (won't be pivoted)",
      not {"frame", "bodypart", "x", "y"}.issubset(set(wide_df.columns)))

n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122hw_fdlc_long_format: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
