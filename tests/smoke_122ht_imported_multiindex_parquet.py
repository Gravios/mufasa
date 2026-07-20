"""Smoke test for patch 122ht — smoother reads imported MultiIndex parquets.

The FreeDLC/SimBA importer writes sources/pose parquets with a 3-level
MultiIndex column header (scorer/bodypart/coords =
IMPORTED_POSE/IMPORTED_POSE/<bp>_x), via write_df's multi_idx_header path. The
Kalman smoother's inline loader did pd.read_parquet then looked for columns
ending in "_x"; on a MultiIndex the columns come back as tuples, so it found no
markers, fell through to the DLC-CSV diagnostic parser, and failed with
"Could not load ... Tried direct read and DLC multi-row header parsing" — which
is what the user hit adding sessions with smoothing.

122ht flattens a MultiIndex column header to its last level (which holds the
flat <bp>_x/_y/_p names) right after the direct read, leaving plain columns
untouched.

The smoother module imports heavy deps (pyarrow/cv2/h5py) absent in the
sandbox, so this test (1) verifies the fix logic against a parquet written
exactly as the importer writes it, and (2) AST-checks the fix is present in the
loader ahead of the marker detection.
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

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


def _write_imported_parquet(path: Path, markers: list[str]) -> None:
    """Write a parquet exactly as the FDLC importer does: 3-level MultiIndex
    (IMPORTED_POSE, IMPORTED_POSE, <bp>_x/_y/_p)."""
    flat: list[str] = []
    for m in markers:
        flat += [f"{m}_x", f"{m}_y", f"{m}_p"]
    df = pd.DataFrame(np.random.rand(20, len(flat)), columns=flat)
    df.columns = pd.MultiIndex.from_tuples(
        [("IMPORTED_POSE", "IMPORTED_POSE", c) for c in flat],
        names=("scorer", "bodypart", "coords"),
    )
    df.to_parquet(path)


def _load_markers(path: Path) -> set[str]:
    """Replicate the smoother's fixed direct-read marker detection."""
    df_direct = pd.read_parquet(path)
    # THE FIX:
    if isinstance(df_direct.columns, pd.MultiIndex):
        df_direct.columns = [col[-1] for col in df_direct.columns]
    markers_found = set()
    for col in df_direct.columns:
        if isinstance(col, str) and col.endswith("_x"):
            base = col[:-2]
            if f"{base}_y" in df_direct.columns and not base.endswith("_var"):
                markers_found.add(base)
    return markers_found


MARKERS = ["back_T4", "back_T8", "head_nose", "tail_V6", "hip_left"]

with tempfile.TemporaryDirectory() as d:
    # imported 3-level MultiIndex parquet -> markers recovered
    p = Path(d) / "imported.parquet"
    _write_imported_parquet(p, MARKERS)
    found = _load_markers(p)
    check("imported MultiIndex parquet: all markers recovered",
          found == set(MARKERS))
    check("marker names preserve case (back_T4 not back_t4)",
          "back_T4" in found)

    # a plain-column parquet (e.g. a prior smoothed-flat output) still works
    p2 = Path(d) / "flat.parquet"
    flat = []
    for m in MARKERS:
        flat += [f"{m}_x", f"{m}_y", f"{m}_p"]
    pd.DataFrame(np.random.rand(10, len(flat)),
                 columns=flat).to_parquet(p2)
    check("plain-column parquet still loads (guard skips flatten)",
          _load_markers(p2) == set(MARKERS))

    # _var columns are excluded (variance overlay columns aren't markers)
    p3 = Path(d) / "withvar.parquet"
    cols = []
    for m in MARKERS:
        cols += [f"{m}_x", f"{m}_y", f"{m}_p",
                 f"{m}_var_x", f"{m}_var_y"]
    df3 = pd.DataFrame(np.random.rand(10, len(cols)), columns=cols)
    df3.columns = pd.MultiIndex.from_tuples(
        [("IMPORTED_POSE", "IMPORTED_POSE", c) for c in cols])
    df3.to_parquet(p3)
    found3 = _load_markers(p3)
    check("variance columns are not mistaken for markers",
          found3 == set(MARKERS))

# ------------------------------------------------------------------ #
# AST: the fix is present in the loader, before the marker detection
# ------------------------------------------------------------------ #
src = (REPO / "mufasa/data_processors/kalman_pose_smoother_v2.py").read_text()
check("loader flattens a MultiIndex column header",
      "isinstance(df_direct.columns, pd.MultiIndex)" in src
      and "col[-1] for col in df_direct.columns" in src)
# the flatten must come BEFORE the marker check (col.endswith('_x'))
flatten_pos = src.find("isinstance(df_direct.columns, pd.MultiIndex)")
# the marker detection loop over df_direct in the direct-read branch
marker_pos = src.find('if col.endswith("_x"):', flatten_pos)
check("flatten precedes the direct-read marker detection",
      0 <= flatten_pos < marker_pos)

n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122ht_imported_multiindex_parquet: "
      f"{n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
