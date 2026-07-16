"""Smoke test for patch 122hb — marker-name case + layout resolution order.

Two bugs, both of which presented as "the smoother can't find my markers":

1. Every pose loader did ``[str(c).lower() for c in df.columns]``, which
   renamed markers. A project with ``back_T4``/``tail_V6`` got ``back_t4``/
   ``tail_v6`` back and matched nothing. Markers already lower-case survived,
   so the damage looked like a *partial* rename.
2. ``smooth_pose_v2`` defaulted ``layout`` to ``standard_rat_layout()`` and
   validated the data against it BEFORE ``load_model`` opened the .npz that
   carries the model's real layout. Load-mode calls therefore died citing
   ``nose``/``back1``/``tailbase`` — names from the built-in rig.

These are real tests: actual parquet/CSV files through the actual loaders.
"""
from __future__ import annotations

import ast
import pathlib
import sys
import tempfile
import types

# mufasa.utils.errors does a legacy `from tkinter import messagebox` at
# import time; the Tk GUI is retired but the import lingers.
_tk = types.ModuleType("tkinter")
_tk.messagebox = types.ModuleType("tkinter.messagebox")
_tk.messagebox.showerror = lambda *a, **k: None
sys.modules.setdefault("tkinter", _tk)
sys.modules.setdefault("tkinter.messagebox", _tk.messagebox)

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mufasa.data_processors.kalman_diagnostic import (  # noqa: E402
    load_pose_file,
    normalize_pose_columns,
)

# The user's real marker set: 8 of 15 carry an upper-case segment code.
MARKERS = [
    "head_nose", "head_mid", "head_left", "head_right", "head_back",
    "back_T4", "back_T8", "back_L2", "back_L6", "back_V2",
    "hip_left", "hip_right", "tail_V6", "tail_V18", "tail_V32",
]
CASE_SENSITIVE = sorted(m for m in MARKERS if m != m.lower())

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


def _write_parquet(path: pathlib.Path, markers=MARKERS, n=8) -> None:
    cols = {}
    for m in markers:
        cols[f"{m}_x"] = np.arange(n, dtype=float)
        cols[f"{m}_y"] = np.arange(n, dtype=float)
        cols[f"{m}_p"] = np.ones(n)
    pd.DataFrame(cols).to_parquet(path)


tmp = pathlib.Path(tempfile.mkdtemp())

# ---------------------------------------------------------------- #
# 1. normalize_pose_columns: base preserved, suffix folded
# ---------------------------------------------------------------- #
out = normalize_pose_columns(
    ["back_T4_x", "back_T4_Y", "tail_V6_P", "head_nose_x", "scorer", "_x"]
)
check("suffix _x preserves base case", out[0] == "back_T4_x")
check("suffix _Y folded, base kept", out[1] == "back_T4_y")
check("suffix _P folded, base kept", out[2] == "tail_V6_p")
check("lower-case marker unchanged", out[3] == "head_nose_x")
check("non-marker column still folded", out[4] == "scorer")
check("bare '_x' not treated as marker", out[5] == "_x")
check("normalizer is idempotent", normalize_pose_columns(out) == out)

# ---------------------------------------------------------------- #
# 2. parquet round-trip: all 15 markers survive
# ---------------------------------------------------------------- #
pq = tmp / "sess.parquet"
_write_parquet(pq)
_, markers_pq = load_pose_file(str(pq))
check("parquet: all 15 markers read back", sorted(markers_pq) == sorted(MARKERS))
check(
    "parquet: no case-folded marker lost",
    not (set(CASE_SENSITIVE) - set(markers_pq)),
)

# ---------------------------------------------------------------- #
# 3. flat CSV round-trip
# ---------------------------------------------------------------- #
csv = tmp / "sess.csv"
pd.read_parquet(pq).to_csv(csv)
_, markers_csv = load_pose_file(str(csv))
check("csv: all 15 markers read back", sorted(markers_csv) == sorted(MARKERS))

# ---------------------------------------------------------------- #
# 4. DLC 3-row multi-index CSV: bodypart level keeps its case
# ---------------------------------------------------------------- #
dlc = tmp / "dlc.csv"
tuples, data = [], {}
for m in ("back_T4", "tail_V6", "head_nose"):
    for level in ("x", "y", "likelihood"):
        tuples.append(("model", m, level))
        data[("model", m, level)] = np.arange(4, dtype=float)
df_dlc = pd.DataFrame(data)
df_dlc.columns = pd.MultiIndex.from_tuples(tuples)
df_dlc.to_csv(dlc)
_, markers_dlc = load_pose_file(str(dlc))
check(
    "dlc multi-header: bodypart case preserved",
    sorted(markers_dlc) == ["back_T4", "head_nose", "tail_V6"],
)

# ---------------------------------------------------------------- #
# 5. the smoother's own direct reader
# ---------------------------------------------------------------- #
src = (REPO / "mufasa/data_processors/kalman_pose_smoother_v2.py").read_text()
check(
    "smoother reader uses normalize_pose_columns",
    "df_direct.columns = normalize_pose_columns(df_direct.columns)" in src,
)
check(
    "no blanket column .lower() left in the smoother reader",
    "str(c).lower() for c in df_direct.columns" not in src,
)
diag_src = (REPO / "mufasa/data_processors/kalman_diagnostic.py").read_text()
check(
    "no blanket column .lower() left in the diagnostic loaders",
    "df.columns = [str(c).lower()" not in diag_src,
)

# ---------------------------------------------------------------- #
# 6. layout resolution order inside smooth_pose_v2
# ---------------------------------------------------------------- #
tree = ast.parse(src)
fn = next(
    n for n in ast.walk(tree)
    if isinstance(n, ast.FunctionDef) and n.name == "smooth_pose_v2"
)
args = [a.arg for a in fn.args.args + fn.args.kwonlyargs]
check("smooth_pose_v2 gained config_path", "config_path" in args)


def _first_line_of_call(func_name: str) -> int | None:
    hits = [
        n.lineno for n in ast.walk(fn)
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == func_name
    ]
    return min(hits) if hits else None


ln_load = _first_line_of_call("load_model_v2")
ln_validate = _first_line_of_call("_validate_layout_against_data")
check("smooth_pose_v2 calls load_model_v2", ln_load is not None)
check("smooth_pose_v2 calls _validate_layout_against_data", ln_validate is not None)
check(
    "model is loaded BEFORE the data is validated against a layout",
    ln_load is not None and ln_validate is not None and ln_load < ln_validate,
)
check(
    "load_model_v2 is unpacked exactly once (no duplicate load)",
    sum(
        1 for n in ast.walk(fn)
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "load_model_v2"
    ) == 1,
)
check(
    "smooth_pose_v2 no longer hard-defaults to standard_rat_layout()",
    not any(
        isinstance(n, ast.Call)
        and getattr(n.func, "id", "") == "standard_rat_layout"
        for n in ast.walk(fn)
    ),
)

# ---------------------------------------------------------------- #
# 7. the guard: any missing marker is fatal (closes the 122gv hole)
# ---------------------------------------------------------------- #
from mufasa.data_processors.kalman_pose_smoother_v2 import (  # noqa: E402
    _resolve_layout,
    _validate_layout_against_data,
    standard_rat_layout,
)

rig = standard_rat_layout()
full = list(rig.marker_names)

ok = True
try:
    _validate_layout_against_data(rig, full)
except ValueError:
    ok = False
check("exact match passes the guard", ok)

partial_msg = ""
try:
    _validate_layout_against_data(rig, full[:-3])
    partial = False
except ValueError as exc:
    partial, partial_msg = True, str(exc)
check("PARTIAL mismatch now raises (was a warning — the 122gv hole)", partial)
check("partial error reports the matched fraction", "/15 layout markers found" in partial_msg)

case_msg = ""
try:
    _validate_layout_against_data(rig, [m.upper() for m in full])
except ValueError as exc:
    case_msg = str(exc)
check("case-only mismatch is called out by name", "CASE ONLY" in case_msg)

try:
    _validate_layout_against_data(rig, ["completely", "different"])
    zero = False
except ValueError as exc:
    zero = "every value would be NaN" in str(exc)
check("zero overlap still raises", zero)

# ---------------------------------------------------------------- #
# 8. _resolve_layout never silently invents the rat rig
# ---------------------------------------------------------------- #
try:
    _resolve_layout(None, MARKERS)
    resolved = False
except ValueError as exc:
    resolved = "not the built-in rat rig" in str(exc)
check("no config + non-canonical markers -> explicit error, not the rat rig", resolved)
check(
    "no config + canonical markers -> built-in rig (legacy path kept)",
    _resolve_layout(None, full).n_markers == rig.n_markers,
)

# ---------------------------------------------------------------- #
# 9. every GUI call site resolves a layout
# ---------------------------------------------------------------- #
gui = ast.parse((REPO / "mufasa/ui_qt/forms/pose_cleanup.py").read_text())
calls = [
    n for n in ast.walk(gui)
    if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "smooth_pose_v2"
]
check("all 4 GUI smoothing call sites found", len(calls) == 4)
check(
    "no GUI call site leaves the layout to chance",
    all(
        {"layout", "load_model"} & {k.arg for k in c.keywords}
        and "config_path" in {k.arg for k in c.keywords}
        for c in calls
    ),
)

# ---------------------------------------------------------------- #
n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122hb_marker_case_and_layout: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
