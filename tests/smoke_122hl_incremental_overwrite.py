"""Smoke test for patch 122hl — incremental smoothing / overwrite option.

smooth_pose_v2 gains an `overwrite` parameter (default True = current
behaviour, re-smooth everything). With overwrite=False, any session whose
output already exists (<stem>_smoothed_v2.parquet, or the .csv fallback) is
skipped BEFORE the per-session smoother runs, in both the parallel and serial
paths — so a re-run only processes files with no output yet, and skipped
sessions cost no compute. With no output_dir there is nothing to skip.

Exposed as --skip-existing / --overwrite on the CLI (mutually exclusive,
default overwrite) and as an "Overwrite existing smoothed output" checkbox in
the Kalman v2 GUI form, threaded to both train and load modes.
"""
from __future__ import annotations

import ast
import pathlib
import sys
import tempfile
import types
from pathlib import Path

_tk = types.ModuleType("tkinter")
_tk.messagebox = types.ModuleType("tkinter.messagebox")
_tk.messagebox.showerror = lambda *a, **k: None
sys.modules.setdefault("tkinter", _tk)
sys.modules.setdefault("tkinter.messagebox", _tk.messagebox)

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


# ---------------------------------------------------------------- #
# 1. the skip decision (replicates the exact in-function logic)
# ---------------------------------------------------------------- #
def compute_skips(stems, existing, output_dir, overwrite):
    """Mirror of the skip_session computation in smooth_pose_v2."""
    n = len(stems)
    skip = [False] * n
    if output_dir is not None and not overwrite:
        for i in range(n):
            pq = Path(output_dir) / f"{stems[i]}_smoothed_v2.parquet"
            csv = Path(output_dir) / f"{stems[i]}_smoothed_v2.csv"
            if pq.exists() or csv.exists():
                skip[i] = True
    return skip


with tempfile.TemporaryDirectory() as d:
    stems = ["A", "B", "C", "D"]
    (Path(d) / "A_smoothed_v2.parquet").write_text("x")
    (Path(d) / "C_smoothed_v2.csv").write_text("x")   # csv fallback counts

    # overwrite=True -> process all
    sk = compute_skips(stems, None, d, overwrite=True)
    check("overwrite=True skips nothing", sk == [False, False, False, False])

    # overwrite=False -> skip A (parquet) and C (csv), keep B, D
    sk = compute_skips(stems, None, d, overwrite=False)
    check("overwrite=False skips existing parquet output (A)", sk[0] is True)
    check("overwrite=False skips existing csv output (C)", sk[2] is True)
    check("overwrite=False keeps files with no output (B, D)",
          sk[1] is False and sk[3] is False)
    smoothed = [stems[i] for i in range(4) if not sk[i]]
    check("overwrite=False processes exactly the un-smoothed files",
          smoothed == ["B", "D"])

    # No output_dir -> nothing to skip even with overwrite=False
    sk = compute_skips(stems, None, None, overwrite=False)
    check("no output_dir -> nothing skipped (in-memory use)",
          sk == [False, False, False, False])

# ---------------------------------------------------------------- #
# 2. the parameter and both dispatch paths honour it
# ---------------------------------------------------------------- #
src = (REPO / "mufasa/data_processors/kalman_pose_smoother_v2.py").read_text()
tree = ast.parse(src)
sig = None
for fn in ast.walk(tree):
    if isinstance(fn, ast.FunctionDef) and fn.name == "smooth_pose_v2":
        sig = [a.arg for a in fn.args.args]
        break
check("smooth_pose_v2 has an 'overwrite' parameter",
      sig is not None and "overwrite" in sig)

check("skip is computed once before dispatch",
      "skip_session = [False] * n_sess_final" in src)
# parallel path filters task_args
check("parallel path filters task_args on skip_session",
      "if not skip_session[sess_idx]" in src)
# serial path skips in the loop
check("serial path continues on skip_session",
      "if skip_session[sess_idx]:" in src
      and src.count("if skip_session[sess_idx]:") >= 1)
# the skip only applies when output_dir is set and not overwrite
check("skip gated on output_dir present and not overwrite",
      "if output_dir is not None and not overwrite:" in src)

# ---------------------------------------------------------------- #
# 3. CLI flags
# ---------------------------------------------------------------- #
check("--skip-existing flag defined", '"--skip-existing"' in src)
check("--overwrite flag defined", '"--overwrite"' in src)
check("--skip-existing sets overwrite False (store_false)",
      'dest="overwrite", action="store_false"' in src)
check("overwrite defaults True at the CLI",
      "parser.set_defaults(overwrite=True)" in src)
check("main passes overwrite to smooth_pose_v2",
      "overwrite=args.overwrite" in src)

# ---------------------------------------------------------------- #
# 4. GUI wiring
# ---------------------------------------------------------------- #
gui = (REPO / "mufasa/ui_qt/forms/pose_cleanup.py").read_text()
gtree = ast.parse(gui)
form = ""
for c in ast.walk(gtree):
    if isinstance(c, ast.ClassDef) and c.name == "KalmanV2SmoothingForm":
        form = ast.get_source_segment(gui, c) or ""
        break
check("GUI has an overwrite checkbox",
      "self.overwrite = QCheckBox" in form)
check("GUI train collect_args emits overwrite",
      '"overwrite": bool(self.overwrite.isChecked())' in form)
check("GUI load collect_args emits overwrite",
      '"overwrite":            bool(self.overwrite.isChecked())' in form)
# exactly the 3 output-writing calls pass overwrite (pass-1 writes nothing)
check("GUI output-writing smooth calls pass overwrite (3: pass2/single/load)",
      form.count('overwrite=kwargs["overwrite"]') == 3)

# ---------------------------------------------------------------- #
n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122hl_incremental_overwrite: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
