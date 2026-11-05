"""Smoke test for patch 122ja — feature kernels with the compiled kernels loaded.

User report: Compute Features → HEAD VELOCITY IN BODY FRAME failed on every
video with "NameError: name 'warnings' is not defined". The new kernel used
``warnings``, which was only imported inside the fallback branch that runs
when the compiled (Cython) kernels are missing. In development (no compiled
kernels) the fallback ran and the name existed, so tests passed; on an
installed copy the compiled kernels load, the branch is skipped and the name
was never bound.

Checks, statically and at run time with the compiled kernels pretended
present:

1. every module-level name a function in feature_subset_kernels uses is
   bound unconditionally (top-level import / def / assignment), or by both
   arms of the native-kernel try/except;
2. ``warnings`` is imported at top level;
3. the head-velocity kernel runs after importing the module with a stub
   ``mufasa._native`` package, i.e. down the path the user's machine takes.
"""
from __future__ import annotations

import ast
import builtins
import importlib
import sys
import types
from pathlib import Path

_tk = types.ModuleType("tkinter")
_tk.messagebox = types.ModuleType("tkinter.messagebox")
sys.modules.setdefault("tkinter", _tk)
sys.modules.setdefault("tkinter.messagebox", _tk.messagebox)
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
SRC = REPO / "mufasa" / "feature_extractors" / "feature_subset_kernels.py"

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


tree = ast.parse(SRC.read_text())


def bound(nodes) -> set[str]:
    out: set[str] = set()
    for n in nodes:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            out |= {(a.asname or a.name).split(".")[0] for a in n.names}
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            out.add(n.name)
        elif isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            for t in targets:
                out |= {x.id for x in ast.walk(t) if isinstance(x, ast.Name)}
    return out


always = bound(tree.body)
for n in tree.body:
    if isinstance(n, ast.Try):           # names bound in try AND every handler
        both = bound(n.body)
        for h in n.handlers:
            both &= bound(h.body)
        always |= both
    if isinstance(n, ast.If):
        always |= bound(n.body) & bound(n.orelse)
check("warnings is imported at module top level",
      any(isinstance(n, ast.Import) and any(a.name == "warnings" for a in n.names)
          for n in tree.body))

missing: dict[str, set[str]] = {}
for fn in tree.body:
    if not isinstance(fn, ast.FunctionDef):
        continue
    local = {a.arg for a in ast.walk(fn.args) if isinstance(a, ast.arg)}
    for x in ast.walk(fn):
        if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store):
            local.add(x.id)
        elif isinstance(x, (ast.FunctionDef, ast.Lambda)) and x is not fn:
            if isinstance(x, ast.FunctionDef):
                local.add(x.name)
            local |= {a.arg for a in ast.walk(x.args) if isinstance(a, ast.arg)}
        elif isinstance(x, (ast.Import, ast.ImportFrom)):
            local |= {(a.asname or a.name).split(".")[0] for a in x.names}
        elif isinstance(x, ast.comprehension):
            local |= {t.id for t in ast.walk(x.target) if isinstance(t, ast.Name)}
        elif isinstance(x, ast.ExceptHandler) and x.name:
            local.add(x.name)
        elif isinstance(x, (ast.With,)):
            for it in x.items:
                if it.optional_vars is not None:
                    local |= {t.id for t in ast.walk(it.optional_vars)
                              if isinstance(t, ast.Name)}
    used = {x.id for x in ast.walk(fn) if isinstance(x, ast.Name)
            and isinstance(x.ctx, ast.Load)}
    free = used - local - always - set(dir(builtins))
    if free:
        missing[fn.name] = free
check(f"every global a kernel function uses is bound on both import paths "
      f"(unbound: {missing or 'none'})", not missing)

# ---- run time, down the compiled-kernels path -----------------------------
for k in [m for m in sys.modules if m.startswith("mufasa")]:
    del sys.modules[k]
native = types.ModuleType("mufasa._native")
native.__path__ = []
sys.modules["mufasa._native"] = native
stub_names = {
    "angle3pt": ["angle3pt_vectorized"],
    "border_distances": ["border_distances"],
    "euclidean_distance": ["framewise_euclidean_distance",
                           "framewise_euclidean_distance_roi"],
    "hull": ["jitted_hull"], "inside_circle": ["is_inside_circle"],
    "inside_polygon": ["framewise_inside_polygon_roi"],
    "inside_rectangle": ["framewise_inside_rectangle_roi"],
}
src = SRC.read_text()
for mod in {m.group(1) for m in __import__("re").finditer(r"mufasa\._native\.(\w+)", src)}:
    sm = types.ModuleType(f"mufasa._native.{mod}")
    for attr in __import__("re").findall(rf"mufasa\._native\.{mod} import \(?\s*(\w+)", src):
        setattr(sm, attr, lambda *a, **k: None)
    for attr in stub_names.get(mod, []):
        setattr(sm, attr, lambda *a, **k: None)
    sys.modules[f"mufasa._native.{mod}"] = sm
import warnings as _w  # noqa: E402

with _w.catch_warnings(record=True) as rec:
    _w.simplefilter("always")
    K = importlib.import_module("mufasa.feature_extractors.feature_subset_kernels")
check("with mufasa._native present the module takes the compiled path "
      "(no fallback warning)",
      K._NATIVE_AVAILABLE and not any("falling back" in str(r.message) for r in rec))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

T = 50
t = np.arange(T, dtype=float)
df = pd.DataFrame({"h_x": 120 + 3 * t, "h_y": np.full(T, 100.), "b_x": np.full(T, 100.),
                   "b_y": np.full(T, 100.)})
df.loc[5, "h_x"] = np.nan
try:
    out = K.compute_head_velocity_body_frame(df, {"A": (["h"], ["b"], "x")}, 2.0, 30.0)
    ok = np.allclose(list(out.values())[0][10:40], 45)
except NameError as exc:
    ok = False
    print("NameError:", exc)
check("head-velocity kernel runs on the compiled-kernels path", ok)

n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122ja_kernels_module_imports: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
