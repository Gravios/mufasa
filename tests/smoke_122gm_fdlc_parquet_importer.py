"""
tests/smoke_122gm_fdlc_parquet_importer.py
==========================================

Patch 122gm — import FreeDLC long/tidy Parquet pose files.

FreeDLC (modified DeepLabCut) writes pose as a long table
(frame|individual|bodypart|x|y|likelihood) named <stem>.fdlc.parquet.
FDLCParquetImporter mirrors DLCSingleAnimalH5Importer but reads that layout,
aligns nodes to the project body-parts BY NAME (exact -> case-insensitive ->
positional fallback -> error), clamps the -1.0 "no detection" likelihood
sentinel to 0.0, and writes the standard IMPORTED_POSE multi-index output.
Registered in the pose-import UI registry.

Functional checks exercise the pure static pivot (long_to_wide) and node
resolver (_resolve_bodypart_order) exec'd from source (pandas/numpy only;
the full module needs heavy deps).
"""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
MOD = REPO / "mufasa" / "pose_importers" / "fdlc_parquet_importer.py"
FORM = REPO / "mufasa" / "ui_qt" / "forms" / "pose_import.py"

P = T = 0


def check(label, cond, *, detail=""):
    global P, T
    T += 1
    if cond:
        P += 1
    else:
        print(f"FAIL: {label}" + (f" — {detail}" if detail else ""))


def _load_pure(src):
    """Exec long_to_wide + _resolve_bodypart_order + _LONG_COLUMNS in
    isolation with a stub error class."""
    tree = ast.parse(src)
    ns = {"pd": pd, "np": np}

    class BodypartColumnNotFoundError(Exception):
        def __init__(self, msg, source=None):
            super().__init__(msg)

    ns["BodypartColumnNotFoundError"] = BodypartColumnNotFoundError
    for n in tree.body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "_LONG_COLUMNS":
            exec(compile(ast.Module([n], []), "<c>", "exec"), ns)
        if isinstance(n, ast.FunctionDef) and n.name == "_resolve_bodypart_order":
            exec(compile(ast.Module([n], []), "<r>", "exec"), ns)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FDLCParquetImporter")
    ltw = next(m for m in cls.body if isinstance(m, ast.FunctionDef) and m.name == "long_to_wide")
    ltw.decorator_list = []
    exec(compile(ast.Module([ltw], []), "<l>", "exec"), ns)
    return ns


def _long(bps, n=3, individual="single"):
    rows = []
    for f in range(n):
        for i, bp in enumerate(bps):
            rows.append({"frame": f, "individual": individual, "bodypart": bp,
                         "x": float(i), "y": float(i + 10),
                         "likelihood": -1.0 if (f == 0 and i == 0) else 0.9})
    return pd.DataFrame(rows)


def main():
    src = MOD.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        print(f"FAIL: parse — {e}")
        print("smoke_122gm_fdlc_parquet_importer: 0/8 checks passed")
        return 1
    check("fdlc_parquet_importer.py parses", True)

    cls = next((n for n in tree.body if isinstance(n, ast.ClassDef)
                and n.name == "FDLCParquetImporter"), None)
    check("FDLCParquetImporter extends ConfigReader + PoseImporterMixin",
          cls is not None and {b.id for b in cls.bases if isinstance(b, ast.Name)}
          >= {"ConfigReader", "PoseImporterMixin"})

    init = next((m for m in cls.body if isinstance(m, ast.FunctionDef)
                 and m.name == "__init__"), None) if cls else None
    args = {a.arg for a in init.args.args} if init else set()
    check("__init__ mirrors h5 signature",
          {"config_path", "data_folder", "interpolation_settings",
           "smoothing_settings", "p_threshold"} <= args)

    check("dispatches on *.fdlc.parquet suffix", ".fdlc.parquet" in src)

    ns = _load_pure(src)
    ltw, resolve = ns["long_to_wide"], ns["_resolve_bodypart_order"]
    bps = ["nose", "ear_left", "tailbase"]

    wide = ltw(_long(bps), bps)
    check("long_to_wide pivots to <bp>_x/_y/_p in project order",
          list(wide.columns) == [f"{bp}_{s}" for bp in bps for s in ("x", "y", "p")]
          and len(wide) == 3)

    check("likelihood -1.0 sentinel clamped to 0.0",
          float(wide[[c for c in wide.columns if c.endswith("_p")]].min().min()) >= 0.0
          and wide.iloc[0]["nose_p"] == 0.0)

    # name-based reorder + case-insensitive + multi-animal reject
    rev = ltw(_long(bps), bps[::-1])
    ci = ltw(_long([b.upper() for b in bps]), bps)  # data upper, project lower
    multi_ok = False
    try:
        ltw(_long(bps, individual="a").assign(individual="a").pipe(
            lambda d: pd.concat([d, _long(bps, individual="b")])), bps)
    except ns["BodypartColumnNotFoundError"]:
        multi_ok = True
    check("name alignment: reorder + case-insensitive; multi-animal rejected",
          rev.columns[0] == "tailbase_x" and list(ci.columns[:1]) == ["nose_x"] and multi_ok)

    form = FORM.read_text(encoding="utf-8")
    check("registered in pose-import UI registry",
          "FreeDLC parquet (single animal)" in form and "FDLCParquetImporter" in form)

    print(f"smoke_122gm_fdlc_parquet_importer: {P}/{T} checks passed")
    return 0 if P == T else 1


if __name__ == "__main__":
    sys.exit(main())
