"""
tests/smoke_122ha_import_validation.py
======================================

Patch 122ha — pose-mismatch validation moved to the import step.

A project has one pose model, so an imported file either speaks its marker
names or it does not belong. That check belongs at the door: a mismatched
file does not fail on its own, it writes columns nothing downstream can find
and only surfaces later as all-NaN arrays and an "EKF has diverged" message
(cf. 122gv/122gx).

THE HOLE THIS CLOSES: FDLCParquetImporter._resolve_bodypart_order fell back
to POSITIONAL alignment whenever the marker counts matched — it zipped the
two lists and trusted the data's node order. So after renaming a project's
15 markers, an old 15-marker file still imported "successfully", mapping
nose->head_nose, headmid->head_mid, ... and writing plausible-looking but
wrong columns. Counts matching is not the same as poses matching.

Shared helpers now live on PoseImporterMixin — the base every importer
inherits — so pre-flight checks and real imports can't disagree:
markers_from_pose_file, describe_marker_mismatch, validate_pose_markers.
session_ingest delegates to them rather than keeping a second copy.
"""
import sys
import tempfile
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

if "tkinter" not in sys.modules:
    _tk = types.ModuleType("tkinter")
    _tk.messagebox = types.ModuleType("tkinter.messagebox")
    _tk.messagebox.showerror = lambda *a, **k: None
    sys.modules["tkinter"] = _tk
    sys.modules["tkinter.messagebox"] = _tk.messagebox

P = T = 0


def check(label, cond, *, detail=""):
    global P, T
    T += 1
    if cond:
        P += 1
    else:
        print(f"FAIL: {label}" + (f" — {detail}" if detail else ""))


def main():
    from mufasa.mixins.pose_importer_mixin import (
        describe_marker_mismatch,
        markers_from_pose_file,
        validate_pose_markers,
    )
    from mufasa.pose_importers.fdlc_parquet_importer import _resolve_bodypart_order
    from mufasa.utils.errors import BodypartColumnNotFoundError

    PROJ = ["head_nose", "head_mid", "head_back"]
    OLD = ["nose", "headmid", "neck"]

    # --- the regression this patch exists for ---
    raised = ""
    try:
        _resolve_bodypart_order(OLD, PROJ, source="old.fdlc.parquet")
    except BodypartColumnNotFoundError as e:
        raised = str(e)
    check("same-count/different-name file is REJECTED (no positional fallback)",
          bool(raised) and "head_nose" in raised and "nose" in raised,
          detail=raised[:70])

    src = (REPO / "mufasa" / "pose_importers"
           / "fdlc_parquet_importer.py").read_text()
    check("the positional zip is gone from the resolver",
          "return list(zip(project_bps, data_bps))" not in src)

    # --- legitimate alignments still work ---
    check("exact names resolve",
          _resolve_bodypart_order(PROJ, PROJ, source="x")
          == [(b, b) for b in PROJ])
    check("marker order doesn't matter (aligned by name)",
          sorted(_resolve_bodypart_order(PROJ[::-1], PROJ, source="x"))
          == sorted([(b, b) for b in PROJ]))
    check("case-insensitive match still resolves",
          _resolve_bodypart_order([b.upper() for b in PROJ], PROJ, source="x")
          == [(b, b.upper()) for b in PROJ])
    check("extra nodes in the data are tolerated (project is a subset)",
          _resolve_bodypart_order(PROJ + ["spare"], PROJ, source="x")
          == [(b, b) for b in PROJ])

    # --- shared helpers ---
    check("describe_marker_mismatch: names the difference, empty on match",
          "missing" in describe_marker_mismatch(OLD, PROJ)
          and describe_marker_mismatch(PROJ[::-1], PROJ) == "")

    ok = True
    try:
        validate_pose_markers(PROJ[::-1], PROJ, source="x")   # must not raise
    except BodypartColumnNotFoundError:
        ok = False
    raised2 = False
    try:
        validate_pose_markers(OLD, PROJ, source="x")
    except BodypartColumnNotFoundError:
        raised2 = True
    check("validate_pose_markers: passes a reorder, raises a mismatch",
          ok and raised2)

    # --- marker reading, and session_ingest delegating to it ---
    import numpy as np
    import pandas as pd
    d = Path(tempfile.mkdtemp())
    long = pd.DataFrame([{"frame": f, "individual": "single", "bodypart": b,
                          "x": 1.0, "y": 2.0, "likelihood": 0.9}
                         for f in range(2) for b in PROJ])
    long.to_parquet(d / "a.fdlc.parquet")
    wide = pd.DataFrame(np.zeros((2, 9)), columns=pd.MultiIndex.from_tuples(
        [("IMPORTED_POSE", "IMPORTED_POSE", f"{b}_{s}")
         for b in PROJ for s in ("x", "y", "p")]))
    wide.to_parquet(d / "b.parquet")
    check("markers_from_pose_file reads both long and wide layouts",
          sorted(markers_from_pose_file(str(d / "a.fdlc.parquet"))) == sorted(PROJ)
          and sorted(markers_from_pose_file(str(d / "b.parquet"))) == sorted(PROJ))

    ing = (REPO / "mufasa" / "model" / "session_ingest.py").read_text()
    check("session_ingest delegates instead of duplicating the logic",
          "markers_from_pose_file" in ing
          and "describe_marker_mismatch" in ing
          and "pyarrow.compute" not in ing)

    print(f"smoke_122ha_import_validation: {P}/{T} checks passed")
    return 0 if P == T else 1


if __name__ == "__main__":
    sys.exit(main())
