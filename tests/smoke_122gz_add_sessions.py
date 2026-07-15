"""
tests/smoke_122gz_add_sessions.py
=================================

Patch 122gz — Add sessions: import new recordings matching the project's
pose and refresh their derived data with the latest model.

The contract is "same format and same pose as the originals", and the pose
check is the substance: a mismatched import does not fail loudly, it
produces columns nothing downstream can find and surfaces much later as
all-NaN arrays (cf. 122gv/122gx). So files are validated before anything is
written, and rejects name the exact difference.

* check_pose_compatibility(source, config) — file or folder; accepts only
  when the marker set EQUALS the project's body_parts (order-independent —
  importers align by name). Reads markers without loading whole files, and
  understands both FreeDLC long format (a `bodypart` column) and the wide
  <bp>_x/_y/_p layout (flat or IMPORTED_POSE MultiIndex).
* find_latest_smoothing_model(config) — newest models/<name>/model.npz by
  mtime, or None (caller must train; this never trains).
* ingest_sessions(...) — import the accepted files, then optionally smooth
  ONLY those, reusing the latest model into a fresh run dir.
"""
import sys
import tempfile
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

if "tkinter" not in sys.modules:  # read_write imports tkinter (legacy)
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
    import numpy as np
    import pandas as pd

    from mufasa.model.session_ingest import (
        check_pose_compatibility,
        find_latest_smoothing_model,
        ingest_sessions,
    )
    from mufasa.project_layout import (
        PROJECT_LAYOUT_VERSION,
        project_paths_from_config,
        write_project_toml,
    )
    from mufasa.utils.read_write import write_df

    BP = ["head_nose", "head_mid", "head_back"]
    d = Path(tempfile.mkdtemp())
    cp = d / "project.toml"
    write_project_toml(cp, {"project_layout_version": PROJECT_LAYOUT_VERSION,
                            "pose": {"body_parts": BP, "file_type": "parquet"}})
    inc = Path(tempfile.mkdtemp())

    def wide(names, path, n=10):
        cols = pd.MultiIndex.from_tuples(
            [("IMPORTED_POSE", "IMPORTED_POSE", f"{b}_{s}")
             for b in names for s in ("x", "y", "p")])
        write_df(pd.DataFrame(np.zeros((n, len(names) * 3)), columns=cols),
                 "parquet", str(path), multi_idx_header=True)

    wide(BP, inc / "good.parquet")
    wide(BP[::-1], inc / "reordered.parquet")          # same set, other order
    wide(["nose", "headmid", "neck"], inc / "oldpose.parquet")
    wide(BP[:2], inc / "partial.parquet")
    pd.DataFrame([{"frame": f, "individual": "single", "bodypart": b,
                   "x": 1.0, "y": 2.0, "likelihood": 0.9}
                  for f in range(3) for b in BP]).to_parquet(
        inc / "new.fdlc.parquet")

    rep = check_pose_compatibility(inc, cp)
    acc = {Path(p).name for p in rep["accepted"]}
    check("accepts matching wide + FreeDLC long files",
          {"good.parquet", "new.fdlc.parquet"} <= acc, detail=str(acc))

    check("marker ORDER doesn't matter (importers align by name)",
          "reordered.parquet" in acc)

    rej = {Path(p).name: w for p, w in rep["rejected"].items()}
    check("rejects a file whose pose differs, naming the difference",
          "oldpose.parquet" in rej and "missing" in rej["oldpose.parquet"],
          detail=str(rej))
    check("rejects a file missing a marker",
          "partial.parquet" in rej and "head_back" in rej["partial.parquet"])

    check("no model yet -> None (never invents one)",
          find_latest_smoothing_model(cp) is None)

    models = Path(project_paths_from_config(cp)["models_dir"])
    (models / "old_model").mkdir(parents=True)
    (models / "old_model" / "model.npz").write_bytes(b"x")
    import os
    import time
    time.sleep(0.01)
    (models / "new_model").mkdir(parents=True)
    (models / "new_model" / "model.npz").write_bytes(b"x")
    os.utime(models / "new_model" / "model.npz", (time.time() + 10,) * 2)
    latest = find_latest_smoothing_model(cp)
    check("latest model = most recently modified",
          latest is not None and Path(latest).parent.name == "new_model",
          detail=str(latest))

    s = ingest_sessions(cp, inc, dry_run=True)
    check("dry run reports without importing",
          s["dry_run"] and s["imported"] == [] and len(s["accepted"]) >= 2)

    empty = Path(tempfile.mkdtemp())
    s2 = ingest_sessions(cp, empty, dry_run=True)
    check("nothing matching -> a note, not a crash",
          s2["imported"] == [] and s2["notes"])

    page = (REPO / "mufasa" / "ui_qt" / "pages" / "data_import_page.py").read_text()
    form = (REPO / "mufasa" / "ui_qt" / "forms" / "add_sessions.py").read_text()
    check("form wired into the Import pose data section",
          "AddSessionsForm" in page
          and "check_pose_compatibility" in form and "ingest_sessions" in form)

    print(f"smoke_122gz_add_sessions: {P}/{T} checks passed")
    return 0 if P == T else 1


if __name__ == "__main__":
    sys.exit(main())
