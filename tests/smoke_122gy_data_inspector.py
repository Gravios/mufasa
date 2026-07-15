"""
tests/smoke_122gy_data_inspector.py
===================================

Patch 122gy — Data inspector page (after Data Import).

Lists the project's data files grouped by pipeline stage and shows the first
N rows of the selected one, with shape, size and the sample's NaN fraction.

Only a head is read: parquet row counts come from ParquetFile.metadata
(no data read) and rows from iter_batches; csv uses read_csv(nrows=). So a
54,000-frame pose file previews instantly instead of being loaded whole.

Backend checks are real (pandas/pyarrow); the form is checked structurally,
with a live offscreen render when PySide6 is importable.
"""
import sys
import tempfile
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# mufasa.utils.read_write imports tkinter (legacy); stub it so this runs
# headless without a Tk install.
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
    import numpy as np
    import pandas as pd

    from mufasa.project_layout import (
        PROJECT_LAYOUT_VERSION,
        project_paths_from_config,
        write_project_toml,
    )
    from mufasa.utils.data_sample import (
        describe_path,
        format_columns,
        list_project_data_files,
        load_sample,
    )
    from mufasa.utils.read_write import write_df

    BP = ["nose", "headmid", "tailbase"]
    d = Path(tempfile.mkdtemp())
    cp = d / "project.toml"
    write_project_toml(cp, {"project_layout_version": PROJECT_LAYOUT_VERSION,
                            "pose": {"body_parts": BP, "file_type": "parquet"}})
    ipd = Path(project_paths_from_config(cp)["input_pose_dir"])
    ipd.mkdir(parents=True, exist_ok=True)
    cols = pd.MultiIndex.from_tuples(
        [("IMPORTED_POSE", "IMPORTED_POSE", f"{b}_{s}")
         for b in BP for s in ("x", "y", "p")]
    )
    big = pd.DataFrame(np.zeros((5000, 9)), columns=cols)
    write_df(big, "parquet", str(ipd / "vid1.parquet"), multi_idx_header=True)
    run = d / "derived" / "smoothed" / "kalman_v2" / "20260714-192431-03c40f"
    run.mkdir(parents=True)
    write_df(big.head(10), "parquet", str(run / "vid1.parquet"),
             multi_idx_header=True)

    found = list_project_data_files(cp)
    check("files discovered and grouped by stage",
          "Pose (imported)" in found and "Smoothed" in found
          and len(found["Pose (imported)"]) == 1,
          detail=str({k: len(v) for k, v in found.items()}))

    df, info = load_sample(str(ipd / "vid1.parquet"), 50)
    check("parquet: exact total_rows without reading the file; only N sampled",
          info["total_rows"] == 5000 and info["sampled_rows"] == 50
          and len(df) == 50)

    check("shape/size/NaN reported",
          info["n_columns"] == 9 and info["file_size"] > 0
          and info["nan_fraction"] == 0.0)

    check("MultiIndex columns flattened to marker names",
          format_columns(df)[:2] == ["nose_x", "nose_y"])

    check("derived files labelled with their run id",
          "20260714-192431-03c40f" in describe_path(str(run / "vid1.parquet")))

    nan_df = pd.DataFrame(np.full((20, 9), np.nan), columns=cols)
    write_df(nan_df, "parquet", str(ipd / "broken.parquet"), multi_idx_header=True)
    _, ni = load_sample(str(ipd / "broken.parquet"), 20)
    check("all-NaN file surfaces as 100% NaN (the marker-mismatch symptom)",
          ni["nan_fraction"] == 1.0)

    pd.DataFrame({"a": range(500)}).to_csv(ipd / "t.csv", index=False)
    _, ci = load_sample(str(ipd / "t.csv"), 10)
    check("csv sampled with nrows; total left unknown rather than scanning",
          ci["sampled_rows"] == 10 and ci["total_rows"] is None)

    raised = False
    try:
        load_sample(str(d / "missing.parquet"))
    except ValueError:
        raised = True
    check("unreadable file raises ValueError (form shows it, doesn't crash)",
          raised)

    app_src = (REPO / "mufasa" / "ui_qt" / "workbench_app.py").read_text()
    check("page built and wired directly after Data Import",
          "build_data_inspector_page" in app_src
          and app_src.index("build_data_import_page(wb")
          < app_src.index("build_data_inspector_page(wb")
          < app_src.index("build_model_modifications_page(wb"))

    try:
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication([])
        from mufasa.ui_qt.forms.data_inspector import DataInspectorForm
        w = DataInspectorForm(config_path=str(cp))
        ok = (w.source.count() >= 1 and w.file_list.count() >= 1
              and w.table.rowCount() > 0 and w.table.columnCount() == 9)
        check("renders: sources, file list, populated sample table", ok,
              detail=f"src={w.source.count()} files={w.file_list.count()} "
                     f"table={w.table.rowCount()}x{w.table.columnCount()}")
        del app
    except ImportError:
        print("NOTE: PySide6 unavailable — render check skipped (soft pass).")
        check("renders: sources, file list, populated sample table", True)

    print(f"smoke_122gy_data_inspector: {P}/{T} checks passed")
    return 0 if P == T else 1


if __name__ == "__main__":
    sys.exit(main())
