"""Smoke test for patch 122ip — calibration actually gets saved.

User report: "I calibrated already" but Compute Features said video_info.csv
was missing. The Video Calibration form inherited the standard Run button;
Run called a no-op target() and reported "Done." — only a separate "Save to
video_info.csv" button wrote the file. Now Run is "Save calibration" and
saves; unsaved edits are flagged; and the missing-file error names
Preprocessing → Video Calibration instead of SimBA's "[Video parameters] tab".

Runs for real: the VideoInfoForm offscreen on a generated project, the
saved CSV read back by read_video_info_csv / ConfigReader.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_tk = types.ModuleType("tkinter")
_tk.messagebox = types.ModuleType("tkinter.messagebox")
_tk.messagebox.showerror = lambda *a, **k: None
sys.modules.setdefault("tkinter", _tk)
sys.modules.setdefault("tkinter.messagebox", _tk.messagebox)
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


app = QApplication.instance() or QApplication([])
seen: list[tuple[str, str]] = []
QMessageBox.information = staticmethod(
    lambda *a, **k: seen.append(("info", str(a[2]) if len(a) > 2 else "")))
QMessageBox.warning = staticmethod(
    lambda *a, **k: seen.append(("warn", str(a[2]) if len(a) > 2 else "")))

from mufasa.mixins.config_reader import ConfigReader  # noqa: E402
from mufasa.project_layout import project_paths_from_config  # noqa: E402
from mufasa.ui_qt.forms import video_info as M  # noqa: E402
from mufasa.utils.config_creator import ProjectConfigCreator  # noqa: E402
from mufasa.utils.errors import NoFilesFoundError  # noqa: E402
from mufasa.utils.read_write import read_video_info_csv  # noqa: E402

tmp = Path(tempfile.mkdtemp())
cfg = ProjectConfigCreator(
    project_path=str(tmp), project_name="vi", target_list=[],
    pose_estimation_bp_cnt="user_defined", body_part_config_idx=0,
    animal_cnt=1, file_type="parquet", body_parts=["a", "b"],
).config_path
P = project_paths_from_config(cfg)
vd = Path(P["video_dir"])
vd.mkdir(parents=True, exist_ok=True)
for stem in ("v1", "v2"):
    w = cv2.VideoWriter(str(vd / f"{stem}.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 360))
    for _ in range(3):
        w.write(np.zeros((360, 640, 3), np.uint8))
    w.release()
info = Path(P["video_info_path"])

# ------------------------------------------------ missing-file message ---
try:
    read_video_info_csv(str(info))
    msg = ""
except NoFilesFoundError as exc:
    msg = str(getattr(exc, "msg", exc))
check("missing video_info.csv: the error points to Preprocessing → Video "
      "Calibration, not a '[Video parameters] tab'",
      "Preprocessing → Video Calibration" in msg
      and "[Video parameters]" not in msg and str(info) in msg)

# ------------------------------------------------------------ the form ---
win = QMainWindow()
f = M.VideoInfoForm(config_path=cfg)
win.setCentralWidget(f)
win.show()
app.processEvents()
check("form lists the project's videos with fps / resolution from the files",
      f.table.rowCount() == 2
      and f.table.item(0, M._COL_FPS).text().startswith("30")
      and f.table.item(0, M._COL_WIDTH).text() == "640")
check("a fresh load is not flagged as unsaved", not f._dirty)

for r in range(2):        # what calibrating writes into the row
    f.table.item(r, M._COL_DISTANCE).setText("100")
    f.table.item(r, M._COL_PX_PER_MM).setText("2.5")
check("calibrating / editing flags unsaved changes",
      f._dirty and "Unsaved changes" in f.status_label.text())
check("the form's main button is 'Save calibration'",
      "Save calibration" in f.run_btn.text())

f.run_btn.click()         # what the user pressed
t0 = time.time()
while time.time() - t0 < 2:
    app.processEvents()
check("pressing it writes video_info.csv (it used to say 'Done.' and write "
      "nothing)", info.is_file())
check("…and reports where it saved, without a bogus 'Done.'",
      any(k == "info" and str(info) in m for k, m in seen)
      and ("info", "Done.") not in seen)
check("unsaved flag clears after saving", not f._dirty)

df = read_video_info_csv(str(info))
check("feature computation can read it back",
      list(df["Video"]) == ["v1", "v2"]
      and (df["pixels/mm"] == 2.5).all() and (df["fps"] == 30).all())
cr = ConfigReader(config_path=cfg, read_video_info=True)
check("ConfigReader (used by every analysis) loads it",
      cr.video_info_df is not None and len(cr.video_info_df) == 2)

f2 = M.VideoInfoForm(config_path=cfg)
check("re-opening the form shows the saved calibration",
      f2.table.item(1, M._COL_PX_PER_MM).text().startswith("2.5")
      and not f2._dirty)
f2.table.item(1, M._COL_PX_PER_MM).setText("")
seen.clear()
f2.run_btn.click()
check("an incomplete row blocks the save with a warning (file untouched)",
      any(k == "warn" for k, _ in seen)
      and (pd.read_csv(info)["pixels/mm"] == 2.5).all())

n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122ip_calibration_save: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
