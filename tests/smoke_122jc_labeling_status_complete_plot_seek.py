"""Smoke test for patch 122jc — Frame Labeling status table, "Mark video
complete", and click-to-seek in the label plot.

User request: in the Frame Labeling section a table showing which videos
already have annotations; in the labeller a mark-complete checkbox; and
left-clicking in the label plot moves to that time point.

Runs for real, offscreen: a generated project with three videos (one
labelled, one labelled and marked complete, one untouched), the Frame
Labeling form, and a FrameLabellerWidget on a real video file:

* table rows / states / counts / summary, numeric sorting, live refresh
  when the labeller saves or marks complete;
* "Label" opens the selected video (no file dialog); "New" on a labelled
  video asks first and can switch to Continue;
* the checkbox reflects and writes derived/labels/annotation_status.toml,
  saving unsaved labels before marking complete;
* plot clicks map to the drawn frame, seek the video, ignore the name
  column; dragging scrubs;
* TOML writer quotes keys that are not bare keys (video names with dots,
  classifier names with spaces) — they used to produce unreadable files.
"""
from __future__ import annotations

import os
import sys
import tempfile
import tomllib
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
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


app = QApplication.instance() or QApplication([])

import mufasa.ui_qt.frame_labeller as FLB  # noqa: E402
from mufasa.project_layout import (  # noqa: E402
    project_paths_from_config,
    read_project_toml,
    write_project_toml,
)
from mufasa.ui_qt.forms.annotation import FrameLabellingLauncher  # noqa: E402
from mufasa.utils import annotation_status as AS  # noqa: E402
from mufasa.utils.config_creator import ProjectConfigCreator  # noqa: E402

# ---- TOML keys -----------------------------------------------------------------
tmp = Path(tempfile.mkdtemp())
cfg = ProjectConfigCreator(
    project_path=str(tmp), project_name="lab", target_list=["rear", "head dip"],
    pose_estimation_bp_cnt="user_defined", body_part_config_idx=0,
    animal_cnt=1, file_type="parquet", body_parts=["a", "b"]).config_path
data = read_project_toml(Path(cfg))
data.setdefault("classifiers", {})["keys"] = {"rear": "r", "head dip": "h"}
data["pose"]["segments"] = [{"name": "body", "markers": ["a"]},
                            {"name": "head", "parent": "body", "markers": ["b"]}]
write_project_toml(Path(cfg), data)
back = read_project_toml(Path(cfg))
check("project.toml with a spaced classifier key round-trips (quoted key)",
      back["classifiers"]["keys"] == {"rear": "r", "head dip": "h"}
      and '"head dip" = "h"' in Path(cfg).read_text()
      and back["pose"]["segments"][1]["parent"] == "body")

check("video-name extension stripping keeps dotted names",
      [AS.strip_media_ext(x) for x in ("rec.mp4", "a/b/rec.MP4", "rec.v2", "rec.v2.parquet")]
      == ["rec", "rec", "rec.v2", "rec.v2"])
from mufasa.utils import classification_io, csv_export, feature_io, label_io  # noqa: E402

check("label / feature / classification / CSV-export helpers all use it",
      all(m._strip_video_ext("rec.v2") == "rec.v2" and m._strip_video_ext("rec.v2.mp4") == "rec.v2"
          for m in (label_io, feature_io, classification_io, csv_export)))

# ---- project with three videos ---------------------------------------------
P = project_paths_from_config(cfg)
vd = Path(P["video_dir"])
vd.mkdir(parents=True, exist_ok=True)
N = 120
for v in ("v_done", "v_prog", "v.new"):
    w = cv2.VideoWriter(str(vd / f"{v}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    for k in range(N):
        w.write(np.full((48, 64, 3), k % 255, np.uint8))
    w.release()
ld = Path(P["derived_labels_dir"])
ld.mkdir(parents=True, exist_ok=True)
lab = np.zeros(N, int)
lab[10:40] = 1
pd.DataFrame({"rear": lab, "head dip": np.zeros(N, int)}).astype("Int64").to_parquet(ld / "v_done.parquet")
lab2 = np.zeros(N, int)
lab2[5:11] = 1
pd.DataFrame({"rear": lab2, "head dip": lab2}).astype("Int64").to_parquet(ld / "v_prog.parquet")
AS.set_complete(cfg, "v_done", True)
AS.set_complete(cfg, "v.new", False)
st = tomllib.loads(AS.status_path(cfg).read_text())
check("status file is valid TOML with dotted video names quoted",
      st["videos"]["v_done"]["complete"] is True and "v.new" in st["videos"])
rows = {r.video: r for r in AS.summarize(cfg, ["rear", "head dip"])}
check("summary: states complete / in progress / not started",
      rows["v_done"].state == "complete" and rows["v_prog"].state == "in progress"
      and rows["v.new"].state == "not started")
check("summary: labelled frame counts per behaviour",
      rows["v_done"].labelled == {"rear": 30, "head dip": 0}
      and rows["v_prog"].labelled == {"rear": 6, "head dip": 6} and rows["v_done"].n_frames == N)
check("status file is not mistaken for a labelled video",
      "annotation_status" not in rows)

# ---- the Frame Labeling form ------------------------------------------------
form = FrameLabellingLauncher(config_path=cfg)
form.resize(900, 600)
form.show()
app.processEvents()
t = form.status_table


def table() -> dict[str, list[str]]:
    return {t.item(i, 0).text(): [t.item(i, c).text() for c in range(t.columnCount())]
            for i in range(t.rowCount())}


tb = table()
check("one row per video", set(tb) == {"v_done", "v_prog", "v.new"})
check("status column shows complete / in progress / not started",
      tb["v_done"][1].endswith("complete") and tb["v_prog"][1].endswith("in progress")
      and tb["v.new"][1].endswith("not started"))
check("frames-labelled column (after Status): frames with any behaviour / total",
      t.horizontalHeaderItem(2).text() == "Frames labelled"
      and tb["v_done"][2] == "30 / 120 (25.0%)" and tb["v.new"][2] == "")
check("…a frame carrying two behaviours counts once (v_prog: 6 + 6 on the same frames → 6)",
      tb["v_prog"][2] == "6 / 120 (5.0%)" and rows["v_prog"].any_labelled == 6)
check("per-behaviour column shows counts and percentages",
      tb["v_done"][3] == "rear 30 (25.0%) · head dip 0 (0.0%)" and tb["v.new"][3] == "")
check("last-saved shown for labelled videos only", tb["v_done"][4] and not tb["v.new"][4])
check("summary line counts the states",
      "3</b> videos" in form.status_summary.text() and "1</b> complete" in form.status_summary.text()
      and "1</b> in progress" in form.status_summary.text())
t.sortItems(1, Qt.DescendingOrder)
check("status sorts by progress (complete first, descending)",
      t.item(0, 0).text() == "v_done" and t.item(2, 0).text() == "v.new")
t.sortItems(2, Qt.DescendingOrder)
check("frames labelled sorts numerically", t.item(0, 0).text() == "v_done"
      and t.item(1, 0).text() == "v_prog")

launched = []
FLB_launch = FLB.launch_frame_labeller
FLB.launch_frame_labeller = lambda *a, **k: launched.append(k)
form.select_video("v.new")
form.mode_cb.setCurrentIndex(0)                      # New
form._launch()
check("Label opens the selected video directly (no file dialog)",
      launched and launched[-1]["video_path"].endswith("v.new.mp4")
      and launched[-1]["mode"] == "new")
asked = []
orig_q = QMessageBox.question
QMessageBox.question = lambda *a, **k: (asked.append(a[2]), QMessageBox.Yes)[1]
form.select_video("v_prog")
form._launch()
check("New on a labelled video asks first and can switch to Continue",
      asked and "already has labels" in asked[-1] and launched[-1]["mode"] == "continue")
QMessageBox.question = lambda *a, **k: QMessageBox.Cancel
n_before = len(launched)
form._launch()
check("…and Cancel opens nothing", len(launched) == n_before)
QMessageBox.question = orig_q
r_done = next(i for i in range(t.rowCount()) if t.item(i, 0).text() == "v_done")
form.mode_cb.setCurrentIndex(1)                      # Continue
t.cellDoubleClicked.emit(r_done, 0)
check("double-clicking a row opens that video", launched[-1]["video_path"].endswith("v_done.mp4")
      and launched[-1]["mode"] == "continue")
FLB.launch_frame_labeller = FLB_launch

# ---- the labeller: mark complete --------------------------------------------
W = FLB.FrameLabellerWidget(cfg, str(vd / "v.new.mp4"), mode="new")
W.resize(1000, 700)
W.show()
app.processEvents()
check("checkbox starts unchecked for an unfinished video", not W.complete_cb.isChecked())
W._labels["rear"][50:60] = 1
W._dirty = True
W.complete_cb.setChecked(True)
app.processEvents()
check("marking complete saves the unsaved labels first",
      not W.is_dirty() and (ld / "v.new.parquet").is_file()
      and int(pd.read_parquet(ld / "v.new.parquet")["rear"].sum()) == 10)
check("…and records it in annotation_status.toml", AS.is_complete(cfg, "v.new"))
check("the open Frame Labeling table refreshes live",
      table()["v.new"][1].endswith("complete") and table()["v.new"][2] == "10 / 120 (8.3%)"
      and table()["v.new"][3].startswith("rear 10"))
W.complete_cb.setChecked(False)
app.processEvents()
check("unchecking reopens it", not AS.is_complete(cfg, "v.new")
      and table()["v.new"][1].endswith("in progress"))
W2 = FLB.FrameLabellerWidget(cfg, str(vd / "v_done.mp4"), mode="continue")
check("checkbox starts checked for a completed video", W2.complete_cb.isChecked())
W2.cleanup()

# ---- the labeller: click / drag in the label plot ---------------------------
plot = W.timeseries_plot
W.scrubber.seek(60)
app.processEvents()
x_mid = plot._label_col_w + (plot.width() - plot._label_col_w - 8) / 2
check("plot centre maps to the current frame", plot.frame_at(x_mid) == 60)
half = int(round(plot._window_seconds * plot._fps))
x_right = plot._label_col_w + 0.75 * (plot.width() - plot._label_col_w - 8)
expect = 60 + int(round(0.25 * 2 * half))
QTest.mouseClick(plot, Qt.LeftButton, Qt.NoModifier, QPoint(int(x_right), 20))
app.processEvents()
check(f"left-click right of centre seeks forward to the clicked frame "
      f"(→ {W.scrubber.current_frame}, expected ≈{expect})",
      abs(W.scrubber.current_frame - expect) <= 1)
QTest.mouseClick(plot, Qt.LeftButton, Qt.NoModifier, QPoint(20, 20))
app.processEvents()
check("a click on the behaviour-name column does nothing",
      abs(W.scrubber.current_frame - expect) <= 1)
W.scrubber.seek(60)
app.processEvents()
QTest.mouseClick(plot, Qt.LeftButton, Qt.NoModifier,
                 QPoint(plot._label_col_w + 1, 20))
app.processEvents()
check("clicking the window's left edge goes back by the window half-width "
      "(clamped at frame 0)", W.scrubber.current_frame == max(0, 60 - half))
W.scrubber.seek(60)
app.processEvents()
x0 = int(x_mid)
QTest.mousePress(plot, Qt.LeftButton, Qt.NoModifier, QPoint(x0, 20))
for dx in (10, 20, 40):
    QTest.mouseMove(plot, QPoint(x0 + dx, 20))
    app.processEvents()
QTest.mouseRelease(plot, Qt.LeftButton, Qt.NoModifier, QPoint(x0 + 40, 20))
fpp = 2 * half / (plot.width() - plot._label_col_w - 8)
check(f"dragging scrubs by the distance dragged "
      f"(→ {W.scrubber.current_frame}, expected ≈{60 + 40 * fpp:.0f})",
      abs(W.scrubber.current_frame - (60 + 40 * fpp)) <= 1.5)
W.scrubber.seek(N - 3)
app.processEvents()
QTest.mouseClick(plot, Qt.LeftButton, Qt.NoModifier,
                 QPoint(plot.width() - 10, 20))
app.processEvents()
check("seeking past the end clamps to the last frame",
      W.scrubber.current_frame == W.scrubber.total_frames - 1)
W.cleanup()

n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122jc_labeling_status_complete_plot_seek: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
