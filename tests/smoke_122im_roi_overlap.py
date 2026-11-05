"""Smoke test for patch 122im — overlapping ROIs stay reachable.

Before: a click resolved to the top-most (most recently drawn) shape whose
area contained it, so an ROI lying under a newer one could not be selected,
and its edges / polygon vertices could not be dragged where the newer one
covered them.

Now: handles (vertices, corners, circle rim handle) of every ROI come first,
then the nearest outline of any ROI (rectangle side, circle rim, polygon
edge), then interiors smallest-first; clicking the same spot again cycles
through the overlapping ROIs.

Runs for real: the ROI editor offscreen on a generated project, driven by
QTest mouse events; geometry read back from ROILogic.
"""
from __future__ import annotations

import os
import sys
import tempfile
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
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


app = QApplication.instance() or QApplication([])

from mufasa.project_layout import project_paths_from_config  # noqa: E402
from mufasa.roi_tools.roi_logic import ROILogic  # noqa: E402
from mufasa.ui_qt.dialogs.roi_define_panel import ROIDefineWidget  # noqa: E402
from mufasa.utils.config_creator import ProjectConfigCreator  # noqa: E402

tmp = Path(tempfile.mkdtemp())
cfg = ProjectConfigCreator(
    project_path=str(tmp), project_name="ov", target_list=[],
    pose_estimation_bp_cnt="user_defined", body_part_config_idx=0,
    animal_cnt=1, file_type="parquet", body_parts=["a", "b"],
).config_path
vd = Path(project_paths_from_config(cfg)["video_dir"])
vd.mkdir(parents=True, exist_ok=True)
w = cv2.VideoWriter(str(vd / "v.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                    30, (640, 360))
for _ in range(3):
    w.write(np.full((360, 640, 3), 40, np.uint8))
w.release()

# Older ROIs first, then a big one drawn last on top of all of them.
L = ROILogic(config_path=cfg, video_path=str(vd / "v.mp4"))
S = dict(color_name="Red", bgr=(0, 0, 255), thickness=2, ear_tag_size=8)
L.add_rectangle(name="small", top_left=(100, 100), bottom_right=(160, 160), **S)
L.add_polygon(name="poly", vertices=[(200, 100), (260, 100), (260, 160),
                                     (200, 160)], **S)
L.add_circle(name="cup", center=(130, 230), radius=25, **S)
L.add_rectangle(name="big", top_left=(50, 50), bottom_right=(350, 320), **S)
L.save()

W = ROIDefineWidget(config_path=cfg)
W.resize(1500, 900)
W.show()
app.processEvents()
C = W.preview
QTest.keyClick(C, Qt.Key_V)


def wpt(fx, fy) -> QPoint:
    return QPoint(*C._mapping.frame_to_widget(fx, fy))


def click(fx, fy):
    QTest.mouseClick(C, Qt.LeftButton, Qt.NoModifier, wpt(fx, fy))


def drag(a, b):
    QTest.mousePress(C, Qt.LeftButton, Qt.NoModifier, wpt(*a))
    QTest.mouseMove(C, wpt((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))
    QTest.mouseMove(C, wpt(*b))
    QTest.mouseRelease(C, Qt.LeftButton, Qt.NoModifier, wpt(*b))


def wobbly_click(fx, fy, jitter=(2, 1)):
    """A real click: the mouse moves a pixel or two between press and
    release (QTest.mouseClick never moves, which hid the 122in bug)."""
    p = wpt(fx, fy)
    QTest.mousePress(C, Qt.LeftButton, Qt.NoModifier, p)
    QTest.mouseMove(C, p + QPoint(1, 0))
    QTest.mouseMove(C, p + QPoint(*jitter))
    QTest.mouseRelease(C, Qt.LeftButton, Qt.NoModifier, p + QPoint(*jitter))


def selected() -> str | None:
    i = C.selected_index()
    return None if i is None else W._overlay_idx_to_kind_name[i][1]


def g(name):
    return W.logic.get_roi(name).geometry


click(130, 130)
check("a click inside an ROI covered by a newer, larger one selects the "
      "covered (smaller) ROI", selected() == "small")
check("…and highlights its table row",
      W._overlay_idx_to_kind_name[W._selected_row()][1] == "small")
click(130, 130)
check("clicking the same spot again cycles to the ROI beneath",
      selected() == "big")
click(130, 130)
check("…and around again", selected() == "small")

# 122in: the same with realistic, slightly moving clicks.
before = (dict(g("small")), dict(g("big")))
wobbly_click(130, 130)
check("a wobbly click (cursor moves 2 px) still cycles to the ROI beneath",
      selected() == "big")
wobbly_click(130, 130)
check("…and back", selected() == "small")
check("wobbly clicks never nudge the ROIs",
      (dict(g("small")), dict(g("big"))) == before)
wobbly_click(160, 130)                # on small's right side
check("a wobbly click on a side doesn't resize it",
      dict(g("small")) == before[0])
wobbly_click(130, 130)                # 'small' is selected: cycle to 'big'
wobbly_click(130, 130)                # and back to 'small'

drag((130, 130), (140, 135))
check("dragging inside moves the covered ROI, not the newer one on top",
      (g("small")["topLeftX"], g("small")["topLeftY"]) == (110, 105)
      and g("big")["topLeftX"] == 50)

big_before = dict(g("big"))
drag((170, 120), (190, 120))          # small's right side, inside 'big'
check("a side of a covered rectangle can be dragged",
      g("small")["Bottom_right_X"] == 190 and g("small")["topLeftX"] == 110
      and g("big") == big_before)
drag((110, 105), (90, 85))            # small's top-left corner
check("a corner of a covered rectangle can be dragged",
      (g("small")["topLeftX"], g("small")["topLeftY"]) == (90, 85)
      and g("small")["Bottom_right_X"] == 190)

drag((260, 160), (280, 180))          # poly vertex 2, under 'big'
check("a vertex of a covered polygon can be dragged directly",
      g("poly")["vertices"][2] == [280, 180]
      and g("poly")["vertices"][0] == [200, 100])
drag((230, 100), (230, 80))           # poly top edge (v0-v1)
check("dragging a polygon edge moves both of its vertices",
      g("poly")["vertices"][0] == [200, 80] and g("poly")["vertices"][1] == [260, 80]
      and g("poly")["vertices"][2] == [280, 180])

drag((130, 205), (130, 195))          # top of the cup's rim
check("the rim of a covered circle can be dragged to resize it",
      g("cup")["radius"] == 35 and (g("cup")["centerX"], g("cup")["centerY"]) == (130, 230))

drag((350, 200), (380, 200))          # big's right side
check("the newer ROI's own edges still work",
      g("big")["Bottom_right_X"] == 380)
click(330, 300)
check("a click only inside the big ROI selects it", selected() == "big")
click(600, 340)
check("a click on empty frame deselects", selected() is None)

n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122im_roi_overlap: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
