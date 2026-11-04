"""Smoke test for patch 122ij — ROI editor redesign.

The headline bug: committing a polygon ran it through
shapely.simplify(tolerance=20), silently dropping every vertex within 20 frame
px of the line through its neighbours — on a 640x360 frame polygons came out
with ~4-8 vertices however many were clicked. Also redesigned: sticky tools
with automatic names (no "type a name, then press Draw" per shape), Esc
abandons instead of committing, a Select tool instead of a separate Edit
toggle, polygon vertex editing, editable name / colour / width in the table,
undo/redo, zoom, and a confirmation before "Apply to all" overwrites other
videos' ROIs.

Runs for real: a ProjectConfigCreator project with generated videos, the
ROIDefineWidget offscreen driven by QTest mouse / key events, ROILogic save,
and a fresh ROILogic reading the saved H5 back.
"""
from __future__ import annotations

import math
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
from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QWheelEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


app = QApplication.instance() or QApplication([])
asked: list[str] = []
answer = {"q": QMessageBox.No}
QMessageBox.question = staticmethod(
    lambda *a, **k: (asked.append(str(a[2]) if len(a) > 2 else ""),
                     answer["q"])[1])
QMessageBox.critical = staticmethod(lambda *a, **k: asked.append("critical"))

from mufasa.project_layout import project_paths_from_config  # noqa: E402
from mufasa.roi_tools.roi_logic import POLYGON, RECTANGLE, ROILogic  # noqa: E402
from mufasa.ui_qt.dialogs.roi_canvas import SELECT  # noqa: E402
from mufasa.ui_qt.dialogs.roi_define_panel import ROIDefineWidget  # noqa: E402
from mufasa.utils.config_creator import ProjectConfigCreator  # noqa: E402

tmp = Path(tempfile.mkdtemp())
cfg = ProjectConfigCreator(
    project_path=str(tmp), project_name="roi", target_list=[],
    pose_estimation_bp_cnt="user_defined", body_part_config_idx=0,
    animal_cnt=1, file_type="parquet", body_parts=["a", "b"],
).config_path
P = project_paths_from_config(cfg)
vd = Path(P["video_dir"])
vd.mkdir(parents=True, exist_ok=True)
for stem in ("vidA", "vidB", "vidC"):
    w = cv2.VideoWriter(str(vd / f"{stem}.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 360))
    for i in range(20):
        f = np.full((360, 640, 3), 40, np.uint8)
        cv2.circle(f, (100 + i * 10, 180), 20, (200, 200, 200), -1)
        w.write(f)
    w.release()

W = ROIDefineWidget(config_path=cfg)
W.resize(1500, 900)
W.show()
app.processEvents()
C = W.preview


def wpt(fx, fy) -> QPoint:
    return QPoint(*C._mapping.frame_to_widget(fx, fy))


def click(fx, fy, button=Qt.LeftButton):
    QTest.mouseClick(C, button, Qt.NoModifier, wpt(fx, fy))


def drag(a, b):
    QTest.mousePress(C, Qt.LeftButton, Qt.NoModifier, wpt(*a))
    QTest.mouseMove(C, wpt((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))
    QTest.mouseMove(C, wpt(*b))
    QTest.mouseRelease(C, Qt.LeftButton, Qt.NoModifier, wpt(*b))


def key(k, mod=Qt.NoModifier, widget=None):
    QTest.keyClick(widget or C, k, mod)


def names():
    return list(W.logic.all_roi_names)


# ------------------------------------------------- polygon: no vertex cap ---
check("panel opens on the first video in the Select tool",
      W.logic.video_name == "vidA" and C.tool() == SELECT)
key(Qt.Key_P)
check("P arms the polygon tool", C.tool() == POLYGON)
N = 24
ring = [(320 + 120 * math.cos(2 * math.pi * k / N),
         180 + 120 * math.sin(2 * math.pi * k / N)) for k in range(N)]
for x, y in ring:
    click(x, y)
key(Qt.Key_Return)
polys = W.logic.defs[POLYGON]
check("a 24-vertex polygon keeps all 24 vertices (was simplified to ~8)",
      len(polys) == 1
      and len(next(iter(polys.values())).geometry["vertices"]) == N)
check("it is auto-named polygon_1 (no name needed before drawing)",
      "polygon_1" in polys)
check("the polygon tool stays armed for the next shape", C.tool() == POLYGON)

for x, y in [(50, 50), (120, 50), (120, 120)]:
    click(x, y)
key(Qt.Key_Escape)
check("Esc abandons a polygon in progress (used to COMMIT it)",
      len(W.logic.defs[POLYGON]) == 1 and not C.is_drawing())

W.name_edit.setText("nest")
for x, y in [(30, 300), (90, 300), (90, 340), (30, 340)]:
    click(x, y)
click(30, 300)        # click the first vertex to close
check("clicking the first vertex closes; the Name box names it",
      "nest" in W.logic.defs[POLYGON]
      and len(W.logic.defs[POLYGON]["nest"].geometry["vertices"]) == 4
      and W.name_edit.text() == "")

# ---------------------------------------------------- rectangle / circle ---
key(Qt.Key_R)
drag((400, 40), (560, 120))
r = W.logic.defs[RECTANGLE].get("rectangle_1")
check("rectangle drawn by press-drag-release",
      r is not None and r.geometry["topLeftX"] == 400
      and r.geometry["Bottom_right_Y"] == 120)
key(Qt.Key_C)
drag((560, 280), (600, 280))
check("circle drawn from centre outward",
      W.logic.get_roi("circle_1") is not None
      and W.logic.get_roi("circle_1").geometry["radius"] == 40)
check("the table lists every ROI", W.shape_table.rowCount() == 4)

# ----------------------------------------------------------- select / edit ---
key(Qt.Key_V)
click(470, 80)
row = W._selected_row()
check("Select: clicking a shape selects its table row",
      row is not None and W._overlay_idx_to_kind_name[row][1] == "rectangle_1")
drag((470, 80), (480, 90))
g = W.logic.get_roi("rectangle_1").geometry
check("dragging the body moves it", (g["topLeftX"], g["topLeftY"]) == (410, 50))
drag((410 + 80, 50), (490, 30))       # 'n' edge-midpoint handle
g = W.logic.get_roi("rectangle_1").geometry
check("the top-edge handle resizes only the top edge",
      g["topLeftY"] == 30 and g["topLeftX"] == 410 and g["Bottom_right_Y"] == 130)
key(Qt.Key_Right)
key(Qt.Key_Down, Qt.ShiftModifier)
g = W.logic.get_roi("rectangle_1").geometry
check("arrow keys nudge 1 px, Shift+arrow 10 px",
      (g["topLeftX"], g["topLeftY"]) == (411, 40))

nest = lambda: W.logic.get_roi("nest").geometry["vertices"]  # noqa: E731
click(60, 320)                        # inside 'nest'
drag((90, 340), (100, 350))           # its vertex 2
check("dragging a polygon vertex moves just that vertex",
      nest()[2] == [100, 350] and nest()[0] == [30, 300])
QTest.mouseDClick(C, Qt.LeftButton, Qt.NoModifier, wpt(60, 300))
check("double-clicking an edge of the selected polygon inserts a vertex",
      len(nest()) == 5 and [60, 300] in nest())
click(60, 300, Qt.RightButton)
check("right-clicking a vertex removes it", len(nest()) == 4
      and [60, 300] not in nest())

# ---------------------------------------------------------- table editing ---
nrow = next(i for i, (_, n) in enumerate(W._overlay_idx_to_kind_name)
            if n == "nest")
W.shape_table.item(nrow, 0).setText("home")
app.processEvents()
check("renaming in the table renames the ROI",
      "home" in names() and "nest" not in names())
hrow = next(i for i, (_, n) in enumerate(W._overlay_idx_to_kind_name)
            if n == "home")
W.shape_table.item(hrow, 0).setText("polygon_1")
app.processEvents()
check("a clashing name is refused and reverted",
      "home" in names() and W.shape_table.item(hrow, 0).text() == "home")
W.shape_table.cellWidget(hrow, 2).setCurrentText("Blue")
check("the colour cell recolours the ROI",
      W.logic.get_roi("home").color_name == "Blue"
      and W.logic.get_roi("home").color_bgr == (255, 0, 0))

# ------------------------------------------------------------ undo / redo ---
W.shape_table.selectRow(hrow)
C.setFocus()
key(Qt.Key_Delete)
check("Delete removes the selected ROI", "home" not in names())
key(Qt.Key_Z, Qt.ControlModifier)
check("Ctrl+Z brings it back", "home" in names()
      and W.logic.get_roi("home").color_name == "Blue")
key(Qt.Key_Z, Qt.ControlModifier | Qt.ShiftModifier)
check("Ctrl+Shift+Z redoes the delete", "home" not in names())
key(Qt.Key_Z, Qt.ControlModifier)
check("unsaved changes are flagged", W.dirty_label.isVisibleTo(W))

# ------------------------------------------------------------------- zoom ---
anchor = QPointF(*C._mapping.frame_to_widget(200, 100))
before = C._mapping.widget_to_frame(anchor.x(), anchor.y())
ev = QWheelEvent(anchor, C.mapToGlobal(anchor), QPoint(0, 0), QPoint(0, 240),
                 Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
QApplication.sendEvent(C, ev)
after = C._mapping.widget_to_frame(anchor.x(), anchor.y())
check("wheel zooms in about the cursor (frame point under it stays put)",
      C._zoom > 1.4 and before is not None and after is not None
      and abs(after[0] - before[0]) <= 1 and abs(after[1] - before[1]) <= 1)
key(Qt.Key_F)
check("F fits the frame again", C._zoom == 1.0)

# ------------------------------------------------------------ persistence ---
W._on_save_clicked()
check("Save clears the unsaved flag", not W._dirty
      and not W.dirty_label.isVisibleTo(W))
back = ROILogic(config_path=cfg, video_path=str(vd / "vidA.mp4"))
check("saved H5 reloads with the 24-vertex polygon intact",
      len(back.defs[POLYGON]["polygon_1"].geometry["vertices"]) == N)
check("saved H5 has the renamed, recoloured polygon",
      back.get_roi("home") is not None
      and back.get_roi("home").color_name == "Blue")

# ------------------------------------------------- apply-to-all guarded ---
W._load_video(1)
key(Qt.Key_R)
drag((10, 10), (60, 60))
W._on_save_clicked()
W._load_video(0)
asked.clear()
answer["q"] = QMessageBox.No
W._on_apply_all_clicked()
vb = ROILogic(config_path=cfg, video_path=str(vd / "vidB.mp4"))
check("Apply to all asks first and says it replaces existing ROIs",
      asked and "replaces the existing ROIs of 1 video" in asked[-1])
check("declining leaves the other videos untouched",
      vb.all_roi_names == ["rectangle_1"]
      and vb.get_roi("rectangle_1").geometry["topLeftX"] == 10)

import ast  # noqa: E402

tree = ast.parse((REPO / "mufasa/ui_qt/dialogs/roi_canvas.py").read_text())
check("no polygon simplification call is left in the canvas",
      not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
              and n.func.attr == "simplify" for n in ast.walk(tree)))

n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122ij_roi_editor: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
