"""Smoke test for patch 122jh — "Apply to all" copies only the selected ROI;
Manage classifiers sits above Features.

User requests: "in ROI I would like apply to all to only apply the selected
roi to the other videos", and move the Manage classifiers tab above the
Features tab.

ROI part runs for real, offscreen: a project with three videos (one at a
different resolution), ROIs drawn in the editor, then "Apply '<roi>' to
all…": the selected ROI reaches every other video (rescaled to its frame),
a same-named ROI there is replaced in place, the other videos' other ROIs
are untouched, the source video is unchanged, and nothing happens without
a selection or when the user declines. Also RoiStore.copy_roi directly.
"""
from __future__ import annotations

import ast
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
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


app = QApplication.instance() or QApplication([])
asked: list[str] = []
answer = {"q": QMessageBox.Yes}
QMessageBox.question = staticmethod(
    lambda *a, **k: (asked.append(str(a[2]) if len(a) > 2 else ""), answer["q"])[1])
QMessageBox.critical = staticmethod(lambda *a, **k: asked.append("critical: " + str(a[2])))

from mufasa.project_layout import project_paths_from_config  # noqa: E402
from mufasa.roi_tools.roi_logic import ROILogic  # noqa: E402
from mufasa.roi_tools.roi_store import RECTANGLE, Roi, RoiStore, load_rois  # noqa: E402
from mufasa.ui_qt.dialogs.roi_define_panel import ROIDefineWidget  # noqa: E402
from mufasa.utils.config_creator import ProjectConfigCreator  # noqa: E402

# ---- RoiStore.copy_roi ----------------------------------------------------------
st = RoiStore()
st.set_video("a", [Roi("arena", RECTANGLE, {"x": 0, "y": 0, "w": 100, "h": 50}),
                   Roi("cup", RECTANGLE, {"x": 10, "y": 10, "w": 20, "h": 20})], (200, 100))
st.set_video("b", [Roi("cup", RECTANGLE, {"x": 1, "y": 1, "w": 2, "h": 2}),
                   Roi("nest", RECTANGLE, {"x": 5, "y": 5, "w": 5, "h": 5})], (400, 200))
done = st.copy_roi("a", "cup", ["a", "b", "c"], {"b": (400, 200), "c": (200, 100)})
check("copy_roi: targets other than the source", done == ["b", "c"])
check("…replaces a same-named ROI in place, keeps the others, rescales",
      [r.name for r in st.rois("b")] == ["cup", "nest"]
      and st.rois("b")[0].geometry == {"x": 20, "y": 20, "w": 40, "h": 40})
check("…adds it to a video without ROIs", [r.name for r in st.rois("c")] == ["cup"])
check("…leaves the source alone", [r.name for r in st.rois("a")] == ["arena", "cup"])
try:
    st.copy_roi("a", "nope", ["b"])
    ok = False
except ValueError:
    ok = True
check("…an unknown ROI name is an error", ok)

# ---- the editor ------------------------------------------------------------------
tmp = Path(tempfile.mkdtemp())
cfg = ProjectConfigCreator(
    project_path=str(tmp), project_name="r", target_list=[],
    pose_estimation_bp_cnt="user_defined", body_part_config_idx=0,
    animal_cnt=1, file_type="parquet", body_parts=["a", "b"]).config_path
vd = Path(project_paths_from_config(cfg)["video_dir"])
vd.mkdir(parents=True, exist_ok=True)
for v, (w, h) in {"A": (640, 360), "B": (640, 360), "C": (1280, 720)}.items():
    wr = cv2.VideoWriter(str(vd / f"{v}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    for _ in range(3):
        wr.write(np.zeros((h, w, 3), np.uint8))
    wr.release()
S = dict(color_name="Red", bgr=(0, 0, 255), thickness=2, ear_tag_size=8)
L = ROILogic(config_path=cfg, video_path=str(vd / "A.mp4"))
L.add_rectangle(name="arena", top_left=(0, 0), bottom_right=(600, 340), **S)
L.add_circle(name="cup", center=(100, 100), radius=30, **S)
L.save()
L = ROILogic(config_path=cfg, video_path=str(vd / "B.mp4"))
L.add_rectangle(name="nest", top_left=(400, 200), bottom_right=(500, 300), **S)
L.add_circle(name="cup", center=(50, 50), radius=10, **S)
L.save()

W = ROIDefineWidget(config_path=cfg)
W.resize(1400, 900)
W.show()
app.processEvents()
names = [Path(v).stem for v in W._videos]
W._load_video(names.index("A"))
app.processEvents()
W.shape_table.clearSelection()
app.processEvents()
check("button is disabled until a ROI is selected", not W.apply_all_btn.isEnabled())
asked.clear()
W._on_apply_all_clicked()
check("without a selection nothing is asked or copied, and the status says to select one",
      not asked and load_rois(project_paths_from_config(cfg)["roi_definitions_path"]).rois("C") == []
      and "Select the ROI" in W.save_status.text())
row = [k for k, (_kind, n) in enumerate(W._overlay_idx_to_kind_name) if n == "cup"][0]
W.shape_table.selectRow(row)
app.processEvents()
check("selecting a ROI names it on the button",
      W.apply_all_btn.isEnabled() and W.apply_all_btn.text() == "Apply 'cup' to all…")
answer["q"] = QMessageBox.No
W._on_apply_all_clicked()
roi_file = project_paths_from_config(cfg)["roi_definitions_path"]
check("the question names the ROI, says other ROIs are kept and which videos "
      "already have one of that name",
      asked and "Copy ROI <b>cup</b>" in asked[-1] and "Their other ROIs are kept" in asked[-1]
      and "1 video(s) already have a ROI named 'cup'" in asked[-1] and "B" in asked[-1])
check("declining copies nothing", load_rois(roi_file).rois("C") == [])
answer["q"] = QMessageBox.Yes
W._on_apply_all_clicked()
app.processEvents()
store = load_rois(roi_file)
check("A (source) unchanged", [r.name for r in store.rois("A")] == ["arena", "cup"])
b = {r.name: r for r in store.rois("B")}
check("B keeps its own 'nest'; its 'cup' is replaced by A's (same place in the list)",
      [r.name for r in store.rois("B")] == ["nest", "cup"]
      and b["cup"].geometry["center"] == [100, 100] and b["cup"].geometry["radius"] == 30
      and b["nest"].geometry["x"] == 400)
c = {r.name: r for r in store.rois("C")}
check("C (twice the resolution) gets only 'cup', rescaled",
      list(c) == ["cup"] and c["cup"].geometry["center"] == [200, 200]
      and c["cup"].geometry["radius"] == 60)
check("'arena' was not copied anywhere",
      "arena" not in [r.name for r in store.rois("B")] + [r.name for r in store.rois("C")])
check("the status says what happened",
      "Copied 'cup' to 2 videos (1 replaced)" in W.save_status.text())
check("the other copy button is relabelled to say it copies all ROIs",
      W.apply_selected_btn.text() == "Copy all ROIs to chosen videos…")

# ---- page order ----------------------------------------------------------------
src = (REPO / "mufasa/ui_qt/workbench_app.py").read_text()
calls = [n.func.id for n in ast.walk(ast.parse(src))
         if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
         and n.func.id.startswith("build_") and n.func.id.endswith("_page")]
check(f"Manage classifiers is built right before Features ({calls[5:8]})",
      calls.index("build_manage_classifiers_page") + 1 == calls.index("build_features_page")
      and calls.index("build_roi_page") < calls.index("build_manage_classifiers_page"))
check("annotation / train / validate / inference still follow Features",
      calls.index("build_features_page") < calls.index("build_annotation_page")
      < calls.index("build_train_classifier_page"))

n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122jh_roi_apply_one_and_page_order: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
