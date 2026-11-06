"""Smoke test for patch 122je — Features → Feature Visualization.

Runs for real, offscreen, on a generated project (three videos, two feature
families written the way Compute Features writes them, labels for one
behaviour):

* feature_io: schema-only catalogue per family, column-projected loads;
* the form: video / family / search filtering, selection that survives
  filtering, distributions (this video and all videos) with per-feature
  AUC against the labelled behaviour, the joint plot, the whole-session
  plot, and the guards on the number of selected features;
* the video viewer: traces for the selected features, labels strip,
  click / drag seek, wheel zoom, cursor value readout;
* plotting helpers: AUC against sklearn, decimated session traces;
* the section is registered on the Features page.
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
import pandas as pd  # noqa: E402
from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QWheelEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


app = QApplication.instance() or QApplication([])

from mufasa.project_layout import project_paths_from_config  # noqa: E402
from mufasa.tools import feature_plots as FP  # noqa: E402
from mufasa.ui_qt.feature_viewer import FeatureViewerWidget  # noqa: E402
from mufasa.ui_qt.forms.feature_visualization import FeatureVisualizationForm  # noqa: E402
from mufasa.utils.config_creator import ProjectConfigCreator  # noqa: E402
from mufasa.utils.feature_io import feature_catalog, load_feature_columns  # noqa: E402

tmp = Path(tempfile.mkdtemp())
cfg = ProjectConfigCreator(
    project_path=str(tmp), project_name="fv", target_list=["rear"],
    pose_estimation_bp_cnt="user_defined", body_part_config_idx=0,
    animal_cnt=1, file_type="parquet", body_parts=["a", "b"]).config_path
P = project_paths_from_config(cfg)
vd, fd, ld = Path(P["video_dir"]), Path(P["derived_features_dir"]), Path(P["derived_labels_dir"])
for d in (vd, ld, fd / "two_point_body_part_distances_mm", fd / "head_velocity_in_body_frame_mm_s"):
    d.mkdir(parents=True, exist_ok=True)
N = 600
r = np.random.default_rng(4)
H_COL = "Animal_1 head velocity forward along body-head axis (mm/s)_FEATURE_SUBSET"
D_COL = "Animal_1 distance (mm) a b_FEATURE_SUBSET"
truth = {}
for v in ("v1", "v2", "v3"):
    w = cv2.VideoWriter(str(vd / f"{v}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    for k in range(N):
        w.write(np.full((48, 64, 3), k % 250, np.uint8))
    w.release()
    rear = np.zeros(N, int)
    for s in r.integers(0, N - 60, 6):
        rear[s:s + 30] = 1
    truth[v] = rear
    # distance separates rearing well; noise columns do not
    pd.DataFrame({D_COL: np.where(rear, r.normal(30, 3, N), r.normal(45, 3, N)),
                  "Animal_1 distance (mm) a nose_FEATURE_SUBSET": r.normal(20, 2, N)}
                 ).to_parquet(fd / "two_point_body_part_distances_mm" / f"{v}.parquet")
    pd.DataFrame({H_COL: r.normal(0, 30, N)}
                 ).to_parquet(fd / "head_velocity_in_body_frame_mm_s" / f"{v}.parquet")
    if v != "v3":
        pd.DataFrame({"rear": rear}).astype("Int64").to_parquet(ld / f"{v}.parquet")

# ---- feature_io helpers -------------------------------------------------------
cat = feature_catalog("v1", cfg)
check("catalogue lists columns per family from schemas",
      cat == {"head_velocity_in_body_frame_mm_s": [H_COL],
              "two_point_body_part_distances_mm":
                  [D_COL, "Animal_1 distance (mm) a nose_FEATURE_SUBSET"]})
df = load_feature_columns("v1", cfg, [H_COL, D_COL, "nope"])
check("projected load returns just the requested columns in order, across families",
      list(df.columns) == [H_COL, D_COL] and len(df) == N)

# ---- plotting helpers ---------------------------------------------------------
from sklearn.metrics import roc_auc_score  # noqa: E402

x = r.normal(size=500)
m = (x + r.normal(size=500)) > 0.5
check("AUC matches sklearn", abs(FP.auc(x, m) - roc_auc_score(m, x)) < 1e-9)
check("AUC is NaN without both groups", np.isnan(FP.auc(x, np.zeros(500, bool))))

# ---- the form -----------------------------------------------------------------
form = FeatureVisualizationForm(config_path=cfg)
form.resize(1100, 1000)
form.show()
app.processEvents()
check("videos with features are listed", [form.video_cb.itemText(i)
      for i in range(form.video_cb.count())] == ["v1", "v2", "v3"])
T = form.tree


def tops():
    return [T.topLevelItem(i) for i in range(T.topLevelItemCount())]


def kids(top):
    return [top.child(i) for i in range(top.childCount())]


def visible_features():
    return [k.data(0, Qt.UserRole) for t_ in tops() if not t_.isHidden()
            for k in kids(t_) if not k.isHidden()]


check("features are a tree: one branch per family, columns underneath",
      [t_.data(0, Qt.UserRole + 1)[1] for t_ in tops()]
      == ["head_velocity_in_body_frame_mm_s", "two_point_body_part_distances_mm"]
      and [len(kids(t_)) for t_ in tops()] == [1, 2])
check("family titles show the feature count",
      tops()[1].text(0) == "two point body part distances mm   (2)")
check("behaviours offered for colouring",
      [form.colour_cb.itemText(i) for i in range(form.colour_cb.count())] == ["(none)", "rear"])
check("features shown without the _FEATURE_SUBSET suffix",
      kids(tops()[0])[0].text(0) == H_COL.replace("_FEATURE_SUBSET", ""))
form.search.setText("distance nose")
form._apply_filter()
check("search matches all words and hides empty families",
      visible_features() == ["Animal_1 distance (mm) a nose_FEATURE_SUBSET"]
      and tops()[0].isHidden() and tops()[1].isExpanded()
      and "1/2 shown" in tops()[1].text(0))
form.search.setText("velocity")
form._apply_filter()
check("search also matches the family name",
      visible_features() == [H_COL])
form.search.setText("")
form._apply_filter()
dist, hv = tops()[1], tops()[0]
kids(dist)[0].setCheckState(0, Qt.Checked)
check("ticking one feature makes its family partially checked, with a count",
      dist.checkState(0) == Qt.PartiallyChecked and "1/2 selected" in dist.text(0))
hv.setCheckState(0, Qt.Checked)
check("ticking a family selects all its features",
      kids(hv)[0].checkState(0) == Qt.Checked and hv.checkState(0) == Qt.Checked)
check("selection keeps tick order across families", form.selected_features() == [D_COL, H_COL])
form.search.setText("nose")
form._apply_filter()
dist.setCheckState(0, Qt.Checked)
form.search.setText("")
form._apply_filter()
check("ticking a family while searching takes only its shown features",
      form.selected_features() == [D_COL, H_COL, "Animal_1 distance (mm) a nose_FEATURE_SUBSET"]
      and dist.checkState(0) == Qt.Checked)
dist.setCheckState(0, Qt.Unchecked)
check("unticking a family clears its features", form.selected_features() == [H_COL])
form.set_selected([D_COL, H_COL])
check("selection survives rebuilding the tree",
      tops()[1].checkState(0) == Qt.PartiallyChecked and tops()[0].checkState(0) == Qt.Checked)
form.tree.collapseAll()
form.tree.topLevelItem(1).setExpanded(True)
form.set_selected([D_COL, H_COL])
check("expansion state survives a rebuild", form.tree.topLevelItem(1).isExpanded()
      and not form.tree.topLevelItem(0).isExpanded())
check("selection summary", "2 selected" in form.sel_label.text())

form.colour_cb.setCurrentText("rear")
aucs = form.plot_distributions()
check("distributions: one panel per feature", len(form.figure.axes) == 2)
check("…AUC per feature against the labelled behaviour (distance separates rearing, "
      f"head velocity does not: {aucs})",
      aucs[D_COL] < 0.05 and 0.35 < aucs[H_COL] < 0.65)
check("…summary names the best separating feature",
      "Best separating: Animal_1 distance (mm) a b (AUC 0.0" in form.plot_info.text()
      and f"{N:,} frames from 1 video" in form.plot_info.text())
form.scope_cb.setCurrentIndex(1)
form.plot_distributions()
lab_all = truth["v1"].sum() + truth["v2"].sum()
check("all-videos scope pools the labelled videos; the unlabelled one is left out of "
      "the split and the summary says so",
      f"{2 * N:,} frames from 2 video(s)" in form.plot_info.text()
      and "1 video(s) without 'rear' labels left out" in form.plot_info.text()
      and f"{lab_all:,} labelled 'rear'" in form.plot_info.text())
form.colour_cb.setCurrentIndex(0)
form.plot_distributions()
check("without a behaviour, all videos are pooled",
      f"{3 * N:,} frames from 3 video(s)" in form.plot_info.text())
form.colour_cb.setCurrentText("rear")
form.plot_joint()
ax = form.figure.axes[0]
check("joint plot: 2-D density with the behaviour frames overlaid",
      len(form.figure.axes) in (1, 2) and ax.collections and "rear (" in
      ax.get_legend().get_texts()[0].get_text())
form.scope_cb.setCurrentIndex(0)
form.plot_over_time()
check("over time: one panel per feature, rearing bouts shaded",
      len(form.figure.axes) == 2 and len(form.figure.axes[0].patches) >= 1
      and "Double-click" in form.plot_info.text())

msgs = []
orig = QMessageBox.information
QMessageBox.information = lambda *a, **k: msgs.append(a[2])
form.set_selected([D_COL])
form.plot_joint()
check("joint asks for exactly two features", msgs and "exactly 2" in msgs[-1])
form.set_selected([])
form.plot_distributions()
check("nothing selected → a message, not an empty plot", "at least 1" in msgs[-1]
      or "1–12" in msgs[-1])
QMessageBox.information = orig

# ---- the viewer -----------------------------------------------------------------
W = FeatureViewerWidget(cfg, str(vd / "v1.mp4"), [D_COL, H_COL], start_frame=100)
W.resize(1000, 700)
W.show()
app.processEvents()
pl = W.plot
check("viewer opens at the requested frame", W.scrubber.current_frame == 100
      and pl._frame == 100)
check("one trace lane per feature, plus the labels strip",
      list(pl._data) == [D_COL, H_COL] and list(pl._labels) == ["rear"])
xmid = pl.NAME_W + pl._plot_w() / 2
check("plot centre = current frame", pl.frame_at(xmid) == 100)
half = pl._half()
QTest.mouseClick(pl, Qt.LeftButton, Qt.NoModifier,
                 QPoint(int(pl.NAME_W + 0.75 * pl._plot_w()), 40))
app.processEvents()
check("click right of centre seeks forward",
      abs(W.scrubber.current_frame - (100 + round(0.5 * half))) <= 1)
QTest.mouseClick(pl, Qt.LeftButton, Qt.NoModifier, QPoint(10, 40))
check("click on the names column does nothing",
      abs(W.scrubber.current_frame - (100 + round(0.5 * half))) <= 1)
W.scrubber.seek(300)
app.processEvents()
QTest.mousePress(pl, Qt.LeftButton, Qt.NoModifier, QPoint(int(xmid), 40))
QTest.mouseMove(pl, QPoint(int(xmid) - 60, 40))
QTest.mouseRelease(pl, Qt.LeftButton, Qt.NoModifier, QPoint(int(xmid) - 60, 40))
check("dragging scrubs backwards",
      abs(W.scrubber.current_frame - (300 - 60 * 2 * half / pl._plot_w())) <= 1.5)
w0 = pl.window_seconds()
ev = QWheelEvent(QPointF(xmid, 40), pl.mapToGlobal(QPointF(xmid, 40)), QPoint(0, 0),
                 QPoint(0, 120), Qt.NoButton, Qt.NoModifier, Qt.ScrollPhase.NoScrollPhase,
                 False)
QApplication.sendEvent(pl, ev)
check("wheel narrows the window and updates the spin box",
      pl.window_seconds() < w0 and abs(W.window_spin.value() - pl.window_seconds()) < 0.05)
W.window_spin.setValue(20)
check("spin box sets the window", pl.window_seconds() == 20)
img = pl.grab()
check("traces render", not img.isNull() and img.width() > 100)
W.cleanup()

# ---- page registration -----------------------------------------------------------
src = (REPO / "mufasa/ui_qt/pages/features_page.py").read_text()
check("section registered on the Features page",
      'add_section("Feature Visualization"' in src and "FeatureVisualizationForm" in src)

n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122je_feature_visualization: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
