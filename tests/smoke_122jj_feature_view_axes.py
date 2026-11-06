"""Smoke test for patch 122jj — Feature Visualization: swap axes, linear / log10.

User request: "under feature view, I would like to be able to swap axis and
change between linear and log10".

New controls under the plot buttons: "Swap X ↔ Y" and an X and a Y scale
(linear / log10) for the displayed axes. They apply to the plot on screen at
once, from cached data (nothing re-read). Distributions: values on x,
density on y (swap → sideways histograms); log on the value axis uses
log-spaced bins and leaves out values ≤ 0, counted in the panel. Joint:
features exchange axes; log axes get log-spaced bins, frames ≤ 0 on a log
axis are counted in the title. Over time: time on x (swap → time running
down the y axis, panels side by side); double-click maps the right axis to
time.

Runs offscreen on a generated project with real video files.
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
sys.modules.setdefault("tkinter", _tk)
sys.modules.setdefault("tkinter.messagebox", _tk.messagebox)
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


app = QApplication.instance() or QApplication([])

from mufasa.project_layout import project_paths_from_config  # noqa: E402
from mufasa.ui_qt.forms.feature_visualization import FeatureVisualizationForm  # noqa: E402
from mufasa.utils.config_creator import ProjectConfigCreator  # noqa: E402

tmp = Path(tempfile.mkdtemp())
cfg = ProjectConfigCreator(
    project_path=str(tmp), project_name="ax", target_list=["rear"],
    pose_estimation_bp_cnt="user_defined", body_part_config_idx=0,
    animal_cnt=1, file_type="parquet", body_parts=["a", "b"]).config_path
P = project_paths_from_config(cfg)
vd, fd, ld = Path(P["video_dir"]), Path(P["derived_features_dir"]), Path(P["derived_labels_dir"])
fam = fd / "frame_by_frame_body_part_movements_mm"
for d in (vd, ld, fam):
    d.mkdir(parents=True, exist_ok=True)
N = 3000
r = np.random.default_rng(1)
w = cv2.VideoWriter(str(vd / "v1.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
for _ in range(10):
    w.write(np.zeros((48, 64, 3), np.uint8))
w.release()
A = "Animal_1 movement a (mm)_FEATURE_SUBSET"
B = "Animal_1 movement b (mm)_FEATURE_SUBSET"
a = r.lognormal(0, 1, N)
a[:50] = 0.0                                           # 50 zeros
b = a * r.lognormal(0, 0.3, N)
b[100:110] = -1.0                                       # 10 negatives
pd.DataFrame({A: a, B: b}).to_parquet(fam / "v1.parquet")
rear = np.zeros(N, int)
rear[500:800] = 1
pd.DataFrame({"rear": rear}).astype("Int64").to_parquet(ld / "v1.parquet")

F = FeatureVisualizationForm(config_path=cfg)
F.resize(1000, 900)
F.show()
app.processEvents()
F.set_selected([A, B])
loads = []
orig = F._load_uncached
F._load_uncached = lambda *a_, **k_: (loads.append(1), orig(*a_, **k_))[1]
check("controls exist with defaults: no swap, linear / linear",
      not F.swap_btn.isChecked() and F.xscale_cb.currentText() == "linear"
      and F.yscale_cb.currentText() == "linear")

# ---- joint ------------------------------------------------------------------------
F.plot_joint()
ax = F.figure.axes[0]
check("joint: A on x, B on y, linear", ax.get_xscale() == "linear"
      and ax.get_xlabel().startswith("Animal_1 movement a") and ax.get_ylabel().startswith("Animal_1 movement b"))
F.swap_btn.setChecked(True)
ax = F.figure.axes[0]
check("swap redraws at once with the features exchanged",
      ax.get_xlabel().startswith("Animal_1 movement b") and ax.get_ylabel().startswith("Animal_1 movement a"))
F.xscale_cb.setCurrentText("log10")
F.yscale_cb.setCurrentText("log10")
ax = F.figure.axes[0]
check("log10 on both displayed axes", ax.get_xscale() == "log" and ax.get_yscale() == "log")
check("frames with a value ≤ 0 on a log axis are left out and counted (50 zeros + 10 negatives)",
      "60 with a value ≤ 0 on a log axis not shown" in ax.get_title())
lo, hi = ax.get_xlim()
check("log axis limits are positive", lo > 0 and hi > lo)
check("axis changes re-draw from cached data (one load)", len(loads) == 1)

# ---- distributions ----------------------------------------------------------------
F.swap_btn.setChecked(False)
F.xscale_cb.setCurrentText("linear")
F.yscale_cb.setCurrentText("linear")
F.colour_cb.setCurrentText("rear")
aucs = F.plot_distributions()
n_loads = len(loads)
ax0 = F.figure.axes[0]
bars = [p for p in ax0.patches]
check("distributions: vertical bars, values on x", bars and bars[0].get_width() > 0
      and ax0.get_xscale() == "linear")
F.xscale_cb.setCurrentText("log10")
ax0 = F.figure.axes[0]
w = np.array([p.get_width() for p in ax0.patches[:40]])
x0 = np.array([p.get_x() for p in ax0.patches[:40]])
check("log10 value axis: log scale with log-spaced bins (widths grow with x)",
      ax0.get_xscale() == "log" and np.corrcoef(x0, w)[0, 1] > 0.9)
check("…and the panel counts the values ≤ 0 it leaves out",
      any("50 ≤ 0 not shown" in t.get_text() for t in ax0.texts))
F.yscale_cb.setCurrentText("log10")
check("log10 density axis", F.figure.axes[0].get_yscale() == "log")
F.swap_btn.setChecked(True)
ax0 = F.figure.axes[0]
check("swap: sideways histograms — values on y (log), density on x (log)",
      ax0.get_yscale() == "log" and ax0.get_xscale() == "log"
      and ax0.patches[0].get_height() < ax0.patches[0].get_width() * 1e6)
aucs2 = F.plot_distributions()
check("AUC does not depend on the display", aucs2 == aucs)

# ---- over time ----------------------------------------------------------------------
F.swap_btn.setChecked(False)
F.xscale_cb.setCurrentText("linear")
F.yscale_cb.setCurrentText("linear")
F.plot_over_time()
axs = F.figure.axes
check("over time: panels stacked, time on x",
      len(axs) == 2 and axs[-1].get_xlabel() == "time (min)"
      and axs[0].get_position().y0 > axs[1].get_position().y0)
F.yscale_cb.setCurrentText("log10")
check("log10 on the value axis", all(x.get_yscale() == "log" for x in F.figure.axes))
F.swap_btn.setChecked(True)
axs = F.figure.axes
check("swap: time on y running downwards, panels side by side; the value axis keeps "
      "its log scale (scale settings swap with the axes)",
      axs[0].get_ylabel() == "time (min)" and axs[0].yaxis_inverted()
      and axs[0].get_position().x0 < axs[1].get_position().x0
      and axs[0].get_xscale() == "log" and axs[0].get_yscale() == "linear")
opened = []
F.open_viewer = lambda start_frame=0: opened.append(start_frame)


class _Ev:
    dblclick = True
    inaxes = axs[0]
    xdata = 5.0          # a value, not time
    ydata = 0.5          # minutes


F._on_canvas_click(_Ev())
check("double-click with swapped axes reads time from y (0.5 min → frame 900)",
      opened == [900])
check("distribution / over-time redraws never re-read the features",
      len(loads) == n_loads)
F.swap_btn.setChecked(False)
check("swapping back returns the log scale to the value axis",
      F.yscale_cb.currentText() == "log10" and F.xscale_cb.currentText() == "linear"
      and F.figure.axes[0].get_yscale() == "log")

n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122jj_feature_view_axes: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
