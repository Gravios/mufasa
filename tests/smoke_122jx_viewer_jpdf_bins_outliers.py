"""Smoke test for patch 122jx — bins and outliers of the viewer's JPDF.

User request: "I would also like control over the outlier and bin size in
the video jpdf".

The JPDF row gets "Bins" (per axis) and "Outliers" (none / percentile tails
/ MAD with its threshold): outlying background frames are left out and the
axes fitted to the rest; the current-frame marker is pinned to the edge when
outside. Changes redraw the panel at once.
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

from mufasa.project_layout import (  # noqa: E402
    generate_run_id,
    project_paths_from_config,
    write_skeleton,
)
from mufasa.utils.config_creator import ProjectConfigCreator  # noqa: E402

tmp = Path(tempfile.mkdtemp())
cfg = ProjectConfigCreator(
    project_path=str(tmp), project_name="po", target_list=["rear"],
    pose_estimation_bp_cnt="user_defined", body_part_config_idx=0,
    animal_cnt=1, file_type="parquet", body_parts=["nose", "tail"]).config_path
root = Path(cfg).parent
P = project_paths_from_config(cfg)
vd, fd = Path(P["video_dir"]), Path(P["derived_features_dir"])
fam = fd / "frame_by_frame_body_part_movements_mm"
for d in (vd, fam, root / "sources" / "pose"):
    d.mkdir(parents=True, exist_ok=True)
W, H, N = 320, 240, 40
w = cv2.VideoWriter(str(vd / "v1.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20, (W, H))
for i in range(N):
    w.write(np.full((H, W, 3), 40, np.uint8))
w.release()
F = "Animal_1 movement nose (mm)_FEATURE_SUBSET"
G = "Animal_1 movement tail (mm)_FEATURE_SUBSET"
pd.DataFrame({F: np.sin(np.arange(N) / 5), G: np.cos(np.arange(N) / 5)}
             ).to_parquet(fam / "v1.parquet")
rng = np.random.default_rng(0)
wild = rng.normal(0, 0.7, 2000)
wild[:10] = 400.0                                    # a few wild values
pd.DataFrame({F: wild, G: rng.normal(0, 0.7, 2000)}
             ).to_parquet(fam / "v9.parquet")       # another video: pooled only
# FDLC long format: nose moves right, tail fixed; tail low likelihood on odd frames
rows = []
for i in range(N):
    rows.append((i, "single", "nose", 40 + 5 * i, 120, 0.95))
    rows.append((i, "single", "tail", 250, 60, 0.2 if i % 2 else 0.9))
pd.DataFrame(rows, columns=["frame", "individual", "bodypart", "x", "y", "likelihood"]
             ).to_parquet(root / "sources" / "pose" / "v1.fdlc.parquet")
run = root / "derived" / "smoothed" / "kalman_v2" / generate_run_id()
run.mkdir(parents=True)
pd.DataFrame({"nose_x": 40 + 5.0 * np.arange(N), "nose_y": np.full(N, 180.0),
              "tail_x": np.full(N, 250.0), "tail_y": np.full(N, 200.0),
              "nose_var_x": np.ones(N), "nose_var_y": np.ones(N),
              "tail_var_x": np.ones(N), "tail_var_y": np.ones(N)}
             ).to_parquet(run / "v1_smoothed_v2.deadbeef.parquet")
write_skeleton(cfg, ["nose", "tail"], [("nose", "tail")])

from mufasa.ui_qt import jpdf_panel as jp  # noqa: E402
from mufasa.ui_qt.feature_viewer import FeatureViewerWidget  # noqa: E402

pool = jp.pooled_values(cfg, [F, G])
xv, yv = np.sin(np.arange(N) / 5), np.cos(np.arange(N) / 5)
p_none = jp.JpdfPanel(cfg, F, G, xv, yv, pooled=pool, outliers="none")
p_mad = jp.JpdfPanel(cfg, F, G, xv, yv, pooled=pool, outliers="mad", outlier_value=5)
p_pct = jp.JpdfPanel(cfg, F, G, xv, yv, pooled=pool, outliers="percentile",
                     outlier_value=2)
bn, bm, bp = p_none.background(300), p_mad.background(300), p_pct.background(300)
check("none: the wild values stretch the x axis", bn.xlim[1] > 300)
check("mad: wild values left out, axes fitted to the rest",
      bm.xlim[1] < 5 and p_mad.n_outliers >= 10)
check("percentile 2 %: ~4 % per feature left out, axes tighter",
      bp.xlim[1] < 5 and 0.03 * len(pool) <= p_pct.n_outliers <= 0.1 * len(pool))
pdef = jp.JpdfPanel(cfg, F, G, xv, yv, pooled=pool)
check("default is 0.5 % tails, 90 bins",
      (pdef.outliers, pdef.outlier_value, pdef.bins) == ("percentile", 0.5, 90))
far = xv.copy()
far[3] = 1000
pf = jp.JpdfPanel(cfg, F, G, far, yv, pooled=pool, outliers="mad")
mx, my = pf.marker_px(300, 3)
edge_x = pf.background(300).to_px(np.array([pf.background(300).xlim[1]]), np.array([0.0]))[0][0]
check("marker outside the axes pinned to the right edge", abs(mx - edge_x) <= 1)
img_a = p_mad.panel(300, 5)
p_mad.set_options(bins=20)
img_b = p_mad.panel(300, 5)
check("bins change redraws the density", p_mad.bins == 20 and not np.array_equal(img_a, img_b))
p_mad.set_options(outliers="none")
check("outlier change redraws (axes widen again)", p_mad.background(300).xlim[1] > 300)

V = FeatureViewerWidget(cfg, str(vd / "v1.mp4"), [F, G])
V.resize(1200, 760)
check("viewer: Bins and Outliers controls (defaults 90 / percentile 0.5 %)",
      V.bins_spin.value() == 90 and V.jout_cb.currentText() == "percentile"
      and V.jout_val.value() == 0.5)
V.scrubber.seek(6)
V.jpdf_cb.setChecked(True)
f0 = V.scrubber.last_display_frame.copy()
V.bins_spin.setValue(30)
check("Bins applied and redrawn", V.jpdf.bins == 30
      and not np.array_equal(V.scrubber.last_display_frame, f0))
V.jout_cb.setCurrentText("mad")
check("MAD: threshold 5 × MAD", V.jpdf.outliers == "mad" and V.jpdf.outlier_value == 5.0
      and V.jout_val.suffix().strip() == "× MAD")
V.jout_val.setValue(3)
check("threshold applied", V.jpdf.outlier_value == 3.0)
V.jout_cb.setCurrentText("none")
check("none: threshold disabled, nothing left out",
      not V.jout_val.isEnabled() and V.jpdf.outliers == "none"
      and (V.jpdf.background(200) and V.jpdf.n_outliers == 0))
V.set_jpdf_pair(G, F)
check("a new pair keeps the settings", V.jpdf.bins == 30 and V.jpdf.outliers == "none")
V.cleanup()

failed = [n for n, ok in checks if not ok]
for n, ok in checks:
    print(("PASS " if ok else "FAIL ") + n)
print(f"smoke_122jx_viewer_jpdf_bins_outliers: {len(checks) - len(failed)}/{len(checks)} "
      "checks passed")
sys.exit(1 if failed else 0)
