"""Smoke test for patch 122jv — JPDF beside the video in the feature viewer.

User request: "Also add an option to show the video side by side with a
selected jpdf where the current position in the jpdf space is marked by a
visible, solid, red marker. When the jpdf is active the frame collection
should include the video and the jpdf, so I can make a video of the rat's
behavior alongside the marker trajectory of its current position in the
jpdf space."

"JPDF beside the video" + x / y feature combos + Trail: the 2-D log density
of the pair over all videos is composed to the right of the displayed frame,
with the frame's position as a solid red marker and a fading red trail;
names / log10 scales from feature_names.toml. The composed frame is what
"● Record" saves.
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
pd.DataFrame({F: rng.normal(0, 0.7, 2000), G: rng.normal(0, 0.7, 2000)}
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
from mufasa.utils import feature_names as fn  # noqa: E402

pool = jp.pooled_values(cfg, [F, G])
check("pooled over all videos with both features", len(pool) == N + 2000)
fn.save_names(cfg, {F: "nose speed"})
panel = jp.JpdfPanel(cfg, F, G, np.sin(np.arange(N) / 5), np.cos(np.arange(N) / 5),
                     trail=10, pooled=pool)
img = panel.panel(300, 12)
check("panel square at the frame height", img.shape == (300, 300, 3))
mx, my = panel.marker_px(300, 12)
px = img[my, mx]
check("current frame: solid red marker at its position",
      px[2] > 200 and px[1] < 60 and px[0] < 60)
mx0, my0 = panel.marker_px(300, 0)
check("marker moves with the frame", (mx0, my0) != (mx, my))
seg = img.astype(int)
red = (seg[:, :, 2] > 180) & (seg[:, :, 1] < 200) & (seg[:, :, 0] < 200)
red_trail = red.sum()
panel.trail = 0
red_none = ((lambda a: (a[:, :, 2] > 180) & (a[:, :, 1] < 200) & (a[:, :, 0] < 200))(
    panel.panel(300, 12).astype(int))).sum()
check("trail drawn behind the marker (and can be 0)", red_trail > red_none + 20)
panel.trail = 10
check("axes use the figure names", panel.name_x == "nose speed")
fn.save_names(cfg, {}, {G: "log10"})
p2 = jp.JpdfPanel(cfg, F, G, np.sin(np.arange(N) / 5), np.abs(np.cos(np.arange(N) / 5)) + .1,
                  pooled=pool)
check("log10 scale applied and labelled", p2.name_y.startswith("log10 ")
      and np.nanmax(p2.vy) <= np.log10(1.1) + 1e-9)
fn.save_names(cfg, {}, {G: "linear"})
nan_vals = np.sin(np.arange(N) / 5)
nan_vals[12] = -1
p3 = jp.JpdfPanel(cfg, F, G, nan_vals, np.cos(np.arange(N) / 5), pooled=pool)
check("-1 / missing value: no marker", p3.marker_px(300, 12) is None)
comp = panel(np.zeros((240, 320, 3), np.uint8), 5)
check("composer puts the JPDF to the right of the frame",
      comp.shape == (240, 320 + 240, 3) and (comp[:, :320] == 0).all())

V = FeatureViewerWidget(cfg, str(vd / "v1.mp4"), [F, G])
V.resize(1100, 760)
check("viewer: JPDF option with x / y combos of its features",
      V.jpdf_cb.isEnabled() and V.jpdf_x.count() == 2 and V.jpdf_x.currentData() == F
      and V.jpdf_y.currentData() == G and V.jpdf_x.itemText(0) == "nose speed")
V.scrubber.seek(8)
w0 = V.scrubber.last_display_frame.shape[1]
V.jpdf_cb.setChecked(True)
fr = V.scrubber.last_display_frame
check("JPDF shown beside the video", fr.shape[1] == w0 + fr.shape[0] and V.jpdf is not None)
mxy = V.jpdf.marker_px(fr.shape[0], 8)
pix = fr[mxy[1], w0 + mxy[0]]
check("red marker at frame 8's position", pix[2] > 200 and pix[1] < 60)
V.kalman_cb.setChecked(True)
check("pose overlay and JPDF together", V.scrubber.last_display_frame.shape == fr.shape
      and not np.array_equal(V.scrubber.last_display_frame, fr))
V.set_jpdf_pair(G, F)
check("changing the pair redraws", V.jpdf.x == G and V.jpdf.y == F)
V.trail_spin.setValue(5)
check("trail length applied", V.jpdf.trail == 5)
V.record_btn.setChecked(True)
for f in range(9, 20):
    V.scrubber.seek(f)
rec = V.recorder
V.record_btn.setChecked(False)
t0 = time.time()
while not V.record_btn.isEnabled() and time.time() - t0 < 60:
    app.processEvents()
    time.sleep(0.05)
png = cv2.imread(str(sorted(rec.frames_dir.glob("[0-9]*.png"))[0]))
check("recorded frames contain video + JPDF", png.shape[1] == w0 + png.shape[0])
cap = cv2.VideoCapture(str(rec.video_path))
vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
n_out = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
check("recorded video is side by side, all frames", n_out == 12 and vw >= w0 + png.shape[0] - 1)
V.jpdf_cb.setChecked(False)
check("JPDF off: video alone again", V.scrubber.last_display_frame.shape[1] == w0)
V.set_feature_columns([F])
check("fewer than two features: JPDF option disabled", not V.jpdf_cb.isEnabled())
V.cleanup()

failed = [n for n, ok in checks if not ok]
for n, ok in checks:
    print(("PASS " if ok else "FAIL ") + n)
print(f"smoke_122jv_viewer_jpdf_beside_video: {len(checks) - len(failed)}/{len(checks)} "
      "checks passed")
sys.exit(1 if failed else 0)
