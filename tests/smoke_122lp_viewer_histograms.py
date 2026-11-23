"""Smoke test for patch 122lp — feature histograms, the 3-D view in the feature viewer's window,
features in the 3-D viewer, and the viewers in windows of their own.

User: "add a feature view histogram to the 3D viewer, or even consolidate the current feature
viewer with it so it has its own window? I have no need to dock it".

* the feature viewer opens in a window of its own (not a workbench dock);
* each feature has a histogram beside its trace, lane for lane: the whole video, the window, each
  behaviour (rearing separates on the distance), the current value; clicking a bin moves to the
  next frame in it;
* ``3-D…`` shows the 3-D view beside the video in the same window, on the same playback;
* the 3-D viewer opens in a window of its own; its *Features* lists the 3-D file's numbers and
  those of the gait file beside it, as traces with histograms, following the playback;
* closing a window cleans it up.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from PySide6.QtWidgets import QApplication, QDockWidget, QMainWindow  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


app = QApplication.instance() or QApplication([])

from mufasa.project_layout import project_paths_from_config  # noqa: E402
from mufasa.ui_qt import feature_panels as FPN  # noqa: E402
from mufasa.ui_qt.feature_viewer import FeatureViewerWidget, open_feature_viewer  # noqa: E402
from mufasa.ui_qt.pose3d_viewer import Pose3DViewerWidget, open_pose3d_viewer  # noqa: E402
from mufasa.utils.config_creator import ProjectConfigCreator  # noqa: E402

tmp = Path(tempfile.mkdtemp())
cfg = ProjectConfigCreator(
    project_path=str(tmp), project_name="hv", target_list=["rear"],
    pose_estimation_bp_cnt="user_defined", body_part_config_idx=0,
    animal_cnt=1, file_type="parquet", body_parts=["a", "b"]).config_path
P = project_paths_from_config(cfg)
vd, fd, ld = Path(P["video_dir"]), Path(P["derived_features_dir"]), Path(P["derived_labels_dir"])
for d in (vd, ld, fd / "two_point_body_part_distances_mm"):
    d.mkdir(parents=True, exist_ok=True)
N = 300
r = np.random.default_rng(2)
D_COL = "Animal_1 distance (mm) a b_FEATURE_SUBSET"
video = vd / "v1.mp4"
w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
for k in range(N):
    w.write(np.full((48, 64, 3), k % 250, np.uint8))
w.release()
rear = np.zeros(N, int)
for s in (20, 120, 220):
    rear[s:s + 30] = 1
dist = np.where(rear, r.normal(30, 2, N), r.normal(45, 2, N))
pd.DataFrame({D_COL: dist}).to_parquet(fd / "two_point_body_part_distances_mm" / "v1.parquet")
pd.DataFrame({"rear": rear}).astype("Int64").to_parquet(ld / "v1.parquet")

# ---- the feature viewer, in a window of its own ------------------------------------------------
win = open_feature_viewer(None, cfg, str(video), [D_COL], 0)
app.processEvents()
fv = win.viewer
check("the feature viewer opens in a window of its own (not a dock); its viewer inside",
      isinstance(win, QMainWindow) and not isinstance(win, QDockWidget) and win.isWindow()
      and isinstance(fv, FeatureViewerWidget) and win.isVisible())

H = fv.hist.histograms(D_COL)
edges = H["edges"]
mid = (edges[:-1] + edges[1:]) / 2
peak_all = mid[np.argmax(H["all"])]
peak_rear = mid[np.argmax(H["labels"]["rear"])]
check(f"a histogram beside the trace: the whole video peaks at {peak_all:.0f} mm, rearing at "
      f"{peak_rear:.0f} mm (dashed); {FPN.HIST_BINS} bins over the trace's range",
      len(H["all"]) == FPN.HIST_BINS and abs(peak_rear - 30) < 3 and abs(peak_all - 45) < 3
      and fv.hist.isVisible() and fv.hist._lanes()[0][1] == 6 + fv.plot.LABEL_LANE_H + 2)
fv.window_spin.setValue(0.5)                              # ±15 frames
fv._seek(135)
app.processEvents()
Hw = fv.hist.histograms(D_COL)
check(f"the ±0.5 s window around frame 135 (inside a rear): its histogram peaks at "
      f"{mid[np.argmax(Hw['window'])]:.0f} mm; the trace plot and histogram follow the frame",
      fv.plot._frame == 135 and abs(mid[np.argmax(Hw["window"])] - 30) < 4)
k45 = int(np.argmin(np.abs(mid - 45)))
f = fv.hist.next_frame_in_bin(D_COL, k45)
check(f"a click on the 45 mm bin: the next frame with a value in it ({f}, after 135)",
      f is not None and f > 135 and abs(dist[f] - mid[k45]) <= (edges[1] - edges[0]))
fv.hist_cb.setChecked(False)
check("Histograms: off hides them", not fv.hist.isVisible())
fv.hist_cb.setChecked(True)

# ---- a 3-D file and its gait file -------------------------------------------------------------
names = ["nose", "ear_l", "ear_r"]
fr = np.arange(N)
xyz = np.zeros((N, 3, 3))
xyz[:, 0] = np.c_[np.linspace(-50, 50, N), np.zeros(N), 40 + 20 * np.sin(fr / 20)]
xyz[:, 1] = xyz[:, 0] + [-20, 10, 0]
xyz[:, 2] = xyz[:, 0] + [-20, -10, 0]
cols = {"frame": fr, "pelvis_pitch_deg": np.linspace(0, 30, N)}
for j, n in enumerate(names):
    cols[n + "_x"], cols[n + "_y"], cols[n + "_z"] = xyz[:, j].T
meta = {"units": "mm", "box_x": [-200, 200], "box_y": [-150, 150], "box_z": [0, 300], "edges": [],
        "camera": {"height_mm": 400.0, "focal_px": 200.0, "principal_px": [32, 24]}}
t = pa.Table.from_pandas(pd.DataFrame(cols), preserve_index=False)
t = t.replace_schema_metadata({**t.schema.metadata, b"mufasa_pose3d": json.dumps(meta).encode()})
pose = vd / "v1_3d.parquet"
pq.write_table(t, pose)
pd.DataFrame({"frame": fr[10:], "speed_mm_s": np.linspace(0, 200, N - 10),
              "LH_contact": (fr[10:] % 10 < 6).astype(int)}).to_parquet(vd / "v1_gait.parquet")

tab = FPN.session_features(pose)
check(f"the numbers that go with a 3-D file, by frame: {sorted(tab.columns)[:6]}…",
      {"3d:pelvis_pitch_deg", "3d:nose height", "gait:speed_mm_s", "gait:LH_contact"} <= set(tab.columns)
      and len(tab) == N and np.isnan(tab["gait:speed_mm_s"][5]) and abs(tab["gait:speed_mm_s"][10]) < 1e-9
      and abs(tab["3d:nose height"][0] - 40) < 1e-9)

# ---- 3-D beside the video, in the feature viewer's window --------------------------------------
v3 = fv.open_3d(str(pose))
app.processEvents()
fv._seek(77)
app.processEvents()
check("3-D… : the 3-D view beside the video in the same window, on the same playback (frame 77)",
      isinstance(v3, Pose3DViewerWidget) and v3.scrubber is fv.scrubber and v3.parent() is fv.top_split
      and v3.volume.frame == 77 and fv.view3d_btn.isChecked() and not v3.reproj_cb.isEnabled())
fv.view3d_btn.setChecked(False)
check("3-D… again: hidden (kept for the next time)", not v3.isVisible() and fv.view3d is v3)

# ---- the 3-D viewer, in a window of its own, with features -------------------------------------
w3 = open_pose3d_viewer(None, str(video), pose)
app.processEvents()
pv = w3.viewer
check("the 3-D viewer opens in a window of its own", isinstance(w3, QMainWindow) and w3.isWindow()
      and isinstance(pv, Pose3DViewerWidget) and pv.scrubber is not fv.scrubber)
pv.features_cb.setChecked(True)
app.processEvents()
panel = pv.features
pre = panel.selected()
panel.select(["3d:nose height", "gait:speed_mm_s"])
pv.scrubber.seek(150)
app.processEvents()
check(f"Features: {panel.list.count()} listed (preselected {pre}); two ticked → two lanes with histograms, "
      "following the playback (frame 150)",
      panel.isVisible() and list(panel.traces._data) == ["3d:nose height", "gait:speed_mm_s"]
      and panel.traces._frame == 150 and len(panel.hist._lanes()) == 2 and "gait:speed_mm_s" in pre)
panel.filter.setText("height")
hidden = sum(panel.list.item(i).isHidden() for i in range(panel.list.count()))
check(f"the list filters ('height': {panel.list.count() - hidden} shown)",
      0 < panel.list.count() - hidden < panel.list.count())
seen = []
panel.frame_clicked.connect(seen.append)
panel.hist.frame_clicked.emit(42)
app.processEvents()
check("a click in the panel moves the playback", seen == [42] and pv.scrubber.current_frame == 42)

# ---- the JPDF beside the 3-D view -----------------------------------------------------------
pv.jpdf_cb.setChecked(True)
app.processEvents()
pv.set_jpdf_pair("3d:nose height", "gait:speed_mm_s")
pv.jpdf_trail.setValue(20)
pv.scrubber.seek(200)
app.processEvents()
jp = pv.jpdf.jpdf
img = pv.jpdf.image()
mk = jp.marker_px(img.shape[0], 200) if jp is not None else None
red = img is not None and mk is not None and img[mk[1], mk[0], 2] > 200 and img[mk[1], mk[0], 1] < 80
check(f"JPDF beside the 3-D view: {jp.x if jp else None} × {jp.y if jp else None}, the current frame's point "
      f"(red at {mk}), a {jp.trail if jp else None}-frame trail, this video's frames as the density",
      pv.jpdf.isVisible() and pv.jpdf.parent() is pv.view_split and pv.volume.parent() is pv.view_split
      and jp is not None and (jp.x, jp.y) == ("3d:nose height", "gait:speed_mm_s") and jp.trail == 20
      and pv.jpdf.frame == 200 and red and jp.title == "this video" and pv.jpdf_bar.isVisible())
pv.jpdf_cb.setChecked(False)
check("JPDF off: hidden, the 3-D view takes the room", not pv.jpdf.isVisible() and not pv.jpdf_bar.isVisible())

n_open = len(FPN._OPEN)
w3.close()
win.close()
app.processEvents()
check("closing the windows cleans them up", len(FPN._OPEN) == n_open - 2)

doc = (REPO / "docs" / "pose3d.md").read_text()
check("docs/pose3d.md: histograms, 3-D beside the video, features and the JPDF in the 3-D viewer",
      "Histograms" in doc and "beside the video" in doc and "Features" in doc and "JPDF beside the 3-D view" in doc)

for name, ok in checks:
    print(("PASS  " if ok else "FAIL  ") + name)
n_ok = sum(ok for _, ok in checks)
print(f"smoke_122lp_viewer_histograms: {n_ok}/{len(checks)} checks passed")
sys.exit(0 if n_ok == len(checks) else 1)
