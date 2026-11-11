"""Smoke test for patch 122ke — 3-D marker viewer with the video below.

User: "Generate a 3d viewer in mufasa for the markers, with the rat video as a
view port below the reconstruction volume."

* load_pose3d: <m>_x/_y/_z columns, ``frame`` column, likelihoods, and the
  ``mufasa_pose3d`` metadata (box, edges, camera);
* the straight-down camera's projection matrix reproduces u = cx + f x/(h-z),
  v = cy - f y/(h-z); its centre is the camera position;
* the viewer: volume above, FrameScrubberWidget below; seeking the video moves
  the 3-D frame; the Camera preset puts the eye at the recording camera;
  reprojection draws markers on the video where the camera sees them;
* find_pose3d (beside the video), mufasa-view3d argument handling, the
  entry point, and the feature viewer's "3-D…" button.
"""
from __future__ import annotations

import json
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
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


tmp = Path(tempfile.mkdtemp())
video = tmp / "rat.mp4"
w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 180))
for k in range(40):
    w.write(np.full((180, 320, 3), 60 + k, np.uint8))
w.release()
H, F, CX, CY = 400.0, 200.0, 160.0, 90.0
names = ["nose", "ear_l", "ear_r"]
T = 30
fr = np.arange(5, 5 + T)
xyz = np.zeros((T, 3, 3))
xyz[:, 0] = np.c_[np.linspace(-50, 50, T), np.zeros(T), np.linspace(20, 200, T)]
xyz[:, 1] = xyz[:, 0] + [-20, 10, 0]
xyz[:, 2] = xyz[:, 0] + [-20, -10, 0]
cols = {"frame": fr}
for j, n in enumerate(names):
    cols[n + "_x"], cols[n + "_y"], cols[n + "_z"] = xyz[:, j].T
    cols[n + "_likelihood"] = np.full(T, 0.9)
meta = {"units": "mm", "box_x": [-200, 200], "box_y": [-150, 150], "box_z": [0, 300],
        "edges": [["nose", "ear_l"], ["nose", "ear_r"], ["nose", "missing"]],
        "camera": {"height_mm": H, "focal_px": F, "principal_px": [CX, CY]}, "method": "test"}
t = pa.Table.from_pandas(pd.DataFrame(cols), preserve_index=False)
t = t.replace_schema_metadata({**t.schema.metadata, b"mufasa_pose3d": json.dumps(meta).encode()})
pose_path = tmp / "rat_head3d.parquet"
pq.write_table(t, pose_path)

from mufasa.ui_qt import pose3d_viewer as V  # noqa: E402

P3 = V.load_pose3d(pose_path)
check("load: markers, frames, likelihood, box, edges (unknown marker dropped)",
      P3.names == names and P3.xyz.shape == (T, 3, 3) and P3.frames[0] == 5
      and P3.likelihood is not None and P3.box == ((-200, 200), (-150, 150), (0, 300))
      and P3.edges == [("nose", "ear_l"), ("nose", "ear_r")])
check("row(): video frame -> row; frames outside the file -> None",
      P3.row(5) == 0 and P3.row(34) == T - 1 and P3.row(4) is None and P3.row(99) is None)
M = P3.projection_matrix()
X = np.array([30.0, -40.0, 120.0])
h = M @ np.r_[X, 1]
u, v = h[:2] / h[2]
check("camera matrix: u = cx + f x/(h-z), v = cy - f y/(h-z)",
      abs(u - (CX + F * 30 / (H - 120))) < 1e-9 and abs(v - (CY + F * 40 / (H - 120))) < 1e-9)
check("camera position from the matrix", np.allclose(P3.camera_position(), [0, 0, H]))
df = pd.DataFrame({"a_x": [1.0], "a_y": [2.0], "a_z": [3.0]})
df.to_csv(tmp / "plain.csv", index=False)
P2 = V.load_pose3d(tmp / "plain.csv")
check("plain CSV: rows are frames, extent from the data, no camera",
      P2.frames.tolist() == [0] and P2.box is None and P2.projection_matrix() is None)

from PySide6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])
W = V.Pose3DViewerWidget(str(video), pose_path)
W.resize(700, 700)
W.show()
app.processEvents()
check("layout: volume above, video scrubber below (one splitter)",
      W.volume.parent() is W.scrubber.parent() and
      W.volume.parent().indexOf(W.volume) == 0 and W.volume.parent().indexOf(W.scrubber) == 1)
check("opens at the first 3-D frame", W.scrubber.current_frame == 5 and W.volume.frame == 5)
W.scrubber.seek(20)
app.processEvents()
check("seeking the video moves the 3-D view", W.volume.frame == 20)
vol = W.volume
(pc,), _ = vol.project(vol.target[None])
check("the orbit target is drawn at the volume's centre",
      abs(pc[0] - vol.width() / 2) < 1e-6 and abs(pc[1] - vol.height() / 2) < 1e-6)
vol.reset_view("Camera")
eye = vol._basis()[0]
check("Camera preset: the eye is at the recording camera, looking down",
      np.allclose(eye, [0, 0, H], atol=1.0) and vol.el > np.radians(89))
img = W.grab().toImage()
check("renders", not img.isNull() and img.width() > 0)
W.reproj_cb.setChecked(True)
over = V.reprojection_overlay(P3, vol.colors)
frame = np.zeros((180, 320, 3), np.uint8)
out = over(frame, 20)
i = P3.row(20)
hh = M @ np.r_[P3.xyz[i, 0], 1]
uu, vv = (hh[:2] / hh[2]).round().astype(int)
check("reprojection draws the nose where the camera sees it",
      out[vv, uu].any() and not frame.any() and over(frame, 2) is frame)
for name in (*V.VIEWS, "Camera"):
    vol.reset_view(name)
    vol.grab()
check("every preset view draws", True)
W.cleanup()

check("find_pose3d: <stem>*3d*.parquet beside the video",
      V.find_pose3d(str(video)) == pose_path and V.find_pose3d(str(tmp / "other.mp4")) is None)
from mufasa.tools import view3d  # noqa: E402

a = view3d.build_parser().parse_args([str(video)])
check("mufasa-view3d VIDEO finds the 3-D file", view3d.resolve(a)[1] == str(pose_path))
a = view3d.build_parser().parse_args([str(pose_path)])
check("mufasa-view3d POSE3D alone: volume only", view3d.resolve(a)[:2] == (None, str(pose_path)))
check("mufasa-view3d without a file: error code 2",
      view3d.main([str(tmp / "other.mp4")]) == 2)
toml = (REPO / "pyproject.toml").read_text()
check("entry point mufasa-view3d", 'mufasa-view3d    = "mufasa.tools.view3d:main"' in toml)
fv = (REPO / "mufasa/ui_qt/feature_viewer.py").read_text()
check("feature viewer: 3-D… button opens the viewer for its video",
      'QPushButton("3-D…", self)' in fv and "open_pose3d_viewer(self, self.video_path, path)" in fv)

failed = [n for n, ok in checks if not ok]
for n, ok in checks:
    print(("PASS " if ok else "FAIL ") + n)
print(f"smoke_122ke_pose3d_viewer: {len(checks) - len(failed)}/{len(checks)} checks passed")
sys.exit(1 if failed else 0)
