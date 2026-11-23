"""Smoke test for patch 122ls — the viewers' windows fit a screen: their tool rows wrap.

User: "the 3d view port is too wide and I cannot change it's width below a certain point which is
wider than my screen".

* the 3-D viewer's window, everything on (features, JPDF), narrows to well under a laptop screen
  (before: 1612 px, its tool row in one line);
* narrowed, its tool row wraps onto more rows, and the playback controls (also wrapping) stay
  below the video picture rather than over it;
* the feature viewer's window, with the 3-D view beside the video, narrows too (before: 1049 px);
* the wrapping layout itself: rows by width, hidden widgets take no room.
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
from PySide6.QtWidgets import QApplication, QPushButton, QWidget  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


app = QApplication.instance() or QApplication([])


def settle(n: int = 8) -> None:
    for _ in range(n):
        app.processEvents()


from mufasa.project_layout import project_paths_from_config  # noqa: E402
from mufasa.ui_qt.feature_viewer import open_feature_viewer  # noqa: E402
from mufasa.ui_qt.flow_layout import FlowLayout  # noqa: E402
from mufasa.ui_qt.pose3d_viewer import open_pose3d_viewer  # noqa: E402
from mufasa.utils.config_creator import ProjectConfigCreator  # noqa: E402

# ---- the wrapping layout ---------------------------------------------------------------------
host = QWidget()
fl = FlowLayout(host)
btns = [QPushButton(f"button {k}", host) for k in range(6)]
for b in btns:
    b.setFixedSize(100, 24)
    fl.addWidget(b)
one, two = fl.heightForWidth(700), fl.heightForWidth(330)
btns[5].hide()
check(f"the wrapping layout: one row at 700 px ({one} px high), more at 330 ({two}); its narrowest is "
      f"its widest widget ({fl.minimumSize().width()} px); a hidden widget takes no room",
      one == 24 and two > 2 * 24 and fl.minimumSize().width() == 100
      and fl.heightForWidth(530) == 24)

# ---- a project, a video, a 3-D file ----------------------------------------------------------
tmp = Path(tempfile.mkdtemp())
cfg = ProjectConfigCreator(
    project_path=str(tmp), project_name="nw", target_list=["rear"],
    pose_estimation_bp_cnt="user_defined", body_part_config_idx=0,
    animal_cnt=1, file_type="parquet", body_parts=["a", "b"]).config_path
P = project_paths_from_config(cfg)
vd, fd = Path(P["video_dir"]), Path(P["derived_features_dir"]) / "two_point_body_part_distances_mm"
for d in (vd, fd):
    d.mkdir(parents=True, exist_ok=True)
N = 120
video = vd / "v1.mp4"
w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
for k in range(N):
    w.write(np.full((240, 320, 3), k % 250, np.uint8))
w.release()
D_COL = "Animal_1 distance (mm) a b_FEATURE_SUBSET"
pd.DataFrame({D_COL: np.random.default_rng(0).normal(40, 3, N)}).to_parquet(fd / "v1.parquet")
fr = np.arange(N)
cols = {"frame": fr, "nose_x": np.linspace(-50, 50, N), "nose_y": 0.0, "nose_z": 40 + 10 * np.sin(fr / 9),
        "pelvis_pitch_deg": np.linspace(0, 20, N)}
meta = {"units": "mm", "box_x": [-200, 200], "box_y": [-150, 150], "box_z": [0, 300], "edges": [],
        "camera": {"height_mm": 400.0, "focal_px": 200.0, "principal_px": [160, 120]}}
t = pa.Table.from_pandas(pd.DataFrame(cols), preserve_index=False)
t = t.replace_schema_metadata({**t.schema.metadata, b"mufasa_pose3d": json.dumps(meta).encode()})
pose = vd / "v1_3d.parquet"
pq.write_table(t, pose)

# ---- the 3-D viewer --------------------------------------------------------------------------
win = open_pose3d_viewer(None, str(video), pose)
pv = win.viewer
pv.features_cb.setChecked(True)
pv.jpdf_cb.setChecked(True)
settle()
mw = win.minimumSizeHint().width()
check(f"the 3-D viewer's window, features and JPDF on, narrows to {mw} px (was 1612)", mw < 600)
win.resize(560, 760)
win.show()
settle()
rows = sorted({b.y() for b in pv.findChildren(QPushButton) if b.parent() is pv and b.isVisible()})
check(f"narrowed to 560 px: its tool row wraps ({len(rows)} rows of buttons) and fits ({pv.width()} px)",
      len(rows) >= 2 and pv.width() <= 560)
sc = pv.scrubber
disp = sc._display.geometry()
ctl = [b.geometry() for b in (sc._b_prev100, sc._b_next100) + tuple(
    b for b in sc.findChildren(QPushButton) if b.isVisible())]
low = max(g.bottom() for g in ctl)
check(f"the playback controls wrap below the picture (picture ends at {disp.bottom()}, controls from "
      f"{min(g.top() for g in ctl)} to {low}; the scrubber is {sc.height()} high)",
      min(g.top() for g in ctl) > disp.bottom() and low < sc.height()
      and len({g.top() for g in ctl}) >= 2)
win.close()

# ---- the feature viewer, the 3-D view beside the video ---------------------------------------
fw = open_feature_viewer(None, cfg, str(video), [D_COL], 0)
fw.viewer.open_3d(str(pose))
fw.viewer.jpdf_cb.setChecked(True)
settle()
mf = fw.minimumSizeHint().width()
check(f"the feature viewer's window with the 3-D view and the JPDF bar narrows to {mf} px (was 1049)", mf < 760)
fw.resize(720, 900)
fw.show()
settle()
check(f"narrowed to 720 px: it fits ({fw.width()} px)", fw.width() <= 720)
fw.close()
settle()

doc = (REPO / "docs" / "pose3d.md").read_text()
check("docs/pose3d.md: narrow windows", "wrap onto more" in doc and "Narrow windows" in doc)

for name, ok in checks:
    print(("PASS  " if ok else "FAIL  ") + name)
n_ok = sum(ok for _, ok in checks)
print(f"smoke_122ls_narrow_windows: {n_ok}/{len(checks)} checks passed")
sys.exit(0 if n_ok == len(checks) else 1)
