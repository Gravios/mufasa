"""Smoke test for patch 122lu — the camera animation's *Render video…* with every panel shown.

User: "I ran render videos, but still only see the 3d view".

* *All panels* (on by default): each rendered frame holds the 3-D view moving along the camera
  path, the video at that frame, the JPDF and the feature traces, laid out as on the screen, the
  3-D view at the chosen width; the viewer's view and frame are put back after;
* the 3-D view inside the feature viewer: the feature viewer's panels too (its traces);
* *All panels* off: the 3-D view alone at width × height, as before (checked in 122kp).
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
from PySide6.QtWidgets import QApplication  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


app = QApplication.instance() or QApplication([])


def settle(n: int = 6) -> None:
    for _ in range(n):
        app.processEvents()


from mufasa.pose3d.animation import CameraKey  # noqa: E402
from mufasa.project_layout import project_paths_from_config  # noqa: E402
from mufasa.ui_qt.feature_viewer import open_feature_viewer  # noqa: E402
from mufasa.ui_qt.pose3d_viewer import open_pose3d_viewer  # noqa: E402
from mufasa.utils.config_creator import ProjectConfigCreator  # noqa: E402

# ---- a project, a video, a 3-D file ----------------------------------------------------------
tmp = Path(tempfile.mkdtemp())
cfg = ProjectConfigCreator(
    project_path=str(tmp), project_name="ra", target_list=["rear"],
    pose_estimation_bp_cnt="user_defined", body_part_config_idx=0,
    animal_cnt=1, file_type="parquet", body_parts=["a", "b"]).config_path
P = project_paths_from_config(cfg)
vd, fd = Path(P["video_dir"]), Path(P["derived_features_dir"]) / "two_point_body_part_distances_mm"
for d in (vd, fd):
    d.mkdir(parents=True, exist_ok=True)
N, VW, VH = 60, 320, 240
video = vd / "v1.mp4"
w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (VW, VH))
for k in range(N):
    w.write(np.full((VH, VW, 3), 40 + 3 * k, np.uint8))
w.release()
D_COL = "Animal_1 distance (mm) a b_FEATURE_SUBSET"
pd.DataFrame({D_COL: np.random.default_rng(0).normal(40, 3, N)}).to_parquet(fd / "v1.parquet")
fr = np.arange(N)
cols = {"frame": fr, "nose_x": np.linspace(-150, 150, N), "nose_y": 0.0, "nose_z": 40 + 30 * np.sin(fr / 5),
        "pelvis_pitch_deg": np.linspace(0, 20, N)}
meta = {"units": "mm", "box_x": [-200, 200], "box_y": [-150, 150], "box_z": [0, 300], "edges": [],
        "camera": {"height_mm": 400.0, "focal_px": 200.0, "principal_px": [160, 120]}}
t = pa.Table.from_pandas(pd.DataFrame(cols), preserve_index=False)
t = t.replace_schema_metadata({**t.schema.metadata, b"mufasa_pose3d": json.dumps(meta).encode()})
pose = vd / "v1_3d.parquet"
pq.write_table(t, pose)


def frames_of(path):  # noqa: ANN001
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


# ---- the 3-D viewer's window -----------------------------------------------------------------
w3 = open_pose3d_viewer(None, str(video), pose)
pv = w3.viewer
pv.features_cb.setChecked(True)
pv.jpdf_cb.setChecked(True)
pv.anim_cb.setChecked(True)
settle()
pv.features.select(["3d:nose height"])
pv.set_jpdf_pair("3d:nose height", "3d:pelvis_pitch_deg")
w3.resize(1000, 1000)
w3.show()
settle()
pv.scrubber.seek(3)
settle()
A = pv.animation
v0 = pv.volume.view_state()
A.path.keys = [CameraKey(5, v0, 20), CameraKey(25, dict(v0, az=v0["az"] + 120), 20)]
A.width_spin.setValue(800)
out = A.render_to(tmp / "anim_all.mp4")
fs = frames_of(out)
vol = pv.volume
shown = [p for p in pv.record_panels() if p.isVisible()]
x0 = min(p.mapTo(pv, p.rect().topLeft()).x() for p in shown)
y0 = min(p.mapTo(pv, p.rect().topLeft()).y() for p in shown)
sc = max(0.5, min(4.0, 800 / vol.width()))
vx, vy = round((vol.mapTo(pv, vol.rect().topLeft()).x() - x0) * sc), round((vol.mapTo(pv, vol.rect().topLeft()).y() - y0) * sc)
vw, vh = round(vol.width() * sc), round(vol.height() * sc)
turn = float(np.abs(fs[0][vy:vy + vh, vx:vx + vw].astype(int) - fs[-1][vy:vy + vh, vx:vx + vw].astype(int)).mean())
d = pv.scrubber._display
dx, dy = round((d.mapTo(pv, d.rect().center()).x() - x0) * sc), round((d.mapTo(pv, d.rect().center()).y() - y0) * sc)
g0, g1 = int(fs[0][dy, dx, 1]), int(fs[-1][dy, dx, 1])
check(f"All panels (on by default): {len(fs)} frames ({A.path.n_output()} expected) of {fs[0].shape[1]}×{fs[0].shape[0]}: "
      f"{len(shown)} panels, the 3-D view {vw} wide (asked 800) turning along the path (change {turn:.1f}), the video "
      f"at each frame (grey {g0} → {g1}: frames 5 → 25)",
      A.panels_cb.isChecked() and len(fs) == A.path.n_output() and len(shown) == 5 and abs(vw - 800) <= 2
      and fs[0].shape[1] > vw + 100 and fs[0].shape[0] > vh + 100 and turn > 1.0
      and abs(g0 - (40 + 3 * 5)) < 8 and abs(g1 - (40 + 3 * 25)) < 8)
check("the viewer's view and frame put back after rendering",
      vol.view_state()["az"] == v0["az"] and pv.scrubber.current_frame == 3 and vol.frame == 3)
on_off = (A.height_spin.isEnabled(), A.below_cb.isEnabled())
A.panels_cb.setChecked(False)
off = (A.height_spin.isEnabled(), A.below_cb.isEnabled())
A.panels_cb.setChecked(True)
check(f"Height and Video below apply to the 3-D-only render: disabled with All panels {on_off}, enabled without {off}",
      on_off == (False, False) and off == (True, True))
w3.close()
settle()

# ---- the 3-D view inside the feature viewer ---------------------------------------------------
fw = open_feature_viewer(None, cfg, str(video), [D_COL], 0)
fv = fw.viewer
v3 = fv.open_3d(str(pose))
v3.anim_cb.setChecked(True)
fw.resize(1200, 1000)
fw.show()
settle()
B = v3.animation
B.path.keys = [CameraKey(2, v3.volume.view_state(), 6), CameraKey(8, v3.volume.view_state(), 6)]
B.width_spin.setValue(400)
out2 = B.render_to(tmp / "anim_fv.mp4")
fs2 = frames_of(out2)
P2 = [p for p in fv.record_panels() if p.isVisible()]
check(f"inside the feature viewer: its panels too ({len(P2)}: video, 3-D view, traces, histograms), "
      f"{len(fs2)} frames of {fs2[0].shape[1]}×{fs2[0].shape[0]}",
      fv.plot in P2 and v3.volume in P2 and len(fs2) == B.path.n_output()
      and fs2[0].shape[0] > round(v3.volume.height() * 400 / v3.volume.width()) + 50)
fw.close()
settle()

doc = (REPO / "docs" / "pose3d.md").read_text()
check("docs/pose3d.md: the animation renders all panels", "Render video…" in doc and "122lu" in doc)

for name, ok in checks:
    print(("PASS  " if ok else "FAIL  ") + name)
n_ok = sum(ok for _, ok in checks)
print(f"smoke_122lu_animation_all_panels: {n_ok}/{len(checks)} checks passed")
sys.exit(0 if n_ok == len(checks) else 1)
