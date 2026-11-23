"""Smoke test for patch 122lt — recording every visible panel into the video.

User: "I would like each of the visible panels rendered in the video, such as if I have 3d,
video, and jpdf, then I want all of them."

* the panels shown, drawn as they sit on the screen, cropped to them; a hidden one left out; the
  video at its own resolution (its frame's pixels, the other panels drawn again at that scale);
* the feature viewer's Record, *All panels* on: the video, the 3-D view beside it, the traces and
  histograms, in each frame of the recording, following the playback; off: the video alone;
* the 3-D viewer's own Record: the 3-D view, the JPDF, the video panel, the feature traces;
* resizing the window while recording keeps the video's size.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QWidget  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


app = QApplication.instance() or QApplication([])


def settle(n: int = 6) -> None:
    for _ in range(n):
        app.processEvents()


def wait_saved(btn) -> None:  # noqa: ANN001
    t0 = time.time()
    while not btn.isEnabled() and time.time() - t0 < 90:
        app.processEvents()
        time.sleep(0.05)


from mufasa.project_layout import project_paths_from_config  # noqa: E402
from mufasa.ui_qt.feature_viewer import open_feature_viewer  # noqa: E402
from mufasa.ui_qt.panel_capture import compose, video_scale  # noqa: E402
from mufasa.ui_qt.pose3d_viewer import open_pose3d_viewer  # noqa: E402
from mufasa.utils.config_creator import ProjectConfigCreator  # noqa: E402

# ---- composing panels ------------------------------------------------------------------------
host = QWidget()
row = QHBoxLayout(host)
row.setContentsMargins(10, 10, 10, 10)
row.setSpacing(0)
panes = []
for colour in ("#ff0000", "#00ff00", "#0000ff"):
    lab = QLabel(host)
    lab.setFixedSize(100, 80)
    lab.setStyleSheet(f"background: {colour};")
    row.addWidget(lab)
    panes.append(lab)
host.show()
settle()
img = compose(host, panes)
mid = [tuple(int(c) for c in img[40, 50 + 100 * k]) for k in range(3)]
panes[1].hide()
settle()
img2 = compose(host, [panes[0], panes[1]])
frame = np.zeros((160, 200, 3), np.uint8)
frame[:, :, 2] = np.arange(200, dtype=np.uint8)[None, :]   # a ramp: its own pixels must come through
panes[1].show()
settle()
img3 = compose(host, panes, video=panes[0], frame=frame)
s = video_scale(panes[0], frame)
check(f"the panels as on the screen, cropped to them: {img.shape[1]}×{img.shape[0]}, red | green | blue "
      f"(BGR {mid}); one hidden: left out ({img2.shape[1]}×{img2.shape[0]}); with a 200×160 video in the "
      f"first, drawn at its scale ({s:g}): {img3.shape[1]}×{img3.shape[0]}, the video's own pixels",
      img.shape[:2] == (80, 300) and mid == [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
      and img2.shape[:2] == (80, 100) and s == 2.0 and img3.shape[:2] == (160, 600)
      and np.array_equal(img3[:, :200], frame) and tuple(int(c) for c in img3[80, 300]) == (0, 255, 0))
host.close()

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

# ---- the feature viewer: video, 3-D beside it, traces ----------------------------------------
fw = open_feature_viewer(None, cfg, str(video), [D_COL], 0)
fv = fw.viewer
v3 = fv.open_3d(str(pose))
v3.video3d_cb.setChecked(False)
fw.resize(1100, 800)
fw.show()
settle()
panels = fv.record_panels()
check("the 3-D view inside the feature viewer has no Record of its own (the feature viewer's records it)",
      not v3.record_btn.isVisible())
check("Record, All panels (on by default): the video, the 3-D view and its JPDF, the traces and histograms",
      fv.all_panels_cb.isChecked() and fv.scrubber._display in panels and v3.volume in panels
      and fv.plot in panels and fv.hist in panels)
fv.record_btn.setChecked(True)
for f in range(1, 21):
    fv.scrubber.seek(f)
rec = fv.recorder
fv.record_btn.setChecked(False)
wait_saved(fv.record_btn)
pngs = sorted(rec.frames_dir.glob("[0-9]*.png"))
a, b = cv2.imread(str(pngs[2])), cv2.imread(str(pngs[18]))
disp = fv.scrubber._display
sc = rec.scale
dx = round((disp.mapTo(fv, disp.rect().center()).x() - min(p.mapTo(fv, p.rect().topLeft()).x()
                                                              for p in panels if p.isVisible())) * sc)
vol = v3.volume
vx = round((vol.mapTo(fv, vol.rect().topLeft()).x() - min(p.mapTo(fv, p.rect().topLeft()).x()
                                                           for p in panels if p.isVisible())) * sc)
vy = round((vol.mapTo(fv, vol.rect().topLeft()).y() - min(p.mapTo(fv, p.rect().topLeft()).y()
                                                           for p in panels if p.isVisible())) * sc)
vh, vw = round(vol.height() * sc), round(vol.width() * sc)
grey_a, grey_b = int(a[a.shape[0] // 8, dx, 1]), int(b[b.shape[0] // 8, dx, 1])
moved = float(np.abs(a[vy:vy + vh, vx:vx + vw].astype(int) - b[vy:vy + vh, vx:vx + vw].astype(int)).mean())
cap = cv2.VideoCapture(str(rec.video_path))
n_out, out_w = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cap.release()
check(f"the recording: {len(pngs)} frames of {a.shape[1]}×{a.shape[0]} (scale {sc:g}: the video at least at its own "
      f"size), the video in it (grey {grey_a} → {grey_b} as it plays), the 3-D view beside it changing "
      f"(mean change {moved:.1f}); the video: {n_out} frames {out_w} wide",
      len(pngs) == 21 and n_out == 21 and a.shape[1] > VW + 200 and a.shape[0] > VH + 100
      and grey_b > grey_a + 30 and moved > 0.2 and out_w == a.shape[1]
      and abs(sc - max(1.0, VW / disp.width(), VH / disp.height())) < 1e-9)

fv.all_panels_cb.setChecked(False)
fv.record_btn.setChecked(True)
fv.scrubber.seek(30)
rec2 = fv.recorder
fv.record_btn.setChecked(False)
wait_saved(fv.record_btn)
one = cv2.imread(str(sorted(rec2.frames_dir.glob("[0-9]*.png"))[0]))
check(f"All panels off: the video alone ({one.shape[1]}×{one.shape[0]})", one.shape[:2] == (VH, VW))
fv.all_panels_cb.setChecked(True)

fv.record_btn.setChecked(True)
fv.scrubber.seek(31)
fw.resize(900, 700)
settle()
fv.scrubber.seek(32)
rec3 = fv.recorder
fv.record_btn.setChecked(False)
wait_saved(fv.record_btn)
sizes = {cv2.imread(str(p)).shape for p in rec3.frames_dir.glob("[0-9]*.png")}
check(f"the window resized while recording: every frame the first one's size {sizes}", len(sizes) == 1)
fw.close()
settle()

# ---- the 3-D viewer's own Record -------------------------------------------------------------
w3 = open_pose3d_viewer(None, str(video), pose)
pv = w3.viewer
pv.features_cb.setChecked(True)
pv.jpdf_cb.setChecked(True)
pv.set_jpdf_pair("3d:nose height", "3d:pelvis_pitch_deg")
pv.features.select(["3d:nose height"])
w3.resize(1000, 900)
w3.show()
settle()
pv.record_btn.setChecked(True)
for f in range(5, 15):
    pv.scrubber.seek(f)
r3 = pv.recorder
pv.record_btn.setChecked(False)
wait_saved(pv.record_btn)
png = cv2.imread(str(sorted(r3.frames_dir.glob("[0-9]*.png"))[0]))
shown = [p for p in pv.record_panels() if p.isVisible()]
check(f"the 3-D viewer's Record: {len(shown)} panels (3-D view, JPDF, video, traces, histograms) in "
      f"{png.shape[1]}×{png.shape[0]} frames; saved under derived/frames/annotated ('{pv.info.text()[:40]}…')",
      pv.record_btn.isVisible() and len(shown) == 5 and r3.video_path.is_file()
      and r3.video_path.parent == Path(cfg).parent / "derived" / "frames" / "annotated"
      and "Saved:" in pv.info.text() and png.shape[1] > VW)
w3.close()
settle()

doc = (REPO / "docs" / "pose3d.md").read_text()
check("docs/pose3d.md: recording all panels", "All panels" in doc and "Recording what is shown" in doc)

for name, ok in checks:
    print(("PASS  " if ok else "FAIL  ") + name)
n_ok = sum(ok for _, ok in checks)
print(f"smoke_122lt_record_all_panels: {n_ok}/{len(checks)} checks passed")
sys.exit(0 if n_ok == len(checks) else 1)
