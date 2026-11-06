"""Smoke test for patch 122ju — pose overlays and recording in the feature
viewer.

User request: "In the feature viewer, in the video I would like the option
to add overlays of the fdlc and the kalman filtered markers. In addition, I
would like to be able to record, extract frames while playing until the
record button is pressed again and then convert the frames into a video."

* the viewer finds the video's FDLC pose (sources/pose, long format) and its
  file in the latest Kalman v2 run; "FDLC markers" / "Kalman markers"
  (+ Skeleton, likelihood threshold, pose scale) draw them on the frame, in
  video coordinates, before the display rotation;
* "● Record" keeps every frame shown (overlays included) as numbered PNGs
  under derived/frames/extracted/<video>_rec_<time>/ with frames.csv;
  pressing it again encodes derived/frames/annotated/<video>_rec_<time>.mp4.
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
pd.DataFrame({F: np.sin(np.arange(N) / 5)}).to_parquet(fam / "v1.parquet")
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

from mufasa.ui_qt import pose_overlay as po  # noqa: E402

check("FDLC pose found under sources/pose",
      po.find_fdlc_pose(cfg, "v1") == root / "sources" / "pose" / "v1.fdlc.parquet")
check("Kalman pose found in the latest run",
      po.find_kalman_pose(cfg, "v1") == run / "v1_smoothed_v2.deadbeef.parquet")
check("unknown video: none", po.find_fdlc_pose(cfg, "zz") is None
      and po.find_kalman_pose(cfg, "zz") is None)
ov = po.PoseOverlay.for_video(cfg, "v1", str(vd / "v1.mp4"))
check("both sources loaded, skeleton from the project",
      ov.available == {"fdlc": True, "kalman": True} and ("nose", "tail") in ov.edges)
blank = np.full((H, W, 3), 40, np.uint8)
img = ov(blank, 10)
check("overlay draws on a copy", (blank == 40).all() and not (img == 40).all())
check("Kalman dot at its position, FDLC ring at its own",
      not (img[180, 90] == 40).all() and not (img[120 - 6:120 + 7, 90 - 6:90 + 7] == 40).all())
check("FDLC point below the likelihood threshold not drawn",
      (ov(blank, 11)[60 - 8:60 + 9, 250 - 8:250 + 9] == 40).all()
      and not (ov(blank, 10)[60 - 8:60 + 9, 250 - 8:250 + 9] == 40).all())
mid = (int((90 + 250) / 2), int((180 + 200) / 2))
check("skeleton line drawn, and can be turned off",
      not (img[mid[1] - 2:mid[1] + 3, mid[0]] == 40).all())
ov.show_skeleton = False
check("…off", (ov(blank, 10)[mid[1] - 2:mid[1] + 3, mid[0]] == 40).all())
ov.show_skeleton = True
ov.scale = 0.5
check("pose scale moves the markers", not (ov(blank, 10)[90, 45] == 40).all())
ov.scale = 1.0
ov.show_fdlc = ov.show_kalman = False
check("nothing enabled: frame unchanged", ov(blank, 10) is blank)
ov.show_fdlc = ov.show_kalman = True

# ---------------------------------------------------------------- viewer
from mufasa.ui_qt.feature_viewer import FeatureViewerWidget  # noqa: E402

V = FeatureViewerWidget(cfg, str(vd / "v1.mp4"), [F])
V.resize(900, 700)
check("viewer: overlay checkboxes enabled where pose exists",
      V.fdlc_cb.isEnabled() and V.kalman_cb.isEnabled() and not V.fdlc_cb.isChecked())
V.scrubber.seek(10)
plain = V.scrubber.last_display_frame.copy()
V.kalman_cb.setChecked(True)
check("Kalman overlay shown on the current frame",
      not np.array_equal(V.scrubber.last_display_frame, plain) and V.overlay is not None)
V.fdlc_cb.setChecked(True)
V.p_spin.setValue(0.1)
check("threshold applied live", V.overlay.p_threshold == 0.1)
V.scrubber.rotate_cw()
check("rotation applies after the overlay (markers rotate with the video)",
      V.scrubber.last_display_frame.shape[:2] == (W, H))
V.scrubber.reset_orientation()

V.record_btn.setChecked(True)
rec = V.recorder
check("record: frames folder under derived/frames/extracted",
      rec is not None and rec.frames_dir.parent == root / "derived" / "frames" / "extracted"
      and rec.n == 1 and V.record_btn.text().startswith("■"))
for f in range(11, 31):           # play 20 frames
    V.scrubber.seek(f)
V.scrubber.seek(30)               # same frame again: not duplicated
check("each shown frame saved once as a PNG", rec.n == 21
      and len(list(rec.frames_dir.glob("[0-9]*.png"))) == 21)
V.record_btn.setChecked(False)
t0 = time.time()
while not V.record_btn.isEnabled() and time.time() - t0 < 60:
    app.processEvents()
    time.sleep(0.05)
idx = pd.read_csv(rec.frames_dir / "frames.csv")
check("frames.csv maps images to video frames",
      list(idx.video_frame) == list(range(10, 31)) and idx.image.iloc[0] == "000001.png")
cap = cv2.VideoCapture(str(rec.video_path))
n_out = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
ok, fr = cap.read()
cap.release()
check("stop: video made under derived/frames/annotated with all frames",
      rec.video_path.parent == root / "derived" / "frames" / "annotated"
      and rec.video_path.is_file() and n_out == 21 and "Saved:" in V.rec_status.text())
check("recorded frames include the overlay", ok and fr.std() > 3)
check("frames kept by default", rec.frames_dir.is_dir())
d2 = Path(tempfile.mkdtemp())
for i in range(3):
    cv2.imwrite(str(d2 / f"{i + 1:06d}.png"), np.full((51, 63, 3), 90, np.uint8))
ok2, msg2 = po.encode_frames(d2, d2 / "o.mp4", 10, prefer_ffmpeg=False)
cap = cv2.VideoCapture(str(d2 / "o.mp4"))
check("OpenCV fallback (odd frame size made even)",
      ok2 and "OpenCV" in msg2 and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 3)
cap.release()
check("nothing recorded: reported", po.encode_frames(Path(tempfile.mkdtemp()),
                                                    d2 / "x.mp4", 10)[0] is False)
V.keep_frames_cb.setChecked(False)
V.record_btn.setChecked(True)
r2 = V.recorder
V.scrubber.seek(2)
V.record_btn.setChecked(False)
t0 = time.time()
while not V.record_btn.isEnabled() and time.time() - t0 < 60:
    app.processEvents()
    time.sleep(0.05)
check("'Keep frames' off: only the video stays",
      r2.video_path.is_file() and not r2.frames_dir.exists())
V.cleanup()

# viewer for a video without pose: boxes disabled, no crash
cv2.imwrite(str(Path(tempfile.mkdtemp()) / "x.png"), blank)
w = cv2.VideoWriter(str(vd / "v2.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20, (W, H))
for i in range(5):
    w.write(blank)
w.release()
pd.DataFrame({F: np.zeros(5)}).to_parquet(fam / "v2.parquet")
V2 = FeatureViewerWidget(cfg, str(vd / "v2.mp4"), [F])
check("no pose for the video: overlay boxes disabled, with a reason",
      not V2.fdlc_cb.isEnabled() and not V2.kalman_cb.isEnabled()
      and "No " in V2.fdlc_cb.toolTip())
V2.cleanup()

failed = [n for n, ok in checks if not ok]
for n, ok in checks:
    print(("PASS " if ok else "FAIL ") + n)
print(f"smoke_122ju_viewer_pose_overlay_and_record: {len(checks) - len(failed)}/{len(checks)} "
      "checks passed")
sys.exit(1 if failed else 0)
