"""Smoke test for patch 122lr — the 3-D viewer finds a 3-D file's video by itself.

User: "The new viewer doesn't find the videos."; "It automatically found them before".

* the project's videos are listed from subfolders of ``sources/videos`` too;
* the 3-D viewer's launcher lists every 3-D file (runs, motion capture, ``mufasa run``'s beside
  the 2-D labels) but not the gait / dynamics / joint files written beside them, and by default
  opens each with the video it was made from, found by name (``K7_cam`` for
  ``K7_camDLC_resnet50…_3d.parquet``; or by its 2-D labels' name);
* a video chosen lists only its 3-D files; ``(no video)`` opens none;
* the viewer's *Video…* shows another video; ``mufasa-view3d --pose`` finds the video too;
* finding a video's 3-D file skips the gait file written after it.
"""
from __future__ import annotations

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
from PySide6.QtWidgets import QApplication  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


app = QApplication.instance() or QApplication([])

from mufasa.pose3d.io import write_pose3d  # noqa: E402
from mufasa.project_layout import project_paths_from_config  # noqa: E402
from mufasa.ui_qt import pose3d_viewer as PV  # noqa: E402
from mufasa.ui_qt.forms.pose3d import AUTO, Pose3DViewerLauncher, _videos  # noqa: E402
from mufasa.utils.config_creator import ProjectConfigCreator  # noqa: E402


def video(path: Path, n: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    for k in range(n):
        w.write(np.full((48, 64, 3), k % 250, np.uint8))
    w.release()
    return path


def pose3d(path: Path, n: int, **meta) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"frame": np.arange(n), "nose_x": 0.0, "nose_y": 0.0, "nose_z": 40.0})
    return write_pose3d(df, path, {"units": "mm", **meta})


tmp = Path(tempfile.mkdtemp())
cfg = ProjectConfigCreator(
    project_path=str(tmp), project_name="vv", target_list=["rear"],
    pose_estimation_bp_cnt="user_defined", body_part_config_idx=0,
    animal_cnt=1, file_type="parquet", body_parts=["a", "b"]).config_path
root = Path(cfg).parent
vd = Path(project_paths_from_config(cfg)["video_dir"])
v_k7 = video(vd / "rat6" / "K7_cam.mp4", 40)                       # in a subfolder
v_k7b = video(vd / "K7_cam_2.mp4", 50)                             # a longer name beginning the same
v_r2 = video(vd / "R2.MP4", 30)
run = root / "derived" / "pose3d" / "20261123-120000-abcdef"
f_k7 = pose3d(run / "K7_camDLC_resnet50_shuffle1_3d.parquet", 40)
f_k7b = pose3d(run / "K7_cam_2DLC_resnet50_3d.parquet", 50)
time.sleep(0.05)
pd.DataFrame({"frame": np.arange(40), "speed_mm_s": 1.0}).to_parquet(run / "K7_camDLC_resnet50_shuffle1_gait.parquet")
f_r2 = pose3d(root / "sources" / "pose" / "R2_3d.parquet", 30)     # `mufasa run`'s, beside the labels
f_lab = pose3d(root / "sources" / "pose3d" / "session9_3d.parquet", 30,   # named after its 2-D labels
               source_pose=str(root / "sources" / "pose" / "R2.parquet"))
f_mocap = pose3d(root / "sources" / "pose3d" / "mocap_rat1.parquet", 30)

vids = _videos(root, cfg)
check(f"the project's videos, subfolders too: {[str(v.relative_to(vd)) for v in vids]}",
      set(vids) == {v_k7, v_k7b, v_r2})

L = Pose3DViewerLauncher(None, cfg)
files = [L.file.itemData(i) for i in range(L.file.count())]
check(f"the launcher, by default: every 3-D file ({len(files)}), not the gait file beside one",
      L.video.currentData() == AUTO and set(map(Path, files)) == {f_k7, f_k7b, f_r2, f_lab, f_mocap})
got = {Path(f).name: L.video_for(f) for f in files}
want = {f_k7.name: str(v_k7), f_k7b.name: str(v_k7b), f_r2.name: str(v_r2), f_lab.name: str(v_r2),
        f_mocap.name: None}
check("each 3-D file with its own video, found by name: "
      + ", ".join(f"{k} → {Path(v).name if v else None}" for k, v in got.items()), got == want)
L.file.setCurrentIndex(files.index(str(f_k7)))
check(f"the launcher says which video: '{L.found.text()}'", L.found.text() == "Video: K7_cam.mp4")
win = L.open()
app.processEvents()
pv = win.viewer
check(f"opened: the video plays with it ({pv.scrubber.total_frames} frames), its name on Video…",
      pv.video_path == str(v_k7) and pv.scrubber.total_frames == 40 and "K7_cam.mp4" in pv.video_btn.text())
pv.set_video(v_k7b)
app.processEvents()
check("Video…: another video shown", pv.video_path == str(v_k7b) and pv.scrubber.total_frames == 50)
win.close()

L.video.setCurrentIndex(L.video.findData(str(v_k7b)))
sub = [L.file.itemData(i) for i in range(L.file.count())]
check(f"a video chosen: its 3-D files only ({[Path(s).name for s in sub]})", sub == [str(f_k7b)])
L.video.setCurrentIndex(L.video.findData(None))
L.file.setCurrentIndex(L.file.findData(str(f_mocap)))
win = L.open()
app.processEvents()
check("(no video): motion capture alone, Video… offered", win.viewer.video_path is None
      and win.viewer.video_btn.text() == "Video…" and L.found.text() == "")
win.close()

from mufasa.tools import view3d  # noqa: E402

a = view3d.build_parser().parse_args(["--pose", str(f_k7)])
b = view3d.build_parser().parse_args(["--pose", str(f_k7), "--no-video"])
check("mufasa-view3d --pose: the video found (--no-video: none)",
      view3d.resolve(a)[0] == str(v_k7) and view3d.resolve(b)[0] is None)
check("a video's 3-D file: not the gait file written after it, nor K7_cam_2's for K7_cam",
      PV.find_pose3d(str(v_k7), cfg) == f_k7)

for name, ok in checks:
    print(("PASS  " if ok else "FAIL  ") + name)
n_ok = sum(ok for _, ok in checks)
print(f"smoke_122lr_viewer_videos: {n_ok}/{len(checks)} checks passed")
sys.exit(0 if n_ok == len(checks) else 1)
