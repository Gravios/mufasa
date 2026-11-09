"""Smoke test for patch 122jw — style / visibility popup for the FDLC and
Kalman overlays.

User request: "I also need control over the fdlc and kalman filtered
markers/skeletons, can you make a configuration popup which allows me to
change the visual style/visibilty of the markers/segments?"

"Style…" in the viewer opens a non-modal popup with a tab per source:
markers (show, shape, size, line width, opacity, outline, palette / single
colour, names; per-marker show + own colour) and skeleton (show, colour,
width, opacity; per-segment show + own colour), a legend switch and Reset.
Edits redraw the video at once; Close saves to
logs/measures/pose_overlay_style.toml, used next time.
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
    animal_cnt=1, file_type="parquet", body_parts=["nose", "tail", "mid"]).config_path
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

from PySide6.QtCore import Qt  # noqa: E402

from mufasa.ui_qt import overlay_style as os_  # noqa: E402
from mufasa.ui_qt.feature_viewer import FeatureViewerWidget  # noqa: E402

# ---------------------------------------------------------------- style model
st = os_.OverlayStyle()
check("defaults: FDLC rings without skeleton, Kalman filled dots with skeleton",
      st.fdlc.shape == "ring" and not st.fdlc.show_skeleton
      and st.kalman.shape == "filled circle" and st.kalman.show_skeleton)
st.kalman.hidden_markers = ["tail"]
st.kalman.marker_colors = {"nose": "#00ff00"}
st.fdlc.edge_colors = {"nose|tail": "#123456"}
os_.save_style(cfg, st)
back = os_.load_style(cfg)
check("style saved / loaded (TOML)", back == st
      and os_.style_path(cfg).parts[-3:] == ("logs", "measures", "pose_overlay_style.toml"))
os_.style_path(cfg).write_text("[kalman]\nshape = 'blob'\nsize = 7\nbogus = 1\n")
b2 = os_.load_style(cfg)
check("bad values fall back, unknown keys ignored, known kept",
      b2.kalman.shape == "filled circle" and b2.kalman.size == 7)
os_.style_path(cfg).unlink()

blank = np.full((H, W, 3), 40, np.uint8)
pts = {"nose": (100, 100), "tail": (200, 100)}


def drawn(sst, **kw):
    out = blank.copy()
    os_.draw_source(out, pts, ["nose", "tail"], [("nose", "tail")], sst, 6, **kw)
    return out


s0 = os_.default_kalman()
o = drawn(s0)
check("draws markers and the segment", not (o[100, 100] == 40).all()
      and not (o[100, 150] == 40).all())
s1 = os_.default_kalman()
s1.hidden_markers = ["nose"]
s1.hidden_edges = ["nose|tail"]
o = drawn(s1)
check("hidden marker / segment not drawn", (o[100, 100] == 40).all()
      and (o[100, 150] == 40).all() and not (o[100, 200] == 40).all())
s2 = os_.default_kalman()
s2.marker_colors = {"nose": "#00ff00"}
s2.outline = False
check("own marker colour", tuple(drawn(s2)[100, 100]) == (0, 255, 0))
s3 = os_.default_kalman()
s3.color_mode, s3.color, s3.outline = "single", "#0000ff", False
o = drawn(s3)
check("single colour for all markers", tuple(o[100, 100]) == (255, 0, 0)
      and tuple(o[100, 200]) == (255, 0, 0))
s3.opacity = 0.5
s3.show_skeleton = False
check("opacity blends with the frame", 90 <= drawn(s3)[100, 100][0] <= 160)
s4 = os_.default_kalman()
s4.size, s4.show_skeleton = 15, False
o = drawn(s4)
check("size in px; skeleton off", not (o[100, 113] == 40).all() and (o[100, 150] == 40).all())
for shape in os_.SHAPES:
    s5 = os_.default_kalman()
    s5.shape = shape
    s5.show_skeleton = False
    check(f"shape '{shape}' draws", not (drawn(s5) == 40).all())
s6 = os_.default_kalman()
s6.labels = True
s6.show_skeleton = False
check("marker names drawn when asked",
      (drawn(s6)[85:100, 108:150] != 40).any() and (drawn(s0)[85:94, 110:150] == 40).all())
s7 = os_.default_kalman()
s7.skeleton_color, s7.skeleton_width = "#ff0000", 5
s7.show_markers = False
o = drawn(s7)
check("skeleton colour / width; markers off", tuple(o[100, 150]) == (0, 0, 255)
      and tuple(o[102, 150]) == (0, 0, 255) and (o[100, 95] == 40).all())

# ---------------------------------------------------------------- popup in the viewer
V = FeatureViewerWidget(cfg, str(vd / "v1.mp4"), [F])
V.resize(900, 700)
V.scrubber.seek(10)
V.kalman_cb.setChecked(True)
V.fdlc_cb.setChecked(True)
before = V.scrubber.last_display_frame.copy()
d = V.open_style_dialog()
check("Style… opens a popup with a tab per source",
      d.isVisible() and d.tabs.count() == 2 and d.tabs.tabText(0).startswith("FDLC"))
kp = d.pages["kalman"]
check("marker and segment tables list the pose's markers and skeleton",
      kp.mtable.rowCount() == len(V.overlay.kalman.markers)
      and kp.etable.rowCount() == len(V.overlay.edges) >= 1)
kp.shape.setCurrentText("square")
check("edit redraws the video at once",
      V.overlay.style.kalman.shape == "square"
      and not np.array_equal(V.scrubber.last_display_frame, before))
row = [kp.mtable.item(r, 1).data(Qt.UserRole) for r in range(kp.mtable.rowCount())].index("nose")
kp.mtable.item(row, 0).setCheckState(Qt.Unchecked)
check("unticking a marker hides it", V.overlay.style.kalman.hidden_markers == ["nose"])
kp.etable.item(0, 0).setCheckState(Qt.Unchecked)
check("unticking a segment hides it", V.overlay.style.kalman.hidden_edges == ["nose|tail"])
kp.cmode.setCurrentIndex(1)
check("single colour mode enables the colour button",
      kp.color.isEnabled() and V.overlay.style.kalman.color_mode == "single")
fp = d.pages["fdlc"]
fp.op.setValue(0.4)
fp.show_s.setChecked(True)
check("FDLC page edits FDLC only", V.overlay.style.fdlc.opacity == 0.4
      and V.overlay.style.fdlc.show_skeleton and V.overlay.style.kalman.opacity == 1.0)
d.legend.setChecked(False)
check("legend switch", V.overlay.style.legend is False)
d.close()
saved = os_.load_style(cfg)
check("Close saves to the project",
      saved.kalman.shape == "square" and saved.kalman.hidden_markers == ["nose"]
      and saved.fdlc.opacity == 0.4 and saved.legend is False)
V.cleanup()
V2 = FeatureViewerWidget(cfg, str(vd / "v1.mp4"), [F])
V2.kalman_cb.setChecked(True)
check("next viewer uses the saved style", V2.overlay.style.kalman.shape == "square")
d2 = V2.open_style_dialog()
d2.reset()
check("Reset restores the defaults", V2.overlay.style.kalman == os_.default_kalman()
      and d2.pages["kalman"].shape.currentText() == "filled circle")
d2.close()
V2.cleanup()

failed = [n for n, ok in checks if not ok]
for n, ok in checks:
    print(("PASS " if ok else "FAIL ") + n)
print(f"smoke_122jw_overlay_style_popup: {len(checks) - len(failed)}/{len(checks)} "
      "checks passed")
sys.exit(1 if failed else 0)
