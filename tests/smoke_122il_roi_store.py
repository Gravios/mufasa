"""Smoke test for patch 122il — ROIs stored in rois.json.

The SimBA ROI_definitions.h5 (three pickled HDF5 tables) is replaced by
``logs/measures/rois.json``: geometry + style only, per video, with the
frame size it was drawn on; everything derived is computed on read. No
H5 is read or written any more.

Runs for real: a ProjectConfigCreator project with generated videos of two
resolutions, ROILogic load/save (including two editors open at once),
read_roi_data, ConfigReader.read_roi_data, multiply_ROIs (with rescaling),
reset_video_ROIs, write_frames, and a corrupt file.
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
_tk.messagebox.showerror = lambda *a, **k: None
sys.modules.setdefault("tkinter", _tk)
sys.modules.setdefault("tkinter.messagebox", _tk.messagebox)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mufasa.project_layout import project_paths_from_config  # noqa: E402
from mufasa.roi_tools import roi_store as S  # noqa: E402
from mufasa.roi_tools.roi_logic import ROILogic  # noqa: E402
from mufasa.utils.config_creator import ProjectConfigCreator  # noqa: E402
from mufasa.utils.read_write import read_roi_data  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


tmp = Path(tempfile.mkdtemp())
cfg = ProjectConfigCreator(
    project_path=str(tmp), project_name="rs", target_list=[],
    pose_estimation_bp_cnt="user_defined", body_part_config_idx=0,
    animal_cnt=1, file_type="parquet", body_parts=["a", "b"],
).config_path
P = project_paths_from_config(cfg)
vd = Path(P["video_dir"])
vd.mkdir(parents=True, exist_ok=True)
for stem, size in (("vidA", (640, 360)), ("vidB", (640, 360)),
                   ("vidHD", (1280, 720))):
    w = cv2.VideoWriter(str(vd / f"{stem}.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"), 30, size)
    for _ in range(3):
        w.write(np.zeros((size[1], size[0], 3), np.uint8))
    w.release()
pd.DataFrame({"Video": ["vidA", "vidB", "vidHD"], "fps": [30] * 3,
              "Resolution_width": [640, 640, 1280],
              "Resolution_height": [360, 360, 720],
              "Distance_in_mm": [0] * 3, "pixels/mm": [2.0, 2.0, 4.0]},
             ).to_csv(P["video_info_path"], index=False)
roi_path = Path(P["roi_definitions_path"])
STYLE = dict(color_name="Red", bgr=(0, 0, 255), thickness=2, ear_tag_size=8)
ring = [(int(320 + 100 * np.cos(t)), int(180 + 100 * np.sin(t)))
        for t in np.linspace(0, 2 * np.pi, 40, endpoint=False)]

check("the project's ROI file is logs/measures/rois.json",
      roi_path.name == "rois.json" and roi_path.parent.name == "measures")

# ---------------------------------------------------------------- write ---
A = ROILogic(config_path=cfg, video_path=str(vd / "vidA.mp4"))
A.add_rectangle(name="arena", top_left=(40, 20), bottom_right=(600, 340), **STYLE)
A.add_polygon(name="ring", vertices=ring, **STYLE)
A.add_circle(name="cup", center=(560, 280), radius=40,
             color_name="Blue", bgr=(255, 0, 0), thickness=3, ear_tag_size=8)
A.save()
doc = json.loads(roi_path.read_text())
va = doc["videos"]["vidA"]
check("file is plain JSON with format/version and the video's frame size",
      doc["format"] == "mufasa-rois" and doc["version"] == 1
      and va["frame_size"] == [640, 360])
check("stores geometry + style only (no derived values)",
      va["rois"][0] == {"name": "arena", "type": "rectangle", "x": 40,
                        "y": 20, "w": 560, "h": 320, "color": "#ff0000",
                        "line_width": 2, "tag_size": 8})
check("a 40-vertex polygon is stored with all 40 vertices",
      next(r for r in va["rois"] if r["name"] == "ring")["vertices"]
      == [list(v) for v in ring])
check("colour stored as #rrggbb", next(r for r in va["rois"]
                                       if r["name"] == "cup")["color"] == "#0000ff")
check("no HDF5 file is written", not list(Path(P["project_root"]).rglob("*.h5")))

# ------------------------------------------------- read back / editors ---
A2 = ROILogic(config_path=cfg, video_path=str(vd / "vidA.mp4"))
check("previously saved ROIs load back into the editor",
      sorted(A2.all_roi_names) == ["arena", "cup", "ring"]
      and A2.get_roi("ring").geometry["vertices"] == [list(v) for v in ring]
      and A2.get_roi("cup").color_name == "Blue")
B = ROILogic(config_path=cfg, video_path=str(vd / "vidB.mp4"))
A3 = ROILogic(config_path=cfg, video_path=str(vd / "vidA.mp4"))
B.add_rectangle(name="nest", top_left=(1, 1), bottom_right=(50, 50), **STYLE)
B.save()
A3.delete_roi("cup")
A3.save()          # A3 was opened before B saved
store = S.load_rois(roi_path)
check("two editors open at once: saving one never drops the other's video",
      store.videos_with_rois() == ["vidA", "vidB"]
      and store.rois("vidB")[0].name == "nest"
      and sorted(r.name for r in store.rois("vidA")) == ["arena", "ring"])

# ---------------------------------------------------- analysis tables ---
r, c, p = read_roi_data(roi_path)
ring_row = p[p.Name == "ring"].iloc[0]
arena = r[r.Name == "arena"].iloc[0]
check("read_roi_data gives the analysis tables with derived columns",
      isinstance(ring_row["vertices"], np.ndarray)
      and ring_row["vertices"].shape == (40, 2)
      and isinstance(arena["Tags"], dict) and "Left tag" in arena["Tags"]
      and arena["Center_X"] == 320 and arena["width"] == 560
      and arena["Color BGR"] == (0, 0, 255))
check("cm sizes come from the CURRENT pixels/mm in video_info.csv",
      abs(arena["width_cm"] - 560 / 20.0) < 1e-9
      and abs(arena["area_cm"] - (560 * 320) / 400.0) < 1e-9)
check("empty tables still carry their full column set",
      len(c) == 0 and "radius" in c.columns and "centerX" in c.columns)

from mufasa.mixins.config_reader import ConfigReader  # noqa: E402

cr = ConfigReader(config_path=cfg, read_video_info=False)
cr.read_roi_data()
check("ConfigReader.read_roi_data reads rois.json (shape names, roi_dict)",
      sorted(cr.shape_names) == ["arena", "nest", "ring"]
      and set(cr.video_names_w_rois) == {"vidA", "vidB"}
      and len(cr.roi_dict["polygons"]) == 1)

# -------------------------------------------------- multiply / reset ---
from mufasa.roi_tools.roi_utils import multiply_ROIs, reset_video_ROIs  # noqa: E402

multiply_ROIs(config_path=cfg, filename=str(vd / "vidA.mp4"))
store = S.load_rois(roi_path)
hd = {x.name: x for x in store.rois("vidHD")}
check("Apply to all copies to every other video (replacing their ROIs)",
      store.videos_with_rois() == ["vidA", "vidB", "vidHD"]
      and sorted(x.name for x in store.rois("vidB")) == ["arena", "ring"])
check("…rescaled to a target with a different resolution (640x360 → 1280x720)",
      store.frame_size("vidHD") == (1280, 720)
      and hd["arena"].geometry == {"x": 80, "y": 40, "w": 1120, "h": 640}
      and hd["ring"].geometry["vertices"][0] == [ring[0][0] * 2, ring[0][1] * 2])
reset_video_ROIs(config_path=cfg, filename=str(vd / "vidB.mp4"))
check("reset_video_ROIs removes only that video",
      S.load_rois(roi_path).videos_with_rois() == ["vidA", "vidHD"])

# ------------------------------------------------------- write_frames ---
r, c, p = read_roi_data(roi_path)
r2 = r[r.Video == "vidA"].copy()
r2.loc[:, "topLeftX"] = 10
S.write_frames(roi_path, r2, c.iloc[0:0], p.iloc[0:0], whole_project=False)
store = S.load_rois(roi_path)
check("write_frames(whole_project=False) replaces only the tables' videos",
      store.videos_with_rois() == ["vidA", "vidHD"]
      and [x.name for x in store.rois("vidA")] == ["arena"]
      and store.rois("vidA")[0].geometry["x"] == 10
      and store.frame_size("vidA") == (640, 360))
S.write_frames(roi_path, r2, None, None, whole_project=True)
check("write_frames(whole_project=True) makes the tables the whole set",
      S.load_rois(roi_path).videos_with_rois() == ["vidA"])

# ---------------------------------------------------------- bad file ---
roi_path.write_text("{ not json")
L = ROILogic(config_path=cfg, video_path=str(vd / "vidA.mp4"))
check("an unreadable rois.json is reported on load", bool(L.load_error))
L.add_rectangle(name="x", top_left=(1, 1), bottom_right=(9, 9), **STYLE)
try:
    L.save()
    refused = False
except S.RoiFileError:
    refused = True
check("…and is never overwritten by a save",
      refused and roi_path.read_text() == "{ not json")
try:
    S.Roi("p", "polygon", {"vertices": [[0, 0], [1, 1], [0, 0]]})
    bad_poly = False
except ValueError:
    bad_poly = True
check("polygons need 3 distinct vertices; no upper limit", bad_poly
      and len(S.Roi("p", "polygon", {"vertices": ring * 1}).geometry["vertices"]) == 40)

# ------------------------------------- a real ROI analysis on rois.json ---
tmp2 = Path(tempfile.mkdtemp())
cfg2 = ProjectConfigCreator(
    project_path=str(tmp2), project_name="an", target_list=[],
    pose_estimation_bp_cnt="user_defined", body_part_config_idx=0,
    animal_cnt=1, file_type="parquet", body_parts=["nose", "tail"],
).config_path
P2 = project_paths_from_config(cfg2)
vd2 = Path(P2["video_dir"])
vd2.mkdir(parents=True, exist_ok=True)
w = cv2.VideoWriter(str(vd2 / "v1.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                    30, (640, 360))
for _ in range(3):
    w.write(np.zeros((360, 640, 3), np.uint8))
w.release()
pd.DataFrame({"Video": ["v1"], "fps": [30], "Resolution_width": [640],
              "Resolution_height": [360], "Distance_in_mm": [0],
              "pixels/mm": [2.0]}).to_csv(P2["video_info_path"], index=False)
L2 = ROILogic(config_path=cfg2, video_path=str(vd2 / "v1.mp4"))
L2.add_rectangle(name="left", top_left=(0, 0), bottom_right=(320, 360), **STYLE)
L2.add_polygon(name="ring", vertices=[
    (int(480 + 60 * np.cos(t)), int(180 + 60 * np.sin(t)))
    for t in np.linspace(0, 2 * np.pi, 30, endpoint=False)], **STYLE)
L2.add_circle(name="cup", center=(480, 180), radius=20, **STYLE)
L2.save()
xs = np.where(np.arange(300) < 150, 100.0, 480.0)
cr2 = ConfigReader(config_path=cfg2, read_video_info=False)
os.makedirs(cr2.outlier_corrected_dir, exist_ok=True)
pd.DataFrame({"nose_x": xs, "nose_y": 180.0, "nose_p": 1.0, "tail_x": xs,
              "tail_y": 180.0, "tail_p": 1.0}).to_parquet(
    os.path.join(cr2.outlier_corrected_dir, "v1.parquet"))
try:
    from mufasa.roi_tools.roi_aggregate_statistics_analyzer import (
        ROIAggregateStatisticsAnalyzer,
    )
    an = ROIAggregateStatisticsAnalyzer(config_path=cfg2, body_parts=["nose"],
                                        total_time=True, entry_counts=True)
    an.run()
    an.save()
    out = sorted(Path(P2["logs_dir"]).glob("ROI_descriptive_statistics_*.csv"))
    res = pd.read_csv(out[-1])
    t = res[res.MEASUREMENT == "TOTAL ROI TIME (S)"].set_index("SHAPE")["VALUE"]
    check("ROI aggregate analysis runs on rois.json: 5 s in the rectangle, "
          "5 s in the 30-vertex polygon and the circle",
          dict(t) == {"left": 5.0, "ring": 5.0, "cup": 5.0})
except ModuleNotFoundError as exc:      # optional analysis deps
    check(f"ROI analysis skipped: {exc}", True)

n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122il_roi_store: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
