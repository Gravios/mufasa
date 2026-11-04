"""Smoke test for patch 122ik — saving ROIs never loses other videos' ROIs.

ROILogic.save rewrites the whole ROI_definitions.h5. Before 122ik:

* it read the existing file through read_roi_data, which drops every
  rectangle / circle row with a NaN, and swallowed read errors — then wrote
  a file holding only the current video;
* its own rows used a private format (string colour / tags / vertices,
  circles with extra Center_X / Center_Y columns). Mixed with SimBA-format
  rows that left NaN cells, and the next read dropped those rows. Measured:
  saving one circle on video A deleted video B's SimBA-format circle;
* string vertices / tags broke ROI analysis (``vertices.astype``) and
  visualisation (``Tags.items()``).

Runs for real: a ProjectConfigCreator project with generated videos, real H5
files in SimBA's format and in the old Mufasa format, ROILogic load / save,
read_roi_data, ConfigReader-style reading, and multiply_ROIs.
"""
from __future__ import annotations

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
from mufasa.roi_tools.roi_logic import ROILogic  # noqa: E402
from mufasa.roi_tools.roi_utils import (  # noqa: E402
    get_circle_df_headers,
    get_polygon_df_headers,
    get_rectangle_df_headers,
)
from mufasa.utils.config_creator import ProjectConfigCreator  # noqa: E402
from mufasa.utils.read_write import read_roi_data  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


def project(name: str):
    tmp = Path(tempfile.mkdtemp())
    cfg = ProjectConfigCreator(
        project_path=str(tmp), project_name=name, target_list=[],
        pose_estimation_bp_cnt="user_defined", body_part_config_idx=0,
        animal_cnt=1, file_type="parquet", body_parts=["a", "b"],
    ).config_path
    P = project_paths_from_config(cfg)
    vd = Path(P["video_dir"])
    vd.mkdir(parents=True, exist_ok=True)
    for s in ("vidA", "vidB", "vidC"):
        w = cv2.VideoWriter(str(vd / f"{s}.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 360))
        for _ in range(3):
            w.write(np.zeros((360, 640, 3), np.uint8))
        w.release()
    h5 = P["roi_definitions_path"]
    os.makedirs(os.path.dirname(h5), exist_ok=True)
    return cfg, vd, h5


def simba_rows(video: str):
    rect = dict(zip(get_rectangle_df_headers(), [
        video, "Rectangle", "arena", "Red", (0, 0, 255), 3, 50, 50, 0, 0,
        100, 100, 100, 100, 10.0, 10.0, 100.0, {"Center tag": (50, 50)}, 15],
        strict=True))
    circ = dict(zip(get_circle_df_headers(), [
        video, "Circle", "cup", "Blue", (255, 0, 0), 3, 200, 200, 30, 3.0,
        28.0, {"Center tag": (200, 200), "Border tag": (170, 200)}, 15],
        strict=True))
    poly = dict(zip(get_polygon_df_headers(), [
        video, "Polygon", "nest", "Green", (0, 255, 0), 3, 300, 300,
        np.array([[280, 280], [320, 280], [320, 320], [280, 320]]),
        (300, 300), 1600.0, 56, 16.0, {"Center_tag": (300, 300)}, 15],
        strict=True))
    return rect, circ, poly


def write(h5, rects, circs, polys):
    pd.DataFrame(rects).to_hdf(h5, key="rectangles", mode="w")
    pd.DataFrame(circs).to_hdf(h5, key="circleDf", mode="a")
    pd.DataFrame(polys).to_hdf(h5, key="polygons", mode="a")


def on_disk(h5):
    r, c, p = read_roi_data(h5)
    return sorted((str(x.Video), str(x.Name))
                  for df in (r, c, p) for _, x in df.iterrows())


STYLE = dict(color_name="Red", bgr=(0, 0, 255), thickness=2, ear_tag_size=8)

# ------------------------------------------- SimBA-format neighbours ---
cfg, vd, h5 = project("simba")
r, c, p = simba_rows("vidB")
write(h5, [r], [c], [p])
B = ROILogic(config_path=cfg, video_path=str(vd / "vidB.mp4"))
check("previously annotated SimBA-format ROIs are detected on load",
      sorted(B.all_roi_names) == ["arena", "cup", "nest"]
      and B.get_roi("nest").geometry["vertices"][2] == [320, 320])
A = ROILogic(config_path=cfg, video_path=str(vd / "vidA.mp4"))
check("other videos' ROIs are visible to this video as other_video_rois",
      set(A.other_video_rois.get("vidB", {})) == {"arena", "cup", "nest"})
A.add_circle(name="c1", center=(10, 10), radius=5, **STYLE)
A.save()
A.save()
check("saving a circle on vidA (twice) keeps vidB's SimBA circle "
      "(it used to be deleted)",
      on_disk(h5) == [("vidA", "c1"), ("vidB", "arena"), ("vidB", "cup"),
                      ("vidB", "nest")])
r2, c2, p2 = read_roi_data(h5)
check("read_roi_data's NaN-dropping now has nothing to drop",
      len(r2) == 1 and len(c2) == 2 and not c2.isna().any().any()
      and not r2.isna().any().any())
check("rows are written in SimBA's native types",
      isinstance(p2.iloc[0]["vertices"], np.ndarray)
      and isinstance(c2.iloc[0]["Tags"], dict)
      and isinstance(c2.iloc[1]["Tags"], dict)
      and isinstance(c2.iloc[1]["Color BGR"], tuple))
check("circle table keeps SimBA's column set (no extra NaN-prone columns)",
      list(c2.columns) == get_circle_df_headers())
A.add_polygon(name="pg", vertices=[(1, 1), (50, 1), (50, 50), (25, 60),
                                   (1, 50), (0, 25)], **STYLE)
A.add_rectangle(name="box", top_left=(5, 5), bottom_right=(40, 30), **STYLE)
A.save()
r3, c3, p3 = read_roi_data(h5)
pg = p3[p3.Video == "vidA"].iloc[0]
check("ROI analysis can use the polygon (vertices.astype) with all 6 vertices",
      pg["vertices"].astype(np.int32).shape == (6, 2))
bx = r3[r3.Video == "vidA"].iloc[0]
check("visualisation tags exist (rectangle 'Left tag', circle 'Border tag')",
      "Left tag" in bx["Tags"] and "Border tag" in c3[c3.Video == "vidA"].iloc[0]["Tags"]
      and bx["Center_X"] == 22 and bx["width"] == 35)
B2 = ROILogic(config_path=cfg, video_path=str(vd / "vidB.mp4"))
B2.delete_roi("cup")
B2.save()
check("vidB editing its own ROIs leaves vidA's untouched",
      on_disk(h5) == [("vidA", "box"), ("vidA", "c1"), ("vidA", "pg"),
                      ("vidB", "arena"), ("vidB", "nest")])

# ------------------------------------------- old Mufasa-format file ---
cfg, vd, h5 = project("oldmufasa")
old_rect = {"Video": "vidB", "Shape_type": "Rectangle", "Name": "zone",
            "Color name": "Red", "Color BGR": "(0, 0, 255)", "Thickness": 3,
            "Center_X": 20, "Center_Y": 20, "topLeftX": 10, "topLeftY": 10,
            "Bottom_right_X": 30, "Bottom_right_Y": 30, "width": 20,
            "height": 20, "width_cm": 0.0, "height_cm": 0.0, "area_cm": 0.0,
            "Tags": "{'Center tag': (20, 20)}", "Ear_tag_size": 8}
old_circ = {"Video": "vidB", "Shape_type": "Circle", "Name": "cup2",
            "Color name": "Blue", "Color BGR": "(255, 0, 0)", "Thickness": 3,
            "centerX": 80, "centerY": 80, "radius": 10, "radius_cm": 0.0,
            "area_cm": 0.0, "Tags": "{'Center tag': (80, 80)}",
            "Ear_tag_size": 8, "Center_X": 80, "Center_Y": 80}
old_poly = {"Video": "vidB", "Shape_type": "Polygon", "Name": "tri",
            "Color name": "Red", "Color BGR": "(0, 0, 255)", "Thickness": 3,
            "Center_X": 5, "Center_Y": 5, "vertices": "[[0, 0], [10, 0], [5, 9]]",
            "center": "(5, 3)", "area": 45.0, "max_vertice_distance": 10.0,
            "area_cm": 0.0, "Tags": "{'Center tag': (5, 3)}", "Ear_tag_size": 8}
_, simba_circ, _ = simba_rows("vidC")
write(h5, [old_rect], [old_circ, simba_circ], [old_poly])
B = ROILogic(config_path=cfg, video_path=str(vd / "vidB.mp4"))
check("files saved by older Mufasa versions still load",
      sorted(B.all_roi_names) == ["cup2", "tri", "zone"]
      and B.get_roi("tri").geometry["vertices"] == [[0, 0], [10, 0], [5, 9]])
C = ROILogic(config_path=cfg, video_path=str(vd / "vidC.mp4"))
check("a mixed old/SimBA file shows every video's ROIs on load",
      C.all_roi_names == ["cup"])
A = ROILogic(config_path=cfg, video_path=str(vd / "vidA.mp4"))
A.add_rectangle(name="r", top_left=(1, 1), bottom_right=(9, 9), **STYLE)
A.save()
r4, c4, p4 = read_roi_data(h5)
check("saving repairs old rows to native types and keeps all of them",
      on_disk(h5) == [("vidA", "r"), ("vidB", "cup2"), ("vidB", "tri"),
                      ("vidB", "zone"), ("vidC", "cup")]
      and all(isinstance(t, dict) for t in c4["Tags"])
      and isinstance(p4.iloc[0]["vertices"], np.ndarray))

# ------------------------------------------------ apply to all works ---
from mufasa.roi_tools.roi_utils import multiply_ROIs  # noqa: E402

multiply_ROIs(config_path=cfg, filename=str(vd / "vidA.mp4"))
check("multiply_ROIs reads the saved file and copies to every video",
      on_disk(h5) == [("vidA", "r"), ("vidB", "r"), ("vidC", "r")])

# ------------------------------------------------- unreadable file ---
cfg, vd, h5 = project("corrupt")
Path(h5).write_bytes(b"this is not an HDF5 file")
A = ROILogic(config_path=cfg, video_path=str(vd / "vidA.mp4"))
check("an unreadable ROI file is reported on load", bool(A.load_error))
A.add_rectangle(name="r", top_left=(1, 1), bottom_right=(9, 9), **STYLE)
try:
    A.save()
    refused = False
except RuntimeError:
    refused = True
check("…and save refuses to overwrite it (it used to replace it)",
      refused and Path(h5).read_bytes() == b"this is not an HDF5 file")

n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122ik_roi_save_preserves: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
