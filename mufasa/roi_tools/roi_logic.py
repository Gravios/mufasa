"""
mufasa.roi_tools.roi_logic
==========================

UI-framework-independent ROI definition logic. Extracted from
:class:`mufasa.roi_tools.roi_ui_mixin.ROI_mixin` so the new Qt
panel (:mod:`mufasa.ui_qt.dialogs.roi_define_panel`) and the legacy
Tk panel can both build on the same primitives.

What's in here
--------------

* Frame buffer (which frame of the video is currently showing,
  navigation, on-disk h5 read/write).
* In-memory ROI dict (rectangles / circles / polygons for the
  current video).
* Add / delete / rename / duplicate operations.
* Persistence (read existing definitions, write back on save).
* Image rendering — overlay all current ROIs onto the active frame
  for display in the canvas.

What's NOT in here
------------------

* Tk widgets, dropdowns, button handlers.
* Qt widgets.
* OpenCV mouse callbacks. The shape-drawing OpenCV loop is run via
  :class:`mufasa.video_processors.roi_selector.ROISelector` (and the
  circle/polygon variants) — those classes are themselves clean
  (no Tk coupling), they just need values rather than dropdown
  references.

The Qt panel calls into this module as plain Python: ``logic.
add_rectangle(name=..., bgr=..., thickness=...)``.
"""
from __future__ import annotations

import configparser
import copy
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from mufasa.utils.checks import check_file_exist_and_readable
from mufasa.utils.read_write import (get_fn_ext, get_video_meta_data,
                                     read_frm_of_video, read_roi_data)


# Shape-type constants. Mirror the values used by the legacy code so
# in-memory dicts stay compatible with read_roi_data() / write_df.
RECTANGLE = "rectangle"
CIRCLE    = "circle"
POLYGON   = "polygon"

# H5 keys for persistence.
KEY_RECT = "rectangles"
KEY_CIRC = "circleDf"
KEY_POLY = "polygons"
KEY_VINFO = "video_info"

# Bounded frame-cache size (LRU eviction). 50 frames is a balance:
# enough to handle moderate scrubbing without re-decoding, low enough
# to keep memory under ~1.25 GB even at 4K resolution.
FRAME_CACHE_MAX = 50


@dataclass
class ROIDefinition:
    """Single ROI's in-memory representation. Mirrors the dict shape
    that :func:`mufasa.roi_tools.roi_utils.create_rectangle_entry`
    (and the circle / polygon equivalents) emit, but typed."""
    shape_type: Literal["rectangle", "circle", "polygon"]
    video: str
    name: str
    color_name: str
    color_bgr: Tuple[int, int, int]
    thickness: int
    ear_tag_size: int
    # Geometry — meaning depends on shape_type:
    #   rectangle: top-left + bottom-right + width + height
    #   circle:    center + radius
    #   polygon:   list of (x, y) vertices
    geometry: Dict[str, Any] = field(default_factory=dict)
    # Computed at save time
    px_per_mm: Optional[float] = None


class ROILogic:
    """Plain-Python ROI state for one video.

    Owns the active frame, the dict of currently-defined ROIs, and
    persistence. UI layers (Qt panel) wire their controls to this
    class's methods.

    Lifecycle:

    1. ``__init__(config_path, video_path)`` reads the project's
       existing ROI_definitions.h5 (if any), filters by this video
       name, and loads frame 0 into the buffer.
    2. UI calls ``add_rectangle()`` / ``add_circle()`` /
       ``add_polygon()`` after each successful draw.
    3. UI calls ``advance_frame(stride)`` / ``goto_frame(idx)`` to
       navigate.
    4. UI calls ``rendered_frame()`` whenever it needs an image to
       display (returns the current frame with all ROIs overlaid).
    5. UI calls ``save()`` to persist all definitions back to disk.
    """

    def __init__(self, config_path: str, video_path: str) -> None:
        check_file_exist_and_readable(file_path=config_path)
        check_file_exist_and_readable(file_path=video_path)
        self.config_path = config_path
        self.video_path = video_path
        _, video_name, _ = get_fn_ext(filepath=video_path)
        self.video_name = video_name

        # Read project_path from the config — same approach every
        # other Qt form uses. Falls back to the config file's parent
        # directory if the [General settings]/project_path key is
        # missing (legacy projects).
        cfg = configparser.ConfigParser()
        cfg.read(config_path)
        cfg_project_path = cfg.get(
            "General settings", "project_path", fallback="",
        ).strip()
        if cfg_project_path and os.path.isdir(cfg_project_path):
            self.project_path = cfg_project_path
        else:
            self.project_path = os.path.dirname(config_path)
        self.roi_h5_path = os.path.join(
            self.project_path, "logs", "measures", "ROI_definitions.h5",
        )

        # Video metadata
        self.video_meta = get_video_meta_data(video_path=video_path)
        self.frame_count = int(self.video_meta["frame_count"])
        self.fps = float(self.video_meta["fps"])
        self.width = int(self.video_meta["width"])
        self.height = int(self.video_meta["height"])

        # Per-pixel calibration (set elsewhere in the project; read it
        # if available so saved ROIs include cm² area). Defaults to 1.0
        # if not configured — area_cm columns will be wrong in that
        # case, but the H5 is still readable and ROIs still functional.
        self.px_per_mm = 1.0
        try:
            video_info_csv = os.path.join(
                self.project_path, "logs", "video_info.csv",
            )
            if os.path.isfile(video_info_csv):
                vi = pd.read_csv(video_info_csv)
                row = vi[vi["Video"] == self.video_name]
                if not row.empty and "pixels/mm" in row.columns:
                    self.px_per_mm = float(row["pixels/mm"].iloc[0])
        except Exception:
            pass  # Leave at 1.0 default

        # Frame buffer — bounded LRU cache. A 4K video frame is
        # ~25 MB; capping at 50 frames keeps worst-case memory at
        # ~1.25 GB even if the user scrubs through a long video.
        # Cache holds decoded BGR frames; oldest evicted first.
        self.frame_idx = 0
        self._frame_cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._cur_frame: Optional[np.ndarray] = self._read_frame(0)

        # ROI state — defs[shape_type] = {name: ROIDefinition}
        self.defs: Dict[str, Dict[str, ROIDefinition]] = {
            RECTANGLE: {}, CIRCLE: {}, POLYGON: {},
        }
        # Other-video ROIs (read-only, for the "apply from other video"
        # action). Keys are video names.
        self.other_video_rois: Dict[str, Dict[str, Any]] = {}

        self._load_existing_definitions()

    # ------------------------------------------------------------------ #
    # Frame buffer
    # ------------------------------------------------------------------ #
    def _read_frame(self, idx: int) -> Optional[np.ndarray]:
        if idx in self._frame_cache:
            # Touch — move to end to mark recent
            self._frame_cache.move_to_end(idx)
            return self._frame_cache[idx]
        try:
            frm = read_frm_of_video(
                video_path=self.video_path, frame_index=idx,
                greyscale=False, use_ffmpeg=False, raise_error=True,
            )
            self._frame_cache[idx] = frm
            # Evict oldest while over cap
            while len(self._frame_cache) > FRAME_CACHE_MAX:
                self._frame_cache.popitem(last=False)
            return frm
        except Exception:
            return None

    def goto_frame(self, idx: int) -> None:
        idx = max(0, min(idx, self.frame_count - 1))
        self.frame_idx = idx
        self._cur_frame = self._read_frame(idx)

    def advance_frame(self, stride: int) -> None:
        """Step ``stride`` frames forward (negative steps backward)."""
        self.goto_frame(self.frame_idx + stride)

    def jump_seconds(self, seconds: float) -> None:
        """Step by an absolute number of seconds (negative steps back)."""
        self.advance_frame(int(seconds * self.fps))

    def first_frame(self) -> None:
        self.goto_frame(0)

    def last_frame(self) -> None:
        self.goto_frame(self.frame_count - 1)

    # ------------------------------------------------------------------ #
    # ROI state
    # ------------------------------------------------------------------ #
    @property
    def all_roi_names(self) -> List[str]:
        out = []
        for d in self.defs.values():
            out.extend(d.keys())
        return out

    def has_roi(self, name: str) -> bool:
        return any(name in d for d in self.defs.values())

    def get_roi(self, name: str) -> Optional[ROIDefinition]:
        for d in self.defs.values():
            if name in d:
                return d[name]
        return None

    def add_rectangle(self, name: str, top_left: Tuple[int, int],
                      bottom_right: Tuple[int, int],
                      color_name: str, bgr: Tuple[int, int, int],
                      thickness: int, ear_tag_size: int) -> None:
        if self.has_roi(name):
            raise ValueError(f"ROI named {name!r} already exists for this video.")
        x1, y1 = top_left; x2, y2 = bottom_right
        w = abs(x2 - x1); h = abs(y2 - y1)
        cx = (x1 + x2) // 2; cy = (y1 + y2) // 2
        self.defs[RECTANGLE][name] = ROIDefinition(
            shape_type=RECTANGLE, video=self.video_name, name=name,
            color_name=color_name, color_bgr=bgr, thickness=thickness,
            ear_tag_size=ear_tag_size,
            geometry={
                "topLeftX": x1, "topLeftY": y1,
                "Bottom_right_X": x2, "Bottom_right_Y": y2,
                "Center_X": cx, "Center_Y": cy,
                "width": w, "height": h,
            },
            px_per_mm=self.px_per_mm,
        )

    def add_circle(self, name: str, center: Tuple[int, int], radius: int,
                   color_name: str, bgr: Tuple[int, int, int],
                   thickness: int, ear_tag_size: int) -> None:
        if self.has_roi(name):
            raise ValueError(f"ROI named {name!r} already exists for this video.")
        cx, cy = center
        self.defs[CIRCLE][name] = ROIDefinition(
            shape_type=CIRCLE, video=self.video_name, name=name,
            color_name=color_name, color_bgr=bgr, thickness=thickness,
            ear_tag_size=ear_tag_size,
            geometry={
                "centerX": cx, "centerY": cy, "radius": int(radius),
                "Center_X": cx, "Center_Y": cy,
            },
            px_per_mm=self.px_per_mm,
        )

    def add_polygon(self, name: str, vertices: List[Tuple[int, int]],
                    color_name: str, bgr: Tuple[int, int, int],
                    thickness: int, ear_tag_size: int) -> None:
        if self.has_roi(name):
            raise ValueError(f"ROI named {name!r} already exists for this video.")
        if len(vertices) < 3:
            raise ValueError(
                f"Polygon must have at least 3 vertices, got {len(vertices)}."
            )
        verts = np.asarray(vertices, dtype=np.int32)
        center = verts.mean(axis=0).astype(int)
        self.defs[POLYGON][name] = ROIDefinition(
            shape_type=POLYGON, video=self.video_name, name=name,
            color_name=color_name, color_bgr=bgr, thickness=thickness,
            ear_tag_size=ear_tag_size,
            geometry={
                "vertices": verts.tolist(),
                "Center_X": int(center[0]), "Center_Y": int(center[1]),
            },
            px_per_mm=self.px_per_mm,
        )

    def delete_roi(self, name: str) -> bool:
        for d in self.defs.values():
            if name in d:
                del d[name]
                return True
        return False

    def delete_all(self) -> None:
        for d in self.defs.values():
            d.clear()

    def rename_roi(self, old: str, new: str) -> None:
        if self.has_roi(new):
            raise ValueError(f"An ROI named {new!r} already exists.")
        for d in self.defs.values():
            if old in d:
                roi = d.pop(old)
                roi.name = new
                d[new] = roi
                return
        raise KeyError(f"No ROI named {old!r}.")

    def duplicate_roi(self, source_name: str, dest_name: str,
                      offset: Tuple[int, int] = (20, 20)) -> None:
        """Copy an ROI, shifting its position by ``offset``. The duplicate
        gets the same shape-type, color, and dimensions; just a new name
        and shifted location. Useful for placing several similar ROIs
        without redrawing each one."""
        roi = self.get_roi(source_name)
        if roi is None:
            raise KeyError(f"No ROI named {source_name!r}.")
        if self.has_roi(dest_name):
            raise ValueError(f"An ROI named {dest_name!r} already exists.")
        dx, dy = offset
        new_roi = copy.deepcopy(roi)
        new_roi.name = dest_name
        # Shift geometry
        if new_roi.shape_type == RECTANGLE:
            new_roi.geometry["topLeftX"] += dx
            new_roi.geometry["topLeftY"] += dy
            new_roi.geometry["Bottom_right_X"] += dx
            new_roi.geometry["Bottom_right_Y"] += dy
            new_roi.geometry["Center_X"] += dx
            new_roi.geometry["Center_Y"] += dy
        elif new_roi.shape_type == CIRCLE:
            new_roi.geometry["centerX"] += dx
            new_roi.geometry["centerY"] += dy
            new_roi.geometry["Center_X"] += dx
            new_roi.geometry["Center_Y"] += dy
        elif new_roi.shape_type == POLYGON:
            verts = np.asarray(new_roi.geometry["vertices"]) + (dx, dy)
            new_roi.geometry["vertices"] = verts.tolist()
            new_roi.geometry["Center_X"] += dx
            new_roi.geometry["Center_Y"] += dy
        self.defs[new_roi.shape_type][dest_name] = new_roi

    # ------------------------------------------------------------------ #
    # Rendering — return current frame with all ROIs drawn on top
    # ------------------------------------------------------------------ #
    def rendered_frame(self) -> Optional[np.ndarray]:
        """Return a copy of the current frame with every defined ROI
        overlaid, ready for display in either the OpenCV preview or
        a Qt QLabel."""
        if self._cur_frame is None:
            return None
        img = self._cur_frame.copy()
        # Rectangles
        for roi in self.defs[RECTANGLE].values():
            g = roi.geometry
            cv2.rectangle(
                img,
                (g["topLeftX"], g["topLeftY"]),
                (g["Bottom_right_X"], g["Bottom_right_Y"]),
                roi.color_bgr, roi.thickness,
            )
        # Circles
        for roi in self.defs[CIRCLE].values():
            g = roi.geometry
            cv2.circle(img, (g["centerX"], g["centerY"]),
                       g["radius"], roi.color_bgr, roi.thickness)
        # Polygons
        for roi in self.defs[POLYGON].values():
            verts = np.asarray(roi.geometry["vertices"], dtype=np.int32)
            cv2.polylines(img, [verts], isClosed=True,
                          color=roi.color_bgr, thickness=roi.thickness)
        return img

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def _load_existing_definitions(self) -> None:
        """Read project_folder/logs/measures/ROI_definitions.h5, filter by
        this video, and populate ``self.defs``. ROIs for other videos
        are stashed in ``self.other_video_rois`` for the
        'apply-from-other-video' action."""
        if not os.path.isfile(self.roi_h5_path):
            return
        try:
            rect_df, circ_df, poly_df = read_roi_data(roi_path=self.roi_h5_path)
        except Exception:
            return
        for df, key in ((rect_df, RECTANGLE), (circ_df, CIRCLE),
                        (poly_df, POLYGON)):
            if df is None or "Video" not in df.columns:
                continue
            for _, row in df.iterrows():
                video = row.get("Video", "")
                if not isinstance(video, str):
                    continue
                roi = self._row_to_definition(row, key)
                if roi is None:
                    continue
                if video == self.video_name:
                    self.defs[key][roi.name] = roi
                else:
                    self.other_video_rois.setdefault(video, {})[roi.name] = roi

    @staticmethod
    def _row_to_definition(row: pd.Series, shape_type: str
                           ) -> Optional[ROIDefinition]:
        try:
            common = dict(
                shape_type=shape_type,
                video=str(row["Video"]),
                name=str(row.get("Name", "")),
                color_name=str(row.get("Color name", "Red")),
                color_bgr=ROILogic._parse_bgr(row.get("Color BGR", "(0, 0, 255)")),
                thickness=int(row.get("Thickness", 2)),
                ear_tag_size=int(row.get("Ear_tag_size", 15)),
            )
            if shape_type == RECTANGLE:
                geom = {
                    "topLeftX": int(row["topLeftX"]),
                    "topLeftY": int(row["topLeftY"]),
                    "Bottom_right_X": int(row["Bottom_right_X"]),
                    "Bottom_right_Y": int(row["Bottom_right_Y"]),
                    "Center_X": int(row.get("Center_X", 0)),
                    "Center_Y": int(row.get("Center_Y", 0)),
                    "width": int(row.get("width", 0)),
                    "height": int(row.get("height", 0)),
                }
            elif shape_type == CIRCLE:
                geom = {
                    "centerX": int(row["centerX"]),
                    "centerY": int(row["centerY"]),
                    "radius": int(row["radius"]),
                    "Center_X": int(row.get("Center_X", row["centerX"])),
                    "Center_Y": int(row.get("Center_Y", row["centerY"])),
                }
            elif shape_type == POLYGON:
                v = row["vertices"]
                if isinstance(v, str):
                    # ast.literal_eval — only Python literals, never
                    # arbitrary code. The legacy SimBA writer dumps
                    # vertex lists via str(...), so this round-trips
                    # correctly while refusing arbitrary expressions.
                    import ast
                    v = ast.literal_eval(v)
                geom = {
                    "vertices": [[int(p[0]), int(p[1])] for p in v],
                    "Center_X": int(row.get("Center_X", 0)),
                    "Center_Y": int(row.get("Center_Y", 0)),
                }
            else:
                return None
            return ROIDefinition(**common, geometry=geom)
        except Exception:
            return None

    @staticmethod
    def _parse_bgr(value: Any) -> Tuple[int, int, int]:
        if isinstance(value, tuple) or isinstance(value, list):
            return (int(value[0]), int(value[1]), int(value[2]))
        if isinstance(value, str):
            v = value.strip()
            if v.startswith("(") and v.endswith(")"):
                parts = [p.strip() for p in v[1:-1].split(",")]
                return (int(parts[0]), int(parts[1]), int(parts[2]))
        return (0, 0, 255)  # fallback red

    def save(self) -> None:
        """Write every ROI for this video back to ROI_definitions.h5,
        preserving any other video's ROIs that are already on disk.

        Two correctness concerns:

        * **Concurrency** — multiple ROIDefinePanel instances may be
          open simultaneously (one per video). Without a lock, two
          panels can race on read-modify-write and lose each other's
          updates. We acquire an exclusive ``fcntl.flock`` on a
          ``.lock`` sidecar file for the whole read-modify-write
          window. Only one process can hold the lock at a time;
          others block briefly.

        * **Atomicity** — write to ``.tmp`` first, then ``os.replace``
          to the final path. Crash mid-write leaves the previous
          good file intact rather than a half-written H5.

        Compatible with the SimBA H5 format read by
        :func:`mufasa.utils.read_write.read_roi_data`."""
        os.makedirs(os.path.dirname(self.roi_h5_path), exist_ok=True)
        lock_path = self.roi_h5_path + ".lock"

        # fcntl is Linux/macOS only. On Windows we'd use msvcrt.locking,
        # but Mufasa 6.0+ is Linux-only so this is fine.
        import fcntl
        with open(lock_path, "w") as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            try:
                self._save_locked()
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    def _save_locked(self) -> None:
        """The actual read-modify-write. Caller holds the lock."""
        # Read existing data so we can preserve other-video rows.
        # This re-read is intentional (and within the lock): the on-
        # disk state may have changed since __init__ if another panel
        # for a different video saved while ours was open.
        rect_existing, circ_existing, poly_existing = (
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        )
        if os.path.isfile(self.roi_h5_path):
            try:
                r, c, p = read_roi_data(roi_path=self.roi_h5_path)
                rect_existing = r if r is not None else pd.DataFrame()
                circ_existing = c if c is not None else pd.DataFrame()
                poly_existing = p if p is not None else pd.DataFrame()
            except Exception:
                pass
        # Drop our video's rows from the existing frames
        for df in (rect_existing, circ_existing, poly_existing):
            if "Video" in df.columns:
                df.drop(df[df["Video"] == self.video_name].index,
                        inplace=True)

        # Build new rows from our in-memory state
        new_rect_rows = [self._rect_row(r) for r in self.defs[RECTANGLE].values()]
        new_circ_rows = [self._circ_row(r) for r in self.defs[CIRCLE].values()]
        new_poly_rows = [self._poly_row(r) for r in self.defs[POLYGON].values()]

        rect_out = pd.concat(
            [rect_existing, pd.DataFrame(new_rect_rows)], ignore_index=True,
        ) if new_rect_rows else rect_existing
        circ_out = pd.concat(
            [circ_existing, pd.DataFrame(new_circ_rows)], ignore_index=True,
        ) if new_circ_rows else circ_existing
        poly_out = pd.concat(
            [poly_existing, pd.DataFrame(new_poly_rows)], ignore_index=True,
        ) if new_poly_rows else poly_existing

        # Write atomically — to a tmp path then rename. Prevents
        # corrupt H5 on power loss / crash mid-write.
        tmp_path = self.roi_h5_path + ".tmp"
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)
        rect_out.to_hdf(tmp_path, key=KEY_RECT, mode="w")
        circ_out.to_hdf(tmp_path, key=KEY_CIRC, mode="a")
        poly_out.to_hdf(tmp_path, key=KEY_POLY, mode="a")
        os.replace(tmp_path, self.roi_h5_path)

    @staticmethod
    def _rect_row(r: ROIDefinition) -> Dict[str, Any]:
        g = r.geometry
        # Compute area in cm² (best-effort; uses px_per_mm if known)
        area_cm = 0.0
        if r.px_per_mm and r.px_per_mm > 0:
            area_cm = (g["width"] / (r.px_per_mm * 10.0)) * \
                      (g["height"] / (r.px_per_mm * 10.0))
        return {
            "Video": r.video, "Shape_type": "Rectangle", "Name": r.name,
            "Color name": r.color_name, "Color BGR": str(r.color_bgr),
            "Thickness": r.thickness,
            "Center_X": g["Center_X"], "Center_Y": g["Center_Y"],
            "topLeftX": g["topLeftX"], "topLeftY": g["topLeftY"],
            "Bottom_right_X": g["Bottom_right_X"],
            "Bottom_right_Y": g["Bottom_right_Y"],
            "width": g["width"], "height": g["height"],
            "width_cm": g["width"] / (r.px_per_mm * 10.0)
                if r.px_per_mm else 0.0,
            "height_cm": g["height"] / (r.px_per_mm * 10.0)
                if r.px_per_mm else 0.0,
            "area_cm": area_cm,
            "Tags": str({"Center tag": (g["Center_X"], g["Center_Y"]),
                         "Top left tag": (g["topLeftX"], g["topLeftY"]),
                         "Bottom right tag": (g["Bottom_right_X"],
                                              g["Bottom_right_Y"])}),
            "Ear_tag_size": r.ear_tag_size,
        }

    @staticmethod
    def _circ_row(r: ROIDefinition) -> Dict[str, Any]:
        g = r.geometry
        area_cm = 0.0
        if r.px_per_mm and r.px_per_mm > 0:
            radius_cm = g["radius"] / (r.px_per_mm * 10.0)
            area_cm = float(np.pi * radius_cm ** 2)
        return {
            "Video": r.video, "Shape_type": "Circle", "Name": r.name,
            "Color name": r.color_name, "Color BGR": str(r.color_bgr),
            "Thickness": r.thickness,
            "centerX": g["centerX"], "centerY": g["centerY"],
            "radius": g["radius"],
            "radius_cm": g["radius"] / (r.px_per_mm * 10.0)
                if r.px_per_mm else 0.0,
            "area_cm": area_cm,
            "Tags": str({"Center tag": (g["centerX"], g["centerY"])}),
            "Ear_tag_size": r.ear_tag_size,
            "Center_X": g["Center_X"], "Center_Y": g["Center_Y"],
        }

    @staticmethod
    def _poly_row(r: ROIDefinition) -> Dict[str, Any]:
        g = r.geometry
        verts = np.asarray(g["vertices"], dtype=np.int32)
        # Polygon area via shoelace
        x = verts[:, 0]; y = verts[:, 1]
        area_px2 = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        area_cm = 0.0
        if r.px_per_mm and r.px_per_mm > 0:
            area_cm = float(area_px2 / ((r.px_per_mm * 10.0) ** 2))
        # Max distance between any two vertices
        max_dist = 0.0
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                d = float(np.linalg.norm(verts[i] - verts[j]))
                if d > max_dist:
                    max_dist = d
        return {
            "Video": r.video, "Shape_type": "Polygon", "Name": r.name,
            "Color name": r.color_name, "Color BGR": str(r.color_bgr),
            "Thickness": r.thickness,
            "Center_X": g["Center_X"], "Center_Y": g["Center_Y"],
            "vertices": str(g["vertices"]),
            "center": str((g["Center_X"], g["Center_Y"])),
            "area": float(area_px2),
            "max_vertice_distance": float(max_dist),
            "area_cm": area_cm,
            "Tags": str({"Center tag": (g["Center_X"], g["Center_Y"])}),
            "Ear_tag_size": r.ear_tag_size,
        }


__all__ = ["ROILogic", "ROIDefinition", "RECTANGLE", "CIRCLE", "POLYGON"]
