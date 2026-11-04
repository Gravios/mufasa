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

import copy
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Literal

import cv2
import numpy as np
import pandas as pd

from mufasa.roi_tools.roi_store import (
    Roi,
    RoiFileError,
    color_name,
    load_rois,
    update_rois,
)
from mufasa.utils.checks import check_file_exist_and_readable
from mufasa.utils.read_write import (
    get_fn_ext,
    get_video_meta_data,
    read_frm_of_video,
)

# Shape-type constants. Mirror the values used by the legacy code so
# in-memory dicts stay compatible with read_roi_data() / write_df.
RECTANGLE = "rectangle"
CIRCLE    = "circle"
POLYGON   = "polygon"


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
    color_bgr: tuple[int, int, int]
    thickness: int
    ear_tag_size: int
    # Geometry — meaning depends on shape_type:
    #   rectangle: top-left + bottom-right + width + height
    #   circle:    center + radius
    #   polygon:   list of (x, y) vertices
    geometry: dict[str, Any] = field(default_factory=dict)
    # Computed at save time
    px_per_mm: float | None = None


class ROILogic:
    """Plain-Python ROI state for one video.

    Owns the active frame, the dict of currently-defined ROIs, and
    persistence. UI layers (Qt panel) wire their controls to this
    class's methods.

    Lifecycle:

    1. ``__init__(config_path, video_path)`` reads the project's
       existing rois.json (if any), filters by this video
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

        # Patch 122ai: route through project_paths_from_config so
        # both v1 (project.toml) and legacy (project.toml)
        # projects resolve cleanly. Prior to 122ai this block read
        # '[General settings].project_path' directly via
        # configparser, which silently parses zero sections out of
        # a TOML file then crashes with MissingSectionHeaderError
        # when the user clicks any video in the ROI Definitions
        # panel. Same bug shape that 122ab fixed for frame_labeller
        # and 122ag fixed for clip_review / roi_define_panel /
        # VideoInfoForm — this file was outside the 122ah audit
        # scope (mufasa/ui_qt/) and got missed. 122ai extends the
        # audit guard to cover mufasa/roi_tools/ and adjacent
        # backend modules so this family of bug stops surfacing.
        from mufasa.project_layout import project_paths_from_config
        try:
            paths = project_paths_from_config(config_path)
            self.project_path = paths["project_root"]
            self.roi_path = paths["roi_definitions_path"]
            video_info_csv = paths["video_info_path"]
        except Exception:
            # Defensive fallback for malformed configs. Same shape
            # as the VideoInfoForm fallback in 122ag — preserves
            # the pre-fix behaviour rather than crashing the dialog.
            self.project_path = os.path.dirname(config_path)
            self.roi_path = os.path.join(
                self.project_path, "logs", "measures", "rois.json",
            )
            video_info_csv = os.path.join(
                self.project_path, "logs", "video_info.csv",
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
        # Patch 122ai: video_info_csv path now resolved via the layout
        # helper above (was hardcoded '<project>/logs/video_info.csv',
        # wrong for v1 where the canonical path is
        # '<root>/sources/video_info.csv').
        self.px_per_mm = 1.0
        try:
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
        self._frame_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cur_frame: np.ndarray | None = self._read_frame(0)

        # ROI state — defs[shape_type] = {name: ROIDefinition}
        self.defs: dict[str, dict[str, ROIDefinition]] = {
            RECTANGLE: {}, CIRCLE: {}, POLYGON: {},
        }
        # Other-video ROIs (read-only, for the "apply from other video"
        # action). Keys are video names.
        self.other_video_rois: dict[str, dict[str, Any]] = {}

        self._load_existing_definitions()

    # ------------------------------------------------------------------ #
    # Frame buffer
    # ------------------------------------------------------------------ #
    def _read_frame(self, idx: int) -> np.ndarray | None:
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

    @property
    def current_frame(self) -> np.ndarray | None:
        """The currently-displayed frame as a raw BGR ndarray (no
        ROI overlays). Returns None if the frame couldn't be read.

        Use this when the caller wants to render ROIs themselves
        (e.g. the Qt canvas does its own painting). For the
        OpenCV-rendered version with ROIs already burned in, see
        :meth:`rendered_frame`."""
        return self._cur_frame

    # ------------------------------------------------------------------ #
    # ROI state
    # ------------------------------------------------------------------ #
    @property
    def all_roi_names(self) -> list[str]:
        out = []
        for d in self.defs.values():
            out.extend(d.keys())
        return out

    def has_roi(self, name: str) -> bool:
        return any(name in d for d in self.defs.values())

    def get_roi(self, name: str) -> ROIDefinition | None:
        for d in self.defs.values():
            if name in d:
                return d[name]
        return None

    def add_rectangle(self, name: str, top_left: tuple[int, int],
                      bottom_right: tuple[int, int],
                      color_name: str, bgr: tuple[int, int, int],
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

    def add_circle(self, name: str, center: tuple[int, int], radius: int,
                   color_name: str, bgr: tuple[int, int, int],
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

    def add_polygon(self, name: str, vertices: list[tuple[int, int]],
                    color_name: str, bgr: tuple[int, int, int],
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

    # ------------------------------------------------------------------ #
    # Patch 122dm — geometry updates for drag-to-adjust (Proposal 2).
    # Each method preserves the existing ROIDefinition's color /
    # thickness / ear_tag_size metadata and writes the new geometry
    # using the SAME key conventions as the matching add_* method.
    # This keeps the H5 serializer and the painter's legacy-form
    # readers (which use topLeftX / Bottom_right_X / centerX etc.)
    # backward-compatible. Returns True on success, False if the
    # named ROI doesn't exist or is of the wrong kind.
    # ------------------------------------------------------------------ #
    def update_rectangle_geometry(
        self, name: str,
        top_left: tuple[int, int],
        bottom_right: tuple[int, int],
    ) -> bool:
        if name not in self.defs[RECTANGLE]:
            return False
        x1, y1 = int(top_left[0]), int(top_left[1])
        x2, y2 = int(bottom_right[0]), int(bottom_right[1])
        w = abs(x2 - x1); h = abs(y2 - y1)
        cx = (x1 + x2) // 2; cy = (y1 + y2) // 2
        self.defs[RECTANGLE][name].geometry = {
            "topLeftX": x1, "topLeftY": y1,
            "Bottom_right_X": x2, "Bottom_right_Y": y2,
            "Center_X": cx, "Center_Y": cy,
            "width": w, "height": h,
        }
        return True

    def update_circle_geometry(
        self, name: str,
        center: tuple[int, int],
        radius: int,
    ) -> bool:
        if name not in self.defs[CIRCLE]:
            return False
        cx, cy = int(center[0]), int(center[1])
        self.defs[CIRCLE][name].geometry = {
            "centerX": cx, "centerY": cy, "radius": int(radius),
            "Center_X": cx, "Center_Y": cy,
        }
        return True

    def update_polygon_geometry(
        self, name: str,
        vertices: list[tuple[int, int]],
    ) -> bool:
        if name not in self.defs[POLYGON]:
            return False
        if len(vertices) < 3:
            return False
        verts = np.asarray(vertices, dtype=np.int32)
        center = verts.mean(axis=0).astype(int)
        self.defs[POLYGON][name].geometry = {
            "vertices": verts.tolist(),
            "Center_X": int(center[0]), "Center_Y": int(center[1]),
        }
        return True

    def update_style(
        self, name: str, *,
        color_name: str | None = None,
        bgr: tuple[int, int, int] | None = None,
        thickness: int | None = None,
        ear_tag_size: int | None = None,
    ) -> bool:
        """Change an ROI's colour / line thickness / tag size in place
        (patch 122ij — the redesigned panel edits these per row).
        Returns False if no ROI has that name."""
        roi = self.get_roi(name)
        if roi is None:
            return False
        if color_name is not None:
            roi.color_name = str(color_name)
        if bgr is not None:
            roi.color_bgr = tuple(int(c) for c in bgr)
        if thickness is not None:
            roi.thickness = max(1, int(thickness))
        if ear_tag_size is not None:
            roi.ear_tag_size = max(1, int(ear_tag_size))
        return True

    def snapshot(self) -> dict[str, dict[str, ROIDefinition]]:
        """Deep copy of this video's ROI definitions, for undo."""
        return copy.deepcopy(self.defs)

    def restore(self, snap: dict[str, dict[str, ROIDefinition]]) -> None:
        """Replace this video's ROI definitions with a :meth:`snapshot`."""
        self.defs = copy.deepcopy(snap)

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
                      offset: tuple[int, int] = (20, 20)) -> None:
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
    def rendered_frame(self) -> np.ndarray | None:
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
        """Load this video's ROIs from ``rois.json`` (patch 122il); keep the
        other videos' in ``self.other_video_rois``. An unreadable file is
        recorded in ``self.load_error`` and :meth:`save` refuses to touch
        it."""
        self.load_error: str | None = None
        try:
            store = load_rois(self.roi_path)
        except RoiFileError as exc:
            self.load_error = str(exc)
            return
        for video, vr in store.videos.items():
            for roi in vr.rois:
                d = _definition_from_roi(video, roi, self.px_per_mm)
                if video == self.video_name:
                    self.defs[roi.type][roi.name] = d
                else:
                    self.other_video_rois.setdefault(video, {})[roi.name] = d

    def to_rois(self) -> list[Roi]:
        """This video's ROIs as :class:`~mufasa.roi_tools.roi_store.Roi`."""
        return [_roi_from_definition(r)
                for kind in (RECTANGLE, CIRCLE, POLYGON)
                for r in self.defs[kind].values()]

    def save(self) -> None:
        """Write this video's ROIs to ``rois.json``.

        A locked read-modify-write (:func:`update_rois`): the file is
        re-read inside the lock and only this video's entry is replaced, so
        other videos' ROIs — including ones saved by another editor since
        this one was opened — are never lost. An unreadable file raises and
        is left as it is."""
        rois = self.to_rois()
        size = (self.width, self.height)
        update_rois(self.roi_path,
                    lambda store: store.set_video(self.video_name, rois, size))


def _definition_from_roi(video: str, roi: Roi,
                         px_per_mm: float | None) -> ROIDefinition:
    g = roi.geometry
    if roi.type == RECTANGLE:
        x1, y1 = g["x"], g["y"]
        x2, y2 = x1 + g["w"], y1 + g["h"]
        geom = {"topLeftX": x1, "topLeftY": y1, "Bottom_right_X": x2,
                "Bottom_right_Y": y2, "Center_X": (x1 + x2) // 2,
                "Center_Y": (y1 + y2) // 2, "width": g["w"], "height": g["h"]}
    elif roi.type == CIRCLE:
        cx, cy = g["center"]
        geom = {"centerX": cx, "centerY": cy, "radius": g["radius"],
                "Center_X": cx, "Center_Y": cy}
    else:
        cx, cy = roi.center
        geom = {"vertices": [list(v) for v in g["vertices"]],
                "Center_X": cx, "Center_Y": cy}
    return ROIDefinition(
        shape_type=roi.type, video=video, name=roi.name,
        color_name=color_name(roi.color_bgr), color_bgr=tuple(roi.color_bgr),
        thickness=roi.line_width, ear_tag_size=roi.tag_size,
        geometry=geom, px_per_mm=px_per_mm)


def _roi_from_definition(d: ROIDefinition) -> Roi:
    g = d.geometry
    if d.shape_type == RECTANGLE:
        x1, x2 = sorted((g["topLeftX"], g["Bottom_right_X"]))
        y1, y2 = sorted((g["topLeftY"], g["Bottom_right_Y"]))
        geom = {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}
    elif d.shape_type == CIRCLE:
        geom = {"center": [g["centerX"], g["centerY"]], "radius": g["radius"]}
    else:
        geom = {"vertices": g["vertices"]}
    return Roi(d.name, d.shape_type, geom, d.color_bgr, d.thickness,
               d.ear_tag_size)


__all__ = ["ROILogic", "ROIDefinition", "RECTANGLE", "CIRCLE", "POLYGON"]
