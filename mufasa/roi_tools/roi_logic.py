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
            self.roi_h5_path = paths["roi_definitions_path"]
            video_info_csv = paths["video_info_path"]
        except Exception:
            # Defensive fallback for malformed configs. Same shape
            # as the VideoInfoForm fallback in 122ag — preserves
            # the pre-fix behaviour rather than crashing the dialog.
            self.project_path = os.path.dirname(config_path)
            self.roi_h5_path = os.path.join(
                self.project_path, "logs", "measures",
                "ROI_definitions.h5",
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
        """Read ROI_definitions.h5, keep this video's ROIs in ``self.defs``
        and stash the others in ``self.other_video_rois``.

        Patch 122ik: read raw instead of through :func:`read_roi_data`,
        whose NaN-dropping hid rows from mixed-format files; each row is
        normalised with :func:`_to_native_row` first. An unreadable file is
        recorded in ``self.load_error`` (and :meth:`save` refuses to
        overwrite it)."""
        self.load_error: str | None = None
        if not os.path.isfile(self.roi_h5_path):
            return
        try:
            with pd.HDFStore(self.roi_h5_path, mode="r") as store:
                keys = {k.lstrip("/") for k in store}
                frames = {key: (store[key] if key in keys else pd.DataFrame())
                          for key in (KEY_RECT, KEY_CIRC, KEY_POLY)}
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            return
        for key, kind in ((KEY_RECT, RECTANGLE), (KEY_CIRC, CIRCLE),
                          (KEY_POLY, POLYGON)):
            df = frames[key]
            if df is None or "Video" not in df.columns:
                continue
            for _, row in df.iterrows():
                video = row.get("Video", "")
                if not isinstance(video, str):
                    continue
                try:
                    native = pd.Series(_to_native_row(kind, dict(row)))
                except Exception:
                    native = row
                roi = self._row_to_definition(native, kind)
                if roi is None:
                    continue
                if video == self.video_name:
                    self.defs[kind][roi.name] = roi
                else:
                    self.other_video_rois.setdefault(video, {})[roi.name] = roi

    @staticmethod
    def _row_to_definition(row: pd.Series, shape_type: str
                           ) -> ROIDefinition | None:
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
                    # 122ik: a mixed file can hold NaN here; the centre is
                    # authoritative anyway.
                    "Center_X": int(row["centerX"]),
                    "Center_Y": int(row["centerY"]),
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
    def _parse_bgr(value: Any) -> tuple[int, int, int]:
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
        """The actual read-modify-write. Caller holds the lock.

        Patch 122ik — this used to lose other videos' ROIs:

        * The existing file was read with :func:`read_roi_data`, which
          drops every rectangle / circle row containing a NaN, and any read
          failure was swallowed (``except Exception: pass``) so the save
          went on to write a file holding only this video.
        * Rows were written in a private format (``Color BGR`` / ``Tags`` /
          ``vertices`` as strings, circles with extra ``Center_X`` /
          ``Center_Y`` columns). Concatenating them with SimBA-format rows
          left NaN in the columns one side lacked, and the next read
          silently dropped those rows — saving one circle on video A
          deleted video B's circles. String ``vertices`` / ``Tags`` also
          broke ROI analysis and visualisation (``vertices.astype``,
          ``Tags.items()``).

        Now: the file is read raw (no NaN dropping) and an unreadable file
        aborts the save; every row — ours and the existing ones — is written
        in SimBA's native shape (tuple colour, dict tags, ndarray vertices,
        SimBA's column set); missing standard values are filled rather than
        left NaN; and a non-standard column is kept only if no row lacks it.
        """
        existing = {KEY_RECT: pd.DataFrame(), KEY_CIRC: pd.DataFrame(),
                    KEY_POLY: pd.DataFrame()}
        if os.path.isfile(self.roi_h5_path):
            try:
                with pd.HDFStore(self.roi_h5_path, mode="r") as store:
                    keys = {k.lstrip("/") for k in store}
                    for key in existing:
                        if key in keys:
                            existing[key] = store[key]
            except Exception as exc:
                raise RuntimeError(
                    f"Could not read the existing ROI file "
                    f"{self.roi_h5_path} ({type(exc).__name__}: {exc}). "
                    f"Not saving, so the ROIs of other videos in it are not "
                    f"overwritten. Back the file up and check it.") from exc

        new_rows = {
            KEY_RECT: [self._rect_row(r) for r in self.defs[RECTANGLE].values()],
            KEY_CIRC: [self._circ_row(r) for r in self.defs[CIRCLE].values()],
            KEY_POLY: [self._poly_row(r) for r in self.defs[POLYGON].values()],
        }
        out = {}
        for key, kind in ((KEY_RECT, RECTANGLE), (KEY_CIRC, CIRCLE),
                          (KEY_POLY, POLYGON)):
            df = existing[key]
            if "Video" in df.columns:
                df = df[df["Video"].astype(str) != self.video_name]
            others = [_to_native_row(kind, dict(row)) for _, row in df.iterrows()]
            out[key] = _conform_table(kind, others + new_rows[key])

        # Write atomically — to a tmp path then rename. Prevents
        # corrupt H5 on power loss / crash mid-write.
        tmp_path = self.roi_h5_path + ".tmp"
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)
        out[KEY_RECT].to_hdf(tmp_path, key=KEY_RECT, mode="w")
        out[KEY_CIRC].to_hdf(tmp_path, key=KEY_CIRC, mode="a")
        out[KEY_POLY].to_hdf(tmp_path, key=KEY_POLY, mode="a")
        os.replace(tmp_path, self.roi_h5_path)

    @staticmethod
    def _rect_row(r: ROIDefinition) -> dict[str, Any]:
        g = r.geometry
        return _to_native_row(RECTANGLE, {
            "Video": r.video, "Shape_type": "Rectangle", "Name": r.name,
            "Color name": r.color_name, "Color BGR": r.color_bgr,
            "Thickness": r.thickness,
            "topLeftX": g["topLeftX"], "topLeftY": g["topLeftY"],
            "Bottom_right_X": g["Bottom_right_X"],
            "Bottom_right_Y": g["Bottom_right_Y"],
            "Ear_tag_size": r.ear_tag_size,
        }, px_per_mm=r.px_per_mm)

    @staticmethod
    def _circ_row(r: ROIDefinition) -> dict[str, Any]:
        g = r.geometry
        return _to_native_row(CIRCLE, {
            "Video": r.video, "Shape_type": "Circle", "Name": r.name,
            "Color name": r.color_name, "Color BGR": r.color_bgr,
            "Thickness": r.thickness,
            "centerX": g["centerX"], "centerY": g["centerY"],
            "radius": g["radius"], "Ear_tag_size": r.ear_tag_size,
        }, px_per_mm=r.px_per_mm)

    @staticmethod
    def _poly_row(r: ROIDefinition) -> dict[str, Any]:
        return _to_native_row(POLYGON, {
            "Video": r.video, "Shape_type": "Polygon", "Name": r.name,
            "Color name": r.color_name, "Color BGR": r.color_bgr,
            "Thickness": r.thickness,
            "vertices": r.geometry["vertices"],
            "Ear_tag_size": r.ear_tag_size,
        }, px_per_mm=r.px_per_mm)


# --------------------------------------------------------------------------- #
# Patch 122ik — SimBA-native ROI rows.
# --------------------------------------------------------------------------- #
_HEADERS = {
    RECTANGLE: ['Video', 'Shape_type', 'Name', 'Color name', 'Color BGR',
                'Thickness', 'Center_X', 'Center_Y', 'topLeftX', 'topLeftY',
                'Bottom_right_X', 'Bottom_right_Y', 'width', 'height',
                'width_cm', 'height_cm', 'area_cm', 'Tags', 'Ear_tag_size'],
    CIRCLE: ['Video', 'Shape_type', 'Name', 'Color name', 'Color BGR',
             'Thickness', 'centerX', 'centerY', 'radius', 'radius_cm',
             'area_cm', 'Tags', 'Ear_tag_size'],
    POLYGON: ['Video', 'Shape_type', 'Name', 'Color name', 'Color BGR',
              'Thickness', 'Center_X', 'Center_Y', 'vertices', 'center',
              'area', 'max_vertice_distance', 'area_cm', 'Tags',
              'Ear_tag_size'],
}
_SHAPE_TYPE = {RECTANGLE: "Rectangle", CIRCLE: "Circle", POLYGON: "Polygon"}


def _missing(v: Any) -> bool:
    if v is None:
        return True
    try:
        return bool(pd.isna(v)) if np.ndim(v) == 0 else False
    except (TypeError, ValueError):
        return False


def _literal(v: Any) -> Any:
    """Parse the string form older Mufasa versions wrote; else return v."""
    if isinstance(v, str):
        import ast
        try:
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return v
    return v


def _to_native_row(kind: str, row: dict[str, Any],
                   px_per_mm: float | None = None) -> dict[str, Any]:
    """Return ``row`` in SimBA's native shape for ``kind``.

    Converts string-encoded ``Color BGR`` / ``Tags`` / ``vertices`` /
    ``center`` back to tuple / dict / ndarray / tuple, and fills every
    standard column that is missing or NaN from the geometry (or a neutral
    default), so no row ever carries a NaN that a reader would drop.
    Derived metric columns are recomputed only when absent.
    """
    r = dict(row)
    r["Shape_type"] = _SHAPE_TYPE[kind]
    for col in ("Color BGR", "Tags", "vertices", "center"):
        if col in r and not _missing(r[col]):
            r[col] = _literal(r[col])
    bgr = r.get("Color BGR")
    r["Color BGR"] = (tuple(int(c) for c in bgr)
                      if isinstance(bgr, (tuple, list, np.ndarray)) and len(bgr) == 3
                      else (0, 0, 255))
    defaults = {"Color name": "Red", "Thickness": 3, "Ear_tag_size": 15}
    for col, val in defaults.items():
        if _missing(r.get(col)):
            r[col] = val
    r["Thickness"] = int(r["Thickness"])
    r["Ear_tag_size"] = int(r["Ear_tag_size"])
    mm = (px_per_mm * 10.0) if px_per_mm else None

    def need(col: str) -> bool:
        return _missing(r.get(col))

    if kind == RECTANGLE:
        x1, y1 = int(r["topLeftX"]), int(r["topLeftY"])
        x2, y2 = int(r["Bottom_right_X"]), int(r["Bottom_right_Y"])
        w, h = abs(x2 - x1), abs(y2 - y1)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        r.update(topLeftX=x1, topLeftY=y1, Bottom_right_X=x2, Bottom_right_Y=y2)
        for col, val in (("width", w), ("height", h),
                         ("Center_X", cx), ("Center_Y", cy)):
            if need(col):
                r[col] = val
        if need("width_cm"):
            r["width_cm"] = w / mm if mm else 0.0
        if need("height_cm"):
            r["height_cm"] = h / mm if mm else 0.0
        if need("area_cm"):
            r["area_cm"] = (w / mm) * (h / mm) if mm else 0.0
        if not isinstance(r.get("Tags"), dict):
            tags = {
                "Center tag": (cx, cy), "Top left tag": (x1, y1),
                "Bottom right tag": (x2, y2), "Top right tag": (x2, y1),
                "Bottom left tag": (x1, y2), "Top tag": (cx, y1),
                "Right tag": (x2, cy), "Left tag": (x1, cy),
                "Bottom tag": (cx, y2),
            }
            r["Tags"] = tags
    elif kind == CIRCLE:
        cx, cy, rad = int(r["centerX"]), int(r["centerY"]), int(r["radius"])
        r.update(centerX=cx, centerY=cy, radius=rad)
        if need("radius_cm"):
            r["radius_cm"] = rad / mm if mm else 0.0
        if need("area_cm"):
            r["area_cm"] = float(np.pi * (rad / mm) ** 2) if mm else 0.0
        if not isinstance(r.get("Tags"), dict):
            r["Tags"] = {"Center tag": (cx, cy), "Border tag": (cx - rad, cy)}
    elif kind == POLYGON:
        verts = np.asarray(r["vertices"], dtype=np.int32).reshape(-1, 2)
        r["vertices"] = verts
        cx, cy = (int(v) for v in verts.mean(axis=0))
        for col, val in (("Center_X", cx), ("Center_Y", cy)):
            if need(col):
                r[col] = val
        if not isinstance(r.get("center"), (tuple, list, np.ndarray)):
            r["center"] = (int(r["Center_X"]), int(r["Center_Y"]))
        x, y = verts[:, 0].astype(float), verts[:, 1].astype(float)
        area_px2 = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        if need("area"):
            r["area"] = float(area_px2)
        if need("max_vertice_distance"):
            d = verts[:, None, :].astype(float) - verts[None, :, :]
            r["max_vertice_distance"] = int(np.sqrt((d ** 2).sum(-1)).max())
        if need("area_cm"):
            r["area_cm"] = float(area_px2 / mm ** 2) if mm else 0.0
        if not isinstance(r.get("Tags"), dict):
            tags = {f"Tag_{i}": (int(v[0]), int(v[1]))
                    for i, v in enumerate(verts)}
            tags["Center_tag"] = (int(r["Center_X"]), int(r["Center_Y"]))
            r["Tags"] = tags
    return r


def _conform_table(kind: str, rows: list[dict[str, Any]]) -> pd.DataFrame:
    """One H5 table: SimBA's columns first, then any extra column that every
    row has a value for. An extra column some rows lack is dropped rather
    than written with NaN (readers drop NaN rows)."""
    headers = _HEADERS[kind]
    if not rows:
        return pd.DataFrame(columns=headers)
    extra: list[str] = []
    for row in rows:
        for c in row:
            if c not in headers and c not in extra:
                extra.append(c)
    extra = [c for c in extra
             if all(c in row and not _missing(row[c]) for row in rows)]
    cols = headers + extra
    return pd.DataFrame([{c: row.get(c) for c in cols} for row in rows],
                        columns=cols)


__all__ = ["ROILogic", "ROIDefinition", "RECTANGLE", "CIRCLE", "POLYGON"]
