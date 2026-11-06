"""
mufasa.roi_tools.roi_store
==========================

The project's ROI definitions, stored as one JSON file (patch 122il).

File
----

``<project>/logs/measures/rois.json``::

    {
      "format": "mufasa-rois",
      "version": 1,
      "videos": {
        "K7_cam_A10-0-051": {
          "frame_size": [640, 360],
          "rois": [
            {"name": "arena", "type": "rectangle",
             "x": 40, "y": 20, "w": 560, "h": 320,
             "color": "#ff0000", "line_width": 3, "tag_size": 8},
            {"name": "cup", "type": "circle",
             "center": [560, 280], "radius": 40,
             "color": "#0000ff", "line_width": 3, "tag_size": 8},
            {"name": "nest", "type": "polygon",
             "vertices": [[30, 300], [90, 300], [100, 350], [30, 340]],
             "color": "#00ff00", "line_width": 3, "tag_size": 8}
          ]
        }
      }
    }

Only geometry and style are stored, in frame pixels of the video named by the
key; ``frame_size`` records the frame the ROIs were drawn on. ROIs keep their
drawing order. Everything derived — centres, areas, cm sizes, tag points — is
computed when read, from the current calibration, so it can never go stale.

Replaces the SimBA ``ROI_definitions.h5`` (three pickled HDF5 tables) with no
compatibility layer: that file is no longer read or written.

API
---

* :func:`load_rois` / :func:`save_rois` — the file <-> :class:`RoiStore`.
* :func:`update_rois` — locked read-modify-write; use it for every change so
  concurrent editors never drop each other's work.
* :func:`rois_to_frames` — the three in-memory tables (rectangles, circles,
  polygons) the analysis / plotting code consumes, with all derived columns.
* :func:`frames_to_store` — the reverse, for tools that edit those tables.
"""
from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

FORMAT = "mufasa-rois"
VERSION = 1
ROI_FILE_NAME = "rois.json"

RECTANGLE = "rectangle"
CIRCLE = "circle"
POLYGON = "polygon"
ROI_TYPES = (RECTANGLE, CIRCLE, POLYGON)

# Keys of the analysis-facing table dict (ConfigReader.roi_dict).
FRAME_KEYS = {RECTANGLE: "rectangles", CIRCLE: "circleDf", POLYGON: "polygons"}
# Lower-case, as SimBA's own writers produce and the feature / ROI-classifier
# kernels compare against (ROI_SETTINGS.*.value). Capitalised values made the
# "body-parts inside ROIs" feature silently produce nothing (patch 122io).
_SHAPE_TYPE = {RECTANGLE: "rectangle", CIRCLE: "circle", POLYGON: "polygon"}

# Named colours offered by the editor (BGR).
COLORS: list[tuple[str, tuple[int, int, int]]] = [
    ("Red",     (0, 0, 255)),
    ("Green",   (0, 255, 0)),
    ("Blue",    (255, 0, 0)),
    ("Yellow",  (0, 255, 255)),
    ("Cyan",    (255, 255, 0)),
    ("Magenta", (255, 0, 255)),
    ("White",   (255, 255, 255)),
    ("Orange",  (0, 165, 255)),
    ("Pink",    (203, 192, 255)),
]

FRAME_COLUMNS = {
    RECTANGLE: ["Video", "Shape_type", "Name", "Color name", "Color BGR",
                "Thickness", "Center_X", "Center_Y", "topLeftX", "topLeftY",
                "Bottom_right_X", "Bottom_right_Y", "width", "height",
                "width_cm", "height_cm", "area_cm", "Tags", "Ear_tag_size"],
    CIRCLE: ["Video", "Shape_type", "Name", "Color name", "Color BGR",
             "Thickness", "Center_X", "Center_Y", "centerX", "centerY",
             "radius", "radius_cm", "area_cm", "Tags", "Ear_tag_size"],
    POLYGON: ["Video", "Shape_type", "Name", "Color name", "Color BGR",
              "Thickness", "Center_X", "Center_Y", "vertices", "center",
              "area", "max_vertice_distance", "area_cm", "Tags",
              "Ear_tag_size"],
}


class RoiFileError(ValueError):
    """The ROI file exists but is not a valid rois.json."""


# --------------------------------------------------------------------------- #
# Colours
# --------------------------------------------------------------------------- #
def bgr_to_hex(bgr: Iterable[int]) -> str:
    b, g, r = (max(0, min(255, int(c))) for c in bgr)
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_to_bgr(text: str) -> tuple[int, int, int]:
    s = str(text).strip().lstrip("#")
    if len(s) != 6:
        raise ValueError(f"not a #rrggbb colour: {text!r}")
    r, g, b = (int(s[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def color_name(bgr: tuple[int, int, int]) -> str:
    """Palette name of ``bgr``, else its ``#rrggbb``."""
    bgr = tuple(int(c) for c in bgr)
    return next((n for n, c in COLORS if c == bgr), bgr_to_hex(bgr))


def color_bgr(name: str) -> tuple[int, int, int]:
    """BGR of a palette name or ``#rrggbb`` (unknown -> red)."""
    for n, c in COLORS:
        if n.lower() == str(name).strip().lower():
            return c
    try:
        return hex_to_bgr(name)
    except ValueError:
        return (0, 0, 255)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@dataclass
class Roi:
    """One ROI. Geometry, in frame pixels:

    * rectangle: ``{"x", "y", "w", "h"}`` (top-left corner and size)
    * circle:    ``{"center": [x, y], "radius"}``
    * polygon:   ``{"vertices": [[x, y], ...]}`` (>= 3, any number)
    """
    name: str
    type: str
    geometry: dict[str, Any]
    color_bgr: tuple[int, int, int] = (0, 0, 255)
    line_width: int = 3
    tag_size: int = 8

    def __post_init__(self) -> None:
        self.name = str(self.name).strip()
        if not self.name:
            raise ValueError("ROI name must not be empty")
        if self.type not in ROI_TYPES:
            raise ValueError(f"unknown ROI type {self.type!r}")
        self.geometry = _clean_geometry(self.type, self.geometry)
        self.color_bgr = tuple(int(c) for c in self.color_bgr)
        self.line_width = max(1, int(self.line_width))
        self.tag_size = max(1, int(self.tag_size))

    # ---- derived geometry --------------------------------------------- #
    @property
    def center(self) -> tuple[int, int]:
        g = self.geometry
        if self.type == RECTANGLE:
            return (g["x"] + g["w"] // 2, g["y"] + g["h"] // 2)
        if self.type == CIRCLE:
            return (g["center"][0], g["center"][1])
        v = np.asarray(g["vertices"], float)
        return (int(round(v[:, 0].mean())), int(round(v[:, 1].mean())))

    @property
    def area_px(self) -> float:
        g = self.geometry
        if self.type == RECTANGLE:
            return float(g["w"] * g["h"])
        if self.type == CIRCLE:
            return float(math.pi * g["radius"] ** 2)
        v = np.asarray(g["vertices"], float)
        x, y = v[:, 0], v[:, 1]
        return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))

    def translated(self, dx: int, dy: int) -> Roi:
        g = dict(self.geometry)
        if self.type == RECTANGLE:
            g.update(x=g["x"] + dx, y=g["y"] + dy)
        elif self.type == CIRCLE:
            g["center"] = [g["center"][0] + dx, g["center"][1] + dy]
        else:
            g["vertices"] = [[x + dx, y + dy] for x, y in g["vertices"]]
        return Roi(self.name, self.type, g, self.color_bgr,
                   self.line_width, self.tag_size)

    # ---- (de)serialisation -------------------------------------------- #
    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "type": self.type}
        out.update(self.geometry)
        out.update(color=bgr_to_hex(self.color_bgr),
                   line_width=self.line_width, tag_size=self.tag_size)
        return out

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Roi:
        t = d.get("type")
        if t == RECTANGLE:
            geom = {k: d[k] for k in ("x", "y", "w", "h")}
        elif t == CIRCLE:
            geom = {"center": d["center"], "radius": d["radius"]}
        elif t == POLYGON:
            geom = {"vertices": d["vertices"]}
        else:
            raise ValueError(f"unknown ROI type {t!r}")
        return cls(name=d["name"], type=t, geometry=geom,
                   color_bgr=hex_to_bgr(d.get("color", "#ff0000")),
                   line_width=d.get("line_width", 3),
                   tag_size=d.get("tag_size", 8))


def _clean_geometry(kind: str, g: dict[str, Any]) -> dict[str, Any]:
    if kind == RECTANGLE:
        x, y, w, h = (int(round(float(g[k]))) for k in ("x", "y", "w", "h"))
        if w < 0:
            x, w = x + w, -w
        if h < 0:
            y, h = y + h, -h
        if w < 1 or h < 1:
            raise ValueError("rectangle must be at least 1x1 px")
        return {"x": x, "y": y, "w": w, "h": h}
    if kind == CIRCLE:
        c = g["center"]
        r = int(round(float(g["radius"])))
        if r < 1:
            raise ValueError("circle radius must be >= 1 px")
        return {"center": [int(round(float(c[0]))), int(round(float(c[1])))],
                "radius": r}
    verts = [[int(round(float(p[0]))), int(round(float(p[1])))]
             for p in np.asarray(g["vertices"]).reshape(-1, 2)]
    if len({tuple(v) for v in verts}) < 3:
        raise ValueError("polygon needs at least 3 distinct vertices")
    return {"vertices": verts}


@dataclass
class VideoRois:
    rois: list[Roi] = field(default_factory=list)
    frame_size: tuple[int, int] | None = None

    def names(self) -> list[str]:
        return [r.name for r in self.rois]


@dataclass
class RoiStore:
    """All ROIs of a project, keyed by video name (file stem)."""
    videos: dict[str, VideoRois] = field(default_factory=dict)

    def rois(self, video: str) -> list[Roi]:
        v = self.videos.get(video)
        return list(v.rois) if v else []

    def frame_size(self, video: str) -> tuple[int, int] | None:
        v = self.videos.get(video)
        return v.frame_size if v else None

    def set_video(self, video: str, rois: list[Roi],
                  frame_size: tuple[int, int] | None = None) -> None:
        """Replace one video's ROIs (an empty list removes the video)."""
        names = [r.name for r in rois]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate ROI names for {video}: {dupes}")
        if not rois:
            self.videos.pop(video, None)
            return
        if frame_size is None:
            frame_size = self.frame_size(video)
        self.videos[str(video)] = VideoRois(
            list(rois), tuple(int(s) for s in frame_size) if frame_size else None)

    def remove_video(self, video: str) -> None:
        self.videos.pop(video, None)

    def fingerprint(self, video: str) -> str | None:
        """Short content hash of one video's ROIs (names, shapes,
        geometry, order) — None when it has none. Stamped into ROI-feature
        outputs so the Compute Features form can tell when the ROIs they
        were computed from have changed (patch 122ji)."""
        import hashlib
        rois = self.rois(video)
        if not rois:
            return None
        blob = json.dumps([r.to_json() for r in rois], sort_keys=True,
                          separators=(",", ":"))
        return hashlib.sha1(blob.encode()).hexdigest()[:16]

    def videos_with_rois(self) -> list[str]:
        return sorted(v for v, vr in self.videos.items() if vr.rois)

    def copy_video(self, source: str, targets: Iterable[str],
                   frame_sizes: dict[str, tuple[int, int]] | None = None) -> list[str]:
        """Give each target video a copy of ``source``'s ROIs (replacing the
        target's own). With ``frame_sizes``, ROIs are rescaled when a
        target's frame differs from the source's. Returns the targets."""
        src = self.videos.get(source)
        if src is None or not src.rois:
            raise ValueError(f"{source} has no ROIs to copy")
        done = []
        for t in targets:
            if t == source:
                continue
            dst_size = (frame_sizes or {}).get(t) or src.frame_size
            rois = [_rescaled(r, src.frame_size, dst_size) for r in src.rois]
            self.set_video(t, rois, dst_size)
            done.append(t)
        return done

    def copy_roi(self, source: str, name: str, targets: Iterable[str],
                 frame_sizes: dict[str, tuple[int, int]] | None = None) -> list[str]:
        """Give each target video a copy of one ROI of ``source`` (patch
        122jh). A target ROI with the same name is replaced in place; the
        target's other ROIs are kept. Rescaled like :meth:`copy_video`.
        Returns the targets."""
        src = self.videos.get(source)
        roi = next((r for r in (src.rois if src else []) if r.name == name), None)
        if roi is None:
            raise ValueError(f"{source} has no ROI named {name!r}")
        done = []
        for t in targets:
            if t == source:
                continue
            dst_size = (frame_sizes or {}).get(t) or self.frame_size(t) or src.frame_size
            new = _rescaled(roi, src.frame_size, dst_size)
            rois = self.rois(t)
            idx = next((i for i, r in enumerate(rois) if r.name == name), None)
            if idx is None:
                rois.append(new)
            else:
                rois[idx] = new
            self.set_video(t, rois, dst_size)
            done.append(t)
        return done

    # ---- (de)serialisation -------------------------------------------- #
    def to_json(self) -> dict[str, Any]:
        return {
            "format": FORMAT, "version": VERSION,
            "videos": {
                name: {"frame_size": list(vr.frame_size) if vr.frame_size else None,
                       "rois": [r.to_json() for r in vr.rois]}
                for name, vr in sorted(self.videos.items()) if vr.rois
            },
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> RoiStore:
        if not isinstance(d, dict) or d.get("format") != FORMAT:
            raise RoiFileError(f"not a {FORMAT} file")
        if int(d.get("version", 0)) > VERSION:
            raise RoiFileError(
                f"written by a newer Mufasa (format version {d['version']})")
        store = cls()
        for name, vd in (d.get("videos") or {}).items():
            rois = []
            for i, rd in enumerate(vd.get("rois") or []):
                try:
                    rois.append(Roi.from_json(rd))
                except (KeyError, TypeError, ValueError) as exc:
                    raise RoiFileError(
                        f"video {name!r}, ROI #{i + 1}: {exc}") from exc
            fs = vd.get("frame_size")
            store.set_video(name, rois, tuple(fs) if fs else None)
        return store


def _rescaled(roi: Roi, src: tuple[int, int] | None,
              dst: tuple[int, int] | None) -> Roi:
    if not src or not dst or tuple(src) == tuple(dst):
        return Roi(roi.name, roi.type, dict(roi.geometry), roi.color_bgr,
                   roi.line_width, roi.tag_size)
    sx, sy = dst[0] / src[0], dst[1] / src[1]
    g = roi.geometry
    if roi.type == RECTANGLE:
        geom = {"x": g["x"] * sx, "y": g["y"] * sy,
                "w": g["w"] * sx, "h": g["h"] * sy}
    elif roi.type == CIRCLE:
        geom = {"center": [g["center"][0] * sx, g["center"][1] * sy],
                "radius": g["radius"] * (sx + sy) / 2}
    else:
        geom = {"vertices": [[x * sx, y * sy] for x, y in g["vertices"]]}
    return Roi(roi.name, roi.type, geom, roi.color_bgr, roi.line_width,
               roi.tag_size)


# --------------------------------------------------------------------------- #
# File I/O
# --------------------------------------------------------------------------- #
def load_rois(path: str | os.PathLike) -> RoiStore:
    """Read ``rois.json``. A missing file is an empty store; an invalid one
    raises :class:`RoiFileError` (never silently treated as empty)."""
    path = os.fspath(path)
    if not os.path.isfile(path):
        return RoiStore()
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise RoiFileError(f"{path}: {exc}") from exc
    try:
        return RoiStore.from_json(data)
    except RoiFileError as exc:
        raise RoiFileError(f"{path}: {exc}") from exc


def _dumps(store: RoiStore) -> str:
    """JSON with one ROI per line block and vertex lists kept compact."""
    text = json.dumps(store.to_json(), indent=2)
    # Collapse innermost [x, y] pairs onto single lines.
    import re
    text = re.sub(r"\[\s+(-?\d+),\s+(-?\d+)\s+\]", r"[\1, \2]", text)
    return text + "\n"


def save_rois(store: RoiStore, path: str | os.PathLike) -> None:
    """Write atomically (temp file + rename) under an exclusive lock."""
    with _locked(path):
        _write(store, path)


def _write(store: RoiStore, path: str | os.PathLike) -> None:
    path = os.fspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(_dumps(store))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


class _locked:
    """Exclusive advisory lock on ``<path>.lock`` (Linux/macOS)."""

    def __init__(self, path: str | os.PathLike) -> None:
        self.lock_path = os.fspath(path) + ".lock"

    def __enter__(self):
        import fcntl
        os.makedirs(os.path.dirname(self.lock_path) or ".", exist_ok=True)
        self._fh = open(self.lock_path, "w")  # noqa: SIM115
        fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc) -> None:
        import fcntl
        fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        self._fh.close()


def update_rois(path: str | os.PathLike,
                change: Callable[[RoiStore], Any]) -> RoiStore:
    """Locked read-modify-write: load, apply ``change(store)``, save.

    Re-reads the file inside the lock, so another editor's save between our
    load and this call is never lost. An unreadable file raises
    :class:`RoiFileError` and is left untouched."""
    with _locked(path):
        store = load_rois(path)
        change(store)
        _write(store, path)
    return store


# --------------------------------------------------------------------------- #
# Analysis tables
# --------------------------------------------------------------------------- #
def _cm(px: float, px_per_mm: float | None) -> float:
    return px / (px_per_mm * 10.0) if px_per_mm else 0.0


def roi_row(video: str, roi: Roi, px_per_mm: float | None = None) -> dict[str, Any]:
    """One analysis-table row for ``roi`` with every derived column."""
    cx, cy = roi.center
    base = {"Video": video, "Shape_type": _SHAPE_TYPE[roi.type],
            "Name": roi.name, "Color name": color_name(roi.color_bgr),
            "Color BGR": tuple(roi.color_bgr), "Thickness": roi.line_width,
            "Center_X": cx, "Center_Y": cy, "Ear_tag_size": roi.tag_size}
    g = roi.geometry
    area_cm = roi.area_px / (px_per_mm * 10.0) ** 2 if px_per_mm else 0.0
    if roi.type == RECTANGLE:
        x1, y1, w, h = g["x"], g["y"], g["w"], g["h"]
        x2, y2 = x1 + w, y1 + h
        base.update({
            "topLeftX": x1, "topLeftY": y1, "Bottom_right_X": x2,
            "Bottom_right_Y": y2, "width": w, "height": h,
            "width_cm": _cm(w, px_per_mm), "height_cm": _cm(h, px_per_mm),
            "area_cm": area_cm,
            "Tags": {"Center tag": (cx, cy), "Top left tag": (x1, y1),
                     "Bottom right tag": (x2, y2), "Top right tag": (x2, y1),
                     "Bottom left tag": (x1, y2), "Top tag": (cx, y1),
                     "Right tag": (x2, cy), "Left tag": (x1, cy),
                     "Bottom tag": (cx, y2)},
        })
    elif roi.type == CIRCLE:
        r = g["radius"]
        base.update({
            "centerX": cx, "centerY": cy, "radius": r,
            "radius_cm": _cm(r, px_per_mm), "area_cm": area_cm,
            "Tags": {"Center tag": (cx, cy), "Border tag": (cx - r, cy)},
        })
    else:
        v = np.asarray(g["vertices"], dtype=np.int32)
        d = v[:, None, :].astype(float) - v[None, :, :]
        tags = {f"Tag_{i}": (int(p[0]), int(p[1])) for i, p in enumerate(v)}
        tags["Center_tag"] = (cx, cy)
        base.update({
            "vertices": v, "center": (cx, cy), "area": roi.area_px,
            "max_vertice_distance": int(np.sqrt((d ** 2).sum(-1)).max()),
            "area_cm": area_cm, "Tags": tags,
        })
    return base


def rois_to_frames(store: RoiStore,
                   px_per_mm: dict[str, float] | float | None = None,
                   videos: Iterable[str] | None = None,
                   ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(rectangles, circles, polygons) tables for the analysis code.

    ``px_per_mm`` (one value, or per video) fills the cm columns; without it
    they are 0. Tables always have their full column set, even when empty.
    """
    rows: dict[str, list[dict[str, Any]]] = {t: [] for t in ROI_TYPES}
    wanted = set(videos) if videos is not None else None
    for video, vr in sorted(store.videos.items()):
        if wanted is not None and video not in wanted:
            continue
        ppm = px_per_mm.get(video) if isinstance(px_per_mm, dict) else px_per_mm
        for roi in vr.rois:
            rows[roi.type].append(roi_row(video, roi, ppm))
    return tuple(pd.DataFrame(rows[t], columns=FRAME_COLUMNS[t])
                 for t in ROI_TYPES)  # type: ignore[return-value]


def frames_to_store(rectangles: pd.DataFrame | None,
                    circles: pd.DataFrame | None,
                    polygons: pd.DataFrame | None,
                    base: RoiStore | None = None,
                    replace_videos: Iterable[str] | None = None) -> RoiStore:
    """Build a store from analysis-style tables.

    The tables' videos replace those videos in ``base`` (frame sizes kept);
    ``replace_videos`` also clears listed videos absent from the tables.
    Row order within a video follows the tables (rectangles, circles,
    polygons).
    """
    out = RoiStore({k: VideoRois(list(v.rois), v.frame_size)
                    for k, v in (base.videos.items() if base else [])})
    per_video: dict[str, list[Roi]] = {}
    for kind, df in ((RECTANGLE, rectangles), (CIRCLE, circles),
                     (POLYGON, polygons)):
        if df is None or len(df) == 0:
            continue
        for _, row in df.iterrows():
            per_video.setdefault(str(row["Video"]), []).append(
                _roi_from_row(kind, row))
    for video in set(replace_videos or []) - set(per_video):
        out.remove_video(video)
    for video, rois in per_video.items():
        out.set_video(video, rois)
    return out


def write_frames(path: str | os.PathLike,
                 rectangles: pd.DataFrame | None,
                 circles: pd.DataFrame | None,
                 polygons: pd.DataFrame | None,
                 whole_project: bool) -> RoiStore:
    """Save analysis-style tables to ``rois.json`` (locked update).

    ``whole_project=True``: the tables are the complete set — videos missing
    from them are removed. ``False``: only the videos in the tables are
    replaced; every other video is kept. Frame sizes are preserved.
    """
    def change(store: RoiStore) -> None:
        drop = list(store.videos) if whole_project else None
        new = frames_to_store(rectangles, circles, polygons, base=store,
                              replace_videos=drop)
        store.videos = new.videos

    return update_rois(path, change)


def _roi_from_row(kind: str, row: pd.Series) -> Roi:
    bgr = row.get("Color BGR")
    if isinstance(bgr, str):
        import ast
        bgr = ast.literal_eval(bgr)
    if not isinstance(bgr, (tuple, list, np.ndarray)) or len(bgr) != 3:
        bgr = color_bgr(str(row.get("Color name", "Red")))
    style = dict(color_bgr=tuple(int(c) for c in bgr),
                 line_width=int(row.get("Thickness", 3) or 3),
                 tag_size=int(row.get("Ear_tag_size", 8) or 8))
    if kind == RECTANGLE:
        x1, y1 = float(row["topLeftX"]), float(row["topLeftY"])
        x2, y2 = float(row["Bottom_right_X"]), float(row["Bottom_right_Y"])
        geom = {"x": min(x1, x2), "y": min(y1, y2),
                "w": abs(x2 - x1), "h": abs(y2 - y1)}
    elif kind == CIRCLE:
        geom = {"center": [row["centerX"], row["centerY"]],
                "radius": row["radius"]}
    else:
        v = row["vertices"]
        if isinstance(v, str):
            import ast
            v = ast.literal_eval(v)
        geom = {"vertices": np.asarray(v).reshape(-1, 2).tolist()}
    return Roi(str(row["Name"]), kind, geom, **style)


def read_frame_sizes(video_paths: Iterable[str]) -> dict[str, tuple[int, int]]:
    """Frame size of each video (by stem), for rescaling copied ROIs."""
    import cv2
    out = {}
    for p in video_paths:
        cap = cv2.VideoCapture(os.fspath(p))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if w and h:
            out[os.path.splitext(os.path.basename(os.fspath(p)))[0]] = (w, h)
    return out


__all__ = [
    "FORMAT", "VERSION", "ROI_FILE_NAME", "RECTANGLE", "CIRCLE", "POLYGON",
    "ROI_TYPES", "FRAME_KEYS", "FRAME_COLUMNS", "COLORS", "Roi", "VideoRois",
    "RoiStore", "RoiFileError", "load_rois", "save_rois", "update_rois",
    "rois_to_frames", "frames_to_store", "write_frames", "roi_row", "read_frame_sizes",
    "bgr_to_hex", "hex_to_bgr", "color_name", "color_bgr",
]
