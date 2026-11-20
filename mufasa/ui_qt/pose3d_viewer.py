"""
mufasa.ui_qt.pose3d_viewer
==========================

Patch 122ke: a 3-D marker viewer — the reconstruction volume on top, the
video below as a viewport, kept in step (scrub or play the video and the 3-D
pose follows).

The volume is drawn with QPainter (no OpenGL or extra packages): the box
(when known), a floor grid, each marker as a depth-sorted dot, the skeleton,
the markers' shadows on the floor and drop lines that show height, and an
optional trail. Drag to orbit, right/middle-drag to pan, wheel to zoom,
double-click to reset; preset views: perspective, top, front, side, and
"camera" (the recording camera's own view, when the file says where it is).

Input: a parquet/CSV with ``<marker>_x/_y/_z`` columns (any units, mm
expected), optional ``frame`` (video frame of each row; default row =
frame) and ``<marker>_likelihood``. Optional parquet metadata
``mufasa_pose3d`` (JSON) supplies ``box_x/box_y/box_z`` limits, ``edges``,
``units`` and ``camera`` ({height_mm, focal_px, principal_px} for a
straight-down pinhole, or a 3×4 ``P``); with a camera, the reconstruction
can be reprojected onto the video ("Reprojection") to check it.

Patch 122kn: **video in the 3-D view.** With a camera in the file, the
current video frame is drawn on the floor (``z = 0``) of the volume —
mapped through the calibration (image → floor homography, undistorted
first when the camera has lens distortion), then through the view (floor →
screen, a second homography; one projective ``QTransform``). Markers on
the floor sit on their image; raised markers show their height as
parallax. The **Camera** view renders through the calibration itself: the
frame fills the view as recorded and every marker sits on the animal at any
height (zoom and pan keep the alignment; dragging to orbit leaves it).

Patch 122kv: **models and view options.** *View & models* opens a panel.
* **View:** markers on or off and their size, marker names, the skeleton,
  the floor grid, an axes gizmo, perspective or orthographic projection,
  the field of view and the background.
* **Models:** load a body-model rig (``.rig.npz``) or the built-in virtual
  rodent, posed to the 3-D markers of each frame, or a static mesh
  (``.obj``, ``.stl``, ``.ply``; an arena, objects) with its own scale,
  rotation and offset. Each model is drawn solid, transparent, as a
  wireframe or as points, in a colour and opacity of its own.

See :mod:`mufasa.pose3d.scene_models`.

Open from the feature viewer ("3-D…") or ``mufasa-view3d VIDEO [POSE3D] [--model …]``.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF, QTransform
from PySide6.QtWidgets import (
    QCheckBox,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

META_KEY = b"mufasa_pose3d"
PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
           "#f032e6", "#9a6324", "#469990", "#808000", "#000075", "#bfef45",
           "#fabed4", "#dcbeff", "#aaffc3"]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass
class Pose3D:
    names: list[str]
    xyz: np.ndarray                         # (T, M, 3)
    frames: np.ndarray                      # (T,) video frame of each row
    likelihood: np.ndarray | None = None    # (T, M)
    edges: list[tuple[str, str]] = field(default_factory=list)
    box: tuple | None = None                # ((x0,x1),(y0,y1),(z0,z1))
    camera: dict | None = None
    units: str = "mm"
    meta: dict = field(default_factory=dict)

    def row(self, frame: int) -> int | None:
        """Row for a video frame (None when the file has no such frame)."""
        i = int(np.searchsorted(self.frames, frame))
        if i < len(self.frames) and self.frames[i] == frame:
            return i
        return None

    def extent(self) -> tuple:
        if self.box is not None:
            return self.box
        key = (id(self.xyz), self.xyz.shape)                 # 122ky: once, not on every paint
        cached = getattr(self, "_extent_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        out = self._extent()
        object.__setattr__(self, "_extent_cache", (key, out))
        return out

    def _extent(self) -> tuple:
        v = self.xyz.reshape(-1, 3)
        v = v[np.isfinite(v).all(1)]
        if not len(v):
            return ((-1, 1), (-1, 1), (0, 1))
        lo, hi = np.percentile(v, 0.5, 0), np.percentile(v, 99.5, 0)
        return tuple((float(a), float(b)) for a, b in zip(lo, hi))

    def projection_matrix(self) -> np.ndarray | None:
        """3×4 matrix from world to video pixels, if the file says."""
        c = self.camera or {}
        if "P" in c:
            return np.asarray(c["P"], float).reshape(3, 4)
        if {"height_mm", "focal_px", "principal_px"} <= set(c):
            f, (cx, cy), h = float(c["focal_px"]), c["principal_px"], float(c["height_mm"])
            # straight down: u = cx + f x/(h-z), v = cy - f y/(h-z)
            return np.array([[f, 0, -cx, cx * h], [0, -f, -cy, cy * h], [0, 0, -1, h]], float)
        return None

    def camera_position(self) -> np.ndarray | None:
        P = self.projection_matrix()
        if P is None:
            return None
        _, _, vt = np.linalg.svd(P)
        c = vt[-1]
        return c[:3] / c[3] if abs(c[3]) > 1e-12 else None

    def image_size(self) -> tuple[int, int] | None:
        """(width, height) of the video the camera describes, if known."""
        c = self.camera or {}
        w, h = int(c.get("image_width_px") or 0), int(c.get("image_height_px") or 0)
        if w > 0 and h > 0:
            return w, h
        if "principal_px" in c and h > 0:
            return int(round(2 * float(c["principal_px"][0]))), h
        return None

    def floor_homography(self, z: float = 0.0) -> np.ndarray | None:
        """3×3: floor point (x, y, 1) at height ``z`` → video pixel (homogeneous)."""
        P = self.projection_matrix()
        if P is None:
            return None
        return np.c_[P[:, 0], P[:, 1], P[:, 2] * z + P[:, 3]]

    def undistorter(self):
        """``fn(frame_bgr) -> frame_bgr`` removing lens distortion (the
        calibration's ``K`` and ``dist``), or None when there is none."""
        c = self.camera or {}
        dist = np.asarray(c.get("dist") or [], float)
        if "K" not in c or not dist.size or not np.any(np.abs(dist) > 0):
            return None
        import cv2
        K = np.asarray(c["K"], float)
        cache: dict = {}

        def undistort(img: np.ndarray) -> np.ndarray:
            h, w = img.shape[:2]
            if (w, h) not in cache:
                cache[(w, h)] = cv2.initUndistortRectifyMap(K, dist, None, K, (w, h), cv2.CV_16SC2)
            m1, m2 = cache[(w, h)]
            return cv2.remap(img, m1, m2, cv2.INTER_LINEAR)
        return undistort


def load_pose3d(path) -> Pose3D:
    path = Path(path)
    meta: dict = {}
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq
        t = pq.read_table(path)
        raw = (t.schema.metadata or {}).get(META_KEY)
        if raw:
            try:
                meta = json.loads(raw)
            except ValueError:
                meta = {}
        df = t.to_pandas()
    else:
        df = pd.read_csv(path)
    from mufasa.pose3d.io import marker_names
    names = marker_names(df)                                  # 122kz: the shared rule
    if not names:
        raise ValueError(f"{path.name}: no <marker>_x/_y/_z columns")
    xyz = np.stack([df[[n + "_x", n + "_y", n + "_z"]].to_numpy(float) for n in names], 1)
    frames = (df["frame"].to_numpy(int) if "frame" in df.columns
              else np.arange(len(df)))
    order = np.argsort(frames, kind="stable")
    lk = None
    if all(n + "_likelihood" in df.columns for n in names):
        lk = np.stack([df[n + "_likelihood"].to_numpy(float) for n in names], 1)[order]
    edges = [tuple(e) for e in meta.get("edges", []) if e[0] in names and e[1] in names]
    box = None
    if all(k in meta for k in ("box_x", "box_y", "box_z")):
        box = tuple(tuple(float(v) for v in meta[k]) for k in ("box_x", "box_y", "box_z"))
    return Pose3D(names, xyz[order], frames[order], lk, edges, box,
                  meta.get("camera"), meta.get("units", "mm"), meta)


def find_pose3d(video_path: str | None, config_path: str | None = None) -> Path | None:
    """A 3-D pose file for the video: beside it (``<stem>*3d*.parquet``) or
    under the project's ``derived/pose3d/``; the newest wins."""
    if not video_path:
        return None
    v = Path(video_path)
    stem = v.stem
    hits = list(v.parent.glob(f"{stem}*3d*.parquet")) if v.parent.is_dir() else []
    if config_path:
        for d in (Path(config_path).parent / "derived" / "pose3d",
                  Path(config_path).parent / "sources" / "pose3d"):    # + motion capture (122kl)
            if d.is_dir():
                hits += list(d.rglob(f"{stem}*.parquet"))
    hits = [h for h in hits if h.is_file()]
    return max(hits, key=lambda h: h.stat().st_mtime) if hits else None


# --------------------------------------------------------------------------- #
# Volume view
# --------------------------------------------------------------------------- #
VIEWS = {  # azimuth, elevation (degrees)
    "Perspective": (-60.0, 28.0),
    "Top": (-90.0, 89.0),
    "Front": (-90.0, 2.0),
    "Side": (0.0, 2.0),
}


class Volume3DWidget(QWidget):
    """Draws one frame of a :class:`Pose3D` in its volume."""

    def __init__(self, pose: Pose3D, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.pose = pose
        self.setMinimumSize(320, 240)
        self.setMouseTracking(False)
        self.frame = int(pose.frames[0]) if len(pose.frames) else 0
        self.show_box = True
        self.show_shadows = True
        self.show_drops = True
        self.trail = 0                       # frames of trail behind each marker
        self.p_threshold = 0.0
        self.colors = {n: QColor(PALETTE[i % len(PALETTE)]) for i, n in enumerate(pose.names)}
        (x0, x1), (y0, y1), (z0, z1) = pose.extent()
        self._home_target = np.array([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 3])
        self._home_dist = 2.2 * max(x1 - x0, y1 - y0, z1 - z0, 1e-6)
        self.fov = math.radians(40)
        # 122kn: video in the volume; exact recording-camera view
        self.video_img: QImage | None = None
        self.show_video = False
        self.video_opacity = 0.9
        self._P = pose.projection_matrix()
        self._P_sign = float(np.sign(np.linalg.det(self._P[:, :3]))) if self._P is not None else 1.0
        self._undistort = pose.undistorter()
        self.cam_mode = False
        self.cam_zoom = 1.0
        self.cam_pan = np.zeros(2)
        self.show_label = True               # "frame N · …" text (off for rendered animations)
        self.cam_blend = 0.0                 # 122kp: 0 orbit … 1 recording camera (animation moves)
        # 122kv: view options and models
        self.show_markers = True
        self.marker_scale = 1.0
        self.show_names = False
        self.show_skeleton = True
        self.show_grid = True
        self.show_axes = False
        self.ortho = False
        self.background = "theme"           # theme | dark | light | black | white
        self.models: list = []               # scene_models.PosedRig / SceneMesh
        self.sync_fit = False                # fit rigs while painting (offscreen rendering)
        self.on_fitted = None                # called after rigs were posed to the current frame
        self._fit_timer = QTimer(self)
        self._fit_timer.setSingleShot(True)
        self._fit_timer.setInterval(60)
        self._fit_timer.timeout.connect(self._fit_now)
        self.bg_fitting = False              # 122ky: rigs are being posed in the background
        self.gpu = True                      # 122ky: draw models with OpenGL when it works
        self.gpu_note = ""
        self.reset_view()
        self._drag = None

    # 122kp: the whole view as plain values (camera animation keys)
    def view_state(self) -> dict:
        return {"az": float(self.az), "el": float(self.el), "dist": float(self.dist),
                "target": [float(v) for v in self.target], "fov": float(self.fov),
                "cam_mode": bool(self.cam_mode), "cam_zoom": float(self.cam_zoom),
                "cam_pan": [float(v) for v in self.cam_pan]}

    def set_view_state(self, s: dict) -> None:
        self.az, self.el, self.dist = float(s["az"]), float(s["el"]), float(s["dist"])
        self.target = np.asarray(s["target"], float).copy()
        self.fov = float(s.get("fov", self.fov))
        self.cam_mode = bool(s.get("cam_mode")) and self._P is not None and self._image_wh() is not None
        self.cam_zoom = float(s.get("cam_zoom", 1.0))
        self.cam_pan = np.asarray(s.get("cam_pan", (0.0, 0.0)), float).copy()
        ok = self._P is not None and self._image_wh() is not None
        self.cam_blend = float(s.get("cam_blend", 0.0)) if ok and not self.cam_mode else 0.0
        self.update()

    # -- camera ----------------------------------------------------------- #
    def reset_view(self, name: str = "Perspective") -> None:
        self.cam_blend = 0.0
        self.target = self._home_target.copy()
        self.dist = self._home_dist
        self.fov = math.radians(40)
        self.cam_mode = False
        if name == "Camera" and self.pose.camera_position() is not None:
            self.view_from_recording_camera()           # orbit pose to leave from
            self.cam_mode = self._image_wh() is not None
            self.cam_zoom, self.cam_pan = 1.0, np.zeros(2)
            self.update()
            return
        self.az, self.el = (math.radians(a) for a in VIEWS.get(name, VIEWS["Perspective"]))
        self.update()

    def view_from_recording_camera(self) -> None:
        """Look from where the video camera is, along its axis and with its
        field of view, so the volume lines up with the video below."""
        c = self.pose.camera_position()
        P = self.pose.projection_matrix()
        axis = P[2, :3] / np.linalg.norm(P[2, :3])          # viewing direction
        if axis @ (self._home_target - c) < 0:
            axis = -axis
        self.dist = float(abs((self._home_target - c) @ axis)) or 1.0
        self.target = c + axis * self.dist
        self.el = math.asin(max(-1.0, min(1.0, -axis[2])))
        self.el = max(math.radians(-89.9), min(math.radians(89.9), self.el))
        # image x runs along world x for a straight-down camera; keep that
        self.az = math.atan2(-axis[1], -axis[0]) if abs(axis[2]) < 0.99 else math.radians(-90)
        f = (self.pose.camera or {}).get("focal_px")
        if not f and "K" in (self.pose.camera or {}):
            f = float(np.asarray(self.pose.camera["K"], float)[1, 1])
        if not f and P is not None:
            from scipy.linalg import rq
            Kp, _ = rq(P[:, :3])
            f = abs(Kp[1, 1] / Kp[2, 2]) if abs(Kp[2, 2]) > 1e-12 else None
        if f:
            h = float((self.pose.camera or {}).get("image_height_px") or 360)
            self.fov = 2 * math.atan(h / 2 / float(f))
        self.update()

    # -- video ------------------------------------------------------------ #
    def _image_wh(self) -> tuple[int, int] | None:
        if self.video_img is not None and not self.video_img.isNull():
            return self.video_img.width(), self.video_img.height()
        return self.pose.image_size()

    def set_video_frame(self, frame_bgr: np.ndarray | None) -> None:
        """The video frame to draw into the volume (BGR, as read)."""
        if frame_bgr is None:
            self.video_img = None
        else:
            if self._undistort is not None:
                frame_bgr = self._undistort(frame_bgr)
            f = np.ascontiguousarray(frame_bgr)
            h, w = f.shape[:2]
            self.video_img = QImage(f.data, w, h, 3 * w, QImage.Format_BGR888).copy()
        self.update()

    def _cam_fit(self) -> tuple[float, float, float]:
        """Camera view: video pixel → widget pixel (scale k, offset ox, oy)."""
        iw, ih = self._image_wh()
        k = self.cam_zoom * min(self.width() / iw, self.height() / ih)
        ox = self.width() / 2 - k * iw / 2 + self.cam_pan[0]
        oy = self.height() / 2 - k * ih / 2 + self.cam_pan[1]
        return k, ox, oy

    def video_transform(self) -> QTransform | None:
        """Video pixel → widget pixel for the frame drawn on the floor, or
        None when it cannot be drawn in this view (no camera, or the floor
        it covers reaches behind the eye)."""
        if self._P is None or self._image_wh() is None:
            return None
        k, ox, oy = self._cam_fit()
        if self.cam_mode:
            return QTransform(k, 0, 0, 0, k, 0, ox, oy, 1)
        T = self._orbit_video_transform()
        if T is None or self.cam_blend <= 0:
            return T
        # 122kp: moving between the camera view and an orbit view — blend the
        # two image→widget homographies (each scaled to map the image centre
        # with weight 1), continuous at both ends
        A = np.array([[k, 0, ox], [0, k, oy], [0, 0, 1.0]])
        B = np.array([[T.m11(), T.m21(), T.m31()], [T.m12(), T.m22(), T.m32()], [T.m13(), T.m23(), T.m33()]])
        iw, ih = self._image_wh()
        c = np.array([iw / 2, ih / 2, 1.0])
        B = B / (B[2] @ c)
        b = self.cam_blend
        G = b * A + (1 - b) * B
        return QTransform(G[0, 0], G[1, 0], G[2, 0], G[0, 1], G[1, 1], G[2, 1], G[0, 2], G[1, 2], G[2, 2])

    def _orbit_video_transform(self) -> QTransform | None:
        H = self.pose.floor_homography(0.0)
        try:
            Hinv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return None
        eye, right, up, fwd = self._basis()
        sc = 0.5 * min(self.width(), self.height()) / math.tan(self.fov / 2)
        R_ = np.r_[right[:2], -right @ eye]
        U_ = np.r_[up[:2], -up @ eye]
        F_ = np.r_[fwd[:2], -fwd @ eye]
        V = np.vstack([self.width() / 2 * F_ + sc * R_, self.height() / 2 * F_ - sc * U_, F_])
        G = V @ Hinv
        iw, ih = self._image_wh()
        corners = np.array([[0, 0, 1], [iw, 0, 1], [0, ih, 1], [iw, ih, 1]], float)
        fl = corners @ Hinv.T                          # floor points (homogeneous)
        if np.any(np.abs(fl[:, 2]) < 1e-12):
            return None
        X = np.c_[fl[:, :2] / fl[:, 2:], np.zeros(4)]
        depth_cam = (np.c_[X, np.ones(4)] @ self._P.T)[:, 2] * self._P_sign
        depth_view = (X - eye) @ fwd
        if np.any(depth_cam <= 0) or np.any(depth_view <= 1e-6):
            return None
        return QTransform(G[0, 0], G[1, 0], G[2, 0], G[0, 1], G[1, 1], G[2, 1], G[0, 2], G[1, 2], G[2, 2])

    def _draw_video(self, p: QPainter) -> bool:
        if not self.show_video or self.video_img is None:
            return False
        T = self.video_transform()
        if T is None:
            return False
        p.save()
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.setOpacity(self.video_opacity)
        p.setTransform(T)
        p.drawImage(0, 0, self.video_img)
        p.restore()
        return True

    def _basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ce, se, ca, sa = math.cos(self.el), math.sin(self.el), math.cos(self.az), math.sin(self.az)
        fwd = -np.array([ce * ca, ce * sa, se])               # eye -> target
        eye = self.target - fwd * self.dist
        right = np.cross(fwd, [0.0, 0.0, 1.0])
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        return eye, right, up, fwd

    def project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World (N,3) -> widget pixels (N,2) and depth (N,); depth <= 0 is
        behind the eye."""
        if self.cam_mode and self._P is not None:          # through the calibration
            X3 = np.asarray(pts, float).reshape(-1, 3)
            h = np.c_[X3, np.ones(len(X3))] @ self._P.T
            k, ox, oy = self._cam_fit()
            with np.errstate(divide="ignore", invalid="ignore"):
                uv = h[:, :2] / h[:, 2:]
            return np.c_[ox + k * uv[:, 0], oy + k * uv[:, 1]], h[:, 2] * self._P_sign
        eye, right, up, fwd = self._basis()
        d = np.asarray(pts, float).reshape(-1, 3) - eye
        x, y, z = d @ right, d @ up, d @ fwd
        s = 0.5 * min(self.width(), self.height()) / math.tan(self.fov / 2)
        if self.ortho:                                     # 122kv: as the target plane looks in perspective
            so = s / max(self.dist, 1e-9)
            px = self.width() / 2 + so * x
            py = self.height() / 2 - so * y
            return np.c_[px, py], z + 10 * self.dist       # all in front; depth keeps the order
        with np.errstate(divide="ignore", invalid="ignore"):
            px = self.width() / 2 + s * x / z
            py = self.height() / 2 - s * y / z
        if self.cam_blend > 0 and self._P is not None and self._image_wh() is not None:
            # 122kp: on the way to / from the camera view: mix the two
            self.cam_mode = True
            try:
                pc, zc = self.project(pts)
            finally:
                self.cam_mode = False
            b = self.cam_blend
            px, py = b * pc[:, 0] + (1 - b) * px, b * pc[:, 1] + (1 - b) * py
            z = np.where(zc > 0, z, -1.0)
        return np.c_[px, py], z

    # -- data ------------------------------------------------------------- #
    def set_frame(self, frame: int) -> None:
        self.frame = int(frame)
        self.update()

    def current(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        i = self.pose.row(self.frame)
        if i is None:
            return None, None
        X = self.pose.xyz[i].copy()
        if self.pose.likelihood is not None and self.p_threshold > 0:
            X[self.pose.likelihood[i] < self.p_threshold] = np.nan
        return X, i

    # -- painting --------------------------------------------------------- #
    def _line(self, p: QPainter, a, b, pen: QPen) -> None:
        (pa, pb), z = self.project(np.vstack([a, b]))
        if (z > 1e-6).all() and np.isfinite(pa).all() and np.isfinite(pb).all():
            p.setPen(pen)
            p.drawLine(QPointF(*pa), QPointF(*pb))

    def paintEvent(self, _ev) -> None:  # noqa: ANN001, N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        dark = self.palette().window().color().lightness() < 128
        if self.background != "theme":
            dark = self.background in ("dark", "black")
        bg = QColor("#1b1d21") if dark else QColor("#fafafa")
        if self.background in ("black", "white"):
            bg = QColor("#000000" if self.background == "black" else "#ffffff")
        ink = QColor("#c9ccd1") if dark else QColor("#3a3d42")
        faint = QColor("#3a3e45") if dark else QColor("#dcdfe3")
        p.fillRect(self.rect(), bg)
        video_drawn = self._draw_video(p)
        (x0, x1), (y0, y1), (z0, z1) = self.pose.extent()
        # floor grid
        step = _nice_step(max(x1 - x0, y1 - y0) / 8)
        pen = QPen(faint, 1)
        if self.show_grid:
            for gx in np.arange(math.ceil(x0 / step) * step, x1 + 1e-9, step):
                self._line(p, (gx, y0, z0), (gx, y1, z0), pen)
            for gy in np.arange(math.ceil(y0 / step) * step, y1 + 1e-9, step):
                self._line(p, (x0, gy, z0), (x1, gy, z0), pen)
        if self.show_box and self.pose.box is not None:
            pen = QPen(ink, 1.2)
            c = [(x, y, z) for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)]
            for a, b in ((0, 1), (2, 3), (0, 2), (1, 3), (4, 5), (6, 7), (4, 6), (5, 7),
                         (0, 4), (1, 5), (2, 6), (3, 7)):
                self._line(p, c[a], c[b], pen)
        cam = self.pose.camera_position()
        if cam is not None:
            (cp,), cz = self.project(cam[None])
            if cz[0] > 0 and np.isfinite(cp).all():
                p.setPen(QPen(ink, 1))
                p.setBrush(Qt.NoBrush)
                p.drawRect(int(cp[0]) - 5, int(cp[1]) - 4, 10, 8)
                p.drawText(QPointF(cp[0] + 8, cp[1] + 4), "camera")
        model_note = self._draw_models(p)
        X, i = self.current()
        if X is not None:
            names = self.pose.names
            # trails
            if self.trail > 0:
                lo = max(0, i - self.trail)
                seg = self.pose.xyz[lo:i + 1]
                for j, n in enumerate(names):
                    pts, z = self.project(seg[:, j])
                    col = QColor(self.colors[n])
                    for k in range(1, len(pts)):
                        if (z[k - 1] > 0 and z[k] > 0 and np.isfinite(pts[k - 1]).all()
                                and np.isfinite(pts[k]).all()):
                            col.setAlphaF(0.15 + 0.6 * k / len(pts))
                            p.setPen(QPen(col, 1.2))
                            p.drawLine(QPointF(*pts[k - 1]), QPointF(*pts[k]))
            ok = np.isfinite(X).all(1)
            shadow = X.copy()
            shadow[:, 2] = z0
            if self.show_shadows:
                sh, sz = self.project(shadow)
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(0, 0, 0, 70 if not dark else 110))
                for j in np.flatnonzero(ok & (sz > 0)):
                    p.drawEllipse(QPointF(*sh[j]), 3.0, 1.8)
            if self.show_drops:
                pen = QPen(faint.darker(130) if not dark else faint.lighter(140), 1, Qt.DashLine)
                for j in np.flatnonzero(ok):
                    self._line(p, X[j], shadow[j], pen)
            pen = QPen(ink, 2)
            if self.show_skeleton:
                for a, b in self.pose.edges:
                    ja, jb = names.index(a), names.index(b)
                    if ok[ja] and ok[jb]:
                        self._line(p, X[ja], X[jb], pen)
            pts, z = self.project(np.where(ok[:, None], X, 0.0))
            if self.show_markers:
                for j in sorted(np.flatnonzero(ok & (z > 0)), key=lambda k: -z[k]):
                    r = 5.0 if self.ortho else max(3.0, min(9.0, 900.0 / max(z[j], 1e-6) * 4))
                    r *= self.marker_scale
                    p.setPen(QPen(bg, 1))
                    p.setBrush(self.colors[names[j]])
                    p.drawEllipse(QPointF(*pts[j]), r, r)
            if self.show_names:
                p.setPen(ink)
                for j in np.flatnonzero(ok & (z > 0)):
                    p.drawText(QPointF(pts[j][0] + 6, pts[j][1] - 6), names[j])
        # text
        p.setPen(ink)
        f = QFont(p.font())
        f.setPointSizeF(max(8.0, f.pointSizeF() - 1))
        p.setFont(f)
        txt = f"frame {self.frame}"
        if self.cam_mode:
            txt += "  · camera view"
        if self.show_video and self.video_img is not None and not video_drawn:
            txt += "  · video: turn the view to look down at the floor"
        if model_note:
            txt += "  · " + model_note
        if X is None:
            txt += "  · no 3-D data for this frame"
        else:
            z = X[:, 2]
            if np.isfinite(z).any():
                txt += f"  · height {np.nanmin(z):.0f}–{np.nanmax(z):.0f} {self.pose.units}"
        if self.show_label:
            p.drawText(10, 18, txt)
        if self.show_axes:
            self._draw_axes(p)
        p.end()

    # -- 122kv: models and the axes gizmo -------------------------------- #
    def add_model(self, model) -> None:
        self.models.append(model)
        self.update()

    def _fit_now(self) -> None:
        for m in self.models:
            if getattr(m, "kind", "") == "rig" and m.look.visible and not m.cached(self.frame):
                m.fit_frame(self.frame)
        self.update()
        if self.on_fitted is not None:
            self.on_fitted()

    def _gl_view(self, items) -> dict:
        """The projection as ``project`` does it, for the GPU's vertex shader."""
        eye, right, up, fwd = self._basis()
        s = 0.5 * min(self.width(), self.height()) / math.tan(self.fov / 2)
        v = {"eye": eye, "right": right, "up": up, "fwd": fwd, "s": s, "so": s / max(self.dist, 1e-9),
             "dist": self.dist, "blend": 0.0, "psign": self._P_sign}
        cam_ok = self._P is not None and self._image_wh() is not None
        if self.cam_mode and self._P is not None:
            v["mode"] = "camera"
        elif self.ortho:
            v["mode"] = "ortho"
        else:
            v["mode"] = "perspective"
            if self.cam_blend > 0 and cam_ok:
                v["blend"] = float(self.cam_blend)
        if cam_ok or v["mode"] == "camera":
            k, ox, oy = self._cam_fit()
            v.update(P=self._P, k=k, ox=ox, oy=oy)
        # depth range of what is drawn
        Vall = np.vstack([V for V, _, _ in items])
        if v["mode"] == "camera":
            z = (np.c_[Vall, np.ones(len(Vall))] @ self._P[2]) * self._P_sign
        else:
            z = (Vall - eye) @ fwd + (10 * self.dist if v["mode"] == "ortho" else 0.0)
        z = z[np.isfinite(z) & (z > 0)]
        lo, hi = (float(z.min()), float(z.max())) if len(z) else (1.0, 2.0)
        pad = 0.02 * (hi - lo) + 1e-3
        v["depth_range"] = (max(lo - pad, 1e-6), hi + pad)
        return v

    def _draw_models(self, p: QPainter) -> str:
        notes = []
        meshes = []                                          # (V, F, look) of this frame
        for m in self.models:
            if not m.look.visible:
                continue
            frame = self.frame
            if getattr(m, "kind", "") == "gait":
                self._draw_gait(p, m)
                continue
            if getattr(m, "kind", "") == "rig":
                if not m.cached(frame):
                    if self.sync_fit:
                        m.fit_frame(frame)
                    else:
                        near = m.nearest_cached(frame)
                        if not self.bg_fitting:          # 122ky: the background fit will get there
                            self._fit_timer.start()
                        if near is None:
                            notes.append(f"{m.name}: fitting…")
                            continue
                        frame = near
                if m.fit_frame(frame) is None:
                    notes.append(f"{m.name}: too few markers")
                    continue
            V = m.world(frame)
            if V is not None:
                meshes.append((V, m.faces(), m.look))
        if meshes and self.gpu:
            from mufasa.ui_qt.gl_models import shared_renderer
            gl = shared_renderer()
            dpr = max(1.0, float(self.devicePixelRatioF()))
            view = self._gl_view(meshes)
            if dpr != 1.0:
                view = dict(view, s=view["s"] * dpr, so=view["so"] * dpr, k=view.get("k", 0) * dpr,
                            ox=view.get("ox", 0) * dpr, oy=view.get("oy", 0) * dpr)
            img = gl.render(int(self.width() * dpr), int(self.height() * dpr), view, meshes)
            if img is not None:
                img.setDevicePixelRatio(dpr)
                p.drawImage(0, 0, img)
                self.gpu_note = f"GPU: {gl.info}"
                return ", ".join(notes)
            self.gpu_note = f"GPU unavailable ({gl.error}); drawing on the CPU"
        for V, F, look in meshes:
            self._draw_mesh(p, V, F, look)
        return ", ".join(notes)

    def _draw_mesh(self, p: QPainter, V: np.ndarray, F: np.ndarray, look) -> None:
        P, z = self.project(V)
        okv = (z > 1e-6) & np.isfinite(P).all(1)
        base = QColor(look.color)
        if look.style == "points":
            p.setPen(Qt.NoPen)
            c = QColor(base)
            c.setAlphaF(max(0.05, min(1.0, look.opacity)))
            p.setBrush(c)
            for j in np.flatnonzero(okv):
                p.drawEllipse(QPointF(*P[j]), 1.3, 1.3)
            return
        F = F[okv[F].all(1)]
        if not len(F):
            return
        if look.style == "wireframe":
            E = np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
            E = np.unique(E, axis=0)
            c = QColor(base)
            c.setAlphaF(max(0.05, min(1.0, look.opacity)))
            p.setPen(QPen(c, 0.7))
            for a, b in E:
                p.drawLine(QPointF(*P[a]), QPointF(*P[b]))
            return
        # solid / transparent: back to front, flat shaded towards the eye
        T = V[F]
        n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        eye, _, _, fwd = self._basis()
        shade = 0.35 + 0.65 * np.abs(n @ (-fwd))
        order = np.argsort(-z[F].mean(1))
        alpha = max(0.05, min(1.0, look.opacity)) if look.style == "transparent" else 1.0
        p.setPen(Qt.NoPen)
        r0, g0, b0 = base.red(), base.green(), base.blue()
        for k in order:
            sh = shade[k]
            c = QColor(int(r0 * sh), int(g0 * sh), int(b0 * sh))
            c.setAlphaF(alpha)
            p.setBrush(c)
            a, b, cc = P[F[k]]
            p.drawConvexPolygon(QPolygonF([QPointF(*a), QPointF(*b), QPointF(*cc)]))

    GAIT_COLORS = {"LF": "#e6194b", "RF": "#3cb44b", "LH": "#4363d8", "RH": "#f58231"}

    def _draw_gait(self, p: QPainter, m) -> None:
        """122kw: planted paws (size by load), the support polygon, the centre of mass over the
        floor and the zero-moment point."""
        from mufasa.pose3d.gait import _hull
        g = m.at(self.frame)
        if g is None:
            return
        z0 = 0.0
        on = [(k, v) for k, v in g["paws"].items() if v[3] == 1 and np.isfinite(v[0])]
        if len(on) >= 2:
            H = _hull(np.array([[v[0], v[1]] for _, v in on]))
            P, z = self.project(np.c_[H, np.full(len(H), z0)])
            if (z > 0).all():
                c = QColor(m.look.color)
                c.setAlphaF(0.25 * m.look.opacity)
                p.setBrush(c)
                p.setPen(QPen(QColor(m.look.color).darker(150), 1.4))
                p.drawPolygon(QPolygonF([QPointF(*q) for q in P]))
        for k, v in on:
            (q,), z = self.project(np.array([[v[0], v[1], z0]]))
            if z[0] > 0:
                r = 3 + 9 * max(0.0, min(1.0, v[2]))
                p.setPen(QPen(QColor("#202020"), 1))
                p.setBrush(QColor(self.GAIT_COLORS[k]))
                p.drawEllipse(QPointF(*q), r, r * 0.6)
        com = g["com"]
        if np.isfinite(com).all():
            (cp, cf), z = self.project(np.array([com, [com[0], com[1], z0]]))
            if (z > 0).all():
                p.setPen(QPen(QColor("#202020"), 1, Qt.DashLine))
                p.drawLine(QPointF(*cp), QPointF(*cf))
                p.setPen(QPen(QColor("#202020"), 1))
                p.setBrush(QColor("#ffd400"))
                p.drawEllipse(QPointF(*cp), 6, 6)
                p.drawEllipse(QPointF(*cf), 3, 3)
        zm = g["zmp"]
        if np.isfinite(zm).all():
            (zp,), z = self.project(np.array([[zm[0], zm[1], z0]]))
            if z[0] > 0:
                col = QColor("#1a9850" if g["margin"] >= 0 else "#d73027")
                p.setPen(QPen(col, 2))
                p.drawLine(QPointF(zp[0] - 5, zp[1] - 5), QPointF(zp[0] + 5, zp[1] + 5))
                p.drawLine(QPointF(zp[0] - 5, zp[1] + 5), QPointF(zp[0] + 5, zp[1] - 5))

    def _draw_axes(self, p: QPainter) -> None:
        """A small x/y/z gizmo in the lower left, turning with the view."""
        eye, right, up, fwd = self._basis()
        o = QPointF(40, self.height() - 40)
        for v, name, col in ((np.array([1.0, 0, 0]), "x", "#e6194b"), (np.array([0, 1.0, 0]), "y", "#3cb44b"),
                             (np.array([0, 0, 1.0]), "z", "#4363d8")):
            d = QPointF(28 * float(v @ right), -28 * float(v @ up))
            p.setPen(QPen(QColor(col), 2))
            p.drawLine(o, o + d)
            p.drawText(o + d * 1.2 + QPointF(-3, 4), name)

    # -- interaction ------------------------------------------------------ #
    def mousePressEvent(self, ev) -> None:  # noqa: ANN001, N802
        self.cam_blend = 0.0
        if self.cam_mode and ev.button() == Qt.LeftButton:
            self.cam_mode = False                      # orbiting leaves the camera view
        self._drag = (ev.position(), ev.button(), self.az, self.el, self.target.copy(),
                      self.cam_pan.copy())

    def mouseMoveEvent(self, ev) -> None:  # noqa: ANN001, N802
        if self._drag is None:
            return
        pos0, btn, az0, el0, t0, pan0 = self._drag
        d = ev.position() - pos0
        if self.cam_mode:                              # pan the image, keep alignment
            self.cam_pan = pan0 + np.array([d.x(), d.y()])
            self.update()
            return
        if btn == Qt.LeftButton:
            self.az = az0 - d.x() * 0.01
            self.el = max(math.radians(-89), min(math.radians(89), el0 + d.y() * 0.01))
        else:
            _, right, up, _ = self._basis()
            k = self.dist * 2 * math.tan(self.fov / 2) / max(1, min(self.width(), self.height()))
            self.target = t0 - right * d.x() * k + up * d.y() * k
        self.update()

    def mouseReleaseEvent(self, _ev) -> None:  # noqa: ANN001, N802
        self._drag = None

    def mouseDoubleClickEvent(self, _ev) -> None:  # noqa: ANN001, N802
        self.reset_view()

    def wheelEvent(self, ev) -> None:  # noqa: ANN001, N802
        self.cam_blend = 0.0
        if self.cam_mode:
            self.cam_zoom = max(0.2, min(20.0, self.cam_zoom * 1.1 ** (ev.angleDelta().y() / 120)))
            self.update()
            return
        self.dist *= 0.9 ** (ev.angleDelta().y() / 120)
        self.dist = max(self._home_dist * 0.05, min(self._home_dist * 10, self.dist))
        self.update()


def _nice_step(x: float) -> float:
    if x <= 0 or not math.isfinite(x):
        return 1.0
    e = 10 ** math.floor(math.log10(x))
    for m in (1, 2, 5, 10):
        if m * e >= x:
            return m * e
    return 10 * e


# --------------------------------------------------------------------------- #
# Reprojection onto the video
# --------------------------------------------------------------------------- #
def reprojection_overlay(pose: Pose3D, colors: dict):
    """``fn(frame_bgr, frame_idx) -> frame`` drawing the 3-D markers as the
    camera sees them (needs a camera in the file), or None. With lens
    distortion in the file the points are distorted to match the video
    (122kn; before, they were drawn undistorted)."""
    P = pose.projection_matrix()
    if P is None:
        return None
    import cv2
    distort = None
    c = pose.camera or {}
    dist = np.asarray(c.get("dist") or [], float)
    if "K" in c and dist.size and np.any(np.abs(dist) > 0):
        from mufasa.pose3d.camera import _distort_normalized
        K = np.asarray(c["K"], float)

        def distort(uv: np.ndarray) -> np.ndarray:
            xn = (uv - K[:2, 2]) / np.array([K[0, 0], K[1, 1]])
            return _distort_normalized(xn, dist) * np.array([K[0, 0], K[1, 1]]) + K[:2, 2]

    def draw(img: np.ndarray, frame: int) -> np.ndarray:
        i = pose.row(frame)
        if i is None:
            return img
        X = pose.xyz[i]
        ok = np.isfinite(X).all(1)
        if not ok.any():
            return img
        h = np.c_[X, np.ones(len(X))] @ P.T
        uv = h[:, :2] / h[:, 2:]
        if distort is not None:                    # the video is not undistorted (122kn)
            uv = distort(uv)
        out = img.copy()
        for a, b in pose.edges:
            ja, jb = pose.names.index(a), pose.names.index(b)
            if ok[ja] and ok[jb]:
                cv2.line(out, tuple(int(round(v)) for v in uv[ja]),
                         tuple(int(round(v)) for v in uv[jb]), (255, 255, 255), 1, cv2.LINE_AA)
        for j in np.flatnonzero(ok):
            c = colors[pose.names[j]]
            cv2.circle(out, tuple(int(round(v)) for v in uv[j]), 4,
                       (c.blue(), c.green(), c.red()), -1, cv2.LINE_AA)
        return out
    return draw


# --------------------------------------------------------------------------- #
# Viewer: volume above (optionally with the video on its floor), video below
# --------------------------------------------------------------------------- #
class Pose3DViewerWidget(QWidget):
    def __init__(self, video_path: str | None, pose_path, parent: QWidget | None = None,
                 models: list | None = None) -> None:
        super().__init__(parent)
        from mufasa.ui_qt.frame_scrubber import FrameScrubberWidget
        self.pose_path = Path(pose_path)
        self.pose = load_pose3d(self.pose_path)
        self.video_path = video_path
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        bar = QHBoxLayout()
        for name in (*VIEWS, "Camera"):
            b = QPushButton(name, self)
            b.setToolTip("Recording camera's view" if name == "Camera" else f"{name} view")
            if name == "Camera":
                b.setEnabled(self.pose.camera_position() is not None)
            b.clicked.connect(lambda _=False, n=name: self.volume.reset_view(n))
            bar.addWidget(b)
        bar.addSpacing(12)
        self.box_cb = QCheckBox("Box", self)
        self.shadow_cb = QCheckBox("Shadows", self)
        self.drop_cb = QCheckBox("Height lines", self)
        self.reproj_cb = QCheckBox("Reprojection", self)
        self.reproj_cb.setToolTip("Draw the 3-D markers on the video as the camera sees them")
        for cb in (self.box_cb, self.shadow_cb, self.drop_cb):
            cb.setChecked(True)
            bar.addWidget(cb)
        bar.addWidget(QLabel("Trail", self))
        self.trail_spin = QSpinBox(self)
        self.trail_spin.setRange(0, 300)
        self.trail_spin.setSuffix(" fr")
        bar.addWidget(self.trail_spin)
        bar.addWidget(self.reproj_cb)
        bar.addSpacing(12)
        self.video3d_cb = QCheckBox("Video in 3-D", self)
        self.video3d_cb.setToolTip(
            "Draw the video frame on the floor of the volume, mapped through the camera calibration. "
            "Camera view: every marker sits on the animal; other views: floor-level markers do, "
            "raised ones show their height as parallax.")
        self.opacity_spin = QSpinBox(self)
        self.opacity_spin.setRange(10, 100)
        self.opacity_spin.setSuffix(" %")
        self.opacity_spin.setValue(90)
        self.opacity_spin.setToolTip("Opacity of the video in the 3-D view")
        self.panel_cb = QCheckBox("Video panel", self)
        self.panel_cb.setToolTip("The video picture below the volume (the playback controls stay)")
        self.panel_cb.setChecked(True)
        for w in (self.video3d_cb, self.opacity_spin, self.panel_cb):
            bar.addWidget(w)
        bar.addSpacing(12)
        self.anim_cb = QCheckBox("Animation", self)
        self.anim_cb.setToolTip("Camera animation: key views at frames, smooth moves between them, "
                                "rendered to a video")
        bar.addWidget(self.anim_cb)
        self.models_cb = QCheckBox("View && models", self)
        self.models_cb.setToolTip("View options (markers, names, grid, projection, background) and models: "
                                  "a body-model rig posed to the markers, or a mesh (arena, objects)")
        bar.addWidget(self.models_cb)
        bar.addStretch()
        lay.addLayout(bar)
        split = QSplitter(Qt.Vertical, self)
        self.volume = Volume3DWidget(self.pose, split)
        split.addWidget(self.volume)
        self.scrubber = FrameScrubberWidget(split)
        split.addWidget(self.scrubber)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        lay.addWidget(split, 1)
        from mufasa.ui_qt.pose3d_animation import CameraAnimationPanel
        self.animation = CameraAnimationPanel(self, self)          # 122kp
        self.animation.setVisible(False)
        self.anim_cb.toggled.connect(self.animation.setVisible)
        lay.addWidget(self.animation)
        from mufasa.ui_qt.pose3d_models import ViewModelsPanel
        self.models_panel = ViewModelsPanel(self, self)              # 122kv
        self.models_panel.setVisible(False)
        self.models_cb.toggled.connect(self.models_panel.setVisible)
        lay.addWidget(self.models_panel)
        for mp in models or ():
            self.models_panel.add(mp)
        if models:
            self.models_cb.setChecked(True)
        self.info = QLabel(self._info_text(), self)
        self.info.setStyleSheet("color: palette(placeholder-text);")
        self.info.setWordWrap(True)
        lay.addWidget(self.info)
        self.box_cb.toggled.connect(lambda v: self._set("show_box", v))
        self.shadow_cb.toggled.connect(lambda v: self._set("show_shadows", v))
        self.drop_cb.toggled.connect(lambda v: self._set("show_drops", v))
        self.trail_spin.valueChanged.connect(lambda v: self._set("trail", int(v)))
        self._reproj = reprojection_overlay(self.pose, self.volume.colors)
        self.reproj_cb.setEnabled(self._reproj is not None)
        self.reproj_cb.toggled.connect(self._reproj_changed)
        self.box_cb.setEnabled(self.pose.box is not None)
        self.scrubber.frame_changed.connect(self._on_frame)
        can_video = bool(video_path) and self.pose.projection_matrix() is not None
        self.video3d_cb.setEnabled(can_video)
        self.opacity_spin.setEnabled(can_video)
        self.panel_cb.setEnabled(bool(video_path))
        self.video3d_cb.toggled.connect(self._video3d_changed)
        self.opacity_spin.valueChanged.connect(lambda v: self._set("video_opacity", v / 100.0))
        self.panel_cb.toggled.connect(self.scrubber.set_picture_visible)
        if video_path:
            self.scrubber.load(str(video_path))
            self.scrubber.seek(int(self.pose.frames[0]))
        if can_video:
            self.video3d_cb.setChecked(True)

    def closeEvent(self, ev) -> None:  # noqa: ANN001, N802
        self.models_panel.stop_fitting()                 # 122ky: no thread left running
        super().closeEvent(ev)

    def _on_frame(self, frame: int) -> None:
        self.volume.set_frame(frame)
        self.models_panel._update_info()
        if self.volume.show_video:
            self.volume.set_video_frame(self.scrubber.raw_frame())

    def _video3d_changed(self, on: bool) -> None:
        self.volume.show_video = bool(on)
        self.volume.set_video_frame(self.scrubber.raw_frame() if on else None)

    def _set(self, attr: str, v) -> None:  # noqa: ANN001
        setattr(self.volume, attr, v)
        self.volume.update()

    def _reproj_changed(self, on: bool) -> None:
        self.scrubber.set_overlay(self._reproj if on else None)

    def _info_text(self) -> str:
        m = self.pose.meta
        s = (f"{self.pose_path.name} · {len(self.pose.names)} markers · "
             f"{len(self.pose.frames):,} frames · {self.pose.units}")
        if m.get("method"):
            s += f" · {m['method']}"
        return s

    def cleanup(self) -> None:
        self.animation.timer.stop()
        self.volume._fit_timer.stop()
        try:
            self.scrubber.close_video()
        except Exception:
            pass


def choose_pose3d(parent: QWidget | None, start_dir: str = "") -> str | None:
    path, _ = QFileDialog.getOpenFileName(parent, "3-D pose file", start_dir,
                                          "3-D pose (*.parquet *.csv)")
    return path or None


def open_pose3d_viewer(parent: QWidget | None, video_path: str | None, pose_path, models: list | None = None
                       ) -> QWidget:
    """Show the viewer in a floating workbench dock (window fallback). ``models``: paths of models
    to load (``.rig.npz``, ``virtual_rodent``, ``.obj``/``.stl``/``.ply``)."""
    w = Pose3DViewerWidget(video_path, pose_path, models=models)
    main = parent.window() if parent is not None else None
    title = f"3-D — {Path(pose_path).stem}"
    if isinstance(main, QMainWindow):
        dock = QDockWidget(title, main)
        dock.setWidget(w)
        dock.setFloating(True)
        dock.resize(1000, 900)
        main.addDockWidget(Qt.RightDockWidgetArea, dock)
        orig = dock.closeEvent

        def _close(ev):  # noqa: ANN001
            w.cleanup()
            orig(ev)
        dock.closeEvent = _close
        dock.show()
        return dock
    w.setWindowTitle(title)
    w.resize(1000, 900)
    w.show()
    return w


__all__ = ["Pose3D", "Pose3DViewerWidget", "Volume3DWidget", "VIEWS", "find_pose3d",
           "load_pose3d", "open_pose3d_viewer", "reprojection_overlay"]
