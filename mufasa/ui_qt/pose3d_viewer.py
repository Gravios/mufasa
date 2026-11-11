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

Open from the feature viewer ("3-D…") or ``mufasa-view3d VIDEO [POSE3D]``.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
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
    names = [c[:-2] for c in df.columns
             if c.endswith("_x") and c[:-2] + "_y" in df.columns and c[:-2] + "_z" in df.columns]
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
        d = Path(config_path).parent / "derived" / "pose3d"
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
        self.reset_view()
        self._drag = None

    # -- camera ----------------------------------------------------------- #
    def reset_view(self, name: str = "Perspective") -> None:
        self.target = self._home_target.copy()
        self.dist = self._home_dist
        self.fov = math.radians(40)
        if name == "Camera" and self.pose.camera_position() is not None:
            self.view_from_recording_camera()
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
        if f:
            h = float((self.pose.camera or {}).get("image_height_px", 360))
            self.fov = 2 * math.atan(h / 2 / float(f))
        self.update()

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
        eye, right, up, fwd = self._basis()
        d = np.asarray(pts, float).reshape(-1, 3) - eye
        x, y, z = d @ right, d @ up, d @ fwd
        s = 0.5 * min(self.width(), self.height()) / math.tan(self.fov / 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            px = self.width() / 2 + s * x / z
            py = self.height() / 2 - s * y / z
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
        bg = QColor("#1b1d21") if dark else QColor("#fafafa")
        ink = QColor("#c9ccd1") if dark else QColor("#3a3d42")
        faint = QColor("#3a3e45") if dark else QColor("#dcdfe3")
        p.fillRect(self.rect(), bg)
        (x0, x1), (y0, y1), (z0, z1) = self.pose.extent()
        # floor grid
        step = _nice_step(max(x1 - x0, y1 - y0) / 8)
        pen = QPen(faint, 1)
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
            for a, b in self.pose.edges:
                ja, jb = names.index(a), names.index(b)
                if ok[ja] and ok[jb]:
                    self._line(p, X[ja], X[jb], pen)
            pts, z = self.project(np.where(ok[:, None], X, 0.0))
            for j in sorted(np.flatnonzero(ok & (z > 0)), key=lambda j: -z[j]):
                r = max(3.0, min(9.0, 900.0 / max(z[j], 1e-6) * 4))
                p.setPen(QPen(bg, 1))
                p.setBrush(self.colors[names[j]])
                p.drawEllipse(QPointF(*pts[j]), r, r)
        # text
        p.setPen(ink)
        f = QFont(p.font())
        f.setPointSizeF(max(8.0, f.pointSizeF() - 1))
        p.setFont(f)
        txt = f"frame {self.frame}"
        if X is None:
            txt += "  · no 3-D data for this frame"
        else:
            z = X[:, 2]
            if np.isfinite(z).any():
                txt += f"  · height {np.nanmin(z):.0f}–{np.nanmax(z):.0f} {self.pose.units}"
        p.drawText(10, 18, txt)
        p.end()

    # -- interaction ------------------------------------------------------ #
    def mousePressEvent(self, ev) -> None:  # noqa: ANN001, N802
        self._drag = (ev.position(), ev.button(), self.az, self.el, self.target.copy())

    def mouseMoveEvent(self, ev) -> None:  # noqa: ANN001, N802
        if self._drag is None:
            return
        pos0, btn, az0, el0, t0 = self._drag
        d = ev.position() - pos0
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
    camera sees them (needs a camera in the file), or None."""
    P = pose.projection_matrix()
    if P is None:
        return None
    import cv2

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
# Viewer: volume above, video below
# --------------------------------------------------------------------------- #
class Pose3DViewerWidget(QWidget):
    def __init__(self, video_path: str | None, pose_path, parent: QWidget | None = None) -> None:
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
        self.scrubber.frame_changed.connect(self.volume.set_frame)
        if video_path:
            self.scrubber.load(str(video_path))
            self.scrubber.seek(int(self.pose.frames[0]))

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
        try:
            self.scrubber.close_video()
        except Exception:
            pass


def choose_pose3d(parent: QWidget | None, start_dir: str = "") -> str | None:
    path, _ = QFileDialog.getOpenFileName(parent, "3-D pose file", start_dir,
                                          "3-D pose (*.parquet *.csv)")
    return path or None


def open_pose3d_viewer(parent: QWidget | None, video_path: str | None, pose_path
                       ) -> QWidget:
    """Show the viewer in a floating workbench dock (window fallback)."""
    w = Pose3DViewerWidget(video_path, pose_path)
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
