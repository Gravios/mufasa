"""Recording every visible panel of a viewer into one video (122lt).

User: "I would like each of the visible panels rendered in the video, such as if I have 3d,
video, and jpdf, then I want all of them."

:func:`compose` draws the panels as they sit on the screen (the video, the 3-D view, the JPDF,
the traces and histograms) into one picture, cropped to them; toolbars and lists are left out.
It is drawn at the video's own resolution: the video picture is the video's frame itself and the
other panels are drawn again at that scale (not enlarged screenshots), so they stay sharp.
:class:`PanelRecorder` keeps one such picture a video frame, all of the first one's size, and makes
the video when stopped (in the background).
"""
from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QRegion
from PySide6.QtWidgets import QWidget

__all__ = ["MAX_SCALE", "PanelRecorder", "compose", "showing", "video_scale"]

MAX_SCALE = 3.0                         # at most this many video pixels per screen pixel


def showing(w: QWidget | None) -> bool:
    """On the screen with some size (hidden, collapsed in a splitter or not yet laid out: not)."""
    return w is not None and w.isVisible() and w.width() > 4 and w.height() > 4


def _fit(img: np.ndarray, w: int, h: int, bg=(34, 34, 34)) -> np.ndarray:
    """``img`` fitted into w × h, its aspect kept, centred (as the video panel shows it)."""
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = max(1, round(iw * s)), max(1, round(ih * s))
    out = np.empty((h, w, 3), np.uint8)
    out[:] = bg
    x, y = (w - nw) // 2, (h - nh) // 2
    out[y:y + nh, x:x + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1
                                         else cv2.INTER_LINEAR)
    return out


def video_scale(video: QWidget, frame: np.ndarray) -> float:
    """Output pixels per screen pixel that show ``frame`` at its own size in ``video``'s panel
    (fitted, aspect kept), between 1 and :data:`MAX_SCALE`."""
    fh, fw = frame.shape[:2]
    return min(MAX_SCALE, max(1.0, fw / max(1, video.width()), fh / max(1, video.height())))


def _to_bgr(img: QImage) -> np.ndarray:
    img = img.convertToFormat(QImage.Format_BGR888)
    w, h, bpl = img.width(), img.height(), img.bytesPerLine()
    arr = np.frombuffer(img.constBits(), np.uint8, count=bpl * h)
    return arr.reshape(h, bpl)[:, : 3 * w].reshape(h, w, 3).copy()


def compose(root: QWidget, panels: Iterable[QWidget | None], *, video: QWidget | None = None,
            frame: np.ndarray | None = None, scale: float | None = None) -> np.ndarray | None:
    """The visible ``panels`` drawn where they are in ``root``, cropped to them, as a BGR image.

    ``video``, ``frame``: the panel showing the video, and the frame it shows (its own pixels go
    there). ``scale``: output pixels per screen pixel; by default the video's (its frame's width
    over the width it is shown at, at most :data:`MAX_SCALE`), else 1."""
    shown = [p for p in panels if showing(p)]
    if not shown:
        return None
    rects = [QRect(p.mapTo(root, QPoint(0, 0)), p.size()) for p in shown]
    box = rects[0]
    for r in rects[1:]:
        box = box.united(r)
    use_frame = frame is not None and video in shown
    if scale is None:
        scale = video_scale(video, frame) if use_frame else 1.0
    W, H = max(2, round(box.width() * scale)), max(2, round(box.height() * scale))
    img = QImage(W, H, QImage.Format_RGB888)
    pal = root.palette()
    img.fill(pal.window().color() if pal is not None else QColor("white"))
    painter = QPainter(img)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setRenderHint(QPainter.TextAntialiasing)
    painter.setRenderHint(QPainter.SmoothPixmapTransform)
    for p, r in zip(shown, rects, strict=True):
        if use_frame and p is video:
            continue
        painter.save()
        painter.translate(QPointF((r.x() - box.x()) * scale, (r.y() - box.y()) * scale))
        painter.scale(scale, scale)
        p.render(painter, QPoint(0, 0), QRegion(), QWidget.RenderFlag.DrawChildren)
        painter.restore()
    painter.end()
    out = _to_bgr(img)
    if use_frame:
        r = rects[shown.index(video)]
        x0, y0 = round((r.x() - box.x()) * scale), round((r.y() - box.y()) * scale)
        x1, y1 = min(W, round((r.right() + 1 - box.x()) * scale)), min(H, round((r.bottom() + 1 - box.y()) * scale))
        out[y0:y1, x0:x1] = _fit(frame, x1 - x0, y1 - y0)
    return out


class PanelRecorder:
    """Records a viewer: at each video frame, the visible panels as one picture (:func:`compose`);
    :meth:`stop` makes the video.

    ``panels``: called at each frame for the panels to record (those hidden then are left out);
    ``video`` and ``frame_source``: the video's panel and its current frame (overlays and the JPDF
    beside the video drawn in); ``all_panels`` off: that frame alone, as before 122lt."""

    def __init__(self, root: QWidget, panels: Callable[[], list], config_path, name: str, fps: float,
                 keep_frames: bool = True, all_panels: bool = True, video: QWidget | None = None,
                 frame_source: Callable[[], np.ndarray | None] | None = None) -> None:
        from mufasa.ui_qt.pose_overlay import FrameRecorder
        self.root, self.panels, self.all_panels = root, panels, all_panels
        self.video, self.frame_source = video, frame_source
        self.rec = FrameRecorder(config_path, name, fps, keep_frames=keep_frames)
        self.size: tuple[int, int] | None = None
        self.scale: float | None = None

    @property
    def n(self) -> int:
        return self.rec.n

    @property
    def frames_dir(self) -> Path:
        return self.rec.frames_dir

    @property
    def video_path(self) -> Path:
        return self.rec.video_path

    def picture(self) -> np.ndarray | None:
        frame = self.frame_source() if self.frame_source is not None else None
        if not self.all_panels:
            img = frame
        else:
            if self.scale is None:                               # the first frame's, throughout
                self.scale = video_scale(self.video, frame) if frame is not None and showing(self.video) else 1.0
            img = compose(self.root, self.panels(), video=self.video, frame=frame, scale=self.scale)
        if img is None:
            return None
        if self.size is None:
            h, w = img.shape[:2]
            self.size = (w - w % 2, h - h % 2) if self.all_panels else (w, h)
        if img.shape[1::-1] != self.size:          # a panel shown or hidden, or the window resized
            img = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
        return img

    def capture(self, frame: int) -> None:
        """Keep this video frame (the panels already show it)."""
        self.rec.add(self.picture(), frame)

    def stop(self, done: Callable[[bool, str], None], parent: QWidget) -> None:
        """Make the video in the background; ``done(ok, message)`` after."""
        result: dict = {}
        t = threading.Thread(target=lambda: result.update(r=self.rec.stop()), daemon=True)
        t.start()
        timer = QTimer(parent)

        def poll() -> None:
            if t.is_alive():
                return
            timer.stop()
            done(*result.get("r", (False, "encoding failed")))

        timer.timeout.connect(poll)
        timer.start(100)
        self.thread = t

