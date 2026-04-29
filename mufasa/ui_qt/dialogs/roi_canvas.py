"""
mufasa.ui_qt.dialogs.roi_canvas
===============================

Native Qt drawing canvas for ROI definition. Replaces the OpenCV
subprocess-based approach in earlier versions of the panel.

Why this exists
---------------

OpenCV's pip wheel on Linux is built with the Qt GUI backend
(``window_QT.cpp``). Running OpenCV's drawing windows from inside
a host PySide6 application produces a long tail of conflicts:

* ``cv2.namedWindow`` succeeding but ``cv2.setMouseCallback`` failing
  with NULL handle errors.
* ``cv2.getWindowProperty(WND_PROP_VISIBLE)`` race conditions on
  X11 / Wayland that close the window before the user can interact.
* Subprocess isolation works for some shapes but not others.

After 5 iterations of working around these issues, the cumulative
fragility outweighed the cost of a from-scratch native Qt
implementation. This is that implementation.

Design
------

Single ``ROICanvas`` widget that:

1. Renders the current video frame as a ``QPixmap`` via ``paintEvent``.
2. Renders existing ROIs (rectangles / circles / polygons) on top.
3. Accepts mouse + keyboard events for drawing new shapes.
4. Maintains a small state machine per shape type (idle / dragging /
   vertexing).
5. Emits ``shape_committed`` carrying the captured geometry in
   ORIGINAL FRAME pixel coordinates (not widget coordinates) when
   the user finishes drawing.

Coordinate spaces
-----------------

Two coordinate systems coexist:

* **Frame coords**: pixel positions in the original image. ROIs are
  stored and persisted in these.
* **Widget coords**: pixel positions in the displayed widget. The
  frame is scaled with aspect-preservation to fit the widget, with
  black bars top/bottom or left/right.

Mouse events come in widget coords; we translate to frame coords via
``_widget_to_frame()``. Painting goes the other way via
``_frame_to_widget()``.

Failure modes I expect (and have tried to prevent)
--------------------------------------------------

* **Off-by-one bugs** in coordinate transforms — visible in saved H5
  ROIs that are subtly wrong size.
* **Black bar clicks** — when the widget is wider than the frame
  aspect, clicks in the side bars should be ignored. Handled by
  ``_widget_to_frame`` returning None outside the frame area.
* **Resize-during-draw** — if the widget resizes mid-draw, the
  in-progress shape's widget coords become stale. State is stored
  in frame coords; re-paint translates back to widget coords.
* **Focus stealing** — keyboard shortcuts (ESC, Space, etc.) need
  the canvas to have focus. ``setFocus()`` is called when entering
  drawing mode.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import (QColor, QImage, QKeyEvent, QMouseEvent, QPainter,
                           QPaintEvent, QPen, QPixmap)
from PySide6.QtWidgets import QWidget

from mufasa.roi_tools.roi_logic import CIRCLE, POLYGON, RECTANGLE


class _DrawMode(enum.Enum):
    """Active drawing state of the canvas."""
    IDLE = "idle"               # not drawing; preview is read-only
    RECT_PENDING = "rect_pending"     # tool selected, waiting for mouse-down
    RECT_DRAGGING = "rect_dragging"   # mouse down, capturing rectangle
    CIRCLE_PENDING = "circle_pending"
    CIRCLE_DRAGGING = "circle_dragging"
    POLY_PENDING = "poly_pending"     # tool selected, waiting for first vertex
    POLY_VERTEXING = "poly_vertexing" # accumulating polygon vertices


# Visual constants
_LIVE_LINE_WIDTH = 2
_VERTEX_DOT_RADIUS = 5
# Distance (in widget pixels) within which a polygon click is treated
# as "snap to existing vertex" — clicking on the first vertex closes
# the polygon, clicking on an existing vertex is a no-op (avoids
# accidental degenerate edges).
_VERTEX_SNAP_DISTANCE_WIDGET_PX = 10

# NOTE on update strategy:
# The canvas uses full-widget self.update() calls rather than
# update(QRect) for bounding-rect updates. Region-based updates would
# be faster on slow X11 connections (ThinLinc, especially) but require
# tracking previous-position rects to invalidate both old and new
# locations. Getting that wrong leaves drawing artifacts.
# If repaint flicker becomes a real problem, the right place to
# implement is in mouseMoveEvent (rect_dragging / circle_dragging /
# poly rubber-band updates): compute oldRect.united(newRect).adjusted(-pad)
# and pass to update(). Until then, full repaint is safer.


@dataclass
class _ImageMapping:
    """Cached transform between widget and frame pixel coordinates."""
    scale: float
    offset_x: int
    offset_y: int
    frame_w: int
    frame_h: int

    def widget_to_frame(self, wx: int, wy: int
                        ) -> Optional[Tuple[int, int]]:
        """Convert widget coords to frame coords. Returns None if the
        widget point is in the black bars (outside the frame)."""
        if self.scale <= 0:
            return None
        fx = (wx - self.offset_x) / self.scale
        fy = (wy - self.offset_y) / self.scale
        if 0 <= fx <= self.frame_w and 0 <= fy <= self.frame_h:
            return int(round(fx)), int(round(fy))
        return None

    def frame_to_widget(self, fx: float, fy: float
                        ) -> Tuple[int, int]:
        """Convert frame coords to widget coords (for painting)."""
        return (
            int(round(fx * self.scale + self.offset_x)),
            int(round(fy * self.scale + self.offset_y)),
        )


@dataclass
class _DrawState:
    """All currently-being-drawn shape state. Coordinates are
    always in FRAME pixel space, never widget pixel space."""
    # Rectangle: corner anchors
    rect_start: Optional[Tuple[int, int]] = None
    rect_end: Optional[Tuple[int, int]] = None
    # Circle: center + current radius point
    circle_center: Optional[Tuple[int, int]] = None
    circle_edge: Optional[Tuple[int, int]] = None
    # Polygon: committed vertices + current rubber-band end point
    poly_vertices: List[Tuple[int, int]] = field(default_factory=list)
    poly_rubber_end: Optional[Tuple[int, int]] = None

    def reset(self) -> None:
        self.rect_start = None
        self.rect_end = None
        self.circle_center = None
        self.circle_edge = None
        self.poly_vertices = []
        self.poly_rubber_end = None


class ROICanvas(QWidget):
    """Native Qt widget for drawing ROIs on a video frame.

    Usage from the panel::

        canvas = ROICanvas(self)
        canvas.shape_committed.connect(self._on_shape_committed)
        canvas.set_frame(frame_bgr_array)

        # Later, when user clicks Draw → in the panel:
        canvas.start_draw(kind=RECTANGLE, color_bgr=(0, 0, 255),
                          thickness=3)

        # User draws on the canvas. shape_committed fires with the
        # captured geometry, or shape_cancelled fires on ESC.
    """

    # Emitted on successful shape capture. Args: kind, geometry dict
    # in frame-pixel coords. Geometry shape mirrors what
    # _run_selector_in_process used to return, for drop-in replacement:
    #   rectangle: {"top_left": [x, y], "bottom_right": [x, y]}
    #   circle:    {"center": [x, y], "radius": int}
    #   polygon:   {"vertices": [[x, y], ...]}
    shape_committed = Signal(str, dict)

    # Emitted on user cancel (ESC outside dragging, or window close).
    shape_cancelled = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._frame_bgr: Optional[np.ndarray] = None
        self._frame_pixmap: Optional[QPixmap] = None
        self._mapping: Optional[_ImageMapping] = None

        # Existing ROIs to display (overlay only, not editable here).
        # List of dicts: {"kind": str, "color_bgr": (b,g,r), "thickness": int,
        #                 "geometry": dict (same shape as committed signal)}
        self._existing_rois: list[dict] = []

        # Active draw state
        self._mode = _DrawMode.IDLE
        self._draw_state = _DrawState()
        self._draw_color_bgr: Tuple[int, int, int] = (0, 0, 255)
        self._draw_thickness: int = 3

        # Widget config
        self.setMinimumSize(640, 360)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setStyleSheet(
            "background: #1a1a1a; border: 1px solid palette(mid);"
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def set_frame(self, bgr: Optional[np.ndarray]) -> None:
        """Replace the displayed frame.

        Importantly, this does NOT cancel any in-progress drawing.
        Frame nav (slider, +1f, +1s, etc.) often happens during a
        polygon draw — the user navigates to the right frame to
        place vertices around a feature visible only there. ROI
        geometry is stored in frame-pixel space, not widget-pixel
        space, so the in-progress shape stays correct across frame
        changes; only the underlying frame image changes.
        """
        self._frame_bgr = bgr
        if bgr is None:
            self._frame_pixmap = None
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
            self._frame_pixmap = QPixmap.fromImage(qimg)
        self._recompute_mapping()
        self.update()

    def set_existing_rois(self, rois: list[dict]) -> None:
        """Replace the list of ROIs displayed as overlays.

        Each dict: {kind, color_bgr (B,G,R tuple), thickness, geometry}.
        The geometry shape matches the shape_committed signal payload.
        """
        self._existing_rois = list(rois)
        self.update()

    def start_draw(self, kind: str,
                   color_bgr: Tuple[int, int, int],
                   thickness: int) -> None:
        """Enter drawing mode for ``kind`` (one of RECTANGLE / CIRCLE /
        POLYGON from mufasa.roi_tools.roi_logic).

        After this call, the next user mouse/keyboard interaction
        starts capturing a shape. The canvas grabs keyboard focus so
        ESC/Q/Space work.
        """
        if self._mode != _DrawMode.IDLE:
            self._cancel_draw(emit_signal=False)
        self._draw_color_bgr = color_bgr
        self._draw_thickness = max(1, int(thickness))
        if kind == RECTANGLE:
            self._mode = _DrawMode.RECT_PENDING
        elif kind == CIRCLE:
            self._mode = _DrawMode.CIRCLE_PENDING
        elif kind == POLYGON:
            self._mode = _DrawMode.POLY_PENDING
        else:
            return
        self.setFocus(Qt.OtherFocusReason)
        self.setCursor(Qt.CrossCursor)
        self.update()

    def is_drawing(self) -> bool:
        """True when a shape capture is in progress (any non-IDLE)."""
        return self._mode != _DrawMode.IDLE

    def cancel_draw(self) -> None:
        """Abort any in-progress draw. Emits shape_cancelled."""
        if self._mode != _DrawMode.IDLE:
            self._cancel_draw(emit_signal=True)

    # ------------------------------------------------------------------ #
    # Coordinate mapping
    # ------------------------------------------------------------------ #
    def _recompute_mapping(self) -> None:
        if self._frame_bgr is None:
            self._mapping = None
            return
        h, w = self._frame_bgr.shape[:2]
        wW = max(1, self.width())
        wH = max(1, self.height())
        scale = min(wW / w, wH / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        offset_x = (wW - new_w) // 2
        offset_y = (wH - new_h) // 2
        self._mapping = _ImageMapping(
            scale=scale, offset_x=offset_x, offset_y=offset_y,
            frame_w=w, frame_h=h,
        )

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        self._recompute_mapping()
        self.update()

    # ------------------------------------------------------------------ #
    # Painting
    # ------------------------------------------------------------------ #
    def paintEvent(self, ev: QPaintEvent) -> None:
        p = QPainter(self)
        try:
            # 1. Background
            p.fillRect(self.rect(), QColor(26, 26, 26))

            # 2. Frame (aspect-preserving scaled)
            if self._frame_pixmap is not None and self._mapping is not None:
                m = self._mapping
                target = QRect(
                    m.offset_x, m.offset_y,
                    int(round(m.frame_w * m.scale)),
                    int(round(m.frame_h * m.scale)),
                )
                p.drawPixmap(target, self._frame_pixmap,
                             self._frame_pixmap.rect())

                # 3. Existing ROIs overlay
                for roi in self._existing_rois:
                    self._paint_existing_roi(p, roi)

                # 4. In-progress drawing
                self._paint_in_progress(p)
        finally:
            p.end()

    def _paint_existing_roi(self, p: QPainter, roi: dict) -> None:
        kind = roi.get("kind")
        color_bgr = roi.get("color_bgr", (0, 0, 255))
        thickness = roi.get("thickness", 2)
        geom = roi.get("geometry", {})
        m = self._mapping
        if m is None:
            return
        pen = QPen(self._bgr_to_qcolor(color_bgr))
        pen.setWidth(max(1, int(thickness)))
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)

        if kind == RECTANGLE:
            tl = geom.get("top_left") or (
                geom.get("topLeftX", 0), geom.get("topLeftY", 0),
            )
            br = geom.get("bottom_right") or (
                geom.get("Bottom_right_X", 0), geom.get("Bottom_right_Y", 0),
            )
            x1, y1 = m.frame_to_widget(tl[0], tl[1])
            x2, y2 = m.frame_to_widget(br[0], br[1])
            p.drawRect(QRect(min(x1, x2), min(y1, y2),
                             abs(x2 - x1), abs(y2 - y1)))
        elif kind == CIRCLE:
            cx, cy = geom.get("center") or (
                geom.get("centerX", 0), geom.get("centerY", 0),
            )
            r = geom.get("radius", 0)
            wcx, wcy = m.frame_to_widget(cx, cy)
            wr = int(round(r * m.scale))
            p.drawEllipse(QPoint(wcx, wcy), wr, wr)
        elif kind == POLYGON:
            verts = geom.get("vertices") or []
            if len(verts) >= 2:
                pts = [QPoint(*m.frame_to_widget(v[0], v[1])) for v in verts]
                # Closed polygon — connect last to first too
                p.drawPolygon(pts)

    def _paint_in_progress(self, p: QPainter) -> None:
        m = self._mapping
        if m is None:
            return
        pen = QPen(self._bgr_to_qcolor(self._draw_color_bgr))
        pen.setWidth(_LIVE_LINE_WIDTH)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)

        if self._mode == _DrawMode.RECT_DRAGGING:
            s = self._draw_state
            if s.rect_start is not None and s.rect_end is not None:
                x1, y1 = m.frame_to_widget(s.rect_start[0], s.rect_start[1])
                x2, y2 = m.frame_to_widget(s.rect_end[0], s.rect_end[1])
                p.drawRect(QRect(min(x1, x2), min(y1, y2),
                                 abs(x2 - x1), abs(y2 - y1)))

        elif self._mode == _DrawMode.CIRCLE_DRAGGING:
            s = self._draw_state
            if s.circle_center is not None and s.circle_edge is not None:
                cx, cy = m.frame_to_widget(*s.circle_center)
                ex, ey = m.frame_to_widget(*s.circle_edge)
                r = int(round(((ex - cx) ** 2 + (ey - cy) ** 2) ** 0.5))
                p.drawEllipse(QPoint(cx, cy), r, r)
                # also draw a tiny center marker
                p.drawEllipse(QPoint(cx, cy), 2, 2)

        elif self._mode == _DrawMode.POLY_VERTEXING:
            s = self._draw_state
            verts = s.poly_vertices
            if len(verts) >= 1:
                # Draw committed segments
                for i in range(len(verts) - 1):
                    a = m.frame_to_widget(*verts[i])
                    b = m.frame_to_widget(*verts[i + 1])
                    p.drawLine(QPoint(*a), QPoint(*b))
                # Vertex markers
                for v in verts:
                    wx, wy = m.frame_to_widget(*v)
                    p.drawEllipse(QPoint(wx, wy),
                                  _VERTEX_DOT_RADIUS, _VERTEX_DOT_RADIUS)
                # Closing-edge preview when polygon would already be
                # valid (≥3 vertices). Faint dashed line back to the
                # first vertex shows what shape will commit. Without
                # this, drawing a triangle looks like an open V-shape
                # until ESC, which is confusing.
                if len(verts) >= 3:
                    first = m.frame_to_widget(*verts[0])
                    last = m.frame_to_widget(*verts[-1])
                    closing_pen = QPen(self._bgr_to_qcolor(
                        self._draw_color_bgr))
                    closing_pen.setWidth(_LIVE_LINE_WIDTH)
                    closing_pen.setStyle(Qt.DotLine)
                    p.setPen(closing_pen)
                    p.drawLine(QPoint(*last), QPoint(*first))
                # Rubber band — dashed line from last vertex to mouse
                if s.poly_rubber_end is not None:
                    last = m.frame_to_widget(*verts[-1])
                    end = m.frame_to_widget(*s.poly_rubber_end)
                    rubber_pen = QPen(self._bgr_to_qcolor(
                        self._draw_color_bgr))
                    rubber_pen.setWidth(_LIVE_LINE_WIDTH)
                    rubber_pen.setStyle(Qt.DashLine)
                    p.setPen(rubber_pen)
                    p.drawLine(QPoint(*last), QPoint(*end))

    # ------------------------------------------------------------------ #
    # Mouse events
    # ------------------------------------------------------------------ #
    def mousePressEvent(self, ev: QMouseEvent) -> None:
        if self._mapping is None or ev.button() != Qt.LeftButton:
            return
        frame_pt = self._mapping.widget_to_frame(ev.position().x(),
                                                  ev.position().y())
        if frame_pt is None:
            # Click outside the frame area (in black bars)
            return

        if self._mode == _DrawMode.RECT_PENDING:
            self._draw_state.rect_start = frame_pt
            self._draw_state.rect_end = frame_pt
            self._mode = _DrawMode.RECT_DRAGGING
            self.update()

        elif self._mode == _DrawMode.CIRCLE_PENDING:
            self._draw_state.circle_center = frame_pt
            self._draw_state.circle_edge = frame_pt
            self._mode = _DrawMode.CIRCLE_DRAGGING
            self.update()

        elif self._mode == _DrawMode.POLY_PENDING:
            self._draw_state.poly_vertices = [frame_pt]
            self._draw_state.poly_rubber_end = frame_pt
            self._mode = _DrawMode.POLY_VERTEXING
            self.update()

        elif self._mode == _DrawMode.POLY_VERTEXING:
            # Snap to first vertex if clicked nearby — closes polygon.
            # Snap to last vertex if clicked nearby — no-op (avoids
            # accidental duplicate edges from double-clicks landing
            # on the same place).
            verts = self._draw_state.poly_vertices
            if verts:
                # Distance computed in widget pixels for predictable
                # snap behavior regardless of frame zoom level.
                wx_click = ev.position().x()
                wy_click = ev.position().y()
                m = self._mapping
                wx_first, wy_first = m.frame_to_widget(*verts[0])
                wx_last, wy_last = m.frame_to_widget(*verts[-1])
                d_first = ((wx_click - wx_first) ** 2
                           + (wy_click - wy_first) ** 2) ** 0.5
                d_last = ((wx_click - wx_last) ** 2
                          + (wy_click - wy_last) ** 2) ** 0.5
                # First-vertex snap: close polygon if ≥3 vertices
                if (d_first <= _VERTEX_SNAP_DISTANCE_WIDGET_PX
                        and len(verts) >= 3):
                    self._commit_polygon()
                    return
                # Last-vertex snap: ignore (no degenerate edge)
                if d_last <= _VERTEX_SNAP_DISTANCE_WIDGET_PX:
                    return
            # Otherwise add another vertex
            self._draw_state.poly_vertices.append(frame_pt)
            self.update()

    def mouseDoubleClickEvent(self, ev: QMouseEvent) -> None:
        """Double-click commits the polygon if it has ≥3 vertices.

        This matches the convention from many polygon-drawing tools
        (GIMP's Free Select, image-annotation tools, etc.). Without
        this, users frequently try double-click and are confused
        when nothing happens — they only learn ESC/Q/Space close
        from documentation.
        """
        if (self._mode == _DrawMode.POLY_VERTEXING
                and ev.button() == Qt.LeftButton
                and len(self._draw_state.poly_vertices) >= 3):
            # The single-click that started this double-click already
            # added a vertex via mousePressEvent. Now commit; the
            # extra vertex is fine (it's just before the close).
            self._commit_polygon()
            return
        super().mouseDoubleClickEvent(ev)

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:
        if self._mapping is None:
            return
        # Skip the coordinate transform entirely when not drawing.
        # mouseMoveEvent fires on every mouse motion (setMouseTracking
        # is on); the panel doesn't care about IDLE motion.
        if self._mode not in (_DrawMode.RECT_DRAGGING,
                              _DrawMode.CIRCLE_DRAGGING,
                              _DrawMode.POLY_VERTEXING):
            return
        frame_pt = self._mapping.widget_to_frame(ev.position().x(),
                                                  ev.position().y())
        # For move events we DO want updates even when mouse leaves
        # the frame — clamp to frame bounds rather than ignore.
        if frame_pt is None:
            # Use the closest frame edge
            wx, wy = ev.position().x(), ev.position().y()
            m = self._mapping
            fx = (wx - m.offset_x) / m.scale
            fy = (wy - m.offset_y) / m.scale
            fx = max(0, min(m.frame_w, fx))
            fy = max(0, min(m.frame_h, fy))
            frame_pt = (int(round(fx)), int(round(fy)))

        if self._mode == _DrawMode.RECT_DRAGGING:
            self._draw_state.rect_end = frame_pt
            self.update()
        elif self._mode == _DrawMode.CIRCLE_DRAGGING:
            self._draw_state.circle_edge = frame_pt
            self.update()
        elif self._mode == _DrawMode.POLY_VERTEXING:
            self._draw_state.poly_rubber_end = frame_pt
            self.update()

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:
        if ev.button() != Qt.LeftButton:
            return
        # Rectangle and circle commit on release.
        # Polygon stays in vertexing mode — no commit on release.
        if self._mode == _DrawMode.RECT_DRAGGING:
            self._commit_rectangle()
        elif self._mode == _DrawMode.CIRCLE_DRAGGING:
            self._commit_circle()

    # ------------------------------------------------------------------ #
    # Keyboard events
    # ------------------------------------------------------------------ #
    def keyPressEvent(self, ev: QKeyEvent) -> None:
        key = ev.key()
        if self._mode == _DrawMode.IDLE:
            super().keyPressEvent(ev)
            return

        # ESC commits polygon if ≥3 vertices, cancels otherwise.
        # For rectangle/circle modes ESC always cancels.
        if key == Qt.Key_Escape:
            if self._mode == _DrawMode.POLY_VERTEXING and \
                    len(self._draw_state.poly_vertices) >= 3:
                # In polygon mode, ESC also commits if we have enough
                # vertices (consistent with the legacy OpenCV behavior).
                self._commit_polygon()
            else:
                self._cancel_draw(emit_signal=True)
            return

        if self._mode == _DrawMode.POLY_VERTEXING:
            if key in (Qt.Key_Q, Qt.Key_Space, Qt.Key_Return,
                        Qt.Key_Enter):
                if len(self._draw_state.poly_vertices) >= 3:
                    self._commit_polygon()
                else:
                    self._cancel_draw(emit_signal=True)
                return
            if key == Qt.Key_Backspace:
                if self._draw_state.poly_vertices:
                    self._draw_state.poly_vertices.pop()
                    if not self._draw_state.poly_vertices:
                        # Removed the last vertex — back to pending
                        self._mode = _DrawMode.POLY_PENDING
                self.update()
                return

        super().keyPressEvent(ev)

    # ------------------------------------------------------------------ #
    # Commit / cancel
    # ------------------------------------------------------------------ #
    def _commit_rectangle(self) -> None:
        s = self._draw_state
        if s.rect_start is None or s.rect_end is None:
            self._cancel_draw(emit_signal=True)
            return
        x1, y1 = s.rect_start
        x2, y2 = s.rect_end
        if abs(x2 - x1) < 2 or abs(y2 - y1) < 2:
            # Too small — treat as a click rather than a drag
            self._cancel_draw(emit_signal=True)
            return
        geom = {
            "top_left": [int(min(x1, x2)), int(min(y1, y2))],
            "bottom_right": [int(max(x1, x2)), int(max(y1, y2))],
        }
        self._reset_state()
        self.shape_committed.emit(RECTANGLE, geom)

    def _commit_circle(self) -> None:
        s = self._draw_state
        if s.circle_center is None or s.circle_edge is None:
            self._cancel_draw(emit_signal=True)
            return
        cx, cy = s.circle_center
        ex, ey = s.circle_edge
        r = int(round(((ex - cx) ** 2 + (ey - cy) ** 2) ** 0.5))
        if r < 2:
            self._cancel_draw(emit_signal=True)
            return
        geom = {
            "center": [int(cx), int(cy)],
            "radius": r,
        }
        self._reset_state()
        self.shape_committed.emit(CIRCLE, geom)

    def _commit_polygon(self) -> None:
        verts = self._draw_state.poly_vertices
        if len(verts) < 3:
            self._cancel_draw(emit_signal=True)
            return
        # Optional Shapely simplification — matches the legacy
        # OpenCV path's polygon vertex tolerance handling.
        try:
            from shapely.geometry import Polygon as _Poly
            simp = _Poly(verts).simplify(tolerance=20,
                                          preserve_topology=True)
            if simp.is_valid and not simp.is_empty:
                # exterior.coords includes a closing duplicate; drop it
                final = [(int(x), int(y))
                         for x, y in list(simp.exterior.coords)[:-1]]
                if len(final) >= 3:
                    verts = final
        except Exception:
            # Shapely failed (degenerate polygon, etc.); use raw verts
            pass

        geom = {"vertices": [[int(x), int(y)] for x, y in verts]}
        self._reset_state()
        self.shape_committed.emit(POLYGON, geom)

    def _cancel_draw(self, emit_signal: bool) -> None:
        self._reset_state()
        if emit_signal:
            self.shape_cancelled.emit()

    def _reset_state(self) -> None:
        self._draw_state.reset()
        self._mode = _DrawMode.IDLE
        self.unsetCursor()
        self.update()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _bgr_to_qcolor(bgr: Tuple[int, int, int]) -> QColor:
        b, g, r = bgr
        return QColor(int(r), int(g), int(b))


__all__ = ["ROICanvas"]
