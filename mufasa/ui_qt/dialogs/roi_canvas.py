"""
mufasa.ui_qt.dialogs.roi_canvas
===============================

Native Qt drawing canvas for ROI definition.

Why native Qt
-------------

OpenCV's pip wheel on Linux is built with the Qt GUI backend. Running its
drawing windows from inside a PySide6 application produced a long tail of
conflicts (NULL mouse-callback handles, visibility races on X11/Wayland), so
the canvas is a plain ``QWidget`` that paints the frame and the ROIs itself.

Tools (patch 122ij redesign)
----------------------------

The canvas has one active *tool* at a time, chosen by the panel:

``select``
    Click a shape to select it. Nothing is hidden by drawing order: handles
    (vertices, corners) and outlines (rectangle sides, circle rims, polygon
    edges) of every ROI are grabbable even where another ROI covers them,
    and a click inside overlapping ROIs picks the smallest — click again on
    the same spot to cycle to the next one (patch 122im). Drag the inside to
    move; drag a vertex, corner, side, rim or polygon edge to reshape. On a selected
    polygon, double-click an edge to insert a vertex there and right-click a
    vertex to remove it. Arrow keys nudge the selection by 1 px (Shift: 10).
    Delete/Backspace removes it; Esc deselects.
``rectangle`` / ``circle``
    Press and drag. Circles are drawn from the centre outward.
``polygon``
    Click to place vertices — **any number of them**. Finish by clicking the
    first vertex, double-clicking, or pressing Enter. Backspace removes the
    last vertex; Esc abandons the polygon.

Drawing tools are *sticky*: after a shape is committed the same tool stays
armed, so several shapes can be drawn in a row. Esc during a draw abandons
only the shape in progress.

Before 122ij, committing a polygon ran it through
``shapely.simplify(tolerance=20)``, which silently deleted every vertex lying
within 20 frame pixels of the line through its neighbours. On a 640x360
frame that capped real polygons at roughly 4-8 vertices however many were
clicked. Vertices are now kept exactly as placed (only exact consecutive
duplicates are dropped).

View
----

Mouse wheel zooms about the cursor, middle-drag pans, ``F`` (or ``0``) fits
the frame back into the widget. ROI geometry is always held in *frame*
pixels; :class:`_ImageMapping` converts to and from widget pixels.

Signals
-------

* ``shape_committed(kind, geometry)`` — a new shape was drawn.
* ``shape_cancelled()`` — a shape in progress was abandoned.
* ``shape_edited(idx, geometry)`` — an existing shape was moved/reshaped.
* ``shape_deleted(idx)`` — Delete pressed on the selected shape.
* ``shape_selected(idx)`` — selection changed by a click (-1 = none).
* ``hint_changed(text)`` — what the current tool expects next.
* ``zoom_changed(factor)`` — 1.0 = fit to widget.

Geometry dicts are canonical: ``{"top_left", "bottom_right"}`` for
rectangles, ``{"center", "radius"}`` for circles, ``{"vertices"}`` for
polygons. Legacy per-axis keys (``topLeftX``, ``centerX``, ...) are accepted
on input.
"""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPixmap,
    QPolygonF,
    QWheelEvent,
)
from PySide6.QtWidgets import QWidget

from mufasa.roi_tools.roi_logic import CIRCLE, POLYGON, RECTANGLE

SELECT = "select"
TOOLS = (SELECT, RECTANGLE, CIRCLE, POLYGON)


class _DrawMode(enum.Enum):
    """Active interaction state of the canvas."""
    IDLE = "idle"                       # no tool; read-only preview
    RECT_PENDING = "rect_pending"       # rectangle tool, waiting for press
    RECT_DRAGGING = "rect_dragging"
    CIRCLE_PENDING = "circle_pending"
    CIRCLE_DRAGGING = "circle_dragging"
    POLY_PENDING = "poly_pending"       # polygon tool, no vertex yet
    POLY_VERTEXING = "poly_vertexing"   # accumulating vertices
    SELECT = "select"                   # select tool, nothing being dragged
    SHAPE_MOVING = "shape_moving"       # dragging a shape's body
    HANDLE_DRAGGING = "handle_dragging"  # dragging a corner/edge/rim/vertex


_PENDING_FOR_TOOL = {
    RECTANGLE: _DrawMode.RECT_PENDING,
    CIRCLE: _DrawMode.CIRCLE_PENDING,
    POLYGON: _DrawMode.POLY_PENDING,
    SELECT: _DrawMode.SELECT,
}

# Visual / interaction constants (widget pixels).
_LIVE_LINE_WIDTH = 2
_VERTEX_DOT_RADIUS = 4
# Clicking within this distance of the first vertex closes the polygon.
_CLOSE_SNAP_WIDGET_PX = 10
# A click this close to the previous vertex is ignored (stray double hits).
_DUPLICATE_SNAP_WIDGET_PX = 3
_HANDLE_HIT_TOLERANCE_WIDGET_PX = 8
_HANDLE_BOX_HALF_PX = 5
_EDGE_HIT_TOLERANCE_WIDGET_PX = 6
_MIN_ZOOM, _MAX_ZOOM = 1.0, 32.0
_SELECTION_COLOR = QColor(255, 215, 0)

_RECT_HANDLES = ("nw", "n", "ne", "e", "se", "s", "sw", "w")
_HANDLE_CURSORS = {
    "nw": Qt.SizeFDiagCursor, "se": Qt.SizeFDiagCursor,
    "ne": Qt.SizeBDiagCursor, "sw": Qt.SizeBDiagCursor,
    "n": Qt.SizeVerCursor, "s": Qt.SizeVerCursor,
    "e": Qt.SizeHorCursor, "w": Qt.SizeHorCursor,
    "edge": Qt.SizeHorCursor,
}

_TOOL_HINTS = {
    SELECT: ("Select: click a shape (click again to reach the one beneath) · "
             "drag inside to move · drag a vertex, corner, side or edge to "
             "reshape · polygon: double-click an edge to add a vertex, "
             "right-click a vertex to remove it · arrows nudge · Del deletes"),
    RECTANGLE: "Rectangle: press and drag",
    CIRCLE: "Circle: press at the centre and drag out the radius",
    POLYGON: "Polygon: click to place vertices (no limit)",
}


@dataclass
class _ImageMapping:
    """Cached transform between widget and frame pixel coordinates."""
    scale: float
    offset_x: int
    offset_y: int
    frame_w: int
    frame_h: int

    def widget_to_frame(self, wx: int, wy: int
                        ) -> tuple[int, int] | None:
        """Convert widget coords to frame coords. Returns None if the
        widget point is outside the frame (black bars / panned away)."""
        if self.scale <= 0:
            return None
        fx = (wx - self.offset_x) / self.scale
        fy = (wy - self.offset_y) / self.scale
        if 0 <= fx <= self.frame_w and 0 <= fy <= self.frame_h:
            return int(round(fx)), int(round(fy))
        return None

    def frame_to_widget(self, fx: float, fy: float
                        ) -> tuple[int, int]:
        """Convert frame coords to widget coords (for painting)."""
        return (
            int(round(fx * self.scale + self.offset_x)),
            int(round(fy * self.scale + self.offset_y)),
        )


@dataclass
class _DrawState:
    """The shape being drawn. Coordinates are FRAME pixels."""
    rect_start: tuple[int, int] | None = None
    rect_end: tuple[int, int] | None = None
    circle_center: tuple[int, int] | None = None
    circle_edge: tuple[int, int] | None = None
    poly_vertices: list[tuple[int, int]] = field(default_factory=list)
    poly_rubber_end: tuple[int, int] | None = None

    def reset(self) -> None:
        self.rect_start = None
        self.rect_end = None
        self.circle_center = None
        self.circle_edge = None
        self.poly_vertices = []
        self.poly_rubber_end = None


def canonical_geometry(kind: str, geom: dict) -> dict:
    """Normalise legacy per-axis geometry keys to the canonical form."""
    if kind == RECTANGLE:
        tl = geom.get("top_left") or (
            geom.get("topLeftX", 0), geom.get("topLeftY", 0))
        br = geom.get("bottom_right") or (
            geom.get("Bottom_right_X", 0), geom.get("Bottom_right_Y", 0))
        x1, x2 = sorted((int(tl[0]), int(br[0])))
        y1, y2 = sorted((int(tl[1]), int(br[1])))
        return {"top_left": [x1, y1], "bottom_right": [x2, y2]}
    if kind == CIRCLE:
        c = geom.get("center") or (
            geom.get("centerX", 0), geom.get("centerY", 0))
        return {"center": [int(c[0]), int(c[1])],
                "radius": int(geom.get("radius", 0))}
    if kind == POLYGON:
        return {"vertices": [[int(v[0]), int(v[1])]
                             for v in (geom.get("vertices") or [])]}
    return dict(geom)


def _dedupe_consecutive(verts: list) -> list[tuple[int, int]]:
    """Drop exact consecutive duplicates (incl. last == first)."""
    out: list[tuple[int, int]] = []
    for v in verts:
        t = (int(v[0]), int(v[1]))
        if not out or out[-1] != t:
            out.append(t)
    while len(out) > 1 and out[-1] == out[0]:
        out.pop()
    return out


def _dist_point_segment(p, a, b) -> tuple[float, float]:
    """Distance from p to segment ab, and the projection parameter t."""
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy), t


class ROICanvas(QWidget):
    """Native Qt widget for drawing and editing ROIs on a video frame."""

    shape_committed = Signal(str, dict)
    shape_cancelled = Signal()
    shape_edited = Signal(int, dict)
    shape_deleted = Signal(int)
    shape_selected = Signal(int)
    hint_changed = Signal(str)
    zoom_changed = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._frame_bgr: np.ndarray | None = None
        self._frame_pixmap: QPixmap | None = None
        self._mapping: _ImageMapping | None = None
        # [{kind, color_bgr, thickness, geometry, name?}]
        self._existing_rois: list[dict] = []
        self._show_labels = True

        self._tool: str | None = None
        self._mode = _DrawMode.IDLE
        self._draw_state = _DrawState()
        self._draw_color_bgr: tuple[int, int, int] = (0, 0, 255)
        self._draw_thickness: int = 3

        # Selection / edit state.
        self._selected_idx: int | None = None
        self._selected_handle: str | None = None
        self._drag_offset_frame: tuple[float, float] | None = None
        self._pre_drag_geom: dict | None = None
        self._drag_anchor_geom: dict | None = None
        self._press_frame: tuple[int, int] | None = None
        self._cycle_from: int | None = None

        # View state: zoom relative to fit, pan in widget px.
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._pan_last: QPointF | None = None

        self.setMinimumSize(480, 270)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setStyleSheet("background: #1a1a1a;")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def set_frame(self, bgr: np.ndarray | None) -> None:
        """Replace the displayed frame. Does NOT cancel an in-progress
        draw — geometry is in frame pixels, so a polygon can be continued
        after stepping to another frame."""
        same_size = (self._frame_bgr is not None and bgr is not None
                     and self._frame_bgr.shape[:2] == bgr.shape[:2])
        self._frame_bgr = bgr
        if bgr is None:
            self._frame_pixmap = None
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
            self._frame_pixmap = QPixmap.fromImage(qimg)
        if not same_size:
            self._zoom, self._pan = 1.0, QPointF(0.0, 0.0)
            self.zoom_changed.emit(self._zoom)
        self._recompute_mapping()
        self.update()

    def set_existing_rois(self, rois: list[dict]) -> None:
        """Replace the displayed ROIs. Each dict: ``kind``, ``color_bgr``
        (B,G,R), ``thickness``, ``geometry`` and optionally ``name``."""
        self._existing_rois = [dict(r) for r in rois]
        if (self._selected_idx is not None
                and self._selected_idx >= len(self._existing_rois)):
            self._selected_idx = None
        self.update()

    def set_show_labels(self, show: bool) -> None:
        self._show_labels = bool(show)
        self.update()

    def set_tool(self, tool: str | None,
                 color_bgr: tuple[int, int, int] | None = None,
                 thickness: int | None = None) -> None:
        """Arm a tool: ``"select"``, RECTANGLE, CIRCLE, POLYGON or None.

        Switching tools abandons a shape in progress (without emitting
        ``shape_cancelled``) and clears the selection when leaving select.
        """
        if color_bgr is not None:
            self._draw_color_bgr = tuple(color_bgr)
        if thickness is not None:
            self._draw_thickness = max(1, int(thickness))
        self._draw_state.reset()
        if tool not in TOOLS:
            tool = None
        if tool != SELECT and self._selected_idx is not None:
            self._selected_idx = None
            self.shape_selected.emit(-1)
        self._selected_handle = None
        self._tool = tool
        self._mode = _PENDING_FOR_TOOL.get(tool, _DrawMode.IDLE)
        self._update_cursor()
        self._emit_hint()
        if tool is not None:
            self.setFocus(Qt.OtherFocusReason)
        self.update()

    def tool(self) -> str | None:
        return self._tool

    def set_draw_style(self, color_bgr: tuple[int, int, int],
                       thickness: int) -> None:
        """Colour / thickness used for the live preview of new shapes."""
        self._draw_color_bgr = tuple(color_bgr)
        self._draw_thickness = max(1, int(thickness))
        self.update()

    def start_draw(self, kind: str, color_bgr: tuple[int, int, int],
                   thickness: int) -> None:
        """Compatibility alias for :meth:`set_tool` with a drawing tool."""
        self.set_tool(kind, color_bgr, thickness)

    def is_drawing(self) -> bool:
        """True while a shape is actually being drawn (not merely armed)."""
        return self._mode in (_DrawMode.RECT_DRAGGING,
                              _DrawMode.CIRCLE_DRAGGING,
                              _DrawMode.POLY_VERTEXING)

    def cancel_draw(self) -> None:
        """Abandon the shape in progress; the tool stays armed."""
        if self.is_drawing():
            self._abandon_shape(emit_signal=True)

    def start_select(self) -> None:
        self.set_tool(SELECT)

    def stop_select(self) -> None:
        if self.is_selecting():
            self.set_tool(None)

    def is_selecting(self) -> bool:
        return self._mode in (_DrawMode.SELECT, _DrawMode.SHAPE_MOVING,
                              _DrawMode.HANDLE_DRAGGING)

    def select_shape(self, idx: int | None) -> None:
        """Select a shape programmatically (e.g. from the ROI table).
        Does not emit ``shape_selected``."""
        if idx is not None and not (0 <= idx < len(self._existing_rois)):
            idx = None
        self._selected_idx = idx
        self._selected_handle = None
        self.update()

    def selected_index(self) -> int | None:
        return self._selected_idx

    def fit_view(self) -> None:
        self._zoom, self._pan = 1.0, QPointF(0.0, 0.0)
        self._recompute_mapping()
        self.zoom_changed.emit(self._zoom)
        self.update()

    def zoom_by(self, factor: float, anchor: QPointF | None = None) -> None:
        """Zoom by ``factor`` keeping the frame point under ``anchor``
        (widget coords; default: widget centre) fixed."""
        if self._frame_bgr is None or self._mapping is None:
            return
        new_zoom = max(_MIN_ZOOM, min(_MAX_ZOOM, self._zoom * factor))
        if new_zoom == self._zoom:
            return
        if anchor is None:
            anchor = QPointF(self.width() / 2.0, self.height() / 2.0)
        m = self._mapping
        fx = (anchor.x() - m.offset_x) / m.scale
        fy = (anchor.y() - m.offset_y) / m.scale
        self._zoom = new_zoom
        self._recompute_mapping()
        m2 = self._mapping
        self._pan += QPointF(anchor.x() - (fx * m2.scale + m2.offset_x),
                             anchor.y() - (fy * m2.scale + m2.offset_y))
        self._recompute_mapping()
        self.zoom_changed.emit(self._zoom)
        self.update()

    # ------------------------------------------------------------------ #
    # Geometry helpers
    # ------------------------------------------------------------------ #
    def _canonical_geom(self, kind: str, geom: dict) -> dict:
        return canonical_geometry(kind, geom)

    def _geom(self, idx: int) -> tuple[str, dict]:
        roi = self._existing_rois[idx]
        kind = roi.get("kind")
        return kind, canonical_geometry(kind, roi.get("geometry", {}))

    def _tol_frame(self, widget_px: float) -> float:
        m = self._mapping
        return widget_px / max(m.scale, 1e-6) if m else widget_px

    def _handles(self, kind: str, geom: dict) -> dict[str, tuple[float, float]]:
        """Handle name -> frame position, for the given shape."""
        if kind == RECTANGLE:
            (x1, y1), (x2, y2) = geom["top_left"], geom["bottom_right"]
            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            return {"nw": (x1, y1), "n": (mx, y1), "ne": (x2, y1),
                    "e": (x2, my), "se": (x2, y2), "s": (mx, y2),
                    "sw": (x1, y2), "w": (x1, my)}
        if kind == CIRCLE:
            c, r = geom["center"], geom["radius"]
            return {"edge": (c[0] + r, c[1])}
        if kind == POLYGON:
            return {f"v{i}": (v[0], v[1])
                    for i, v in enumerate(geom["vertices"])}
        return {}

    def _edge_hit(self, kind: str, geom: dict, fx: float, fy: float,
                  tol: float) -> tuple[float, str] | None:
        """Nearest point of the shape's OUTLINE within ``tol``: returns
        (distance, handle) — a rectangle side ("n"/"e"/"s"/"w"), the
        circle rim ("edge") or a polygon edge ("pe<i>": vertex i to i+1)."""
        best: tuple[float, str] | None = None

        def consider(d: float, h: str) -> None:
            nonlocal best
            if d <= tol and (best is None or d < best[0]):
                best = (d, h)

        if kind == RECTANGLE:
            (x1, y1), (x2, y2) = geom["top_left"], geom["bottom_right"]
            for h, a_, b_ in (("n", (x1, y1), (x2, y1)), ("s", (x1, y2), (x2, y2)),
                              ("w", (x1, y1), (x1, y2)), ("e", (x2, y1), (x2, y2))):
                consider(_dist_point_segment((fx, fy), a_, b_)[0], h)
        elif kind == CIRCLE:
            c, r = geom["center"], geom["radius"]
            consider(abs(math.hypot(fx - c[0], fy - c[1]) - r), "edge")
        elif kind == POLYGON:
            v = geom["vertices"]
            n = len(v)
            for i in range(n):
                consider(_dist_point_segment((fx, fy), v[i], v[(i + 1) % n])[0],
                         f"pe{i}")
        return best

    def _interior_hit(self, kind: str, geom: dict, fx: float, fy: float) -> bool:
        if kind == RECTANGLE:
            (x1, y1), (x2, y2) = geom["top_left"], geom["bottom_right"]
            return x1 <= fx <= x2 and y1 <= fy <= y2
        if kind == CIRCLE:
            c, r = geom["center"], geom["radius"]
            return math.hypot(fx - c[0], fy - c[1]) <= r
        if kind == POLYGON:
            return self._point_in_polygon(fx, fy, geom["vertices"])
        return False

    @staticmethod
    def _area(kind: str, geom: dict) -> float:
        if kind == RECTANGLE:
            (x1, y1), (x2, y2) = geom["top_left"], geom["bottom_right"]
            return float((x2 - x1) * (y2 - y1))
        if kind == CIRCLE:
            return math.pi * geom["radius"] ** 2
        v = np.asarray(geom["vertices"], float)
        if len(v) < 3:
            return 0.0
        x, y = v[:, 0], v[:, 1]
        return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))

    def _interior_candidates(self, frame_pt) -> list[int]:
        """Shapes whose interior contains the point, smallest first (so a
        small ROI under a larger, newer one is still reachable)."""
        fx, fy = frame_pt
        hits = []
        for idx in range(len(self._existing_rois)):
            kind, geom = self._geom(idx)
            if self._interior_hit(kind, geom, fx, fy):
                hits.append((self._area(kind, geom), -idx, idx))
        return [idx for _, _, idx in sorted(hits)]

    def _hit_test(
        self, frame_pt: tuple[float, float]
    ) -> tuple[int | None, str | None]:
        """Shape (and handle) under the cursor.

        Patch 122im — z-order no longer hides anything. Priority:

        1. vertex / corner / rim handles (the selected shape's first, then
           every shape's) — a handle is always grabbable;
        2. the nearest OUTLINE of any shape — rectangle sides, circle rims,
           polygon edges — so an edge can be dragged even where another ROI
           covers it;
        3. interiors, smallest area first, so an ROI inside a larger, newer
           one is selected by clicking inside it. Clicking the same spot
           again cycles through the overlapping shapes (see
           mouseReleaseEvent).

        Returns (idx, handle) — handle None for an interior hit — or
        (None, None) on a miss.
        """
        if self._mapping is None:
            return (None, None)
        fx, fy = frame_pt
        tol = self._tol_frame(_HANDLE_HIT_TOLERANCE_WIDGET_PX)
        n = len(self._existing_rois)

        def handle_at(idx: int) -> str | None:
            kind, geom = self._geom(idx)
            best = None
            for name, (hx, hy) in self._handles(kind, geom).items():
                d = max(abs(fx - hx), abs(fy - hy))
                if d <= tol and (best is None or d < best[0]):
                    best = (d, name)
            return best[1] if best else None

        order = list(range(n - 1, -1, -1))
        if self._selected_idx is not None and self._selected_idx < n:
            order.remove(self._selected_idx)
            order.insert(0, self._selected_idx)
        for idx in order:
            h = handle_at(idx)
            if h is not None:
                return (idx, h)

        edge_tol = self._tol_frame(_EDGE_HIT_TOLERANCE_WIDGET_PX)
        best_edge: tuple[float, int, str] | None = None
        for idx in order:
            kind, geom = self._geom(idx)
            e = self._edge_hit(kind, geom, fx, fy, edge_tol)
            if e is not None and (best_edge is None or e[0] < best_edge[0]):
                best_edge = (e[0], idx, e[1])
        if best_edge is not None:
            return (best_edge[1], best_edge[2])

        cands = self._interior_candidates(frame_pt)
        return (cands[0], None) if cands else (None, None)

    @staticmethod
    def _point_in_polygon(px: float, py: float, verts: list) -> bool:
        """Ray-cast point-in-polygon test."""
        n = len(verts)
        if n < 3:
            return False
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = verts[i][0], verts[i][1]
            xj, yj = verts[j][0], verts[j][1]
            if ((yi > py) != (yj > py)) and (
                px < (xj - xi) * (py - yi) / ((yj - yi) or 1e-9) + xi
            ):
                inside = not inside
            j = i
        return inside

    def _apply_edit_geom(self, idx: int, new_geom: dict) -> None:
        """Update the displayed shape during a drag (emitted on release)."""
        if 0 <= idx < len(self._existing_rois):
            roi = self._existing_rois[idx]
            roi["geometry"] = canonical_geometry(roi.get("kind"), new_geom)
            self.update()

    def _commit_edit(self, idx: int, new_geom: dict) -> None:
        """Apply and emit an edit in one step (keyboard / vertex ops)."""
        self._apply_edit_geom(idx, new_geom)
        kind, geom = self._geom(idx)
        self.shape_edited.emit(idx, geom)

    def _clamp_frame(self, fx: float, fy: float) -> tuple[int, int]:
        m = self._mapping
        return (int(round(max(0, min(m.frame_w, fx)))),
                int(round(max(0, min(m.frame_h, fy)))))

    def _event_frame_pt(self, ev, clamp: bool) -> tuple[int, int] | None:
        m = self._mapping
        if m is None:
            return None
        wx, wy = ev.position().x(), ev.position().y()
        pt = m.widget_to_frame(wx, wy)
        if pt is None and clamp:
            pt = self._clamp_frame((wx - m.offset_x) / m.scale,
                                   (wy - m.offset_y) / m.scale)
        return pt

    # ------------------------------------------------------------------ #
    # Coordinate mapping
    # ------------------------------------------------------------------ #
    def _recompute_mapping(self) -> None:
        if self._frame_bgr is None:
            self._mapping = None
            return
        h, w = self._frame_bgr.shape[:2]
        wW, wH = max(1, self.width()), max(1, self.height())
        scale = min(wW / w, wH / h) * self._zoom
        new_w, new_h = w * scale, h * scale
        if self._zoom <= 1.0:
            self._pan = QPointF(0.0, 0.0)
        else:
            # Keep at least half the widget covered by the frame.
            max_px = max(0.0, (new_w - wW) / 2.0 + wW / 2.0)
            max_py = max(0.0, (new_h - wH) / 2.0 + wH / 2.0)
            self._pan = QPointF(max(-max_px, min(max_px, self._pan.x())),
                                max(-max_py, min(max_py, self._pan.y())))
        self._mapping = _ImageMapping(
            scale=scale,
            offset_x=int(round((wW - new_w) / 2.0 + self._pan.x())),
            offset_y=int(round((wH - new_h) / 2.0 + self._pan.y())),
            frame_w=w, frame_h=h,
        )

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        self._recompute_mapping()
        self.update()

    # ------------------------------------------------------------------ #
    # Painting
    # ------------------------------------------------------------------ #
    def _pen_width(self, thickness: int) -> int:
        """ROI thickness is in frame px; show it scaled, within reason."""
        s = self._mapping.scale if self._mapping else 1.0
        return max(1, int(round(thickness * min(s, 2.0))))

    def paintEvent(self, ev: QPaintEvent) -> None:
        p = QPainter(self)
        try:
            p.fillRect(self.rect(), QColor(26, 26, 26))
            if self._frame_pixmap is None or self._mapping is None:
                p.setPen(QColor(140, 140, 140))
                p.drawText(self.rect(), Qt.AlignCenter, "No frame")
                return
            m = self._mapping
            p.setRenderHint(QPainter.SmoothPixmapTransform, m.scale < 4.0)
            p.drawPixmap(QRect(m.offset_x, m.offset_y,
                               int(round(m.frame_w * m.scale)),
                               int(round(m.frame_h * m.scale))),
                         self._frame_pixmap, self._frame_pixmap.rect())
            p.setRenderHint(QPainter.Antialiasing, True)
            for idx, roi in enumerate(self._existing_rois):
                self._paint_existing_roi(p, roi)
                if self.is_selecting() and idx == self._selected_idx:
                    self._paint_selection_chrome(p, roi)
            self._paint_in_progress(p)
        finally:
            p.end()

    def _paint_existing_roi(self, p: QPainter, roi: dict) -> None:
        m = self._mapping
        kind = roi.get("kind")
        geom = canonical_geometry(kind, roi.get("geometry", {}))
        color = self._bgr_to_qcolor(roi.get("color_bgr", (0, 0, 255)))
        pen = QPen(color)
        pen.setWidth(self._pen_width(roi.get("thickness", 2)))
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        label_at: tuple[int, int] | None = None
        if kind == RECTANGLE:
            x1, y1 = m.frame_to_widget(*geom["top_left"])
            x2, y2 = m.frame_to_widget(*geom["bottom_right"])
            p.drawRect(QRect(x1, y1, x2 - x1, y2 - y1))
            label_at = (x1 + 3, y1 - 4)
        elif kind == CIRCLE:
            wcx, wcy = m.frame_to_widget(*geom["center"])
            wr = int(round(geom["radius"] * m.scale))
            p.drawEllipse(QPoint(wcx, wcy), wr, wr)
            label_at = (wcx - wr + 3, wcy - wr - 4)
        elif kind == POLYGON:
            verts = geom["vertices"]
            if len(verts) >= 2:
                p.drawPolygon(QPolygonF([QPointF(*m.frame_to_widget(*v))
                                         for v in verts]))
                top = min(verts, key=lambda v: v[1])
                wx, wy = m.frame_to_widget(*top)
                label_at = (wx + 3, wy - 4)
        name = roi.get("name")
        if self._show_labels and name and label_at is not None:
            f = QFont(self.font())
            f.setPointSizeF(max(7.0, f.pointSizeF() - 1))
            f.setBold(True)
            p.setFont(f)
            x, y = label_at
            y = max(y, 12)
            p.setPen(QColor(0, 0, 0, 200))
            p.drawText(QPoint(x + 1, y + 1), name)
            p.setPen(color)
            p.drawText(QPoint(x, y), name)

    def _draw_handle(self, p: QPainter, wx: float, wy: float,
                     filled: bool = True) -> None:
        r = QRect(int(wx - _HANDLE_BOX_HALF_PX), int(wy - _HANDLE_BOX_HALF_PX),
                  _HANDLE_BOX_HALF_PX * 2, _HANDLE_BOX_HALF_PX * 2)
        p.setPen(QPen(QColor(0, 0, 0), 1))
        p.setBrush(_SELECTION_COLOR if filled else Qt.NoBrush)
        p.drawRect(r)
        p.setBrush(Qt.NoBrush)

    def _paint_selection_chrome(self, p: QPainter, roi: dict) -> None:
        """Dashed outline plus the shape's handles."""
        m = self._mapping
        kind = roi.get("kind")
        geom = canonical_geometry(kind, roi.get("geometry", {}))
        pen = QPen(_SELECTION_COLOR)
        pen.setWidth(1)
        pen.setStyle(Qt.DashLine)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        if kind == RECTANGLE:
            x1, y1 = m.frame_to_widget(*geom["top_left"])
            x2, y2 = m.frame_to_widget(*geom["bottom_right"])
            p.drawRect(QRect(x1 - 3, y1 - 3, (x2 - x1) + 6, (y2 - y1) + 6))
        elif kind == CIRCLE:
            wcx, wcy = m.frame_to_widget(*geom["center"])
            wr = int(round(geom["radius"] * m.scale))
            p.drawEllipse(QPoint(wcx, wcy), wr + 3, wr + 3)
            p.drawLine(QPoint(wcx - 4, wcy), QPoint(wcx + 4, wcy))
            p.drawLine(QPoint(wcx, wcy - 4), QPoint(wcx, wcy + 4))
        elif kind == POLYGON and len(geom["vertices"]) >= 2:
            p.drawPolygon(QPolygonF([QPointF(*m.frame_to_widget(*v))
                                     for v in geom["vertices"]]))
        for name, (hx, hy) in self._handles(kind, geom).items():
            wx, wy = m.frame_to_widget(hx, hy)
            self._draw_handle(p, wx, wy, filled=name != self._selected_handle)

    def _paint_in_progress(self, p: QPainter) -> None:
        m = self._mapping
        color = self._bgr_to_qcolor(self._draw_color_bgr)
        pen = QPen(color)
        pen.setWidth(_LIVE_LINE_WIDTH)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        s = self._draw_state

        if self._mode == _DrawMode.RECT_DRAGGING and s.rect_start and s.rect_end:
            x1, y1 = m.frame_to_widget(*s.rect_start)
            x2, y2 = m.frame_to_widget(*s.rect_end)
            p.drawRect(QRect(min(x1, x2), min(y1, y2),
                             abs(x2 - x1), abs(y2 - y1)))
            self._paint_readout(p, (max(x1, x2), max(y1, y2)),
                                f"{abs(s.rect_end[0] - s.rect_start[0])} × "
                                f"{abs(s.rect_end[1] - s.rect_start[1])} px")

        elif (self._mode == _DrawMode.CIRCLE_DRAGGING
              and s.circle_center and s.circle_edge):
            cx, cy = m.frame_to_widget(*s.circle_center)
            r_frame = math.hypot(s.circle_edge[0] - s.circle_center[0],
                                 s.circle_edge[1] - s.circle_center[1])
            r = int(round(r_frame * m.scale))
            p.drawEllipse(QPoint(cx, cy), r, r)
            p.drawLine(QPoint(cx - 4, cy), QPoint(cx + 4, cy))
            p.drawLine(QPoint(cx, cy - 4), QPoint(cx, cy + 4))
            self._paint_readout(p, (cx + r, cy), f"r = {int(round(r_frame))} px")

        elif self._mode == _DrawMode.POLY_VERTEXING and s.poly_vertices:
            pts = [QPointF(*m.frame_to_widget(*v)) for v in s.poly_vertices]
            for a, b in zip(pts, pts[1:], strict=False):
                p.drawLine(a, b)
            if s.poly_rubber_end is not None:
                rubber = QPen(color)
                rubber.setWidth(_LIVE_LINE_WIDTH)
                rubber.setStyle(Qt.DashLine)
                p.setPen(rubber)
                end = QPointF(*m.frame_to_widget(*s.poly_rubber_end))
                p.drawLine(pts[-1], end)
                if len(pts) >= 2:
                    closing = QPen(color)
                    closing.setWidth(1)
                    closing.setStyle(Qt.DotLine)
                    p.setPen(closing)
                    p.drawLine(end, pts[0])
            p.setPen(QPen(color, 1))
            p.setBrush(color)
            for pt in pts:
                p.drawEllipse(pt, _VERTEX_DOT_RADIUS, _VERTEX_DOT_RADIUS)
            p.setBrush(Qt.NoBrush)
            # Ring around the first vertex when a click would close.
            if len(pts) >= 3 and self._near_first_vertex(s.poly_rubber_end):
                ring = QPen(_SELECTION_COLOR)
                ring.setWidth(2)
                p.setPen(ring)
                p.drawEllipse(pts[0], _CLOSE_SNAP_WIDGET_PX,
                              _CLOSE_SNAP_WIDGET_PX)
            self._paint_readout(p, (int(pts[-1].x()), int(pts[-1].y())),
                                f"{len(pts)} vertices")

    def _paint_readout(self, p: QPainter, at: tuple[int, int],
                       text: str) -> None:
        x, y = at[0] + 8, at[1] + 16
        fm = p.fontMetrics()
        r = fm.boundingRect(text).adjusted(-4, -2, 4, 2)
        r.moveTo(x - 4, y - fm.ascent() - 2)
        p.fillRect(r, QColor(0, 0, 0, 170))
        p.setPen(QColor(235, 235, 235))
        p.drawText(QPoint(x, y), text)

    def _near_first_vertex(self, frame_pt) -> bool:
        verts = self._draw_state.poly_vertices
        if frame_pt is None or not verts or self._mapping is None:
            return False
        a = self._mapping.frame_to_widget(*verts[0])
        b = self._mapping.frame_to_widget(*frame_pt)
        return math.hypot(a[0] - b[0], a[1] - b[1]) <= _CLOSE_SNAP_WIDGET_PX

    # ------------------------------------------------------------------ #
    # Mouse
    # ------------------------------------------------------------------ #
    def mousePressEvent(self, ev: QMouseEvent) -> None:
        if self._mapping is None:
            return
        if ev.button() == Qt.MiddleButton:
            self._pan_last = ev.position()
            self.setCursor(Qt.ClosedHandCursor)
            return
        if ev.button() == Qt.RightButton:
            self._on_right_click(ev)
            return
        if ev.button() != Qt.LeftButton:
            return
        self.setFocus(Qt.MouseFocusReason)

        if self._mode == _DrawMode.SELECT:
            frame_pt = self._event_frame_pt(ev, clamp=True)
            idx, handle = self._hit_test(frame_pt)
            if idx is None:
                if self._selected_idx is not None:
                    self._selected_idx = None
                    self._selected_handle = None
                    self.shape_selected.emit(-1)
                self.update()
                return
            # Clicking again inside the selected shape (not on a handle or
            # outline) may mean "the one underneath": decided on release,
            # if the mouse didn't move.
            self._cycle_from = (idx if handle is None
                                and self._selected_idx is not None
                                and self._selected_idx in
                                self._interior_candidates(frame_pt) else None)
            if self._cycle_from is not None:
                idx = self._selected_idx
            self._press_frame = frame_pt
            if idx != self._selected_idx:
                self._selected_idx = idx
                self.shape_selected.emit(idx)
            self._selected_handle = handle
            kind, geom = self._geom(idx)
            self._pre_drag_geom = dict(geom)
            self._drag_anchor_geom = dict(geom)
            if handle is None:
                ref = self._reference_point_for_drag(idx)
                self._drag_offset_frame = (frame_pt[0] - ref[0],
                                           frame_pt[1] - ref[1])
                self._mode = _DrawMode.SHAPE_MOVING
            else:
                self._drag_offset_frame = None
                self._mode = _DrawMode.HANDLE_DRAGGING
            self._update_cursor(handle if handle is not None else "body")
            self.update()
            return

        frame_pt = self._event_frame_pt(ev, clamp=False)
        if frame_pt is None:
            if self._mode != _DrawMode.POLY_VERTEXING:
                return
            frame_pt = self._event_frame_pt(ev, clamp=True)

        if self._mode == _DrawMode.RECT_PENDING:
            self._draw_state.rect_start = frame_pt
            self._draw_state.rect_end = frame_pt
            self._mode = _DrawMode.RECT_DRAGGING
        elif self._mode == _DrawMode.CIRCLE_PENDING:
            self._draw_state.circle_center = frame_pt
            self._draw_state.circle_edge = frame_pt
            self._mode = _DrawMode.CIRCLE_DRAGGING
        elif self._mode == _DrawMode.POLY_PENDING:
            self._draw_state.poly_vertices = [frame_pt]
            self._draw_state.poly_rubber_end = frame_pt
            self._mode = _DrawMode.POLY_VERTEXING
            self._emit_hint()
        elif self._mode == _DrawMode.POLY_VERTEXING:
            verts = self._draw_state.poly_vertices
            if len(verts) >= 3 and self._near_first_vertex(frame_pt):
                self._commit_polygon()
                return
            last_w = self._mapping.frame_to_widget(*verts[-1])
            if math.hypot(ev.position().x() - last_w[0],
                          ev.position().y() - last_w[1]) <= _DUPLICATE_SNAP_WIDGET_PX:
                return
            verts.append(frame_pt)
            self._emit_hint()
        self.update()

    def _on_right_click(self, ev: QMouseEvent) -> None:
        """Right-click: remove a vertex of the selected polygon; while
        drawing a polygon, remove the last vertex."""
        if self._mode == _DrawMode.POLY_VERTEXING:
            self._pop_vertex()
            return
        if self._mode != _DrawMode.SELECT or self._selected_idx is None:
            return
        frame_pt = self._event_frame_pt(ev, clamp=True)
        idx, handle = self._hit_test(frame_pt)
        if idx != self._selected_idx or not (handle or "").startswith("v"):
            return
        kind, geom = self._geom(idx)
        verts = geom["vertices"]
        if len(verts) <= 3:
            self.hint_changed.emit("A polygon needs at least 3 vertices.")
            return
        del verts[int(handle[1:])]
        self._commit_edit(idx, {"vertices": verts})

    def _reference_point_for_drag(self, idx: int) -> tuple[float, float]:
        kind, geom = self._geom(idx)
        if kind == RECTANGLE:
            return tuple(geom["top_left"])
        if kind == CIRCLE:
            return tuple(geom["center"])
        verts = geom.get("vertices") or [[0, 0]]
        return tuple(verts[0])

    def mouseDoubleClickEvent(self, ev: QMouseEvent) -> None:
        if ev.button() != Qt.LeftButton or self._mapping is None:
            return super().mouseDoubleClickEvent(ev)
        if self._mode == _DrawMode.POLY_VERTEXING:
            # The first click of the double-click already placed a vertex.
            if len(self._draw_state.poly_vertices) >= 3:
                self._commit_polygon()
            return
        if self._mode == _DrawMode.SELECT and self._selected_idx is not None:
            kind, geom = self._geom(self._selected_idx)
            if kind != POLYGON:
                return
            pt = self._event_frame_pt(ev, clamp=True)
            verts = geom["vertices"]
            n = len(verts)
            best = min(range(n), key=lambda i: _dist_point_segment(
                pt, verts[i], verts[(i + 1) % n])[0])
            d, _ = _dist_point_segment(pt, verts[best], verts[(best + 1) % n])
            if d <= self._tol_frame(_EDGE_HIT_TOLERANCE_WIDGET_PX):
                verts.insert(best + 1, [int(pt[0]), int(pt[1])])
                self._selected_handle = f"v{best + 1}"
                self._commit_edit(self._selected_idx, {"vertices": verts})
            return
        super().mouseDoubleClickEvent(ev)

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:
        if self._mapping is None:
            return
        if self._pan_last is not None:
            d = ev.position() - self._pan_last
            self._pan_last = ev.position()
            self._pan += d
            self._recompute_mapping()
            self.update()
            return
        if self._mode == _DrawMode.SELECT:
            pt = self._event_frame_pt(ev, clamp=True)
            idx, handle = self._hit_test(pt)
            self._update_cursor(None if idx is None
                                else (handle if handle is not None else "body"))
            return
        if self._mode not in (_DrawMode.RECT_DRAGGING,
                              _DrawMode.CIRCLE_DRAGGING,
                              _DrawMode.POLY_VERTEXING,
                              _DrawMode.SHAPE_MOVING,
                              _DrawMode.HANDLE_DRAGGING):
            return
        frame_pt = self._event_frame_pt(ev, clamp=True)

        if self._mode == _DrawMode.RECT_DRAGGING:
            self._draw_state.rect_end = frame_pt
        elif self._mode == _DrawMode.CIRCLE_DRAGGING:
            self._draw_state.circle_edge = frame_pt
        elif self._mode == _DrawMode.POLY_VERTEXING:
            self._draw_state.poly_rubber_end = frame_pt
        elif self._mode == _DrawMode.SHAPE_MOVING:
            self._move_selected_to(frame_pt)
            return
        elif self._mode == _DrawMode.HANDLE_DRAGGING:
            self._drag_handle_to(frame_pt)
            return
        self.update()

    def _move_selected_to(self, frame_pt) -> None:
        if self._selected_idx is None or self._drag_offset_frame is None:
            return
        kind, geom = self._geom(self._selected_idx)
        ref = self._reference_point_for_drag(self._selected_idx)
        tx = frame_pt[0] - self._drag_offset_frame[0]
        ty = frame_pt[1] - self._drag_offset_frame[1]
        dx, dy = int(round(tx - ref[0])), int(round(ty - ref[1]))
        self._apply_edit_geom(self._selected_idx,
                              self._translated(kind, geom, dx, dy))

    @staticmethod
    def _translated(kind: str, geom: dict, dx: int, dy: int) -> dict:
        if kind == RECTANGLE:
            (x1, y1), (x2, y2) = geom["top_left"], geom["bottom_right"]
            return {"top_left": [x1 + dx, y1 + dy],
                    "bottom_right": [x2 + dx, y2 + dy]}
        if kind == CIRCLE:
            c = geom["center"]
            return {"center": [c[0] + dx, c[1] + dy], "radius": geom["radius"]}
        return {"vertices": [[v[0] + dx, v[1] + dy] for v in geom["vertices"]]}

    def _drag_handle_to(self, frame_pt) -> None:
        if self._selected_idx is None or self._selected_handle is None:
            return
        idx, h = self._selected_idx, self._selected_handle
        kind, geom = self._geom(idx)
        fx, fy = self._clamp_frame(*frame_pt)
        if kind == RECTANGLE:
            # Resize against the geometry captured at press time so a
            # handle dragged past the opposite edge flips cleanly.
            a = self._drag_anchor_geom or geom
            (x1, y1), (x2, y2) = a["top_left"], a["bottom_right"]
            if "w" in h:
                x1 = fx
            if "e" in h:
                x2 = fx
            if "n" in h:
                y1 = fy
            if "s" in h:
                y2 = fy
            new = {"top_left": [x1, y1], "bottom_right": [x2, y2]}
        elif kind == CIRCLE:
            c = geom["center"]
            new = {"center": c,
                   "radius": max(1, int(round(math.hypot(fx - c[0], fy - c[1]))))}
        elif kind == POLYGON and h.startswith("v"):
            verts = geom["vertices"]
            verts[int(h[1:])] = [fx, fy]
            new = {"vertices": verts}
        elif kind == POLYGON and h.startswith("pe"):
            # Drag a polygon edge: both its vertices follow the cursor.
            a = (self._drag_anchor_geom or geom)["vertices"]
            px, py = self._press_frame or (fx, fy)
            dx, dy = int(round(fx - px)), int(round(fy - py))
            i = int(h[2:])
            j = (i + 1) % len(a)
            verts = [list(v) for v in a]
            verts[i] = [a[i][0] + dx, a[i][1] + dy]
            verts[j] = [a[j][0] + dx, a[j][1] + dy]
            new = {"vertices": verts}
        else:
            return
        self._apply_edit_geom(idx, new)

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:
        if ev.button() == Qt.MiddleButton and self._pan_last is not None:
            self._pan_last = None
            self._update_cursor()
            return
        if ev.button() != Qt.LeftButton:
            return
        if self._mode == _DrawMode.RECT_DRAGGING:
            self._commit_rectangle()
        elif self._mode == _DrawMode.CIRCLE_DRAGGING:
            self._commit_circle()
        elif self._mode in (_DrawMode.SHAPE_MOVING, _DrawMode.HANDLE_DRAGGING):
            if self._selected_idx is not None:
                idx = self._selected_idx
                kind, final_geom = self._geom(idx)
                moved = (self._pre_drag_geom is not None and
                         canonical_geometry(kind, self._pre_drag_geom)
                         != final_geom)
                if not moved and getattr(self, "_cycle_from", None) is not None:
                    cands = self._interior_candidates(self._press_frame)
                    if len(cands) > 1:
                        nxt = cands[(cands.index(idx) + 1) % len(cands)]
                        self._selected_idx = nxt
                        self.shape_selected.emit(nxt)
                self._cycle_from = None
                if (self._pre_drag_geom is not None
                        and canonical_geometry(kind, self._pre_drag_geom)
                        != final_geom):
                    self.shape_edited.emit(idx, final_geom)
            self._mode = _DrawMode.SELECT
            self._drag_offset_frame = None
            self._pre_drag_geom = None
            self._drag_anchor_geom = None
            self._update_cursor()
            self.update()

    def wheelEvent(self, ev: QWheelEvent) -> None:
        steps = ev.angleDelta().y() / 120.0
        if steps == 0 or self._mapping is None:
            return super().wheelEvent(ev)
        self.zoom_by(1.25 ** steps, ev.position())
        ev.accept()

    # ------------------------------------------------------------------ #
    # Keyboard
    # ------------------------------------------------------------------ #
    def keyPressEvent(self, ev: QKeyEvent) -> None:
        key = ev.key()
        if key in (Qt.Key_F, Qt.Key_0) and not ev.modifiers():
            self.fit_view()
            return
        if key in (Qt.Key_Plus, Qt.Key_Equal):
            self.zoom_by(1.25)
            return
        if key == Qt.Key_Minus:
            self.zoom_by(0.8)
            return

        if self._mode == _DrawMode.SELECT:
            if key in (Qt.Key_Delete, Qt.Key_Backspace) and self._selected_idx is not None:
                idx = self._selected_idx
                self._selected_idx = None
                self._selected_handle = None
                self.update()
                self.shape_deleted.emit(idx)
                return
            if key == Qt.Key_Escape:
                if self._selected_idx is not None:
                    self._selected_idx = None
                    self._selected_handle = None
                    self.shape_selected.emit(-1)
                    self.update()
                return
            arrows = {Qt.Key_Left: (-1, 0), Qt.Key_Right: (1, 0),
                      Qt.Key_Up: (0, -1), Qt.Key_Down: (0, 1)}
            if key in arrows and self._selected_idx is not None:
                step = 10 if ev.modifiers() & Qt.ShiftModifier else 1
                dx, dy = arrows[key]
                kind, geom = self._geom(self._selected_idx)
                self._commit_edit(self._selected_idx,
                                  self._translated(kind, geom, dx * step, dy * step))
                return
            return super().keyPressEvent(ev)

        if self._mode == _DrawMode.POLY_VERTEXING:
            if key in (Qt.Key_Return, Qt.Key_Enter):
                if len(self._draw_state.poly_vertices) >= 3:
                    self._commit_polygon()
                else:
                    self.hint_changed.emit(
                        "A polygon needs at least 3 vertices — keep "
                        "clicking, or Esc to abandon it.")
                return
            if key == Qt.Key_Backspace:
                self._pop_vertex()
                return
        if key == Qt.Key_Escape and self.is_drawing():
            self._abandon_shape(emit_signal=True)
            return
        super().keyPressEvent(ev)

    def _pop_vertex(self) -> None:
        verts = self._draw_state.poly_vertices
        if verts:
            verts.pop()
        if not verts:
            self._mode = _DrawMode.POLY_PENDING
            self._draw_state.reset()
        self._emit_hint()
        self.update()

    # ------------------------------------------------------------------ #
    # Commit / cancel
    # ------------------------------------------------------------------ #
    def _rearm(self) -> None:
        """Back to the armed state of the current tool (sticky tools)."""
        self._draw_state.reset()
        self._mode = _PENDING_FOR_TOOL.get(self._tool, _DrawMode.IDLE)
        self._update_cursor()
        self._emit_hint()
        self.update()

    def _commit_rectangle(self) -> None:
        s = self._draw_state
        (x1, y1), (x2, y2) = s.rect_start, s.rect_end
        if abs(x2 - x1) < 2 or abs(y2 - y1) < 2:
            self._abandon_shape(emit_signal=True)
            return
        geom = {"top_left": [min(x1, x2), min(y1, y2)],
                "bottom_right": [max(x1, x2), max(y1, y2)]}
        self._rearm()
        self.shape_committed.emit(RECTANGLE, geom)

    def _commit_circle(self) -> None:
        s = self._draw_state
        (cx, cy), (ex, ey) = s.circle_center, s.circle_edge
        r = int(round(math.hypot(ex - cx, ey - cy)))
        if r < 2:
            self._abandon_shape(emit_signal=True)
            return
        self._rearm()
        self.shape_committed.emit(CIRCLE, {"center": [cx, cy], "radius": r})

    def _commit_polygon(self) -> None:
        # Patch 122ij: no simplification. Shapely's simplify(tolerance=20)
        # used to run here and silently delete vertices — the cause of the
        # "polygon can only have ~6 vertices" behaviour.
        verts = _dedupe_consecutive(self._draw_state.poly_vertices)
        if len(verts) < 3:
            self._abandon_shape(emit_signal=True)
            return
        self._rearm()
        self.shape_committed.emit(
            POLYGON, {"vertices": [[x, y] for x, y in verts]})

    def _abandon_shape(self, emit_signal: bool) -> None:
        self._rearm()
        if emit_signal:
            self.shape_cancelled.emit()

    # Kept for older callers.
    def _cancel_draw(self, emit_signal: bool) -> None:
        self._abandon_shape(emit_signal)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _emit_hint(self) -> None:
        if self._mode == _DrawMode.POLY_VERTEXING:
            n = len(self._draw_state.poly_vertices)
            text = (f"Polygon: {n} {'vertex' if n == 1 else 'vertices'} — keep "
                    f"clicking; " + ("finish: click the first vertex, "
                                     "double-click or Enter · "
                                     if n >= 3 else "")
                    + "Backspace/right-click removes the last · Esc abandons")
        else:
            text = _TOOL_HINTS.get(self._tool, "")
        if text:
            text += " · wheel zooms, middle-drag pans, F fits"
        self.hint_changed.emit(text)

    def _update_cursor(self, hover: str | None = None) -> None:
        if self._tool in (RECTANGLE, CIRCLE, POLYGON):
            self.setCursor(Qt.CrossCursor)
        elif self._tool == SELECT:
            if hover == "body":
                self.setCursor(Qt.SizeAllCursor)
            elif hover is not None and hover.startswith("v"):
                self.setCursor(Qt.PointingHandCursor)
            elif hover is not None and hover.startswith("pe"):
                self.setCursor(Qt.SizeAllCursor)
            elif hover in _HANDLE_CURSORS:
                self.setCursor(_HANDLE_CURSORS[hover])
            else:
                self.setCursor(Qt.ArrowCursor)
        else:
            self.unsetCursor()

    @staticmethod
    def _bgr_to_qcolor(bgr: tuple[int, int, int]) -> QColor:
        b, g, r = bgr
        return QColor(int(r), int(g), int(b))


__all__ = ["ROICanvas", "SELECT", "TOOLS", "canonical_geometry"]
