"""
mufasa.ui_qt.dialogs.pixel_calibration
=======================================

Qt-native dialog for deriving pixels-per-mm from a video frame.

Replaces the OpenCV-based ``CalculatePixelDistanceTool`` workflow
when called from the Data Import → Video parameters & calibration
section of the workbench. The OpenCV widget had two issues:

1. **Truncated instruction text** — the side panel that showed
   "Press ESC to proceed" / "Double-click to move circle" got
   clipped on the user's display, leaving them with a window
   that displayed only the question "Are you happy with the
   displayed choice?" and no visible way to confirm.

2. **No native confirm/cancel** — the workflow used ESC for
   confirm and double-clicks for everything else. Modal-but-
   not-really, no Cancel button, surprising semantics.

This dialog uses a QDialog with explicit OK/Cancel buttons,
inline distance/ppm readouts, and a single-click point-placement
flow. The original OpenCV class is left in place for any other
callers and as a fallback path.

Workflow
--------

1. Dialog opens showing the first frame of the video.
2. Click anywhere on the image to place point A.
3. Click again to place point B. A line is drawn between them.
4. The pixel distance and computed pixels/mm are shown live.
5. **Reset points** wipes the markers; **OK** accepts; **Cancel**
   discards.
6. The known-distance value is set in the dialog (defaulting to
   whatever was passed in) and is editable.

Returned via the ``ppm`` attribute on the dialog after
``exec() == QDialog.Accepted``. ``None`` if Cancelled.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import (QColor, QImage, QMouseEvent, QPaintEvent, QPainter,
                           QPen, QPixmap)
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFormLayout, QHBoxLayout, QLabel, QPushButton,
                               QSizePolicy, QVBoxLayout, QWidget)


# Same coordinate-mapping pattern as the ROI canvas: a dataclass
# holding the scale + offset between widget and frame pixel space.
# Duplicated rather than imported because importing roi_canvas
# brings in the whole drawing state machine; we only need the
# transform here. If a future refactor extracts _ImageMapping to a
# shared utilities module both can use it.
@dataclass
class _ImageMapping:
    scale: float
    offset_x: int
    offset_y: int
    frame_w: int
    frame_h: int

    def widget_to_frame(self, wx: int, wy: int) -> Optional[Tuple[int, int]]:
        if self.scale <= 0:
            return None
        fx = (wx - self.offset_x) / self.scale
        fy = (wy - self.offset_y) / self.scale
        if 0 <= fx <= self.frame_w and 0 <= fy <= self.frame_h:
            return int(round(fx)), int(round(fy))
        return None

    def frame_to_widget(self, fx: float, fy: float) -> Tuple[int, int]:
        return (
            int(round(fx * self.scale + self.offset_x)),
            int(round(fy * self.scale + self.offset_y)),
        )


class _CalibrationCanvas(QWidget):
    """Image viewport that accepts two click-points and draws a
    line between them. Emits a signal when the second point is
    placed (so the host dialog can refresh its readouts).

    Coordinates are stored in FRAME pixel space (so the px-distance
    calculation is independent of widget size / aspect bars).
    """

    points_changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCursor(Qt.CrossCursor)
        # Frame data
        self._frame_bgr: Optional[np.ndarray] = None
        self._frame_pixmap: Optional[QPixmap] = None
        self._mapping: Optional[_ImageMapping] = None
        # Click points in FRAME coords
        self._point_a: Optional[Tuple[int, int]] = None
        self._point_b: Optional[Tuple[int, int]] = None

    def set_frame(self, bgr: np.ndarray) -> None:
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

    def reset_points(self) -> None:
        self._point_a = None
        self._point_b = None
        self.points_changed.emit()
        self.update()

    def get_points(
        self,
    ) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """Returns (point_a, point_b) in frame pixel coords. Either
        may be None if not yet placed."""
        return self._point_a, self._point_b

    def pixel_distance(self) -> Optional[float]:
        """Euclidean distance between A and B in frame pixels.
        Returns None if both points aren't placed yet."""
        if self._point_a is None or self._point_b is None:
            return None
        ax, ay = self._point_a
        bx, by = self._point_b
        return float(np.hypot(bx - ax, by - ay))

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

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        if (
            ev.button() != Qt.LeftButton
            or self._mapping is None
        ):
            return
        # ev.position() is float in Qt6; widget coords
        pos = ev.position()
        frame_pt = self._mapping.widget_to_frame(
            int(pos.x()), int(pos.y()),
        )
        if frame_pt is None:
            # Click landed in the aspect-ratio bars; ignore. Drawing
            # a point at the click site would be visually misleading
            # since the same widget x,y can correspond to different
            # frame coords across resizes.
            return
        # Two-click cycle: A → B → (next click resets and starts over)
        if self._point_a is None:
            self._point_a = frame_pt
        elif self._point_b is None:
            self._point_b = frame_pt
        else:
            # Both already set; treat as a fresh first click.
            # Saves the user from clicking Reset just to redo.
            self._point_a = frame_pt
            self._point_b = None
        self.points_changed.emit()
        self.update()

    def paintEvent(self, ev: QPaintEvent) -> None:
        p = QPainter(self)
        try:
            p.fillRect(self.rect(), QColor(26, 26, 26))
            if self._frame_pixmap is None or self._mapping is None:
                # No frame loaded yet — show a hint
                p.setPen(QColor(160, 160, 160))
                p.drawText(
                    self.rect(),
                    Qt.AlignCenter,
                    "Loading first frame…",
                )
                return
            m = self._mapping
            target = QRect(
                m.offset_x, m.offset_y,
                int(round(m.frame_w * m.scale)),
                int(round(m.frame_h * m.scale)),
            )
            p.drawPixmap(target, self._frame_pixmap,
                         self._frame_pixmap.rect())
            # Draw markers + connecting line
            radius = max(4, int(8 * m.scale)) if m.scale > 0 else 8
            green = QColor(0, 230, 0)
            pen = QPen(green)
            pen.setWidth(max(2, int(3 * m.scale)) if m.scale > 0 else 3)
            p.setPen(pen)
            p.setBrush(green)
            if self._point_a is not None:
                ax, ay = m.frame_to_widget(*self._point_a)
                p.drawEllipse(QPoint(ax, ay), radius, radius)
                # Label "A"
                self._draw_label(p, ax + radius + 4, ay - radius, "A")
            if self._point_b is not None:
                bx, by = m.frame_to_widget(*self._point_b)
                p.drawEllipse(QPoint(bx, by), radius, radius)
                self._draw_label(p, bx + radius + 4, by - radius, "B")
            if (
                self._point_a is not None
                and self._point_b is not None
            ):
                ax, ay = m.frame_to_widget(*self._point_a)
                bx, by = m.frame_to_widget(*self._point_b)
                p.setBrush(Qt.NoBrush)
                pen2 = QPen(green)
                pen2.setWidth(max(2, int(2 * m.scale)) if m.scale > 0 else 2)
                p.setPen(pen2)
                p.drawLine(ax, ay, bx, by)
        finally:
            p.end()

    def _draw_label(self, p: QPainter, x: int, y: int, text: str) -> None:
        p.setPen(QColor(0, 230, 0))
        p.setBrush(Qt.NoBrush)
        font = p.font()
        font.setBold(True)
        font.setPointSize(max(font.pointSize(), 12))
        p.setFont(font)
        # Outline-style: white background label for legibility on
        # any frame brightness
        p.setPen(QColor(0, 0, 0))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    p.drawText(x + dx, y + dy, text)
        p.setPen(QColor(0, 230, 0))
        p.drawText(x, y, text)


class PixelCalibrationDialog(QDialog):
    """Modal dialog for deriving pixels-per-mm from a video frame.

    Usage:

        dlg = PixelCalibrationDialog(
            video_path="/path/to/video.mp4",
            known_mm_distance=100.0,
            parent=self,
        )
        if dlg.exec() == QDialog.Accepted:
            ppm = dlg.ppm   # float, > 0

    The dialog reads the first frame of the video on construction
    and shows it for the user to click two reference points on.
    If reading the frame fails (codec issue, missing file), the
    constructor raises RuntimeError before the dialog is shown.
    """

    def __init__(
        self,
        video_path: str,
        known_mm_distance: float = 100.0,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(
            f"Calibrate pixels/mm — {os.path.basename(video_path)}"
        )
        self.setModal(True)
        self.resize(900, 700)
        # Result attributes set on accept. Initialized to None so
        # callers can distinguish "user cancelled" from "user
        # accepted" via the attribute being None vs. set.
        self.ppm: Optional[float] = None
        self.known_mm_distance: Optional[float] = None
        self._video_path = video_path

        # Read first frame
        first_frame = self._load_first_frame(video_path)
        if first_frame is None:
            raise RuntimeError(
                f"Could not read first frame of {video_path}. The "
                f"video may be corrupted, missing, or in an "
                f"unsupported codec."
            )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        # Instructions
        instructions = QLabel(
            "<b>Click two points</b> on the image that span a "
            "known real-world distance (e.g. the two ends of a "
            "ruler placed in the frame). Adjust the known "
            "distance below if needed, then click OK to save the "
            "pixels/mm value.",
            self,
        )
        instructions.setWordWrap(True)
        outer.addWidget(instructions)

        # Image canvas
        self._canvas = _CalibrationCanvas(self)
        self._canvas.set_frame(first_frame)
        self._canvas.points_changed.connect(self._update_readouts)
        outer.addWidget(self._canvas, stretch=1)

        # Known-distance input + readouts row
        controls = QFormLayout()
        controls.setLabelAlignment(Qt.AlignRight)

        self._distance_input = QDoubleSpinBox(self)
        self._distance_input.setMinimum(0.001)
        self._distance_input.setMaximum(1_000_000.0)
        self._distance_input.setDecimals(3)
        self._distance_input.setValue(float(known_mm_distance))
        self._distance_input.setSuffix(" mm")
        self._distance_input.setToolTip(
            "The real-world length of the line you'll draw on the "
            "image. Common values: 100 mm for a 10 cm ruler, the "
            "diagonal of a known-size arena, etc."
        )
        self._distance_input.valueChanged.connect(self._update_readouts)
        controls.addRow("Known distance:", self._distance_input)

        self._pixel_readout = QLabel("—", self)
        controls.addRow("Pixel distance:", self._pixel_readout)

        self._ppm_readout = QLabel("—", self)
        # Bold so the actual value being saved is visually
        # prominent (the user's eye should land here when
        # deciding whether to click OK)
        self._ppm_readout.setStyleSheet(
            "font-weight: bold; font-size: 11pt;"
        )
        controls.addRow("Pixels per mm:", self._ppm_readout)

        outer.addLayout(controls)

        # Bottom row: Reset button (left) + OK / Cancel (right)
        bottom = QHBoxLayout()
        self._reset_btn = QPushButton("Reset points", self)
        self._reset_btn.setToolTip("Clear both placed points")
        self._reset_btn.clicked.connect(self._canvas.reset_points)
        bottom.addWidget(self._reset_btn)
        bottom.addStretch(1)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            parent=self,
        )
        self._buttons.accepted.connect(self._on_accept)
        self._buttons.rejected.connect(self.reject)
        bottom.addWidget(self._buttons)
        outer.addLayout(bottom)

        # OK starts disabled; enabled once both points placed
        self._update_readouts()

    @staticmethod
    def _load_first_frame(video_path: str) -> Optional[np.ndarray]:
        """Read the first frame of the video as a BGR ndarray.
        Returns None on any failure."""
        if not os.path.isfile(video_path):
            return None
        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                return None
            ok, frame = cap.read()
            return frame if ok else None
        finally:
            cap.release()

    def _update_readouts(self) -> None:
        """Refresh the pixel-distance and px/mm labels based on
        current click points + known distance."""
        px = self._canvas.pixel_distance()
        if px is None:
            self._pixel_readout.setText("—  (click two points)")
            self._ppm_readout.setText("—")
            self._buttons.button(QDialogButtonBox.Ok).setEnabled(False)
            return
        self._pixel_readout.setText(f"{px:.2f} px")
        known_mm = self._distance_input.value()
        if known_mm <= 0:
            self._ppm_readout.setText("—  (set known distance)")
            self._buttons.button(QDialogButtonBox.Ok).setEnabled(False)
            return
        ppm = px / known_mm
        self._ppm_readout.setText(f"{ppm:.4f}  px/mm")
        self._buttons.button(QDialogButtonBox.Ok).setEnabled(True)

    def _on_accept(self) -> None:
        """Validate and store the result, then close with Accepted.

        Persists both ``self.ppm`` AND ``self.known_mm_distance`` —
        the user may have edited the spinbox during the calibration
        flow (e.g. realized the reference was 150mm not 100mm), so
        the caller needs to read both values back to keep the form's
        Distance cell consistent with the px/mm cell.
        """
        px = self._canvas.pixel_distance()
        known_mm = self._distance_input.value()
        if px is None or known_mm <= 0:
            # Shouldn't happen — OK is disabled in this state — but
            # be defensive.
            return
        ppm = px / known_mm
        if ppm <= 0:
            return
        self.ppm = round(float(ppm), 4)
        self.known_mm_distance = float(known_mm)
        self.accept()


__all__ = ["PixelCalibrationDialog"]
