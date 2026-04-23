"""
mufasa.ui_qt.frame_scrubber
===========================

A reusable Qt widget that displays a single video frame with a
scrub bar, jog buttons, and a frame-number editor. Designed for
forms that need to let the user pinpoint a specific frame of a
video without running a full playback loop.

Used by :mod:`mufasa.ui_qt.frame_labeller` (behavioural annotation)
and can be embedded in future clip-review or classifier-validation
dialogs.

Design
------

* Frames are loaded on demand via :class:`cv2.VideoCapture` with
  ``CAP_PROP_POS_FRAMES`` seeking — no background playback thread,
  no threading concerns. Random access only.
* The ``frame_changed`` signal emits whenever the shown frame
  index changes (from any source: slider, jog, edit). Downstream
  widgets subscribe to keep their state in sync.
* OpenCV BGR → Qt RGB conversion happens once per seek via
  :class:`QImage` with ``Format_BGR888`` — no per-pixel copy.
* Aspect ratio preserved; the displayed pixmap is scaled to fit
  the label's current size (respects widget resizing).
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QPushButton, QSlider,
                               QSpinBox, QVBoxLayout, QWidget)


class FrameScrubberWidget(QWidget):
    """Single-frame video preview with seek controls.

    Signals
    -------
    frame_changed(int)
        Emitted after the display updates to a new frame index.
    """

    frame_changed = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._cap: Optional[cv2.VideoCapture] = None
        self._total_frames: int = 0
        self._fps: float = 30.0
        self._current_frame: int = 0
        self._last_pixmap: Optional[QPixmap] = None
        self._build_ui()

    def load(self, video_path: str) -> None:
        """Open ``video_path`` and jump to frame 0."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        self._cap = cap
        self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        self._fps = fps if fps and fps > 0 else 30.0
        self._slider.setRange(0, max(0, self._total_frames - 1))
        self._frame_box.setRange(0, max(0, self._total_frames - 1))
        self._total_lbl.setText(f"/ {self._total_frames - 1}")
        self.seek(0)

    def close_video(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def seek(self, frame_idx: int) -> None:
        if self._cap is None:
            return
        frame_idx = max(0, min(frame_idx, self._total_frames - 1))
        if frame_idx == self._current_frame and self._last_pixmap is not None:
            return
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return
        self._current_frame = frame_idx
        self._render(frame)
        self._slider.blockSignals(True)
        self._slider.setValue(frame_idx)
        self._slider.blockSignals(False)
        self._frame_box.blockSignals(True)
        self._frame_box.setValue(frame_idx)
        self._frame_box.blockSignals(False)
        self._time_lbl.setText(self._format_time(frame_idx))
        self.frame_changed.emit(frame_idx)

    @property
    def current_frame(self) -> int:
        return self._current_frame

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def fps(self) -> float:
        return self._fps

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self._display = QLabel(self)
        self._display.setAlignment(Qt.AlignCenter)
        self._display.setMinimumSize(QSize(320, 180))
        self._display.setStyleSheet("background-color: #222;")
        self._display.setText("(no video loaded)")
        self._display.setScaledContents(False)
        outer.addWidget(self._display, 1)

        self._slider = QSlider(Qt.Horizontal, self)
        self._slider.valueChanged.connect(self._on_slider)
        outer.addWidget(self._slider)

        ctrl = QHBoxLayout()
        # Compact button padding is structural, not cosmetic — the
        # scrubber row has to fit six seek buttons plus a frame counter
        # inline, so we shave the default padding.
        self._b_prev100 = QPushButton("\u27EA 100", self)
        self._b_prev10  = QPushButton("\u27E8 10",  self)
        self._b_prev    = QPushButton("\u25C0",     self)
        self._b_next    = QPushButton("\u25B6",     self)
        self._b_next10  = QPushButton("10 \u27E9",  self)
        self._b_next100 = QPushButton("100 \u27EB", self)
        for b, delta in [(self._b_prev100, -100), (self._b_prev10, -10),
                         (self._b_prev, -1), (self._b_next, 1),
                         (self._b_next10, 10), (self._b_next100, 100)]:
            b.setStyleSheet("padding: 2px 8px;")
            b.clicked.connect(lambda _=False, d=delta: self.seek(self._current_frame + d))
            ctrl.addWidget(b)
        ctrl.addSpacing(12)

        ctrl.addWidget(QLabel("Frame:", self))
        self._frame_box = QSpinBox(self)
        self._frame_box.setRange(0, 0)
        self._frame_box.valueChanged.connect(self.seek)
        ctrl.addWidget(self._frame_box)
        self._total_lbl = QLabel("/ 0", self)
        ctrl.addWidget(self._total_lbl)

        ctrl.addSpacing(12)
        ctrl.addWidget(QLabel("Time:", self))
        self._time_lbl = QLabel("00:00.000", self)
        self._time_lbl.setMinimumWidth(80)
        ctrl.addWidget(self._time_lbl)
        ctrl.addStretch()

        outer.addLayout(ctrl)

    def _on_slider(self, val: int) -> None:
        self.seek(val)

    def _render(self, frame_bgr: np.ndarray) -> None:
        h, w = frame_bgr.shape[:2]
        if not frame_bgr.flags["C_CONTIGUOUS"]:
            frame_bgr = np.ascontiguousarray(frame_bgr)
        qimg = QImage(frame_bgr.data, w, h, 3 * w, QImage.Format_BGR888)
        pix = QPixmap.fromImage(qimg)
        self._last_pixmap = pix
        self._rescale_display()

    def _rescale_display(self) -> None:
        if self._last_pixmap is None:
            return
        target = self._display.size()
        self._display.setPixmap(
            self._last_pixmap.scaled(
                target, Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
        )

    def resizeEvent(self, ev) -> None:  # noqa: N802
        super().resizeEvent(ev)
        self._rescale_display()

    def _format_time(self, frame_idx: int) -> str:
        seconds = frame_idx / max(self._fps, 1.0)
        mm = int(seconds // 60)
        ss = seconds - mm * 60
        return f"{mm:02d}:{ss:06.3f}"


__all__ = ["FrameScrubberWidget"]
