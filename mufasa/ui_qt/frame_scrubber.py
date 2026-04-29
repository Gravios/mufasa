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
from PySide6.QtCore import QSize, Qt, QTimer, Signal
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
        # Playback state. _play_direction is +1 (forward), -1
        # (backward), or 0 (paused). The single timer drives both
        # directions; the direction flips by clicking the opposite
        # button or by pressing the same direction's button again to
        # pause.
        self._play_direction: int = 0
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._on_play_tick)
        self._build_ui()

    def load(self, video_path: str) -> None:
        """Open ``video_path`` and jump to frame 0."""
        # Stop any in-progress playback so we don't tick into a
        # newly-loaded video at the old timer interval.
        self._stop_playback()
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
        # Set timer interval based on actual video FPS so playback
        # speed matches the recorded rate. 1000/fps ms per frame.
        self._play_timer.setInterval(int(1000.0 / self._fps))
        self._slider.setRange(0, max(0, self._total_frames - 1))
        self._frame_box.setRange(0, max(0, self._total_frames - 1))
        self._total_lbl.setText(f"/ {self._total_frames - 1}")
        self.seek(0)

    def close_video(self) -> None:
        self._stop_playback()
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
        # scrubber row has to fit six seek buttons plus play/pause
        # plus a frame counter inline, so we shave the default padding.
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
        # Play-backward / play-forward buttons. Sit between the
        # single-frame [◀] [▶] buttons. Each button toggles its own
        # direction: clicking play-fwd starts forward playback;
        # clicking it again pauses; clicking play-back while playing
        # forward switches to backward playback. The label changes
        # to ⏸ when that direction is active. Using ◀◀/▶▶ as the
        # idle glyph (vs. simple ⏵) to distinguish from single-step.
        self._b_play_back = QPushButton("\u23EA", self)  # ⏪
        self._b_play_fwd  = QPushButton("\u23E9", self)  # ⏩
        self._b_play_back.setToolTip(
            "Play backward (frame index -1 per tick, wraps at 0). "
            "Click again to pause."
        )
        self._b_play_fwd.setToolTip(
            "Play forward (frame index +1 per tick, wraps at end). "
            "Click again to pause."
        )
        self._b_play_back.setStyleSheet("padding: 2px 8px;")
        self._b_play_fwd.setStyleSheet("padding: 2px 8px;")
        self._b_play_back.clicked.connect(
            lambda: self._toggle_play(direction=-1)
        )
        self._b_play_fwd.clicked.connect(
            lambda: self._toggle_play(direction=+1)
        )

        # Layout order:
        #   [⟪100] [⟨10] [◀] [⏪] [⏩] [▶] [10⟩] [100⟫]
        # Play buttons sit between the single-step buttons so the
        # single-step buttons remain edge-adjacent to the play
        # direction they relate to (◀ next to ⏪).
        ctrl.addWidget(self._b_prev100)
        ctrl.addWidget(self._b_prev10)
        ctrl.addWidget(self._b_prev)
        ctrl.addWidget(self._b_play_back)
        ctrl.addWidget(self._b_play_fwd)
        ctrl.addWidget(self._b_next)
        ctrl.addWidget(self._b_next10)
        ctrl.addWidget(self._b_next100)
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

    # ------------------------------------------------------------------ #
    # Playback controls
    # ------------------------------------------------------------------ #
    def _toggle_play(self, direction: int) -> None:
        """Start/stop playback in the requested direction.

        Three transitions:
          - Idle → playing(direction): start the timer
          - Playing(direction) → idle: same button pressed; pause
          - Playing(other_direction) → playing(direction): switch
            direction without stopping the timer

        The timer interval is set in ``load`` based on the video's
        FPS so playback runs at recorded speed.
        """
        if self._cap is None or self._total_frames <= 0:
            return
        if self._play_direction == direction:
            # Same-direction click while playing → pause
            self._stop_playback()
        else:
            # Either idle (start) or other direction (switch).
            # The timer's already running in the switch case; just
            # change the direction so the next tick goes the new way.
            self._play_direction = direction
            self._refresh_play_button_glyphs()
            if not self._play_timer.isActive():
                self._play_timer.start()

    def _stop_playback(self) -> None:
        """Halt the playback timer and reset glyphs."""
        if self._play_timer.isActive():
            self._play_timer.stop()
        self._play_direction = 0
        self._refresh_play_button_glyphs()

    def _refresh_play_button_glyphs(self) -> None:
        """Show the pause glyph on the active direction's button and
        the idle glyph on the other. Only relevant after ``_build_ui``
        has run (guarded by hasattr in case _stop_playback is called
        from ``__init__`` before the buttons exist)."""
        if not hasattr(self, "_b_play_fwd"):
            return
        # \u23F8 is ⏸ (pause). The idle glyphs are ⏪ (play-back) and
        # ⏩ (play-fwd).
        if self._play_direction == 1:
            self._b_play_fwd.setText("\u23F8")
            self._b_play_back.setText("\u23EA")
        elif self._play_direction == -1:
            self._b_play_back.setText("\u23F8")
            self._b_play_fwd.setText("\u23E9")
        else:
            self._b_play_back.setText("\u23EA")
            self._b_play_fwd.setText("\u23E9")

    def _on_play_tick(self) -> None:
        """Timer tick: advance frame index by ``_play_direction``,
        wrapping at the file boundaries.

        Wrap behavior: end → 0 (forward), 0 → end (backward). User
        explicitly requested "wrap-around to play the video."
        """
        if self._cap is None or self._total_frames <= 0 or self._play_direction == 0:
            return
        next_idx = self._current_frame + self._play_direction
        if next_idx >= self._total_frames:
            next_idx = 0
        elif next_idx < 0:
            next_idx = self._total_frames - 1
        # Use seek() so all the existing book-keeping (slider,
        # frame_box, time label, frame_changed signal) updates
        # consistently. seek() clamps the index for safety, but our
        # next_idx is already within [0, total-1] after the wrap.
        self.seek(next_idx)

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
