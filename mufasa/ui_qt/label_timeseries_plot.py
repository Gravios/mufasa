"""
mufasa.ui_qt.label_timeseries_plot
===================================

Active label-timeseries plot for Frame labelling.

Patch 122fg — User request (May 26, 2026):

  > I still need an active plot, +- 2 range with lines where a
  > behavior has been labeled.

Shows a fixed-window view of the label state around the current
playback frame. The window defaults to ±2 seconds and is
adjustable via an instance setter (``set_window_seconds``).
Each classifier's label series is drawn as a horizontal lane;
vertical bars / spans within each lane mark frames where the
label is 1.

The widget is render-only — it doesn't own playback state or
the label data. The host (``FrameLabellerWidget``) feeds it
updates:

* ``set_labels(labels: dict[str, ndarray])`` — one-shot, on
  load / mode change.
* ``set_fps(fps: float)`` — once on video load.
* ``set_current_frame(frame_idx: int)`` — once per
  ``frame_changed`` event from the scrubber.

The widget repaints whenever any of the three setters fires.
A small status label below the plot shows the current frame
position and the window range.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QPainter,
    QPen,
)
from PySide6.QtWidgets import (
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

# Per-lane colour palette. Reused if there are more classifiers
# than colours. Chosen to be high-contrast against both light
# and dark Qt palette themes.
_LANE_COLORS: Sequence[tuple[int, int, int]] = (
    (66, 133, 244),   # blue
    (234, 67, 53),    # red
    (251, 188, 5),    # yellow / amber
    (52, 168, 83),    # green
    (171, 71, 188),   # purple
    (255, 112, 67),   # orange
    (3, 169, 244),    # cyan
    (216, 27, 96),    # magenta
)


class LabelTimeseriesPlot(QWidget):
    """Patch 122fg — Active ±-N-seconds plot of label state.

    The plot has one horizontal lane per classifier. Within each
    lane, contiguous spans where ``labels[name][i] == 1`` are
    rendered as filled coloured rectangles. A vertical "now"
    cursor marks the current frame position.

    Defaults: window = ±2 seconds, lane height = 18 px, label
    column width = 80 px (left margin). All adjustable via the
    setter methods below.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._labels: dict[str, np.ndarray] = {}
        self._classifier_names: list[str] = []
        self._classifier_keys: dict[str, str] = {}
        self._fps: float = 30.0
        self._current_frame: int = 0
        self._window_seconds: float = 2.0
        self._lane_height: int = 22
        self._label_col_w: int = 100
        self._top_pad: int = 8
        self._bot_pad: int = 18

        self.setMinimumHeight(80)
        self.setSizePolicy(QSizePolicy.Expanding,
                            QSizePolicy.Minimum)

        # Build a small status row at the bottom (handled
        # entirely in paintEvent — simpler than a nested QLabel).
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        # No child widgets — we paint everything ourselves.

    # ------------------------------------------------------------------ #
    # Public setters
    # ------------------------------------------------------------------ #
    def set_labels(
        self, labels: dict[str, np.ndarray],
        classifier_names: list[str] | None = None,
        classifier_keys: dict[str, str] | None = None,
    ) -> None:
        """Replace the label arrays. ``classifier_names`` controls
        lane ordering; defaults to the keys of ``labels``.
        ``classifier_keys`` is an optional name→hotkey map used to
        annotate each lane's label."""
        self._labels = dict(labels)
        if classifier_names is None:
            self._classifier_names = list(labels.keys())
        else:
            self._classifier_names = list(classifier_names)
        self._classifier_keys = (
            dict(classifier_keys) if classifier_keys else {}
        )
        # Auto-size to fit lanes + status row.
        n = max(1, len(self._classifier_names))
        self.setMinimumHeight(
            self._top_pad + n * self._lane_height + self._bot_pad,
        )
        self.update()

    def set_fps(self, fps: float) -> None:
        if fps > 0:
            self._fps = float(fps)
            self.update()

    def set_current_frame(self, frame_idx: int) -> None:
        if frame_idx != self._current_frame:
            self._current_frame = int(frame_idx)
            self.update()

    def set_window_seconds(self, seconds: float) -> None:
        """Set the half-width of the window in seconds. The full
        window spans ±``seconds`` around the current frame."""
        if seconds > 0:
            self._window_seconds = float(seconds)
            self.update()

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def paintEvent(self, _ev) -> None:  # noqa: ANN001
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)

        bg_brush = self.palette().alternateBase()
        painter.fillRect(self.rect(), bg_brush)

        if not self._classifier_names:
            self._paint_placeholder(painter)
            painter.end()
            return

        w = self.width()
        plot_x = self._label_col_w
        plot_w = max(40, w - plot_x - 8)

        # Window math: full window = ±window_seconds. Total
        # frames = 2 * window_seconds * fps.
        half_frames = max(1, int(round(
            self._window_seconds * self._fps,
        )))
        start_f = self._current_frame - half_frames
        end_f = self._current_frame + half_frames
        total = end_f - start_f
        if total <= 0:
            painter.end()
            return

        # Pixels per frame.
        ppf = plot_w / total

        # Lanes.
        text_pen = QPen(self.palette().text().color())
        painter.setPen(text_pen)
        font = self.font()
        fm = QFontMetrics(font)
        painter.setFont(font)

        for lane_i, name in enumerate(self._classifier_names):
            y_top = self._top_pad + lane_i * self._lane_height
            y_lane = y_top + 2
            h = self._lane_height - 4

            # Lane background — subtle stripe to separate.
            if lane_i % 2 == 0:
                stripe = QColor(255, 255, 255, 8)
                painter.fillRect(
                    plot_x, y_top, plot_w, self._lane_height, stripe,
                )

            # Lane label (name + optional key).
            key = self._classifier_keys.get(name, "")
            label_text = (
                f"{name} ({key})" if key else name
            )
            # Truncate if it overflows the label column.
            elided = fm.elidedText(
                label_text, Qt.ElideRight, self._label_col_w - 6,
            )
            painter.setPen(text_pen)
            painter.drawText(
                QRect(4, y_top, self._label_col_w - 6,
                        self._lane_height),
                Qt.AlignVCenter | Qt.AlignLeft,
                elided,
            )

            # Label spans within the lane.
            arr = self._labels.get(name)
            if arr is None or arr.size == 0:
                continue
            colour_tuple = _LANE_COLORS[lane_i % len(_LANE_COLORS)]
            colour = QColor(*colour_tuple)
            painter.setBrush(QBrush(colour))
            painter.setPen(Qt.NoPen)

            # Iterate windowed range and draw filled spans where
            # arr == 1.
            in_span = False
            span_start = 0
            for f in range(start_f, end_f):
                if 0 <= f < arr.shape[0]:
                    v = int(arr[f]) != 0
                else:
                    v = False
                if v and not in_span:
                    in_span = True
                    span_start = f
                elif not v and in_span:
                    # Close span at f-1 → f
                    sx = plot_x + (span_start - start_f) * ppf
                    ex = plot_x + (f - start_f) * ppf
                    painter.drawRect(
                        int(sx), y_lane,
                        max(1, int(ex - sx)), h,
                    )
                    in_span = False
            # Trailing span (extends to end of window).
            if in_span:
                sx = plot_x + (span_start - start_f) * ppf
                ex = plot_x + (end_f - start_f) * ppf
                painter.drawRect(
                    int(sx), y_lane,
                    max(1, int(ex - sx)), h,
                )

        # "Now" cursor — vertical line at current frame position.
        now_x = plot_x + (self._current_frame - start_f) * ppf
        now_pen = QPen(QColor(255, 80, 80), 2)
        painter.setPen(now_pen)
        n_lanes = len(self._classifier_names)
        y_top = self._top_pad
        y_bot = y_top + n_lanes * self._lane_height
        painter.drawLine(
            int(now_x), y_top, int(now_x), y_bot,
        )

        # Status row at the bottom.
        painter.setPen(QPen(self.palette().placeholderText().color()))
        small = QFont(self.font())
        small.setPointSizeF(max(7.0, self.font().pointSizeF() - 1))
        small.setItalic(True)
        painter.setFont(small)
        status = (
            f"frame {self._current_frame}  ·  "
            f"window ±{self._window_seconds:g}s  ·  "
            f"fps {self._fps:g}"
        )
        painter.drawText(
            QRect(8, y_bot + 2, w - 16, self._bot_pad),
            Qt.AlignVCenter | Qt.AlignLeft,
            status,
        )

        painter.end()

    def _paint_placeholder(self, painter: QPainter) -> None:
        painter.setPen(QPen(self.palette().placeholderText().color()))
        f = QFont(self.font()); f.setItalic(True)
        painter.setFont(f)
        painter.drawText(
            self.rect(),
            Qt.AlignCenter,
            "(no classifiers defined — label-timeseries plot empty)",
        )


__all__ = ["LabelTimeseriesPlot"]
