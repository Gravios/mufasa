"""
mufasa.ui_qt.feature_viewer
===========================

Video + feature traces, synchronised (patch 122je).

:class:`FeatureTracePlot` draws one lane per selected feature over a window
centred on the current frame — each lane scaled to that feature's
0.5–99.5 percentile range over the whole video, with its value at the
cursor printed beside the name — plus a strip of behaviour labels when the
video has any. Left-click / drag seeks (as in the labeller's label plot);
the mouse wheel widens or narrows the window.

:class:`FeatureViewerWidget` puts a :class:`FrameScrubberWidget` above the
traces; :func:`open_feature_viewer` shows it in a floating dock of the
workbench (or a dialog without one).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtCore import QPointF, QRect, Qt, Signal
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QDialog,
    QDockWidget,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

TRACE_COLORS = [(66, 133, 244), (219, 68, 55), (15, 157, 88), (244, 160, 0),
                (171, 71, 188), (0, 172, 193), (255, 112, 67), (124, 179, 66),
                (92, 107, 192), (240, 98, 146), (0, 137, 123), (141, 110, 99)]
LABEL_COLORS = [(66, 133, 244, 150), (219, 68, 55, 150), (15, 157, 88, 150),
                (244, 160, 0, 150), (171, 71, 188, 150), (0, 172, 193, 150)]


def short_feature_name(col: str) -> str:
    """Display name: drop the legacy ``_FEATURE_SUBSET`` suffix."""
    return col[: -len("_FEATURE_SUBSET")] if col.endswith("_FEATURE_SUBSET") else col


def robust_range(v: np.ndarray) -> tuple[float, float]:
    fin = v[np.isfinite(v)]
    if fin.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(fin, [0.5, 99.5])
    if not hi > lo:
        lo, hi = float(fin.min()), float(fin.max())
        if not hi > lo:
            lo, hi = lo - 0.5, hi + 0.5
    return float(lo), float(hi)


class FeatureTracePlot(QWidget):
    """Windowed feature lanes with a cursor; click / drag to seek."""

    frame_clicked = Signal(int)
    window_changed = Signal(float)

    LANE_H = 46
    LABEL_LANE_H = 14
    NAME_W = 230

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._data: dict[str, np.ndarray] = {}
        self._ranges: dict[str, tuple[float, float]] = {}
        self._labels: dict[str, np.ndarray] = {}
        self._frame = 0
        self._fps = 30.0
        self._window_s = 5.0
        self._drag: tuple[float, int] | None = None
        self.setMouseTracking(False)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Click or drag to move to that time · wheel = zoom the window")

    # ---- data ------------------------------------------------------------
    def set_features(self, df: pd.DataFrame) -> None:
        self._data = {c: pd.to_numeric(df[c], errors="coerce").to_numpy(np.float64)
                      for c in df.columns}
        self._ranges = {c: robust_range(v) for c, v in self._data.items()}
        self._update_height()
        self.update()

    def set_labels(self, labels: dict[str, np.ndarray]) -> None:
        self._labels = {k: np.asarray(v) for k, v in labels.items()
                        if np.nansum(np.asarray(v, dtype=float) == 1) > 0}
        self._update_height()
        self.update()

    def set_current_frame(self, f: int) -> None:
        self._frame = int(f)
        self.update()

    def set_fps(self, fps: float) -> None:
        self._fps = float(fps) if fps and fps > 0 else 30.0
        self.update()

    def set_window_seconds(self, s: float) -> None:
        self._window_s = float(min(600.0, max(0.5, s)))
        self.update()

    def window_seconds(self) -> float:
        return self._window_s

    def _update_height(self) -> None:
        h = 8 + len(self._labels) * self.LABEL_LANE_H + len(self._data) * self.LANE_H + 18
        self.setMinimumHeight(max(80, h))

    # ---- geometry --------------------------------------------------------
    def _plot_w(self) -> int:
        return max(40, self.width() - self.NAME_W - 8)

    def _half(self) -> int:
        return max(1, int(round(self._window_s * self._fps)))

    def frame_at(self, x: float) -> int | None:
        if x < self.NAME_W:
            return None
        half = self._half()
        f = self._frame - half + (min(x, self.NAME_W + self._plot_w()) - self.NAME_W) \
            * 2 * half / self._plot_w()
        return max(0, int(round(f)))

    # ---- mouse -----------------------------------------------------------
    def mousePressEvent(self, ev) -> None:  # noqa: ANN001
        if ev.button() == Qt.LeftButton:
            f = self.frame_at(ev.position().x())
            if f is not None:
                self._drag = (ev.position().x(), f)
                self.frame_clicked.emit(f)
                ev.accept()
                return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev) -> None:  # noqa: ANN001
        if self._drag and ev.buttons() & Qt.LeftButton:
            x0, f0 = self._drag
            f = f0 + (ev.position().x() - x0) * 2 * self._half() / self._plot_w()
            self.frame_clicked.emit(max(0, int(round(f))))
            ev.accept()
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev) -> None:  # noqa: ANN001
        self._drag = None
        super().mouseReleaseEvent(ev)

    def wheelEvent(self, ev) -> None:  # noqa: ANN001
        steps = ev.angleDelta().y() / 120.0
        if steps:
            self.set_window_seconds(self._window_s * (0.8 ** steps))
            self.window_changed.emit(self._window_s)
            ev.accept()

    # ---- painting --------------------------------------------------------
    def paintEvent(self, _ev) -> None:  # noqa: ANN001
        p = QPainter(self)
        p.fillRect(self.rect(), self.palette().base())
        fm = QFontMetrics(self.font())
        text = self.palette().text().color()
        dim = QColor(text)
        dim.setAlpha(140)
        x0, pw = self.NAME_W, self._plot_w()
        half = self._half()
        start = self._frame - half
        n = 2 * half
        if not self._data and not self._labels:
            p.setPen(dim)
            p.drawText(self.rect(), Qt.AlignCenter, "Select features to plot")
            p.end()
            return
        y = 6
        # behaviour strip
        for k, (name, arr) in enumerate(self._labels.items()):
            p.setPen(dim)
            p.drawText(QRect(4, y, x0 - 8, self.LABEL_LANE_H),
                       Qt.AlignVCenter | Qt.AlignLeft,
                       fm.elidedText(name, Qt.ElideRight, x0 - 8))
            lo, hi = max(0, start), min(len(arr), start + n + 1)
            if hi > lo:
                seg = (arr[lo:hi] == 1).astype(np.int8)
                d = np.diff(np.r_[0, seg, 0])
                on, off = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(*LABEL_COLORS[k % len(LABEL_COLORS)]))
                for a, b in zip(on, off):
                    xa = x0 + (lo + a - start) * pw / n
                    xb = x0 + (lo + b - start) * pw / n
                    p.drawRect(QRect(int(xa), y + 2, max(1, int(xb - xa)),
                                     self.LABEL_LANE_H - 4))
            y += self.LABEL_LANE_H
        # feature lanes
        for k, (name, v) in enumerate(self._data.items()):
            top = y + 2
            h = self.LANE_H - 6
            if k % 2 == 0:
                p.fillRect(x0, y, pw, self.LANE_H, self.palette().alternateBase())
            col = QColor(*TRACE_COLORS[k % len(TRACE_COLORS)])
            cur = v[self._frame] if 0 <= self._frame < len(v) else np.nan
            p.setPen(text)
            p.drawText(QRect(4, top, x0 - 8, h // 2 + 4), Qt.AlignLeft | Qt.AlignTop,
                       fm.elidedText(short_feature_name(name), Qt.ElideMiddle, x0 - 8))
            p.setPen(col)
            p.drawText(QRect(4, top + h // 2, x0 - 8, h // 2 + 2),
                       Qt.AlignLeft | Qt.AlignVCenter,
                       "—" if not np.isfinite(cur) else f"{cur:.4g}")
            lo_v, hi_v = self._ranges[name]
            lo, hi = max(0, start), min(len(v), start + n + 1)
            if hi - lo >= 2:
                xs = x0 + (np.arange(lo, hi) - start) * pw / n
                seg = v[lo:hi]
                ys = top + h - (np.clip(seg, lo_v, hi_v) - lo_v) / (hi_v - lo_v) * h
                # decimate to ≤ 2 points per pixel column (min / max)
                if len(xs) > 2 * pw:
                    bins = np.minimum(((xs - x0) * 1.0).astype(int), pw - 1)
                    order = np.argsort(bins, kind="stable")
                    xs, ys, bins = xs[order], ys[order], bins[order]
                    path = QPainterPath()
                    started = False
                    for b in np.unique(bins):
                        m = bins == b
                        yy = ys[m]
                        yy = yy[np.isfinite(yy)]
                        if yy.size == 0:
                            started = False
                            continue
                        xb = x0 + b + 0.5
                        if not started:
                            path.moveTo(QPointF(xb, yy.min()))
                            started = True
                        path.lineTo(QPointF(xb, yy.min()))
                        path.lineTo(QPointF(xb, yy.max()))
                else:
                    path = QPainterPath()
                    started = False
                    for xx, yy in zip(xs, ys):
                        if not np.isfinite(yy):
                            started = False
                            continue
                        if started:
                            path.lineTo(QPointF(xx, yy))
                        else:
                            path.moveTo(QPointF(xx, yy))
                            started = True
                p.setPen(QPen(col, 1.4))
                p.setBrush(Qt.NoBrush)
                p.drawPath(path)
            y += self.LANE_H
        # cursor + time axis
        p.setPen(QPen(QColor(220, 40, 40), 1.5))
        xc = x0 + pw / 2
        p.drawLine(int(xc), 4, int(xc), y)
        p.setPen(dim)
        p.drawText(QRect(x0, y + 2, pw, 14), Qt.AlignLeft,
                   f"−{self._window_s:.1f} s")
        p.drawText(QRect(x0, y + 2, pw, 14), Qt.AlignHCenter,
                   f"frame {self._frame}  ·  {self._frame / self._fps:.2f} s")
        p.drawText(QRect(x0, y + 2, pw, 14), Qt.AlignRight, f"+{self._window_s:.1f} s")
        p.end()


class FeatureViewerWidget(QWidget):
    """Video scrubber with synchronised feature traces."""

    def __init__(self, config_path: str, video_path: str, features: list[str],
                 start_frame: int = 0, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from pathlib import Path

        from mufasa.ui_qt.frame_scrubber import FrameScrubberWidget
        from mufasa.utils.feature_io import load_feature_columns
        self.config_path = config_path
        self.video_name = Path(video_path).stem
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        self.scrubber = FrameScrubberWidget(self)
        self.scrubber.load(video_path)
        lay.addWidget(self.scrubber, 3)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Window ±", self))
        self.window_spin = QDoubleSpinBox(self)
        self.window_spin.setRange(0.5, 600.0)
        self.window_spin.setDecimals(1)
        self.window_spin.setSuffix(" s")
        self.window_spin.setValue(5.0)
        bar.addWidget(self.window_spin)
        self.info = QLabel("", self)
        self.info.setStyleSheet("color: palette(placeholder-text);")
        bar.addWidget(self.info, 1)
        lay.addLayout(bar)
        self.plot = FeatureTracePlot(self)
        self.plot.set_fps(self.scrubber.fps)
        df = load_feature_columns(self.video_name, config_path, features)
        self.plot.set_features(df)
        self.plot.set_labels(self._labels())
        lay.addWidget(self.plot, 2)
        n_feat = len(df)
        n_vid = self.scrubber.total_frames
        self.info.setText(
            f"{df.shape[1]} feature(s) · {n_feat:,} rows"
            + (f" — video has {n_vid:,} frames" if n_feat != n_vid else ""))
        self.window_spin.valueChanged.connect(self.plot.set_window_seconds)
        self.plot.window_changed.connect(self._sync_spin)
        self.plot.frame_clicked.connect(self._seek)
        self.scrubber.frame_changed.connect(self.plot.set_current_frame)
        self._seek(start_frame)
        self.plot.set_current_frame(self.scrubber.current_frame)

    def _labels(self) -> dict[str, np.ndarray]:
        try:
            from mufasa.utils.label_io import load_labels_for_video
            df = load_labels_for_video(self.video_name, self.config_path)
        except Exception:
            return {}
        return {c: pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy()
                for c in df.columns}

    def _sync_spin(self, s: float) -> None:
        self.window_spin.blockSignals(True)
        self.window_spin.setValue(s)
        self.window_spin.blockSignals(False)

    def _seek(self, f: int) -> None:
        n = self.scrubber.total_frames
        if n > 0:
            f = max(0, min(int(f), n - 1))
            if f != self.scrubber.current_frame:
                self.scrubber.seek(f)

    def cleanup(self) -> None:
        try:
            self.scrubber.close_video()
        except Exception:
            pass


def open_feature_viewer(parent: QWidget, config_path: str, video_path: str,
                        features: list[str], start_frame: int = 0) -> QWidget:
    """Show the viewer in a floating workbench dock (dialog fallback)."""
    w = FeatureViewerWidget(config_path, video_path, features, start_frame)
    main = parent.window() if parent is not None else None
    title = f"Features — {w.video_name}"
    if isinstance(main, QMainWindow):
        dock = QDockWidget(title, main)
        dock.setWidget(w)
        dock.setAllowedAreas(Qt.AllDockWidgetAreas)
        dock.setFloating(True)
        dock.resize(1150, 780)
        main.addDockWidget(Qt.RightDockWidgetArea, dock)
        orig = dock.closeEvent

        def _close(ev):  # noqa: ANN001
            w.cleanup()
            orig(ev)

        dock.closeEvent = _close
        if not hasattr(main, "_feature_viewer_docks"):
            main._feature_viewer_docks = []
        main._feature_viewer_docks.append(dock)
        dock.show()
        return dock
    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    QVBoxLayout(dlg).addWidget(w)
    dlg.resize(1150, 780)
    dlg.finished.connect(lambda *_: w.cleanup())
    dlg.show()
    return dlg


__all__ = [
    "FeatureTracePlot",
    "FeatureViewerWidget",
    "open_feature_viewer",
    "robust_range",
    "short_feature_name",
]
