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

from pathlib import Path

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
    GRID_LINES = 5          # light horizontal grid per lane (0/25/50/75/100 %)
    GRID_ALPHA = 38
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
    def grid_ys(self, top: int, h: int) -> list[float]:
        """y of the lane's grid lines: GRID_LINES evenly spaced from the
        top (lane maximum) to the bottom (lane minimum)."""
        n = self.GRID_LINES
        return [top + h * i / (n - 1) for i in range(n)] if n > 1 else []

    def _draw_grid(self, p, x0: int, pw: int, top: int, h: int, text) -> None:  # noqa: ANN001
        g = QColor(text)
        g.setAlpha(self.GRID_ALPHA)
        pen = QPen(g)
        pen.setWidth(0)
        p.setPen(pen)
        for gy in self.grid_ys(top, h):
            p.drawLine(QPointF(x0, gy), QPointF(x0 + pw, gy))

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
            self._draw_grid(p, x0, pw, top, h, text)
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
        self.config_path = config_path
        self.video_name = Path(video_path).stem
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        self.scrubber = FrameScrubberWidget(self)
        self.scrubber.load(video_path)
        lay.addWidget(self.scrubber, 3)
        self.video_path = video_path
        self._build_overlay_bar(lay)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Window ±", self))
        self.window_spin = QDoubleSpinBox(self)
        self.window_spin.setRange(0.5, 600.0)
        self.window_spin.setDecimals(1)
        self.window_spin.setSuffix(" s")
        self.window_spin.setValue(5.0)
        bar.addWidget(self.window_spin)
        # Patch 122jk: follow the Feature Visualization selection
        from PySide6.QtWidgets import QCheckBox
        self.follow_cb = QCheckBox("Follow selection", self)
        self.follow_cb.setChecked(True)
        self.follow_cb.setToolTip("Update the traces when features are ticked or "
                                  "unticked in Feature Visualization")
        bar.addWidget(self.follow_cb)
        self.info = QLabel("", self)
        self.info.setStyleSheet("color: palette(placeholder-text);")
        bar.addWidget(self.info, 1)
        lay.addLayout(bar)
        self.plot = FeatureTracePlot(self)
        self.plot.set_fps(self.scrubber.fps)
        self._loaded: dict[str, np.ndarray] = {}
        self.features: list[str] = []
        self.set_feature_columns(features)
        self.plot.set_labels(self._labels())
        lay.addWidget(self.plot, 2)
        self.window_spin.valueChanged.connect(self.plot.set_window_seconds)
        self.plot.window_changed.connect(self._sync_spin)
        self.plot.frame_clicked.connect(self._seek)
        self.scrubber.frame_changed.connect(self.plot.set_current_frame)
        self._seek(start_frame)
        self.plot.set_current_frame(self.scrubber.current_frame)

    def set_feature_columns(self, features: list[str]) -> None:
        """Show these features (in this order); only columns not shown
        before are read from disk."""
        from mufasa.utils.feature_io import load_feature_columns
        features = list(dict.fromkeys(features))
        new = [c for c in features if c not in self._loaded]
        if new:
            df = load_feature_columns(self.video_name, self.config_path, new)
            for c in df.columns:
                self._loaded[c] = df[c].to_numpy()
        self.features = [c for c in features if c in self._loaded]
        if hasattr(self, "jpdf_x"):
            self._fill_jpdf_combos()
            if self.jpdf is not None and (self.jpdf.x not in self.features
                                          or self.jpdf.y not in self.features):
                self.jpdf_cb.setChecked(False)
        self.plot.set_features(pd.DataFrame({c: self._loaded[c] for c in self.features}))
        n_feat = len(next(iter(self._loaded.values()))) if self._loaded else 0
        n_vid = self.scrubber.total_frames
        self.info.setText(
            f"{len(self.features)} feature(s) · {n_feat:,} rows"
            + (f" — video has {n_vid:,} frames" if n_feat and n_feat != n_vid else ""))

    def follow(self, features: list[str]) -> bool:
        """Selection changed elsewhere: update when following; True if it did."""
        if not self.follow_cb.isChecked() or list(features) == self.features:
            return False
        self.set_feature_columns(features)
        return True

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

    # ------------------------------------------------------------------ #
    # Patch 122ju — pose overlays and recording
    # ------------------------------------------------------------------ #
    def _build_overlay_bar(self, lay) -> None:  # noqa: ANN001
        from PySide6.QtWidgets import QCheckBox, QPushButton

        from mufasa.ui_qt.pose_overlay import find_fdlc_pose, find_kalman_pose
        self.overlay = None
        self.recorder = None
        self._encode_thread = None
        self.fdlc_path = find_fdlc_pose(self.config_path, self.video_name)
        self.kalman_path = find_kalman_pose(self.config_path, self.video_name)
        bar = QHBoxLayout()
        self.fdlc_cb = QCheckBox("FDLC markers", self)
        self.kalman_cb = QCheckBox("Kalman markers", self)
        for cb, path, what in ((self.fdlc_cb, self.fdlc_path, "tracker (FDLC) pose under "
                                "sources/pose"),
                               (self.kalman_cb, self.kalman_path, "Kalman v2 smoothed pose "
                                "(latest run)")):
            cb.setEnabled(path is not None)
            cb.setToolTip(str(path) if path else f"No {what} for this video")
            cb.toggled.connect(self._overlay_changed)
            bar.addWidget(cb)
        self.skeleton_cb = QCheckBox("Skeleton", self)
        self.skeleton_cb.setChecked(True)
        self.skeleton_cb.setToolTip("Join the Kalman markers by the skeleton")
        self.skeleton_cb.toggled.connect(self._overlay_changed)
        bar.addWidget(self.skeleton_cb)
        bar.addWidget(QLabel("p ≥", self))
        self.p_spin = QDoubleSpinBox(self)
        self.p_spin.setRange(0.0, 1.0)
        self.p_spin.setSingleStep(0.05)
        self.p_spin.setValue(0.5)
        self.p_spin.setToolTip("FDLC markers below this likelihood are not drawn")
        self.p_spin.valueChanged.connect(self._overlay_changed)
        bar.addWidget(self.p_spin)
        bar.addWidget(QLabel("Pose scale", self))
        self.scale_spin = QDoubleSpinBox(self)
        self.scale_spin.setRange(0.1, 16.0)
        self.scale_spin.setSingleStep(0.5)
        self.scale_spin.setValue(1.0)
        self.scale_spin.setToolTip("Multiply pose coordinates (2 = pose estimated on a "
                                   "half-resolution video)")
        self.scale_spin.valueChanged.connect(self._overlay_changed)
        bar.addWidget(self.scale_spin)
        self.style_btn = QPushButton("Style…", self)
        self.style_btn.setToolTip("Marker / skeleton style and visibility for FDLC and "
                                  "Kalman (remembered per project)")
        self.style_btn.setEnabled(self.fdlc_path is not None or self.kalman_path is not None)
        self.style_btn.clicked.connect(self.open_style_dialog)
        self._style_dlg = None
        bar.addWidget(self.style_btn)
        # Patch 122ke: 3-D marker viewer (volume above, this video below)
        self.view3d_btn = QPushButton("3-D…", self)
        self.view3d_btn.setToolTip(
            "Open the 3-D marker viewer for this video (a <video>*3d*.parquet beside "
            "the video or under derived/pose3d/; otherwise choose a file)")
        self.view3d_btn.clicked.connect(lambda: self.open_3d())
        bar.addWidget(self.view3d_btn)
        bar.addStretch()
        self.keep_frames_cb = QCheckBox("Keep frames", self)
        self.keep_frames_cb.setChecked(True)
        self.keep_frames_cb.setToolTip("Keep the extracted PNG frames after the video "
                                       "is made")
        bar.addWidget(self.keep_frames_cb)
        self.record_btn = QPushButton("● Record", self)
        self.record_btn.setCheckable(True)
        self.record_btn.setToolTip(
            "Start: every frame shown from now on (playing or stepping) is saved as a "
            "PNG, overlays included. Stop: the frames are made into a video under "
            "derived/frames/annotated/")
        self.record_btn.toggled.connect(self._toggle_record)
        bar.addWidget(self.record_btn)
        lay.addLayout(bar)
        self._build_jpdf_bar(lay)
        self.rec_status = QLabel("", self)
        self.rec_status.setStyleSheet("color: palette(placeholder-text);")
        self.rec_status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.rec_status)
        self.scrubber.frame_changed.connect(self._record_frame)

    # ------------------------------------------------------------------ #
    # Patch 122jv — JPDF beside the video
    # ------------------------------------------------------------------ #
    def _build_jpdf_bar(self, lay) -> None:  # noqa: ANN001
        from PySide6.QtWidgets import QCheckBox, QComboBox, QSpinBox
        self.jpdf = None
        self._pooled: dict = {}
        bar = QHBoxLayout()
        self.jpdf_cb = QCheckBox("JPDF beside the video", self)
        self._jpdf_tip = (
            "Show the joint distribution of two features (all videos) to the right of "
            "the video, with this frame's position as a red marker and its recent "
            "path as a trail. Recording then saves video and JPDF together.")
        self.jpdf_cb.setToolTip(self._jpdf_tip)
        self.jpdf_cb.toggled.connect(self._jpdf_changed)
        bar.addWidget(self.jpdf_cb)
        bar.addWidget(QLabel("x:", self))
        self.jpdf_x = QComboBox(self)
        bar.addWidget(self.jpdf_x, 1)
        bar.addWidget(QLabel("y:", self))
        self.jpdf_y = QComboBox(self)
        bar.addWidget(self.jpdf_y, 1)
        bar.addWidget(QLabel("Trail", self))
        self.trail_spin = QSpinBox(self)
        self.trail_spin.setRange(0, 3000)
        self.trail_spin.setValue(30)
        self.trail_spin.setSuffix(" frames")
        self.trail_spin.setToolTip("Frames of the path drawn behind the marker")
        bar.addWidget(self.trail_spin)
        for w in (self.jpdf_x, self.jpdf_y):
            w.currentIndexChanged.connect(self._jpdf_changed)
        self.trail_spin.valueChanged.connect(self._trail_changed)
        # Patch 122jx: density bins and outliers
        from PySide6.QtWidgets import QDoubleSpinBox as _QDS
        bar.addWidget(QLabel("Bins", self))
        self.bins_spin = QSpinBox(self)
        self.bins_spin.setRange(10, 400)
        self.bins_spin.setValue(90)
        self.bins_spin.setToolTip("Density bins per axis")
        bar.addWidget(self.bins_spin)
        bar.addWidget(QLabel("Outliers", self))
        self.jout_cb = QComboBox(self)
        self.jout_cb.addItems(["none", "percentile", "mad"])
        self.jout_cb.setCurrentText("percentile")
        self.jout_cb.setToolTip("Background frames left out (and the axes fitted to the "
                                "rest): percentile tails of either feature, or beyond N "
                                "scaled MADs from the median")
        bar.addWidget(self.jout_cb)
        self.jout_val = _QDS(self)
        self.jout_val.setDecimals(2)
        bar.addWidget(self.jout_val)
        self._jout_defaults = {"percentile": 0.5, "mad": 5.0}

        def _sync_out(m: str) -> None:
            self.jout_val.blockSignals(True)
            self.jout_val.setEnabled(m != "none")
            if m == "percentile":
                self.jout_val.setRange(0.01, 25.0)
                self.jout_val.setSuffix(" % tails")
            elif m == "mad":
                self.jout_val.setRange(1.0, 50.0)
                self.jout_val.setSuffix(" × MAD")
            if m != "none":
                self.jout_val.setValue(self._jout_defaults[m])
            self.jout_val.blockSignals(False)

        self.jout_cb.currentTextChanged.connect(_sync_out)
        _sync_out("percentile")
        self.jout_cb.currentTextChanged.connect(self._jpdf_options)
        self.jout_val.valueChanged.connect(self._jpdf_options)
        self.bins_spin.valueChanged.connect(self._jpdf_options)
        lay.addLayout(bar)
        self._fill_jpdf_combos()

    def _fill_jpdf_combos(self) -> None:
        from mufasa.utils.feature_names import display, load_names
        names = load_names(self.config_path) if self.config_path else {}
        for k, cb in enumerate((self.jpdf_x, self.jpdf_y)):
            cur = cb.currentData()
            cb.blockSignals(True)
            cb.clear()
            for c in getattr(self, "features", []):
                cb.addItem(display(c, names), c)
            i = cb.findData(cur) if cur is not None else -1
            cb.setCurrentIndex(i if i >= 0 else min(k, cb.count() - 1))
            cb.blockSignals(False)
        ok = len(getattr(self, "features", [])) >= 2
        self.jpdf_cb.setEnabled(ok)
        self.jpdf_cb.setToolTip(self._jpdf_tip if ok else
                                "Needs two or more features in the viewer")

    def set_jpdf_pair(self, x: str, y: str) -> None:
        self.jpdf_x.setCurrentIndex(self.jpdf_x.findData(x))
        self.jpdf_y.setCurrentIndex(self.jpdf_y.findData(y))

    def _jpdf_kw(self) -> dict:
        m = self.jout_cb.currentText()
        return dict(bins=self.bins_spin.value(), outliers=m,
                    outlier_value=self.jout_val.value() if m != "none" else 0.0)

    def _jpdf_options(self, *_a) -> None:
        if self.jpdf is not None:
            self.jpdf.set_options(**self._jpdf_kw())
            self.scrubber._rerender()

    def _trail_changed(self, v: int) -> None:
        if self.jpdf is not None:
            self.jpdf.trail = int(v)
            self.scrubber._rerender()

    def _jpdf_changed(self, *_a) -> None:
        x, y = self.jpdf_x.currentData(), self.jpdf_y.currentData()
        if not self.jpdf_cb.isChecked() or x is None or y is None:
            self.jpdf = None
            self.scrubber.set_composer(None)
            return
        from mufasa.ui_qt.jpdf_panel import JpdfPanel, pooled_values
        key = (x, y)
        if key not in self._pooled:
            self._pooled[key] = pooled_values(self.config_path, [x, y])
        self.jpdf = JpdfPanel(self.config_path, x, y, self._loaded[x], self._loaded[y],
                              trail=self.trail_spin.value(), pooled=self._pooled[key],
                              **self._jpdf_kw())
        self.scrubber.set_composer(self.jpdf)

    def _ensure_overlay(self) -> None:
        if self.overlay is None:
            from mufasa.ui_qt.pose_overlay import PoseOverlay
            self.overlay = PoseOverlay.for_video(self.config_path, self.video_name,
                                                 self.video_path)
            for name, cb in (("fdlc", self.fdlc_cb), ("kalman", self.kalman_cb)):
                if name in self.overlay.errors:
                    cb.setChecked(False)
                    cb.setEnabled(False)
                    cb.setToolTip("Could not load: " + self.overlay.errors[name])

    def open_style_dialog(self):
        """Patch 122jw: the style popup; edits show on the video at once and
        are saved to the project when it is closed."""
        from mufasa.ui_qt.overlay_style import OverlayStyleDialog
        self._ensure_overlay()
        ov = self.overlay
        if self._style_dlg is None:
            markers = {k: list(pf.markers) for k, pf in (("fdlc", ov.fdlc),
                                                          ("kalman", ov.kalman)) if pf}
            self._style_dlg = OverlayStyleDialog(self.config_path, ov.style, markers,
                                                 ov.edges, self)
            self._style_dlg.changed.connect(self.scrubber._rerender)
        self._style_dlg.show()
        self._style_dlg.raise_()
        return self._style_dlg

    def _overlay_changed(self, *_a) -> None:
        want = self.fdlc_cb.isChecked() or self.kalman_cb.isChecked()
        if want:
            self._ensure_overlay()
        ov = self.overlay
        if ov is not None:
            ov.show_fdlc = self.fdlc_cb.isChecked()
            ov.show_kalman = self.kalman_cb.isChecked()
            ov.show_skeleton = self.skeleton_cb.isChecked()
            ov.p_threshold = self.p_spin.value()
            ov.scale = self.scale_spin.value()
        self.scrubber.set_overlay(ov if want and ov is not None else None)

    def _toggle_record(self, on: bool) -> None:
        from mufasa.ui_qt.pose_overlay import FrameRecorder
        if on:
            self.recorder = FrameRecorder(self.config_path, self.video_name,
                                          self.scrubber.fps,
                                          keep_frames=self.keep_frames_cb.isChecked())
            self.record_btn.setText("■ Stop")
            self.record_btn.setStyleSheet("color: #d03030; font-weight: bold;")
            self.recorder.add(self.scrubber.last_display_frame, self.scrubber.current_frame)
            self.rec_status.setText(f"Recording → {self.recorder.frames_dir}")
            return
        rec, self.recorder = self.recorder, None
        self.record_btn.setText("● Record")
        self.record_btn.setStyleSheet("")
        if rec is None:
            return
        self.rec_status.setText(f"Making the video from {rec.n} frame(s)…")
        self.record_btn.setEnabled(False)
        import threading
        result: dict = {}
        t = threading.Thread(target=lambda: result.update(r=rec.stop()), daemon=True)
        t.start()
        from PySide6.QtCore import QTimer
        timer = QTimer(self)

        def poll() -> None:
            if t.is_alive():
                return
            timer.stop()
            ok, msg = result.get("r", (False, "encoding failed"))
            self.rec_status.setText(("Saved: " if ok else "Recording failed: ") + msg)
            self.record_btn.setEnabled(True)
            self.last_recording = rec.video_path if ok else None

        timer.timeout.connect(poll)
        timer.start(100)
        self._encode_thread = t

    def _record_frame(self, f: int) -> None:
        if self.recorder is not None:
            self.recorder.add(self.scrubber.last_display_frame, f)
            if self.recorder.n % 25 == 0:
                self.rec_status.setText(f"Recording → {self.recorder.frames_dir} · "
                                        f"{self.recorder.n} frame(s)")

    def open_3d(self, pose_path: str | None = None):
        """Patch 122ke: the 3-D viewer for this video (found, given, or chosen)."""
        from mufasa.ui_qt.pose3d_viewer import choose_pose3d, find_pose3d, open_pose3d_viewer
        path = pose_path or find_pose3d(self.video_path, self.config_path)
        if path is None:
            path = choose_pose3d(self, str(Path(self.video_path).parent))
        if not path:
            return None
        self._view3d = open_pose3d_viewer(self, self.video_path, path)
        return self._view3d

    def cleanup(self) -> None:
        if self.recorder is not None:
            self.record_btn.setChecked(False)
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
        dock.viewer = w
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
    dlg.viewer = w
    dlg.show()
    return dlg


__all__ = [
    "FeatureTracePlot",
    "FeatureViewerWidget",
    "open_feature_viewer",
    "robust_range",
    "short_feature_name",
]
