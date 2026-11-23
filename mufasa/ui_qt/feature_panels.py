"""
mufasa.ui_qt.feature_panels
===========================

Patch 122lp: feature histograms beside the traces, a feature panel for the 3-D viewer, and the
viewers in windows of their own.

* :class:`FeatureHistogramPlot` sits beside a :class:`~mufasa.ui_qt.feature_viewer.FeatureTracePlot`,
  one lane per feature at the same height. Each lane is drawn over the same value range as its
  trace, with:
  * the feature's distribution over the whole video (filled);
  * its distribution within the trace's window around the current frame (outlined);
  * for each behaviour label, its distribution in the labelled frames (dashed);
  * the current frame's value (the red line).

  Clicking a bin moves to the next frame whose value falls in it.
* :func:`session_features` gathers the numbers that go with a 3-D pose file, by frame: the pose
  file's own derived columns (pelvis pitch, spine length, …) and each marker's height, and the
  columns of the files beside it (``<video>_gait.parquet``, ``_dynamics.parquet``,
  ``_joints.parquet``): paw loads, the centre of mass's speed, joint angles and torques,
  muscle activations.
* :class:`FeaturePanel`: a searchable list of those, ticked into traces and histograms (the 3-D
  viewer's *Features*).
* :class:`JpdfView`: the joint PDF of two of them, with the current frame's point and its recent
  path, beside the 3-D view (the 3-D viewer's *JPDF*).
* :class:`ViewerWindow`: a window of its own for a viewer (not a dock of the workbench), closed
  and cleaned up on its own.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import QPointF, QRect, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

HIST_BINS = 48
MAX_PANEL_FEATURES = 12
SIBLINGS = ("_gait.parquet", "_dynamics.parquet", "_joints.parquet")


class FeatureHistogramPlot(QWidget):
    """Per-feature value distributions, lane for lane beside a trace plot."""

    frame_clicked = Signal(int)
    W = 240

    def __init__(self, traces, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.traces = traces
        self._cache: dict[str, tuple] = {}
        self.setMinimumWidth(160)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Each feature's values: the whole video (filled), the window around the "
                        "current frame (outline), each behaviour (dashed), now (red). Click a bin: "
                        "the next frame with a value in it.")
        traces.changed.connect(self._refresh)

    def _refresh(self) -> None:
        self.setMinimumHeight(self.traces.minimumHeight())
        self.update()

    def _bins(self, name: str):
        v = self.traces._data[name]
        lo, hi = self.traces._ranges[name]
        key = (id(v), lo, hi)
        c = self._cache.get(name)
        if c is None or c[0] != key:
            edges = np.linspace(lo, hi, HIST_BINS + 1)
            idx = np.clip(np.digitize(v, edges) - 1, 0, HIST_BINS - 1)
            idx = np.where(np.isfinite(v), idx, -1)
            full = np.bincount(idx[idx >= 0], minlength=HIST_BINS).astype(float)
            c = (key, edges, idx, full)
            self._cache[name] = c
        return c[1:]

    def _lanes(self):
        """(name, top, height) per feature lane, as the trace plot lays them out."""
        t = self.traces
        y = 6 + len(t._labels) * t.LABEL_LANE_H
        out = []
        for name in t._data:
            out.append((name, y + 2, t.LANE_H - 6))
            y += t.LANE_H
        return out

    def histograms(self, name: str) -> dict:
        """The lane's histograms: whole video, window, per label; normalised to their peaks."""
        edges, idx, full = self._bins(name)
        t = self.traces
        half = t._half()
        a, b = max(0, t._frame - half), max(0, t._frame + half + 1)
        w = idx[a:b]
        win = np.bincount(w[w >= 0], minlength=HIST_BINS).astype(float)
        labs = {}
        for lab, arr in list(t._labels.items())[:3]:
            m = (np.asarray(arr)[: len(idx)] == 1) & (idx[: len(arr)] >= 0)
            if m.any():
                labs[lab] = np.bincount(idx[: len(arr)][m], minlength=HIST_BINS).astype(float)
        norm = lambda h: h / h.max() if h.max() > 0 else h          # noqa: E731
        return {"edges": edges, "all": norm(full), "window": norm(win),
                "labels": {k: norm(v) for k, v in labs.items()}}

    def mousePressEvent(self, ev) -> None:  # noqa: ANN001, N802
        if ev.button() != Qt.LeftButton:
            return super().mousePressEvent(ev)
        x, y = ev.position().x(), ev.position().y()
        for name, top, h in self._lanes():
            if top <= y <= top + h:
                k = int(np.clip((x - 4) / max(1, self.width() - 8) * HIST_BINS, 0, HIST_BINS - 1))
                f = self.next_frame_in_bin(name, k)
                if f is not None:
                    self.frame_clicked.emit(f)
                ev.accept()
                return
        super().mousePressEvent(ev)

    def next_frame_in_bin(self, name: str, k: int) -> int | None:
        """The first frame after the current one (wrapping round) whose value is in bin ``k``."""
        _, idx, _ = self._bins(name)
        hits = np.flatnonzero(idx == k)
        if not len(hits):
            return None
        after = hits[hits > self.traces._frame]
        return int(after[0] if len(after) else hits[0])

    def paintEvent(self, _ev) -> None:  # noqa: ANN001, N802
        from mufasa.ui_qt.feature_viewer import LABEL_COLORS, TRACE_COLORS
        p = QPainter(self)
        p.fillRect(self.rect(), self.palette().base())
        text = self.palette().text().color()
        x0, pw = 4, max(10, self.width() - 8)
        t = self.traces
        if not t._data:
            p.end()
            return
        for k, (name, top, h) in enumerate(self._lanes()):
            if k % 2 == 0:
                p.fillRect(0, top - 2, self.width(), t.LANE_H, self.palette().alternateBase())
            H = self.histograms(name)
            col = QColor(*TRACE_COLORS[k % len(TRACE_COLORS)])
            bw = pw / HIST_BINS
            fill = QColor(col)
            fill.setAlpha(70)
            p.setPen(Qt.NoPen)
            p.setBrush(fill)
            for i, v in enumerate(H["all"]):
                if v > 0:
                    bh = v * h
                    p.drawRect(QRect(int(x0 + i * bw), int(top + h - bh), max(1, int(bw)), int(bh)))
            p.setBrush(Qt.NoBrush)

            def outline(hv, pen):
                path = QPainterPath()
                path.moveTo(QPointF(x0, top + h))
                for i, v in enumerate(hv):
                    yv = top + h - v * h
                    path.lineTo(QPointF(x0 + i * bw, yv))
                    path.lineTo(QPointF(x0 + (i + 1) * bw, yv))
                path.lineTo(QPointF(x0 + pw, top + h))
                p.setPen(pen)
                p.drawPath(path)

            outline(H["window"], QPen(col, 1.4))
            for j, (_, hv) in enumerate(H["labels"].items()):
                lc = QColor(*LABEL_COLORS[j % len(LABEL_COLORS)][:3])
                pen = QPen(lc, 1.2, Qt.DashLine)
                outline(hv, pen)
            v = t._data[name]
            cur = v[t._frame] if 0 <= t._frame < len(v) else np.nan
            if np.isfinite(cur):
                lo, hi = t._ranges[name]
                xc = x0 + (np.clip(cur, lo, hi) - lo) / (hi - lo) * pw
                p.setPen(QPen(QColor(220, 40, 40), 1.5))
                p.drawLine(QPointF(xc, top), QPointF(xc, top + h))
            dim = QColor(text)
            dim.setAlpha(120)
            p.setPen(dim)
            lo, hi = t._ranges[name]
            fm = p.fontMetrics()
            p.drawText(QRect(x0, top + h - fm.height(), pw, fm.height()), Qt.AlignLeft, f"{lo:.3g}")
            p.drawText(QRect(x0, top + h - fm.height(), pw, fm.height()), Qt.AlignRight, f"{hi:.3g}")
        p.end()


# ---------------------------------------------------------------------------------------------
# the numbers that go with a 3-D pose file


def _stem(pose_path: Path) -> str:
    s = pose_path.stem
    for suf in ("_3d_heights", "_3d_body", "_3d_smooth", "_3d_h", "_3d"):
        if s.endswith(suf):
            return s[: -len(suf)]
    return s


def session_features(pose_path, n_frames: int | None = None) -> pd.DataFrame:
    """Numeric columns by frame (row = frame number) from a 3-D pose file and the files beside
    it: ``<file>:<column>`` names (``3d:``, ``gait:``, ``dynamics:``, ``joints:``)."""
    from mufasa.pose3d.io import marker_names, read_pose3d
    pose_path = Path(pose_path)
    df, _ = read_pose3d(pose_path)
    names = marker_names(df)
    marker_cols = {f"{n}_{s}" for n in names for s in ("x", "y", "z", "likelihood", "reproj_px")}
    tables = []
    fr = df["frame"].to_numpy(int) if "frame" in df else np.arange(len(df))
    own = {f"3d:{c}": df[c] for c in df.columns if c != "frame" and c not in marker_cols
           and pd.api.types.is_numeric_dtype(df[c])}
    own.update({f"3d:{n} height": df[f"{n}_z"] for n in names})
    tables.append((fr, pd.DataFrame(own)))
    stem = _stem(pose_path)
    for suf in SIBLINGS:
        f = pose_path.with_name(stem + suf)
        if not f.exists():
            continue
        try:
            t = pd.read_parquet(f)
        except Exception:
            continue
        tag = suf.split(".")[0].strip("_")
        if "frame" not in t:
            continue
        cols = {f"{tag}:{c}": t[c] for c in t.columns if c != "frame"
                and (pd.api.types.is_numeric_dtype(t[c]) or pd.api.types.is_bool_dtype(t[c]))}
        tables.append((t["frame"].to_numpy(int), pd.DataFrame(cols)))
    N = int(max([n_frames or 0] + [int(f.max()) + 1 for f, _ in tables if len(f)]))
    out = {}
    for f, t in tables:
        ok = (f >= 0) & (f < N)
        for c in t.columns:
            a = np.full(N, np.nan)
            a[f[ok]] = pd.to_numeric(t[c], errors="coerce").to_numpy(float)[ok]
            out[c] = a
    return pd.DataFrame(out)


class FeaturePanel(QWidget):
    """A searchable list of features, the ticked ones as traces with histograms beside them."""

    frame_clicked = Signal(int)

    def __init__(self, table: pd.DataFrame, fps: float = 30.0, labels: dict | None = None,
                 parent: QWidget | None = None, preselect=()) -> None:
        super().__init__(parent)
        from mufasa.ui_qt.feature_viewer import FeatureTracePlot
        self.table = table
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        side = QVBoxLayout()
        self.filter = QLineEdit(self)
        self.filter.setPlaceholderText(f"Filter {table.shape[1]} features…")
        self.filter.setClearButtonEnabled(True)
        side.addWidget(self.filter)
        self.list = QListWidget(self)
        for c in table.columns:
            it = QListWidgetItem(c)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if c in preselect else Qt.Unchecked)
            self.list.addItem(it)
        side.addWidget(self.list, 1)
        self.count = QLabel("", self)
        self.count.setStyleSheet("color: palette(placeholder-text);")
        side.addWidget(self.count)
        self.hist_cb = QCheckBox("Histograms", self)
        self.hist_cb.setChecked(True)
        side.addWidget(self.hist_cb)
        sw = QWidget(self)
        sw.setLayout(side)
        sw.setMaximumWidth(280)
        split = QSplitter(Qt.Horizontal, self)
        split.addWidget(sw)
        self.traces = FeatureTracePlot(split)
        self.traces.set_fps(fps)
        if labels:
            self.traces.set_labels(labels)
        split.addWidget(self.traces)
        self.hist = FeatureHistogramPlot(self.traces, split)
        split.addWidget(self.hist)
        split.setStretchFactor(1, 4)
        split.setStretchFactor(2, 1)
        lay.addWidget(split)
        self.filter.textChanged.connect(self._filter)
        self.list.itemChanged.connect(self._ticked)
        self.hist_cb.toggled.connect(self.hist.setVisible)
        self.traces.frame_clicked.connect(self.frame_clicked)
        self.hist.frame_clicked.connect(self.frame_clicked)
        self._ticked()

    def _filter(self, s: str) -> None:
        s = s.lower().strip()
        for i in range(self.list.count()):
            it = self.list.item(i)
            it.setHidden(bool(s) and s not in it.text().lower())

    def selected(self) -> list[str]:
        return [self.list.item(i).text() for i in range(self.list.count())
                if self.list.item(i).checkState() == Qt.Checked]

    def select(self, cols) -> None:
        self.list.blockSignals(True)
        for i in range(self.list.count()):
            it = self.list.item(i)
            it.setCheckState(Qt.Checked if it.text() in cols else Qt.Unchecked)
        self.list.blockSignals(False)
        self._ticked()

    def _ticked(self, *_a) -> None:
        cols = self.selected()[:MAX_PANEL_FEATURES]
        self.traces.set_features(self.table[cols] if cols else pd.DataFrame())
        self.count.setText(f"{len(cols)} shown" + (f" (at most {MAX_PANEL_FEATURES})"
                                                    if len(self.selected()) > MAX_PANEL_FEATURES else ""))

    def set_current_frame(self, f: int) -> None:
        self.traces.set_current_frame(f)


class JpdfView(QWidget):
    """122lp: a joint PDF of two features with the current frame's point and its recent path, as
    a panel of its own (beside the 3-D view): :class:`mufasa.ui_qt.jpdf_panel.JpdfPanel` drawn
    square at the panel's height."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.jpdf = None
        self.frame = 0
        self.setMinimumSize(200, 200)

    def set_pair(self, table: pd.DataFrame, x: str, y: str, trail: int = 30, **kw) -> None:
        from mufasa.ui_qt.jpdf_panel import JpdfPanel
        if x not in table or y not in table:
            self.jpdf = None
        else:
            self.jpdf = JpdfPanel(None, x, y, table[x].to_numpy(float), table[y].to_numpy(float),
                                  trail=trail, title="this video", **kw)
        self.update()

    def set_trail(self, n: int) -> None:
        if self.jpdf is not None:
            self.jpdf.trail = int(n)
            self.update()

    def set_current_frame(self, f: int) -> None:
        self.frame = int(f)
        if self.isVisible():
            self.update()

    def sizeHint(self):  # noqa: N802
        from PySide6.QtCore import QSize
        h = max(200, self.height())
        return QSize(h, h)

    def image(self):
        """The panel at the widget's height (BGR array), or None."""
        if self.jpdf is None:
            return None
        h = max(120, min(self.height(), self.width()))
        return self.jpdf.panel(h, self.frame)

    def paintEvent(self, _ev) -> None:  # noqa: ANN001, N802
        from PySide6.QtGui import QImage
        p = QPainter(self)
        p.fillRect(self.rect(), self.palette().base())
        img = self.image()
        if img is None:
            p.setPen(self.palette().text().color())
            p.drawText(self.rect(), Qt.AlignCenter, "Choose two features for the JPDF")
            p.end()
            return
        rgb = np.ascontiguousarray(img[:, :, ::-1])
        q = QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1], QImage.Format_RGB888)
        x = (self.width() - rgb.shape[1]) // 2
        y = (self.height() - rgb.shape[0]) // 2
        p.drawImage(x, y, q)
        p.end()


# ---------------------------------------------------------------------------------------------
# windows of their own

_OPEN: list = []


class ViewerWindow(QMainWindow):
    """A viewer in a window of its own (not docked in the workbench); cleans up on close."""

    def __init__(self, widget: QWidget, title: str, size=(1150, 800)) -> None:
        super().__init__(None)
        self.viewer = widget
        self.setWindowTitle(title)
        self.setCentralWidget(widget)
        self.resize(*size)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        _OPEN.append(self)

    def closeEvent(self, ev) -> None:  # noqa: ANN001, N802
        try:
            if hasattr(self.viewer, "cleanup"):
                self.viewer.cleanup()
            self.viewer.close()
        finally:
            if self in _OPEN:
                _OPEN.remove(self)
        super().closeEvent(ev)


def open_window(widget: QWidget, title: str, size=(1150, 800)) -> ViewerWindow:
    w = ViewerWindow(widget, title, size)
    w.show()
    return w


__all__ = ["FeatureHistogramPlot", "FeaturePanel", "HIST_BINS", "JpdfView", "ViewerWindow", "open_window",
           "session_features"]
