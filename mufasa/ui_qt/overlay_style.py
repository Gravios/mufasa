"""
mufasa.ui_qt.overlay_style
==========================

Visual style of the pose overlays in the feature viewer (patch 122jw), per
source (FDLC / Kalman), kept per project in
``logs/measures/pose_overlay_style.toml``:

* **markers** — shown, shape (filled circle / ring / cross / square /
  diamond), size, line width, opacity, outline, colour (palette per marker
  or one colour), marker names next to them; each marker can be hidden or
  given its own colour;
* **skeleton** — shown, colour, line width, opacity; each segment can be
  hidden or given its own colour.

:class:`OverlayStyleDialog` edits it live (changes show on the video at
once) and saves on Close; Reset restores the defaults.
"""
from __future__ import annotations

import functools
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import cv2
import numpy as np

FILENAME = "pose_overlay_style.toml"
SHAPES = ("filled circle", "ring", "cross", "square", "diamond")
COLOR_MODES = ("palette", "single")


def edge_key(a: str, b: str) -> str:
    return f"{a}|{b}"


@dataclass
class SourceStyle:
    show_markers: bool = True
    shape: str = "filled circle"
    size: int = 0                      # radius in px; 0 = from the frame size
    line_width: int = 2
    opacity: float = 1.0
    outline: bool = True               # thin black rim (filled shapes)
    color_mode: str = "palette"
    color: str = "#ffffff"
    labels: bool = False
    hidden_markers: list[str] = field(default_factory=list)
    marker_colors: dict[str, str] = field(default_factory=dict)
    show_skeleton: bool = True
    skeleton_color: str = "#ebebeb"
    skeleton_width: int = 2
    skeleton_opacity: float = 1.0
    hidden_edges: list[str] = field(default_factory=list)
    edge_colors: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict, base: "SourceStyle") -> "SourceStyle":
        out = cls(**asdict(base))
        names = {f.name for f in fields(cls)}
        for k, v in (d or {}).items():
            if k in names:
                setattr(out, k, v)
        if out.shape not in SHAPES:
            out.shape = base.shape
        if out.color_mode not in COLOR_MODES:
            out.color_mode = base.color_mode
        out.hidden_markers = [str(x) for x in out.hidden_markers]
        out.hidden_edges = [str(x) for x in out.hidden_edges]
        out.marker_colors = {str(k): str(v) for k, v in dict(out.marker_colors).items()}
        out.edge_colors = {str(k): str(v) for k, v in dict(out.edge_colors).items()}
        return out


def default_fdlc() -> SourceStyle:
    return SourceStyle(shape="ring", outline=False, show_skeleton=False,
                       skeleton_color="#ffd27f", skeleton_width=1)


def default_kalman() -> SourceStyle:
    return SourceStyle(shape="filled circle", outline=True, show_skeleton=True)


@dataclass
class OverlayStyle:
    fdlc: SourceStyle = field(default_factory=default_fdlc)
    kalman: SourceStyle = field(default_factory=default_kalman)
    legend: bool = True

    def to_dict(self) -> dict:
        return {"legend": self.legend, "fdlc": asdict(self.fdlc),
                "kalman": asdict(self.kalman)}

    @classmethod
    def from_dict(cls, d: dict) -> "OverlayStyle":
        return cls(fdlc=SourceStyle.from_dict(d.get("fdlc", {}), default_fdlc()),
                   kalman=SourceStyle.from_dict(d.get("kalman", {}), default_kalman()),
                   legend=bool(d.get("legend", True)))


def style_path(config_path) -> Path:
    from mufasa.utils.feature_names import names_path
    return names_path(config_path).with_name(FILENAME)


def load_style(config_path) -> OverlayStyle:
    if not config_path:
        return OverlayStyle()
    try:
        with open(style_path(config_path), "rb") as fh:
            return OverlayStyle.from_dict(tomllib.load(fh))
    except (OSError, tomllib.TOMLDecodeError, TypeError, ValueError):
        return OverlayStyle()


def save_style(config_path, style: OverlayStyle) -> Path:
    from mufasa.project_layout import write_project_toml
    p = style_path(config_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    write_project_toml(p, style.to_dict())
    return p


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def hex_bgr(h: str) -> tuple[int, int, int]:
    h = str(h).lstrip("#")
    if len(h) != 6:
        return (255, 255, 255)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def palette_hex(k: int) -> str:
    from mufasa.tools.pose_viewer import MARKER_PALETTE
    r, g, b = MARKER_PALETTE[k % len(MARKER_PALETTE)]
    return f"#{r:02x}{g:02x}{b:02x}"


def marker_hex(st: SourceStyle, k: int, name: str) -> str:
    if name in st.marker_colors:
        return st.marker_colors[name]
    return st.color if st.color_mode == "single" else palette_hex(k)


def _blend(out: np.ndarray, layer: np.ndarray, mask: np.ndarray, alpha: float) -> None:
    if alpha >= 0.999:
        out[mask] = layer[mask]
    elif alpha > 0:
        out[mask] = (alpha * layer[mask] + (1 - alpha) * out[mask]).astype(out.dtype)


def _shape(img, c, r, shape, color, lw, outline) -> None:  # noqa: ANN001
    x, y = c
    aa = cv2.LINE_AA
    if shape == "filled circle":
        cv2.circle(img, c, r, color, -1, aa)
        if outline:
            cv2.circle(img, c, r, (0, 0, 0), 1, aa)
    elif shape == "ring":
        cv2.circle(img, c, r, color, max(1, lw), aa)
    elif shape == "cross":
        cv2.line(img, (x - r, y - r), (x + r, y + r), color, max(1, lw), aa)
        cv2.line(img, (x - r, y + r), (x + r, y - r), color, max(1, lw), aa)
    elif shape == "square":
        cv2.rectangle(img, (x - r, y - r), (x + r, y + r), color, -1, aa)
        if outline:
            cv2.rectangle(img, (x - r, y - r), (x + r, y + r), (0, 0, 0), 1, aa)
    elif shape == "diamond":
        pts = np.array([(x, y - r), (x + r, y), (x, y + r), (x - r, y)], np.int32)
        cv2.fillPoly(img, [pts], color, aa)
        if outline:
            cv2.polylines(img, [pts], True, (0, 0, 0), 1, aa)


def draw_source(out: np.ndarray, pts: dict[str, tuple[int, int]], markers: list[str],
                edges: list[tuple[str, str]], st: SourceStyle, auto_r: int,
                skeleton: bool = True) -> None:
    """Draw one source's skeleton then markers onto ``out`` (in place)."""
    if skeleton and st.show_skeleton and edges:
        layer, mask = out.copy(), np.zeros(out.shape[:2], np.uint8)
        hidden = set(st.hidden_edges)
        for a, b in edges:
            if a in pts and b in pts and edge_key(a, b) not in hidden:
                col = hex_bgr(st.edge_colors.get(edge_key(a, b), st.skeleton_color))
                lw = max(1, int(st.skeleton_width))
                cv2.line(layer, pts[a], pts[b], col, lw, cv2.LINE_AA)
                cv2.line(mask, pts[a], pts[b], 255, lw + 2)
        _blend(out, layer, mask > 0, float(st.skeleton_opacity))
    if st.show_markers:
        layer, mask = out.copy(), np.zeros(out.shape[:2], np.uint8)
        r = int(st.size) if st.size > 0 else auto_r
        hidden = set(st.hidden_markers)
        h = out.shape[0]
        fs = max(0.35, h / 1400)
        for k, m in enumerate(markers):
            if m not in pts or m in hidden:
                continue
            col = hex_bgr(marker_hex(st, k, m))
            _shape(layer, pts[m], r, st.shape, col, int(st.line_width), st.outline)
            cv2.circle(mask, pts[m], r + 2, 255, -1)
            if st.labels:
                org = (pts[m][0] + r + 3, pts[m][1] - r)
                cv2.putText(layer, m, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 3,
                            cv2.LINE_AA)
                cv2.putText(layer, m, org, cv2.FONT_HERSHEY_SIMPLEX, fs, col, 1, cv2.LINE_AA)
                (tw, th), _ = cv2.getTextSize(m, cv2.FONT_HERSHEY_SIMPLEX, fs, 3)
                cv2.rectangle(mask, (org[0] - 2, org[1] - th - 3), (org[0] + tw + 2,
                                                                     org[1] + 4), 255, -1)
        _blend(out, layer, mask > 0, float(st.opacity))


# --------------------------------------------------------------------------- #
# Dialog
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=1)
def _dialog_classes():  # Qt imported lazily so the drawing code stays headless
    from PySide6.QtCore import Qt, Signal
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import (
        QAbstractItemView,
        QCheckBox,
        QColorDialog,
        QComboBox,
        QDialog,
        QDoubleSpinBox,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QPushButton,
        QSpinBox,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )

    class ColorButton(QPushButton):
        changed = Signal(str)

        def __init__(self, color: str, parent=None) -> None:  # noqa: ANN001
            super().__init__(parent)
            self.setFixedWidth(56)
            self.set_color(color)
            self.clicked.connect(self._pick)

        def set_color(self, color: str) -> None:
            self.color = color
            self.setStyleSheet(f"background-color: {color}; border: 1px solid #555;")

        def _pick(self) -> None:
            c = QColorDialog.getColor(QColor(self.color), self, "Colour")
            if c.isValid():
                self.set_color(c.name())
                self.changed.emit(self.color)

    class SourcePage(QWidget):
        """Editor for one SourceStyle; emits ``changed`` after each edit."""

        changed = Signal()

        def __init__(self, style: SourceStyle, markers: list[str],
                     edges: list[tuple[str, str]], parent=None) -> None:  # noqa: ANN001
            super().__init__(parent)
            self.st, self.markers, self.edges = style, list(markers), list(edges)
            lay = QHBoxLayout(self)
            # -- markers
            mg = QGroupBox("Markers", self)
            ml = QVBoxLayout(mg)
            f = QFormLayout()
            self.show_m = QCheckBox("Show markers", mg)
            f.addRow(self.show_m)
            self.shape = QComboBox(mg)
            self.shape.addItems(list(SHAPES))
            f.addRow("Shape:", self.shape)
            self.size = QSpinBox(mg)
            self.size.setRange(0, 60)
            self.size.setSpecialValueText("auto")
            self.size.setSuffix(" px")
            self.size.setToolTip("Radius; auto = from the frame size")
            f.addRow("Size:", self.size)
            self.lw = QSpinBox(mg)
            self.lw.setRange(1, 20)
            self.lw.setSuffix(" px")
            self.lw.setToolTip("Line width of rings and crosses")
            f.addRow("Line width:", self.lw)
            self.op = QDoubleSpinBox(mg)
            self.op.setRange(0.05, 1.0)
            self.op.setSingleStep(0.05)
            f.addRow("Opacity:", self.op)
            self.outline = QCheckBox("Black outline (filled shapes)", mg)
            f.addRow(self.outline)
            crow = QHBoxLayout()
            self.cmode = QComboBox(mg)
            self.cmode.addItems(["palette (one colour per marker)", "single colour"])
            crow.addWidget(self.cmode)
            self.color = ColorButton(style.color, mg)
            crow.addWidget(self.color)
            f.addRow("Colour:", crow)
            self.labels = QCheckBox("Marker names", mg)
            f.addRow(self.labels)
            ml.addLayout(f)
            self.mtable = self._table(["Show", "Marker", "Colour"], len(self.markers), mg)
            ml.addWidget(self.mtable, 1)
            ml.addLayout(self._all_none(self.mtable))
            lay.addWidget(mg, 1)
            # -- skeleton
            sg = QGroupBox("Skeleton", self)
            sl = QVBoxLayout(sg)
            f2 = QFormLayout()
            self.show_s = QCheckBox("Show skeleton", sg)
            f2.addRow(self.show_s)
            self.scolor = ColorButton(style.skeleton_color, sg)
            f2.addRow("Colour:", self.scolor)
            self.sw = QSpinBox(sg)
            self.sw.setRange(1, 20)
            self.sw.setSuffix(" px")
            f2.addRow("Line width:", self.sw)
            self.sop = QDoubleSpinBox(sg)
            self.sop.setRange(0.05, 1.0)
            self.sop.setSingleStep(0.05)
            f2.addRow("Opacity:", self.sop)
            sl.addLayout(f2)
            self.etable = self._table(["Show", "Segment", "Colour"], len(self.edges), sg)
            sl.addWidget(self.etable, 1)
            sl.addLayout(self._all_none(self.etable))
            lay.addWidget(sg, 1)
            self.load(style)
            for w, sig in ((self.show_m, "toggled"), (self.shape, "currentIndexChanged"),
                           (self.size, "valueChanged"), (self.lw, "valueChanged"),
                           (self.op, "valueChanged"), (self.outline, "toggled"),
                           (self.cmode, "currentIndexChanged"), (self.labels, "toggled"),
                           (self.show_s, "toggled"), (self.sw, "valueChanged"),
                           (self.sop, "valueChanged")):
                getattr(w, sig).connect(self._edited)
            self.color.changed.connect(lambda _c: self._edited())
            self.scolor.changed.connect(lambda _c: self._edited())
            self.mtable.itemChanged.connect(lambda _i: self._edited())
            self.etable.itemChanged.connect(lambda _i: self._edited())
            self.mtable.cellDoubleClicked.connect(lambda r, c: self._pick_row(self.mtable, r))
            self.etable.cellDoubleClicked.connect(lambda r, c: self._pick_row(self.etable, r))

        @staticmethod
        def _table(headers, n, parent):  # noqa: ANN001
            t = QTableWidget(n, 3, parent)
            t.setHorizontalHeaderLabels(headers)
            t.verticalHeader().setVisible(False)
            t.setSelectionMode(QAbstractItemView.NoSelection)
            t.setEditTriggers(QAbstractItemView.NoEditTriggers)
            hd = t.horizontalHeader()
            hd.setSectionResizeMode(0, QHeaderView.ResizeToContents)
            hd.setSectionResizeMode(1, QHeaderView.Stretch)
            hd.setSectionResizeMode(2, QHeaderView.ResizeToContents)
            t.setToolTip("Untick to hide · double-click a row to choose its own colour "
                         "(Shift+double-click: back to the default)")
            return t

        def _all_none(self, table):  # noqa: ANN001
            row = QHBoxLayout()
            for txt, state in (("All", Qt.Checked), ("None", Qt.Unchecked)):
                b = QPushButton(txt, self)
                b.clicked.connect(lambda _c=False, s=state, t=table: self._set_all(t, s))
                row.addWidget(b)
            row.addStretch()
            return row

        def _set_all(self, table, state) -> None:  # noqa: ANN001
            table.blockSignals(True)
            for r in range(table.rowCount()):
                table.item(r, 0).setCheckState(state)
            table.blockSignals(False)
            self._edited()

        def _pick_row(self, table, r: int) -> None:  # noqa: ANN001
            from PySide6.QtWidgets import QApplication
            key = table.item(r, 1).data(Qt.UserRole)
            colors = self.st.marker_colors if table is self.mtable else self.st.edge_colors
            if QApplication.keyboardModifiers() & Qt.ShiftModifier:
                colors.pop(key, None)
            else:
                cur = table.item(r, 2).data(Qt.UserRole) or "#ffffff"
                c = QColorDialog.getColor(QColor(cur), self, f"Colour of {key}")
                if not c.isValid():
                    return
                colors[key] = c.name()
            self._fill_colours()
            self.changed.emit()

        def load(self, st: SourceStyle) -> None:
            self.st = st
            widgets = (self.show_m, self.shape, self.size, self.lw, self.op, self.outline,
                       self.cmode, self.labels, self.show_s, self.sw, self.sop,
                       self.mtable, self.etable)
            for w in widgets:
                w.blockSignals(True)
            self.show_m.setChecked(st.show_markers)
            self.shape.setCurrentText(st.shape)
            self.size.setValue(int(st.size))
            self.lw.setValue(int(st.line_width))
            self.op.setValue(float(st.opacity))
            self.outline.setChecked(st.outline)
            self.cmode.setCurrentIndex(COLOR_MODES.index(st.color_mode))
            self.color.set_color(st.color)
            self.labels.setChecked(st.labels)
            self.show_s.setChecked(st.show_skeleton)
            self.scolor.set_color(st.skeleton_color)
            self.sw.setValue(int(st.skeleton_width))
            self.sop.setValue(float(st.skeleton_opacity))
            for t, keys, hidden in ((self.mtable, self.markers, set(st.hidden_markers)),
                                    (self.etable, [edge_key(a, b) for a, b in self.edges],
                                     set(st.hidden_edges))):
                for r, key in enumerate(keys):
                    show = QTableWidgetItem("")
                    show.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                    show.setCheckState(Qt.Unchecked if key in hidden else Qt.Checked)
                    t.setItem(r, 0, show)
                    name = QTableWidgetItem(key.replace("|", " – "))
                    name.setData(Qt.UserRole, key)
                    t.setItem(r, 1, name)
                    t.setItem(r, 2, QTableWidgetItem(""))
            self._fill_colours()
            for w in widgets:
                w.blockSignals(False)
            self._sync_enabled()

        def _fill_colours(self) -> None:
            for t in (self.mtable, self.etable):
                t.blockSignals(True)
            for r, m in enumerate(self.markers):
                c = marker_hex(self.st, r, m)
                it = self.mtable.item(r, 2)
                it.setBackground(QColor(c))
                it.setData(Qt.UserRole, c)
                it.setText("own" if m in self.st.marker_colors else "")
            for r, (a, b) in enumerate(self.edges):
                k = edge_key(a, b)
                c = self.st.edge_colors.get(k, self.st.skeleton_color)
                it = self.etable.item(r, 2)
                it.setBackground(QColor(c))
                it.setData(Qt.UserRole, c)
                it.setText("own" if k in self.st.edge_colors else "")
            for t in (self.mtable, self.etable):
                t.blockSignals(False)

        def _sync_enabled(self) -> None:
            self.color.setEnabled(self.cmode.currentIndex() == 1)
            self.lw.setEnabled(self.shape.currentText() in ("ring", "cross"))
            self.outline.setEnabled(self.shape.currentText() in ("filled circle", "square",
                                                                 "diamond"))

        def _edited(self, *_a) -> None:
            st = self.st
            st.show_markers = self.show_m.isChecked()
            st.shape = self.shape.currentText()
            st.size = self.size.value()
            st.line_width = self.lw.value()
            st.opacity = round(self.op.value(), 3)
            st.outline = self.outline.isChecked()
            st.color_mode = COLOR_MODES[self.cmode.currentIndex()]
            st.color = self.color.color
            st.labels = self.labels.isChecked()
            st.show_skeleton = self.show_s.isChecked()
            st.skeleton_color = self.scolor.color
            st.skeleton_width = self.sw.value()
            st.skeleton_opacity = round(self.sop.value(), 3)
            st.hidden_markers = [self.markers[r] for r in range(self.mtable.rowCount())
                                 if self.mtable.item(r, 0).checkState() != Qt.Checked]
            st.hidden_edges = [edge_key(*self.edges[r]) for r in range(self.etable.rowCount())
                               if self.etable.item(r, 0).checkState() != Qt.Checked]
            self._fill_colours()
            self._sync_enabled()
            self.changed.emit()

    class OverlayStyleDialog(QDialog):
        """Non-modal editor; ``changed`` fires on every edit, the style is
        saved to the project on Close."""

        changed = Signal()

        def __init__(self, config_path, style: OverlayStyle,
                     markers: dict[str, list[str]], edges: list[tuple[str, str]],
                     parent=None) -> None:  # noqa: ANN001
            super().__init__(parent)
            self.setWindowTitle("Pose overlay style")
            self.resize(980, 640)
            self.config_path, self.style = config_path, style
            lay = QVBoxLayout(self)
            self.tabs = QTabWidget(self)
            self.pages = {}
            for key, title in (("fdlc", "FDLC (tracker)"), ("kalman", "Kalman (smoothed)")):
                page = SourcePage(getattr(style, key), markers.get(key, []), edges, self)
                page.changed.connect(self.changed.emit)
                self.pages[key] = page
                self.tabs.addTab(page, title)
            lay.addWidget(self.tabs, 1)
            row = QHBoxLayout()
            self.legend = QCheckBox("Legend in the corner", self)
            self.legend.setChecked(style.legend)
            self.legend.toggled.connect(self._legend)
            row.addWidget(self.legend)
            row.addStretch()
            reset = QPushButton("Reset to defaults", self)
            reset.clicked.connect(self.reset)
            row.addWidget(reset)
            close = QPushButton("Close", self)
            close.clicked.connect(self.close)
            row.addWidget(close)
            lay.addLayout(row)

        def _legend(self, on: bool) -> None:
            self.style.legend = on
            self.changed.emit()

        def reset(self) -> None:
            fresh = OverlayStyle()
            self.style.fdlc, self.style.kalman, self.style.legend = (fresh.fdlc,
                                                                      fresh.kalman, True)
            self.pages["fdlc"].load(self.style.fdlc)
            self.pages["kalman"].load(self.style.kalman)
            self.legend.setChecked(True)
            self.changed.emit()

        def save(self) -> None:
            if self.config_path:
                save_style(self.config_path, self.style)

        def closeEvent(self, ev) -> None:  # noqa: ANN001
            self.save()
            super().closeEvent(ev)

    return OverlayStyleDialog


def OverlayStyleDialog(*args, **kwargs):  # noqa: N802
    """Create the style dialog (see :func:`_dialog_classes`)."""
    return _dialog_classes()(*args, **kwargs)


__all__ = ["COLOR_MODES", "FILENAME", "OverlayStyle", "OverlayStyleDialog", "SHAPES",
           "SourceStyle", "draw_source", "edge_key", "load_style", "save_style",
           "style_path"]
