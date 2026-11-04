"""
mufasa.ui_qt.dialogs.roi_define_panel
=====================================

Unified Qt panel for defining ROIs across all videos in a project.

Layout (patch 122ij redesign)
-----------------------------

::

    ┌ Videos ──────┬ [Select V][Rect R][Circle C][Polygon P]  Fit 100% ☑Labels ┬ ROIs on this video ─────────┐
    │ filter…      │                                                         │ Name     Type   Colour  Px │
    │ ✓ vid_01  3  │                                                         │ arena    Rect   Red     3  │
    │   vid_02     │                    video frame + ROIs                   │ nest     Poly   Blue    3  │
    │ ✓ vid_03  2  │                                                         │ [Duplicate] [Delete]       │
    │              │ hint: what the current tool expects next                │ New shape                  │
    │              │ ⏮ −1s −1f ━━━━━━●━━━━━━━ +1f +1s ⏭   Frame 47 / 900     │  Name  [auto]              │
    │              │                                                         │  Colour [Red ▾] Width [3]  │
    │              │                                                         │ Copy to other videos       │
    │              │                                                         │  [Apply to all…]           │
    │              │                                                         │  [Apply to selected…]      │
    │              │                                                         │ [Import / standardize…]    │
    │              │                                                         │ [Clear this video…]        │
    ├──────────────┴─────────────────────────────────────────────────────────┴────────────────────────────┤
    │ status                         ● Unsaved changes   [Undo] [Redo]    [Close] [Save] [Save && close]  │
    └──────────────────────────────────────────────────────────────────────────────────────────────────────┘

How it works
------------

* **Pick a tool and draw.** Rectangle / circle / polygon tools are sticky —
  draw as many shapes as you like. Each new shape is named from the
  "New shape → Name" box, or automatically (``polygon_1``, ``polygon_2``…)
  when the box is empty. Rename any ROI by double-clicking its name.
* **Polygons have no vertex limit.** (Before 122ij the canvas simplified
  every polygon with a 20 px tolerance, which silently capped them at a
  handful of vertices.)
* **Select tool** moves / reshapes shapes, including individual polygon
  vertices; colour and line width are edited per row in the table.
* **Undo / redo** (Ctrl+Z / Ctrl+Shift+Z) cover every change on the current
  video since it was opened.
* Switching videos auto-saves. "Apply to all…" now says how many other
  videos' existing ROIs it will replace before doing it.

Keyboard: V/R/C/P tools · , / . step a frame (Shift: a second) · PgUp/PgDn
switch video · Ctrl+S save · F fit · wheel zoom · middle-drag pan.

Compatibility
-------------

Same H5 file (``ROI_definitions.h5``), same :class:`ROILogic` backend, same
public API (:class:`ROIDefineWidget`, :class:`ROIDefinePanel`,
``rois_modified``, ``set_config_path``, ``set_embedded_mode``).
"""
from __future__ import annotations

import contextlib
from pathlib import Path

from PySide6.QtCore import QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mufasa.roi_tools.roi_logic import CIRCLE, POLYGON, RECTANGLE, ROILogic
from mufasa.ui_qt.dialogs.roi_canvas import SELECT, ROICanvas, canonical_geometry

# Patch 122ag: project paths resolve through the v1-aware layout helper
# (_project_paths_lite returns a paths dict; _list_project_videos takes the
# video directory directly).
from mufasa.ui_qt.dialogs.roi_video_table import (
    _list_project_videos,
    _project_paths_lite,
    _videos_with_rois,
)

# Standard color palette (BGR for OpenCV consistency).
_COLORS: list[tuple[str, tuple[int, int, int]]] = [
    ("Red",     (0, 0, 255)),
    ("Green",   (0, 255, 0)),
    ("Blue",    (255, 0, 0)),
    ("Yellow",  (0, 255, 255)),
    ("Cyan",    (255, 255, 0)),
    ("Magenta", (255, 0, 255)),
    ("White",   (255, 255, 255)),
    ("Orange",  (0, 165, 255)),
    ("Pink",    (203, 192, 255)),
]
_KIND_LABEL = {RECTANGLE: "Rectangle", CIRCLE: "Circle", POLYGON: "Polygon"}
_UNDO_LIMIT = 100

# Shape-table columns.
_COL_NAME, _COL_TYPE, _COL_COLOR, _COL_WIDTH = range(4)


def _bgr_to_qcolor(bgr: tuple[int, int, int]) -> QColor:
    b, g, r = bgr
    return QColor(r, g, b)


def _swatch(bgr: tuple[int, int, int]) -> QIcon:
    pm = QPixmap(12, 12)
    pm.fill(_bgr_to_qcolor(bgr))
    return QIcon(pm)


def _color_combo(parent: QWidget) -> QComboBox:
    cb = QComboBox(parent)
    for name, bgr in _COLORS:
        cb.addItem(_swatch(bgr), name)
    return cb


def _color_bgr(name: str) -> tuple[int, int, int]:
    return next((b for n, b in _COLORS if n == name), (0, 0, 255))


def _section_label(text: str, parent: QWidget) -> QLabel:
    lbl = QLabel(text, parent)
    lbl.setStyleSheet("font-weight: bold; padding: 2px 0;")
    return lbl


class ROIDefineWidget(QWidget):
    """Unified Qt widget for ROI definition across an entire project.

    Construct with ``ROIDefineWidget(config_path, video_path=None)``. With no
    project, or a project without videos, it shows a placeholder instead of
    raising, so the workbench can lay it out at startup.
    """

    rois_modified = Signal()   # emitted when ROIs are written to disk

    def __init__(self, config_path: str | None = None,
                 video_path: str | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.config_path: str | None = config_path
        self.project_path: str = ""
        self.video_dir: str = ""
        self._videos: list[str] = []
        self._cur_idx: int = 0
        self.logic: ROILogic | None = None
        self._dirty = False
        self._undo: list = []
        self._redo: list = []
        self._overlay_idx_to_kind_name: list[tuple[str, str]] = []
        self._root_layout = QVBoxLayout(self)
        self._root_layout.setContentsMargins(0, 0, 0, 0)
        self._root_layout.setSpacing(0)
        self._content_widget: QWidget | None = None
        self._requested_video_path = video_path
        self._initialize_for_project()

    # ------------------------------------------------------------------ #
    # Project lifecycle
    # ------------------------------------------------------------------ #
    def _initialize_for_project(self) -> None:
        """Bring up the full UI for ``config_path`` or a placeholder."""
        if not self.config_path:
            self._show_placeholder(
                "No project loaded. Use <b>File → Open project…</b> "
                "to load a project with videos, then return here."
            )
            return
        try:
            paths = _project_paths_lite(self.config_path)
        except Exception as exc:
            self._show_placeholder(
                f"Could not read project config — "
                f"{type(exc).__name__}: {exc}"
            )
            return
        self.project_path = paths.get("project_root", "")
        self.video_dir = paths.get("video_dir", "")
        try:
            self._videos = _list_project_videos(self.video_dir)
        except Exception as exc:
            self._show_placeholder(
                f"Could not list project videos — "
                f"{type(exc).__name__}: {exc}"
            )
            return
        if not self._videos:
            self._show_placeholder(
                f"No videos found in <code>{self.video_dir or '&lt;no project&gt;'}</code>.<br>"
                "Import videos via <b>Data Import → Video Parameters</b> first."
            )
            return
        video_path = self._requested_video_path
        self._cur_idx = (self._videos.index(video_path)
                         if video_path in self._videos else 0)
        self.logic = None
        self._dirty = False
        self._clear_content()
        self._build_ui()
        self._refresh_video_list()
        self._load_video(self._cur_idx)

    def set_config_path(self, config_path: str | None) -> None:
        """Switch to a different project (or to no-project mode)."""
        self.config_path = config_path
        self._requested_video_path = None
        self._initialize_for_project()

    def set_embedded_mode(self, embedded: bool) -> None:
        """Embedded in a workbench page: hide "Close" / "Save && close"."""
        self._embedded = bool(embedded)
        if hasattr(self, "close_btn"):
            self._apply_embedded_visibility()

    def _apply_embedded_visibility(self) -> None:
        embedded = getattr(self, "_embedded", False)
        self.close_btn.setVisible(not embedded)
        self.save_close_btn.setVisible(not embedded)

    def _clear_content(self) -> None:
        if self._content_widget is not None:
            self._root_layout.removeWidget(self._content_widget)
            self._content_widget.deleteLater()
            self._content_widget = None

    def _show_placeholder(self, message_html: str) -> None:
        self._clear_content()
        self.project_path = ""
        self.video_dir = ""
        self._videos = []
        self._cur_idx = 0
        self.logic = None
        self._dirty = False
        host = QWidget(self)
        layout = QVBoxLayout(host)
        layout.setContentsMargins(40, 60, 40, 60)
        layout.addStretch()
        lbl = QLabel(message_html, host)
        lbl.setTextFormat(Qt.RichText)
        lbl.setWordWrap(True)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("color: palette(placeholder-text); "
                          "font-size: 11pt; padding: 12px;")
        layout.addWidget(lbl)
        layout.addStretch()
        self._root_layout.addWidget(host)
        self._content_widget = host

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        self._clear_content()
        content = QWidget(self)
        outer = QVBoxLayout(content)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        splitter = QSplitter(Qt.Horizontal, content)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_video_pane(content))
        splitter.addWidget(self._build_canvas_pane(content))
        splitter.addWidget(self._build_roi_pane(content))
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([220, 900, 320])
        outer.addWidget(splitter, 1)
        outer.addLayout(self._build_bottom_bar(content))

        self._root_layout.addWidget(content)
        self._content_widget = content
        if getattr(self, "_embedded", False):
            self._apply_embedded_visibility()
        self._build_shortcuts()
        self._set_tool(SELECT)

    def _build_video_pane(self, parent: QWidget) -> QWidget:
        pane = QWidget(parent)
        lay = QVBoxLayout(pane)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(_section_label("Videos", pane))
        self.video_filter = QLineEdit(pane)
        self.video_filter.setPlaceholderText("Filter…")
        self.video_filter.setClearButtonEnabled(True)
        self.video_filter.textChanged.connect(self._apply_video_filter)
        lay.addWidget(self.video_filter)
        self.video_list = QListWidget(pane)
        self.video_list.setAlternatingRowColors(True)
        self.video_list.currentRowChanged.connect(self._on_video_picked)
        lay.addWidget(self.video_list, 1)
        hint = QLabel("✓ = has ROIs · PgUp / PgDn to switch\n"
                      "Switching saves automatically.", pane)
        hint.setStyleSheet("color: palette(placeholder-text); font-size: 9pt;")
        lay.addWidget(hint)
        pane.setMinimumWidth(160)
        return pane

    def _build_canvas_pane(self, parent: QWidget) -> QWidget:
        pane = QWidget(parent)
        lay = QVBoxLayout(pane)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        # Tool bar.
        bar = QHBoxLayout()
        bar.setSpacing(4)
        self._tool_group = QButtonGroup(pane)
        self._tool_group.setExclusive(True)
        self._tool_buttons: dict[str, QToolButton] = {}
        for tool, text, key, tip in (
            (SELECT,    "Select",    "V", "Select, move and reshape ROIs"),
            (RECTANGLE, "Rectangle", "R", "Draw rectangles"),
            (CIRCLE,    "Circle",    "C", "Draw circles (centre, then radius)"),
            (POLYGON,   "Polygon",   "P", "Draw polygons — any number of vertices"),
        ):
            btn = QToolButton(pane)
            btn.setText(f"{text}  {key}")
            btn.setToolTip(f"{tip} ({key})")
            btn.setCheckable(True)
            btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
            btn.setStyleSheet(
                "QToolButton { padding: 4px 10px; }"
                "QToolButton:checked { background: palette(highlight); "
                "color: palette(highlighted-text); }")
            btn.clicked.connect(lambda _=False, t=tool: self._set_tool(t))
            self._tool_group.addButton(btn)
            self._tool_buttons[tool] = btn
            bar.addWidget(btn)
        bar.addStretch(1)
        self.fit_btn = QToolButton(pane)
        self.fit_btn.setText("Fit")
        self.fit_btn.setToolTip("Fit the frame to the view (F)")
        bar.addWidget(self.fit_btn)
        self.zoom_label = QLabel("100%", pane)
        self.zoom_label.setMinimumWidth(48)
        self.zoom_label.setStyleSheet("color: palette(placeholder-text);")
        bar.addWidget(self.zoom_label)
        self.labels_cb = QCheckBox("Labels", pane)
        self.labels_cb.setChecked(True)
        self.labels_cb.setToolTip("Show ROI names on the frame")
        bar.addWidget(self.labels_cb)
        lay.addLayout(bar)

        # Canvas.
        self.preview = ROICanvas(pane)
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview.shape_committed.connect(self._on_shape_committed)
        self.preview.shape_cancelled.connect(self._on_shape_cancelled)
        self.preview.shape_edited.connect(self._on_shape_edited)
        self.preview.shape_deleted.connect(self._on_shape_deleted)
        self.preview.shape_selected.connect(self._on_shape_selected)
        self.preview.hint_changed.connect(self._on_hint)
        self.preview.zoom_changed.connect(
            lambda z: self.zoom_label.setText(f"{z * 100:.0f}%"))
        self.fit_btn.clicked.connect(self.preview.fit_view)
        self.labels_cb.toggled.connect(self.preview.set_show_labels)
        lay.addWidget(self.preview, 1)

        self.hint_label = QLabel("", pane)
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet(
            "color: palette(placeholder-text); font-size: 9pt; padding: 0 2px;")
        lay.addWidget(self.hint_label)

        # Frame navigation.
        nav = QHBoxLayout()
        nav.setSpacing(2)

        def tb(text: str, tip: str, slot) -> QToolButton:
            b = QToolButton(pane)
            b.setText(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            nav.addWidget(b)
            return b

        self.first_btn = tb("⏮", "First frame", self._first_frame)
        self.back_s_btn = tb("−1s", "Back one second (Shift+,)",
                             lambda: self._step_seconds(-1.0))
        self.back_f_btn = tb("−1f", "Back one frame (,)",
                             lambda: self._step_frames(-1))
        self.frame_slider = QSlider(Qt.Horizontal, pane)
        self.frame_slider.valueChanged.connect(self._on_slider_changed)
        nav.addWidget(self.frame_slider, 1)
        self.fwd_f_btn = tb("+1f", "Forward one frame (.)",
                            lambda: self._step_frames(1))
        self.fwd_s_btn = tb("+1s", "Forward one second (Shift+.)",
                            lambda: self._step_seconds(1.0))
        self.last_btn = tb("⏭", "Last frame", self._last_frame)
        self.frame_label = QLabel("Frame 0", pane)
        self.frame_label.setMinimumWidth(150)
        self.frame_label.setStyleSheet("color: palette(placeholder-text);")
        nav.addWidget(self.frame_label)
        lay.addLayout(nav)
        return pane

    def _build_roi_pane(self, parent: QWidget) -> QWidget:
        pane = QWidget(parent)
        lay = QVBoxLayout(pane)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        lay.addWidget(_section_label("ROIs on this video", pane))
        self.shape_table = QTableWidget(pane)
        self.shape_table.setColumnCount(4)
        self.shape_table.setHorizontalHeaderLabels(
            ["Name", "Type", "Colour", "Width"])
        self.shape_table.verticalHeader().setVisible(False)
        self.shape_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.shape_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.shape_table.setEditTriggers(QAbstractItemView.DoubleClicked
                                         | QAbstractItemView.EditKeyPressed)
        h = self.shape_table.horizontalHeader()
        h.setSectionResizeMode(_COL_NAME, QHeaderView.Stretch)
        for c in (_COL_TYPE, _COL_COLOR, _COL_WIDTH):
            h.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.shape_table.itemChanged.connect(self._on_table_item_changed)
        self.shape_table.itemSelectionChanged.connect(
            self._on_table_selection_changed)
        lay.addWidget(self.shape_table, 1)

        row = QHBoxLayout()
        self.duplicate_btn = QPushButton("Duplicate", pane)
        self.duplicate_btn.setToolTip("Copy the selected ROI, offset by 20 px")
        self.duplicate_btn.clicked.connect(self._on_duplicate_clicked)
        row.addWidget(self.duplicate_btn)
        self.delete_btn = QPushButton("Delete", pane)
        self.delete_btn.setToolTip("Delete the selected ROI (Del)")
        self.delete_btn.clicked.connect(self._on_delete_selected)
        row.addWidget(self.delete_btn)
        lay.addLayout(row)

        new_box = QGroupBox("New shape", pane)
        form = QFormLayout(new_box)
        form.setContentsMargins(8, 6, 8, 6)
        self.name_edit = QLineEdit(new_box)
        self.name_edit.setPlaceholderText("auto (e.g. polygon_1)")
        self.name_edit.setToolTip(
            "Name for the next shape you draw. Leave empty to name it "
            "automatically; rename any time by double-clicking in the table.")
        form.addRow("Name", self.name_edit)
        self.color_cb = _color_combo(new_box)
        self.color_cb.setCurrentText("Red")
        form.addRow("Colour", self.color_cb)
        self.thickness_spin = QSpinBox(new_box)
        self.thickness_spin.setRange(1, 30)
        self.thickness_spin.setValue(3)
        self.thickness_spin.setSuffix(" px")
        form.addRow("Line width", self.thickness_spin)
        self.marker_spin = QSpinBox(new_box)
        self.marker_spin.setRange(2, 30)
        self.marker_spin.setValue(8)
        self.marker_spin.setSuffix(" px")
        self.marker_spin.setToolTip(
            "Size of the corner/centre tags drawn on ROI visualisations")
        form.addRow("Tag size", self.marker_spin)
        self.color_cb.currentTextChanged.connect(self._push_draw_style)
        self.thickness_spin.valueChanged.connect(self._push_draw_style)
        lay.addWidget(new_box)

        copy_box = QGroupBox("Copy to other videos", pane)
        cl = QVBoxLayout(copy_box)
        cl.setContentsMargins(8, 6, 8, 6)
        self.apply_all_btn = QPushButton("Apply to all…", copy_box)
        self.apply_all_btn.setToolTip(
            "Replace the ROIs of every other video with this video's ROIs")
        self.apply_all_btn.clicked.connect(self._on_apply_all_clicked)
        cl.addWidget(self.apply_all_btn)
        # Patch 122dl: subset-apply via DuplicateRoisDialog.
        self.apply_selected_btn = QPushButton("Apply to selected…", copy_box)
        self.apply_selected_btn.setToolTip(
            "Copy this video's ROIs to videos you pick by name / filter")
        self.apply_selected_btn.clicked.connect(self._on_apply_selected_clicked)
        cl.addWidget(self.apply_selected_btn)
        lay.addWidget(copy_box)

        # Patch 122fb: bulk import / standardize lives in a popup.
        self.maintenance_btn = QPushButton("Import / standardize…", pane)
        self.maintenance_btn.setToolTip(
            "Import ROI definitions from CSV, or standardize all ROIs to a "
            "reference video's pixels-per-mm calibration")
        self.maintenance_btn.clicked.connect(self._on_maintenance_clicked)
        lay.addWidget(self.maintenance_btn)

        self.reset_btn = QPushButton("Clear this video…", pane)
        self.reset_btn.setToolTip("Delete every ROI on the current video")
        self.reset_btn.clicked.connect(self._on_reset_clicked)
        lay.addWidget(self.reset_btn)
        pane.setMinimumWidth(260)
        return pane

    def _build_bottom_bar(self, parent: QWidget) -> QHBoxLayout:
        bot = QHBoxLayout()
        self.save_status = QLabel("", parent)
        self.save_status.setStyleSheet("color: palette(placeholder-text);")
        bot.addWidget(self.save_status, 1)
        self.dirty_label = QLabel("● Unsaved changes", parent)
        self.dirty_label.setStyleSheet("color: #d08a00; font-weight: bold;")
        self.dirty_label.setVisible(False)
        bot.addWidget(self.dirty_label)
        bot.addSpacing(12)
        self.undo_btn = QPushButton("Undo", parent)
        self.undo_btn.setToolTip("Undo (Ctrl+Z)")
        self.undo_btn.clicked.connect(self._undo_action)
        bot.addWidget(self.undo_btn)
        self.redo_btn = QPushButton("Redo", parent)
        self.redo_btn.setToolTip("Redo (Ctrl+Shift+Z)")
        self.redo_btn.clicked.connect(self._redo_action)
        bot.addWidget(self.redo_btn)
        sep = QFrame(parent)
        sep.setFrameShape(QFrame.VLine)
        bot.addWidget(sep)
        self.close_btn = QPushButton("Close", parent)
        self.close_btn.clicked.connect(self._on_close_clicked)
        bot.addWidget(self.close_btn)
        self.save_btn = QPushButton("Save", parent)
        self.save_btn.setToolTip("Save (Ctrl+S)")
        self.save_btn.clicked.connect(self._on_save_clicked)
        bot.addWidget(self.save_btn)
        self.save_close_btn = QPushButton("Save && close", parent)
        self.save_close_btn.setDefault(True)
        self.save_close_btn.clicked.connect(self._on_save_and_close)
        bot.addWidget(self.save_close_btn)
        return bot

    def _build_shortcuts(self) -> None:
        def sc(seq: str, slot, widget: QWidget | None = None,
               ctx=Qt.WidgetWithChildrenShortcut) -> QShortcut:
            s = QShortcut(QKeySequence(seq), widget or self, activated=slot)
            s.setContext(ctx)
            return s

        for tool, key in ((SELECT, "V"), (RECTANGLE, "R"),
                          (CIRCLE, "C"), (POLYGON, "P")):
            sc(key, lambda t=tool: self._set_tool(t))
        sc("PgUp", self._prev_video)
        sc("PgDown", self._next_video)
        sc(",", lambda: self._step_frames(-1))
        sc(".", lambda: self._step_frames(1))
        sc("Shift+,", lambda: self._step_seconds(-1.0))
        sc("Shift+.", lambda: self._step_seconds(1.0))
        sc("Ctrl+Z", self._undo_action)
        sc("Ctrl+Shift+Z", self._redo_action)
        sc("Ctrl+Y", self._redo_action)
        sc("Ctrl+S", self._on_save_clicked)
        sc("Del", self._on_delete_selected, self.shape_table, Qt.WidgetShortcut)

    # ------------------------------------------------------------------ #
    # Tools
    # ------------------------------------------------------------------ #
    def _selected_color(self) -> tuple[str, tuple[int, int, int]]:
        name = self.color_cb.currentText()
        return name, _color_bgr(name)

    def _set_tool(self, tool: str) -> None:
        btn = self._tool_buttons.get(tool)
        if btn is not None and not btn.isChecked():
            btn.setChecked(True)
        _, bgr = self._selected_color()
        self.preview.set_tool(tool, bgr, self.thickness_spin.value())
        if tool == SELECT:
            row = self._selected_row()
            self.preview.select_shape(row)

    def _push_draw_style(self, *_args) -> None:
        _, bgr = self._selected_color()
        self.preview.set_draw_style(bgr, self.thickness_spin.value())

    def _on_hint(self, text: str) -> None:
        self.hint_label.setText(text)

    # ------------------------------------------------------------------ #
    # Video list
    # ------------------------------------------------------------------ #
    def _refresh_video_list(self) -> None:
        """Rebuild the video list with each video's ROI status."""
        # Patch 122ag: the layout helper gives the right H5 path for v1
        # and legacy projects alike.
        roi_h5 = _project_paths_lite(self.config_path).get(
            "roi_definitions_path", "")
        videos_with_rois = _videos_with_rois(roi_h5)
        with QSignalBlocker(self.video_list):
            self.video_list.clear()
            for vpath in self._videos:
                stem = Path(vpath).stem
                has_rois = stem in videos_with_rois
                item = QListWidgetItem(f"{'✓' if has_rois else '  '}  {stem}")
                item.setToolTip(vpath)
                if has_rois:
                    f = item.font()
                    f.setBold(True)
                    item.setFont(f)
                self.video_list.addItem(item)
            self.video_list.setCurrentRow(self._cur_idx)
        self._apply_video_filter(self.video_filter.text())

    def _apply_video_filter(self, text: str) -> None:
        t = text.strip().lower()
        for i in range(self.video_list.count()):
            item = self.video_list.item(i)
            stem = Path(self._videos[i]).stem.lower()
            item.setHidden(bool(t) and t not in stem and i != self._cur_idx)

    def _on_video_picked(self, row: int) -> None:
        if row < 0 or row >= len(self._videos) or row == self._cur_idx:
            return
        if self._dirty and self.logic is not None:
            try:
                self.logic.save()
                self._set_dirty(False)
                self._flash_status("Auto-saved.")
            except Exception as exc:
                QMessageBox.critical(
                    self, "Auto-save failed",
                    f"Could not save current video before switching: "
                    f"{type(exc).__name__}: {exc}\n\n"
                    f"Staying on the current video.",
                )
                with QSignalBlocker(self.video_list):
                    self.video_list.setCurrentRow(self._cur_idx)
                return
            self._refresh_video_list_marks()
        self._cur_idx = row
        self._load_video(row)

    def _refresh_video_list_marks(self) -> None:
        """Update the ✓ marks without losing the user's position."""
        cur = self._cur_idx
        self._refresh_video_list()
        with QSignalBlocker(self.video_list):
            self.video_list.setCurrentRow(cur)

    def _load_video(self, idx: int, keep_frame: bool = False) -> None:
        """Open video[idx]: fresh ROILogic, preview and table."""
        vpath = self._videos[idx]
        frame = self.logic.frame_idx if (keep_frame and self.logic) else 0
        try:
            self.logic = ROILogic(config_path=self.config_path,
                                  video_path=vpath)
        except Exception as exc:
            QMessageBox.critical(
                self, "Load failed",
                f"Could not load {Path(vpath).name}: "
                f"{type(exc).__name__}: {exc}",
            )
            return
        if frame:
            self.logic.goto_frame(frame)
        if getattr(self.logic, "load_error", None):
            # Patch 122ik: an unreadable ROI file used to look like "no ROIs
            # yet" — and the next save would have replaced it.
            QMessageBox.warning(
                self, "ROI file unreadable",
                f"Could not read the project's ROI file:\n\n"
                f"{self.logic.load_error}\n\n"
                f"Existing ROIs are not shown, and saving is blocked so the "
                f"file isn't overwritten. Back it up and check it.")
        self._set_dirty(False)
        self._undo.clear()
        self._redo.clear()
        self._update_undo_buttons()
        self.setWindowTitle(f"ROI Definitions — {self.logic.video_name}")
        with QSignalBlocker(self.frame_slider):
            self.frame_slider.setRange(0, max(0, self.logic.frame_count - 1))
            self.frame_slider.setValue(self.logic.frame_idx)
        # Abandon a shape in progress from the previous video.
        self._set_tool(self.preview.tool() or SELECT)
        self._sync_preview()
        self._sync_table()

    def _prev_video(self) -> None:
        if self._cur_idx > 0:
            self.video_list.setCurrentRow(self._cur_idx - 1)

    def _next_video(self) -> None:
        if self._cur_idx < len(self._videos) - 1:
            self.video_list.setCurrentRow(self._cur_idx + 1)

    # ------------------------------------------------------------------ #
    # Preview / table sync
    # ------------------------------------------------------------------ #
    def _roi_entries(self) -> list[tuple[str, str, object]]:
        """(kind, name, ROIDefinition) in display order. The canvas overlay
        index and the table row index are both positions in this list."""
        if self.logic is None:
            return []
        return [(kind, name, roi)
                for kind, d in self.logic.defs.items()
                for name, roi in d.items()]

    def _sync_preview(self) -> None:
        if self.logic is None:
            return
        if self.logic.current_frame is not None:
            self.preview.set_frame(self.logic.current_frame)
        overlay_rois = []
        self._overlay_idx_to_kind_name = []
        for kind, name, roi in self._roi_entries():
            overlay_rois.append({
                "kind": kind, "name": name,
                "color_bgr": roi.color_bgr,
                "thickness": roi.thickness,
                "geometry": roi.geometry,
            })
            self._overlay_idx_to_kind_name.append((kind, name))
        self.preview.set_existing_rois(overlay_rois)
        self._update_frame_label()

    def _update_frame_label(self) -> None:
        lg = self.logic
        if lg is None:
            return
        secs = lg.frame_idx / lg.fps if lg.fps else 0.0
        self.frame_label.setText(
            f"Frame {lg.frame_idx} / {lg.frame_count - 1}  ({secs:.1f} s)")

    def _sync_preview_and_slider(self) -> None:
        if self.logic is None:
            return
        with QSignalBlocker(self.frame_slider):
            self.frame_slider.setValue(self.logic.frame_idx)
        self._sync_preview()

    def _sync_table(self, select_name: str | None = None) -> None:
        if select_name is None:
            row = self._selected_row()
            if row is not None and row < len(self._overlay_idx_to_kind_name):
                select_name = self._overlay_idx_to_kind_name[row][1]
        entries = self._roi_entries()
        with QSignalBlocker(self.shape_table):
            self.shape_table.clearContents()
            self.shape_table.setRowCount(len(entries))
            for i, (kind, name, roi) in enumerate(entries):
                name_item = QTableWidgetItem(name)
                name_item.setData(Qt.UserRole, name)
                name_item.setToolTip("Double-click to rename")
                self.shape_table.setItem(i, _COL_NAME, name_item)
                type_item = QTableWidgetItem(self._describe(kind, roi))
                type_item.setFlags(type_item.flags() & ~Qt.ItemIsEditable)
                self.shape_table.setItem(i, _COL_TYPE, type_item)
                cb = _color_combo(self.shape_table)
                if roi.color_name not in [c for c, _ in _COLORS]:
                    cb.addItem(_swatch(roi.color_bgr), roi.color_name)
                cb.setCurrentText(roi.color_name)
                cb.currentTextChanged.connect(
                    lambda text, n=name: self._on_row_color(n, text))
                self.shape_table.setCellWidget(i, _COL_COLOR, cb)
                sp = QSpinBox(self.shape_table)
                sp.setRange(1, 30)
                sp.setValue(int(roi.thickness))
                sp.valueChanged.connect(
                    lambda v, n=name: self._on_row_width(n, v))
                self.shape_table.setCellWidget(i, _COL_WIDTH, sp)
            row = next((i for i, (_, n, _r) in enumerate(entries)
                        if n == select_name), None)
            if row is not None:
                self.shape_table.selectRow(row)
            else:
                self.shape_table.clearSelection()
        self._update_row_buttons()

    @staticmethod
    def _describe(kind: str, roi) -> str:
        if kind == POLYGON:
            n = len(roi.geometry.get("vertices") or [])
            return f"Polygon ({n})"
        return _KIND_LABEL.get(kind, kind)

    def _selected_row(self) -> int | None:
        rows = self.shape_table.selectionModel().selectedRows() \
            if self.shape_table.selectionModel() else []
        return rows[0].row() if rows else None

    def _selected_name(self) -> str | None:
        row = self._selected_row()
        if row is None or row >= len(self._overlay_idx_to_kind_name):
            return None
        return self._overlay_idx_to_kind_name[row][1]

    def _update_row_buttons(self) -> None:
        has = self._selected_row() is not None
        self.duplicate_btn.setEnabled(has)
        self.delete_btn.setEnabled(has)

    # ------------------------------------------------------------------ #
    # Undo / dirty state
    # ------------------------------------------------------------------ #
    def _push_undo(self) -> None:
        if self.logic is None:
            return
        self._undo.append(self.logic.snapshot())
        del self._undo[:-_UNDO_LIMIT]
        self._redo.clear()
        self._last_width_edit = None
        self._update_undo_buttons()

    def _changed(self, message: str, select_name: str | None = None) -> None:
        """Common tail of every edit: mark dirty, resync, report."""
        self._set_dirty(True)
        self._sync_preview()
        self._sync_table(select_name)
        if self.preview.is_selecting():
            self.preview.select_shape(self._selected_row())
        self._flash_status(message)

    def _undo_action(self) -> None:
        if self.logic is None or not self._undo:
            return
        if self.preview.is_drawing():
            self.preview.cancel_draw()
            return
        self._redo.append(self.logic.snapshot())
        self.logic.restore(self._undo.pop())
        self._update_undo_buttons()
        self._changed("Undone.")

    def _redo_action(self) -> None:
        if self.logic is None or not self._redo:
            return
        self._undo.append(self.logic.snapshot())
        self.logic.restore(self._redo.pop())
        self._update_undo_buttons()
        self._changed("Redone.")

    def _update_undo_buttons(self) -> None:
        if hasattr(self, "undo_btn"):
            self.undo_btn.setEnabled(bool(self._undo))
            self.redo_btn.setEnabled(bool(self._redo))

    def _set_dirty(self, dirty: bool) -> None:
        self._dirty = bool(dirty)
        if hasattr(self, "dirty_label"):
            self.dirty_label.setVisible(self._dirty)

    # ------------------------------------------------------------------ #
    # Frame navigation
    # ------------------------------------------------------------------ #
    def _step_frames(self, n: int) -> None:
        if self.logic is None:
            return
        self.logic.advance_frame(n)
        self._sync_preview_and_slider()

    def _step_seconds(self, s: float) -> None:
        if self.logic is None:
            return
        self.logic.jump_seconds(s)
        self._sync_preview_and_slider()

    def _first_frame(self) -> None:
        if self.logic is None:
            return
        self.logic.first_frame()
        self._sync_preview_and_slider()

    def _last_frame(self) -> None:
        if self.logic is None:
            return
        self.logic.last_frame()
        self._sync_preview_and_slider()

    def _on_slider_changed(self, val: int) -> None:
        if self.logic is None:
            return
        self.logic.goto_frame(val)
        self._sync_preview()

    # ------------------------------------------------------------------ #
    # Drawing
    # ------------------------------------------------------------------ #
    def _next_auto_name(self, kind: str) -> str:
        i = 1
        while self.logic.has_roi(f"{kind}_{i}"):
            i += 1
        return f"{kind}_{i}"

    def _on_shape_committed(self, kind: str, geom: dict) -> None:
        """A new shape was drawn: name it, style it, add it."""
        if self.logic is None:
            return
        typed = self.name_edit.text().strip()
        if typed and self.logic.has_roi(typed):
            self._flash_status(
                f"An ROI named {typed!r} already exists — the new shape was "
                f"named automatically instead.", error=True)
            typed = ""
        name = typed or self._next_auto_name(kind)
        clr_name, bgr = self._selected_color()
        style = dict(color_name=clr_name, bgr=bgr,
                     thickness=self.thickness_spin.value(),
                     ear_tag_size=self.marker_spin.value())
        self._push_undo()
        try:
            if kind == RECTANGLE:
                self.logic.add_rectangle(
                    name=name, top_left=geom["top_left"],
                    bottom_right=geom["bottom_right"], **style)
            elif kind == CIRCLE:
                self.logic.add_circle(
                    name=name, center=geom["center"],
                    radius=geom["radius"], **style)
            elif kind == POLYGON:
                self.logic.add_polygon(
                    name=name, vertices=geom["vertices"], **style)
        except Exception as exc:
            self._undo.pop()
            self._update_undo_buttons()
            self._flash_status(f"Add failed: {exc}", error=True)
            return
        self.name_edit.clear()
        extra = (f" ({len(geom['vertices'])} vertices)"
                 if kind == POLYGON else "")
        self._changed(f"Added {name}{extra} — double-click its name to "
                      f"rename.", select_name=name)

    def _on_shape_cancelled(self) -> None:
        self._flash_status("Shape abandoned.")

    # ------------------------------------------------------------------ #
    # Editing placed shapes
    # ------------------------------------------------------------------ #
    def _on_shape_edited(self, idx: int, geom: dict) -> None:
        """Canvas moved / reshaped overlay ``idx``: write it to the logic."""
        if self.logic is None:
            return
        if not (0 <= idx < len(self._overlay_idx_to_kind_name)):
            return
        kind, name = self._overlay_idx_to_kind_name[idx]
        geom = canonical_geometry(kind, geom)
        self._push_undo()
        ok = False
        try:
            if kind == RECTANGLE:
                ok = self.logic.update_rectangle_geometry(
                    name=name, top_left=tuple(geom["top_left"]),
                    bottom_right=tuple(geom["bottom_right"]))
            elif kind == CIRCLE:
                ok = self.logic.update_circle_geometry(
                    name=name, center=tuple(geom["center"]),
                    radius=int(geom["radius"]))
            elif kind == POLYGON:
                ok = self.logic.update_polygon_geometry(
                    name=name, vertices=[tuple(v) for v in geom["vertices"]])
        except Exception as exc:
            ok = False
            self._flash_status(f"Edit failed: {exc}", error=True)
        if not ok:
            self._undo.pop()
            self._update_undo_buttons()
            self._sync_preview()
            return
        self._changed(f"Adjusted {name}.", select_name=name)

    def _on_shape_deleted(self, idx: int) -> None:
        """Delete pressed on the canvas selection."""
        if self.logic is None:
            return
        if not (0 <= idx < len(self._overlay_idx_to_kind_name)):
            return
        _kind, name = self._overlay_idx_to_kind_name[idx]
        self._delete_names([name])

    def _on_shape_selected(self, idx: int) -> None:
        """Canvas click selected overlay ``idx`` (-1 = none): mirror it in
        the table."""
        with QSignalBlocker(self.shape_table):
            if idx < 0:
                self.shape_table.clearSelection()
            elif 0 <= idx < self.shape_table.rowCount():
                self.shape_table.selectRow(idx)
                item = self.shape_table.item(idx, 0)
                if item is not None:
                    self.shape_table.scrollToItem(item)
        self._update_row_buttons()

    def _on_table_selection_changed(self) -> None:
        """Table row picked: select that shape on the canvas."""
        row = self._selected_row()
        if row is not None and not self.preview.is_selecting():
            self._set_tool(SELECT)
        self.preview.select_shape(row)
        self._update_row_buttons()

    def _on_table_item_changed(self, item: QTableWidgetItem) -> None:
        """Name cell edited: rename the ROI (or revert on a bad name)."""
        if self.logic is None or item.column() != _COL_NAME:
            return
        old = item.data(Qt.UserRole)
        new = item.text().strip()
        if not old or new == old:
            return
        if not new or self.logic.has_roi(new):
            self._flash_status(
                "Name can't be empty." if not new
                else f"An ROI named {new!r} already exists.", error=True)
            with QSignalBlocker(self.shape_table):
                item.setText(old)
            return
        self._push_undo()
        self.logic.rename_roi(old, new)
        # Rebuild after this signal returns: the table must not delete the
        # item that is emitting itemChanged.
        QTimer.singleShot(0, lambda: self._changed(
            f"Renamed {old} → {new}.", select_name=new))

    def _on_row_color(self, name: str, color_name: str) -> None:
        if self.logic is None or not self.logic.has_roi(name):
            return
        self._push_undo()
        self.logic.update_style(name, color_name=color_name,
                                bgr=_color_bgr(color_name))
        # The combo box stays (no table rebuild while it is emitting).
        self._set_dirty(True)
        self._sync_preview()
        self._flash_status(f"{name}: colour {color_name}.")

    def _on_row_width(self, name: str, width: int) -> None:
        if self.logic is None or not self.logic.has_roi(name):
            return
        # Coalesce spin-box scrubbing into a single undo step.
        last = getattr(self, "_last_width_edit", None)
        if last != name or not self._undo:
            self._push_undo()
        self._last_width_edit = name
        self.logic.update_style(name, thickness=width)
        self._set_dirty(True)
        self._sync_preview()
        self._flash_status(f"{name}: line width {width} px.")

    def _on_duplicate_clicked(self) -> None:
        name = self._selected_name()
        if self.logic is None or name is None:
            return
        i, base = 2, f"{name}_copy"
        new = base
        while self.logic.has_roi(new):
            new = f"{base}{i}"
            i += 1
        self._push_undo()
        self.logic.duplicate_roi(name, new)
        self._changed(f"Duplicated {name} → {new}.", select_name=new)

    def _on_delete_selected(self) -> None:
        name = self._selected_name()
        if name is not None:
            self._delete_names([name])

    def _on_delete_roi(self, name: str) -> None:
        """Delete one ROI by name (undoable, so no confirmation)."""
        self._delete_names([name])

    def _delete_names(self, names: list[str]) -> None:
        if self.logic is None:
            return
        names = [n for n in names if self.logic.has_roi(n)]
        if not names:
            return
        self._push_undo()
        for n in names:
            self.logic.delete_roi(n)
        self.preview.select_shape(None)
        self._changed(f"Deleted {', '.join(names)} (Ctrl+Z to undo).")

    # ------------------------------------------------------------------ #
    # Project-wide actions
    # ------------------------------------------------------------------ #
    def _on_reset_clicked(self) -> None:
        if self.logic is None or not self.logic.all_roi_names:
            return
        if QMessageBox.question(
            self, "Clear this video",
            f"Delete all {len(self.logic.all_roi_names)} ROIs on "
            f"<b>{self.logic.video_name}</b>? (Undo is available until you "
            f"switch video.)",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        ) != QMessageBox.Yes:
            return
        self._push_undo()
        self.logic.delete_all()
        self._changed("All ROIs on this video cleared (Ctrl+Z to undo).")

    def _on_apply_all_clicked(self) -> None:
        if self.logic is None:
            return
        if not self.logic.all_roi_names:
            self._flash_status("No ROIs to apply.", error=True)
            return
        others = [Path(v).stem for v in self._videos
                  if Path(v).stem != self.logic.video_name]
        roi_h5 = _project_paths_lite(self.config_path).get(
            "roi_definitions_path", "")
        overwritten = sorted(set(others) & set(_videos_with_rois(roi_h5)))
        msg = (f"Copy the {len(self.logic.all_roi_names)} ROIs of "
               f"<b>{self.logic.video_name}</b> to all {len(others)} other "
               f"videos?")
        if overwritten:
            msg += (f"<br><br><b>This replaces the existing ROIs of "
                    f"{len(overwritten)} video(s)</b>, e.g. "
                    f"{', '.join(overwritten[:3])}"
                    f"{'…' if len(overwritten) > 3 else ''}. "
                    f"Use <i>Apply to selected…</i> to pick videos instead.")
        if QMessageBox.question(
            self, "Apply to all videos", msg,
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        ) != QMessageBox.Yes:
            return
        try:
            self.logic.save()
            self._set_dirty(False)
            self._flash_status("Saved — applying to all videos…")
            QApplication.processEvents()
            from mufasa.roi_tools.roi_utils import multiply_ROIs
            multiply_ROIs(config_path=self.config_path,
                          filename=self._videos[self._cur_idx])
            self._refresh_video_list_marks()
            self.rois_modified.emit()
            self._flash_status(f"Applied ROIs to {len(others)} videos.")
        except Exception as exc:
            QMessageBox.critical(
                self, "Apply-all failed",
                f"Could not apply ROIs: {type(exc).__name__}: {exc}",
            )

    def _on_apply_selected_clicked(self) -> None:
        """Patch 122dl: copy this video's ROIs to a chosen subset."""
        if self.logic is None:
            return
        if not self.logic.all_roi_names:
            self._flash_status("No ROIs to apply.", error=True)
            return
        # Save first so the dialog reads the freshest H5 state.
        try:
            self.logic.save()
            self._set_dirty(False)
            QApplication.processEvents()
        except Exception as exc:
            QMessageBox.critical(
                self, "Apply-to-selected failed",
                f"Could not save current ROIs before opening "
                f"the subset-apply dialog: "
                f"{type(exc).__name__}: {exc}",
            )
            return
        from mufasa.ui_qt.dialogs.duplicate_rois_source_target import DuplicateRoisDialog
        dlg = DuplicateRoisDialog(
            config_path=self.config_path,
            parent=self,
            default_source=Path(self._videos[self._cur_idx]).stem,
            window_title="Apply ROIs to selected videos",
        )
        if dlg.init_failed():
            return
        if dlg.exec() == QDialog.Accepted:
            self._refresh_video_list_marks()
            self.rois_modified.emit()
            self._flash_status("Applied ROIs to selected videos.")

    def _on_maintenance_clicked(self) -> None:
        """Patch 122fb: ROIManageForm (CSV import / standardize) in a
        popup. Afterwards the current video is reloaded from disk so
        imported ROIs appear."""
        from mufasa.ui_qt.forms.roi import ROIManageForm

        cfg = (getattr(self.logic, "config_path", None)
               if self.logic is not None else None)
        if not cfg:
            self._flash_status(
                "No project loaded — open a project before "
                "importing or standardizing ROIs.", error=True)
            return
        if self._dirty:
            try:
                self.logic.save()
                self._set_dirty(False)
            except Exception as exc:
                QMessageBox.critical(
                    self, "Save failed",
                    f"Could not save current ROIs first: "
                    f"{type(exc).__name__}: {exc}")
                return
        dlg = QDialog(self)
        dlg.setWindowTitle("Import ROIs / Standardize sizes")
        dlg.setMinimumSize(640, 420)
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(8, 8, 8, 8)
        form = ROIManageForm(parent=dlg, config_path=cfg)
        with contextlib.suppress(Exception):
            form.completed.connect(dlg.accept)
        layout.addWidget(form)
        dlg.exec()
        try:
            self._load_video(self._cur_idx, keep_frame=True)
            self._refresh_video_list_marks()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Save / close
    # ------------------------------------------------------------------ #
    def _on_save_clicked(self) -> None:
        if self.logic is None:
            return
        try:
            self.logic.save()
            self._set_dirty(False)
            self._flash_status("Saved.")
            self._refresh_video_list_marks()
            self.rois_modified.emit()
        except Exception as exc:
            QMessageBox.critical(
                self, "Save failed",
                f"Could not save ROIs: {type(exc).__name__}: {exc}",
            )

    def _on_save_and_close(self) -> None:
        self._on_save_clicked()
        if not self._dirty:
            self._close_window()

    def _on_close_clicked(self) -> None:
        if self._dirty:
            ans = QMessageBox.question(
                self, "Unsaved changes",
                "You have unsaved ROI changes for the current video. "
                "Save before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save,
            )
            if ans == QMessageBox.Save:
                self._on_save_clicked()
                if self._dirty:
                    return
            elif ans == QMessageBox.Cancel:
                return
            else:
                self._set_dirty(False)   # discarded: don't ask again
        self._close_window()

    def _close_window(self) -> None:
        """Close the enclosing dialog (popup mode) or this widget."""
        win = self.window()
        (win if isinstance(win, QDialog) else self).close()

    def _flash_status(self, msg: str, error: bool = False) -> None:
        self.save_status.setStyleSheet(
            "color: #c44;" if error else "color: palette(placeholder-text);")
        self.save_status.setText(msg)


# --------------------------------------------------------------------------- #
# Patch 122dn — backward-compatible QDialog wrapper.
# --------------------------------------------------------------------------- #
class ROIDefinePanel(QDialog):
    """Thin QDialog wrapper around :class:`ROIDefineWidget` for the popup
    entry points (``roi_video_table`` and ``ROIManageForm``).

    Raises ``RuntimeError`` when the project has no videos, as the
    pre-122dn dialog did.
    """

    rois_modified = Signal()

    def __init__(self, config_path: str,
                 video_path: str | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("ROI Definitions — Mufasa")
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.resize(1440, 860)
        self.config_path = config_path
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.widget = ROIDefineWidget(
            config_path=config_path, video_path=video_path, parent=self)
        outer.addWidget(self.widget)
        self.widget.rois_modified.connect(self.rois_modified)
        if not self.widget._videos:
            raise RuntimeError(
                f"No videos found in "
                f"{self.widget.video_dir or '<no project>'}"
            )

    def closeEvent(self, ev) -> None:
        """Window-manager close: same unsaved-changes prompt as Close."""
        w = self.widget
        if getattr(w, "_dirty", False):
            ans = QMessageBox.question(
                self, "Unsaved changes",
                "You have unsaved ROI changes for the current video. "
                "Save before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save,
            )
            if ans == QMessageBox.Cancel:
                ev.ignore()
                return
            if ans == QMessageBox.Save:
                w._on_save_clicked()
                if w._dirty:
                    ev.ignore()
                    return
            w._set_dirty(False)
        super().closeEvent(ev)


__all__ = ["ROIDefinePanel", "ROIDefineWidget"]
