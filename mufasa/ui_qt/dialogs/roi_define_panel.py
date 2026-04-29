"""
mufasa.ui_qt.dialogs.roi_define_panel
=====================================

Unified Qt panel for defining ROIs across all videos in a project.

Layout
------

::

    ┌─────────────────────────────────────────────────────────────┐
    │ Tools: [▭][◯][△]  Color:[Red ▾]  Thickness:3  Marker:8       │
    │        Name:[___________]  [Draw →]                          │
    ├──────────────────┬──────────────────────────────────────────┤
    │ Project videos   │ [⏮][−1s][−1f] ━━●━━ [+1f][+1s][⏭] 47/60  │
    │                  ├──────────────────────────────────────────┤
    │ vid_01           │                                          │
    │ vid_02 ✓         │   [video frame with ROI overlays]        │
    │ vid_03 ✓ ←       │                                          │
    │ vid_04 ✓         │                                          │
    │ vid_05           │                                          │
    │ ...              │                                          │
    │                  ├──────────────────────────────────────────┤
    │                  │ # Type     Name        Color    ✕        │
    │                  │ 1 Rectangle center_zone Red     ✕        │
    │                  │ 2 Circle   left_obj    Blue     ✕        │
    ├──────────────────┴──────────────────────────────────────────┤
    │ status flash    [Reset] [Apply to all] [Save] [Save&close]   │
    └─────────────────────────────────────────────────────────────┘

Differences from the previous version
-------------------------------------

* **Single window**. The video table dialog is gone; the panel
  contains a project-wide video list as a left sidebar.
* **Page Up / Page Down** step between videos.
* **Auto-save on video switch**. Switching videos with unsaved
  changes saves them silently before loading the next video.
* **Synchronous OpenCV drawing**. The selector runs on the main
  thread (cv2.imshow / cv2.waitKey are not thread-safe on Linux/X11
  — running them in a QThread caused a black canvas in earlier
  versions). The Qt panel is briefly frozen during drawing, but the
  user is interacting with the OpenCV window during that time
  anyway.

Compatibility
-------------

* Same H5 file format as the previous version (``project_folder/
  logs/measures/ROI_definitions.h5``).
* Same shape constants and ROILogic backend.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (QApplication, QButtonGroup, QComboBox,
                               QDialog, QHBoxLayout, QHeaderView,
                               QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMessageBox, QPushButton,
                               QSizePolicy, QSlider, QSpinBox, QSplitter,
                               QTableWidget, QTableWidgetItem, QToolButton,
                               QVBoxLayout, QWidget)

from mufasa.roi_tools.roi_logic import (CIRCLE, POLYGON, RECTANGLE,
                                        ROILogic)
from mufasa.ui_qt.dialogs.roi_canvas import ROICanvas
from mufasa.ui_qt.dialogs.roi_video_table import (_list_project_videos,
                                                  _project_path_from_config,
                                                  _videos_with_rois)


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


def _bgr_to_qcolor(bgr: tuple[int, int, int]) -> QColor:
    b, g, r = bgr
    return QColor(r, g, b)


class ROIDefinePanel(QDialog):
    """Unified Qt panel for ROI definition across an entire project.

    Open with ``ROIDefinePanel(config_path, video_path=None)`` to start
    on the first video, or pass a specific ``video_path`` to start
    there.

    Page Up / Page Down step between videos. Switching videos with
    unsaved changes auto-saves them first.
    """

    rois_modified = Signal()   # emitted when any save / reset happens

    def __init__(self, config_path: str,
                 video_path: Optional[str] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.config_path = config_path
        self.project_path = _project_path_from_config(config_path) or ""

        # Discover videos
        self._videos: list[str] = _list_project_videos(self.project_path)
        if not self._videos:
            raise RuntimeError(
                f"No videos found in {self.project_path}/videos/"
            )
        # Pick starting video
        if video_path is not None and video_path in self._videos:
            self._cur_idx = self._videos.index(video_path)
        else:
            self._cur_idx = 0

        # Logic for the currently-selected video. Created lazily —
        # _load_video() initializes it.
        self.logic: Optional[ROILogic] = None
        self._dirty = False

        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.resize(1300, 850)

        self._build_ui()
        self._refresh_video_list()
        self._load_video(self._cur_idx)

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        # ---- Tool palette + options (top row, full width) ---- #
        tool_row = QHBoxLayout()
        tool_row.setSpacing(2)
        self._tool_buttons: dict[str, QToolButton] = {}
        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(True)
        for kind, glyph, tip in (
            (RECTANGLE, "▭", "Rectangle (R)"),
            (CIRCLE,    "◯", "Circle (C)"),
            (POLYGON,   "△", "Polygon (P)"),
        ):
            btn = QToolButton(self)
            btn.setText(glyph)
            btn.setToolTip(tip)
            btn.setCheckable(True)
            btn.setFixedSize(32, 32)
            btn.setStyleSheet(
                "QToolButton { font-size: 18pt; padding: 0; }"
                "QToolButton:checked { background: palette(highlight); "
                "color: palette(highlighted-text); }"
            )
            self._tool_buttons[kind] = btn
            self._tool_group.addButton(btn)
            tool_row.addWidget(btn)
        self._tool_buttons[RECTANGLE].setChecked(True)
        tool_row.addSpacing(20)

        tool_row.addWidget(QLabel("Color:"))
        self.color_cb = QComboBox(self)
        for name, _ in _COLORS:
            self.color_cb.addItem(name)
        self.color_cb.setCurrentText("Red")
        tool_row.addWidget(self.color_cb)

        tool_row.addWidget(QLabel("Thickness:"))
        self.thickness_spin = QSpinBox(self)
        self.thickness_spin.setRange(1, 30)
        self.thickness_spin.setValue(3)
        self.thickness_spin.setFixedWidth(60)
        tool_row.addWidget(self.thickness_spin)

        tool_row.addWidget(QLabel("Marker:"))
        self.marker_spin = QSpinBox(self)
        self.marker_spin.setRange(2, 30)
        self.marker_spin.setValue(8)
        self.marker_spin.setFixedWidth(60)
        self.marker_spin.setToolTip("Vertex marker size for polygons")
        tool_row.addWidget(self.marker_spin)

        tool_row.addStretch(1)

        tool_row.addWidget(QLabel("Name:"))
        self.name_edit = QLineEdit(self)
        self.name_edit.setPlaceholderText("e.g. center_zone")
        self.name_edit.setMinimumWidth(160)
        tool_row.addWidget(self.name_edit)

        self.draw_btn = QPushButton("Draw  →", self)
        self.draw_btn.setStyleSheet("font-weight: bold; padding: 4px 12px;")
        self.draw_btn.clicked.connect(self._on_draw_clicked)
        tool_row.addWidget(self.draw_btn)
        outer.addLayout(tool_row)

        # ---- Splitter: video list (left) | preview/table (right) ---- #
        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setChildrenCollapsible(False)

        # Left: video list
        left_panel = QWidget(self)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(2)
        left_label = QLabel("Project videos (PgUp/PgDn to nav)")
        left_label.setStyleSheet("font-weight: bold; padding: 2px 4px;")
        left_layout.addWidget(left_label)
        self.video_list = QListWidget(self)
        self.video_list.setAlternatingRowColors(True)
        self.video_list.currentRowChanged.connect(self._on_video_picked)
        left_layout.addWidget(self.video_list)
        splitter.addWidget(left_panel)

        # Right: frame nav + preview + shape table
        right_panel = QWidget(self)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        # Frame nav
        nav_row = QHBoxLayout()
        nav_row.setSpacing(2)

        self.first_btn = QToolButton(self); self.first_btn.setText("⏮")
        self.first_btn.setToolTip("First frame")
        self.first_btn.clicked.connect(self._first_frame)
        nav_row.addWidget(self.first_btn)

        self.back_s_btn = QToolButton(self); self.back_s_btn.setText("−1s")
        self.back_s_btn.clicked.connect(lambda: self._step_seconds(-1.0))
        nav_row.addWidget(self.back_s_btn)

        self.back_f_btn = QToolButton(self); self.back_f_btn.setText("−1f")
        self.back_f_btn.clicked.connect(lambda: self._step_frames(-1))
        nav_row.addWidget(self.back_f_btn)

        self.frame_slider = QSlider(Qt.Horizontal, self)
        self.frame_slider.valueChanged.connect(self._on_slider_changed)
        nav_row.addWidget(self.frame_slider, 1)

        self.fwd_f_btn = QToolButton(self); self.fwd_f_btn.setText("+1f")
        self.fwd_f_btn.clicked.connect(lambda: self._step_frames(1))
        nav_row.addWidget(self.fwd_f_btn)

        self.fwd_s_btn = QToolButton(self); self.fwd_s_btn.setText("+1s")
        self.fwd_s_btn.clicked.connect(lambda: self._step_seconds(1.0))
        nav_row.addWidget(self.fwd_s_btn)

        self.last_btn = QToolButton(self); self.last_btn.setText("⏭")
        self.last_btn.clicked.connect(self._last_frame)
        nav_row.addWidget(self.last_btn)

        self.frame_label = QLabel("Frame 0", self)
        self.frame_label.setMinimumWidth(120)
        self.frame_label.setStyleSheet("color: palette(mid);")
        nav_row.addWidget(self.frame_label)

        right_layout.addLayout(nav_row)

        # Canvas — native Qt. Replaces the OpenCV subprocess approach.
        # The canvas displays the current frame, overlays existing
        # ROIs, and accepts mouse/keyboard input for drawing new ROIs.
        self.preview = ROICanvas(self)
        self.preview.setSizePolicy(QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        self.preview.shape_committed.connect(self._on_shape_committed)
        self.preview.shape_cancelled.connect(self._on_shape_cancelled)
        right_layout.addWidget(self.preview, 1)

        # Shape table
        self.shape_table = QTableWidget(self)
        self.shape_table.setColumnCount(5)
        self.shape_table.setHorizontalHeaderLabels(
            ["#", "Type", "Name", "Color", ""],
        )
        self.shape_table.verticalHeader().setVisible(False)
        self.shape_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.shape_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.shape_table.setMaximumHeight(150)
        h = self.shape_table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(2, QHeaderView.Stretch)
        h.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        right_layout.addWidget(self.shape_table)

        splitter.addWidget(right_panel)
        splitter.setSizes([260, 1040])   # left:right = 1:4
        outer.addWidget(splitter, 1)

        # ---- Bottom action bar ---- #
        bot = QHBoxLayout()

        self.reset_btn = QPushButton("Reset video", self)
        self.reset_btn.setToolTip("Delete all ROIs from the current video")
        self.reset_btn.clicked.connect(self._on_reset_clicked)
        bot.addWidget(self.reset_btn)

        self.apply_all_btn = QPushButton("Apply to all", self)
        self.apply_all_btn.setToolTip(
            "Copy this video's ROIs to every other video in the project"
        )
        self.apply_all_btn.clicked.connect(self._on_apply_all_clicked)
        bot.addWidget(self.apply_all_btn)

        bot.addStretch(1)

        self.save_status = QLabel("", self)
        self.save_status.setStyleSheet("color: palette(mid);")
        bot.addWidget(self.save_status)
        bot.addSpacing(20)

        cancel_btn = QPushButton("Close", self)
        cancel_btn.clicked.connect(self._on_close_clicked)
        bot.addWidget(cancel_btn)

        self.save_btn = QPushButton("Save", self)
        self.save_btn.clicked.connect(self._on_save_clicked)
        bot.addWidget(self.save_btn)

        save_close_btn = QPushButton("Save && close", self)
        save_close_btn.setDefault(True)
        save_close_btn.clicked.connect(self._on_save_and_close)
        bot.addWidget(save_close_btn)
        outer.addLayout(bot)

        # Keyboard shortcuts: R/C/P switch tools; PgUp/PgDn nav videos
        QShortcut(QKeySequence("R"), self,
                  activated=lambda: self._tool_buttons[RECTANGLE].setChecked(True))
        QShortcut(QKeySequence("C"), self,
                  activated=lambda: self._tool_buttons[CIRCLE].setChecked(True))
        QShortcut(QKeySequence("P"), self,
                  activated=lambda: self._tool_buttons[POLYGON].setChecked(True))
        QShortcut(QKeySequence("PgUp"), self,
                  activated=self._prev_video)
        QShortcut(QKeySequence("PgDown"), self,
                  activated=self._next_video)

    # ------------------------------------------------------------------ #
    # Video list
    # ------------------------------------------------------------------ #
    def _refresh_video_list(self) -> None:
        """Rebuild the left-side video list with current ROI status."""
        self.video_list.blockSignals(True)
        self.video_list.clear()
        roi_h5 = os.path.join(
            self.project_path, "logs", "measures", "ROI_definitions.h5",
        )
        videos_with_rois = _videos_with_rois(roi_h5)
        for vpath in self._videos:
            stem = Path(vpath).stem
            has_rois = stem in videos_with_rois
            mark = "✓ " if has_rois else "  "
            item = QListWidgetItem(f"{mark}{stem}")
            if has_rois:
                f = item.font(); f.setBold(True); item.setFont(f)
                item.setForeground(Qt.darkGreen)
            self.video_list.addItem(item)
        self.video_list.setCurrentRow(self._cur_idx)
        self.video_list.blockSignals(False)

    def _on_video_picked(self, row: int) -> None:
        if row < 0 or row >= len(self._videos) or row == self._cur_idx:
            return
        # Auto-save before switching
        if self._dirty and self.logic is not None:
            try:
                self.logic.save()
                self._dirty = False
                self._flash_status("Auto-saved.")
            except Exception as exc:
                # Save failed — abort the switch and keep the user on
                # the current video so their work isn't silently lost.
                QMessageBox.critical(
                    self, "Auto-save failed",
                    f"Could not save current video before switching: "
                    f"{type(exc).__name__}: {exc}\n\n"
                    f"Staying on the current video.",
                )
                self.video_list.blockSignals(True)
                self.video_list.setCurrentRow(self._cur_idx)
                self.video_list.blockSignals(False)
                return
        self._cur_idx = row
        self._load_video(row)

    def _load_video(self, idx: int) -> None:
        """Construct a fresh ROILogic for video[idx] and rebuild the
        preview / table."""
        vpath = self._videos[idx]
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
        self._dirty = False
        self.setWindowTitle(
            f"ROI Definitions — {self.logic.video_name}"
        )
        # Update slider range for this video
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, max(0, self.logic.frame_count - 1))
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)
        self._sync_preview()
        self._sync_table()

    def _prev_video(self) -> None:
        if self._cur_idx > 0:
            self.video_list.setCurrentRow(self._cur_idx - 1)

    def _next_video(self) -> None:
        if self._cur_idx < len(self._videos) - 1:
            self.video_list.setCurrentRow(self._cur_idx + 1)

    # ------------------------------------------------------------------ #
    # Preview / shape table sync
    # ------------------------------------------------------------------ #
    def _sync_preview(self) -> None:
        if self.logic is None:
            return
        # The canvas now renders ROIs natively; pass the RAW frame
        # (no overlays) and let the canvas draw the ROIs on top via
        # set_existing_rois.
        if self.logic.current_frame is not None:
            self.preview.set_frame(self.logic.current_frame)
        # Build the overlay list for the canvas
        overlay_rois = []
        for kind, d in self.logic.defs.items():
            for name, roi in d.items():
                overlay_rois.append({
                    "kind": kind,
                    "color_bgr": roi.color_bgr,
                    "thickness": roi.thickness,
                    "geometry": roi.geometry,
                })
        self.preview.set_existing_rois(overlay_rois)
        self.frame_label.setText(
            f"Frame {self.logic.frame_idx} / "
            f"{self.logic.frame_count - 1}"
        )

    def _sync_preview_and_slider(self) -> None:
        if self.logic is None:
            return
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.logic.frame_idx)
        self.frame_slider.blockSignals(False)
        self._sync_preview()

    def _sync_table(self) -> None:
        if self.logic is None:
            self.shape_table.setRowCount(0)
            return
        all_rois = []
        for kind, d in self.logic.defs.items():
            for name, roi in d.items():
                all_rois.append((kind, name, roi))
        self.shape_table.setRowCount(len(all_rois))
        for i, (kind, name, roi) in enumerate(all_rois):
            idx_item = QTableWidgetItem(str(i + 1))
            idx_item.setTextAlignment(Qt.AlignCenter)
            self.shape_table.setItem(i, 0, idx_item)
            self.shape_table.setItem(i, 1, QTableWidgetItem(kind.capitalize()))
            self.shape_table.setItem(i, 2, QTableWidgetItem(name))
            clr_item = QTableWidgetItem(roi.color_name)
            clr_item.setForeground(_bgr_to_qcolor(roi.color_bgr))
            self.shape_table.setItem(i, 3, clr_item)
            del_btn = QPushButton("✕", self)
            del_btn.setFixedWidth(28)
            del_btn.setToolTip(f"Delete {name}")
            del_btn.clicked.connect(
                lambda _=False, n=name: self._on_delete_roi(n),
            )
            self.shape_table.setCellWidget(i, 4, del_btn)

    # ------------------------------------------------------------------ #
    # Frame nav
    # ------------------------------------------------------------------ #
    def _step_frames(self, n: int) -> None:
        if self.logic is None: return
        self.logic.advance_frame(n)
        self._sync_preview_and_slider()

    def _step_seconds(self, s: float) -> None:
        if self.logic is None: return
        self.logic.jump_seconds(s)
        self._sync_preview_and_slider()

    def _first_frame(self) -> None:
        if self.logic is None: return
        self.logic.first_frame()
        self._sync_preview_and_slider()

    def _last_frame(self) -> None:
        if self.logic is None: return
        self.logic.last_frame()
        self._sync_preview_and_slider()

    def _on_slider_changed(self, val: int) -> None:
        if self.logic is None: return
        self.logic.goto_frame(val)
        self._sync_preview()

    # ------------------------------------------------------------------ #
    # Drawing — synchronous on the main thread
    # ------------------------------------------------------------------ #
    def _selected_kind(self) -> str:
        for kind, btn in self._tool_buttons.items():
            if btn.isChecked():
                return kind
        return RECTANGLE

    def _selected_color(self) -> tuple[str, tuple[int, int, int]]:
        name = self.color_cb.currentText()
        bgr = next((b for n, b in _COLORS if n == name), (0, 0, 255))
        return name, bgr

    def _on_draw_clicked(self) -> None:
        if self.logic is None:
            return
        name = self.name_edit.text().strip()
        if not name:
            self._flash_status("Type a shape name first.", error=True)
            self.name_edit.setFocus()
            return
        if self.logic.has_roi(name):
            self._flash_status(
                f"An ROI named {name!r} already exists for this video.",
                error=True,
            )
            return
        if self.preview.is_drawing():
            self._flash_status(
                "A draw is already in progress. Press ESC to cancel "
                "the current one before starting a new shape.",
                error=True,
            )
            return
        kind = self._selected_kind()
        clr_name, bgr = self._selected_color()
        thickness = self.thickness_spin.value()

        # Stash params for the commit callback to consume
        self._pending_shape = {
            "name": name, "kind": kind,
            "color_name": clr_name, "bgr": bgr,
            "thickness": thickness,
            "ear_tag_size": self.marker_spin.value(),
        }

        # Hand off to the canvas. The next mouse interaction starts
        # capturing the shape; ESC/Q/Space cancel/commit per shape
        # type. shape_committed (or shape_cancelled) fires when done.
        self.preview.start_draw(kind=kind, color_bgr=bgr,
                                thickness=thickness)
        self._flash_status(
            {
                RECTANGLE: f"Drawing rectangle: {name} — "
                           f"drag to define, ESC to cancel",
                CIRCLE: f"Drawing circle: {name} — "
                        f"click center, drag radius, ESC to cancel",
                POLYGON: f"Drawing polygon: {name} — "
                         f"click vertices, ESC/Q/Space to close",
            }.get(kind, f"Drawing {kind}: {name}…"),
        )

    def _on_shape_committed(self, kind: str, geom: dict) -> None:
        """Slot for ROICanvas.shape_committed. Adds the captured shape
        to the logic, refreshes preview + table."""
        params = getattr(self, "_pending_shape", None)
        if params is None:
            self._flash_status(
                "Got a shape but had no pending request — ignoring.",
                error=True,
            )
            return
        name = params["name"]
        clr_name = params["color_name"]
        bgr = params["bgr"]
        thickness = params["thickness"]
        marker = params["ear_tag_size"]
        try:
            if kind == RECTANGLE:
                self.logic.add_rectangle(
                    name=name, top_left=geom["top_left"],
                    bottom_right=geom["bottom_right"],
                    color_name=clr_name, bgr=bgr,
                    thickness=thickness, ear_tag_size=marker,
                )
            elif kind == CIRCLE:
                self.logic.add_circle(
                    name=name, center=geom["center"],
                    radius=geom["radius"],
                    color_name=clr_name, bgr=bgr,
                    thickness=thickness, ear_tag_size=marker,
                )
            elif kind == POLYGON:
                self.logic.add_polygon(
                    name=name, vertices=geom["vertices"],
                    color_name=clr_name, bgr=bgr,
                    thickness=thickness, ear_tag_size=marker,
                )
        except Exception as exc:
            self._flash_status(f"Add failed: {exc}", error=True)
            return
        self._dirty = True
        self.name_edit.clear()
        self._pending_shape = None
        self._sync_preview()
        self._sync_table()
        self._flash_status(f"Added {kind}: {name}")

    def _on_shape_cancelled(self) -> None:
        """Slot for ROICanvas.shape_cancelled."""
        self._pending_shape = None
        self._flash_status("Drawing cancelled.")

    def _on_delete_roi(self, name: str) -> None:
        if self.logic is None:
            return
        if QMessageBox.question(
            self, "Delete ROI", f"Delete ROI {name!r}?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        ) != QMessageBox.Yes:
            return
        self.logic.delete_roi(name)
        self._dirty = True
        self._sync_preview()
        self._sync_table()
        self._flash_status(f"Deleted {name}.")

    # ------------------------------------------------------------------ #
    # Bottom-bar actions
    # ------------------------------------------------------------------ #
    def _on_reset_clicked(self) -> None:
        if self.logic is None:
            return
        if QMessageBox.question(
            self, "Reset video",
            f"Delete every ROI for <b>{self.logic.video_name}</b>?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        ) != QMessageBox.Yes:
            return
        self.logic.delete_all()
        self._dirty = True
        self._sync_preview()
        self._sync_table()
        self._flash_status("All ROIs cleared.")

    def _on_apply_all_clicked(self) -> None:
        if self.logic is None:
            return
        if not self.logic.all_roi_names:
            self._flash_status("No ROIs to apply.", error=True)
            return
        # Save first so the H5 has the source ROIs, then call the
        # existing multiply_ROIs backend (it reads the H5).
        try:
            self.logic.save()
            self._dirty = False
            self._flash_status("Saved — applying to all videos…")
            QApplication.processEvents()
            from mufasa.roi_tools.roi_utils import multiply_ROIs
            multiply_ROIs(config_path=self.config_path,
                          filename=self._videos[self._cur_idx])
            self._refresh_video_list()
            self.rois_modified.emit()
            self._flash_status("Applied ROIs to every video.")
        except Exception as exc:
            QMessageBox.critical(
                self, "Apply-all failed",
                f"Could not apply ROIs: {type(exc).__name__}: {exc}",
            )

    # ------------------------------------------------------------------ #
    # Save / close
    # ------------------------------------------------------------------ #
    def _on_save_clicked(self) -> None:
        if self.logic is None:
            return
        try:
            self.logic.save()
            self._dirty = False
            self._flash_status("Saved.")
            self._refresh_video_list()
            self.rois_modified.emit()
        except Exception as exc:
            QMessageBox.critical(
                self, "Save failed",
                f"Could not save ROIs: {type(exc).__name__}: {exc}",
            )

    def _on_save_and_close(self) -> None:
        self._on_save_clicked()
        if not self._dirty:
            self.close()

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
        self.close()

    def _flash_status(self, msg: str, error: bool = False) -> None:
        if error:
            self.save_status.setStyleSheet("color: #c44;")
        else:
            self.save_status.setStyleSheet("color: palette(mid);")
        self.save_status.setText(msg)


__all__ = ["ROIDefinePanel"]
