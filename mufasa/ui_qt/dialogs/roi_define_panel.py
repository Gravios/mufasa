"""
mufasa.ui_qt.dialogs.roi_define_panel
=====================================

GIMP-style Qt panel for defining ROIs on a single video.

Layout (top to bottom)
----------------------

1. **Tool palette** — three small toggle buttons: Rectangle / Circle /
   Polygon. Mutually exclusive. Compact icons (24x24), like GIMP's
   tool dock.
2. **Tool options** — color, line thickness, vertex marker size,
   shape name. Single row of compact controls.
3. **Frame navigation** — slider with -1s / +1s / -1f / +1f /
   first / last buttons. The slider scrubs the underlying video.
4. **Image preview** — current frame with all defined ROIs overlaid.
   Self-updates as user adds / deletes ROIs and changes frame.
5. **Shape list** — table of currently-defined ROIs with delete-row
   and rename-on-doubleclick. Click row → highlight in preview.
6. **Save bar** — sticky bottom: Save / Save & Close / Cancel.

Differences from the legacy Tk panel
------------------------------------

* No "ear_tag_size" dropdown (auto-derived from thickness)
* No "polygon tolerance" knob (uses sensible default)
* No buried "apply from other video" dropdown — that's now a separate
  action in the parent video-table dialog
* No status bar (status messages flash inline near the action that
  produced them)
* No info / move / ruler buttons (interaction is direct: click a row
  to select; resize via OpenCV canvas if needed)

The actual mouse-driven shape drawing still happens in OpenCV via the
existing :class:`mufasa.video_processors.roi_selector.ROISelector`
classes — running as a :class:`QThread` so the panel stays responsive
during drawing.

Lifecycle
---------

The panel owns one :class:`mufasa.roi_tools.roi_logic.ROILogic`
instance per video. UI events translate to logic-method calls;
``rendered_frame()`` is queried after every state change to refresh
the preview.

Save persists via :meth:`ROILogic.save` to the project's
``ROI_definitions.h5``. Reads/writes are compatible with all
downstream Mufasa tools.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import (QObject, Qt, QThread, Signal)
from PySide6.QtGui import (QColor, QImage, QKeySequence, QPixmap, QShortcut)
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QDialog, QFrame,
                               QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QMessageBox, QPushButton,
                               QSizePolicy, QSlider, QSpinBox,
                               QTableWidget, QTableWidgetItem, QToolButton,
                               QVBoxLayout, QWidget)

from mufasa.roi_tools.roi_logic import (CIRCLE, POLYGON, RECTANGLE,
                                        ROILogic)


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


class _SelectorThread(QThread):
    """Run an OpenCV ROI selector in a background thread so the Qt
    panel stays responsive while the user click-drags on the OpenCV
    canvas. Emits ``finished_with_attrs`` carrying the selector's
    captured attributes (or None if cancelled)."""

    finished_with_attrs = Signal(object)   # Optional[dict]

    def __init__(self, selector_kind: str, image: np.ndarray,
                 thickness: int, bgr: tuple[int, int, int],
                 ear_tag_size: int = 15,
                 parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.selector_kind = selector_kind
        self.image = image.copy()
        self.thickness = thickness
        self.bgr = bgr
        self.ear_tag_size = ear_tag_size

    def run(self) -> None:
        result: Optional[dict] = None
        try:
            if self.selector_kind == RECTANGLE:
                from mufasa.video_processors.roi_selector import ROISelector
                sel = ROISelector(path=self.image, thickness=self.thickness,
                                  clr=self.bgr,
                                  title="Draw rectangle — drag with mouse, "
                                        "ESC to confirm")
                sel.run()
                # Rectangle's run_checks() sets self.complete=True after
                # clamping roi_start/roi_end into self.top_left/
                # self.bottom_right. Use the clamped values, not the raw
                # mouse positions (which may be off-canvas).
                if (getattr(sel, "complete", False)
                        and getattr(sel, "top_left", None) is not None
                        and getattr(sel, "bottom_right", None) is not None
                        and getattr(sel, "width", 0) > 0
                        and getattr(sel, "height", 0) > 0):
                    result = {
                        "kind": RECTANGLE,
                        "top_left": tuple(sel.top_left),
                        "bottom_right": tuple(sel.bottom_right),
                    }
            elif self.selector_kind == CIRCLE:
                from mufasa.video_processors.roi_selector_circle import (
                    ROISelectorCircle,
                )
                sel = ROISelectorCircle(
                    path=self.image, thickness=self.thickness, clr=self.bgr,
                    title="Draw circle — click center, drag radius, "
                          "ESC to confirm",
                )
                sel.run()
                # Circle has no `complete` attribute. Success indicator:
                # terminate=True (the run loop only breaks via that flag
                # AFTER run_checks() returns True), AND a non-zero
                # radius. Geometry is in `circle_center` / `circle_radius`.
                if (getattr(sel, "terminate", False)
                        and getattr(sel, "circle_radius", 0) > 0
                        and getattr(sel, "circle_center", (-1, -1))[0] >= 0):
                    cx, cy = sel.circle_center
                    result = {
                        "kind": CIRCLE,
                        "center": (int(cx), int(cy)),
                        "radius": int(sel.circle_radius),
                    }
            elif self.selector_kind == POLYGON:
                from mufasa.video_processors.roi_selector_polygon import (
                    ROISelectorPolygon,
                )
                sel = ROISelectorPolygon(
                    path=self.image, thickness=self.thickness, clr=self.bgr,
                    vertice_size=self.ear_tag_size,
                    title="Draw polygon — click vertices, "
                          "ESC / Q / Space to close",
                )
                sel.run()
                # Polygon also has no `complete` attribute. Success:
                # terminate=True (set after run_checks() returns True,
                # i.e. >= 3 unique vertices). After run_checks(),
                # self.polygon_vertices is replaced with the simplified
                # numpy array, and self.polygon_arr holds the int32
                # version ready for cv2 drawing.
                verts_attr = (getattr(sel, "polygon_arr", None)
                              if hasattr(sel, "polygon_arr")
                              else getattr(sel, "polygon_vertices", None))
                if (getattr(sel, "terminate", False)
                        and verts_attr is not None
                        and len(verts_attr) >= 3):
                    result = {
                        "kind": POLYGON,
                        "vertices": [(int(v[0]), int(v[1]))
                                     for v in verts_attr],
                    }
        except Exception as exc:
            print(f"ROI selector raised {type(exc).__name__}: {exc}")
        self.finished_with_attrs.emit(result)


class _PreviewLabel(QLabel):
    """Label that shows the current frame with ROI overlays. Aspect-
    preserving scaling — adjusts as the dialog resizes."""
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 360)
        self.setStyleSheet(
            "background: #1a1a1a; border: 1px solid palette(mid);"
        )
        self._pix: Optional[QPixmap] = None

    def set_frame(self, bgr: Optional[np.ndarray]) -> None:
        if bgr is None:
            self.clear()
            self._pix = None
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
        self._pix = QPixmap.fromImage(qimg)
        self._refresh_scaled()

    def _refresh_scaled(self) -> None:
        if self._pix is None:
            return
        scaled = self._pix.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        self.setPixmap(scaled)

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        self._refresh_scaled()


class ROIDefinePanel(QDialog):
    """The Qt-native ROI definition panel.

    Construct with a config_path + video_path and call ``.show()``.
    The dialog stays open until the user closes it; ROIs are saved
    to the project's ROI_definitions.h5 when the user clicks Save.
    """

    saved = Signal(str)   # video name when save completes

    def __init__(self, config_path: str, video_path: str,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.logic = ROILogic(config_path=config_path, video_path=video_path)
        self.video_name = self.logic.video_name

        self.setWindowTitle(f"ROI Definitions — {self.video_name}")
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.resize(1100, 800)

        # State for the active drawing operation
        self._selector_thread: Optional[_SelectorThread] = None
        self._dirty = False

        self._build_ui()
        self._sync_preview()
        self._sync_table()

    # ------------------------------------------------------------------ #
    # UI construction — top to bottom
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        # ---- 1. Tool palette ---- #
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

        # Tool options inline with the palette
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

        # The DRAW button is the action — kicks off the OpenCV canvas
        self.draw_btn = QPushButton("Draw  →", self)
        self.draw_btn.setStyleSheet("font-weight: bold; padding: 4px 12px;")
        self.draw_btn.clicked.connect(self._on_draw_clicked)
        tool_row.addWidget(self.draw_btn)

        outer.addLayout(tool_row)

        # ---- thin separator ---- #
        sep = QFrame(self)
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        outer.addWidget(sep)

        # ---- 2. Frame nav ---- #
        nav_row = QHBoxLayout()
        nav_row.setSpacing(2)

        self.first_btn = QToolButton(self); self.first_btn.setText("⏮")
        self.first_btn.setToolTip("First frame")
        self.first_btn.clicked.connect(self.logic.first_frame)
        self.first_btn.clicked.connect(self._sync_preview_and_slider)
        nav_row.addWidget(self.first_btn)

        self.back_s_btn = QToolButton(self); self.back_s_btn.setText("−1s")
        self.back_s_btn.setToolTip("Back 1 second")
        self.back_s_btn.clicked.connect(lambda: self._step_seconds(-1.0))
        nav_row.addWidget(self.back_s_btn)

        self.back_f_btn = QToolButton(self); self.back_f_btn.setText("−1f")
        self.back_f_btn.setToolTip("Back 1 frame")
        self.back_f_btn.clicked.connect(lambda: self._step_frames(-1))
        nav_row.addWidget(self.back_f_btn)

        self.frame_slider = QSlider(Qt.Horizontal, self)
        self.frame_slider.setRange(0, max(0, self.logic.frame_count - 1))
        self.frame_slider.setValue(0)
        self.frame_slider.valueChanged.connect(self._on_slider_changed)
        nav_row.addWidget(self.frame_slider, 1)

        self.fwd_f_btn = QToolButton(self); self.fwd_f_btn.setText("+1f")
        self.fwd_f_btn.setToolTip("Forward 1 frame")
        self.fwd_f_btn.clicked.connect(lambda: self._step_frames(1))
        nav_row.addWidget(self.fwd_f_btn)

        self.fwd_s_btn = QToolButton(self); self.fwd_s_btn.setText("+1s")
        self.fwd_s_btn.setToolTip("Forward 1 second")
        self.fwd_s_btn.clicked.connect(lambda: self._step_seconds(1.0))
        nav_row.addWidget(self.fwd_s_btn)

        self.last_btn = QToolButton(self); self.last_btn.setText("⏭")
        self.last_btn.setToolTip("Last frame")
        self.last_btn.clicked.connect(self.logic.last_frame)
        self.last_btn.clicked.connect(self._sync_preview_and_slider)
        nav_row.addWidget(self.last_btn)

        self.frame_label = QLabel("Frame 0", self)
        self.frame_label.setMinimumWidth(120)
        self.frame_label.setStyleSheet("color: palette(mid);")
        nav_row.addWidget(self.frame_label)

        outer.addLayout(nav_row)

        # ---- 3. Preview ---- #
        self.preview = _PreviewLabel(self)
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        outer.addWidget(self.preview, 1)

        # ---- 4. Shape list ---- #
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
        outer.addWidget(self.shape_table)

        # ---- 5. Save bar ---- #
        save_row = QHBoxLayout()
        save_row.addStretch(1)
        self.save_status = QLabel("", self)
        self.save_status.setStyleSheet("color: palette(mid);")
        save_row.addWidget(self.save_status)
        save_row.addSpacing(20)

        cancel_btn = QPushButton("Close", self)
        cancel_btn.clicked.connect(self._on_close_clicked)
        save_row.addWidget(cancel_btn)

        self.save_btn = QPushButton("Save", self)
        self.save_btn.clicked.connect(self._on_save_clicked)
        save_row.addWidget(self.save_btn)

        save_close_btn = QPushButton("Save && close", self)
        save_close_btn.setDefault(True)
        save_close_btn.clicked.connect(self._on_save_and_close)
        save_row.addWidget(save_close_btn)
        outer.addLayout(save_row)

        # Keyboard shortcuts: R/C/P switch tools
        QShortcut(QKeySequence("R"), self,
                  activated=lambda: self._tool_buttons[RECTANGLE].setChecked(True))
        QShortcut(QKeySequence("C"), self,
                  activated=lambda: self._tool_buttons[CIRCLE].setChecked(True))
        QShortcut(QKeySequence("P"), self,
                  activated=lambda: self._tool_buttons[POLYGON].setChecked(True))

    # ------------------------------------------------------------------ #
    # State synchronization
    # ------------------------------------------------------------------ #
    def _sync_preview(self) -> None:
        self.preview.set_frame(self.logic.rendered_frame())
        self.frame_label.setText(
            f"Frame {self.logic.frame_idx} / {self.logic.frame_count - 1}"
        )

    def _sync_preview_and_slider(self) -> None:
        # Block signals to avoid recursion
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.logic.frame_idx)
        self.frame_slider.blockSignals(False)
        self._sync_preview()

    def _sync_table(self) -> None:
        all_rois = []
        for kind, d in self.logic.defs.items():
            for name, roi in d.items():
                all_rois.append((kind, name, roi))
        self.shape_table.setRowCount(len(all_rois))
        for i, (kind, name, roi) in enumerate(all_rois):
            idx = QTableWidgetItem(str(i + 1))
            idx.setTextAlignment(Qt.AlignCenter)
            self.shape_table.setItem(i, 0, idx)
            type_item = QTableWidgetItem(kind.capitalize())
            self.shape_table.setItem(i, 1, type_item)
            name_item = QTableWidgetItem(name)
            self.shape_table.setItem(i, 2, name_item)
            clr_item = QTableWidgetItem(roi.color_name)
            clr_item.setForeground(_bgr_to_qcolor(roi.color_bgr))
            self.shape_table.setItem(i, 3, clr_item)
            del_btn = QPushButton("✕", self)
            del_btn.setFixedWidth(28)
            del_btn.setToolTip(f"Delete {name}")
            del_btn.clicked.connect(
                lambda _=False, n=name: self._on_delete_roi(n)
            )
            self.shape_table.setCellWidget(i, 4, del_btn)

    # ------------------------------------------------------------------ #
    # Frame nav handlers
    # ------------------------------------------------------------------ #
    def _step_frames(self, n: int) -> None:
        self.logic.advance_frame(n)
        self._sync_preview_and_slider()

    def _step_seconds(self, s: float) -> None:
        self.logic.jump_seconds(s)
        self._sync_preview_and_slider()

    def _on_slider_changed(self, val: int) -> None:
        self.logic.goto_frame(val)
        self._sync_preview()

    # ------------------------------------------------------------------ #
    # Drawing
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
        if self._selector_thread is not None and self._selector_thread.isRunning():
            QMessageBox.information(
                self, "Drawing in progress",
                "Finish or cancel the current drawing before starting a new one."
            )
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
        kind = self._selected_kind()
        clr_name, bgr = self._selected_color()
        thickness = self.thickness_spin.value()
        marker = self.marker_spin.value()
        # Pass the current rendered frame so the user sees existing ROIs
        # while drawing.
        img = self.logic.rendered_frame()
        if img is None:
            self._flash_status("Cannot read current frame.", error=True)
            return

        self._selector_thread = _SelectorThread(
            selector_kind=kind, image=img,
            thickness=thickness, bgr=bgr, ear_tag_size=marker,
            parent=self,
        )
        self._selector_thread.finished_with_attrs.connect(
            lambda attrs: self._on_selector_done(name, clr_name, bgr,
                                                  thickness, marker, attrs),
        )
        self._flash_status(f"Drawing {kind}: {name}…")
        self.draw_btn.setEnabled(False)
        self._selector_thread.start()

    def _on_selector_done(self, name: str, clr_name: str,
                          bgr: tuple[int, int, int], thickness: int,
                          marker: int, attrs: Optional[dict]) -> None:
        self.draw_btn.setEnabled(True)
        self._selector_thread = None
        if attrs is None:
            self._flash_status("Drawing cancelled.")
            return
        try:
            if attrs["kind"] == RECTANGLE:
                self.logic.add_rectangle(
                    name=name, top_left=attrs["top_left"],
                    bottom_right=attrs["bottom_right"],
                    color_name=clr_name, bgr=bgr,
                    thickness=thickness, ear_tag_size=marker,
                )
            elif attrs["kind"] == CIRCLE:
                self.logic.add_circle(
                    name=name, center=attrs["center"],
                    radius=attrs["radius"],
                    color_name=clr_name, bgr=bgr,
                    thickness=thickness, ear_tag_size=marker,
                )
            elif attrs["kind"] == POLYGON:
                self.logic.add_polygon(
                    name=name, vertices=attrs["vertices"],
                    color_name=clr_name, bgr=bgr,
                    thickness=thickness, ear_tag_size=marker,
                )
        except Exception as exc:
            self._flash_status(f"Add failed: {exc}", error=True)
            return
        self._dirty = True
        self.name_edit.clear()
        self._sync_preview()
        self._sync_table()
        self._flash_status(f"Added {attrs['kind']}: {name}")

    def _on_delete_roi(self, name: str) -> None:
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
    # Save / close
    # ------------------------------------------------------------------ #
    def _on_save_clicked(self) -> None:
        try:
            self.logic.save()
            self._dirty = False
            self._flash_status("Saved.")
            self.saved.emit(self.video_name)
        except Exception as exc:
            QMessageBox.critical(
                self, "Save failed",
                f"Could not save ROIs: {type(exc).__name__}: {exc}"
            )

    def _on_save_and_close(self) -> None:
        self._on_save_clicked()
        if not self._dirty:
            self.close()

    def _on_close_clicked(self) -> None:
        if self._dirty:
            ans = QMessageBox.question(
                self, "Unsaved changes",
                "You have unsaved ROI changes. Save before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save,
            )
            if ans == QMessageBox.Save:
                self._on_save_clicked()
                if self._dirty:
                    return  # save failed
            elif ans == QMessageBox.Cancel:
                return
        self.close()

    def closeEvent(self, ev) -> None:
        if self._selector_thread is not None and self._selector_thread.isRunning():
            self._selector_thread.wait(2000)
        super().closeEvent(ev)

    # ------------------------------------------------------------------ #
    # Status flash
    # ------------------------------------------------------------------ #
    def _flash_status(self, msg: str, error: bool = False) -> None:
        if error:
            self.save_status.setStyleSheet("color: #c44;")
        else:
            self.save_status.setStyleSheet("color: palette(mid);")
        self.save_status.setText(msg)


__all__ = ["ROIDefinePanel"]
