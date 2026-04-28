"""
mufasa.ui_qt.dialogs.roi_video_table
====================================

Native Qt replacement for :class:`mufasa.ui.pop_ups.roi_video_table_pop_up.ROIVideoTable`.

A modal-but-not-blocking dialog that lists every video in
``project_folder/videos/`` and lets the user:

* Draw ROIs on a video (launches the existing OpenCV-based
  :class:`ROI_ui` canvas in a subprocess so the Qt event loop stays
  responsive).
* Reset the ROIs for a single video.
* Apply the ROIs from one video to all other videos.
* Access the file-menu actions: Standardise / Duplicate / Import CSV /
  Min-max draw size / Delete all. Each opens its existing Tk popup in
  a subprocess; they can be ported to Qt later if needed.

The dialog auto-refreshes its ROI status column when
``ROI_definitions.h5`` changes on disk (via :class:`QFileSystemWatcher`),
so closing the OpenCV window after drawing visibly flips the status
from "NO ROIs" to "ROIs defined" without manual reload.

Why this exists
---------------

The pre-existing Tk version was reported as "window appears but is
broken — missing buttons, can't pick a video" on user testing. Rather
than chase Tk-on-Qt event loop issues, this is a native Qt replacement
for the picker layer. The actual drawing canvas is still OpenCV — that
layer is well-tested and there's no functional gain from porting it
to QGraphicsView.

Compatibility
-------------

* Reads the same ``project_folder/logs/measures/ROI_definitions.h5``.
* Calls the same backend functions for reset
  (:func:`reset_video_ROIs`) and apply-all (:func:`multiply_ROIs`).
* Subprocess-launched ``ROI_ui`` writes to the same file as before.
* No data migration needed — existing ROI definitions are picked up
  directly.
"""
from __future__ import annotations

import configparser
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import (QFileSystemWatcher, QSize, Qt, QTimer, Signal)
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QHeaderView, QLabel,
                               QMenuBar, QMessageBox, QPushButton,
                               QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)


# Column indices for the table — referenced in several places.
COL_INDEX = 0
COL_THUMB = 1
COL_NAME = 2
COL_DRAW = 3
COL_RESET = 4
COL_APPLY = 5
COL_STATUS = 6
COLUMNS = ["#", "Preview", "Video", "", "", "", "Status"]


def _project_path_from_config(config_path: str) -> Optional[str]:
    """Light-weight lookup of project_path that doesn't pull
    :class:`ConfigReader` (and therefore numba) into the import graph."""
    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    return cfg.get("General settings", "project_path", fallback=None)


def _list_project_videos(project_path: str) -> list[str]:
    """Return absolute paths of all videos in ``project_folder/videos/``,
    sorted alphabetically. Non-recursive."""
    vid_dir = os.path.join(project_path, "videos")
    if not os.path.isdir(vid_dir):
        return []
    out = []
    for name in sorted(os.listdir(vid_dir)):
        if name.startswith("."):
            continue
        if name.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
            out.append(os.path.join(vid_dir, name))
    return out


def _videos_with_rois(roi_h5_path: str) -> set[str]:
    """Read ``ROI_definitions.h5`` and return the set of video stems
    that have at least one ROI of any kind defined.

    Returns empty set if the file doesn't exist or is unreadable —
    same as "no ROIs defined for any video"."""
    if not os.path.isfile(roi_h5_path):
        return set()
    try:
        from mufasa.utils.read_write import read_roi_data
        rect_df, circ_df, poly_df = read_roi_data(roi_path=roi_h5_path)
    except Exception:
        return set()
    out: set[str] = set()
    for df in (rect_df, circ_df, poly_df):
        if df is not None and "Video" in df.columns:
            out.update(df["Video"].dropna().astype(str).unique().tolist())
    return out


def _build_thumbnail(video_path: str,
                     target_size: tuple[int, int] = (160, 90)) -> QPixmap:
    """Read frame 0 of ``video_path`` and return a QPixmap scaled to
    ``target_size``. Returns a placeholder grey pixmap on read failure."""
    try:
        cap = cv2.VideoCapture(video_path)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError("could not read frame 0")
        h, w = frame.shape[:2]
        # Aspect-preserving resize into target_size
        target_w, target_h = target_size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h),
                             interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Pad to target_size with black bars
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        x_off = (target_w - new_w) // 2
        y_off = (target_h - new_h) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = rgb
        qimg = QImage(canvas.data, target_w, target_h, target_w * 3,
                      QImage.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)
    except Exception:
        # Placeholder: solid grey rectangle
        canvas = np.full((target_size[1], target_size[0], 3), 80,
                         dtype=np.uint8)
        qimg = QImage(canvas.data, target_size[0], target_size[1],
                      target_size[0] * 3,
                      QImage.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)


class ROIVideoTableDialog(QDialog):
    """Modal-but-not-blocking dialog showing the project's videos with
    Draw / Reset / Apply-all action buttons per row.

    Use ``.show()`` (not ``.exec()``) to keep it non-blocking. The
    workbench typically does this via launch_dialog() so the dialog
    isn't GC'd."""

    # Emitted after a backend action completes (reset / apply / file-menu
    # operation). The page can use this to refresh anything dependent.
    rois_modified = Signal()

    def __init__(self, config_path: str, parent: Optional[QWidget] = None
                 ) -> None:
        super().__init__(parent)
        self.config_path = config_path
        self.project_path = _project_path_from_config(config_path) or ""
        self.roi_h5_path = os.path.join(
            self.project_path, "logs", "measures", "ROI_definitions.h5",
        )
        self._child_procs: list = []   # keep refs so they aren't GC'd

        self.setWindowTitle("ROI Definitions — Project Videos")
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.resize(1100, 700)

        self._build_ui()
        self._populate_table()

        # Auto-refresh status column when ROI_definitions.h5 changes.
        # The watcher needs the file to exist; if it doesn't yet, watch
        # the parent directory and re-arm once the file appears.
        self._watcher = QFileSystemWatcher(self)
        self._wire_watcher()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        # Menu bar (file menu mimics SimBA's ROI table File menu)
        self._menu = QMenuBar(self)
        file_menu = self._menu.addMenu("&File")

        act_std = QAction("Standardise ROI sizes by metric conversion factor…", self)
        act_std.triggered.connect(self._action_standardize)
        file_menu.addAction(act_std)

        act_dup = QAction("Duplicate ROIs from source video to target videos…", self)
        act_dup.triggered.connect(self._action_duplicate)
        file_menu.addAction(act_dup)

        act_imp = QAction("Import SimBA ROI CSV definitions…", self)
        act_imp.triggered.connect(self._action_import_csv)
        file_menu.addAction(act_imp)

        act_size = QAction("Set min/max draw window size…", self)
        act_size.triggered.connect(self._action_min_max_draw_size)
        file_menu.addAction(act_size)

        file_menu.addSeparator()
        act_del = QAction("Delete all ROIs…", self)
        act_del.triggered.connect(self._action_delete_all)
        file_menu.addAction(act_del)

        outer.setMenuBar(self._menu)

        # Header band with project info
        header = QLabel(self)
        header.setTextFormat(Qt.RichText)
        header.setWordWrap(True)
        header.setStyleSheet("padding: 8px 12px; "
                             "background: palette(alternate-base); "
                             "border-bottom: 1px solid palette(mid);")
        if self.project_path:
            header.setText(
                f"<b>Project:</b> <code>{self.project_path}</code><br>"
                f"<b>ROI definitions file:</b> "
                f"<code>{self.roi_h5_path}</code>"
            )
        else:
            header.setText("<b>No project loaded.</b>")
        outer.addWidget(header)

        # Video table
        self.table = QTableWidget(self)
        self.table.setColumnCount(len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        self.table.setIconSize(QSize(160, 90))
        # Column sizing
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(COL_INDEX,  QHeaderView.ResizeToContents)
        h.setSectionResizeMode(COL_THUMB,  QHeaderView.ResizeToContents)
        h.setSectionResizeMode(COL_NAME,   QHeaderView.Stretch)
        h.setSectionResizeMode(COL_DRAW,   QHeaderView.ResizeToContents)
        h.setSectionResizeMode(COL_RESET,  QHeaderView.ResizeToContents)
        h.setSectionResizeMode(COL_APPLY,  QHeaderView.ResizeToContents)
        h.setSectionResizeMode(COL_STATUS, QHeaderView.ResizeToContents)
        outer.addWidget(self.table)

        # Bottom band with refresh + close
        btm = QHBoxLayout()
        btm.setContentsMargins(8, 4, 8, 8)
        self.refresh_btn = QPushButton("Refresh", self)
        self.refresh_btn.clicked.connect(self._populate_table)
        btm.addWidget(self.refresh_btn)
        btm.addStretch(1)
        close_btn = QPushButton("Close", self)
        close_btn.clicked.connect(self.close)
        btm.addWidget(close_btn)
        outer.addLayout(btm)

    def _populate_table(self) -> None:
        """Build / rebuild the table from disk state."""
        self.table.setRowCount(0)
        if not self.project_path:
            return
        videos = _list_project_videos(self.project_path)
        if not videos:
            return
        videos_with_rois = _videos_with_rois(self.roi_h5_path)
        self.table.setRowCount(len(videos))
        for row, vpath in enumerate(videos):
            stem = Path(vpath).stem
            self._fill_row(row, vpath, stem, has_rois=stem in videos_with_rois)
        # Row height: enough for the 90-px thumbnail
        for r in range(self.table.rowCount()):
            self.table.setRowHeight(r, 100)

    def _fill_row(self, row: int, video_path: str, stem: str,
                  has_rois: bool) -> None:
        # Index
        idx = QTableWidgetItem(str(row + 1))
        idx.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, COL_INDEX, idx)

        # Thumbnail
        thumb_lbl = QLabel(self)
        thumb_lbl.setPixmap(_build_thumbnail(video_path))
        thumb_lbl.setAlignment(Qt.AlignCenter)
        self.table.setCellWidget(row, COL_THUMB, thumb_lbl)

        # Name
        name_item = QTableWidgetItem(stem)
        name_item.setToolTip(video_path)
        # Bold / green if ROIs are defined, regular otherwise
        font = name_item.font()
        if has_rois:
            font.setBold(True)
            name_item.setForeground(Qt.darkGreen)
        name_item.setFont(font)
        self.table.setItem(row, COL_NAME, name_item)

        # Draw / Reset / Apply-all buttons
        draw_btn = QPushButton("DRAW", self)
        draw_btn.clicked.connect(lambda _, v=video_path: self._draw(v))
        self.table.setCellWidget(row, COL_DRAW, draw_btn)

        reset_btn = QPushButton("RESET", self)
        reset_btn.clicked.connect(lambda _, v=video_path: self._reset(v))
        self.table.setCellWidget(row, COL_RESET, reset_btn)

        apply_btn = QPushButton("APPLY TO ALL", self)
        apply_btn.clicked.connect(lambda _, v=video_path: self._apply_all(v))
        self.table.setCellWidget(row, COL_APPLY, apply_btn)

        # Status
        status_text = "ROIs defined" if has_rois else "NO ROIs defined"
        status_item = QTableWidgetItem(status_text)
        status_item.setTextAlignment(Qt.AlignCenter)
        if has_rois:
            f = status_item.font(); f.setBold(True); status_item.setFont(f)
            status_item.setForeground(Qt.darkGreen)
        else:
            status_item.setForeground(Qt.gray)
        self.table.setItem(row, COL_STATUS, status_item)

    # ------------------------------------------------------------------ #
    # Watcher — auto-refresh on ROI_definitions.h5 change
    # ------------------------------------------------------------------ #
    def _wire_watcher(self) -> None:
        # Watch the file if it exists; otherwise watch its parent dir
        # so we get notified when it's first created.
        watch_targets = []
        if os.path.isfile(self.roi_h5_path):
            watch_targets.append(self.roi_h5_path)
        parent_dir = os.path.dirname(self.roi_h5_path)
        if os.path.isdir(parent_dir):
            watch_targets.append(parent_dir)
        if watch_targets:
            self._watcher.addPaths(watch_targets)
        # Coalesce rapid file modifications into a single refresh
        # (h5 writes can hit several events per save).
        self._watcher.fileChanged.connect(self._schedule_refresh)
        self._watcher.directoryChanged.connect(self._schedule_refresh)
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(300)   # 300 ms debounce
        self._refresh_timer.timeout.connect(self._do_watcher_refresh)

    def _schedule_refresh(self, *_) -> None:
        self._refresh_timer.start()

    def _do_watcher_refresh(self) -> None:
        # Re-arm watcher on the file if it was just created.
        if (os.path.isfile(self.roi_h5_path)
                and self.roi_h5_path not in self._watcher.files()):
            self._watcher.addPath(self.roi_h5_path)
        self._populate_table()

    # ------------------------------------------------------------------ #
    # Row actions
    # ------------------------------------------------------------------ #
    def _draw(self, video_path: str) -> None:
        """Open the Qt-native ROI definition panel for this video.

        Replaces the previous subprocess-launched ROI_ui (which used
        Tk + OpenCV with a clunky giant-icon SimBA panel). The new
        panel is :class:`mufasa.ui_qt.dialogs.roi_define_panel.ROIDefinePanel`
        — a GIMP-style compact Qt widget that runs in-process and shares
        the workbench's event loop. The actual mouse-driven shape
        drawing still happens in OpenCV (via ROISelector running in a
        QThread inside the panel), but the surrounding controls are
        all native Qt.

        Multiple panels can be open simultaneously — one per video.
        Each saves to the same project_folder/logs/measures/
        ROI_definitions.h5; the QFileSystemWatcher on this dialog
        refreshes the status column when any of them save.
        """
        try:
            from mufasa.ui_qt.dialogs.roi_define_panel import (
                ROIDefinePanel,
            )
            panel = ROIDefinePanel(
                config_path=self.config_path, video_path=video_path,
                parent=self,
            )
            # Connect the panel's saved signal to our refresh, so the
            # status column updates immediately on save (in addition to
            # the QFileSystemWatcher's debounced callback).
            panel.saved.connect(lambda _name: self._populate_table())
            panel.show()
            self._child_procs.append(panel)   # keep ref so it isn't GC'd
        except Exception as exc:
            QMessageBox.critical(
                self, "Panel open failed",
                f"Could not open ROI definition panel: "
                f"{type(exc).__name__}: {exc}"
            )

    def _reset(self, video_path: str) -> None:
        from pathlib import Path as _P
        stem = _P(video_path).stem
        ok = QMessageBox.question(
            self, "Reset ROIs",
            f"Delete all ROI definitions for video <b>{stem}</b>?<br><br>"
            f"This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if ok != QMessageBox.Yes:
            return
        try:
            from mufasa.roi_tools.roi_utils import reset_video_ROIs
            reset_video_ROIs(config_path=self.config_path, filename=video_path)
            self._populate_table()
            self.rois_modified.emit()
        except Exception as exc:
            QMessageBox.critical(
                self, "Reset failed",
                f"Could not reset ROIs: {type(exc).__name__}: {exc}"
            )

    def _apply_all(self, video_path: str) -> None:
        from pathlib import Path as _P
        stem = _P(video_path).stem
        ok = QMessageBox.question(
            self, "Apply ROIs to all videos",
            f"Copy the ROI definitions from video <b>{stem}</b> to "
            f"<b>every other video in this project</b>? This will "
            f"overwrite existing ROIs on the target videos.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if ok != QMessageBox.Yes:
            return
        try:
            from mufasa.roi_tools.roi_utils import multiply_ROIs
            multiply_ROIs(config_path=self.config_path, filename=video_path)
            self._populate_table()
            self.rois_modified.emit()
        except Exception as exc:
            QMessageBox.critical(
                self, "Apply-all failed",
                f"Could not apply ROIs: {type(exc).__name__}: {exc}"
            )

    # ------------------------------------------------------------------ #
    # File-menu actions — each launches its existing Tk popup in a
    # subprocess. Ports of these to Qt are out of scope for this patch.
    # ------------------------------------------------------------------ #
    def _action_standardize(self) -> None:
        self._launch_tk_popup(
            "from mufasa.ui.pop_ups.roi_size_standardizer_popup import ROISizeStandardizerPopUp\n"
            "ROISizeStandardizerPopUp(config_path=sys.argv[1])\n"
        )

    def _action_duplicate(self) -> None:
        self._launch_tk_popup(
            "from mufasa.ui.pop_ups.duplicate_rois_by_source_target_popup import DuplicateROIsBySourceTarget\n"
            "DuplicateROIsBySourceTarget(config_path=sys.argv[1], roi_data_path=None, roi_table_popup=None)\n"
        )

    def _action_import_csv(self) -> None:
        self._launch_tk_popup(
            "from mufasa.ui.pop_ups.import_roi_csv_popup import ROIDefinitionsCSVImporterPopUp\n"
            "ROIDefinitionsCSVImporterPopUp(config_path=sys.argv[1], roi_table_frm=None)\n"
        )

    def _action_min_max_draw_size(self) -> None:
        self._launch_tk_popup(
            "from mufasa.ui.pop_ups.min_max_draw_size_popup import SetMinMaxDrawWindowSize\n"
            "SetMinMaxDrawWindowSize(config_path=sys.argv[1])\n"
        )

    def _action_delete_all(self) -> None:
        ok = QMessageBox.question(
            self, "Delete ALL ROIs",
            "Delete every ROI definition in this project?<br><br>"
            "This affects every video and cannot be undone.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if ok != QMessageBox.Yes:
            return
        # delete_all_rois_pop_up is itself a Tk popup that asks for
        # confirmation again. We've already confirmed; just delete the
        # file directly. Faster, no second confirmation prompt.
        try:
            if os.path.isfile(self.roi_h5_path):
                os.remove(self.roi_h5_path)
            self._populate_table()
            self.rois_modified.emit()
        except Exception as exc:
            QMessageBox.critical(
                self, "Delete failed",
                f"Could not delete: {type(exc).__name__}: {exc}"
            )

    def _launch_tk_popup(self, body: str) -> None:
        """Run a Tk popup in a subprocess so it doesn't deadlock Qt."""
        import subprocess
        launcher = "import sys\n" + body
        try:
            proc = subprocess.Popen(
                [sys.executable, "-c", launcher, self.config_path],
                stdin=subprocess.DEVNULL,
            )
            self._child_procs.append(proc)
        except Exception as exc:
            QMessageBox.critical(
                self, "Launch failed",
                f"Could not launch the popup: "
                f"{type(exc).__name__}: {exc}"
            )

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def closeEvent(self, ev) -> None:
        # Clean up references but don't kill the child subprocesses —
        # they may still be open and that's fine; the user closed our
        # picker, not the drawing canvas.
        super().closeEvent(ev)


__all__ = ["ROIVideoTableDialog"]
