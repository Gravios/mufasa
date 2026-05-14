"""
mufasa.ui_qt.forms.batch_pre_process
=====================================

Qt port of :class:`mufasa.video_processors.batch_process_menus.BatchProcessFrame`
— the multi-step video pre-processing wizard (crop → downsample →
greyscale → flip/rotate → clip → FPS → CLAHE). Replaces the
:class:`BatchPreProcessLauncher` placeholder.

Patch 122al (this file)
-----------------------
The legacy Tk widget was 588 lines of grid layouts with one
``Entry_Box`` per per-video cell and three separate "headings"
rows. Qt's ``QTableWidget`` collapses that into a single
declarative table — same operations, but the per-row machinery
is uniform.

In-frame + dockable
-------------------
The form is an :class:`OperationForm` so it lives inline on the
Preprocessing page like any other section. It also exposes
:meth:`pop_out_to_dock` (wired to a "Pop out" button) which
re-parents the form into a :class:`QDockWidget` attached to the
workbench's main window — same pattern as 122aj's frame
labeller. Floating, dockable to any area, and re-dockable back
into the workbench.

Operations supported (per Tk parity)
------------------------------------
Per-video, in execution order (matches FFMPEGCommandCreator):

1. clahe         (CLAHE contrast enhancement)
2. frame_cnt     (overlay frame counter)
3. grayscale     (convert to greyscale)
4. fps           (resample to target FPS)
5. downsample    (resize to width × height)
6. clip          (trim to [start, end])
7. crop          (crop to ROI rectangle picked via OpenCV)

Quick-settings buttons apply a value across all rows at once;
per-row entries override.

Deferred from Tk parity
-----------------------
* Preferences dialog (codec, font, text location, CLAHE clip
  limit, crop colour/thickness). The Tk widget exposed a
  Preferences pop-up to tweak these; the Qt port uses the
  baked-in :data:`SETTINGS` defaults from the legacy module.
  Adding a Preferences dialog is a 10-minute follow-up if
  needed.
* Per-row video thumbnail-on-hover. Pure cosmetic.
"""
from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDockWidget, QFileDialog,
                               QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QMessageBox, QPushButton,
                               QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

from mufasa.ui_qt.workbench import OperationForm


# --------------------------------------------------------------------------- #
# Module-level settings — mirror the Tk module's SETTINGS dict
# --------------------------------------------------------------------------- #
SETTINGS = {
    "codec": "libx264",          # Formats.BATCH_CODEC.value in Tk
    "crop_color": "Pink",
    "font": "Arial",
    "crop_thickness": 10,
    "txt_loc": "bottom_middle",
    "font_size": 25,
    "clahe_clip_limit": 2,
    "clahe_tile_size": 16,
}

# Operation order matches FFMPEGCommandCreator's pipeline
_OPS_IN_EXEC_ORDER = (
    "clahe", "frame_cnt", "grayscale", "fps", "downsample", "clip", "crop",
)

# Column layout for the per-video QTableWidget
COL_VIDEO     = 0
COL_CROP      = 1
COL_START     = 2
COL_END       = 3
COL_CLIP_CB   = 4
COL_WIDTH     = 5
COL_HEIGHT    = 6
COL_DS_CB     = 7
COL_FPS       = 8
COL_FPS_CB    = 9
COL_GREY_CB   = 10
COL_FRMCNT_CB = 11
COL_CLAHE_CB  = 12
COL_QUALITY   = 13
N_COLS = 14

_COL_HEADERS = [
    "Video", "Crop", "Start", "End", "Clip",
    "Width", "Height", "DS",
    "FPS", "FPS?",
    "Grey", "FrameCnt", "CLAHE",
    "Quality %",
]

# Quality options — same set as Tk's quick-set dropdown
_QUALITY_VALUES = [str(v) for v in range(10, 110, 10)]


class BatchPreProcessForm(OperationForm):
    """In-frame Qt port of the batch video pre-process wizard.

    Lives as a section on the Preprocessing page. Can also be
    popped out to a floating dockable window via the "Pop out"
    button (same pattern as the frame labeller in 122aj).
    """

    title = "Preprocess Videos"
    description = (
        "Batch pre-process a directory of videos with crop, clip, "
        "downsample, FPS resample, greyscale, frame-counter overlay, "
        "and CLAHE in one pass. Quick-apply settings broadcast values "
        "to all rows; per-row entries override."
    )

    # ----------------------------------------------------------- State
    def __init__(self,
                 parent: Optional[QWidget] = None,
                 config_path: Optional[str] = None) -> None:
        # State must be set BEFORE build() runs (super().__init__
        # calls self.build).
        self.input_dir: Optional[str] = None
        self.output_dir: Optional[str] = None
        self.videos_in_dir_dict: dict = {}
        self.crop_dict: dict = {}
        self.settings = deepcopy(SETTINGS)
        self._docked_widget: Optional[QDockWidget] = None
        super().__init__(parent=parent, config_path=config_path)

    # ----------------------------------------------------------- UI
    def build(self) -> None:
        # --- Top: input / output directory pickers --------------- #
        dirs_row = QHBoxLayout()
        self.in_edit = QLineEdit(self)
        self.in_edit.setPlaceholderText("Input directory (videos)")
        self.in_edit.setReadOnly(True)
        in_browse = QPushButton("Browse…", self)
        in_browse.clicked.connect(self._on_pick_input_dir)
        out_browse = QPushButton("Browse…", self)
        out_browse.clicked.connect(self._on_pick_output_dir)
        self.out_edit = QLineEdit(self)
        self.out_edit.setPlaceholderText("Output directory")
        self.out_edit.setReadOnly(True)

        dirs_row.addWidget(QLabel("Input:", self))
        dirs_row.addWidget(self.in_edit, 3)
        dirs_row.addWidget(in_browse)
        dirs_row.addWidget(QLabel("Output:", self))
        dirs_row.addWidget(self.out_edit, 3)
        dirs_row.addWidget(out_browse)
        self.body_layout.addLayout(dirs_row)

        # --- Quick settings (4 grouped controls) ----------------- #
        quick_row = QHBoxLayout()
        quick_row.addWidget(self._build_quick_clip_group())
        quick_row.addWidget(self._build_quick_downsample_group())
        quick_row.addWidget(self._build_quick_fps_group())
        quick_row.addWidget(self._build_quick_quality_group())
        quick_row.addStretch()
        self.body_layout.addLayout(quick_row)

        # --- Per-video table ------------------------------------- #
        self.table = QTableWidget(0, N_COLS, self)
        self.table.setHorizontalHeaderLabels(_COL_HEADERS)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents,
        )
        self.table.horizontalHeader().setSectionResizeMode(
            COL_VIDEO, QHeaderView.Stretch,
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setMinimumHeight(220)
        self.body_layout.addWidget(self.table, 1)

        # --- Action row (Reset all / Reset crop / Pop out) ------ #
        actions = QHBoxLayout()
        self.reset_all_btn = QPushButton("Reset all", self)
        self.reset_all_btn.clicked.connect(self._on_reset_all)
        self.reset_crop_btn = QPushButton("Reset crop", self)
        self.reset_crop_btn.clicked.connect(self._on_reset_crop)
        self.pop_out_btn = QPushButton("Pop out ⇱", self)
        self.pop_out_btn.setToolTip(
            "Detach this form into a floating dockable window. "
            "Click again to re-dock into the workbench.",
        )
        self.pop_out_btn.clicked.connect(self._toggle_pop_out)
        actions.addWidget(self.reset_all_btn)
        actions.addWidget(self.reset_crop_btn)
        actions.addStretch()
        actions.addWidget(self.pop_out_btn)
        self.body_layout.addLayout(actions)

        # The shell's "Run" button drives execute. Override its
        # label to clarify intent.
        self.run_btn.setText("  Execute pipeline")

    # ----------------------------------------------------------- Quick groups
    def _build_quick_clip_group(self) -> QGroupBox:
        box = QGroupBox("Clip", self)
        layout = QVBoxLayout(box)
        self.q_clip_start = QLineEdit("00:00:00", self)
        self.q_clip_end = QLineEdit("00:00:00", self)
        for w, label in (
            (self.q_clip_start, "Start (HH:MM:SS):"),
            (self.q_clip_end, "End (HH:MM:SS):"),
        ):
            row = QHBoxLayout()
            row.addWidget(QLabel(label, box))
            row.addWidget(w)
            layout.addLayout(row)
        btn = QPushButton("Apply to all", box)
        btn.clicked.connect(self._apply_clip_to_all)
        layout.addWidget(btn)
        return box

    def _build_quick_downsample_group(self) -> QGroupBox:
        box = QGroupBox("Downsample", self)
        layout = QVBoxLayout(box)
        self.q_ds_width = QLineEdit("400", self)
        self.q_ds_height = QLineEdit("600", self)
        for w, label in (
            (self.q_ds_width, "Width (px):"),
            (self.q_ds_height, "Height (px):"),
        ):
            row = QHBoxLayout()
            row.addWidget(QLabel(label, box))
            row.addWidget(w)
            layout.addLayout(row)
        btn = QPushButton("Apply to all", box)
        btn.clicked.connect(self._apply_downsample_to_all)
        layout.addWidget(btn)
        return box

    def _build_quick_fps_group(self) -> QGroupBox:
        box = QGroupBox("FPS", self)
        layout = QVBoxLayout(box)
        self.q_fps = QLineEdit("15", self)
        row = QHBoxLayout()
        row.addWidget(QLabel("FPS:", box))
        row.addWidget(self.q_fps)
        layout.addLayout(row)
        btn = QPushButton("Apply to all", box)
        btn.clicked.connect(self._apply_fps_to_all)
        layout.addWidget(btn)
        return box

    def _build_quick_quality_group(self) -> QGroupBox:
        box = QGroupBox("Quality", self)
        layout = QVBoxLayout(box)
        self.q_gpu = QComboBox(box)
        self.q_gpu.addItems(["FALSE", "TRUE"])
        self.q_quality = QComboBox(box)
        self.q_quality.addItems(_QUALITY_VALUES)
        self.q_quality.setCurrentText("60")
        for w, label in (
            (self.q_gpu, "Use GPU:"),
            (self.q_quality, "Quality %:"),
        ):
            row = QHBoxLayout()
            row.addWidget(QLabel(label, box))
            row.addWidget(w)
            layout.addLayout(row)
        btn = QPushButton("Apply quality to all", box)
        btn.clicked.connect(self._apply_quality_to_all)
        layout.addWidget(btn)
        return box

    # ----------------------------------------------------------- Dir pickers
    def _on_pick_input_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select input video directory",
        )
        if not path:
            return
        self.input_dir = path
        self.in_edit.setText(path)
        if not self.output_dir:
            default_out = os.path.join(path, "batch_out")
            self.output_dir = default_out
            self.out_edit.setText(default_out)
        self._reload_videos()

    def _on_pick_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select output directory",
        )
        if not path:
            return
        self.output_dir = path
        self.out_edit.setText(path)

    # ----------------------------------------------------------- Video discovery
    def _reload_videos(self) -> None:
        """Scan ``self.input_dir`` for videos and rebuild the table.
        Mirrors :meth:`BatchProcessFrame.get_input_files`.
        """
        from mufasa.utils.read_write import get_video_meta_data
        if not self.input_dir or not os.path.isdir(self.input_dir):
            return
        VIDEO_EXTS = (".mp4", ".avi", ".mov", ".flv", ".m4v")
        files = sorted(
            f for f in os.listdir(self.input_dir)
            if not f.startswith(".")
            and os.path.splitext(f.lower())[1] in VIDEO_EXTS
            and os.path.isfile(os.path.join(self.input_dir, f))
        )
        self.videos_in_dir_dict.clear()
        self.crop_dict.clear()
        for fname in files:
            full = os.path.join(self.input_dir, fname)
            stem = os.path.splitext(fname)[0]
            try:
                meta = get_video_meta_data(video_path=full)
            except Exception as exc:
                # Skip un-probeable files (corrupt headers, etc.) —
                # don't crash the table build over one bad video.
                print(f"[batch-preproc] skipped {fname}: {exc}")
                continue
            self.videos_in_dir_dict[stem] = {
                "file_path":   full,
                "video_length": meta.get("video_length_s", "00:00:00"),
                "fps":         meta.get("fps", 30.0),
                "width":       meta.get("width", 640),
                "height":      meta.get("height", 480),
            }
        self._populate_table()

    def _populate_table(self) -> None:
        self.table.setRowCount(0)
        for stem, data in self.videos_in_dir_dict.items():
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(
                row, COL_VIDEO, QTableWidgetItem(stem),
            )
            crop_btn = QPushButton("CROP", self)
            crop_btn.clicked.connect(
                lambda _, r=row, s=stem: self._on_crop_row(r, s),
            )
            self.table.setCellWidget(row, COL_CROP, crop_btn)
            self.table.setItem(
                row, COL_START, QTableWidgetItem("00:00:00"),
            )
            self.table.setItem(
                row, COL_END, QTableWidgetItem(str(data["video_length"])),
            )
            self.table.setCellWidget(
                row, COL_CLIP_CB, _centered_checkbox(self),
            )
            self.table.setItem(
                row, COL_WIDTH, QTableWidgetItem(str(data["width"])),
            )
            self.table.setItem(
                row, COL_HEIGHT, QTableWidgetItem(str(data["height"])),
            )
            self.table.setCellWidget(
                row, COL_DS_CB, _centered_checkbox(self),
            )
            self.table.setItem(
                row, COL_FPS,
                QTableWidgetItem(f"{round(float(data['fps']), 4)}"),
            )
            self.table.setCellWidget(
                row, COL_FPS_CB, _centered_checkbox(self),
            )
            self.table.setCellWidget(
                row, COL_GREY_CB, _centered_checkbox(self),
            )
            self.table.setCellWidget(
                row, COL_FRMCNT_CB, _centered_checkbox(self),
            )
            self.table.setCellWidget(
                row, COL_CLAHE_CB, _centered_checkbox(self),
            )
            qcombo = QComboBox(self)
            qcombo.addItems(_QUALITY_VALUES)
            qcombo.setCurrentText("60")
            self.table.setCellWidget(row, COL_QUALITY, qcombo)

    # ----------------------------------------------------------- Quick-apply
    def _apply_clip_to_all(self) -> None:
        start = self.q_clip_start.text().strip()
        end = self.q_clip_end.text().strip()
        from mufasa.utils.checks import (
            check_if_string_value_is_valid_video_timestamp,
            check_that_hhmmss_start_is_before_end,
        )
        try:
            check_if_string_value_is_valid_video_timestamp(
                value=start, name="Quick-apply clip START",
            )
            check_if_string_value_is_valid_video_timestamp(
                value=end, name="Quick-apply clip END",
            )
            check_that_hhmmss_start_is_before_end(
                start_time=start, end_time=end,
                name="Quick-apply clip START/END",
            )
        except Exception as exc:
            QMessageBox.warning(
                self, "Invalid clip times", str(exc),
            )
            return
        for row in range(self.table.rowCount()):
            self.table.item(row, COL_START).setText(start)
            self.table.item(row, COL_END).setText(end)

    def _apply_downsample_to_all(self) -> None:
        try:
            w = int(self.q_ds_width.text())
            h = int(self.q_ds_height.text())
            if w <= 0 or h <= 0:
                raise ValueError("Width and height must be positive.")
        except ValueError as exc:
            QMessageBox.warning(
                self, "Invalid downsample dims", str(exc),
            )
            return
        for row in range(self.table.rowCount()):
            self.table.item(row, COL_WIDTH).setText(str(w))
            self.table.item(row, COL_HEIGHT).setText(str(h))

    def _apply_fps_to_all(self) -> None:
        try:
            fps = float(self.q_fps.text())
            if fps <= 0:
                raise ValueError("FPS must be positive.")
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid FPS", str(exc))
            return
        for row in range(self.table.rowCount()):
            self.table.item(row, COL_FPS).setText(str(fps))

    def _apply_quality_to_all(self) -> None:
        quality = self.q_quality.currentText()
        for row in range(self.table.rowCount()):
            widget = self.table.cellWidget(row, COL_QUALITY)
            if isinstance(widget, QComboBox):
                widget.setCurrentText(quality)

    # ----------------------------------------------------------- Per-row
    def _on_crop_row(self, row: int, stem: str) -> None:
        """Open an OpenCV ROISelector for one video. Reuses the
        same ROISelector the Tk widget called — that's a native
        OpenCV dialog, host-agnostic."""
        try:
            from mufasa.video_processors.roi_selector import ROISelector
            from mufasa.utils.lookups import get_color_dict
        except Exception as exc:
            QMessageBox.critical(
                self, "Crop unavailable",
                f"Could not import ROISelector: {exc}",
            )
            return
        data = self.videos_in_dir_dict.get(stem)
        if data is None:
            return
        clrs = get_color_dict()
        clr = clrs.get(self.settings["crop_color"], (255, 105, 180))
        try:
            sel = ROISelector(
                path=data["file_path"],
                title=f"CROP {stem} — press ESC to commit",
                clr=clr,
                thickness=self.settings["crop_thickness"],
            )
            sel.run()
        except Exception as exc:
            QMessageBox.critical(self, "Crop failed", str(exc))
            return
        self.crop_dict[stem] = {
            "top_left_x":     sel.top_left[0],
            "top_left_y":     sel.top_left[1],
            "width":          sel.width,
            "height":         sel.height,
            "bottom_right_x": sel.bottom_right[0],
            "bottom_right_y": sel.bottom_right[1],
        }
        # Visual feedback — change button label / colour.
        btn = self.table.cellWidget(row, COL_CROP)
        if isinstance(btn, QPushButton):
            btn.setText("CROPPED ✓")
            btn.setStyleSheet(
                "color: white; background-color: #2e7d32; font-weight: bold;"
            )

    # ----------------------------------------------------------- Reset
    def _on_reset_all(self) -> None:
        self._populate_table()
        self.crop_dict.clear()

    def _on_reset_crop(self) -> None:
        self.crop_dict.clear()
        for row in range(self.table.rowCount()):
            btn = self.table.cellWidget(row, COL_CROP)
            if isinstance(btn, QPushButton):
                btn.setText("CROP")
                btn.setStyleSheet("")

    # ----------------------------------------------------------- Pop-out
    def _toggle_pop_out(self) -> None:
        """Re-parent the form between the inline section and a
        floating QDockWidget. Same pattern as 122aj's frame
        labeller dock."""
        if self._docked_widget is None:
            # Move into a floating dock
            main_window = self._find_main_window()
            if main_window is None:
                QMessageBox.information(
                    self, "Pop out",
                    "No main workbench window available; the "
                    "form must stay inline.",
                )
                return
            dock = QDockWidget("Preprocess Videos", main_window)
            dock.setAllowedAreas(Qt.AllDockWidgetAreas)
            dock.setFeatures(
                QDockWidget.DockWidgetMovable
                | QDockWidget.DockWidgetFloatable
                | QDockWidget.DockWidgetClosable
            )
            self.setParent(dock)
            dock.setWidget(self)
            main_window.addDockWidget(Qt.RightDockWidgetArea, dock)
            dock.setFloating(True)
            dock.show()
            self._docked_widget = dock
            self.pop_out_btn.setText("Re-dock ⇲")
            # Keep a ref on the main window so the dock isn't GC'd
            store = getattr(main_window, "_batch_preproc_docks", [])
            store.append(dock)
            main_window._batch_preproc_docks = store
        else:
            # Send the form back to its original parent
            dock = self._docked_widget
            self._docked_widget = None
            # The original parent is the section's body — Qt
            # automatically restores layout on setParent + show.
            # Best-effort: rely on the section's layout host.
            section_host = getattr(self, "_section_host", None)
            if section_host is not None:
                self.setParent(section_host)
                section_host.layout().addWidget(self)
            dock.setWidget(None)
            dock.close()
            self.pop_out_btn.setText("Pop out ⇱")

    def _find_main_window(self) -> Optional[QWidget]:
        from PySide6.QtWidgets import QMainWindow
        w = self.parentWidget()
        while w is not None:
            if isinstance(w, QMainWindow):
                return w
            w = w.parentWidget()
        return None

    # ----------------------------------------------------------- Execute
    def collect_args(self) -> dict:
        """Build the same JSON structure :meth:`BatchProcessFrame.execute`
        does, validate it, and return the on-disk path of the
        emitted ``batch_process_log.json``."""
        if not self.input_dir:
            raise ValueError("Pick an input directory first.")
        if not self.output_dir:
            raise ValueError("Pick an output directory first.")
        if not self.videos_in_dir_dict:
            raise ValueError(
                "No videos found in the input directory.",
            )
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        out_video_dict = {
            "meta_data": {
                "in_dir":  self.input_dir,
                "out_dir": self.output_dir,
                "gpu":     self.q_gpu.currentText() == "TRUE",
            },
            "video_data": {},
        }
        for row in range(self.table.rowCount()):
            stem = self.table.item(row, COL_VIDEO).text()
            data = self.videos_in_dir_dict.get(stem)
            if data is None:
                continue
            quality_combo = self.table.cellWidget(row, COL_QUALITY)
            quality_pct = (
                quality_combo.currentText()
                if isinstance(quality_combo, QComboBox)
                else "60"
            )
            # Quality % → CRF — use the lookup the Tk widget did.
            try:
                from mufasa.utils.lookups import percent_to_crf_lookup
                crf = percent_to_crf_lookup()[quality_pct]
            except Exception:
                crf = 23  # libx264 default
            entry: dict = {
                "video_info":     data,
                "output_quality": crf,
                "last_operation": None,
            }
            entry["crop"] = stem in self.crop_dict
            entry["crop_settings"] = (
                self.crop_dict[stem] if entry["crop"] else None
            )
            # Toggle pairs
            for op, col, kind in (
                ("clip",       COL_CLIP_CB,   "clip"),
                ("downsample", COL_DS_CB,     "downsample"),
                ("fps",        COL_FPS_CB,    "fps"),
                ("grayscale",  COL_GREY_CB,   "noargs"),
                ("frame_cnt",  COL_FRMCNT_CB, "noargs"),
                ("clahe",      COL_CLAHE_CB,  "noargs"),
            ):
                cb_widget = self.table.cellWidget(row, col)
                checked = (
                    isinstance(cb_widget, QWidget)
                    and _checkbox_is_checked(cb_widget)
                )
                entry[op] = checked
                if not checked:
                    entry[f"{op}_settings"] = None
                    continue
                if kind == "clip":
                    entry[f"{op}_settings"] = {
                        "start": self.table.item(row, COL_START).text(),
                        "stop":  self.table.item(row, COL_END).text(),
                    }
                elif kind == "downsample":
                    w = int(self.table.item(row, COL_WIDTH).text())
                    h = int(self.table.item(row, COL_HEIGHT).text())
                    # FFMPEG needs even dims for most codecs.
                    w += w % 2
                    h += h % 2
                    entry[f"{op}_settings"] = {
                        "width": str(w), "height": str(h),
                    }
                elif kind == "fps":
                    entry[f"{op}_settings"] = {
                        "fps": self.table.item(row, COL_FPS).text(),
                    }
                else:
                    entry[f"{op}_settings"] = None
            for op in _OPS_IN_EXEC_ORDER:
                if entry.get(op):
                    entry["last_operation"] = op
            out_video_dict["video_data"][stem] = entry

        save_path = os.path.join(
            self.output_dir, "batch_process_log.json",
        )
        with open(save_path, "w") as fp:
            json.dump(out_video_dict, fp)
        return {
            "save_path": save_path,
            "codec":     self.settings["codec"],
        }

    def target(self, *, save_path: str, codec: str) -> None:
        """Drive the FFMPEGCommandCreator pipeline. Runs in a worker
        thread (handled by OperationForm.on_run)."""
        from mufasa.video_processors.batch_process_create_ffmpeg_commands \
            import FFMPEGCommandCreator
        runner = FFMPEGCommandCreator(json_path=save_path, codec=codec)
        # The Tk widget calls each op explicitly so that exceptions
        # surface per-op. Mirror that.
        runner.crop_videos()
        runner.clip_videos()
        runner.downsample_videos()
        runner.apply_fps()
        runner.apply_grayscale()
        if hasattr(runner, "apply_clahe"):
            runner.apply_clahe()
        if hasattr(runner, "apply_frame_count_overlay"):
            runner.apply_frame_count_overlay()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _centered_checkbox(parent: QWidget) -> QWidget:
    """Wrap a QCheckBox in a centering layout for table cells."""
    host = QWidget(parent)
    layout = QHBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    cb = QCheckBox(host)
    cb.setObjectName("table_cb")
    layout.addStretch()
    layout.addWidget(cb)
    layout.addStretch()
    return host


def _checkbox_is_checked(host: QWidget) -> bool:
    cb = host.findChild(QCheckBox, "table_cb")
    return bool(cb is not None and cb.isChecked())


__all__ = ["BatchPreProcessForm", "SETTINGS"]
