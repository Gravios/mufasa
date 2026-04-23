"""
mufasa.ui_qt.targeted_clips
===========================

Interactive multi-row editor for targeted annotation clips. Replaces
the legacy ``TargetedAnnotationsWClipsPopUp`` — a Tk window that
showed a fixed-width table of ``HH:MM:SS`` start/end fields and
required the user to type timestamps verbatim.

The Qt version wraps the existing :class:`FrameScrubberWidget` with
a :class:`QTableWidget` of clip rows. New rows are added by clicking
"Mark start" at the current scrubber position, then "Mark end" once
the scrubber is at the end of the clip. Each row's timestamps remain
editable if the user wants to type exact values.

Workflow
--------

1. Pick a video (the project's video_dir is searched first for a
   matching name).
2. Scrub to the desired start frame → **Mark start** creates a new
   row and stores the frame number + timestamp.
3. Scrub to the end frame → **Mark end** completes the row.
4. Repeat for as many clips as needed; rows can be reordered or
   deleted.
5. **Run** extracts both video clips and the matching slices of the
   machine_results CSV into ``input_frames_dir/advanced_clip_annotator/{video}/``.

Persistence
-----------

Clip definitions are saved in the target dir alongside the outputs
as ``clips.json`` so a partial session can be resumed — the dialog
loads any existing ``clips.json`` on open.
"""
from __future__ import annotations

import configparser
import json
import os
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QDialog, QDialogButtonBox,
                               QFileDialog, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QMessageBox, QPushButton,
                               QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

from mufasa.ui_qt.frame_scrubber import FrameScrubberWidget


# --------------------------------------------------------------------------- #
# Time formatting helpers
# --------------------------------------------------------------------------- #
def _frame_to_hhmmss(frame_idx: int, fps: float) -> str:
    """``5123, 30.0`` → ``"00:02:50.767"``."""
    total_s = frame_idx / max(fps, 1.0)
    hh = int(total_s // 3600)
    mm = int((total_s % 3600) // 60)
    ss = total_s % 60
    return f"{hh:02d}:{mm:02d}:{ss:06.3f}"


def _hhmmss_to_frame(s: str, fps: float) -> int:
    """``"00:02:50.767", 30.0`` → ``5123``.

    Accepts ``HH:MM:SS`` or ``HH:MM:SS.mmm``. Raises ``ValueError`` on
    malformed input.
    """
    parts = s.strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"Expected HH:MM:SS format, got {s!r}")
    hh = int(parts[0]); mm = int(parts[1])
    ss = float(parts[2])
    if hh < 0 or mm < 0 or mm >= 60 or ss < 0 or ss >= 60:
        raise ValueError(f"Invalid time components in {s!r}")
    total_s = hh * 3600 + mm * 60 + ss
    return int(round(total_s * fps))


# --------------------------------------------------------------------------- #
# Dialog
# --------------------------------------------------------------------------- #
class TargetedClipsDialog(QDialog):
    """Multi-row clip-range editor with scrubber-driven row creation."""

    _COLS = ["#", "Start frame", "End frame", "Start HH:MM:SS",
             "End HH:MM:SS", "Length (s)"]

    def __init__(self,
                 config_path: str,
                 video_path: str,
                 *,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.config_path = config_path
        self.video_path = video_path
        self.video_name = Path(video_path).stem
        self._pending_start: Optional[int] = None
        self._dirty: bool = False

        self.setWindowTitle(f"Clip ranges — {self.video_name}")
        self.resize(1100, 760)
        self._load_project_metadata()
        self._build_ui()
        self.scrubber.load(video_path)
        self._load_existing_clips()

    # ------------------------------------------------------------------ #
    # Project metadata
    # ------------------------------------------------------------------ #
    def _load_project_metadata(self) -> None:
        cfg = configparser.ConfigParser()
        cfg.read(self.config_path)
        project_path = cfg.get("General settings", "project_path")
        self.file_type = cfg.get("General settings", "workflow_file_type",
                                 fallback="csv")
        # Staging dir mirrors the legacy popup's behaviour
        self.target_dir = os.path.join(
            project_path, "frames", "input", "advanced_clip_annotator",
            self.video_name,
        )
        os.makedirs(self.target_dir, exist_ok=True)
        self.clips_json = os.path.join(self.target_dir, "clips.json")
        self.machine_results_dir = os.path.join(
            project_path, "csv", "machine_results",
        )

    def _load_existing_clips(self) -> None:
        if not os.path.isfile(self.clips_json):
            return
        try:
            with open(self.clips_json) as f:
                data = json.load(f)
        except Exception:
            return
        for entry in data.get("clips", []):
            self._append_row(entry["start_frame"], entry["end_frame"])
        self._dirty = False
        self.status.setText(
            f"Loaded {self.table.rowCount()} existing clip(s) from {self.clips_json}."
        )

    # ------------------------------------------------------------------ #
    # UI
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)

        self.scrubber = FrameScrubberWidget(self)
        outer.addWidget(self.scrubber, 1)

        # Mark start / end buttons
        mark_row = QHBoxLayout()
        self.mark_start_btn = QPushButton("\u2315 Mark start", self)
        self.mark_end_btn = QPushButton("Mark end \u2315", self)
        self.mark_end_btn.setEnabled(False)
        self.cancel_pending_btn = QPushButton("Cancel pending", self)
        self.cancel_pending_btn.setEnabled(False)
        self.mark_start_btn.clicked.connect(self._mark_start)
        self.mark_end_btn.clicked.connect(self._mark_end)
        self.cancel_pending_btn.clicked.connect(self._cancel_pending)
        mark_row.addWidget(self.mark_start_btn)
        mark_row.addWidget(self.mark_end_btn)
        mark_row.addWidget(self.cancel_pending_btn)
        mark_row.addStretch()
        self.pending_lbl = QLabel("", self)
        self.pending_lbl.setStyleSheet("color: #a86400;")
        mark_row.addWidget(self.pending_lbl)
        outer.addLayout(mark_row)

        # Table of clips
        self.table = QTableWidget(0, len(self._COLS), self)
        self.table.setHorizontalHeaderLabels(self._COLS)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents,
        )
        self.table.horizontalHeader().setSectionResizeMode(
            len(self._COLS) - 1, QHeaderView.Stretch,
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.cellChanged.connect(self._on_cell_changed)
        outer.addWidget(self.table, 2)

        # Row controls
        row_btns = QHBoxLayout()
        self.goto_start_btn = QPushButton("Scrub to start of selected", self)
        self.goto_end_btn = QPushButton("Scrub to end of selected", self)
        self.delete_btn = QPushButton("Delete selected", self)
        self.clear_btn = QPushButton("Clear all", self)
        self.goto_start_btn.clicked.connect(lambda: self._scrub_to_selected("start"))
        self.goto_end_btn.clicked.connect(lambda: self._scrub_to_selected("end"))
        self.delete_btn.clicked.connect(self._delete_selected)
        self.clear_btn.clicked.connect(self._clear_all)
        row_btns.addWidget(self.goto_start_btn)
        row_btns.addWidget(self.goto_end_btn)
        row_btns.addWidget(self.delete_btn)
        row_btns.addWidget(self.clear_btn)
        row_btns.addStretch()
        outer.addLayout(row_btns)

        self.status = QLabel("Ready. Scrub to a start frame and click "
                             "<b>Mark start</b>.", self)
        outer.addWidget(self.status)

        # Save / Run / Close
        btns = QDialogButtonBox(self)
        self.save_btn = btns.addButton("Save list", QDialogButtonBox.ApplyRole)
        self.save_btn.clicked.connect(self._save_clips_json)
        self.run_btn = btns.addButton("Run (extract clips)",
                                      QDialogButtonBox.AcceptRole)
        self.run_btn.clicked.connect(self._run)
        close_btn = btns.addButton("Close", QDialogButtonBox.RejectRole)
        close_btn.clicked.connect(self.reject)
        outer.addWidget(btns)

    # ------------------------------------------------------------------ #
    # Mark-start / mark-end state machine
    # ------------------------------------------------------------------ #
    def _mark_start(self) -> None:
        self._pending_start = self.scrubber.current_frame
        self.mark_end_btn.setEnabled(True)
        self.cancel_pending_btn.setEnabled(True)
        self.pending_lbl.setText(
            f"Pending clip starting at frame {self._pending_start} "
            f"({_frame_to_hhmmss(self._pending_start, self.scrubber.fps)})"
        )

    def _mark_end(self) -> None:
        end_frame = self.scrubber.current_frame
        if self._pending_start is None:
            return
        if end_frame <= self._pending_start:
            QMessageBox.warning(
                self, "Invalid end",
                f"End frame ({end_frame}) must be greater than start "
                f"frame ({self._pending_start}).",
            )
            return
        self._append_row(self._pending_start, end_frame)
        self._cancel_pending()
        self._dirty = True

    def _cancel_pending(self) -> None:
        self._pending_start = None
        self.mark_end_btn.setEnabled(False)
        self.cancel_pending_btn.setEnabled(False)
        self.pending_lbl.setText("")

    # ------------------------------------------------------------------ #
    # Row manipulation
    # ------------------------------------------------------------------ #
    def _append_row(self, start_frame: int, end_frame: int) -> None:
        row = self.table.rowCount()
        self.table.blockSignals(True)
        self.table.insertRow(row)
        fps = self.scrubber.fps
        length_s = (end_frame - start_frame) / max(fps, 1.0)
        values = [
            f"{row + 1}",
            str(start_frame), str(end_frame),
            _frame_to_hhmmss(start_frame, fps),
            _frame_to_hhmmss(end_frame, fps),
            f"{length_s:.3f}",
        ]
        for col, text in enumerate(values):
            it = QTableWidgetItem(text)
            # Clip# and Length columns are read-only; others are editable
            if col in (0, 5):
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, col, it)
        self.table.blockSignals(False)

    def _delete_selected(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedItems()},
                      reverse=True)
        for r in rows:
            self.table.removeRow(r)
        self._renumber_rows()
        self._dirty = True

    def _clear_all(self) -> None:
        if self.table.rowCount() == 0:
            return
        reply = QMessageBox.question(
            self, "Clear all",
            f"Remove all {self.table.rowCount()} clip row(s)?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self.table.setRowCount(0)
        self._dirty = True

    def _renumber_rows(self) -> None:
        self.table.blockSignals(True)
        for r in range(self.table.rowCount()):
            self.table.item(r, 0).setText(f"{r + 1}")
        self.table.blockSignals(False)

    def _scrub_to_selected(self, which: str) -> None:
        rows = sorted({i.row() for i in self.table.selectedItems()})
        if not rows:
            return
        col = 1 if which == "start" else 2
        try:
            frame = int(self.table.item(rows[0], col).text())
        except (ValueError, AttributeError):
            return
        self.scrubber.seek(frame)

    # ------------------------------------------------------------------ #
    # Manual cell-edit validation (HH:MM:SS + frame columns)
    # ------------------------------------------------------------------ #
    def _on_cell_changed(self, row: int, col: int) -> None:
        """Keep the HH:MM:SS and frame-number columns in sync when the
        user edits either."""
        if col not in (1, 2, 3, 4):
            return
        self._dirty = True
        fps = self.scrubber.fps
        try:
            if col in (1, 2):  # frame changed; update HH:MM:SS + length
                frame = int(self.table.item(row, col).text())
                self.table.blockSignals(True)
                self.table.item(row, col + 2).setText(
                    _frame_to_hhmmss(frame, fps)
                )
                self._recompute_length(row)
                self.table.blockSignals(False)
            else:                # HH:MM:SS changed; update frame + length
                ts = self.table.item(row, col).text()
                frame = _hhmmss_to_frame(ts, fps)
                self.table.blockSignals(True)
                self.table.item(row, col - 2).setText(str(frame))
                self._recompute_length(row)
                self.table.blockSignals(False)
        except (ValueError, AttributeError):
            # Keep editing open — user will correct on their own. The
            # run-button validates before it tries to extract.
            pass

    def _recompute_length(self, row: int) -> None:
        try:
            start_f = int(self.table.item(row, 1).text())
            end_f = int(self.table.item(row, 2).text())
            length_s = (end_f - start_f) / max(self.scrubber.fps, 1.0)
            self.table.item(row, 5).setText(f"{length_s:.3f}")
        except (ValueError, AttributeError):
            pass

    # ------------------------------------------------------------------ #
    # Save list (json) + Run (extract clips)
    # ------------------------------------------------------------------ #
    def _collect_clips(self) -> list[dict]:
        """Return the validated clip list as [{start_frame, end_frame}, …].
        Raises ValueError on invalid rows."""
        clips = []
        total = self.scrubber.total_frames
        for r in range(self.table.rowCount()):
            try:
                s = int(self.table.item(r, 1).text())
                e = int(self.table.item(r, 2).text())
            except (ValueError, AttributeError):
                raise ValueError(f"Row {r + 1}: start/end must be integers.")
            if s < 0 or e < 0:
                raise ValueError(f"Row {r + 1}: negative frame.")
            if s >= total or e >= total:
                raise ValueError(
                    f"Row {r + 1}: frame(s) exceed video length ({total})."
                )
            if e <= s:
                raise ValueError(
                    f"Row {r + 1}: end ({e}) must be greater than start ({s})."
                )
            clips.append({"start_frame": s, "end_frame": e})
        return clips

    def _save_clips_json(self) -> None:
        try:
            clips = self._collect_clips()
        except ValueError as exc:
            QMessageBox.critical(self, "Invalid clips", str(exc))
            return
        try:
            with open(self.clips_json, "w") as f:
                json.dump({"video": self.video_name,
                           "fps": self.scrubber.fps,
                           "total_frames": self.scrubber.total_frames,
                           "clips": clips},
                          f, indent=2)
        except OSError as exc:
            QMessageBox.critical(self, "Save failed",
                                 f"Could not write {self.clips_json}: {exc}")
            return
        self._dirty = False
        self.status.setText(
            f"Saved {len(clips)} clip(s) → {self.clips_json}"
        )

    def _run(self) -> None:
        """Extract video + data slices for each clip."""
        try:
            clips = self._collect_clips()
        except ValueError as exc:
            QMessageBox.critical(self, "Invalid clips", str(exc))
            return
        if not clips:
            QMessageBox.information(
                self, "Nothing to run",
                "Add at least one clip row before running.",
            )
            return
        # Persist JSON first so a crash mid-run doesn't lose the list
        try:
            self._save_clips_json()
        except Exception:
            pass
        # Extract
        try:
            self._extract_clips(clips)
            self.status.setText(
                f"Extracted {len(clips)} clip(s) + data slices → "
                f"{self.target_dir}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Extraction failed", str(exc))

    def _extract_clips(self, clips: list[dict]) -> None:
        """Call the backend video splitter + slice machine_results."""
        fps = self.scrubber.fps
        start_times = [_frame_to_hhmmss(c["start_frame"], fps) for c in clips]
        end_times = [_frame_to_hhmmss(c["end_frame"], fps) for c in clips]
        # Video extraction — use the same backend the legacy popup used
        from mufasa.video_processors.video_processing import multi_split_video
        multi_split_video(
            file_path=self.video_path,
            start_times=start_times,
            end_times=end_times,
            out_dir=self.target_dir,
            include_clip_time_in_filename=True,
        )
        # Data slicing — read machine_results CSV and slice by frame range
        mr_path = os.path.join(
            self.machine_results_dir,
            f"{self.video_name}.{self.file_type}",
        )
        if not os.path.isfile(mr_path):
            # Non-fatal: a user may want video-only clips
            return
        try:
            import pandas as pd
            try:
                from mufasa.utils.read_write import read_df
                df = read_df(mr_path, self.file_type)
            except Exception:
                if self.file_type.lower() == "csv":
                    df = pd.read_csv(mr_path)
                else:
                    raise
        except Exception:
            return
        for i, c in enumerate(clips):
            s, e = c["start_frame"], c["end_frame"]
            if e > len(df):
                e = len(df)
            sliced = df.iloc[s:e + 1, :]
            start_txt = start_times[i].replace(":", "-")
            end_txt = end_times[i].replace(":", "-")
            save_path = os.path.join(
                self.target_dir,
                f"{self.video_name}_{start_txt}_{end_txt}.{self.file_type}",
            )
            if self.file_type.lower() == "csv":
                sliced.to_csv(save_path, index=False)
            else:
                try:
                    from mufasa.utils.read_write import write_df
                    write_df(df=sliced, file_type=self.file_type,
                             save_path=save_path)
                except Exception:
                    sliced.to_csv(save_path + ".csv", index=False)

    # ------------------------------------------------------------------ #
    # Close
    # ------------------------------------------------------------------ #
    def reject(self) -> None:
        if self._dirty and self.table.rowCount() > 0:
            reply = QMessageBox.question(
                self, "Unsaved changes",
                "You have unsaved clip definitions. Save list before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Cancel:
                return
            if reply == QMessageBox.Save:
                self._save_clips_json()
        self.scrubber.close_video()
        super().reject()


# --------------------------------------------------------------------------- #
# Launcher
# --------------------------------------------------------------------------- #
def launch_targeted_clips(parent: QWidget, config_path: str) -> None:
    if not config_path:
        QMessageBox.warning(
            parent, "No project",
            "Load a project before defining annotation clips.",
        )
        return
    video_path, _ = QFileDialog.getOpenFileName(
        parent, "Select video to annotate", "",
        "Videos (*.mp4 *.avi *.mov *.mkv *.webm)",
    )
    if not video_path:
        return
    try:
        dlg = TargetedClipsDialog(
            config_path=config_path, video_path=video_path, parent=parent,
        )
    except Exception as exc:
        QMessageBox.critical(parent, "Could not open clip editor", str(exc))
        return
    if not hasattr(parent, "_active_targeted_clips_dialogs"):
        parent._active_targeted_clips_dialogs = []
    parent._active_targeted_clips_dialogs.append(dlg)
    dlg.show()


__all__ = [
    "TargetedClipsDialog",
    "launch_targeted_clips",
    "_frame_to_hhmmss",
    "_hhmmss_to_frame",
]
