"""
mufasa.ui_qt.clip_review
========================

Interactive bout-by-bout review of classifier output (or existing
annotations). Built on top of :class:`FrameScrubberWidget` — same
random-access seek mechanism, plus a :class:`GanttStrip` that shows
where bouts are in the video and a :class:`BoutList` that lets the
user step through them and mark each one valid / invalid / unsure.

This dialog is **new functionality** rather than a 1:1 port of a
legacy popup: the existing :class:`ClassifierValidationClipsPopUp`
only batch-renders clips to disk. Reviewing the output is much
faster when it stays interactive — no disk round-trips, no waiting
for video encodes.

The :class:`ClassifierValidationLauncher` form on the Annotation page
invokes this dialog.

Review output
-------------

Per-bout ratings are saved to
``{project}/csv/validation_results/{video_name}.csv`` with columns:

* ``bout_id`` (int, 1-based)
* ``start_frame``, ``end_frame`` (int)
* ``classifier`` (str)
* ``rating`` (one of ``valid`` / ``invalid`` / ``unsure`` / ``unreviewed``)
* ``notes`` (str, optional)

Existing rating files are loaded on dialog open so reviews can resume.
"""
from __future__ import annotations

import configparser
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDialog,
                               QDialogButtonBox, QFileDialog, QFormLayout,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QMessageBox, QPushButton, QSizePolicy,
                               QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

from mufasa.ui_qt.forms.analysis import _load_classifier_names
from mufasa.ui_qt.frame_scrubber import FrameScrubberWidget


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Bout:
    """A contiguous run of frames where a classifier predicted present."""
    bout_id: int
    classifier: str
    start_frame: int
    end_frame: int   # inclusive
    rating: str = "unreviewed"  # valid | invalid | unsure | unreviewed
    notes: str = ""

    @property
    def length_frames(self) -> int:
        return self.end_frame - self.start_frame + 1


def _detect_bouts(values: np.ndarray,
                  classifier: str,
                  threshold: float = 0.5,
                  min_bout_frames: int = 1) -> list[Bout]:
    """Scan a binary / probability column and emit a list of bouts.

    Accepts both 0/1 arrays (``values > 0``) and probability arrays
    (``values > threshold``). Bouts shorter than ``min_bout_frames``
    are dropped — matches the legacy shortest-bout filter used in
    classifier validation.
    """
    if values.ndim != 1:
        raise ValueError(f"Expected 1-D array, got shape {values.shape}")
    mask = values > threshold
    if not mask.any():
        return []
    # Edges: where mask flips 0→1 (starts) and 1→0 (ends)
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1  # inclusive
    bouts: list[Bout] = []
    for s, e in zip(starts, ends):
        if (e - s + 1) >= min_bout_frames:
            bouts.append(Bout(
                bout_id=len(bouts) + 1,
                classifier=classifier,
                start_frame=int(s),
                end_frame=int(e),
            ))
    return bouts


# --------------------------------------------------------------------------- #
# Gantt strip — visual bout timeline, clickable to seek
# --------------------------------------------------------------------------- #
class GanttStrip(QWidget):
    """Horizontal bar showing bout positions along the video timeline.

    Colour encodes rating: grey (unreviewed), green (valid),
    red (invalid), yellow (unsure). A vertical line marks the current
    playhead; clicking on the strip seeks to that frame.
    """

    frame_seek_requested = Signal(int)

    _RATING_COLORS = {
        "unreviewed": QColor(160, 160, 160),
        "valid":      QColor( 80, 180,  80),
        "invalid":    QColor(200,  80,  80),
        "unsure":     QColor(220, 200,  60),
    }

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._bouts: list[Bout] = []
        self._total_frames: int = 1
        self._current_frame: int = 0
        self.setMinimumHeight(30)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_state(self, bouts: list[Bout], total_frames: int) -> None:
        self._bouts = bouts
        self._total_frames = max(1, total_frames)
        self.update()

    def set_current_frame(self, idx: int) -> None:
        self._current_frame = idx
        self.update()

    def paintEvent(self, _ev) -> None:  # noqa: N802
        p = QPainter(self)
        w, h = self.width(), self.height()
        # Background track
        p.fillRect(0, 0, w, h, QColor(40, 40, 40))
        # Bout rectangles
        for b in self._bouts:
            x0 = int(w * b.start_frame / self._total_frames)
            x1 = int(w * (b.end_frame + 1) / self._total_frames)
            col = self._RATING_COLORS.get(b.rating, QColor(160, 160, 160))
            p.fillRect(QRect(x0, 2, max(1, x1 - x0), h - 4), col)
        # Playhead line
        playhead_x = int(w * self._current_frame / self._total_frames)
        p.setPen(QPen(QColor(255, 255, 255), 2))
        p.drawLine(playhead_x, 0, playhead_x, h)

    def mousePressEvent(self, ev) -> None:  # noqa: N802
        if self._total_frames <= 1:
            return
        frac = max(0.0, min(1.0, ev.position().x() / max(1, self.width())))
        self.frame_seek_requested.emit(int(frac * (self._total_frames - 1)))


# --------------------------------------------------------------------------- #
# Bout list (table) — rateable, sortable, navigable
# --------------------------------------------------------------------------- #
class BoutTable(QTableWidget):
    """Tabular bout list with in-row rating dropdown.

    Emits:

    * ``bout_selected(int)`` — the ``start_frame`` of the clicked row.
    * ``rating_changed(int, str)`` — a (``bout_id``, new rating) pair.
    """
    bout_selected = Signal(int)
    rating_changed = Signal(int, str)

    _COLS = ["Bout", "Start", "End", "Length (s)", "Rating", "Notes"]

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setColumnCount(len(self._COLS))
        self.setHorizontalHeaderLabels(self._COLS)
        self.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents,
        )
        self.horizontalHeader().setSectionResizeMode(
            len(self._COLS) - 1, QHeaderView.Stretch,
        )
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setAlternatingRowColors(True)
        self.cellClicked.connect(self._on_click)
        self._fps: float = 30.0
        self._bouts: list[Bout] = []

    def set_bouts(self, bouts: list[Bout], fps: float) -> None:
        self._bouts = bouts
        self._fps = max(fps, 1.0)
        self.setRowCount(len(bouts))
        for r, b in enumerate(bouts):
            self._write_row(r, b)

    def _write_row(self, r: int, b: Bout) -> None:
        length_s = b.length_frames / self._fps
        items = [
            f"#{b.bout_id}", str(b.start_frame), str(b.end_frame),
            f"{length_s:.2f}",
        ]
        for c, text in enumerate(items):
            it = QTableWidgetItem(text)
            it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            it.setData(Qt.UserRole, b.bout_id)
            self.setItem(r, c, it)
        # Rating: dropdown
        cb = QComboBox(self)
        cb.addItems(["unreviewed", "valid", "invalid", "unsure"])
        cb.setCurrentText(b.rating)
        cb.currentTextChanged.connect(
            lambda new_rating, bid=b.bout_id:
                self.rating_changed.emit(bid, new_rating)
        )
        self.setCellWidget(r, 4, cb)
        # Notes: free-text (editable)
        notes = QTableWidgetItem(b.notes)
        notes.setFlags(notes.flags() | Qt.ItemIsEditable)
        self.setItem(r, 5, notes)

    def _on_click(self, row: int, _col: int) -> None:
        if 0 <= row < len(self._bouts):
            self.bout_selected.emit(self._bouts[row].start_frame)

    def goto_bout_for_frame(self, frame_idx: int) -> None:
        """Highlight the row whose bout contains ``frame_idx``."""
        for r, b in enumerate(self._bouts):
            if b.start_frame <= frame_idx <= b.end_frame:
                self.selectRow(r)
                return

    def current_notes(self) -> dict[int, str]:
        """Collect per-bout notes from the editable text cells."""
        out: dict[int, str] = {}
        for r, b in enumerate(self._bouts):
            cell = self.item(r, 5)
            out[b.bout_id] = cell.text() if cell else ""
        return out


# --------------------------------------------------------------------------- #
# Main dialog
# --------------------------------------------------------------------------- #
class ClipReviewDialog(QDialog):
    """Bout-by-bout review dialog.

    The classifier dropdown picks which column of the machine_results
    CSV drives bout detection. Threshold slider re-derives bouts
    live; rating dropdowns + notes persist to the validation CSV on
    save.
    """

    def __init__(self,
                 config_path: str,
                 video_path: str,
                 machine_results_path: str,
                 *,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.config_path = config_path
        self.video_path = video_path
        self.machine_results_path = machine_results_path
        self.video_name = Path(video_path).stem
        self._bouts: list[Bout] = []
        self._classifier_names: list[str] = []
        self._mr_df = None   # pandas DataFrame, loaded lazily
        self._dirty: bool = False

        self.setWindowTitle(f"Review clips — {self.video_name}")
        self.resize(1200, 800)
        self._load_project_metadata()
        self._build_ui()
        self._load_machine_results()
        self.scrubber.load(video_path)
        self.scrubber.frame_changed.connect(self._on_frame_changed)
        self._load_existing_ratings()
        self._recompute_bouts()

    # ------------------------------------------------------------------ #
    # Project + data loading
    # ------------------------------------------------------------------ #
    def _load_project_metadata(self) -> None:
        self._classifier_names = _load_classifier_names(self.config_path)
        if not self._classifier_names:
            raise RuntimeError(
                "No classifiers defined in project_config.ini. Add at "
                "least one via the Classifier → Manage page."
            )
        cfg = configparser.ConfigParser()
        cfg.read(self.config_path)
        project_path = cfg.get("General settings", "project_path")
        self.file_type = cfg.get("General settings", "workflow_file_type",
                                 fallback="csv")
        self.validation_dir = os.path.join(project_path, "csv",
                                           "validation_results")
        os.makedirs(self.validation_dir, exist_ok=True)
        self.ratings_path = os.path.join(
            self.validation_dir, f"{self.video_name}.csv",
        )

    def _load_machine_results(self) -> None:
        """Load the machine_results CSV into ``self._mr_df``.

        Uses pandas directly for CSV to dodge an upstream quirk in
        ``mufasa.utils.read_write.read_df`` which unconditionally
        strips the first column of CSV input regardless of
        ``has_index``. For .parquet / .pickle the project reader is
        used since those formats handle index correctly.
        """
        try:
            import pandas as pd
            if self.file_type.lower() == "csv":
                self._mr_df = pd.read_csv(self.machine_results_path)
            else:
                from mufasa.utils.read_write import read_df
                self._mr_df = read_df(self.machine_results_path,
                                      self.file_type)
        except Exception as exc:
            raise RuntimeError(
                f"Could not read machine_results at "
                f"{self.machine_results_path}: {exc}"
            )
        # Trim the classifier list to those present in the dataframe
        present = [n for n in self._classifier_names
                   if n in self._mr_df.columns]
        if not present:
            raise RuntimeError(
                f"No project classifiers found as columns in "
                f"{self.machine_results_path}. Columns = "
                f"{list(self._mr_df.columns)[:8]}…"
            )
        self._classifier_names = present

    def _load_existing_ratings(self) -> None:
        """Populate an internal {bout_id: (rating, notes)} dict from
        an existing validation CSV."""
        self._existing_ratings: dict[tuple[str, int, int], tuple[str, str]] = {}
        if not os.path.isfile(self.ratings_path):
            return
        try:
            import pandas as pd
            df = pd.read_csv(self.ratings_path)
        except Exception:
            return
        for _, row in df.iterrows():
            key = (str(row.get("classifier", "")),
                   int(row.get("start_frame", -1)),
                   int(row.get("end_frame", -1)))
            self._existing_ratings[key] = (
                str(row.get("rating", "unreviewed")),
                str(row.get("notes", "")) if "notes" in df.columns else "",
            )

    # ------------------------------------------------------------------ #
    # Bout derivation
    # ------------------------------------------------------------------ #
    def _recompute_bouts(self) -> None:
        """Derive bouts for the currently-selected classifier."""
        clf = self.clf_cb.currentText()
        if not clf or self._mr_df is None or clf not in self._mr_df.columns:
            return
        values = self._mr_df[clf].to_numpy()
        threshold = self.threshold_cb.value()
        min_frames = self.min_bout_frames.value()
        bouts = _detect_bouts(values, classifier=clf,
                              threshold=threshold,
                              min_bout_frames=min_frames)
        # Re-attach any previously-saved ratings
        for b in bouts:
            key = (clf, b.start_frame, b.end_frame)
            if key in self._existing_ratings:
                b.rating, b.notes = self._existing_ratings[key]
        self._bouts = bouts
        self.bout_table.set_bouts(bouts, fps=self.scrubber.fps)
        self.gantt.set_state(bouts, total_frames=self.scrubber.total_frames)
        self.status.setText(
            f"Classifier {clf!r}: {len(bouts)} bouts "
            f"(threshold = {threshold:.2f}, min {min_frames} frames)."
        )

    # ------------------------------------------------------------------ #
    # UI
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        from PySide6.QtWidgets import QDoubleSpinBox, QSpinBox

        outer = QVBoxLayout(self)

        # Controls row (classifier + thresholds)
        ctrl_form = QFormLayout()
        self.clf_cb = QComboBox(self)
        self.clf_cb.addItems(self._classifier_names)
        self.clf_cb.currentIndexChanged.connect(lambda _: self._recompute_bouts())
        ctrl_form.addRow("Classifier:", self.clf_cb)

        self.threshold_cb = QDoubleSpinBox(self)
        self.threshold_cb.setRange(0.0, 1.0)
        self.threshold_cb.setSingleStep(0.05)
        self.threshold_cb.setValue(0.5)
        self.threshold_cb.valueChanged.connect(lambda _: self._recompute_bouts())
        ctrl_form.addRow("Probability threshold:", self.threshold_cb)

        self.min_bout_frames = QSpinBox(self)
        self.min_bout_frames.setRange(1, 10000)
        self.min_bout_frames.setValue(1)
        self.min_bout_frames.valueChanged.connect(lambda _: self._recompute_bouts())
        ctrl_form.addRow("Minimum bout length (frames):", self.min_bout_frames)
        outer.addLayout(ctrl_form)

        # Scrubber + gantt strip stacked
        self.scrubber = FrameScrubberWidget(self)
        outer.addWidget(self.scrubber, 1)
        self.gantt = GanttStrip(self)
        self.gantt.frame_seek_requested.connect(self.scrubber.seek)
        outer.addWidget(self.gantt)

        # Prev/next bout buttons
        nav = QHBoxLayout()
        self.prev_btn = QPushButton("\u25C0 Prev bout", self)
        self.next_btn = QPushButton("Next bout \u25B6", self)
        self.prev_btn.clicked.connect(lambda: self._step_bout(-1))
        self.next_btn.clicked.connect(lambda: self._step_bout(+1))
        nav.addWidget(self.prev_btn); nav.addWidget(self.next_btn)
        nav.addStretch()
        outer.addLayout(nav)

        # Bout table
        self.bout_table = BoutTable(self)
        self.bout_table.bout_selected.connect(self.scrubber.seek)
        self.bout_table.rating_changed.connect(self._on_rating_changed)
        outer.addWidget(self.bout_table, 2)

        # Status + save/close
        self.status = QLabel("Ready.", self)
        outer.addWidget(self.status)

        btns = QDialogButtonBox(self)
        self.save_btn = btns.addButton("Save ratings", QDialogButtonBox.ApplyRole)
        self.save_btn.clicked.connect(self._save)
        close_btn = btns.addButton("Close", QDialogButtonBox.RejectRole)
        close_btn.clicked.connect(self.reject)
        outer.addWidget(btns)

    # ------------------------------------------------------------------ #
    # Event handlers
    # ------------------------------------------------------------------ #
    def _on_frame_changed(self, idx: int) -> None:
        self.gantt.set_current_frame(idx)
        self.bout_table.goto_bout_for_frame(idx)

    def _step_bout(self, delta: int) -> None:
        if not self._bouts:
            return
        cur = self.scrubber.current_frame
        # Find current bout's index, or nearest
        candidate = None
        for i, b in enumerate(self._bouts):
            if b.start_frame <= cur <= b.end_frame:
                candidate = i; break
            if b.start_frame > cur:
                candidate = i - 1 if delta > 0 else i
                break
        if candidate is None:
            candidate = len(self._bouts) - 1 if delta > 0 else 0
        new_idx = candidate + delta
        new_idx = max(0, min(new_idx, len(self._bouts) - 1))
        self.scrubber.seek(self._bouts[new_idx].start_frame)

    def _on_rating_changed(self, bout_id: int, new_rating: str) -> None:
        for b in self._bouts:
            if b.bout_id == bout_id:
                b.rating = new_rating
                self._dirty = True
                self.setWindowTitle(f"Review clips — {self.video_name} *")
                self.gantt.update()   # colour the strip
                return

    def _save(self) -> None:
        """Write ratings to ``validation_results/{video_name}.csv``."""
        try:
            import pandas as pd
            # Sync notes from the table
            notes_by_id = self.bout_table.current_notes()
            for b in self._bouts:
                b.notes = notes_by_id.get(b.bout_id, "")

            # Merge with prior ratings from other classifiers — the
            # file is one CSV per video, all classifiers side by side
            all_rows: list[dict] = []
            clf_now = self.clf_cb.currentText()
            # Keep other classifiers' prior ratings intact
            for (clf, s, e), (rating, notes) in self._existing_ratings.items():
                if clf == clf_now:
                    continue  # replaced below
                all_rows.append({
                    "classifier": clf, "start_frame": s, "end_frame": e,
                    "bout_id": 0, "rating": rating, "notes": notes,
                })
            # Current classifier's ratings
            for b in self._bouts:
                all_rows.append({
                    "classifier": b.classifier,
                    "start_frame": b.start_frame,
                    "end_frame":   b.end_frame,
                    "bout_id":     b.bout_id,
                    "rating":      b.rating,
                    "notes":       b.notes,
                })
            df = pd.DataFrame(all_rows, columns=[
                "classifier", "bout_id", "start_frame", "end_frame",
                "rating", "notes",
            ])
            df.to_csv(self.ratings_path, index=False)
            # Refresh the existing-ratings cache so subsequent
            # classifier switches still see current ratings
            self._existing_ratings.update({
                (b.classifier, b.start_frame, b.end_frame):
                    (b.rating, b.notes)
                for b in self._bouts
            })
            self._dirty = False
            self.setWindowTitle(f"Review clips — {self.video_name}")
            self.status.setText(f"Saved {len(self._bouts)} ratings → {self.ratings_path}")
        except Exception as exc:
            QMessageBox.critical(
                self, "Save failed", f"Could not save ratings: {exc}",
            )

    def reject(self) -> None:
        if self._dirty:
            reply = QMessageBox.question(
                self, "Unsaved changes",
                "You have unsaved ratings. Save before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Cancel:
                return
            if reply == QMessageBox.Save:
                self._save()
        self.scrubber.close_video()
        super().reject()


# --------------------------------------------------------------------------- #
# Launcher helper
# --------------------------------------------------------------------------- #
def launch_clip_review(parent: QWidget, config_path: str) -> None:
    """Prompt for a video + machine_results CSV, then open the review dialog."""
    if not config_path:
        QMessageBox.warning(
            parent, "No project",
            "Load a project (project_config.ini) before reviewing clips.",
        )
        return
    video_path, _ = QFileDialog.getOpenFileName(
        parent, "Select video", "",
        "Videos (*.mp4 *.avi *.mov *.mkv *.webm)",
    )
    if not video_path:
        return
    # Try to auto-locate the machine_results CSV next to the project
    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    project_path = cfg.get("General settings", "project_path", fallback=None)
    file_type = cfg.get("General settings", "workflow_file_type",
                        fallback="csv")
    default_mr = None
    if project_path:
        auto = os.path.join(
            project_path, "csv", "machine_results",
            f"{Path(video_path).stem}.{file_type}",
        )
        if os.path.isfile(auto):
            default_mr = auto
    if default_mr is None:
        mr_path, _ = QFileDialog.getOpenFileName(
            parent, "Select machine_results CSV", "",
            "CSV (*.csv);;All files (*)",
        )
        if not mr_path:
            return
    else:
        mr_path = default_mr

    try:
        dlg = ClipReviewDialog(
            config_path=config_path,
            video_path=video_path,
            machine_results_path=mr_path,
            parent=parent,
        )
    except Exception as exc:
        QMessageBox.critical(parent, "Could not open review dialog", str(exc))
        return
    if not hasattr(parent, "_active_review_dialogs"):
        parent._active_review_dialogs = []
    parent._active_review_dialogs.append(dlg)
    dlg.show()


__all__ = ["ClipReviewDialog", "launch_clip_review",
           "Bout", "_detect_bouts"]
