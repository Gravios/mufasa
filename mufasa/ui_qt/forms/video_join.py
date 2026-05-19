"""
mufasa.ui_qt.forms.video_join
=============================

Inline forms for joining and transitioning between videos:

* :class:`JoinVideosForm` — N-video temporal (end-to-end),
  horizontal (side-by-side), vertical (stacked), or mosaic (grid)
  concatenation. Replaces:
  - :class:`ConcatenatingVideosPopUp` (2-video concat special case)
  - :class:`ConcatenatorPopUp` (N-video horizontal/vertical/mosaic)
  - :class:`VideoTemporalJoinPopUp` (automatic end-to-end)
  - :class:`ManualTemporalJoinPopUp` (manual ordering — collapses
    to "temporal mode + drag-reorder the list").

* :class:`CrossfadeVideosForm` — 2-video crossfade transition with
  configurable method (18 ffmpeg xfade modes), duration, and
  offset. Replaces :class:`CrossfadeVideosPopUp`. Patch 122t.

Mode selector drives which backend is called; the same file list
feeds each mode of the join form. The crossfade form is separate
because it has a strict 2-video constraint and a transition-style
parameter set (method / duration / offset) that doesn't fit
naturally as a fifth mode on the join form.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QFileDialog, QFormLayout,
                               QHBoxLayout, QLineEdit,
                               QListWidget, QListWidgetItem,
                               QPushButton, QSpinBox, QVBoxLayout,
                               QWidget)

from mufasa.ui_qt.workbench import OperationForm


# --------------------------------------------------------------------------- #
# Helper: multi-file picker
# --------------------------------------------------------------------------- #
class _VideoListPicker(QWidget):
    """Re-orderable list of video paths. Subsumes the
    2-video-special-case + N-video-list pattern.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)

        self.list = QListWidget(self)
        self.list.setDragDropMode(QListWidget.InternalMove)
        self.list.setMinimumHeight(100)
        outer.addWidget(self.list)

        btn_row = QHBoxLayout()
        self.add_btn = QPushButton("Add file(s)…", self)
        self.rem_btn = QPushButton("Remove selected", self)
        self.clear_btn = QPushButton("Clear", self)
        self.add_btn.clicked.connect(self._add)
        self.rem_btn.clicked.connect(self._remove_selected)
        self.clear_btn.clicked.connect(self.list.clear)
        btn_row.addWidget(self.add_btn)
        btn_row.addWidget(self.rem_btn)
        btn_row.addWidget(self.clear_btn)
        btn_row.addStretch()
        outer.addLayout(btn_row)

    def _add(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add videos", "",
            "Video files (*.mp4 *.avi *.mov *.mkv *.webm);;All files (*)",
        )
        for p in paths:
            self.list.addItem(QListWidgetItem(p))

    def _remove_selected(self) -> None:
        for item in self.list.selectedItems():
            self.list.takeItem(self.list.row(item))

    def paths(self) -> list[str]:
        return [self.list.item(i).text() for i in range(self.list.count())]


# --------------------------------------------------------------------------- #
# JoinVideosForm
# --------------------------------------------------------------------------- #
class JoinVideosForm(OperationForm):
    """Combine multiple videos using one of four join modes.

    Modes
    -----
    * **Temporal (end-to-end)** — classic concatenation in the order
      listed. Subsumes ``ConcatenatingVideosPopUp`` (2-video special
      case) and ``VideoTemporalJoinPopUp``.
    * **Horizontal (side-by-side)** — ``horizontal_video_concatenator``.
    * **Vertical (stacked)** — ``vertical_video_concatenator``.
    * **Mosaic (grid)** — ``mosaic_concatenator`` with configurable rows.

    Legacy ``ConcatenatorPopUp`` routed to horizontal/vertical/mosaic
    based on a dropdown — same here, but without a separate window.
    ``ManualTemporalJoinPopUp`` collapses to "temporal mode + re-order
    the list via drag-and-drop" (the list widget is reorderable).

    Patch 122t notes:

    * The previously-collected ``normalize`` flag was a stub — never
      passed to any backend. Removed.
    * Mosaic backend takes grid shape as (width_idx, height_idx) in
      tiles, not a single ``rows`` kwarg. UI ``rows`` maps to
      height_idx; width_idx is derived by packing the videos.
    """

    title = "Join videos"
    description = ("Combine two or more videos temporally (end-to-end), "
                   "horizontally, vertically, or as a mosaic grid. "
                   "Drag list entries to reorder.")

    MODES = [
        ("Temporal (end-to-end)", "temporal"),
        ("Horizontal (side-by-side)", "horizontal"),
        ("Vertical (stacked)",    "vertical"),
        ("Mosaic (grid)",         "mosaic"),
    ]

    def build(self) -> None:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.mode_cb = QComboBox(self)
        for label, _ in self.MODES:
            self.mode_cb.addItem(label)
        self.mode_cb.currentIndexChanged.connect(self._on_mode_changed)
        form.addRow("Join mode:", self.mode_cb)

        self.picker = _VideoListPicker(self)
        form.addRow("Videos:", self.picker)

        # Mosaic-only: rows count
        self.mosaic_rows = QSpinBox(self)
        self.mosaic_rows.setRange(1, 16); self.mosaic_rows.setValue(2)
        form.addRow("Mosaic rows:", self.mosaic_rows)
        self._mosaic_row_index = form.rowCount() - 1

        self.body_layout.addLayout(form)
        self._on_mode_changed(0)

    def _on_mode_changed(self, index: int) -> None:
        self.mosaic_rows.setVisible(index == 3)

    def collect_args(self) -> dict:
        paths = self.picker.paths()
        mode = self.MODES[self.mode_cb.currentIndex()][1]
        if mode == "temporal" and len(paths) < 2:
            raise ValueError("Temporal join needs at least 2 videos.")
        if mode in ("horizontal", "vertical") and len(paths) < 2:
            raise ValueError(f"{mode.title()} join needs at least 2 videos.")
        if mode == "mosaic" and len(paths) < 4:
            raise ValueError("Mosaic join needs at least 4 videos.")
        return {
            "paths": paths,
            "mode":  mode,
            "rows":  int(self.mosaic_rows.value()),
        }

    def target(self, *, paths: list[str], mode: str,
               rows: int) -> None:
        import math
        from mufasa.video_processors import video_processing as _vp

        # All four concatenators require explicit save_path (no default
        # in the backend). Derive next to the first input video.
        first = Path(paths[0])
        save_path = str(first.parent / f"{first.stem}_joined_{mode}.mp4")

        if mode == "temporal":
            _vp.temporal_concatenation(video_paths=paths, save_path=save_path)
        elif mode == "horizontal":
            _vp.horizontal_video_concatenator(video_paths=paths,
                                              save_path=save_path)
        elif mode == "vertical":
            _vp.vertical_video_concatenator(video_paths=paths,
                                            save_path=save_path)
        elif mode == "mosaic":
            # Mosaic backend takes grid shape as (width_idx, height_idx)
            # in tiles, not a `rows` kwarg. Map UI `rows` → height_idx;
            # derive width_idx by packing the videos.
            height_idx = max(1, int(rows))
            width_idx = max(1, math.ceil(len(paths) / height_idx))
            _vp.mosaic_concatenator(video_paths=paths, save_path=save_path,
                                    height_idx=height_idx,
                                    width_idx=width_idx)


# --------------------------------------------------------------------------- #
# CrossfadeVideosForm (patch 122t)
# --------------------------------------------------------------------------- #
class CrossfadeVideosForm(OperationForm):
    """Apply a 2-video crossfade transition.

    Replaces :class:`CrossfadeVideosPopUp`. Backend:
    :func:`mufasa.video_processors.video_processing.crossfade_two_videos`,
    which wraps ffmpeg's ``xfade`` filter — 18+ transition methods
    (fade, fadeblack, wipe*, smooth*, circlecrop, rectcrop, …).

    Strict 2-video constraint: this is a transition, not a
    concatenation, so it's intentionally NOT a fifth mode on
    :class:`JoinVideosForm`.

    Offset parameter accepts the legacy HH:MM:SS string form (matches
    the popup's behaviour and the backend's
    ``check_if_hhmmss_timestamp_is_valid_part_of_video`` validator).
    The backend type hint says ``int`` but accepts the string form.
    """

    title = "Crossfade two videos"
    description = (
        "Smoothly transition between two videos using one of 18+ "
        "ffmpeg xfade methods. Duration is the visible crossfade "
        "length; offset is where in video 1 the transition starts "
        "(HH:MM:SS)."
    )

    _OFFSET_RE = re.compile(r"^\d{2}:\d{2}:\d{2}$")

    def build(self) -> None:
        from mufasa.utils.lookups import get_ffmpeg_crossfade_methods

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        # Two file pickers — strict 2-video.
        self.video_1_edit = self._build_file_row(form, "Video 1:")
        self.video_2_edit = self._build_file_row(form, "Video 2:")

        # Crossfade method — pulled from lookups so we stay in sync
        # with the canonical list.
        self.method_cb = QComboBox(self)
        try:
            methods = list(get_ffmpeg_crossfade_methods())
        except Exception:
            # If lookup fails (test sandbox, etc.) fall back to a
            # minimal set rather than blocking form build.
            methods = ["fade", "fadeblack", "fadewhite", "wipeleft"]
        for m in methods:
            self.method_cb.addItem(m)
        # Default to 'fade' if present.
        if "fade" in methods:
            self.method_cb.setCurrentText("fade")
        form.addRow("Crossfade method:", self.method_cb)

        # Duration (seconds). Legacy popup offered 2..20 step 2;
        # widen to 1..60 with step 1 so unusual durations are reachable.
        self.duration = QSpinBox(self)
        self.duration.setRange(1, 60)
        self.duration.setValue(6)
        self.duration.setSuffix(" s")
        form.addRow("Crossfade duration:", self.duration)

        # Offset — HH:MM:SS string. The backend validates it against
        # the actual length of video 1, so the form doesn't pre-clamp.
        self.offset_edit = QLineEdit(self)
        self.offset_edit.setText("00:00:00")
        self.offset_edit.setPlaceholderText("HH:MM:SS")
        form.addRow("Crossfade offset:", self.offset_edit)

        # Output options
        self.quality = QSpinBox(self)
        self.quality.setRange(10, 100)
        self.quality.setSingleStep(10)
        self.quality.setValue(60)
        self.quality.setSuffix(" %")
        form.addRow("Output quality:", self.quality)

        self.format_cb = QComboBox(self)
        self.format_cb.addItems(["mp4", "avi", "webm"])
        form.addRow("Output format:", self.format_cb)

        self.body_layout.addLayout(form)

    def _build_file_row(self, form: QFormLayout,
                        label: str) -> QLineEdit:
        """Helper: file picker row with Browse button. Returns the
        QLineEdit so collect_args can read the value."""
        edit = QLineEdit(self)
        edit.setReadOnly(True)
        edit.setPlaceholderText("Pick a video file…")
        browse = QPushButton("Browse…", self)

        def _pick() -> None:
            path, _ = QFileDialog.getOpenFileName(
                self, f"Pick {label.rstrip(':')}", "",
                "Video files (*.mp4 *.avi *.mov *.mkv *.webm);;"
                "All files (*)",
            )
            if path:
                edit.setText(path)
        browse.clicked.connect(_pick)

        row = QHBoxLayout()
        row.addWidget(edit, 1)
        row.addWidget(browse)
        form.addRow(label, row)
        return edit

    def collect_args(self) -> dict:
        v1 = self.video_1_edit.text().strip()
        v2 = self.video_2_edit.text().strip()
        if not v1:
            raise ValueError("Pick a Video 1 path.")
        if not v2:
            raise ValueError("Pick a Video 2 path.")
        if not Path(v1).is_file():
            raise ValueError(f"Video 1 is not a file: {v1}")
        if not Path(v2).is_file():
            raise ValueError(f"Video 2 is not a file: {v2}")
        if Path(v1).resolve() == Path(v2).resolve():
            raise ValueError(
                "Video 1 and Video 2 are the same file. Pick two "
                "different videos.",
            )
        offset = self.offset_edit.text().strip()
        if not self._OFFSET_RE.match(offset):
            raise ValueError(
                f"Crossfade offset {offset!r} must be HH:MM:SS "
                "(e.g. 00:00:05). Backend validates against video 1's "
                "length on run.",
            )
        return {
            "video_path_1":       v1,
            "video_path_2":       v2,
            "crossfade_method":   self.method_cb.currentText(),
            "crossfade_duration": int(self.duration.value()),
            "crossfade_offset":   offset,
            "quality":            int(self.quality.value()),
            "out_format":         self.format_cb.currentText(),
        }

    def target(self, *, video_path_1: str, video_path_2: str,
               crossfade_method: str, crossfade_duration: int,
               crossfade_offset: str, quality: int,
               out_format: str) -> None:
        from mufasa.video_processors import video_processing as _vp
        _vp.crossfade_two_videos(
            video_path_1=video_path_1,
            video_path_2=video_path_2,
            crossfade_duration=crossfade_duration,
            crossfade_method=crossfade_method,
            crossfade_offset=crossfade_offset,
            quality=quality,
            out_format=out_format,
        )


__all__ = ["JoinVideosForm", "CrossfadeVideosForm"]
