"""
mufasa.ui_qt.forms.video_join
=============================

Inline form for joining/combining multiple videos.

Replaces:

* :class:`ConcatenatingVideosPopUp` (2-video concat special case)
* :class:`ConcatenatorPopUp` (N-video horizontal/vertical/mosaic)
* :class:`VideoTemporalJoinPopUp` (automatic end-to-end)
* :class:`ManualTemporalJoinPopUp` (manual ordering)

Mode selector drives which backend is called; the same file list /
scope picker feeds each mode.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QFormLayout,
                               QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                               QPushButton, QSpinBox, QVBoxLayout, QWidget)

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

        # Normalisation
        self.normalize = QCheckBox(
            "Normalise input resolution / fps before join", self,
        )
        self.normalize.setChecked(True)
        form.addRow("", self.normalize)

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
            "paths":     paths,
            "mode":      mode,
            "rows":      int(self.mosaic_rows.value()),
            "normalize": bool(self.normalize.isChecked()),
        }

    def target(self, *, paths: list[str], mode: str, rows: int,
               normalize: bool) -> None:
        import math
        from pathlib import Path as _P
        from mufasa.video_processors import video_processing as _vp

        # All four concatenators require explicit save_path (no default
        # in the backend). Derive next to the first input video.
        first = _P(paths[0])
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


__all__ = ["JoinVideosForm"]
