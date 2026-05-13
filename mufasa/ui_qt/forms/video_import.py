"""
mufasa.ui_qt.forms.video_import
================================

Inline form for importing video files into the currently-open
project. Sits on the Data Import workbench page alongside the
pose-import and video-calibration surfaces.

Two source modes:

* **Single video file** — copy (or symlink) one ``.mp4`` /
  ``.avi`` / ``.mov`` / ``.mkv`` / ``.webm`` into the project's
  videos directory.
* **Directory of videos** — walk a directory (optionally
  recursive) and copy every video file matching the allowed
  formats. Existing destinations skipped with a warning rather
  than overwritten.

Destination resolution is layout-aware: v1 projects land in
``<root>/sources/videos/``; legacy ``project_config.ini``
projects land in ``<project>/videos/`` to match SimBA's tree.
:func:`mufasa.project_layout.project_paths_from_config`
encapsulates the branching; this form (and the underlying
``copy_*_video*`` helpers in ``read_write``) just call into it.

Patch 122o: introduced. Replaces the legacy
:class:`mufasa.ui.import_videos_frame.ImportVideosFrame` Tk
popup with an inline Qt surface on the Data Import page, and
piggybacks layout-aware destination resolution into the existing
``copy_single_video_to_project`` / ``copy_multiple_videos_to_project``
helpers (which previously hardcoded ``<project>/videos/``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox,
                               QFileDialog, QFormLayout, QHBoxLayout,
                               QLabel, QLineEdit, QPushButton,
                               QRadioButton, QVBoxLayout)

from mufasa.ui_qt.workbench import OperationForm


_VIDEO_FORMATS = ("mp4", "avi", "mov", "mkv", "webm", "m4v")


class VideoImportForm(OperationForm):
    """Inline Qt form: import a single video or a directory of videos
    into the project's video tree.

    Validation surfaces as RuntimeError from ``collect_args``; the
    OperationForm base class catches it and shows a non-blocking
    error in the status bar (consistent with every other form on
    the page).
    """

    title = "Import video"
    description = (
        "Bring one video or a directory of videos into this project. "
        "Files copy into <code>sources/videos/</code> (v1) or "
        "<code>videos/</code> (legacy). Symlink mode is available "
        "when disk space matters more than portability."
    )

    def build(self) -> None:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        # ----- Source-type radio ----- #
        # QButtonGroup is exclusive by default; pin the radios into
        # it so they behave as a pair without a parent groupbox.
        self._mode_single = QRadioButton("Single video file", self)
        self._mode_directory = QRadioButton("Directory of videos", self)
        self._mode_single.setChecked(True)
        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self._mode_single, 0)
        self._mode_group.addButton(self._mode_directory, 1)
        self._mode_single.toggled.connect(self._on_mode_changed)

        mode_row = QHBoxLayout()
        mode_row.addWidget(self._mode_single)
        mode_row.addWidget(self._mode_directory)
        mode_row.addStretch()
        form.addRow("Source type:", mode_row)

        # ----- Source path + Browse ----- #
        self._source_edit = QLineEdit(self)
        self._source_edit.setReadOnly(True)
        self._source_edit.setPlaceholderText(
            "Pick a video file or a directory…",
        )
        browse = QPushButton("Browse…", self)
        browse.clicked.connect(self._pick_source)
        src_row = QHBoxLayout()
        src_row.addWidget(self._source_edit, 1)
        src_row.addWidget(browse)
        form.addRow("Source:", src_row)

        # ----- Directory-mode options ----- #
        # File-type combo restricts the directory scan to one
        # extension at a time (matches the legacy helper's
        # behaviour where file_type is a single string). The
        # recursive checkbox flips between flat and walked scans.
        self._dir_file_type = QComboBox(self)
        self._dir_file_type.addItems(_VIDEO_FORMATS)
        self._recursive = QCheckBox("Search subdirectories", self)
        form.addRow("Directory format:", self._dir_file_type)
        form.addRow("", self._recursive)

        # ----- Common options ----- #
        # Symlink mode is the right choice when the source videos
        # are on a fast local disk that the project can reference
        # without copying. The cost is portability — a symlinked
        # project can't be moved off the machine without breaking
        # the references.
        self._symlink = QCheckBox(
            "Create symbolic links instead of copying "
            "(faster; non-portable)",
            self,
        )
        form.addRow("", self._symlink)

        self.body_layout.addLayout(form)
        # Initial visibility / enable state of mode-specific rows
        self._on_mode_changed()

    # ------------------------------------------------------------------ #
    # Slots
    # ------------------------------------------------------------------ #
    def _on_mode_changed(self) -> None:
        """Show only the options relevant to the current source mode.

        QFormLayout doesn't expose per-row hide directly in older Qt
        APIs; setEnabled is the portable compromise. The form-type +
        recursive options are still visible when in single-file mode
        so the user can see what's available, but they're grayed.
        """
        is_dir = self._mode_directory.isChecked()
        self._dir_file_type.setEnabled(is_dir)
        self._recursive.setEnabled(is_dir)
        # Reset the path field on mode change — switching from a
        # picked file to directory mode would otherwise leave a
        # stale single-file path that doesn't match the new mode.
        self._source_edit.clear()

    def _pick_source(self) -> None:
        if self._mode_single.isChecked():
            # Build a Qt-friendly filter string from _VIDEO_FORMATS so
            # the dialog defaults to showing only video files.
            ext_glob = " ".join(f"*.{e}" for e in _VIDEO_FORMATS)
            path, _ = QFileDialog.getOpenFileName(
                self, "Pick a video file", "",
                f"Video files ({ext_glob});;All files (*)",
            )
        else:
            path = QFileDialog.getExistingDirectory(
                self, "Pick a directory of videos", "",
            )
        if path:
            self._source_edit.setText(path)

    # ------------------------------------------------------------------ #
    # OperationForm contract
    # ------------------------------------------------------------------ #
    def collect_args(self) -> dict:
        if not self.config_path:
            raise RuntimeError("No project loaded.")
        source = self._source_edit.text().strip()
        if not source:
            raise RuntimeError("Pick a video file or directory first.")
        mode_single = self._mode_single.isChecked()
        if mode_single:
            if not Path(source).is_file():
                raise RuntimeError(
                    f"Source is not a file: {source}",
                )
        else:
            if not Path(source).is_dir():
                raise RuntimeError(
                    f"Source is not a directory: {source}",
                )

        return {
            "config_path":  self.config_path,
            "source":       source,
            "mode_single":  mode_single,
            "file_type":    self._dir_file_type.currentText(),
            "recursive":    self._recursive.isChecked(),
            "symlink":      self._symlink.isChecked(),
        }

    def target(self, *, config_path: str, source: str,
               mode_single: bool, file_type: str,
               recursive: bool, symlink: bool) -> None:
        from mufasa.utils.read_write import (
            copy_multiple_videos_to_project,
            copy_single_video_to_project,
        )
        if mode_single:
            copy_single_video_to_project(
                simba_ini_path=config_path,
                source_path=source,
                symlink=symlink,
                allowed_video_formats=_VIDEO_FORMATS,
                overwrite=False,
            )
        else:
            copy_multiple_videos_to_project(
                config_path=config_path,
                source=source,
                file_type=file_type,
                symlink=symlink,
                recursive_search=recursive,
                allowed_video_formats=_VIDEO_FORMATS,
            )


__all__ = ["VideoImportForm"]
