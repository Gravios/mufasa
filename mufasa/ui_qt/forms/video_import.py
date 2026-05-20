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

Patch history
-------------
* **122o**: introduced. Replaced the legacy
  :class:`mufasa.ui.import_videos_frame.ImportVideosFrame` Tk
  popup with an inline Qt surface on the Data Import page, and
  piggybacked layout-aware destination resolution into the
  existing ``copy_*_video*`` helpers (which previously
  hardcoded ``<project>/videos/``).
* **122v**: symlink mode is now the default (the typical
  workflow re-references existing video stores rather than
  duplicating multi-GB files); pre-flight duplicate detection
  warns before invoking the copy so users aren't surprised by
  silent skips; a read-only "Already imported" table lists the
  current project video tree (filename, size, modified date,
  symlink target if applicable) and refreshes after each
  successful import.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
)

from mufasa.ui_qt.workbench import OperationForm

_VIDEO_FORMATS = ("mp4", "avi", "mov", "mkv", "webm", "m4v")


def _humanize_bytes(n: int) -> str:
    """Format a byte count as a short human string.

    Inline implementation so the form doesn't pull a stdlib
    dependency. Symmetric rounding at 1024 makes 1023 → '1023 B'
    rather than 1.0 KB, which feels right for file sizes.
    """
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return (f"{n} {unit}" if unit == "B"
                    else f"{n:.1f} {unit}")
        n /= 1024
    return f"{n:.1f} TB"


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
        "<code>videos/</code> (legacy). <b>Symlink mode is on by "
        "default</b> — saves disk space at the cost of portability."
    )

    def build(self) -> None:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        # ----- Source-type radio ----- #
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
        self._dir_file_type = QComboBox(self)
        self._dir_file_type.addItems(_VIDEO_FORMATS)
        self._recursive = QCheckBox("Search subdirectories", self)
        form.addRow("Directory format:", self._dir_file_type)
        form.addRow("", self._recursive)

        # ----- Common options ----- #
        # Patch 122v: symlink ON by default. The typical workflow
        # re-references existing video stores rather than duplicating
        # multi-GB files. The non-portability tradeoff is now opt-out
        # instead of opt-in.
        self._symlink = QCheckBox(
            "Create symbolic links instead of copying "
            "(default; non-portable)",
            self,
        )
        self._symlink.setChecked(True)
        form.addRow("", self._symlink)

        self.body_layout.addLayout(form)

        # ----- Already-imported table ----- #
        # Patch 122v: read-only view of the current project's video
        # tree. Updated on form show + after each import.
        self.body_layout.addSpacing(8)
        header_row = QHBoxLayout()
        header_row.addWidget(QLabel("<b>Already imported</b>", self))
        header_row.addStretch()
        refresh_btn = QPushButton("Refresh", self)
        refresh_btn.clicked.connect(self._refresh_table)
        header_row.addWidget(refresh_btn)
        self.body_layout.addLayout(header_row)

        self._table = QTableWidget(0, 4, self)
        self._table.setHorizontalHeaderLabels([
            "Filename", "Size", "Modified", "Symlink target",
        ])
        self._table.setEditTriggers(
            QAbstractItemView.NoEditTriggers,
        )
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectRows,
        )
        self._table.verticalHeader().setVisible(False)
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.Stretch)
        self._table.setMinimumHeight(140)
        self.body_layout.addWidget(self._table)

        self._on_mode_changed()
        # Try populating once at build time — project may already
        # be loaded.
        self._refresh_table()

    # ------------------------------------------------------------------ #
    # Slots
    # ------------------------------------------------------------------ #
    def _on_mode_changed(self) -> None:
        is_dir = self._mode_directory.isChecked()
        self._dir_file_type.setEnabled(is_dir)
        self._recursive.setEnabled(is_dir)
        self._source_edit.clear()

    def _pick_source(self) -> None:
        if self._mode_single.isChecked():
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
    # Duplicate detection (patch 122v)
    # ------------------------------------------------------------------ #
    def _videos_dir(self) -> Path | None:
        """Resolve the project's videos directory, or None if no
        project is loaded or the layout helper isn't reachable.

        Sandboxed gracefully — the project_layout helper imports
        heavy dependencies that may not be available in all
        environments, so the import is local."""
        if not self.config_path:
            return None
        try:
            from mufasa.project_layout import (
                project_paths_from_config,
            )
            paths = project_paths_from_config(self.config_path)
            return Path(paths["video_dir"])
        except Exception:
            return None

    def _existing_video_names(self) -> set[str]:
        """Filenames currently sitting in the project's video tree.

        Used by both the duplicate-check pre-flight and the
        already-imported table.
        """
        vd = self._videos_dir()
        if vd is None or not vd.is_dir():
            return set()
        return {p.name for p in vd.iterdir() if p.is_file()}

    def _candidate_basenames(self, source: str, mode_single: bool,
                             file_type: str,
                             recursive: bool) -> list[str]:
        """Names that *would* be written by an import of `source`.

        Lets the duplicate-check pre-flight enumerate clashes
        without running the copy. For directory mode, mirrors the
        backend's filter (single file_type extension, recursive or
        flat).
        """
        sp = Path(source)
        if mode_single:
            return [sp.name]
        if not sp.is_dir():
            return []
        pattern = f"*.{file_type}"
        iterator = sp.rglob(pattern) if recursive else sp.glob(pattern)
        return [p.name for p in iterator if p.is_file()]

    def _confirm_duplicates(self,
                            duplicates: list[str]) -> bool:
        """Pop a confirmation dialog listing the duplicate filenames.

        Returns True if the user wants to proceed (backends will
        skip the existing destinations), False to cancel. Limited
        to first 20 names in the dialog body to avoid an unreadable
        wall of text.
        """
        head = duplicates[:20]
        tail_n = len(duplicates) - len(head)
        body = (
            f"{len(duplicates)} video"
            f"{'s' if len(duplicates) != 1 else ''} already exist in "
            "the project's video directory:\n\n"
            + "\n".join(f"  • {n}" for n in head)
            + (f"\n  … and {tail_n} more" if tail_n else "")
            + "\n\nProceed anyway? The existing destinations will be "
            "skipped — the source files are not touched."
        )
        reply = QMessageBox.question(
            self, "Duplicate videos detected", body,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return reply == QMessageBox.Yes

    # ------------------------------------------------------------------ #
    # Already-imported table (patch 122v)
    # ------------------------------------------------------------------ #
    def _refresh_table(self) -> None:
        """Repopulate the already-imported table from the videos dir.

        Listing failure (no project, missing dir) clears the table
        and shows a single italic placeholder row so the user sees
        what's going on rather than an empty grid.
        """
        self._table.setRowCount(0)
        vd = self._videos_dir()
        if vd is None:
            self._set_placeholder_row(
                "(no project loaded — open a project to see "
                "imported videos)"
            )
            return
        if not vd.is_dir():
            self._set_placeholder_row(f"(no videos directory at {vd})")
            return
        rows = []
        for p in sorted(vd.iterdir()):
            if not p.is_file():
                continue
            try:
                st = p.lstat()
                size = _humanize_bytes(int(st.st_size))
                mtime = datetime.fromtimestamp(
                    st.st_mtime,
                ).strftime("%Y-%m-%d %H:%M")
                if p.is_symlink():
                    try:
                        target = str(os.readlink(p))
                    except OSError:
                        target = "(broken symlink)"
                else:
                    target = ""
                rows.append((p.name, size, mtime, target))
            except OSError:
                rows.append((p.name, "?", "?", ""))
        if not rows:
            self._set_placeholder_row("(no videos imported yet)")
            return
        self._table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                item = QTableWidgetItem(val)
                if c == 3 and val.startswith("("):
                    # broken-symlink hint stays grey
                    item.setForeground(Qt.GlobalColor.gray)
                self._table.setItem(r, c, item)

    def _set_placeholder_row(self, text: str) -> None:
        """Render a single full-width grey row with hint text."""
        self._table.setRowCount(1)
        item = QTableWidgetItem(text)
        item.setForeground(Qt.GlobalColor.gray)
        self._table.setItem(0, 0, item)
        self._table.setSpan(0, 0, 1, 4)

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

        # Patch 122v: pre-flight duplicate detection. Enumerate the
        # candidate basenames and compare against the existing video
        # tree; if any overlap, prompt the user before invoking the
        # backend (which would silently skip them).
        file_type = self._dir_file_type.currentText()
        recursive = self._recursive.isChecked()
        candidates = self._candidate_basenames(
            source=source, mode_single=mode_single,
            file_type=file_type, recursive=recursive,
        )
        existing = self._existing_video_names()
        duplicates = sorted(set(candidates) & existing)
        if duplicates and not self._confirm_duplicates(duplicates):
            raise RuntimeError(
                "Import cancelled because of duplicate filenames.",
            )

        return {
            "config_path":  self.config_path,
            "source":       source,
            "mode_single":  mode_single,
            "file_type":    file_type,
            "recursive":    recursive,
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
                mufasa_ini_path=config_path,
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
        # Refresh the table after a successful import so the user
        # sees the new rows immediately.
        self._refresh_table()


__all__ = ["VideoImportForm"]
