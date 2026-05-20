"""
mufasa.ui_qt.forms.video_bg_removal
====================================

Consolidates two legacy Tk popups into a single workbench
section under Video Processing → "Background removal":

* :class:`BackgroundRemoverSingleVideoPopUp` (single video)
* :class:`BackgroundRemoverDirectoryPopUp`   (directory batch)

Why pulled out of :class:`VideoFiltersForm`?  Background
subtraction has 6+ parameters (bg colour, fg colour, threshold,
optional reference video/dir, time window, core count), dwarfing
the 0–2 parameters of the other VideoFiltersForm operations
(CLAHE, brightness/contrast, blur, greyscale, B&W). The stub
that existed in :class:`VideoFiltersForm._BgRemoverPanel` was
also non-functional — it passed a ``bg_method`` kwarg the
backend doesn't accept and never wired the colour / threshold /
time-window parameters.

Backends
--------
* :func:`mufasa.video_processors.video_processing.video_bg_subtraction`
  (single-process; used when ``core_cnt == 1``)
* :func:`mufasa.video_processors.video_processing.video_bg_subtraction_mp`
  (multi-process; used when ``core_cnt > 1``)

Both take the same shape of kwargs except the mp version also
takes ``core_cnt``.
"""
from __future__ import annotations

import os
from copy import deepcopy

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from mufasa.ui_qt.workbench import OperationForm


# --------------------------------------------------------------------------- #
# Helper: optional reference path that follows the scope mode
# --------------------------------------------------------------------------- #
class _OptionalRefPath(QWidget):
    """A label + line edit + browse button. Browse opens a file
    dialog or directory dialog depending on the caller's ``is_dir``
    flag. Path can be left empty (optional).

    Used for the optional "background reference" path: a separate
    video or directory whose frames serve as the static baseline,
    instead of the source itself.
    """

    def __init__(self, parent: QWidget | None = None,
                 file_filter: str = ("Video files (*.mp4 *.avi *.mov "
                                      "*.mkv *.webm);;All files (*)")
                 ) -> None:
        super().__init__(parent)
        self._is_dir = False
        self._path = ""
        self._file_filter = file_filter

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.line = QLineEdit(self)
        self.line.setPlaceholderText(
            "(optional — defaults to using source as own baseline)")
        self.line.textChanged.connect(self._on_text)
        lay.addWidget(self.line)
        self.btn = QPushButton("Browse…", self)
        self.btn.clicked.connect(self._on_browse)
        lay.addWidget(self.btn)

    @property
    def path(self) -> str:
        return self._path

    def set_is_dir(self, is_dir: bool) -> None:
        """Switch between file-vs-directory mode. Clears the path
        so users can't carry over an incompatible selection.
        """
        if self._is_dir != is_dir:
            self._is_dir = is_dir
            self.line.clear()
            self._path = ""

    def _on_text(self, text: str) -> None:
        self._path = text.strip()

    def _on_browse(self) -> None:
        if self._is_dir:
            p = QFileDialog.getExistingDirectory(
                self, "Select background reference directory")
        else:
            p, _ = QFileDialog.getOpenFileName(
                self, "Select background reference video",
                "", self._file_filter)
        if p:
            self.line.setText(p)


# --------------------------------------------------------------------------- #
# Helper: scope picker (single file / directory)
# --------------------------------------------------------------------------- #
class _BgScopePicker(QWidget):
    """File-or-directory toggle + path. Emits ``is_dir_changed`` so
    the reference-path widget can adapt.

    This is a local helper rather than the shared
    :class:`mufasa.ui_qt.forms.video_processing._ScopePicker`
    because we need to wire an outbound signal and the shared widget
    doesn't expose one (other VideoFiltersForm callers don't need it).
    Keeping it local avoids changing the shared widget's contract.
    """

    def __init__(self, parent: QWidget | None = None,
                 on_scope_change=None,
                 file_filter: str = ("Video files (*.mp4 *.avi *.mov "
                                      "*.mkv *.webm);;All files (*)")
                 ) -> None:
        super().__init__(parent)
        self._path = ""
        self._is_dir = False
        self._file_filter = file_filter
        self._on_scope_change = on_scope_change

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.rb_single = QRadioButton("Single video", self)
        self.rb_dir = QRadioButton("Directory of videos", self)
        self.rb_single.setChecked(True)
        self.rb_single.toggled.connect(self._on_toggle)
        self.grp = QButtonGroup(self)
        self.grp.addButton(self.rb_single)
        self.grp.addButton(self.rb_dir)
        row.addWidget(self.rb_single)
        row.addWidget(self.rb_dir)
        row.addStretch(1)
        lay.addLayout(row)

        prow = QHBoxLayout()
        self.line = QLineEdit(self)
        self.line.setPlaceholderText("Select source video…")
        self.line.textChanged.connect(self._on_text)
        prow.addWidget(self.line)
        self.btn = QPushButton("Browse…", self)
        self.btn.clicked.connect(self._on_browse)
        prow.addWidget(self.btn)
        lay.addLayout(prow)

    @property
    def path(self) -> str:
        return self._path

    @property
    def is_dir(self) -> bool:
        return self._is_dir

    def _on_text(self, text: str) -> None:
        self._path = text.strip()

    def _on_browse(self) -> None:
        if self._is_dir:
            p = QFileDialog.getExistingDirectory(
                self, "Select directory of videos")
        else:
            p, _ = QFileDialog.getOpenFileName(
                self, "Select video", "", self._file_filter)
        if p:
            self.line.setText(p)

    def _on_toggle(self, checked: bool) -> None:
        # toggled fires for both buttons in a button group; only
        # react to one to avoid double-processing.
        self._is_dir = self.rb_dir.isChecked()
        self.line.clear()
        self._path = ""
        placeholder = ("Select directory of videos…" if self._is_dir
                       else "Select source video…")
        self.line.setPlaceholderText(placeholder)
        if self._on_scope_change:
            self._on_scope_change(self._is_dir)


# --------------------------------------------------------------------------- #
# BackgroundRemovalForm
# --------------------------------------------------------------------------- #
class BackgroundRemovalForm(OperationForm):
    """Subtract a static background from one or many videos.

    Replaces the two legacy Tk popups
    :class:`BackgroundRemoverSingleVideoPopUp` and
    :class:`BackgroundRemoverDirectoryPopUp` via a "Source"
    scope toggle (Single video / Directory).
    """

    title = "Background removal (subtraction)"
    description = (
        "Replace the background of a video with a flat colour, "
        "leaving moving subjects intact. Optional separate "
        "reference video / directory to compute the baseline; "
        "leave empty to use the source itself."
    )

    # Default core count: half the machine's logical CPUs; capped at 1
    # if cpu_count fails (which is rare on Linux/macOS but defensive).
    @staticmethod
    def _default_core_count() -> int:
        try:
            n = os.cpu_count() or 1
        except Exception:
            n = 1
        return max(1, n // 2)

    @staticmethod
    def _max_core_count() -> int:
        try:
            return max(1, os.cpu_count() or 1)
        except Exception:
            return 1

    def build(self) -> None:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        # ---------- scope ---------- #
        self.scope = _BgScopePicker(
            self, on_scope_change=self._on_scope_change)
        form.addRow("Source:", self.scope)

        # ---------- optional reference ---------- #
        self.ref_path = _OptionalRefPath(self)
        form.addRow("Background reference (optional):", self.ref_path)

        # ---------- colours ---------- #
        # Loaded synchronously — the lookup is a plain dict with
        # ~30 entries. Adding "Original" as a sentinel maps to
        # backend's ``fg_color=None`` (keep source colours where the
        # subject was).
        from mufasa.utils.lookups import get_color_dict
        self._colors = get_color_dict()
        color_names = list(self._colors.keys())

        self.bg_color = QComboBox(self)
        self.bg_color.addItems(color_names)
        idx = self.bg_color.findText("White")
        if idx >= 0:
            self.bg_color.setCurrentIndex(idx)
        form.addRow("Background colour:", self.bg_color)

        self.fg_color = QComboBox(self)
        # "Original" first so it's the visible default for users
        # who just want the subject in true colour against a flat
        # background.
        self.fg_color.addItems(["Original"] + color_names)
        self.fg_color.setCurrentIndex(0)
        form.addRow("Foreground colour:", self.fg_color)

        # ---------- threshold ---------- #
        # 1-99 (percentage of 255). Backend multiplies by 255/100.
        self.threshold = QSpinBox(self)
        self.threshold.setRange(1, 99)
        self.threshold.setValue(30)
        self.threshold.setSuffix(" %")
        self.threshold.setToolTip(
            "Pixel-difference threshold to flag a pixel as "
            "foreground. Lower = more sensitive (more pixels "
            "classified as moving subject). Default 30 (≈76/255).")
        form.addRow("Threshold:", self.threshold)

        # ---------- time window ---------- #
        self.entire_video_cb = QCheckBox(
            "Use entire video as background reference", self)
        self.entire_video_cb.setChecked(True)
        self.entire_video_cb.toggled.connect(self._on_entire_toggled)
        form.addRow("", self.entire_video_cb)

        self.bg_start = QLineEdit(self)
        self.bg_start.setText("00:00:00")
        self.bg_start.setEnabled(False)
        self.bg_start.setToolTip(
            "HH:MM:SS or a frame number. Marks the start of the "
            "window used to compute the baseline.")
        form.addRow("Background window start:", self.bg_start)

        self.bg_end = QLineEdit(self)
        self.bg_end.setText("00:00:20")
        self.bg_end.setEnabled(False)
        self.bg_end.setToolTip(
            "HH:MM:SS or a frame number. End of the baseline window.")
        form.addRow("Background window end:", self.bg_end)

        # ---------- parallelism ---------- #
        self.core_cnt = QSpinBox(self)
        self.core_cnt.setRange(1, self._max_core_count())
        self.core_cnt.setValue(self._default_core_count())
        self.core_cnt.setToolTip(
            "CPU cores. 1 = single-process. >1 = multi-process "
            "frame chunks.")
        form.addRow("CPU cores:", self.core_cnt)

        self.body_layout.addLayout(form)

    # ------------------------------------------------------------------ #
    # Internal slots
    # ------------------------------------------------------------------ #
    def _on_scope_change(self, is_dir: bool) -> None:
        """Reference-path widget follows the scope toggle."""
        self.ref_path.set_is_dir(is_dir)

    def _on_entire_toggled(self, checked: bool) -> None:
        """Disable bg-window fields when 'entire video' is on."""
        self.bg_start.setEnabled(not checked)
        self.bg_end.setEnabled(not checked)

    # ------------------------------------------------------------------ #
    # OperationForm contract
    # ------------------------------------------------------------------ #
    def collect_args(self) -> dict:
        if not self.scope.path:
            raise ValueError("No source selected.")

        bg_name = self.bg_color.currentText()
        bg_color = self._colors[bg_name]
        fg_name = self.fg_color.currentText()
        fg_color = None if fg_name == "Original" else self._colors[fg_name]

        if bg_color == fg_color:
            raise ValueError(
                f"Background and foreground colour must differ "
                f"(both set to {fg_name}).")

        # Threshold: percentage of 255.
        threshold = int(self.threshold.value() / 100.0 * 255)

        # Time window
        entire = self.entire_video_cb.isChecked()
        bg_start_frm = bg_end_frm = None
        bg_start_time = bg_end_time = None
        if not entire:
            start_str = self.bg_start.text().strip()
            end_str = self.bg_end.text().strip()
            # Distinguish frame-number vs HH:MM:SS by attempting int
            # conversion. Both Tk popups did the same.
            try:
                si, ei = int(start_str), int(end_str)
                if si >= ei:
                    raise ValueError(
                        f"Start frame ({si}) must be before end "
                        f"frame ({ei}).")
                bg_start_frm, bg_end_frm = si, ei
            except ValueError:
                # Not integers — treat as HH:MM:SS.
                # Validation happens in the backend.
                if not start_str or not end_str:
                    raise ValueError(
                        "Background window start and end are required "
                        "when 'use entire video' is unchecked.")
                bg_start_time, bg_end_time = start_str, end_str

        return {
            "src_path":     self.scope.path,
            "is_dir":       self.scope.is_dir,
            "ref_path":     self.ref_path.path,
            "bg_color":     bg_color,
            "fg_color":     fg_color,
            "threshold":    threshold,
            "bg_start_frm": bg_start_frm,
            "bg_end_frm":   bg_end_frm,
            "bg_start_time": bg_start_time,
            "bg_end_time":  bg_end_time,
            "core_cnt":     int(self.core_cnt.value()),
        }

    def target(self, *, src_path: str, is_dir: bool, ref_path: str,
               bg_color, fg_color, threshold: int,
               bg_start_frm, bg_end_frm, bg_start_time, bg_end_time,
               core_cnt: int) -> None:
        """Dispatch to single-process or multi-process backend,
        per-video for directory scope.
        """
        from mufasa.utils.read_write import (
            find_all_videos_in_directory,
            get_video_meta_data,
        )
        from mufasa.video_processors.video_processing import (
            video_bg_subtraction,
            video_bg_subtraction_mp,
        )

        def _resolve_bg_window(video_path: str, bg_video_path: str
                                ) -> tuple:
            """If 'entire video' was selected (frames+times all
            None), default the window to the full bg-video span.
            Otherwise pass through what the user set.
            """
            if (bg_start_frm is None and bg_end_frm is None
                    and bg_start_time is None and bg_end_time is None):
                meta = get_video_meta_data(video_path=bg_video_path)
                return 0, meta["frame_count"], None, None
            return (bg_start_frm, bg_end_frm,
                    bg_start_time, bg_end_time)

        def _call_one(video_path: str, bg_video_path: str) -> None:
            sf, ef, st, et = _resolve_bg_window(video_path, bg_video_path)
            if core_cnt == 1:
                video_bg_subtraction(
                    video_path=video_path,
                    bg_video_path=bg_video_path,
                    bg_start_frm=sf, bg_end_frm=ef,
                    bg_start_time=st, bg_end_time=et,
                    bg_color=bg_color, fg_color=fg_color,
                    threshold=threshold,
                )
            else:
                video_bg_subtraction_mp(
                    video_path=video_path,
                    bg_video_path=bg_video_path,
                    bg_start_frm=sf, bg_end_frm=ef,
                    bg_start_time=st, bg_end_time=et,
                    bg_color=bg_color, fg_color=fg_color,
                    core_cnt=core_cnt, threshold=threshold,
                )

        if not is_dir:
            # Single-video mode. Reference video defaults to source.
            bg_video_path = (ref_path if ref_path
                             and os.path.isfile(ref_path)
                             else deepcopy(src_path))
            _call_one(src_path, bg_video_path)
            return

        # Directory mode.
        video_paths = find_all_videos_in_directory(
            directory=src_path, as_dict=True, raise_error=True)

        if ref_path and os.path.isdir(ref_path):
            ref_paths = find_all_videos_in_directory(
                directory=ref_path, as_dict=True, raise_error=True)
            missing = [n for n in video_paths if n not in ref_paths]
            if missing:
                raise ValueError(
                    "Reference directory is missing matching files "
                    f"for: {missing}")
        else:
            ref_paths = deepcopy(video_paths)

        for video_name, video_path in video_paths.items():
            _call_one(video_path, ref_paths[video_name])


__all__ = ["BackgroundRemovalForm"]
