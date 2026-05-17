"""
mufasa.ui_qt.forms.image_conversion
===================================

Inline forms for image-format conversion and for the few compute-and-
save metadata operations that don't fit the Tools-menu model.

Replaces:

* :class:`Convert2PNGPopUp`, :class:`Convert2TIFFPopUp`,
  :class:`Convert2WEBPPopUp`, :class:`Convert2bmpPopUp`,
  :class:`Convert2jpegPopUp`, :class:`ChangeImageFormatPopUp` (6 popups)
  → :class:`ImageFormatConverterForm`
* :class:`CreateAverageFramePopUp` (1 popup)
  → :class:`AverageFrameForm` (rewritten in patch 122cc to match
    the actual ``create_average_frm`` backend signature)

Read-only ``PrintVideoMetaDataPopUp`` lives on the Tools menu —
takes no parameters beyond a file picker. ``CheckVideoSeekablePopUp``
is now :class:`CheckVideoSeekableForm` in
``mufasa.ui_qt.forms.video_utilities`` under Video Processing →
Utilities (patch 122bx).
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog,
                               QFormLayout, QGroupBox, QHBoxLayout,
                               QLabel, QLineEdit, QPushButton, QSpinBox,
                               QStackedWidget, QTimeEdit, QVBoxLayout,
                               QWidget)
from PySide6.QtCore import QTime

from mufasa.ui_qt.forms.video_processing import _ScopePicker
from mufasa.ui_qt.workbench import OperationForm


# --------------------------------------------------------------------------- #
# ImageFormatConverterForm — 6 popups → 1
# --------------------------------------------------------------------------- #
class ImageFormatConverterForm(OperationForm):
    """Convert image file(s) to a target format: PNG, JPEG, BMP, WEBP, TIFF.

    Replaces six legacy Tk popups (``Convert2PNGPopUp``,
    ``Convert2jpegPopUp``, ``Convert2bmpPopUp``,
    ``Convert2WEBPPopUp``, ``Convert2TIFFPopUp``,
    ``ChangeImageFormatPopUp``). Source format is auto-detected from
    file extensions — declaring it explicitly (as
    ``ChangeImageFormatPopUp`` did) was redundant.

    Per-format options live on a :class:`QStackedWidget` so only the
    relevant controls are visible:

    * **PNG** — lossless, no extras.
    * **JPEG** — quality 1–100.
    * **BMP** — uncompressed, no extras.
    * **WEBP** — quality 1–100.
    * **TIFF** — compression (raw / tiff_deflate / tiff_lzw) +
      multi-frame stacking checkbox.

    Backend constraint: :func:`mufasa.video_processors.video_processing.convert_to_tiff`
    only accepts a directory (it writes one file per source image
    and optionally stacks them). When the user selects TIFF +
    Single-file source, the form raises a validation error pointing
    them at the Directory option. Relaxing the backend to accept a
    single-file input as a degenerate directory-of-one is a
    separate change.

    Patch 122r: rewritten from a single-quality-spin / no-save-dir
    / no-TIFF-compression draft into a full per-format options
    surface that exposes every flag the underlying backends accept.
    """

    title = "Convert image format"
    description = (
        "Re-encode image file(s) to PNG, JPEG, BMP, WEBP, or TIFF. "
        "Source is one image or a directory; output format drives "
        "the extra options (quality, compression, stacking)."
    )

    _IMG_EXTS = "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff"
    _FORMATS = ["PNG", "JPEG", "BMP", "WEBP", "TIFF"]

    def build(self) -> None:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        # Scope picker — same widget the video forms use. Image
        # file-filter rather than the default video one.
        self.scope = _ScopePicker(
            self,
            file_filter=(
                f"Image files ({self._IMG_EXTS});;All files (*)"
            ),
        )
        form.addRow("Source:", self.scope)

        self.fmt_cb = QComboBox(self)
        self.fmt_cb.addItems(self._FORMATS)
        self.fmt_cb.currentTextChanged.connect(self._on_fmt_changed)
        form.addRow("Output format:", self.fmt_cb)

        # Per-format options stack. Pages are added in the same
        # order as _FORMATS so the combo index maps directly.
        self.options = QStackedWidget(self)
        self.options.addWidget(self._make_png_panel())   # idx 0
        self.options.addWidget(self._make_jpeg_panel())  # idx 1
        self.options.addWidget(self._make_bmp_panel())   # idx 2
        self.options.addWidget(self._make_webp_panel())  # idx 3
        self.options.addWidget(self._make_tiff_panel())  # idx 4
        form.addRow("Options:", self.options)

        # Optional save_dir — every backend except TIFF respects it.
        # TIFF writes alongside the source dir; surface that in the
        # TIFF panel's note rather than disabling the field here.
        self.save_dir_edit = QLineEdit(self)
        self.save_dir_edit.setReadOnly(True)
        self.save_dir_edit.setPlaceholderText(
            "Optional — defaults to alongside source",
        )
        sd_browse = QPushButton("Browse…", self)
        sd_browse.clicked.connect(self._pick_save_dir)
        sd_row = QHBoxLayout()
        sd_row.addWidget(self.save_dir_edit, 1)
        sd_row.addWidget(sd_browse)
        form.addRow("Output directory:", sd_row)

        self.verbose = QCheckBox("Verbose progress logging", self)
        self.verbose.setChecked(True)
        form.addRow("", self.verbose)

        self.body_layout.addLayout(form)

    # ------------------------------------------------------------------ #
    # Per-format option panels
    # ------------------------------------------------------------------ #
    def _make_png_panel(self) -> QWidget:
        w = QWidget(self)
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        msg = QLabel(
            "<i>PNG: lossless; no extra options.</i>", w,
        )
        msg.setStyleSheet("color: palette(placeholder-text);")
        lay.addWidget(msg)
        lay.addStretch()
        return w

    def _make_bmp_panel(self) -> QWidget:
        w = QWidget(self)
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        msg = QLabel(
            "<i>BMP: uncompressed; no extra options.</i>", w,
        )
        msg.setStyleSheet("color: palette(placeholder-text);")
        lay.addWidget(msg)
        lay.addStretch()
        return w

    def _make_jpeg_panel(self) -> QWidget:
        w = QWidget(self)
        lay = QFormLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        self.jpeg_quality = QSpinBox(w)
        self.jpeg_quality.setRange(1, 100)
        self.jpeg_quality.setValue(95)
        self.jpeg_quality.setSuffix(" %")
        lay.addRow("Quality:", self.jpeg_quality)
        return w

    def _make_webp_panel(self) -> QWidget:
        w = QWidget(self)
        lay = QFormLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        self.webp_quality = QSpinBox(w)
        self.webp_quality.setRange(1, 100)
        self.webp_quality.setValue(95)
        self.webp_quality.setSuffix(" %")
        lay.addRow("Quality:", self.webp_quality)
        return w

    def _make_tiff_panel(self) -> QWidget:
        w = QWidget(self)
        lay = QFormLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        self.tiff_compression = QComboBox(w)
        self.tiff_compression.addItems(
            ["raw", "tiff_deflate", "tiff_lzw"],
        )
        lay.addRow("Compression:", self.tiff_compression)
        self.tiff_stack = QCheckBox(
            "Combine pages into a single multi-frame TIFF", w,
        )
        lay.addRow("", self.tiff_stack)
        note = QLabel(
            "<i>TIFF requires a directory source — the backend "
            "writes per-image. Output directory field is ignored; "
            "files land alongside the source dir.</i>", w,
        )
        note.setStyleSheet("color: palette(placeholder-text);")
        note.setWordWrap(True)
        lay.addRow("", note)
        return w

    # ------------------------------------------------------------------ #
    # Slots
    # ------------------------------------------------------------------ #
    def _on_fmt_changed(self, fmt: str) -> None:
        try:
            idx = self._FORMATS.index(fmt)
        except ValueError:
            return
        self.options.setCurrentIndex(idx)

    def _pick_save_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Pick an output directory", "",
        )
        if d:
            self.save_dir_edit.setText(d)

    # ------------------------------------------------------------------ #
    # OperationForm contract
    # ------------------------------------------------------------------ #
    def collect_args(self) -> dict:
        if not self.scope.path:
            raise ValueError("No source selected.")
        fmt = self.fmt_cb.currentText()
        # TIFF backend is directory-only — surface the constraint as
        # a validation error rather than letting it fail downstream.
        if fmt == "TIFF" and not self.scope.is_dir:
            raise ValueError(
                "TIFF conversion requires a directory source. "
                "Switch the Source picker to Directory mode, or "
                "pick a different output format.",
            )
        save_dir = self.save_dir_edit.text().strip() or None
        kwargs: dict = {
            "path":     self.scope.path,
            "is_dir":   self.scope.is_dir,
            "fmt":      fmt,
            "save_dir": save_dir,
            "verbose":  bool(self.verbose.isChecked()),
        }
        if fmt == "JPEG":
            kwargs["quality"] = int(self.jpeg_quality.value())
        elif fmt == "WEBP":
            kwargs["quality"] = int(self.webp_quality.value())
        elif fmt == "TIFF":
            kwargs["compression"] = self.tiff_compression.currentText()
            kwargs["stack"] = bool(self.tiff_stack.isChecked())
        return kwargs

    def target(self, *, path: str, is_dir: bool, fmt: str,
               save_dir: Optional[str], verbose: bool,
               quality: Optional[int] = None,
               compression: Optional[str] = None,
               stack: Optional[bool] = None) -> None:
        # Backend dispatch — per-format functions live in
        # video_processors.video_processing. Each has a slightly
        # different signature, so kwargs are built per-fn rather
        # than passed uniformly.
        from mufasa.video_processors import video_processing as _vp

        if fmt == "PNG":
            _vp.convert_to_png(
                path=path, save_dir=save_dir, verbose=verbose,
            )
        elif fmt == "JPEG":
            _vp.convert_to_jpeg(
                path=path, quality=quality or 95,
                save_dir=save_dir, verbose=verbose,
            )
        elif fmt == "BMP":
            _vp.convert_to_bmp(
                path=path, save_dir=save_dir, verbose=verbose,
            )
        elif fmt == "WEBP":
            _vp.convert_to_webp(
                path=path, quality=quality or 95,
                save_dir=save_dir, verbose=verbose,
            )
        elif fmt == "TIFF":
            # is_dir guaranteed True by collect_args; convert_to_tiff
            # takes a directory positional, not 'path'.
            _vp.convert_to_tiff(
                directory=path,
                stack=bool(stack),
                compression=compression or "raw",
                verbose=verbose,
            )
        else:
            raise ValueError(f"Unsupported output format: {fmt!r}")


# --------------------------------------------------------------------------- #
# AverageFrameForm — 1 popup
# --------------------------------------------------------------------------- #
class AverageFrameForm(OperationForm):
    """Compute and save a per-pixel average frame across a video (or
    a frame / time window within it).

    Rewritten in patch 122cc to match the actual backend signature
    :func:`mufasa.video_processors.video_processing.create_average_frm`.
    The legacy form had two issues identified in
    :doc:`backend_audit` §2a:

    * Looked up the backend as ``create_average_frame`` (with `e`)
      — the actual name is ``create_average_frm`` (no `e`). Always
      returned ``AttributeError`` → ``NotImplementedError``.
    * Surfaced ``method`` (Mean/Median) and ``stride`` parameters
      the backend doesn't accept. The backend is mean-only over all
      frames in the requested window; no stride / median support.

    The rewrite drops the unsupported fields and adds the ones the
    backend actually exposes: an optional frame-or-time window
    (via a mode selector) and an optional explicit save path.

    Replaces :class:`CreateAverageFramePopUp`.
    """

    title = "Compute average frame"
    description = (
        "Compute a per-pixel mean across the full video, or across "
        "a frame / time window. Output is a single image (PNG / "
        "JPG / TIFF) usable as a baseline for background "
        "subtraction or as a reference frame."
    )

    WINDOW_MODES = [
        ("Whole video", "whole"),
        ("Frame range", "frames"),
        ("Time range (HH:MM:SS)", "time"),
    ]

    def build(self) -> None:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.scope = _ScopePicker(self, allow_multiple=False)
        form.addRow("Source video:", self.scope)

        # ----- Window mode + stacked panels ----- #
        self.window_cb = QComboBox(self)
        for label, _ in self.WINDOW_MODES:
            self.window_cb.addItem(label)
        self.window_cb.currentIndexChanged.connect(self._on_window_changed)
        form.addRow("Average over:", self.window_cb)

        self.window_stack = QStackedWidget(self)

        # Mode 0: whole video — no extra controls
        whole_panel = QWidget(self)
        wp_lay = QVBoxLayout(whole_panel)
        wp_lay.setContentsMargins(0, 0, 0, 0)
        wp_lay.addWidget(QLabel(
            "<i>All frames in the source video.</i>", whole_panel))
        self.window_stack.addWidget(whole_panel)

        # Mode 1: frame range
        frames_panel = QWidget(self)
        fp_form = QFormLayout(frames_panel)
        fp_form.setContentsMargins(0, 0, 0, 0)
        self.start_frm = QSpinBox(self)
        self.start_frm.setRange(0, 10_000_000); self.start_frm.setValue(0)
        fp_form.addRow("Start frame:", self.start_frm)
        self.end_frm = QSpinBox(self)
        self.end_frm.setRange(0, 10_000_000); self.end_frm.setValue(100)
        fp_form.addRow("End frame:", self.end_frm)
        self.window_stack.addWidget(frames_panel)

        # Mode 2: time range
        time_panel = QWidget(self)
        tp_form = QFormLayout(time_panel)
        tp_form.setContentsMargins(0, 0, 0, 0)
        self.start_time = QTimeEdit(QTime(0, 0, 0), time_panel)
        self.start_time.setDisplayFormat("HH:mm:ss")
        tp_form.addRow("Start time:", self.start_time)
        self.end_time = QTimeEdit(QTime(0, 0, 10), time_panel)
        self.end_time.setDisplayFormat("HH:mm:ss")
        tp_form.addRow("End time:", self.end_time)
        self.window_stack.addWidget(time_panel)

        form.addRow("Window:", self.window_stack)

        # ----- Save path ----- #
        self.save_path = QLineEdit(self)
        self.save_path.setPlaceholderText(
            "Optional — defaults to alongside source with a "
            "timestamped name")
        sp_browse = QPushButton("Browse…", self)
        sp_browse.clicked.connect(self._pick_save_path)
        sp_row = QHBoxLayout()
        sp_row.addWidget(self.save_path, 1)
        sp_row.addWidget(sp_browse)
        form.addRow("Save image to:", sp_row)

        self.body_layout.addLayout(form)

    def _on_window_changed(self, idx: int) -> None:
        self.window_stack.setCurrentIndex(idx)

    def _pick_save_path(self) -> None:
        p, _ = QFileDialog.getSaveFileName(
            self, "Save average frame as", "",
            "Image files (*.png *.jpg *.jpeg *.tiff *.bmp);;"
            "All files (*)")
        if p:
            self.save_path.setText(p)

    def collect_args(self) -> dict:
        if not self.scope.path:
            raise ValueError("No source video selected.")
        mode = self.WINDOW_MODES[self.window_cb.currentIndex()][1]
        args: dict = {
            "video_path": self.scope.path,
            "save_path":  self.save_path.text().strip() or None,
        }
        if mode == "frames":
            sf = int(self.start_frm.value())
            ef = int(self.end_frm.value())
            if sf > ef:
                raise ValueError(
                    f"Start frame ({sf}) must be ≤ end frame ({ef}).")
            args["start_frm"] = sf
            args["end_frm"] = ef
        elif mode == "time":
            st: QTime = self.start_time.time()
            et: QTime = self.end_time.time()
            args["start_time"] = st.toString("HH:mm:ss")
            args["end_time"] = et.toString("HH:mm:ss")
            if st >= et:
                raise ValueError(
                    f"Start time ({args['start_time']}) must be "
                    f"before end time ({args['end_time']}).")
        # mode == "whole" → no window kwargs; backend uses whole video
        return args

    def target(self, *, video_path: str,
               save_path: Optional[str],
               start_frm: Optional[int] = None,
               end_frm: Optional[int] = None,
               start_time: Optional[str] = None,
               end_time: Optional[str] = None) -> None:
        from mufasa.video_processors.video_processing import (
            create_average_frm,
        )
        # If no explicit save_path, default to alongside source with
        # a timestamped name. Backend returns the np.ndarray only when
        # save_path is None — we always supply one so the user gets a
        # file artifact.
        if not save_path:
            base = Path(video_path)
            stamp = datetime.now().strftime("%Y%m%d%H%M%S")
            save_path = str(
                base.parent / f"{base.stem}_avgframe_{stamp}.png")
        create_average_frm(
            video_path=video_path,
            start_frm=start_frm, end_frm=end_frm,
            start_time=start_time, end_time=end_time,
            save_path=save_path,
            verbose=False,
        )


__all__ = ["ImageFormatConverterForm", "AverageFrameForm"]
