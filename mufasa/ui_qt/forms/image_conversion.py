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
  → :class:`AverageFrameForm`

Read-only inspection popups (``PrintVideoMetaDataPopUp``,
``CheckVideoSeekablePopUp``) live on the Tools menu — they take no
parameters beyond a file picker and have no config to configure.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog,
                               QFormLayout, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QSpinBox,
                               QStackedWidget, QWidget)

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
    """Compute and save an average / median frame of a video. Used as
    the baseline for background subtraction.
    """

    title = "Compute average frame"
    description = ("Save a per-pixel mean or median frame over the full "
                   "video. Typical use: baseline for background "
                   "subtraction.")

    def build(self) -> None:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.scope = _ScopePicker(self, allow_multiple=False)
        form.addRow("Source video:", self.scope)

        self.stat_cb = QComboBox(self)
        self.stat_cb.addItems(["Mean", "Median"])
        form.addRow("Statistic:", self.stat_cb)

        self.stride = QSpinBox(self)
        self.stride.setRange(1, 1000); self.stride.setValue(1)
        self.stride.setPrefix("every ")
        self.stride.setSuffix(" frame(s)")
        form.addRow("Sampling:", self.stride)

        self.body_layout.addLayout(form)

    def collect_args(self) -> dict:
        if not self.scope.path:
            raise ValueError("No source video selected.")
        return {
            "path":   self.scope.path,
            "method": self.stat_cb.currentText().lower(),
            "stride": int(self.stride.value()),
        }

    def target(self, *, path: str, method: str, stride: int) -> None:
        # create_average_frm / create_average_frm_cupy live in
        # mufasa.data_processors.cuda.image; the CPU equivalent is
        # in mufasa.video_processors.video_processing but naming
        # diverges between branches. Raise explicitly if not found.
        try:
            from mufasa.video_processors import video_processing as _vp
            fn = getattr(_vp, "create_average_frame", None)
            if fn is None:
                raise AttributeError("create_average_frame not found")
            fn(video_path=path, method=method, frame_stride=stride)
        except (AttributeError, ImportError) as exc:
            raise NotImplementedError(
                f"create_average_frame backend not present in this fork "
                f"(looked for video_processing.create_average_frame). "
                f"Underlying cause: {exc}"
            )


__all__ = ["ImageFormatConverterForm", "AverageFrameForm"]
