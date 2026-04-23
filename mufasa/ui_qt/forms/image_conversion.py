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
from PySide6.QtWidgets import (QComboBox, QFormLayout, QSpinBox, QWidget)

from mufasa.ui_qt.forms.video_processing import _ScopePicker
from mufasa.ui_qt.workbench import OperationForm


# --------------------------------------------------------------------------- #
# ImageFormatConverterForm — 6 popups → 1
# --------------------------------------------------------------------------- #
class ImageFormatConverterForm(OperationForm):
    """Convert images to a target format. Replaces 5 specific-format
    popups (Convert2PNG/TIFF/WEBP/bmp/jpeg) plus the generic
    ChangeImageFormatPopUp. Source image format is auto-detected from
    files in the directory; target chosen by dropdown.
    """

    title = "Convert image format"
    description = ("Convert images to a different format. Source format "
                   "auto-detected from the files in the directory.")

    TARGETS = ["PNG", "JPEG", "TIFF", "BMP", "WEBP"]

    def build(self) -> None:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.scope = _ScopePicker(
            self,
            file_filter="Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp);;All files (*)",
        )
        form.addRow("Source:", self.scope)

        self.fmt_cb = QComboBox(self)
        self.fmt_cb.addItems(self.TARGETS)
        form.addRow("Target format:", self.fmt_cb)

        # Format-sensitive extra: JPEG/WEBP want a quality spin
        self.quality = QSpinBox(self)
        self.quality.setRange(1, 100); self.quality.setValue(90)
        self.quality.setSuffix(" %")
        form.addRow("Quality (JPEG/WEBP):", self.quality)

        self.body_layout.addLayout(form)

    def collect_args(self) -> dict:
        if not self.scope.path:
            raise ValueError("No source selected.")
        return {
            "path":    self.scope.path,
            "is_dir":  self.scope.is_dir,
            "fmt":     self.fmt_cb.currentText().lower(),
            "quality": int(self.quality.value()),
        }

    def target(self, *, path: str, is_dir: bool, fmt: str,
               quality: int) -> None:
        from pathlib import Path as _P
        from mufasa.video_processors import video_processing as _vp
        dispatch = {
            "png":  _vp.convert_to_png,
            "tiff": _vp.convert_to_tiff,
            "webp": _vp.convert_to_webp,
            "bmp":  _vp.convert_to_bmp,
        }
        if fmt == "jpeg":
            # `change_img_format` requires explicit in/out file types.
            # Detect source format from directory contents; fall back
            # to "png" if nothing obvious.
            file_type_in = "png"
            if is_dir:
                for ext in ("png", "tiff", "bmp", "webp", "jpg"):
                    if any(f.suffix.lower().lstrip(".") == ext
                           for f in _P(path).iterdir()):
                        file_type_in = ext
                        break
            # Note: this backend doesn't honour a `quality` kwarg —
            # JPEG quality is baked into its ffmpeg invocation. If
            # quality control is needed, extend the backend upstream.
            _vp.change_img_format(directory=path,
                                  file_type_in=file_type_in,
                                  file_type_out="jpg")
        elif fmt in dispatch:
            fn = dispatch[fmt]
            # convert_to_png/webp/bmp take `path=`; convert_to_tiff
            # takes `directory=`.
            if fn is _vp.convert_to_tiff:
                fn(directory=path)
            else:
                fn(path=path)


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
