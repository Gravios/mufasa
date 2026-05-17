"""
mufasa.ui_qt.forms.video_filters
================================

Inline form replacing five filter/enhancement popups:

* :class:`CLAHEPopUp` and :class:`InteractiveClahePopUp`
* :class:`BrightnessContrastPopUp`
* :class:`BoxBlurPopUp`
* :class:`GreyscaleSingleVideoPopUp`
* :class:`Convert2BlackWhitePopUp`

Background removal (the two ``BackgroundRemover*PopUp`` classes)
was previously a stub here. As of patch 122bu it lives in its
own section :mod:`mufasa.ui_qt.forms.video_bg_removal` because
it has 6+ parameters that don't fit the lightweight stacked-panel
layout used for the other five filters.

A single "operation" dropdown picks the filter; operation-specific
parameters live in a :class:`QStackedWidget` that swaps contents when
the operation changes. The scope picker (single file / directory) is
shared.

Design notes
------------

Interactive previewing (the one thing that justified the separate
:class:`InteractiveClahePopUp` Tk window) still needs a live video
surface, so it remains a button that launches a :class:`QDialog` with
the preview. That's the only case in this form where we fall back to
a window.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,
                               QFormLayout, QLabel, QPushButton, QSpinBox,
                               QStackedWidget, QVBoxLayout, QWidget)

from mufasa.ui_qt.forms.video_processing import _ScopePicker
from mufasa.ui_qt.workbench import OperationForm


# --------------------------------------------------------------------------- #
# Per-filter parameter panels
# --------------------------------------------------------------------------- #
class _ClahePanel(QWidget):
    """CLAHE (contrast-limited adaptive histogram equalization)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        self.clip_limit = QDoubleSpinBox(self)
        self.clip_limit.setRange(0.1, 40.0); self.clip_limit.setValue(2.0)
        self.clip_limit.setSingleStep(0.5)
        form.addRow("Clip limit:", self.clip_limit)
        self.tile_size = QSpinBox(self); self.tile_size.setRange(2, 64)
        self.tile_size.setValue(8)
        form.addRow("Tile grid size:", self.tile_size)
        self.interactive = QCheckBox("Show live preview before running", self)
        form.addRow("", self.interactive)

    def to_kwargs(self) -> dict:
        return {
            "clip_limit":  float(self.clip_limit.value()),
            "tile_grid":   int(self.tile_size.value()),
            "interactive": bool(self.interactive.isChecked()),
        }


class _BrightnessContrastPanel(QWidget):
    """Manual brightness / contrast."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        self.brightness = QDoubleSpinBox(self)
        self.brightness.setRange(-1.0, 1.0); self.brightness.setSingleStep(0.05)
        self.brightness.setValue(0.0)
        form.addRow("Brightness (−1 … +1):", self.brightness)
        self.contrast = QDoubleSpinBox(self)
        self.contrast.setRange(0.0, 3.0); self.contrast.setSingleStep(0.05)
        self.contrast.setValue(1.0)
        form.addRow("Contrast (0 … 3):", self.contrast)

    def to_kwargs(self) -> dict:
        return {"brightness": float(self.brightness.value()),
                "contrast":   float(self.contrast.value())}


class _BoxBlurPanel(QWidget):
    """Box / Gaussian blur — one-parameter."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        self.kernel = QSpinBox(self); self.kernel.setRange(3, 101)
        self.kernel.setSingleStep(2); self.kernel.setValue(5)
        form.addRow("Kernel size (odd):", self.kernel)

    def to_kwargs(self) -> dict:
        k = int(self.kernel.value())
        if k % 2 == 0:
            k += 1  # OpenCV wants odd kernel
        return {"kernel_size": k}


class _GreyscalePanel(QWidget):
    """No parameters — just a note."""
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QLabel(
            "<i>Converts videos to 8-bit greyscale. Reduces disk use by ~3×.</i>",
            self,
        ))

    def to_kwargs(self) -> dict:
        return {}


class _BlackWhitePanel(QWidget):
    """Binary threshold (not greyscale).

    Patch 122ca: the legacy ``invert`` checkbox was removed because
    the rewired backend (:func:`video_to_bw`) doesn't support
    inversion. Keeping the checkbox while ignoring it would be a
    silent-failure UX trap. If invert is later needed, add a small
    OpenCV post-processing step in :meth:`target` and re-introduce
    the field here.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        self.threshold = QSpinBox(self); self.threshold.setRange(0, 255)
        self.threshold.setValue(127)
        form.addRow("Threshold (0–255):", self.threshold)

    def to_kwargs(self) -> dict:
        return {"threshold": int(self.threshold.value())}


# --------------------------------------------------------------------------- #
# VideoFiltersForm
# --------------------------------------------------------------------------- #
class VideoFiltersForm(OperationForm):
    """Apply an image filter to video(s): CLAHE, brightness/contrast,
    blur, greyscale, or black-and-white threshold.

    Replaces 5 popups; the "operation" dropdown drives which panel of
    parameters is shown. Background removal moved out to its own
    section (:class:`mufasa.ui_qt.forms.video_bg_removal.BackgroundRemovalForm`)
    in patch 122bu because of its larger parameter surface.
    """

    title = "Filter / enhance video(s)"
    description = ("Apply image filters to videos. Choose the operation; "
                   "its parameters appear below.")

    # (label, stack-index, backend-fn name)
    OPS: list[tuple[str, int, str]] = [
        ("CLAHE (adaptive histogram)",  0, "clahe"),
        ("Brightness / contrast",       1, "brightness_contrast"),
        ("Box / Gaussian blur",         2, "blur"),
        ("Greyscale (8-bit)",           3, "greyscale"),
        ("Black & white (binarise)",    4, "black_white"),
    ]

    def build(self) -> None:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.scope = _ScopePicker(self)
        form.addRow("Source:", self.scope)

        self.op_cb = QComboBox(self)
        for label, _, _ in self.OPS:
            self.op_cb.addItem(label)
        self.op_cb.currentIndexChanged.connect(self._on_op_changed)
        form.addRow("Operation:", self.op_cb)

        self.panels = QStackedWidget(self)
        self.panels.addWidget(_ClahePanel(self))
        self.panels.addWidget(_BrightnessContrastPanel(self))
        self.panels.addWidget(_BoxBlurPanel(self))
        self.panels.addWidget(_GreyscalePanel(self))
        self.panels.addWidget(_BlackWhitePanel(self))
        form.addRow("Parameters:", self.panels)

        self.body_layout.addLayout(form)

    def _on_op_changed(self, index: int) -> None:
        self.panels.setCurrentIndex(index)

    def collect_args(self) -> dict:
        if not self.scope.path:
            raise ValueError("No source selected.")
        op_idx = self.op_cb.currentIndex()
        _, _, op_name = self.OPS[op_idx]
        panel = self.panels.widget(op_idx)
        return {
            "path":    self.scope.path,
            "is_dir":  self.scope.is_dir,
            "op":      op_name,
            "params":  panel.to_kwargs(),
        }

    def target(self, *, path: str, is_dir: bool, op: str,
               params: dict) -> None:
        from mufasa.video_processors import video_processing as _vp
        # Dispatch table. Where a backend doesn't match the kwargs
        # shape exactly, adapt here rather than in the panel.
        if op == "clahe":
            # Interactive preview is deliberately NOT wired through the
            # thread-off-GUI runner — it needs the main event loop.
            # Surface that mismatch with an explicit error for now;
            # when the preview dialog lands, this branch dispatches to it.
            if params.pop("interactive"):
                raise NotImplementedError(
                    "Interactive preview launches a dialog; not yet wired.")
            fn = _vp.clahe_enhance_video_mp
            fn(file_path=path, **params)
        elif op == "greyscale":
            if is_dir:
                _vp.batch_video_to_greyscale(path=path)
            else:
                _vp.video_to_greyscale(file_path=path)
        elif op == "black_white":
            # Patch 122ca: rewired to existing `video_to_bw` backend
            # (per docs/backend_audit.md §2b). The form's threshold
            # is integer 0–255; the backend takes float 0.0–1.0.
            # Scale here. No batch variant exists, so iterate for the
            # directory case.
            threshold = float(params["threshold"]) / 255.0
            if is_dir:
                from mufasa.utils.read_write import (
                    find_all_videos_in_directory)
                for vp in find_all_videos_in_directory(
                        directory=path, raise_error=True):
                    _vp.video_to_bw(video_path=vp, threshold=threshold)
            else:
                _vp.video_to_bw(video_path=path, threshold=threshold)
        elif op == "blur":
            raise NotImplementedError(
                "Box blur: backend wiring pending (convert_to_bw_blur).")
        elif op == "brightness_contrast":
            raise NotImplementedError(
                "Brightness/contrast: backend wiring pending.")


__all__ = ["VideoFiltersForm"]
