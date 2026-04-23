"""
mufasa.ui_qt.forms.video_filters
================================

Inline form replacing the seven filter/enhancement popups:

* :class:`CLAHEPopUp` and :class:`InteractiveClahePopUp`
* :class:`BrightnessContrastPopUp`
* :class:`BoxBlurPopUp`
* :class:`GreyscaleSingleVideoPopUp`
* :class:`Convert2BlackWhitePopUp`
* :class:`BackgroundRemoverSingleVideoPopUp`
* :class:`BackgroundRemoverDirectoryPopUp`

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
    """Binary threshold (not greyscale)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        self.threshold = QSpinBox(self); self.threshold.setRange(0, 255)
        self.threshold.setValue(127)
        form.addRow("Threshold (0–255):", self.threshold)
        self.invert = QCheckBox("Invert (dark → white)", self)
        form.addRow("", self.invert)

    def to_kwargs(self) -> dict:
        return {"threshold": int(self.threshold.value()),
                "invert":    bool(self.invert.isChecked())}


class _BgRemoverPanel(QWidget):
    """Background subtraction — static baseline."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        self.method = QComboBox(self)
        self.method.addItems(["Mean", "Median", "KNN", "MOG2"])
        form.addRow("Baseline method:", self.method)
        self.parallel = QCheckBox("Multi-process (parallel frames)", self)
        self.parallel.setChecked(True)
        form.addRow("", self.parallel)

    def to_kwargs(self) -> dict:
        return {"method":   self.method.currentText().lower(),
                "parallel": bool(self.parallel.isChecked())}


# --------------------------------------------------------------------------- #
# VideoFiltersForm
# --------------------------------------------------------------------------- #
class VideoFiltersForm(OperationForm):
    """Apply an image filter to video(s): CLAHE, brightness/contrast,
    blur, greyscale, black-and-white threshold, or background removal.

    Replaces 7 popups; the "operation" dropdown drives which panel of
    parameters is shown.
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
        ("Background subtraction",      5, "bg_remove"),
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
        self.panels.addWidget(_BgRemoverPanel(self))
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
        elif op == "bg_remove":
            method = params.get("method", "mean")
            parallel = params.get("parallel", True)
            fn = _vp.video_bg_subtraction_mp if parallel else _vp.video_bg_subtraction
            fn(video_path=path, bg_method=method)
        elif op == "black_white":
            # No dedicated backend fn on newer versions; treat as B&W
            # via greyscale + threshold. Keep the branch but mark for
            # backend wiring.
            raise NotImplementedError(
                "Black & white threshold: backend wiring pending "
                "(convert_to_black_and_white is in an older branch)."
            )
        elif op == "blur":
            raise NotImplementedError(
                "Box blur: backend wiring pending (convert_to_bw_blur).")
        elif op == "brightness_contrast":
            raise NotImplementedError(
                "Brightness/contrast: backend wiring pending.")


__all__ = ["VideoFiltersForm"]
