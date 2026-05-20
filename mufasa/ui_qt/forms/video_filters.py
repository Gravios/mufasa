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

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from mufasa.ui_qt.forms.video_processing import _ScopePicker
from mufasa.ui_qt.workbench import OperationForm


# --------------------------------------------------------------------------- #
# Per-filter parameter panels
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# CLAHE interactive-preview dialog (patch 122ci)
# --------------------------------------------------------------------------- #
def _cv2_to_qpixmap(img) -> QPixmap:
    """Convert a CV2/numpy image (2D grayscale or 3D RGB888) to a
    QPixmap. Mirrors the helper in
    :mod:`mufasa.ui_qt.forms.blob_quick_check` — kept local here
    to avoid a cross-module dependency.
    """
    import numpy as np
    if img.ndim == 2:
        h, w = img.shape
        qimg = QImage(img.tobytes(), w, h, w, QImage.Format_Grayscale8)
    elif img.ndim == 3 and img.shape[2] == 3:
        h, w, _ = img.shape
        qimg = QImage(np.ascontiguousarray(img).tobytes(), w, h,
                      w * 3, QImage.Format_RGB888)
    else:
        raise ValueError(
            f"Unsupported image shape {img.shape}")
    return QPixmap.fromImage(qimg)


class _ClahePreviewDialog(QDialog):
    """Live-preview tuning dialog for CLAHE. Lets the user iterate
    on `clip_limit` and `tile_size` against a chosen frame, then
    confirms the final values (OK) or aborts (Cancel).

    Patch 122ci. Same architecture pattern as
    :class:`_BlobCheckDialog`:

    * QLabel-backed image display, auto-scaled.
    * Live `clip_limit` (QDoubleSpinBox) + `tile_size` (QSpinBox);
      changes trigger immediate re-render of the displayed frame.
    * Frame slider for navigation across the video; CLAHE re-runs
      on the new frame whenever the slider moves.
    * Standard OK / Cancel buttons via QDialogButtonBox.

    The CLAHE application is OpenCV's ``cv2.createCLAHE`` —
    matches the actual backend (`clahe_enhance_video_mp`'s per-
    frame transform) so the preview is faithful, not a stand-in.
    """

    def __init__(self, video_path: str,
                 init_clip_limit: float,
                 init_tile_size: int,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("CLAHE — live preview")
        self.video_path = video_path

        # Lazy-import the backend pieces — AST tests don't need
        # cv2 / utils to load this module.
        from mufasa.utils.read_write import get_video_meta_data, read_frm_of_video
        self._read_frm = read_frm_of_video

        meta = get_video_meta_data(video_path=video_path)
        self.frame_count = int(meta["frame_count"])
        self.img_idx = 0

        self._build_ui(init_clip_limit, init_tile_size)
        self._refresh()

    # ------------------------------------------------------------------ #
    # UI
    # ------------------------------------------------------------------ #
    def _build_ui(self, init_clip: float, init_tile: int) -> None:
        root = QVBoxLayout(self)

        # Image canvas
        self.image_lbl = QLabel(self)
        self.image_lbl.setAlignment(Qt.AlignCenter)
        self.image_lbl.setMinimumSize(640, 360)
        root.addWidget(self.image_lbl, 1)

        # Tunables
        params = QFormLayout()
        params.setLabelAlignment(Qt.AlignRight)
        self.clip_sp = QDoubleSpinBox(self)
        self.clip_sp.setRange(0.1, 40.0); self.clip_sp.setSingleStep(0.5)
        self.clip_sp.setValue(init_clip)
        self.clip_sp.valueChanged.connect(self._refresh)
        params.addRow("Clip limit:", self.clip_sp)
        self.tile_sp = QSpinBox(self)
        self.tile_sp.setRange(2, 64); self.tile_sp.setValue(init_tile)
        self.tile_sp.valueChanged.connect(self._refresh)
        params.addRow("Tile grid size:", self.tile_sp)
        root.addLayout(params)

        # Frame nav
        nav_row = QHBoxLayout()
        self.frame_slider = QSlider(Qt.Horizontal, self)
        self.frame_slider.setRange(0, max(0, self.frame_count - 1))
        self.frame_slider.setValue(0)
        self.frame_slider.valueChanged.connect(self._on_frame_changed)
        nav_row.addWidget(self.frame_slider, 1)
        self.frame_lbl = QLabel(
            f"frame 0 / {self.frame_count - 1}", self)
        nav_row.addWidget(self.frame_lbl)
        root.addLayout(nav_row)

        # OK / Cancel
        btns = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        # Rename OK to "Apply & run" so the action is unambiguous
        ok_btn = btns.button(QDialogButtonBox.Ok)
        if ok_btn is not None:
            ok_btn.setText("Apply && run")
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

    # ------------------------------------------------------------------ #
    # Render
    # ------------------------------------------------------------------ #
    def _on_frame_changed(self, idx: int) -> None:
        self.img_idx = int(idx)
        self.frame_lbl.setText(
            f"frame {self.img_idx} / {self.frame_count - 1}")
        self._refresh()

    def _refresh(self, *_args) -> None:
        import cv2
        try:
            frame = self._read_frm(video_path=self.video_path,
                                   frame_index=self.img_idx)
        except Exception:
            return
        # CLAHE operates on grayscale; convert if needed
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        clahe = cv2.createCLAHE(
            clipLimit=float(self.clip_sp.value()),
            tileGridSize=(int(self.tile_sp.value()),
                          int(self.tile_sp.value())),
        )
        result = clahe.apply(gray)
        rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        pm = _cv2_to_qpixmap(rgb)
        self.image_lbl.setPixmap(pm.scaled(
            self.image_lbl.width(), self.image_lbl.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation))

    # ------------------------------------------------------------------ #
    # Accessors — read after exec() returns Accepted
    # ------------------------------------------------------------------ #
    @property
    def clip_limit(self) -> float:
        return float(self.clip_sp.value())

    @property
    def tile_size(self) -> int:
        return int(self.tile_sp.value())


# --------------------------------------------------------------------------- #
# _ClahePanel — small inline parameter widget shown in the form
# --------------------------------------------------------------------------- #
class _ClahePanel(QWidget):
    """CLAHE (contrast-limited adaptive histogram equalization)."""

    def __init__(self, parent: QWidget | None = None) -> None:
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

    def __init__(self, parent: QWidget | None = None) -> None:
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

    def __init__(self, parent: QWidget | None = None) -> None:
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
    def __init__(self, parent: QWidget | None = None) -> None:
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

    def __init__(self, parent: QWidget | None = None) -> None:
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

    def on_run(self) -> None:
        """Override to intercept the interactive-CLAHE-preview case.

        Patch 122ci. When `op == "clahe"` AND `params["interactive"]`
        is True, opens :class:`_ClahePreviewDialog` modally so the
        user can tune `clip_limit` / `tile_size` against a live
        preview frame. On Accept, mutates the kwargs with the
        dialog's final tuning and proceeds to the worker-thread
        dispatch. On Reject (user cancelled), aborts silently.

        All other ops fall through to the standard worker-thread
        path inherited from :class:`OperationForm`.
        """
        try:
            kwargs = self.collect_args()
        except Exception as exc:
            QMessageBox.warning(
                self, f"{self.title}: invalid input", str(exc))
            return

        params = kwargs.get("params") or {}
        if kwargs.get("op") == "clahe" and params.get("interactive"):
            # Resolve a sample video for the preview. In directory
            # scope, pick the first video in the directory — same
            # video the worker would hit first anyway.
            sample_path = kwargs["path"]
            if kwargs["is_dir"]:
                from mufasa.utils.read_write import find_all_videos_in_directory
                vids = find_all_videos_in_directory(
                    directory=sample_path, as_dict=True,
                    raise_error=True)
                sample_path = next(iter(vids.values()))
            try:
                dlg = _ClahePreviewDialog(
                    video_path=sample_path,
                    init_clip_limit=float(params["clip_limit"]),
                    init_tile_size=int(params["tile_grid"]),
                    parent=self.window(),
                )
            except Exception as exc:
                QMessageBox.critical(
                    self, "Could not open CLAHE preview",
                    f"{type(exc).__name__}: {exc}")
                return
            result = dlg.exec()
            if result != QDialog.Accepted:
                return  # user cancelled the preview
            # Mutate params with the dialog's final values; drop
            # the `interactive` flag so target()'s pop() never
            # sees True again.
            params["clip_limit"] = dlg.clip_limit
            params["tile_grid"] = dlg.tile_size
            params["interactive"] = False
            kwargs["params"] = params

        # Standard worker-thread dispatch (lifted from
        # OperationForm.on_run; can't call super().on_run()
        # because that would re-run collect_args).
        from mufasa.ui_qt.progress import run_with_progress

        def _work() -> None:
            self.target(**kwargs)

        run_with_progress(
            parent=self.window(),
            title=f"{self.title}…",
            target=_work,
            on_success=lambda: (
                self.completed.emit(),
                QMessageBox.information(self, self.title, "Done."),
            ),
        )

    def target(self, *, path: str, is_dir: bool, op: str,
               params: dict) -> None:
        from mufasa.video_processors import video_processing as _vp
        # Dispatch table. Where a backend doesn't match the kwargs
        # shape exactly, adapt here rather than in the panel.
        if op == "clahe":
            # Patch 122ci: interactive preview is handled by on_run
            # (opens a dialog on the main thread, then mutates
            # params to interactive=False before reaching target).
            # If interactive is still True here, it's a logic bug
            # in on_run — drop it defensively rather than raising.
            params.pop("interactive", None)
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
                from mufasa.utils.read_write import find_all_videos_in_directory
                for vp in find_all_videos_in_directory(
                        directory=path, raise_error=True):
                    _vp.video_to_bw(video_path=vp, threshold=threshold)
            else:
                _vp.video_to_bw(video_path=path, threshold=threshold)
        elif op == "blur":
            # Patch 122cb: rewired to new `video_blur` backend.
            # Form's `kernel_size` is the FFmpeg sigma / box radius.
            # `video_blur` handles file-or-dir internally; pass path
            # through directly.
            _vp.video_blur(video_path=path,
                            kernel_size=params["kernel_size"],
                            method="gaussian")
        elif op == "brightness_contrast":
            # Patch 122cb: rewired to new `video_brightness_contrast`
            # backend. Form's brightness range (−1..+1) and contrast
            # range (0..3) map directly to FFmpeg's `eq` filter args.
            _vp.video_brightness_contrast(
                video_path=path,
                brightness=params["brightness"],
                contrast=params["contrast"])


__all__ = ["VideoFiltersForm"]
