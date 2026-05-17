"""
mufasa.ui_qt.forms.blob_quick_check
====================================

Qt port of the legacy :class:`BlobQuickChecker` Tk interactive
tool (``mufasa/ui/blob_quick_check_interface.py``). Replaces the
Tk ``Toplevel`` window + Tk ``Label``-based PIL.ImageTk frame
display with a native Qt dialog.

What this tool does
-------------------
Visual confirmation that a video is amenable to blob tracking
*before* committing to the full tracker. The user picks a source
video and a background reference video; the dialog shows the
threshold-difference image (the frame minus the bg-video mean
frame, thresholded) so they can verify that the moving subject
shows up as a clean blob against a quiet background.

If the difference image is mostly noise, blob tracking won't
work and the user needs to retake the background or rethink
the recording setup. The "quick check" name reflects this
fail-fast spirit.

Architecture
------------
* :class:`BlobQuickCheckForm` — minimal parameter form (just
  the two video paths). Overrides
  :meth:`OperationForm.on_run` so the Run button opens the
  preview dialog on the Qt main thread instead of running a
  worker. Same pattern as :class:`ROIManageForm`'s "Draw"
  action.
* :class:`_BlobCheckDialog` — interactive preview. Live
  controls for method / threshold / kernel sizes /
  frame-index. Background average frame is computed once at
  dialog construction (synchronous; can take several seconds
  on a long bg video — see Caveats).

The Tk source's static-parameters model (method/threshold
fixed at construction) is intentionally NOT reproduced. The
whole point of a "quick check" is iterating on parameters
until the difference image looks right; making them live in
the dialog removes the need to close and reopen the tool.
"""
from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog,
                               QFileDialog, QFormLayout, QGridLayout,
                               QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QPushButton, QSlider,
                               QSpinBox, QVBoxLayout, QWidget)

from mufasa.ui_qt.workbench import OperationForm


# --------------------------------------------------------------------------- #
# Image conversion helper
# --------------------------------------------------------------------------- #
def _cv2_to_qpixmap(img):
    """Convert a CV2/numpy image (BGR or grayscale) to a QPixmap.

    The diff-image pipeline returns RGB (it converts grayscale →
    RGB before drawing inclusion zones); accept both colour and
    monochrome and dispatch on shape.
    """
    import numpy as np
    if img.ndim == 2:
        h, w = img.shape
        qimg = QImage(img.tobytes(), w, h, w, QImage.Format_Grayscale8)
    elif img.ndim == 3 and img.shape[2] == 3:
        h, w, _ = img.shape
        # The diff pipeline's `cv2.cvtColor(..., COLOR_GRAY2RGB)`
        # leaves us in RGB888; if a caller hands us BGR they'd
        # see channel swap, but BlobQuickChecker's output is
        # explicitly RGB.
        qimg = QImage(np.ascontiguousarray(img).tobytes(), w, h,
                      w * 3, QImage.Format_RGB888)
    else:
        raise ValueError(
            f"Unsupported image shape {img.shape} — expected "
            "2D grayscale or 3D RGB.")
    return QPixmap.fromImage(qimg)


# --------------------------------------------------------------------------- #
# _BlobCheckDialog — interactive viewer
# --------------------------------------------------------------------------- #
class _BlobCheckDialog(QDialog):
    """Modal-but-not-blocking preview window. Holds the average
    background frame and recomputes the diff image whenever any
    parameter changes.

    Backend hooks:
    * :func:`mufasa.video_processors.video_processing.create_average_frm`
      computes the bg-mean once at __init__.
    * :meth:`mufasa.mixins.image_mixin.ImageMixin.img_diff`
      computes the per-frame difference.
    * :func:`mufasa.utils.read_write.read_frm_of_video` reads
      individual frames on demand.
    """

    def __init__(self, video_path: str, bg_video_path: str,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Blob quick-check — preview")
        self.video_path = video_path
        self.bg_video_path = bg_video_path

        # Lazy-import the backend pieces so this module can be
        # AST-parsed without cv2/PIL/etc. on the path. They're
        # required to ACTUALLY render, but not to define the form.
        from mufasa.utils.read_write import (get_video_meta_data,
                                             read_frm_of_video)
        from mufasa.video_processors.video_processing import (
            create_average_frm,
        )
        self._read_frm = read_frm_of_video
        self._get_meta = get_video_meta_data
        self._create_avg = create_average_frm

        self.video_meta = get_video_meta_data(video_path=video_path)
        bg_meta = get_video_meta_data(video_path=bg_video_path)
        if self.video_meta["resolution_str"] != bg_meta["resolution_str"]:
            raise ValueError(
                f"Source video and background reference have "
                f"different resolutions: "
                f"{self.video_meta['resolution_str']} vs "
                f"{bg_meta['resolution_str']}.")

        self.frame_count = int(self.video_meta["frame_count"])
        self.fps = float(self.video_meta["fps"]) or 30.0
        self.img_idx = 0
        self.avg_frm = None  # populated below

        self._build_ui()

        # Compute the average background. This is synchronous; for
        # long videos it can take several seconds. Acceptable for
        # v1 — a future patch could move it to a QThread + show a
        # progress spinner, but the existing OperationForm worker
        # machinery isn't structured around in-dialog operations.
        self._set_status(f"Computing background for "
                          f"{self.video_meta['video_name']}…",
                          color="blue")
        # Force repaint so the status text appears before the
        # blocking compute kicks in.
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()
        self.avg_frm = create_average_frm(
            video_path=self.bg_video_path, verbose=False)
        self._set_status(
            f"Background ready — showing frame {self.img_idx}.",
            color="green")
        self._refresh_image()

    # ------------------------------------------------------------------ #
    # UI
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        # ----- Image display ----- #
        self.image_lbl = QLabel(self)
        self.image_lbl.setAlignment(Qt.AlignCenter)
        self.image_lbl.setMinimumSize(640, 480)
        root.addWidget(self.image_lbl, 1)

        # ----- Status ----- #
        self.status_lbl = QLabel("", self)
        self.status_lbl.setWordWrap(True)
        root.addWidget(self.status_lbl)

        # ----- Tunables ----- #
        params = QFormLayout()
        params.setLabelAlignment(Qt.AlignRight)

        self.method_cb = QComboBox(self)
        self.method_cb.addItems(["absolute", "light", "dark"])
        self.method_cb.currentIndexChanged.connect(self._on_params_changed)
        params.addRow("Difference method:", self.method_cb)

        self.threshold_sp = QSpinBox(self)
        self.threshold_sp.setRange(1, 255)
        self.threshold_sp.setValue(70)
        self.threshold_sp.valueChanged.connect(self._on_params_changed)
        params.addRow("Threshold (1–255):", self.threshold_sp)

        # Close / open kernel toggles + sizes. The Tk version
        # accepts a tuple; we surface a single integer N which we
        # expand to (N, N). 0 disables the kernel.
        self.close_chk = QCheckBox("Apply morphological CLOSE", self)
        self.close_chk.toggled.connect(self._on_params_changed)
        params.addRow("", self.close_chk)
        close_row = QHBoxLayout()
        self.close_size_sp = QSpinBox(self)
        self.close_size_sp.setRange(1, 99); self.close_size_sp.setValue(5)
        self.close_size_sp.valueChanged.connect(self._on_params_changed)
        close_row.addWidget(QLabel("size:", self))
        close_row.addWidget(self.close_size_sp)
        self.close_iters_sp = QSpinBox(self)
        self.close_iters_sp.setRange(1, 20); self.close_iters_sp.setValue(3)
        self.close_iters_sp.valueChanged.connect(self._on_params_changed)
        close_row.addWidget(QLabel(" iterations:", self))
        close_row.addWidget(self.close_iters_sp)
        close_row.addStretch(1)
        params.addRow("Close params:", close_row)

        self.open_chk = QCheckBox("Apply morphological OPEN", self)
        self.open_chk.toggled.connect(self._on_params_changed)
        params.addRow("", self.open_chk)
        open_row = QHBoxLayout()
        self.open_size_sp = QSpinBox(self)
        self.open_size_sp.setRange(1, 99); self.open_size_sp.setValue(5)
        self.open_size_sp.valueChanged.connect(self._on_params_changed)
        open_row.addWidget(QLabel("size:", self))
        open_row.addWidget(self.open_size_sp)
        self.open_iters_sp = QSpinBox(self)
        self.open_iters_sp.setRange(1, 20); self.open_iters_sp.setValue(3)
        self.open_iters_sp.valueChanged.connect(self._on_params_changed)
        open_row.addWidget(QLabel(" iterations:", self))
        open_row.addWidget(self.open_iters_sp)
        open_row.addStretch(1)
        params.addRow("Open params:", open_row)

        root.addLayout(params)

        # ----- Frame nav ----- #
        nav_row = QHBoxLayout()

        self.frame_slider = QSlider(Qt.Horizontal, self)
        self.frame_slider.setRange(0, max(0, self.frame_count - 1))
        self.frame_slider.setValue(0)
        self.frame_slider.valueChanged.connect(self._on_slider)
        nav_row.addWidget(self.frame_slider, 1)

        self.frame_lbl = QLabel(f"frame 0 / {self.frame_count - 1}", self)
        nav_row.addWidget(self.frame_lbl)
        root.addLayout(nav_row)

        step_row = QHBoxLayout()
        for label, stride in [
            ("⏮ First",  "first"),
            ("-1s",       -int(self.fps)),
            ("-N s",      "custom_back"),
            ("+N s",      "custom_fwd"),
            ("+1s",       int(self.fps)),
            ("⏭ Last",   "last"),
        ]:
            btn = QPushButton(label, self)
            btn.clicked.connect(
                lambda _checked=False, s=stride: self._step(s))
            step_row.addWidget(btn)

        step_row.addStretch(1)
        step_row.addWidget(QLabel("N seconds:", self))
        self.custom_s_sp = QSpinBox(self)
        self.custom_s_sp.setRange(1, 99999)
        self.custom_s_sp.setValue(10)
        step_row.addWidget(self.custom_s_sp)
        root.addLayout(step_row)

        # ----- Close button ----- #
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        close_btn = QPushButton("Close", self)
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

    # ------------------------------------------------------------------ #
    # Slots
    # ------------------------------------------------------------------ #
    def _on_params_changed(self, *_) -> None:
        self._refresh_image()

    def _on_slider(self, idx: int) -> None:
        self.img_idx = int(idx)
        self.frame_lbl.setText(
            f"frame {self.img_idx} / {self.frame_count - 1}")
        self._refresh_image()

    def _step(self, stride) -> None:
        if stride == "first":
            new_idx = 0
        elif stride == "last":
            new_idx = self.frame_count - 1
        elif stride == "custom_fwd":
            new_idx = self.img_idx + int(self.custom_s_sp.value() * self.fps)
        elif stride == "custom_back":
            new_idx = self.img_idx - int(self.custom_s_sp.value() * self.fps)
        else:
            new_idx = self.img_idx + int(stride)
        new_idx = max(0, min(self.frame_count - 1, new_idx))
        # setValue triggers _on_slider which updates img_idx + refresh
        self.frame_slider.setValue(new_idx)

    # ------------------------------------------------------------------ #
    # Image pipeline
    # ------------------------------------------------------------------ #
    def _refresh_image(self) -> None:
        if self.avg_frm is None:
            return
        import cv2
        from mufasa.mixins.image_mixin import ImageMixin

        try:
            img = self._read_frm(video_path=self.video_path,
                                 frame_index=self.img_idx)
        except Exception as exc:
            self._set_status(f"Could not read frame: {exc}",
                              color="red")
            return

        method = self.method_cb.currentText()
        threshold = int(self.threshold_sp.value())

        try:
            diff = ImageMixin.img_diff(
                x=img, y=self.avg_frm,
                threshold=threshold, method=method,
            )
        except Exception as exc:
            self._set_status(f"img_diff failed: {exc}", color="red")
            return

        diff = cv2.cvtColor(diff, cv2.COLOR_GRAY2RGB)

        if self.open_chk.isChecked():
            n = int(self.open_size_sp.value())
            iters = int(self.open_iters_sp.value())
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
            diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, k,
                                    iterations=iters)
        if self.close_chk.isChecked():
            n = int(self.close_size_sp.value())
            iters = int(self.close_iters_sp.value())
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
            diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, k,
                                    iterations=iters)

        pm = _cv2_to_qpixmap(diff)
        # Scale down for display while keeping aspect.
        self.image_lbl.setPixmap(pm.scaled(
            self.image_lbl.width(), self.image_lbl.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _set_status(self, text: str, color: str = "blue") -> None:
        self.status_lbl.setText(text)
        self.status_lbl.setStyleSheet(f"color: {color};")


# --------------------------------------------------------------------------- #
# BlobQuickCheckForm — parameter entry
# --------------------------------------------------------------------------- #
class BlobQuickCheckForm(OperationForm):
    """Open the interactive blob quick-check preview.

    Replaces the legacy Tk :class:`BlobQuickChecker` tool.
    """

    title = "Blob quick-check"
    description = (
        "Visualise the threshold-difference image used by blob "
        "tracking so you can verify a video is trackable before "
        "running the full pipeline. Select a source video and a "
        "background reference video, then explore method / "
        "threshold / morphological kernel parameters live in the "
        "preview dialog."
    )

    def build(self) -> None:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.video_path_le = QLineEdit(self)
        self.video_path_le.setPlaceholderText(
            "Select the video to check…")
        v_row = QHBoxLayout()
        v_row.addWidget(self.video_path_le)
        v_btn = QPushButton("Browse…", self)
        v_btn.clicked.connect(self._browse_video)
        v_row.addWidget(v_btn)
        form.addRow("Source video:", v_row)

        self.bg_path_le = QLineEdit(self)
        self.bg_path_le.setPlaceholderText(
            "Select the background reference video…")
        b_row = QHBoxLayout()
        b_row.addWidget(self.bg_path_le)
        b_btn = QPushButton("Browse…", self)
        b_btn.clicked.connect(self._browse_bg)
        b_row.addWidget(b_btn)
        form.addRow("Background reference:", b_row)

        self.body_layout.addLayout(form)

    def _browse_video(self) -> None:
        p, _ = QFileDialog.getOpenFileName(
            self, "Select video", "",
            "Video files (*.mp4 *.avi *.mov *.mkv *.webm);;"
            "All files (*)")
        if p:
            self.video_path_le.setText(p)

    def _browse_bg(self) -> None:
        p, _ = QFileDialog.getOpenFileName(
            self, "Select background reference video", "",
            "Video files (*.mp4 *.avi *.mov *.mkv *.webm);;"
            "All files (*)")
        if p:
            self.bg_path_le.setText(p)

    # ------------------------------------------------------------------ #
    # on_run override: open the interactive dialog on the Qt main
    # thread instead of running through the worker-thread machinery.
    # The dialog itself is a viewer, not a finite operation, so the
    # standard "collect_args → target() in worker" flow doesn't fit.
    # Same pattern as ROIManageForm's "draw" action.
    # ------------------------------------------------------------------ #
    def collect_args(self) -> dict:
        video_path = self.video_path_le.text().strip()
        bg_path = self.bg_path_le.text().strip()
        if not video_path:
            raise ValueError("Source video path is required.")
        if not bg_path:
            raise ValueError(
                "Background reference video path is required.")
        if not os.path.isfile(video_path):
            raise ValueError(
                f"Source video not found: {video_path}")
        if not os.path.isfile(bg_path):
            raise ValueError(
                f"Background reference video not found: {bg_path}")
        return {"video_path": video_path, "bg_video_path": bg_path}

    def on_run(self) -> None:
        try:
            kwargs = self.collect_args()
        except Exception as exc:
            QMessageBox.warning(
                self, f"{self.title}: invalid input", str(exc))
            return
        top = self.window()
        try:
            dlg = _BlobCheckDialog(
                video_path=kwargs["video_path"],
                bg_video_path=kwargs["bg_video_path"],
                parent=top,
            )
        except Exception as exc:
            QMessageBox.critical(
                self, "Could not open blob quick-check",
                f"{type(exc).__name__}: {exc}",
            )
            return
        dlg.show()
        # Stash on the workbench so it isn't GC'd before the user
        # closes it.
        top._dialog_refs = getattr(top, "_dialog_refs", [])
        top._dialog_refs.append(dlg)

    def target(self, **_kwargs) -> None:
        # Not used — on_run handles dispatch directly.
        raise RuntimeError(
            "BlobQuickCheckForm.target() should not be called; "
            "on_run opens the dialog directly.")


__all__ = ["BlobQuickCheckForm"]
