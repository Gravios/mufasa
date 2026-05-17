"""
mufasa.ui_qt.forms.video_utilities
==================================

Small inline forms for single-utility video operations.

* :class:`ReverseVideoForm` — reverse playback of one or many
  videos. Replaces :class:`ReverseVideoPopUp`.
* :class:`ChangeSpeedForm` — change playback speed (0.1×–160×) of
  one or many videos. Replaces :class:`ChangeSpeedPopup`.
* :class:`PixelsPerMMForm` — interactive pixel-to-millimetre
  calibration. Replaces :class:`CalculatePixelsPerMMInVideoPopUp`.
  The calibration UI itself is the legacy
  :class:`GetPixelsPerMillimeterInterface`, an OpenCV-window
  click-to-pick affair — this form just orchestrates the
  inputs (video path + known distance) and reports the result.
* :class:`CheckVideoSeekableForm` — verify that frame seeks work
  correctly on a video or directory of videos. Replaces
  :class:`CheckVideoSeekablePopUp` (patch 122bx).

Each form was previously a single Tk popup. Patch 122u
consolidates the first three; 122bx adds the fourth.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QHBoxLayout,
                               QLabel, QLineEdit, QMessageBox,
                               QPushButton, QSpinBox)

from mufasa.ui_qt.forms.video_processing import _ScopePicker
from mufasa.ui_qt.workbench import OperationForm


# =========================================================================== #
# ReverseVideoForm
# =========================================================================== #
class ReverseVideoForm(OperationForm):
    """Reverse playback of one or many videos.

    Replaces :class:`ReverseVideoPopUp`. Backend:
    :func:`mufasa.video_processors.video_processing.reverse_videos`,
    which accepts either a file or a directory via its ``path``
    parameter.
    """

    title = "Reverse video(s)"
    description = (
        "Produce a reversed-playback copy of one or many videos. "
        "The backend writes alongside the source by default; pick "
        "a save directory below to override."
    )

    def build(self) -> None:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.scope = _ScopePicker(self)
        form.addRow("Source:", self.scope)

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
        form.addRow("Save directory:", sd_row)

        self.quality = QSpinBox(self)
        self.quality.setRange(10, 100)
        self.quality.setSingleStep(10)
        self.quality.setValue(60)
        self.quality.setSuffix(" %")
        form.addRow("Output quality:", self.quality)

        self.gpu = QCheckBox(
            "Use GPU encoder if available", self,
        )
        form.addRow("", self.gpu)

        self.body_layout.addLayout(form)

    def _pick_save_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Pick a save directory", "",
        )
        if d:
            self.save_dir_edit.setText(d)

    def collect_args(self) -> dict:
        if not self.scope.path:
            raise ValueError("No source selected.")
        return {
            "path":     self.scope.path,
            "save_dir": self.save_dir_edit.text().strip() or None,
            "quality":  int(self.quality.value()),
            "gpu":      bool(self.gpu.isChecked()),
        }

    def target(self, *, path: str, save_dir: Optional[str],
               quality: int, gpu: bool) -> None:
        from mufasa.video_processors import video_processing as _vp
        _vp.reverse_videos(
            path=path, save_dir=save_dir,
            quality=quality, gpu=gpu,
        )


# =========================================================================== #
# ChangeSpeedForm
# =========================================================================== #
class ChangeSpeedForm(OperationForm):
    """Change playback speed of one or many videos.

    Replaces :class:`ChangeSpeedPopup`. ``speed > 1.0`` speeds up;
    ``speed < 1.0`` slows down. Backends:
    :func:`change_playback_speed` (single video) and
    :func:`change_playback_speed_dir` (directory) — picked by the
    scope toggle.
    """

    title = "Change playback speed"
    description = (
        "Re-encode one or many videos at a different playback speed. "
        "Values > 1.0 speed up; values < 1.0 slow down. Audio (if "
        "present) is re-timed by the backend."
    )

    def build(self) -> None:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.scope = _ScopePicker(self)
        form.addRow("Source:", self.scope)

        # Speed: 0.1× through 160× in 0.1 steps (matches legacy
        # SPEED_OPTIONS range). QDoubleSpinBox gives precise control.
        self.speed = QDoubleSpinBox(self)
        self.speed.setRange(0.1, 160.0)
        self.speed.setSingleStep(0.1)
        self.speed.setDecimals(1)
        self.speed.setValue(1.5)
        self.speed.setSuffix("×")
        form.addRow("Speed:", self.speed)

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
        form.addRow("Save directory:", sd_row)

        self.quality = QSpinBox(self)
        self.quality.setRange(10, 100)
        self.quality.setSingleStep(10)
        self.quality.setValue(60)
        self.quality.setSuffix(" %")
        form.addRow("Output quality:", self.quality)

        self.gpu = QCheckBox(
            "Use GPU encoder if available", self,
        )
        form.addRow("", self.gpu)

        self.body_layout.addLayout(form)

    def _pick_save_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Pick a save directory", "",
        )
        if d:
            self.save_dir_edit.setText(d)

    def collect_args(self) -> dict:
        if not self.scope.path:
            raise ValueError("No source selected.")
        speed = float(self.speed.value())
        if speed == 1.0:
            raise ValueError(
                "Speed = 1.0× is a no-op. Pick something else.",
            )
        return {
            "path":     self.scope.path,
            "is_dir":   self.scope.is_dir,
            "speed":    speed,
            "save_dir": self.save_dir_edit.text().strip() or None,
            "quality":  int(self.quality.value()),
            "gpu":      bool(self.gpu.isChecked()),
        }

    def target(self, *, path: str, is_dir: bool, speed: float,
               save_dir: Optional[str], quality: int,
               gpu: bool) -> None:
        from mufasa.video_processors import video_processing as _vp
        if is_dir:
            _vp.change_playback_speed_dir(
                data_dir=path, speed=speed, save_dir=save_dir,
                quality=quality, gpu=gpu,
            )
        else:
            # change_playback_speed takes save_path (a file) not
            # save_dir (a directory). Derive when save_dir picked.
            if save_dir:
                stem = Path(path).stem
                ext = Path(path).suffix or ".mp4"
                save_path = str(
                    Path(save_dir) / f"{stem}_speed_{speed}{ext}"
                )
            else:
                save_path = None
            _vp.change_playback_speed(
                video_path=path, speed=speed, save_path=save_path,
                quality=quality, gpu=gpu,
            )


# =========================================================================== #
# PixelsPerMMForm
# =========================================================================== #
class PixelsPerMMForm(OperationForm):
    """Calibrate pixels-per-millimetre by clicking two points of
    known real-world distance.

    Replaces :class:`CalculatePixelsPerMMInVideoPopUp`. The
    calibration UI itself is the legacy
    :class:`mufasa.ui.px_to_mm_ui.GetPixelsPerMillimeterInterface`,
    an OpenCV-window click-to-pick affair. This form supplies its
    inputs and reports the resulting ratio.

    Output is printed to stdout (matching the legacy popup's
    behaviour) and surfaced as a QMessageBox when the OpenCV
    window closes so the user sees the value without scanning a
    terminal.
    """

    title = "Calculate pixels per millimetre"
    description = (
        "Click two points of known real-world distance in a video "
        "frame; this form converts the pixel distance to a "
        "pixels-per-mm ratio you can record in the project's "
        "video metadata."
    )

    def build(self) -> None:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.video_path_edit = QLineEdit(self)
        self.video_path_edit.setReadOnly(True)
        self.video_path_edit.setPlaceholderText("Pick a video file…")
        vp_browse = QPushButton("Browse…", self)
        vp_browse.clicked.connect(self._pick_video)
        vp_row = QHBoxLayout()
        vp_row.addWidget(self.video_path_edit, 1)
        vp_row.addWidget(vp_browse)
        form.addRow("Video:", vp_row)

        self.known_mm = QDoubleSpinBox(self)
        self.known_mm.setRange(0.001, 100000.0)
        self.known_mm.setDecimals(3)
        self.known_mm.setSingleStep(1.0)
        self.known_mm.setValue(100.0)
        self.known_mm.setSuffix(" mm")
        form.addRow("Known distance:", self.known_mm)

        hint = QLabel(
            "<i>An OpenCV window will open showing the first frame "
            "of the video. Click two points spanning the known "
            "distance; close the window when done.</i>",
            self,
        )
        hint.setTextFormat(Qt.RichText)
        hint.setWordWrap(True)
        hint.setStyleSheet("color: palette(placeholder-text);")
        form.addRow("", hint)

        self.body_layout.addLayout(form)

    def _pick_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Pick a video", "",
            "Video files (*.mp4 *.avi *.mov *.mkv *.webm);;"
            "All files (*)",
        )
        if path:
            self.video_path_edit.setText(path)

    def collect_args(self) -> dict:
        v = self.video_path_edit.text().strip()
        if not v:
            raise ValueError("Pick a video file.")
        if not Path(v).is_file():
            raise ValueError(f"Not a file: {v}")
        return {
            "video_path":     v,
            "known_metric_mm": float(self.known_mm.value()),
        }

    def target(self, *, video_path: str,
               known_metric_mm: float) -> None:
        # The interactive UI pops its own OpenCV window; the form
        # blocks until the user closes it. Print + dialog the
        # result.
        from mufasa.ui.px_to_mm_ui import (
            GetPixelsPerMillimeterInterface,
        )
        iface = GetPixelsPerMillimeterInterface(
            video_path=video_path,
            known_metric_mm=known_metric_mm,
        )
        iface.run()
        ppm = float(iface.ppm)
        name = os.path.basename(video_path)
        msg = (
            f"One (1) pixel represents {ppm:.4f} millimetres in "
            f"video {name}."
        )
        print(msg)
        QMessageBox.information(
            self, "Calibration result", msg,
        )


__all__ = [
    "ReverseVideoForm",
    "ChangeSpeedForm",
    "PixelsPerMMForm",
    "CheckVideoSeekableForm",
]


# =========================================================================== #
# CheckVideoSeekableForm
# =========================================================================== #
class CheckVideoSeekableForm(OperationForm):
    """Verify that frame seeks work correctly across a video or
    directory of videos.

    Replaces :class:`CheckVideoSeekablePopUp` (patch 122bx).
    Backend:
    :func:`mufasa.video_processors.video_processing.is_video_seekable`.

    The backend writes a CSV report listing every video plus a
    pass/fail flag and any error encountered. The report path
    defaults to ``Desktop`` (falling back to ``Downloads`` if the
    Desktop directory is unavailable, matching the Tk popup's
    behaviour) with a timestamped filename. The user can override
    by selecting a save path explicitly.
    """

    title = "Check video seekability"
    description = (
        "Test whether videos support reliable frame-seek "
        "operations. Slow CFR sources or some VFR re-encodes can "
        "silently mis-seek; this scan catches them. Output is a "
        "CSV with one row per video flagged pass / fail."
    )

    def build(self) -> None:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.scope = _ScopePicker(self)
        form.addRow("Source:", self.scope)

        self.gpu = QCheckBox("Use GPU (faster on large videos)", self)
        form.addRow("", self.gpu)

        # Batch size 0 → no batching (load all frames). >0 → process
        # `batch_size` frames at a time. Default mirrors the Tk
        # popup's default of 400 (a balance between RAM and IO).
        self.batch_size = QSpinBox(self)
        self.batch_size.setRange(0, 5000)
        self.batch_size.setSingleStep(100)
        self.batch_size.setValue(400)
        self.batch_size.setSpecialValueText("Disabled — load all frames")
        form.addRow("Frame batch size:", self.batch_size)

        # Optional save path. Empty → defaults to Desktop /
        # Downloads + timestamped filename in target().
        self.save_path_edit = QLineEdit(self)
        self.save_path_edit.setPlaceholderText(
            "Optional — defaults to Desktop / Downloads with "
            "timestamped filename")
        sp_browse = QPushButton("Browse…", self)
        sp_browse.clicked.connect(self._pick_save_path)
        sp_row = QHBoxLayout()
        sp_row.addWidget(self.save_path_edit, 1)
        sp_row.addWidget(sp_browse)
        form.addRow("Save CSV to:", sp_row)

        self.body_layout.addLayout(form)

    def _pick_save_path(self) -> None:
        p, _ = QFileDialog.getSaveFileName(
            self, "Save seekability report as", "",
            "CSV files (*.csv);;All files (*)",
        )
        if p:
            self.save_path_edit.setText(p)

    def collect_args(self) -> dict:
        if not self.scope.path:
            raise ValueError("No source selected.")
        bs = int(self.batch_size.value())
        return {
            "path":       self.scope.path,
            "gpu":        bool(self.gpu.isChecked()),
            # 0 = disabled in the UI; pass None to the backend
            # which interprets that as "no batching".
            "batch_size": bs if bs > 0 else None,
            "save_path":  self.save_path_edit.text().strip() or None,
        }

    def target(self, *, path: str, gpu: bool,
               batch_size: Optional[int],
               save_path: Optional[str]) -> None:
        from datetime import datetime

        from mufasa.utils.read_write import (get_desktop_path,
                                             get_downloads_path)
        from mufasa.video_processors.video_processing import (
            is_video_seekable,
        )

        if save_path is None:
            save_dir = get_desktop_path()
            if save_dir is None:
                save_dir = get_downloads_path(raise_error=True)
            stamp = datetime.now().strftime("%Y%m%d%H%M%S")
            save_path = os.path.join(
                save_dir, f"seekability_test_{stamp}.csv")
        is_video_seekable(
            data_path=path, gpu=gpu, batch_size=batch_size,
            verbose=False, save_path=save_path,
        )
