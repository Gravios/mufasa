"""
mufasa.ui_qt.forms.video_frames
===============================

Inline form for the frame-extraction popups:

* :class:`ExtractAllFramesPopUp`
* :class:`ExtractSpecificFramesPopUp`
* :class:`ExtractSEQFramesPopUp` (Norpix SEQ format)
* :class:`SingleVideo2FramesPopUp`
* :class:`MultipleVideos2FramesPopUp`

The "range" field distinguishes "extract all" from "extract specific";
the scope picker handles single-video vs. directory; the SEQ format is
picked up automatically from the file extension.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QLabel,
                               QLineEdit, QSpinBox, QWidget)

from mufasa.ui_qt.forms.video_processing import _ScopePicker
from mufasa.ui_qt.workbench import OperationForm


class ExtractFramesForm(OperationForm):
    """Extract frames from one or more videos.

    Replaces :class:`ExtractAllFramesPopUp`,
    :class:`ExtractSpecificFramesPopUp`, :class:`ExtractSEQFramesPopUp`,
    :class:`SingleVideo2FramesPopUp`, :class:`MultipleVideos2FramesPopUp`.

    The range fields are optional — leave them blank to extract every
    frame (subsumes Extract*All* popups). SEQ format is auto-detected
    from file extension.
    """

    title = "Extract frames from video(s)"
    description = ("Extract frames to an image directory. Leave the "
                   "frame range blank to extract every frame.")

    def build(self) -> None:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.scope = _ScopePicker(
            self,
            file_filter=(
                "Videos & SEQ (*.mp4 *.avi *.mov *.mkv *.webm *.seq);;All files (*)"
            ),
        )
        form.addRow("Source:", self.scope)

        self.start_frame = QSpinBox(self)
        self.start_frame.setRange(0, 10_000_000)
        self.start_frame.setSpecialValueText("(from start)")
        form.addRow("Start frame (optional):", self.start_frame)

        self.end_frame = QSpinBox(self)
        self.end_frame.setRange(0, 10_000_000)
        self.end_frame.setSpecialValueText("(to end)")
        form.addRow("End frame (optional):", self.end_frame)

        self.fmt_cb = QComboBox(self)
        self.fmt_cb.addItems(["PNG", "JPEG", "TIFF", "BMP"])
        form.addRow("Output image format:", self.fmt_cb)

        self.body_layout.addLayout(form)

    def collect_args(self) -> dict:
        if not self.scope.path:
            raise ValueError("No source selected.")
        start = int(self.start_frame.value()) if self.start_frame.value() else None
        end = int(self.end_frame.value()) if self.end_frame.value() else None
        if start is not None and end is not None and start >= end:
            raise ValueError(f"Start frame ({start}) must be < end ({end}).")
        return {
            "path":   self.scope.path,
            "is_dir": self.scope.is_dir,
            "start":  start,
            "end":    end,
            "fmt":    self.fmt_cb.currentText().lower(),
        }

    def target(self, *, path: str, is_dir: bool, start: Optional[int],
               end: Optional[int], fmt: str) -> None:
        from pathlib import Path as _P
        from mufasa.video_processors import video_processing as _vp
        # SEQ detection by extension
        is_seq = (not is_dir) and path.lower().endswith(".seq")
        if is_seq:
            raise NotImplementedError(
                "SEQ frame extraction: backend wiring pending.")

        def _frames_dir(video_path: str) -> str:
            # Sibling directory `{stem}_frames/` next to the video.
            # The backend requires save_dir (no default) — derive here.
            p = _P(video_path)
            out = p.parent / f"{p.stem}_frames"
            out.mkdir(parents=True, exist_ok=True)
            return str(out)

        def _extract_one(vp: str) -> None:
            if start is None and end is None:
                _vp.extract_frames_single_video(file_path=vp,
                                                save_dir=_frames_dir(vp))
            else:
                _vp.extract_frame_range(
                    file_path=vp,
                    start_frame=start or 0,
                    end_frame=end or 10_000_000,
                )

        if is_dir:
            exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
            for vp in sorted(_P(path).iterdir()):
                if vp.suffix.lower() in exts:
                    _extract_one(str(vp))
        else:
            _extract_one(path)


__all__ = ["ExtractFramesForm"]
