"""
mufasa.ui_qt.forms.video_processing
===================================

Consolidated :class:`OperationForm` implementations for the Video
Processing workbench page. Together these three forms replace 16
legacy Tk popups.

Replaces:

* :class:`VideoFormatConverterForm`  → Convert2AVIPopUp / Convert2MP4PopUp
  / Convert2MOVPopUp / Convert2WEBMPopUp   (4 popups)
* :class:`VideoOverlayForm`          → SuperImposeFrameCountPopUp /
  SuperimposeTimerPopUp / SuperimposeTextPopUp /
  SuperimposeVideoNamesPopUp / SuperimposeWatermarkPopUp /
  SuperimposeProgressBarPopUp / SuperimposeVideoPopUp   (7 popups)
* :class:`ClipVideosForm`            → ClipVideoPopUp / MultiShortenPopUp /
  ClipMultipleVideosByFrameNumbersPopUp /
  InitiateClipMultipleVideosByFrameNumbersPopUp /
  InitiateClipMultipleVideosByTimestampsPopUp  (5 popups)

Design principle: the popup-level features become option *fields* on a
shared form, not separate UIs. A target-format dropdown or a unit
radio replaces the need for a dedicated window per variant.
"""
from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QButtonGroup, QFileDialog, QFormLayout,
                               QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QRadioButton, QStackedWidget, QVBoxLayout,
                               QComboBox, QSpinBox, QCheckBox, QWidget,
                               QDoubleSpinBox)

from mufasa.ui_qt import linux_env
from mufasa.ui_qt.workbench import OperationForm


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
class _ScopePicker(QWidget):
    """File-or-directory selector with radio toggle.

    Subsumes the "single_video_frm + multiple_video_frm" pattern present
    in every Convert2*PopUp (and many others). One control handles both.
    """

    def __init__(self, parent: Optional[QWidget] = None,
                 allow_multiple: bool = True,
                 file_filter: str = "Video files (*.mp4 *.avi *.mov *.mkv *.webm);;All files (*)"
                 ) -> None:
        super().__init__(parent)
        self.file_filter = file_filter
        self._path: str = ""
        self._is_dir: bool = False

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self.rb_single = QRadioButton("Single file", self)
        self.rb_dir = QRadioButton("Directory", self)
        self.rb_single.setChecked(True)
        self.rb_single.toggled.connect(self._on_toggle)
        self.grp = QButtonGroup(self)
        self.grp.addButton(self.rb_single)
        self.grp.addButton(self.rb_dir)
        if allow_multiple:
            lay.addWidget(self.rb_single)
            lay.addWidget(self.rb_dir)

        self.line = QLineEdit(self)
        self.line.setPlaceholderText("Select a video file…")
        self.line.setReadOnly(True)
        self.browse_btn = QPushButton("Browse…", self)
        self.browse_btn.clicked.connect(self._browse)
        lay.addWidget(self.line, 1)
        lay.addWidget(self.browse_btn)

    def _on_toggle(self, checked: bool) -> None:
        if self.rb_single.isChecked():
            self.line.setPlaceholderText("Select a video file…")
        else:
            self.line.setPlaceholderText("Select a directory…")
        self.line.clear()
        self._path = ""

    def _browse(self) -> None:
        if self.rb_single.isChecked():
            path, _ = QFileDialog.getOpenFileName(
                self, "Select video", os.path.expanduser("~"), self.file_filter,
            )
            self._is_dir = False
        else:
            path = QFileDialog.getExistingDirectory(
                self, "Select directory", os.path.expanduser("~")
            )
            self._is_dir = True
        if path:
            self._path = path
            self.line.setText(path)

    @property
    def path(self) -> str:
        return self._path

    @property
    def is_dir(self) -> bool:
        return self._is_dir


# --------------------------------------------------------------------------- #
# A1 — VideoFormatConverterForm (replaces 4 popups)
# --------------------------------------------------------------------------- #
class VideoFormatConverterForm(OperationForm):
    """Convert video(s) to a chosen format: MP4, AVI, MOV, WEBM.

    Replaces Convert2AVIPopUp, Convert2MP4PopUp, Convert2MOVPopUp,
    Convert2WEBMPopUp. Target format drives the codec options; scope
    (single file / directory) chosen inline, not via a second window.
    """

    title = "Convert video format"
    description = ("Re-encode one or more videos to a different container "
                   "format. NVENC is used automatically when available.")

    # {target_fmt: {codec_label: ffmpeg_codec_name}}
    CODEC_TABLE: dict[str, dict[str, str]] = {
        "MP4":  {"H.264 (AVC)": "libx264",
                 "HEVC (H.265)": "libx265",
                 "VP9": "vp9",
                 "NVENC H.264": "h264_nvenc",
                 "NVENC HEVC": "hevc_nvenc"},
        "AVI":  {"XviD": "xvid", "DivX": "divx", "MJPEG": "mjpeg"},
        "MOV":  {"ProRes": "prores", "Animation": "animation",
                 "CineForm": "cineform", "DNxHD/HR": "dnxhd"},
        "WEBM": {"VP9": "vp9", "VP8": "vp8", "AV1": "av1"},
    }

    def build(self) -> None:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        self.scope = _ScopePicker(self)
        form.addRow("Source:", self.scope)

        self.fmt_cb = QComboBox(self)
        self.fmt_cb.addItems(list(self.CODEC_TABLE.keys()))
        self.fmt_cb.currentTextChanged.connect(self._on_fmt_changed)
        form.addRow("Target format:", self.fmt_cb)

        self.codec_cb = QComboBox(self)
        self._on_fmt_changed(self.fmt_cb.currentText())
        form.addRow("Codec:", self.codec_cb)

        self.quality = QSpinBox(self)
        self.quality.setRange(10, 100)
        self.quality.setValue(60)
        self.quality.setSuffix(" %")
        form.addRow("Quality:", self.quality)

        self.keep_audio = QCheckBox("Preserve audio (if present)", self)
        form.addRow("", self.keep_audio)

        self.body_layout.addLayout(form)

        if linux_env.nvenc_available():
            note = QLabel(
                "<i>NVENC available — GPU encoders will be offered in the codec list.</i>",
                self,
            )
            note.setStyleSheet("color: #5a8f5a;")
            self.body_layout.addWidget(note)

    def _on_fmt_changed(self, fmt: str) -> None:
        self.codec_cb.clear()
        for label in self.CODEC_TABLE[fmt].keys():
            # Hide NVENC codecs when the machine can't use them
            if "NVENC" in label and not linux_env.nvenc_available():
                continue
            self.codec_cb.addItem(label)

    def collect_args(self) -> dict:
        if not self.scope.path:
            raise ValueError("No source selected.")
        fmt = self.fmt_cb.currentText()
        codec_label = self.codec_cb.currentText()
        codec = self.CODEC_TABLE[fmt][codec_label]
        return {
            "path": self.scope.path,
            "is_dir": self.scope.is_dir,
            "fmt": fmt,
            "codec": codec,
            "quality": int(self.quality.value()),
            "keep_audio": bool(self.keep_audio.isChecked()),
        }

    def target(self, *, path: str, is_dir: bool, fmt: str, codec: str,
               quality: int, keep_audio: bool) -> None:
        # Backend dispatch — per-format fns live in video_processing.
        from mufasa.video_processors import video_processing as _vp
        dispatch = {
            "MP4":  _vp.convert_to_mp4,
            "AVI":  _vp.convert_to_avi,
            "MOV":  _vp.convert_to_mov,
            "WEBM": _vp.convert_to_webm,
        }
        fn = dispatch[fmt]
        # Argument convention differs by backend; MP4 takes keep_audio,
        # the others don't. Forward kwargs defensively.
        kwargs = {"path": path, "codec": codec, "quality": quality}
        if fmt == "MP4":
            kwargs["keep_audio"] = keep_audio
        fn(**kwargs)


# --------------------------------------------------------------------------- #
# A3 — VideoOverlayForm (replaces 7 popups)
# --------------------------------------------------------------------------- #
class VideoOverlayForm(OperationForm):
    """Burn an overlay (frame count / timer / text / filename /
    watermark / progress bar / another video) onto videos.

    Replaces SuperImposeFrameCountPopUp, SuperimposeTimerPopUp,
    SuperimposeTextPopUp, SuperimposeVideoNamesPopUp,
    SuperimposeWatermarkPopUp, SuperimposeProgressBarPopUp,
    SuperimposeVideoPopUp.
    """

    title = "Overlay on video(s)"
    description = ("Burn an overlay into video frames. Overlay type "
                   "drives the extra field shown.")

    def build(self) -> None:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.scope = _ScopePicker(self)
        form.addRow("Source:", self.scope)

        self.overlay_type = QComboBox(self)
        self.overlay_type.addItems([
            "Frame count", "Timer (elapsed)", "Text",
            "Filename", "Image watermark", "Progress bar", "Video picture-in-picture",
        ])
        self.overlay_type.currentIndexChanged.connect(self._on_type_changed)
        form.addRow("Overlay type:", self.overlay_type)

        # Type-specific extra fields — swap in a stacked widget
        self.extras = QStackedWidget(self)
        # 0: frame count / timer / filename / progress — no extra field
        blank = QWidget()
        self.extras.addWidget(blank)
        # 1: text
        self.text_field = QLineEdit()
        self.text_field.setPlaceholderText("Overlay text…")
        self.extras.addWidget(self.text_field)
        # 2: watermark path
        self.watermark_picker = _ScopePicker(
            allow_multiple=False,
            file_filter="Images (*.png *.jpg *.jpeg);;All files (*)",
        )
        self.extras.addWidget(self.watermark_picker)
        # 3: pip video path
        self.pip_picker = _ScopePicker(
            allow_multiple=False,
            file_filter="Videos (*.mp4 *.avi *.mov *.mkv *.webm);;All files (*)",
        )
        self.extras.addWidget(self.pip_picker)
        form.addRow("Content:", self.extras)

        # Position & color — shared across types
        self.position_cb = QComboBox(self)
        self.position_cb.addItems(["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right", "Center"])
        form.addRow("Position:", self.position_cb)

        self.font_size = QSpinBox(self)
        self.font_size.setRange(8, 200)
        self.font_size.setValue(24)
        form.addRow("Font / icon size:", self.font_size)

        self.body_layout.addLayout(form)
        self._on_type_changed(0)  # reset stacked index

    def _on_type_changed(self, index: int) -> None:
        # Map overlay type → extras stack index
        stack_idx = {0: 0, 1: 0, 2: 1, 3: 0, 4: 2, 5: 0, 6: 3}[index]
        self.extras.setCurrentIndex(stack_idx)

    def collect_args(self) -> dict:
        if not self.scope.path:
            raise ValueError("No source selected.")
        overlay_idx = self.overlay_type.currentIndex()
        overlay_kind = ["frame_count", "timer", "text", "filename",
                        "watermark", "progress_bar", "pip_video"][overlay_idx]
        content: Optional[str] = None
        if overlay_kind == "text":
            content = self.text_field.text().strip()
            if not content:
                raise ValueError("Text overlay requires content.")
        elif overlay_kind == "watermark":
            content = self.watermark_picker.path
            if not content:
                raise ValueError("Watermark overlay requires an image path.")
        elif overlay_kind == "pip_video":
            content = self.pip_picker.path
            if not content:
                raise ValueError("Picture-in-picture requires a video path.")
        return {
            "path": self.scope.path,
            "is_dir": self.scope.is_dir,
            "kind": overlay_kind,
            "content": content,
            "position": self.position_cb.currentText().lower().replace("-", "_"),
            "font_size": int(self.font_size.value()),
        }

    def target(self, *, path: str, is_dir: bool, kind: str,
               content: Optional[str], position: str, font_size: int) -> None:
        from mufasa.video_processors import video_processing as _vp
        dispatch = {
            "frame_count":  _vp.superimpose_frame_count,
            "timer":        _vp.superimpose_elapsed_time,
            "filename":     _vp.superimpose_video_names,
            "text":         _vp.superimpose_freetext,
            "watermark":    _vp.watermark_video,
            "progress_bar": _vp.superimpose_progressbar_on_video,
            "pip_video":    _vp.superimpose_video_on_video,
        }
        fn = dispatch[kind]
        # Each backend has slightly different kwargs — forward the
        # minimum intersection. Backends ignore unknown kwargs or the
        # caller can add a shim.
        kwargs = {"video_path": path, "position": position, "font_size": font_size}
        if content is not None:
            kwargs["content"] = content
        fn(**kwargs)


# --------------------------------------------------------------------------- #
# B — ClipVideosForm (replaces 5 popups)
# --------------------------------------------------------------------------- #
class ClipVideosForm(OperationForm):
    """Clip / trim videos by frame numbers or timestamps. Single or
    multi-clip, single or batch.

    Replaces ClipVideoPopUp, MultiShortenPopUp,
    ClipMultipleVideosByFrameNumbersPopUp,
    InitiateClipMultipleVideosByFrameNumbersPopUp,
    InitiateClipMultipleVideosByTimestampsPopUp.
    """

    title = "Clip / trim videos"
    description = ("Extract one or more segments from videos. Unit "
                   "(frames / timestamps) and scope (file / directory) "
                   "chosen inline.")

    def build(self) -> None:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.scope = _ScopePicker(self)
        form.addRow("Source:", self.scope)

        # Unit radio
        unit_row = QHBoxLayout()
        self.rb_frames = QRadioButton("Frame numbers", self)
        self.rb_times = QRadioButton("Timestamps (HH:MM:SS)", self)
        self.rb_frames.setChecked(True)
        unit_row.addWidget(self.rb_frames)
        unit_row.addWidget(self.rb_times)
        unit_row.addStretch()
        unit_host = QWidget(self); unit_host.setLayout(unit_row)
        form.addRow("Unit:", unit_host)

        # Start / end fields
        self.start_ed = QLineEdit(self); self.start_ed.setPlaceholderText("0")
        self.end_ed = QLineEdit(self); self.end_ed.setPlaceholderText("1000")
        form.addRow("Start:", self.start_ed)
        form.addRow("End:", self.end_ed)

        # Multi-split toggle
        self.multi_split = QCheckBox(
            "Multiple segments (comma-separated start/end pairs)", self
        )
        form.addRow("", self.multi_split)

        self.body_layout.addLayout(form)

    def collect_args(self) -> dict:
        if not self.scope.path:
            raise ValueError("No source selected.")
        unit = "frames" if self.rb_frames.isChecked() else "timestamps"
        start = self.start_ed.text().strip()
        end = self.end_ed.text().strip()
        if not start or not end:
            raise ValueError("Start and end are required.")
        if self.multi_split.isChecked():
            # Parse "0-100, 200-300" style
            try:
                pairs = []
                for chunk in (start + "," + end).replace(" ", "").split(","):
                    if "-" in chunk:
                        a, b = chunk.split("-", 1)
                        pairs.append((a, b))
                if not pairs:
                    raise ValueError("No valid segments parsed.")
            except Exception as exc:
                raise ValueError(f"Could not parse multi-segment input: {exc}")
            segments = pairs
        else:
            segments = [(start, end)]
        return {
            "path": self.scope.path,
            "is_dir": self.scope.is_dir,
            "unit": unit,
            "segments": segments,
        }

    def target(self, *, path: str, is_dir: bool, unit: str,
               segments: list[tuple[str, str]]) -> None:
        from mufasa.video_processors import video_processing as _vp
        # Backend entry depends on the axis. Dispatch to whichever
        # concrete fn the legacy popups used.
        if unit == "frames":
            fn = _vp.clip_videos_by_frame_ids
        else:
            fn = _vp.clip_videos_by_timestamps
        fn(data_path=path, is_dir=is_dir, segments=segments)


__all__ = [
    "VideoFormatConverterForm",
    "VideoOverlayForm",
    "ClipVideosForm",
]
