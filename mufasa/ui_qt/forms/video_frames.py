"""
mufasa.ui_qt.forms.video_frames
===============================

Three inline forms covering the frame-operations cluster:

* :class:`ExtractFramesForm` — frames OUT of one or more videos.
  Replaces :class:`ExtractAllFramesPopUp`,
  :class:`ExtractSpecificFramesPopUp`, :class:`ExtractSEQFramesPopUp`,
  :class:`SingleVideo2FramesPopUp`, :class:`MultipleVideos2FramesPopUp`.
* :class:`MergeFramesToVideoForm` — frames INTO a video file.
  Replaces :class:`MergeFrames2VideoPopUp`.
* :class:`ImportFrameDirectoryForm` — copy a directory of frames
  (typically PNGs) into a new location, normally the project's
  ``frames/input/<basename>/`` tree. Replaces
  :class:`ImportFrameDirectoryPopUp`.

Patch 122s notes:

* :class:`ExtractFramesForm` previously claimed (in its docstring)
  to cover ExtractSpecificFramesPopUp + ExtractSEQFramesPopUp but
  the body had stubs: SEQ extraction raised ``NotImplementedError``,
  the chosen image format was collected but never passed to the
  backend, and per-video looping was only implemented on the
  "extract everything" path. Rewritten to actually deliver.

* :class:`ImportFrameDirectoryForm` ignores the v1 project layout
  by design — there's no ``frames_dir`` key in
  :func:`mufasa.project_layout.project_paths_from_config`. The
  user picks a destination directly; the helper just copies. If
  v1 layout grows a canonical frames-dir later, this form can
  pre-fill the destination from it without changing the public
  contract.

Backend functions used:

* :func:`mufasa.video_processors.video_processing.extract_frames_single_video`
  — all-frames extraction; PNG only by backend constraint.
* :func:`mufasa.video_processors.video_processing.extract_frame_range`
  — start/end frame, honours img_format, greyscale, clahe,
  include_fn.
* :func:`mufasa.video_processors.extract_seqframes.extract_seq_frames`
  — Norpix SEQ container reader; takes just a filename.
* :func:`mufasa.video_processors.video_processing.frames_to_movie`
  — directory of numerically-named images → mp4/avi/webm.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog,
                               QFormLayout, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QSpinBox,
                               QWidget)

from mufasa.ui_qt.forms.video_processing import _ScopePicker
from mufasa.ui_qt.workbench import OperationForm


_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


# =========================================================================== #
# ExtractFramesForm — frames OUT of videos (5 popups)
# =========================================================================== #
class ExtractFramesForm(OperationForm):
    """Extract frames from one or more videos.

    Replaces :class:`ExtractAllFramesPopUp`,
    :class:`ExtractSpecificFramesPopUp`, :class:`ExtractSEQFramesPopUp`,
    :class:`SingleVideo2FramesPopUp`, :class:`MultipleVideos2FramesPopUp`.

    Behaviour:

    * Leave start + end blank → every frame goes through
      :func:`extract_frames_single_video` (PNG only — the backend
      doesn't expose a format parameter on the all-frames path).
    * Either start or end set → :func:`extract_frame_range` is called,
      which honours the format dropdown, greyscale, clahe, and the
      include-video-name-in-image-name toggle.
    * Source path ending in ``.seq`` → :func:`extract_seq_frames` is
      called regardless of start/end (SEQ has its own pipeline; the
      range fields are ignored with a UI note).
    * Directory mode → each video in the directory is processed in
      turn with the same options.
    """

    title = "Extract frames from video(s)"
    description = (
        "Extract frames to an image directory. Leave the frame range "
        "blank to extract every frame (PNG only). With a range set, "
        "image format / greyscale / CLAHE / include-filename options "
        "apply. SEQ files use a separate backend; range options are "
        "ignored for SEQ."
    )

    def build(self) -> None:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.scope = _ScopePicker(
            self,
            file_filter=(
                "Videos & SEQ (*.mp4 *.avi *.mov *.mkv *.webm *.m4v *.seq);;All files (*)"
            ),
        )
        form.addRow("Source:", self.scope)

        # Save dir (optional). Backend default = sibling
        # `<stem>_frames/` directory next to the source video.
        self.save_dir_edit = QLineEdit(self)
        self.save_dir_edit.setReadOnly(True)
        self.save_dir_edit.setPlaceholderText(
            "Optional — defaults to <video-stem>_frames/ next to source",
        )
        sd_browse = QPushButton("Browse…", self)
        sd_browse.clicked.connect(self._pick_save_dir)
        sd_row = QHBoxLayout()
        sd_row.addWidget(self.save_dir_edit, 1)
        sd_row.addWidget(sd_browse)
        form.addRow("Save directory:", sd_row)

        # Range fields
        self.start_frame = QSpinBox(self)
        self.start_frame.setRange(0, 10_000_000)
        self.start_frame.setSpecialValueText("(from start)")
        form.addRow("Start frame:", self.start_frame)

        self.end_frame = QSpinBox(self)
        self.end_frame.setRange(0, 10_000_000)
        self.end_frame.setSpecialValueText("(to end)")
        form.addRow("End frame:", self.end_frame)

        # Format + filters (range mode only — see _on_run for why)
        self.fmt_cb = QComboBox(self)
        self.fmt_cb.addItems(["png", "jpeg", "webp"])
        form.addRow("Image format:", self.fmt_cb)

        self.greyscale = QCheckBox("Greyscale output", self)
        form.addRow("", self.greyscale)
        self.clahe = QCheckBox(
            "CLAHE (adaptive histogram equalisation)", self,
        )
        form.addRow("", self.clahe)
        self.include_fn = QCheckBox(
            "Include video filename in image names "
            "(e.g. 'myvideo_0.png' not '0.png')", self,
        )
        form.addRow("", self.include_fn)

        note = QLabel(
            "<i>Image format / greyscale / CLAHE / include-filename "
            "options apply only when a frame range is set. With no "
            "range, all frames are written as plain PNGs (backend "
            "constraint).</i>",
            self,
        )
        note.setTextFormat(Qt.RichText)
        note.setWordWrap(True)
        note.setStyleSheet("color: palette(placeholder-text);")
        form.addRow("", note)

        self.body_layout.addLayout(form)

    def _pick_save_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Pick a save directory for frames", "",
        )
        if d:
            self.save_dir_edit.setText(d)

    def collect_args(self) -> dict:
        if not self.scope.path:
            raise ValueError("No source selected.")
        # special-value sentinel: spinbox at min => "blank"
        start = (int(self.start_frame.value())
                 if self.start_frame.value() else None)
        end = (int(self.end_frame.value())
               if self.end_frame.value() else None)
        if start is not None and end is not None and start >= end:
            raise ValueError(
                f"Start frame ({start}) must be < end ({end}).",
            )
        save_dir = self.save_dir_edit.text().strip() or None
        return {
            "path":        self.scope.path,
            "is_dir":      self.scope.is_dir,
            "save_dir":    save_dir,
            "start":       start,
            "end":         end,
            "fmt":         self.fmt_cb.currentText(),
            "greyscale":   bool(self.greyscale.isChecked()),
            "clahe":       bool(self.clahe.isChecked()),
            "include_fn":  bool(self.include_fn.isChecked()),
        }

    def target(self, *, path: str, is_dir: bool,
               save_dir: Optional[str],
               start: Optional[int], end: Optional[int],
               fmt: str, greyscale: bool, clahe: bool,
               include_fn: bool) -> None:
        from mufasa.video_processors import video_processing as _vp
        from mufasa.video_processors.extract_seqframes import (
            extract_seq_frames,
        )

        def _frames_dir_for(video_path: str) -> str:
            """Sibling `<stem>_frames/` dir next to the source video.
            Used as the default when the user didn't pick one."""
            if save_dir:
                # Per-video subdir when iterating a directory of videos
                if is_dir:
                    sub = Path(save_dir) / f"{Path(video_path).stem}_frames"
                else:
                    sub = Path(save_dir)
            else:
                p = Path(video_path)
                sub = p.parent / f"{p.stem}_frames"
            sub.mkdir(parents=True, exist_ok=True)
            return str(sub)

        def _extract_one(vp: str) -> None:
            # SEQ branch — own pipeline, ignores range fields.
            if vp.lower().endswith(".seq"):
                extract_seq_frames(vp)
                return
            if start is None and end is None:
                _vp.extract_frames_single_video(
                    file_path=vp,
                    save_dir=_frames_dir_for(vp),
                )
            else:
                _vp.extract_frame_range(
                    file_path=vp,
                    start_frame=start or 0,
                    end_frame=end or 10_000_000,
                    save_dir=_frames_dir_for(vp),
                    img_format=fmt,
                    greyscale=greyscale,
                    clahe=clahe,
                    include_fn=include_fn,
                )

        if is_dir:
            for vp in sorted(Path(path).iterdir()):
                ext = vp.suffix.lower()
                if ext in _VIDEO_EXTS or ext == ".seq":
                    _extract_one(str(vp))
        else:
            _extract_one(path)


# =========================================================================== #
# MergeFramesToVideoForm — frames INTO a video (1 popup)
# =========================================================================== #
class MergeFramesToVideoForm(OperationForm):
    """Merge a directory of numerically-named image frames into a video.

    Replaces :class:`MergeFrames2VideoPopUp`. Backend:
    :func:`mufasa.video_processors.video_processing.frames_to_movie`,
    which expects images named ``1.png``, ``2.png``, … or
    ``0.png``, ``1.png``, … (any zero-padding scheme).

    Output video is written into the input frames' parent directory
    using the directory basename + chosen extension (the backend's
    convention; not currently overrideable).
    """

    title = "Merge frames into video"
    description = (
        "Combine a directory of numerically-named image frames into "
        "an MP4 / AVI / WEBM video. The output is written alongside "
        "the input frames directory."
    )

    def build(self) -> None:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.dir_edit = QLineEdit(self)
        self.dir_edit.setReadOnly(True)
        self.dir_edit.setPlaceholderText(
            "Pick a directory containing numerically-named frames "
            "(1.png, 2.png, …)",
        )
        browse = QPushButton("Browse…", self)
        browse.clicked.connect(self._pick_dir)
        d_row = QHBoxLayout()
        d_row.addWidget(self.dir_edit, 1)
        d_row.addWidget(browse)
        form.addRow("Image directory:", d_row)

        self.fps = QSpinBox(self)
        self.fps.setRange(1, 240)
        self.fps.setValue(30)
        self.fps.setSuffix(" fps")
        form.addRow("Frame rate:", self.fps)

        self.quality = QSpinBox(self)
        self.quality.setRange(10, 100)
        self.quality.setSingleStep(10)
        self.quality.setValue(60)
        self.quality.setSuffix(" %")
        form.addRow("Quality:", self.quality)

        self.fmt_cb = QComboBox(self)
        self.fmt_cb.addItems(["mp4", "avi", "webm"])
        form.addRow("Output format:", self.fmt_cb)

        self.gpu = QCheckBox(
            "Use GPU encoder if available (NVENC)", self,
        )
        form.addRow("", self.gpu)

        self.body_layout.addLayout(form)

    def _pick_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Pick frames directory", "",
        )
        if d:
            self.dir_edit.setText(d)

    def collect_args(self) -> dict:
        d = self.dir_edit.text().strip()
        if not d:
            raise ValueError("No frames directory selected.")
        if not Path(d).is_dir():
            raise ValueError(f"Not a directory: {d}")
        return {
            "directory":  d,
            "fps":        int(self.fps.value()),
            "quality":    int(self.quality.value()),
            "out_format": self.fmt_cb.currentText(),
            "gpu":        bool(self.gpu.isChecked()),
        }

    def target(self, *, directory: str, fps: int, quality: int,
               out_format: str, gpu: bool) -> None:
        from mufasa.video_processors import video_processing as _vp
        _vp.frames_to_movie(
            directory=directory, fps=fps, quality=quality,
            out_format=out_format, gpu=gpu,
        )


# =========================================================================== #
# ImportFrameDirectoryForm — copy a frames directory into the project (1 popup)
# =========================================================================== #
class ImportFrameDirectoryForm(OperationForm):
    """Copy a directory of frame images into the project's tree.

    Replaces :class:`ImportFrameDirectoryPopUp`. Legacy backend
    :func:`mufasa.video_processors.video_processing.copy_img_folder`
    resolves the destination via the SimBA INI key
    ``General settings.project_path`` and writes to
    ``<project>/frames/input/<basename>/``. v1 project layout
    doesn't model a frames directory; rather than coercing v1 to
    fit, this form takes an explicit destination directory from the
    user.

    Validation:

    * source must exist and contain at least one image file
      (``.png`` / ``.jpg`` / ``.jpeg`` / ``.bmp`` / ``.tif`` /
      ``.tiff`` / ``.webp``);
    * destination's parent must exist;
    * destination must not exist yet (the helper refuses to merge
      into an existing tree — overwrite-on-purpose is a separate UX
      and worth being explicit about).
    """

    title = "Import frame directory"
    description = (
        "Copy a directory of frame images into a destination of your "
        "choice (typically a project's frames tree). The destination "
        "directory is created fresh; merging into an existing tree "
        "is not supported here."
    )

    _IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp",
                 ".tif", ".tiff", ".webp"}

    def build(self) -> None:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.src_edit = QLineEdit(self)
        self.src_edit.setReadOnly(True)
        self.src_edit.setPlaceholderText(
            "Folder containing image frames",
        )
        src_browse = QPushButton("Browse…", self)
        src_browse.clicked.connect(self._pick_src)
        s_row = QHBoxLayout()
        s_row.addWidget(self.src_edit, 1)
        s_row.addWidget(src_browse)
        form.addRow("Source directory:", s_row)

        self.dst_edit = QLineEdit(self)
        self.dst_edit.setReadOnly(True)
        self.dst_edit.setPlaceholderText(
            "Destination path (must not exist yet)",
        )
        dst_browse = QPushButton("Browse…", self)
        dst_browse.clicked.connect(self._pick_dst)
        d_row = QHBoxLayout()
        d_row.addWidget(self.dst_edit, 1)
        d_row.addWidget(dst_browse)
        form.addRow("Destination:", d_row)

        self.symlink = QCheckBox(
            "Symlink instead of copying "
            "(faster; output isn't portable)", self,
        )
        form.addRow("", self.symlink)

        note = QLabel(
            "<i>Legacy SimBA placed imported frames under "
            "<code>&lt;project&gt;/frames/input/&lt;basename&gt;/</code>. "
            "v1 project layout doesn't model a frames directory; pick "
            "any destination you want.</i>",
            self,
        )
        note.setTextFormat(Qt.RichText)
        note.setWordWrap(True)
        note.setStyleSheet("color: palette(placeholder-text);")
        form.addRow("", note)

        self.body_layout.addLayout(form)

    def _pick_src(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Pick source frames directory", "",
        )
        if d:
            self.src_edit.setText(d)
            # Helpful: if the destination field is empty AND a
            # project is loaded with a parent path on disk, default
            # the destination to <project>/frames/input/<basename>/
            # so the legacy-SimBA workflow is one click instead of
            # two browses. Skipped when no config_path on the form.
            if not self.dst_edit.text().strip() and self.config_path:
                try:
                    project_root = (
                        Path(self.config_path).parent.resolve()
                    )
                    basename = Path(d).name
                    default = project_root / "frames" / "input" / basename
                    self.dst_edit.setText(str(default))
                except OSError:
                    pass

    def _pick_dst(self) -> None:
        # We pick a *parent* directory and let the user name the
        # destination inline. QFileDialog.getSaveFileName is overkill
        # for a directory; use a directory picker + ask user to type
        # the leaf in the read-only field? Compromise: open the dir
        # picker, append the source basename. The user can still
        # override by manually editing… except the field is read-only.
        # So: ask for the *parent* and append the source basename.
        parent = QFileDialog.getExistingDirectory(
            self, "Pick destination parent directory", "",
        )
        if not parent:
            return
        src = self.src_edit.text().strip()
        leaf = Path(src).name if src else "imported_frames"
        self.dst_edit.setText(str(Path(parent) / leaf))

    def collect_args(self) -> dict:
        src = self.src_edit.text().strip()
        dst = self.dst_edit.text().strip()
        if not src:
            raise ValueError("Pick a source directory.")
        if not dst:
            raise ValueError("Pick a destination.")
        src_p = Path(src)
        dst_p = Path(dst)
        if not src_p.is_dir():
            raise ValueError(f"Source is not a directory: {src}")
        # Has at least one image?
        has_image = any(
            f.suffix.lower() in self._IMG_EXTS
            for f in src_p.iterdir() if f.is_file()
        )
        if not has_image:
            raise ValueError(
                f"No image files found in {src}. Looked for: "
                f"{', '.join(sorted(self._IMG_EXTS))}."
            )
        if dst_p.exists():
            raise ValueError(
                f"Destination already exists: {dst}. Pick a new path "
                "or move the existing tree out of the way first."
            )
        if not dst_p.parent.is_dir():
            raise ValueError(
                f"Destination parent doesn't exist: {dst_p.parent}",
            )
        return {
            "source":      str(src_p),
            "destination": str(dst_p),
            "symlink":     bool(self.symlink.isChecked()),
        }

    def target(self, *, source: str, destination: str,
               symlink: bool) -> None:
        if symlink:
            # symlink the whole directory, not contents — copytree
            # semantic equivalent.
            os.symlink(source, destination,
                       target_is_directory=True)
        else:
            shutil.copytree(source, destination)


__all__ = [
    "ExtractFramesForm",
    "MergeFramesToVideoForm",
    "ImportFrameDirectoryForm",
]
