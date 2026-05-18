"""
mufasa.ui_qt.forms.yolo_inference
=================================

Qt port of the legacy Tk YOLO pose-inference popup
(:class:`YOLOPoseInferencePopUP`). Lives on the Classifier
workbench page alongside the SimBA-classifier RunInferenceForm.

Two execution paths share the same parameters:

* :class:`YOLOPoseInference` — plain pose estimation (no tracking).
* :class:`YOLOPoseTrackInference` — tracking-enabled inference;
  selected by supplying a tracker `.yml` config.

The form has a Mode selector (Single video / Video directory) and
a stacked picker that switches between a file-picker and a
folder-picker accordingly. Otherwise the parameter set is the same
as the legacy popup, with each Tk MufasaDropDown / FileSelect /
FolderSelect mapped to its Qt-idiomatic substitute:

==================================  ================================
Tk widget                            Qt substitute
==================================  ================================
MufasaDropDown (TRUE/FALSE)          QCheckBox
MufasaDropDown (int options)         QSpinBox
MufasaDropDown (float options)       QDoubleSpinBox
MufasaDropDown (str enum + 'None')   QComboBox with special-value
FileSelect                           QLineEdit + Browse… QPushButton
FolderSelect                         QLineEdit + Browse… QPushButton
==================================  ================================

Backend availability (CUDA + ultralytics) is checked in
:meth:`collect_args` rather than :meth:`build`, so the form can
still be displayed inert on machines without these. The check
raises a ValueError caught by the OperationForm base, which
surfaces as a `QMessageBox.warning` — friendlier than a
construction-time crash that would tear down the workbench page.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QHBoxLayout,
                               QLabel, QLineEdit, QPushButton,
                               QSpinBox, QStackedWidget, QWidget)

from mufasa.ui_qt.workbench import OperationForm


# Same option lists as the legacy popup, kept lazy where possible
# so importing this module doesn't trigger find_core_cnt() etc.
_MAX_TRACKS_OPTIONS = ["None", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
_IMG_SIZE_OPTIONS = [256, 288, 320, 416, 480, 512, 640, 720, 768, 960, 1280]
_SMOOTHING_OPTIONS = ["None", 50, 100, 200, 300, 400, 500]


def _lazy_yolo_formats() -> list[str]:
    """Return YOLO format options. Lazy because Options.VALID_YOLO_FORMATS
    pulls in enum-loading machinery; defer to call time."""
    try:
        from mufasa.utils.enums import Options
        return list(Options.VALID_YOLO_FORMATS.value) + ["None"]
    except Exception:
        return ["None"]


def _lazy_thresholds() -> list[float]:
    """Float threshold options 0.05..1.00 step 0.05. Materialise
    once at form build."""
    import numpy as np
    return [float(x) for x in
            np.arange(0.05, 1.05, 0.05).astype("float32")]


def _lazy_core_cnt() -> list[int]:
    """CPU-worker option list. Wraps find_core_cnt to avoid the
    import at module top."""
    try:
        from mufasa.utils.read_write import find_core_cnt
        return list(range(1, find_core_cnt()[0]))
    except Exception:
        return [1, 2, 4]


def _lazy_devices() -> tuple[bool, list[str]]:
    """Return (cuda_available, device_options) — CPU plus any GPUs
    detected. Wrapped so this module imports clean on no-CUDA boxes."""
    try:
        from mufasa.data_processors.cuda.utils import _is_cuda_available
        gpu_available, gpus = _is_cuda_available()
    except Exception:
        return False, ["CPU"]
    devices = ["CPU"]
    if gpu_available:
        devices.extend(
            f"{x} : {y['model']}" for x, y in gpus.items()
        )
    return gpu_available, devices


def _seven_bp_default() -> str:
    """Default body-part config path — `<mufasa>/<YOLO_SCHEMATICS>/yolo_7bps.csv`."""
    try:
        import mufasa
        from mufasa.utils.enums import Paths
        return os.path.join(
            os.path.dirname(mufasa.__file__),
            Paths.YOLO_SCHEMATICS_DIR.value,
            "yolo_7bps.csv",
        )
    except Exception:
        return ""


# =========================================================================== #
# YOLOPoseInferenceForm
# =========================================================================== #
class YOLOPoseInferenceForm(OperationForm):
    """Run inference with a trained YOLO pose-estimation model.

    Replaces :class:`YOLOPoseInferencePopUP`. Selects between
    :class:`YOLOPoseInference` (no tracking) and
    :class:`YOLOPoseTrackInference` (tracking) based on whether a
    tracker `.yml` is supplied.
    """

    title = "YOLO pose — inference"
    description = (
        "Run a trained YOLO pose model on a video or a directory "
        "of videos. Supply a tracker .yml to switch from plain "
        "pose estimation to tracking. Requires CUDA + ultralytics; "
        "the form is inert without them."
    )

    def build(self) -> None:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        # ----- Paths frame ------------------------------------- #
        self.weights_path = self._make_file_row(
            form, "Model weights (.pt):",
            "YOLO weights file",
            file_filter="YOLO model (*.pt *.onnx *.engine);;"
                        "All files (*)",
        )
        self.save_dir = self._make_dir_row(
            form, "Save directory:",
            "Where pose results will be written",
        )
        self.bp_config_csv = self._make_file_row(
            form, "Body-part names (.csv):",
            "CSV mapping YOLO keypoints to names",
            file_filter="CSV (*.csv);;All files (*)",
            initial_path=_seven_bp_default(),
        )
        self.tracker_path = self._make_file_row(
            form, "Tracker config (optional):",
            "Empty → plain inference; set → tracking inference",
            file_filter="YAML (*.yml *.yaml);;All files (*)",
        )

        # ----- Mode selector + stacked video picker ------------ #
        self.mode_combo = QComboBox(self)
        self.mode_combo.addItems(["Single video", "Video directory"])
        form.addRow("Mode:", self.mode_combo)

        # Stacked: single-video file picker / directory picker
        self.video_stack = QStackedWidget(self)
        self.video_path_edit = QLineEdit(self)
        self.video_path_edit.setReadOnly(True)
        self.video_path_edit.setPlaceholderText("Video file…")
        video_browse = QPushButton("Browse…", self)
        def _pick_video() -> None:
            f, _ = QFileDialog.getOpenFileName(
                self, "Pick video", "",
                "Video (*.mp4 *.avi *.mov *.mkv *.webm);;"
                "All files (*)",
            )
            if f:
                self.video_path_edit.setText(f)
        video_browse.clicked.connect(_pick_video)
        single_widget = QWidget(self)
        single_row = QHBoxLayout(single_widget)
        single_row.setContentsMargins(0, 0, 0, 0)
        single_row.addWidget(self.video_path_edit, 1)
        single_row.addWidget(video_browse)

        self.video_dir_edit = QLineEdit(self)
        self.video_dir_edit.setReadOnly(True)
        self.video_dir_edit.setPlaceholderText("Folder of videos…")
        dir_browse = QPushButton("Browse…", self)
        def _pick_dir() -> None:
            d = QFileDialog.getExistingDirectory(
                self, "Pick video directory", "")
            if d:
                self.video_dir_edit.setText(d)
        dir_browse.clicked.connect(_pick_dir)
        dir_widget = QWidget(self)
        dir_row = QHBoxLayout(dir_widget)
        dir_row.setContentsMargins(0, 0, 0, 0)
        dir_row.addWidget(self.video_dir_edit, 1)
        dir_row.addWidget(dir_browse)

        self.video_stack.addWidget(single_widget)
        self.video_stack.addWidget(dir_widget)
        self.mode_combo.currentIndexChanged.connect(
            self.video_stack.setCurrentIndex)
        form.addRow("Input:", self.video_stack)

        # ----- Devices + workers ------------------------------- #
        self._cuda_available, devices = _lazy_devices()
        self.device_combo = QComboBox(self)
        self.device_combo.addItems(devices)
        # Default to first GPU if present, CPU otherwise
        if len(devices) > 1:
            self.device_combo.setCurrentIndex(1)
        form.addRow("Device:", self.device_combo)

        core_options = _lazy_core_cnt()
        self.workers = QSpinBox(self)
        self.workers.setRange(min(core_options), max(core_options))
        self.workers.setValue(
            int(math.ceil(max(core_options) / 2)))
        form.addRow("CPU workers:", self.workers)

        # ----- Numeric / size knobs --------------------------- #
        self.batch_size = QSpinBox(self)
        self.batch_size.setRange(50, 1000)
        self.batch_size.setSingleStep(50)
        self.batch_size.setValue(250)
        form.addRow("Batch size:", self.batch_size)

        self.img_size = QComboBox(self)
        self.img_size.addItems(str(s) for s in _IMG_SIZE_OPTIONS)
        self.img_size.setCurrentText("256")
        form.addRow("Image size:", self.img_size)

        thresholds = _lazy_thresholds()
        self.threshold = QDoubleSpinBox(self)
        self.threshold.setRange(min(thresholds), max(thresholds))
        self.threshold.setSingleStep(0.05)
        self.threshold.setValue(0.1)
        self.threshold.setDecimals(2)
        form.addRow("Box threshold:", self.threshold)

        self.iou = QDoubleSpinBox(self)
        self.iou.setRange(min(thresholds), max(thresholds))
        self.iou.setSingleStep(0.05)
        self.iou.setValue(0.8)
        self.iou.setDecimals(2)
        form.addRow("IoU:", self.iou)

        # ----- Format + tracking-only --------------------------- #
        self.fmt = QComboBox(self)
        self.fmt.addItems(_lazy_yolo_formats())
        self.fmt.setCurrentText("None")
        form.addRow("Output format:", self.fmt)

        self.max_tracks = QComboBox(self)
        self.max_tracks.addItems(str(o) for o in _MAX_TRACKS_OPTIONS)
        self.max_tracks.setCurrentText("None")
        form.addRow("Max tracks:", self.max_tracks)

        self.max_per_id = QComboBox(self)
        self.max_per_id.addItems(str(o) for o in _MAX_TRACKS_OPTIONS)
        self.max_per_id.setCurrentText("1")
        form.addRow("Max tracks per ID:", self.max_per_id)

        self.smoothing = QComboBox(self)
        self.smoothing.addItems(str(o) for o in _SMOOTHING_OPTIONS)
        # Tk default = SMOOTHING_OPTIONS[2] = 100
        self.smoothing.setCurrentText("100")
        form.addRow("Smoothing (ms):", self.smoothing)

        # ----- Toggles ----------------------------------------- #
        self.verbose = QCheckBox("Verbose progress logging", self)
        self.verbose.setChecked(True)
        form.addRow("", self.verbose)
        self.interpolate = QCheckBox("Interpolate gaps", self)
        self.interpolate.setChecked(True)
        form.addRow("", self.interpolate)
        self.stream = QCheckBox("Stream (lower memory use)", self)
        self.stream.setChecked(True)
        form.addRow("", self.stream)
        self.recursive = QCheckBox(
            "Recursive video search (directory mode)", self)
        form.addRow("", self.recursive)

        # ----- Availability hint -------------------------------- #
        hint = self._availability_hint()
        if hint:
            warn = QLabel(hint, self)
            warn.setStyleSheet("color: #c0392b; font-style: italic;")
            warn.setWordWrap(True)
            form.addRow("", warn)

        self.body_layout.addLayout(form)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _availability_hint(self) -> Optional[str]:
        """Return a human-readable hint if CUDA or ultralytics is
        missing, else None."""
        missing: list[str] = []
        if not self._cuda_available:
            missing.append("CUDA GPU")
        try:
            from mufasa.utils.enums import PackageNames
            from mufasa.utils.read_write import get_pkg_version
            ver = get_pkg_version(pkg=PackageNames.ULTRALYTICS.value)
            if ver is None:
                missing.append("ultralytics package")
        except Exception:
            missing.append("ultralytics package")
        if missing:
            return (f"Note: YOLO inference requires "
                    f"{' and '.join(missing)}. The Run button will "
                    f"fail until both are available.")
        return None

    def _make_dir_row(self, form: QFormLayout, label: str,
                      placeholder: str) -> QLineEdit:
        edit = QLineEdit(self)
        edit.setReadOnly(True)
        edit.setPlaceholderText(placeholder)
        browse = QPushButton("Browse…", self)

        def _pick() -> None:
            d = QFileDialog.getExistingDirectory(
                self, f"Pick {label.rstrip(':')}", "")
            if d:
                edit.setText(d)
        browse.clicked.connect(_pick)
        row = QHBoxLayout()
        row.addWidget(edit, 1)
        row.addWidget(browse)
        form.addRow(label, row)
        return edit

    def _make_file_row(self, form: QFormLayout, label: str,
                       placeholder: str, file_filter: str,
                       initial_path: str = "") -> QLineEdit:
        edit = QLineEdit(self)
        edit.setReadOnly(True)
        edit.setPlaceholderText(placeholder)
        if initial_path and Path(initial_path).is_file():
            edit.setText(initial_path)
        browse = QPushButton("Browse…", self)

        def _pick() -> None:
            f, _ = QFileDialog.getOpenFileName(
                self, f"Pick {label.rstrip(':')}",
                edit.text() or "", file_filter,
            )
            if f:
                edit.setText(f)
        browse.clicked.connect(_pick)
        row = QHBoxLayout()
        row.addWidget(edit, 1)
        row.addWidget(browse)
        form.addRow(label, row)
        return edit

    # ------------------------------------------------------------------ #
    # OperationForm contract
    # ------------------------------------------------------------------ #
    def collect_args(self) -> dict:
        # Availability check — surfaces as a friendly QMessageBox via
        # the OperationForm base, NOT a workbench-tearing crash.
        if not self._cuda_available:
            raise ValueError(
                "No CUDA GPU detected. YOLO pose inference needs "
                "an NVIDIA GPU.")
        try:
            from mufasa.utils.enums import PackageNames
            from mufasa.utils.read_write import get_pkg_version
            if get_pkg_version(
                    pkg=PackageNames.ULTRALYTICS.value) is None:
                raise ValueError(
                    "ultralytics package not installed. "
                    "`pip install ultralytics`.")
        except ImportError:
            raise ValueError(
                "ultralytics package not installed. "
                "`pip install ultralytics`.")

        weights = self.weights_path.text().strip()
        save_dir = self.save_dir.text().strip()
        bp_csv = self.bp_config_csv.text().strip()
        tracker = self.tracker_path.text().strip()
        if not weights or not Path(weights).is_file():
            raise ValueError("Pick a valid model weights file.")
        if not save_dir or not Path(save_dir).is_dir():
            raise ValueError("Pick a valid save directory.")
        if not bp_csv or not Path(bp_csv).is_file():
            raise ValueError("Pick a valid body-part names CSV.")
        tracker_arg = (tracker if tracker
                       and Path(tracker).is_file() else None)

        mode = self.mode_combo.currentText()
        if mode == "Single video":
            v = self.video_path_edit.text().strip()
            if not v or not Path(v).is_file():
                raise ValueError(
                    "Pick a valid video file in Single video mode.")
            video_paths: list[str] = [v]
            recursive = False
        else:
            d = self.video_dir_edit.text().strip()
            if not d or not Path(d).is_dir():
                raise ValueError(
                    "Pick a valid video directory in Directory mode.")
            # Defer the actual file discovery to target() — keep
            # collect_args fast and avoid pulling in the whole
            # read_write module here.
            video_paths = []  # marker → target() will populate
            video_paths = [d]  # actually pass the directory; target
                               # will expand it. Avoid two-arg case.
            recursive = bool(self.recursive.isChecked())

        # Device parsing — "0 : NVIDIA RTX..." → int(0)
        dev_text = self.device_combo.currentText()
        device = ("cpu" if dev_text == "CPU"
                  else int(dev_text.split(":", 1)[0]))

        # Combo → optional int / str conversions
        max_tracks_text = self.max_tracks.currentText()
        max_tracks = (None if max_tracks_text == "None"
                      else int(max_tracks_text))
        max_per_id_text = self.max_per_id.currentText()
        max_per_id = (None if max_per_id_text == "None"
                      else int(max_per_id_text))
        smoothing_text = self.smoothing.currentText()
        smoothing = (None if smoothing_text == "None"
                     else int(smoothing_text))
        fmt_text = self.fmt.currentText()
        fmt = None if fmt_text == "None" else fmt_text

        return {
            "weights":       weights,
            "save_dir":      save_dir,
            "bp_csv":        bp_csv,
            "tracker_path":  tracker_arg,
            "mode":          mode,
            "video_target":  video_paths[0],  # file or directory
            "recursive":     recursive,
            "device":        device,
            "workers":       int(self.workers.value()),
            "batch_size":    int(self.batch_size.value()),
            "img_size":      int(self.img_size.currentText()),
            "threshold":     float(self.threshold.value()),
            "iou":           float(self.iou.value()),
            "fmt":           fmt,
            "max_tracks":    max_tracks,
            "max_per_id":    max_per_id,
            "smoothing":     smoothing,
            "verbose":       bool(self.verbose.isChecked()),
            "interpolate":   bool(self.interpolate.isChecked()),
            "stream":        bool(self.stream.isChecked()),
        }

    def target(self, *, weights: str, save_dir: str, bp_csv: str,
               tracker_path: Optional[str], mode: str,
               video_target: str, recursive: bool, device,
               workers: int, batch_size: int, img_size: int,
               threshold: float, iou: float, fmt: Optional[str],
               max_tracks: Optional[int],
               max_per_id: Optional[int],
               smoothing: Optional[int], verbose: bool,
               interpolate: bool, stream: bool) -> None:
        from mufasa.model.yolo_pose_inference import YOLOPoseInference
        from mufasa.model.yolo_pose_track_inference import (
            YOLOPoseTrackInference,
        )
        from mufasa.utils.read_write import (
            find_files_of_filetypes_in_directory,
            get_video_meta_data, read_yolo_bp_names_file,
            recursive_file_search,
        )
        from mufasa.utils.enums import Options

        keypoint_names = read_yolo_bp_names_file(file_path=bp_csv)

        # Expand directory → list of video files. Single mode already
        # has the file path in video_target.
        if mode == "Video directory":
            if recursive:
                video_paths = recursive_file_search(
                    directory=video_target,
                    extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value,
                    as_dict=False,
                )
            else:
                video_paths = find_files_of_filetypes_in_directory(
                    directory=video_target,
                    extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value,
                    raise_error=True,
                )
        else:
            video_paths = [video_target]

        # Validate each video readable
        for vp in video_paths:
            _ = get_video_meta_data(video_path=vp)

        if tracker_path is None:
            runner = YOLOPoseInference(
                weights=weights, video_path=video_paths,
                verbose=verbose, save_dir=save_dir, device=device,
                format=fmt, batch_size=batch_size,
                torch_threads=workers, box_threshold=threshold,
                max_tracks=max_tracks, max_per_class=max_per_id,
                interpolate=interpolate, imgsz=img_size, iou=iou,
                stream=stream, smoothing=smoothing,
                keypoint_names=keypoint_names, recursive=recursive,
            )
        else:
            runner = YOLOPoseTrackInference(
                weights_path=weights, video_path=video_paths,
                config_path=tracker_path, verbose=verbose,
                save_dir=save_dir, device=device, format=fmt,
                batch_size=batch_size, torch_threads=workers,
                half_precision=True, stream=stream,
                interpolate=interpolate, threshold=threshold,
                max_tracks=max_tracks, smoothing=smoothing,
                imgsz=img_size, iou=iou,
                keypoint_names=keypoint_names,
            )
        runner.run()


__all__ = ["YOLOPoseInferenceForm"]
