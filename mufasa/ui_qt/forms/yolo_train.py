"""
mufasa.ui_qt.forms.yolo_train
=============================

Qt port of :class:`YOLOPoseTrainPopUP`. Trains a YOLO pose model
by firing off `python -m mufasa.model.yolo_fit` as a detached
subprocess — long-running, hardware-bound (typically hours on
GPU), so the workbench doesn't wait for it. After launch, the
form shows an info dialog telling the user where to watch
progress.

Lives on the Classifier workbench page beside
:class:`YOLOPoseInferenceForm`. Like the inference form, requires
CUDA + ultralytics; availability checked in `collect_args` for
the friendly-error path, with an in-form red hint when missing.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog,
                               QFormLayout, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QPushButton,
                               QSpinBox)

from mufasa.ui_qt.workbench import OperationForm


_IMG_SIZE_OPTIONS = [256, 320, 416, 480, 512, 640, 720, 768, 960, 1280]
_BATCH_SIZE_OPTIONS = [2, 4, 8, 16, 32, 64, 128]


def _lazy_format_options() -> list[str]:
    try:
        from mufasa.utils.enums import Options
        return ["None"] + list(Options.VALID_YOLO_FORMATS.value)
    except Exception:
        return ["None"]


def _lazy_devices() -> tuple[bool, list[str]]:
    try:
        from mufasa.data_processors.cuda.utils import _is_cuda_available
        gpu_available, gpus = _is_cuda_available()
    except Exception:
        return False, ["CPU"]
    devices = ["CPU"]
    if gpu_available:
        devices.extend(f"{x} : {y['model']}"
                       for x, y in gpus.items())
    return gpu_available, devices


def _lazy_max_workers() -> int:
    """Worker-count ceiling. Windows pytorch DataLoader deadlocks
    above 8; cap there on win32 to match the Tk popup."""
    try:
        from mufasa.utils.read_write import find_core_cnt
        n = find_core_cnt()[0]
    except Exception:
        return 4
    return min(n, 8) if sys.platform == "win32" else n


# =========================================================================== #
# YOLOPoseTrainForm
# =========================================================================== #
class YOLOPoseTrainForm(OperationForm):
    """Train a YOLO pose-estimation model. Detached subprocess —
    the workbench doesn't wait. Replaces :class:`YOLOPoseTrainPopUP`."""

    title = "YOLO pose — train"
    description = (
        "Train a YOLO pose model from a YAML map + (optional) "
        "initial weights. Runs in a detached subprocess; progress "
        "appears in a console window (Windows) or the parent "
        "terminal (Unix). Requires CUDA + ultralytics."
    )

    def build(self) -> None:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.yolo_map_edit = self._make_file_row(
            form, "YOLO map (.yaml):",
            "Dataset YAML describing classes + train/val splits",
            file_filter="YAML (*.yaml *.yml);;All files (*)",
        )
        self.weights_edit = self._make_file_row(
            form, "Initial weights (optional, .pt):",
            "Empty → train from scratch / pretrained default",
            file_filter="YOLO model (*.pt *.onnx);;All files (*)",
        )
        self.save_dir_edit = self._make_dir_row(
            form, "Save directory:",
            "Where training runs + checkpoints will be written",
        )

        # Devices + workers
        self._cuda_available, devices = _lazy_devices()
        self.device_combo = QComboBox(self)
        self.device_combo.addItems(devices)
        if len(devices) > 1:
            self.device_combo.setCurrentIndex(1)
        form.addRow("Device:", self.device_combo)

        max_w = _lazy_max_workers()
        self.workers = QSpinBox(self)
        self.workers.setRange(1, max(max_w, 1))
        self.workers.setValue(max(1, max_w // 2))
        form.addRow("CPU workers:", self.workers)

        # Numerics
        self.epochs = QSpinBox(self)
        self.epochs.setRange(100, 5500)
        self.epochs.setSingleStep(250)
        self.epochs.setValue(500)
        form.addRow("Epochs:", self.epochs)

        self.img_size = QComboBox(self)
        self.img_size.addItems(str(s) for s in _IMG_SIZE_OPTIONS)
        self.img_size.setCurrentText("640")
        form.addRow("Image size:", self.img_size)

        self.batch_size = QComboBox(self)
        self.batch_size.addItems(str(s) for s in _BATCH_SIZE_OPTIONS)
        self.batch_size.setCurrentText("16")
        form.addRow("Batch size:", self.batch_size)

        self.patience = QSpinBox(self)
        self.patience.setRange(50, 1000)
        self.patience.setSingleStep(50)
        self.patience.setValue(100)
        form.addRow("Patience:", self.patience)

        self.fmt = QComboBox(self)
        self.fmt.addItems(_lazy_format_options())
        self.fmt.setCurrentText("None")
        form.addRow("Export format:", self.fmt)

        # Toggles
        self.plots = QCheckBox("Generate plots", self)
        self.plots.setChecked(True)
        form.addRow("", self.plots)
        self.verbose = QCheckBox("Verbose progress logging", self)
        self.verbose.setChecked(True)
        form.addRow("", self.verbose)

        # Availability hint (same pattern as inference form)
        hint = self._availability_hint()
        if hint:
            warn = QLabel(hint, self)
            warn.setStyleSheet("color: #c0392b; font-style: italic;")
            warn.setWordWrap(True)
            form.addRow("", warn)

        self.body_layout.addLayout(form)

    # ------------------------------------------------------------------ #
    # Helpers (mirroring yolo_inference.py — same idioms)
    # ------------------------------------------------------------------ #
    def _availability_hint(self) -> Optional[str]:
        missing: list[str] = []
        if not self._cuda_available:
            missing.append("CUDA GPU")
        try:
            from mufasa.utils.enums import PackageNames
            from mufasa.utils.read_write import get_pkg_version
            if get_pkg_version(
                    pkg=PackageNames.ULTRALYTICS.value) is None:
                missing.append("ultralytics package")
        except Exception:
            missing.append("ultralytics package")
        if missing:
            return (f"Note: YOLO training requires "
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
                       placeholder: str, file_filter: str
                       ) -> QLineEdit:
        edit = QLineEdit(self)
        edit.setReadOnly(True)
        edit.setPlaceholderText(placeholder)
        browse = QPushButton("Browse…", self)

        def _pick() -> None:
            f, _ = QFileDialog.getOpenFileName(
                self, f"Pick {label.rstrip(':')}", "", file_filter,
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
        if not self._cuda_available:
            raise ValueError(
                "No CUDA GPU detected. YOLO training needs an "
                "NVIDIA GPU.")
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
                "ultralytics package not installed.")

        yaml_path = self.yolo_map_edit.text().strip()
        weights = self.weights_edit.text().strip()
        save_dir = self.save_dir_edit.text().strip()
        if not yaml_path or not Path(yaml_path).is_file():
            raise ValueError("Pick a valid YOLO map .yaml file.")
        if not save_dir or not Path(save_dir).is_dir():
            raise ValueError("Pick a valid save directory.")
        # Validate the YAML structure via existing backend helper
        from mufasa.third_party_label_appenders.transform.utils import (
            check_valid_yolo_map,
        )
        check_valid_yolo_map(yolo_map=yaml_path)
        # weights is optional — empty string → None
        weights_arg = weights if weights and Path(weights).is_file() else None

        dev_text = self.device_combo.currentText()
        device_str = ("cpu" if dev_text == "CPU"
                      else dev_text.split(":", 1)[0])
        fmt_text = self.fmt.currentText()
        fmt_arg = None if fmt_text == "None" else fmt_text

        return {
            "yolo_map":    yaml_path,
            "weights":     weights_arg,
            "save_dir":    save_dir,
            "device":      device_str,
            "workers":     int(self.workers.value()),
            "epochs":      int(self.epochs.value()),
            "img_size":    int(self.img_size.currentText()),
            "batch_size":  int(self.batch_size.currentText()),
            "patience":    int(self.patience.value()),
            "fmt":         fmt_arg,
            "plots":       bool(self.plots.isChecked()),
            "verbose":     bool(self.verbose.isChecked()),
        }

    def target(self, *, yolo_map: str, weights: Optional[str],
               save_dir: str, device: str, workers: int,
               epochs: int, img_size: int, batch_size: int,
               patience: int, fmt: Optional[str], plots: bool,
               verbose: bool) -> None:
        """Fire off `python -m mufasa.model.yolo_fit` as a detached
        subprocess. The workbench doesn't wait for completion."""
        # Windows: cap workers at 8 (DataLoader deadlock)
        workers_subproc = (min(workers, 8)
                           if sys.platform == "win32" else workers)
        cmd = [
            sys.executable, "-m", "mufasa.model.yolo_fit",
            "--model_yaml", yolo_map,
            "--save_path", save_dir,
            "--epochs", str(epochs),
            "--batch", str(batch_size),
            "--plots", "True" if plots else "False",
            "--imgsz", str(img_size),
            "--device", device,
            "--verbose", "True" if verbose else "False",
            "--workers", str(workers_subproc),
            "--patience", str(patience),
        ]
        if weights is not None:
            cmd.extend(["--weights_path", weights])
        if fmt is not None:
            cmd.extend(["--format", fmt])

        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"

        try:
            if sys.platform == "win32":
                # Wrap in a .bat with `pause` so the console
                # stays open for the user to see results
                cmd_line = subprocess.list2cmdline(cmd)
                with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".bat", delete=False,
                        newline="") as f:
                    f.write("@echo off\n" + cmd_line + "\npause\n")
                    bat_path = f.name
                subprocess.Popen(
                    [bat_path],
                    creationflags=subprocess.CREATE_NEW_CONSOLE,
                    env=env,
                )
            else:
                subprocess.Popen(cmd, env=env)
        except Exception as exc:
            QMessageBox.critical(
                self.window(), "YOLO training",
                f"Failed to start training process:\n{exc}")
            return

        QMessageBox.information(
            self.window(), "YOLO training started",
            "YOLO training has been started in a separate process "
            "to avoid memory issues.\n\n"
            "On Windows a new console window will show training "
            "progress. On other platforms, check the terminal "
            "from which Mufasa was launched.\n\n"
            f"Results will be saved to:\n{save_dir}",
        )


__all__ = ["YOLOPoseTrainForm"]
