"""
mufasa.ui_qt.forms.pose_tools
=============================

Project-independent pose utilities — two inline Qt forms that live
on the Tools workbench page:

* :class:`PoseReorganizerForm` — re-order keypoints in a directory
  of DLC / maDLC pose files. Two-stage UX: pick the data folder +
  tracking tool + file format, click Load order to discover the
  current keypoint list, then re-map each body-part to a new
  position. Replaces :class:`PoseReorganizerPopUp`.

* :class:`SLEAPToYoloForm` — convert SLEAP CSV predictions into
  YOLOv8 keypoint-format annotations. Many tuning knobs (per-
  video frame count, train/test split, threshold, padding,
  greyscale, CLAHE, single-vs-multi-instance) match the legacy
  popup one-to-one. Replaces :class:`SLEAPcsvInference2Yolo`.

Patch 122ac: both forms moved off the legacy Tk popups onto the
Tools page so v1 users have a stable surface for these
operations. Neither form needs an open project — they take their
inputs (directories of pose files / SLEAP CSVs / videos) directly
from the user, which is also why they live on Tools and not on
Data Import.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QPushButton, QScrollArea,
                               QSpinBox, QVBoxLayout, QWidget)

from mufasa.ui_qt.workbench import OperationForm


# =========================================================================== #
# PoseReorganizerForm
# =========================================================================== #
class PoseReorganizerForm(OperationForm):
    """Re-order keypoints in a directory of DLC / maDLC pose files.

    Replaces :class:`PoseReorganizerPopUp`. The legacy popup
    surfaces the operation in two stages — pick data
    folder/tool/format, click Confirm to load the current
    order, then choose a new order from dropdowns. The Qt port
    preserves this two-stage flow:

    1. **Configure** — data folder picker + tracking tool combo
       (DLC / maDLC) + file format combo (csv / h5). Click
       *Load order* to spin up the
       :class:`KeypointReorganizer` and surface the existing
       body-part list.
    2. **Reorder** — for each existing slot, a combo box lets
       the user pick which body-part should land there. The
       Run button (base-class) triggers the backend
       ``run(bp_lst, animal_list)`` call.

    No backend changes; just a project-less Qt surface for the
    same operation.
    """

    title = "Re-order pose keypoints"
    description = (
        "Re-arrange the keypoint column order in a directory of "
        "DeepLabCut or multi-animal DLC pose files. Useful when "
        "different recording sessions used different body-part "
        "orderings and need to be unified before feature "
        "extraction. Works on CSV or H5; pick a tool / format "
        "above, click <b>Load order</b>, then re-map each "
        "body-part to its new position."
    )

    def build(self) -> None:
        # ----- Stage 1: configure ----- #
        cfg = QGroupBox("Configure", self)
        cfg_layout = QFormLayout(cfg)

        self.data_folder_edit = QLineEdit(self)
        self.data_folder_edit.setReadOnly(True)
        self.data_folder_edit.setPlaceholderText(
            "Folder of pose-estimation files",
        )
        browse = QPushButton("Browse…", self)
        browse.clicked.connect(self._pick_data_folder)
        df_row = QHBoxLayout()
        df_row.addWidget(self.data_folder_edit, 1)
        df_row.addWidget(browse)
        cfg_layout.addRow("Data folder:", df_row)

        self.tool_cb = QComboBox(self)
        self.tool_cb.addItems(["DLC", "maDLC"])
        cfg_layout.addRow("Tracking tool:", self.tool_cb)

        self.format_cb = QComboBox(self)
        self.format_cb.addItems(["csv", "h5"])
        cfg_layout.addRow("File format:", self.format_cb)

        load_btn = QPushButton("Load order", self)
        load_btn.clicked.connect(self._load_order)
        cfg_layout.addRow("", load_btn)

        self.body_layout.addWidget(cfg)

        # ----- Stage 2: reorder (populated by _load_order) ----- #
        self.reorder_group = QGroupBox("Set new order", self)
        self.reorder_group.setVisible(False)
        # Scroll area in case there are many body-parts.
        scroll = QScrollArea(self.reorder_group)
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(220)
        self._reorder_host = QWidget(scroll)
        self._reorder_layout = QFormLayout(self._reorder_host)
        scroll.setWidget(self._reorder_host)
        rg_layout = QVBoxLayout(self.reorder_group)
        rg_layout.addWidget(scroll)
        self.body_layout.addWidget(self.reorder_group)

        # Storage populated by _load_order
        self._original_animals: Optional[list[str]] = None
        self._original_bps: list[str] = []
        self._row_combos: list[
            tuple[Optional[QComboBox], QComboBox]
        ] = []
        self._reorganizer = None

    def _pick_data_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Pick the data folder", "",
        )
        if d:
            self.data_folder_edit.setText(d)

    def _load_order(self) -> None:
        """Stage-1 → stage-2: discover the existing keypoint list."""
        data_folder = self.data_folder_edit.text().strip()
        if not data_folder or not Path(data_folder).is_dir():
            QMessageBox.warning(
                self, "Pose Reorganizer",
                "Pick an existing data folder first.",
            )
            return
        try:
            from mufasa.pose_processors.reorganize_keypoint import (
                KeypointReorganizer,
            )
            self._reorganizer = KeypointReorganizer(
                data_folder=data_folder,
                pose_tool=self.tool_cb.currentText(),
                file_format=self.format_cb.currentText(),
            )
        except Exception as exc:
            QMessageBox.critical(
                self, "Pose Reorganizer",
                f"Couldn't read keypoints from {data_folder!r}:\n\n"
                f"{type(exc).__name__}: {exc}",
            )
            return

        self._original_animals = self._reorganizer.animal_list
        self._original_bps = list(self._reorganizer.bp_list)
        self._populate_reorder_panel()

    def _populate_reorder_panel(self) -> None:
        """Build a row per existing slot. maDLC mode adds a parallel
        animal-id combo; DLC mode just shows the body-part combo."""
        # Clear any prior panel state
        while self._reorder_layout.rowCount():
            self._reorder_layout.removeRow(0)
        self._row_combos.clear()

        is_madlc = bool(self._original_animals)
        for i, bp in enumerate(self._original_bps):
            row_widget = QWidget(self._reorder_host)
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)

            animal_cb: Optional[QComboBox] = None
            if is_madlc:
                animal_cb = QComboBox(row_widget)
                # de-dupe while preserving order
                seen, animals_unique = set(), []
                for a in self._original_animals:
                    if a not in seen:
                        seen.add(a)
                        animals_unique.append(a)
                animal_cb.addItems(animals_unique)
                animal_cb.setCurrentText(
                    self._original_animals[i],
                )
                row_layout.addWidget(animal_cb)

            bp_cb = QComboBox(row_widget)
            bp_cb.addItems(self._original_bps)
            bp_cb.setCurrentText(bp)
            row_layout.addWidget(bp_cb, 1)

            current_label = QLabel(
                f"<i>was: {bp}</i>" if not is_madlc
                else f"<i>was: {self._original_animals[i]}/{bp}</i>",
                row_widget,
            )
            current_label.setStyleSheet(
                "color: palette(placeholder-text);",
            )
            row_layout.addWidget(current_label)

            self._reorder_layout.addRow(
                f"Position {i + 1}:", row_widget,
            )
            self._row_combos.append((animal_cb, bp_cb))

        self.reorder_group.setVisible(True)

    def collect_args(self) -> dict:
        if not self._reorganizer:
            raise ValueError(
                "Click 'Load order' before running."
            )
        new_bps = [bp_cb.currentText() for _, bp_cb in self._row_combos]
        new_animals = (
            [animal_cb.currentText()
             for animal_cb, _ in self._row_combos
             if animal_cb is not None]
            if self._original_animals else None
        )
        return {"bp_lst": new_bps, "animal_list": new_animals}

    def target(self, *, bp_lst: list[str],
               animal_list: Optional[list[str]]) -> None:
        if self._reorganizer is None:
            raise RuntimeError(
                "Reorganizer not initialised — re-load order.",
            )
        self._reorganizer.run(bp_lst=bp_lst, animal_list=animal_list)


# =========================================================================== #
# SLEAPToYoloForm
# =========================================================================== #
class SLEAPToYoloForm(OperationForm):
    """Convert SLEAP CSV inference into YOLOv8 keypoint annotations.

    Replaces :class:`SLEAPcsvInference2Yolo`. The legacy popup has
    ~12 form controls; the Qt port surfaces all of them with the
    same defaults. Backend:
    :class:`mufasa.third_party_label_appenders.transform.sleap_csv_to_yolo.Sleap2Yolo`.

    Body-part flip indices (``flip_idx``) and category names
    (``names``) are computed from the SLEAP files at run time —
    no separate UI surface.
    """

    title = "SLEAP → YOLO conversion"
    description = (
        "Convert a directory of SLEAP CSV predictions into "
        "YOLOv8 keypoint-format training annotations. Sub-samples "
        "frames per video, optionally splits into train/val, and "
        "writes images + label files to the chosen output folder."
    )

    def build(self) -> None:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        # Three directory pickers — share a small helper.
        self.sleap_dir_edit = self._make_dir_row(
            form, "SLEAP data directory:",
            "Folder of SLEAP CSV files",
        )
        self.video_dir_edit = self._make_dir_row(
            form, "Video directory:",
            "Folder of source videos",
        )
        self.save_dir_edit = self._make_dir_row(
            form, "Save directory:",
            "Where YOLO annotations will be written",
        )

        # Numeric / dropdown options — defaults match the legacy popup.
        self.frames_per_video = QSpinBox(self)
        self.frames_per_video.setRange(50, 5_000)
        self.frames_per_video.setSingleStep(50)
        self.frames_per_video.setValue(100)
        form.addRow("Frames per video:", self.frames_per_video)

        self.train_size = QSpinBox(self)
        self.train_size.setRange(10, 100)
        self.train_size.setSingleStep(10)
        self.train_size.setValue(70)
        self.train_size.setSuffix(" %")
        form.addRow("Train size:", self.train_size)

        self.threshold = QSpinBox(self)
        self.threshold.setRange(10, 100)
        self.threshold.setSingleStep(10)
        self.threshold.setValue(90)
        self.threshold.setSuffix(" %")
        form.addRow("Detection threshold:", self.threshold)

        self.padding = QDoubleSpinBox(self)
        self.padding.setRange(0.0, 10.0)
        self.padding.setSingleStep(0.05)
        self.padding.setDecimals(2)
        self.padding.setValue(0.0)
        self.padding.setSpecialValueText("None")
        form.addRow("Padding:", self.padding)

        self.greyscale = QCheckBox("Greyscale output", self)
        form.addRow("", self.greyscale)
        self.clahe = QCheckBox("Apply CLAHE", self)
        form.addRow("", self.clahe)
        self.single_id = QCheckBox(
            "Single-instance (collapse animals)", self,
        )
        form.addRow("", self.single_id)
        self.verbose = QCheckBox("Verbose progress logging", self)
        self.verbose.setChecked(True)
        form.addRow("", self.verbose)

        self.body_layout.addLayout(form)

    def _make_dir_row(self, form: QFormLayout, label: str,
                      placeholder: str) -> QLineEdit:
        edit = QLineEdit(self)
        edit.setReadOnly(True)
        edit.setPlaceholderText(placeholder)
        browse = QPushButton("Browse…", self)

        def _pick() -> None:
            d = QFileDialog.getExistingDirectory(
                self, f"Pick {label.rstrip(':')}", "",
            )
            if d:
                edit.setText(d)
        browse.clicked.connect(_pick)
        row = QHBoxLayout()
        row.addWidget(edit, 1)
        row.addWidget(browse)
        form.addRow(label, row)
        return edit

    def collect_args(self) -> dict:
        sleap = self.sleap_dir_edit.text().strip()
        video = self.video_dir_edit.text().strip()
        save = self.save_dir_edit.text().strip()
        if not sleap or not Path(sleap).is_dir():
            raise ValueError("Pick a valid SLEAP data directory.")
        if not video or not Path(video).is_dir():
            raise ValueError("Pick a valid video directory.")
        if not save or not Path(save).is_dir():
            raise ValueError("Pick a valid save directory.")

        # Padding: QDoubleSpinBox special-value (min) maps to None
        pad_val = self.padding.value()
        padding = None if pad_val == self.padding.minimum() else pad_val

        return {
            "data_dir":           sleap,
            "video_dir":          video,
            "save_dir":           save,
            "frms_cnt":           int(self.frames_per_video.value()),
            "train_size":         int(self.train_size.value()),
            "instance_threshold": int(self.threshold.value()),
            "padding":            padding,
            "greyscale":          bool(self.greyscale.isChecked()),
            "clahe":              bool(self.clahe.isChecked()),
            "single_id":          bool(self.single_id.isChecked()),
            "verbose":            bool(self.verbose.isChecked()),
        }

    def target(self, *, data_dir: str, video_dir: str,
               save_dir: str, frms_cnt: int, train_size: int,
               instance_threshold: int, padding,
               greyscale: bool, clahe: bool, single_id: bool,
               verbose: bool) -> None:
        from mufasa.third_party_label_appenders.transform.sleap_csv_to_yolo import (
            Sleap2Yolo,
        )
        from mufasa.third_party_label_appenders.transform.utils import (
            get_yolo_keypoint_flip_idx,
        )
        # flip_idx + names derived from the SLEAP data files at
        # run time. The backend's own helpers know how to read
        # them; we pass through with sensible defaults.
        runner = Sleap2Yolo(
            data_dir=data_dir, video_dir=video_dir, save_dir=save_dir,
            frms_cnt=frms_cnt, verbose=verbose,
            instance_threshold=instance_threshold,
            train_size=train_size, flip_idx=None, names=None,
            greyscale=greyscale, clahe=clahe, padding=padding,
            single_id=single_id,
        )
        runner.run()


__all__ = ["PoseReorganizerForm", "SLEAPToYoloForm"]
