"""
mufasa.ui_qt.create_project_dialog
==================================

Native Qt dialog for creating a new Mufasa project. Wraps
:class:`mufasa.utils.config_creator.ProjectConfigCreator` with a form
that matches the legacy Tk flow:

* project parent directory (Browse…)
* project name
* body-part preset (dropdown from pose_config_names.csv)
* animal count (auto-populated from no_animals.csv but editable)
* classifier names (newline-separated; blank lines skipped)
* file type (csv / parquet)

On success returns the full path to the generated ``project_config.ini``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox,
                               QFileDialog, QFormLayout, QHBoxLayout,
                               QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
                               QPushButton, QSpinBox, QVBoxLayout, QWidget)

import mufasa
from mufasa.utils.enums import Paths
from mufasa.utils.lookups import get_bp_config_codes


class CreateProjectDialog(QDialog):
    """Modal dialog that creates a new project. Use ``config_path``
    after ``exec()`` returns ``QDialog.Accepted`` to retrieve the path
    to the generated ``project_config.ini``."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Create new Mufasa project")
        self.setModal(True)
        self.resize(560, 520)
        self.config_path: Optional[str] = None

        # Resolve pose-preset metadata (same source of truth as the Tk UI)
        simba_dir = Path(mufasa.__file__).parent
        names_path = simba_dir / Paths.PROJECT_POSE_CONFIG_NAMES.value
        animals_path = simba_dir / Paths.SIMBA_NO_ANIMALS_PATH.value
        self._preset_names: List[str] = list(
            pd.read_csv(names_path, header=None)[0]
        )
        self._preset_animal_counts: List[int] = list(
            pd.read_csv(animals_path, header=None)[0]
        )
        self._preset_codes = get_bp_config_codes()

        # ------------------ Widgets ----------------------------------- #
        self._dir_edit = QLineEdit(); self._dir_edit.setReadOnly(True)
        self._dir_edit.setPlaceholderText("Parent directory for the project folder")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._pick_dir)
        dir_row = QWidget()
        dl = QHBoxLayout(dir_row); dl.setContentsMargins(0, 0, 0, 0)
        dl.addWidget(self._dir_edit, 1); dl.addWidget(browse)

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g. my_experiment_2026")

        self._preset_combo = QComboBox()
        self._preset_combo.addItems(self._preset_names)
        self._preset_combo.currentIndexChanged.connect(self._preset_changed)

        # Autodetect-from-DLC path: populates self._autodetected_bps.
        # While set, the preset dropdown is disabled and the animal-
        # count spinner is forced to 1. Accept handler uses user_defined.
        self._autodetected_bps: list[str] = []
        self._autodetect_label = QLabel("—")
        self._autodetect_label.setStyleSheet("color: palette(mid);")
        autodetect_btn = QPushButton("Auto-detect from DLC file…")
        autodetect_btn.clicked.connect(self._autodetect_from_dlc)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_autodetect)
        autodetect_row = QWidget()
        ar = QHBoxLayout(autodetect_row); ar.setContentsMargins(0, 0, 0, 0)
        ar.addWidget(self._autodetect_label, 1)
        ar.addWidget(autodetect_btn); ar.addWidget(clear_btn)

        self._animal_count = QSpinBox()
        self._animal_count.setRange(1, 100)
        self._animal_count.setValue(self._preset_animal_counts[0])

        self._clf_edit = QPlainTextEdit()
        self._clf_edit.setPlaceholderText(
            "One classifier per line, e.g.:\n"
            "Attack\n"
            "Groom\n"
            "Rear"
        )
        self._clf_edit.setFixedHeight(110)

        self._file_type_combo = QComboBox()
        self._file_type_combo.addItems(["csv", "parquet"])

        # ------------------ Layout ------------------------------------ #
        form = QFormLayout()
        form.addRow("Project parent directory:", dir_row)
        form.addRow("Project name:", self._name_edit)
        form.addRow("Body-part preset:", self._preset_combo)
        form.addRow("Or auto-detect:", autodetect_row)
        form.addRow("Animal count:", self._animal_count)
        form.addRow("Classifier names:", self._clf_edit)
        form.addRow("Workflow file type:", self._file_type_combo)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            parent=self,
        )
        buttons.button(QDialogButtonBox.Ok).setText("Create")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)

        root = QVBoxLayout(self)
        hint = QLabel(
            "This creates a new project directory tree and "
            "<code>project_config.ini</code>.<br>"
            "After creation, the workbench will reopen pointing at the "
            "new project."
        )
        hint.setWordWrap(True)
        hint.setTextFormat(Qt.RichText)
        root.addWidget(hint)
        root.addLayout(form)
        root.addStretch(1)
        root.addWidget(buttons)

    # -------------- Slots -------------------------------------------- #
    def _pick_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Select parent directory for the project", ""
        )
        if d:
            self._dir_edit.setText(d)

    def _preset_changed(self, idx: int) -> None:
        try:
            self._animal_count.setValue(int(self._preset_animal_counts[idx]))
        except (IndexError, ValueError):
            pass

    def _autodetect_from_dlc(self) -> None:
        """Pick a DLC file and pre-populate the body-part list. When
        this path is used, the preset dropdown is ignored and the new
        project is created with ``user_defined`` + detected bps."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Pick a DLC output file", "",
            "DLC output (*.h5 *.csv);;All files (*)",
        )
        if not path:
            return
        # Deferred import: the autodetect module imports pandas, which
        # we shouldn't load unless the user actually uses this path.
        from mufasa.pose_importers.dlc_autodetect import (
            DLCAutodetectError, extract_bodyparts,
        )
        try:
            bps = extract_bodyparts(path)
        except DLCAutodetectError as exc:
            QMessageBox.warning(
                self, "Could not read DLC file", str(exc),
            )
            return
        except Exception as exc:
            QMessageBox.critical(
                self, "Unexpected error reading DLC file",
                f"{type(exc).__name__}: {exc}",
            )
            return

        self._autodetected_bps = bps
        self._autodetect_label.setText(
            f"{len(bps)} body-parts from {Path(path).name}"
        )
        # When autodetect wins, preset selection is irrelevant.
        self._preset_combo.setEnabled(False)
        self._animal_count.setValue(1)
        self._animal_count.setEnabled(False)

    def _clear_autodetect(self) -> None:
        self._autodetected_bps = []
        self._autodetect_label.setText("—")
        self._preset_combo.setEnabled(True)
        self._animal_count.setEnabled(True)
        # Restore the preset's default animal count
        self._preset_changed(self._preset_combo.currentIndex())

    def _accept(self) -> None:
        parent_dir = self._dir_edit.text().strip()
        name = self._name_edit.text().strip()
        file_type = self._file_type_combo.currentText()
        classifiers = [
            line.strip()
            for line in self._clf_edit.toPlainText().splitlines()
            if line.strip()
        ]

        # Common validation
        if not parent_dir:
            QMessageBox.warning(self, "Missing field",
                                "Pick a parent directory.")
            return
        if not Path(parent_dir).is_dir():
            QMessageBox.warning(self, "Not a directory",
                                f"{parent_dir} does not exist.")
            return
        if not name:
            QMessageBox.warning(self, "Missing field",
                                "Enter a project name.")
            return
        bad_chars = set(' /\\?%*:|"<>\t\n')
        if any(c in bad_chars for c in name):
            QMessageBox.warning(
                self, "Invalid project name",
                "Project name can't contain spaces, slashes, or "
                "shell-unfriendly characters. Use letters, digits, "
                "underscore, dash.",
            )
            return
        target = Path(parent_dir) / name
        if target.exists():
            QMessageBox.warning(
                self, "Already exists",
                f"{target} already exists. Pick a different name or "
                "parent directory.",
            )
            return
        if not classifiers:
            QMessageBox.warning(self, "Missing field",
                                "Enter at least one classifier name.")
            return

        # Branch on autodetect vs preset. We still go through
        # ProjectConfigCreator for both paths (it builds the directory
        # tree etc.), then for autodetect we reconfigure the freshly-
        # created project to user_defined + our detected bps.
        if self._autodetected_bps:
            # Pick an arbitrary 1-animal preset just to get the tree
            # created; we'll overwrite the pose_estimation fields right
            # after. '1 animal; 7 body-parts' is first, exists in all
            # forks — doesn't matter which, it's getting overwritten.
            preset = "1 animal; 7 body-parts"
            config_code = self._preset_codes.get(preset, "7")
            preset_idx = (self._preset_names.index(preset)
                          if preset in self._preset_names else 1)
            animal_cnt = 1
        else:
            preset = self._preset_combo.currentText()
            animal_cnt = int(self._animal_count.value())
            config_code = self._preset_codes.get(preset)
            if config_code is None:
                QMessageBox.critical(
                    self, "Unknown preset",
                    f"Preset {preset!r} has no associated config code. "
                    "This is a Mufasa bug — file an issue.",
                )
                return
            preset_idx = self._preset_names.index(preset)

        # Actually create it
        from mufasa.utils.config_creator import ProjectConfigCreator
        try:
            creator = ProjectConfigCreator(
                project_path=parent_dir,
                project_name=name,
                target_list=classifiers,
                pose_estimation_bp_cnt=config_code,
                body_part_config_idx=preset_idx,
                animal_cnt=animal_cnt,
                file_type=file_type,
            )
        except Exception as exc:  # broad: config_creator raises many flavours
            QMessageBox.critical(
                self, "Project creation failed",
                f"{type(exc).__name__}: {exc}",
            )
            return

        # If autodetect was used, immediately reconfigure the fresh
        # project to user_defined + detected body parts. This keeps
        # the two paths (create + reconfigure) sharing the same
        # reconfigure helper, which has its own tests.
        if self._autodetected_bps:
            from mufasa.utils.project_reconfigure import (
                ProjectReconfigureError,
                reconfigure_project_user_defined,
            )
            try:
                reconfigure_project_user_defined(
                    config_path=creator.config_path,
                    body_parts=self._autodetected_bps,
                    animal_cnt=1,
                )
            except ProjectReconfigureError as exc:
                QMessageBox.critical(
                    self, "Autodetect reconfigure failed",
                    f"Project directory was created, but body-part "
                    f"reconfigure failed: {exc}. You can still open the "
                    f"project and run File → Reconfigure project from "
                    f"DLC file… to retry.",
                )
                # Still accept — the project exists, it's just on a
                # preset that doesn't match the data.

        self.config_path = creator.config_path
        self.accept()
