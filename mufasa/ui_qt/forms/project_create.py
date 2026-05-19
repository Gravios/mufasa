"""
mufasa.ui_qt.forms.project_create
=================================

Inline form widget for creating a new v1-layout Mufasa project
(``project.toml`` + ``sources/``, ``derived/``, ``models/``,
``logs/``). Same field set the legacy Tk popup and the
:class:`mufasa.ui_qt.create_project_dialog.CreateProjectDialog`
modal expose:

* project parent directory (Browse…)
* project name
* body-part preset (dropdown from pose_config_names.csv) OR
  auto-detect body parts from a DLC output file
* animal count (auto-populated from no_animals.csv; editable
  unless autodetect is active, in which case forced to 1)
* classifier names (optional; one per line)
* workflow file type (csv / parquet)

Patch 122l: extracted from CreateProjectDialog so the same
fields can be embedded inline on the Projects page's
"Create or open project" frame without forcing a modal dialog.
CreateProjectDialog now wraps an instance of this widget.

The form emits :pyattr:`project_created` with the absolute path
to the generated ``project.toml`` on success. Validation
errors surface as :class:`QMessageBox` dialogs (modal but
non-blocking-after-dismissal), matching the legacy UX.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QComboBox, QFileDialog, QFormLayout,
                               QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QPlainTextEdit,
                               QPushButton, QSpinBox,
                               QVBoxLayout, QWidget)

import mufasa
from mufasa.utils.enums import Paths
from mufasa.utils.lookups import get_bp_config_codes


class ProjectCreateForm(QWidget):
    """Inline form for creating a new Mufasa project.

    Behaviour matches the legacy CreateProjectDialog; the only
    structural difference is that this is a :class:`QWidget`
    instead of a :class:`QDialog`, so it can be embedded inside
    another page or wrapped by a modal dialog.

    Parameters
    ----------
    parent
        Parent widget; standard Qt.
    show_create_button
        When True (the default), the form draws its own
        "Create project" button at the bottom of the field
        column. When False, the embedding widget is expected to
        supply its own action button and trigger
        :meth:`submit` directly. The :class:`CreateProjectDialog`
        wrapper uses this to put the Create action inside a
        QDialogButtonBox.
    """

    # Emitted on successful project creation with the absolute
    # path to the new ``project.toml``.
    project_created = Signal(str)

    def __init__(self,
                 parent: Optional[QWidget] = None,
                 *,
                 show_create_button: bool = True) -> None:
        super().__init__(parent)
        self._show_create_button = show_create_button

        # ----- Source of truth for presets ------------------------- #
        # Same lookup files the legacy Tk UI uses; if they ever move
        # to project.toml, only this block changes.
        mufasa_dir = Path(mufasa.__file__).parent
        names_path = mufasa_dir / Paths.PROJECT_POSE_CONFIG_NAMES.value
        animals_path = mufasa_dir / Paths.SIMBA_NO_ANIMALS_PATH.value
        self._preset_names: List[str] = list(
            pd.read_csv(names_path, header=None)[0],
        )
        self._preset_animal_counts: List[int] = list(
            pd.read_csv(animals_path, header=None)[0],
        )
        self._preset_codes = get_bp_config_codes()

        # Autodetect state — populated when the user picks a DLC
        # file. While non-empty the preset combo is disabled and
        # animal_count is pinned to 1 (multi-animal autodetect is
        # not supported by the legacy parser).
        self._autodetected_bps: list[str] = []

        self._build_shell()

    # ------------------------------------------------------------------ #
    # UI scaffolding
    # ------------------------------------------------------------------ #
    def _build_shell(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        # ----- Widgets ----- #
        self._dir_edit = QLineEdit(self)
        self._dir_edit.setReadOnly(True)
        self._dir_edit.setPlaceholderText(
            "Parent directory for the new project folder",
        )
        browse = QPushButton("Browse…", self)
        browse.clicked.connect(self._pick_dir)
        dir_row = QWidget(self)
        dl = QHBoxLayout(dir_row)
        dl.setContentsMargins(0, 0, 0, 0)
        dl.addWidget(self._dir_edit, 1)
        dl.addWidget(browse)

        self._name_edit = QLineEdit(self)
        self._name_edit.setPlaceholderText("e.g. my_experiment_2026")

        self._preset_combo = QComboBox(self)
        self._preset_combo.addItems(self._preset_names)
        self._preset_combo.currentIndexChanged.connect(self._preset_changed)

        # Autodetect-from-DLC path
        self._autodetect_label = QLabel("—", self)
        self._autodetect_label.setStyleSheet(
            "color: palette(placeholder-text);",
        )
        autodetect_btn = QPushButton("Auto-detect from DLC file…", self)
        autodetect_btn.clicked.connect(self._autodetect_from_dlc)
        clear_btn = QPushButton("Clear", self)
        clear_btn.clicked.connect(self._clear_autodetect)
        autodetect_row = QWidget(self)
        ar = QHBoxLayout(autodetect_row)
        ar.setContentsMargins(0, 0, 0, 0)
        ar.addWidget(self._autodetect_label, 1)
        ar.addWidget(autodetect_btn)
        ar.addWidget(clear_btn)

        self._animal_count = QSpinBox(self)
        self._animal_count.setRange(1, 100)
        self._animal_count.setValue(self._preset_animal_counts[0])

        self._clf_edit = QPlainTextEdit(self)
        self._clf_edit.setPlaceholderText(
            "Optional — one classifier per line. You can add these "
            "later from the Classifier page. Examples:\n"
            "Attack\n"
            "Groom\n"
            "Rear"
        )
        self._clf_edit.setFixedHeight(110)

        self._file_type_combo = QComboBox(self)
        self._file_type_combo.addItems(["csv", "parquet"])

        # ----- Layout ----- #
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)
        form.addRow("Project parent directory:", dir_row)
        form.addRow("Project name:", self._name_edit)
        form.addRow("Body-part preset:", self._preset_combo)
        form.addRow("Or auto-detect:", autodetect_row)
        form.addRow("Animal count:", self._animal_count)
        form.addRow("Classifier names (optional):", self._clf_edit)
        form.addRow("Workflow file type:", self._file_type_combo)
        outer.addLayout(form)

        if self._show_create_button:
            btn_row = QHBoxLayout()
            btn_row.addStretch()
            self.create_btn = QPushButton("Create project", self)
            self.create_btn.setMinimumWidth(160)
            self.create_btn.clicked.connect(self.submit)
            btn_row.addWidget(self.create_btn)
            outer.addLayout(btn_row)
        else:
            # Embedder is responsible for the Create button. Expose
            # the submit() method so the embedder can wire its own
            # button to it.
            self.create_btn = None

    # ------------------------------------------------------------------ #
    # Slots
    # ------------------------------------------------------------------ #
    def _pick_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Select parent directory for the project", "",
        )
        if d:
            self._dir_edit.setText(d)

    def _preset_changed(self, idx: int) -> None:
        try:
            self._animal_count.setValue(
                int(self._preset_animal_counts[idx]),
            )
        except (IndexError, ValueError):
            pass

    def _autodetect_from_dlc(self) -> None:
        """Pick a DLC file and pre-populate the body-part list.
        When this path is used, the preset dropdown is ignored
        and the new project is created with ``user_defined`` +
        the parsed body parts.
        """
        path, _ = QFileDialog.getOpenFileName(
            self, "Pick a DLC output file", "",
            "DLC output (*.h5 *.csv);;All files (*)",
        )
        if not path:
            return
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
            f"{len(bps)} body-parts from {Path(path).name}",
        )
        self._preset_combo.setEnabled(False)
        self._animal_count.setValue(1)
        self._animal_count.setEnabled(False)

    def _clear_autodetect(self) -> None:
        self._autodetected_bps = []
        self._autodetect_label.setText("—")
        self._preset_combo.setEnabled(True)
        self._animal_count.setEnabled(True)
        self._preset_changed(self._preset_combo.currentIndex())

    # ------------------------------------------------------------------ #
    # Submission
    # ------------------------------------------------------------------ #
    def submit(self) -> Optional[str]:
        """Validate fields and attempt to create the project.

        :returns: the absolute path to the new ``project.toml`` on
            success, or ``None`` if validation failed. Emits
            :pyattr:`project_created` on success so caller
            widgets can react without polling.
        """
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
            QMessageBox.warning(
                self, "Missing field", "Pick a parent directory.",
            )
            return None
        if not Path(parent_dir).is_dir():
            QMessageBox.warning(
                self, "Not a directory", f"{parent_dir} does not exist.",
            )
            return None
        if not name:
            QMessageBox.warning(
                self, "Missing field", "Enter a project name.",
            )
            return None
        bad_chars = set(' /\\?%*:|"<>\t\n')
        if any(c in bad_chars for c in name):
            QMessageBox.warning(
                self, "Invalid project name",
                "Project name can't contain spaces, slashes, or "
                "shell-unfriendly characters. Use letters, digits, "
                "underscore, dash.",
            )
            return None
        target = Path(parent_dir) / name
        if target.exists():
            QMessageBox.warning(
                self, "Already exists",
                f"{target} already exists. Pick a different name or "
                "parent directory.",
            )
            return None

        # Branch on autodetect vs preset. Both paths funnel into
        # the same v1 ProjectConfigCreator call.
        if self._autodetected_bps:
            config_code = "user_defined"
            preset_idx = 0
            animal_cnt = 1
            body_parts = list(self._autodetected_bps)
        else:
            preset = self._preset_combo.currentText()
            animal_cnt = int(self._animal_count.value())
            config_code = self._preset_codes.get(preset)
            if config_code is None:
                QMessageBox.critical(
                    self, "Unknown preset",
                    f"Preset {preset!r} has no associated config "
                    "code. This is a Mufasa bug — file an issue.",
                )
                return None
            preset_idx = self._preset_names.index(preset)
            body_parts = None  # ProjectConfigCreator looks up the
                               # preset row in bp_names.csv.

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
                body_parts=body_parts,
            )
        except Exception as exc:  # broad: creator raises many flavours
            QMessageBox.critical(
                self, "Project creation failed",
                f"{type(exc).__name__}: {exc}",
            )
            return None

        self.project_created.emit(creator.config_path)
        return creator.config_path


__all__ = ["ProjectCreateForm"]
