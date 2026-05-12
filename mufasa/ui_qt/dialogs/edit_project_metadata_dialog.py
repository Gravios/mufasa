"""
mufasa.ui_qt.dialogs.edit_project_metadata_dialog
=================================================

Modal dialog for editing the metadata of an already-loaded
project (file type, body parts, animal count, animal IDs,
classifier targets). Reached from the Project information
section's Edit button on the Projects page.

Patch 122n: covers the common case where project.toml's body
parts list got out of sync with the imported pose data —
e.g. the project was created with one preset but DLC files
with different body-part names were imported later. The dialog
also surfaces an **Auto-detect from pose file** button that
reads body parts from an existing pose file rather than
forcing the user to retype them.

v1 only. For legacy ``project_config.ini`` projects, the
File → Reconfigure project from DLC file… menu action remains
the editing entry point; this dialog refuses to open against a
legacy project rather than silently corrupting the INI.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox,
                               QFileDialog, QFormLayout, QHBoxLayout,
                               QLabel, QLineEdit, QMessageBox,
                               QPlainTextEdit, QPushButton, QSpinBox,
                               QVBoxLayout, QWidget)


class EditProjectMetadataDialog(QDialog):
    """Modal editor for a project's ``[pose]`` and ``[classifiers]``
    TOML sections.

    Reads current values from ``project.toml`` on construction
    and pre-populates each field. On Save, validates inputs and
    writes them back via
    :func:`mufasa.project_layout.write_project_toml`.

    Emits :pyattr:`metadata_updated` with the config_path on
    successful save; callers can subscribe to refresh dependent
    views (e.g. :class:`ProjectInfoForm._populate`).
    """

    metadata_updated = Signal(str)

    def __init__(self,
                 config_path: str,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit project information")
        self.setModal(True)
        self.resize(560, 580)
        self.config_path = str(config_path)
        self._is_v1 = self.config_path.lower().endswith(".toml")

        self._build_shell()
        self._populate_from_disk()

    # ------------------------------------------------------------------ #
    # UI scaffolding
    # ------------------------------------------------------------------ #
    def _build_shell(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(10)

        if not self._is_v1:
            # Legacy projects: refuse rather than silently scribble
            # over project_config.ini. The File → Reconfigure menu
            # action is the legacy editing path.
            warning = QLabel(
                "<b>Legacy project.</b><br>"
                "Inline editing isn't supported for legacy "
                "<code>project_config.ini</code> projects. Use "
                "<b>File → Reconfigure project from DLC file…</b> "
                "or convert this project to v1 first.",
                self,
            )
            warning.setTextFormat(Qt.RichText)
            warning.setWordWrap(True)
            root.addWidget(warning)
            close_btn = QPushButton("Close", self)
            close_btn.clicked.connect(self.reject)
            root.addStretch()
            row = QHBoxLayout()
            row.addStretch()
            row.addWidget(close_btn)
            root.addLayout(row)
            return

        # Hint at the top — clarify scope so users don't expect
        # this dialog to add new animals' pose data or rename the
        # project root.
        hint = QLabel(
            "Edits the project's <code>project.toml</code> in place. "
            "Affects metadata only — not the imported pose data on "
            "disk. Use <b>Auto-detect</b> to refresh the body-part "
            "list from one of your imported pose files.",
            self,
        )
        hint.setTextFormat(Qt.RichText)
        hint.setWordWrap(True)
        hint.setStyleSheet("color: palette(placeholder-text);")
        root.addWidget(hint)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)

        # File type
        self._file_type_combo = QComboBox(self)
        self._file_type_combo.addItems(["csv", "parquet", "h5"])
        form.addRow("File type:", self._file_type_combo)

        # Animal count
        self._animal_count = QSpinBox(self)
        self._animal_count.setRange(1, 100)
        form.addRow("Animal count:", self._animal_count)

        # Animal IDs — comma-separated. Empty entry → auto-fill
        # with Animal_1, Animal_2, … on save.
        self._animal_ids_edit = QLineEdit(self)
        self._animal_ids_edit.setPlaceholderText(
            "e.g. Mouse_A, Mouse_B (leave blank to auto-name "
            "Animal_1, Animal_2, …)"
        )
        form.addRow("Animal IDs:", self._animal_ids_edit)

        # Body parts — multiline. With an Auto-detect button on the
        # right.
        bp_widget = QWidget(self)
        bp_layout = QVBoxLayout(bp_widget)
        bp_layout.setContentsMargins(0, 0, 0, 0)
        bp_layout.setSpacing(4)

        self._body_parts_edit = QPlainTextEdit(bp_widget)
        self._body_parts_edit.setPlaceholderText(
            "One body part per line, e.g.\n"
            "Nose\nEar_left\nEar_right\nCenter\nTail_base"
        )
        self._body_parts_edit.setMinimumHeight(110)
        bp_layout.addWidget(self._body_parts_edit)

        bp_btn_row = QHBoxLayout()
        bp_btn_row.addStretch()
        autodetect_btn = QPushButton(
            "Auto-detect from pose file…", bp_widget,
        )
        autodetect_btn.clicked.connect(self._autodetect_body_parts)
        bp_btn_row.addWidget(autodetect_btn)
        bp_layout.addLayout(bp_btn_row)

        form.addRow("Body parts:", bp_widget)

        # Classifier targets
        self._classifiers_edit = QPlainTextEdit(self)
        self._classifiers_edit.setPlaceholderText(
            "One classifier name per line. Optional — can be "
            "added later from the Classifier page."
        )
        self._classifiers_edit.setMinimumHeight(90)
        form.addRow("Classifier targets:", self._classifiers_edit)

        root.addLayout(form)

        # OK / Cancel
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel, parent=self)
        save_btn = buttons.addButton(
            "Save", QDialogButtonBox.AcceptRole,
        )
        save_btn.clicked.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    # ------------------------------------------------------------------ #
    # Population from disk
    # ------------------------------------------------------------------ #
    def _populate_from_disk(self) -> None:
        if not self._is_v1:
            return
        try:
            from mufasa.project_layout import project_metadata_from_config
            meta = project_metadata_from_config(self.config_path)
        except (ValueError, OSError) as exc:
            QMessageBox.critical(
                self, "Cannot read project",
                f"Could not parse project.toml: {exc}",
            )
            self.reject()
            return

        ft = meta.get("file_type", "csv")
        idx = self._file_type_combo.findText(ft)
        if idx >= 0:
            self._file_type_combo.setCurrentIndex(idx)
        else:
            # Unknown file type — preserve it as a new entry so we
            # don't silently rewrite the field on save.
            self._file_type_combo.addItem(ft)
            self._file_type_combo.setCurrentText(ft)

        self._animal_count.setValue(int(meta.get("animal_count", 1)))
        ids = meta.get("animal_ids") or []
        self._animal_ids_edit.setText(", ".join(ids))
        self._body_parts_edit.setPlainText(
            "\n".join(meta.get("body_parts") or []),
        )
        self._classifiers_edit.setPlainText(
            "\n".join(meta.get("classifier_targets") or []),
        )

    # ------------------------------------------------------------------ #
    # Slots
    # ------------------------------------------------------------------ #
    def _autodetect_body_parts(self) -> None:
        """Open a file dialog (defaulting to <project>/sources/pose/),
        read body-part names from the picked file, and replace the
        text-area contents. Tries multiple parser strategies so it
        works for raw DLC h5/csv files as well as already-imported
        flat CSVs in sources/pose/."""
        try:
            from mufasa.project_layout import project_paths_from_config
            paths = project_paths_from_config(self.config_path)
            start_dir = paths["input_pose_dir"]
            if not Path(start_dir).is_dir():
                start_dir = paths["project_root"]
        except (ValueError, OSError):
            start_dir = ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Pick a pose file to read body parts from",
            start_dir,
            "Pose data (*.csv *.parquet *.h5 *.hdf5);;All files (*)",
        )
        if not path:
            return

        bps = self._extract_bodyparts_from_pose_file(path)
        if not bps:
            QMessageBox.warning(
                self, "Could not parse body parts",
                f"Couldn't extract body-part names from {Path(path).name}. "
                "If this is a raw DLC h5/csv file the multi-index header "
                "may be missing or in an unexpected format. Try a "
                "different file, or enter the body parts manually.",
            )
            return
        self._body_parts_edit.setPlainText("\n".join(bps))

    @staticmethod
    def _extract_bodyparts_from_pose_file(path: str) -> List[str]:
        """Strategy chain — try DLC's extract_bodyparts first
        (handles raw DLC h5/csv with multi-index headers), fall
        back to a flat-CSV parser that strips ``_x``/``_y``/
        ``_likelihood`` suffixes (handles post-import sources/pose/
        files), then to a parquet column scan.

        Returns an empty list if nothing could be extracted.
        """
        p = Path(path)
        ext = p.suffix.lower()

        # Strategy 1: dlc_autodetect (handles raw DLC files)
        try:
            from mufasa.pose_importers.dlc_autodetect import (
                extract_bodyparts,
            )
            bps = extract_bodyparts(str(p))
            if bps:
                # De-dup preserving order
                seen: list[str] = []
                for bp in bps:
                    if bp not in seen:
                        seen.append(bp)
                return seen
        except Exception:
            pass

        # Strategy 2: flat-CSV with x/y/likelihood suffixes
        if ext == ".csv":
            try:
                import pandas as pd
                df = pd.read_csv(p, nrows=0)
                cols = [str(c) for c in df.columns]
                bps: list[str] = []
                for c in cols:
                    for suffix in ("_x", "_y", "_likelihood", "_p"):
                        if c.endswith(suffix):
                            bp = c[: -len(suffix)]
                            if bp and bp not in bps:
                                bps.append(bp)
                            break
                if bps:
                    return bps
            except Exception:
                pass

        # Strategy 3: parquet — same suffix logic, columns read
        # lazily.
        if ext == ".parquet":
            try:
                import pyarrow.parquet as pq
                schema = pq.read_schema(str(p))
                cols = [f.name for f in schema]
                bps = []
                for c in cols:
                    for suffix in ("_x", "_y", "_likelihood", "_p"):
                        if c.endswith(suffix):
                            bp = c[: -len(suffix)]
                            if bp and bp not in bps:
                                bps.append(bp)
                            break
                if bps:
                    return bps
            except Exception:
                pass

        return []

    def _on_save(self) -> None:
        if not self._is_v1:
            self.reject()
            return

        # ----- Gather + validate fields ----- #
        file_type = self._file_type_combo.currentText().strip()
        animal_count = int(self._animal_count.value())
        animal_ids = [
            s.strip()
            for s in self._animal_ids_edit.text().split(",")
            if s.strip()
        ]
        body_parts = [
            s.strip()
            for s in self._body_parts_edit.toPlainText().splitlines()
            if s.strip()
        ]
        classifier_targets = [
            s.strip()
            for s in self._classifiers_edit.toPlainText().splitlines()
            if s.strip()
        ]

        if not file_type:
            QMessageBox.warning(
                self, "Missing field", "Pick a file type.",
            )
            return
        if animal_count < 1:
            QMessageBox.warning(
                self, "Bad value",
                "Animal count must be at least 1.",
            )
            return
        # If animal_ids supplied, count must match animal_count.
        # If empty, auto-fill on save (handled below).
        if animal_ids and len(animal_ids) != animal_count:
            QMessageBox.warning(
                self, "Animal ID count mismatch",
                f"Animal count is {animal_count} but you provided "
                f"{len(animal_ids)} animal ID(s). Either fill in "
                f"exactly {animal_count} or leave the field blank "
                "and Mufasa will auto-name them.",
            )
            return
        if not animal_ids:
            animal_ids = [f"Animal_{i+1}" for i in range(animal_count)]

        # Duplicate-name guards
        if len(set(animal_ids)) != len(animal_ids):
            QMessageBox.warning(
                self, "Duplicate animal IDs",
                "Animal IDs must be unique.",
            )
            return
        if len(set(body_parts)) != len(body_parts):
            QMessageBox.warning(
                self, "Duplicate body parts",
                "Body-part names must be unique.",
            )
            return
        if len(set(classifier_targets)) != len(classifier_targets):
            QMessageBox.warning(
                self, "Duplicate classifier targets",
                "Classifier target names must be unique.",
            )
            return

        # ----- Read-modify-write project.toml ----- #
        try:
            from mufasa.project_layout import (
                read_project_toml, write_project_toml,
            )
            data = read_project_toml(self.config_path)
        except (ValueError, OSError) as exc:
            QMessageBox.critical(
                self, "Cannot read project.toml",
                f"{type(exc).__name__}: {exc}",
            )
            return

        pose = dict(data.get("pose", {}))
        pose["file_type"] = file_type
        pose["animal_count"] = animal_count
        pose["animal_ids"] = animal_ids
        pose["body_parts"] = body_parts
        data["pose"] = pose

        classifiers = dict(data.get("classifiers", {}))
        classifiers["targets"] = classifier_targets
        data["classifiers"] = classifiers

        try:
            write_project_toml(Path(self.config_path), data)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(
                self, "Could not write project.toml",
                f"{type(exc).__name__}: {exc}",
            )
            return

        self.metadata_updated.emit(self.config_path)
        self.accept()


__all__ = ["EditProjectMetadataDialog"]
